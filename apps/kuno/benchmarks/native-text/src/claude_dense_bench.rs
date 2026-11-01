//! Dense prefill projection GEMM benchmark (synthetic public data, no model).
//!
//! `--public-dense-bench SETTINGS`: times the production `strata_matrix::dense`
//! GEMM at the model's real dense projection shapes and reports TFLOPS, so GEMM
//! kernels can be iterated on in seconds instead of full model runs.
use crate::private_arena::PrivateArena;
use crate::strata_prompt::allocation::ROOT_SIZES;
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Instant;

type R<T> = std::result::Result<T, String>;

/// (label, tokens, K = input columns, N = output rows, bf16)
pub(crate) const SHAPES: [(&str, usize, usize, usize, bool); 11] = [
    ("gdn_qkv", 4096, 2560, 10240, false),
    ("gdn_gate", 4096, 2560, 6144, false),
    ("gdn_out", 4096, 6144, 2560, false),
    ("qsa_q", 4096, 2560, 12288, false),
    ("shared_gate", 4096, 2560, 640, false),
    ("shared_down", 4096, 640, 2560, false),
    ("tail_qkv", 148, 2560, 10240, false),
    ("router_bf16", 4096, 2560, 512, true),
    ("hc_down_bf16", 4096, 10240, 320, true),
    ("hc_up_bf16", 4096, 320, 10240, true),
    ("alpha_bf16", 4096, 2560, 48, true),
];

fn sync(client: &ComputeClient<WgpuRuntime>) -> R<()> {
    cubecl::future::block_on(client.sync()).map_err(|e| format!("sync: {e:?}"))
}

fn f16_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(|v| half::f16::from_f32(v).to_le_bytes()).collect()
}

/// Deterministic small-magnitude public values.
fn pattern(n: usize, salt: usize) -> impl Iterator<Item = f32> {
    (0..n).map(move |i| (((i * 2654435761 + salt * 97) >> 7) % 61) as f32 / 61.0 - 0.5)
}

pub(crate) fn run(client: &ComputeClient<WgpuRuntime>) -> R<()> {
    if std::env::var_os("KUNO_DPAS_BENCH").is_some() {
        return crate::claude_gemm_i8::dpas_bench(client);
    }
    if std::env::var_os("KUNO_I8_BENCH").is_some() {
        return crate::claude_gemm_i8::bench(client);
    }
    client
        .configure_public_target_request_io_with_backlog(Default::default())
        .map_err(|e| format!("request io: {e:?}"))?;
    let budget = ROOT_SIZES.iter().sum::<u64>() as usize + 64 * 1024 * 1024;
    let mut arena = PrivateArena::new_transient_on_configured(
        client.clone(),
        budget,
        Arc::new(AtomicBool::new(false)),
    )
    .map_err(|e| format!("arena: {e:?}"))?;
    let ids = [
        arena.allocate_zeroed(ROOT_SIZES[0] as usize / 4).map_err(|e| format!("root0: {e:?}"))?,
        arena.allocate_zeroed(ROOT_SIZES[1] as usize / 4).map_err(|e| format!("root1: {e:?}"))?,
        arena.allocate_zeroed(ROOT_SIZES[2] as usize / 4).map_err(|e| format!("root2: {e:?}"))?,
    ];
    let _scope = crate::strata_prompt::Scope::new(&arena, ids).map_err(|e| format!("scope: {e:?}"))?;
    let max_tokens = SHAPES.iter().map(|s| s.1).max().unwrap();
    let max_k = SHAPES.iter().map(|s| s.2).max().unwrap();
    let max_kn = SHAPES.iter().map(|s| s.2 * s.3).max().unwrap();
    let max_out = SHAPES.iter().map(|s| s.1 * s.3).max().unwrap();
    let input = client.create_from_slice(&f16_bytes(pattern(max_tokens * max_k, 1)));
    let weight = client.create_from_slice(&f16_bytes(pattern(max_kn, 2)));
    let output = client.empty(max_out * 4);
    let fault = client.create_from_slice(&0u32.to_le_bytes());
    let candidate_out = client.empty(max_out * 4);
    for (label, tokens, k, n, bf16) in SHAPES {
        // Correctness: candidate vs the original chain on identical inputs.
        let bypass = |on: bool| crate::claude_gemm::BYPASS.store(on, std::sync::atomic::Ordering::Relaxed);
        bypass(true);
        crate::strata_matrix::dense(
            client,
            [input.clone(), weight.clone(), output.clone(), fault.clone()],
            tokens, k, n, 0, 0, n, bf16,
        )
        .map_err(|e| format!("{label}: reference: {e:?}"))?;
        bypass(false);
        let ran = crate::claude_gemm::try_dense(
            client,
            [input.clone(), weight.clone(), candidate_out.clone(), fault.clone()],
            tokens, k, n, 0, 0, n, bf16,
        )
        .map_err(|e| format!("{label}: candidate: {e:?}"))?;
        if ran {
            sync(client)?;
            let (mut max_diff, mut max_ref) = (0f32, 0f32);
            // 1 MiB windows spread over the whole tokens x n output.
            let total = (tokens * n * 4) as u64;
            let window = 1u64 << 20;
            let windows = 4u64;
            for w in 0..windows {
                let start = (total - window) / (windows - 1) * w / 256 * 256;
                let view = |h: &cubecl_runtime::server::Handle| {
                    h.clone()
                        .offset_start(start)
                        .offset_end(h.size_in_used() - start - window)
                };
                let reference = client.read_one(view(&output)).map_err(|e| format!("read: {e:?}"))?;
                let actual = client.read_one(view(&candidate_out)).map_err(|e| format!("read: {e:?}"))?;
                for i in 0..(window as usize / 4) {
                    let r = f32::from_le_bytes(reference[i * 4..i * 4 + 4].try_into().unwrap());
                    let a = f32::from_le_bytes(actual[i * 4..i * 4 + 4].try_into().unwrap());
                    max_diff = max_diff.max((r - a).abs());
                    max_ref = max_ref.max(r.abs());
                }
            }
            let launch = || {
                crate::claude_gemm::try_dense(
                    client,
                    [input.clone(), weight.clone(), candidate_out.clone(), fault.clone()],
                    tokens, k, n, 0, 0, n, bf16,
                )
                .map(|_| ())
                .map_err(|e| format!("{label}: candidate: {e:?}"))
            };
            launch()?;
            sync(client)?;
            let reps = 20;
            let started = Instant::now();
            for _ in 0..reps {
                launch()?;
            }
            sync(client)?;
            let seconds = started.elapsed().as_secs_f64() / reps as f64;
            let tflops = 2.0 * tokens as f64 * k as f64 * n as f64 / seconds / 1e12;
            eprintln!(
                "public_dense_bench candidate {label} ms={:.3} tflops={tflops:.1} max_abs_diff={max_diff:.3e} max_abs_ref={max_ref:.3e}",
                seconds * 1e3
            );
        } else {
            eprintln!("public_dense_bench candidate {label} not_applicable");
        }
        let launch = || {
            crate::strata_matrix::dense(
                client,
                [input.clone(), weight.clone(), output.clone(), fault.clone()],
                tokens,
                k,
                n,
                0,
                0,
                n,
                bf16,
            )
            .map_err(|e| format!("{label}: dense: {e:?}"))
        };
        // Original chain timing. Warm up (compiles pipelines), then time.
        bypass(true);
        launch()?;
        sync(client)?;
        let reps = 20;
        let started = Instant::now();
        for _ in 0..reps {
            launch()?;
        }
        sync(client)?;
        bypass(false);
        let seconds = started.elapsed().as_secs_f64() / reps as f64;
        let tflops = 2.0 * tokens as f64 * k as f64 * n as f64 / seconds / 1e12;
        eprintln!(
            "public_dense_bench {label} tokens={tokens} k={k} n={n} ms={:.3} tflops={tflops:.1}",
            seconds * 1e3
        );
    }
    Ok(())
}
