//! Public synthetic Rust-authored kernel probe, not a model engine wrapper.
mod bank96_probe;
mod bank_slots;
mod executor;
mod expert_banks;
mod gdn_chunk;
mod gdn_chunk_probe;
mod gdn_executor;
mod gdn_gpu;
mod gdn_pointwise;
mod gr_batch_probe;
mod main_admission;
mod matmul_probe;
mod ordered_public;
mod private_arena;
mod qsa_gpu;
mod quant_matvec;
mod resident_gdn;
mod resident_gr;
mod resident_model;
mod resident_moe;
mod resident_probe;
mod resident_qsa;
mod resident_util;
mod residual_batch_gpu;
mod residual_gpu;
mod small_cpu;
mod transient_probe;
use cubecl::prelude::*;
use cubecl::wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use kuno_text::model::{MatrixExecutor, Qwen4Exp, RoutedWeights, greedy};
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TensorType, TokenizerLimits};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zeroize::Zeroize;

#[cube(launch)]
fn matvec<F: Float>(matrix: &Array<F>, input: &Array<F>, output: &mut Array<F>, columns: usize) {
    let row = ABSOLUTE_POS;
    if row < output.len() {
        let mut sum = F::new(0.0);
        for col in 0..columns {
            sum += matrix[row * columns + col] * input[col];
        }
        output[row] = sum;
    }
}

fn transient_resident_mode(mode: &str, raw: Option<&str>) -> Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1")
            if matches!(
                mode,
                "resident-bang" | "resident-chat" | "resident-rainwater"
            ) =>
        {
            Ok(true)
        }
        Some("1") => Err("owned runtime I/O requires a full resident fixture"),
        Some(_) => Err("KUNO_LAB_TRANSIENT_IO must be 0 or 1"),
    }
}

fn token_io_cadence(raw: Option<&str>, owned_resident: bool) -> Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") if owned_resident => Ok(true),
        Some("1") => Err("token I/O checkpoint requires owned resident I/O"),
        Some(_) => Err("KUNO_LAB_IO_TOKEN_CHECKPOINT must be 0 or 1"),
    }
}

fn public_owned_zero_audit_mode(transient: bool, raw: Option<&str>) -> Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") if transient => Ok(true),
        Some("1") => Err("owned zero audit requires the transient fixed-public resident fixture"),
        Some(_) => Err("KUNO_LAB_OWNED_ZERO_AUDIT must be 0 or 1"),
    }
}
fn report_public_owned_zero_audit(
    resident: &resident_model::ResidentModel<'_, '_>,
    enabled: bool,
    reports: &mut Vec<serde_json::Value>,
) -> Result<(), &'static str> {
    if enabled {
        let (main, source) = resident
            .public_owned_zero_audit_receipt()
            .ok_or("owned main zero audit did not accept complete cleanup")?;
        println!(
            "public_resident_owned_zero_audit main_buffers={} main_bytes={} source_buffers={} source_bytes={}",
            main.buffers, main.bytes, source.buffers, source.bytes
        );
        reports.push(serde_json::json!({
            "main_buffers":main.buffers,"main_bytes":main.bytes,
            "source_buffers":source.buffers,"source_bytes":source.bytes
        }));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vendor = match std::env::args().nth(1).as_deref() {
        Some("intel") => 0x8086,
        Some("nvidia") => 0x10de,
        _ => return Err("expected intel or nvidia".into()),
    };
    let mode = std::env::args().nth(2).unwrap_or("probe".into());
    let transient_setting = match std::env::var("KUNO_LAB_TRANSIENT_IO") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_TRANSIENT_IO must be 0 or 1".into()),
    };
    let transient_resident = transient_resident_mode(&mode, transient_setting.as_deref())?;
    let io_token_setting = match std::env::var("KUNO_LAB_IO_TOKEN_CHECKPOINT") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_IO_TOKEN_CHECKPOINT must be 0 or 1".into()),
    };
    let token_checkpoint = token_io_cadence(io_token_setting.as_deref(), transient_resident)?;
    let owned_audit_setting = match std::env::var("KUNO_LAB_OWNED_ZERO_AUDIT") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_OWNED_ZERO_AUDIT must be 0 or 1".into()),
    };
    let public_owned_zero_audit =
        public_owned_zero_audit_mode(transient_resident, owned_audit_setting.as_deref())?;
    let ordered_setting = match std::env::var("KUNO_LAB_ORDERED_PUBLIC_UPLOAD") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_ORDERED_PUBLIC_UPLOAD must be 0 or 1".into()),
    };
    let ordered_public_upload = ordered_public::configured(
        ordered_setting.as_deref(),
        transient_resident,
        std::env::var("KUNO_LAB_BATCH_EXPERTS").as_deref() == Ok("1"),
    )?;
    // One retirement flag belongs to the configured physical stream, across
    // every fresh logical owner/executor in the cancellation diagnostic.
    let transient_context = transient_resident.then(|| std::sync::Arc::new(AtomicBool::new(false)));
    let requested_slots = match std::env::var("KUNO_LAB_EXPERT_SLOTS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_EXPERT_SLOTS must be an integer 10..72 or 96".into()),
    };
    // Reject malformed/out-of-range overrides before creating a GPU device.
    let expert_slots = bank_slots::configured_capacity(
        requested_slots.as_deref(),
        if vendor == 0x8086 { 64 } else { 16 },
    )
    .map_err(|_| "KUNO_LAB_EXPERT_SLOTS must be an integer 10..72 or 96")?;
    let static_cache_setting = match std::env::var("KUNO_LAB_STATIC_CACHE_BYTES") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_STATIC_CACHE_BYTES must be integer bytes".into()),
    };
    let cache_budget = expert_banks::configured_static_cache(
        static_cache_setting.as_deref(),
        if vendor == 0x8086 { 8 } else { 6 } * 1024 * 1024 * 1024,
    )?;
    expert_banks::validate_segmented_configuration(
        expert_slots,
        static_cache_setting.as_ref().map(|_| cache_budget),
        cache_budget,
        std::env::var("KUNO_LAB_BLOCK_MOE").as_deref() == Ok("1"),
        std::env::var("KUNO_LAB_BATCH_EXPERTS").as_deref() == Ok("1"),
        std::env::var("KUNO_LAB_WORD_DOWN").as_deref() == Ok("1"),
        matches!(
            mode.as_str(),
            "resident-bang" | "resident-chat" | "resident-rainwater"
        ),
    )?;
    expert_banks::validate_ordered_segmented(expert_slots)?;
    cubecl::future::block_on(async move {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let mut candidates: Vec<_> = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .await
            .into_iter()
            .filter(|a| {
                a.get_info().vendor == vendor
                    && a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
            })
            .collect();
        if candidates.len() != 1 {
            return Err("physical GPU selection is ambiguous".into());
        }
        let adapter = candidates.remove(0);
        let info = adapter.get_info();
        println!(
            "adapter={} vendor={:04x} device={:04x} backend={:?}",
            info.name, info.vendor, info.device, info.backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Kuno public synthetic Rust kernel"),
                required_features: adapter.features()
                    & (wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await?;
        let gr_storage_alignment = adapter.limits().min_storage_buffer_offset_alignment as usize;
        let selected = init_device(
            WgpuSetup {
                instance,
                adapter,
                device,
                queue,
                backend: wgpu::Backend::Vulkan,
            },
            RuntimeOptions {
                tasks_max: 32,
                memory_config: private_arena::required_memory_configuration(),
            },
        );
        let client = WgpuRuntime::client(&selected);
        if mode == "gr-batch-probe" {
            gr_batch_probe::run(&client, gr_storage_alignment)?;
            return Ok(());
        }
        if mode == "gdn-chunk-probe" {
            gdn_chunk_probe::run(&client)?;
            return Ok(());
        }
        if mode == "bank96-probe" {
            bank96_probe::run(&client)?;
            return Ok(());
        }
        if mode == "matmul-probe" {
            matmul_probe::run(&client)?;
            return Ok(());
        }
        if let Some(retired) = &transient_context {
            let configured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                client.configure_transient_io(executor::RESIDENT_RUNTIME_IO_MAX)
            }));
            if !matches!(configured, Ok(Ok(()))) {
                retired.store(true, std::sync::atomic::Ordering::Release);
                return Err("resident runtime I/O initialization failed".into());
            }
        }
        if mode == "transient-io-probe" {
            transient_probe::run(&client)?;
            return Ok(());
        }
        for (rows, cols) in [(3usize, 5usize), (137, 257)] {
            let weights: Vec<f32> = (0..rows * cols)
                .map(|n| ((n % 23) as f32 - 11.0) / 23.0)
                .collect();
            let input: Vec<f32> = (0..cols).map(|n| ((n % 7) as f32 - 3.0) / 7.0).collect();
            let expected: Vec<f32> = weights
                .chunks(cols)
                .map(|row| row.iter().zip(&input).map(|(a, b)| a * b).sum())
                .collect();
            let w = client.create_from_slice(f32::as_bytes(&weights));
            let x = client.create_from_slice(f32::as_bytes(&input));
            let y = client.empty(rows * 4);
            matvec::launch::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static((rows as u32).div_ceil(64), 1, 1),
                CubeDim::new_1d(64),
                // SAFETY: handles are allocated immediately above for exactly these f32 element counts.
                unsafe { ArrayArg::from_raw_parts(w, weights.len()) },
                unsafe { ArrayArg::from_raw_parts(x, input.len()) },
                unsafe { ArrayArg::from_raw_parts(y.clone(), rows) },
                cols,
            );
            let bytes = client.read_one(y)?;
            let actual = f32::from_bytes(&bytes);
            if actual.len() != expected.len() {
                return Err("output length mismatch".into());
            }
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f32, f32::max);
            if !max_error.is_finite() || max_error > 1e-4 {
                return Err("Rust GPU numeric fixture failed".into());
            }
            println!("rust_matvec rows={rows} cols={cols} max_abs_error={max_error}");
            drop(bytes);
            if transient_resident {
                executor::sanitize_resident_io(&client)?;
            }
        }
        if mode == "resident-qsa-probe" {
            for report in resident_qsa::public_parity_probe(&client)? {
                println!(
                    "public_resident_qsa visible={} complete={} selected={} norm_error={} score_error={} output_error={} owned_bytes={}",
                    report.visible,
                    report.complete_blocks,
                    report.selected_tokens,
                    report.max_norm_error,
                    report.max_score_error,
                    report.max_output_error,
                    report.charged_bytes
                );
            }
            return Ok(());
        }
        if mode == "resident-gdn-probe" {
            resident_probe::gdn(&client)?;
            return Ok(());
        }
        if mode == "resident-probe" {
            resident_probe::run(&client)?;
            return Ok(());
        }
        if mode == "gdn-probe" {
            for report in gdn_executor::parity_probe(&client)? {
                println!(
                    "rust_gdn shape={:?} steps={} max_state_abs_error={} max_output_abs_error={} elapsed_ms={}",
                    report.shape,
                    report.steps,
                    report.max_state_abs_error,
                    report.max_output_abs_error,
                    report.elapsed.as_millis()
                );
            }
            return Ok(());
        }
        if let Some(path) = std::env::var_os("KUNO_TEST_GGUF") {
            let file = GgufFile::open(path, GgufLimits::default())?;
            for kind in [
                TensorType::F32,
                TensorType::F16,
                TensorType::Bf16,
                TensorType::Q4K,
                TensorType::Q5_0,
                TensorType::Q6K,
                TensorType::Q8_0,
            ] {
                let tensor = if kind == TensorType::Q5_0 {
                    file.tensor("per_layer_token_embd.weight").unwrap()
                } else {
                    let info = file
                        .tensors()
                        .values()
                        .find(|t| t.dtype() == kind)
                        .ok_or("missing format")?;
                    file.tensor(info.name()).unwrap()
                };
                let rows = 7usize.min(tensor.info().row_count() as usize);
                let cols = tensor.info().row_len() as usize;
                let bytes_count = rows * tensor.info().row_bytes() as usize;
                if bytes_count > 512 * 1024 || cols > 65536 {
                    return Err("fixture slice exceeds bound".into());
                }
                let mut packed = vec![0u8; bytes_count];
                tensor.read_range(0, &mut packed)?;
                let input: Vec<f32> = (0..cols).map(|n| ((n % 7) as f32 - 3.0) / 7.0).collect();
                let mut decoded = vec![0f32; cols];
                let mut expected = vec![0f32; rows];
                for (row, out) in expected.iter_mut().enumerate() {
                    let row_bytes = tensor.info().row_bytes() as usize;
                    kind.decode(
                        &packed[row * row_bytes..(row + 1) * row_bytes],
                        &mut decoded,
                    )?;
                    *out = decoded.iter().zip(&input).map(|(w, x)| w * x).sum();
                    let tree = quant_matvec::cooperative::reference_tree_dot(&decoded, &input)?;
                    if !tree.is_finite() || (tree - *out).abs() > 1e-4 + 3e-4 * out.abs() {
                        return Err("CPU reduction schedule oracle failed".into());
                    }
                }
                let dtype = kind.gguf_id();
                quant_matvec::validate_layout(dtype, rows, cols, packed.len(), input.len(), rows)?;
                let words = quant_matvec::pack_le_words(&packed);
                let upload = quant_matvec::words_to_le_upload(&words);
                let w = client.create_from_slice(&upload);
                let x = client.create_from_slice(f32::as_bytes(&input));
                let y = client.empty(rows * 4);
                // SAFETY: validated upload sizes, handle element types and exact logical lengths.
                quant_matvec::block::block_matvec::launch::<WgpuRuntime>(
                    &client,
                    CubeCount::Static(rows as u32, 1, 1),
                    CubeDim::new_1d(64),
                    unsafe { ArrayArg::from_raw_parts(w, words.len()) },
                    unsafe { ArrayArg::from_raw_parts(x, input.len()) },
                    unsafe { ArrayArg::from_raw_parts(y.clone(), rows) },
                    rows,
                    cols,
                    dtype,
                    0,
                );
                let actual_bytes = client.read_one(y)?;
                let actual = f32::from_bytes(&actual_bytes);
                let max_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0f32, f32::max);
                let magnitude = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                if actual.len() != rows
                    || actual.iter().any(|v| !v.is_finite())
                    || max_error > 1e-4 + 3e-4 * magnitude
                {
                    return Err("packed Rust GPU numeric fixture failed".into());
                }
                println!(
                    "rust_packed_matvec dtype={dtype} rows={rows} cols={cols} max_abs_error={max_error}"
                );
                drop(actual_bytes);
                if transient_resident {
                    executor::sanitize_resident_io(&client)?;
                }
            }
        }
        if mode == "matrix-bench" {
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let file = GgufFile::open(path, GgufLimits::default())?;
            let cancel = AtomicBool::new(false);
            let mut executor = executor::GpuExecutor::new(
                &file,
                client.clone(),
                2 * 1024 * 1024 * 1024,
                16,
                4 * 1024 * 1024 * 1024,
            )?;
            executor.small_cpu_enabled = false;
            executor.block_large_only = false;
            let mut results = Vec::new();
            for name in [
                "output.weight",
                "blk.0.hc_attn_down.weight",
                "blk.0.hc_ffn_down.weight",
            ] {
                let tensor = file.tensor(name).ok_or("missing public test matrix")?;
                let input: Vec<f32> = (0..tensor.info().row_len())
                    .map(|i| ((i % 13) as f32 - 6.) / 13.)
                    .collect();
                executor.block_kernel = false;
                let mut oracle = executor.run(tensor, 0, &input, &cancel)?;
                for variant in [true, false] {
                    executor.block_kernel = variant;
                    let mut warm = executor.run(tensor, 0, &input, &cancel)?;
                    warm.zeroize();
                }
                let mut generic = Vec::new();
                let mut block = Vec::new();
                let mut max_error = 0f32;
                // Alternating ABBA order within one resident process bounds
                // cache/JIT/clock drift better than independent tiny graph runs.
                for _ in 0..5 {
                    for variant in [false, true, true, false] {
                        executor.block_kernel = variant;
                        let t = Instant::now();
                        let mut out = executor.run(tensor, 0, &input, &cancel)?;
                        let ns = t.elapsed().as_nanos() as u64;
                        if variant {
                            block.push(ns);
                        } else {
                            generic.push(ns);
                        }
                        for (&a, &e) in out.iter().zip(&oracle) {
                            let error = (a - e).abs();
                            max_error = max_error.max(error);
                            if !a.is_finite() || error > 1e-4 + 3e-4 * e.abs() {
                                return Err("block numeric parity failed".into());
                            }
                        }
                        out.zeroize();
                    }
                }
                generic.sort_unstable();
                block.sort_unstable();
                println!(
                    "public_matrix_bench tensor={name} generic_median_us={} block_median_us={} max_abs_error={max_error}",
                    generic[5] / 1000,
                    block[5] / 1000
                );
                results.push(serde_json::json!({"tensor":name,"generic_ns":generic,"block_ns":block,"max_abs_error":max_error}));
                oracle.zeroize();
            }
            if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
                use std::io::Write;
                let receipt = serde_json::json!({"scope":"fixed-public-synthetic-operator-test-only","fixture":"matrix-bench","vendor":vendor,"results":results});
                let mut out = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?;
                out.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
            }
            return Ok(());
        }
        if matches!(
            mode.as_str(),
            "moe-probe" | "moe-bench" | "moe-word-probe" | "moe-word-bench"
        ) {
            let word_probe = mode.starts_with("moe-word-");
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let file = GgufFile::open(path, GgufLimits::default())?;
            let cancel = AtomicBool::new(false);
            let mut executor = executor::GpuExecutor::new(
                &file,
                client.clone(),
                64 * 1024 * 1024,
                16,
                2 * 1024 * 1024 * 1024,
            )?;
            if word_probe {
                executor.word_down = true;
            }
            let mut results = Vec::new();
            for kind in [TensorType::Q5_0, TensorType::Q8_0] {
                let down = file
                    .tensors()
                    .values()
                    .find(|t| t.dtype() == kind && t.name().ends_with("ffn_down_exps.weight"))
                    .ok_or("missing expert dtype")?;
                let base = down.name().strip_suffix("ffn_down_exps.weight").unwrap();
                let weights = RoutedWeights {
                    gate: file.tensor(&format!("{base}ffn_gate_exps.weight")).unwrap(),
                    up: file.tensor(&format!("{base}ffn_up_exps.weight")).unwrap(),
                    down: file.tensor(down.name()).unwrap(),
                };
                let input: Vec<f32> = (0..2560).map(|i| ((i % 23) as f32 - 11.) / 23.).collect();
                let gate = kuno_text::cpu::Matrix::new(weights.gate)?;
                let up = kuno_text::cpu::Matrix::new(weights.up)?;
                let down = kuno_text::cpu::Matrix::new(weights.down)?;
                for pass in 0..3 {
                    let ids = if pass == 2 {
                        [503, 71, 6, 97, 12, 31, 78, 491, 99, 311]
                    } else {
                        [0, 17, 511, 71, 3, 31, 255, 491, 97, 119]
                    };
                    let selected: Vec<_> = ids
                        .into_iter()
                        .enumerate()
                        .map(|(i, e)| (e, (i + 1) as f32 / 55.))
                        .collect();
                    let mut expected = vec![0.; 2560];
                    for &(e, p) in &selected {
                        let mut g = vec![0.; 640];
                        let mut u = vec![0.; 640];
                        let mut d = vec![0.; 2560];
                        gate.apply(e, &input, &mut g, &cancel)?;
                        up.apply(e, &input, &mut u, &cancel)?;
                        for (a, b) in g.iter_mut().zip(u) {
                            *a = kuno_text::ops::silu(*a) * b;
                        }
                        down.apply(e, &g, &mut d, &cancel)?;
                        for (y, x) in expected.iter_mut().zip(d) {
                            *y += p * x;
                        }
                    }
                    let actual = executor
                        .routed_moe(weights, &input, &selected, &cancel)?
                        .ok_or("missing routed backend")?;
                    let max_error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let magnitude = expected.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    if actual.len() != expected.len() || max_error > 1e-4 + 3e-4 * magnitude {
                        return Err("expert GPU parity failed".into());
                    }
                    println!(
                        "rust_moe dtype={} pass={pass} max_abs_error={max_error} hits={} uploads={} gdn_ms={} gdn_calls={}",
                        kind.gguf_id(),
                        executor.expert_hits,
                        executor.expert_uploads,
                        executor.gdn_ns / 1_000_000,
                        executor.gdn_calls
                    );
                    if word_probe {
                        executor.word_down = false;
                        let mut reference = executor
                            .routed_moe(weights, &input, &selected, &cancel)?
                            .ok_or("missing routed backend")?;
                        executor.word_down = true;
                        if reference.len() != actual.len() {
                            return Err("word down MoE output extent".into());
                        }
                        let mut max_difference = 0.0_f32;
                        for (&a, &e) in actual.iter().zip(&reference) {
                            let error = (a - e).abs();
                            max_difference = max_difference.max(error);
                            if !a.is_finite() || !e.is_finite() || error > 1e-4 + 3e-4 * e.abs() {
                                return Err("word down MoE numerical parity failed".into());
                            }
                        }
                        reference.zeroize();
                        println!(
                            "public_word_down dtype={} pass={pass} selected=10 physical_lanes=16 virtual_lanes=64 max_abs_difference={max_difference}",
                            kind.gguf_id()
                        );
                        results.push(serde_json::json!({"down_dtype":kind.gguf_id(),"pass":pass,"selected":10,"max_abs_difference":max_difference,"physical_lanes":16,"virtual_lanes":64,"rows_per_cube":1}));
                        if mode == "moe-word-bench" && pass == 1 {
                            for variant in [false, true] {
                                executor.word_down = variant;
                                let mut warm = executor
                                    .routed_moe(weights, &input, &selected, &cancel)?
                                    .ok_or("missing routed backend")?;
                                warm.zeroize();
                            }
                            let misses = executor.expert_uploads;
                            let mut generic = Vec::new();
                            let mut word = Vec::new();
                            for _ in 0..5 {
                                for variant in [false, true, true, false] {
                                    executor.word_down = variant;
                                    let start = Instant::now();
                                    let mut output = executor
                                        .routed_moe(weights, &input, &selected, &cancel)?
                                        .ok_or("missing routed backend")?;
                                    let ns = start.elapsed().as_nanos() as u64;
                                    if variant {
                                        word.push(ns);
                                    } else {
                                        generic.push(ns);
                                    }
                                    if output.len() != actual.len() {
                                        return Err("word down timed MoE output extent".into());
                                    }
                                    for (&a, &e) in output.iter().zip(&actual) {
                                        let error = (a - e).abs();
                                        max_difference = max_difference.max(error);
                                        if !a.is_finite()
                                            || !e.is_finite()
                                            || error > 1e-4 + 3e-4 * e.abs()
                                        {
                                            return Err(
                                                "word down timed MoE numerical parity failed"
                                                    .into(),
                                            );
                                        }
                                    }
                                    output.zeroize();
                                }
                            }
                            executor.word_down = true;
                            if executor.expert_uploads != misses {
                                return Err("MoE cache changed during word down bench".into());
                            }
                            generic.sort_unstable();
                            word.sort_unstable();
                            println!(
                                "public_word_down_bench dtype={} generic_median_us={} word_median_us={} max_abs_difference={max_difference}",
                                kind.gguf_id(),
                                generic[5] / 1000,
                                word[5] / 1000
                            );
                            results.push(serde_json::json!({"down_dtype":kind.gguf_id(),"generic_ns":generic,"word_ns":word,"max_abs_difference":max_difference,"cache_misses_during_measurement":0,"selected":10,"physical_lanes":16,"virtual_lanes":64,"rows_per_cube":1,"block_gate_up":executor.block_moe}));
                        }
                    }
                    if mode == "moe-bench" && pass == 1 {
                        for variant in [false, true] {
                            executor.block_moe = variant;
                            let mut warm = executor
                                .routed_moe(weights, &input, &selected, &cancel)?
                                .ok_or("missing routed backend")?;
                            warm.zeroize();
                        }
                        let misses = executor.expert_uploads;
                        let mut generic = Vec::new();
                        let mut block = Vec::new();
                        let mut max_difference = 0f32;
                        for _ in 0..5 {
                            for variant in [false, true, true, false] {
                                executor.block_moe = variant;
                                let start = Instant::now();
                                let mut out = executor
                                    .routed_moe(weights, &input, &selected, &cancel)?
                                    .ok_or("missing routed backend")?;
                                let ns = start.elapsed().as_nanos() as u64;
                                if variant {
                                    block.push(ns);
                                } else {
                                    generic.push(ns);
                                }
                                for (&a, &e) in out.iter().zip(&actual) {
                                    let error = (a - e).abs();
                                    max_difference = max_difference.max(error);
                                    if !a.is_finite() || error > 1e-4 + 3e-4 * e.abs() {
                                        return Err("block MoE numeric parity failed".into());
                                    }
                                }
                                out.zeroize();
                            }
                        }
                        if executor.expert_uploads != misses {
                            return Err("MoE cache changed during fixed bench".into());
                        }
                        generic.sort_unstable();
                        block.sort_unstable();
                        println!(
                            "public_moe_bench dtype={} generic_median_us={} block_median_us={} max_abs_difference={max_difference}",
                            kind.gguf_id(),
                            generic[5] / 1000,
                            block[5] / 1000
                        );
                        results.push(serde_json::json!({"down_dtype":kind.gguf_id(),"generic_ns":generic,"block_ns":block,"max_abs_difference":max_difference,"cache_misses_during_measurement":0}));
                    }
                }
            }
            if (mode == "moe-bench" || word_probe)
                && let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT")
            {
                use std::io::Write;
                let receipt = serde_json::json!({"scope":"fixed-public-synthetic-operator-test-only","fixture":mode,"vendor":vendor,"word_down":executor.word_down,"results":results});
                let mut out = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?;
                out.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
            }
            return Ok(());
        }
        if mode != "probe" {
            let fixture: serde_json::Value =
                serde_json::from_str(include_str!("../../text/qwen38-baseline.json"))?;
            let resident_enabled = mode.starts_with("resident-");
            let prefill_last_head =
                std::env::var("KUNO_LAB_PREFILL_LAST_HEAD").as_deref() == Ok("1");
            if prefill_last_head && !resident_enabled {
                return Err("last-head prefill requires a resident fixture".into());
            }
            let fixture_mode = mode.strip_prefix("resident-").unwrap_or(&mode);
            let (prompt, limit) = match fixture_mode {
                "bang" => ("!", 4usize),
                "chat" => (
                    "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    8,
                ),
                "rainwater" => (
                    fixture["prompt"].as_str().ok_or("invalid public fixture")?,
                    128,
                ),
                _ => return Err("unknown public fixture".into()),
            };
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let load = Instant::now();
            let file = GgufFile::open(path, GgufLimits::default())?;
            let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default())?;
            let model = Qwen4Exp::bind(&file)?;
            let ids = tokenizer.encode(prompt, true)?;
            let mut sequence = model.sequence(ids.len() + limit)?;
            let device_budget = if vendor == 0x8086 { 26 } else { 10 } * 1024 * 1024 * 1024;
            let bank_charge = expert_banks::preflight_residency(
                model.bound_graph(),
                expert_slots,
                cache_budget,
                device_budget,
            )?;
            let mut executor = executor::GpuExecutor::new(
                &file,
                client.clone(),
                cache_budget,
                expert_slots,
                device_budget,
            )?;
            if let Some(retired) = &transient_context {
                executor.enable_transient_resident(retired.clone())?;
            }
            let cancel = AtomicBool::new(false);
            let mut resident = if resident_enabled {
                Some(
                    resident_model::ResidentModel::new_with_public_owned_zero_audit(
                        model.bound_graph(),
                        client.clone(),
                        ids.len() + limit,
                        &mut executor,
                        public_owned_zero_audit,
                    )?,
                )
            } else {
                None
            };
            if let Some(r) = &resident {
                println!("resident_private_owned_bytes={}", r.owned_bytes());
            }
            let model_bind_ms = load.elapsed().as_millis() as u64;
            println!(
                "gpu_fixture={mode} prompt_tokens={} seed=424242 temperature=0 load_ms={}",
                ids.len(),
                load.elapsed().as_millis()
            );
            let start = Instant::now();
            let mut logits = Vec::new();
            for (n, token) in ids.iter().enumerate() {
                logits.zeroize();
                logits = if let Some(r) = &mut resident {
                    if prefill_last_head && n + 1 < ids.len() {
                        r.advance_without_logits(&mut executor, *token, &cancel)?;
                        Vec::new()
                    } else {
                        let mut output = r.step(&mut executor, *token, &cancel)?;
                        std::mem::take(&mut *output)
                    }
                } else {
                    model.step(&mut executor, &mut sequence, *token, &cancel, |_| {})?
                };
                println!(
                    "gpu_prefill_tokens={} elapsed_ms={}",
                    n + 1,
                    start.elapsed().as_millis()
                );
            }
            let prefill = start.elapsed();
            let pre_stats = (
                executor.matrix_ns,
                executor.moe_ns,
                executor.gdn_ns,
                executor.weight_read_ns,
                executor.weight_upload_ns,
                executor.gpu_wait_ns,
                executor.uploaded_bytes,
                executor.expert_hits,
                executor.expert_uploads,
            );
            executor.by_tensor.clear();
            if let Some(r) = &mut resident {
                r.phase_ns.clear();
            }
            let start = Instant::now();
            let mut generated = Vec::new();
            let mut decoder = tokenizer.stream_decoder();
            for n in 0..limit {
                let token = greedy(&logits)?;
                if token == model.eos_token() {
                    break;
                }
                generated.push(token);
                let text = decoder.push_bytes(tokenizer.token_bytes(token)?)?;
                println!("synthetic_gpu_token={token} synthetic_gpu_text={text:?}");
                if n + 1 < limit {
                    logits.zeroize();
                    logits = if let Some(r) = &mut resident {
                        let mut output = r.step(&mut executor, token, &cancel)?;
                        std::mem::take(&mut *output)
                    } else {
                        model.step(&mut executor, &mut sequence, token, &cancel, |_| {})?
                    };
                }
            }
            println!(
                "gpu_prefill_ms={} gpu_generation_ms={} synthetic_gpu_tokens={generated:?} cache_bytes={} reserved_bytes={} weight_upload_bytes={} cache_hits={} gpu_matvec_calls={}",
                prefill.as_millis(),
                start.elapsed().as_millis(),
                executor.cache_bytes(),
                executor.reserved_bytes()?,
                executor.uploaded_bytes,
                executor.cache_hits,
                executor.calls
            );
            println!(
                "matrix_ms={} weight_read_ms={} weight_upload_ms={} gpu_wait_ms={} moe_ms={} moe_calls={} expert_hits={} expert_uploads={} gdn_ms={} gdn_calls={}",
                executor.matrix_ns / 1_000_000,
                executor.weight_read_ns / 1_000_000,
                executor.weight_upload_ns / 1_000_000,
                executor.gpu_wait_ns / 1_000_000,
                executor.moe_ns / 1_000_000,
                executor.moe_calls,
                executor.expert_hits,
                executor.expert_uploads,
                executor.gdn_ns / 1_000_000,
                executor.gdn_calls
            );
            let generation_ms = start.elapsed().as_millis() as u64;
            if let Some(r) = &resident {
                println!("public_resident_warm_phase_ns={:?}", r.phase_ns);
            }
            let mut pending_receipt = None;
            let mut owned_zero_audits = Vec::new();
            if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
                let mut receipt = serde_json::json!({
                    "scope":"fixed-public-synthetic-operator-test-only", "fixture":mode,"vendor":vendor,
                    "seed":424242,"temperature":0,"model_bind_ms":model_bind_ms,"prompt_tokens":ids.len(),"public_generated_ids":generated,
                    "prefill_ms":prefill.as_millis() as u64,"generation_ms":generation_ms,
                    "cache_accounted_bytes":executor.cache_bytes(),"allocator_reserved_bytes":executor.reserved_bytes()?,
                    "matrix_calls":executor.calls,"moe_calls":executor.moe_calls,"gdn_calls":executor.gdn_calls,
                    "total_matrix_ms":(executor.matrix_ns/1_000_000) as u64,
                    "total_moe_ms":(executor.moe_ns/1_000_000) as u64,"total_gdn_ms":(executor.gdn_ns/1_000_000) as u64,
                    "warm_matrix_ms":((executor.matrix_ns-pre_stats.0)/1_000_000) as u64,
                    "warm_moe_ms":((executor.moe_ns-pre_stats.1)/1_000_000) as u64,
                    "warm_gdn_ms":((executor.gdn_ns-pre_stats.2)/1_000_000) as u64,
                    "warm_weight_read_ms":((executor.weight_read_ns-pre_stats.3)/1_000_000) as u64,
                    "warm_weight_upload_ms":((executor.weight_upload_ns-pre_stats.4)/1_000_000) as u64,
                    "warm_gpu_wait_ms":((executor.gpu_wait_ns-pre_stats.5)/1_000_000) as u64,
                    "warm_uploaded_bytes":executor.uploaded_bytes-pre_stats.6,
                    "warm_expert_hits":executor.expert_hits-pre_stats.7,"warm_expert_uploads":executor.expert_uploads-pre_stats.8,
                    "public_warm_matrix_ns":executor.by_tensor,
                    "gpu_gdn_cpu_state_owner":executor.gpu_gdn&&!resident_enabled,
                    "public_resident_warm_phase_ns":resident.as_ref().map(|r|&r.phase_ns),"resident_private_owned_bytes":resident.as_ref().map(|r|r.owned_bytes()),"block_kernel":executor.block_kernel,
                    "block_large_only":executor.block_large_only,"block_moe":executor.block_moe,
                    "small_cpu_enabled":executor.small_cpu_enabled,
                    "small_cpu_ms":(executor.small_cpu_ns/1_000_000) as u64,
                    "small_cpu_calls":executor.small_cpu_calls,
                    "small_cpu_cache_bytes":executor.small_cpu.cached_bytes(),
                    "small_cpu_cache_matrices":executor.small_cpu.cached_matrices(), "resident_graph":resident_enabled,"weights":"borrowed-unmodified-ollama-blob"
                });
                receipt["word_down"] = serde_json::json!(executor.word_down);
                receipt["resident_main_admitted_upper_bytes"] =
                    serde_json::json!(executor.admission_upper_bytes()?);
                receipt["prefill_last_head"] = serde_json::json!(prefill_last_head);
                receipt["resident_transient_io"] = serde_json::json!(transient_resident);
                receipt["resident_owned_zero_audit_enabled"] =
                    serde_json::json!(public_owned_zero_audit);
                receipt["io_token_checkpoint"] = serde_json::json!(token_checkpoint);
                receipt["resident_runtime_io_limit_bytes"] = serde_json::json!(
                    transient_resident.then_some(executor::RESIDENT_RUNTIME_IO_MAX)
                );
                receipt["ordered_public_upload"] = serde_json::json!(ordered_public_upload);
                receipt["ordered_public_source_limit_bytes"] = serde_json::json!(
                    ordered_public_upload.then_some(ordered_public::SOURCE_LIMIT)
                );
                receipt["ordered_public_source_peak_bytes"] =
                    serde_json::json!(executor.ordered_source_peak_bytes());
                receipt["batch_experts"] = serde_json::json!(executor.batch_experts);
                receipt["expert_batch_window_bytes"] =
                    serde_json::json!(executor.expert_batch_bytes);
                receipt["expert_scatter_mode"] = serde_json::json!(executor.expert_scatter.label());
                receipt["expert_bank_slots"] = serde_json::json!(expert_slots);
                receipt["expert_bank_accounted_limit_bytes"] = serde_json::json!(bank_charge);
                receipt["static_cache_budget_bytes"] = serde_json::json!(cache_budget);
                receipt["explicit_static_cache"] =
                    serde_json::json!(static_cache_setting.is_some());
                receipt["expert_bank_segments"] = serde_json::json!(if expert_slots == 96 {
                    vec![64, 32]
                } else {
                    vec![expert_slots]
                });
                receipt["device_reserved_limit_bytes"] = serde_json::json!(device_budget);
                // Timings/public IDs remain in RAM until explicit owned cleanup
                // and optional cancellation/audit acceptance complete.
                pending_receipt = Some((path, receipt));
            }
            let tail = decoder.finish();
            if !tail.is_empty() {
                println!("synthetic_gpu_tail={tail:?}");
            }
            let mut matrix_profile: Vec<_> = executor.by_tensor.iter().collect();
            matrix_profile.sort_by_key(|(_, ns)| std::cmp::Reverse(**ns));
            for (tensor, ns) in matrix_profile.into_iter().take(12) {
                println!(
                    "public_warm_matrix tensor={tensor} total_ms={}",
                    ns / 1_000_000
                );
            }
            if std::env::var("KUNO_LAB_CANCEL_CHECK").as_deref() == Ok("1") {
                let Some(r) = &mut resident else {
                    return Err("resident cancellation check requires resident fixture".into());
                };
                cancel.store(true, std::sync::atomic::Ordering::Release);
                if r.step(&mut executor, ids[0], &cancel).is_ok() || !r.erased_after_failure() {
                    return Err("resident cancellation cleanup failed".into());
                }
                cancel.store(false, std::sync::atomic::Ordering::Release);
                if r.step(&mut executor, ids[0], &cancel).is_ok() {
                    return Err("poisoned resident sequence reused".into());
                }
                report_public_owned_zero_audit(r, public_owned_zero_audit, &mut owned_zero_audits)?;
                // A poisoned admission ledger is never reset/reused. The
                // diagnostic explicitly drops the erased owner and executor;
                // executor Drop drains its public cache before a fresh binding.
                resident.take();
                executor = executor::GpuExecutor::new(
                    &file,
                    client.clone(),
                    cache_budget,
                    expert_slots,
                    device_budget,
                )?;
                if let Some(retired) = &transient_context {
                    executor.enable_transient_resident(retired.clone())?;
                }
                let mut fresh = resident_model::ResidentModel::new_with_public_owned_zero_audit(
                    model.bound_graph(),
                    client.clone(),
                    ids.len() + limit,
                    &mut executor,
                    public_owned_zero_audit,
                )?;
                let mut fresh_logits = zeroize::Zeroizing::new(Vec::new());
                for (n, token) in ids.iter().enumerate() {
                    if prefill_last_head && n + 1 < ids.len() {
                        fresh.advance_without_logits(&mut executor, *token, &cancel)?;
                    } else {
                        fresh_logits = fresh.step(&mut executor, *token, &cancel)?;
                    }
                }
                if greedy(&fresh_logits)? != generated[0] {
                    return Err("fresh resident sequence mismatch".into());
                }
                if public_owned_zero_audit {
                    fresh.close()?;
                    report_public_owned_zero_audit(&fresh, true, &mut owned_zero_audits)?;
                }
                drop(fresh);
                for layer in [0, 3] {
                    executor = executor::GpuExecutor::new(
                        &file,
                        client.clone(),
                        cache_budget,
                        expert_slots,
                        device_budget,
                    )?;
                    if let Some(retired) = &transient_context {
                        executor.enable_transient_resident(retired.clone())?;
                    }
                    let mut partial =
                        resident_model::ResidentModel::new_with_public_owned_zero_audit(
                            model.bound_graph(),
                            client.clone(),
                            ids.len() + limit,
                            &mut executor,
                            public_owned_zero_audit,
                        )?;
                    partial.arm_public_mid_token_cancel(layer)?;
                    let failed = if prefill_last_head && ids.len() > 1 {
                        partial
                            .advance_without_logits(&mut executor, ids[0], &cancel)
                            .is_err()
                    } else {
                        partial.step(&mut executor, ids[0], &cancel).is_err()
                    };
                    if !failed || !partial.erased_after_failure() {
                        return Err("partially queued resident cancellation failed".into());
                    }
                    report_public_owned_zero_audit(
                        &partial,
                        public_owned_zero_audit,
                        &mut owned_zero_audits,
                    )?;
                    cancel.store(false, std::sync::atomic::Ordering::Release);
                    println!("public_mid_token_cancel layer={layer} erased=true");
                }
                println!(
                    "public_resident_cancel_erased=true poisoned_retry_rejected=true fresh_first_token_matches=true"
                );
            }
            logits.zeroize();
            generated.zeroize();
            sequence.reset();
            decoder.clear();
            if transient_resident {
                // Cancellation diagnostics already explicitly checked and
                // dropped this owner before their fresh logical bindings.
                if let Some(r) = resident.as_mut() {
                    r.close()?;
                    if !r.erased_after_failure() {
                        return Err("resident full-span cleanup acceptance failed".into());
                    }
                    report_public_owned_zero_audit(
                        r,
                        public_owned_zero_audit,
                        &mut owned_zero_audits,
                    )?;
                }
                executor.close_transient_resident()?;
                if executor.ordered_source_pending_bytes() != 0 {
                    return Err("ordered public source cleanup incomplete".into());
                }
                println!("public_resident_owned_cleanup=true context_retired=false");
            }
            if let Some((path, mut receipt)) = pending_receipt {
                use std::io::Write;
                receipt["owned_io_cleanup_accepted"] = serde_json::json!(transient_resident);
                receipt["owned_zero_audits"] = serde_json::json!(owned_zero_audits);
                if ordered_public_upload {
                    receipt["ordered_public_source_pending_bytes_after_cleanup"] =
                        serde_json::json!(executor.ordered_source_pending_bytes());
                }
                let mut out = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?;
                out.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
            }
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}

#[cfg(test)]
mod transient_mode_tests {
    #[test]
    fn owned_zero_audit_is_explicit_and_requires_secure_fixed_public_mode() {
        use super::public_owned_zero_audit_mode;
        for transient in [false, true] {
            assert_eq!(public_owned_zero_audit_mode(transient, None), Ok(false));
            assert_eq!(
                public_owned_zero_audit_mode(transient, Some("0")),
                Ok(false)
            );
            assert_eq!(
                public_owned_zero_audit_mode(transient, Some("1")).is_ok(),
                transient
            );
            for raw in ["", "true", "01", "2"] {
                assert!(public_owned_zero_audit_mode(transient, Some(raw)).is_err());
            }
        }
        assert_eq!(public_owned_zero_audit_mode(true, Some("1")), Ok(true));
    }

    use super::{token_io_cadence, transient_resident_mode};
    #[test]
    fn token_checkpoint_requires_explicit_owned_resident_path() {
        assert_eq!(token_io_cadence(None, false), Ok(false));
        assert_eq!(token_io_cadence(Some("0"), false), Ok(false));
        assert!(token_io_cadence(Some("1"), false).is_err());
        assert_eq!(token_io_cadence(Some("1"), true), Ok(true));
        for raw in ["", "true", "2", " 1"] {
            assert!(token_io_cadence(Some(raw), true).is_err());
        }
    }
    #[test]
    fn owned_io_is_an_explicit_full_resident_fixture_option() {
        for mode in ["resident-bang", "resident-chat", "resident-rainwater"] {
            assert_eq!(transient_resident_mode(mode, None), Ok(false));
            assert_eq!(transient_resident_mode(mode, Some("0")), Ok(false));
            assert_eq!(transient_resident_mode(mode, Some("1")), Ok(true));
        }
        for mode in ["probe", "bang", "matrix-bench", "resident-qsa-probe"] {
            assert!(transient_resident_mode(mode, Some("1")).is_err());
        }
        for raw in ["", "true", "01", "2"] {
            assert!(transient_resident_mode("resident-bang", Some(raw)).is_err());
        }
    }
}
