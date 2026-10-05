//! Public synthetic Rust-authored kernel probe, not a model engine wrapper.
mod executor;
mod quant_matvec;
use cubecl::prelude::*;
use cubecl::wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use cubecl_runtime::memory_management::{MemoryPoolOptions, PoolType};
use kuno_text::model::{Qwen4Exp, greedy};
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vendor = match std::env::args().nth(1).as_deref() {
        Some("intel") => 0x8086,
        Some("nvidia") => 0x10de,
        _ => return Err("expected intel or nvidia".into()),
    };
    let mode = std::env::args().nth(2).unwrap_or("probe".into());
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
                memory_config: cubecl::MemoryConfiguration::Custom {
                    pool_options: (15..=26)
                        .map(|power| MemoryPoolOptions {
                            pool_type: PoolType::ExclusivePages {
                                max_alloc_size: 1u64 << power,
                            },
                            dealloc_period: Some(1),
                        })
                        .collect(),
                },
            },
        );
        let client = WgpuRuntime::client(&selected);
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
                }
                let dtype = kind.gguf_id();
                quant_matvec::validate_layout(dtype, rows, cols, packed.len(), input.len(), rows)?;
                let words = quant_matvec::pack_le_words(&packed);
                let upload = quant_matvec::words_to_le_upload(&words);
                let w = client.create_from_slice(&upload);
                let x = client.create_from_slice(f32::as_bytes(&input));
                let y = client.empty(rows * 4);
                // SAFETY: validated upload sizes, handle element types and exact logical lengths.
                quant_matvec::packed_matvec::launch::<WgpuRuntime>(
                    &client,
                    CubeCount::Static((rows as u32).div_ceil(64), 1, 1),
                    CubeDim::new_1d(64),
                    unsafe { ArrayArg::from_raw_parts(w, words.len()) },
                    unsafe { ArrayArg::from_raw_parts(x, input.len()) },
                    unsafe { ArrayArg::from_raw_parts(y.clone(), rows) },
                    rows,
                    cols,
                    dtype,
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
            }
        }
        if mode != "probe" {
            let fixture: serde_json::Value =
                serde_json::from_str(include_str!("../../text/qwen38-baseline.json"))?;
            let (prompt, limit) = match mode.as_str() {
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
            let mut executor = executor::GpuExecutor::new(
                client.clone(),
                if vendor == 0x8086 {
                    8 * 1024 * 1024 * 1024
                } else {
                    4 * 1024 * 1024 * 1024
                },
            )?;
            let cancel = AtomicBool::new(false);
            println!(
                "gpu_fixture={mode} prompt_tokens={} seed=424242 temperature=0 load_ms={}",
                ids.len(),
                load.elapsed().as_millis()
            );
            let start = Instant::now();
            let mut logits = Vec::new();
            for (n, token) in ids.iter().enumerate() {
                logits.zeroize();
                logits = model.step(&mut executor, &mut sequence, *token, &cancel, |_| {})?;
                println!(
                    "gpu_prefill_tokens={} elapsed_ms={}",
                    n + 1,
                    start.elapsed().as_millis()
                );
            }
            let prefill = start.elapsed();
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
                    logits = model.step(&mut executor, &mut sequence, token, &cancel, |_| {})?;
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
                "matrix_ms={} weight_read_ms={} weight_upload_ms={} gpu_wait_ms={}",
                executor.matrix_ns / 1_000_000,
                executor.weight_read_ns / 1_000_000,
                executor.weight_upload_ns / 1_000_000,
                executor.gpu_wait_ns / 1_000_000
            );
            logits.zeroize();
            generated.zeroize();
            sequence.reset();
            decoder.clear();
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })
}
