//! Protected production caller for the selected native engine. All private
//! bytes cross inherited pipes; diagnostics, receipts and profiling are absent.
use crate::{native_protocol as wire, strata_engine as engine};
use cubecl::{
    prelude::*,
    wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device},
};
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, model::Qwen4Exp};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Instant,
};
use zeroize::Zeroizing;
#[path = "../../../src/image_privacy.rs"]
#[allow(dead_code)]
mod protection;
type Result<T> = std::result::Result<T, &'static str>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    shards: Vec<Artifact>,
    mtp: Artifact,
    device_id: u32,
}
fn verify(artifact: &Artifact) -> Result<()> {
    if !artifact.path.is_absolute() || artifact.sha256.len() != 64 {
        return Err("native_artifact_invalid");
    }
    let mut file =
        std::fs::File::open(&artifact.path).map_err(|_| "native_artifact_unavailable")?;
    if !file
        .metadata()
        .map_err(|_| "native_artifact_unavailable")?
        .is_file()
    {
        return Err("native_artifact_invalid");
    }
    let mut hash = Sha256::new();
    let mut window = vec![0; 1024 * 1024];
    loop {
        let n = file
            .read(&mut window)
            .map_err(|_| "native_artifact_unavailable")?;
        if n == 0 {
            break;
        }
        hash.update(&window[..n]);
    }
    if format!("{:x}", hash.finalize()) != artifact.sha256 {
        return Err("native_artifact_changed");
    }
    Ok(())
}
/// Called before any runtime thread exists. Inherited experiment configuration
/// cannot enable public diagnostics or alter the installed execution policy.
pub(crate) fn run(path: &Path) -> Result<()> {
    let worker_started = Instant::now();
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
    CubeClRuntimeConfig::set(Default::default());
    std::panic::set_hook(Box::new(|_| eprintln!("native_worker_failed")));
    protection::protect_process().map_err(|_| "native_worker_protection_failed")?;
    protection::require_no_swap().map_err(|_| "native_worker_requires_no_swap")?;
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"KUNO_LAB_")
            || name.as_encoded_bytes().starts_with(b"KUNO_STRATA_")
            || name.as_encoded_bytes().starts_with(b"KUNO_TEST_")
            || name.as_encoded_bytes().starts_with(b"KUNO_PUBLIC_")
        {
            // SAFETY: the dedicated worker enters here before constructing any threads.
            unsafe {
                std::env::remove_var(name);
            }
        }
    }
    // SAFETY: same single-threaded startup boundary. These are policy constants,
    // not public-fixture authorization; LANGUAGE_PUBLIC_ONLY remains absent.
    unsafe {
        std::env::set_var("KUNO_STRATA_INTEL_CORE", "protected-worker");
        std::env::set_var(
            "KUNO_LAB_STATIC_CACHE_BYTES",
            (5u64 * 1024 * 1024 * 1024).to_string(),
        );
    }
    let raw = std::fs::read(path).map_err(|_| "native_settings_unavailable")?;
    if raw.len() > 64 * 1024 {
        return Err("native_settings_invalid");
    }
    let settings: Settings = serde_json::from_slice(&raw).map_err(|_| "native_settings_invalid")?;
    if settings.shards.len() != 2
        || settings.device_id != 0xe223
        || !crate::strata_intel_core::enabled()
        || !cfg!(feature = "strata-gpu-adaptive-usage")
    {
        return Err("native_settings_unsupported");
    }
    protection::restrict_filesystem().map_err(|_| "native_worker_filesystem_protection_failed")?;
    for artifact in &settings.shards {
        verify(artifact)?;
    }
    verify(&settings.mtp)?;
    cubecl::future::block_on(serve(settings, worker_started))
}
async fn serve(settings: Settings, worker_started: Instant) -> Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapters: Vec<_> = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .filter(|a| {
            let info = a.get_info();
            info.vendor == 0x8086
                && info.device == settings.device_id
                && info.device_type == wgpu::DeviceType::DiscreteGpu
        })
        .collect();
    if adapters.len() != 1 {
        return Err("native_gpu_unavailable");
    }
    let adapter = adapters.remove(0);
    let limits = adapter.limits();
    crate::strata_port::validate_limits(&limits).map_err(|_| "native_gpu_unsupported")?;
    let features = crate::strata_intel_core::device_features() | wgpu::Features::SHADER_F16;
    if !adapter.features().contains(features) {
        return Err("native_gpu_unsupported");
    }
    let (device, queue) = crate::strata_matrix::request_device(
        &adapter,
        &wgpu::DeviceDescriptor {
            label: Some("Kuno native text"),
            required_features: features,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            ..Default::default()
        },
    )
    .await
    .map_err(|_| "native_gpu_unavailable")?;
    // A real physical heap query, not a caller assertion or fixed capacity.
    // Do not use budget (which can be a discretionary driver target) as total.
    {
        let h =
            unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or("native_gpu_unsupported")?;
        let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties =
            ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            h.shared_instance()
                .raw_instance()
                .get_physical_device_memory_properties2(h.raw_physical_device(), &mut properties);
        }
        let memory = properties.memory_properties;
        let heaps: Vec<_> = (0..memory.memory_heap_count as usize)
            .filter(|&i| {
                memory.memory_heaps[i]
                    .flags
                    .contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .collect();
        if heaps.len() != 1
            || memory.memory_heaps[heaps[0]]
                .size
                .saturating_sub(budget.heap_usage[heaps[0]])
                < 31 * 1024 * 1024 * 1024 + 512 * 1024 * 1024
        {
            return Err("native_gpu_memory_busy");
        }
    }
    let target_device = device.clone();
    let target_queue = queue.clone();
    let selected = init_device(
        WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: wgpu::Backend::Vulkan,
        },
        RuntimeOptions {
            tasks_max: 32768,
            memory_config: crate::private_arena::required_memory_configuration(),
        },
    );
    let client = WgpuRuntime::client(&selected);
    let paths: Vec<_> = settings.shards.iter().map(|a| a.path.as_path()).collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())
        .map_err(|_| "native_artifact_invalid")?;
    let sidecar = GgufFile::open(&settings.mtp.path, GgufLimits::default())
        .map_err(|_| "native_artifact_invalid")?;
    if file.file_len() != 83_617_662_656 {
        return Err("native_artifact_invalid");
    }
    crate::daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(name, t)| (name.as_str(), t.dtype().gguf_id())),
    )
    .map_err(|_| "native_artifact_invalid")?;
    crate::mtp_lab_profile::Profile::StrataQ2
        .validate_sidecar(&sidecar)
        .map_err(|_| "native_artifact_invalid")?;
    // SAFETY: operator-configured read-only original artifacts were authenticated
    // above; all backing owners remain live until the complete Session closes.
    unsafe {
        file.enable_mapped_weights()
            .map_err(|_| "native_artifact_invalid")?;
        sidecar
            .enable_mapped_weights()
            .map_err(|_| "native_artifact_invalid")?;
    }
    let graph = Qwen4Exp::bind(&file).map_err(|_| "native_artifact_invalid")?;
    let draft = crate::mtp_lab_profile::Profile::StrataQ2
        .bind(&sidecar, graph.bound_graph())
        .map_err(|_| "native_artifact_invalid")?;
    let tokenizer =
        Qwen35Tokenizer::from_gguf(&file, engine::input::tokenizer_limits(192 * 1024 * 1024))
            .map_err(|_| "native_artifact_invalid")?;
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    wire::write_response(&mut output, wire::READY, &[]).map_err(|_| "native_worker_pipe_failed")?;
    let Some(first) = wire::read_request(&mut input).map_err(|_| "native_request_invalid")? else {
        return Ok(());
    };
    let first_preload = first.outputs == 0;
    let request = prepare(first, &tokenizer)?;
    let cancel = AtomicBool::new(false);

    // SAFETY: the private worker has exclusive parent admission, protected
    // process memory and model backing, one physical runtime and original MAIN/
    // HOST bounds. No diagnostics/public receipt path is entered for this caller.
    let mut session = unsafe {
        engine::Session::load(
            &file,
            graph.bound_graph(),
            Some(&draft),
            &client,
            &limits,
            &target_device,
            &target_queue,
            cubecl_runtime::server::PublicTargetSubmissionBacklog::default(),
            &request,
            &cancel,
            &mut |_| {},
        )
    }
    .map_err(|_| "native_load_failed")?;
    let mut load_ns = worker_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let mut current = Some(request);
    let mut preload = first_preload;
    let attempted: Result<()> = (|| {
        while let Some(request) = current.take() {
            let prompt_count = request.geometry().map_err(|_| "native_request_invalid")?.0;
            let start = Instant::now();
            let mut prompt_ns = 0;
            let mut generation_ns = None;
            let mut decoder = tokenizer.stream_decoder();
            let result = session.run(
                request,
                &cancel,
                &mut |token| {
                    if !preload && token != graph.eos_token() {
                        let text =
                            Zeroizing::new(decoder.push_bytes(tokenizer.token_bytes(token)?)?);
                        if !text.is_empty() {
                            wire::write_response(&mut output, wire::TEXT, text.as_bytes())
                                .map_err(|e| kuno_text::Error::Io(e.kind()))?;
                        }
                    }
                    Ok(())
                },
                &mut |phase| {
                    if phase == "request_private_erasure" {
                        generation_ns =
                            Some(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                    }
                    if phase == "last_prompt_input" {
                        prompt_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    }
                },
            );
            let done = result.map_err(|_| "native_generation_failed")?;
            let total_ns = generation_ns.ok_or("native_generation_failed")?;
            let tail = Zeroizing::new(decoder.finish());
            if !tail.is_empty() {
                wire::write_response(&mut output, wire::TEXT, tail.as_bytes())
                    .map_err(|_| "native_worker_pipe_failed")?;
            }
            let mut receipt = [0u8; 33];
            receipt[..4]
                .copy_from_slice(&(if preload { 0 } else { prompt_count as u32 }).to_le_bytes());
            receipt[4..8]
                .copy_from_slice(&(if preload { 0 } else { done.outputs as u32 }).to_le_bytes());
            receipt[8..16].copy_from_slice(&load_ns.to_le_bytes());
            receipt[16..24].copy_from_slice(&prompt_ns.to_le_bytes());
            receipt[24..32].copy_from_slice(&total_ns.saturating_sub(prompt_ns).to_le_bytes());
            receipt[32] = u8::from(done.eos);
            // DONE is only emitted after the engine's checked request erasure.
            wire::write_response(&mut output, wire::DONE, &receipt)
                .map_err(|_| "native_worker_pipe_failed")?;
            load_ns = 0;
            let next = wire::read_request(&mut input).map_err(|_| "native_request_invalid")?;
            preload = next.as_ref().is_some_and(|r| r.outputs == 0);
            current = next.map(|r| prepare(r, &tokenizer)).transpose()?;
        }
        Ok(())
    })();
    // Checked close even after rejected input or a disconnected output pipe.
    session.close().map_err(|_| "native_recovery_required")?;
    attempted?;
    wire::write_response(&mut output, wire::CLOSED, &[]).map_err(|_| "native_worker_pipe_failed")
}
fn prepare(mut request: wire::Generate, tokenizer: &Qwen35Tokenizer) -> Result<engine::Request> {
    if request.outputs == 0 {
        // Ordinary model preload performs one bounded, fixed public warmup. Its
        // token is discarded and its private owners undergo the same full erase.
        request.prompt = Zeroizing::new(
            b"<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                .to_vec(),
        );
        request.outputs = 1;
    }
    engine::input::request(
        &mut std::io::Cursor::new(request.prompt.as_slice()),
        tokenizer,
        engine::input::Controls {
            outputs: request.outputs as usize,
            eos: tokenizer.eos_token_id(),
            sampling: crate::strata_sampler::Params {
                seed: request.seed,
                temperature: request.temperature,
                greedy: request.temperature == 0.,
                top_k: request.top_k,
                top_p: request.top_p,
                min_p: request.min_p,
                ..Default::default()
            },
            speculation: crate::strata_speculation::Config::native(4),
            minimum: 0.5,
        },
    )
    .map_err(|_| "native_request_invalid")
}
