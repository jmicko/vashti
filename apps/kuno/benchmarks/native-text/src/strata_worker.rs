//! Protected production caller for the selected native engine. All private
//! bytes cross inherited pipes; diagnostics, receipts and profiling are absent.
use crate::{native_protocol as wire, strata_engine as engine};
use cubecl::{
    prelude::*,
    wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device},
};
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, model::Qwen4Exp};
use std::{io::Read, path::Path, sync::atomic::AtomicBool, time::Instant};
use zeroize::Zeroizing;
#[cfg(feature = "strata-deep-matrix")]
#[path = "strata_deep_capability.rs"]
pub(crate) mod deep_capability;
#[path = "../../../src/image_privacy.rs"]
#[allow(dead_code)]
mod protection;
type Result<T> = std::result::Result<T, &'static str>;
use crate::native_artifact::{Settings, verify};
// PUBLIC_CAPACITY_ENTRY_BEGIN
fn compiled_public_capacity_entry(public_check: bool, prefill_check: Option<bool>) -> bool {
    public_check && prefill_check == Some(true)
}
// PUBLIC_CAPACITY_ENTRY_END
#[cfg(feature = "strata-public-host-clock")]
fn public_pipeline_snapshot(phase: &'static str, request: usize) {
    if let Some(totals) = cubecl::wgpu::native_strata_pipeline_creation_profile() {
        eprintln!("public_native_pipeline_creation {}", serde_json::json!({
            "compiled_public_fixture_only":true,"content_saved":false,
            "phase":phase,"request":request,
            "labels":cubecl::wgpu::NATIVE_PIPELINE_CREATION_LABELS,
            "columns":cubecl::wgpu::NATIVE_PIPELINE_CREATION_COLUMNS,
            "cumulative":totals,
            "host_wall_time_only":true,"extra_gpu_fences":false,
            "completion_or_correctness_claim":false,
        }));
    }
}
// Exercise the actual production guard/aggregation helper without a device.
#[cfg(all(test, feature = "strata-public-host-clock"))]
#[path = "../vendor/cubecl-wgpu/src/compute/strata_pipeline_profile.rs"]
mod pipeline_profile_cpu_tests;
/// Called before any runtime thread exists. Inherited experiment configuration
/// cannot enable public diagnostics or alter the installed execution policy.
pub(crate) fn run(path: &Path) -> Result<()> {
    run_mode(path, false, None, false, false, false, false, false, false, false)
}
/// Dedicated laboratory entry: its only request is compiled public text, never
/// stdin. Only this entry reports static engine phase labels for diagnosis.
pub(crate) fn public_long_prompt_check(path: &Path) -> Result<()> {
    run_mode(path, true, None, false, false, false, false, false, false, false)
}
static DENSE_BENCH: AtomicBool = AtomicBool::new(false);
static REGULAR_ONLY: AtomicBool = AtomicBool::new(false);
/// Fixed-public regular-cohort GPU numerics only (subset of the matrix check).
pub(crate) fn public_regular_check(path: &Path) -> Result<()> {
    REGULAR_ONLY.store(true, std::sync::atomic::Ordering::Release);
    run_mode(path, true, None, true, false, false, false, false, false, false)
}
/// Fixed-public dense GEMM benchmark: synthetic data, no model load or requests.
pub(crate) fn public_dense_bench(path: &Path) -> Result<()> {
    DENSE_BENCH.store(true, std::sync::atomic::Ordering::Release);
    run_mode(path, true, None, false, false, false, false, false, false, false)
}
pub(crate) fn public_prefill_check(path: &Path, profile: bool) -> Result<()> {
    run_mode(path, true, Some(profile), false, false, false, false, false, false, false)
}
#[cfg(feature = "strata-public-host-clock")]
pub(crate) fn public_prefill_host_check(path: &Path) -> Result<()> {
    run_mode(path, true, Some(true), false, false, false, false, true, false, false)
}
/// Bounded fixed-public last-request host trace across both chunks and all layers.
#[cfg(feature = "strata-public-host-clock")]
pub(crate) fn public_prefill_host_trace(path: &Path) -> Result<()> {
    run_mode(path, true, Some(true), false, false, false, false, true, true, false)
}


/// Three identical compiled-public short requests exercise cold and two warm
/// prefix paths. Ordinary request erasure and final session close are unchanged.
pub(crate) fn public_prefill_short_check(path: &Path) -> Result<()> {
    run_mode(path, true, Some(false), false, true, false, false, false, false, false)
}
static CODE_FILL: AtomicBool = AtomicBool::new(false);
/// Install-time code cache fill: fixed public prompts covering the windowed
/// short path, batched prefill size buckets and decode, so serving workers can
/// prewarm every recorded kernel. Writes GPU code only, never request content.
pub(crate) fn public_code_cache_fill(path: &Path) -> Result<()> {
    CODE_FILL.store(true, std::sync::atomic::Ordering::Release);
    run_mode(path, true, Some(false), false, false, false, false, false, false, false)
}
fn fill_request(index: usize) -> Option<wire::Generate> {
    let tiny = b"<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_vec();
    let (prompt, outputs) = match index {
        0 => (tiny, 64),
        1 => (public_prompt(0), 64),
        2 => (public_prompt(2), 8),
        3 => (public_prompt(10), 8),
        4 => (public_prompt(30), 8),
        5 => (public_prompt(100), 8),
        6 => (public_prompt(300), 8),
        // A full-length decode (the tiny prompt stops early at EOS).
        7 => (b"<|im_start|>user\nWrite a Python function that merges two sorted lists of integers into one sorted list. Explain its time and space complexity.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_vec(), 200),
        // Every short-window row count (1..=8 rows per window) and their
        // two-window combinations: raw public prompts of 1..=17 words. The
        // first is a one-token prompt following a completed request.
        8..=24 => (" one".repeat(index - 7).into_bytes(), 3),
        _ => return None,
    };
    Some(wire::Generate {
        prompt: Zeroizing::new(prompt),
        outputs,
        context: 32768,
        temperature: 0.,
        seed: 0,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.,
    })
}
pub(crate) fn public_matrix_check(path: &Path) -> Result<()> {
    run_mode(path, true, None, true, false, false, false, false, false, false)
}
/// Fixed-public original expert matrices only; no Session or private input.
pub(crate) fn public_matrix_timing_check(path: &Path) -> Result<()> {
    run_mode(path, true, None, true, false, true, false, false, false, false)
}
/// Explicit compiled-public terminal negative check. It never reads stdin and
/// cannot be selected through KNT2 or the inference API. An actual submitted
/// prefix is interrupted; owning process exit is part of this check's recovery.
pub(crate) fn public_rolling_failure_check(path: &Path) -> Result<()> {
    run_mode(path, true, Some(false), false, false, false, true, false, false, false)
}
#[cfg(feature="strata-host-expert-blobs")]
pub(crate) fn public_host_blob_check(path:&Path)->Result<()>{
    run_mode(path,true,Some(true),false,false,false,false,false,false,true)
}
fn public_prompt(repeats: usize) -> Vec<u8> {
    let mut text = String::from(
        "<|im_start|>system\nYou are a helpful assistant. Reference material follows.\n",
    );
    for _ in 0..repeats {
        text.push_str("A calculator evaluates arithmetic. Notes organize information. Images illustrate ideas.\n");
    }
    text.push_str("<|im_end|>\n<|im_start|>user\nWhat is 17 times 19? Reply with the numerical answer.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    text.into_bytes()
}
fn host_blob_mode_allowed(public:bool,prefill:Option<bool>,matrix:bool,short:bool,timing:bool,rolling:bool,host_clock:bool,full_trace:bool)->bool{
    cfg!(feature="strata-host-expert-blobs") && public && prefill==Some(true) && !matrix && !short && !timing && !rolling && !host_clock && !full_trace
}
enum WgslCache {
    Record(std::path::PathBuf),
    Prewarm(std::path::PathBuf),
}
static WGSL_CACHE: std::sync::OnceLock<WgslCache> = std::sync::OnceLock::new();
/// Private RAM copy of the compiled Mesa shader cache for a serving worker.
/// Holds GPU code only; the directory contents are removed when dropped.
struct RamCodeCache {
    root: std::path::PathBuf,
}
impl RamCodeCache {
    fn create(source: &Path) -> Option<Self> {
        if !source.is_dir() {
            return None;
        }
        let root = std::path::PathBuf::from(format!("/dev/shm/kuno-code-{}", std::process::id()));
        std::fs::create_dir(&root).ok()?;
        let cache = Self { root };
        copy_tree(source, &cache.mesa_dir()).ok()?;
        // Recorded WGSL kernel sources (GPU code only) for load-time prewarm.
        if let Some(wgsl) = source.parent().map(|p| p.join("wgsl")).filter(|p| p.is_dir()) {
            let _ = copy_tree(&wgsl, &cache.wgsl_dir());
        }
        Some(cache)
    }
    fn mesa_dir(&self) -> std::path::PathBuf {
        self.root.join("mesa")
    }
    fn wgsl_dir(&self) -> std::path::PathBuf {
        self.root.join("wgsl")
    }
}
impl Drop for RamCodeCache {
    fn drop(&mut self) {
        // The worker may remove entries beneath its writable root, not the root
        // itself; the empty directory is left for the launcher or reboot.
        let _ = std::fs::remove_dir_all(self.mesa_dir());
        let _ = std::fs::remove_dir_all(self.wgsl_dir());
    }
}
fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
fn run_mode(
    path: &Path,
    public_check: bool,
    prefill_check: Option<bool>,
    matrix_check: bool,
    short_check: bool,
    matrix_timing: bool,
    rolling_failure: bool,
    host_clock_only: bool,
    full_host_trace: bool,
    host_blob_audit: bool,
) -> Result<()> {
    if host_blob_audit && !host_blob_mode_allowed(public_check,prefill_check,matrix_check,short_check,matrix_timing,rolling_failure,host_clock_only,full_host_trace){return Err("public_host_blob_configuration");}
    if full_host_trace && !host_clock_only {return Err("public_full_host_trace_configuration");}
    if rolling_failure
        && (!public_check
            || prefill_check != Some(false)
            || matrix_check
            || short_check
            || matrix_timing)
    {
        return Err("public_rolling_failure_configuration");
    }
    if short_check && (!public_check || prefill_check != Some(false) || matrix_check) {
        return Err("public_short_check_configuration");
    }
    if matrix_timing && (!public_check || !matrix_check || prefill_check.is_some() || short_check) {
        return Err("public_matrix_timing_configuration");
    }
    if host_clock_only
        && (!cfg!(feature = "strata-public-host-clock")
            || !public_check
            || prefill_check != Some(true)
            || matrix_check
            || matrix_timing
            || short_check
            || rolling_failure)
    {
        return Err("public_host_clock_configuration");
    }
    if public_check {
        crate::resident_util::enable_compiled_public_diagnostics();
    }
    // Capture this fixed-public selector before the original inherited-env
    // scrub. Private entry cannot select it, and no diagnostic env survives.
    let qsa_stage_only = cfg!(feature = "strata-qsa-xmx")
        && public_check
        && matrix_check
        && std::env::var("KUNO_LAB_PUBLIC_QSA_XMX_STAGE_ONLY").as_deref() == Ok("1");
    let worker_started = Instant::now();
    // Model identity re-checks (stat) at most every 250 ms on hot paths.
    kuno_text::gguf::set_identity_check_interval(std::time::Duration::from_millis(250));
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
    // Kuno's Mesa build, installed next to the worker, is always preferred.
    if let Some(icd) = crate::claude_driver::bundled_icd() {
        // SAFETY: single-threaded startup; no runtime thread exists yet.
        unsafe { std::env::set_var("VK_DRIVER_FILES", icd) };
    }
    // Compiled-code cache next to the worker: GPU code only, built from model
    // shapes and fixed public prompts, never request content. Mesa's ISA cache,
    // CubeCL's SPIR-V cache and the recorded WGSL kernels live under it. Only
    // fixed-public modes may write it (see the filesystem policy); the install
    // fill (--public-code-cache-fill) creates it.
    let code_cache = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("code-cache")))
        .filter(|dir| {
            if public_check && CODE_FILL.load(std::sync::atomic::Ordering::Acquire) {
                let _ = std::fs::create_dir_all(dir);
            }
            dir.is_absolute() && dir.is_dir()
        });
    if std::env::var_os("KUNO_ZERO_AUDIT").is_some_and(|v| v == "1") {
        crate::private_arena::enable_zero_readback_audit();
    }
    let mut runtime_config = cubecl_runtime::config::CubeClRuntimeConfig::default();
    if let Some(dir) = &code_cache {
        runtime_config.compilation.cache =
            Some(cubecl_runtime::config::cache::CacheConfig::File(dir.join("cubecl")));
        // SAFETY: still single-threaded startup; no runtime thread exists yet.
        unsafe {
            std::env::remove_var("MESA_SHADER_CACHE_DISABLE");
            std::env::set_var("MESA_SHADER_CACHE_DIR", dir.join("mesa"));
            std::env::set_var("MESA_SHADER_CACHE_MAX_SIZE", "4G");
        }
    }
    CubeClRuntimeConfig::set(runtime_config);
    if public_check {
        // Fixed-public diagnostics: report only the static source location.
        std::panic::set_hook(Box::new(|info| {
            let at = info.location().map(|l| format!("{}:{}", l.file(), l.line()));
            eprintln!("native_worker_failed");
            eprintln!("public_panic_location {}", at.unwrap_or_default());
        }));
    } else {
        std::panic::set_hook(Box::new(|_| eprintln!("native_worker_failed")));
    }
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
        if cfg!(feature = "strata-native-push-descriptors") {
            std::env::set_var("KUNO_STRATA_PUSH_DESCRIPTORS", "command-owned");
        }
        std::env::set_var(
            "KUNO_LAB_STATIC_CACHE_BYTES",
            (5u64 * 1024 * 1024 * 1024).to_string(),
        );
    }
    if public_check && (prefill_check == Some(true) || rolling_failure) {
        // SAFETY: still before any runtime thread; only the compiled public
        // diagnostic can select this fixed one-layer query scope. Private entry
        // has already scrubbed inherited profile/authorization variables above.
        unsafe {
            if !rolling_failure {
                std::env::set_var(
                    "KUNO_STRATA_NATIVE_PROFILE",
                    if host_clock_only {
                        "prefill-host-layer"
                    } else {
                        "prefill-layer"
                    },
                );
            }
            std::env::set_var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "1");
        }
        if !rolling_failure {
            crate::strata_prefill_profile::enable_compiled_public();
            if full_host_trace {crate::strata_prefill_profile::enable_full_host_trace();}
        }
    }
    #[cfg(feature = "strata-public-host-clock")]
    let _pipeline_timing = if compiled_public_capacity_entry(public_check, prefill_check)
        && !rolling_failure {
        Some(cubecl::wgpu::begin_native_strata_public_pipeline_profile()
            .ok_or("public_pipeline_profile_configuration")?)
    } else { None };
    let mut settings_file = std::fs::File::open(path).map_err(|_| "native_settings_unavailable")?;
    let trust_identity = crate::native_artifact::trusted_settings(&settings_file);
    let mut raw = Vec::new();
    settings_file
        .by_ref()
        .take(64 * 1024 + 1)
        .read_to_end(&mut raw)
        .map_err(|_| "native_settings_unavailable")?;
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
    // Only the fixed-public check may populate a compiled shader cache; it
    // compiles GPU code from public fixtures and writes no request content.
    // Serving workers keep the fully read-only policy and only read cache hits.
    // Serving workers never write the persistent cache. Mesa's cache needs a
    // writable index, so they get a private RAM copy that is removed on exit.
    let cache_writer = public_check
        && code_cache.as_ref().is_some_and(|dir| {
            use std::os::unix::ffi::OsStrExt;
            let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok();
            path.is_some_and(|p| unsafe { libc::access(p.as_ptr(), libc::W_OK) } == 0)
        });
    let ram_cache = match &code_cache {
        Some(dir) if !cache_writer => RamCodeCache::create(&dir.join("mesa")),
        _ => None,
    };
    if let Some(ram) = &ram_cache {
        // SAFETY: still single-threaded startup; no runtime thread exists yet.
        unsafe { std::env::set_var("MESA_SHADER_CACHE_DIR", ram.mesa_dir()) };
    }
    if cache_writer {
        if let Some(dir) = &code_cache {
            let _ = WGSL_CACHE.set(WgslCache::Record(dir.join("wgsl")));
        }
    } else if let Some(ram) = &ram_cache {
        let _ = WGSL_CACHE.set(WgslCache::Prewarm(ram.wgsl_dir()));
    }
    let writable = if cache_writer {
        code_cache.clone()
    } else {
        ram_cache.as_ref().map(|ram| ram.root.clone())
    };
    protection::restrict_filesystem_with_writable(writable.as_deref())
        .map_err(|_| "native_worker_filesystem_protection_failed")?;
    for artifact in &settings.shards {
        verify(artifact, trust_identity)?;
    }
    verify(&settings.mtp, trust_identity)?;
    cubecl::future::block_on(serve(
        settings,
        worker_started,
        public_check,
        prefill_check,
        matrix_check,
        short_check,
        matrix_timing,
        rolling_failure,
        qsa_stage_only,
        host_blob_audit,
    ))
}
#[allow(clippy::too_many_arguments)] // Explicit internal diagnostic selectors; no wire fields.
async fn serve(
    settings: Settings,
    worker_started: Instant,
    public_check: bool,
    prefill_check: Option<bool>,
    matrix_check: bool,
    short_check: bool,
    matrix_timing: bool,
    rolling_failure: bool,
    qsa_stage_only: bool,
    host_blob_audit: bool,
) -> Result<()> {
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
    crate::claude_driver::record(&adapter.get_info());
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
                < crate::target_lab_build::NATIVE_SHARED_MAIN_BYTES + 512 * 1024 * 1024
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
    // Compile recorded WGSL kernels on idle CPU cores while weights load.
    match WGSL_CACHE.get() {
        Some(WgslCache::Record(dir)) => cubecl::wgpu::set_wgsl_record_dir(dir.clone()),
        Some(WgslCache::Prewarm(dir)) => {
            cubecl::wgpu::prewarm_wgsl(dir, target_device.clone(), 8)
        }
        None => {}
    }
    if public_check && DENSE_BENCH.load(std::sync::atomic::Ordering::Acquire) {
        return crate::claude_dense_bench::run(&client).map_err(|error| {
            eprintln!("public_dense_bench_error {error}");
            "public_dense_bench_failed"
        });
    }
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
    if matrix_check {
        client
            .configure_public_target_request_io_with_backlog(Default::default())
            .map_err(|_| "public_matrix_io_failed")?;
        // Matrix fixtures carry full-capacity PART/GU owners (450 + 22.5 MiB at 4608 rows).
        let mut arena = crate::private_arena::PrivateArena::new_transient_on_configured(
            client.clone(),
            640 * 1024 * 1024,
            std::sync::Arc::new(AtomicBool::new(false)),
        )
        .map_err(|_| "public_matrix_arena_failed")?;
        arena
            .enable_public_owned_zero_audit()
            .map_err(|_| "public_matrix_audit_failed")?;
        #[cfg(feature = "strata-qsa-xmx")]
        if qsa_stage_only {
            let report = crate::strata_qsa::prompt_xmx::public_stage_probe(&mut arena)
                .map_err(|_| "public_qsa_xmx_stage_failed")?;
            arena.close().map_err(|_| "public_matrix_close_failed")?;
            if arena.owned_zero_audit_receipt().is_none() {
                return Err("public_matrix_erasure_unconfirmed");
            }
            eprintln!(
                "public_prefill_qsa_xmx_stage {}",
                serde_json::json!({"report":report,"closed":true,"content_saved":false})
            );
            return Ok(());
        }
        if matrix_timing {
            let cases = crate::strata_matrix::public_timing(&client, &mut arena, &file)
                .map_err(|_| "public_matrix_timing_failed")?;
            arena.close().map_err(|_| "public_matrix_close_failed")?;
            if arena.owned_zero_audit_receipt().is_none() {
                return Err("public_matrix_erasure_unconfirmed");
            }
            eprintln!(
                "public_prefill_matrix_timing {}",
                serde_json::json!({"cases":cases,"closed":true,"content_saved":false})
            );
            return Ok(());
        }
        #[cfg(feature="strata-regular-four-cell")]
        if REGULAR_ONLY.load(std::sync::atomic::Ordering::Acquire) {
            let started = Instant::now();
            let mut checks = crate::strata_matrix::cohorts::public_quad_regular_pairs(&client, &mut arena)
                .map_err(|e| { eprintln!("public_quad_regular_error {e}"); "public_quad_regular_failed" })?;
            #[cfg(feature="strata-large-shared-panels")]
            checks.extend(crate::strata_matrix::cohorts::public_large_shared_pairs(&client, &mut arena)
                .map_err(|e| { eprintln!("public_shared_large_matrix_error {e}"); "public_shared_large_matrix_failed" })?);
            #[cfg(all(feature = "strata-dequant-direct8", feature = "strata-expert-cohorts"))]
            {
                let fused = crate::strata_prompt::experts::cohorts::fused::public_cases(&client, &mut arena, &file)
                    .map_err(|e| { eprintln!("public_small_fused_error {e}"); "public_small_fused_failed" })?;
                eprintln!("public_small_fused {}", serde_json::json!(fused));
            }
            arena.close().map_err(|_| "public_matrix_close_failed")?;
            eprintln!("public_regular_check {}", serde_json::json!({
                "passed": true, "reports": checks.len(), "seconds": started.elapsed().as_secs_f64(),
                "claude_kernel": true,
            }));
            return Ok(());
        }
        #[cfg(feature="strata-large-shared-panels")]
        let shared_large_matrix_checks=crate::strata_matrix::cohorts::public_large_shared_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_shared_large_matrix_error {e}");"public_shared_large_matrix_failed"})?;
        #[cfg(feature="strata-regular-four-cell")]
        let quad_regular_checks=crate::strata_matrix::cohorts::public_quad_regular_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_quad_regular_error {e}");"public_quad_regular_failed"})?;
        // Exercise the new complete producer/reference family first. Every
        // original control still follows; a probe setup failure can be fixed
        // without waiting through the entire original numeric population.
        #[cfg(feature="strata-large-packed-tiles")]
        let packed_large_dequant_checks=crate::strata_prompt::experts::regular::pub_packed_large_dequant(&client,&mut arena,&file)
            .map_err(|e|{eprintln!("public_packed_large_dequant_error {e}");"public_packed_large_dequant_failed"})?;
        #[cfg(feature="strata-large-packed-tiles")]
        let packed_large_activation_checks=crate::strata_prompt::experts::regular::pub_packed_large_activations(&client,&mut arena)
            .map_err(|e|{eprintln!("public_packed_large_activation_error {e}");"public_packed_large_activation_failed"})?;
        #[cfg(feature="strata-large-packed-tiles")]
        let packed_large_matrix_checks=crate::strata_matrix::cohorts::public_large_packed_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_packed_large_matrix_error {e}");"public_packed_large_matrix_failed"})?;
        #[cfg(feature="strata-large-packed-tiles")]
        let packed_layer_gather_checks=crate::strata_prompt::layer_gather::public_packed_producers(&client,&mut arena)
            .map_err(|e|{eprintln!("public_packed_layer_gather_error {e}");"public_packed_layer_gather_failed"})?;
        #[cfg(feature="strata-layer-gather")]
        let layer_gather_checks=crate::strata_prompt::layer_gather::public_producers(&client,&mut arena)
            .map_err(|e|{eprintln!("public_layer_gather_error {e}");"public_layer_gather_failed"})?;
        #[cfg(feature="strata-layer-gather")]
        let layer_activation_checks=crate::strata_prompt::experts::regular::public_layer_activations(&client,&mut arena)
            .map_err(|e|{eprintln!("public_layer_activation_error {e}");"public_layer_activation_failed"})?;
        #[cfg(feature="strata-layer-gather")]
        let layer_small_checks=crate::strata_matrix::cohorts::public_layer_small_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_layer_small_error {e}");"public_layer_small_failed"})?;
        #[cfg(feature="strata-layer-gather")]
        let layer_regular_large_checks=crate::strata_matrix::cohorts::public_layer_regular_large_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_layer_regular_large_error {e}");"public_layer_regular_large_failed"})?;
        #[cfg(feature="strata-large-cohorts")]
        let large_producer_checks=crate::strata_prompt::experts::regular::public_large_producers(&client,&mut arena)
            .map_err(|e|{eprintln!("public_large_cohort_producer_error {e}");"public_large_cohort_producer_failed"})?;
        #[cfg(feature="strata-large-cohorts")]
        let large_dequant_checks=crate::strata_prompt::experts::regular::public_large_multi_dequant(&client,&mut arena,&file)
            .map_err(|e|{eprintln!("public_large_cohort_dequant_error {e}");"public_large_cohort_dequant_failed"})?;
        #[cfg(feature="strata-large-cohorts")]
        let large_matrix_checks=crate::strata_matrix::cohorts::public_large_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_large_cohort_matrix_error {e}");"public_large_cohort_matrix_failed"})?;
        #[cfg(feature="strata-small-dual-cells")]
        let dual_regular_checks=crate::strata_matrix::cohorts::public_dual_regular_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_dual_regular_error {e}");"public_dual_regular_failed"})?;
        #[cfg(feature="strata-small-dual-cells")]
        let dual_small_checks=crate::strata_matrix::cohorts::public_dual_small_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_dual_small_error {e}");"public_dual_small_failed"})?;
        #[cfg(feature="strata-regular-cohorts")]
        let regular_cohort_producer_checks=crate::strata_prompt::experts::regular::public_producers(&client,&mut arena)
            .map_err(|e|{eprintln!("public_regular_cohort_producer_error {e}");"public_regular_cohort_producer_failed"})?;
        #[cfg(feature="strata-regular-cohorts")]
        eprintln!("public_regular_cohort_control_complete family=producer cases={}",regular_cohort_producer_checks.len());
        #[cfg(feature="strata-regular-cohorts")]
        let regular_cohort_dequant_checks=crate::strata_prompt::experts::regular::public_multi_dequant(&client,&mut arena,&file)
            .map_err(|e|{eprintln!("public_regular_cohort_dequant_error {e}");"public_regular_cohort_dequant_failed"})?;
        #[cfg(feature="strata-regular-cohorts")]
        eprintln!("public_regular_cohort_control_complete family=dequant cases={}",regular_cohort_dequant_checks.len());
        #[cfg(feature="strata-regular-cohorts")]
        let regular_cohort_matrix_checks=crate::strata_matrix::cohorts::public_regular_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_regular_cohort_matrix_error {e}");"public_regular_cohort_matrix_failed"})?;
        #[cfg(feature="strata-regular-cohorts")]
        eprintln!("public_regular_cohort_control_complete family=matrix cases={}",regular_cohort_matrix_checks.len());
        #[cfg(feature = "strata-native-prepared")]
        let native_prepared_checks = crate::strata_native_prepared_probe::cases(&target_device)
            .map_err(|_| "public_native_prepared_admission_failed")?;
        #[cfg(not(feature = "strata-native-prepared"))]
        let native_prepared_checks: Vec<serde_json::Value> = Vec::new();
        #[cfg(feature="strata-small-granular")]
        let granular_checks=crate::strata_matrix::granular_small::public_numeric(&client,&mut arena)
            .map_err(|e|{eprintln!("public_small_granular_error {e}");"public_small_granular_failed"})?;
        #[cfg(not(feature="strata-small-granular"))]
        let granular_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-granular")]
        let granular_pair_checks=crate::strata_matrix::cohorts::public_granular_pairs(&client,&mut arena)
            .map_err(|e|{eprintln!("public_small_granular_pair_error {e}");"public_small_granular_pair_failed"})?;
        #[cfg(not(feature="strata-small-granular"))]
        let granular_pair_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-local-carry")]
        let small_local_carry_checks=crate::strata_matrix::local_small::public_carry_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_small_local_carry_error {error}");"public_small_local_carry_failed"})?;
        #[cfg(not(feature="strata-small-local-carry"))]
        let small_local_carry_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-local-carry")]
        let small_local_carry_pair_checks=crate::strata_matrix::cohorts::public_small_local_carry_pairs(&client,&mut arena)
            .map_err(|error|{eprintln!("public_small_local_carry_pair_error {error}");"public_small_local_carry_pair_failed"})?;
        #[cfg(not(feature="strata-small-local-carry"))]
        let small_local_carry_pair_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-local-prefetch")]
        let local_small_checks=crate::strata_matrix::local_small::public_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_local_small_error {error}");"public_local_small_failed"})?;
        #[cfg(not(feature="strata-local-prefetch"))]
        let local_small_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-local-prefetch")]
        let local_small_pair_checks=crate::strata_matrix::cohorts::public_local_small_pairs(&client,&mut arena)
            .map_err(|error|{eprintln!("public_balanced_pair_error {error}");"public_balanced_pair_failed"})?;
        #[cfg(not(feature="strata-local-prefetch"))]
        let local_small_pair_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-balanced-global")]
        let balanced_global_checks=crate::strata_matrix::balanced_global::public_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_balanced_global_error {error}");"public_balanced_global_failed"})?;
        #[cfg(not(feature="strata-balanced-global"))]
        let balanced_global_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-balanced-global")]
        let balanced_global_pair_checks=crate::strata_matrix::cohorts::public_balanced_pairs(&client,&mut arena)
            .map_err(|error|{eprintln!("public_balanced_pair_error {error}");"public_balanced_pair_failed"})?;
        #[cfg(not(feature="strata-balanced-global"))]
        let balanced_global_pair_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-padded-global")]
        let padded_global_checks=crate::strata_matrix::padded_global::public_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_padded_global_error {error}");"public_padded_global_failed"})?;
        #[cfg(not(feature="strata-padded-global"))]
        let padded_global_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-padded-global")]
        let medium_padding_checks=crate::strata_prompt::experts::public_medium_padding(&client,&mut arena)
            .map_err(|error|{eprintln!("public_medium_padding_error {error}");"public_medium_padding_failed"})?;
        #[cfg(not(feature="strata-padded-global"))]
        let medium_padding_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-full-walk")]
        let full_walk_checks=crate::strata_matrix::small_full_walk::public_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_full_walk_error {error}");"public_full_walk_failed"})?;
        #[cfg(not(feature="strata-small-full-walk"))]
        let full_walk_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-full-walk")]
        let full_walk_pair_checks=crate::strata_matrix::cohorts::public_full_walk_pairs(&client,&mut arena)
            .map_err(|error|{eprintln!("public_full_walk_pair_error {error}");"public_full_walk_pair_failed"})?;
        #[cfg(not(feature="strata-small-full-walk"))]
        let full_walk_pair_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-global-small-products")]
        let global_small_checks=crate::strata_matrix::global_small::public_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_global_small_error {error}");"public_global_small_failed"})?;
        #[cfg(not(feature="strata-global-small-products"))]
        let global_small_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-global-small-products")]
        let global_small_pair_checks=crate::strata_matrix::cohorts::public_global_small_pairs(&client,&mut arena)
            .map_err(|error|{eprintln!("public_global_small_pair_error {error}");"public_global_small_pair_failed"})?;
        #[cfg(not(feature="strata-global-small-products"))]
        let global_small_pair_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-local-prefetch")]
        let local_wide32_expert_checks=crate::strata_matrix::public_local_wide32_expert_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_local_wide32_expert_error {error}");"public_local_wide32_expert_failed"})?;
        #[cfg(not(feature="strata-local-prefetch"))]
        let local_wide32_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-local-prefetch")]
        let local_wide32_checks=if crate::strata_matrix::public_local_wide32_available(&client){crate::strata_matrix::public_local_wide32_numeric(&client,&mut arena,&file).map_err(|error|{eprintln!("public_local_wide32_error {error}");"public_local_wide32_failed"})?}else{Vec::new()};
        #[cfg(feature="strata-local-prefetch")]
        let local_wide32_capabilities=serde_json::json!({"compiled":true,"available":crate::strata_matrix::public_local_wide32_available(&client),"required_shared_bytes":0,"original_backend_shared_gate":49152,"device_shared_bytes":client.properties().hardware.max_shared_memory_size,"workgroup":256,"subgroup":16,"expected_cases":21,"executed_cases":local_wide32_checks.len()});
        #[cfg(not(feature="strata-local-prefetch"))]
        let local_wide32_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(not(feature="strata-local-prefetch"))]
        let local_wide32_capabilities=serde_json::json!({"compiled":false,"available":false,"expected_cases":0,"executed_cases":0});
        #[cfg(feature="strata-balanced-wide32")]
        let balanced_wide32_expert_checks=crate::strata_matrix::public_balanced_wide32_expert_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_balanced_wide32_expert_error {error}");"public_balanced_wide32_expert_failed"})?;
        #[cfg(not(feature="strata-balanced-wide32"))]
        let balanced_wide32_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-balanced-wide32")]
        let balanced_wide32_checks=if crate::strata_matrix::public_balanced_wide32_available(&client){crate::strata_matrix::public_balanced_wide32_numeric(&client,&mut arena,&file).map_err(|error|{eprintln!("public_balanced_wide32_error {error}");"public_balanced_wide32_failed"})?}else{Vec::new()};
        #[cfg(feature="strata-balanced-wide32")]
        let balanced_wide32_capabilities=serde_json::json!({"compiled":true,"available":crate::strata_matrix::public_balanced_wide32_available(&client),"required_shared_bytes":0,"original_backend_shared_gate":49152,"device_shared_bytes":client.properties().hardware.max_shared_memory_size,"workgroup":512,"subgroup":16,"expected_cases":21,"executed_cases":balanced_wide32_checks.len()});
        #[cfg(not(feature="strata-balanced-wide32"))]
        let balanced_wide32_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(not(feature="strata-balanced-wide32"))]
        let balanced_wide32_capabilities=serde_json::json!({"compiled":false,"available":false,"expected_cases":0,"executed_cases":0});
        #[cfg(feature="strata-global-wide32")]
        let global_wide32_expert_checks=crate::strata_matrix::public_global_wide32_expert_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_global_wide32_expert_error {error}");"public_global_wide32_expert_failed"})?;
        #[cfg(not(feature="strata-global-wide32"))]
        let global_wide32_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-global-wide32")]
        let global_wide32_checks=if crate::strata_matrix::public_global_wide32_available(&client){crate::strata_matrix::public_global_wide32_numeric(&client,&mut arena,&file).map_err(|error|{eprintln!("public_global_wide32_error {error}");"public_global_wide32_failed"})?}else{Vec::new()};
        #[cfg(feature="strata-global-wide32")]
        let global_wide32_capabilities=serde_json::json!({"compiled":true,"available":crate::strata_matrix::public_global_wide32_available(&client),"required_shared_bytes":0,"original_backend_shared_gate":49152,"device_shared_bytes":client.properties().hardware.max_shared_memory_size,"workgroup":256,"subgroup":16,"expected_cases":21,"executed_cases":global_wide32_checks.len()});
        #[cfg(not(feature="strata-global-wide32"))]
        let global_wide32_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(not(feature="strata-global-wide32"))]
        let global_wide32_capabilities=serde_json::json!({"compiled":false,"available":false,"expected_cases":0,"executed_cases":0});
        #[cfg(feature="strata-global-transpose")]
        let global_transpose_expert_checks=crate::strata_matrix::public_global_transpose_expert_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_global_transpose_expert_error {error}");"public_global_transpose_expert_failed"})?;
        #[cfg(not(feature="strata-global-transpose"))]
        let global_transpose_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-global-transpose")]
        let global_transpose_checks=if crate::strata_matrix::public_global_transpose_available(&client){crate::strata_matrix::public_global_transpose_numeric(&client,&mut arena,&file).map_err(|error|{eprintln!("public_global_transpose_error {error}");"public_global_transpose_failed"})?}else{Vec::new()};
        #[cfg(feature="strata-global-transpose")]
        let global_transpose_capabilities=serde_json::json!({"compiled":true,"available":crate::strata_matrix::public_global_transpose_available(&client),"required_shared_bytes":0,"original_backend_shared_gate":49152,"device_shared_bytes":client.properties().hardware.max_shared_memory_size,"workgroup":512,"subgroup":16,"expected_cases":21,"executed_cases":global_transpose_checks.len()});
        #[cfg(not(feature="strata-global-transpose"))]
        let global_transpose_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(not(feature="strata-global-transpose"))]
        let global_transpose_capabilities=serde_json::json!({"compiled":false,"available":false,"expected_cases":0,"executed_cases":0});
        #[cfg(feature="strata-global-register-n64")]
        let global_n64_expert_checks=crate::strata_matrix::public_global_n64_expert_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_global_n64_expert_error {error}");"public_global_n64_expert_failed"})?;
        #[cfg(not(feature="strata-global-register-n64"))]
        let global_n64_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-global-register-n64")]
        let global_n64_checks=if crate::strata_matrix::public_global_n64_available(&client){crate::strata_matrix::public_global_n64_numeric(&client,&mut arena,&file).map_err(|error|{eprintln!("public_global_n64_error {error}");"public_global_n64_failed"})?}else{Vec::new()};
        #[cfg(feature="strata-global-register-n64")]
        let global_n64_capabilities=serde_json::json!({"compiled":true,"available":crate::strata_matrix::public_global_n64_available(&client),"required_shared_bytes":0,"original_backend_shared_gate":49152,"device_shared_bytes":client.properties().hardware.max_shared_memory_size,"workgroup":512,"subgroup":16,"expected_cases":21,"executed_cases":global_n64_checks.len()});
        #[cfg(not(feature="strata-global-register-n64"))]
        let global_n64_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(not(feature="strata-global-register-n64"))]
        let global_n64_capabilities=serde_json::json!({"compiled":false,"available":false,"expected_cases":0,"executed_cases":0});
        #[cfg(feature="strata-global-image-experts")]
        let global_image_expert_checks=crate::strata_matrix::public_global_image_expert_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_global_image_expert_error {error}");"public_global_image_expert_failed"})?;
        #[cfg(not(feature="strata-global-image-experts"))]
        let global_image_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-quantized-matrix")]
        let small_quant_half_checks=crate::strata_matrix::quant_cohort::half_probe::cases(&client,&mut arena)
            .map_err(|error|{eprintln!("public_small_quant_half_error {error}");"public_small_quant_half_failed"})?;
        #[cfg(not(feature="strata-small-quantized-matrix"))]
        let small_quant_half_checks:Vec<serde_json::Value>=Vec::new();
        let cases = crate::strata_matrix::public_numeric(&client, &mut arena, &file)
            .map_err(|error| { eprintln!("public_native_matrix_numeric_error {error}"); "public_matrix_numeric_failed" })?;
        // The immutable original operator harness still sees exactly its
        // original36 cases. Additional actual GR-tail controls have a separate
        // population and must be checked by the successor's outer aggregate.
        let (gr_wide_tail_checks, cases): (Vec<_>, Vec<_>) = cases
            .into_iter()
            .partition(|r| r.get("gr_wide_tail_case").and_then(serde_json::Value::as_bool) == Some(true));
        #[cfg(feature="strata-tiny-output")]
        let tiny_output_checks=crate::strata_matrix::public_tiny_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_tiny_output_error {error}");"public_tiny_output_failed"})?;
        #[cfg(not(feature="strata-tiny-output"))]
        let tiny_output_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-dense-direct-store")]
        let dense_direct_checks=crate::strata_matrix::public_direct_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_dense_direct_error {error}");"public_dense_direct_failed"})?;
        #[cfg(not(feature="strata-dense-direct-store"))]
        let dense_direct_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-dense-slm40")]
        let dense_slm40_checks=crate::strata_matrix::public_slm40_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_dense_slm40_error {error}");"public_dense_slm40_failed"})?;
        #[cfg(not(feature="strata-dense-slm40"))]
        let dense_slm40_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-dense-prefix")]
        let dense_prefix_checks=crate::strata_matrix::public_prefix_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_dense_prefix_error {error}");"public_dense_prefix_failed"})?;
        #[cfg(not(feature="strata-dense-prefix"))]
        let dense_prefix_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-gr-register-norm")]
        let gr_register_norm_checks=crate::strata_prompt::register_norm::public_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_gr_register_norm_error {error}");"public_gr_register_norm_failed"})?;
        #[cfg(not(feature="strata-gr-register-norm"))]
        let gr_register_norm_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-deep-matrix")]
        let actual_deep_capability=deep_capability::Capability::actual(&client);
        #[cfg(feature="strata-deep-matrix")]
        let deep_matrix_capabilities=actual_deep_capability.report();
        #[cfg(not(feature="strata-deep-matrix"))]
        let deep_matrix_capabilities=serde_json::json!({"compiled":false});
        #[cfg(feature="strata-deep-matrix")]
        let deep_matrix_unavailable_checks:Vec<serde_json::Value>=actual_deep_capability.unavailable_dense().into_iter().collect();
        #[cfg(not(feature="strata-deep-matrix"))]
        let deep_matrix_unavailable_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-deep-matrix")]
        let deep_dense_checks=if actual_deep_capability.dense_selected() {
            let checks=crate::strata_matrix::public_deep_dense_numeric(&client,&mut arena,&file)
                .map_err(|error|{eprintln!("public_deep_dense_error {error}");"public_deep_dense_failed"})?;
            if checks.len()!=deep_capability::DEEP_DENSE_CASES {
                return Err("public_deep_dense_case_population");
            }
            checks
        } else {
            // This is unavailable GPU coverage, not a successful empty suite.
            // Every original control below still runs through its real fallback.
            Vec::new()
        };
        #[cfg(not(feature="strata-deep-matrix"))]
        let deep_dense_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-deep-matrix")]
        let asymmetric_dense_checks=if crate::strata_matrix::public_asymmetric_available(&client) {
            let checks=crate::strata_matrix::public_asymmetric_dense_numeric(&client,&mut arena,&file)
                .map_err(|error|{eprintln!("public_asymmetric_dense_error {error}");"public_asymmetric_dense_failed"})?;
            if checks.len()!=15{return Err("public_asymmetric_dense_case_population");} checks
        } else { Vec::new() };
        #[cfg(feature="strata-deep-matrix")]
        let asymmetric_dense_capabilities=serde_json::json!({"compiled":true,"available":crate::strata_matrix::public_asymmetric_available(&client),"required_shared_bytes":49152,"device_shared_bytes":client.properties().hardware.max_shared_memory_size,"workgroup":512,"subgroup":16,"expected_cases":15,"executed_cases":asymmetric_dense_checks.len()});
        #[cfg(not(feature="strata-deep-matrix"))]
        let asymmetric_dense_capabilities=serde_json::json!({"compiled":false,"available":false,"expected_cases":15,"executed_cases":0});
        #[cfg(not(feature="strata-deep-matrix"))]
        let asymmetric_dense_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-deep-matrix")]
        let global_register_checks=if crate::strata_matrix::public_global_register_available(&client) {
            let checks=crate::strata_matrix::public_global_register_numeric(&client,&mut arena,&file)
                .map_err(|error|{eprintln!("public_global_register_error {error}");"public_global_register_failed"})?;
            if checks.len()!=21{return Err("public_global_register_case_population");} checks
        } else {Vec::new()};
        #[cfg(feature="strata-deep-matrix")]
        let global_register_capabilities=serde_json::json!({"compiled":true,"available":crate::strata_matrix::public_global_register_available(&client),"required_shared_bytes":0,"original_backend_shared_gate":49152,"device_shared_bytes":client.properties().hardware.max_shared_memory_size,"workgroup":512,"subgroup":16,"expected_cases":21,"executed_cases":global_register_checks.len()});
        #[cfg(not(feature="strata-deep-matrix"))]
        let global_register_capabilities=serde_json::json!({"compiled":false,"available":false,"expected_cases":21,"executed_cases":0});
        #[cfg(not(feature="strata-deep-matrix"))]
        let global_register_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-hc-down-inject")]
        let hc_down_inject_checks=crate::strata_matrix::public_hc_down_inject_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_hc_down_inject_error {error}");"public_hc_down_inject_failed"})?;
        #[cfg(not(feature="strata-hc-down-inject"))]
        let hc_down_inject_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-expert-ordered-output")]
        let ordered_expert_checks=crate::strata_matrix::public_ordered_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_ordered_expert_error {error}");"public_ordered_expert_failed"})?;
        #[cfg(not(feature="strata-expert-ordered-output"))]
        let ordered_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-expert-ordered-output")]
        let ordered_prompt_checks=crate::strata_global_moe::public_prompt_ordered(&client,&mut arena)
            .map_err(|error|{eprintln!("public_ordered_prompt_error {error}");"public_ordered_prompt_failed"})?;
        #[cfg(not(feature="strata-expert-ordered-output"))]
        let ordered_prompt_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-cohort-row-order")]
        let row_order_checks=crate::strata_global_moe::public_prompt_row_order(&client,&mut arena)
            .map_err(|error|{eprintln!("public_row_order_error {error}");"public_row_order_failed"})?;
        #[cfg(not(feature="strata-cohort-row-order"))]
        let row_order_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-wide-shared-reuse")]
        let shared_reuse_checks=crate::strata_matrix::public_shared_reuse_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_shared_reuse_error {error}");"public_shared_reuse_failed"})?;
        #[cfg(not(feature="strata-wide-shared-reuse"))]
        let shared_reuse_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-direct-publish")]
        let small_direct_checks=crate::strata_matrix::public_small_direct_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_small_direct_error {error}");"public_small_direct_failed"})?;
        #[cfg(not(feature="strata-small-direct-publish"))]
        let small_direct_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-direct-publish")]
        let small_direct_cohort_checks=crate::strata_matrix::cohorts::small_direct::public_numeric(&client,&mut arena)
            .map_err(|error|{eprintln!("public_small_direct_cohort_error {error}");"public_small_direct_cohort_failed"})?;
        #[cfg(not(feature="strata-small-direct-publish"))]
        let small_direct_cohort_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-quantized-matrix")]
        let small_quant_compact_checks=crate::strata_prompt::experts::cohorts::compact::public_cases(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_small_quant_compact_error {error}");"public_small_quant_compact_failed"})?;
        #[cfg(not(feature="strata-small-quantized-matrix"))]
        let small_quant_compact_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-quantized-matrix")]
        let small_quant_matrix_checks=crate::strata_matrix::quant_cohort::public_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_small_quant_matrix_error {error}");"public_small_quant_matrix_failed"})?;
        #[cfg(not(feature="strata-small-quantized-matrix"))]
        let small_quant_matrix_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-small-quantized-matrix")]
        let small_quant_pipeline_checks=crate::strata_matrix::quant_cohort::public_pipeline_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_small_quant_pipeline_error {error}");"public_small_quant_pipeline_failed"})?;
        #[cfg(not(feature="strata-small-quantized-matrix"))]
        let small_quant_pipeline_checks:Vec<serde_json::Value>=Vec::new();
        let mut operator_checks =
            crate::strata_global_moe::public_prompt_numeric(&client, &mut arena)
                .map_err(|_| "public_prompt_route_numeric_failed")?;
        operator_checks.extend(
            crate::strata_prompt::experts::public_dequant_numeric(&client, &mut arena, &file)
                .map_err(|_| "public_prompt_dequant_numeric_failed")?,
        );
        #[cfg(feature = "strata-expert-scratch-ring")]
        let expert_scratch_image_checks =
            crate::strata_prompt::experts::public_scratch_dequant_numeric(
                &client, &mut arena, &file,
            )
            .map_err(|_| "public_scratch_dequant_numeric_failed")?;
        #[cfg(not(feature = "strata-expert-scratch-ring"))]
        let expert_scratch_image_checks: Vec<serde_json::Value> = Vec::new();
        #[cfg(feature = "strata-narrow-global")]
        let narrow_padding_checks = crate::strata_prompt::experts::public_narrow_padding(&client,&mut arena)
            .map_err(|e| {eprintln!("public_narrow_padding_error {e}");"public_narrow_padding_failed"})?;
        #[cfg(not(feature = "strata-narrow-global"))]
        let narrow_padding_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature = "strata-narrow-global")]
        let narrow_expert_checks = crate::strata_matrix::public_native_narrow_numeric(&client,&mut arena)
            .map_err(|e| {eprintln!("public_narrow_expert_error {e}");"public_narrow_expert_failed"})?;
        #[cfg(not(feature = "strata-narrow-global"))]
        let narrow_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-expert-cohorts")]
        let cohort_producer_checks=crate::strata_prompt::experts::cohorts::public_producers(&client,&mut arena).map_err(|e|{eprintln!("public_cohort_producer_error {e}");"public_cohort_producer_failed"})?;
        #[cfg(not(feature="strata-expert-cohorts"))]
        let cohort_producer_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-expert-cohorts")]
        let cohort_dequant_checks=crate::strata_prompt::experts::public_cohort_dequant_numeric(&client,&mut arena,&file).map_err(|e|{eprintln!("public_cohort_dequant_error {e}");"public_cohort_dequant_failed"})?;
        #[cfg(not(feature="strata-expert-cohorts"))]
        let cohort_dequant_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-expert-cohorts")]
        let cohort_matrix_checks=crate::strata_matrix::cohorts::public_numeric(&client,&mut arena).map_err(|e|{eprintln!("public_cohort_matrix_error {e}");"public_cohort_matrix_failed"})?;
        #[cfg(not(feature="strata-expert-cohorts"))]
        let cohort_matrix_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-expert-cohorts")]
        let cohort_matrix_pair_checks=crate::strata_matrix::cohorts::public_pairs(&client,&mut arena).map_err(|e|{eprintln!("public_cohort_matrix_pair_error {e}");"public_cohort_matrix_pair_failed"})?;
        #[cfg(not(feature="strata-expert-cohorts"))]
        let cohort_matrix_pair_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-expert-cohorts")]
        let cohort_multi_dequant_checks=crate::strata_prompt::experts::cohorts::public_multi_dequant(&client,&mut arena,&file).map_err(|e|{eprintln!("public_cohort_multi_dequant_error {e}");"public_cohort_multi_dequant_failed"})?;
        #[cfg(not(feature="strata-expert-cohorts"))]
        let cohort_multi_dequant_checks:Vec<serde_json::Value>=Vec::new();
        let attention_checks =
            crate::strata_qsa::selection::public_numeric(&mut arena).map_err(|error| {
                // This branch accepts only compiled public diagnostic inputs.
                eprintln!("public_attention_numeric_error {error}");
                "public_prompt_attention_numeric_failed"
            })?;
        let gdn_math_checks = crate::strata_gdn::prompt::public_recurrence_numeric(&mut arena)
            .map_err(|error| {
                eprintln!("public_gdn_math_numeric_error {error}");
                "public_prompt_gdn_math_numeric_failed"
            })?;
        #[cfg(feature = "strata-cubek-gemm")]
        let expert_matrix_checks = crate::strata_matrix::public_expert_numeric(&client, &mut arena)
            .map_err(|error| {
                eprintln!("public_expert_matrix_numeric_error {error}");
                "public_prompt_expert_matrix_numeric_failed"
            })?;
        #[cfg(not(feature = "strata-cubek-gemm"))]
        let expert_matrix_checks: Vec<serde_json::Value> = Vec::new();
        #[cfg(feature = "strata-native-expert-wide")]
        let native_expert_matrix_checks = crate::strata_matrix::public_native_expert_numeric(
            &client, &mut arena,
        )
        .map_err(|error| {
            eprintln!("public_native_expert_matrix_numeric_error {error}");
            "public_native_expert_matrix_numeric_failed"
        })?;
        #[cfg(not(feature = "strata-native-expert-wide"))]
        let native_expert_matrix_checks: Vec<serde_json::Value> = Vec::new();
        #[cfg(feature="strata-expert-macro64")]
        let macro64_expert_checks = crate::strata_matrix::public_native_macro64_numeric(&client,&mut arena)
            .map_err(|error| {eprintln!("public_native_macro64_numeric_error {error}");"public_native_macro64_numeric_failed"})?;
        #[cfg(not(feature="strata-expert-macro64"))]
        let macro64_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-native-expert-medium")]
        let medium_expert_checks = crate::strata_matrix::public_native_medium_numeric(&client,&mut arena)
            .map_err(|error| {
                eprintln!("public_native_medium_numeric_error {error}");
                "public_native_medium_numeric_failed"
            })?;
        #[cfg(not(feature="strata-native-expert-medium"))]
        let medium_expert_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature = "strata-native-direct-gu")]
        let direct_gu_checks = crate::strata_matrix::public_direct_gu_numeric(&client, &mut arena)
            .map_err(|error| {
                eprintln!("public_direct_gu_numeric_error {error}");
                "public_direct_gu_numeric_failed"
            })?;
        #[cfg(not(feature = "strata-native-direct-gu"))]
        let direct_gu_checks: Vec<serde_json::Value> = Vec::new();
        #[cfg(feature="strata-native-expert-medium")]
        let medium_direct_gu_checks = crate::strata_matrix::public_medium_direct_gu_numeric(&client,&mut arena)
            .map_err(|error| {
                eprintln!("public_native_medium_direct_gu_error {error}");
                "public_native_medium_direct_gu_failed"
            })?;
        #[cfg(not(feature="strata-native-expert-medium"))]
        let medium_direct_gu_checks:Vec<serde_json::Value>=Vec::new();
        let qsa_math_checks =
            crate::strata_qsa::public_attention_numeric(&mut arena).map_err(|error| {
                eprintln!("public_attention_math_numeric_error {error}");
                "public_prompt_attention_math_numeric_failed"
            })?;
        eprintln!(
            "public_attention_math_controls_complete {}",
            qsa_math_checks.len()
        );
        #[cfg(any(feature = "strata-qsa-packed-cache", feature = "strata-qsa-scale-finite"))]
        let packed_cache_checks = crate::strata_qsa::public_packed_cache_numeric(&mut arena)
            .map_err(|error| {
                eprintln!("public_packed_cache_numeric_error {error}");
                "public_packed_cache_numeric_failed"
            })?;
        #[cfg(not(any(feature = "strata-qsa-packed-cache", feature = "strata-qsa-scale-finite")))]
        let packed_cache_checks: Vec<serde_json::Value> = Vec::new();
        #[cfg(feature = "strata-qsa-xmx")]
        let qsa_xmx_checks =
            crate::strata_qsa::public_attention_xmx_numeric(&mut arena).map_err(|error| {
                eprintln!("public_attention_xmx_numeric_error {error}");
                "public_prompt_attention_xmx_failed"
            })?;
        #[cfg(not(feature = "strata-qsa-xmx"))]
        let qsa_xmx_checks: Vec<serde_json::Value> = Vec::new();
        arena.close().map_err(|_| "public_matrix_close_failed")?;
        if arena.owned_zero_audit_receipt().is_none() {
            return Err("public_matrix_erasure_unconfirmed");
        }
        eprintln!(
            "public_prefill_matrix {}",
            {let mut report=serde_json::json!({"small_local_carry_selected":cfg!(feature="strata-small-local-carry"),"small_local_carry_checks":small_local_carry_checks,"small_local_carry_pair_checks":small_local_carry_pair_checks,"local_prefetch_selected":cfg!(feature="strata-local-prefetch"),"local_wide32_checks":local_wide32_checks,"local_wide32_capabilities":local_wide32_capabilities,"local_wide32_expert_checks":local_wide32_expert_checks,"local_small_checks":local_small_checks,"local_small_pair_checks":local_small_pair_checks,"balanced_wide32_selected":cfg!(feature="strata-balanced-wide32"),"balanced_wide32_checks":balanced_wide32_checks,"balanced_wide32_capabilities":balanced_wide32_capabilities,"balanced_wide32_expert_checks":balanced_wide32_expert_checks,"row_order_selected":cfg!(feature="strata-cohort-row-order"),"row_order_checks":row_order_checks,"balanced_carry_selected":cfg!(feature="strata-balanced-carry"),"balanced_preload_selected":cfg!(feature="strata-balanced-preload"),"cases":cases,"gr_wide_tail_checks":gr_wide_tail_checks,"tiny_output_checks":tiny_output_checks,"dense_direct_checks":dense_direct_checks,"dense_slm40_checks":dense_slm40_checks,"dense_prefix_checks":dense_prefix_checks,"gr_register_norm_checks":gr_register_norm_checks,"hc_down_inject_checks":hc_down_inject_checks,"deep_dense_checks":deep_dense_checks,"asymmetric_dense_checks":asymmetric_dense_checks,"asymmetric_dense_capabilities":asymmetric_dense_capabilities,"global_register_checks":global_register_checks,"global_image_expert_checks":global_image_expert_checks,"balanced_global_checks":balanced_global_checks,"balanced_global_pair_checks":balanced_global_pair_checks,"padded_global_checks":padded_global_checks,"medium_padding_checks":medium_padding_checks,"full_walk_checks":full_walk_checks,"full_walk_pair_checks":full_walk_pair_checks,"global_small_checks":global_small_checks,"global_small_pair_checks":global_small_pair_checks,"global_wide32_checks":global_wide32_checks,"global_wide32_capabilities":global_wide32_capabilities,"global_wide32_expert_checks":global_wide32_expert_checks,"global_transpose_checks":global_transpose_checks,"global_transpose_capabilities":global_transpose_capabilities,"global_transpose_expert_checks":global_transpose_expert_checks,"global_n64_checks":global_n64_checks,"global_n64_capabilities":global_n64_capabilities,"global_n64_expert_checks":global_n64_expert_checks,"global_register_capabilities":global_register_capabilities,"ordered_expert_checks":ordered_expert_checks,"ordered_prompt_checks":ordered_prompt_checks,"shared_reuse_checks":shared_reuse_checks,"small_direct_checks":small_direct_checks,"small_direct_cohort_checks":small_direct_cohort_checks,"small_quant_compact_checks":small_quant_compact_checks,"small_quant_matrix_checks":small_quant_matrix_checks,"small_quant_pipeline_checks":small_quant_pipeline_checks,"small_quant_half_checks":small_quant_half_checks,"deep_matrix_capabilities":deep_matrix_capabilities,"deep_matrix_unavailable_checks":deep_matrix_unavailable_checks,"native_prepared_checks":native_prepared_checks,"operator_checks":operator_checks,"expert_scratch_image_checks":expert_scratch_image_checks,"attention_checks":attention_checks,"qsa_math_checks":qsa_math_checks,"qsa_xmx_checks":qsa_xmx_checks,"packed_cache_checks":packed_cache_checks,"gdn_math_checks":gdn_math_checks,"expert_matrix_checks":expert_matrix_checks,"native_expert_matrix_checks":native_expert_matrix_checks,"direct_gu_checks":direct_gu_checks,"medium_expert_checks":medium_expert_checks,"macro64_expert_checks":macro64_expert_checks,"medium_direct_gu_checks":medium_direct_gu_checks,"narrow_padding_checks":narrow_padding_checks,"narrow_expert_checks":narrow_expert_checks,"cohort_producer_checks":cohort_producer_checks,"cohort_dequant_checks":cohort_dequant_checks,"cohort_matrix_checks":cohort_matrix_checks,"cohort_matrix_pair_checks":cohort_matrix_pair_checks,"cohort_multi_dequant_checks":cohort_multi_dequant_checks,"closed":true,"content_saved":false});
             report["small_granular_selected"]=cfg!(feature="strata-small-granular").into();
             report["granular_checks"]=granular_checks.into();
             report["granular_pair_checks"]=granular_pair_checks.into();
             #[cfg(feature="strata-regular-four-cell")]
             {report["quad_regular_checks"]=quad_regular_checks.into();report["quad_regular_selected"]=true.into();}
             #[cfg(feature="strata-large-shared-panels")]
             {report["shared_large_matrix_checks"]=shared_large_matrix_checks.into();report["shared_large_selected"]=true.into();}
             report["layer_gather_selected"]=cfg!(feature="strata-layer-gather").into();
             #[cfg(feature="strata-layer-gather")]
             {report["layer_gather_checks"]=layer_gather_checks.into();report["layer_activation_checks"]=layer_activation_checks.into();report["layer_small_checks"]=layer_small_checks.into();report["layer_regular_large_checks"]=layer_regular_large_checks.into();}
             report["large_cohorts_selected"]=cfg!(feature="strata-large-cohorts").into();
             #[cfg(feature="strata-large-packed-tiles")]
             {report["packed_layer_gather_checks"]=packed_layer_gather_checks.into();report["packed_large_matrix_checks"]=packed_large_matrix_checks.into();report["packed_large_dequant_checks"]=packed_large_dequant_checks.into();report["packed_large_activation_checks"]=packed_large_activation_checks.into();}
             #[cfg(feature="strata-large-cohorts")] {report["large_cohort_producer_checks"]=large_producer_checks.into();report["large_cohort_matrix_checks"]=large_matrix_checks.into();report["large_cohort_dequant_checks"]=large_dequant_checks.into();}
             report["regular_cohorts_selected"]=cfg!(feature="strata-regular-cohorts").into();
             #[cfg(feature="strata-regular-cohorts")] {
                #[cfg(feature="strata-small-dual-cells")]
                {report["dual_regular_checks"]=dual_regular_checks.into();report["dual_small_checks"]=dual_small_checks.into();}
                report["regular_cohort_producer_checks"]=regular_cohort_producer_checks.into();
                report["regular_cohort_dequant_checks"]=regular_cohort_dequant_checks.into();
                report["regular_cohort_matrix_checks"]=regular_cohort_matrix_checks.into();
             }

             report}

        );
        return Ok(());
    }
    let tokenizer =
        Qwen35Tokenizer::from_gguf(&file, engine::input::tokenizer_limits(192 * 1024 * 1024))
            .map_err(|_| "native_artifact_invalid")?;
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let setup_ns = worker_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    wire::write_response(&mut output, wire::READY, &[]).map_err(|_| "native_worker_pipe_failed")?;
    // Fixed-public quality check: each public text with int8, then f16, small experts.
    let quality = public_check && prefill_check == Some(false) && !short_check && crate::claude_quality::requested();
    let quality_texts = crate::claude_quality::texts();
    let quality_plan: Vec<(usize, bool)> = if quality {
        crate::claude_quality::activate();
        [false, true].iter().flat_map(|&f16| (0..quality_texts.len()).map(move |t| (t, f16))).collect()
    } else {
        Vec::new()
    };
    let quality_request = |i: usize| wire::Generate {
        prompt: Zeroizing::new(quality_texts[quality_plan[i].0].bytes.to_vec()),
        outputs: 1,
        context: 32768,
        temperature: 0.,
        seed: 0,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.,
    };
    let mut quality_rows: Vec<Vec<(usize, u32, Vec<f32>)>> = Vec::new();
    let first = if quality {
        Some((prepare(quality_request(0), &tokenizer)?, false))
    } else if public_check && CODE_FILL.load(std::sync::atomic::Ordering::Acquire) {
        Some((prepare(fill_request(0).ok_or("public_code_fill")?, &tokenizer)?, false))
    } else if public_check {
        let request = wire::Generate {
            // The long-prefix operator profile warms the ordinary prompt path
            // with73 tokens. Its two measured4245-token inputs remain exact.
            // The independent short check keeps its original45-token inputs.
            prompt: Zeroizing::new(public_prompt(if rolling_failure {
                300
            } else {
                match prefill_check {
                    Some(true) => 2,
                    Some(false) => 0,
                    None => 600,
                }
            })),
            outputs: 32,
            context: 32768,
            temperature: 0.,
            seed: 0,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.,
        };
        Some((prepare(request, &tokenizer)?, false))
    } else {
        next_request(&mut input, &mut output, &tokenizer)?
    };
    let Some((request, first_preload)) = first else {
        return Ok(());
    };
    let cancel = AtomicBool::new(false);

    // SAFETY: the private worker has exclusive parent admission, protected
    // process memory and model backing, one physical runtime and original MAIN/
    // HOST bounds. No diagnostics/public receipt path is entered for this caller.
    let model_started = Instant::now();
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
            &mut |phase| {
                if public_check {
                    eprintln!("public_prefill_phase {phase}");
                }
            },
        )
    }
    .map_err(|_| "native_load_failed")?;
    if rolling_failure {
        cubecl::wgpu::arm_native_strata_public_rolling_failure()
            .map_err(|_| "public_rolling_failure_arm_rejected")?;
    }
    let mut load_ns =
        setup_ns.saturating_add(model_started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    #[cfg(feature = "strata-public-host-clock")]
    if compiled_public_capacity_entry(public_check, prefill_check) {
        public_pipeline_snapshot("load_complete", 0);
    }
    let mut current = Some(request);
    let mut preload = first_preload;
    let mut public_index = 0;
    let attempted: Result<()> = (|| {
        if compiled_public_capacity_entry(public_check, prefill_check) {
            let capacity = session
                .compiled_public_capacity()
                .map_err(|_| "public_prefill_capacity_unavailable")?;
            eprintln!("public_prefill_capacity {capacity}");
        }
        if host_blob_audit {
            #[cfg(feature="strata-host-expert-blobs")]
            {
                let audit=session.compiled_public_host_blob_probe().map_err(|_|"public_host_blob_audit_failed")?;
                eprintln!("public_host_blob_audit {audit}");
                return Ok(());
            }
            #[cfg(not(feature="strata-host-expert-blobs"))]
            return Err("public_host_blob_feature_disabled");
        }
        while let Some(request) = current.take() {
            let prompt_count = request.geometry().map_err(|_| "native_request_invalid")?.0;
            #[cfg(feature = "strata-dequant-direct8")]
            if quality {
                let second = quality_plan[quality_rows.len()].1;
                if crate::claude_quality::chunk_mode() {
                    crate::claude_quality::CHUNK_CAP
                        .store(if second { 4096 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
                } else {
                    crate::strata_prompt::experts::cohorts::fused::FORCE_OFF
                        .store(second, std::sync::atomic::Ordering::Relaxed);
                }
            }
            crate::strata_prefill_profile::begin();
            #[cfg(feature = "strata-public-host-clock")]
            if compiled_public_capacity_entry(public_check, prefill_check) {
                public_pipeline_snapshot("request_begin", public_index + 1);
            }
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
                    if public_check {
                        eprintln!("public_prefill_phase {phase}");
                        if prefill_check == Some(true) {
                            eprintln!(
                                "public_prefill_phase_time {phase} {}",
                                start.elapsed().as_nanos()
                            );
                        }
                    }
                    if phase == "request_private_erasure" {
                        generation_ns =
                            Some(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                    }
                    if phase == "last_prompt_input" {
                        prompt_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                        #[cfg(feature = "strata-public-host-clock")]
                        if compiled_public_capacity_entry(public_check, prefill_check) {
                            public_pipeline_snapshot("last_prompt_input", public_index + 1);
                        }
                    }
                },
            );
            #[cfg(feature = "strata-public-host-clock")]
            if compiled_public_capacity_entry(public_check, prefill_check) {
                public_pipeline_snapshot("request_end", public_index + 1);
            }
            if rolling_failure {
                let [fired, waited, failed] =
                    cubecl::wgpu::native_strata_public_rolling_failure_stats();
                let expected = result.is_err() && fired == 1 && waited > 0 && failed == 0;
                eprintln!(
                    "public_rolling_terminal_failure {}",
                    serde_json::json!({
                        "compiled_public_fixture_only":true,"content_saved":false,
                        "expected_terminal_error":expected,"fired":fired,
                        "actual_waited_recovery_queues":waited,"failed_recovery_queues":failed,
                        "ordinary_reusable_session_claimed":false,"owned_zero_claimed":false,
                        "protected_process_exit_required":true
                    })
                );
                return Err(if expected {
                    "public_rolling_failure_expected_terminal_exit"
                } else {
                    "public_rolling_failure_not_exercised"
                });
            }
            let done = result.map_err(|_| "native_generation_failed")?;
            #[cfg(feature = "strata-native-prepared")]
            if public_check {
                eprintln!("public_native_prepared_pipeline {}", serde_json::json!({
                    "compiled_public_fixture_only":true,"content_saved":false,
                    "prepared_ordinary_native_graph_original_graph":
                        cubecl::wgpu::native_strata_prepared_pipeline_stats(),
                    "ordinary_root_initialization_retained":true,
                    "original_core_and_hal_stage_admission_retained":true,
                }));
            }
            if public_check || std::env::var_os("KUNO_CLAUDE_PIPE_TRACE").is_some() {
                let t = cubecl::wgpu::claude_pipeline_time();
                eprintln!("public_pipeline_time server_misses={} server_s={:.3} native_creates={} native_s={:.3} modules={} module_s={:.3} pipelines={} pipeline_s={:.3} prewarm_hits={} prewarm_failures={} wgsl_compiles={} wgsl_s={:.3}", t[0], t[1] as f64 / 1e9, t[2], t[3] as f64 / 1e9, t[4], t[5] as f64 / 1e9, t[6], t[7] as f64 / 1e9, t[8], t[9], t[10], t[11] as f64 / 1e9);
            }
            crate::strata_prefill_profile::report();
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
            // DONE is delivered as soon as generation completes; the request's
            // private state is then erased before the next request is read.
            wire::write_response(&mut output, wire::DONE, &receipt)
                .map_err(|_| "native_worker_pipe_failed")?;
            {
                use std::io::Write;
                output.flush().map_err(|_| "native_worker_pipe_failed")?;
            }
            session
                .erase_completed(&cancel, &mut |phase| {
                    if public_check {
                        eprintln!("public_prefill_phase {phase}");
                    }
                })
                .map_err(|_| "native_request_erasure_failed")?;
            load_ns = 0;
            if quality {
                let mut rows = crate::claude_quality::take();
                rows.sort_by_key(|r| r.0);
                eprintln!("public_quality_request {}", serde_json::json!({"index":quality_rows.len(),"rows":rows.len(),"prompt_tokens":prompt_count}));
                quality_rows.push(rows);
                if quality_rows.len() < quality_plan.len() {
                    current = Some(prepare(quality_request(quality_rows.len()), &tokenizer)?);
                } else {
                    let n = quality_texts.len();
                    // GPU-only comparisons first; the CPU reference is slow.
                    for (t, text) in quality_texts.iter().enumerate() {
                        crate::claude_quality::report(&graph, text.name, &quality_rows[t], &quality_rows[n + t], None)
                            .map_err(|_| "public_quality_report_failed")?;
                    }
                    for (t, text) in quality_texts.iter().enumerate().filter(|(_, x)| x.reference) {
                        let tokens: Vec<u32> = quality_rows[t].iter().map(|r| r.1).take(crate::claude_quality::REFERENCE_TOKENS).collect();
                        let reference = crate::claude_quality::reference(&graph, &tokens).map_err(|_| "public_quality_reference_failed")?;
                        let cut = tokens.len();
                        crate::claude_quality::report(&graph, text.name, &quality_rows[t][..cut], &quality_rows[n + t][..cut], Some(&reference))
                            .map_err(|_| "public_quality_report_failed")?;
                    }
                }
                continue;
            }
            if CODE_FILL.load(std::sync::atomic::Ordering::Acquire) {
                public_index += 1;
                if let Some(request) = fill_request(public_index) {
                    current = Some(prepare(request, &tokenizer)?);
                }
            } else if prefill_check.is_some() && public_index < 2 {
                public_index += 1;
                current = Some(prepare(
                    wire::Generate {
                        prompt: Zeroizing::new(public_prompt(if short_check { 0 } else { 300 })),
                        outputs: 32,
                        context: 32768,
                        temperature: 0.,
                        seed: 0,
                        top_k: 20,
                        top_p: 0.95,
                        min_p: 0.,
                    },
                    &tokenizer,
                )?);
            }
            if !public_check
                && let Some((request, is_preload)) =
                    next_request(&mut input, &mut output, &tokenizer)?
            {
                preload = is_preload;
                current = Some(request);
            }
        }
        Ok(())
    })();
    if rolling_failure {
        // A partial submitted=false flight is intentionally quarantined. The
        // real recovery fences above have joined attempted queues; do not claim
        // reusable-session close/owned-zero. The parent MUST verify process exit
        // and actual physical recovery before accepting this terminal fixture.
        return attempted.and(Err("public_rolling_failure_not_exercised"));
    }
    // Checked close even after rejected input or a disconnected output pipe.
    session.close().map_err(|_| "native_recovery_required")?;
    attempted?;
    wire::write_response(&mut output, wire::CLOSED, &[]).map_err(|_| "native_worker_pipe_failed")
}
fn next_request(
    input: &mut impl std::io::Read,
    output: &mut impl std::io::Write,
    tokenizer: &Qwen35Tokenizer,
) -> Result<Option<(engine::Request, bool)>> {
    loop {
        let Some(request) = wire::read_request(input).map_err(|_| "native_request_invalid")? else {
            return Ok(None);
        };
        let preload = request.outputs == 0;
        match prepare(request, tokenizer) {
            Ok(request) => return Ok(Some((request, preload))),
            Err(code) => wire::write_response(output, wire::REJECTED, code.as_bytes())
                .map_err(|_| "native_worker_pipe_failed")?,
        }
    }
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
    let prepared = engine::input::request(
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
    .map_err(|_| "native_request_invalid")?;
    let (prompt, outputs, _) = prepared.geometry().map_err(|_| "native_request_invalid")?;
    if prompt
        .checked_add(outputs)
        .and_then(|n| n.checked_add(engine::CONTEXT_HEADROOM))
        .is_none_or(|n| n > request.context as usize)
    {
        return Err("native_context_exceeded");
    }
    Ok(prepared)
}

#[cfg(test)]
mod host_blob_entry_tests {
 #[test]
 fn host_blob_probe_only_enters_an_explicit_public_mode_without_private_or_other_diagnostics(){
  for bits in 0..128u32 {for p in [None,Some(false),Some(true)] {
   let b=|i:u32|bits&(1u32<<i)!=0u32;
   let allowed=super::host_blob_mode_allowed(b(0),p,b(1),b(2),b(3),b(4),b(5),b(6));
   assert_eq!(allowed,cfg!(feature="strata-host-expert-blobs") && bits==1 && p==Some(true));
  }}
 }
}
