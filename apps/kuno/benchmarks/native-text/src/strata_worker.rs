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
/// Called before any runtime thread exists. Inherited experiment configuration
/// cannot enable public diagnostics or alter the installed execution policy.
pub(crate) fn run(path: &Path) -> Result<()> {
    run_mode(path, false, None, false, false, false, false, false)
}
/// Dedicated laboratory entry: its only request is compiled public text, never
/// stdin. Only this entry reports static engine phase labels for diagnosis.
pub(crate) fn public_long_prompt_check(path: &Path) -> Result<()> {
    run_mode(path, true, None, false, false, false, false, false)
}
pub(crate) fn public_prefill_check(path: &Path, profile: bool) -> Result<()> {
    run_mode(path, true, Some(profile), false, false, false, false, false)
}
#[cfg(feature = "strata-public-host-clock")]
pub(crate) fn public_prefill_host_check(path: &Path) -> Result<()> {
    run_mode(path, true, Some(true), false, false, false, false, true)
}

/// Three identical compiled-public short requests exercise cold and two warm
/// prefix paths. Ordinary request erasure and final session close are unchanged.
pub(crate) fn public_prefill_short_check(path: &Path) -> Result<()> {
    run_mode(path, true, Some(false), false, true, false, false, false)
}
pub(crate) fn public_matrix_check(path: &Path) -> Result<()> {
    run_mode(path, true, None, true, false, false, false, false)
}
/// Fixed-public original expert matrices only; no Session or private input.
pub(crate) fn public_matrix_timing_check(path: &Path) -> Result<()> {
    run_mode(path, true, None, true, false, true, false, false)
}
/// Explicit compiled-public terminal negative check. It never reads stdin and
/// cannot be selected through KNT2 or the inference API. An actual submitted
/// prefix is interrupted; owning process exit is part of this check's recovery.
pub(crate) fn public_rolling_failure_check(path: &Path) -> Result<()> {
    run_mode(path, true, Some(false), false, false, false, true, false)
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
fn run_mode(
    path: &Path,
    public_check: bool,
    prefill_check: Option<bool>,
    matrix_check: bool,
    short_check: bool,
    matrix_timing: bool,
    rolling_failure: bool,
    host_clock_only: bool,
) -> Result<()> {
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
        }
    }
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
    protection::restrict_filesystem().map_err(|_| "native_worker_filesystem_protection_failed")?;
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
    if matrix_check {
        client
            .configure_public_target_request_io_with_backlog(Default::default())
            .map_err(|_| "public_matrix_io_failed")?;
        let mut arena = crate::private_arena::PrivateArena::new_transient_on_configured(
            client.clone(),
            512 * 1024 * 1024,
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
        #[cfg(feature = "strata-native-prepared")]
        let native_prepared_checks = crate::strata_native_prepared_probe::cases(&target_device)
            .map_err(|_| "public_native_prepared_admission_failed")?;
        #[cfg(not(feature = "strata-native-prepared"))]
        let native_prepared_checks: Vec<serde_json::Value> = Vec::new();
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
        let deep_dense_checks=crate::strata_matrix::public_deep_dense_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_deep_dense_error {error}");"public_deep_dense_failed"})?;
        #[cfg(not(feature="strata-deep-matrix"))]
        let deep_dense_checks:Vec<serde_json::Value>=Vec::new();
        #[cfg(feature="strata-hc-down-inject")]
        let hc_down_inject_checks=crate::strata_matrix::public_hc_down_inject_numeric(&client,&mut arena,&file)
            .map_err(|error|{eprintln!("public_hc_down_inject_error {error}");"public_hc_down_inject_failed"})?;
        #[cfg(not(feature="strata-hc-down-inject"))]
        let hc_down_inject_checks:Vec<serde_json::Value>=Vec::new();
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
            serde_json::json!({"cases":cases,"gr_wide_tail_checks":gr_wide_tail_checks,"tiny_output_checks":tiny_output_checks,"dense_direct_checks":dense_direct_checks,"dense_slm40_checks":dense_slm40_checks,"dense_prefix_checks":dense_prefix_checks,"gr_register_norm_checks":gr_register_norm_checks,"hc_down_inject_checks":hc_down_inject_checks,"deep_dense_checks":deep_dense_checks,"native_prepared_checks":native_prepared_checks,"operator_checks":operator_checks,"expert_scratch_image_checks":expert_scratch_image_checks,"attention_checks":attention_checks,"qsa_math_checks":qsa_math_checks,"qsa_xmx_checks":qsa_xmx_checks,"packed_cache_checks":packed_cache_checks,"gdn_math_checks":gdn_math_checks,"expert_matrix_checks":expert_matrix_checks,"native_expert_matrix_checks":native_expert_matrix_checks,"direct_gu_checks":direct_gu_checks,"medium_expert_checks":medium_expert_checks,"macro64_expert_checks":macro64_expert_checks,"medium_direct_gu_checks":medium_direct_gu_checks,"narrow_padding_checks":narrow_padding_checks,"narrow_expert_checks":narrow_expert_checks,"cohort_producer_checks":cohort_producer_checks,"cohort_dequant_checks":cohort_dequant_checks,"cohort_matrix_checks":cohort_matrix_checks,"cohort_matrix_pair_checks":cohort_matrix_pair_checks,"cohort_multi_dequant_checks":cohort_multi_dequant_checks,"closed":true,"content_saved":false})
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
    let first = if public_check {
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
        while let Some(request) = current.take() {
            let prompt_count = request.geometry().map_err(|_| "native_request_invalid")?.0;
            crate::strata_prefill_profile::begin();
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
                    }
                },
            );
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
            // DONE is only emitted after the engine's checked request erasure.
            wire::write_response(&mut output, wire::DONE, &receipt)
                .map_err(|_| "native_worker_pipe_failed")?;
            load_ns = 0;
            if prefill_check.is_some() && public_index < 2 {
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
