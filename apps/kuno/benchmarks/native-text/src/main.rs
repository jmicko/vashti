//! Public synthetic Rust-authored kernel probe, not a model engine wrapper.
mod bank96_probe;
mod bank_slots;
pub mod cpu_gpu_expert_probe;
mod cpu_gpu_expert_schedule;
mod executor;
mod expert_banks;
mod gdn_batch_pointwise;
mod gdn_batch_pointwise_probe;
mod gdn_chunk;
mod gdn_chunk_probe;
mod gdn_executor;
mod gdn_gpu;
mod gdn_layer_batch;
mod gdn_layer_batch_probe;
mod gdn_pointwise;
mod gr_batch_probe;
mod grouped_moe_chunk;
mod grouped_moe_chunk_probe;
mod info_slab_probe;
mod main_admission;
mod matmul_probe;
mod next_shared;
mod next_shared_moe;
mod ordered_public;
mod ple_batch_gpu;
mod ple_chunk;
mod ple_chunk_probe;
mod private_arena;
mod public_resources;
mod qsa_chunk;
mod qsa_chunk_probe;
mod qsa_gpu;
mod qsa_layer_batch;
mod qsa_layer_batch_probe;
mod quant_matvec;
mod resident_batch_projection;
mod resident_gdn;
mod resident_gr;
mod resident_model;
mod resident_moe;
mod resident_probe;
mod resident_qsa;
mod resident_util;
mod residual_batch_gpu;
mod residual_gpu;
mod router_gpu;
mod small_cpu;
mod token_waves;
mod transient_probe;
use cubecl::prelude::*;
use cubecl::wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use kuno_text::model::{MatrixExecutor, Qwen4Exp, RoutedWeights, greedy};
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TensorType, TokenizerLimits};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zeroize::Zeroize;

struct ScrubPublicRead(cubecl::bytes::Bytes);
impl Drop for ScrubPublicRead {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

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
                "resident-bang"
                    | "resident-chat"
                    | "resident-count1-hybrid-chat"
                    | "resident-rainwater"
                    | "resident-lifecycle-probe"
                    | "resident-numeric-boundaries-probe"
                    | "resident-cpu-gpu-expert-probe"
                    | "resident-cpu-gpu-expert-calibration"
                    | "resident-staged-hybrid-expert"
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
fn protected_model_diagnostic(
    mode: &str,
    transient: bool,
    audit: bool,
) -> Result<(), &'static str> {
    if matches!(
        mode,
        "resident-count1-hybrid-chat"
            | "resident-lifecycle-probe"
            | "resident-numeric-boundaries-probe"
            | "resident-cpu-gpu-expert-probe"
            | "resident-cpu-gpu-expert-calibration"
            | "resident-staged-hybrid-expert"
    ) && !(transient && audit)
    {
        return Err("model diagnostic requires owned transient I/O and complete zero audits");
    }
    Ok(())
}

// This numeric-only entry has no model/executor policy, variable route count,
// private input, cache layout or timing selection. Runtime INFO slab policy is
// still the independently protected transient-I/O setting.
const STAGED_HYBRID_MODEL_SETTINGS: [&str; 20] = [
    "KUNO_LAB_NATIVE_MAX8",
    "KUNO_LAB_NATIVE_DECODE",
    "KUNO_LAB_NATIVE_DECODE_TILE",
    "KUNO_LAB_NATIVE_DECODE_KERNELS",
    "KUNO_LAB_NATIVE_BATCH_COUNT",
    "KUNO_LAB_PREFILL_LAST_HEAD",
    "KUNO_LAB_CANCEL_CHECK",
    "KUNO_LAB_ORDERED_PUBLIC_UPLOAD",
    "KUNO_LAB_BATCH_EXPERTS",
    "KUNO_LAB_BLOCK_MOE",
    "KUNO_LAB_WORD_DOWN",
    "KUNO_LAB_SMALL_CPU",
    "KUNO_LAB_FUSED_MATRIX_FAULT",
    "KUNO_LAB_CPU_GDN",
    "KUNO_LAB_IO_TOKEN_CHECKPOINT",
    "KUNO_LAB_BLOCK_KERNEL",
    "KUNO_LAB_EXPERT_SLOTS",
    "KUNO_LAB_STATIC_CACHE_BYTES",
    "KUNO_LAB_EXPERT_BATCH_BYTES",
    "KUNO_LAB_EXPERT_SCATTER",
];
fn count1_hybrid_registration(
    argc: usize,
    receipt: bool,
    cap: Option<&str>,
) -> Result<usize, &'static str> {
    if argc != 4 || !receipt {
        return Err("count1 hybrid requires card, fixed mode, cap and receipt destination");
    }
    match cap {
        Some(raw @ ("0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10")) => {
            raw.parse().map_err(|_| "count1 hybrid bounded cap")
        }
        _ => Err("count1 hybrid cap must be a canonical integer 0 through 10"),
    }
}
fn count1_worker_pool(mode: &str, raw: Option<&str>) -> Result<bool, &'static str> {
    match (mode, raw) {
        (_, None) => Ok(false),
        ("resident-count1-hybrid-chat", Some("scoped")) => Ok(false),
        ("resident-count1-hybrid-chat", Some("pooled")) => Ok(true),
        _ => Err("CPU workers require protected count1 hybrid mode and scoped or pooled"),
    }
}
fn count1_overlap(mode: &str, pooled: bool, raw: Option<&str>) -> Result<bool, &'static str> {
    match (mode, pooled, raw) {
        (_, _, None) => Ok(false),
        ("resident-count1-hybrid-chat", _, Some("0")) => Ok(false),
        ("resident-count1-hybrid-chat", true, Some("1")) => Ok(true),
        _ => Err("CPU/GPU overlap requires protected pooled count1 mode and 0 or 1"),
    }
}
fn count1_opt_in(mode: &str, raw: Option<&str>) -> Result<bool, &'static str> {
    match (mode, raw) {
        (_, None) => Ok(false),
        ("resident-count1-hybrid-chat", Some("0")) => Ok(false),
        ("resident-count1-hybrid-chat", Some("1")) => Ok(true),
        _ => Err("native optimization requires protected count1 mode and 0 or 1"),
    }
}
fn count1_environment_opt_in(mode: &str, name: &str) -> Result<bool, &'static str> {
    let raw = std::env::var_os(name);
    count1_opt_in(
        mode,
        raw.as_ref()
            .map(|v| {
                v.to_str()
                    .ok_or("native optimization setting encoding rejected")
            })
            .transpose()?,
    )
}
fn count1_dispatch(
    mode: &str,
    cap: Option<usize>,
    raw: Option<&str>,
    compiled_inline: bool,
) -> Result<&'static str, &'static str> {
    match (raw, compiled_inline) {
        (None | Some("threaded"), false) => Ok("threaded"),
        (Some("inline"), true)
            if mode == "resident-count1-hybrid-chat" && cap.is_some_and(|value| value <= 10) =>
        {
            Ok("inline")
        }
        _ => Err("GPU dispatch must match the compiled protected count1 artifact"),
    }
}

fn count1_hybrid_setting(name: &str, raw: Option<&str>) -> Result<(), &'static str> {
    let accepted = match name {
        "KUNO_LAB_NATIVE_MAX8" | "KUNO_LAB_NATIVE_DECODE" | "KUNO_LAB_CANCEL_CHECK" => {
            raw == Some("1")
        }
        "KUNO_LAB_NATIVE_DECODE_TILE" => raw == Some("1"),
        "KUNO_LAB_NATIVE_DECODE_KERNELS" => raw == Some("single-q4-all"),
        "KUNO_LAB_NATIVE_BATCH_COUNT" => matches!(raw, None | Some("8")),
        "KUNO_LAB_BLOCK_KERNEL" => matches!(raw, None | Some("1")),
        "KUNO_LAB_EXPERT_SLOTS"
        | "KUNO_LAB_STATIC_CACHE_BYTES"
        | "KUNO_LAB_EXPERT_BATCH_BYTES"
        | "KUNO_LAB_EXPERT_SCATTER" => raw.is_none(),
        _ if STAGED_HYBRID_MODEL_SETTINGS.contains(&name) => matches!(raw, None | Some("0")),
        _ => false,
    };
    if accepted {
        Ok(())
    } else {
        Err("count1 hybrid fixed model settings rejected")
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Count1Banks {
    Standard,
    Intel72,
    Intel96,
    Intel128,
    Intel144,
    Intel160,
}
impl Count1Banks {
    fn parse(mode: &str, vendor: u32, raw: Option<&str>) -> Result<Self, &'static str> {
        match raw {
            None | Some("standard") => Ok(Self::Standard),
            Some("intel72") if mode == "resident-count1-hybrid-chat" && vendor == 0x8086 => {
                Ok(Self::Intel72)
            }
            Some("intel96") if mode == "resident-count1-hybrid-chat" && vendor == 0x8086 => {
                Ok(Self::Intel96)
            }
            Some("intel128") if mode == "resident-count1-hybrid-chat" && vendor == 0x8086 => {
                Ok(Self::Intel128)
            }
            Some("intel144") if mode == "resident-count1-hybrid-chat" && vendor == 0x8086 => {
                Ok(Self::Intel144)
            }
            Some("intel160") if mode == "resident-count1-hybrid-chat" && vendor == 0x8086 => {
                Ok(Self::Intel160)
            }
            _ => Err("fixed count1 bank profile rejected"),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Intel72 => "intel72",
            Self::Intel96 => "intel96",
            Self::Intel128 => "intel128",
            Self::Intel144 => "intel144",
            Self::Intel160 => "intel160",
        }
    }
    fn setting(self, name: &str, raw: Option<&str>) -> Result<(), &'static str> {
        if matches!(
            self,
            Self::Intel96 | Self::Intel128 | Self::Intel144 | Self::Intel160
        ) && name == "KUNO_LAB_BLOCK_MOE"
        {
            return if raw == Some("1") {
                Ok(())
            } else {
                Err("segmented count1 bank block setting rejected")
            };
        }
        if matches!(
            self,
            Self::Intel96 | Self::Intel128 | Self::Intel144 | Self::Intel160
        ) && name == "KUNO_LAB_STATIC_CACHE_BYTES"
        {
            return if raw
                == Some(if self == Self::Intel160 {
                    "3221225472"
                } else if matches!(self, Self::Intel128 | Self::Intel144) {
                    "5368709120"
                } else {
                    "6442450944"
                }) {
                Ok(())
            } else {
                Err("segmented count1 bank static cache setting rejected")
            };
        }
        count1_hybrid_setting(name, raw)
    }
    fn slots(self, vendor: u32) -> usize {
        match self {
            Self::Intel72 => 72,
            Self::Intel96 => 96,
            Self::Intel128 => 128,
            Self::Intel144 => 144,
            Self::Intel160 => 160,
            Self::Standard => {
                if vendor == 0x8086 {
                    64
                } else {
                    16
                }
            }
        }
    }
}

fn staged_hybrid_registration(argc: usize, receipt: bool) -> Result<(), &'static str> {
    if argc != 3 || !receipt {
        return Err("staged hybrid numeric probe requires only card, mode and receipt destination");
    }
    Ok(())
}
fn staged_hybrid_setting(name: &str, raw: Option<&str>) -> Result<(), &'static str> {
    let accepted = match name {
        "KUNO_LAB_NATIVE_DECODE_TILE" => matches!(raw, None | Some("8")),
        "KUNO_LAB_NATIVE_DECODE_KERNELS" => matches!(raw, None | Some("chunk")),
        "KUNO_LAB_NATIVE_BATCH_COUNT"
        | "KUNO_LAB_EXPERT_SLOTS"
        | "KUNO_LAB_STATIC_CACHE_BYTES"
        | "KUNO_LAB_EXPERT_BATCH_BYTES"
        | "KUNO_LAB_EXPERT_SCATTER" => raw.is_none(),
        _ if STAGED_HYBRID_MODEL_SETTINGS.contains(&name) => matches!(raw, None | Some("0")),
        _ => false,
    };
    if !accepted {
        return Err("staged hybrid numeric probe accepts no model or native policy selection");
    }
    Ok(())
}

/// Independent publication inventory, not receipt-selected configurations.
fn staged_hybrid_numeric_cases() -> [cpu_gpu_expert_probe::Config; 8] {
    std::array::from_fn(|i| cpu_gpu_expert_probe::Config {
        down: if i < 4 {
            TensorType::Q5_0
        } else {
            TensorType::Q8_0
        },
        workers: 16,
        backend: kuno_text::cpu_expert::Backend::Avx2,
        misses: if i % 4 == 3 { 10 } else { 5 },
        cpu_misses: [0, 1, 5, 10][i % 4],
        group: 0,
        vector: 1,
        mode: cpu_gpu_expert_schedule::Mode::Concurrent,
    })
}
fn staged_hybrid_numeric_reports(
    reports: cpu_gpu_expert_probe::StagedHybridReport,
) -> Result<Vec<serde_json::Value>, &'static str> {
    if reports.cases.len() != 8 {
        return Err("staged hybrid numeric inventory is not the fixed eight cases");
    }
    staged_hybrid_numeric_cases()
        .into_iter()
        .zip(reports.cases)
        .map(|(config, report)| {
            if report.metadata_bytes == 0
                || report.metadata_bytes > 256 * 1024
                || report.snapshot_bytes == 0
            {
                return Err("staged hybrid retained plan metadata exceeds its fixed bound");
            }
            cpu_gpu_expert_numeric_report(config, report)
        })
        .collect()
}

/// Compiled public subset only: ten original ranks, both actual down formats,
/// mixed work plus GPU-only/CPU-only endpoints. No timing permit is consumed.
fn cpu_gpu_expert_numeric_cases() -> [cpu_gpu_expert_probe::Config; 4] {
    [
        (TensorType::Q5_0, 0, 1),
        (TensorType::Q5_0, 5, 1),
        (TensorType::Q8_0, 5, 0),
        (TensorType::Q8_0, 10, 1),
    ]
    .map(|(down, cpu_misses, vector)| cpu_gpu_expert_probe::Config {
        down,
        workers: 16,
        backend: kuno_text::cpu_expert::Backend::Avx2,
        misses: 10,
        cpu_misses,
        group: 0,
        vector,
        mode: cpu_gpu_expert_schedule::Mode::Concurrent,
    })
}

/// The producer has already completed owning maps, numerical checks, full MAIN
/// erasure, I/O audits and file verification. Refuse incomplete/timed reports at
/// this narrow publication seam; never expose its returned calibration permit.
fn cpu_gpu_expert_numeric_report(
    config: cpu_gpu_expert_probe::Config,
    report: cpu_gpu_expert_probe::Report,
) -> Result<serde_json::Value, &'static str> {
    use cpu_gpu_expert_probe::{BANK_BYTES, IO_COMPONENT_BYTES, PRIVATE_BYTES};
    let gpu_misses = config
        .misses
        .checked_sub(config.cpu_misses)
        .ok_or("fixed-public CPU/GPU numeric report split invalid")?;
    let errors = [
        report.max_tree_error,
        report.max_legacy_error,
        report.max_core_error,
    ];
    if report.timing.is_some()
        || report.bridge_and_merge.is_some()
        || report.layer >= 48
        || report.down_dtype != config.down.gguf_id()
        || report.workers != config.workers
        || report.misses != config.misses
        || report.cpu_misses != config.cpu_misses
        || report.gpu_misses != gpu_misses
        || report.private_pages != 18
        || report.private_bytes != PRIVATE_BYTES
        || report.bank_bytes != BANK_BYTES
        || report.source_precharge != report.gpu_misses as u64 * 4 * 1024 * 1024
        || report.io_inspected_bytes == 0
        || report.io_inspected_bytes > IO_COMPONENT_BYTES / 2
        || report.zero_buffers != 18
        || report.zero_bytes != PRIVATE_BYTES
        || report.metadata_bytes > 1024 * 1024
        || report.snapshot_bytes > 2 * 1024 * 1024
        || errors.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err("fixed-public CPU/GPU numeric report did not accept complete cleanup");
    }
    Ok(serde_json::json!({
        "layer":report.layer,"down_dtype":report.down_dtype,
        "workers":report.workers,"backend":"avx2","mode":"concurrent",
        "ranks":10,"misses":report.misses,"cpu_misses":report.cpu_misses,
        "gpu_misses":report.gpu_misses,"group":config.group,"vector":config.vector,
        "max_tree_error":report.max_tree_error,
        "max_legacy_error":report.max_legacy_error,"max_core_error":report.max_core_error,
        "private_pages":report.private_pages,"private_bytes":report.private_bytes,
        "bank_bytes":report.bank_bytes,"source_precharge":report.source_precharge,
        "io_inspected_bytes":report.io_inspected_bytes,
        "zero_buffers":report.zero_buffers,"zero_bytes":report.zero_bytes,
        "metadata_bytes":report.metadata_bytes,"snapshot_bytes":report.snapshot_bytes,
        "operator_numeric_accepted":true,"owned_cleanup_accepted":true
    }))
}

fn public_duration_ns(duration: std::time::Duration) -> kuno_text::Result<u64> {
    u64::try_from(duration.as_nanos())
        .map_err(|_| kuno_text::Error::InvalidFormat("public calibration duration overflow"))
}

fn cpu_gpu_calibration_metrics(
    report: &cpu_gpu_expert_probe::Report,
) -> kuno_text::Result<serde_json::Value> {
    let mut value = serde_json::json!({
        "layer":report.layer,"down_dtype":report.down_dtype,"workers":report.workers,
        "misses":report.misses,"cpu_misses":report.cpu_misses,"gpu_misses":report.gpu_misses,
        "max_tree_error":report.max_tree_error,"max_legacy_error":report.max_legacy_error,
        "max_core_error":report.max_core_error,"private_pages":report.private_pages,
        "private_bytes":report.private_bytes,"bank_bytes":report.bank_bytes,
        "source_precharge":report.source_precharge,"io_inspected_bytes":report.io_inspected_bytes,
        "zero_buffers":report.zero_buffers,"zero_bytes":report.zero_bytes,
        "metadata_bytes":report.metadata_bytes,"snapshot_bytes":report.snapshot_bytes,
    });
    match (report.timing, report.bridge_and_merge) {
        (Some(times), Some(bridge)) => {
            value["timing_ns"] = serde_json::json!({
                "cpu":public_duration_ns(times.cpu)?,"gpu":public_duration_ns(times.gpu)?,
                "joined":public_duration_ns(times.joined)?,"bridge_and_merge":public_duration_ns(bridge)?,
            });
        }
        (None, None) => value["timing_ns"] = serde_json::Value::Null,
        _ => {
            return Err(kuno_text::Error::InvalidFormat(
                "public calibration timing fields incomplete",
            ));
        }
    }
    Ok(value)
}

fn cpu_gpu_calibration_points(
    report: &cpu_gpu_expert_probe::FixedCalibrationReport,
) -> kuno_text::Result<Vec<serde_json::Value>> {
    let mut points = Vec::with_capacity(cpu_gpu_expert_probe::FIXED_CALIBRATION_POINTS);
    for point in &report.points {
        let numeric = point
            .numeric
            .iter()
            .map(cpu_gpu_calibration_metrics)
            .collect::<kuno_text::Result<Vec<_>>>()?;
        let mut samples = Vec::with_capacity(cpu_gpu_expert_probe::FIXED_CALIBRATION_SAMPLES);
        for sample in &point.samples {
            let mode = match sample.mode {
                cpu_gpu_expert_schedule::Mode::Concurrent => "concurrent",
                cpu_gpu_expert_schedule::Mode::CpuFirst => "cpu-first",
                _ => {
                    return Err(kuno_text::Error::InvalidFormat(
                        "public calibration mode rejected",
                    ));
                }
            };
            samples.push(serde_json::json!({
                "mode":mode,"report":cpu_gpu_calibration_metrics(&sample.report)?,
                "candidate_completed_wall_ns":public_duration_ns(sample.candidate_completed_wall)?,
                "acceptance_cleanup_wall_ns":public_duration_ns(sample.acceptance_cleanup_wall)?,
                "full_run_including_references_wall_ns":public_duration_ns(sample.full_run_including_references_wall)?,
            }));
        }
        points.push(
            serde_json::json!({"down_dtype":point.down_dtype,"cpu_misses":point.cpu_misses,
            "numeric_modes":["concurrent","cpu-first"],"numeric":numeric,"samples":samples}),
        );
    }
    Ok(points)
}

fn native_decode_mode(
    mode: &str,
    raw: Option<&str>,
    native_max8: bool,
    transient: bool,
    audit: bool,
) -> Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1")
            if matches!(
                mode,
                "resident-bang"
                    | "resident-chat"
                    | "resident-rainwater"
                    | "resident-count1-hybrid-chat"
            ) && native_max8
                && transient
                && audit =>
        {
            Ok(true)
        }
        Some("1") => Err(
            "native decode requires a fixed resident fixture, native MAX8, owned transient I/O and complete zero audits",
        ),
        Some(_) => Err("KUNO_LAB_NATIVE_DECODE must be 0 or 1"),
    }
}

fn native_decode_tile(
    raw: Option<&str>,
    protected_native_decode: bool,
) -> Result<usize, &'static str> {
    match raw {
        None | Some("8") => Ok(8),
        Some("1") if protected_native_decode => Ok(1),
        Some("1") => Err("native decode tile1 requires protected native decode"),
        Some(_) => Err("KUNO_LAB_NATIVE_DECODE_TILE must be 1 or 8"),
    }
}

fn native_decode_kernels(
    raw: Option<&str>,
    protected_native_decode: bool,
    projection_tile: usize,
) -> kuno_text::Result<resident_model::batch::NativeDecodeKernels> {
    let policy = resident_model::batch::NativeDecodeKernels::parse(raw.unwrap_or("chunk"))?;
    if policy.label() != "chunk" && (!protected_native_decode || projection_tile != 1) {
        return Err(kuno_text::Error::InvalidFormat(
            "native decode kernels require protected native decode and projection tile1",
        ));
    }
    Ok(policy)
}

fn apply_native_decode_kernels<'f>(
    resident: &mut resident_model::ResidentModel<'_, 'f>,
    executor: &mut executor::GpuExecutor<'f>,
    selected: &str,
    cancel: &AtomicBool,
) -> kuno_text::Result<&'static str> {
    if selected != "chunk"
        && resident
            .set_lab_native_decode_kernels(
                executor,
                resident_model::batch::NativeDecodeKernels::parse(selected)?,
                cancel,
            )?
            .label()
            != selected
    {
        return Err(kuno_text::Error::InvalidFormat(
            "native decode kernel selection mismatch",
        ));
    }
    let observed = resident.native_decode_kernels()?.label();
    if observed != selected {
        return Err(kuno_text::Error::InvalidFormat(
            "native decode kernel observation mismatch",
        ));
    }
    Ok(observed)
}

fn apply_count1_hybrid_cap<'f>(
    resident: &mut resident_model::ResidentModel<'_, 'f>,
    executor: &mut executor::GpuExecutor<'f>,
    selected: Option<usize>,
    cancel: &AtomicBool,
) -> kuno_text::Result<()> {
    if let Some(cap) = selected
        && (resident.completed_cpu_misses(executor)? != 0
            || resident.set_lab_count1_cpu_misses(executor, cap, cancel)? != cap
            || resident.native_cpu_miss_cap()? != Some(cap))
    {
        return Err(kuno_text::Error::InvalidFormat(
            "count1 hybrid completed prefill policy rejected",
        ));
    }
    Ok(())
}

// The generation and terminal cancellation checks use the same dispatch.
// A native continuation never falls back to scalar: its PLE cursor must stay
// aligned with the model position through the entire accepted sequence.
fn resident_decode_step<'f>(
    resident: &mut resident_model::ResidentModel<'_, 'f>,
    executor: &mut executor::GpuExecutor<'f>,
    token: u32,
    cancel: &AtomicBool,
    native_decode: bool,
) -> kuno_text::Result<zeroize::Zeroizing<Vec<f32>>> {
    if native_decode {
        resident.prefill_chunk(
            executor,
            &[token],
            resident_model::batch::Purpose::LastLogits,
            cancel,
        )
    } else {
        resident.step(executor, token, cancel)
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
    protected_model_diagnostic(&mode, transient_resident, public_owned_zero_audit)?;
    if mode == "resident-staged-hybrid-expert" {
        staged_hybrid_registration(
            std::env::args().count(),
            std::env::var_os("KUNO_PUBLIC_RECEIPT").is_some(),
        )?;
        for name in STAGED_HYBRID_MODEL_SETTINGS {
            let raw = std::env::var_os(name);
            let value = raw
                .as_ref()
                .map(|value| {
                    value
                        .to_str()
                        .ok_or("staged hybrid model setting encoding rejected")
                })
                .transpose()?;
            staged_hybrid_setting(name, value)?;
        }
    }
    let cpu_workers_raw = std::env::var_os("KUNO_LAB_CPU_WORKERS");
    let count1_pooled = count1_worker_pool(
        &mode,
        cpu_workers_raw
            .as_ref()
            .map(|v| v.to_str().ok_or("CPU worker setting encoding rejected"))
            .transpose()?,
    )?;
    let overlap_raw = std::env::var_os("KUNO_LAB_CPU_GPU_OVERLAP");
    let count1_overlap = count1_overlap(
        &mode,
        count1_pooled,
        overlap_raw
            .as_ref()
            .map(|v| v.to_str().ok_or("CPU/GPU overlap encoding rejected"))
            .transpose()?,
    )?;
    let banks_raw = std::env::var_os("KUNO_LAB_COUNT1_BANKS");
    let count1_banks = Count1Banks::parse(
        &mode,
        vendor,
        banks_raw
            .as_ref()
            .map(|v| v.to_str().ok_or("bank profile encoding rejected"))
            .transpose()?,
    )?;
    let count1_hybrid_cap = if mode == "resident-count1-hybrid-chat" {
        let cap = count1_hybrid_registration(
            std::env::args().count(),
            std::env::var_os("KUNO_PUBLIC_RECEIPT").is_some(),
            std::env::args().nth(3).as_deref(),
        )?;
        for name in STAGED_HYBRID_MODEL_SETTINGS {
            let raw = std::env::var_os(name);
            let value = raw
                .as_ref()
                .map(|v| v.to_str().ok_or("count1 hybrid setting encoding rejected"))
                .transpose()?;
            count1_banks.setting(name, value)?;
        }
        Some(cap)
    } else {
        None
    };
    if count1_banks != Count1Banks::Standard && count1_hybrid_cap.is_none_or(|cap| cap > 10) {
        return Err("Expanded count1 banks require protected count1 mode".into());
    }
    let mapped_weights = count1_environment_opt_in(&mode, "KUNO_LAB_MAPPED_WEIGHTS")?;
    let raw_dispatch = std::env::var_os("KUNO_LAB_DEVICE_DISPATCH");
    let inline_dispatch = cubecl_common::device_handle::DeviceHandle::<
        <WgpuRuntime as cubecl::prelude::Runtime>::Server,
    >::is_blocking();
    if inline_dispatch != cfg!(feature = "direct-device") {
        return Err("Compiled GPU dispatch does not match the original runtime type".into());
    }
    let device_dispatch = count1_dispatch(
        &mode,
        count1_hybrid_cap,
        raw_dispatch
            .as_ref()
            .map(|v| v.to_str().ok_or("GPU dispatch setting encoding rejected"))
            .transpose()?,
        inline_dispatch,
    )?;
    println!("gpu_device_dispatch mode={device_dispatch} blocking={inline_dispatch}");
    let kernel_profile = count1_environment_opt_in(&mode, "KUNO_LAB_KERNEL_PROFILE")?;
    let fast_gpu = count1_environment_opt_in(&mode, "KUNO_LAB_FAST_GPU")?;
    let fast_gpu_fault = count1_environment_opt_in(&mode, "KUNO_LAB_FAST_GPU_FAULT")?;
    let next_shared = count1_environment_opt_in(&mode, "KUNO_LAB_NEXT_SHARED")?;
    let shared_gdn = count1_environment_opt_in(&mode, "KUNO_LAB_NEXT_SHARED_GDN")?;
    if shared_gdn && !next_shared {
        return Err("Shared GDN requires protected shared Next kernels".into());
    }
    if next_shared && (!fast_gpu || fast_gpu_fault || count1_hybrid_cap.is_none_or(|cap| cap > 10))
    {
        return Err(
            "Shared Next kernels require protected count1 fast GPU mode without fused faults"
                .into(),
        );
    }
    if fast_gpu_fault && !fast_gpu {
        return Err("Fused subgroup finite checks require the explicit fast GPU path".into());
    }
    let long_fixture = count1_environment_opt_in(&mode, "KUNO_LAB_LONG_FIXTURE")?;
    let cpu_buffered_weights = cfg!(feature = "cpu-buffered-weights");
    if cpu_buffered_weights && (!mapped_weights || !count1_pooled) {
        return Err("CPU buffered weights require mapped pooled hybrid mode".into());
    }
    let cpu_mapped_prefetch = count1_environment_opt_in(&mode, "KUNO_LAB_CPU_MAPPED_PREFETCH")?;
    if cpu_mapped_prefetch && (!mapped_weights || !count1_pooled || cpu_buffered_weights) {
        return Err("CPU source prefetch requires explicitly mapped pooled workers".into());
    }
    let cpu_threads = match std::env::var("KUNO_LAB_CPU_THREADS").as_deref() {
        Ok("4") if mode == "resident-count1-hybrid-chat" => 4,
        Ok("8") if mode == "resident-count1-hybrid-chat" => 8,
        Ok("16") | Err(_) => 16,
        _ => return Err("CPU thread choice requires explicit protected 4,8,16 geometry".into()),
    };
    if cpu_threads != 16 && !count1_pooled {
        return Err("CPU thread tuning requires pooled protected workers".into());
    }
    let cpu_q8 = count1_environment_opt_in(&mode, "KUNO_LAB_CPU_Q8")?;
    let cpu_q8_residual = count1_environment_opt_in(&mode, "KUNO_LAB_CPU_Q8_RESIDUAL")?;
    let cpu_q8_two_pass = count1_environment_opt_in(&mode, "KUNO_LAB_CPU_Q8_TWO_PASS")?;
    if cpu_q8_two_pass != cfg!(feature = "cpu-q8-two-pass")
        || (cpu_q8_two_pass && (!cpu_q8 || !cpu_q8_residual || !count1_pooled))
    {
        return Err("Compiled two-pass Q8 requires explicit pooled residual Q8 selection".into());
    }
    if cpu_q8_residual && !cpu_q8 {
        return Err("Residual Q8 requires explicitly selected Q8 activations".into());
    }
    if cfg!(feature = "cpu-fma")
        && (mode != "resident-count1-hybrid-chat"
            || !count1_pooled
            || cpu_q8
            || !kuno_text::cpu_expert::fma_available())
    {
        return Err("Compiled CPU FMA requires explicit pooled protected count1 float mode and AVX2/F16C/FMA".into());
    }
    let large_weight_cache = count1_environment_opt_in(&mode, "KUNO_LAB_LARGE_WEIGHT_CACHE")?;
    let reclaim_gpu_sources = count1_environment_opt_in(&mode, "KUNO_LAB_RECLAIM_GPU_SOURCES")?;
    let pageout_gpu_sources = count1_environment_opt_in(&mode, "KUNO_LAB_PAGEOUT_GPU_SOURCES")?;
    if pageout_gpu_sources && !reclaim_gpu_sources {
        return Err("Mapped GPU source pageout requires explicit source reclaim".into());
    }
    if reclaim_gpu_sources && (!mapped_weights || count1_hybrid_cap != Some(10)) {
        return Err(
            "GPU source cache hint requires mapped static cap10 protected count1 mode".into(),
        );
    }
    let native_scratch_clear = count1_environment_opt_in(&mode, "KUNO_LAB_NATIVE_SCRATCH_CLEAR")?;
    let compact_info = count1_environment_opt_in(&mode, "KUNO_LAB_COMPACT_INFO")?;
    let info_batch_upload = count1_environment_opt_in(&mode, "KUNO_LAB_INFO_BATCH_UPLOAD")?;
    if info_batch_upload && (!compact_info || count1_hybrid_cap.is_none_or(|cap| cap > 10)) {
        return Err("Batched INFO requires protected compact count1 mode".into());
    }
    let batch_tasks = count1_environment_opt_in(&mode, "KUNO_LAB_BATCH_TASKS")?;
    let gpu_router = count1_environment_opt_in(&mode, "KUNO_LAB_GPU_ROUTER")?;
    if next_shared && !gpu_router {
        return Err("Shared Next kernels require GPU routing".into());
    }
    let retained_metadata = count1_environment_opt_in(&mode, "KUNO_LAB_RETAINED_METADATA")?;
    if retained_metadata && count1_hybrid_cap.is_none_or(|cap| cap > 10) {
        return Err("Retained metadata requires protected count1 hybrid mode".into());
    }
    if gpu_router && count1_hybrid_cap.is_none_or(|cap| cap > 10) {
        return Err("GPU routing requires protected count1 hybrid mode".into());
    }
    let defer_router_io = count1_environment_opt_in(&mode, "KUNO_LAB_DEFER_ROUTER_IO")?;
    if defer_router_io && (!compact_info || count1_hybrid_cap.is_none_or(|cap| cap > 10)) {
        return Err("Deferred router IO requires compact protected count1 hybrid mode".into());
    }
    if mode == "resident-count1-hybrid-chat" {
        let raw = std::env::var_os("KUNO_LAB_INFO_SLAB_BYTES");
        let expected = if compact_info { "4096" } else { "0" };
        if raw.as_ref().is_some_and(|v| v != expected) || (compact_info && raw.is_none()) {
            return Err("Compact INFO policy must match explicit bounded runtime slab".into());
        }
    }
    if large_weight_cache && !mapped_weights {
        return Err("Larger public weight cache requires explicit mapped weights".into());
    }
    if cpu_q8 && !count1_pooled {
        return Err("Q8 CPU activation arithmetic requires explicit pooled workers".into());
    }
    if kernel_profile {
        use cubecl_runtime::config::{
            CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel,
        };
        let mut config = CubeClRuntimeConfig::default();
        config.profiling.logger.level = ProfilingLogLevel::Basic;
        config.profiling.logger.stdout = true;
        // Programmatic config avoids compile-feature-dependent file/env lookup.
        // Only fixed-public lab kernels are profiled; no request-derived keys.
        CubeClRuntimeConfig::set(config);
    }
    // This fixed operator accepts no additional case/configuration arguments.
    // Its receipt destination is required before any device or model access.
    let cpu_gpu_expert_receipt = if matches!(
        mode.as_str(),
        "resident-cpu-gpu-expert-probe"
            | "resident-cpu-gpu-expert-calibration"
            | "resident-staged-hybrid-expert"
    ) {
        if std::env::args().count() != 3 {
            return Err("CPU/GPU expert probe accepts only its compiled numeric cases".into());
        }
        Some(
            std::env::var_os("KUNO_PUBLIC_RECEIPT")
                .ok_or("fixed-public CPU/GPU expert receipt path required")?,
        )
    } else {
        None
    };
    let native_max8 = std::env::var("KUNO_LAB_NATIVE_MAX8").as_deref() == Ok("1");
    let native_decode_setting = match std::env::var("KUNO_LAB_NATIVE_DECODE") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_NATIVE_DECODE must be 0 or 1".into()),
    };
    // Reject an unprotected or malformed opt-in before creating a device or
    // reading model payload. The ordinary native/scalar defaults are unchanged.
    let native_decode = native_decode_mode(
        &mode,
        native_decode_setting.as_deref(),
        native_max8,
        transient_resident,
        public_owned_zero_audit,
    )?;
    let native_decode_tile_setting = match std::env::var("KUNO_LAB_NATIVE_DECODE_TILE") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_NATIVE_DECODE_TILE must be 1 or 8".into()),
    };
    let native_decode_tile =
        native_decode_tile(native_decode_tile_setting.as_deref(), native_decode)?;
    let native_decode_kernel_setting = match std::env::var("KUNO_LAB_NATIVE_DECODE_KERNELS") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("invalid public native decode kernel profile".into()),
    };
    let native_decode_kernels = native_decode_kernels(
        native_decode_kernel_setting.as_deref(),
        native_decode,
        native_decode_tile,
    )?
    .label();
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
    let expert_slots =
        bank_slots::configured_capacity(requested_slots.as_deref(), count1_banks.slots(vendor))
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
            "resident-bang"
                | "resident-chat"
                | "resident-rainwater"
                | "resident-count1-hybrid-chat"
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
        if next_shared
            && (adapter.limits().max_compute_invocations_per_workgroup < 256
                || adapter.limits().max_compute_workgroup_size_x < 256)
        {
            return Err("shared Next router requires256 workgroup lanes".into());
        }
        if fast_gpu
            && (!adapter.features().contains(wgpu::Features::SUBGROUP)
                || info.subgroup_min_size < 16
                || info.subgroup_max_size > 32
                || adapter.limits().max_compute_invocations_per_workgroup < 128
                || adapter.limits().max_compute_workgroup_size_x < 128)
        {
            return Err(
                "fast count1 GPU requires enabled16..32 subgroups and128 workgroup lanes".into(),
            );
        }
        println!(
            "adapter={} vendor={:04x} device={:04x} backend={:?}",
            info.name, info.vendor, info.device, info.backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Kuno public synthetic Rust kernel"),
                required_features: adapter.features()
                    & (wgpu::Features::SHADER_F16
                        | wgpu::Features::SUBGROUP
                        | if kernel_profile {
                            wgpu::Features::TIMESTAMP_QUERY
                                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                        } else {
                            wgpu::Features::empty()
                        }),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await?;
        let grouped_limits = adapter.limits();
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
                tasks_max: if batch_tasks { 256 } else { 32 },
                memory_config: private_arena::required_memory_configuration(),
            },
        );
        let client = WgpuRuntime::client(&selected);
        if kernel_profile {
            client.set_profiling_paused(true);
        }
        if mode == "grouped-moe-chunk-probe" {
            return grouped_moe_chunk_probe::run(&client, &grouped_limits);
        }
        if mode == "qsa-layer-batch-probe" {
            return qsa_layer_batch_probe::run(&client);
        }
        if mode == "ple-chunk-probe" {
            return ple_chunk_probe::run(&client, gr_storage_alignment);
        }
        if mode == "gdn-layer-batch-probe" {
            gdn_layer_batch_probe::run(&client)?;
            return Ok(());
        }
        if mode == "gdn-batch-pointwise-probe" {
            gdn_batch_pointwise_probe::run(&client)?;
            return Ok(());
        }
        if mode == "gr-batch-probe" {
            gr_batch_probe::run(&client, gr_storage_alignment)?;
            return Ok(());
        }
        if mode == "qsa-chunk-probe" {
            return qsa_chunk_probe::run(&client);
        }
        if mode == "gdn-chunk-probe" {
            gdn_chunk_probe::run(&client)?;
            return Ok(());
        }
        if mode == "bank96-probe" {
            bank96_probe::run(&client)?;
            return Ok(());
        }
        if mode == "info-slab-probe" {
            info_slab_probe::run(&client)?;
            return Ok(());
        }
        if mode == "fused-matrix-fault-probe" {
            return matmul_probe::run_fused_fault(&client);
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
        if mode == "resident-cpu-gpu-expert-probe" {
            let target = cpu_gpu_expert_receipt
                .ok_or("fixed-public CPU/GPU expert receipt path required")?;
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let file = GgufFile::open(path, GgufLimits::default())?;
            let model = Qwen4Exp::bind(&file)?;
            let retired = transient_context
                .clone()
                .ok_or("CPU/GPU expert probe requires the configured transient stream")?;
            let cancel = AtomicBool::new(false);
            let mut reports = Vec::with_capacity(4);
            for config in cpu_gpu_expert_numeric_cases() {
                let (report, permit) = cpu_gpu_expert_probe::run(
                    &client,
                    &file,
                    model.bound_graph(),
                    &grouped_limits,
                    retired.clone(),
                    config,
                    cpu_gpu_expert_probe::Purpose::Numeric,
                    &cancel,
                )?;
                // Numeric evidence is not authorization to run Calibration.
                // The returned permit retains only this file/client/config.
                drop(permit);
                reports.push(cpu_gpu_expert_numeric_report(config, report)?);
            }
            file.verify_file_unchanged()?;
            if cancel.load(std::sync::atomic::Ordering::Acquire)
                || retired.load(std::sync::atomic::Ordering::Acquire)
            {
                return Err("CPU/GPU expert probe did not accept complete cleanup".into());
            }
            // No partial receipt: every sequential operator has returned only
            // after owning maps, full erasure, I/O audit and final consistency.
            let receipt = serde_json::json!({
                "scope":"fixed-public-original-cpu-gpu-expert-numeric",
                "vendor":vendor,"weights":"borrowed-unmodified-ollama-blob",
                "fixed_public_cases":4,"purpose":"numeric","ranks_per_case":10,
                "private_bytes_per_case":cpu_gpu_expert_probe::PRIVATE_BYTES,
                "physical_limit_bytes":cpu_gpu_expert_probe::PHYSICAL_LIMIT,
                "runtime_io_limit_bytes":executor::RESIDENT_RUNTIME_IO_MAX,
                "cases":reports,"operator_numeric_accepted":true,
                "owned_cleanup_accepted":true,"context_retired":false,
                "whole_model_numeric_accepted":false,"performance_accepted":false,
                "automatic_split_policy_enabled":false
            });
            let output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(target)?;
            serde_json::to_writer_pretty(output, &receipt)?;
            println!(
                "public_cpu_gpu_expert_numeric_complete cases=4 ranks=10 owned_cleanup=true context_retired=false performance_accepted=false"
            );
            return Ok(());
        }
        if mode == "resident-cpu-gpu-expert-calibration" {
            let target = cpu_gpu_expert_receipt
                .ok_or("fixed-public CPU/GPU calibration receipt path required")?;
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let file = GgufFile::open(path, GgufLimits::default())?;
            let model = Qwen4Exp::bind(&file)?;
            let retired = transient_context
                .clone()
                .ok_or("CPU/GPU calibration requires configured transient stream")?;
            let cancel = AtomicBool::new(false);
            let reports = cpu_gpu_expert_probe::run_fixed_calibration(
                &client,
                &file,
                model.bound_graph(),
                &grouped_limits,
                retired.clone(),
                &cancel,
            )?;
            let points = cpu_gpu_calibration_points(&reports)?;
            file.verify_file_unchanged()?;
            if cancel.load(std::sync::atomic::Ordering::Acquire)
                || retired.load(std::sync::atomic::Ordering::Acquire)
            {
                return Err("CPU/GPU calibration did not accept complete cleanup".into());
            }
            let receipt = serde_json::json!({
                "scope":"fixed-public-original-cpu-gpu-expert-calibration",
                "vendor":vendor,"weights":"borrowed-unmodified-ollama-blob",
                "fixed_public_points":6,"numeric_gates":12,"timing_samples":24,
                "ranks_per_case":10,"group":0,"vector":0,"backend":"avx2","workers":16,
                "cache_scope":"warmed-original-read-operator-only",
                "source_kind":"original-packed-gguf","points":points,
                "owned_cleanup_accepted":true,"context_retired":false,
                "full_model_or_automatic_placement_accepted":false,
            });
            use std::io::Write;
            let mut output = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(target)?;
            output.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
            println!(
                "public_cpu_gpu_expert_calibration_complete points=6 numeric_gates=12 samples=24 owned_cleanup=true context_retired=false full_model_accepted=false"
            );
            return Ok(());
        }
        if mode == "resident-staged-hybrid-expert" {
            let target =
                cpu_gpu_expert_receipt.ok_or("fixed-public staged hybrid receipt path required")?;
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let file = GgufFile::open(path, GgufLimits::default())?;
            let model = Qwen4Exp::bind(&file)?;
            let retired = transient_context
                .clone()
                .ok_or("staged hybrid probe requires configured transient stream")?;
            let cancel = AtomicBool::new(false);
            let reports = cpu_gpu_expert_probe::run_staged_hybrid_acceptance(
                &client,
                &file,
                model.bound_graph(),
                &grouped_limits,
                retired.clone(),
                &cancel,
            )?;
            let cases = staged_hybrid_numeric_reports(reports)?;
            file.verify_file_unchanged()?;
            if cancel.load(std::sync::atomic::Ordering::Acquire)
                || retired.load(std::sync::atomic::Ordering::Acquire)
            {
                return Err("staged hybrid probe did not accept complete cleanup".into());
            }
            let receipt = serde_json::json!({
                "scope":"fixed-public-original-staged-cpu-gpu-expert-numeric",
                "vendor":vendor,"weights":"borrowed-unmodified-ollama-blob",
                "source_kind":"original-packed-gguf","fixed_public_cases":8,
                "purpose":"numeric","ranks_per_case":10,"group":0,"vector":1,
                "backend":"avx2","workers":16,"bank_slots_per_owner":16,
                "private_pages_per_case":18,
                "private_bytes_per_case":cpu_gpu_expert_probe::PRIVATE_BYTES,
                "bank_bytes_per_case":cpu_gpu_expert_probe::BANK_BYTES,
                "physical_limit_bytes":cpu_gpu_expert_probe::PHYSICAL_LIMIT,
                "source_window_limit_bytes":cpu_gpu_expert_probe::SOURCE_BYTES,
                "runtime_io_limit_bytes":executor::RESIDENT_RUNTIME_IO_MAX,
                "runtime_io_component_limit_bytes":cpu_gpu_expert_probe::IO_COMPONENT_BYTES,
                "private_host_limit_bytes":cpu_gpu_expert_probe::HOST_PRIVATE_LIMIT,
                "cpu_private_host_limit_bytes":kuno_text::cpu_expert::MAX_PRIVATE_HOST_BYTES,
                "public_cpu_packed_limit_bytes":kuno_text::cpu_expert::MAX_PUBLIC_PACKED_BYTES,
                "bridge_host_limit_bytes":1048576,"oracle_host_limit_bytes":2097152,
                "retained_plan_limit_bytes":262144,"cases":cases,
                "operator_numeric_accepted":true,"owned_cleanup_accepted":true,
                "context_retired":false,"cache_publication_accepted":false,
                "whole_model_numeric_accepted":false,"timing_accepted":false,
                "performance_accepted":false,"production_accepted":false,
                "automatic_split_policy_enabled":false
            });
            // Every operator's complete cleanup precedes publication. Storage
            // failure can leave an incomplete public file but emits no marker.
            let output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(target)?;
            serde_json::to_writer_pretty(output, &receipt)?;
            println!(
                "public_staged_hybrid_expert_numeric_complete cases=8 ranks=10 owned_cleanup=true context_retired=false performance_accepted=false"
            );
            return Ok(());
        }
        if mode == "resident-lifecycle-probe" {
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let file = GgufFile::open(path, GgufLimits::default())?;
            let retired = transient_context
                .clone()
                .ok_or("lifecycle probe requires the configured transient stream")?;
            let mut receipt = resident_model::lifecycle_probe::run(
                &file,
                resident_model::lifecycle_probe::RunConfig {
                    client: client.clone(),
                    limits: grouped_limits.clone(),
                    vendor,
                    retired,
                },
            )?;
            receipt["vendor"] = serde_json::json!(vendor);
            receipt["weights"] = serde_json::json!("borrowed-unmodified-ollama-blob");
            receipt["fixed_public_input"] = serde_json::json!("original tokenizer ! repeated4/8");
            receipt["seed"] = serde_json::json!(424242);
            receipt["performance_claim"] = serde_json::json!(false);
            // The fixed-public runner returned only aggregate statistics after
            // all actual owners closed. Never overwrite a prior lab receipt.
            let target = std::env::var_os("KUNO_PUBLIC_RECEIPT")
                .ok_or("fixed-public lifecycle receipt path required")?;
            let output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(target)?;
            serde_json::to_writer_pretty(output, &receipt)?;
            println!(
                "public_actual_model_lifecycle_complete cases=22 owned_cleanup=true context_retired=false performance_claim=false"
            );
            return Ok(());
        }
        if mode == "resident-numeric-boundaries-probe" {
            let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?;
            let file = GgufFile::open(path, GgufLimits::default())?;
            let retired = transient_context
                .clone()
                .ok_or("numeric boundaries require the configured transient stream")?;
            let mut receipt = resident_model::numeric_boundaries::run(
                &file,
                resident_model::lifecycle_probe::RunConfig {
                    client: client.clone(),
                    limits: grouped_limits.clone(),
                    vendor,
                    retired,
                },
            )?;
            receipt["vendor"] = serde_json::json!(vendor);
            receipt["weights"] = serde_json::json!("borrowed-unmodified-ollama-blob");
            receipt["fixed_public_input"] =
                serde_json::json!("original tokenizer ! position0 count1");
            receipt["seed"] = serde_json::json!(424242);
            let target = std::env::var_os("KUNO_PUBLIC_RECEIPT")
                .ok_or("fixed-public numeric boundary receipt path required")?;
            let output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(target)?;
            serde_json::to_writer_pretty(output, &receipt)?;
            println!(
                "public_actual_numeric_boundaries_complete pairs=4 owners=8 owned_cleanup=true whole_model_numeric_accepted=false performance_accepted=false"
            );
            return Ok(());
        }
        if gpu_router {
            router_gpu::probe(&client)?;
            cubecl::future::block_on(client.sanitize_transient_io_audited())?;
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
                    unsafe { ArrayArg::from_raw_parts(w.clone(), words.len()) },
                    unsafe { ArrayArg::from_raw_parts(x.clone(), input.len()) },
                    unsafe { ArrayArg::from_raw_parts(y.clone(), rows) },
                    rows,
                    cols,
                    dtype,
                    0,
                );
                let actual_bytes = client.read_one(y.clone())?;
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
                if fast_gpu {
                    quant_matvec::fast::packed_single::launch::<WgpuRuntime>(
                        &client,
                        quant_matvec::fast::grid(rows),
                        quant_matvec::fast::cube(),
                        unsafe { ArrayArg::from_raw_parts(w.clone(), words.len()) },
                        unsafe { ArrayArg::from_raw_parts(x.clone(), input.len()) },
                        unsafe { ArrayArg::from_raw_parts(y.clone(), rows) },
                        1,
                        0,
                        0,
                        0,
                        rows,
                        cols,
                        dtype,
                        tensor.info().row_bytes() as usize,
                    );
                    let fast_bytes = ScrubPublicRead(client.read_one(y.clone())?);
                    let fast_values = f32::from_bytes(&fast_bytes.0);
                    if fast_values.len() != rows {
                        return Err("fixed public subgroup packed result extent".into());
                    }
                    let mut fast_error = 0f32;
                    for (&got, &reference) in fast_values.iter().zip(&expected) {
                        fast_error = fast_error.max((got - reference).abs());
                        if !got.is_finite()
                            || (got - reference).abs() > 1e-4 + 3e-4 * reference.abs()
                        {
                            println!(
                                "fixed_public_fast_numeric_failure dtype={dtype} got_bits={} expected_bits={}",
                                got.to_bits(),
                                reference.to_bits()
                            );
                            return Err("fixed public subgroup packed numeric parity".into());
                        }
                    }
                    println!(
                        "rust_fast_packed_matvec dtype={dtype} rows={rows} cols={cols} max_abs_error={fast_error}"
                    );
                    if fast_gpu_fault {
                        for case in 0..4 {
                            let mut initial = vec![0f32; 16];
                            initial[2..2 + rows].fill(f32::NAN);
                            if case == 1 {
                                initial[15] = f32::NAN;
                            }
                            let mut fixture_input = input.clone();
                            if case == 2 {
                                fixture_input[0] = f32::NAN;
                            }
                            let initial_fault = if case == 3 { 8u32 } else { 0 };
                            let out = client.create_from_slice(f32::as_bytes(&initial));
                            let fixture_x = client.create_from_slice(f32::as_bytes(&fixture_input));
                            let fault = client.create_from_slice(&initial_fault.to_le_bytes());
                            quant_matvec::fast::packed_single_fault::launch::<WgpuRuntime>(
                                &client,
                                quant_matvec::fast::fault_grid(rows, 16, true),
                                quant_matvec::fast::cube(),
                                unsafe { ArrayArg::from_raw_parts(w.clone(), words.len()) },
                                unsafe { ArrayArg::from_raw_parts(fixture_x, input.len()) },
                                unsafe { ArrayArg::from_raw_parts(out.clone(), 16) },
                                unsafe { ArrayArg::from_raw_parts(fault.clone(), 1) },
                                1,
                                0,
                                0,
                                2,
                                2,
                                rows,
                                cols,
                                dtype,
                                tensor.info().row_bytes() as usize,
                                rows,
                                true,
                            );
                            let out_bytes = ScrubPublicRead(client.read_one(out)?);
                            let values = f32::from_bytes(&out_bytes.0);
                            let fault_bytes = ScrubPublicRead(client.read_one(fault)?);
                            let expected_fault: u32 = match case {
                                0 => 0,
                                3 => 8,
                                _ => 1,
                            };
                            if fault_bytes.0[..] != expected_fault.to_le_bytes()
                                || values.len() != 16
                            {
                                return Err(
                                    "Fixed public fused packed fault/tail parity failed".into()
                                );
                            }
                            if case != 2
                                && values[2..2 + rows].iter().zip(&expected).any(
                                    |(&got, &reference)| {
                                        !got.is_finite()
                                            || (got - reference).abs()
                                                > 1e-4 + 3e-4 * reference.abs()
                                    },
                                )
                            {
                                return Err(
                                    "Fixed public fused packed numeric parity failed".into()
                                );
                            }
                        }
                        println!("rust_fast_packed_fused_fault dtype={dtype} cases=4 passed=true");
                    }
                }
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
            // Startup operators intentionally use public fixture arenas and
            // retain their original submission semantics. Only this guarded
            // complete-model fixture selects active transient metadata lookup.
            resident_util::select_retained_metadata(retained_metadata);
            resident_util::select_native_scratch_clear(native_scratch_clear);
            let fixture: serde_json::Value =
                serde_json::from_str(include_str!("../../text/qwen38-baseline.json"))?;
            let resident_enabled = mode.starts_with("resident-");
            if native_max8 && (!resident_enabled || transient_context.is_none()) {
                return Err(
                    "native MAX8 requires fixed resident fixture and owned transient I/O".into(),
                );
            }
            let native_count = if native_max8 {
                let count = std::env::var("KUNO_LAB_NATIVE_BATCH_COUNT")
                    .unwrap_or_else(|_| "8".to_owned())
                    .parse::<usize>()?;
                if !(1..=8).contains(&count) {
                    return Err("invalid native MAX8 public fixture count".into());
                }
                count
            } else {
                8
            };
            let native_config = if native_max8 {
                let config = resident_model::batch::Config::new(&client, &grouped_limits, 8)?;
                Some(if count1_hybrid_cap.is_some() {
                    if count1_pooled {
                        config.reserve_count1_hybrid_arithmetic(
                            true,
                            count1_overlap,
                            cpu_q8,
                            cpu_q8_residual,
                            cpu_mapped_prefetch,
                            cpu_threads,
                        )?
                    } else {
                        config.reserve_count1_hybrid()?
                    }
                } else {
                    config
                })
            } else {
                None
            };
            let prefill_last_head =
                std::env::var("KUNO_LAB_PREFILL_LAST_HEAD").as_deref() == Ok("1");
            if prefill_last_head && !resident_enabled {
                return Err("last-head prefill requires a resident fixture".into());
            }
            let fixture_mode = if long_fixture {
                "rainwater"
            } else {
                mode.strip_prefix("resident-").unwrap_or(&mode)
            };
            let (prompt, limit) = match fixture_mode {
                "bang" => ("!", 4usize),
                "chat" | "count1-hybrid-chat" => (
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
            if mapped_weights {
                // SAFETY: this exclusive fixed-public lab borrows the original
                // immutable Ollama content-addressed blob. No model installer or
                // in-place writer may run during the complete borrowed session.
                unsafe {
                    file.enable_mapped_weights()?;
                }
            }
            let model = Qwen4Exp::bind(&file)?;
            let ids = tokenizer.encode(prompt, true)?;
            let mut sequence = model.sequence(ids.len() + limit)?;
            let device_budget = if matches!(
                count1_banks,
                Count1Banks::Intel128 | Count1Banks::Intel144 | Count1Banks::Intel160
            ) {
                31
            } else if vendor == 0x8086 {
                26
            } else {
                10
            } * 1024
                * 1024
                * 1024;
            if next_shared {
                next_shared::probe(&client, &file)?;
                if shared_gdn {
                    next_shared::probe_gdn(&client, model.bound_graph().geometry().gdn)?;
                }
            }
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
            executor.gpu_router = gpu_router;
            executor.fast_dense = fast_gpu;
            executor.fast_dense_fault = fast_gpu_fault;
            executor.shared_kernels = next_shared;
            executor.shared_gdn = shared_gdn;
            executor.token_io_checkpoint = token_checkpoint || defer_router_io;
            let cancel = AtomicBool::new(false);
            let mut resident = if resident_enabled {
                Some(resident_model::ResidentModel::new_for_lab_fixture(
                    model.bound_graph(),
                    client.clone(),
                    ids.len() + limit,
                    &mut executor,
                    public_owned_zero_audit,
                    native_config.clone(),
                )?)
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
            if native_max8 {
                let r = resident.as_mut().ok_or("native fixture owner missing")?;
                let chunks = ids.len().div_ceil(native_count);
                for (n, tokens) in ids.chunks(native_count).enumerate() {
                    logits.zeroize();
                    let purpose = if n + 1 == chunks {
                        resident_model::batch::Purpose::LastLogits
                    } else {
                        resident_model::batch::Purpose::Advance
                    };
                    let mut output = r.prefill_chunk(&mut executor, tokens, purpose, &cancel)?;
                    logits = std::mem::take(&mut *output);
                }
            } else {
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
            }
            let prefill = start.elapsed();
            if kernel_profile {
                client.sync().await?;
            }
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
            let native_prefill_projection_tile = executor.native_projection_tile();
            let mut native_decode_projection_tile = native_prefill_projection_tile;
            let mut observed_native_decode_kernels = "chunk";
            executor.by_tensor.clear();
            if let Some(r) = &mut resident {
                if native_decode {
                    // A public laboratory policy, selected once at a completed
                    // prefill boundary. Prefill remains tile8; no private count
                    // chooses the projection specialization or shader key.
                    native_decode_projection_tile =
                        r.set_lab_native_projection_tile(&mut executor, native_decode_tile)?;
                    observed_native_decode_kernels = apply_native_decode_kernels(
                        r,
                        &mut executor,
                        native_decode_kernels,
                        &cancel,
                    )?;
                }
                apply_count1_hybrid_cap(r, &mut executor, count1_hybrid_cap, &cancel)?;
                r.phase_ns.clear();
            }
            let pre_cpu_phase_ns = if count1_hybrid_cap.is_some() {
                resident
                    .as_ref()
                    .ok_or("CPU phase owner absent")?
                    .native_cpu_phase_ns(&executor)?
                    .ok_or("CPU phases absent")?
            } else {
                [0; 4]
            };
            let pre_cpu_buffered_reads = if count1_hybrid_cap.is_some() {
                resident
                    .as_ref()
                    .ok_or("CPU buffered owner absent")?
                    .native_cpu_buffered_weights(&executor)?
                    .ok_or("CPU buffered operator absent")?
                    .1
            } else {
                0
            };
            let pre_cpu_prefetch_calls = if count1_hybrid_cap.is_some() {
                resident
                    .as_ref()
                    .ok_or("CPU prefetch owner absent")?
                    .native_cpu_mapped_prefetch(&executor)?
                    .ok_or("CPU prefetch operator absent")?
                    .1
            } else {
                0
            };
            let pre_metadata_calls = resident_util::metadata_calls();
            let pre_scratch_clear_calls = resident_util::native_scratch_clear_calls();
            let reclaim_start = Instant::now();
            let gpu_source_advised_bytes = if reclaim_gpu_sources {
                resident
                    .as_ref()
                    .ok_or("GPU source cache hint requires complete resident model")?
                    .advise_gpu_sources_not_needed(&executor, pageout_gpu_sources)?
            } else {
                0
            };
            let gpu_source_advice_ms = reclaim_start.elapsed().as_millis();
            let start = Instant::now();
            let mut generated = Vec::new();
            // Count actual successful generation continuations only. Prompt
            // chunks, cancellation attempts and fresh replay are excluded.
            let mut native_decode_steps = 0u64;
            let mut profiled_native_decode_steps = 0u64;
            let mut scalar_decode_steps = 0u64;
            let resources_before = public_resources::snapshot();
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
                    struct PausedProfileOnDrop<'a>(&'a cubecl::prelude::ComputeClient<WgpuRuntime>);
                    impl Drop for PausedProfileOnDrop<'_> {
                        fn drop(&mut self) {
                            self.0.set_profiling_paused(true);
                        }
                    }
                    let profile_guard = if kernel_profile && n == 0 {
                        client.set_profiling_paused(false);
                        Some(PausedProfileOnDrop(&client))
                    } else {
                        None
                    };
                    logits.zeroize();
                    logits = if let Some(r) = &mut resident {
                        let mut output =
                            resident_decode_step(r, &mut executor, token, &cancel, native_decode)?;
                        if native_decode {
                            native_decode_steps += 1;
                        } else {
                            scalar_decode_steps += 1;
                        }
                        std::mem::take(&mut *output)
                    } else {
                        let output =
                            model.step(&mut executor, &mut sequence, token, &cancel, |_| {})?;
                        scalar_decode_steps += 1;
                        output
                    };
                    if profile_guard.is_some() {
                        profiled_native_decode_steps += 1;
                    }
                    drop(profile_guard);
                }
            }
            let warm_resources =
                public_resources::delta(resources_before, public_resources::snapshot());
            let observed_cpu_buffered = if count1_hybrid_cap.is_some() {
                let (enabled, calls) = resident
                    .as_ref()
                    .ok_or("CPU buffered owner absent")?
                    .native_cpu_buffered_weights(&executor)?
                    .ok_or("CPU buffered operator absent")?;
                (
                    enabled,
                    calls
                        .checked_sub(pre_cpu_buffered_reads)
                        .ok_or("CPU buffered counter overflow")?,
                )
            } else {
                (false, 0)
            };
            let observed_cpu_prefetch = if count1_hybrid_cap.is_some() {
                let (enabled, calls) = resident
                    .as_ref()
                    .ok_or("CPU prefetch owner absent")?
                    .native_cpu_mapped_prefetch(&executor)?
                    .ok_or("CPU prefetch operator absent")?;
                (
                    enabled,
                    calls
                        .checked_sub(pre_cpu_prefetch_calls)
                        .ok_or("CPU prefetch counter overflow")?,
                )
            } else {
                (false, 0)
            };
            if let Some(r) = &resident
                && native_max8
            {
                observed_native_decode_kernels = r.native_decode_kernels()?.label();
            }
            if observed_native_decode_kernels != native_decode_kernels {
                return Err("completed native decode kernel observation mismatch".into());
            }
            let completed_cpu_misses = if let Some(cap) = count1_hybrid_cap {
                let r = resident.as_ref().ok_or("count1 hybrid owner absent")?;
                let completed = r.completed_cpu_misses(&executor)?;
                if r.native_cpu_miss_cap()? != Some(cap)
                    || completed > native_decode_steps * 480
                    || (cap == 0 && completed != 0)
                    || (cap > 0 && completed == 0)
                {
                    return Err("count1 hybrid did not complete actual CPU miss work".into());
                }
                Some(completed)
            } else {
                None
            };
            let mut fresh_completed_cpu_misses = None;
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
            if kernel_profile {
                client.sync().await?;
            }
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
                receipt["fused_matrix_fault"] = serde_json::json!(executor.fused_matrix_fault);
                receipt["mapped_weights"] = serde_json::json!(mapped_weights);
                receipt["count1_bank_profile"] = serde_json::json!(count1_banks.label());
                receipt["observed_mapped_weights"] = serde_json::json!(file.weights_mapped());
                receipt["kernel_profile"] = serde_json::json!(kernel_profile);
                receipt["kernel_profile_scope"] = serde_json::json!(if kernel_profile {
                    "first-following-decode"
                } else {
                    "disabled"
                });
                receipt["profiled_native_decode_steps"] =
                    serde_json::json!(profiled_native_decode_steps);
                receipt["observed_shared_overlap_calls"] =
                    serde_json::json!(executor.shared_overlap_calls);
                receipt["gpu_router"] = serde_json::json!(gpu_router);
                receipt["retained_metadata"] = serde_json::json!(retained_metadata);
                receipt["observed_retained_metadata_calls"] = serde_json::json!(
                    resident_util::metadata_calls()
                        .checked_sub(pre_metadata_calls)
                        .ok_or("metadata counter regressed")?
                );
                receipt["observed_gpu_router_calls"] = serde_json::json!(executor.gpu_router_calls);
                receipt["fast_gpu"] = serde_json::json!(fast_gpu);
                receipt["next_shared_kernels"] = serde_json::json!(next_shared);
                receipt["next_shared_gdn"] = serde_json::json!(shared_gdn);
                receipt["observed_shared_gdn_calls"] = serde_json::json!(
                    resident
                        .as_ref()
                        .ok_or("missing native shared GDN owner")?
                        .shared_gdn_calls()?
                );
                receipt["observed_shared_dense_calls"] =
                    serde_json::json!(executor.shared_dense_calls);
                receipt["observed_fast_dense_calls"] = serde_json::json!(executor.fast_dense_calls);
                receipt["fast_gpu_fault"] = serde_json::json!(fast_gpu_fault);
                receipt["observed_fast_dense_fault_calls"] =
                    serde_json::json!(executor.fast_dense_fault_calls);
                receipt["long_fixture"] = serde_json::json!(long_fixture);
                receipt["cpu_q8_activations"] = serde_json::json!(cpu_q8);
                receipt["cpu_q8_residual"] = serde_json::json!(cpu_q8_residual);
                receipt["cpu_q8_two_pass"] = serde_json::json!(cpu_q8_two_pass);
                receipt["cpu_threads"] = serde_json::json!(cpu_threads);
                if count1_hybrid_cap.is_some() {
                    let (threads, passes) = resident
                        .as_ref()
                        .ok_or("CPU arithmetic owner absent")?
                        .native_cpu_arithmetic_geometry(&executor)?
                        .ok_or("CPU arithmetic operator absent")?;
                    receipt["observed_cpu_threads"] = serde_json::json!(threads);
                    receipt["observed_cpu_q8_residual_passes"] = serde_json::json!(passes);
                }
                receipt["cpu_buffered_weights"] = serde_json::json!(cpu_buffered_weights);
                receipt["observed_cpu_buffered_weights"] =
                    serde_json::json!(observed_cpu_buffered.0);
                if count1_hybrid_cap.is_some() {
                    let current = resident
                        .as_ref()
                        .ok_or("CPU phase owner absent")?
                        .native_cpu_phase_ns(&executor)?
                        .ok_or("CPU phases absent")?;
                    let mut delta = [0u64; 4];
                    for i in 0..4 {
                        delta[i] = current[i]
                            .checked_sub(pre_cpu_phase_ns[i])
                            .ok_or("CPU phase counter regression")?;
                    }
                    receipt["warm_cpu_operator_host_phase_ns"] = serde_json::json!(delta);
                }
                receipt["warm_cpu_buffered_reads"] = serde_json::json!(observed_cpu_buffered.1);
                receipt["cpu_mapped_prefetch"] = serde_json::json!(cpu_mapped_prefetch);
                receipt["observed_cpu_mapped_prefetch"] =
                    serde_json::json!(observed_cpu_prefetch.0);
                receipt["warm_cpu_prefetch_calls"] = serde_json::json!(observed_cpu_prefetch.1);
                receipt["cpu_fma_arithmetic"] = serde_json::json!(cfg!(feature = "cpu-fma"));
                receipt["warm_host_resource_delta"] = serde_json::json!(warm_resources);
                receipt["reclaim_gpu_sources"] = serde_json::json!(reclaim_gpu_sources);
                receipt["pageout_gpu_sources"] = serde_json::json!(pageout_gpu_sources);
                receipt["gpu_source_mapping_advice"] = serde_json::json!(if pageout_gpu_sources {
                    "discard-clean-file-ptes"
                } else if reclaim_gpu_sources {
                    "descriptor-only"
                } else {
                    "disabled"
                });
                receipt["gpu_source_advised_bytes"] = serde_json::json!(gpu_source_advised_bytes);
                receipt["gpu_source_advice_ms"] = serde_json::json!(gpu_source_advice_ms);
                receipt["large_weight_cache"] = serde_json::json!(large_weight_cache);
                receipt["native_scratch_clear"] = serde_json::json!(native_scratch_clear);
                receipt["observed_native_scratch_clear_calls"] = serde_json::json!(
                    resident_util::native_scratch_clear_calls()
                        .checked_sub(pre_scratch_clear_calls)
                        .ok_or("scratch clear counter regression")?
                );
                receipt["compact_info"] = serde_json::json!(compact_info);
                receipt["info_batch_upload"] = serde_json::json!(info_batch_upload);
                receipt["device_dispatch"] = serde_json::json!(device_dispatch);
                receipt["observed_inline_device_dispatch"] = serde_json::json!(inline_dispatch);
                receipt["info_slab_bytes"] = serde_json::json!(if compact_info { 4096 } else { 0 });
                receipt["batch_tasks"] = serde_json::json!(batch_tasks);
                receipt["defer_router_io"] = serde_json::json!(defer_router_io);
                receipt["runtime_tasks_max"] =
                    serde_json::json!(if batch_tasks { 256 } else { 32 });
                if long_fixture {
                    receipt["fixture"] = serde_json::json!("resident-count1-hybrid-rainwater");
                }
                receipt["word_down"] = serde_json::json!(executor.word_down);
                receipt["resident_main_admitted_upper_bytes"] =
                    serde_json::json!(executor.admission_upper_bytes()?);
                receipt["prefill_last_head"] = serde_json::json!(prefill_last_head);
                if let Some(cap) = count1_hybrid_cap {
                    let (observed, reads) = resident
                        .as_ref()
                        .ok_or("count1 hybrid owner absent")?
                        .native_deferred_router_reads(&executor)?;
                    receipt["observed_defer_router_io"] = serde_json::json!(observed);
                    receipt["observed_deferred_router_reads"] = serde_json::json!(reads);
                    receipt["observed_cpu_q8_residual"] = serde_json::json!(
                        resident
                            .as_ref()
                            .ok_or("count1 hybrid owner absent")?
                            .native_cpu_q8_residual(&executor)?
                            .ok_or("count1 hybrid arithmetic absent")?
                    );
                    receipt["observed_cpu_q8_activations"] = serde_json::json!(
                        resident
                            .as_ref()
                            .ok_or("count1 hybrid owner absent")?
                            .native_cpu_q8_activations(&executor)?
                            .ok_or("count1 hybrid arithmetic absent")?
                    );
                    receipt["count1_hybrid_cpu_miss_cap"] = serde_json::json!(cap);
                    receipt["observed_count1_hybrid_cpu_miss_cap"] = serde_json::json!(
                        resident
                            .as_ref()
                            .ok_or("count1 hybrid owner absent")?
                            .native_cpu_miss_cap()?
                    );
                    receipt["completed_cpu_misses"] = serde_json::json!(completed_cpu_misses);
                    receipt["count1_hybrid_cpu_gpu_overlap"] = serde_json::json!(count1_overlap);
                    receipt["observed_count1_hybrid_cpu_gpu_overlap"] = serde_json::json!(
                        resident
                            .as_ref()
                            .ok_or("count1 hybrid owner absent")?
                            .native_cpu_gpu_overlap(&executor)?
                            .ok_or("count1 hybrid schedule absent")?
                    );
                    receipt["count1_hybrid_cpu_workers"] =
                        serde_json::json!(if count1_pooled { "pooled" } else { "scoped" });
                    receipt["observed_count1_hybrid_cpu_workers"] = serde_json::json!(if resident
                        .as_ref()
                        .ok_or("count1 hybrid owner absent")?
                        .native_cpu_workers_pooled(&executor)?
                        .ok_or("count1 hybrid workers absent")?
                    {
                        "pooled"
                    } else {
                        "scoped"
                    });
                }
                receipt["native_max8"] = serde_json::json!(native_max8);
                receipt["native_decode"] = serde_json::json!(native_decode);
                receipt["native_decode_steps"] = serde_json::json!(native_decode_steps);
                receipt["scalar_decode_steps"] = serde_json::json!(scalar_decode_steps);
                receipt["native_decode_kernels"] = serde_json::json!(native_decode_kernels);
                receipt["observed_native_decode_kernels"] =
                    serde_json::json!(observed_native_decode_kernels);
                receipt["native_public_chunk_count"] = serde_json::json!(if native_max8 {
                    Some(native_count)
                } else {
                    None
                });
                receipt["native_public_chunk_partition"] = serde_json::json!(if native_max8 {
                    Some(
                        ids.chunks(native_count)
                            .map(|chunk| chunk.len())
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                });
                receipt["native_public_max_batch"] = serde_json::json!(8);
                receipt["native_projection_tile"] =
                    serde_json::json!(native_prefill_projection_tile);
                receipt["native_decode_projection_tile"] =
                    serde_json::json!(native_decode_projection_tile);
                receipt["resident_transient_io"] = serde_json::json!(transient_resident);
                receipt["resident_owned_zero_audit_enabled"] =
                    serde_json::json!(public_owned_zero_audit);
                receipt["io_token_checkpoint"] =
                    serde_json::json!(token_checkpoint || defer_router_io);
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
                receipt["expert_bank_segments"] =
                    serde_json::json!(if crate::bank_slots::segmented_capacity(expert_slots) {
                        bank_slots::segmented_capacities(expert_slots)
                            .ok_or("original segment capacities")?
                            .to_vec()
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
                if resident_decode_step(r, &mut executor, ids[0], &cancel, native_decode).is_ok()
                    || !r.erased_after_failure()
                {
                    return Err("resident cancellation cleanup failed".into());
                }
                cancel.store(false, std::sync::atomic::Ordering::Release);
                if resident_decode_step(r, &mut executor, ids[0], &cancel, native_decode).is_ok() {
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
                executor.gpu_router = gpu_router;
                executor.fast_dense = fast_gpu;
                executor.fast_dense_fault = fast_gpu_fault;
                executor.shared_kernels = next_shared;
                executor.shared_gdn = shared_gdn;
                executor.token_io_checkpoint = token_checkpoint || defer_router_io;
                let mut fresh = resident_model::ResidentModel::new_for_lab_fixture(
                    model.bound_graph(),
                    client.clone(),
                    ids.len() + limit,
                    &mut executor,
                    public_owned_zero_audit,
                    native_config.clone(),
                )?;
                let mut fresh_logits = zeroize::Zeroizing::new(Vec::new());
                if native_max8 {
                    let chunks = ids.len().div_ceil(native_count);
                    for (n, tokens) in ids.chunks(native_count).enumerate() {
                        let purpose = if n + 1 == chunks {
                            resident_model::batch::Purpose::LastLogits
                        } else {
                            resident_model::batch::Purpose::Advance
                        };
                        fresh_logits =
                            fresh.prefill_chunk(&mut executor, tokens, purpose, &cancel)?;
                    }
                } else {
                    for (n, token) in ids.iter().enumerate() {
                        if prefill_last_head && n + 1 < ids.len() {
                            fresh.advance_without_logits(&mut executor, *token, &cancel)?;
                        } else {
                            fresh_logits = fresh.step(&mut executor, *token, &cancel)?;
                        }
                    }
                }
                if greedy(&fresh_logits)? != generated[0] {
                    return Err("fresh resident sequence mismatch".into());
                }
                let tile1_recovery = native_decode && native_decode_tile == 1;
                if tile1_recovery {
                    if fresh.set_lab_native_projection_tile(&mut executor, 1)? != 1 {
                        return Err("fresh native tile1 policy mismatch".into());
                    }
                    apply_native_decode_kernels(
                        &mut fresh,
                        &mut executor,
                        native_decode_kernels,
                        &cancel,
                    )?;
                    apply_count1_hybrid_cap(&mut fresh, &mut executor, count1_hybrid_cap, &cancel)?;
                    fresh_logits = resident_decode_step(
                        &mut fresh,
                        &mut executor,
                        generated[0],
                        &cancel,
                        true,
                    )?;
                    if greedy(&fresh_logits)?
                        != *generated
                            .get(1)
                            .ok_or("native tile1 replay requires a second public output")?
                    {
                        return Err("fresh native tile1 continuation mismatch".into());
                    }
                    if fresh.native_decode_kernels()?.label() != native_decode_kernels {
                        return Err("fresh native kernel continuation policy mismatch".into());
                    }
                }
                if let Some(cap) = count1_hybrid_cap {
                    let completed = fresh.completed_cpu_misses(&executor)?;
                    if fresh.native_cpu_miss_cap()? != Some(cap)
                        || completed > 480
                        || (cap == 0 && completed != 0)
                        || (cap > 0 && completed == 0)
                    {
                        return Err("fresh count1 hybrid actual completion missing".into());
                    }
                    fresh_completed_cpu_misses = Some(completed);
                    println!("public_count1_hybrid_fresh_continuation cap={cap} matches=true");
                }
                if public_owned_zero_audit {
                    fresh.close()?;
                    report_public_owned_zero_audit(&fresh, true, &mut owned_zero_audits)?;
                }
                if tile1_recovery {
                    println!("public_native_tile1_fresh_continuation_matches=true");
                    if native_decode_kernels != "chunk" {
                        println!(
                            "public_native_kernel_fresh_continuation profile={native_decode_kernels} matches=true"
                        );
                    }
                }
                drop(fresh);
                for &layer in if defer_router_io {
                    &[0, 3, 47][..]
                } else {
                    &[0, 3][..]
                } {
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
                    executor.gpu_router = gpu_router;
                    executor.fast_dense = fast_gpu;
                    executor.fast_dense_fault = fast_gpu_fault;
                    executor.shared_kernels = next_shared;
                    executor.shared_gdn = shared_gdn;
                    executor.token_io_checkpoint = token_checkpoint || defer_router_io;
                    let mut partial = resident_model::ResidentModel::new_for_lab_fixture(
                        model.bound_graph(),
                        client.clone(),
                        ids.len() + limit,
                        &mut executor,
                        public_owned_zero_audit,
                        native_config.clone(),
                    )?;
                    if tile1_recovery {
                        // Complete a default-tile8 native transaction before
                        // selecting the public tile1 continuation policy.
                        partial.prefill_chunk(
                            &mut executor,
                            &ids[..1],
                            resident_model::batch::Purpose::LastLogits,
                            &cancel,
                        )?;
                        if partial.set_lab_native_projection_tile(&mut executor, 1)? != 1 {
                            return Err("partial native tile1 policy mismatch".into());
                        }
                        apply_native_decode_kernels(
                            &mut partial,
                            &mut executor,
                            native_decode_kernels,
                            &cancel,
                        )?;
                    }
                    apply_count1_hybrid_cap(
                        &mut partial,
                        &mut executor,
                        count1_hybrid_cap,
                        &cancel,
                    )?;
                    partial.arm_public_mid_token_cancel(layer)?;
                    let failed = if tile1_recovery {
                        resident_decode_step(
                            &mut partial,
                            &mut executor,
                            generated[0],
                            &cancel,
                            true,
                        )
                        .is_err()
                    } else if native_max8 {
                        partial
                            .prefill_chunk(
                                &mut executor,
                                &ids[..ids.len().min(native_count)],
                                resident_model::batch::Purpose::LastLogits,
                                &cancel,
                            )
                            .is_err()
                    } else if prefill_last_head && ids.len() > 1 {
                        partial
                            .advance_without_logits(&mut executor, ids[0], &cancel)
                            .is_err()
                    } else {
                        partial.step(&mut executor, ids[0], &cancel).is_err()
                    };
                    if !failed || !partial.erased_after_failure() {
                        return Err("partially queued resident cancellation failed".into());
                    }
                    if tile1_recovery && !cancel.load(std::sync::atomic::Ordering::Acquire) {
                        return Err(
                            "native tile1 cancellation did not reach the requested layer".into(),
                        );
                    }
                    if let Some(cap) = count1_hybrid_cap {
                        if partial.completed_cpu_misses(&executor).is_ok() {
                            return Err("failed count1 hybrid exposed completion".into());
                        }
                        println!(
                            "public_count1_hybrid_mid_token_cancel cap={cap} layer={layer} erased=true"
                        );
                    }
                    report_public_owned_zero_audit(
                        &partial,
                        public_owned_zero_audit,
                        &mut owned_zero_audits,
                    )?;
                    cancel.store(false, std::sync::atomic::Ordering::Release);
                    println!("public_mid_token_cancel layer={layer} erased=true");
                    if tile1_recovery {
                        println!("public_native_tile1_mid_token_cancel layer={layer} erased=true");
                        if native_decode_kernels != "chunk" {
                            println!(
                                "public_native_kernel_mid_token_cancel profile={native_decode_kernels} layer={layer} erased=true"
                            );
                        }
                    }
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
                if count1_hybrid_cap.is_some() {
                    receipt["fresh_completed_cpu_misses"] = serde_json::json!(
                        fresh_completed_cpu_misses
                            .ok_or("count1 hybrid fresh completion missing")?
                    );
                }
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
    #[cfg(feature = "direct-device")]
    struct DirectTestService<const KIND: usize>(usize);
    #[cfg(feature = "direct-device")]
    impl<const KIND: usize> cubecl_common::device::DeviceService for DirectTestService<KIND> {
        fn init(_: cubecl_common::device::DeviceId) -> Self {
            Self(0)
        }
        fn utilities(&self) -> cubecl_common::device::ServerUtilitiesHandle {
            std::sync::Arc::new(())
        }
    }

    #[cfg(feature = "direct-device")]
    #[test]
    fn direct_handle_serializes_concurrent_callers_without_gpu_work() {
        use cubecl_common::{device::DeviceId, device_handle::DeviceHandle};
        let handle = DeviceHandle::<DirectTestService<1>>::new(DeviceId::new(65000, 1));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let handle = handle.clone();
                scope.spawn(move || {
                    for _ in 0..256 {
                        handle.submit(|service| service.0 += 1);
                    }
                });
            }
        });
        assert_eq!(handle.submit_blocking(|service| service.0).unwrap(), 2048);
    }

    #[cfg(feature = "direct-device")]
    #[test]
    fn direct_handle_retains_recursive_service_and_unwind_lock_semantics() {
        use cubecl_common::{device::DeviceId, device_handle::DeviceHandle};
        let device = DeviceId::new(65000, 2);
        let first = DeviceHandle::<DirectTestService<2>>::new(device);
        let second = DeviceHandle::<DirectTestService<3>>::new(device);
        first
            .submit_blocking(|service| {
                service.0 += 1;
                second.submit(|other| other.0 += 2);
            })
            .unwrap();
        assert_eq!(first.submit_blocking(|service| service.0).unwrap(), 1);
        assert_eq!(second.submit_blocking(|service| service.0).unwrap(), 2);
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            first.exclusive::<(), _>(|| panic!("public direct dispatch lock test"))
        }));
        assert!(rejected.is_err());
        first
            .exclusive(|| first.submit(|service| service.0 += 3))
            .unwrap();
        assert_eq!(first.submit_blocking(|service| service.0).unwrap(), 4);
    }

    #[test]
    fn direct_dispatch_requires_explicit_matching_protected_artifact() {
        let mode = "resident-count1-hybrid-chat";
        assert_eq!(
            super::count1_dispatch(mode, Some(10), None, false),
            Ok("threaded")
        );
        for cap in 0..=10 {
            assert_eq!(
                super::count1_dispatch(mode, Some(cap), Some("inline"), true),
                Ok("inline")
            );
        }
        for (mode, cap, raw, compiled) in [
            (mode, Some(10), None, true),
            (mode, Some(10), Some("threaded"), true),
            (mode, Some(10), Some("inline"), false),
            (mode, Some(11), Some("inline"), true),
            (mode, None, Some("inline"), true),
            ("resident-chat", Some(10), Some("inline"), true),
            (mode, Some(10), Some("unknown"), false),
        ] {
            assert!(super::count1_dispatch(mode, cap, raw, compiled).is_err());
        }
        assert_eq!(
            cubecl_common::device_handle::DeviceHandle::<
                <cubecl::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::Server,
            >::is_blocking(),
            cfg!(feature = "direct-device")
        );
    }
    #[test]
    fn expanded_banks_require_an_exact_intel_hybrid_profile_and_preserve_other_guards() {
        use super::Count1Banks as B;
        for (raw, slots) in [
            ("intel72", 72),
            ("intel96", 96),
            ("intel128", 128),
            ("intel144", 144),
            ("intel160", 160),
        ] {
            let p = B::parse("resident-count1-hybrid-chat", 0x8086, Some(raw)).unwrap();
            assert_eq!(p.slots(0x8086), slots);
            assert!(B::parse("resident-chat", 0x8086, Some(raw)).is_err());
            assert!(B::parse("resident-count1-hybrid-chat", 0x10de, Some(raw)).is_err());
            assert!(p.setting("KUNO_LAB_EXPERT_SLOTS", Some("96")).is_err());
            assert!(p.setting("KUNO_LAB_BATCH_EXPERTS", Some("1")).is_err());
        }
        assert!(B::parse("probe", 0x8086, Some("standard")).is_ok());
        for raw in ["", "96", "auto", "true", "Intel96"] {
            assert!(B::parse("resident-count1-hybrid-chat", 0x8086, Some(raw)).is_err());
        }
        assert!(B::Intel96.setting("KUNO_LAB_BLOCK_MOE", Some("1")).is_ok());
        assert!(
            B::Intel96
                .setting("KUNO_LAB_STATIC_CACHE_BYTES", Some("6442450944"))
                .is_ok()
        );
        for raw in [None, Some("0"), Some("8589934592")] {
            assert!(
                B::Intel96
                    .setting("KUNO_LAB_STATIC_CACHE_BYTES", raw)
                    .is_err()
            );
        }
        assert!(
            B::Intel72
                .setting("KUNO_LAB_STATIC_CACHE_BYTES", Some("6442450944"))
                .is_err()
        );
    }
    #[test]
    fn overlap_requires_explicit_pooled_hybrid_selection_before_device_access() {
        for mode in ["resident-count1-hybrid-chat", "resident-chat"] {
            for pooled in [false, true] {
                assert_eq!(super::count1_overlap(mode, pooled, None), Ok(false));
                for raw in ["0", "1", "", "auto", "true"] {
                    let result = super::count1_overlap(mode, pooled, Some(raw));
                    if mode == "resident-count1-hybrid-chat"
                        && (raw == "0" || (raw == "1" && pooled))
                    {
                        assert_eq!(result, Ok(raw == "1"));
                    } else {
                        assert!(result.is_err());
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_worker_selection_is_explicit_and_rejects_before_device_access() {
        for mode in [
            "resident-count1-hybrid-chat",
            "resident-chat",
            "resident-staged-hybrid-expert",
        ] {
            assert_eq!(super::count1_worker_pool(mode, None), Ok(false));
            for raw in ["pooled", "scoped", "", "1", "POOLED", "auto"] {
                let got = super::count1_worker_pool(mode, Some(raw));
                if mode == "resident-count1-hybrid-chat" && ["pooled", "scoped"].contains(&raw) {
                    assert_eq!(got, Ok(raw == "pooled"));
                } else {
                    assert!(got.is_err());
                }
            }
        }
    }

    #[test]
    fn count1_hybrid_rejects_unprotected_arguments_and_overrides_before_device_creation() {
        for argc in [0, 3, 4, 5] {
            for receipt in [false, true] {
                for cap in [
                    None,
                    Some("0"),
                    Some("10"),
                    Some("1"),
                    Some("2"),
                    Some("5"),
                    Some("9"),
                    Some("11"),
                    Some("01"),
                    Some(" 10"),
                    Some("10 "),
                ] {
                    assert_eq!(
                        super::count1_hybrid_registration(argc, receipt, cap).is_ok(),
                        argc == 4
                            && receipt
                            && matches!(
                                cap,
                                Some(
                                    "0" | "1"
                                        | "2"
                                        | "3"
                                        | "4"
                                        | "5"
                                        | "6"
                                        | "7"
                                        | "8"
                                        | "9"
                                        | "10"
                                )
                            )
                    );
                }
            }
        }
        for (name, accepted) in [
            ("KUNO_LAB_NATIVE_MAX8", "1"),
            ("KUNO_LAB_NATIVE_DECODE", "1"),
            ("KUNO_LAB_CANCEL_CHECK", "1"),
            ("KUNO_LAB_NATIVE_DECODE_TILE", "1"),
            ("KUNO_LAB_NATIVE_DECODE_KERNELS", "single-q4-all"),
        ] {
            assert!(super::count1_hybrid_setting(name, Some(accepted)).is_ok());
            for wrong in [None, Some("0"), Some("01"), Some("true"), Some("chunk")] {
                assert!(super::count1_hybrid_setting(name, wrong).is_err());
            }
        }
        for name in [
            "KUNO_LAB_EXPERT_SLOTS",
            "KUNO_LAB_STATIC_CACHE_BYTES",
            "KUNO_LAB_EXPERT_BATCH_BYTES",
            "KUNO_LAB_EXPERT_SCATTER",
        ] {
            assert!(super::count1_hybrid_setting(name, None).is_ok());
            assert!(super::count1_hybrid_setting(name, Some("0")).is_err());
        }
        assert!(super::count1_hybrid_setting("KUNO_LAB_NATIVE_BATCH_COUNT", Some("1")).is_err());
        assert!(super::count1_hybrid_setting("KUNO_LAB_SMALL_CPU", Some("1")).is_err());
        assert!(super::count1_hybrid_setting("unknown", None).is_err());
    }

    #[test]
    fn mapped_and_profile_flags_require_explicit_fixed_public_mode() {
        for mode in [
            "resident-count1-hybrid-chat",
            "resident-chat",
            "matrix-bench",
            "arbitrary",
        ] {
            assert!(!super::count1_opt_in(mode, None).unwrap());
            for raw in ["0", "1", "auto", "false", "", "2"] {
                assert_eq!(
                    super::count1_opt_in(mode, Some(raw)).is_ok(),
                    mode == "resident-count1-hybrid-chat" && matches!(raw, "0" | "1")
                );
            }
        }
    }
    #[test]
    fn native_decode_tile_requires_exact_public_opt_in() {
        for native in [false, true] {
            assert_eq!(super::native_decode_tile(None, native), Ok(8));
            assert_eq!(super::native_decode_tile(Some("8"), native), Ok(8));
            assert_eq!(super::native_decode_tile(Some("1"), native).is_ok(), native);
            for raw in ["", "0", "4", "01", "08", "true", " 1", "1 "] {
                assert!(super::native_decode_tile(Some(raw), native).is_err());
            }
        }
        assert_eq!(super::native_decode_tile(Some("1"), true), Ok(1));
    }

    #[test]
    fn native_decode_kernels_require_explicit_protected_tile_one() {
        for native in [false, true] {
            for tile in [1, 8] {
                for raw in [None, Some("chunk")] {
                    assert_eq!(
                        super::native_decode_kernels(raw, native, tile)
                            .unwrap()
                            .label(),
                        "chunk"
                    );
                }
                for profile in ["single-moe", "single-gdn", "single-all", "single-q4-all"] {
                    let selected = super::native_decode_kernels(Some(profile), native, tile);
                    assert_eq!(selected.is_ok(), native && tile == 1);
                    if let Ok(selected) = selected {
                        assert_eq!(selected.label(), profile);
                    }
                }
                for raw in [
                    "",
                    "0",
                    "1",
                    "CHUNK",
                    "single",
                    " single-all",
                    "single-all ",
                ] {
                    assert!(super::native_decode_kernels(Some(raw), native, tile).is_err());
                }
            }
        }
    }

    #[test]
    fn native_decode_is_explicit_and_requires_all_fixed_public_protections() {
        use super::native_decode_mode;
        for mode in [
            "resident-bang",
            "resident-chat",
            "resident-count1-hybrid-chat",
            "resident-rainwater",
            "resident-lifecycle-probe",
            "resident-numeric-boundaries-probe",
            "resident-cpu-gpu-expert-probe",
            "resident-staged-hybrid-expert",
            "resident-qsa-probe",
            "probe",
            "bang",
        ] {
            for native in [false, true] {
                for transient in [false, true] {
                    for audit in [false, true] {
                        for raw in [None, Some("0")] {
                            assert_eq!(
                                native_decode_mode(mode, raw, native, transient, audit),
                                Ok(false)
                            );
                        }
                        let permitted = matches!(
                            mode,
                            "resident-bang"
                                | "resident-chat"
                                | "resident-rainwater"
                                | "resident-count1-hybrid-chat"
                        ) && native
                            && transient
                            && audit;
                        assert_eq!(
                            native_decode_mode(mode, Some("1"), native, transient, audit).is_ok(),
                            permitted
                        );
                        for raw in ["", "true", "01", "2", " 1", "1 "] {
                            assert!(
                                native_decode_mode(mode, Some(raw), native, transient, audit)
                                    .is_err()
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            native_decode_mode("resident-chat", Some("1"), true, true, true),
            Ok(true)
        );
    }

    #[test]
    fn actual_model_diagnostics_require_both_ownership_and_full_audits() {
        for mode in [
            "resident-count1-hybrid-chat",
            "resident-lifecycle-probe",
            "resident-numeric-boundaries-probe",
            "resident-cpu-gpu-expert-probe",
            "resident-cpu-gpu-expert-calibration",
            "resident-staged-hybrid-expert",
        ] {
            assert_eq!(super::transient_resident_mode(mode, Some("1")), Ok(true));
            for transient in [false, true] {
                for audit in [false, true] {
                    assert_eq!(
                        super::protected_model_diagnostic(mode, transient, audit).is_ok(),
                        transient && audit
                    );
                }
            }
        }
    }
    #[test]
    fn compiled_expert_numeric_subset_keeps_all_ten_ranks_and_fixed_endpoints() {
        use crate::cpu_gpu_expert_schedule::{Mode, Split};
        use kuno_text::{TensorType, cpu_expert::Backend};
        let cases = super::cpu_gpu_expert_numeric_cases();
        let ids = std::array::from_fn(|rank| rank);
        for config in cases {
            assert!(config.backend == Backend::Avx2);
            assert_eq!(config.workers, 16);
            assert_eq!(config.misses, 10);
            assert_eq!(config.group, 0);
            assert_eq!(config.mode, Mode::Concurrent);
            let split = Split::new(&ids, config.misses, config.cpu_misses).unwrap();
            assert_eq!(split.cpu().len() + split.gpu().len(), 10);
            assert_eq!(split.cpu().len(), config.cpu_misses);
        }
        assert_eq!(cases.map(|c| c.cpu_misses), [0, 5, 5, 10]);
        assert_eq!(cases.map(|c| c.vector), [1, 1, 0, 1]);
        assert_eq!(
            cases.map(|c| c.down),
            [
                TensorType::Q5_0,
                TensorType::Q5_0,
                TensorType::Q8_0,
                TensorType::Q8_0
            ]
        );
    }

    fn completed_expert_numeric_report() -> crate::cpu_gpu_expert_probe::Report {
        use crate::cpu_gpu_expert_probe::{BANK_BYTES, PRIVATE_BYTES, Report, SOURCE_BYTES};
        Report {
            layer: 6,
            down_dtype: kuno_text::TensorType::Q5_0.gguf_id(),
            workers: 16,
            misses: 10,
            cpu_misses: 0,
            gpu_misses: 10,
            max_tree_error: 0.,
            max_legacy_error: 0.,
            max_core_error: 0.,
            private_pages: 18,
            private_bytes: PRIVATE_BYTES,
            bank_bytes: BANK_BYTES,
            source_precharge: SOURCE_BYTES,
            io_inspected_bytes: 32768,
            zero_buffers: 18,
            zero_bytes: PRIVATE_BYTES,
            timing: None,
            bridge_and_merge: None,
            metadata_bytes: 32768,
            snapshot_bytes: 32768,
        }
    }

    #[test]
    fn expert_numeric_publication_rejects_timing_incomplete_audits_and_nonfinite_errors() {
        use crate::cpu_gpu_expert_schedule::Times;
        let config = super::cpu_gpu_expert_numeric_cases()[0];
        let report =
            super::cpu_gpu_expert_numeric_report(config, completed_expert_numeric_report())
                .unwrap();
        assert_eq!(report["ranks"], 10);
        assert_eq!(report["owned_cleanup_accepted"], true);
        assert!(report.get("timing").is_none());
        for failure in 0..7 {
            let mut report = completed_expert_numeric_report();
            match failure {
                0 => report.zero_buffers -= 1,
                1 => report.zero_bytes -= 1,
                2 => report.io_inspected_bytes = 0,
                3 => report.max_tree_error = f32::NAN,
                4 => report.source_precharge -= 1,
                5 => {
                    report.timing = Some(Times {
                        cpu: std::time::Duration::ZERO,
                        gpu: std::time::Duration::ZERO,
                        joined: std::time::Duration::ZERO,
                    })
                }
                _ => report.bridge_and_merge = Some(std::time::Duration::ZERO),
            }
            assert!(super::cpu_gpu_expert_numeric_report(config, report).is_err());
        }
    }
    #[test]
    fn staged_hybrid_registration_has_no_variable_arguments_or_model_policies() {
        assert_eq!(super::staged_hybrid_registration(3, true), Ok(()));
        for argc in [0, 1, 2, 4, 5] {
            assert!(super::staged_hybrid_registration(argc, true).is_err());
        }
        assert!(super::staged_hybrid_registration(3, false).is_err());
        for name in super::STAGED_HYBRID_MODEL_SETTINGS {
            assert_eq!(super::staged_hybrid_setting(name, None), Ok(()));
            for raw in ["", "1", "true", "01", " 0", "0 ", "single-all"] {
                assert!(super::staged_hybrid_setting(name, Some(raw)).is_err());
            }
        }
        assert_eq!(
            super::staged_hybrid_setting("KUNO_LAB_NATIVE_DECODE_TILE", Some("8")),
            Ok(())
        );
        assert_eq!(
            super::staged_hybrid_setting("KUNO_LAB_NATIVE_DECODE_KERNELS", Some("chunk")),
            Ok(())
        );
        for name in [
            "KUNO_LAB_NATIVE_BATCH_COUNT",
            "KUNO_LAB_EXPERT_SLOTS",
            "KUNO_LAB_STATIC_CACHE_BYTES",
            "KUNO_LAB_EXPERT_BATCH_BYTES",
            "KUNO_LAB_EXPERT_SCATTER",
        ] {
            for raw in ["0", "8", "16", "64", "1048576"] {
                assert!(super::staged_hybrid_setting(name, Some(raw)).is_err());
            }
        }
        for name in ["KUNO_LAB_NATIVE_MAX8", "KUNO_LAB_BLOCK_MOE"] {
            assert_eq!(super::staged_hybrid_setting(name, Some("0")), Ok(()));
        }
        assert!(super::staged_hybrid_setting("KUNO_PRIVATE_INPUT", None).is_err());
    }
    fn completed_staged_hybrid_reports() -> crate::cpu_gpu_expert_probe::StagedHybridReport {
        crate::cpu_gpu_expert_probe::StagedHybridReport {
            cases: std::array::from_fn(|i| {
                let config = super::staged_hybrid_numeric_cases()[i];
                let mut report = completed_expert_numeric_report();
                report.layer = if i < 4 { 6 } else { 0 };
                report.down_dtype = config.down.gguf_id();
                report.misses = config.misses;
                report.cpu_misses = config.cpu_misses;
                report.gpu_misses = config.misses - config.cpu_misses;
                report.source_precharge = report.gpu_misses as u64 * 4 * 1024 * 1024;
                report
            }),
        }
    }
    #[test]
    fn staged_hybrid_publication_requires_fixed_inventory_without_timings_or_private_data() {
        let reports =
            super::staged_hybrid_numeric_reports(completed_staged_hybrid_reports()).unwrap();
        assert_eq!(reports.len(), 8);
        for (i, report) in reports.iter().enumerate() {
            assert_eq!(report["down_dtype"], if i < 4 { 6 } else { 8 });
            assert_eq!(report["misses"], if i % 4 == 3 { 10 } else { 5 });
            assert_eq!(report["cpu_misses"], [0, 1, 5, 10][i % 4]);
            assert_eq!(report["gpu_misses"], [5, 4, 0, 0][i % 4]);
            assert_eq!(
                report["source_precharge"],
                [20971520, 16777216, 0, 0][i % 4]
            );
            assert_eq!(report["group"], 0);
            assert_eq!(report["vector"], 1);
            assert_eq!(report["ranks"], 10);
            assert_eq!(report["owned_cleanup_accepted"], true);
            for field in [
                "timing",
                "timing_ns",
                "ids",
                "miss_mask",
                "slots",
                "map",
                "input",
                "partials",
            ] {
                assert!(report.get(field).is_none());
            }
        }
        let mut swapped = completed_staged_hybrid_reports();
        swapped.cases.swap(0, 1);
        assert!(super::staged_hybrid_numeric_reports(swapped).is_err());
        let mut swapped = completed_staged_hybrid_reports();
        swapped.cases.swap(0, 4);
        assert!(super::staged_hybrid_numeric_reports(swapped).is_err());
        for index in 0..8 {
            for failure in 0..16 {
                let mut reports = completed_staged_hybrid_reports();
                let report = &mut reports.cases[index];
                match failure {
                    0 => report.misses += 1,
                    1 => report.cpu_misses += 1,
                    2 => report.gpu_misses += 1,
                    3 => report.down_dtype = 0,
                    4 => report.zero_buffers -= 1,
                    5 => report.zero_bytes -= 1,
                    6 => report.source_precharge += 1,
                    7 => report.io_inspected_bytes = 0,
                    8 => {
                        report.io_inspected_bytes = crate::cpu_gpu_expert_probe::IO_COMPONENT_BYTES
                    }
                    9 => report.max_tree_error = f32::NAN,
                    10 => report.max_legacy_error = f32::INFINITY,
                    11 => report.metadata_bytes = 256 * 1024 + 1,
                    12 => report.snapshot_bytes = 2 * 1024 * 1024 + 1,
                    13 => report.bridge_and_merge = Some(std::time::Duration::ZERO),
                    14 => report.metadata_bytes = 0,
                    _ => report.snapshot_bytes = 0,
                }
                assert!(super::staged_hybrid_numeric_reports(reports).is_err());
            }
        }
    }
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
