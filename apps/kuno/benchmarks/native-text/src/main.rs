#![recursion_limit = "256"]
//! Public synthetic Rust-authored kernel probe, not a model engine wrapper.
mod adaptive_experts;
mod bank96_probe;
mod bank_slots;
pub mod cpu_gpu_expert_probe;
mod cpu_gpu_expert_schedule;
mod strata_cache;
mod strata_dense;
mod strata_expert_fused;
pub(crate) use strata_expert_fused::global as strata_global_experts;
mod strata_global_adapt;
#[allow(dead_code)] // Root consumers are the next connected port block.
mod strata_global_gpu;
#[allow(dead_code)] // Native coordinator connection follows this family.
mod strata_global_moe;
#[allow(dead_code)] // Global root/consumer connection continues as one port.
mod strata_global_plan;
#[allow(dead_code)] // Owning source set; coordinator connection follows.
mod strata_global_sources;
mod strata_gr;
mod strata_intel_core;
mod strata_packet;
mod strata_port;
mod strata_stage;
// Shared with the standalone CPU writer; writer-only methods are inert here.
#[allow(dead_code)]
mod daslab_carry_reference;
#[allow(dead_code)] // Separate numeric-only public CPU writer shares this parser.
mod daslab_carry_reference13;
mod daslab_language_probe;
mod daslab_profile;
mod daslab_speculative_probe;
mod daslab_target_repair_probe;
mod daslab_target_rows_probe;
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
mod mtp_gpu_probe;
mod mtp_lab_profile;
#[allow(dead_code)] // Shared numeric oracle writer also uses constructor/writer helpers.
mod mtp_public_reference;
mod mtp_resident_kernels;
mod next_shared;
mod next_shared_moe;
mod next_verify3_dense_probe;
mod next_verify4_dense_probe;
mod oracle_public;
mod oracle_trace;
mod ordered_public;
mod ple_batch_gpu;
mod ple_chunk;
mod ple_chunk_probe;
mod private_arena;
mod public_host_cpu_batch_probe;
#[allow(dead_code)] // Numeric/source admission helpers are public-lab-only.
mod public_host_expert_attachment;
#[allow(dead_code)]
mod public_host_expert_mirror;
#[allow(dead_code)] // Source-only mirror planner; physical consumer is pending.
mod public_host_expert_plan;
mod public_host_weight_cubecl_probe;
mod public_host_weight_pool;
mod public_host_weight_probe;
mod public_qsa_chronological_probe;
mod public_resources;
mod q3_ideal_public;
mod q3_public_reference;
pub mod q3_router_reference;
mod q6_integer_matvec;
mod q6_integer_probe;
mod qsa_chunk;
mod qsa_chunk_probe;
mod qsa_gpu;
mod qsa_layer_batch;
mod qsa_layer_batch_probe;
mod quant_matvec;
mod resident_batch_projection;
mod resident_batch_router_gpu;
mod resident_batch_router_gpu_probe;
mod resident_gdn;
mod resident_gr;
mod resident_model;
mod resident_moe;
mod resident_mtp;
mod resident_mtp_probe;
mod resident_probe;
mod resident_qsa;
mod resident_router_gpu;
mod resident_router_gpu_probe;
mod resident_util;
mod residual_batch_gpu;
mod residual_gpu;
mod router_gpu;
mod scratch_clear_batch;
mod secondary_experts;
mod segmented_iq_moe_probe;
mod small_cpu;
mod status_fold_batch;
mod steady_public;
mod strata_mtp_gpu_probe;
#[allow(dead_code)] // Shared independent public CPU oracle wire.
mod strata_mtp_public_reference;
mod target_lab_build;
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

fn resident_expert_calibration_mode(
    mode: &str,
    raw: Option<&str>,
    protected: bool,
) -> Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1")
            if mode == "resident-cpu-gpu-expert-calibration"
                && protected
                && cfg!(feature = "cpu-fma") =>
        {
            Ok(true)
        }
        _ => Err("Resident expert calibration requires protected compiled CPU FMA entry"),
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
fn resident_router_scope(
    enabled: bool,
    ideal: bool,
    profile: expert_banks::OriginalPackedProfile,
    positions: usize,
    protected: bool,
    arithmetic_compatible: bool,
) -> Result<(), &'static str> {
    if !enabled {
        return Ok(());
    }
    let exact_window = match profile {
        expert_banks::OriginalPackedProfile::UnslothQ3 => positions == 23,
        expert_banks::OriginalPackedProfile::DaslabIq3 => matches!(positions, 13 | 32),
        _ => false,
    };
    if !ideal || !protected || !arithmetic_compatible || !exact_window {
        return Err("resident GPU router requires exclusive supported protected ideal replay");
    }
    Ok(())
}

fn resident_router_probe_registration(
    mode: &str,
    argc: usize,
    gate_present: bool,
) -> Result<bool, &'static str> {
    if mode == "resident-router-gpu-probe" {
        if argc != 3 {
            return Err("resident mapper probe rejects additional arguments");
        }
        return Ok(true);
    }
    if gate_present {
        return Err("resident mapper gate requires its dedicated public probe");
    }
    Ok(false)
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
fn original_packed_profile(
    mode: &str,
    vendor: u32,
    raw: Option<&str>,
) -> Result<expert_banks::OriginalPackedProfile, &'static str> {
    use expert_banks::OriginalPackedProfile as P;
    match raw {
        None | Some("q4") => Ok(P::Q4),
        Some("unsloth-q3") if mode == "resident-count1-hybrid-chat" && vendor == 0x8086 => {
            Ok(P::UnslothQ3)
        }
        Some("daslab-iq3") if mode == "resident-count1-hybrid-chat" && vendor == 0x8086 => {
            Ok(P::DaslabIq3)
        }
        _ => Err("original packed Q3 requires the protected Intel count1 fixture"),
    }
}
fn q3_public_positions(
    mode: &str,
    profile: expert_banks::OriginalPackedProfile,
    protected: bool,
    raw: Option<&str>,
) -> Result<usize, &'static str> {
    match raw {
        None => Ok(4),
        Some(raw @ ("4" | "13" | "23" | "32"))
            if mode == "resident-count1-hybrid-chat"
                && profile.is_low_bit()
                && !matches!(
                    (profile, raw),
                    (expert_banks::OriginalPackedProfile::DaslabIq3, "23")
                        | (expert_banks::OriginalPackedProfile::UnslothQ3, "13")
                )
                && protected =>
        {
            raw.parse().map_err(|_| "public Q3 positions")
        }
        _ => Err(
            "public Q3 positions require protected original Q3 and canonical4, profile-specific13/23-prefix or32",
        ),
    }
}
fn q3_empty_slot_fill(
    mode: &str,
    profile: expert_banks::OriginalPackedProfile,
    protected: bool,
    raw: Option<&str>,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some(raw @ ("0" | "1"))
            if mode == "resident-count1-hybrid-chat" && profile.is_low_bit() && protected =>
        {
            Ok(raw == "1")
        }
        _ => Err("Q3 empty-slot fill requires explicit protected original Q3 and0 or1"),
    }
}
/// Explicit fixed-public speed experiments; no arbitrary request opt-in.
#[allow(clippy::too_many_arguments)]
fn q3_speed_fixture_option(
    mode: &str,
    profile: expert_banks::OriginalPackedProfile,
    protected: bool,
    positions: usize,
    empty_fill: bool,
    option: &'static str,
    raw: Option<&str>,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some("1")
            if mode == "resident-count1-hybrid-chat"
                && profile.is_low_bit()
                && protected
                && empty_fill
                && matches!(positions, 4 | 13 | 23 | 32)
                && matches!(
                    option,
                    "skip-segments"
                        | "subgroup-moe"
                        | "integer-moe"
                        | "shared-original-moe"
                        | "integer-head"
                        | "ideal-prefix"
                )
                && match option {
                    "skip-segments" => true,
                    "subgroup-moe" => profile == expert_banks::OriginalPackedProfile::UnslothQ3,
                    "integer-moe" | "shared-original-moe" | "integer-head" => matches!(
                        (profile, positions),
                        (expert_banks::OriginalPackedProfile::UnslothQ3, 4 | 23 | 32)
                            | (expert_banks::OriginalPackedProfile::DaslabIq3, 4 | 13 | 32)
                    ),
                    "ideal-prefix" => matches!(
                        (profile, positions),
                        (expert_banks::OriginalPackedProfile::UnslothQ3, 23)
                            | (expert_banks::OriginalPackedProfile::DaslabIq3, 13 | 32)
                    ),
                    _ => false,
                } =>
        {
            Ok(true)
        }
        _ => Err("Q3 speed diagnostic requires its explicit protected original fixture"),
    }
}

/// The GPU integer leaf changes activation arithmetic, not CPU policy. Existing
/// supported direct-device/FMA/worker builds remain usable; CPU Q8 and the older
/// Q4 fused/integer paths cannot be selected alongside this fixed experiment.
#[allow(clippy::too_many_arguments)]
fn q3_integer_moe_policy(
    enabled: bool,
    empty_fill: bool,
    skip_segments: bool,
    fast_gpu: bool,
    gpu_router: bool,
    next_shared: bool,
    subgroup_moe: bool,
    incompatible_arithmetic: bool,
) -> Result<(), &'static str> {
    if enabled
        && (!empty_fill
            || !skip_segments
            || !fast_gpu
            || !gpu_router
            || !next_shared
            || subgroup_moe
            || incompatible_arithmetic)
    {
        return Err(
            "Q3 integer MoE requires protected fast count1 empty/segment policy and exclusive original CPU arithmetic",
        );
    }
    Ok(())
}

fn shared_original_iq_probe_registration(
    mode: &str,
    args: usize,
    gate_present: bool,
) -> Result<bool, &'static str> {
    if mode == "iq-shared-original-moe-probe" {
        if args != 3 {
            return Err("fixed shared-original IQ probe rejects additional arguments");
        }
        Ok(true)
    } else if gate_present {
        Err("shared-original IQ gate requires its dedicated public probe")
    } else {
        Ok(false)
    }
}

fn live_member_iq_probe_registration(
    mode: &str,
    args: usize,
    gate_present: bool,
) -> Result<bool, &'static str> {
    if mode == "iq-live-members-moe-probe" {
        if args != 3 {
            return Err("fixed live-member IQ probe rejects additional arguments");
        }
        Ok(true)
    } else if gate_present {
        Err("live-member IQ gate requires its dedicated public probe")
    } else {
        Ok(false)
    }
}

fn q3_integer_moe_shared_limit(enabled: bool, available: u32) -> Result<(), &'static str> {
    if enabled && (available as usize) < quant_matvec::iq_moe_integer::MAX_SHARED_BYTES {
        return Err("Q3 integer MoE requires its complete workgroup storage limit");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn q3_request_binding_scope(
    extended: bool,
    mode: &str,
    profile: expert_banks::OriginalPackedProfile,
    protected: bool,
    positions: usize,
    empty_fill: bool,
    request_reuse: bool,
    binding_reuse: bool,
) -> Result<(), &'static str> {
    if !extended {
        return Ok(());
    }
    if mode != "resident-count1-hybrid-chat"
        || !protected
        || !empty_fill
        || !request_reuse
        || !binding_reuse
        || !matches!(
            (profile, positions),
            (expert_banks::OriginalPackedProfile::UnslothQ3, 4 | 23 | 32)
                | (expert_banks::OriginalPackedProfile::DaslabIq3, 4 | 13 | 32)
        )
    {
        return Err("larger binding cache requires protected fixed original Q3 request reuse");
    }
    Ok(())
}

fn q3_router_reference_path(
    expert_oracle: bool,
    raw: Option<std::ffi::OsString>,
) -> Result<Option<std::path::PathBuf>, &'static str> {
    match raw {
        None => Ok(None),
        Some(path) if expert_oracle && !path.is_empty() => Ok(Some(path.into())),
        _ => Err("fixed public Q3 router reference requires the expert diagnostic"),
    }
}

fn q3_expert_oracle(
    mode: &str,
    profile: expert_banks::OriginalPackedProfile,
    protected: bool,
    positions: usize,
    empty_fill: bool,
    raw: Option<&str>,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some("1")
            if mode == "resident-count1-hybrid-chat"
                && profile == expert_banks::OriginalPackedProfile::UnslothQ3
                && protected
                && positions == 32
                && empty_fill =>
        {
            Ok(true)
        }
        _ => Err("Q3 expert oracle requires explicit protected32-position empty-fill fixture"),
    }
}
fn original_count1_setting(
    profile: expert_banks::OriginalPackedProfile,
    banks: Count1Banks,
    name: &str,
    raw: Option<&str>,
) -> Result<(), &'static str> {
    if profile.is_low_bit() {
        if name == "KUNO_LAB_STATIC_CACHE_BYTES" {
            return if banks == Count1Banks::Intel144
                && matches!(raw, Some("5368709120" | "8589934592"))
            {
                Ok(())
            } else {
                Err("original Q3 cache requires explicit5 or8 GiB")
            };
        }
        let expected = match name {
            "KUNO_LAB_BLOCK_MOE" => Some("0"),
            "KUNO_LAB_NATIVE_DECODE_KERNELS" => Some("single-all"),
            _ => None,
        };
        if let Some(expected) = expected {
            return if banks == Count1Banks::Intel144 && raw == Some(expected) {
                Ok(())
            } else {
                Err("original Q3 requires explicit generic MoE and SingleAll")
            };
        }
    }
    banks.setting(name, raw)
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
            Self::Intel144 => crate::daslab_profile::BANK_SLOTS,
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
    request_io_reuse: bool,
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
    if request_io_reuse {
        resident.begin_lab_request_io_reuse(executor)?;
    }
    executor.enable_secondary_after_prefill()?;
    executor.enable_adaptive_after_prefill()?;
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
    if std::env::args().nth(1).as_deref() == Some("--validate-port-registration") {
        return daslab_language_probe::strata_registration_probe();
    }
    if std::env::args().nth(1).as_deref() == Some("--strata-license") {
        print!("{}", include_str!("strata-port-NOTICE.txt"));
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("--validate-port-shaders") {
        return strata_port::validate_shader_set();
    }
    let vendor = match std::env::args().nth(1).as_deref() {
        Some("intel") => 0x8086,
        Some("nvidia") => 0x10de,
        _ => return Err("expected intel or nvidia".into()),
    };
    let mode = std::env::args().nth(2).unwrap_or("probe".into());
    let public_host_cpu_batch_inputs = if mode == "public-host-cpu-batch-probe" {
        Some(public_host_cpu_batch_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_HOST_CPU_BATCH_PUBLIC_ONLY").is_some() {
            return Err("attached HOST CPU batch requires its dedicated public probe".into());
        }
        None
    };
    let public_host_cpu_overlap_inputs = if mode == "public-host-cpu-overlap-probe" {
        Some(public_host_expert_attachment::overlap_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_HOST_CPU_OVERLAP_PUBLIC_ONLY").is_some() {
            return Err("attached HOST CPU overlap requires its dedicated public probe".into());
        }
        None
    };
    let verify4_finite_probe = mode == "verify4-finite-probe";
    let verify4_dense_receipt = if mode == "verify4-dense-probe" || verify4_finite_probe {
        Some(next_verify4_dense_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled() == target_lab_build::Build::Inline,
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_VERIFY4_DENSE_PUBLIC_ONLY").is_some()
            || std::env::var_os("KUNO_LAB_VERIFY4_FINITE_PUBLIC_ONLY").is_some()
        {
            return Err("Verify4 operator public gate requires its dedicated probe".into());
        }
        None
    };
    let verify3_finite_probe = mode == "verify3-finite-probe";
    let verify3_dense_receipt = if mode == "verify3-dense-probe" || verify3_finite_probe {
        Some(next_verify3_dense_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            mtp_gpu_probe::default_build(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY").is_some()
            || std::env::var_os("KUNO_LAB_VERIFY3_FINITE_PUBLIC_ONLY").is_some()
        {
            return Err("Verify3 operator public gate requires its dedicated probe".into());
        }
        None
    };
    let public_host_weight_receipt = if mode == "public-host-weight-probe" {
        Some(public_host_weight_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled() != target_lab_build::Build::Rejected,
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_PUBLIC_HOST_WEIGHT_ONLY").is_some() {
            return Err("public host weights require their dedicated probe".into());
        }
        None
    };
    let public_host_cubecl_receipt = if mode == "public-host-weight-cubecl-probe" {
        Some(public_host_weight_cubecl_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled() != target_lab_build::Build::Rejected,
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_PUBLIC_HOST_CUBECL_ONLY").is_some() {
            return Err("public host CubeCL weights require their dedicated probe".into());
        }
        None
    };
    let qsa_chronological_receipt = if mode == "qsa-chronological-probe" {
        let receipt = public_qsa_chronological_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled() != target_lab_build::Build::Rejected,
        )?;
        resident_qsa::batch::chronological_probe::validate_configuration()?;
        Some(receipt)
    } else {
        if std::env::var_os("KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY").is_some() {
            return Err("QSA public gate requires its dedicated probe".into());
        }
        None
    };
    let batch_router_receipt = if mode == "batch-router-gpu-probe" {
        let selected: std::collections::BTreeMap<_, _> = std::env::vars_os()
            .filter(|(k, _)| {
                [b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
                    .iter()
                    .any(|prefix| k.as_encoded_bytes().starts_with(prefix))
            })
            .collect();
        if std::env::args_os().count() != 3
            || target_lab_build::Build::compiled() == target_lab_build::Build::Rejected
            || selected.len() != 2
            || selected.get(std::ffi::OsStr::new("KUNO_LAB_BATCH_ROUTER_PUBLIC_ONLY"))
                != Some(&"1".into())
            || selected
                .get(std::ffi::OsStr::new("KUNO_PUBLIC_RECEIPT"))
                .is_none_or(|v| v.is_empty())
        {
            return Err("batch router probe requires exclusive fixed public fixture".into());
        }
        Some(std::path::PathBuf::from(
            selected
                .get(std::ffi::OsStr::new("KUNO_PUBLIC_RECEIPT"))
                .ok_or("batch router receipt missing")?,
        ))
    } else {
        if std::env::var_os("KUNO_LAB_BATCH_ROUTER_PUBLIC_ONLY").is_some() {
            return Err("batch router gate requires dedicated probe".into());
        }
        None
    };
    let speculative_inputs = if mode == "daslab-speculative-probe" {
        Some(daslab_speculative_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY").is_some() {
            return Err("speculative public gate requires its dedicated probe".into());
        }
        None
    };
    let language_inputs = if mode == "daslab-language-probe" {
        Some(daslab_language_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").is_some()
            || std::env::var_os("KUNO_LAB_LANGUAGE_FIXTURE").is_some()
        {
            return Err("language public gate requires dedicated probe".into());
        }
        None
    };
    if std::env::var_os("KUNO_LAB_MTP_PROFILE").is_some()
        && speculative_inputs.is_none()
        && language_inputs.is_none()
    {
        return Err("draft profile requires its fixed public target controller".into());
    }
    if std::env::var_os("KUNO_LAB_TARGET_QSA").is_some()
        && speculative_inputs.is_none()
        && language_inputs.is_none()
    {
        return Err("QSA selector requires its fixed public target controller".into());
    }
    if std::env::var_os("KUNO_LAB_TARGET_HEAD").is_some()
        && speculative_inputs.is_none()
        && language_inputs.is_none()
    {
        return Err("target head selector requires its fixed public target controller".into());
    }
    if std::env::var_os("KUNO_LAB_TARGET_PROFILE").is_some() && speculative_inputs.is_none() {
        return Err("target profiling requires its fixed public warm controller".into());
    }
    let public_target_profile = speculative_inputs
        .as_ref()
        .is_some_and(|input| input.profile_first_warm_chunk)
        || (language_inputs.is_some() && strata_port::profile_target());
    let target_repair_inputs = if mode == "daslab-target-repair-probe" {
        Some(daslab_target_repair_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY").is_some() {
            return Err("target repair public gate requires its dedicated probe".into());
        }
        None
    };
    let strata_mtp_inputs = if mode == "strata-mtp-probe" {
        Some(strata_mtp_gpu_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled() != target_lab_build::Build::Rejected,
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_STRATA_MTP_PUBLIC_ONLY").is_some() {
            return Err("Q2 draft public gate requires its dedicated probe".into());
        }
        None
    };
    let resident_mtp_inputs = if mode == "resident-mtp-probe" {
        Some(resident_mtp_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            mtp_gpu_probe::default_build(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY").is_some() {
            return Err("resident MTP public gate requires its dedicated probe".into());
        }
        None
    };
    let target_rows_inputs = if mode == "daslab-target-rows-probe" {
        Some(daslab_target_rows_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            target_lab_build::Build::compiled(),
        )?)
    } else {
        if std::env::var_os("KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY").is_some() {
            return Err("target row public gate requires its dedicated probe".into());
        }
        None
    };
    let resident_router_probe = resident_router_probe_registration(
        &mode,
        std::env::args_os().count(),
        std::env::var_os("KUNO_LAB_RESIDENT_ROUTER_PUBLIC_ONLY").is_some(),
    )?;
    if resident_router_probe {
        resident_router_gpu_probe::validate_configuration()?;
    }
    let segmented_iq_probe = mode == "segmented-iq-moe-probe";
    let iq_integer_probe = mode == "iq-integer-moe-probe";
    let shared_original_iq_probe = shared_original_iq_probe_registration(
        &mode,
        std::env::args_os().count(),
        std::env::var_os("KUNO_LAB_IQ_SHARED_ORIGINAL_PUBLIC_ONLY").is_some(),
    )?;
    if shared_original_iq_probe {
        segmented_iq_moe_probe::validate_shared_original_configuration()?;
    }
    let live_member_iq_probe = live_member_iq_probe_registration(
        &mode,
        std::env::args_os().count(),
        std::env::var_os("KUNO_LAB_IQ_LIVE_MEMBERS_PUBLIC_ONLY").is_some(),
    )?;
    if live_member_iq_probe {
        segmented_iq_moe_probe::validate_live_members_configuration()?;
    }
    let q6_integer_probe = mode == "q6-integer-probe";
    if q6_integer_probe {
        if std::env::args_os().count() != 3 {
            return Err("fixed public Q6 integer probe rejects additional arguments".into());
        }
        q6_integer_probe::validate_configuration()?;
    } else if std::env::var_os("KUNO_LAB_Q6_INTEGER_PUBLIC_ONLY").is_some() {
        return Err("Q6 integer gate requires its dedicated public probe".into());
    }
    if iq_integer_probe {
        if std::env::args_os().count() != 3 {
            return Err("fixed public integer IQ probe rejects additional arguments".into());
        }
        segmented_iq_moe_probe::validate_integer_configuration()?;
    } else if std::env::var_os("KUNO_LAB_IQ_INTEGER_PUBLIC_ONLY").is_some() {
        return Err("integer IQ gate requires its dedicated public probe".into());
    }
    if segmented_iq_probe {
        if std::env::args_os().count() != 3 {
            return Err("fixed public segmented IQ probe rejects additional arguments".into());
        }
        segmented_iq_moe_probe::validate_configuration()?;
    } else if std::env::var_os("KUNO_LAB_SEGMENTED_IQ_PUBLIC_ONLY").is_some() {
        return Err("segmented IQ gate requires its dedicated public probe".into());
    }
    let mtp_gpu_inputs = if mode == "mtp-gpu-probe" {
        let inputs = mtp_gpu_probe::registration(
            &std::env::args_os().collect::<Vec<_>>(),
            std::env::vars_os(),
            mtp_gpu_probe::default_build(),
        )?;
        mtp_gpu_probe::content_free_panic_hook();
        Some(inputs)
    } else {
        None
    };
    let packed_raw = std::env::var_os("KUNO_LAB_ORIGINAL_PACKED");
    let packed_profile = original_packed_profile(
        &mode,
        vendor,
        packed_raw
            .as_ref()
            .map(|v| {
                v.to_str()
                    .ok_or("original packed profile encoding rejected")
            })
            .transpose()?,
    )?;
    let q3_main = packed_profile.is_low_bit();
    if mode == "iq-quant-probe" {
        matmul_probe::iq::registration(
            std::env::args().count(),
            std::env::var_os("KUNO_PUBLIC_RECEIPT").is_some(),
            std::env::var_os("KUNO_TEST_GGUF").is_some(),
            std::env::var_os("KUNO_LAB_MATMUL_TIMING").is_some(),
        )?;
    }
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
    let q3_positions_raw = std::env::var_os("KUNO_LAB_Q3_PUBLIC_POSITIONS");
    let q3_positions = q3_public_positions(
        &mode,
        packed_profile,
        transient_resident && public_owned_zero_audit,
        q3_positions_raw
            .as_ref()
            .map(|raw| raw.to_str().ok_or("public Q3 positions encoding rejected"))
            .transpose()?,
    )?;
    let q3_empty_raw = std::env::var_os("KUNO_LAB_Q3_EMPTY_SLOT_FILL");
    let q3_empty_slot_fill = q3_empty_slot_fill(
        &mode,
        packed_profile,
        transient_resident && public_owned_zero_audit,
        q3_empty_raw
            .as_ref()
            .map(|raw| raw.to_str().ok_or("Q3 empty-slot fill encoding rejected"))
            .transpose()?,
    )?;
    let speed_option = |name: &str, option| {
        let raw = std::env::var_os(name);
        q3_speed_fixture_option(
            &mode,
            packed_profile,
            transient_resident && public_owned_zero_audit,
            q3_positions,
            q3_empty_slot_fill,
            option,
            raw.as_ref()
                .map(|v| v.to_str().ok_or("Q3 speed option encoding rejected"))
                .transpose()?,
        )
    };
    let q3_skip_empty_segments = speed_option("KUNO_LAB_Q3_SKIP_EMPTY_SEGMENTS", "skip-segments")?;
    let q3_subgroup_moe = speed_option("KUNO_LAB_Q3_SUBGROUP_MOE", "subgroup-moe")?;
    let q3_integer_moe = speed_option("KUNO_LAB_Q3_INTEGER_MOE", "integer-moe")?;
    let q3_shared_original_moe =
        speed_option("KUNO_LAB_Q3_SHARED_ORIGINAL_MOE", "shared-original-moe")?;
    let q6_integer_head = speed_option("KUNO_LAB_Q6_INTEGER_HEAD", "integer-head")?;
    let q3_ideal_public = speed_option("KUNO_LAB_Q3_IDEAL_PUBLIC", "ideal-prefix")?;
    let resident_gpu_router = count1_environment_opt_in(&mode, "KUNO_LAB_RESIDENT_GPU_ROUTER")?;
    let resident_runtime_limit = if resident_gpu_router {
        executor::batch_moe::RESIDENT_ROUTER_RUNTIME_LIMIT
    } else {
        executor::RESIDENT_RUNTIME_IO_MAX
    };
    let q3_oracle_raw = std::env::var_os("KUNO_LAB_Q3_EXPERT_ORACLE");
    let q3_expert_oracle = q3_expert_oracle(
        &mode,
        packed_profile,
        transient_resident && public_owned_zero_audit,
        q3_positions,
        q3_empty_slot_fill,
        q3_oracle_raw
            .as_ref()
            .map(|raw| raw.to_str().ok_or("Q3 expert oracle encoding rejected"))
            .transpose()?,
    )?;
    // Fixed public numeric fixture only; no request data or production path.
    // Load and validate before opening model weights or creating a device.
    let q3_router_reference =
        q3_router_reference_path(q3_expert_oracle, std::env::var_os("KUNO_PUBLIC_Q3_ROUTERS"))?
            .map(q3_router_reference::Reference::load)
            .transpose()?
            .map(std::sync::Arc::new);
    let q3_reference = if q3_main {
        let reference = q3_public_reference::Reference::load(
            std::env::var_os("KUNO_PUBLIC_Q3_REFERENCE")
                .ok_or("fixed public Q3 scalar reference required")?,
        )?;
        reference.require_positions(q3_positions)?;
        Some(reference)
    } else {
        None
    };
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
    let resident_calibration_raw = std::env::var("KUNO_LAB_RESIDENT_EXPERT_CALIBRATION").ok();
    if std::env::var_os("KUNO_LAB_RESIDENT_EXPERT_CALIBRATION").is_some()
        && resident_calibration_raw.is_none()
    {
        return Err("Resident expert calibration setting encoding rejected".into());
    }
    let resident_expert_calibration = resident_expert_calibration_mode(
        &mode,
        resident_calibration_raw.as_deref(),
        transient_resident && public_owned_zero_audit,
    )?;
    let cpu_workers_raw = std::env::var_os("KUNO_LAB_CPU_WORKERS");
    let count1_pooled = count1_worker_pool(
        &mode,
        cpu_workers_raw
            .as_ref()
            .map(|v| v.to_str().ok_or("CPU worker setting encoding rejected"))
            .transpose()?,
    )?;
    let cpu_work_stealing = if resident_expert_calibration && cfg!(feature = "cpu-work-stealing") {
        if std::env::var("KUNO_LAB_CPU_WORK_STEALING").as_deref() != Ok("1") {
            return Err(
                "Resident calibration requires explicit compiled work-stealing selection".into(),
            );
        }
        true
    } else {
        count1_environment_opt_in(&mode, "KUNO_LAB_CPU_WORK_STEALING")?
    };
    if cpu_work_stealing != cfg!(feature = "cpu-work-stealing")
        || cpu_work_stealing && !count1_pooled && !resident_expert_calibration
    {
        return Err("Compiled CPU row scheduling requires explicit pooled count1 selection".into());
    }
    if cpu_work_stealing {
        println!("cpu_row_dispatch mode=work-stealing joined=true original_arithmetic=true");
    }
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
            original_count1_setting(packed_profile, count1_banks, name, value)?;
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
    let public_target_dispatch = speculative_inputs
        .as_ref()
        .map(|p| p.inline_dispatch)
        .or_else(|| target_repair_inputs.as_ref().map(|p| p.inline_dispatch))
        .or_else(|| target_rows_inputs.as_ref().map(|p| p.inline_dispatch))
        .or_else(|| language_inputs.as_ref().map(|p| p.inline_dispatch))
        .or_else(|| {
            public_host_cpu_batch_inputs
                .as_ref()
                .map(|p| p.inline_dispatch)
        })
        .or_else(|| {
            public_host_cpu_overlap_inputs
                .as_ref()
                .map(|p| p.inline_dispatch)
        })
        .or_else(|| verify4_dense_receipt.as_ref().map(|_| inline_dispatch))
        .or_else(|| strata_mtp_inputs.as_ref().map(|_| inline_dispatch))
        .or_else(|| public_host_weight_receipt.as_ref().map(|_| inline_dispatch))
        .or_else(|| public_host_cubecl_receipt.as_ref().map(|_| inline_dispatch))
        .or_else(|| batch_router_receipt.as_ref().map(|_| inline_dispatch))
        .or_else(|| qsa_chronological_receipt.as_ref().map(|_| inline_dispatch));
    let device_dispatch = if let Some(selected) = public_target_dispatch {
        target_lab_build::validate_runtime(selected)?;
        target_lab_build::name(selected)
    } else if resident_expert_calibration && inline_dispatch {
        if raw_dispatch.as_deref() != Some(std::ffi::OsStr::new("inline"))
            || count1_hybrid_cap.is_some()
        {
            return Err(
                "Resident calibration requires explicit matching inline device dispatch".into(),
            );
        }
        "inline"
    } else {
        count1_dispatch(
            &mode,
            count1_hybrid_cap,
            raw_dispatch
                .as_ref()
                .map(|v| v.to_str().ok_or("GPU dispatch setting encoding rejected"))
                .transpose()?,
            inline_dispatch,
        )?
    };
    if mtp_gpu_inputs.is_none() {
        println!("gpu_device_dispatch mode={device_dispatch} blocking={inline_dispatch}");
    }
    let kernel_profile = count1_environment_opt_in(&mode, "KUNO_LAB_KERNEL_PROFILE")?;
    let fast_gpu = count1_environment_opt_in(&mode, "KUNO_LAB_FAST_GPU")?;
    let fast_gpu_fault = count1_environment_opt_in(&mode, "KUNO_LAB_FAST_GPU_FAULT")?;
    let next_shared = count1_environment_opt_in(&mode, "KUNO_LAB_NEXT_SHARED")?;
    next_shared::select_gr(next_shared);
    let shared_gdn = count1_environment_opt_in(&mode, "KUNO_LAB_NEXT_SHARED_GDN")?;
    if shared_gdn && !next_shared {
        return Err("Shared GDN requires protected shared Next kernels".into());
    }
    if next_shared && (!fast_gpu || count1_hybrid_cap.is_none_or(|cap| cap > 10)) {
        return Err("Shared Next kernels require protected count1 fast GPU mode".into());
    }
    if fast_gpu_fault && !fast_gpu {
        return Err("Fused subgroup finite checks require the explicit fast GPU path".into());
    }
    let long_fixture = count1_environment_opt_in(&mode, "KUNO_LAB_LONG_FIXTURE")?;
    let steady_public = count1_environment_opt_in(&mode, "KUNO_LAB_STEADY_PUBLIC")?;
    let oracle_public = count1_environment_opt_in(&mode, "KUNO_LAB_ORACLE_PUBLIC")?;
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
        && (!kuno_text::cpu_expert::fma_available()
            || !resident_expert_calibration
                && (mode != "resident-count1-hybrid-chat" || !count1_pooled || cpu_q8))
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
    let compute_scratch_clear = count1_environment_opt_in(&mode, "KUNO_LAB_COMPUTE_SCRATCH_CLEAR")?;
    let compact_info = count1_environment_opt_in(&mode, "KUNO_LAB_COMPACT_INFO")?;
    let info_batch_upload = count1_environment_opt_in(&mode, "KUNO_LAB_INFO_BATCH_UPLOAD")?;
    if info_batch_upload && (!compact_info || count1_hybrid_cap.is_none_or(|cap| cap > 10)) {
        return Err("Batched INFO requires protected compact count1 mode".into());
    }
    let readback_reuse = count1_environment_opt_in(&mode, "KUNO_LAB_READBACK_REUSE")?;
    if readback_reuse && !info_batch_upload {
        return Err("readback reuse requires protected batched INFO".into());
    }
    let gpu_upload_clear = count1_environment_opt_in(&mode, "KUNO_LAB_GPU_UPLOAD_CLEAR")?;
    if gpu_upload_clear && !info_batch_upload {
        return Err("GPU upload clear requires protected batched INFO".into());
    }
    let inline_readback = count1_environment_opt_in(&mode, "KUNO_LAB_INLINE_READBACK")?;
    if inline_readback && (!info_batch_upload || !inline_dispatch) {
        return Err("inline readback requires protected batched INFO and inline dispatch".into());
    }
    let batch_tasks = count1_environment_opt_in(&mode, "KUNO_LAB_BATCH_TASKS")?;
    let gpu_router = count1_environment_opt_in(&mode, "KUNO_LAB_GPU_ROUTER")?;
    let fused_experts = count1_environment_opt_in(&mode, "KUNO_LAB_FUSED_EXPERTS")?;
    let secondary_experts = count1_environment_opt_in(&mode, "KUNO_LAB_SECONDARY_EXPERTS")?;
    if secondary_experts
        && (vendor != 0x8086
            || count1_banks != Count1Banks::Intel144
            || count1_hybrid_cap != Some(10)
            || !mapped_weights
            || !count1_pooled
            || !count1_overlap
            || !next_shared
            || !inline_dispatch
            || std::env::var("KUNO_LAB_REQUEST_IO_REUSE").as_deref() != Ok("1"))
    {
        return Err(
            "secondary experts require protected intel144 mapped cap10 pooled shared inline mode"
                .into(),
        );
    }
    let adaptive_experts = count1_environment_opt_in(&mode, "KUNO_LAB_ADAPTIVE_EXPERTS")?;
    let adaptive_direct_upload =
        count1_environment_opt_in(&mode, "KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD")?;
    if adaptive_direct_upload && !adaptive_experts {
        return Err("direct expert upload requires protected adaptive cache".into());
    }
    if adaptive_experts
        && (secondary_experts
            || vendor != 0x8086
            || count1_banks != Count1Banks::Intel144
            || count1_hybrid_cap != Some(10)
            || !mapped_weights
            || !count1_pooled
            || !count1_overlap
            || !next_shared
            || !inline_dispatch
            || std::env::var("KUNO_LAB_DEFER_ROUTER_IO").as_deref() != Ok("1")
            || std::env::var("KUNO_LAB_REQUEST_IO_REUSE").as_deref() != Ok("1"))
    {
        return Err(
            "adaptive experts require exclusive protected intel144 mapped cap10 mode".into(),
        );
    }
    let gpu_expert_limit = match std::env::var("KUNO_LAB_GPU_EXPERT_LIMIT") {
        Err(std::env::VarError::NotPresent) => 10,
        Ok(value) if mode == "resident-count1-hybrid-chat" => value
            .parse::<usize>()
            .ok()
            .filter(|&v| v <= 10)
            .ok_or("GPU expert limit must be 0 through 10")?,
        _ => return Err("GPU expert limit requires protected count1 mode".into()),
    };
    if adaptive_experts && gpu_expert_limit != 10 {
        return Err("adaptive requires GPU limit10".into());
    }
    if q3_empty_slot_fill
        && (count1_banks != Count1Banks::Intel144
            || count1_hybrid_cap != Some(10)
            || gpu_expert_limit != 10
            || !mapped_weights
            || !count1_pooled
            || !count1_overlap
            || adaptive_experts
            || adaptive_direct_upload
            || secondary_experts)
    {
        return Err(
            "Q3 empty-slot fill requires original intel144 mapped pooled overlap cap10".into(),
        );
    }
    if q3_subgroup_moe && (!q3_skip_empty_segments || !fast_gpu || !gpu_router || !next_shared) {
        return Err(
            "Q3 subgroup MoE requires protected original fast count1 and segment policy".into(),
        );
    }
    if gpu_expert_limit < 10
        && (count1_hybrid_cap != Some(10) || !mapped_weights || !count1_pooled || !count1_overlap)
    {
        return Err("GPU expert balancing requires mapped cap10 pooled overlap".into());
    }

    if steady_public
        && (!long_fixture
            || secondary_experts
            || vendor != 0x8086
            || count1_banks != Count1Banks::Intel144
            || count1_hybrid_cap != Some(10)
            || gpu_expert_limit != 10
            || !mapped_weights
            || !count1_pooled
            || !count1_overlap
            || !next_shared
            || !inline_dispatch
            || std::env::var("KUNO_LAB_DEFER_ROUTER_IO").as_deref() != Ok("1")
            || std::env::var("KUNO_LAB_REQUEST_IO_REUSE").as_deref() != Ok("1"))
    {
        return Err("steady diagnostic requires protected fixed public intel144 cap10 mode".into());
    }

    if oracle_public
        && (steady_public
            || adaptive_experts
            || adaptive_direct_upload
            || secondary_experts
            || !long_fixture
            || vendor != 0x8086
            || count1_banks != Count1Banks::Intel144
            || count1_hybrid_cap != Some(10)
            || gpu_expert_limit != 10
            || !mapped_weights
            || !count1_pooled
            || !count1_overlap
            || !next_shared
            || !inline_dispatch
            || std::env::var("KUNO_LAB_DEFER_ROUTER_IO").as_deref() != Ok("1")
            || std::env::var("KUNO_LAB_REQUEST_IO_REUSE").as_deref() != Ok("1"))
    {
        return Err(
            "oracle diagnostic requires exclusive protected fixed public intel144 cap10 mode"
                .into(),
        );
    }

    if q3_ideal_public
        && (steady_public
            || oracle_public
            || adaptive_experts
            || secondary_experts
            || count1_banks != Count1Banks::Intel144
            || count1_hybrid_cap != Some(10)
            || gpu_expert_limit != 10
            || !mapped_weights
            || !count1_pooled
            || !count1_overlap
            || !next_shared
            || !inline_dispatch
            || std::env::var("KUNO_LAB_REQUEST_IO_REUSE").as_deref() != Ok("1"))
    {
        return Err("Q3 ideal prefix requires exclusive original protected cap10 execution".into());
    }

    resident_router_scope(
        resident_gpu_router,
        q3_ideal_public,
        packed_profile,
        q3_positions,
        q3_main && transient_resident && public_owned_zero_audit,
        gpu_router
            && fast_gpu
            && !q3_subgroup_moe
            && (!q3_integer_moe || q3_skip_empty_segments)
            && !q3_shared_original_moe
            && !q6_integer_head
            && !q3_expert_oracle
            && !cpu_q8
            && !cpu_q8_residual
            && q3_empty_slot_fill,
    )?;
    let integer_experts = count1_environment_opt_in(&mode, "KUNO_LAB_INTEGER_EXPERTS")?;
    if resident_gpu_router && (fused_experts || integer_experts) {
        return Err("resident GPU routing excludes legacy fused expert arithmetic".into());
    }
    if integer_experts
        && (!fused_experts || gpu_expert_limit != 10 || count1_hybrid_cap != Some(10))
    {
        return Err("integer experts require fused cap10 kernels with GPU limit10".into());
    }
    if fused_experts && !next_shared {
        return Err("Fused experts require protected shared kernels".into());
    }
    if next_shared && !gpu_router {
        return Err("Shared Next kernels require GPU routing".into());
    }
    let retained_metadata = count1_environment_opt_in(&mode, "KUNO_LAB_RETAINED_METADATA")?;
    let grouped_status = count1_environment_opt_in(&mode, "KUNO_LAB_GROUPED_STATUS")?;
    if grouped_status && (!retained_metadata || !next_shared || !compact_info || !batch_tasks) {
        return Err("grouped status requires retained shared compact task batches".into());
    }

    if compute_scratch_clear
        && (native_scratch_clear
            || !retained_metadata
            || !next_shared
            || !compact_info
            || !batch_tasks)
    {
        return Err("compute scratch clear requires retained shared compact task batches without native clear".into());
    }

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
    let request_io_reuse = count1_environment_opt_in(&mode, "KUNO_LAB_REQUEST_IO_REUSE")?;
    let bind_group_reuse = count1_environment_opt_in(&mode, "KUNO_LAB_BIND_GROUP_REUSE")?;
    let binding_capacity_raw = std::env::var("KUNO_LAB_BIND_GROUP_CAPACITY");
    let binding_cache_capacity = match binding_capacity_raw.as_deref() {
        Ok("256") | Err(std::env::VarError::NotPresent) => 256,
        Ok("512") if bind_group_reuse => 512,
        Ok("2048") if bind_group_reuse => 2048,
        Ok("4096") if bind_group_reuse => 4096,
        _ => return Err("binding cache capacity requires explicit bounded request reuse".into()),
    };
    let metadata_raw = std::env::var("KUNO_LAB_REQUEST_REUSE_METADATA_BYTES");
    if matches!(metadata_raw, Err(std::env::VarError::NotUnicode(_))) {
        return Err("request reuse metadata encoding rejected".into());
    }
    let request_reuse_metadata_charge = cubecl::wgpu::transient_request_reuse_metadata_charge(
        bind_group_reuse,
        binding_capacity_raw.as_deref().ok(),
        metadata_raw.as_deref().ok(),
    )
    .map_err(|_| "request reuse metadata plan rejected")?;
    q3_request_binding_scope(
        binding_cache_capacity > 512 || request_reuse_metadata_charge > 262144,
        &mode,
        packed_profile,
        transient_resident && public_owned_zero_audit,
        q3_positions,
        q3_empty_slot_fill,
        request_io_reuse,
        bind_group_reuse,
    )?;
    let native_aggregate_host_plan = resident_model::batch::host_reservation_with_router(
        count1_hybrid_cap.is_some(),
        request_reuse_metadata_charge,
        resident_gpu_router,
    )?;
    if bind_group_reuse && !request_io_reuse {
        return Err("binding reuse requires request I/O reuse".into());
    }
    if request_io_reuse
        && (count1_hybrid_cap != Some(10)
            || !mapped_weights
            || !defer_router_io
            || !info_batch_upload
            || !inline_readback
            || gpu_upload_clear
            || kernel_profile)
    {
        return Err("request I/O reuse requires mapped cap10, deferred batched inline I/O without GPU-clear or kernel profiling".into());
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
    if kernel_profile || public_target_profile {
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
    expert_banks::validate_segmented_configuration_with_profile(
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
        packed_profile,
    )?;
    if q3_main {
        if q3_integer_moe && q3_shared_original_moe {
            return Err("Q3 expert arithmetic choices are exclusive".into());
        }
        q3_integer_moe_policy(
            q3_integer_moe || q3_shared_original_moe || q6_integer_head,
            q3_empty_slot_fill,
            q3_skip_empty_segments,
            fast_gpu,
            gpu_router,
            next_shared,
            q3_subgroup_moe,
            cpu_q8 || cpu_q8_residual || cpu_q8_two_pass || fused_experts || integer_experts,
        )?;
        packed_profile.validate_shortcuts(
            false,
            fused_experts,
            integer_experts,
            std::env::var("KUNO_LAB_WORD_DOWN").as_deref() == Ok("1"),
            std::env::var("KUNO_LAB_BLOCK_MOE").as_deref() == Ok("1"),
        )?;
        if count1_banks != Count1Banks::Intel144
            || !matches!(count1_hybrid_cap, Some(0 | 10))
            || !transient_resident
            || !public_owned_zero_audit
            || !mapped_weights
            || !native_max8
            || !native_decode
            || native_decode_tile != 1
            || native_decode_kernels != "single-all"
            || !count1_pooled
            || !count1_overlap
            || !large_weight_cache
            || long_fixture
            || steady_public
            || oracle_public
            || adaptive_experts
            || secondary_experts
            || cpu_q8
            || ordered_public_upload
        {
            return Err(
                "original Q3 requires its separate short mapped generic count1 fixture".into(),
            );
        }
    }
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
        // Admission precedes request_device; required_limits below requests the
        // complete admitted adapter limits, including this workgroup storage.
        q3_integer_moe_shared_limit(
            q3_integer_moe,
            adapter.limits().max_compute_workgroup_storage_size,
        )?;
        if q3_shared_original_moe
            && (adapter.limits().max_compute_workgroup_storage_size as usize)
                < quant_matvec::iq_moe_shared_original::MAX_SHARED_BYTES
        {
            return Err(
                "Q3 original shared MoE requires its complete workgroup storage limit".into(),
            );
        }
        if language_inputs.is_some() {
            daslab_language_probe::validate_limits(&adapter.limits())?;
        }
        if batch_router_receipt.is_some() {
            resident_batch_router_gpu::validate_limits(&adapter.limits())?;
        }
        if qsa_chronological_receipt.is_some()
            || speculative_inputs
                .as_ref()
                .is_some_and(|p| p.gpu_qsa_selector)
            || language_inputs.as_ref().is_some_and(|p| p.gpu_qsa_selector)
        {
            resident_qsa::batch::chronological_probe::validate_limits(&adapter.limits())?;
        }
        if public_host_weight_receipt.is_some() {
            public_host_weight_probe::validate_limits(&adapter.limits())?;
        }
        if public_host_cubecl_receipt.is_some() {
            public_host_weight_cubecl_probe::validate_limits(&adapter.limits())?;
        }
        if mtp_gpu_inputs.is_some() {
            mtp_gpu_probe::validate_limits(&adapter.limits())?;
        }
        if target_rows_inputs.is_some()
            || target_repair_inputs.is_some()
            || speculative_inputs.is_some()
            || public_host_cpu_batch_inputs.is_some()
            || public_host_cpu_overlap_inputs.is_some()
        {
            daslab_target_rows_probe::validate_limits(&adapter.limits())?;
        }
        if let Some(input) = &resident_mtp_inputs {
            resident_mtp::validate_limits(&adapter.limits())?;
            if input.fused_sweeps {
                resident_mtp::validate_fused_sweep_limits(&adapter.limits())?;
            }
        }
        if let Some(input) = &strata_mtp_inputs {
            resident_mtp::validate_limits(&adapter.limits())?;
            if input.fused_sweeps {
                resident_mtp::validate_fused_sweep_limits(&adapter.limits())?;
            }
        }
        if target_rows_inputs
            .as_ref()
            .is_some_and(|i| i.prepared_experts)
            || target_repair_inputs
                .as_ref()
                .is_some_and(|i| i.prepared_experts)
            || speculative_inputs
                .as_ref()
                .is_some_and(|i| i.prepared_experts)
            || language_inputs.as_ref().is_some_and(|i| i.prepared_experts)
        {
            executor::batch_moe::prepared_verify3::validate_subgroups(
                adapter.features().contains(wgpu::Features::SUBGROUP),
                info.subgroup_min_size,
                info.subgroup_max_size,
            )?;
            executor::batch_moe::prepared_verify3::validate_limits(&adapter.limits())?;
        }
        if verify4_dense_receipt.is_some() {
            next_shared::validate_verify4_limits(
                &adapter.limits(),
                adapter.features().contains(wgpu::Features::SUBGROUP),
                info.subgroup_min_size,
                info.subgroup_max_size,
            )?;
            if verify4_finite_probe {
                next_shared::validate_verify4_finite_limits(
                    &adapter.limits(),
                    adapter.features().contains(wgpu::Features::SUBGROUP),
                    info.subgroup_min_size,
                    info.subgroup_max_size,
                )?;
            }
            if adapter.limits().max_compute_workgroup_storage_size < 16 * 1024 {
                return Err("Verify4 original scalar Q4 operator control storage limit".into());
            }
        }
        if verify3_dense_receipt.is_some()
            || public_host_cpu_batch_inputs.is_some()
            || public_host_cpu_overlap_inputs.is_some()
            || target_rows_inputs.as_ref().is_some_and(|i| i.verify3_dense)
            || target_repair_inputs
                .as_ref()
                .is_some_and(|i| i.verify3_dense)
            || speculative_inputs.as_ref().is_some_and(|i| i.verify3_dense)
            || language_inputs.as_ref().is_some_and(|i| i.verify3_dense)
        {
            next_shared::validate_verify3_limits(
                &adapter.limits(),
                adapter.features().contains(wgpu::Features::SUBGROUP),
                info.subgroup_min_size,
                info.subgroup_max_size,
            )?;
        }
        if verify3_finite_probe
            || public_host_cpu_batch_inputs.is_some()
            || public_host_cpu_overlap_inputs.is_some()
            || target_rows_inputs.as_ref().is_some_and(|i| i.verify3_fused)
            || target_repair_inputs
                .as_ref()
                .is_some_and(|i| i.verify3_fused)
            || speculative_inputs.as_ref().is_some_and(|i| i.verify3_fused)
            || language_inputs.as_ref().is_some_and(|i| i.verify3_fused)
        {
            next_shared::validate_verify3_finite_limits(
                &adapter.limits(),
                adapter.features().contains(wgpu::Features::SUBGROUP),
                info.subgroup_min_size,
                info.subgroup_max_size,
            )?;
        }
        if verify3_dense_receipt.is_some()
            && adapter.limits().max_compute_workgroup_storage_size < 16 * 1024
        {
            return Err("Verify3 original Q4 operator control storage limit".into());
        }
        if q6_integer_probe {
            q6_integer_probe::validate_limits(&adapter.limits())?;
        }
        if resident_router_probe || resident_gpu_router {
            resident_router_gpu_probe::validate_limits(&adapter.limits())?;
        }
        if shared_original_iq_probe
            && (adapter.limits().max_compute_workgroup_storage_size as usize)
                < quant_matvec::iq_moe_shared_original::MAX_SHARED_BYTES
        {
            return Err("shared-original IQ requires its complete workgroup storage limit".into());
        }
        if next_shared
            && (adapter.limits().max_compute_invocations_per_workgroup < 256
                || adapter.limits().max_compute_workgroup_size_x < 256)
        {
            return Err("shared Next router requires256 workgroup lanes".into());
        }
        if quant_matvec::frozen_gu_integer::four::shuffle::PUBLIC_PROBE
            && quant_matvec::frozen_down_integer::four::STAGED_PUBLIC_PROBE
            && (speculative_inputs
                .as_ref()
                .is_some_and(|i| i.frozen_gu_stride_four && i.frozen_down_integer)
                || language_inputs
                    .as_ref()
                    .is_some_and(|i| i.frozen_gu_stride_four && i.frozen_down_integer))
        {
            quant_matvec::frozen_down_integer::four::validate_staged_limits(&adapter.limits())?;
        }
        if (fast_gpu
            || (quant_matvec::frozen_gu_integer::four::shuffle::PUBLIC_PROBE
                && (speculative_inputs
                    .as_ref()
                    .is_some_and(|i| i.frozen_gu_stride_four)
                    || language_inputs
                        .as_ref()
                        .is_some_and(|i| i.frozen_gu_stride_four)))
            || mode == "iq-quant-probe"
            || segmented_iq_probe
            || iq_integer_probe
            || shared_original_iq_probe
            || live_member_iq_probe
            || q6_integer_probe
            || resident_mtp_inputs.is_some()
            || resident_router_probe)
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
        if gpu_upload_clear
            && !adapter
                .features()
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
        {
            return Err("GPU upload clear requires native primary mapping support".into());
        }
        if mtp_gpu_inputs.is_none() {
            println!(
                "adapter={} vendor={:04x} device={:04x} backend={:?}",
                info.name, info.vendor, info.device, info.backend
            );
        }
        if public_target_profile
            && !adapter.features().contains(
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
            )
        {
            return Err("public target profiling requires device timestamp queries".into());
        }
        strata_port::validate_limits(&adapter.limits())?;
        if strata_intel_core::enabled()
            && !adapter.features().contains(wgpu::Features::SHADER_INT64)
        {
            return Err("native expert pointers require shader int64".into());
        }
        let device_result = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Kuno public synthetic Rust kernel"),
                required_features: if mtp_gpu_inputs.is_some() {
                    wgpu::Features::empty()
                } else {
                    adapter.features()
                        & (wgpu::Features::SHADER_F16
                            | if gpu_upload_clear {
                                wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                            } else {
                                wgpu::Features::empty()
                            }
                            | wgpu::Features::SUBGROUP
                            | if strata_intel_core::enabled() {
                                wgpu::Features::SHADER_INT64
                            } else {
                                wgpu::Features::empty()
                            }
                            | if kernel_profile || public_target_profile {
                                wgpu::Features::TIMESTAMP_QUERY
                                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                            } else {
                                wgpu::Features::empty()
                            })
                },
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
            .await;
        let (device, queue) = if mtp_gpu_inputs.is_some() {
            device_result.map_err(|_| "fixed public MTP GPU device setup failed")?
        } else {
            device_result?
        };
        if let Some(receipt) = public_host_weight_receipt {
            std::panic::set_hook(Box::new(|_| {
                eprintln!("public_host_weight_failure panic=true")
            }));
            let mut value =
                public_host_weight_probe::run(&adapter, &device, &queue, &adapter.limits())?;
            value["vendor"] = vendor.into();
            serde_json::to_writer_pretty(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(receipt)?,
                &value,
            )?;
            println!("public_host_weight_operator_complete owners_erased=true");
            return Ok(());
        }
        let grouped_limits = adapter.limits();
        let gr_storage_alignment = adapter.limits().min_storage_buffer_offset_alignment as usize;
        // The public foreign-source probe needs the exact device that owns the
        // initialized runtime. Ordinary model entries retain no extra clones.
        let public_host_setup = public_host_cubecl_receipt
            .as_ref()
            .map(|_| (adapter.clone(), device.clone(), queue.clone()));
        let public_model_host_device = (speculative_inputs
            .as_ref()
            .is_some_and(|p| p.frozen_host_sources)
            || language_inputs
                .as_ref()
                .is_some_and(|p| p.frozen_host_sources)
            || public_host_cpu_batch_inputs.is_some()
            || public_host_cpu_overlap_inputs.is_some())
        .then(|| device.clone());
        let public_model_host_queue = public_model_host_device.as_ref().map(|_| queue.clone());
        let selected = init_device(
            WgpuSetup {
                instance,
                adapter,
                device,
                queue,
                backend: wgpu::Backend::Vulkan,
            },
            RuntimeOptions {
                tasks_max: if strata_intel_core::enabled() {
                    32768
                } else if batch_tasks
                    || target_rows_inputs.is_some()
                    || target_repair_inputs.is_some()
                    || speculative_inputs.is_some()
                    || language_inputs.is_some()
                    || public_host_cpu_batch_inputs.is_some()
                    || public_host_cpu_overlap_inputs.is_some()
                    || resident_mtp_inputs.is_some()
                {
                    256
                } else {
                    32
                },
                memory_config: private_arena::required_memory_configuration(),
            },
        );
        let client = WgpuRuntime::client(&selected);
        if public_target_profile {
            // Configuration is process-once. Each client starts paused; only the
            // target's selected verification round may arm its own logger.
            client.set_profiling_paused(true);
            if client.properties().timing_method != cubecl_common::profile::TimingMethod::Device {
                return Err("public target profiling requires actual device timing".into());
            }
        }
        if let Some(receipt) = public_host_cubecl_receipt {
            std::panic::set_hook(Box::new(|_| {
                eprintln!("public_host_cubecl_failure panic=true")
            }));
            let (adapter, device, queue) =
                public_host_setup.ok_or("public host CubeCL device identity unavailable")?;
            let mut value = public_host_weight_cubecl_probe::run(
                &adapter,
                &device,
                &queue,
                &client,
                &grouped_limits,
            )?;
            value["vendor"] = vendor.into();
            serde_json::to_writer_pretty(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(receipt)?,
                &value,
            )?;
            println!("public_host_cubecl_operator_complete owners_erased=true");
            return Ok(());
        }
        if let Some(receipt) = verify4_dense_receipt {
            std::panic::set_hook(Box::new(|_| {
                eprintln!("public_verify4_operator_failure panic=true")
            }));
            let mut value = if verify4_finite_probe {
                next_verify4_dense_probe::run_fused(&client, &grouped_limits)?
            } else {
                next_verify4_dense_probe::run(&client, &grouped_limits)?
            };
            value["vendor"] = vendor.into();
            serde_json::to_writer_pretty(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(receipt)?,
                &value,
            )?;
            println!(
                "public_verify4_operator_complete normal=20 fault={} owners_erased=true",
                if verify4_finite_probe { 70 } else { 20 }
            );
            return Ok(());
        }
        if let Some(receipt) = verify3_dense_receipt {
            std::panic::set_hook(Box::new(|_| {
                eprintln!("public_verify3_operator_failure panic=true")
            }));
            let mut value = if verify3_finite_probe {
                next_verify3_dense_probe::run_fused(&client, &grouped_limits)?
            } else {
                next_verify3_dense_probe::run(&client, &grouped_limits)?
            };
            value["vendor"] = vendor.into();
            serde_json::to_writer_pretty(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(receipt)?,
                &value,
            )?;
            println!(
                "public_verify3_operator_complete normal=60 fault={} owners_erased=true",
                if verify3_finite_probe { 210 } else { 20 }
            );
            return Ok(());
        }
        if qsa_chronological_receipt.is_some() {
            std::panic::set_hook(Box::new(|_| {
                eprintln!("public_qsa_selector_failure panic=true")
            }));
            resident_qsa::batch::chronological_probe::run(&client, &grouped_limits)?;
            println!("public_qsa_selector_operator_complete owners_erased=true");
            return Ok(());
        }
        if let Some(receipt) = batch_router_receipt {
            let mut value = resident_batch_router_gpu_probe::run(&client, &grouped_limits)?;
            value["vendor"] = vendor.into();
            serde_json::to_writer_pretty(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(receipt)?,
                &value,
            )?;
            println!("public_batch_router_operator_complete owners_erased=true");
            return Ok(());
        }
        if let Some(inputs) = public_host_cpu_batch_inputs {
            return public_host_cpu_batch_probe::run(
                &client,
                &grouped_limits,
                inputs,
                public_model_host_device
                    .as_ref()
                    .ok_or("attached HOST CPU batch actual target device missing")?,
            );
        }
        if let Some(inputs) = public_host_cpu_overlap_inputs {
            return public_host_expert_attachment::overlap_probe::run(
                &client,
                &grouped_limits,
                inputs,
                public_model_host_device
                    .as_ref()
                    .ok_or("attached HOST CPU overlap requires original device-owned HOST pages")?,
            );
        }
        if let Some(inputs) = language_inputs {
            return daslab_language_probe::run(
                &client,
                &grouped_limits,
                inputs,
                public_model_host_device.as_ref(),
                public_model_host_queue.as_ref(),
            );
        }
        if let Some(inputs) = speculative_inputs {
            return daslab_speculative_probe::run(
                &client,
                &grouped_limits,
                inputs,
                public_model_host_device.as_ref(),
            );
        }
        if let Some(inputs) = target_repair_inputs {
            return daslab_target_repair_probe::run(&client, &grouped_limits, inputs);
        }
        if let Some(inputs) = resident_mtp_inputs {
            return resident_mtp_probe::run(&client, &grouped_limits, vendor, inputs);
        }
        if let Some(inputs) = strata_mtp_inputs {
            return strata_mtp_gpu_probe::run(&client, &grouped_limits, vendor, inputs);
        }
        if let Some(inputs) = target_rows_inputs {
            return daslab_target_rows_probe::run(&client, &grouped_limits, inputs);
        }
        if let Some(inputs) = mtp_gpu_inputs {
            return mtp_gpu_probe::run(&client, vendor, inputs);
        }
        if segmented_iq_probe {
            return segmented_iq_moe_probe::run(&client, &grouped_limits);
        }
        if iq_integer_probe {
            return segmented_iq_moe_probe::run_integer(&client, &grouped_limits);
        }
        if shared_original_iq_probe {
            return segmented_iq_moe_probe::run_shared_original(&client, &grouped_limits);
        }
        if live_member_iq_probe {
            return segmented_iq_moe_probe::run_live_members(&client, &grouped_limits);
        }
        if q6_integer_probe {
            return q6_integer_probe::run(&client, &grouped_limits);
        }
        if resident_router_probe {
            return resident_router_gpu_probe::run(&client, &grouped_limits);
        }
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
        if mode == "iq-quant-probe" {
            return matmul_probe::iq::run(&client);
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
                client.configure_transient_io(resident_runtime_limit)
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
            if resident_expert_calibration {
                // SAFETY: this exclusive protected lab borrows the unchanged
                // content-addressed public model. No installer/writer may run.
                unsafe {
                    file.enable_mapped_weights()?;
                }
            }
            let model = Qwen4Exp::bind(&file)?;
            let retired = transient_context
                .clone()
                .ok_or("CPU/GPU calibration requires configured transient stream")?;
            let cancel = AtomicBool::new(false);
            let calibrate = if resident_expert_calibration {
                cpu_gpu_expert_probe::run_resident_calibration
            } else {
                cpu_gpu_expert_probe::run_fixed_calibration
            };
            let reports = calibrate(
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
                "scope":if resident_expert_calibration {"fixed-public-original-resident-expert-calibration"} else {"fixed-public-original-cpu-gpu-expert-calibration"},
                "resident_expert_calibration":resident_expert_calibration,
                "weights_mapped":file.weights_mapped(),
                "cpu_mapped_prefetch":resident_expert_calibration,
                "device_dispatch":device_dispatch,
                "vendor":vendor,"weights":"borrowed-unmodified-ollama-blob",
                "fixed_public_points":6,"numeric_gates":12,"timing_samples":24,
                "ranks_per_case":10,"group":0,"vector":0,"backend":if resident_expert_calibration {"pooled-avx2-fma"} else {"avx2"},"workers":16,
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
            if resident_expert_calibration {
                println!(
                    "public_resident_expert_calibration modes=resident,upload,cpu workers=16 backend=pooled-avx2-fma owned_cleanup=true dual_device_accepted=false"
                );
            }
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
                        for case in 0..if next_shared { 5 } else { 4 } {
                            let mut initial = vec![0f32; 16];
                            initial[2..2 + rows].fill(f32::NAN);
                            if case == 1 {
                                initial[15] = f32::NAN;
                            }
                            if case == 4 {
                                initial[0] = f32::INFINITY;
                            }
                            let mut fixture_input = input.clone();
                            if case == 2 {
                                fixture_input[0] = f32::NAN;
                            }
                            let initial_fault = if case == 3 { 8u32 } else { 0 };
                            let out = client.create_from_slice(f32::as_bytes(&initial));
                            let fixture_x = client.create_from_slice(f32::as_bytes(&fixture_input));
                            let fault = client.create_from_slice(&initial_fault.to_le_bytes());
                            if next_shared {
                                next_shared::matrix_fault(
                                    &client,
                                    next_shared::MatrixSpec::new(rows, cols, dtype)?,
                                    w.clone(),
                                    fixture_x,
                                    out.clone(),
                                    input.len(),
                                    16,
                                    0,
                                    2,
                                    fault.clone(),
                                    rows,
                                    2,
                                    true,
                                )?;
                            } else {
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
                            }
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
                        println!(
                            "{}_packed_fused_fault dtype={dtype} cases={} passed=true",
                            if next_shared {
                                "next_shared"
                            } else {
                                "rust_fast"
                            },
                            if next_shared { 5 } else { 4 }
                        );
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
            resident_util::select_compute_scratch_clear(compute_scratch_clear);
            status_fold_batch::select(grouped_status);
            if grouped_status {
                status_fold_batch::probe(
                    &client,
                    transient_context
                        .as_ref()
                        .ok_or("owned runtime required")?
                        .clone(),
                )?;
            }
            if compute_scratch_clear {
                scratch_clear_batch::probe(
                    &client,
                    transient_context
                        .as_ref()
                        .ok_or("owned runtime required")?
                        .clone(),
                )?;
            }
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
                let config = if count1_hybrid_cap.is_some() {
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
                };
                let config = if resident_gpu_router {
                    config.reserve_resident_router()?
                } else {
                    config
                };
                Some(config.reserve_request_reuse_metadata(request_reuse_metadata_charge)?)
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
            let (prompt, limit) = if q3_main {
                ("!", q3_positions)
            } else {
                match fixture_mode {
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
                }
            };
            let load = Instant::now();
            let before_bind_uploads = cubecl::wgpu::transient_unscoped_upload_transfer_stats();
            let file = if q3_main {
                let dir = std::path::PathBuf::from(
                    std::env::var_os("KUNO_TEST_Q3_DIR")
                        .ok_or("verified Q3 shard directory required")?,
                );
                let daslab = packed_profile == expert_banks::OriginalPackedProfile::DaslabIq3;
                let paths: Vec<_> = (1..=if daslab { 2 } else { 3 })
                    .map(|i| {
                        dir.join(if daslab {
                            format!("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf")
                        } else {
                            format!("Qwen3.8-Flash-Next-UD-Q3_K_XL-{i:05}-of-00003.gguf")
                        })
                    })
                    .collect();
                let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
                if file.file_len()
                    != if daslab {
                        83_617_662_656
                    } else {
                        89_986_353_824
                    }
                {
                    return Err("pinned original Q3 byte count".into());
                }
                file
            } else {
                GgufFile::open(
                    std::env::var_os("KUNO_TEST_GGUF").ok_or("existing GGUF required")?,
                    GgufLimits::default(),
                )?
            };
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
            if q3_main && ids != [0] {
                return Err("fixed public Q3 token binding rejected".into());
            }
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
                if !q3_main {
                    next_shared::probe(&client, &file)?;
                }
                if integer_experts {
                    next_shared_moe::probe_integer(&client, &file)?;
                }
                next_shared::probe_gr(
                    &client,
                    transient_context
                        .as_ref()
                        .ok_or("shared GR requires transient stream")?
                        .clone(),
                )?;
                if shared_gdn {
                    next_shared::probe_gdn(&client, model.bound_graph().geometry().gdn)?;
                }
            }
            let bank_charge = expert_banks::preflight_residency_with_profile(
                model.bound_graph(),
                expert_slots,
                cache_budget,
                device_budget,
                packed_profile,
            )?;
            let mut executor = executor::GpuExecutor::new_with_profile(
                &file,
                client.clone(),
                cache_budget,
                expert_slots,
                device_budget,
                packed_profile,
            )?;
            if let Some(retired) = &transient_context {
                executor.enable_transient_resident(retired.clone())?;
            }
            executor.gpu_router = gpu_router;
            executor.fused_experts = fused_experts;
            executor.integer_experts = integer_experts;
            executor.gpu_expert_limit = gpu_expert_limit;
            executor.q3_empty_slot_fill = q3_empty_slot_fill;
            executor.q3_skip_empty_segments = q3_skip_empty_segments;
            executor.q3_subgroup_moe = q3_subgroup_moe;
            executor.q3_integer_moe = q3_integer_moe;
            executor.q3_shared_original_moe = q3_shared_original_moe;
            executor.q6_integer_head = q6_integer_head;
            executor.resident_router_reserved = resident_gpu_router;
            executor.q3_expert_oracle = q3_expert_oracle;
            executor.q3_router_reference = q3_router_reference.clone();
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
            let after_bind_uploads = cubecl::wgpu::transient_unscoped_upload_transfer_stats();
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
            let q3_prefill_slots = if q3_main {
                Some(executor.q3_occupied_slots()?)
            } else {
                None
            };
            let after_prefill_uploads = cubecl::wgpu::transient_unscoped_upload_transfer_stats();
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
                apply_count1_hybrid_cap(
                    r,
                    &mut executor,
                    count1_hybrid_cap,
                    request_io_reuse,
                    &cancel,
                )?;
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
            let pre_shared_gr_calls = next_shared::gr_calls();
            let pre_scratch_clear_calls = resident_util::native_scratch_clear_calls();
            let pre_compute_scratch_clear_calls = resident_util::compute_scratch_clear_calls();
            let pre_grouped_status_calls = status_fold_batch::calls();
            let pre_readback_reuses = cubecl::wgpu::transient_readback_reuses();
            let pre_gpu_upload_clears = cubecl::wgpu::transient_gpu_upload_clears();
            let pre_inline_read_batches = cubecl::wgpu::transient_inline_readback_batches();
            let pre_request_io_reuses = cubecl::wgpu::transient_request_io_reuses();
            let pre_scoped_read_ns = cubecl::wgpu::transient_scoped_read_ns();
            let pre_info_cache_stats = cubecl::wgpu::transient_info_cache_stats();
            let pre_binding_cache_stats = cubecl::wgpu::transient_binding_cache_stats();
            let pre_fused_expert_calls = executor.fused_expert_calls;
            let pre_cpu_resident_expert_calls = executor.cpu_resident_expert_calls;
            let pre_upload_transfer = cubecl::wgpu::transient_upload_transfer_stats();
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
                if let Some(reference) = &q3_reference {
                    reference.verify(n, &logits)?;
                }
                let token = greedy(&logits)?;
                // Q3 is a fixed-length public numerical sequence, matching the
                // independent writer which also feeds back a possible EOS ID.
                // Ordinary Q4 generation keeps its existing EOS behavior.
                if token == model.eos_token() && !q3_main {
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
                    || (cap > 0 && completed == 0 && !q3_empty_slot_fill)
                {
                    return Err("count1 hybrid did not complete actual CPU miss work".into());
                }
                Some(completed)
            } else {
                None
            };
            if q3_main && generated.len() != q3_positions {
                return Err("Q3 reference requires all selected public positions".into());
            }
            let request_io_checkpoints =
                resident.as_ref().map_or(0, |r| r.request_io_checkpoints());
            let request_io_close_start = Instant::now();
            if request_io_reuse {
                resident
                    .as_ref()
                    .ok_or("request I/O model absent")?
                    .end_lab_request_io_reuse()?;
                println!("public_request_io_reuse closed=true full_audit=true");
            }
            let request_io_close_ms = if request_io_reuse {
                request_io_close_start.elapsed().as_millis() as u64
            } else {
                0
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
                receipt["original_packed_profile"] = serde_json::json!(match packed_profile {
                    expert_banks::OriginalPackedProfile::UnslothQ3 => "unsloth-ud-q3_k_xl",
                    expert_banks::OriginalPackedProfile::DaslabIq3 => "daslab-gsq-rco-iq3_s",
                    expert_banks::OriginalPackedProfile::Q4 => "original-q4",
                });
                if q3_main {
                    receipt["fixture"] = serde_json::json!(if packed_profile
                        == expert_banks::OriginalPackedProfile::DaslabIq3
                    {
                        "daslab-iq3-public-bang"
                    } else {
                        "unsloth-q3-public-bang"
                    });
                    if packed_profile == expert_banks::OriginalPackedProfile::DaslabIq3 {
                        receipt["model_repository"] = serde_json::json!(daslab_profile::REPOSITORY);
                        receipt["model_revision"] = serde_json::json!(daslab_profile::REVISION);
                        receipt["original_header_sha256"] =
                            serde_json::json!(daslab_profile::HEADER_SHA256);
                    }
                    receipt["whole_model_numeric_accepted"] = serde_json::json!(true);
                    receipt["numeric_scope"] = serde_json::json!(match q3_positions {
                        4 => "four-position-original-Q3-public-bang",
                        13 => "13-position-prefix-of-32-original-DASLab-public-bang",
                        23 => "23-position-prefix-of-32-original-Q3-public-bang",
                        _ => "32-position-original-Q3-public-bang",
                    });
                    receipt["numeric_positions"] = serde_json::json!(q3_positions);
                    receipt["full32_reference_accepted"] = serde_json::json!(q3_positions == 32);
                    receipt["prefix_ends_before_known_router_divergence"] =
                        serde_json::json!(matches!(q3_positions, 13 | 23));
                    receipt["weights"] = serde_json::json!("borrowed-unmodified-original-shards");
                    let pre_slots = q3_prefill_slots.ok_or("Q3 prefill slot receipt absent")?;
                    let occupied = executor.q3_occupied_slots()?;
                    receipt["q3_empty_slot_fill"] = serde_json::json!(q3_empty_slot_fill);
                    receipt["q3_skip_empty_segments"] = serde_json::json!(q3_skip_empty_segments);
                    receipt["q3_subgroup_moe"] = serde_json::json!(q3_subgroup_moe);
                    receipt["q3_integer_moe"] = serde_json::json!(q3_integer_moe);
                    receipt["q3_shared_original_moe"] = serde_json::json!(q3_shared_original_moe);
                    if q6_integer_head
                        && (executor.q6_integer_head_prepares == 0
                            || executor.q6_integer_head_tiles
                                != 16 * executor.q6_integer_head_prepares)
                    {
                        return Err("Q6 integer head observed complete tile contract".into());
                    }
                    receipt["q6_integer_head"] = serde_json::json!(q6_integer_head);
                    receipt["q6_integer_head_prepares"] =
                        serde_json::json!(executor.q6_integer_head_prepares);
                    receipt["q6_integer_head_tiles"] =
                        serde_json::json!(executor.q6_integer_head_tiles);
                    receipt["head_alternative_activation_arithmetic"] =
                        serde_json::json!(q6_integer_head);
                    receipt["moe_arithmetic"] = serde_json::json!(if q3_integer_moe {
                        "original-packed-IQ-two-residual-Q8-planes-integer-candidate"
                    } else if q3_shared_original_moe {
                        "original64-F32-shared-codebook"
                    } else {
                        "original64-F32"
                    });
                    receipt["original_f32_full_logits_and_greedy_required"] =
                        serde_json::json!(true);
                    receipt["moe_alternative_activation_arithmetic"] =
                        serde_json::json!(q3_integer_moe);
                    receipt["expert_slots_reserved"] =
                        serde_json::json!(48 * crate::daslab_profile::BANK_SLOTS);
                    receipt["expert_slots_occupied_after_prefill"] = serde_json::json!(pre_slots);
                    receipt["expert_slots_occupied_after_decode"] = serde_json::json!(occupied);
                    receipt["expert_slots_filled_during_decode"] = serde_json::json!(
                        occupied
                            .checked_sub(pre_slots)
                            .ok_or("Q3 slot receipt regressed")?
                    );
                    if q3_empty_slot_fill {
                        let cpu = completed_cpu_misses.ok_or("Q3 CPU rank receipt absent")?;
                        let gpu = native_decode_steps
                            .checked_mul(480)
                            .and_then(|ranks| ranks.checked_sub(cpu))
                            .ok_or("Q3 expert rank receipt overflow")?;
                        if (executor.expert_hits - pre_stats.7)
                            .checked_add(executor.expert_uploads - pre_stats.8)
                            != Some(gpu)
                            || u64::try_from(occupied - pre_slots)?
                                != executor.expert_uploads - pre_stats.8
                        {
                            return Err("Q3 empty-slot rank and upload receipt mismatch".into());
                        }
                        receipt["q3_decode_cpu_expert_ranks"] = serde_json::json!(cpu);
                        receipt["q3_decode_gpu_expert_ranks"] = serde_json::json!(gpu);
                    }
                }
                receipt["fused_matrix_fault"] = serde_json::json!(executor.fused_matrix_fault);
                receipt["mapped_weights"] = serde_json::json!(mapped_weights);
                receipt["count1_bank_profile"] = serde_json::json!(count1_banks.label());
                receipt["observed_mapped_weights"] = serde_json::json!(file.weights_mapped());
                // Private upload command extents only. Packed public weight
                // uploads use a separate path and are not included here.
                for (name, before, after) in [
                    (
                        "model_bind_private_upload_bytes",
                        before_bind_uploads,
                        after_bind_uploads,
                    ),
                    (
                        "prefill_private_upload_bytes",
                        after_bind_uploads,
                        after_prefill_uploads,
                    ),
                ] {
                    receipt[name] = serde_json::json!([
                        after
                            .0
                            .checked_sub(before.0)
                            .ok_or("private upload original counter overflow")?,
                        after
                            .1
                            .checked_sub(before.1)
                            .ok_or("private upload copy counter overflow")?,
                        after
                            .2
                            .checked_sub(before.2)
                            .ok_or("private upload clear counter overflow")?,
                    ]);
                    println!("public_{name}={}", receipt[name]);
                }
                receipt["request_io_reuse"] = serde_json::json!(request_io_reuse);
                receipt["bind_group_reuse"] = serde_json::json!(bind_group_reuse);
                receipt["binding_cache_capacity"] = serde_json::json!(binding_cache_capacity);
                receipt["request_reuse_metadata_reserved_bytes"] =
                    serde_json::json!(request_reuse_metadata_charge);
                receipt["native_aggregate_host_plan"] = serde_json::json!({
                    "reserved_bytes":native_aggregate_host_plan.0,
                    "limit_bytes":native_aggregate_host_plan.1,
                });
                receipt["adaptive_experts"] = serde_json::json!(adaptive_experts);
                if let Some(cache) = &executor.adaptive {
                    receipt["adaptive_direct_upload"] = serde_json::json!(adaptive_direct_upload);
                    receipt["adaptive_direct_batches"] =
                        serde_json::json!(cache.stats.direct_batches);
                    receipt["adaptive_direct_components"] =
                        serde_json::json!(cache.stats.direct_components);
                    receipt["adaptive_direct_bytes"] = serde_json::json!(cache.stats.direct_bytes);
                    receipt["adaptive_capacity"] =
                        serde_json::json!(crate::adaptive_experts::CAPACITY);
                    receipt["adaptive_policy"] = serde_json::json!("decaying-frequency-recency");
                    receipt["adaptive_frequency_half_life_tokens"] =
                        serde_json::json!(crate::adaptive_experts::HALF_LIFE);
                    receipt["adaptive_score_scale"] =
                        serde_json::json!(crate::adaptive_experts::SCALE);
                    receipt["adaptive_admit_score"] =
                        serde_json::json!(crate::adaptive_experts::ADMIT_SCORE);
                    receipt["adaptive_wall_ns"] = serde_json::json!(cache.stats.wall_ns);
                    receipt["adaptive_policy_ns"] = serde_json::json!(cache.stats.policy_ns);
                    receipt["adaptive_fill_ns"] = serde_json::json!(cache.stats.fill_ns);
                    receipt["adaptive_publication_ns"] =
                        serde_json::json!(cache.stats.publication_ns);
                    receipt["adaptive_budget"] = serde_json::json!(crate::adaptive_experts::BUDGET);
                    receipt["adaptive_promotions"] = serde_json::json!(cache.stats.promotions);
                    receipt["adaptive_upload_bytes"] = serde_json::json!(cache.stats.bytes);
                    receipt["adaptive_read_ns"] = serde_json::json!(cache.stats.read_ns);
                    receipt["adaptive_upload_ns"] = serde_json::json!(cache.stats.upload_ns);
                    receipt["adaptive_fence_ns"] = serde_json::json!(cache.stats.fence_ns);
                    receipt["adaptive_boundaries"] = serde_json::json!(cache.stats.boundaries);
                    receipt["adaptive_completed_batches"] =
                        serde_json::json!(cache.stats.completed_batches);
                    receipt["adaptive_private_peak"] = serde_json::json!(cache.stats.private_peak);
                    receipt["adaptive_source_peak"] = serde_json::json!(cache.stats.source_peak);
                    receipt["adaptive_host_bytes"] = serde_json::json!(cache.owned_bytes());
                }
                receipt["secondary_experts"] = serde_json::json!(secondary_experts);
                if let Some(second) = &executor.secondary {
                    receipt["secondary_expert_calls"] = serde_json::json!(second.expert_calls);
                    receipt["secondary_gpu_calls"] = serde_json::json!(second.calls);
                    receipt["secondary_bank_setup_ms"] = serde_json::json!(second.bank_setup_ms);
                    receipt["secondary_host_phase_ns"] = serde_json::json!(second.phase_ns);
                    receipt["secondary_weight_upload_bytes"] =
                        serde_json::json!(second.uploaded_bytes);
                    receipt["secondary_bank_allocation_bytes"] =
                        serde_json::json!(9u64 * 1024 * 1024 * 1024);
                    receipt["secondary_bank_shards"] = serde_json::json!([32, 16]);
                }
                receipt["fused_experts"] = serde_json::json!(fused_experts);
                receipt["integer_experts"] = serde_json::json!(integer_experts);
                receipt["gpu_expert_limit"] = serde_json::json!(gpu_expert_limit);
                receipt["observed_cpu_resident_expert_calls"] = serde_json::json!(
                    executor
                        .cpu_resident_expert_calls
                        .checked_sub(pre_cpu_resident_expert_calls)
                        .ok_or("CPU resident counter overflow")?
                );
                receipt["observed_fused_expert_calls"] = serde_json::json!(
                    executor
                        .fused_expert_calls
                        .checked_sub(pre_fused_expert_calls)
                        .ok_or("fused expert counter overflow")?
                );
                let binding_stats = cubecl::wgpu::transient_binding_cache_stats();
                let binding_delta = binding_stats
                    .into_iter()
                    .zip(pre_binding_cache_stats)
                    .map(|(after, before)| {
                        after.checked_sub(before).ok_or("binding counter overflow")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                receipt["warm_binding_cache_stats"] = serde_json::json!(binding_delta);
                println!(
                    "public_warm_binding_cache_stats={}",
                    receipt["warm_binding_cache_stats"]
                );
                receipt["observed_request_io_checkpoints"] =
                    serde_json::json!(request_io_checkpoints);
                receipt["observed_request_io_reuses"] = serde_json::json!(
                    cubecl::wgpu::transient_request_io_reuses()
                        .checked_sub(pre_request_io_reuses)
                        .ok_or("request I/O counter overflow")?
                );
                receipt["request_io_closed_and_audited"] = serde_json::json!(request_io_reuse);
                receipt["request_io_close_ms"] = serde_json::json!(request_io_close_ms);
                let info_cache_stats = cubecl::wgpu::transient_info_cache_stats();
                receipt["warm_info_cache_stats"] = serde_json::json!([
                    info_cache_stats
                        .0
                        .checked_sub(pre_info_cache_stats.0)
                        .ok_or("INFO cache hit counter overflow")?,
                    info_cache_stats
                        .1
                        .checked_sub(pre_info_cache_stats.1)
                        .ok_or("INFO cache miss counter overflow")?,
                    info_cache_stats
                        .2
                        .checked_sub(pre_info_cache_stats.2)
                        .ok_or("INFO cache page counter overflow")?,
                ]);
                println!(
                    "public_warm_info_cache_stats={}",
                    receipt["warm_info_cache_stats"]
                );
                let upload_transfer = cubecl::wgpu::transient_upload_transfer_stats();
                receipt["warm_upload_transfer_bytes"] = serde_json::json!([
                    upload_transfer
                        .0
                        .checked_sub(pre_upload_transfer.0)
                        .ok_or("upload original byte counter overflow")?,
                    upload_transfer
                        .1
                        .checked_sub(pre_upload_transfer.1)
                        .ok_or("upload copied byte counter overflow")?,
                    upload_transfer
                        .2
                        .checked_sub(pre_upload_transfer.2)
                        .ok_or("upload cleared byte counter overflow")?,
                ]);
                println!(
                    "public_warm_upload_transfer_bytes={}",
                    receipt["warm_upload_transfer_bytes"]
                );
                let scoped_read_ns = cubecl::wgpu::transient_scoped_read_ns();
                receipt["warm_scoped_read_host_ns"] = serde_json::json!([
                    scoped_read_ns
                        .0
                        .checked_sub(pre_scoped_read_ns.0)
                        .ok_or("scoped read setup counter overflow")?,
                    scoped_read_ns
                        .1
                        .checked_sub(pre_scoped_read_ns.1)
                        .ok_or("scoped read flush counter overflow")?,
                    scoped_read_ns
                        .2
                        .checked_sub(pre_scoped_read_ns.2)
                        .ok_or("scoped read map counter overflow")?,
                ]);
                println!(
                    "public_warm_scoped_read_host_ns={}",
                    receipt["warm_scoped_read_host_ns"]
                );
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
                receipt["observed_shared_gr_calls"] = serde_json::json!(
                    next_shared::gr_calls()
                        .checked_sub(pre_shared_gr_calls)
                        .ok_or("shared GR counter regressed")?
                );
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
                receipt["cpu_work_stealing"] = serde_json::json!(cpu_work_stealing);
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
                receipt["grouped_status"] = serde_json::json!(grouped_status);
                receipt["observed_grouped_status_calls"] = serde_json::json!(
                    status_fold_batch::calls()
                        .checked_sub(pre_grouped_status_calls)
                        .ok_or("grouped status counter regression")?
                );
                receipt["compute_scratch_clear"] = serde_json::json!(compute_scratch_clear);
                receipt["observed_compute_scratch_clear_calls"] = serde_json::json!(
                    resident_util::compute_scratch_clear_calls()
                        .checked_sub(pre_compute_scratch_clear_calls)
                        .ok_or("compute clear counter regression")?
                );
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
                receipt["inline_readback"] = serde_json::json!(inline_readback);
                receipt["observed_inline_readback_batches"] = serde_json::json!(
                    cubecl::wgpu::transient_inline_readback_batches()
                        .checked_sub(pre_inline_read_batches)
                        .ok_or("inline read counter regression")?
                );
                receipt["gpu_upload_clear"] = serde_json::json!(gpu_upload_clear);
                receipt["observed_gpu_upload_clears"] = serde_json::json!(
                    cubecl::wgpu::transient_gpu_upload_clears()
                        .checked_sub(pre_gpu_upload_clears)
                        .ok_or("GPU upload clear counter regression")?
                );
                receipt["readback_reuse"] = serde_json::json!(readback_reuse);
                receipt["observed_clean_readback_reuses"] = serde_json::json!(
                    cubecl::wgpu::transient_readback_reuses()
                        .checked_sub(pre_readback_reuses)
                        .ok_or("readback reuse counter regression")?
                );
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
                receipt["resident_runtime_io_limit_bytes"] =
                    serde_json::json!(transient_resident.then_some(resident_runtime_limit));
                receipt["resident_gpu_router_ideal_only"] = serde_json::json!(resident_gpu_router);
                receipt["resident_router_runtime_plan_bytes"] = serde_json::json!(
                    resident_gpu_router
                        .then_some(executor::batch_moe::resident_router_runtime(q3_integer_moe))
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
                executor = executor::GpuExecutor::new_with_profile(
                    &file,
                    client.clone(),
                    cache_budget,
                    expert_slots,
                    device_budget,
                    packed_profile,
                )?;
                if let Some(retired) = &transient_context {
                    executor.enable_transient_resident(retired.clone())?;
                }
                executor.gpu_router = gpu_router;
                executor.fused_experts = fused_experts;
                executor.integer_experts = integer_experts;
                executor.gpu_expert_limit = gpu_expert_limit;
                executor.q3_empty_slot_fill = q3_empty_slot_fill;
                executor.q3_skip_empty_segments = q3_skip_empty_segments;
                executor.q3_subgroup_moe = q3_subgroup_moe;
                executor.q3_integer_moe = q3_integer_moe;
                executor.q3_shared_original_moe = q3_shared_original_moe;
                executor.q6_integer_head = q6_integer_head;
                executor.resident_router_reserved = resident_gpu_router;
                executor.q3_expert_oracle = q3_expert_oracle;
                executor.q3_router_reference = q3_router_reference.clone();
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
                if let Some(reference) = &q3_reference {
                    reference.verify(0, &fresh_logits)?;
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
                    apply_count1_hybrid_cap(
                        &mut fresh,
                        &mut executor,
                        count1_hybrid_cap,
                        request_io_reuse,
                        &cancel,
                    )?;
                    fresh_logits = resident_decode_step(
                        &mut fresh,
                        &mut executor,
                        generated[0],
                        &cancel,
                        true,
                    )?;
                    if let Some(reference) = &q3_reference {
                        reference.verify(1, &fresh_logits)?;
                    }
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
                        || (cap > 0 && completed == 0 && !q3_empty_slot_fill)
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
                    executor = executor::GpuExecutor::new_with_profile(
                        &file,
                        client.clone(),
                        cache_budget,
                        expert_slots,
                        device_budget,
                        packed_profile,
                    )?;
                    if let Some(retired) = &transient_context {
                        executor.enable_transient_resident(retired.clone())?;
                    }
                    executor.gpu_router = gpu_router;
                    executor.fused_experts = fused_experts;
                    executor.integer_experts = integer_experts;
                    executor.gpu_expert_limit = gpu_expert_limit;
                    executor.q3_empty_slot_fill = q3_empty_slot_fill;
                    executor.q3_skip_empty_segments = q3_skip_empty_segments;
                    executor.q3_subgroup_moe = q3_subgroup_moe;
                    executor.q3_integer_moe = q3_integer_moe;
                    executor.q3_shared_original_moe = q3_shared_original_moe;
                    executor.q6_integer_head = q6_integer_head;
                    executor.resident_router_reserved = resident_gpu_router;
                    executor.q3_expert_oracle = q3_expert_oracle;
                    executor.q3_router_reference = q3_router_reference.clone();
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
                        request_io_reuse,
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
                if steady_public {
                    // Original fields and all five original audits are frozen
                    // before this separate extended-context diagnostic begins.
                    println!("public_steady_probe_begin=true");
                    resident.take();
                    executor = executor::GpuExecutor::new_with_profile(
                        &file,
                        client.clone(),
                        cache_budget,
                        expert_slots,
                        device_budget,
                        packed_profile,
                    )?;
                    executor.enable_transient_resident(
                        transient_context
                            .as_ref()
                            .ok_or("steady transient owner missing")?
                            .clone(),
                    )?;
                    executor.gpu_router = gpu_router;
                    executor.fused_experts = fused_experts;
                    executor.integer_experts = integer_experts;
                    executor.gpu_expert_limit = gpu_expert_limit;
                    executor.fast_dense = fast_gpu;
                    executor.fast_dense_fault = fast_gpu_fault;
                    executor.shared_kernels = next_shared;
                    executor.shared_gdn = shared_gdn;
                    executor.token_io_checkpoint = token_checkpoint || defer_router_io;
                    let mut steady = resident_model::ResidentModel::new_for_lab_fixture(
                        model.bound_graph(),
                        client.clone(),
                        steady_public::CONTEXT,
                        &mut executor,
                        true,
                        native_config.clone(),
                    )?;
                    let public_prefix = zeroize::Zeroizing::new(
                        receipt["public_generated_ids"]
                            .as_array()
                            .ok_or("steady public prefix absent")?
                            .iter()
                            .map(|x| {
                                x.as_u64()
                                    .and_then(|n| u32::try_from(n).ok())
                                    .ok_or("steady public prefix invalid")
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    receipt["steady_probe"] = steady_public::run(
                        &mut steady,
                        &mut executor,
                        &ids,
                        &public_prefix,
                        &cancel,
                    )?;
                }
                if oracle_public {
                    // All original acceptance/close fields are frozen first.
                    println!("public_oracle_probe_begin=true");
                    resident.take();
                    let public_prefix = zeroize::Zeroizing::new(
                        receipt["public_generated_ids"]
                            .as_array()
                            .ok_or("oracle public prefix absent")?
                            .iter()
                            .take(oracle_trace::PREFIX)
                            .map(|x| {
                                x.as_u64()
                                    .and_then(|n| u32::try_from(n).ok())
                                    .ok_or("oracle public prefix invalid")
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    receipt["oracle_probe"] = oracle_public::run(
                        |context| {
                            let mut e = executor::GpuExecutor::new_with_profile(
                                &file,
                                client.clone(),
                                cache_budget,
                                expert_slots,
                                device_budget,
                                packed_profile,
                            )?;
                            e.enable_transient_resident(
                                transient_context
                                    .as_ref()
                                    .ok_or("oracle transient owner missing")?
                                    .clone(),
                            )?;
                            e.gpu_router = gpu_router;
                            e.fused_experts = fused_experts;
                            e.integer_experts = integer_experts;
                            e.gpu_expert_limit = gpu_expert_limit;
                            e.fast_dense = fast_gpu;
                            e.fast_dense_fault = fast_gpu_fault;
                            e.shared_kernels = next_shared;
                            e.shared_gdn = shared_gdn;
                            e.token_io_checkpoint = token_checkpoint || defer_router_io;
                            let r = resident_model::ResidentModel::new_for_lab_fixture(
                                model.bound_graph(),
                                client.clone(),
                                context,
                                &mut e,
                                true,
                                native_config.clone(),
                            )?;
                            Ok((r, e))
                        },
                        &ids,
                        &public_prefix,
                        &cancel,
                    )?;
                }
                if q3_ideal_public {
                    resident.take();
                    receipt["q3_ideal_probe"] = q3_ideal_public::run(
                        |context| {
                            let mut e = executor::GpuExecutor::new_with_profile(
                                &file,
                                client.clone(),
                                cache_budget,
                                expert_slots,
                                device_budget,
                                packed_profile,
                            )?;
                            e.enable_transient_resident(
                                transient_context
                                    .as_ref()
                                    .ok_or("oracle transient owner missing")?
                                    .clone(),
                            )?;
                            e.q3_empty_slot_fill = q3_empty_slot_fill;
                            e.q3_skip_empty_segments = q3_skip_empty_segments;
                            e.q3_subgroup_moe = q3_subgroup_moe;
                            e.q3_integer_moe = q3_integer_moe;
                            e.q3_shared_original_moe = q3_shared_original_moe;
                            e.q6_integer_head = q6_integer_head;
                            e.resident_router_reserved = resident_gpu_router;
                            e.gpu_router = gpu_router;
                            e.fused_experts = fused_experts;
                            e.integer_experts = integer_experts;
                            e.gpu_expert_limit = gpu_expert_limit;
                            e.fast_dense = fast_gpu;
                            e.fast_dense_fault = fast_gpu_fault;
                            e.shared_kernels = next_shared;
                            e.shared_gdn = shared_gdn;
                            e.token_io_checkpoint = token_checkpoint || defer_router_io;
                            let r = resident_model::ResidentModel::new_for_lab_fixture(
                                model.bound_graph(),
                                client.clone(),
                                context,
                                &mut e,
                                true,
                                native_config.clone(),
                            )?;
                            Ok((r, e))
                        },
                        q3_reference
                            .as_ref()
                            .ok_or("Q3 original reference absent")?,
                        if packed_profile == expert_banks::OriginalPackedProfile::DaslabIq3 {
                            match q3_positions {
                                13 => oracle_trace::Fixture::DaslabPrefix13,
                                32 => oracle_trace::Fixture::Daslab32,
                                _ => return Err("DASLab ideal fixture window".into()),
                            }
                        } else {
                            oracle_trace::Fixture::Q3Prefix23
                        },
                        resident_gpu_router,
                        &cancel,
                    )?;
                }
                if q3_expert_oracle {
                    return Err(
                        "fixed public expert diagnostic does not publish performance acceptance"
                            .into(),
                    );
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
    fn resident_mapper_probe_registration_is_exclusive_and_has_no_fixture_arguments() {
        for gate in [false, true] {
            assert_eq!(
                super::resident_router_probe_registration("resident-router-gpu-probe", 3, gate),
                Ok(true)
            );
            for argc in [0, 1, 2, 4, usize::MAX] {
                assert!(
                    super::resident_router_probe_registration(
                        "resident-router-gpu-probe",
                        argc,
                        gate
                    )
                    .is_err()
                );
            }
        }
        for mode in ["probe", "resident-count1-hybrid-chat", "iq-quant-probe"] {
            assert_eq!(
                super::resident_router_probe_registration(mode, 3, false),
                Ok(false)
            );
            assert!(super::resident_router_probe_registration(mode, 3, true).is_err());
        }
    }
    #[test]
    fn resident_router_is_exact_supported_protected_ideal_only() {
        use crate::expert_banks::OriginalPackedProfile as P;
        for (profile, positions) in [(P::UnslothQ3, 23), (P::DaslabIq3, 13), (P::DaslabIq3, 32)] {
            assert!(
                super::resident_router_scope(true, true, profile, positions, true, true).is_ok()
            );
            for (ideal, protected, original) in [
                (false, true, true),
                (true, false, true),
                (true, true, false),
            ] {
                assert!(
                    super::resident_router_scope(
                        true, ideal, profile, positions, protected, original
                    )
                    .is_err()
                );
            }
        }
        for (profile, positions) in [
            (P::Q4, 23),
            (P::UnslothQ3, 13),
            (P::UnslothQ3, 32),
            (P::DaslabIq3, 4),
            (P::DaslabIq3, 23),
        ] {
            assert!(
                super::resident_router_scope(true, true, profile, positions, true, true).is_err()
            );
        }
        assert!(super::resident_router_scope(false, false, P::Q4, 0, false, false).is_ok());
    }

    #[test]
    fn extended_binding_scope_requires_its_actual_profile_window_and_owner_policy() {
        use crate::expert_banks::OriginalPackedProfile as P;
        let check = |profile, positions, protected, request, binding| {
            super::q3_request_binding_scope(
                true,
                "resident-count1-hybrid-chat",
                profile,
                protected,
                positions,
                true,
                request,
                binding,
            )
        };
        assert!(check(P::DaslabIq3, 13, true, true, true).is_ok());
        assert!(check(P::UnslothQ3, 23, true, true, true).is_ok());
        for (profile, positions, protected, request, binding) in [
            (P::DaslabIq3, 23, true, true, true),
            (P::UnslothQ3, 13, true, true, true),
            (P::Q4, 13, true, true, true),
            (P::DaslabIq3, 13, false, true, true),
            (P::DaslabIq3, 13, true, false, true),
            (P::DaslabIq3, 13, true, true, false),
        ] {
            assert!(check(profile, positions, protected, request, binding).is_err());
        }
        assert!(
            super::q3_request_binding_scope(true, "chat", P::DaslabIq3, true, 13, true, true, true)
                .is_err()
        );
        assert!(
            super::q3_request_binding_scope(
                true,
                "resident-count1-hybrid-chat",
                P::DaslabIq3,
                true,
                13,
                false,
                true,
                true
            )
            .is_err()
        );
    }
    #[test]
    fn q3_expert_diagnostic_excludes_other_inputs_and_performance_modes() {
        use crate::expert_banks::OriginalPackedProfile as P;
        assert!(!super::q3_expert_oracle("probe", P::Q4, false, 4, false, None).unwrap());
        assert!(
            super::q3_expert_oracle(
                "resident-count1-hybrid-chat",
                P::UnslothQ3,
                true,
                32,
                true,
                Some("1")
            )
            .unwrap()
        );
        for (mode, profile, protected, positions, empty_fill) in [
            ("probe", P::UnslothQ3, true, 32, true),
            ("resident-count1-hybrid-chat", P::Q4, true, 32, true),
            ("resident-count1-hybrid-chat", P::UnslothQ3, false, 32, true),
            ("resident-count1-hybrid-chat", P::UnslothQ3, true, 4, true),
            ("resident-count1-hybrid-chat", P::UnslothQ3, true, 32, false),
        ] {
            assert!(
                super::q3_expert_oracle(mode, profile, protected, positions, empty_fill, Some("1"))
                    .is_err()
            );
        }
        for raw in ["", "0", "true", "01", " 1", "2"] {
            assert!(
                super::q3_expert_oracle(
                    "resident-count1-hybrid-chat",
                    P::UnslothQ3,
                    true,
                    32,
                    true,
                    Some(raw)
                )
                .is_err()
            );
        }
    }
    #[test]
    fn q3_speed_options_reject_other_models_unprotected_and_noncanonical_inputs() {
        use crate::expert_banks::OriginalPackedProfile as P;
        for option in ["skip-segments", "subgroup-moe", "ideal-prefix"] {
            assert!(
                !super::q3_speed_fixture_option("probe", P::Q4, false, 4, false, option, None)
                    .unwrap()
            );
            assert!(
                super::q3_speed_fixture_option(
                    "resident-count1-hybrid-chat",
                    P::UnslothQ3,
                    true,
                    23,
                    true,
                    option,
                    Some("1")
                )
                .unwrap()
            );
            for (mode, profile, protected, positions, empty) in [
                ("probe", P::UnslothQ3, true, 23, true),
                ("resident-count1-hybrid-chat", P::Q4, true, 23, true),
                ("resident-count1-hybrid-chat", P::UnslothQ3, false, 23, true),
                ("resident-count1-hybrid-chat", P::UnslothQ3, true, 23, false),
            ] {
                assert!(
                    super::q3_speed_fixture_option(
                        mode,
                        profile,
                        protected,
                        positions,
                        empty,
                        option,
                        Some("1")
                    )
                    .is_err()
                );
            }
            for raw in ["0", "", "01", "true"] {
                assert!(
                    super::q3_speed_fixture_option(
                        "resident-count1-hybrid-chat",
                        P::UnslothQ3,
                        true,
                        23,
                        true,
                        option,
                        Some(raw)
                    )
                    .is_err()
                );
            }
        }
        for positions in [4, 32] {
            assert!(
                super::q3_speed_fixture_option(
                    "resident-count1-hybrid-chat",
                    P::UnslothQ3,
                    true,
                    positions,
                    true,
                    "ideal-prefix",
                    Some("1")
                )
                .is_err()
            );
        }
    }
    #[test]
    fn q3_router_reference_requires_the_explicit_expert_diagnostic() {
        assert!(
            super::q3_router_reference_path(false, None)
                .unwrap()
                .is_none()
        );
        assert!(super::q3_router_reference_path(false, Some("public.bin".into())).is_err());
        assert!(super::q3_router_reference_path(true, Some("".into())).is_err());
        assert_eq!(
            super::q3_router_reference_path(true, Some("public.bin".into())).unwrap(),
            Some(std::path::PathBuf::from("public.bin"))
        );
    }
    #[test]
    fn q3_empty_slot_fill_rejects_inherited_other_modes_and_noncanonical_settings() {
        use crate::expert_banks::OriginalPackedProfile as P;
        assert_eq!(
            super::q3_empty_slot_fill("probe", P::Q4, false, None),
            Ok(false)
        );
        for raw in ["0", "1"] {
            assert_eq!(
                super::q3_empty_slot_fill(
                    "resident-count1-hybrid-chat",
                    P::UnslothQ3,
                    true,
                    Some(raw),
                ),
                Ok(raw == "1")
            );
            for (mode, profile, protected) in [
                ("probe", P::UnslothQ3, true),
                ("resident-chat", P::UnslothQ3, true),
                ("iq-quant-probe", P::UnslothQ3, true),
                ("resident-count1-hybrid-chat", P::Q4, true),
                ("resident-count1-hybrid-chat", P::UnslothQ3, false),
            ] {
                assert!(super::q3_empty_slot_fill(mode, profile, protected, Some(raw)).is_err());
            }
        }
        for raw in ["", "2", "01", "+1", "true", "1 "] {
            assert!(
                super::q3_empty_slot_fill(
                    "resident-count1-hybrid-chat",
                    P::UnslothQ3,
                    true,
                    Some(raw),
                )
                .is_err()
            );
        }
    }
    #[test]
    fn q3_public_positions_require_exact_count_profile_mode_and_protection() {
        use crate::expert_banks::OriginalPackedProfile as P;
        assert_eq!(
            super::q3_public_positions("probe", P::Q4, false, None),
            Ok(4)
        );
        for raw in [None, Some("4"), Some("23"), Some("32")] {
            assert_eq!(
                super::q3_public_positions("resident-count1-hybrid-chat", P::UnslothQ3, true, raw)
                    .unwrap(),
                match raw {
                    Some("23") => 23,
                    Some("32") => 32,
                    _ => 4,
                }
            );
        }
        for raw in ["4", "23", "32"] {
            for (mode, profile, protected) in [
                ("probe", P::UnslothQ3, true),
                ("resident-bang", P::UnslothQ3, true),
                ("resident-count1-hybrid-chat", P::Q4, true),
                ("resident-count1-hybrid-chat", P::UnslothQ3, false),
            ] {
                assert!(super::q3_public_positions(mode, profile, protected, Some(raw)).is_err());
            }
        }
        for raw in [
            "", "0", "1", "3", "5", "8", "31", "33", "04", "023", "032", "+23", "+32", "23 ", "32 ",
        ] {
            assert!(
                super::q3_public_positions(
                    "resident-count1-hybrid-chat",
                    P::UnslothQ3,
                    true,
                    Some(raw)
                )
                .is_err()
            );
        }
    }
    #[test]
    fn original_q3_profile_is_explicit_and_cannot_select_unrelated_lab_modes() {
        use crate::expert_banks::OriginalPackedProfile as P;
        assert_eq!(
            super::original_packed_profile("probe", 0x10de, None).unwrap(),
            P::Q4
        );
        assert_eq!(
            super::original_packed_profile(
                "resident-count1-hybrid-chat",
                0x8086,
                Some("unsloth-q3")
            )
            .unwrap(),
            P::UnslothQ3
        );
        for (mode, vendor, raw) in [
            ("probe", 0x8086, "unsloth-q3"),
            ("resident-count1-hybrid-chat", 0x10de, "unsloth-q3"),
            ("resident-count1-hybrid-chat", 0x8086, "q3"),
            ("resident-count1-hybrid-chat", 0x8086, ""),
        ] {
            assert!(super::original_packed_profile(mode, vendor, Some(raw)).is_err());
        }
        for raw in [Some("5368709120"), Some("8589934592")] {
            assert!(
                super::original_count1_setting(
                    P::UnslothQ3,
                    super::Count1Banks::Intel144,
                    "KUNO_LAB_STATIC_CACHE_BYTES",
                    raw
                )
                .is_ok()
            );
        }
        assert!(
            super::original_count1_setting(
                P::UnslothQ3,
                super::Count1Banks::Intel144,
                "KUNO_LAB_STATIC_CACHE_BYTES",
                Some("10737418240")
            )
            .is_err()
        );
        assert!(
            super::original_count1_setting(
                P::Q4,
                super::Count1Banks::Intel144,
                "KUNO_LAB_STATIC_CACHE_BYTES",
                Some("8589934592")
            )
            .is_err()
        );
        for raw in [None, Some("1"), Some("")] {
            assert!(
                super::original_count1_setting(
                    P::UnslothQ3,
                    super::Count1Banks::Intel144,
                    "KUNO_LAB_BLOCK_MOE",
                    raw
                )
                .is_err()
            );
        }
        assert!(
            super::original_count1_setting(
                P::UnslothQ3,
                super::Count1Banks::Intel144,
                "KUNO_LAB_BLOCK_MOE",
                Some("0")
            )
            .is_ok()
        );
        assert!(
            super::original_count1_setting(
                P::UnslothQ3,
                super::Count1Banks::Intel144,
                "KUNO_LAB_NATIVE_DECODE_KERNELS",
                Some("single-all")
            )
            .is_ok()
        );
        assert!(
            super::original_count1_setting(
                P::UnslothQ3,
                super::Count1Banks::Intel144,
                "KUNO_LAB_NATIVE_DECODE_KERNELS",
                Some("single-q4-all")
            )
            .is_err()
        );
        assert!(
            super::original_count1_setting(
                P::Q4,
                super::Count1Banks::Intel144,
                "KUNO_LAB_NATIVE_DECODE_KERNELS",
                Some("single-all")
            )
            .is_err()
        );
        assert!(
            super::original_count1_setting(
                P::Q4,
                super::Count1Banks::Intel144,
                "KUNO_LAB_BLOCK_MOE",
                Some("1")
            )
            .is_ok()
        );
    }

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
    fn resident_expert_calibration_rejects_unprotected_or_unrelated_entries() {
        for raw in [None, Some("0")] {
            assert!(!super::resident_expert_calibration_mode("resident-chat", raw, false).unwrap());
        }
        for raw in [Some(""), Some("true"), Some("2")] {
            assert!(
                super::resident_expert_calibration_mode(
                    "resident-cpu-gpu-expert-calibration",
                    raw,
                    true
                )
                .is_err()
            );
        }
        assert!(super::resident_expert_calibration_mode("resident-chat", Some("1"), true).is_err());
        assert!(
            super::resident_expert_calibration_mode(
                "resident-cpu-gpu-expert-calibration",
                Some("1"),
                false
            )
            .is_err()
        );
        assert_eq!(
            super::resident_expert_calibration_mode(
                "resident-cpu-gpu-expert-calibration",
                Some("1"),
                true
            )
            .is_ok(),
            cfg!(feature = "cpu-fma")
        );
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

#[cfg(test)]
mod daslab_connection_tests {
    use super::*;
    #[test]
    fn own_protected_public_fixture_and_unchanged_unsloth_experiments() {
        use expert_banks::OriginalPackedProfile as P;
        let mode = "resident-count1-hybrid-chat";
        assert_eq!(
            original_packed_profile(mode, 0x8086, Some("daslab-iq3")),
            Ok(P::DaslabIq3)
        );
        assert!(original_packed_profile(mode, 0x10de, Some("daslab-iq3")).is_err());
        assert!(original_packed_profile("probe", 0x8086, Some("daslab-iq3")).is_err());
        for positions in ["4", "13", "32"] {
            assert!(q3_public_positions(mode, P::DaslabIq3, true, Some(positions)).is_ok());
        }
        assert!(q3_public_positions(mode, P::DaslabIq3, true, Some("23")).is_err());
        assert!(q3_public_positions(mode, P::DaslabIq3, false, Some("4")).is_err());
        assert_eq!(
            q3_empty_slot_fill(mode, P::DaslabIq3, true, Some("1")),
            Ok(true)
        );
        assert_eq!(
            q3_speed_fixture_option(
                mode,
                P::DaslabIq3,
                true,
                4,
                true,
                "skip-segments",
                Some("1")
            ),
            Ok(true)
        );
        for option in ["subgroup-moe", "ideal-prefix"] {
            assert!(
                q3_speed_fixture_option(mode, P::DaslabIq3, true, 4, true, option, Some("1"))
                    .is_err()
            );
        }
    }
    #[test]
    fn daslab_prefix13_is_explicit_profile_scoped_and_never_unsloth() {
        use expert_banks::OriginalPackedProfile as P;
        let mode = "resident-count1-hybrid-chat";
        assert_eq!(
            q3_public_positions(mode, P::DaslabIq3, true, Some("13")),
            Ok(13)
        );
        assert!(q3_public_positions(mode, P::UnslothQ3, true, Some("13")).is_err());
        assert!(q3_public_positions(mode, P::DaslabIq3, false, Some("13")).is_err());
        assert_eq!(
            q3_speed_fixture_option(
                mode,
                P::DaslabIq3,
                true,
                13,
                true,
                "ideal-prefix",
                Some("1")
            ),
            Ok(true)
        );
        assert!(
            q3_speed_fixture_option(
                mode,
                P::UnslothQ3,
                true,
                13,
                true,
                "ideal-prefix",
                Some("1")
            )
            .is_err()
        );
    }

    #[test]
    fn full_daslab_ideal_replay_requires_its_exact_protected32_position_window() {
        use expert_banks::OriginalPackedProfile as P;
        let mode = "resident-count1-hybrid-chat";
        assert_eq!(
            q3_speed_fixture_option(
                mode,
                P::DaslabIq3,
                true,
                32,
                true,
                "ideal-prefix",
                Some("1")
            ),
            Ok(true)
        );
        for (selected_mode, protected, positions, empty) in [
            (mode, true, 4, true),
            (mode, true, 23, true),
            (mode, false, 32, true),
            (mode, true, 32, false),
            ("probe", true, 32, true),
        ] {
            assert!(
                q3_speed_fixture_option(
                    selected_mode,
                    P::DaslabIq3,
                    protected,
                    positions,
                    empty,
                    "ideal-prefix",
                    Some("1")
                )
                .is_err()
            );
        }
        for positions in [4, 23, 32] {
            assert!(
                q3_speed_fixture_option(
                    mode,
                    P::DaslabIq3,
                    true,
                    positions,
                    true,
                    "subgroup-moe",
                    Some("1")
                )
                .is_err()
            );
        }
        assert_eq!(
            q3_speed_fixture_option(
                mode,
                P::UnslothQ3,
                true,
                23,
                true,
                "ideal-prefix",
                Some("1")
            ),
            Ok(true)
        );
        for positions in [4, 32] {
            assert!(
                q3_speed_fixture_option(
                    mode,
                    P::UnslothQ3,
                    true,
                    positions,
                    true,
                    "ideal-prefix",
                    Some("1")
                )
                .is_err()
            );
        }
        for raw in ["0", "01", "", "true"] {
            assert!(
                q3_speed_fixture_option(
                    mode,
                    P::DaslabIq3,
                    true,
                    32,
                    true,
                    "ideal-prefix",
                    Some(raw)
                )
                .is_err()
            );
        }
    }
}

#[cfg(test)]
mod q3_integer_model_policy_tests {
    use super::*;
    use expert_banks::OriginalPackedProfile as P;

    #[test]
    fn integer_moe_opt_in_has_exact_public_model_and_position_scope() {
        let mode = "resident-count1-hybrid-chat";
        assert_eq!(
            q3_speed_fixture_option("probe", P::Q4, false, 4, false, "integer-moe", None),
            Ok(false)
        );
        for (profile, positions) in [
            (P::UnslothQ3, 4),
            (P::UnslothQ3, 23),
            (P::UnslothQ3, 32),
            (P::DaslabIq3, 4),
            (P::DaslabIq3, 13),
            (P::DaslabIq3, 32),
        ] {
            assert_eq!(
                q3_speed_fixture_option(
                    mode,
                    profile,
                    true,
                    positions,
                    true,
                    "integer-moe",
                    Some("1")
                ),
                Ok(true)
            );
        }
        for (selected, profile, public, positions, empty) in [
            ("resident-chat", P::UnslothQ3, true, 4, true),
            ("probe", P::UnslothQ3, true, 4, true),
            (mode, P::Q4, true, 4, true),
            (mode, P::UnslothQ3, false, 4, true),
            (mode, P::UnslothQ3, true, 4, false),
            (mode, P::UnslothQ3, true, 13, true),
            (mode, P::DaslabIq3, true, 23, true),
            (mode, P::DaslabIq3, true, 128, true),
        ] {
            assert!(
                q3_speed_fixture_option(
                    selected,
                    profile,
                    public,
                    positions,
                    empty,
                    "integer-moe",
                    Some("1")
                )
                .is_err()
            );
        }
        for raw in ["", "0", "01", " 1", "true", "2"] {
            assert!(
                q3_speed_fixture_option(
                    mode,
                    P::DaslabIq3,
                    true,
                    13,
                    true,
                    "integer-moe",
                    Some(raw)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn integer_moe_requires_every_dependency_and_excludes_other_arithmetic() {
        assert!(
            q3_integer_moe_policy(false, false, false, false, false, false, true, true).is_ok()
        );
        assert!(q3_integer_moe_policy(true, true, true, true, true, true, false, false).is_ok());
        for missing in 0..5 {
            let mut required = [true; 5];
            required[missing] = false;
            assert!(
                q3_integer_moe_policy(
                    true,
                    required[0],
                    required[1],
                    required[2],
                    required[3],
                    required[4],
                    false,
                    false
                )
                .is_err()
            );
        }
        assert!(q3_integer_moe_policy(true, true, true, true, true, true, true, false).is_err());
        // Each actual incompatible arithmetic selection is rejected. Supported
        // direct-device/FMA/work-stealing builds add no blanket rejection;
        // existing CPU/dispatch admission still checks their explicit policies.
        for incompatible in [
            [true, false, false, false, false],
            [false, true, false, false, false],
            [false, false, true, false, false],
            [false, false, false, true, false],
            [false, false, false, false, true],
        ] {
            assert!(
                q3_integer_moe_policy(
                    true,
                    true,
                    true,
                    true,
                    true,
                    true,
                    false,
                    incompatible.into_iter().any(|v| v)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn complete_integer_workgroup_storage_is_admitted_before_device_creation() {
        let required = quant_matvec::iq_moe_integer::MAX_SHARED_BYTES as u32;
        assert_eq!(required, 24_196);
        assert!(q3_integer_moe_shared_limit(false, 0).is_ok());
        assert!(q3_integer_moe_shared_limit(true, required - 1).is_err());
        assert!(q3_integer_moe_shared_limit(true, required).is_ok());
        assert!(q3_integer_moe_shared_limit(true, 32_768).is_ok());
    }
}

#[cfg(test)]
mod shared_original_iq_registration_tests {
    use super::shared_original_iq_probe_registration as selected;
    #[test]
    fn exact_mode_and_argument_count_own_the_public_gate() {
        assert_eq!(selected("iq-shared-original-moe-probe", 3, true), Ok(true));
        for args in [0, 2, 4, 5, usize::MAX] {
            assert!(selected("iq-shared-original-moe-probe", args, true).is_err());
        }
        for mode in [
            "probe",
            "iq-integer-moe-probe",
            "segmented-iq-moe-probe",
            "resident-count1-hybrid-chat",
        ] {
            assert_eq!(selected(mode, 3, false), Ok(false));
            assert!(selected(mode, 3, true).is_err());
        }
    }
}

#[cfg(test)]
mod shared_iq_and_head_fixture_tests {
    use super::{expert_banks::OriginalPackedProfile as P, q3_speed_fixture_option as option};
    #[test]
    fn new_choices_are_scoped_to_exact_public_lowbit_windows() {
        for selected in ["shared-original-moe", "integer-head"] {
            for (profile, positions) in [
                (P::DaslabIq3, 4),
                (P::DaslabIq3, 13),
                (P::DaslabIq3, 32),
                (P::UnslothQ3, 4),
                (P::UnslothQ3, 23),
                (P::UnslothQ3, 32),
            ] {
                assert_eq!(
                    option(
                        "resident-count1-hybrid-chat",
                        profile,
                        true,
                        positions,
                        true,
                        selected,
                        Some("1")
                    ),
                    Ok(true)
                );
                assert!(
                    option(
                        "resident-count1-hybrid-chat",
                        profile,
                        false,
                        positions,
                        true,
                        selected,
                        Some("1")
                    )
                    .is_err()
                );
                assert!(
                    option(
                        "resident-count1-hybrid-chat",
                        profile,
                        true,
                        positions,
                        false,
                        selected,
                        Some("1")
                    )
                    .is_err()
                );
                assert!(
                    option("probe", profile, true, positions, true, selected, Some("1")).is_err()
                );
            }
            for (profile, positions) in [(P::Q4, 4), (P::UnslothQ3, 13), (P::DaslabIq3, 23)] {
                assert!(
                    option(
                        "resident-count1-hybrid-chat",
                        profile,
                        true,
                        positions,
                        true,
                        selected,
                        Some("1")
                    )
                    .is_err()
                );
            }
        }
    }
}

#[cfg(test)]
mod live_member_iq_registration_tests {
    use super::live_member_iq_probe_registration as selected;
    #[test]
    fn dedicated_public_gate_rejects_other_modes_and_variable_arguments() {
        assert_eq!(selected("iq-live-members-moe-probe", 3, true), Ok(true));
        for count in [0, 1, 2, 4, 5, usize::MAX] {
            assert!(selected("iq-live-members-moe-probe", count, true).is_err());
        }
        for mode in [
            "probe",
            "segmented-iq-moe-probe",
            "iq-integer-moe-probe",
            "iq-shared-original-moe-probe",
            "daslab-speculative-probe",
        ] {
            assert_eq!(selected(mode, 3, false), Ok(false));
            assert!(selected(mode, 3, true).is_err());
        }
    }
}
