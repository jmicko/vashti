//! Fixed compiled language generation: real tokenizer, normal routes, greedy
//! target verification and actual rejected-prefix state repair. No oracle
//! tokens, HTTP API or private request inputs. Full target packets are retained
//! for this first diagnostic; an independent CPU200 quality gate is not claimed.
#![allow(dead_code)] // Parent owns registration/protected physical acceptance.
use crate::{
    daslab_profile,
    executor::GpuExecutor,
    expert_banks::OriginalPackedProfile,
    mtp_lab_profile::Profile,
    resident_model::{
        ResidentModel,
        batch::{Config, LatestTargetRound},
    },
    resident_mtp,
};
use cubecl::{
    prelude::*,
    wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device},
};
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits, model::Qwen4Exp};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use zeroize::Zeroizing;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const VOCAB: usize = 248320;
const HIDDEN: usize = 10240;
pub(crate) const MAX_OUTPUTS: usize = if crate::strata_port::enabled() {
    512
} else {
    200
};
pub(crate) const FIXTURE_HOST_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const TOKENIZER_HOST_BYTES: usize = 192 * 1024 * 1024;
const ABS: f32 = 0.006;
const REL: f32 = 0.0003;
const TEMPLATE_REVISION: &str = "fb58e0dbc8399662c0e47c76578c6e878b14f6cf";
const TEMPLATE_FILE: &str = "serve/chat_template.jinja";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fixture {
    FirstCode,
    WarmProse,
}
impl Fixture {
    fn parse(value: &OsStr) -> std::result::Result<Self, &'static str> {
        match value.to_str() {
            Some("first-code") => Ok(Self::FirstCode),
            Some("warm-prose") => Ok(Self::WarmProse),
            _ => Err("compiled language fixture rejected"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::FirstCode => "first-code",
            Self::WarmProse => "warm-prose",
        }
    }
    fn prompt(self) -> &'static str {
        match self {
            Self::FirstCode => {
                "Write a Python function that merges two sorted lists of integers into one sorted list. Explain its time and space complexity."
            }
            Self::WarmProse => {
                "Explain how a hash table works, including collisions, resizing, and average lookup complexity."
            }
        }
    }
    pub(crate) fn prompt_tokens(self) -> usize {
        match self {
            Self::FirstCode => 35,
            Self::WarmProse => 30,
        }
    }
    pub(crate) fn context(self) -> usize {
        self.prompt_tokens() + MAX_OUTPUTS
    }
    fn rendered(self) -> Zeroizing<String> {
        Zeroizing::new(format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            self.prompt()
        ))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeMode {
    Speculative,
    TargetOnly,
}
impl DecodeMode {
    fn parse(raw: Option<&OsStr>) -> std::result::Result<Self, &'static str> {
        match raw {
            None => Ok(Self::Speculative),
            Some(raw) if raw == "target-only-public" => Ok(Self::TargetOnly),
            _ => Err("language decode requires canonical public target-only selection"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Speculative => "speculative",
            Self::TargetOnly => "target-only-public",
        }
    }
}
fn retained_check_selection(value: Option<&OsStr>) -> std::result::Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(v) if v == "two-identical-public" && crate::strata_intel_core::enabled() => Ok(true),
        _ => Err("retained check requires the fixed public Intel core case"),
    }
}
pub(crate) struct Inputs {
    retained_check: bool,
    pub(crate) decode_mode: DecodeMode,
    pub(crate) inline_dispatch: bool,
    pub(crate) fused_sweeps: bool,
    draft_profile: Profile,
    mtp_device: crate::target_lab_build::MtpDevice,
    intel_mtp_memory: Option<crate::target_lab_build::IntelMtpMemory>,
    mtp_q8_head_broadcast: bool,
    mtp_q8_head_shared_scales: bool,
    mtp_english_vocab: bool,
    pub(crate) mtp_batch_repair: bool,
    pub(crate) four_lookahead: bool,
    pub(crate) gpu_qsa_selector: bool,
    pub(crate) frozen_host_sources: bool,
    pub(crate) frozen_host_router: bool,
    pub(crate) frozen_partition_exit: bool,
    pub(crate) frozen_private_tables: bool,
    pub(crate) frozen_workgroup_tables: bool,
    pub(crate) frozen_shared_down_scales: bool,
    pub(crate) frozen_shared_gate_up_factors: bool,
    pub(crate) frozen_four_row_down: bool,
    pub(crate) frozen_two_row_gate_up: bool,
    pub(crate) frozen_packed_host_staging: bool,
    pub(crate) frozen_three_members: bool,
    pub(crate) frozen_gu_integer: bool,
    pub(crate) frozen_down_integer: bool,
    pub(crate) public_iq4_decoder: bool,
    pub(crate) public_integer_verify_head: bool,
    pub(crate) bank_fill: crate::executor::PublicHostBankFillPolicy,
    pub(crate) target_submission_backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    pub(crate) frozen_gu_stride_four: bool,
    pub(crate) frozen_down_stride_four: bool,
    pub(crate) three_gdn_streaming: bool,
    pub(crate) public_grouped_ple_reads: bool,
    pub(crate) public_grouped_status_folds: bool,
    pub(crate) verify3_dense: bool,
    pub(crate) verify3_fused: bool,
    pub(crate) prepared_experts: bool,
    public_existing_fill: bool,
    chunk_io: bool,
    request_reuse: bool,
    live_member_reduction: bool,
    pub(crate) fixture: Fixture,
    directory: PathBuf,
    mtp: PathBuf,
    receipt: PathBuf,
}
pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    build: crate::target_lab_build::Build,
) -> std::result::Result<Inputs, &'static str> {
    if args.len() != 3 || args[1] != "intel" || args[2] != "daslab-language-probe" {
        return Err("language lab requires fixed public B70 target");
    }
    let mut values = BTreeMap::new();
    for (key, value) in env {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|p| key.as_encoded_bytes().starts_with(p))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some(
                "KUNO_LAB_LANGUAGE_PUBLIC_ONLY"
                    | "KUNO_LAB_LANGUAGE_DECODE"
                    | "KUNO_LAB_LANGUAGE_FIXTURE"
                    | "KUNO_LAB_LANGUAGE_SESSION"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_TEST_MTP"
                    | "KUNO_PUBLIC_RECEIPT"
                    | "KUNO_LAB_STATIC_CACHE_BYTES"
                    | "KUNO_LAB_DEVICE_DISPATCH"
                    | "KUNO_LAB_MTP_SWEEPS"
                    | "KUNO_LAB_MTP_VOCAB"
                    | "KUNO_LAB_MTP_PROFILE"
                    | "KUNO_LAB_MTP_DEVICE"
                    | "KUNO_LAB_MTP_REPAIR"
                    | "KUNO_LAB_TARGET_LOOKAHEAD"
                    | "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES"
                    | "KUNO_PUBLIC_MTP_INTEL_USED_BYTES"
                    | "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID"
                    | "KUNO_LAB_MTP_Q8_HEAD"
                    | "KUNO_LAB_TARGET_QSA"
                    | "KUNO_LAB_TARGET_SOURCES"
                    | "KUNO_LAB_FROZEN_PARTITION_EXIT"
                    | "KUNO_LAB_FROZEN_TABLES"
                    | "KUNO_LAB_FROZEN_DOWN_SCALES"
                    | "KUNO_LAB_FROZEN_GU_FACTORS"
                    | "KUNO_LAB_FROZEN_DOWN_ROWS"
                    | "KUNO_LAB_FROZEN_GU_ROWS"
                    | "KUNO_LAB_FROZEN_HOST_STAGE"
                    | "KUNO_LAB_FROZEN_MEMBERS"
                    | "KUNO_LAB_FROZEN_GU_MATH"
                    | "KUNO_LAB_FROZEN_DOWN_MATH"
                    | "KUNO_LAB_TARGET_IQ4_DECODER"
                    | "KUNO_LAB_TARGET_HEAD"
                    | "KUNO_LAB_FROZEN_BANK_FILL"
                    | "KUNO_LAB_TARGET_SUBMISSION_BACKLOG"
                    | "KUNO_LAB_FROZEN_GU_SCHEDULE"
                    | "KUNO_LAB_FROZEN_DOWN_SCHEDULE"
                    | "KUNO_LAB_TARGET_GDN"
                    | "KUNO_LAB_TARGET_PLE_READS"
                    | "KUNO_LAB_TARGET_STATUS_FOLDS"
                    | "KUNO_LAB_TARGET_ROUTER"
                    | "KUNO_LAB_TARGET_DENSE"
                    | "KUNO_LAB_TARGET_EXPERTS"
                    | "KUNO_LAB_TARGET_FILL"
                    | "KUNO_LAB_TARGET_IO"
                    | "KUNO_LAB_TARGET_REUSE"
                    | "KUNO_LAB_TARGET_MOE"
            )
        ) || value.is_empty()
            || values.insert(key, value).is_some()
        {
            return Err("language lab rejects inherited experiments");
        }
    }
    if values.remove(OsStr::new("KUNO_LAB_LANGUAGE_PUBLIC_ONLY")) != Some("1".into())
        || values.remove(OsStr::new("KUNO_LAB_STATIC_CACHE_BYTES"))
            != Some(crate::strata_port::static_cache_bytes().to_string().into())
    {
        return Err("compiled language/cache gates missing");
    }
    let retained_check = retained_check_selection(
        values
            .remove(OsStr::new("KUNO_LAB_LANGUAGE_SESSION"))
            .as_deref(),
    )?;
    let decode_mode = DecodeMode::parse(
        values
            .remove(OsStr::new("KUNO_LAB_LANGUAGE_DECODE"))
            .as_deref(),
    )?;
    if decode_mode == DecodeMode::TargetOnly
        && [
            "KUNO_LAB_MTP_SWEEPS",
            "KUNO_LAB_MTP_DEVICE",
            "KUNO_LAB_MTP_REPAIR",
            "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES",
            "KUNO_PUBLIC_MTP_INTEL_USED_BYTES",
            "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID",
            "KUNO_LAB_MTP_VOCAB",
            "KUNO_LAB_MTP_PROFILE",
            "KUNO_LAB_MTP_Q8_HEAD",
        ]
        .iter()
        .any(|key| values.contains_key(OsStr::new(key)))
    {
        return Err("target-only language rejects every MTP selector");
    }
    let raw = values.remove(OsStr::new("KUNO_LAB_DEVICE_DISPATCH"));
    let inline_dispatch = build.selected(raw.as_deref())?;
    let mut choice = |key: &str, canonical: &str| match values.remove(OsStr::new(key)) {
        None => Ok(false),
        Some(v) if v == canonical => Ok(true),
        _ => Err("language policy requires canonical selection"),
    };
    let prepared_experts = choice("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3")?;
    let public_existing_fill = choice("KUNO_LAB_TARGET_FILL", "direct-public")?;
    let chunk_io = choice("KUNO_LAB_TARGET_IO", "chunk")?;
    let fused_sweeps = choice("KUNO_LAB_MTP_SWEEPS", "fused")?;
    let gpu_qsa_selector = match values.remove(OsStr::new("KUNO_LAB_TARGET_QSA")) {
        None => false,
        Some(v) if v == "chronological-public" => true,
        _ => return Err("target QSA requires canonical chronological-public"),
    };
    let frozen_host_sources = match values.remove(OsStr::new("KUNO_LAB_TARGET_SOURCES")) {
        None => false,
        Some(v) if v == "frozen-host-public" => true,
        _ => return Err("target sources require canonical frozen-host-public"),
    };
    let frozen_partition_exit = match values.remove(OsStr::new("KUNO_LAB_FROZEN_PARTITION_EXIT")) {
        None => false,
        Some(v) if v == "1" && frozen_host_sources => true,
        _ => return Err("partition exit requires explicit frozen public sources"),
    };
    let (frozen_private_tables, frozen_workgroup_tables) =
        match values.remove(OsStr::new("KUNO_LAB_FROZEN_TABLES")) {
            None => (false, false),
            Some(v) if v == "private-public" && frozen_host_sources && frozen_partition_exit => {
                (true, false)
            }
            Some(v) if v == "workgroup-public" && frozen_host_sources && frozen_partition_exit => {
                (false, true)
            }
            _ => return Err("tables require canonical frozen public partition exit"),
        };
    let frozen_shared_down_scales = match values.remove(OsStr::new("KUNO_LAB_FROZEN_DOWN_SCALES")) {
        None => false,
        Some(v) if v == "shared-public" && frozen_host_sources && frozen_partition_exit => true,
        _ => return Err("down scales require canonical frozen public partition exit"),
    };
    let frozen_shared_gate_up_factors =
        match values.remove(OsStr::new("KUNO_LAB_FROZEN_GU_FACTORS")) {
            None => false,
            Some(v) if v == "shared-public" && frozen_host_sources && frozen_partition_exit => true,
            _ => return Err("gate/up factors require canonical frozen public partition exit"),
        };
    let frozen_host_router = match values.remove(OsStr::new("KUNO_LAB_TARGET_ROUTER")) {
        None => false,
        Some(v) if v == "device-frozen-public" && frozen_host_sources => true,
        _ => return Err("source router requires canonical device-frozen-public"),
    };
    let frozen_four_row_down = match values.remove(OsStr::new("KUNO_LAB_FROZEN_DOWN_ROWS")) {
        None => false,
        Some(v) if v == "four-public" && frozen_host_sources && frozen_partition_exit => true,
        _ => return Err("four-row down requires canonical public frozen partition exit"),
    };
    let frozen_two_row_gate_up = match values.remove(OsStr::new("KUNO_LAB_FROZEN_GU_ROWS")) {
        None => false,
        Some(v)
            if v == "two-public"
                && frozen_host_sources
                && frozen_partition_exit
                && !frozen_private_tables
                && !frozen_workgroup_tables
                && !frozen_shared_gate_up_factors =>
        {
            true
        }
        _ => return Err("two-row gate/up requires canonical frozen exit without tables/factors"),
    };
    let frozen_packed_host_staging = match values.remove(OsStr::new("KUNO_LAB_FROZEN_HOST_STAGE")) {
        None => false,
        Some(v)
            if v == "packed-public"
                && frozen_host_sources
                && frozen_partition_exit
                && frozen_host_router
                && !frozen_private_tables
                && !frozen_workgroup_tables
                && !frozen_shared_gate_up_factors
                && !frozen_shared_down_scales =>
        {
            true
        }
        _ => {
            return Err(
                "packed staging requires original public frozen GPU router and plain quantization",
            );
        }
    };
    let frozen_three_members = match values.remove(OsStr::new("KUNO_LAB_FROZEN_MEMBERS")) {
        None => false,
        Some(v)
            if v == "three-public"
                && frozen_host_sources
                && frozen_partition_exit
                && frozen_host_router
                && !frozen_private_tables
                && !frozen_workgroup_tables
                && !frozen_shared_gate_up_factors
                && !frozen_shared_down_scales
                && !frozen_two_row_gate_up =>
        {
            true
        }
        _ => {
            return Err(
                "three-member arithmetic requires canonical plain public frozen GPU router",
            );
        }
    };
    let raw_profile = values.remove(OsStr::new("KUNO_LAB_MTP_PROFILE"));
    let draft_profile = Profile::parse(raw_profile.as_deref())?;
    let (mtp_q8_head_broadcast, mtp_q8_head_shared_scales) = match values
        .remove(OsStr::new("KUNO_LAB_MTP_Q8_HEAD"))
    {
        None => (false, false),
        Some(v)
            if v == "broadcast-public"
                && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8) =>
        {
            (true, false)
        }
        Some(v)
            if v == "shared-public"
                && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8) =>
        {
            (false, true)
        }
        _ => return Err("head scales require canonical public self-contained Q8 draft"),
    };
    let raw = values.remove(OsStr::new("KUNO_LAB_TARGET_MOE"));
    let live_member_reduction =
        crate::target_lab_build::live_member_reduction(raw.as_deref(), prepared_experts)?;
    let raw = values.remove(OsStr::new("KUNO_LAB_TARGET_REUSE"));
    let request_reuse =
        crate::target_lab_build::request_reuse(raw.as_deref(), inline_dispatch, chunk_io)?;
    let (verify3_dense, verify3_fused) = match values.remove(OsStr::new("KUNO_LAB_TARGET_DENSE")) {
        None => (false, false),
        Some(v) if v == "verify3" => (true, false),
        Some(v) if v == "verify3-fused" => (true, true),
        _ => return Err("language dense policy requires canonical selection"),
    };
    if frozen_host_sources
        && (!request_reuse || !public_existing_fill || prepared_experts || live_member_reduction)
    {
        return Err("frozen sources require original public language request scope");
    }
    let fixture = Fixture::parse(
        values
            .remove(OsStr::new("KUNO_LAB_LANGUAGE_FIXTURE"))
            .as_deref()
            .ok_or("language fixture missing")?,
    )?;
    let raw_gdn = values.remove(OsStr::new("KUNO_LAB_TARGET_GDN"));
    let three_gdn_streaming = crate::target_lab_build::three_gdn_streaming(
        raw_gdn.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let raw_vocab = values.remove(OsStr::new("KUNO_LAB_MTP_VOCAB"));
    let mtp_english_vocab = draft_profile.parse_public_english(
        raw_vocab.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && fused_sweeps
            && !prepared_experts
            && !live_member_reduction,
        mtp_q8_head_broadcast,
    )? || crate::strata_port::enabled();
    let original_public_target_scope = frozen_host_sources
        && frozen_host_router
        && frozen_partition_exit
        && verify3_dense
        && verify3_fused
        && request_reuse
        && chunk_io
        && inline_dispatch
        && public_existing_fill
        && !frozen_private_tables
        && !frozen_workgroup_tables
        && !frozen_shared_down_scales
        && !frozen_shared_gate_up_factors
        && !frozen_two_row_gate_up
        && !frozen_packed_host_staging
        && !frozen_three_members
        && !prepared_experts
        && !live_member_reduction;
    if decode_mode == DecodeMode::TargetOnly
        && (!original_public_target_scope || three_gdn_streaming)
    {
        return Err("target-only language requires the fixed original frozen public target scope");
    }
    let integer_scope = original_public_target_scope
        && match decode_mode {
            DecodeMode::Speculative => {
                fused_sweeps
                    && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
                    && !mtp_q8_head_broadcast
                    && (!mtp_english_vocab || crate::strata_port::enabled())
            }
            // Target-only has no draft engine, so no draft-sweep prerequisite is
            // fabricated. Every original target prerequisite remains mandatory.
            DecodeMode::TargetOnly => true,
        };
    let raw_gu_math = values.remove(OsStr::new("KUNO_LAB_FROZEN_GU_MATH"));
    let frozen_gu_integer =
        crate::target_lab_build::frozen_gu_integer(raw_gu_math.as_deref(), integer_scope)?;
    let raw_gu_schedule = values.remove(OsStr::new("KUNO_LAB_FROZEN_GU_SCHEDULE"));
    let frozen_gu_stride_four = crate::target_lab_build::frozen_gu_stride_four(
        raw_gu_schedule.as_deref(),
        frozen_gu_integer,
    )?;
    let raw_down_schedule = values.remove(OsStr::new("KUNO_LAB_FROZEN_DOWN_SCHEDULE"));
    let frozen_down_stride_four = crate::target_lab_build::frozen_down_stride_four(
        raw_down_schedule.as_deref(),
        original_public_target_scope && frozen_four_row_down,
    )?;
    let raw_mtp_device = values.remove(OsStr::new("KUNO_LAB_MTP_DEVICE"));
    let mtp_device = crate::target_lab_build::MtpDevice::parse(
        raw_mtp_device.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && fused_sweeps
            && (!mtp_english_vocab || crate::strata_port::enabled()),
    )?;
    let total = values.remove(OsStr::new("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES"));
    let used = values.remove(OsStr::new("KUNO_PUBLIC_MTP_INTEL_USED_BYTES"));
    let device = values.remove(OsStr::new("KUNO_PUBLIC_MTP_INTEL_DEVICE_ID"));
    let intel_mtp_memory = crate::target_lab_build::IntelMtpMemory::parse(
        mtp_device,
        total.as_deref(),
        used.as_deref(),
        device.as_deref(),
    )?;
    let raw_mtp_repair = values.remove(OsStr::new("KUNO_LAB_MTP_REPAIR"));
    let mtp_batch_repair = crate::target_lab_build::public_mtp_batch_repair(
        raw_mtp_repair.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
            && !mtp_q8_head_broadcast
            && (!mtp_english_vocab || crate::strata_port::enabled()),
    )?;
    let raw_down_math = values.remove(OsStr::new("KUNO_LAB_FROZEN_DOWN_MATH"));
    let frozen_down_integer = crate::target_lab_build::frozen_down_integer(
        raw_down_math.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && fused_sweeps
            && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
            && !mtp_q8_head_broadcast
            && (!mtp_english_vocab || crate::strata_port::enabled())
            && !frozen_four_row_down
            && !frozen_down_stride_four,
    )?;
    let raw_iq4_decoder = values.remove(OsStr::new("KUNO_LAB_TARGET_IQ4_DECODER"));
    let public_iq4_decoder = crate::target_lab_build::public_iq4_decoder(
        raw_iq4_decoder.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
            && !mtp_q8_head_broadcast
            && (!mtp_english_vocab || crate::strata_port::enabled())
            && !frozen_down_stride_four,
    )?;
    let raw_bank_fill = values.remove(OsStr::new("KUNO_LAB_FROZEN_BANK_FILL"));
    let bank_fill = crate::target_lab_build::public_host_bank_fill(
        raw_bank_fill.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
            && !mtp_q8_head_broadcast
            && (!mtp_english_vocab || crate::strata_port::enabled()),
    )?;
    if mtp_batch_repair && bank_fill != crate::executor::PublicHostBankFillPolicy::Original {
        return Err("MTP batch repair requires original public bank fill");
    }
    let raw_backlog = values.remove(OsStr::new("KUNO_LAB_TARGET_SUBMISSION_BACKLOG"));
    let target_submission_backlog = crate::target_lab_build::public_target_submission_backlog(
        raw_backlog.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
            && !mtp_q8_head_broadcast
            && (!mtp_english_vocab || crate::strata_port::enabled()),
    )?;
    let raw_ple_reads = values.remove(OsStr::new("KUNO_LAB_TARGET_PLE_READS"));
    let public_grouped_ple_reads = crate::target_lab_build::public_grouped_ple_reads(
        raw_ple_reads.as_deref(),
        original_public_target_scope && (!mtp_english_vocab || crate::strata_port::enabled()),
    )?;
    let raw_status_folds = values.remove(OsStr::new("KUNO_LAB_TARGET_STATUS_FOLDS"));
    let public_grouped_status_folds = crate::target_lab_build::public_grouped_status_folds(
        raw_status_folds.as_deref(),
        original_public_target_scope && (!mtp_english_vocab || crate::strata_port::enabled()),
    )?;
    let raw_lookahead = values.remove(OsStr::new("KUNO_LAB_TARGET_LOOKAHEAD"));
    let four_lookahead = crate::target_lab_build::public_four_lookahead(
        raw_lookahead.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && frozen_gu_integer
            && frozen_down_integer
            && fused_sweeps
            && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
            && !mtp_q8_head_broadcast
            && (!mtp_english_vocab || crate::strata_port::enabled())
            && (!mtp_batch_repair || public_grouped_ple_reads)
            && !public_iq4_decoder
            && !three_gdn_streaming
            && (bank_fill == crate::executor::PublicHostBankFillPolicy::Original
                || crate::strata_port::enabled()),
    )?;
    let raw_head = values.remove(OsStr::new("KUNO_LAB_TARGET_HEAD"));
    let public_integer_verify_head = crate::target_lab_build::public_integer_verify_head(
        raw_head.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && (crate::strata_port::enabled() || draft_profile == Profile::SelfContainedQ8)
            && !mtp_q8_head_broadcast
            && (!mtp_english_vocab || crate::strata_port::enabled())
            && frozen_gu_integer
            && frozen_down_integer
            && public_grouped_ple_reads
            && !public_iq4_decoder
            && (bank_fill == crate::executor::PublicHostBankFillPolicy::Original
                || crate::strata_intel_core::enabled()),
    )?;
    if cfg!(feature = "strata-eight-window")
        && !crate::strata_intel_core::enabled()
        && (public_integer_verify_head
            || crate::strata_port::fused_sources()
            || crate::strata_port::wide_experts()
            || crate::strata_port::expert_lanes() != 8)
    {
        return Err("eight-window isolated integer expert/default dense scope");
    }
    let mut path = |key: &str| {
        values
            .remove(OsStr::new(key))
            .map(PathBuf::from)
            .ok_or("language public path missing")
    };
    Ok(Inputs {
        retained_check,
        decode_mode,
        inline_dispatch,
        fused_sweeps,
        draft_profile,
        mtp_device,
        intel_mtp_memory,
        mtp_q8_head_broadcast,
        mtp_q8_head_shared_scales,
        mtp_english_vocab,
        mtp_batch_repair,
        four_lookahead,
        gpu_qsa_selector,
        frozen_host_sources,
        frozen_host_router,
        frozen_partition_exit,
        frozen_private_tables,
        frozen_workgroup_tables,
        frozen_shared_down_scales,
        frozen_shared_gate_up_factors,
        frozen_four_row_down,
        frozen_two_row_gate_up,
        frozen_packed_host_staging,
        frozen_three_members,
        frozen_gu_integer,
        frozen_down_integer,
        public_iq4_decoder,
        public_integer_verify_head,
        bank_fill,
        target_submission_backlog,
        frozen_gu_stride_four,
        frozen_down_stride_four,
        three_gdn_streaming,
        public_grouped_ple_reads,
        public_grouped_status_folds,
        verify3_dense,
        verify3_fused,
        prepared_experts,
        public_existing_fill,
        chunk_io,
        request_reuse,
        live_member_reduction,
        fixture,
        directory: path("KUNO_TEST_Q3_DIR")?,
        mtp: path("KUNO_TEST_MTP")?,
        receipt: path("KUNO_PUBLIC_RECEIPT")?,
    })
}
pub(crate) fn validate_limits(l: &wgpu::Limits) -> std::result::Result<(), &'static str> {
    crate::daslab_target_rows_probe::validate_limits(l)
}

fn tokenize(
    file: &GgufFile,
    fixture: Fixture,
    eos: u32,
) -> Result<(Qwen35Tokenizer, Zeroizing<Vec<u32>>)> {
    let limits = if crate::strata_intel_core::enabled() {
        crate::strata_engine::input::tokenizer_limits(TOKENIZER_HOST_BYTES)
    } else {
        TokenizerLimits {
            max_vocabulary: VOCAB,
            max_input_bytes: 1024,
            max_output_tokens: 64,
            max_piece_bytes: 1024,
            max_merge_steps: 8192,
            max_decoded_bytes: 4096,
            max_decode_chunk_bytes: 16384,
            max_allocation_bytes: TOKENIZER_HOST_BYTES,
            ..TokenizerLimits::default()
        }
    };
    let token_capacity = limits.max_output_tokens;
    let tokenizer = Qwen35Tokenizer::from_gguf(file, limits)?;
    if tokenizer.vocabulary_len() != VOCAB || tokenizer.eos_token_id() != eos {
        return Err("language original tokenizer identity rejected".into());
    }
    let rendered = fixture.rendered();
    let ids = Zeroizing::new(tokenizer.encode(&rendered, true)?);
    let decoded = Zeroizing::new(tokenizer.decode(&ids)?);
    if ids.len() != fixture.prompt_tokens()
        || ids.capacity() > token_capacity
        || decoded.as_str() != rendered.as_str()
        || ids.iter().any(|&t| t >= VOCAB as u32)
    {
        return Err("language compiled template encoding rejected".into());
    }
    Ok((tokenizer, ids))
}
/// Actual target predictions alone define the accepted prefix. EOS clips it
/// before restoring state, including any provisional suffix after terminalEOS.
fn accepted_prefix(
    inputs: &[u32],
    predicted: &[u32],
    eos: u32,
) -> std::result::Result<usize, &'static str> {
    if !(1..=if crate::strata_port::enabled() {
        crate::strata_port::verify_capacity()
    } else {
        3
    })
        .contains(&inputs.len())
        || predicted.len() != inputs.len()
        || eos >= VOCAB as u32
        || inputs.iter().chain(predicted).any(|&t| t >= VOCAB as u32)
    {
        return Err("language acceptance geometry rejected");
    }
    let mut accepted = inputs.len();
    for row in 1..inputs.len() {
        if inputs[row] != predicted[row - 1] {
            accepted = row;
            break;
        }
    }
    if let Some(index) = predicted[..accepted].iter().position(|&t| t == eos) {
        accepted = index + 1;
    }
    Ok(accepted)
}
fn teacher_input(prompt: &[u32], position: usize, draft_cursor: usize) -> Result<Option<u32>> {
    if !matches!(prompt.len(), 30 | 35)
        || position >= prompt.len()
        || draft_cursor != position
        || prompt.iter().any(|&t| t >= VOCAB as u32)
    {
        return Err("language actual prompt teacher alignment rejected".into());
    }
    Ok(prompt.get(position + 1).copied())
}
fn repair_vector(expected: &[f32], actual: &[f32], width: usize) -> Result<f32> {
    if expected.len() != width || actual.len() != width {
        return Err("language repair complete vector shape rejected".into());
    }
    let mut maximum = 0f32;
    for (&r, &a) in expected.iter().zip(actual) {
        let difference = (a - r).abs();
        if !r.is_finite()
            || !a.is_finite()
            || (!crate::strata_port::enabled() && difference > ABS + REL * r.abs())
        {
            return Err("language real repaired vector rejected".into());
        }
        maximum = maximum.max(difference);
    }
    Ok(maximum)
}
fn owned_fixture_bytes(
    vectors: &[&Zeroizing<Vec<f32>>],
    ids: &[&Zeroizing<Vec<u32>>],
) -> Result<usize> {
    let mut bytes = std::mem::size_of::<Statistics>() + 1024;
    for v in vectors {
        bytes = bytes
            .checked_add(v.capacity() * 4)
            .ok_or("language host extent overflow")?;
    }
    for v in ids {
        bytes = bytes
            .checked_add(v.capacity() * 4)
            .ok_or("language host extent overflow")?;
    }
    if bytes > FIXTURE_HOST_BYTES {
        return Err("language fixed host owner ceiling rejected".into());
    }
    Ok(bytes)
}
fn elapsed(start: Instant) -> Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).map_err(Into::into)
}
#[derive(Default)]
struct Statistics {
    rounds: u64,
    proposed: u64,
    accepted_drafts: u64,
    continuations: u64,
    draft_ns: u64,
    verify_ns: u64,
    repair_ns: u64,
    repairs: u64,
    state_replay_repairs: u64,
    completed_windows: [u64; 9],
    accepted_by_window: [u64; 9],
    repair_logit_max: f32,
    repair_hidden_max: f32,
}
impl Statistics {
    fn target_only_record(
        &self,
        mode: DecodeMode,
        draft_present: bool,
        emitted: usize,
    ) -> Result<Option<serde_json::Value>> {
        if mode == DecodeMode::Speculative {
            if !draft_present {
                return Err("speculative actual draft missing".into());
            }
            return Ok(None);
        }
        if draft_present
            || !(1..=MAX_OUTPUTS).contains(&emitted)
            || self.rounds != self.continuations
            || self.continuations != (emitted - 1) as u64
            || self.proposed != 0
            || self.accepted_drafts != 0
            || self.draft_ns != 0
            || self.repair_ns != 0
            || self.repairs != 0
            || self.repair_logit_max != 0.
            || self.repair_hidden_max != 0.
            || self.continuations > 0 && self.verify_ns == 0
        {
            return Err("target-only actual count1/no-draft accounting rejected".into());
        }
        Ok(Some(serde_json::json!({"count1_chunks":self.rounds,
            "completed_target_rows":self.continuations,"draft_executed":false,
            "draft_steps":0,"target_repair_recomputations":0})))
    }
    fn add(
        &mut self,
        count: usize,
        accepted: usize,
        draft: u64,
        verify: u64,
        repair: u64,
        recomputed: bool,
    ) -> Result<()> {
        self.add_completed(
            false,
            count,
            accepted,
            crate::target_lab_build::LookaheadPhase {
                draft_ns: draft,
                third_follow_ns: 0,
                verify_ns: verify,
                repair_ns: repair,
            },
            recomputed,
        )
    }
    fn add_completed(
        &mut self,
        four: bool,
        count: usize,
        accepted: usize,
        phase: crate::target_lab_build::LookaheadPhase,
        recomputed: bool,
    ) -> Result<()> {
        if !(1..=if four {
            crate::strata_port::verify_capacity()
        } else {
            3
        })
            .contains(&count)
            || accepted == 0
            || accepted > count
        {
            return Err("language count geometry rejected".into());
        }
        let crate::target_lab_build::LookaheadPhase {
            draft_ns: draft,
            verify_ns: verify,
            repair_ns: repair,
            ..
        } = phase;
        fn add(n: &mut u64, v: u64) -> Result<()> {
            *n = n.checked_add(v).ok_or("language counter overflow")?;
            Ok(())
        }
        add(&mut self.completed_windows[count], 1)?;
        add(&mut self.accepted_by_window[count], accepted as u64)?;
        add(&mut self.rounds, 1)?;
        add(&mut self.proposed, (count - 1) as u64)?;
        add(&mut self.accepted_drafts, (accepted - 1) as u64)?;
        add(&mut self.continuations, accepted as u64)?;
        add(&mut self.draft_ns, draft)?;
        add(&mut self.verify_ns, verify)?;
        add(&mut self.repair_ns, repair)?;
        add(&mut self.repairs, recomputed as u64)?;
        Ok(())
    }
}
async fn draft_client(
    fused_sweeps: bool,
    selection: crate::target_lab_build::MtpDevice,
    target_client: &ComputeClient<WgpuRuntime>,
    target_device: Option<&wgpu::Device>,
) -> Result<(ComputeClient<WgpuRuntime>, wgpu::Limits, serde_json::Value)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapters: Vec<_> = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .filter(|a| {
            a.get_info().vendor == selection.vendor()
                && a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
        })
        .collect();
    if adapters.len() != 1 {
        return Err("speculative draft GPU ambiguous".into());
    }
    let adapter = adapters.remove(0);
    let info = adapter.get_info();
    let target_info = target_device.map(wgpu::Device::adapter_info);
    if selection == crate::target_lab_build::MtpDevice::Intel && target_info.as_ref() != Some(&info)
    {
        return Err("Intel MTP adapter differs from the actual target physical GPU".into());
    }
    let limits = adapter.limits();
    resident_mtp::validate_limits(&limits)?;
    if fused_sweeps {
        resident_mtp::validate_fused_sweep_limits(&limits)?;
    }
    if !adapter.features().contains(wgpu::Features::SUBGROUP)
        || info.subgroup_min_size < 16
        || info.subgroup_max_size > 32
    {
        return Err("speculative draft subgroups rejected".into());
    }
    let (device, queue) = crate::strata_matrix::request_device(
        &adapter,
        &wgpu::DeviceDescriptor {
            label: Some("Kuno fixed public native MTP"),
            required_features: wgpu::Features::SUBGROUP
                | if crate::strata_intel_core::enabled() {
                    wgpu::Features::SHADER_INT64
                } else {
                    wgpu::Features::empty()
                },
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            ..Default::default()
        },
    )
    .await
    .map_err(|_| "speculative draft device setup failed")?;
    let distinct_device = target_device
        .map(|target| crate::target_lab_build::mtp_distinct_vulkan_devices(target, &device))
        .transpose()?;
    let selected = init_device(
        WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: wgpu::Backend::Vulkan,
        },
        RuntimeOptions {
            tasks_max: if crate::strata_intel_core::enabled() {
                32768
            } else {
                256
            },
            memory_config: crate::private_arena::required_memory_configuration(),
        },
    );
    let client = WgpuRuntime::client(&selected);
    if crate::strata_port::profile_target() {
        client.set_profiling_paused(true);
    }
    let distinct_runtime = !std::ptr::eq(client.info(), target_client.info());
    let runtime_id = match selected {
        cubecl::wgpu::WgpuDevice::Existing(id) => id,
        _ => return Err("MTP explicit independent runtime identity missing".into()),
    };
    let receipt = crate::target_lab_build::mtp_device_receipt(
        selection,
        target_info.as_ref(),
        &info,
        distinct_device,
        distinct_runtime,
        runtime_id,
    )?;
    Ok((client, limits, receipt))
}

fn process_disk_read_bytes() -> Result<u64> {
    let fields = std::fs::read_to_string("/proc/self/io")?;
    fields
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes: "))
        .ok_or("public fixture process I/O counter missing")?
        .trim()
        .parse()
        .map_err(Into::into)
}

fn fill_counter_record(c: crate::executor::batch_moe::PublicFillCounters) -> serde_json::Value {
    serde_json::json!({"first_touch_fills":c.first_touch_fills,"evictions":c.evictions,
        "direct_windows":c.direct_windows,"direct_components":c.direct_components,
        "direct_bytes":c.direct_bytes})
}

pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    input: Inputs,
    host_device: Option<&wgpu::Device>,
    host_queue: Option<&wgpu::Queue>,
) -> Result<()> {
    if input.frozen_host_sources != host_device.is_some() {
        return Err("public frozen sources require exact parent device selection".into());
    }
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    validate_limits(limits)?;
    std::panic::set_hook(Box::new(|info| {
        if let Some(location) = info.location() {
            eprintln!(
                "public_language_failure panic=true source_file={} source_line={}",
                location.file(),
                location.line()
            );
        } else {
            eprintln!("public_language_failure panic=true");
        }
    }));
    let paths: Vec<_> = (1..=2)
        .map(|i| {
            input.directory.join(format!(
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    if file.file_len() != 83_617_662_656 {
        return Err("language original DAS extent rejected".into());
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(name, t)| (name.as_str(), t.dtype().gguf_id())),
    )
    .map_err(|_| "language original DAS inventory rejected")?;
    let sidecar = if input.decode_mode == DecodeMode::TargetOnly {
        None
    } else {
        let sidecar = GgufFile::open(&input.mtp, GgufLimits::default())?;
        input.draft_profile.validate_sidecar(&sidecar)?;
        Some(sidecar)
    };
    // SAFETY: root's protected runner verifies the immutable original target
    // shards and, only for speculative mode, the explicitly selected sidecar.
    unsafe {
        file.enable_mapped_weights()?;
        if let Some(sidecar) = &sidecar {
            sidecar.enable_mapped_weights()?;
        }
    }
    let graph = Qwen4Exp::bind(&file)?;
    let draft_graph = sidecar
        .as_ref()
        .map(|sidecar| input.draft_profile.bind(sidecar, graph.bound_graph()))
        .transpose()?;
    let (tokenizer, prompt) = tokenize(&file, input.fixture, graph.eos_token())?;
    let intel_mtp_admission = input
        .intel_mtp_memory
        .map(|proof| {
            crate::target_lab_build::intel_mtp_preflight_with_down_math(
                proof,
                host_device.ok_or("Intel MTP actual target device missing")?,
                client,
                graph.bound_graph(),
                input.frozen_gu_integer,
                input.frozen_down_integer,
                input.draft_profile,
            )
        })
        .transpose()?;
    if crate::strata_intel_core::enabled() {
        // The compiled-public fixture exercises the same bounded reader and
        // Rust tokenizer intended for a separately protected arbitrary caller.
        // It still accepts NO private stdin or public-receipt content override.
        drop(prompt);
        let public_request = || {
            let rendered = input.fixture.rendered();
            crate::strata_engine::input::request(
                &mut std::io::Cursor::new(rendered.as_bytes()),
                &tokenizer,
                crate::strata_engine::input::Controls {
                    outputs: if cfg!(feature = "strata-public-cap-check") {
                        16
                    } else {
                        MAX_OUTPUTS
                    },
                    eos: graph.eos_token(),
                    sampling: crate::strata_sampler::Params {
                        greedy: true,
                        temperature: 0.,
                        ..Default::default()
                    },
                    speculation: crate::strata_speculation::Config::native(4),
                    minimum: 0.5,
                },
            )
        };
        let request = public_request()?;
        if request.geometry()?.0 != input.fixture.prompt_tokens() {
            return Err("native fixture input geometry rejected".into());
        }
        let mut emitted = Zeroizing::new(Vec::with_capacity(MAX_OUTPUTS));
        // SAFETY: the protected original-source launcher owns full authentication,
        // memory/driver admission, swap/core protection and a redacted panic hook.
        let cancel = AtomicBool::new(false);
        // The complete selected engine has passed its audit. These process
        // counters are already maintained by native execution; snapshot only
        // three fixed PUBLIC phases/request, with no GPU work or kernel query.
        // Keep the existing static markers as the outer observer's boundaries.
        let mut graph_snapshots = Vec::with_capacity(6);
        let mut phases = |phase| {
            if matches!(
                phase,
                "request_preflight" | "last_prompt_input" | "request_private_erasure"
            ) && graph_snapshots.len() < 6
            {
                graph_snapshots.push(serde_json::json!({
                    "phase":phase,
                    "submissions_dispatches_replays":cubecl::wgpu::native_strata_graph_stats(),
                    "capture_shapes_replay_shapes_capture_ns_wait_ns":cubecl::wgpu::native_strata_graph_execution_stats(),
                }));
            }
            eprintln!("native_public_execution phase={phase}");
        };
        let done = if input.retained_check {
            // SAFETY: the SAME protected caller contract remains active across
            // both requests and checked close; no model/runtime is reconstructed.
            unsafe {
                crate::strata_engine::with_loaded_session(
                    &file,
                    graph.bound_graph(),
                    draft_graph.as_ref(),
                    client,
                    limits,
                    host_device.ok_or("native target device missing")?,
                    host_queue.ok_or("native target queue missing")?,
                    input.target_submission_backlog,
                    request,
                    &cancel,
                    &mut phases,
                    |session, first, phases| {
                        let mut deliver = |token| {
                            if emitted.len() >= MAX_OUTPUTS {
                                return Err(kuno_text::Error::InvalidBufferLength);
                            }
                            emitted.push(token);
                            Ok(())
                        };
                        let first = session.run(first, &cancel, &mut deliver, phases)?;
                        if first.outputs != emitted.len() {
                            return Err(crate::strata_engine::Failure::Inference);
                        }
                        // Fixed PUBLIC fixture tokens only. The existing fixture
                        // host allowance covers this bounded MAX_OUTPUTS*4-byte comparison.
                        let expected = Zeroizing::new(emitted.to_vec());
                        zeroize::Zeroize::zeroize(&mut *emitted);
                        let second = session.run(
                            public_request()?,
                            &cancel,
                            &mut |token| {
                                if emitted.len() >= MAX_OUTPUTS {
                                    return Err(kuno_text::Error::InvalidBufferLength);
                                }
                                emitted.push(token);
                                Ok(())
                            },
                            phases,
                        )?;
                        if second.outputs != emitted.len()
                            || *expected != *emitted
                            || first.eos != second.eos
                            || first.greedy != second.greedy
                        {
                            return Err(crate::strata_engine::Failure::Inference);
                        }
                        Ok(second)
                    },
                )
            }
        } else {
            unsafe {
                crate::strata_engine::execute(
                    &file,
                    graph.bound_graph(),
                    draft_graph.as_ref(),
                    client,
                    limits,
                    host_device.ok_or("native target device missing")?,
                    host_queue.ok_or("native target queue missing")?,
                    input.target_submission_backlog,
                    request,
                    &AtomicBool::new(false),
                    &mut |token| {
                        if emitted.len() >= MAX_OUTPUTS {
                            return Err(kuno_text::Error::InvalidBufferLength);
                        }
                        emitted.push(token);
                        Ok(())
                    },
                    &mut |phase| eprintln!("native_public_execution phase={phase}"),
                )
            }
        }?;
        if let Some(sidecar) = &sidecar {
            sidecar.verify_file_unchanged()?;
        }
        if emitted.len() != done.generation.outputs {
            return Err("native public output delivery count".into());
        }
        if cfg!(feature = "strata-public-cap-check")
            && (done.generation.outputs != 16 || done.generation.eos)
        {
            return Err("fixed public output cap check rejected".into());
        }
        let visible = Zeroizing::new(
            emitted
                .iter()
                .copied()
                .filter(|&t| t != graph.eos_token())
                .collect::<Vec<_>>(),
        );
        let preview = Zeroizing::new(tokenizer.decode(&visible)?);
        // Explicitly public diagnostic only. The protected runner displays this
        // line without writing its content into the persistent receipt/log.
        println!(
            "strata_public_preview={}",
            serde_json::to_string(preview.as_str())?
        );
        let backlog_receipt = crate::target_lab_build::native_shared_backlog_record(
            input.target_submission_backlog,
            Some(done.backlog),
            true,
        )?;
        let draft_additional = done
            .draft_public_bytes
            .map(|n| {
                n.checked_sub(crate::strata_shared_weights::addressable_bytes())
                    .ok_or("native shared public quote")
            })
            .transpose()?;
        let record = serde_json::json!({"controller":"global-original-sources",
            "requested_output_limit":if cfg!(feature = "strata-public-cap-check"){16}else{MAX_OUTPUTS},
            "fixed_public_cap_checked":cfg!(feature = "strata-public-cap-check"),
            "greedy_target_verification":done.generation.greedy,"outputs":done.generation.outputs,
            "windows":done.generation.windows,"state_repairs":done.generation.state_repairs,
            "public_response_nonempty":!preview.trim().is_empty(),
            "eos":done.generation.eos,"request_scope_erased":true,"content_persisted":false,
            "loaded_session_requests":if input.retained_check {2}else{1},
            "identical_public_repeat_matched":input.retained_check,
            "native_graph_phase_counters":graph_snapshots,
            "native_operator_profile_requested":crate::strata_port::profile_native_target(),
            "native_operator_profile":cubecl::wgpu::native_strata_operator_profile(),
            "target_close":{"main_buffers":done.target_main.buffers,"main_bytes":done.target_main.bytes,
                "source_buffers":done.target_sources.buffers,"source_bytes":done.target_sources.bytes},
            "speed_acceptance":false,"full_rewrite_accepted":false});
        let value = serde_json::json!({"scope":"strata-global-connected-execution-check",
            "benchmark":false,"whole_rewrite_accepted":false,"record":record,
            "draft_owners_erased":done.draft_public_bytes.is_some(),"target_owners_erased":true,
            "shared_target_public_bytes":crate::strata_shared_weights::public_bytes(),
            "shared_target_host_embedding_bytes":crate::strata_shared_weights::embedding_bytes(),
            "shared_target_addressable_bytes":crate::strata_shared_weights::addressable_bytes(),
            "draft_public_addressable_bytes":done.draft_public_bytes,
            "draft_public_additional_bytes":draft_additional,"target_submission_backlog":backlog_receipt,
            "intel_mtp_admission":intel_mtp_admission});
        serde_json::to_writer_pretty(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(input.receipt)?,
            &value,
        )?;
        println!("strata_global_execution_complete owners_erased=true speed_acceptance=false");
        return Ok(());
    }
    crate::resident_util::select_retained_metadata(true);
    crate::resident_util::select_compute_scratch_clear(true);
    if input.request_reuse {
        if input.target_submission_backlog
            == cubecl_runtime::server::PublicTargetSubmissionBacklog::Default512
        {
            client.configure_public_target_request_io()?;
        } else {
            client
                .configure_public_target_request_io_with_backlog(input.target_submission_backlog)?;
        }
    } else {
        client.configure_transient_io(128 * 1024 * 1024)?;
    }
    let actual_target_backlog = if input.request_reuse {
        Some(client.public_target_submission_backlog()?)
    } else {
        if client.public_target_submission_backlog().is_ok() {
            return Err("ordinary target stream acquired fixed backlog policy".into());
        }
        None
    };
    let shared_iq_tables = crate::strata_intel_core::Tables::new(client, host_device)?;
    let mut draft_target_backlog_query_rejected = None;
    let draft_gpu = if draft_graph.is_some() && !crate::strata_intel_core::enabled() {
        let (draft_client, draft_limits, mtp_device_receipt) = cubecl::future::block_on(
            draft_client(input.fused_sweeps, input.mtp_device, client, host_device),
        )?;
        draft_client.configure_transient_io(64 * 1024 * 1024)?;
        if draft_client.public_target_request_io_configured()? {
            return Err("language target runtime leaked to draft stream".into());
        }
        let rejected = draft_client.public_target_submission_backlog().is_err();
        if !rejected {
            return Err("language draft stream acquired target backlog policy".into());
        }
        draft_target_backlog_query_rejected = Some(rejected);
        Some((draft_client, draft_limits, mtp_device_receipt))
    } else {
        None
    };
    let target_submission_backlog_receipt = if crate::strata_intel_core::enabled() {
        if !input.request_reuse || !client.public_target_request_io_configured()? {
            return Err("shared native target runtime configuration".into());
        }
        crate::target_lab_build::native_shared_backlog_record(
            input.target_submission_backlog,
            actual_target_backlog,
            input.request_reuse,
        )?
    } else {
        crate::target_lab_build::public_target_submission_backlog_record(
            input.target_submission_backlog,
            actual_target_backlog,
            input.request_reuse,
            draft_target_backlog_query_rejected,
        )?
    };
    let target_retired = Arc::new(AtomicBool::new(false));
    let draft_retired = if crate::strata_intel_core::enabled() {
        target_retired.clone()
    } else {
        Arc::new(AtomicBool::new(false))
    };
    let cancel = AtomicBool::new(false);
    let whole = Instant::now();
    let load_disk_before = process_disk_read_bytes()?;
    let mut mtp_device_receipt = draft_gpu.as_ref().map(|(_, _, receipt)| receipt.clone());
    let mut draft = if let Some((draft_client, draft_limits, _)) = draft_gpu {
        let draft_graph = draft_graph
            .as_ref()
            .ok_or("speculative bound draft graph absent")?;
        let mut draft = if input.mtp_english_vocab && !crate::strata_intel_core::enabled() {
            crate::resident_mtp::ResidentMtp::new_public_english_with_fused_sweeps(
                draft_graph,
                draft_client,
                &draft_limits,
                draft_retired.clone(),
            )?
        } else {
            input.draft_profile.resident(
                draft_graph,
                draft_client,
                &draft_limits,
                draft_retired.clone(),
                input.fused_sweeps,
            )?
        };
        if input.mtp_q8_head_broadcast {
            draft.set_q8_head_broadcast(true)?;
        }
        if input.mtp_q8_head_shared_scales {
            draft.set_q8_head_shared_scales(true)?;
        }
        Some(draft)
    } else {
        None
    };
    if !crate::strata_intel_core::enabled()
        && draft.is_none() != (input.decode_mode == DecodeMode::TargetOnly)
    {
        return Err("language actual draft construction scope rejected".into());
    }
    let mut executor = GpuExecutor::new_with_profile(
        &file,
        client.clone(),
        crate::strata_port::static_cache_bytes(),
        crate::daslab_profile::BANK_SLOTS,
        crate::target_lab_build::NATIVE_SHARED_MAIN_BYTES as usize,
        OriginalPackedProfile::DaslabIq3,
    )?;
    if crate::strata_intel_core::enabled() && draft_graph.is_some() {
        if input.mtp_device != crate::target_lab_build::MtpDevice::Intel
            || input.draft_profile != Profile::StrataQ2
            || !input.fused_sweeps
        {
            return Err("native shared draft requires Intel/original Q2/fused scope".into());
        }
        executor.reserve_native_mtp(resident_mtp::ResidentMtp::shared_capacity()?)?;
    }
    let mut shared_draft_close: Option<kuno_text::Result<resident_mtp::CloseReceipt>> = None;
    executor.target_rows_reserved = true;
    executor.target_batch_router_reserved = input.frozen_host_router;
    executor.target_frozen_host_router_reserved = input.frozen_host_router;
    executor.target_frozen_host_staging_reserved = input.frozen_packed_host_staging;
    executor.target_frozen_gu_integer_reserved = input.frozen_gu_integer;
    executor.target_frozen_down_integer_reserved = input.frozen_down_integer;
    executor.target_verify3_dense = input.verify3_dense;
    executor.target_verify3_fused = input.verify3_fused;
    executor.token_io_checkpoint = input.chunk_io;
    if input.four_lookahead || crate::strata_intel_core::enabled() {
        executor.reserve_public_four_rows()?;
    }
    executor.enable_transient_resident(target_retired.clone())?;
    executor.fast_dense = true;
    executor.fast_dense_fault = true;
    executor.shared_kernels = true;
    executor.shared_gdn = true;
    let attempt = (|| -> Result<serde_json::Value> {
        let mut config = Config::new(client, limits, 8)?
            .reserve_target_rows()?
            .reserve_latest_target_checkpoint()?;
        if input.gpu_qsa_selector {
            config = config.reserve_public_gpu_qsa_selector()?;
        }
        if input.frozen_host_sources {
            config = config.reserve_public_frozen_host_sources()?;
        }
        if input.prepared_experts {
            config = config.reserve_prepared_verify3_experts()?;
        }
        if input.live_member_reduction {
            config = config.with_live_member_reduction()?;
        }
        if input.public_existing_fill {
            config = config.with_public_existing_expert_fill()?;
        }
        if input.request_reuse {
            config = config.reserve_public_language_request_io(prompt.len(), MAX_OUTPUTS)?;
        }
        if input.frozen_host_router {
            config = config.reserve_public_frozen_host_batch_router()?;
        }
        if input.frozen_packed_host_staging {
            config = config.reserve_public_frozen_host_staging()?;
        }
        if input.frozen_gu_integer {
            config = config.reserve_public_frozen_gu_integer()?;
        }
        if input.frozen_down_integer {
            config = config.reserve_public_frozen_down_integer()?;
        }
        if input.public_grouped_ple_reads || crate::strata_intel_core::enabled() {
            config = config.reserve_public_grouped_ple_reads()?;
        }
        if input.public_grouped_status_folds || crate::strata_intel_core::enabled() {
            config = config.reserve_public_grouped_status_folds()?;
        }
        if input.four_lookahead || crate::strata_intel_core::enabled() {
            config = config.reserve_public_four_rows()?;
        }
        let prepared_host = std::cell::OnceCell::new();
        let mut host_attachment = None;
        let mut host_setup = None;
        let mut bank_fill_setup_experts = None;
        if crate::strata_intel_core::enabled() {
            eprintln!("native_public_construct phase=target");
        }
        let mut target = ResidentModel::new_with_native_batch(
            graph.bound_graph(),
            client.clone(),
            input.fixture.context(),
            &mut executor,
            config,
            true,
        )?;
        if crate::strata_intel_core::enabled() {
            eprintln!("native_public_construct phase=global_sources");
            // SAFETY: this protected original-artifact probe retains the full
            // parent hash evidence/RAM envelope and exact target device.
            unsafe {
                target.prepare_global_sources_before_prefill(
                    &mut executor,
                    host_device.ok_or("global target device missing")?,
                    host_queue.ok_or("global target queue missing")?,
                    0, // compiled4096-row fixture: owned prompt, no loan HOST grant
                )?;
            }
        }
        if crate::strata_intel_core::enabled() {
            eprintln!("native_public_construct phase=shared_target_weights");
            executor.prepare_native_shared_target_weights(graph.bound_graph())?;
        }
        if crate::strata_intel_core::enabled()
            && let Some(g) = &draft_graph
        {
            eprintln!("native_public_construct phase=shared_draft");
            let device = host_device.ok_or("shared native target device missing")?;
            if device.adapter_info().vendor != 0x8086
                || g.target_binding_id() != Some(graph.bound_graph().binding_id())
            {
                return Err("shared native target/draft identity rejected".into());
            }
            let admission = executor.shared_mtp_admission()?;
            draft = Some(resident_mtp::ResidentMtp::new_shared_strata_q2(
                g,
                client.clone(),
                limits,
                admission,
            )?);
            mtp_device_receipt = Some(
                serde_json::json!({"device":"intel","shared_target_runtime":true,
                "distinct_vulkan_device":false,"independent_runtime":false,"main_account":"shared"}),
            );
        }
        let runs = (|| -> Result<serde_json::Value> {
            let load_ns = elapsed(whole)?;
            let prefill = Instant::now();
            let before_prefill_repair = draft.as_ref().map(|draft| draft.traffic());
            let mut carry = Zeroizing::new(vec![0f32; HIDDEN]);
            let mut confirmed = 0u32;
            // Teacher cells contain the real next prompt token and actual
            // current target hidden state. The final prompt carry waits for the
            // first generated token; no dummy initial cell is inserted.
            for start in (0..prompt.len()).step_by(3) {
                let count = (prompt.len() - start).min(3);
                let rows = target.prefill_chunk_with_carry(
                    &mut executor,
                    &prompt[start..start + count],
                    &cancel,
                )?;
                if rows.start() != start
                    || rows.count() != count
                    || rows.binding_id() != graph.bound_graph().binding_id()
                {
                    return Err("language prompt output identity rejected".into());
                }
                for row in 0..count {
                    let position = start + row;
                    if rows.position(row)? != position {
                        return Err("language prompt consumed position rejected".into());
                    }
                    if let Some(draft) = draft.as_mut() {
                        if let Some(token) = teacher_input(&prompt, position, draft.position())? {
                            draft.append_kv(token, rows.hidden(row)?, &cancel)?;
                        } else {
                            confirmed = rows.greedy(row)?;
                            carry.copy_from_slice(rows.hidden(row)?);
                        }
                    } else {
                        // Original prompt chunks still execute every target
                        // family. Read full real packets without teacher KV.
                        let predicted = rows.greedy(row)?;
                        let hidden = rows.hidden(row)?;
                        if position + 1 == prompt.len() {
                            confirmed = predicted;
                            carry.copy_from_slice(hidden);
                        }
                    }
                }
                drop(rows);
                if target.completed_target_rows_cursor(&executor)? != start + count {
                    return Err("language prompt actual cursor rejected".into());
                }
            }
            if target.completed_target_rows_cursor(&executor)? != prompt.len()
                || draft
                    .as_ref()
                    .is_some_and(|draft| draft.position() != prompt.len() - 1)
            {
                return Err("language initial target/draft alignment rejected".into());
            }
            if target.native_live_member_reduction(&executor)? != input.live_member_reduction {
                return Err("language frozen live reduction rejected".into());
            }
            let (
                initial_prefill_frozen_down_integer,
                initial_down_integer_preparations,
                initial_down_integer_dispatches,
                initial_down_integer_audits,
                initial_down_integer_sources,
            ) = if executor.target_frozen_host_source_identity.is_some() {
                let counts = target.native_frozen_source_counters(&executor)?;
                (
                    target.native_frozen_source_down_integer(&executor)?,
                    counts.down_integer_preparations,
                    counts.down_integer_dispatches,
                    counts.down_integer_audits,
                    counts.source_dispatches,
                )
            } else {
                (false, 0, 0, 0, 0)
            };
            let initial_prefill_frozen_down_integer_counters =
                crate::target_lab_build::frozen_down_integer_initial_record(
                    false,
                    initial_prefill_frozen_down_integer,
                    initial_down_integer_preparations,
                    initial_down_integer_dispatches,
                    initial_down_integer_audits,
                    initial_down_integer_sources,
                )?;
            let (initial_iq4_decoder, initial_iq4_dispatches) =
                executor.public_target_iq4_decoder()?;
            let initial_prefill_target_iq4_decoder_receipt =
                crate::target_lab_build::public_iq4_decoder_record(
                    false,
                    initial_iq4_decoder,
                    initial_iq4_dispatches,
                    executor.target_verify3_dense_calls,
                )?;
            let initial_bank_fill = target.native_frozen_source_bank_fill(&executor)?;
            let initial_prefill_bank_fill_receipt =
                crate::target_lab_build::public_host_bank_fill_record(
                    crate::executor::PublicHostBankFillPolicy::Original,
                    initial_bank_fill,
                    None,
                )?;
            let prefill_ns = elapsed(prefill)?;
            if input.frozen_host_sources && !crate::strata_intel_core::enabled() {
                println!("public_frozen_host_prepare_start");
                let preparation = Instant::now();
                let (plan, fills) = if input.bank_fill
                    == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
                {
                    target.prepare_public_host_sources_with_fill_policy(
                        &mut executor,
                        &cancel,
                        false,
                        input.bank_fill,
                    )?
                } else {
                    target.prepare_public_host_sources(&mut executor, &cancel, false)?
                };
                if input.bank_fill == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
                {
                    bank_fill_setup_experts = Some(fills.experts);
                }
                println!("public_frozen_host_prepare_complete");
                prepared_host
                    .set(plan)
                    .map_err(|_| "public source plan already initialized")?;
                let prepared = prepared_host.get().ok_or("public host plan absent")?;
                let reserved = prepared.budget().total_host_reserved_bytes;
                println!("public_frozen_host_attach_start");
                let attach_start = Instant::now();
                // SAFETY: protected runner verified full immutable public
                // source hashes before opening the original model descriptors.
                host_attachment = Some(unsafe {
                    if input.frozen_down_integer {
                        target.attach_public_host_sources_with_down_math(
                            &mut executor,
                            prepared,
                            host_device.ok_or("public exact device absent")?,
                            reserved,
                            input.frozen_partition_exit,
                            input.frozen_private_tables,
                            input.frozen_workgroup_tables,
                            input.frozen_shared_down_scales,
                            input.frozen_shared_gate_up_factors,
                            input.frozen_four_row_down,
                            input.frozen_two_row_gate_up,
                            input.frozen_packed_host_staging,
                            input.frozen_three_members,
                            input.frozen_gu_integer,
                            input.frozen_gu_stride_four,
                            input.frozen_down_stride_four,
                            input.frozen_down_integer,
                            &cancel,
                        )?
                    } else if input.frozen_gu_stride_four || input.frozen_down_stride_four {
                        target.attach_public_host_sources_with_schedules(
                            &mut executor,
                            prepared,
                            host_device.ok_or("public exact device absent")?,
                            reserved,
                            input.frozen_partition_exit,
                            input.frozen_private_tables,
                            input.frozen_workgroup_tables,
                            input.frozen_shared_down_scales,
                            input.frozen_shared_gate_up_factors,
                            input.frozen_four_row_down,
                            input.frozen_two_row_gate_up,
                            input.frozen_packed_host_staging,
                            input.frozen_three_members,
                            input.frozen_gu_integer,
                            input.frozen_gu_stride_four,
                            input.frozen_down_stride_four,
                            &cancel,
                        )?
                    } else if input.frozen_gu_integer {
                        target.attach_public_host_sources_with_gu_math(
                            &mut executor,
                            prepared,
                            host_device.ok_or("public exact device absent")?,
                            reserved,
                            input.frozen_partition_exit,
                            input.frozen_private_tables,
                            input.frozen_workgroup_tables,
                            input.frozen_shared_down_scales,
                            input.frozen_shared_gate_up_factors,
                            input.frozen_four_row_down,
                            input.frozen_two_row_gate_up,
                            input.frozen_packed_host_staging,
                            input.frozen_three_members,
                            input.frozen_gu_integer,
                            &cancel,
                        )?
                    } else {
                        target.attach_public_host_sources(
                            &mut executor,
                            prepared,
                            host_device.ok_or("public exact device absent")?,
                            reserved,
                            input.frozen_partition_exit,
                            input.frozen_private_tables,
                            input.frozen_workgroup_tables,
                            input.frozen_shared_down_scales,
                            input.frozen_shared_gate_up_factors,
                            input.frozen_four_row_down,
                            input.frozen_two_row_gate_up,
                            input.frozen_packed_host_staging,
                            input.frozen_three_members,
                            &cancel,
                        )?
                    }
                });
                host_setup = Some(serde_json::json!({"bank_completion_experts":fills.experts,
                    "bank_completion_uploaded_bytes":fills.bytes,"bank_completion_read_ns":fills.read_ns,
                    "bank_completion_upload_ns":fills.upload_ns,
                    "prepare_and_attach_ns":preparation.elapsed().as_nanos(),
                    "attach_ns":attach_start.elapsed().as_nanos(),
                    "separate_host_reserved_bytes":reserved}));
                if input.frozen_host_router {
                    target.activate_public_frozen_host_batch_router(&executor, &cancel)?;
                    println!("public_frozen_host_gpu_router_active");
                }
                println!("public_frozen_host_attach_complete");
            }
            let attached_bank_fill = target.native_frozen_source_bank_fill(&executor)?;
            let attached_bank_fill_receipt = crate::target_lab_build::public_host_bank_fill_record(
                input.bank_fill,
                attached_bank_fill,
                bank_fill_setup_experts,
            )?;
            if input.public_iq4_decoder {
                executor.activate_public_target_iq4_decoder()?;
            }
            let (initial_prefill_target_grouped_status_folds, initial_status_folds) =
                target.completed_public_grouped_status_folds(&executor)?;
            let initial_prefill_target_grouped_status_folds_receipt =
                crate::target_lab_build::public_grouped_status_folds_record(
                    false,
                    initial_prefill_target_grouped_status_folds,
                    initial_status_folds,
                    0,
                    0,
                )?;
            let (initial_four_rows, initial_count4_chunks) =
                target.completed_public_four_rows(&executor)?;
            let initial_four_lookahead_receipt =
                crate::target_lab_build::LookaheadStatistics::default().record(
                    false,
                    initial_four_rows,
                    initial_count4_chunks,
                    executor.target_verify4_dense_calls,
                    executor.target_verify4_fused_calls,
                )?;
            let initial_prefill_mtp_batch_repair =
                crate::target_lab_build::public_mtp_batch_repair_traffic_record(
                    false,
                    before_prefill_repair.unwrap_or_default(),
                    draft
                        .as_ref()
                        .map(|draft| draft.traffic())
                        .unwrap_or_default(),
                    0,
                    0,
                )?;
            let initial_prefill_mtp_fused_four_repair =
                crate::target_lab_build::FourMtpRepairStatistics::default().record(
                    false,
                    initial_four_rows,
                    &crate::target_lab_build::LookaheadStatistics::default(),
                    before_prefill_repair.unwrap_or_default(),
                    draft
                        .as_ref()
                        .map(|draft| draft.traffic())
                        .unwrap_or_default(),
                )?;
            let (
                initial_integer_head,
                initial_head_prepares,
                initial_head_tiles,
                initial_head_members,
            ) = target.completed_public_q6_integer_verify_head(&executor)?;
            let initial_prefill_target_integer_head_receipt =
                crate::target_lab_build::IntegerVerifyHeadStatistics::default().record(
                    false,
                    initial_integer_head,
                    initial_head_prepares,
                    initial_head_tiles,
                    initial_head_members,
                )?;
            let selected_four_lookahead = input.four_lookahead;
            if selected_four_lookahead {
                target.set_public_four_rows(&mut executor, &cancel)?;
            }
            if input.three_gdn_streaming {
                target.set_public_three_gdn_streaming(&mut executor, &cancel)?;
            }
            if input.public_grouped_ple_reads {
                target.set_public_grouped_ple_reads(&executor, true, &cancel)?;
            }
            if input.public_grouped_status_folds {
                target.set_public_grouped_status_folds(&executor, true, &cancel)?;
            }
            let selected_integer_head = input.public_integer_verify_head;
            if selected_integer_head {
                target.activate_public_q6_integer_verify_head(&mut executor, &cancel)?;
            }
            let mut integer_head_stats =
                crate::target_lab_build::IntegerVerifyHeadStatistics::default();
            let load_disk_after = process_disk_read_bytes()?;
            let before_uploads = executor.expert_uploads;
            let before_hits = executor.expert_hits;
            let before_bytes = executor.uploaded_bytes;
            let before_weight_reads = executor.weight_read_ns;
            let before_weight_uploads = executor.weight_upload_ns;
            let before_gpu_routers = executor.gpu_router_calls;
            let mut lookahead_stats = crate::target_lab_build::LookaheadStatistics::default();
            let before_four_dense = executor.target_verify4_dense_calls;
            let before_four_fused = executor.target_verify4_fused_calls;
            let before_dense = executor.target_verify3_dense_calls;
            let before_fused = executor.target_verify3_fused_calls;
            let before_phases = target.phase_ns.clone();
            let before_traffic = draft.as_ref().map(|draft| draft.traffic());
            let mut mtp_batch_repair_cells = 0u64;
            let selected_four_batch_repair = selected_four_lookahead && input.mtp_batch_repair;
            let mut four_batch_repair = crate::target_lab_build::FourMtpRepairStatistics::default();
            let before_host = crate::target_lab_build::host_io_snapshot();
            let (actual_direct, before_fills) = target.native_public_fill_counters(&executor)?;
            if actual_direct != input.public_existing_fill {
                return Err("language frozen direct fill rejected".into());
            }
            let start = Instant::now();
            if input.request_reuse {
                target.begin_public_language_request_io(&mut executor, &cancel)?;
                if !target.public_language_request_io_active()? {
                    return Err("language actual request scope rejected".into());
                }
            }
            let mut emitted = Zeroizing::new(Vec::with_capacity(MAX_OUTPUTS));
            emitted.push(confirmed);
            let mut decoder = tokenizer.stream_decoder();
            let mut nonempty_text = false;
            let mut public_text = Zeroizing::new(String::new());
            if confirmed != graph.eos_token() {
                let text = Zeroizing::new(decoder.push_bytes(tokenizer.token_bytes(confirmed)?)?);
                nonempty_text |= !text.is_empty();
                if crate::strata_port::enabled() {
                    public_text.push_str(&text);
                }
            }
            let mut stats = Statistics::default();
            if crate::strata_port::enabled() {
                println!("strata_decode_start");
            }
            let mut draft_policy = crate::strata_port::DraftPolicy::default();
            let mut fixture_peak = owned_fixture_bytes(&[&carry], &[&prompt, &emitted])?;
            while emitted.len() < MAX_OUTPUTS && confirmed != graph.eos_token() {
                let position = target.completed_target_rows_cursor(&executor)?;
                if position != prompt.len() + stats.continuations as usize
                    || draft
                        .as_ref()
                        .is_some_and(|draft| draft.position() != position - 1)
                {
                    return Err("language real target/draft cursor rejected".into());
                }
                if input.decode_mode == DecodeMode::TargetOnly {
                    let verifying = Instant::now();
                    let rows =
                        target.prefill_chunk_with_carry(&mut executor, &[confirmed], &cancel)?;
                    if rows.start() != position
                        || rows.count() != 1
                        || rows.position(0)? != position
                        || rows.binding_id() != graph.bound_graph().binding_id()
                    {
                        return Err("target-only actual count1 packet identity rejected".into());
                    }
                    confirmed = rows.greedy(0)?;
                    carry.copy_from_slice(rows.hidden(0)?);
                    drop(rows);
                    if target.completed_target_rows_cursor(&executor)? != position + 1 {
                        return Err("target-only actual count1 cursor rejected".into());
                    }
                    emitted.push(confirmed);
                    if confirmed != graph.eos_token() {
                        let text =
                            Zeroizing::new(decoder.push_bytes(tokenizer.token_bytes(confirmed)?)?);
                        nonempty_text |= !text.is_empty();
                        if crate::strata_port::enabled() {
                            public_text.push_str(&text);
                        }
                    }
                    stats.add(1, 1, 0, elapsed(verifying)?, 0, false)?;
                    continue;
                }
                let draft = draft.as_mut().ok_or("speculative actual draft absent")?;
                let mut count = draft_policy.choose_mtp((MAX_OUTPUTS - emitted.len()).min(
                    if cfg!(feature = "strata-eight-window")
                        && std::env::var("KUNO_STRATA_MTP_WINDOW").as_deref() == Ok("eight-public")
                    {
                        8
                    } else if selected_four_lookahead {
                        4
                    } else {
                        3
                    },
                ));
                let mut inputs = Zeroizing::new(Vec::with_capacity(count));
                inputs.push(confirmed);
                let drafting = Instant::now();
                let candidate = crate::strata_port::lookup(
                    &prompt,
                    &emitted,
                    (MAX_OUTPUTS - emitted.len()).min(crate::strata_port::verify_capacity()) - 1,
                );
                let mut lookup = None;
                let mut third_follow_ns = 0;
                let queued_window =
                    crate::strata_port::enabled() && candidate.is_none() && count > 1;
                if queued_window {
                    let output = draft.draft_window(
                        confirmed,
                        &carry,
                        (count - 1).min(crate::strata_port::verify_capacity() - 1),
                        &cancel,
                    )?;
                    if output.start != position - 1
                        || output.binding != draft.binding()
                        || output.tokens().len()
                            != (count - 1).min(crate::strata_port::verify_capacity() - 1)
                        || draft.position()
                            != position + (count - 1).min(crate::strata_port::verify_capacity() - 1)
                                - 1
                    {
                        return Err("language completed queued draft identity rejected".into());
                    }
                    inputs.extend_from_slice(output.tokens());
                    if count == 4 {
                        third_follow_ns = output.last_enqueue_ns;
                    }
                } else if count > 1 {
                    let output = draft.step_draft(confirmed, &carry, &cancel)?;
                    if output.start() != position - 1 {
                        return Err("language first actual draft stamp rejected".into());
                    }
                    inputs.push(output.greedy());
                    drop(output);
                    if let Some(candidate) = candidate.filter(|p| p.tokens.first() == inputs.get(1))
                        && let Some(selected) = draft_policy.choose(count, &candidate)
                    {
                        count = selected;
                        crate::strata_port::credit_lookup(count - 1);
                        for &token in candidate.tokens.iter().skip(1).take(count - 2) {
                            // First cell is the real MTP forward; provisional
                            // follow cells are repaired from real target carries.
                            draft.append_kv(*inputs.last().unwrap(), &carry, &cancel)?;
                            inputs.push(token);
                        }
                        lookup = Some(candidate);
                    }
                    if count > 2 && lookup.is_none() {
                        let output = draft.follow_draft(&cancel)?;
                        if output.start() != position {
                            return Err("language second actual draft stamp rejected".into());
                        }
                        inputs.push(output.greedy());
                        drop(output);
                    }
                }
                if count == 4 && lookup.is_none() && !queued_window {
                    let third_follow = Instant::now();
                    let output = draft.follow_draft(&cancel)?;
                    crate::target_lab_build::public_third_draft_stamp(
                        position,
                        output.start(),
                        draft.position(),
                        draft.binding(),
                        output.binding(),
                    )?;
                    inputs.push(output.greedy());
                    drop(output);
                    third_follow_ns = u64::try_from(third_follow.elapsed().as_nanos())?;
                }
                while inputs.len() < count {
                    let follow = Instant::now();
                    let output = draft.follow_draft(&cancel)?;
                    if output.start() != position + inputs.len() - 2
                        || output.binding() != draft.binding()
                    {
                        return Err("actual extended MTP stamp rejected".into());
                    }
                    inputs.push(output.greedy());
                    third_follow_ns = u64::try_from(follow.elapsed().as_nanos())?;
                }
                let draft_ns = elapsed(drafting)?;
                let verifying = Instant::now();
                let mut round =
                    LatestTargetRound::begin(&mut target, &mut executor, &inputs, &cancel)?;
                let profile_this_round = crate::strata_port::profile_target() && stats.rounds == 8;
                let native_profile_this_round =
                    crate::strata_port::profile_native_target() && stats.rounds == 8;
                if native_profile_this_round {
                    cubecl::future::block_on(client.sync())?;
                    cubecl::wgpu::set_native_strata_profile_active(true);
                }
                if profile_this_round {
                    cubecl::future::block_on(client.sync())?;
                    client.set_profiling_paused(false);
                }
                let verified = round.verify(&cancel);
                if native_profile_this_round {
                    cubecl::wgpu::set_native_strata_profile_active(false);
                }
                if profile_this_round {
                    client.set_profiling_paused(true);
                    cubecl::future::block_on(client.sync())?;
                }
                let rows = verified?;
                if rows.start() != position
                    || rows.count() != count
                    || rows.binding_id() != graph.bound_graph().binding_id()
                {
                    return Err("language actual verification identity rejected".into());
                }
                let mut predicted = Zeroizing::new(Vec::with_capacity(count));
                for row in 0..count {
                    if rows.position(row)? != position + row {
                        return Err("language consumed target position rejected".into());
                    }
                    predicted.push(rows.greedy(row)?);
                }
                let accepted = if count == 4 {
                    crate::target_lab_build::public_four_accepted_prefix(
                        &inputs,
                        &predicted,
                        VOCAB as u32,
                        Some(graph.eos_token()),
                    )?
                } else {
                    accepted_prefix(&inputs, &predicted, graph.eos_token())?
                };
                let mut actual_carries = Zeroizing::new(Vec::with_capacity(accepted * HIDDEN));
                let mut saved_logits = Zeroizing::new(Vec::with_capacity(
                    if accepted < count && !crate::strata_port::enabled() {
                        accepted * VOCAB
                    } else {
                        0
                    },
                ));
                for row in 0..accepted {
                    actual_carries.extend_from_slice(rows.hidden(row)?);
                    if accepted < count && !crate::strata_port::enabled() {
                        saved_logits.extend_from_slice(rows.logits(row)?);
                    }
                }
                fixture_peak = fixture_peak.max(owned_fixture_bytes(
                    &[&carry, &actual_carries, &saved_logits],
                    &[&prompt, &emitted, &inputs, &predicted],
                )?);
                drop(rows);
                let verify_ns = elapsed(verifying)?;
                let repairing = Instant::now();
                let repaired = round.finish(accepted, &cancel)?;
                let recomputed = repaired.is_some();
                if recomputed != (accepted < count && !crate::strata_port::enabled()) {
                    return Err("language actual rejection repair evidence missing".into());
                }
                if crate::strata_port::enabled() && accepted < count {
                    stats.state_replay_repairs += 1;
                }
                if let Some(recomputed) = repaired {
                    if recomputed.start() != position
                        || recomputed.count() != accepted
                        || recomputed.binding_id() != graph.bound_graph().binding_id()
                    {
                        return Err("language recomputed actual identity rejected".into());
                    }
                    for row in 0..accepted {
                        if !crate::strata_port::enabled() {
                            stats.repair_logit_max = stats.repair_logit_max.max(repair_vector(
                                &saved_logits[row * VOCAB..(row + 1) * VOCAB],
                                recomputed.logits(row)?,
                                VOCAB,
                            )?);
                        }
                        stats.repair_hidden_max = stats.repair_hidden_max.max(repair_vector(
                            &actual_carries[row * HIDDEN..(row + 1) * HIDDEN],
                            recomputed.hidden(row)?,
                            HIDDEN,
                        )?);
                        if recomputed.greedy(row)? != predicted[row] {
                            return Err("language repair greedy changed".into());
                        }
                    }
                    drop(recomputed);
                }
                if target.completed_target_rows_cursor(&executor)? != position + accepted {
                    return Err("language repaired actual cursor rejected".into());
                }
                if input.mtp_batch_repair {
                    let repair_before = draft.traffic();
                    if selected_four_batch_repair && count == 4 {
                        draft.repair_accepted_cache_fused_four(
                            position,
                            &inputs[..accepted],
                            &carry,
                            &actual_carries,
                            &cancel,
                        )?;
                    } else {
                        draft.repair_accepted_cache(
                            position,
                            &inputs[..accepted],
                            &carry,
                            &actual_carries,
                            &cancel,
                        )?;
                    }
                    if selected_four_batch_repair {
                        four_batch_repair.observe(
                            count,
                            accepted,
                            repair_before,
                            draft.traffic(),
                        )?;
                    } else {
                        mtp_batch_repair_cells = mtp_batch_repair_cells
                            .checked_add(if count > 1 { accepted - 1 } else { 1 } as u64)
                            .ok_or("language MTP batch repair appended cell count overflow")?;
                    }
                } else if count > 1 {
                    // Retain first real cell; replace only provisional cells
                    // with actual accepted target carry/token conditioning.
                    draft.rewind(position)?;
                    for row in 1..accepted {
                        draft.append_kv(
                            inputs[row],
                            &actual_carries[(row - 1) * HIDDEN..row * HIDDEN],
                            &cancel,
                        )?;
                    }
                } else {
                    draft.append_kv(confirmed, &carry, &cancel)?;
                }
                if draft.position() != position + accepted - 1 {
                    return Err("language repaired draft cell rejected".into());
                }
                carry.copy_from_slice(&actual_carries[(accepted - 1) * HIDDEN..accepted * HIDDEN]);
                confirmed = predicted[accepted - 1];
                for &token in &predicted[..accepted] {
                    emitted.push(token);
                    if token != graph.eos_token() {
                        let text =
                            Zeroizing::new(decoder.push_bytes(tokenizer.token_bytes(token)?)?);
                        nonempty_text |= !text.is_empty();
                        if crate::strata_port::enabled() {
                            public_text.push_str(&text);
                        }
                    }
                }
                let repair_ns = elapsed(repairing)?;
                if selected_integer_head {
                    integer_head_stats.add(count, accepted, recomputed)?;
                }
                let phase = crate::target_lab_build::LookaheadPhase {
                    draft_ns,
                    third_follow_ns,
                    verify_ns,
                    repair_ns,
                };
                if !crate::strata_port::enabled() {
                    lookahead_stats.add(selected_four_lookahead, count, accepted, phase)?;
                }
                stats.add_completed(selected_four_lookahead, count, accepted, phase, recomputed)?;
                if crate::strata_port::enabled() && stats.rounds % 16 == 0 {
                    println!(
                        "strata_public_progress emitted={} continuations={} elapsed_ns={} repairs={}",
                        emitted.len(),
                        stats.continuations,
                        elapsed(start)?,
                        stats.repairs
                    );
                }
                if crate::strata_port::enabled() {
                    draft_policy.observe(
                        lookup.is_some(),
                        count,
                        accepted,
                        lookup.as_ref().map_or(0, |p| p.matched),
                        draft_ns + verify_ns + repair_ns,
                    );
                }
                if crate::strata_port::enabled()
                    && emitted.len() < MAX_OUTPUTS
                    && confirmed != graph.eos_token()
                {
                    if stats.rounds % 4 == 0 {
                        println!("strata_adapt_start round={}", stats.rounds);
                    }
                    target.adapt_strata_cache(
                        &mut executor,
                        host_attachment
                            .as_mut()
                            .ok_or("Strata adaptive attachment absent")?,
                        &cancel,
                        host_queue.ok_or("Strata original target queue absent")?,
                    )?;
                    if stats.rounds % 4 == 0 {
                        println!("strata_adapt_complete round={}", stats.rounds);
                    }
                }
            }
            let tail = Zeroizing::new(decoder.finish());
            nonempty_text |= !tail.is_empty();
            if crate::strata_port::enabled() {
                public_text.push_str(&tail);
            }
            if emitted.len() != stats.continuations as usize + 1
                || emitted.len() > MAX_OUTPUTS
                || target.completed_target_rows_cursor(&executor)?
                    != prompt.len() + emitted.len() - 1
            {
                return Err("language actual emitted/continuation accounting rejected".into());
            }
            let request_close = Instant::now();
            let mut request_scope = None;
            if input.request_reuse {
                let r = target.end_public_language_request_io(&mut executor, &cancel)?;
                if !r.full_io_audit
                    || !r.scope_closed
                    || r.scope != 1
                    || target.public_language_request_io_active()?
                {
                    return Err("language complete request closing audit rejected".into());
                }
                request_scope = Some(
                    serde_json::json!({"scope_sequence":r.scope,"full_io_audit":r.full_io_audit,"scope_closed":r.scope_closed,"closing_audit_ns":r.closing_audit_ns}),
                );
            }
            let request_close_ns = if input.request_reuse {
                elapsed(request_close)?
            } else {
                0
            };
            let elapsed_ns = elapsed(start)?;
            let (actual_integer_head, head_prepares, head_tiles, head_members) =
                target.completed_public_q6_integer_verify_head(&executor)?;
            let target_integer_head_receipt = integer_head_stats.record(
                selected_integer_head,
                actual_integer_head,
                head_prepares,
                head_tiles,
                head_members,
            )?;
            let (actual_four_rows, completed_count4_chunks) =
                target.completed_public_four_rows(&executor)?;
            let four_lookahead_receipt = if crate::strata_port::enabled() {
                serde_json::json!({"selected":selected_four_lookahead,"actual_count4_chunks":completed_count4_chunks,"actual_four_rows":actual_four_rows,"policy":"strata-suffix-or-mtp-target-verified", "capacity":crate::strata_port::verify_capacity(), "completed_windows":stats.completed_windows, "accepted_by_window":stats.accepted_by_window})
            } else {
                lookahead_stats.record_with_mtp_repair(
                    selected_four_lookahead,
                    actual_four_rows,
                    completed_count4_chunks,
                    executor
                        .target_verify4_dense_calls
                        .checked_sub(before_four_dense)
                        .ok_or("Verify4 dense counter regressed")?,
                    executor
                        .target_verify4_fused_calls
                        .checked_sub(before_four_fused)
                        .ok_or("Verify4 fused counter regressed")?,
                    selected_four_batch_repair,
                )?
            };
            let host_io = crate::target_lab_build::host_io_delta(
                before_host,
                crate::target_lab_build::host_io_snapshot(),
            )?;
            if !selected_four_lookahead
                && stats.continuations > 0
                && (input.verify3_dense && executor.target_verify3_dense_calls == before_dense
                    || input.verify3_fused && executor.target_verify3_fused_calls == before_fused)
            {
                return Err("language actual selected projector evidence missing".into());
            }
            let phase_ns: BTreeMap<_, _> = target
                .phase_ns
                .iter()
                .map(|(n, v)| (*n, v - before_phases.get(n).copied().unwrap_or(0)))
                .collect();
            let after_traffic = draft.as_ref().map(|draft| draft.traffic());
            let mtp_fused_four_repair_receipt = four_batch_repair.record(
                selected_four_batch_repair,
                actual_four_rows,
                &lookahead_stats,
                before_traffic.unwrap_or_default(),
                after_traffic.unwrap_or_default(),
            )?;
            let mtp_batch_repair_receipt = if selected_four_batch_repair {
                mtp_fused_four_repair_receipt["original_batched_tails"].clone()
            } else {
                crate::target_lab_build::public_mtp_batch_repair_traffic_record(
                    input.mtp_batch_repair,
                    before_traffic.unwrap_or_default(),
                    after_traffic.unwrap_or_default(),
                    if input.mtp_batch_repair {
                        stats.rounds
                    } else {
                        0
                    },
                    mtp_batch_repair_cells,
                )?
            };
            let draft_traffic = match (before_traffic, after_traffic) {
                (Some(before), Some(after)) => Some(serde_json::json!({
                    "host_read_calls":after.host_read_calls-before.host_read_calls,
                    "host_read_bytes":after.host_read_bytes-before.host_read_bytes,
                    "explicit_dispatches":after.explicit_dispatches-before.explicit_dispatches,
                    "private_upload_calls":after.private_upload_calls-before.private_upload_calls,
                    "private_upload_bytes":after.private_upload_bytes-before.private_upload_bytes,
                    "minimal_packet_reads":after.minimal_packet_reads-before.minimal_packet_reads,
                    "followup_copy_calls":after.followup_copy_calls-before.followup_copy_calls,
                    "draft_window_reads":after.draft_window_reads-before.draft_window_reads,
                    "batched_repair_calls":if selected_four_batch_repair {
                        &mtp_fused_four_repair_receipt["aggregate_completed_calls"]
                    } else {&mtp_batch_repair_receipt["completed_calls"]},
                    "batched_repair_cells":if selected_four_batch_repair {
                        &mtp_fused_four_repair_receipt["aggregate_appended_cells"]
                    } else {&mtp_batch_repair_receipt["appended_cells"]},
                    "batched_repair_fault_gates":if selected_four_batch_repair {
                        &mtp_fused_four_repair_receipt["aggregate_completed_fault_gates"]
                    } else {&mtp_batch_repair_receipt["completed_fault_gates"]}})),
                (None, None) if input.decode_mode == DecodeMode::TargetOnly => None,
                _ => return Err("language actual draft traffic scope changed".into()),
            };
            let target_only_counters =
                stats.target_only_record(input.decode_mode, draft.is_some(), emitted.len())?;
            let (actual_direct, after_fills) = target.native_public_fill_counters(&executor)?;
            let frozen_routes = if executor.target_frozen_host_source_identity.is_some() {
                let counts = target.native_frozen_source_counters(&executor)?;
                Some(serde_json::json!({"vram_routes":counts.vram_routes,
                    "host_routes":counts.host_routes,"source_dispatches":counts.source_dispatches,"staging_dispatches":counts.staging_dispatches,"staged_unique_host_groups":counts.staged_unique_host_groups,"staged_packed_bytes":counts.staged_packed_bytes,"gu_integer_preparations":counts.gu_integer_preparations,"gu_integer_dispatches":counts.gu_integer_dispatches,"gu_integer_audits":counts.gu_integer_audits,"gu_stride_four_dispatches":counts.gu_stride_four_dispatches,"compact_gu_dispatches":counts.compact_gu_dispatches,"compact_down_dispatches":counts.compact_down_dispatches,"compact_audits":counts.compact_audits,"down_stride_four_layers":counts.down_stride_four_layers,"down_stride_four_dispatches":counts.down_stride_four_dispatches}))
            } else {
                None
            };
            let frozen_gu_integer = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_gu_integer(&executor)?
            } else {
                false
            };
            let (gu_preparations, gu_dispatches, gu_audits) =
                if executor.target_frozen_host_source_identity.is_some() {
                    let counts = target.native_frozen_source_counters(&executor)?;
                    (
                        counts.gu_integer_preparations,
                        counts.gu_integer_dispatches,
                        counts.gu_integer_audits,
                    )
                } else {
                    (0, 0, 0)
                };
            let frozen_gu_integer_counters = crate::target_lab_build::frozen_gu_integer_record(
                input.frozen_gu_integer,
                frozen_gu_integer,
                gu_preparations,
                gu_dispatches,
                gu_audits,
            )?;
            if input.decode_mode == DecodeMode::TargetOnly
                && frozen_gu_integer
                && gu_preparations
                    != stats
                        .continuations
                        .checked_mul(48)
                        .ok_or("target-only integer batch count overflow")?
            {
                return Err("target-only actual integer count1 batch accounting rejected".into());
            }
            let frozen_gu_stride_four = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_gu_stride_four(&executor)?
            } else {
                false
            };
            let gu_stride_dispatches = if executor.target_frozen_host_source_identity.is_some() {
                target
                    .native_frozen_source_counters(&executor)?
                    .gu_stride_four_dispatches
            } else {
                0
            };
            let frozen_gu_stride_four_counters =
                crate::target_lab_build::frozen_gu_stride_four_record(
                    input.frozen_gu_stride_four,
                    frozen_gu_stride_four,
                    frozen_gu_integer,
                    gu_dispatches,
                    gu_stride_dispatches,
                )?;
            let frozen_down_stride_four = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_down_stride_four(&executor)?
            } else {
                false
            };
            let (down_layers, down_dispatches, down_source_dispatches) =
                if executor.target_frozen_host_source_identity.is_some() {
                    let counts = target.native_frozen_source_counters(&executor)?;
                    (
                        counts.down_stride_four_layers,
                        counts.down_stride_four_dispatches,
                        counts.source_dispatches,
                    )
                } else {
                    (0, 0, 0)
                };
            let frozen_down_stride_four_counters =
                crate::target_lab_build::frozen_down_stride_four_record(
                    input.frozen_down_stride_four,
                    frozen_down_stride_four,
                    down_layers,
                    down_dispatches,
                    down_source_dispatches,
                )?;
            if input.decode_mode == DecodeMode::TargetOnly
                && frozen_down_stride_four
                && down_layers
                    != stats
                        .continuations
                        .checked_mul(48)
                        .ok_or("target-only DOWN batch count overflow")?
            {
                return Err("target-only actual DOWN count1 batch accounting rejected".into());
            }
            let (public_grouped_ple_reads, grouped_ple_completed_chunks) =
                target.completed_public_grouped_ple_reads(&executor)?;
            let grouped_ple_reads_receipt =
                crate::target_lab_build::public_grouped_ple_reads_record(
                    input.public_grouped_ple_reads,
                    public_grouped_ple_reads,
                    grouped_ple_completed_chunks,
                    stats.rounds,
                    stats.repairs,
                )?;
            let (
                frozen_down_integer,
                down_integer_preparations,
                down_integer_dispatches,
                down_integer_audits,
                down_integer_sources,
            ) = if executor.target_frozen_host_source_identity.is_some() {
                let counts = target.native_frozen_source_counters(&executor)?;
                (
                    target.native_frozen_source_down_integer(&executor)?,
                    counts.down_integer_preparations,
                    counts.down_integer_dispatches,
                    counts.down_integer_audits,
                    counts.source_dispatches,
                )
            } else {
                (false, 0, 0, 0, 0)
            };
            let frozen_down_integer_counters = crate::target_lab_build::frozen_down_integer_record(
                input.frozen_down_integer,
                frozen_down_integer,
                down_integer_preparations,
                down_integer_dispatches,
                down_integer_audits,
                down_integer_sources,
            )?;
            let (actual_iq4_decoder, iq4_dispatches) = executor.public_target_iq4_decoder()?;
            let target_iq4_decoder_receipt = crate::target_lab_build::public_iq4_decoder_record(
                input.public_iq4_decoder,
                actual_iq4_decoder,
                iq4_dispatches,
                executor.target_verify3_dense_calls,
            )?;
            let (public_grouped_status_folds, completed_status_folds) =
                target.completed_public_grouped_status_folds(&executor)?;
            let grouped_status_folds_receipt =
                crate::target_lab_build::public_grouped_status_folds_record(
                    input.public_grouped_status_folds,
                    public_grouped_status_folds,
                    completed_status_folds,
                    stats.rounds,
                    stats.repairs,
                )?;
            let (three_gdn_streaming, three_gdn_completed_chunks) =
                target.completed_public_three_gdn_streaming(&executor)?;
            if three_gdn_streaming != input.three_gdn_streaming {
                return Err("language actual completed GDN policy mismatch".into());
            }
            let frozen_partition_exit = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_partition_exit(&executor)?
            } else {
                false
            };
            let frozen_private_tables = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_private_tables(&executor)?
            } else {
                false
            };
            let frozen_workgroup_tables = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_workgroup_tables(&executor)?
            } else {
                false
            };
            if frozen_workgroup_tables != input.frozen_workgroup_tables {
                return Err("actual frozen workgroup tables policy mismatch".into());
            }
            let frozen_shared_down_scales = if executor.target_frozen_host_source_identity.is_some()
            {
                target.native_frozen_source_shared_down_scales(&executor)?
            } else {
                false
            };
            if frozen_shared_down_scales != input.frozen_shared_down_scales {
                return Err("actual frozen shared down scales policy mismatch".into());
            }
            let frozen_four_row_down = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_four_row_down(&executor)?
            } else {
                false
            };
            if frozen_four_row_down != input.frozen_four_row_down {
                return Err("actual four-row down policy mismatch".into());
            }
            let frozen_two_row_gate_up = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_two_row_gate_up(&executor)?
            } else {
                false
            };
            let frozen_three_members = if executor.target_frozen_host_source_identity.is_some() {
                target.native_frozen_source_three_members(&executor)?
            } else {
                false
            };
            if frozen_three_members != input.frozen_three_members {
                return Err("actual three-member policy mismatch".into());
            }
            let frozen_packed_host_staging =
                if executor.target_frozen_host_source_identity.is_some() {
                    target.native_frozen_source_packed_staging(&executor)?
                } else {
                    false
                };
            if frozen_packed_host_staging != input.frozen_packed_host_staging {
                return Err("actual packed host staging policy mismatch".into());
            }
            if frozen_two_row_gate_up != input.frozen_two_row_gate_up {
                return Err("actual two-row gate/up policy mismatch".into());
            }
            let mtp_english_vocab = draft
                .as_ref()
                .map(|draft| draft.completed_public_english_vocab())
                .transpose()?
                .unwrap_or(false);
            if mtp_english_vocab != input.mtp_english_vocab {
                return Err("actual completed English vocabulary mismatch".into());
            }
            let mtp_english_vocab_receipt = if mtp_english_vocab {
                Some(
                    draft
                        .as_ref()
                        .ok_or("English actual draft absent")?
                        .public_english_vocab_receipt()?,
                )
            } else {
                None
            };
            let mtp_q8_head_shared_scales = draft
                .as_ref()
                .map(|draft| draft.completed_q8_head_shared_scales())
                .transpose()?
                .unwrap_or(false);
            if mtp_q8_head_shared_scales != input.mtp_q8_head_shared_scales {
                return Err("actual MTP shared head scales mismatch".into());
            }
            let mtp_q8_head_broadcast = draft
                .as_ref()
                .map(|draft| draft.completed_q8_head_broadcast())
                .transpose()?
                .unwrap_or(false);
            if mtp_q8_head_broadcast != input.mtp_q8_head_broadcast {
                return Err("actual completed MTP Q8 head policy mismatch".into());
            }
            let frozen_shared_gate_up_factors =
                if executor.target_frozen_host_source_identity.is_some() {
                    target.native_frozen_source_shared_gate_up_factors(&executor)?
                } else {
                    false
                };
            if frozen_shared_gate_up_factors != input.frozen_shared_gate_up_factors {
                return Err("frozen public gate/up factors policy mismatch".into());
            }
            if frozen_private_tables != input.frozen_private_tables {
                return Err("actual frozen private table policy mismatch".into());
            }
            if frozen_partition_exit != input.frozen_partition_exit {
                return Err("frozen source partition exit policy changed".into());
            }
            if actual_direct != input.public_existing_fill
                || target.native_live_member_reduction(&executor)? != input.live_member_reduction
            {
                return Err("language construction-fixed policy changed".into());
            }
            let completed_bank_fill = target.native_frozen_source_bank_fill(&executor)?;
            let bank_fill_receipt = crate::target_lab_build::public_host_bank_fill_record(
                input.bank_fill,
                completed_bank_fill,
                bank_fill_setup_experts,
            )?;
            if bank_fill_receipt != attached_bank_fill_receipt {
                return Err("language bank filler immutable setup receipt changed".into());
            }
            let physical_read_bytes = process_disk_read_bytes()?
                .checked_sub(load_disk_after)
                .ok_or("language process read counter regressed")?;
            crate::strata_port::public_preview(&public_text);
            let mut record = serde_json::json!({"fixture":input.fixture.name(),"prompt_tokens":prompt.len(),"context_capacity":input.fixture.context(),"compiled_prompt_template_revision":TEMPLATE_REVISION,"compiled_prompt_template_file":TEMPLATE_FILE,"load_ns":load_ns,"prefill_ns_including_real_teacher_kv":prefill_ns,"prefill_teacher_kv_executed":draft.is_some(),"prefill_first_output_count":1,"prefill_first_output_generation_pick_timed":false,"first_output_emission_timed":true,"emitted_tokens_including_eos":emitted.len(),"post_prefill_target_continuations":stats.continuations,"stop_reason":if confirmed==graph.eos_token(){"eos"}else{"limit"},"nonempty_text":nonempty_text,"elapsed_ns":elapsed_ns,"continuation_tokens_per_second":stats.continuations as f64*1e9/elapsed_ns as f64,"emitted_tokens_per_second":emitted.len()as f64*1e9/elapsed_ns as f64,"rounds":stats.rounds,"frozen_host_gpu_router":input.frozen_host_router,"frozen_host_inactive_partition_exit":frozen_partition_exit,"frozen_host_private_tables":frozen_private_tables,"frozen_host_workgroup_tables":frozen_workgroup_tables,"frozen_host_shared_down_scales":frozen_shared_down_scales,"frozen_host_shared_gate_up_factors":frozen_shared_gate_up_factors,"mtp_q8_head_broadcast":mtp_q8_head_broadcast,"mtp_q8_head_shared_scales":mtp_q8_head_shared_scales,"mtp_english_vocab":mtp_english_vocab,"mtp_english_vocab_receipt":mtp_english_vocab_receipt,"frozen_host_four_row_down":frozen_four_row_down,"frozen_host_two_row_gate_up":frozen_two_row_gate_up,"frozen_packed_host_staging":frozen_packed_host_staging,"frozen_three_member_arithmetic":frozen_three_members,"frozen_gu_integer":frozen_gu_integer,"frozen_gu_integer_counters":frozen_gu_integer_counters,"frozen_down_integer":frozen_down_integer,"frozen_down_integer_counters":frozen_down_integer_counters,"frozen_gu_stride_four":frozen_gu_stride_four,"frozen_gu_stride_four_counters":frozen_gu_stride_four_counters,"frozen_down_stride_four":frozen_down_stride_four,"frozen_down_stride_four_counters":frozen_down_stride_four_counters,"target_grouped_ple_reads":public_grouped_ple_reads,"target_grouped_ple_reads_receipt":grouped_ple_reads_receipt,"target_grouped_status_folds":public_grouped_status_folds,"target_grouped_status_folds_receipt":grouped_status_folds_receipt,"initial_prefill_target_grouped_status_folds":initial_prefill_target_grouped_status_folds,"initial_prefill_frozen_down_integer":initial_prefill_frozen_down_integer,"initial_prefill_frozen_down_integer_counters":initial_prefill_frozen_down_integer_counters,"initial_prefill_target_grouped_status_folds_receipt":initial_prefill_target_grouped_status_folds_receipt,"target_three_gdn_streaming":three_gdn_streaming,"target_three_gdn_completed_chunks":three_gdn_completed_chunks,"gpu_router_calls":executor.gpu_router_calls-before_gpu_routers,"proposed_drafts":stats.proposed,"accepted_drafts":stats.accepted_drafts,"draft_ns":stats.draft_ns,"verify_ns_full_packets":stats.verify_ns,"repair_ns_including_vector_checks":stats.repair_ns,"actual_target_recomputations":stats.repairs,"state_only_prefix_repairs":stats.state_replay_repairs,"repair_logit_max_abs":stats.repair_logit_max,"repair_hidden_max_abs":stats.repair_hidden_max,"repair_abs_tolerance":ABS,"repair_rel_tolerance":REL,"fixture_peak_host_bytes":fixture_peak,"request_scope":request_scope,"request_close_ns_inside_timing":request_close_ns,"host_io":host_io,"target_host_phase_ns":phase_ns,"verify3_dense_calls":executor.target_verify3_dense_calls-before_dense,"verify3_fused_calls":executor.target_verify3_fused_calls-before_fused,"load_and_prefill_physical_read_bytes":load_disk_after.checked_sub(load_disk_before).ok_or("language process read counter regressed")?,"generation_physical_read_bytes":physical_read_bytes,"expert_uploads":executor.expert_uploads-before_uploads,"expert_hits":executor.expert_hits-before_hits,"uploaded_bytes":executor.uploaded_bytes-before_bytes,"weight_read_ns":executor.weight_read_ns-before_weight_reads,"weight_upload_enqueue_ns":executor.weight_upload_ns-before_weight_uploads,"target_fill_counters_before":fill_counter_record(before_fills),"target_fill_counters_after":fill_counter_record(after_fills),"frozen_source_routes":frozen_routes,"draft_executed":draft.is_some(),"target_only_counters":target_only_counters,"decode_mode":input.decode_mode.name(),"draft_traffic":draft_traffic});
            record["frozen_bank_fill"] =
                crate::target_lab_build::public_host_bank_fill_name(completed_bank_fill.policy)
                    .into();
            record["mtp_batch_repair_requested"] = input.mtp_batch_repair.into();
            record["mtp_batch_repair"] = mtp_batch_repair_receipt;
            record["mtp_fused_four_repair"] = mtp_fused_four_repair_receipt;
            record["initial_prefill_mtp_batch_repair"] = initial_prefill_mtp_batch_repair;
            record["initial_prefill_mtp_fused_four_repair"] = initial_prefill_mtp_fused_four_repair;
            record["frozen_bank_fill_receipt"] = bank_fill_receipt;
            record["target_integer_head_requested"] = input.public_integer_verify_head.into();
            record["target_head_requested"] = if input.public_integer_verify_head {
                "integer-public"
            } else {
                "original"
            }
            .into();
            record["target_integer_head_receipt"] = target_integer_head_receipt;
            record["initial_prefill_target_integer_head_receipt"] =
                initial_prefill_target_integer_head_receipt;
            if crate::strata_port::enabled() {
                record["repair_logit_max_abs"] = serde_json::Value::Null;
                record["strata_draft_policy"] = draft_policy.receipt();
                record["target_packet"] = serde_json::json!({"gpu_argmax":true,"full_logit_readback":false,"vocabulary_rows_evaluated":VOCAB,"carry_rows_retained":true,"packet_bytes":(8+4+4*HIDDEN)*4});
            }
            if crate::strata_port::enabled() {
                record["strata_adaptive_cache"] = host_attachment
                    .as_ref()
                    .ok_or("Strata cache receipt attachment absent")?
                    .strata_cache_receipt()?;
            }
            record["target_public_cache_bytes_including_banks"] = executor.cache_bytes().into();
            record["target_exact_static_cache"] = crate::strata_port::exact_static_cache().into();
            record["target_static_cache_limit_bytes"] =
                crate::strata_port::static_cache_bytes().into();
            record["target_four_lookahead_receipt"] = four_lookahead_receipt;
            record["initial_prefill_target_four_lookahead_receipt"] =
                initial_four_lookahead_receipt;
            record["initial_prefill_frozen_bank_fill"] =
                crate::target_lab_build::public_host_bank_fill_name(initial_bank_fill.policy)
                    .into();
            record["initial_prefill_frozen_bank_fill_receipt"] = initial_prefill_bank_fill_receipt;
            record["target_iq4_decoder"] =
                crate::target_lab_build::public_iq4_decoder_name(actual_iq4_decoder).into();
            record["target_iq4_decoder_receipt"] = target_iq4_decoder_receipt;
            record["initial_prefill_target_iq4_decoder"] =
                crate::target_lab_build::public_iq4_decoder_name(initial_iq4_decoder).into();
            record["initial_prefill_target_iq4_decoder_receipt"] =
                initial_prefill_target_iq4_decoder_receipt;
            Ok(record)
        })();
        let attached_resident_bank_allocation = host_attachment
            .as_ref()
            .map(|owner| owner.resident_bank_allocation())
            .transpose();
        if crate::strata_intel_core::enabled()
            && let Some(d) = draft.as_mut()
        {
            // Peer cleanup uses the live authoritative ledger. Target close
            // deliberately poisons it, so this order is part of the contract.
            shared_draft_close = Some(d.close());
        }
        let close = target.close();
        let audit = target.public_owned_zero_audit_receipt();
        drop(target);
        let cleanup = (|| -> Result<_> {
            close?;
            let (main, sources) = audit.ok_or("language target complete erasure rejected")?;
            let host_close = if let Some(owner) = host_attachment.take() {
                // SAFETY: actual complete private close/audit succeeded and lexical
                // model drop removed every SourceSet consumer.
                Some(unsafe { owner.close()? })
            } else {
                None
            };
            if !crate::strata_intel_core::enabled() {
                crate::target_lab_build::validate_frozen_integer_math_close(
                    input.frozen_gu_integer,
                    input.frozen_down_integer,
                    main.buffers,
                    main.bytes,
                )?;
                if input.decode_mode == DecodeMode::TargetOnly {
                    crate::target_lab_build::validate_public_target_only_close(
                        input.frozen_gu_integer,
                        main.buffers,
                        main.bytes,
                    )?;
                }
            }
            Ok((main, sources, host_close))
        })();
        if runs.is_err() {
            println!(
                "public_language_failed_cleanup complete={}",
                u8::from(cleanup.is_ok())
            );
        }
        let mut record = runs?;
        record["target_submission_backlog"] = target_submission_backlog_receipt.clone();
        let (main, sources, host_close) = cleanup?;
        let resident_bank_allocation =
            crate::target_lab_build::public_resident_bank_allocation_record(
                attached_resident_bank_allocation?,
                host_close
                    .as_ref()
                    .map(|closed| closed.resident_bank_allocation),
            )?;
        record["target_close"] = serde_json::json!({"main_buffers":main.buffers,"main_bytes":main.bytes,"source_buffers":sources.buffers,"source_bytes":sources.bytes});
        record["frozen_host_setup"] = host_setup.into();
        record["resident_bank_allocation"] = resident_bank_allocation.clone();
        record["frozen_host_close"] = host_close
            .map(|closed| {
                serde_json::json!({
                    "registered_pages":closed.registered_pages,
                    "detached_pages":closed.detached_pages,
                    "physical_allocated_bytes":closed.mirror.total_allocated_bytes,
                    "physical_freed_bytes":closed.mirror.total_freed_bytes,
                    "live_backing_bytes":closed.mirror.live_backing_bytes,
                    "metadata_owned_bytes":closed.mirror.metadata_owned_bytes,
                    "source_window_peak_bytes":closed.mirror.source_window_peak_bytes,
                    "resident_bank_allocation":resident_bank_allocation,
                })
            })
            .into();
        Ok(record)
    })();
    let target_close = executor.close_transient_resident();
    drop(executor);
    // This public root belongs to the same client. Release it only after its
    // executor consumers close, and before the final physical-empty audit.
    drop(shared_iq_tables);
    let draft_close = if crate::strata_intel_core::enabled() {
        match shared_draft_close {
            Some(result) => result.map(Some),
            None => draft.as_mut().map(|d| d.close()).transpose(),
        }
        .and_then(|mut closed| {
            if let (Some(d), Some(r)) = (draft.as_ref(), closed.as_mut()) {
                d.finish_shared_root(r)?;
            } else if draft.is_none() {
                crate::executor::audit_native_context_recovery(client, &target_retired)?;
            }
            Ok(closed)
        })
    } else {
        draft.as_mut().map(|d| d.close()).transpose()
    };
    drop(draft);
    let draft_close = draft_close.and_then(|closed| {
        if let Some(closed) = &closed {
            input.draft_profile.validate_close_with_english(
                closed,
                input.mtp_english_vocab && !crate::strata_intel_core::enabled(),
            )?;
        }
        Ok(closed)
    });
    if attempt.is_err() {
        println!(
            "public_language_failed_device_cleanup target={} draft={}",
            u8::from(target_close.is_ok()),
            u8::from(draft_close.is_ok())
        );
    }
    let record = attempt?;
    target_close?;
    let draft_close = draft_close?;
    if draft_close.is_none() != (input.decode_mode == DecodeMode::TargetOnly) {
        return Err("language actual draft closing scope rejected".into());
    }
    if target_retired.load(Ordering::Acquire) || draft_retired.load(Ordering::Acquire) {
        return Err("language physical stream retired".into());
    }
    file.verify_file_unchanged()?;
    if let Some(sidecar) = &sidecar {
        sidecar.verify_file_unchanged()?;
    }
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    if client.public_target_request_io_configured()? != input.request_reuse {
        return Err("language target stream configuration changed".into());
    }
    let final_target_backlog = if input.request_reuse {
        Some(client.public_target_submission_backlog()?)
    } else {
        if client.public_target_submission_backlog().is_ok() {
            return Err("ordinary target stream acquired fixed backlog policy after close".into());
        }
        None
    };
    let final_backlog_receipt = if crate::strata_intel_core::enabled() {
        crate::target_lab_build::native_shared_backlog_record(
            input.target_submission_backlog,
            final_target_backlog,
            input.request_reuse,
        )?
    } else {
        crate::target_lab_build::public_target_submission_backlog_record(
            input.target_submission_backlog,
            final_target_backlog,
            input.request_reuse,
            draft_target_backlog_query_rejected,
        )?
    };
    if final_backlog_receipt != target_submission_backlog_receipt {
        return Err("immutable target backlog policy changed before full close".into());
    }
    if crate::strata_intel_core::enabled() {
        let draft_addressable = draft_close.as_ref().map_or(0, |r| r.public_bytes);
        let draft_additional = if draft_close.is_some() {
            draft_addressable
                .checked_sub(crate::strata_shared_weights::addressable_bytes())
                .ok_or("native shared public quote")?
        } else {
            0
        };
        let value = serde_json::json!({"shared_target_public_bytes":crate::strata_shared_weights::public_bytes(),
            "shared_target_host_embedding_bytes":crate::strata_shared_weights::embedding_bytes(),
            "shared_target_addressable_bytes":crate::strata_shared_weights::addressable_bytes(),
            "draft_public_addressable_bytes":draft_addressable,"draft_public_additional_bytes":draft_additional,
            "scope":"strata-global-connected-execution-check",
            "benchmark":false,"whole_rewrite_accepted":false,"record":record,
            "draft_owners_erased":draft_close.is_some(),"target_owners_erased":true,
            "target_submission_backlog":final_backlog_receipt});
        serde_json::to_writer_pretty(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(input.receipt)?,
            &value,
        )?;
        println!("strata_global_execution_complete owners_erased=true speed_acceptance=false");
        return Ok(());
    }
    let draft_close = draft_close.map(|closed| {
        serde_json::json!({
        "private_buffers":closed.private.buffers,"private_bytes":closed.private.bytes,
        "public_bytes":closed.public_bytes,"main_reserved":closed.main_reserved})
    });
    let draft_executed = draft_close.is_some();
    let mut value = serde_json::json!({"scope":if input.decode_mode == DecodeMode::TargetOnly {
        "compiled-public-language200-real-target-only-greedy"
    } else if input.mtp_device == crate::target_lab_build::MtpDevice::Intel {
        "compiled-public-language200-real-single-physical-device-independent-stream-greedy"
    } else { "compiled-public-language200-real-dual-device-greedy" },"decode_mode":input.decode_mode.name(),"draft_executed":draft_executed,"target_card":"B70","draft_card":if draft_executed {Some(if input.mtp_device == crate::target_lab_build::MtpDevice::Intel {"B70"}else{"RTX3060"})}else{None},"mtp_device_requested":if draft_executed {Some(input.mtp_device.name())}else{None},"mtp_device":mtp_device_receipt,"intel_mtp_admission":intel_mtp_admission,"draft_profile":if draft_executed{Some(input.draft_profile.name())}else{None},"gpu_chronological_qsa_selector":input.gpu_qsa_selector,"qsa_extra_runtime_info_bytes":if input.gpu_qsa_selector{crate::resident_qsa::batch::EXTRA_INFO_BYTES_PER_CHUNK}else{0},"qsa_extra_host_reserved_bytes":if input.gpu_qsa_selector{crate::resident_qsa::batch::EXTRA_HOST_BYTES_PER_CHUNK}else{0},"draft_public_reserved_bytes":if draft_executed{input.draft_profile.public_bytes_with_english(input.mtp_english_vocab)?}else{0},"frozen_gu_integer_requested":input.frozen_gu_integer,"strata_rust_port":crate::strata_port::receipt(),"compiled_count4_compact_down_staged":crate::quant_matvec::frozen_down_integer::four::STAGED_PUBLIC_PROBE,"compiled_count4_compact_shuffle":crate::quant_matvec::frozen_gu_integer::four::shuffle::PUBLIC_PROBE,"compiled_count4_compact_gu_first_plane":crate::quant_matvec::frozen_gu_integer::four::FIRST_PLANE_PUBLIC_PROBE,"frozen_down_math":if input.frozen_down_integer {"integer-public"}else{"original"},"frozen_down_integer_requested":input.frozen_down_integer,"frozen_down_integer_extra_main_bytes":if input.frozen_down_integer{crate::executor::batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES}else{0},"frozen_down_integer_extra_host_bytes":if input.frozen_down_integer{crate::executor::batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES}else{0},"frozen_down_integer_extra_info_bytes":if input.frozen_down_integer{196608}else{0},"frozen_down_integer_extra_runtime_bytes":if input.frozen_down_integer{crate::executor::batch_moe::FROZEN_DOWN_INTEGER_RUNTIME_BYTES}else{0},"frozen_gu_stride_four_requested":input.frozen_gu_stride_four,"frozen_down_schedule":if input.frozen_down_stride_four {"four-public"} else {"original"},"frozen_down_stride_four_requested":input.frozen_down_stride_four,"target_ple_reads":if input.public_grouped_ple_reads {"grouped-public"} else {"original"},"target_grouped_ple_reads_requested":input.public_grouped_ple_reads,"target_status_folds":if input.public_grouped_status_folds {"grouped-public"} else {"original"},"target_grouped_status_folds_requested":input.public_grouped_status_folds,"frozen_gu_integer_extra_main_bytes":if input.frozen_gu_integer {65536} else {0},"frozen_gu_integer_extra_host_bytes":if input.frozen_gu_integer {65536} else {0},"frozen_gu_integer_extra_info_bytes":if input.frozen_gu_integer {196608} else {0},"frozen_gu_integer_extra_runtime_bytes":if input.frozen_gu_integer {462848} else {0},"device_dispatch":crate::target_lab_build::name(input.inline_dispatch),"observed_inline_device_dispatch":crate::target_lab_build::actual_inline(),"temperature":0,"seed":424242,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,"thinking":false,"max_outputs":MAX_OUTPUTS,"verify3_dense":input.verify3_dense,"verify3_logical_finite_fused":input.verify3_fused,"prepared_verify3_experts":input.prepared_experts,"live_member_reduction":input.live_member_reduction,"public_existing_fill":input.public_existing_fill,"chunk_io_checkpoint":input.chunk_io,"request_io_reuse":input.request_reuse,"fused_mtp_sweeps":input.fused_sweeps,"fixture_host_reserved_bytes":FIXTURE_HOST_BYTES,"tokenizer_host_cap_bytes":TOKENIZER_HOST_BYTES,"request_reuse_metadata_reserved_bytes":if input.request_reuse{4*1024*1024}else{0},"full_target_packets_retained":!crate::strata_port::enabled(),"greedy_target_verification":true,"repair_original_prefix_vector_checks":input.decode_mode == DecodeMode::Speculative && !crate::strata_port::enabled(),"independent_cpu200_reference_accepted":false,"forced_oracle_tokens":false,"forced_experts":false,"private_or_production_accepted":false,"record":record,"draft_close":draft_close});
    value["frozen_bank_fill_mode"] =
        crate::target_lab_build::public_host_bank_fill_name(input.bank_fill).into();
    value["frozen_bank_fill_requested"] =
        (input.bank_fill == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller).into();
    value["target_iq4_decoder_mode"] = if input.public_iq4_decoder {
        "packed-registers-public"
    } else {
        "original"
    }
    .into();
    value["target_iq4_decoder_requested"] = input.public_iq4_decoder.into();
    value["target_head_requested"] = if input.public_integer_verify_head {
        "integer-public"
    } else {
        "original"
    }
    .into();
    value["target_submission_backlog"] = final_backlog_receipt;
    value["target_lookahead_requested"] = if input.four_lookahead {
        "four-public"
    } else {
        "original-three"
    }
    .into();
    value["mtp_repair_mode"] = if input.four_lookahead && input.mtp_batch_repair {
        "batch-public-fused-four-with-original-tails"
    } else if input.mtp_batch_repair {
        "batch-public"
    } else {
        "original"
    }
    .into();
    value["mtp_batch_repair_requested"] = input.mtp_batch_repair.into();
    serde_json::to_writer_pretty(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(input.receipt)?,
        &value,
    )?;
    println!("public_language_complete real_target_greedy=true owners_erased=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn retained_public_selector_defaults_off_and_rejects_private_or_unbounded_cases() {
        assert!(!super::retained_check_selection(None).unwrap());
        for raw in ["", "1", "2", "private", "forever", "two-identical-public "] {
            assert!(super::retained_check_selection(Some(std::ffi::OsStr::new(raw))).is_err());
        }
        let selected =
            super::retained_check_selection(Some(std::ffi::OsStr::new("two-identical-public")));
        if crate::strata_intel_core::enabled() {
            assert!(selected.unwrap());
        } else {
            assert!(selected.is_err());
        }
    }
    #[test]
    fn integer_verify_head_default_off_and_original_three_four_composition() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = intel_mtp_env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .public_integer_verify_head
        );
        for four in [false, true] {
            for extra in [false, true] {
                let mut selected = base.clone();
                for (key, value) in [
                    ("KUNO_LAB_TARGET_HEAD", "integer-public"),
                    ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
                    ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
                    ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                ] {
                    selected.push((key.into(), value.into()));
                }
                if four {
                    selected.push(("KUNO_LAB_TARGET_LOOKAHEAD".into(), "four-public".into()));
                }
                if extra {
                    for (key, value) in [
                        ("KUNO_LAB_MTP_REPAIR", "batch-public"),
                        ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                        ("KUNO_LAB_FROZEN_GU_SCHEDULE", "four-public"),
                        ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
                        ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                        ("KUNO_LAB_TARGET_SUBMISSION_BACKLOG", "queued-2048-public"),
                    ] {
                        selected.push((key.into(), value.into()));
                    }
                }
                let got = registration(&args, selected, build).unwrap();
                // ThreeStreaming is a separate approved three-row policy.
                if !four && extra {
                    let mut three = intel_mtp_env();
                    for (key, value) in [
                        ("KUNO_LAB_TARGET_HEAD", "integer-public"),
                        ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
                        ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
                        ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                        ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
                    ] {
                        three.push((key.into(), value.into()));
                    }
                    let three = registration(&args, three, build).unwrap();
                    assert!(three.public_integer_verify_head && three.three_gdn_streaming);
                }
                assert!(
                    got.public_integer_verify_head
                        && got.frozen_gu_integer
                        && got.frozen_down_integer
                        && got.public_grouped_ple_reads
                );
                assert_eq!(got.four_lookahead, four);
                assert_eq!(got.mtp_batch_repair, extra);
            }
        }
    }
    #[test]
    fn integer_verify_head_rejects_missing_scope_unknown_values_and_alternate_math() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = intel_mtp_env();
        for (key, value) in [
            ("KUNO_LAB_TARGET_HEAD", "integer-public"),
            ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
            ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
            ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        for key in [
            "KUNO_LAB_FROZEN_GU_MATH",
            "KUNO_LAB_FROZEN_DOWN_MATH",
            "KUNO_LAB_TARGET_PLE_READS",
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
            "KUNO_LAB_MTP_DEVICE",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_TARGET_HEAD", "integer"),
            ("KUNO_LAB_TARGET_HEAD", "original"),
            ("KUNO_LAB_TARGET_HEAD", ""),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_TARGET_IQ4_DECODER", "packed-registers-public"),
            ("KUNO_LAB_FROZEN_BANK_FILL", "profile-ranked-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCHEDULE", "four-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_LANGUAGE_DECODE", "target-only-public"),
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        assert!(
            registration(
                &args,
                base.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        let mut duplicate = base;
        duplicate.push(("KUNO_LAB_TARGET_HEAD".into(), "integer-public".into()));
        assert!(registration(&args, duplicate, build).is_err());
    }
    #[test]
    fn fused_four_batch_repair_registration_requires_exact_composition() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = intel_mtp_env();
        for (key, value) in [
            ("KUNO_LAB_TARGET_LOOKAHEAD", "four-public"),
            ("KUNO_LAB_MTP_REPAIR", "batch-public"),
            ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
            ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
            ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        for extra in [false, true] {
            let mut selected = base.clone();
            if extra {
                for (key, value) in [
                    ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                    ("KUNO_LAB_FROZEN_GU_SCHEDULE", "four-public"),
                    ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
                    ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                    ("KUNO_LAB_TARGET_SUBMISSION_BACKLOG", "queued-2048-public"),
                ] {
                    selected.push((key.into(), value.into()));
                }
            }
            let got = registration(&args, selected, build).unwrap();
            assert!(got.four_lookahead && got.mtp_batch_repair && got.public_grouped_ple_reads);
            assert!(got.frozen_gu_integer && got.frozen_down_integer);
            assert_eq!(got.frozen_gu_stride_four, extra);
        }
        for key in [
            "KUNO_LAB_FROZEN_GU_MATH",
            "KUNO_LAB_FROZEN_DOWN_MATH",
            "KUNO_LAB_TARGET_PLE_READS",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_TARGET_IQ4_DECODER", "packed-registers-public"),
            ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
            ("KUNO_LAB_FROZEN_BANK_FILL", "profile-ranked-public"),
            ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_LANGUAGE_DECODE", "target-only-public"),
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        let mut scalar_four = base.clone();
        scalar_four.retain(|(k, _)| k != "KUNO_LAB_MTP_REPAIR" && k != "KUNO_LAB_TARGET_PLE_READS");
        let got = registration(&args, scalar_four, build).unwrap();
        assert!(got.four_lookahead && !got.mtp_batch_repair && !got.public_grouped_ple_reads);
        base.retain(|(k, _)| {
            !matches!(
                k.to_str(),
                Some(
                    "KUNO_LAB_TARGET_LOOKAHEAD"
                        | "KUNO_LAB_FROZEN_GU_MATH"
                        | "KUNO_LAB_FROZEN_DOWN_MATH"
                        | "KUNO_LAB_TARGET_PLE_READS"
                )
            )
        });
        let got = registration(&args, base, build).unwrap();
        assert!(
            got.mtp_batch_repair
                && !got.four_lookahead
                && !got.frozen_gu_integer
                && !got.frozen_down_integer
        );
    }
    #[test]
    fn four_completed_statistics_preserve_default_three_and_count_true_four_rows() {
        let phase = crate::target_lab_build::LookaheadPhase {
            draft_ns: 10,
            third_follow_ns: 3,
            verify_ns: 20,
            repair_ns: 30,
        };
        let mut stats = Statistics::default();
        assert!(stats.add(4, 4, 10, 20, 30, false).is_err());
        assert!(stats.add_completed(false, 4, 4, phase, false).is_err());
        for accepted in 1..=4 {
            stats
                .add_completed(true, 4, accepted, phase, accepted < 4)
                .unwrap();
        }
        assert_eq!(
            (
                stats.rounds,
                stats.proposed,
                stats.accepted_drafts,
                stats.continuations,
                stats.repairs
            ),
            (4, 12, 6, 10, 3)
        );
        assert_eq!(
            (stats.draft_ns, stats.verify_ns, stats.repair_ns),
            (40, 80, 120)
        );
        assert!(stats.add_completed(true, 5, 5, phase, false).is_err());
        assert!(stats.add_completed(true, 4, 0, phase, false).is_err());
    }
    #[test]
    fn four_lookahead_registration_preserves_default_and_admits_original_composition() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = intel_mtp_env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .four_lookahead
        );
        for stride in [false, true] {
            let mut selected = base.clone();
            selected.push(("KUNO_LAB_TARGET_LOOKAHEAD".into(), "four-public".into()));
            for (key, value) in [
                ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
            ] {
                selected.push((key.into(), value.into()));
            }
            for (key, value) in [
                ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
                ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
            ] {
                selected.push((key.into(), value.into()));
            }
            if stride {
                selected.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
            }
            let got = registration(&args, selected, build).unwrap();
            assert!(
                got.four_lookahead
                    && !got.mtp_batch_repair
                    && !got.public_iq4_decoder
                    && !got.three_gdn_streaming
            );
            assert!(got.frozen_gu_integer && got.frozen_down_integer);
            assert_eq!(got.frozen_gu_stride_four, stride);
        }
    }
    #[test]
    fn four_lookahead_rejects_missing_scope_unknown_values_and_incompatible_policies() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = intel_mtp_env();
        base.push(("KUNO_LAB_TARGET_LOOKAHEAD".into(), "four-public".into()));
        base.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
        base.push(("KUNO_LAB_FROZEN_DOWN_MATH".into(), "integer-public".into()));
        for key in [
            "KUNO_LAB_FROZEN_GU_MATH",
            "KUNO_LAB_FROZEN_DOWN_MATH",
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
            "KUNO_LAB_MTP_DEVICE",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_TARGET_LOOKAHEAD", "four"),
            ("KUNO_LAB_TARGET_LOOKAHEAD", "4"),
            ("KUNO_LAB_MTP_REPAIR", "batch-public"),
            ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
            ("KUNO_LAB_TARGET_IQ4_DECODER", "packed-registers-public"),
            ("KUNO_LAB_FROZEN_BANK_FILL", "profile-ranked-public"),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        assert!(
            registration(
                &args,
                base.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        let mut duplicate = base;
        duplicate.push(("KUNO_LAB_TARGET_LOOKAHEAD".into(), "four-public".into()));
        assert!(registration(&args, duplicate, build).is_err());
        let mut target_only = target_only_env();
        target_only.push(("KUNO_LAB_TARGET_LOOKAHEAD".into(), "four-public".into()));
        assert!(registration(&args, target_only, build).is_err());
    }
    #[test]
    fn profile_ranked_bank_filler_default_off_and_composes_original_public_policies() {
        use crate::executor::PublicHostBankFillPolicy as P;
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = intel_mtp_env();
        assert_eq!(
            registration(&args, base.clone(), build).unwrap().bank_fill,
            P::Original
        );
        for (gu_integer, down_integer, down_stride, iq4) in [
            (false, false, false, false),
            (true, false, false, true),
            (true, true, false, true),
            (false, false, true, false),
        ] {
            let mut selected = base.clone();
            for (key, value) in [
                ("KUNO_LAB_FROZEN_BANK_FILL", "profile-ranked-public"),
                ("KUNO_LAB_TARGET_SUBMISSION_BACKLOG", "queued-2048-public"),
                ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
                ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
            ] {
                selected.push((key.into(), value.into()));
            }
            if gu_integer {
                selected.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                selected.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
            }
            if down_integer {
                selected.push(("KUNO_LAB_FROZEN_DOWN_MATH".into(), "integer-public".into()));
            } else {
                selected.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                if down_stride {
                    selected.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
                }
            }
            if iq4 {
                selected.push((
                    "KUNO_LAB_TARGET_IQ4_DECODER".into(),
                    "packed-registers-public".into(),
                ));
            }
            let got = registration(&args, selected, build).unwrap();
            assert_eq!(got.bank_fill, P::StrataRankedFiller);
            assert_eq!(got.frozen_gu_integer, gu_integer);
            assert_eq!(got.frozen_down_integer, down_integer);
            assert_eq!(got.frozen_down_stride_four, down_stride);
            assert_eq!(got.public_iq4_decoder, iq4);
            assert!(
                got.public_grouped_ple_reads
                    && got.public_grouped_status_folds
                    && got.three_gdn_streaming
                    && got.gpu_qsa_selector
            );
        }
    }

    #[test]
    fn profile_ranked_bank_filler_rejects_unrelated_scope_and_untrusted_profile_inputs() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = intel_mtp_env();
        base.push((
            "KUNO_LAB_FROZEN_BANK_FILL".into(),
            "profile-ranked-public".into(),
        ));
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_FROZEN_BANK_PLACEMENT", "preserve-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_PUBLIC_BANK_PROFILE_PATH", "/untrusted/profile"),
            ("KUNO_PUBLIC_BANK_PROFILE_SHA256", "untrusted"),
        ] {
            let mut invalid = base.clone();
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        let mut duplicate = base.clone();
        duplicate.push((
            "KUNO_LAB_FROZEN_BANK_FILL".into(),
            "profile-ranked-public".into(),
        ));
        assert!(registration(&args, duplicate, build).is_err());
        assert!(registration(&args, base, crate::target_lab_build::Build::Threaded).is_err());
    }

    #[test]
    fn profile_ranked_bank_filler_does_not_change_unselected_legacy_device_scopes() {
        use crate::executor::PublicHostBankFillPolicy as P;
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut legacy = intel_mtp_env();
        legacy.retain(|(key, _)| {
            ![
                "KUNO_LAB_MTP_DEVICE",
                "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES",
                "KUNO_PUBLIC_MTP_INTEL_USED_BYTES",
                "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID",
            ]
            .iter()
            .any(|removed| key == removed)
        });
        assert_eq!(
            registration(&args, legacy.clone(), build)
                .unwrap()
                .bank_fill,
            P::Original
        );
        legacy.push((
            "KUNO_LAB_FROZEN_BANK_FILL".into(),
            "profile-ranked-public".into(),
        ));
        assert!(registration(&args, legacy, build).is_err());
        let mut target_only = target_only_env();
        assert_eq!(
            registration(&args, target_only.clone(), build)
                .unwrap()
                .bank_fill,
            P::Original
        );
        target_only.push((
            "KUNO_LAB_FROZEN_BANK_FILL".into(),
            "profile-ranked-public".into(),
        ));
        assert!(registration(&args, target_only, build).is_err());
        let mut prose = intel_mtp_env();
        for (key, value) in &mut prose {
            if key == "KUNO_LAB_LANGUAGE_FIXTURE" {
                *value = "warm-prose".into();
            }
        }
        prose.push((
            "KUNO_LAB_FROZEN_BANK_FILL".into(),
            "profile-ranked-public".into(),
        ));
        assert_eq!(
            registration(&args, prose, build).unwrap().bank_fill,
            P::StrataRankedFiller
        );
    }

    #[test]
    fn target_backlog_default_off_and_original_public_compositions() {
        use cubecl_runtime::server::PublicTargetSubmissionBacklog as P;
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = intel_mtp_env();
        assert_eq!(
            registration(&args, base.clone(), build)
                .unwrap()
                .target_submission_backlog,
            P::Default512
        );
        for (gu_integer, down_integer, down_stride, iq4) in [
            (false, false, false, false),
            (true, false, false, true),
            (true, true, false, true),
            (false, false, true, false),
        ] {
            let mut selected = base.clone();
            for (key, value) in [
                ("KUNO_LAB_TARGET_SUBMISSION_BACKLOG", "queued-2048-public"),
                ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
                ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
            ] {
                selected.push((key.into(), value.into()));
            }
            if gu_integer {
                selected.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                selected.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
            }
            if down_integer {
                selected.push(("KUNO_LAB_FROZEN_DOWN_MATH".into(), "integer-public".into()));
            } else {
                selected.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                if down_stride {
                    selected.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
                }
            }
            if iq4 {
                selected.push((
                    "KUNO_LAB_TARGET_IQ4_DECODER".into(),
                    "packed-registers-public".into(),
                ));
            }
            let got = registration(&args, selected, build).unwrap();
            assert_eq!(got.target_submission_backlog, P::Queued2048);
            assert_eq!(got.frozen_gu_integer, gu_integer);
            assert_eq!(got.frozen_down_integer, down_integer);
            assert_eq!(got.frozen_down_stride_four, down_stride);
            assert_eq!(got.public_iq4_decoder, iq4);
            assert!(
                got.public_grouped_ple_reads
                    && got.public_grouped_status_folds
                    && got.three_gdn_streaming
                    && got.gpu_qsa_selector
            );
        }
    }

    #[test]
    fn target_backlog_rejects_missing_scope_and_other_draft_profiles() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = intel_mtp_env();
        base.push((
            "KUNO_LAB_TARGET_SUBMISSION_BACKLOG".into(),
            "queued-2048-public".into(),
        ));
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
        ] {
            let mut invalid = base.clone();
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        let mut nvidia = base.clone();
        nvidia.retain(|(k, _)| {
            !matches!(
                k.to_str(),
                Some(
                    "KUNO_LAB_MTP_DEVICE"
                        | "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES"
                        | "KUNO_PUBLIC_MTP_INTEL_USED_BYTES"
                        | "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID"
                )
            )
        });
        assert!(registration(&args, nvidia, build).is_err());
        assert!(registration(&args, base, crate::target_lab_build::Build::Threaded).is_err());
    }

    #[test]
    fn target_backlog_requires_canonical_value_and_unique_key() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        for raw in [
            "",
            "2048",
            "queued-2048",
            "queued-2048-public ",
            "1",
            "original-512",
        ] {
            let mut invalid = intel_mtp_env();
            invalid.push(("KUNO_LAB_TARGET_SUBMISSION_BACKLOG".into(), raw.into()));
            assert!(registration(&args, invalid, build).is_err());
        }
        let mut duplicate = intel_mtp_env();
        for _ in 0..2 {
            duplicate.push((
                "KUNO_LAB_TARGET_SUBMISSION_BACKLOG".into(),
                "queued-2048-public".into(),
            ));
        }
        assert!(registration(&args, duplicate, build).is_err());
        let mut target_only = intel_mtp_env();
        target_only.push((
            "KUNO_LAB_TARGET_SUBMISSION_BACKLOG".into(),
            "queued-2048-public".into(),
        ));
        target_only.push((
            "KUNO_LAB_LANGUAGE_DECODE".into(),
            "target-only-public".into(),
        ));
        assert!(registration(&args, target_only, build).is_err());
    }

    #[test]
    fn grouped_status_registration_keeps_default_off_and_original_public_compositions() {
        let args = args();
        let build = crate::target_lab_build::Build::Inline;
        let base = frozen_down_stride_four_env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .public_grouped_status_folds
        );
        for intel in [false, true] {
            for (integer, stride) in [(false, false), (true, false), (true, true)] {
                let mut valid = base.clone();
                for (key, value) in [
                    ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
                    ("KUNO_LAB_MTP_SWEEPS", "fused"),
                    ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                    ("KUNO_LAB_FROZEN_DOWN_SCHEDULE", "four-public"),
                    ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                    ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
                ] {
                    valid.push((key.into(), value.into()));
                }
                if integer {
                    valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                }
                if stride {
                    valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                }
                if intel {
                    for (key, value) in [
                        ("KUNO_LAB_MTP_DEVICE", "intel-public"),
                        ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "34242297856"),
                        ("KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "50688000"),
                        ("KUNO_PUBLIC_MTP_INTEL_DEVICE_ID", "57891"),
                    ] {
                        valid.push((key.into(), value.into()));
                    }
                }
                let got = registration(&args, valid, build).unwrap();
                assert!(
                    got.public_grouped_status_folds
                        && got.frozen_down_stride_four
                        && got.three_gdn_streaming
                );
                assert_eq!(got.frozen_gu_integer, integer);
                assert_eq!(got.frozen_gu_stride_four, stride);
                assert_eq!(
                    got.mtp_device == crate::target_lab_build::MtpDevice::Intel,
                    intel
                );
            }
        }
        let mut q2 = base;
        q2.push((
            "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
            "grouped-public".into(),
        ));
        q2.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        assert!(
            registration(&args, q2, build)
                .unwrap()
                .public_grouped_status_folds
        );
    }
    #[test]
    fn grouped_status_registration_rejects_missing_original_gates_and_incompatible_math() {
        let args = args();
        let build = crate::target_lab_build::Build::Inline;
        let mut base = frozen_down_stride_four_env();
        base.push((
            "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
            "grouped-public".into(),
        ));
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_BANK_PLACEMENT", "preserve-public"),
        ] {
            let mut invalid = base.clone();
            invalid.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for raw in ["grouped", "true", "", "grouped-public "] {
            let mut invalid = base.clone();
            invalid.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, invalid, build).is_err());
        }
        base.push((
            "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
            "grouped-public".into(),
        ));
        assert!(registration(&args, base, build).is_err());
        for build in [
            crate::target_lab_build::Build::Threaded,
            crate::target_lab_build::Build::Rejected,
        ] {
            let mut invalid = frozen_down_stride_four_env();
            invalid.push((
                "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
                "grouped-public".into(),
            ));
            assert!(registration(&args, invalid, build).is_err());
        }
    }
    #[test]
    fn grouped_status_registration_accepts_real_target_only_without_a_draft_selector() {
        let build = crate::target_lab_build::Build::Inline;
        for fixture in ["first-code", "warm-prose"] {
            for (integer, stride) in [(false, false), (true, false), (true, true)] {
                let mut valid = target_only_env();
                valid
                    .iter_mut()
                    .find(|(k, _)| k == "KUNO_LAB_LANGUAGE_FIXTURE")
                    .unwrap()
                    .1 = fixture.into();
                valid.push((
                    "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
                    "grouped-public".into(),
                ));
                if integer {
                    valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                }
                if stride {
                    valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                }
                let got = registration(&args(), valid, build).unwrap();
                assert!(
                    got.public_grouped_status_folds && got.decode_mode == DecodeMode::TargetOnly
                );
                assert!(!got.fused_sweeps);
            }
        }
    }
    #[test]
    fn grouped_ple_registration_keeps_default_off_and_original_public_compositions() {
        let args = args();
        let build = crate::target_lab_build::Build::Inline;
        let base = frozen_down_stride_four_env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .public_grouped_ple_reads
        );
        for intel in [false, true] {
            for (integer, stride) in [(false, false), (true, false), (true, true)] {
                let mut valid = base.clone();
                for (key, value) in [
                    ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                    ("KUNO_LAB_MTP_SWEEPS", "fused"),
                    ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                    ("KUNO_LAB_FROZEN_DOWN_SCHEDULE", "four-public"),
                    ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                    ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
                ] {
                    valid.push((key.into(), value.into()));
                }
                if integer {
                    valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                }
                if stride {
                    valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                }
                if intel {
                    for (key, value) in [
                        ("KUNO_LAB_MTP_DEVICE", "intel-public"),
                        ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "34242297856"),
                        ("KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "50688000"),
                        ("KUNO_PUBLIC_MTP_INTEL_DEVICE_ID", "57891"),
                    ] {
                        valid.push((key.into(), value.into()));
                    }
                }
                let got = registration(&args, valid, build).unwrap();
                assert!(
                    got.public_grouped_ple_reads
                        && got.frozen_down_stride_four
                        && got.three_gdn_streaming
                );
                assert_eq!(got.frozen_gu_integer, integer);
                assert_eq!(got.frozen_gu_stride_four, stride);
                assert_eq!(
                    got.mtp_device == crate::target_lab_build::MtpDevice::Intel,
                    intel
                );
            }
        }
        let mut q2 = base;
        q2.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
        q2.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        assert!(
            registration(&args, q2, build)
                .unwrap()
                .public_grouped_ple_reads
        );
    }
    #[test]
    fn grouped_ple_registration_rejects_missing_original_gates_and_incompatible_math() {
        let args = args();
        let build = crate::target_lab_build::Build::Inline;
        let mut base = frozen_down_stride_four_env();
        base.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_BANK_PLACEMENT", "preserve-public"),
        ] {
            let mut invalid = base.clone();
            invalid.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for raw in ["grouped", "true", "", "grouped-public "] {
            let mut invalid = base.clone();
            invalid.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, invalid, build).is_err());
        }
        base.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
        assert!(registration(&args, base, build).is_err());
        for build in [
            crate::target_lab_build::Build::Threaded,
            crate::target_lab_build::Build::Rejected,
        ] {
            let mut invalid = frozen_down_stride_four_env();
            invalid.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
            assert!(registration(&args, invalid, build).is_err());
        }
    }
    #[test]
    fn grouped_ple_registration_accepts_real_target_only_without_a_draft_selector() {
        let build = crate::target_lab_build::Build::Inline;
        for fixture in ["first-code", "warm-prose"] {
            for (integer, stride) in [(false, false), (true, false), (true, true)] {
                let mut valid = target_only_env();
                valid
                    .iter_mut()
                    .find(|(k, _)| k == "KUNO_LAB_LANGUAGE_FIXTURE")
                    .unwrap()
                    .1 = fixture.into();
                valid.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
                if integer {
                    valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                }
                if stride {
                    valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                }
                let got = registration(&args(), valid, build).unwrap();
                assert!(got.public_grouped_ple_reads && got.decode_mode == DecodeMode::TargetOnly);
                assert!(!got.fused_sweeps);
            }
        }
    }
    #[test]
    fn grouped_status_target_only_composes_ple_down_without_a_draft() {
        let build = crate::target_lab_build::Build::Inline;
        for fixture in ["first-code", "warm-prose"] {
            for ple in [false, true] {
                for down in [false, true] {
                    let mut valid = target_only_env();
                    valid
                        .iter_mut()
                        .find(|(k, _)| k == "KUNO_LAB_LANGUAGE_FIXTURE")
                        .unwrap()
                        .1 = fixture.into();
                    valid.push((
                        "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
                        "grouped-public".into(),
                    ));
                    if ple {
                        valid.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
                    }
                    if down {
                        valid.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                        valid.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
                    }
                    let got = registration(&args(), valid, build).unwrap();
                    assert!(
                        got.public_grouped_status_folds
                            && got.decode_mode == DecodeMode::TargetOnly
                    );
                    assert_eq!(got.public_grouped_ple_reads, ple);
                    assert_eq!(got.frozen_down_stride_four, down);
                    assert!(!got.fused_sweeps);
                }
            }
        }
    }
    #[test]
    fn grouped_status_registration_composes_independently_with_ple_and_down() {
        let args = args();
        let build = crate::target_lab_build::Build::Inline;
        for status in [false, true] {
            for ple in [false, true] {
                for down in [false, true] {
                    let mut valid = frozen_down_stride_four_env();
                    if status {
                        valid.push((
                            "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
                            "grouped-public".into(),
                        ));
                    }
                    if ple {
                        valid.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
                    }
                    if down {
                        valid.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
                    }
                    let got = registration(&args, valid, build).unwrap();
                    assert_eq!(got.public_grouped_status_folds, status);
                    assert_eq!(got.public_grouped_ple_reads, ple);
                    assert_eq!(got.frozen_down_stride_four, down);
                }
            }
        }
    }
    #[test]
    fn frozen_down_integer_composes_gu_math_and_reads_on_each_explicit_placement() {
        let build = crate::target_lab_build::Build::Inline;
        for intel in [false, true] {
            for (gu, stride) in [(false, false), (true, false), (true, true)] {
                let mut valid = frozen_down_stride_four_env();
                valid.retain(|(key, _)| key != "KUNO_LAB_FROZEN_DOWN_ROWS");
                for (key, value) in [
                    ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
                    ("KUNO_LAB_MTP_SWEEPS", "fused"),
                    ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                    ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                    ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
                    ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                    ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
                ] {
                    valid.push((key.into(), value.into()));
                }
                if gu {
                    valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                }
                if stride {
                    valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                }
                if intel {
                    for (key, value) in [
                        ("KUNO_LAB_MTP_DEVICE", "intel-public"),
                        ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "34242297856"),
                        ("KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "50688000"),
                        ("KUNO_PUBLIC_MTP_INTEL_DEVICE_ID", "57891"),
                    ] {
                        valid.push((key.into(), value.into()));
                    }
                }
                let got = registration(&args(), valid, build).unwrap();
                assert!(
                    got.frozen_down_integer
                        && got.public_grouped_ple_reads
                        && got.public_grouped_status_folds
                );
                assert_eq!(got.frozen_gu_integer, gu);
                assert_eq!(got.frozen_gu_stride_four, stride);
                assert!(!got.frozen_four_row_down && !got.frozen_down_stride_four);
            }
        }
    }
    #[test]
    fn frozen_down_integer_rejects_target_only_without_altering_existing_gu_scope() {
        let build = crate::target_lab_build::Build::Inline;
        let mut valid = target_only_env();
        valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
        assert!(
            registration(&args(), valid.clone(), build)
                .unwrap()
                .frozen_gu_integer
        );
        valid.push(("KUNO_LAB_FROZEN_DOWN_MATH".into(), "integer-public".into()));
        assert!(registration(&args(), valid, build).is_err());
    }
    use super::*;
    fn args() -> Vec<OsString> {
        ["binary", "intel", "daslab-language-probe"]
            .map(Into::into)
            .to_vec()
    }
    fn env() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "1"),
            ("KUNO_LAB_LANGUAGE_FIXTURE", "first-code"),
            ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
            ("KUNO_TEST_Q3_DIR", "/original"),
            ("KUNO_TEST_MTP", "/original/sidecar"),
            ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .to_vec()
    }
    fn target_only_env() -> Vec<(OsString, OsString)> {
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_LANGUAGE_DECODE", "target-only-public"),
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        base
    }
    #[test]
    fn target_only_registration_is_canonical_and_retains_fixed_original_target_scope() {
        let build = crate::target_lab_build::Build::Inline;
        assert_eq!(
            registration(&args(), env(), crate::target_lab_build::Build::Threaded)
                .unwrap()
                .decode_mode,
            DecodeMode::Speculative
        );
        let base = target_only_env();
        for fixture in ["first-code", "warm-prose"] {
            for integer in [false, true] {
                for stride in [false, true] {
                    let mut valid = base.clone();
                    for (key, value) in &mut valid {
                        if key == "KUNO_LAB_LANGUAGE_FIXTURE" {
                            *value = fixture.into();
                        }
                    }
                    if integer {
                        valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                    }
                    if stride {
                        valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                    }
                    valid.push(("KUNO_LAB_TARGET_QSA".into(), "chronological-public".into()));
                    valid.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                    let got = registration(&args(), valid, build);
                    assert_eq!(got.is_ok(), !stride || integer);
                    if let Ok(got) = got {
                        assert_eq!(got.decode_mode, DecodeMode::TargetOnly);
                        assert_eq!(got.frozen_gu_integer, integer);
                        assert_eq!(got.frozen_gu_stride_four, stride);
                        assert!(got.gpu_qsa_selector && got.frozen_four_row_down);
                        assert!(
                            !got.fused_sweeps
                                && !got.mtp_english_vocab
                                && !got.mtp_q8_head_broadcast
                                && !got.mtp_q8_head_shared_scales
                        );
                    }
                }
            }
        }
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args(), bad, build).is_err(), "{key}");
        }
        for raw in [
            "target-only",
            "speculative",
            "true",
            "",
            "target-only-public ",
        ] {
            let mut bad = base.clone();
            bad.iter_mut()
                .find(|(k, _)| k == "KUNO_LAB_LANGUAGE_DECODE")
                .unwrap()
                .1 = raw.into();
            assert!(registration(&args(), bad, build).is_err());
        }
        let mut duplicate = base.clone();
        duplicate.push((
            "KUNO_LAB_LANGUAGE_DECODE".into(),
            "target-only-public".into(),
        ));
        assert!(registration(&args(), duplicate, build).is_err());
        for build in [
            crate::target_lab_build::Build::Threaded,
            crate::target_lab_build::Build::Rejected,
        ] {
            assert!(registration(&args(), base.clone(), build).is_err());
        }
    }
    #[test]
    fn target_only_registration_rejects_draft_and_nonoriginal_target_experiments() {
        let build = crate::target_lab_build::Build::Inline;
        for (key, value) in [
            ("KUNO_LAB_MTP_PROFILE", "self-contained-q8"),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_SWEEPS", "fused"),
            ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_TARGET_GDN", "three-streaming-public"),
        ] {
            let mut bad = target_only_env();
            bad.push((key.into(), value.into()));
            assert!(registration(&args(), bad, build).is_err(), "{key}");
        }
        // A parsed sidecar path remains a transport field. Registration does
        // not require it to exist; run() never opens it in target-only mode.
        let mut unopened = target_only_env();
        unopened
            .iter_mut()
            .find(|(k, _)| k == "KUNO_TEST_MTP")
            .unwrap()
            .1 = "/does/not/exist".into();
        assert_eq!(
            registration(&args(), unopened, build).unwrap().decode_mode,
            DecodeMode::TargetOnly
        );
    }
    #[test]
    fn target_only_completed_counters_require_real_count1_and_no_draft_or_repair_credit() {
        for continuations in [0, 1, 37, 199] {
            let mut stats = Statistics::default();
            for _ in 0..continuations {
                stats.add(1, 1, 0, 10, 0, false).unwrap();
            }
            assert_eq!(
                stats
                    .target_only_record(DecodeMode::TargetOnly, false, continuations + 1)
                    .unwrap(),
                Some(
                    serde_json::json!({"count1_chunks":continuations,"completed_target_rows":continuations,
                    "draft_executed":false,"draft_steps":0,"target_repair_recomputations":0})
                )
            );
            assert!(
                stats
                    .target_only_record(DecodeMode::TargetOnly, true, continuations + 1)
                    .is_err()
            );
            assert!(
                stats
                    .target_only_record(DecodeMode::TargetOnly, false, continuations + 2)
                    .is_err()
            );
            assert!(
                stats
                    .target_only_record(DecodeMode::Speculative, false, continuations + 1)
                    .is_err()
            );
            assert!(
                stats
                    .target_only_record(DecodeMode::Speculative, true, continuations + 1)
                    .unwrap()
                    .is_none()
            );
        }
        for (count, accepted, draft_ns, repair_ns, repaired) in [
            (2, 1, 0, 0, false),
            (2, 2, 0, 0, false),
            (1, 1, 1, 0, false),
            (1, 1, 0, 1, false),
            (1, 1, 0, 0, true),
        ] {
            let mut stats = Statistics::default();
            stats
                .add(count, accepted, draft_ns, 10, repair_ns, repaired)
                .unwrap();
            assert!(
                stats
                    .target_only_record(DecodeMode::TargetOnly, false, accepted + 1)
                    .is_err()
            );
        }
        let mut missing = Statistics::default();
        missing.add(1, 1, 0, 0, 0, false).unwrap();
        assert!(
            missing
                .target_only_record(DecodeMode::TargetOnly, false, 2)
                .is_err()
        );
    }
    fn intel_mtp_env() -> Vec<(OsString, OsString)> {
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
            ("KUNO_LAB_MTP_SWEEPS", "fused"),
            ("KUNO_LAB_MTP_DEVICE", "intel-public"),
            ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "34242297856"),
            ("KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "50688000"),
            ("KUNO_PUBLIC_MTP_INTEL_DEVICE_ID", "4660"),
        ] {
            base.push((key.into(), value.into()));
        }

        base
    }
    #[test]
    fn mtp_batch_repair_registration_keeps_original_default_and_composes_exact_scope() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = intel_mtp_env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .mtp_batch_repair
        );
        for composed in [false, true] {
            let mut selected = base.clone();
            selected.push(("KUNO_LAB_MTP_REPAIR".into(), "batch-public".into()));
            if composed {
                for (key, value) in [
                    ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
                    ("KUNO_LAB_FROZEN_GU_SCHEDULE", "four-public"),
                    ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
                    ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
                    ("KUNO_LAB_TARGET_STATUS_FOLDS", "grouped-public"),
                    ("KUNO_LAB_TARGET_QSA", "chronological-public"),
                    ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
                ] {
                    selected.push((key.into(), value.into()));
                }
            }
            let got = registration(&args, selected, build).unwrap();
            assert!(got.mtp_batch_repair);
            assert_eq!(got.frozen_down_integer, composed);
        }
    }

    #[test]
    fn mtp_batch_repair_registration_rejects_missing_scope_and_unaccepted_draft_math() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = intel_mtp_env();
        base.push(("KUNO_LAB_MTP_REPAIR".into(), "batch-public".into()));
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
        ] {
            let mut missing = base.clone();
            missing.retain(|(k, _)| k != key);
            assert!(registration(&args, missing, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_BANK_FILL", "profile-ranked-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
        ] {
            let mut invalid = base.clone();
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        let mut nvidia = base.clone();
        nvidia.retain(|(k, _)| {
            !matches!(
                k.to_str(),
                Some(
                    "KUNO_LAB_MTP_DEVICE"
                        | "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES"
                        | "KUNO_PUBLIC_MTP_INTEL_USED_BYTES"
                        | "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID"
                )
            )
        });
        assert!(registration(&args, nvidia, build).is_err());
        assert!(registration(&args, base, crate::target_lab_build::Build::Threaded).is_err());
    }

    #[test]
    fn mtp_batch_repair_registration_rejects_noncanonical_and_duplicate_selection() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        for raw in ["", "batch", "true", "1", "batch-public ", "original"] {
            let mut invalid = intel_mtp_env();
            invalid.push(("KUNO_LAB_MTP_REPAIR".into(), raw.into()));
            assert!(registration(&args, invalid, build).is_err());
        }
        let mut duplicate = intel_mtp_env();
        for _ in 0..2 {
            duplicate.push(("KUNO_LAB_MTP_REPAIR".into(), "batch-public".into()));
        }
        assert!(registration(&args, duplicate, build).is_err());
    }

    #[test]
    fn mtp_batch_repair_never_acquires_target_only_draft_scope() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let mut invalid = env();
        invalid.push((
            "KUNO_LAB_LANGUAGE_DECODE".into(),
            "target-only-public".into(),
        ));
        invalid.push(("KUNO_LAB_MTP_REPAIR".into(), "batch-public".into()));
        assert!(registration(&args, invalid, crate::target_lab_build::Build::Threaded).is_err());
    }

    #[test]
    fn intel_mtp_controller_registration_retains_full144_original_target_and_explicit_profile() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = intel_mtp_env();
        for q2 in [false, true] {
            let mut selected = base.clone();
            if q2 {
                selected.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
            }
            let got = registration(&args, selected, build).unwrap();
            assert_eq!(got.mtp_device, crate::target_lab_build::MtpDevice::Intel);
            assert_eq!(
                got.draft_profile,
                if q2 {
                    Profile::StrataQ2
                } else {
                    Profile::SelfContainedQ8
                }
            );
            assert!(
                got.frozen_host_sources
                    && got.frozen_host_router
                    && got.frozen_partition_exit
                    && got.verify3_fused
                    && got.request_reuse
                    && got.fused_sweeps
            );
            assert!(got.intel_mtp_memory.is_some());
        }
        // Exact first comparison policy: full Q8, shared head, original DOWN,
        // chronological QSA and the already admitted integer/stride GU option.
        let mut q8 = base.clone();
        for (key, value) in [
            ("KUNO_LAB_MTP_Q8_HEAD", "shared-public"),
            ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
            ("KUNO_LAB_TARGET_QSA", "chronological-public"),
            ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
            ("KUNO_LAB_FROZEN_GU_SCHEDULE", "four-public"),
        ] {
            q8.push((key.into(), value.into()));
        }
        let got = registration(&args, q8.clone(), build).unwrap();
        assert!(
            got.frozen_gu_integer
                && got.frozen_gu_stride_four
                && got.mtp_q8_head_shared_scales
                && got.frozen_four_row_down
                && got.gpu_qsa_selector
        );
        q8.push((
            "KUNO_LAB_TARGET_GDN".into(),
            "three-streaming-public".into(),
        ));
        assert!(registration(&args, q8, build).unwrap().three_gdn_streaming);
        let mut q2 = base.clone();
        for (key, value) in [
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
            ("KUNO_LAB_TARGET_QSA", "chronological-public"),
        ] {
            q2.push((key.into(), value.into()));
        }
        let got = registration(&args, q2, build).unwrap();
        assert!(
            !got.frozen_gu_integer
                && !got.mtp_q8_head_shared_scales
                && got.frozen_four_row_down
                && got.gpu_qsa_selector
        );
        let mut default = base.clone();
        default.retain(|(key, _)| {
            !key.to_string_lossy().starts_with("KUNO_PUBLIC_MTP_INTEL_")
                && key != "KUNO_LAB_MTP_DEVICE"
        });
        let got = registration(&args, default.clone(), build).unwrap();
        assert_eq!(got.mtp_device, crate::target_lab_build::MtpDevice::Nvidia);
        assert!(got.intel_mtp_memory.is_none());
        default.push(("KUNO_PUBLIC_MTP_INTEL_USED_BYTES".into(), "50688000".into()));
        assert!(registration(&args, default, build).is_err());
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
            "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES",
            "KUNO_PUBLIC_MTP_INTEL_USED_BYTES",
            "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_MTP_DEVICE", "intel-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
        ] {
            let mut bad = base.clone();
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for value in ["auto", "intel", "nvidia", "intel-public ", ""] {
            let mut bad = base.clone();
            bad.iter_mut()
                .find(|(key, _)| key == "KUNO_LAB_MTP_DEVICE")
                .unwrap()
                .1 = value.into();
            assert!(registration(&args, bad, build).is_err());
        }
        let mut target_only = base;
        target_only.retain(|(key, _)| key != "KUNO_LAB_MTP_SWEEPS");
        target_only.push((
            "KUNO_LAB_LANGUAGE_DECODE".into(),
            "target-only-public".into(),
        ));
        assert!(registration(&args, target_only, build).is_err());
    }

    #[test]
    fn iq4_decoder_registration_requires_fixed_intel_full_q8_and_keeps_old_defaults() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = frozen_down_stride_four_env();
        for (key, value) in [
            ("KUNO_LAB_MTP_SWEEPS", "fused"),
            ("KUNO_LAB_MTP_DEVICE", "intel-public"),
            ("KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES", "34242297856"),
            ("KUNO_PUBLIC_MTP_INTEL_USED_BYTES", "50688000"),
            ("KUNO_PUBLIC_MTP_INTEL_DEVICE_ID", "57891"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .public_iq4_decoder
        );
        base.push((
            "KUNO_LAB_TARGET_IQ4_DECODER".into(),
            "packed-registers-public".into(),
        ));
        for (gu, stride) in [(false, false), (true, false), (true, true)] {
            for down in ["plain", "four", "integer"] {
                for reads in [false, true] {
                    let mut valid = base.clone();
                    if down != "four" {
                        valid.retain(|(key, _)| key != "KUNO_LAB_FROZEN_DOWN_ROWS");
                    }
                    if down == "integer" {
                        valid.push(("KUNO_LAB_FROZEN_DOWN_MATH".into(), "integer-public".into()));
                    }
                    if gu {
                        valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                    }
                    if stride {
                        valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                    }
                    if reads {
                        valid.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
                        valid.push((
                            "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
                            "grouped-public".into(),
                        ));
                    }
                    valid.push(("KUNO_LAB_MTP_Q8_HEAD".into(), "shared-public".into()));
                    let got = registration(&args, valid, build).unwrap();
                    assert!(got.public_iq4_decoder);
                    assert_eq!(got.frozen_gu_integer, gu);
                    assert_eq!(got.frozen_gu_stride_four, stride);
                    assert_eq!(got.frozen_down_integer, down == "integer");
                }
            }
        }
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_TARGET_IQ4_DECODER", "packed-registers-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCHEDULE", "four-public"),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
        ] {
            let mut bad = base.clone();
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        let mut prose = base.clone();
        for (key, value) in &mut prose {
            if key == "KUNO_LAB_LANGUAGE_FIXTURE" {
                *value = "warm-prose".into();
            }
        }
        assert!(
            registration(&args, prose, build)
                .unwrap()
                .public_iq4_decoder
        );
        let mut nvidia = base;
        nvidia.retain(|(key, _)| {
            key != "KUNO_LAB_MTP_DEVICE"
                && !key.to_string_lossy().starts_with("KUNO_PUBLIC_MTP_INTEL_")
        });
        assert!(registration(&args, nvidia.clone(), build).is_err());
        nvidia.retain(|(key, _)| key != "KUNO_LAB_TARGET_IQ4_DECODER");
        assert!(
            !registration(&args, nvidia, build)
                .unwrap()
                .public_iq4_decoder
        );
        let mut target_only = target_only_env();
        assert!(
            !registration(&args, target_only.clone(), build)
                .unwrap()
                .public_iq4_decoder
        );
        target_only.push((
            "KUNO_LAB_TARGET_IQ4_DECODER".into(),
            "packed-registers-public".into(),
        ));
        assert!(registration(&args, target_only, build).is_err());
    }

    fn frozen_down_stride_four_env() -> Vec<(OsString, OsString)> {
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
            ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        base
    }
    #[test]
    fn frozen_down_stride_four_registration_accepts_plain_integer_and_stride_gu() {
        let args = args();
        let build = crate::target_lab_build::Build::Inline;
        let base = frozen_down_stride_four_env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_down_stride_four
        );
        for fixture in ["first-code", "warm-prose"] {
            for (integer, stride) in [(false, false), (true, false), (true, true)] {
                let mut valid = base.clone();
                valid
                    .iter_mut()
                    .find(|(k, _)| k == "KUNO_LAB_LANGUAGE_FIXTURE")
                    .unwrap()
                    .1 = fixture.into();
                valid.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
                if integer {
                    valid.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
                    valid.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                }
                if stride {
                    valid.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                }
                valid.push(("KUNO_LAB_TARGET_QSA".into(), "chronological-public".into()));
                let got = registration(&args, valid, build).unwrap();
                assert!(
                    got.frozen_down_stride_four && got.frozen_four_row_down && got.gpu_qsa_selector
                );
                assert_eq!(got.frozen_gu_integer, integer);
                assert_eq!(got.frozen_gu_stride_four, stride);
            }
        }
        // DOWN scheduling preserves original target math regardless of draft
        // format or sweep choice; it adds no fabricated GU/MTP dependency.
        let mut q2 = base;
        q2.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
        q2.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        let got = registration(&args, q2, build).unwrap();
        assert!(got.frozen_down_stride_four && !got.frozen_gu_integer && !got.fused_sweeps);
        assert_eq!(got.draft_profile, Profile::StrataQ2);
    }
    #[test]
    fn frozen_down_stride_four_registration_requires_exact_original_target_scope() {
        let args = args();
        let build = crate::target_lab_build::Build::Inline;
        let mut base = frozen_down_stride_four_env();
        base.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_FROZEN_DOWN_ROWS",
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
        ] {
            let mut invalid = base.clone();
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        for raw in ["four", "true", "", "four-public "] {
            let mut invalid = base.clone();
            invalid.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, invalid, build).is_err());
        }
        let mut duplicate = base.clone();
        duplicate.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
        assert!(registration(&args, duplicate, build).is_err());
        for build in [
            crate::target_lab_build::Build::Threaded,
            crate::target_lab_build::Build::Rejected,
        ] {
            assert!(registration(&args, base.clone(), build).is_err());
        }
    }
    #[test]
    fn frozen_down_stride_four_target_only_code_prose_keeps_all_original_gu_options() {
        let build = crate::target_lab_build::Build::Inline;
        for fixture in ["first-code", "warm-prose"] {
            for (integer, stride) in [(false, false), (true, false), (true, true)] {
                let mut base = target_only_env();
                base.iter_mut()
                    .find(|(k, _)| k == "KUNO_LAB_LANGUAGE_FIXTURE")
                    .unwrap()
                    .1 = fixture.into();
                base.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                base.push(("KUNO_LAB_FROZEN_DOWN_SCHEDULE".into(), "four-public".into()));
                if integer {
                    base.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
                }
                if stride {
                    base.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
                }
                let got = registration(&args(), base, build).unwrap();
                assert_eq!(got.decode_mode, DecodeMode::TargetOnly);
                assert!(got.frozen_down_stride_four && !got.fused_sweeps);
                assert_eq!(got.frozen_gu_integer, integer);
                assert_eq!(got.frozen_gu_stride_four, stride);
            }
        }
    }
    #[test]
    fn frozen_gu_stride_four_controller_registration_is_explicit_and_integer_only() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
            ("KUNO_LAB_MTP_SWEEPS", "fused"),
        ] {
            base.push((key.into(), value.into()));
        }

        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_gu_stride_four
        );
        let mut no_integer = base.clone();
        no_integer.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
        assert!(registration(&args, no_integer, build).is_err());
        base.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_gu_stride_four
        );
        for raw in ["four-public", "four", "true", "", "four-public "] {
            let mut selected = base.clone();
            selected.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), raw.into()));
            let got = registration(&args, selected, build);
            assert_eq!(got.is_ok(), raw == "four-public");
            if let Ok(got) = got {
                assert!(got.frozen_gu_stride_four && got.frozen_gu_integer);
            }
        }
        base.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
        base.push(("KUNO_LAB_FROZEN_GU_SCHEDULE".into(), "four-public".into()));
        assert!(registration(&args, base, build).is_err());
    }
    #[test]
    fn frozen_down_integer_controller_registration_requires_complete_original_public_scope() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
            ("KUNO_LAB_MTP_SWEEPS", "fused"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_down_integer
        );
        base.push(("KUNO_LAB_FROZEN_DOWN_MATH".into(), "integer-public".into()));
        for qsa in [false, true] {
            for grouped_status in [false, true] {
                for shared_head in [false, true] {
                    for three_gdn in [false, true] {
                        let mut valid = base.clone();
                        if qsa {
                            valid.push((
                                "KUNO_LAB_TARGET_QSA".into(),
                                "chronological-public".into(),
                            ));
                        }
                        if grouped_status {
                            valid.push((
                                "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
                                "grouped-public".into(),
                            ));
                        }
                        if shared_head {
                            valid.push(("KUNO_LAB_MTP_Q8_HEAD".into(), "shared-public".into()));
                        }
                        if three_gdn {
                            valid.push((
                                "KUNO_LAB_TARGET_GDN".into(),
                                "three-streaming-public".into(),
                            ));
                        }
                        let got = registration(&args, valid, build).unwrap();
                        assert!(got.frozen_down_integer);
                        assert_eq!(got.gpu_qsa_selector, qsa);
                        assert_eq!(got.public_grouped_status_folds, grouped_status);
                        assert_eq!(got.mtp_q8_head_shared_scales, shared_head);
                        assert_eq!(got.three_gdn_streaming, three_gdn);
                    }
                }
            }
        }
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCHEDULE", "four-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
        ] {
            let mut bad = base.clone();
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for value in ["", "integer", "true", "integer-public "] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = value.into();
            assert!(registration(&args, bad, build).is_err());
        }
        let mut prose = base.clone();
        for (key, value) in &mut prose {
            if key == "KUNO_LAB_LANGUAGE_FIXTURE" {
                *value = "warm-prose".into();
            }
        }
        assert!(
            registration(&args, prose, build)
                .unwrap()
                .frozen_down_integer
        );
        let mut wrong_dense = base.clone();
        for (key, value) in &mut wrong_dense {
            if key == "KUNO_LAB_TARGET_DENSE" {
                *value = "verify3".into();
            }
        }
        assert!(registration(&args, wrong_dense, build).is_err());
    }

    #[test]
    fn frozen_gu_integer_controller_registration_requires_complete_original_public_scope() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
            ("KUNO_LAB_MTP_SWEEPS", "fused"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_gu_integer
        );
        base.push(("KUNO_LAB_FROZEN_GU_MATH".into(), "integer-public".into()));
        for qsa in [false, true] {
            for four_down in [false, true] {
                for shared_head in [false, true] {
                    for three_gdn in [false, true] {
                        let mut valid = base.clone();
                        if qsa {
                            valid.push((
                                "KUNO_LAB_TARGET_QSA".into(),
                                "chronological-public".into(),
                            ));
                        }
                        if four_down {
                            valid.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                        }
                        if shared_head {
                            valid.push(("KUNO_LAB_MTP_Q8_HEAD".into(), "shared-public".into()));
                        }
                        if three_gdn {
                            valid.push((
                                "KUNO_LAB_TARGET_GDN".into(),
                                "three-streaming-public".into(),
                            ));
                        }
                        let got = registration(&args, valid, build).unwrap();
                        assert!(got.frozen_gu_integer);
                        assert_eq!(got.gpu_qsa_selector, qsa);
                        assert_eq!(got.frozen_four_row_down, four_down);
                        assert_eq!(got.mtp_q8_head_shared_scales, shared_head);
                        assert_eq!(got.three_gdn_streaming, three_gdn);
                    }
                }
            }
        }
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
            ("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public"),
            ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
        ] {
            let mut bad = base.clone();
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for value in ["", "integer", "true", "integer-public "] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = value.into();
            assert!(registration(&args, bad, build).is_err());
        }
        let mut prose = base.clone();
        for (key, value) in &mut prose {
            if key == "KUNO_LAB_LANGUAGE_FIXTURE" {
                *value = "warm-prose".into();
            }
        }
        assert!(registration(&args, prose, build).unwrap().frozen_gu_integer);
        let mut wrong_dense = base.clone();
        for (key, value) in &mut wrong_dense {
            if key == "KUNO_LAB_TARGET_DENSE" {
                *value = "verify3".into();
            }
        }
        assert!(registration(&args, wrong_dense, build).is_err());
    }

    #[test]
    fn english_draft_vocabulary_requires_complete_fixed_public_scope() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
            ("KUNO_LAB_MTP_SWEEPS", "fused"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .mtp_english_vocab
        );
        base.push(("KUNO_LAB_MTP_VOCAB".into(), "english-public".into()));
        for shared in [false, true] {
            let mut valid = base.clone();
            if shared {
                valid.push(("KUNO_LAB_MTP_Q8_HEAD".into(), "shared-public".into()));
            }
            let got = registration(&args, valid, build).unwrap();
            assert!(got.mtp_english_vocab);
            assert_eq!(got.mtp_q8_head_shared_scales, shared);
        }
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_MTP_SWEEPS",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
        ] {
            let mut bad = base.clone();
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for value in ["", "english", "true", "english-public "] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = value.into();
            assert!(registration(&args, bad, build).is_err());
        }
    }

    #[test]
    fn three_gdn_requires_canonical_complete_public_scope_and_stays_default_off() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_DENSE", "verify3-fused"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .three_gdn_streaming
        );
        let flag = (
            OsString::from("KUNO_LAB_TARGET_GDN"),
            OsString::from("three-streaming-public"),
        );
        let mut selected = base.clone();
        selected.push(flag.clone());
        assert!(
            registration(&args, selected.clone(), build)
                .unwrap()
                .three_gdn_streaming
        );
        // Independent original MoE arithmetic/staging and Q2 draft may compose.
        for additions in [
            vec![
                ("KUNO_LAB_FROZEN_MEMBERS", "three-public"),
                ("KUNO_LAB_FROZEN_DOWN_ROWS", "four-public"),
            ],
            vec![("KUNO_LAB_FROZEN_HOST_STAGE", "packed-public")],
            vec![("KUNO_LAB_MTP_PROFILE", "strata-q2")],
        ] {
            let mut composed = selected.clone();
            composed.extend(additions.into_iter().map(|(k, v)| (k.into(), v.into())));
            assert!(
                registration(&args, composed, build)
                    .unwrap()
                    .three_gdn_streaming
            );
        }
        for key in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
        ] {
            let mut bad = selected.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err());
        }
        for (key, value) in [
            ("KUNO_LAB_TARGET_DENSE", "verify3"),
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_DECODE_KERNELS", "single-gdn"),
        ] {
            let mut bad = selected.clone();
            bad.retain(|(k, _)| k != key);
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err());
        }
        for value in [
            "",
            "1",
            "three-streaming",
            "Three-Streaming-Public",
            "three-streaming-public ",
        ] {
            let mut bad = base.clone();
            bad.push(("KUNO_LAB_TARGET_GDN".into(), value.into()));
            assert!(registration(&args, bad, build).is_err());
        }
        selected.push(flag);
        assert!(registration(&args, selected, build).is_err());
    }
    #[test]
    fn frozen_sources_require_original_public_scope_and_fixed_runtime() {
        let build = crate::target_lab_build::Build::Inline;
        let mut selected = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
        ] {
            selected.push((key.into(), value.into()));
        }
        assert!(
            registration(&args(), selected.clone(), build)
                .unwrap()
                .frozen_host_sources
        );
        let mut routed = selected.clone();
        routed.push((
            "KUNO_LAB_TARGET_ROUTER".into(),
            "device-frozen-public".into(),
        ));
        assert!(
            registration(&args(), routed.clone(), build)
                .unwrap()
                .frozen_host_router
        );
        routed.retain(|(k, _)| k != "KUNO_LAB_TARGET_SOURCES");
        assert!(registration(&args(), routed, build).is_err());
        for key in [
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_DEVICE_DISPATCH",
        ] {
            let mut wrong = selected.clone();
            wrong.retain(|(k, _)| k != key);
            assert!(registration(&args(), wrong, build).is_err());
        }
        for value in ["", "1", "frozen-host", "FROZEN-HOST-PUBLIC"] {
            let mut wrong = selected.clone();
            wrong
                .iter_mut()
                .find(|(k, _)| k == "KUNO_LAB_TARGET_SOURCES")
                .unwrap()
                .1 = value.into();
            assert!(registration(&args(), wrong, build).is_err());
        }
        for (key, value) in [
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
        ] {
            let mut wrong = selected.clone();
            wrong.push((key.into(), value.into()));
            assert!(registration(&args(), wrong, build).is_err());
        }
        selected.push((
            "KUNO_LAB_TARGET_SOURCES".into(),
            "frozen-host-public".into(),
        ));
        assert!(registration(&args(), selected, build).is_err());
    }

    #[test]
    fn cooperative_policies_compose_and_reject_scope_or_canonical_errors() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
        ] {
            base.push((key.into(), value.into()));
        }
        for tables in [None, Some("private-public"), Some("workgroup-public")] {
            for scales in [false, true] {
                let mut valid = base.clone();
                if let Some(t) = tables {
                    valid.push(("KUNO_LAB_FROZEN_TABLES".into(), t.into()));
                }
                if scales {
                    valid.push(("KUNO_LAB_FROZEN_DOWN_SCALES".into(), "shared-public".into()));
                }
                let got = registration(&args, valid.clone(), build).unwrap();
                assert_eq!(got.frozen_private_tables, tables == Some("private-public"));
                assert_eq!(
                    got.frozen_workgroup_tables,
                    tables == Some("workgroup-public")
                );
                assert_eq!(got.frozen_shared_down_scales, scales);
                if tables.is_some() || scales {
                    for prerequisite in [
                        "KUNO_LAB_TARGET_SOURCES",
                        "KUNO_LAB_FROZEN_PARTITION_EXIT",
                        "KUNO_LAB_TARGET_REUSE",
                    ] {
                        let mut bad = valid.clone();
                        bad.retain(|(k, _)| k != prerequisite);
                        assert!(registration(&args, bad, build).is_err());
                    }
                }
                if scales {
                    for raw in ["", "shared", "true", "shared-public "] {
                        let mut bad = valid.clone();
                        bad.iter_mut()
                            .find(|(k, _)| k == "KUNO_LAB_FROZEN_DOWN_SCALES")
                            .unwrap()
                            .1 = raw.into();
                        assert!(registration(&args, bad, build).is_err());
                    }
                    valid.push(("KUNO_LAB_FROZEN_DOWN_SCALES".into(), "shared-public".into()));
                    assert!(registration(&args, valid, build).is_err());
                }
            }
        }
    }

    #[test]
    fn q8_head_broadcast_requires_exact_public_q8_profile() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Threaded;
        let base = env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .mtp_q8_head_broadcast
        );
        let mut selected = base.clone();
        selected.push(("KUNO_LAB_MTP_Q8_HEAD".into(), "broadcast-public".into()));
        assert!(
            registration(&args, selected.clone(), build)
                .unwrap()
                .mtp_q8_head_broadcast
        );
        let mut shared = selected.clone();
        shared.last_mut().unwrap().1 = "shared-public".into();
        let got = registration(&args, shared.clone(), build).unwrap();
        assert!(got.mtp_q8_head_shared_scales && !got.mtp_q8_head_broadcast);
        shared.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        assert!(registration(&args, shared, build).is_err());
        let mut q2 = selected.clone();
        q2.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        assert!(registration(&args, q2, build).is_err());
        for raw in ["", "broadcast", "true", "broadcast-public "] {
            let mut bad = selected.clone();
            bad.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, bad, build).is_err());
        }
        selected.push(("KUNO_LAB_MTP_Q8_HEAD".into(), "broadcast-public".into()));
        assert!(registration(&args, selected, build).is_err());
        assert!(
            registration(
                &["lab", "intel", "batch-router-gpu-probe"].map(OsString::from),
                base,
                build
            )
            .is_err()
        );
    }

    #[test]
    fn shared_gate_up_factors_require_exact_public_scope() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_shared_gate_up_factors
        );
        base.push(("KUNO_LAB_FROZEN_GU_FACTORS".into(), "shared-public".into()));
        for tables in [None, Some("private-public"), Some("workgroup-public")] {
            for scales in [false, true] {
                let mut valid = base.clone();
                if let Some(t) = tables {
                    valid.push(("KUNO_LAB_FROZEN_TABLES".into(), t.into()));
                }
                if scales {
                    valid.push(("KUNO_LAB_FROZEN_DOWN_SCALES".into(), "shared-public".into()));
                }
                assert!(
                    registration(&args, valid.clone(), build)
                        .unwrap()
                        .frozen_shared_gate_up_factors
                );
                for key in [
                    "KUNO_LAB_TARGET_SOURCES",
                    "KUNO_LAB_FROZEN_PARTITION_EXIT",
                    "KUNO_LAB_TARGET_REUSE",
                ] {
                    let mut bad = valid.clone();
                    bad.retain(|(k, _)| k != key);
                    assert!(registration(&args, bad, build).is_err());
                }
            }
        }
        for raw in ["", "shared", "true", "shared-public "] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, bad, build).is_err());
        }
        base.push(("KUNO_LAB_FROZEN_GU_FACTORS".into(), "shared-public".into()));
        assert!(registration(&args, base, build).is_err());
    }

    #[test]
    fn three_members_requires_exact_public_router_and_preserves_staging_composition() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_three_members
        );
        base.push(("KUNO_LAB_FROZEN_MEMBERS".into(), "three-public".into()));
        for staged in [false, true] {
            for four in [false, true] {
                let mut valid = base.clone();
                if staged {
                    valid.push(("KUNO_LAB_FROZEN_HOST_STAGE".into(), "packed-public".into()));
                }
                if four {
                    valid.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                }
                assert!(
                    registration(&args, valid, build)
                        .unwrap()
                        .frozen_three_members
                );
            }
        }
        for key in [
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_TARGET_REUSE",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err());
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_ROWS", "two-public"),
        ] {
            let mut bad = base.clone();
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err());
        }
        for raw in ["", "true", "three", "three-public "] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, bad, build).is_err());
        }
        base.push(("KUNO_LAB_FROZEN_MEMBERS".into(), "three-public".into()));
        assert!(registration(&args, base, build).is_err());
    }
    #[test]
    fn packed_staging_requires_exact_public_gpu_router_and_plain_quantization() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_TARGET_ROUTER", "device-frozen-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_packed_host_staging
        );
        base.push(("KUNO_LAB_FROZEN_HOST_STAGE".into(), "packed-public".into()));
        for two in [false, true] {
            for four in [false, true] {
                let mut valid = base.clone();
                if two {
                    valid.push(("KUNO_LAB_FROZEN_GU_ROWS".into(), "two-public".into()));
                }
                if four {
                    valid.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                }
                assert!(
                    registration(&args, valid, build)
                        .unwrap()
                        .frozen_packed_host_staging
                );
            }
        }
        for key in [
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_TARGET_REUSE",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err());
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_DOWN_SCALES", "shared-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
        ] {
            let mut bad = base.clone();
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err());
        }
        for raw in ["", "true", "packed", "packed-public "] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, bad, build).is_err());
        }
        base.push(("KUNO_LAB_FROZEN_HOST_STAGE".into(), "packed-public".into()));
        assert!(registration(&args, base, build).is_err());
    }
    #[test]
    fn two_row_gate_up_requires_exact_public_scope_and_rejects_tables_factors() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_two_row_gate_up
        );
        base.push(("KUNO_LAB_FROZEN_GU_ROWS".into(), "two-public".into()));
        for down_rows in [false, true] {
            for down_scales in [false, true] {
                let mut valid = base.clone();
                if down_rows {
                    valid.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
                }
                if down_scales {
                    valid.push(("KUNO_LAB_FROZEN_DOWN_SCALES".into(), "shared-public".into()));
                }
                let parsed = registration(&args, valid, build).unwrap();
                assert!(parsed.frozen_two_row_gate_up);
                assert_eq!(parsed.frozen_four_row_down, down_rows);
                assert_eq!(parsed.frozen_shared_down_scales, down_scales);
            }
        }
        for (key, value) in [
            ("KUNO_LAB_FROZEN_TABLES", "private-public"),
            ("KUNO_LAB_FROZEN_TABLES", "workgroup-public"),
            ("KUNO_LAB_FROZEN_GU_FACTORS", "shared-public"),
        ] {
            for prepend in [false, true] {
                let mut bad = base.clone();
                if prepend {
                    bad.insert(0, (key.into(), value.into()));
                } else {
                    bad.push((key.into(), value.into()));
                }
                assert!(registration(&args, bad, build).is_err());
            }
        }
        for key in [
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_IO",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err());
        }
        for raw in ["", "two", "true", "1", "two-public ", "four-public"] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, bad, build).is_err());
        }
        base.push(("KUNO_LAB_FROZEN_GU_ROWS".into(), "two-public".into()));
        assert!(registration(&args, base, build).is_err());
    }

    #[test]
    fn four_row_down_requires_exact_public_scope() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
        ] {
            base.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_four_row_down
        );
        base.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
        for tables in [None, Some("private-public"), Some("workgroup-public")] {
            for scales in [false, true] {
                let mut valid = base.clone();
                if let Some(t) = tables {
                    valid.push(("KUNO_LAB_FROZEN_TABLES".into(), t.into()));
                }
                if scales {
                    valid.push(("KUNO_LAB_FROZEN_DOWN_SCALES".into(), "shared-public".into()));
                }
                assert!(
                    registration(&args, valid.clone(), build)
                        .unwrap()
                        .frozen_four_row_down
                );
                for key in [
                    "KUNO_LAB_TARGET_SOURCES",
                    "KUNO_LAB_FROZEN_PARTITION_EXIT",
                    "KUNO_LAB_TARGET_REUSE",
                ] {
                    let mut bad = valid.clone();
                    bad.retain(|(k, _)| k != key);
                    assert!(registration(&args, bad, build).is_err());
                }
            }
        }
        for raw in ["", "shared", "true", "four-public "] {
            let mut bad = base.clone();
            bad.last_mut().unwrap().1 = raw.into();
            assert!(registration(&args, bad, build).is_err());
        }
        base.push(("KUNO_LAB_FROZEN_DOWN_ROWS".into(), "four-public".into()));
        assert!(registration(&args, base, build).is_err());
    }

    #[test]
    fn private_tables_require_explicit_public_frozen_exit_and_reject_duplicates() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut selected = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
        ] {
            selected.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, selected.clone(), build)
                .unwrap()
                .frozen_private_tables
        );
        selected.push(("KUNO_LAB_FROZEN_TABLES".into(), "private-public".into()));
        for router in [None, Some("device-frozen-public")] {
            let mut valid = selected.clone();
            if let Some(router) = router {
                valid.push(("KUNO_LAB_TARGET_ROUTER".into(), router.into()));
            }
            assert!(
                registration(&args, valid, build)
                    .unwrap()
                    .frozen_private_tables
            );
        }
        for raw in ["", "private", "true", "1", "private-public "] {
            let mut bad = selected.clone();
            bad.iter_mut()
                .find(|(k, _)| k == "KUNO_LAB_FROZEN_TABLES")
                .unwrap()
                .1 = raw.into();
            assert!(registration(&args, bad, build).is_err());
        }
        for key in ["KUNO_LAB_TARGET_SOURCES", "KUNO_LAB_FROZEN_PARTITION_EXIT"] {
            let mut bad = selected.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err());
        }
        selected.push(("KUNO_LAB_FROZEN_TABLES".into(), "private-public".into()));
        assert!(registration(&args, selected, build).is_err());
    }

    #[test]
    fn partition_exit_is_explicit_original_frozen_only_and_rejects_duplicate_flags() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut selected = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
        ] {
            selected.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, selected.clone(), build)
                .unwrap()
                .frozen_partition_exit
        );
        let flag = ("KUNO_LAB_FROZEN_PARTITION_EXIT".into(), "1".into());
        selected.push(flag.clone());
        for router in [None, Some("device-frozen-public")] {
            let mut valid = selected.clone();
            if let Some(router) = router {
                valid.push(("KUNO_LAB_TARGET_ROUTER".into(), router.into()));
            }
            assert!(
                registration(&args, valid, build)
                    .unwrap()
                    .frozen_partition_exit
            );
        }
        for raw in ["", "0", "true", "01", "1 ", "-1"] {
            let mut invalid = selected.clone();
            invalid
                .iter_mut()
                .find(|(k, _)| k == "KUNO_LAB_FROZEN_PARTITION_EXIT")
                .unwrap()
                .1 = raw.into();
            assert!(registration(&args, invalid, build).is_err());
        }
        for key in [
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_DEVICE_DISPATCH",
        ] {
            let mut invalid = selected.clone();
            invalid.retain(|(k, _)| k != key);
            assert!(registration(&args, invalid, build).is_err());
        }
        for (key, value) in [
            ("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3"),
            ("KUNO_LAB_TARGET_MOE", "live"),
            ("KUNO_LAB_PROMPT", "private"),
        ] {
            let mut invalid = selected.clone();
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err());
        }
        let mut inherited = env();
        inherited.push(flag.clone());
        assert!(registration(&args, inherited, build).is_err());
        selected.push(flag);
        assert!(registration(&args, selected, build).is_err());
    }
    #[test]
    fn qsa_gpu_selection_is_explicit_and_rejects_malformed_or_duplicate_policy() {
        let args = ["lab", "intel", "daslab-language-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Threaded;
        assert!(!registration(&args, env(), build).unwrap().gpu_qsa_selector);
        let mut selected = env();
        selected.push(("KUNO_LAB_TARGET_QSA".into(), "chronological-public".into()));
        assert!(
            registration(&args, selected.clone(), build)
                .unwrap()
                .gpu_qsa_selector
        );
        selected.push(("KUNO_LAB_TARGET_QSA".into(), "chronological-public".into()));
        assert!(registration(&args, selected, build).is_err());
        for raw in ["", "1", "chronological", "CHRONOLOGICAL-PUBLIC"] {
            let mut invalid = env();
            invalid.push(("KUNO_LAB_TARGET_QSA".into(), raw.into()));
            assert!(registration(&args, invalid, build).is_err());
        }
    }

    #[test]
    fn both_compiled_language_fixtures_require_explicit_bounded_draft_selection() {
        let threaded = crate::target_lab_build::Build::Threaded;
        assert_eq!(
            registration(&args(), env(), threaded)
                .unwrap()
                .draft_profile,
            Profile::SelfContainedQ8
        );
        let mut q2 = env();
        q2.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        for fixture in ["first-code", "warm-prose"] {
            let mut values = q2.clone();
            for (key, value) in &mut values {
                if key == "KUNO_LAB_LANGUAGE_FIXTURE" {
                    *value = fixture.into();
                }
            }
            let selected = registration(&args(), values, threaded).unwrap();
            assert_eq!(selected.draft_profile, Profile::StrataQ2);
            assert_eq!(selected.mtp, PathBuf::from("/original/sidecar"));
            assert_eq!(selected.draft_profile.public_bytes(), 2_539_257_856);
            assert_eq!(
                selected.fixture,
                Fixture::parse(OsStr::new(fixture)).unwrap()
            );
        }
        let mut fused = q2.clone();
        fused.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
        assert!(registration(&args(), fused, threaded).unwrap().fused_sweeps);
        let mut inline = q2.clone();
        inline.push(("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()));
        assert_eq!(
            registration(
                &args(),
                inline.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .draft_profile,
            Profile::StrataQ2
        );
        assert!(registration(&args(), inline, threaded).is_err());
        assert!(
            registration(
                &args(),
                q2.clone(),
                crate::target_lab_build::Build::Rejected
            )
            .is_err()
        );
        for key in [
            "KUNO_LAB_LANGUAGE_PUBLIC_ONLY",
            "KUNO_LAB_LANGUAGE_FIXTURE",
            "KUNO_LAB_STATIC_CACHE_BYTES",
            "KUNO_TEST_Q3_DIR",
            "KUNO_TEST_MTP",
            "KUNO_PUBLIC_RECEIPT",
        ] {
            let missing = q2.iter().filter(|(k, _)| k != key).cloned();
            assert!(registration(&args(), missing, threaded).is_err());
        }
        for raw in [
            "",
            "q8",
            "q2",
            "auto",
            "strata-Q2",
            "strata-q2 ",
            "strata-q2\n",
        ] {
            let mut malformed = env();
            malformed.push(("KUNO_LAB_MTP_PROFILE".into(), raw.into()));
            assert!(registration(&args(), malformed, threaded).is_err());
        }
        let mut repeated = q2.clone();
        repeated.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        assert!(registration(&args(), repeated, threaded).is_err());
        for key in [
            "KUNO_LAB_LANGUAGE_PROMPT",
            "KUNO_LAB_LANGUAGE_MAX_OUTPUTS",
            "KUNO_LAB_WARM_REPLAY",
            "KUNO_TEST_OTHER",
            "KUNO_PUBLIC_MTP_REFERENCE",
        ] {
            let mut foreign = q2.clone();
            foreign.push((key.into(), "1".into()));
            assert!(registration(&args(), foreign, threaded).is_err());
        }
    }
    #[test]
    fn compiled_prompts_bind_exact_thinking_off_template_and_geometry() {
        for (f, p) in [(Fixture::FirstCode, 35), (Fixture::WarmProse, 30)] {
            assert_eq!(f.prompt_tokens(), p);
            assert_eq!(f.context(), p + 200);
            let r = f.rendered();
            assert_eq!(
                r.as_str(),
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    f.prompt()
                )
            );
            assert!(!r.contains("<|im_start|>system"));
            assert!(!r.contains("enable_thinking"));
            assert!(r.len() < 1024);
        }
        assert_eq!(TEMPLATE_REVISION.len(), 40);
        for bad in ["", "user prompt", "first-code ", "warm", "first-code\n"] {
            assert!(Fixture::parse(OsStr::new(bad)).is_err());
        }
    }
    #[test]
    fn actual_greedy_and_terminal_eos_clip_before_state_commit() {
        let e = 248046;
        assert_eq!(accepted_prefix(&[1, 2, 3], &[2, 3, 4], e), Ok(3));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[2, 4, 7], e), Ok(2));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[5, 3, 4], e), Ok(1));
        assert_eq!(accepted_prefix(&[1, e, 3], &[e, 3, 4], e), Ok(1));
        assert_eq!(accepted_prefix(&[1, 2, e], &[2, e, 4], e), Ok(2));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[2, 3, e], e), Ok(3));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[4, e, 7], e), Ok(1));
        assert_eq!(accepted_prefix(&[1], &[e], e), Ok(1));
        for (x, y) in [
            (vec![], vec![]),
            (vec![1, 2, 3, 4], vec![2, 3, 4, 5]),
            (vec![1], vec![2, 3]),
            (vec![1, VOCAB as u32], vec![2, 3]),
        ] {
            assert!(accepted_prefix(&x, &y, e).is_err());
        }
        assert!(accepted_prefix(&[1], &[2], VOCAB as u32).is_err());
    }
    #[test]
    fn prompt_teacher_cell_pairing_survives_chunk_edges_and_has_no_dummy_cell() {
        for p in [30, 35] {
            let prompt: Vec<_> = (7..p as u32 + 7).collect();
            for chunk in [1, 2, 3] {
                let mut draft = 0;
                for start in (0..p).step_by(chunk) {
                    for row in 0..chunk.min(p - start) {
                        let at = start + row;
                        let actual = teacher_input(&prompt, at, draft).unwrap();
                        if at + 1 < p {
                            assert_eq!(actual, Some(prompt[at + 1]));
                            draft += 1;
                        } else {
                            assert_eq!(actual, None);
                        }
                    }
                }
                assert_eq!(draft, p - 1);
            }
            assert!(teacher_input(&prompt, 0, 1).is_err());
            assert!(teacher_input(&prompt, p, 0).is_err());
            assert_eq!(teacher_input(&prompt, p - 1, p - 1).unwrap(), None);
        }
        assert!(teacher_input(&[0; 13], 0, 0).is_err());
        let mut wrong = [0; 30];
        wrong[29] = VOCAB as u32;
        assert!(teacher_input(&wrong, 0, 0).is_err());
    }
    #[test]
    fn repair_requires_complete_finite_vectors_and_frozen_whole_graph_gate() {
        assert_eq!(
            repair_vector(&[0.0, 1.0], &[0.005, 1.006], 2).unwrap(),
            0.006000042
        );
        assert!(repair_vector(&[0.0], &[0.0061], 1).is_err());
        assert!(repair_vector(&[1.0], &[1.007], 1).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(repair_vector(&[1.0], &[bad], 1).is_err());
            assert!(repair_vector(&[bad], &[1.0], 1).is_err());
        }
        assert!(repair_vector(&[0.0], &[0.0], HIDDEN).is_err());
        assert!(repair_vector(&[0.0], &[0.0, 0.0], 1).is_err());
    }
    #[test]
    fn transient_fixture_allocations_and_emitted_continuations_are_bounded() {
        let carry = Zeroizing::new(vec![0.0; HIDDEN]);
        let saved = Zeroizing::new(vec![0.0; 2 * VOCAB]);
        let hc = Zeroizing::new(vec![0.0; 3 * HIDDEN]);
        let ids = Zeroizing::new(Vec::<u32>::with_capacity(MAX_OUTPUTS));
        assert!(owned_fixture_bytes(&[&carry, &saved, &hc], &[&ids]).unwrap() < FIXTURE_HOST_BYTES);
        let huge = Zeroizing::new(vec![0.0; FIXTURE_HOST_BYTES / 4]);
        assert!(owned_fixture_bytes(&[&huge], &[]).is_err());
        let mut stats = Statistics::default();
        for _ in 0..66 {
            stats.add(3, 3, 1, 2, 3, false).unwrap();
        }
        stats.add(1, 1, 1, 2, 3, false).unwrap();
        assert_eq!(stats.continuations, 199);
        assert_eq!(stats.continuations + 1, MAX_OUTPUTS as u64);
        assert_eq!(stats.proposed, 132);
        assert_eq!(stats.accepted_drafts, 132);
        assert!(stats.add(4, 4, 0, 0, 0, false).is_err());
        stats.rounds = u64::MAX;
        assert!(stats.add(1, 1, 0, 0, 0, false).is_err());
    }
    #[test]
    fn registration_rejects_private_prompts_oracles_unbounded_outputs_and_foreign_experiments() {
        let threaded = crate::target_lab_build::Build::Threaded;
        assert_eq!(
            registration(&args(), env(), threaded).unwrap().fixture,
            Fixture::FirstCode
        );
        let mut prose = env();
        for (k, v) in &mut prose {
            if k == "KUNO_LAB_LANGUAGE_FIXTURE" {
                *v = "warm-prose".into();
            }
        }
        assert_eq!(
            registration(&args(), prose, threaded).unwrap().fixture,
            Fixture::WarmProse
        );
        for key in [
            "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE",
            "KUNO_LAB_WARM_REPLAY",
            "KUNO_LAB_LANGUAGE_PROMPT",
            "KUNO_LAB_LANGUAGE_MAX_OUTPUTS",
            "KUNO_LAB_TEST",
            "KUNO_TEST_OTHER",
        ] {
            let mut e = env();
            e.push((key.into(), "0".into()));
            assert!(registration(&args(), e, threaded).is_err());
        }
        for bad in ["", "auto", "First-code", "first-code ", "private text"] {
            let mut e = env();
            for (k, v) in &mut e {
                if k == "KUNO_LAB_LANGUAGE_FIXTURE" {
                    *v = bad.into();
                }
            }
            assert!(registration(&args(), e, threaded).is_err());
        }
        let mut duplicate = env();
        duplicate.push(("KUNO_LAB_LANGUAGE_FIXTURE".into(), "first-code".into()));
        assert!(registration(&args(), duplicate, threaded).is_err());
        let mut selected = env();
        selected.extend([
            ("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()),
            ("KUNO_LAB_TARGET_IO".into(), "chunk".into()),
            ("KUNO_LAB_TARGET_REUSE".into(), "request".into()),
        ]);
        assert!(
            registration(
                &args(),
                selected.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .request_reuse
        );
        assert!(registration(&args(), selected.clone(), threaded).is_err());
        selected.retain(|(k, _)| k != "KUNO_LAB_TARGET_IO");
        assert!(registration(&args(), selected, crate::target_lab_build::Build::Inline).is_err());
        assert!(registration(&args(), env(), crate::target_lab_build::Build::Rejected).is_err());
        let mut wrong = args();
        wrong[2] = "daslab-speculative-probe".into();
        assert!(registration(&wrong, env(), threaded).is_err());
    }
}

/// Final registration-only acceptance for the complete port preset. The caller
/// supplies explicitly synthetic memory fields; this never constructs a device.
pub(crate) fn strata_registration_probe() -> Result<()> {
    if !crate::strata_port::enabled() {
        return Err("port feature required".into());
    }
    let args = vec![
        OsString::from("kuno-strata-rust"),
        OsString::from("intel"),
        OsString::from("daslab-language-probe"),
    ];
    let input = registration(
        &args,
        std::env::vars_os(),
        crate::target_lab_build::Build::compiled(),
    )?;
    println!(
        "{}",
        serde_json::json!({"registration_only":true,"gpu_created":false,"memory_fields_are_synthetic":true,"draft_profile":input.draft_profile.name(),"four_lookahead":input.four_lookahead,"gu_integer":input.frozen_gu_integer,"down_integer":input.frozen_down_integer,"transient_request_io":input.request_reuse,"actual_gpu_admission_still_required":true})
    );
    Ok(())
}
