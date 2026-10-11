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
use kuno_text::{
    GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits,
    model::{Qwen4Exp, greedy},
};
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
pub(crate) const MAX_OUTPUTS: usize = 200;
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
pub(crate) struct Inputs {
    pub(crate) decode_mode: DecodeMode,
    pub(crate) inline_dispatch: bool,
    pub(crate) fused_sweeps: bool,
    draft_profile: Profile,
    mtp_device: crate::target_lab_build::MtpDevice,
    intel_mtp_memory: Option<crate::target_lab_build::IntelMtpMemory>,
    mtp_q8_head_broadcast: bool,
    mtp_q8_head_shared_scales: bool,
    mtp_english_vocab: bool,
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
    pub(crate) frozen_gu_stride_four: bool,
    pub(crate) three_gdn_streaming: bool,
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
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_TEST_MTP"
                    | "KUNO_PUBLIC_RECEIPT"
                    | "KUNO_LAB_STATIC_CACHE_BYTES"
                    | "KUNO_LAB_DEVICE_DISPATCH"
                    | "KUNO_LAB_MTP_SWEEPS"
                    | "KUNO_LAB_MTP_VOCAB"
                    | "KUNO_LAB_MTP_PROFILE"
                    | "KUNO_LAB_MTP_DEVICE"
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
                    | "KUNO_LAB_FROZEN_GU_SCHEDULE"
                    | "KUNO_LAB_TARGET_GDN"
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
        || values.remove(OsStr::new("KUNO_LAB_STATIC_CACHE_BYTES")) != Some("8589934592".into())
    {
        return Err("compiled language/cache gates missing");
    }
    let decode_mode = DecodeMode::parse(
        values
            .remove(OsStr::new("KUNO_LAB_LANGUAGE_DECODE"))
            .as_deref(),
    )?;
    if decode_mode == DecodeMode::TargetOnly
        && [
            "KUNO_LAB_MTP_SWEEPS",
            "KUNO_LAB_MTP_DEVICE",
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
    let (mtp_q8_head_broadcast, mtp_q8_head_shared_scales) =
        match values.remove(OsStr::new("KUNO_LAB_MTP_Q8_HEAD")) {
            None => (false, false),
            Some(v) if v == "broadcast-public" && draft_profile == Profile::SelfContainedQ8 => {
                (true, false)
            }
            Some(v) if v == "shared-public" && draft_profile == Profile::SelfContainedQ8 => {
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
    )?;
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
                    && draft_profile == Profile::SelfContainedQ8
                    && !mtp_q8_head_broadcast
                    && !mtp_english_vocab
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
    let raw_mtp_device = values.remove(OsStr::new("KUNO_LAB_MTP_DEVICE"));
    let mtp_device = crate::target_lab_build::MtpDevice::parse(
        raw_mtp_device.as_deref(),
        decode_mode == DecodeMode::Speculative
            && original_public_target_scope
            && fused_sweeps
            && !mtp_english_vocab,
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
    let mut path = |key: &str| {
        values
            .remove(OsStr::new(key))
            .map(PathBuf::from)
            .ok_or("language public path missing")
    };
    Ok(Inputs {
        decode_mode,
        inline_dispatch,
        fused_sweeps,
        draft_profile,
        mtp_device,
        intel_mtp_memory,
        mtp_q8_head_broadcast,
        mtp_q8_head_shared_scales,
        mtp_english_vocab,
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
        frozen_gu_stride_four,
        three_gdn_streaming,
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
    let limits = TokenizerLimits {
        max_vocabulary: VOCAB,
        max_input_bytes: 1024,
        max_output_tokens: 64,
        max_piece_bytes: 1024,
        max_merge_steps: 8192,
        max_decoded_bytes: 4096,
        max_decode_chunk_bytes: 16384,
        max_allocation_bytes: TOKENIZER_HOST_BYTES,
        ..TokenizerLimits::default()
    };
    let tokenizer = Qwen35Tokenizer::from_gguf(file, limits)?;
    if tokenizer.vocabulary_len() != VOCAB || tokenizer.eos_token_id() != eos {
        return Err("language original tokenizer identity rejected".into());
    }
    let rendered = fixture.rendered();
    let ids = Zeroizing::new(tokenizer.encode(&rendered, true)?);
    let decoded = Zeroizing::new(tokenizer.decode(&ids)?);
    if ids.len() != fixture.prompt_tokens()
        || ids.capacity() > 64
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
    if !(1..=3).contains(&inputs.len())
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
        if !r.is_finite() || !a.is_finite() || difference > ABS + REL * r.abs() {
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
        if !(1..=3).contains(&count) || accepted == 0 || accepted > count {
            return Err("language count geometry rejected".into());
        }
        fn add(n: &mut u64, v: u64) -> Result<()> {
            *n = n.checked_add(v).ok_or("language counter overflow")?;
            Ok(())
        }
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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Kuno fixed public native MTP"),
            required_features: wgpu::Features::SUBGROUP,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            ..Default::default()
        })
        .await
        .map_err(|_| "speculative draft device setup failed")?;
    let distinct_device = target_device.is_none_or(|target| target != &device);
    if !distinct_device {
        return Err("MTP cannot share the target Wgpu device".into());
    }
    let selected = init_device(
        WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: wgpu::Backend::Vulkan,
        },
        RuntimeOptions {
            tasks_max: 256,
            memory_config: crate::private_arena::required_memory_configuration(),
        },
    );
    let client = WgpuRuntime::client(&selected);
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
) -> Result<()> {
    if input.frozen_host_sources != host_device.is_some() {
        return Err("public frozen sources require exact parent device selection".into());
    }
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    validate_limits(limits)?;
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_language_failure panic=true")
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
            crate::target_lab_build::intel_mtp_preflight(
                proof,
                host_device.ok_or("Intel MTP actual target device missing")?,
                client,
                graph.bound_graph(),
                input.frozen_gu_integer,
                input.draft_profile,
            )
        })
        .transpose()?;
    crate::resident_util::select_retained_metadata(true);
    crate::resident_util::select_compute_scratch_clear(true);
    if input.request_reuse {
        client.configure_public_target_request_io()?;
    } else {
        client.configure_transient_io(128 * 1024 * 1024)?;
    }
    let draft_gpu = if draft_graph.is_some() {
        let (draft_client, draft_limits, mtp_device_receipt) = cubecl::future::block_on(
            draft_client(input.fused_sweeps, input.mtp_device, client, host_device),
        )?;
        draft_client.configure_transient_io(64 * 1024 * 1024)?;
        if draft_client.public_target_request_io_configured()? {
            return Err("language target runtime leaked to draft stream".into());
        }
        Some((draft_client, draft_limits, mtp_device_receipt))
    } else {
        None
    };
    let target_retired = Arc::new(AtomicBool::new(false));
    let draft_retired = Arc::new(AtomicBool::new(false));
    let cancel = AtomicBool::new(false);
    let whole = Instant::now();
    let load_disk_before = process_disk_read_bytes()?;
    let mtp_device_receipt = draft_gpu.as_ref().map(|(_, _, receipt)| receipt.clone());
    let mut draft = if let Some((draft_client, draft_limits, _)) = draft_gpu {
        let draft_graph = draft_graph
            .as_ref()
            .ok_or("speculative bound draft graph absent")?;
        let mut draft = if input.mtp_english_vocab {
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
    if draft.is_none() != (input.decode_mode == DecodeMode::TargetOnly) {
        return Err("language actual draft construction scope rejected".into());
    }
    let mut executor = GpuExecutor::new_with_profile(
        &file,
        client.clone(),
        8 * 1024 * 1024 * 1024,
        144,
        31 * 1024 * 1024 * 1024,
        OriginalPackedProfile::DaslabIq3,
    )?;
    executor.target_rows_reserved = true;
    executor.target_batch_router_reserved = input.frozen_host_router;
    executor.target_frozen_host_router_reserved = input.frozen_host_router;
    executor.target_frozen_host_staging_reserved = input.frozen_packed_host_staging;
    executor.target_frozen_gu_integer_reserved = input.frozen_gu_integer;
    executor.target_verify3_dense = input.verify3_dense;
    executor.target_verify3_fused = input.verify3_fused;
    executor.token_io_checkpoint = input.chunk_io;
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
        let prepared_host = std::cell::OnceCell::new();
        let mut host_attachment = None;
        let mut host_setup = None;
        let mut target = ResidentModel::new_with_native_batch(
            graph.bound_graph(),
            client.clone(),
            input.fixture.context(),
            &mut executor,
            config,
            true,
        )?;
        let runs = (|| -> Result<serde_json::Value> {
            let load_ns = elapsed(whole)?;
            let prefill = Instant::now();
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
                            confirmed = greedy(rows.logits(row)?)?;
                            carry.copy_from_slice(rows.hidden(row)?);
                        }
                    } else {
                        // Original prompt chunks still execute every target
                        // family. Read full real packets without teacher KV.
                        let predicted = greedy(rows.logits(row)?)?;
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
            let prefill_ns = elapsed(prefill)?;
            if input.frozen_host_sources {
                println!("public_frozen_host_prepare_start");
                let preparation = Instant::now();
                let (plan, fills) =
                    target.prepare_public_host_sources(&mut executor, &cancel, false)?;
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
                    if input.frozen_gu_stride_four {
                        target.attach_public_host_sources_with_gu_schedule(
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
            if input.three_gdn_streaming {
                target.set_public_three_gdn_streaming(&mut executor, &cancel)?;
            }
            let load_disk_after = process_disk_read_bytes()?;
            let before_uploads = executor.expert_uploads;
            let before_hits = executor.expert_hits;
            let before_bytes = executor.uploaded_bytes;
            let before_weight_reads = executor.weight_read_ns;
            let before_weight_uploads = executor.weight_upload_ns;
            let before_gpu_routers = executor.gpu_router_calls;
            let before_dense = executor.target_verify3_dense_calls;
            let before_fused = executor.target_verify3_fused_calls;
            let before_phases = target.phase_ns.clone();
            let before_traffic = draft.as_ref().map(|draft| draft.traffic());
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
            if confirmed != graph.eos_token() {
                let text = Zeroizing::new(decoder.push_bytes(tokenizer.token_bytes(confirmed)?)?);
                nonempty_text |= !text.is_empty();
            }
            let mut stats = Statistics::default();
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
                    confirmed = greedy(rows.logits(0)?)?;
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
                    }
                    stats.add(1, 1, 0, elapsed(verifying)?, 0, false)?;
                    continue;
                }
                let draft = draft.as_mut().ok_or("speculative actual draft absent")?;
                let count = (MAX_OUTPUTS - emitted.len()).min(3);
                let mut inputs = Zeroizing::new(Vec::with_capacity(count));
                inputs.push(confirmed);
                let drafting = Instant::now();
                if count > 1 {
                    let output = draft.step_draft(confirmed, &carry, &cancel)?;
                    if output.start() != position - 1 {
                        return Err("language first actual draft stamp rejected".into());
                    }
                    inputs.push(output.greedy());
                    drop(output);
                    if count > 2 {
                        let output = draft.follow_draft(&cancel)?;
                        if output.start() != position {
                            return Err("language second actual draft stamp rejected".into());
                        }
                        inputs.push(output.greedy());
                        drop(output);
                    }
                }
                let draft_ns = elapsed(drafting)?;
                let verifying = Instant::now();
                let mut round =
                    LatestTargetRound::begin(&mut target, &mut executor, &inputs, &cancel)?;
                let rows = round.verify(&cancel)?;
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
                    predicted.push(greedy(rows.logits(row)?)?);
                }
                let accepted = accepted_prefix(&inputs, &predicted, graph.eos_token())?;
                let mut actual_carries = Zeroizing::new(Vec::with_capacity(accepted * HIDDEN));
                let mut saved_logits = Zeroizing::new(Vec::with_capacity(if accepted < count {
                    accepted * VOCAB
                } else {
                    0
                }));
                for row in 0..accepted {
                    actual_carries.extend_from_slice(rows.hidden(row)?);
                    if accepted < count {
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
                if recomputed != (accepted < count) {
                    return Err("language actual rejection repair evidence missing".into());
                }
                if let Some(recomputed) = repaired {
                    if recomputed.start() != position
                        || recomputed.count() != accepted
                        || recomputed.binding_id() != graph.bound_graph().binding_id()
                    {
                        return Err("language recomputed actual identity rejected".into());
                    }
                    for row in 0..accepted {
                        stats.repair_logit_max = stats.repair_logit_max.max(repair_vector(
                            &saved_logits[row * VOCAB..(row + 1) * VOCAB],
                            recomputed.logits(row)?,
                            VOCAB,
                        )?);
                        stats.repair_hidden_max = stats.repair_hidden_max.max(repair_vector(
                            &actual_carries[row * HIDDEN..(row + 1) * HIDDEN],
                            recomputed.hidden(row)?,
                            HIDDEN,
                        )?);
                        if greedy(recomputed.logits(row)?)? != predicted[row] {
                            return Err("language repair greedy changed".into());
                        }
                    }
                    drop(recomputed);
                }
                if target.completed_target_rows_cursor(&executor)? != position + accepted {
                    return Err("language repaired actual cursor rejected".into());
                }
                if count > 1 {
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
                    }
                }
                stats.add(
                    count,
                    accepted,
                    draft_ns,
                    verify_ns,
                    elapsed(repairing)?,
                    recomputed,
                )?;
            }
            let tail = Zeroizing::new(decoder.finish());
            nonempty_text |= !tail.is_empty();
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
            let host_io = crate::target_lab_build::host_io_delta(
                before_host,
                crate::target_lab_build::host_io_snapshot(),
            )?;
            if stats.continuations > 0
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
            let draft_traffic = match (before_traffic, after_traffic) {
                (Some(before), Some(after)) => Some(serde_json::json!({
                    "host_read_calls":after.host_read_calls-before.host_read_calls,
                    "host_read_bytes":after.host_read_bytes-before.host_read_bytes,
                    "explicit_dispatches":after.explicit_dispatches-before.explicit_dispatches,
                    "private_upload_calls":after.private_upload_calls-before.private_upload_calls,
                    "private_upload_bytes":after.private_upload_bytes-before.private_upload_bytes,
                    "minimal_packet_reads":after.minimal_packet_reads-before.minimal_packet_reads,
                    "followup_copy_calls":after.followup_copy_calls-before.followup_copy_calls})),
                (None, None) if input.decode_mode == DecodeMode::TargetOnly => None,
                _ => return Err("language actual draft traffic scope changed".into()),
            };
            let target_only_counters =
                stats.target_only_record(input.decode_mode, draft.is_some(), emitted.len())?;
            let (actual_direct, after_fills) = target.native_public_fill_counters(&executor)?;
            let frozen_routes = if executor.target_frozen_host_source_identity.is_some() {
                let counts = target.native_frozen_source_counters(&executor)?;
                Some(serde_json::json!({"vram_routes":counts.vram_routes,
                    "host_routes":counts.host_routes,"source_dispatches":counts.source_dispatches,"staging_dispatches":counts.staging_dispatches,"staged_unique_host_groups":counts.staged_unique_host_groups,"staged_packed_bytes":counts.staged_packed_bytes,"gu_integer_preparations":counts.gu_integer_preparations,"gu_integer_dispatches":counts.gu_integer_dispatches,"gu_integer_audits":counts.gu_integer_audits,"gu_stride_four_dispatches":counts.gu_stride_four_dispatches}))
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
            let physical_read_bytes = process_disk_read_bytes()?
                .checked_sub(load_disk_after)
                .ok_or("language process read counter regressed")?;
            Ok(
                serde_json::json!({"fixture":input.fixture.name(),"prompt_tokens":prompt.len(),"context_capacity":input.fixture.context(),"compiled_prompt_template_revision":TEMPLATE_REVISION,"compiled_prompt_template_file":TEMPLATE_FILE,"load_ns":load_ns,"prefill_ns_including_real_teacher_kv":prefill_ns,"prefill_teacher_kv_executed":draft.is_some(),"prefill_first_output_count":1,"prefill_first_output_generation_pick_timed":false,"first_output_emission_timed":true,"emitted_tokens_including_eos":emitted.len(),"post_prefill_target_continuations":stats.continuations,"stop_reason":if confirmed==graph.eos_token(){"eos"}else{"limit"},"nonempty_text":nonempty_text,"elapsed_ns":elapsed_ns,"continuation_tokens_per_second":stats.continuations as f64*1e9/elapsed_ns as f64,"emitted_tokens_per_second":emitted.len()as f64*1e9/elapsed_ns as f64,"rounds":stats.rounds,"frozen_host_gpu_router":input.frozen_host_router,"frozen_host_inactive_partition_exit":frozen_partition_exit,"frozen_host_private_tables":frozen_private_tables,"frozen_host_workgroup_tables":frozen_workgroup_tables,"frozen_host_shared_down_scales":frozen_shared_down_scales,"frozen_host_shared_gate_up_factors":frozen_shared_gate_up_factors,"mtp_q8_head_broadcast":mtp_q8_head_broadcast,"mtp_q8_head_shared_scales":mtp_q8_head_shared_scales,"mtp_english_vocab":mtp_english_vocab,"mtp_english_vocab_receipt":mtp_english_vocab_receipt,"frozen_host_four_row_down":frozen_four_row_down,"frozen_host_two_row_gate_up":frozen_two_row_gate_up,"frozen_packed_host_staging":frozen_packed_host_staging,"frozen_three_member_arithmetic":frozen_three_members,"frozen_gu_integer":frozen_gu_integer,"frozen_gu_integer_counters":frozen_gu_integer_counters,"frozen_gu_stride_four":frozen_gu_stride_four,"frozen_gu_stride_four_counters":frozen_gu_stride_four_counters,"target_three_gdn_streaming":three_gdn_streaming,"target_three_gdn_completed_chunks":three_gdn_completed_chunks,"gpu_router_calls":executor.gpu_router_calls-before_gpu_routers,"proposed_drafts":stats.proposed,"accepted_drafts":stats.accepted_drafts,"draft_ns":stats.draft_ns,"verify_ns_full_packets":stats.verify_ns,"repair_ns_including_vector_checks":stats.repair_ns,"actual_target_recomputations":stats.repairs,"repair_logit_max_abs":stats.repair_logit_max,"repair_hidden_max_abs":stats.repair_hidden_max,"repair_abs_tolerance":ABS,"repair_rel_tolerance":REL,"fixture_peak_host_bytes":fixture_peak,"request_scope":request_scope,"request_close_ns_inside_timing":request_close_ns,"host_io":host_io,"target_host_phase_ns":phase_ns,"verify3_dense_calls":executor.target_verify3_dense_calls-before_dense,"verify3_fused_calls":executor.target_verify3_fused_calls-before_fused,"load_and_prefill_physical_read_bytes":load_disk_after.checked_sub(load_disk_before).ok_or("language process read counter regressed")?,"generation_physical_read_bytes":physical_read_bytes,"expert_uploads":executor.expert_uploads-before_uploads,"expert_hits":executor.expert_hits-before_hits,"uploaded_bytes":executor.uploaded_bytes-before_bytes,"weight_read_ns":executor.weight_read_ns-before_weight_reads,"weight_upload_enqueue_ns":executor.weight_upload_ns-before_weight_uploads,"target_fill_counters_before":fill_counter_record(before_fills),"target_fill_counters_after":fill_counter_record(after_fills),"frozen_source_routes":frozen_routes,"draft_executed":draft.is_some(),"target_only_counters":target_only_counters,"decode_mode":input.decode_mode.name(),"draft_traffic":draft_traffic}),
            )
        })();
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
            crate::target_lab_build::validate_frozen_gu_integer_close(
                input.frozen_gu_integer,
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
            Ok((main, sources, host_close))
        })();
        if runs.is_err() {
            println!(
                "public_language_failed_cleanup complete={}",
                u8::from(cleanup.is_ok())
            );
        }
        let mut record = runs?;
        let (main, sources, host_close) = cleanup?;
        record["target_close"] = serde_json::json!({"main_buffers":main.buffers,"main_bytes":main.bytes,"source_buffers":sources.buffers,"source_bytes":sources.bytes});
        record["frozen_host_setup"] = host_setup.into();
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
                })
            })
            .into();
        Ok(record)
    })();
    let target_close = executor.close_transient_resident();
    drop(executor);
    let draft_close = draft.as_mut().map(|draft| draft.close()).transpose();
    drop(draft);
    let draft_close = draft_close.and_then(|closed| {
        if let Some(closed) = &closed {
            input
                .draft_profile
                .validate_close_with_english(closed, input.mtp_english_vocab)?;
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
    let draft_close = draft_close.map(|closed| {
        serde_json::json!({
        "private_buffers":closed.private.buffers,"private_bytes":closed.private.bytes,
        "public_bytes":closed.public_bytes,"main_reserved":closed.main_reserved})
    });
    let draft_executed = draft_close.is_some();
    let value = serde_json::json!({"scope":if input.decode_mode == DecodeMode::TargetOnly {
        "compiled-public-language200-real-target-only-greedy"
    } else if input.mtp_device == crate::target_lab_build::MtpDevice::Intel {
        "compiled-public-language200-real-single-physical-device-independent-stream-greedy"
    } else { "compiled-public-language200-real-dual-device-greedy" },"decode_mode":input.decode_mode.name(),"draft_executed":draft_executed,"target_card":"B70","draft_card":if draft_executed {Some(if input.mtp_device == crate::target_lab_build::MtpDevice::Intel {"B70"}else{"RTX3060"})}else{None},"mtp_device_requested":if draft_executed {Some(input.mtp_device.name())}else{None},"mtp_device":mtp_device_receipt,"intel_mtp_admission":intel_mtp_admission,"draft_profile":if draft_executed{Some(input.draft_profile.name())}else{None},"gpu_chronological_qsa_selector":input.gpu_qsa_selector,"qsa_extra_runtime_info_bytes":if input.gpu_qsa_selector{crate::resident_qsa::batch::EXTRA_INFO_BYTES_PER_CHUNK}else{0},"qsa_extra_host_reserved_bytes":if input.gpu_qsa_selector{crate::resident_qsa::batch::EXTRA_HOST_BYTES_PER_CHUNK}else{0},"draft_public_reserved_bytes":if draft_executed{input.draft_profile.public_bytes_with_english(input.mtp_english_vocab)?}else{0},"frozen_gu_integer_requested":input.frozen_gu_integer,"frozen_gu_stride_four_requested":input.frozen_gu_stride_four,"frozen_gu_integer_extra_main_bytes":if input.frozen_gu_integer {65536} else {0},"frozen_gu_integer_extra_host_bytes":if input.frozen_gu_integer {65536} else {0},"frozen_gu_integer_extra_info_bytes":if input.frozen_gu_integer {196608} else {0},"frozen_gu_integer_extra_runtime_bytes":if input.frozen_gu_integer {462848} else {0},"device_dispatch":crate::target_lab_build::name(input.inline_dispatch),"observed_inline_device_dispatch":crate::target_lab_build::actual_inline(),"temperature":0,"seed":424242,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,"thinking":false,"max_outputs":MAX_OUTPUTS,"verify3_dense":input.verify3_dense,"verify3_logical_finite_fused":input.verify3_fused,"prepared_verify3_experts":input.prepared_experts,"live_member_reduction":input.live_member_reduction,"public_existing_fill":input.public_existing_fill,"chunk_io_checkpoint":input.chunk_io,"request_io_reuse":input.request_reuse,"fused_mtp_sweeps":input.fused_sweeps,"fixture_host_reserved_bytes":FIXTURE_HOST_BYTES,"tokenizer_host_cap_bytes":TOKENIZER_HOST_BYTES,"request_reuse_metadata_reserved_bytes":if input.request_reuse{4*1024*1024}else{0},"full_target_packets_retained":true,"greedy_target_verification":true,"repair_original_prefix_vector_checks":input.decode_mode == DecodeMode::Speculative,"independent_cpu200_reference_accepted":false,"forced_oracle_tokens":false,"forced_experts":false,"private_or_production_accepted":false,"record":record,"draft_close":draft_close});
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
