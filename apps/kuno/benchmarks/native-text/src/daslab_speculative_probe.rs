//! Fixed public greedy target/MTP on two original devices. No HTTP path.
//! Only real target-verified continuations enter the accepted-token count.
use crate::{
    daslab_carry_reference13 as reference, daslab_profile,
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
    GgufFile, GgufLimits,
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
pub(crate) struct Inputs {
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
    pub(crate) profile_first_warm_chunk: bool,
    pub(crate) bank_fill: crate::executor::PublicHostBankFillPolicy,
    pub(crate) target_submission_backlog: cubecl_runtime::server::PublicTargetSubmissionBacklog,
    pub(crate) frozen_gu_stride_four: bool,
    pub(crate) frozen_down_stride_four: bool,
    pub(crate) three_gdn_streaming: bool,
    pub(crate) public_grouped_ple_reads: bool,
    pub(crate) public_grouped_status_folds: bool,
    pub(crate) preserve_routed_bank: bool,
    warm_replay: bool,
    pub(crate) verify3_dense: bool,
    pub(crate) verify3_fused: bool,
    pub(crate) prepared_experts: bool,
    public_existing_fill: bool,
    chunk_io: bool,
    request_reuse: bool,
    live_member_reduction: bool,
    device_router: bool,
    directory: PathBuf,
    mtp: PathBuf,
    reference: PathBuf,
    receipt: PathBuf,
}
pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    build: crate::target_lab_build::Build,
) -> std::result::Result<Inputs, &'static str> {
    if args.len() != 3 || args[1] != "intel" || args[2] != "daslab-speculative-probe" {
        return Err("speculative lab requires fixed public B70 target");
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
                "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_TEST_MTP"
                    | "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE"
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
                    | "KUNO_LAB_TARGET_PROFILE"
                    | "KUNO_LAB_FROZEN_BANK_FILL"
                    | "KUNO_LAB_TARGET_SUBMISSION_BACKLOG"
                    | "KUNO_LAB_FROZEN_GU_SCHEDULE"
                    | "KUNO_LAB_FROZEN_DOWN_SCHEDULE"
                    | "KUNO_LAB_TARGET_GDN"
                    | "KUNO_LAB_TARGET_PLE_READS"
                    | "KUNO_LAB_TARGET_STATUS_FOLDS"
                    | "KUNO_LAB_FROZEN_BANK_PLACEMENT"
                    | "KUNO_LAB_WARM_REPLAY"
                    | "KUNO_LAB_TARGET_DENSE"
                    | "KUNO_LAB_TARGET_EXPERTS"
                    | "KUNO_LAB_TARGET_FILL"
                    | "KUNO_LAB_TARGET_IO"
                    | "KUNO_LAB_TARGET_REUSE"
                    | "KUNO_LAB_TARGET_MOE"
                    | "KUNO_LAB_TARGET_ROUTER"
            )
        ) || value.is_empty()
            || values.insert(key, value).is_some()
        {
            return Err("speculative lab rejects inherited experiments");
        }
    }
    if values.remove(OsStr::new("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY")) != Some("1".into())
        || values.remove(OsStr::new("KUNO_LAB_STATIC_CACHE_BYTES")) != Some("8589934592".into())
    {
        return Err("speculative fixed public/cache gates missing");
    }
    let raw_dispatch = values.remove(OsStr::new("KUNO_LAB_DEVICE_DISPATCH"));
    let inline_dispatch = build.selected(raw_dispatch.as_deref())?;
    let prepared_experts = match values.remove(OsStr::new("KUNO_LAB_TARGET_EXPERTS")) {
        None => false,
        Some(value) if value == "prepared-verify3" => true,
        _ => return Err("target expert policy requires canonical prepared-verify3"),
    };
    let raw_moe = values.remove(OsStr::new("KUNO_LAB_TARGET_MOE"));
    let live_member_reduction =
        crate::target_lab_build::live_member_reduction(raw_moe.as_deref(), prepared_experts)?;
    let public_existing_fill = match values.remove(OsStr::new("KUNO_LAB_TARGET_FILL")) {
        None => false,
        Some(value) if value == "direct-public" => true,
        _ => return Err("target fill policy requires canonical direct-public"),
    };
    let chunk_io = match values.remove(OsStr::new("KUNO_LAB_TARGET_IO")) {
        None => false,
        Some(value) if value == "chunk" => true,
        _ => return Err("target I/O policy requires canonical chunk"),
    };
    let raw_reuse = values.remove(OsStr::new("KUNO_LAB_TARGET_REUSE"));
    let request_reuse =
        crate::target_lab_build::request_reuse(raw_reuse.as_deref(), inline_dispatch, chunk_io)?;
    let (verify3_dense, verify3_fused) = match values.remove(OsStr::new("KUNO_LAB_TARGET_DENSE")) {
        None => (false, false),
        Some(value) if value == "verify3" => (true, false),
        Some(value) if value == "verify3-fused" => (true, true),
        _ => return Err("target dense policy requires canonical verify3 or verify3-fused"),
    };
    let warm_replay = match values.remove(OsStr::new("KUNO_LAB_WARM_REPLAY")) {
        None => false,
        Some(value) if value == "once" => true,
        _ => return Err("warm replay requires canonical once"),
    };
    let raw_router = values.remove(OsStr::new("KUNO_LAB_TARGET_ROUTER"));
    let (device_router, frozen_host_router) = match raw_router.as_deref() {
        None => (false, false),
        Some(v)
            if v == "device-warm-public" && warm_replay && request_reuse && !prepared_experts =>
        {
            (true, false)
        }
        Some(v)
            if v == "device-frozen-public" && warm_replay && request_reuse && !prepared_experts =>
        {
            (false, true)
        }
        _ => return Err("device router requires protected original warm request scope"),
    };
    let fused_sweeps = match values.remove(OsStr::new("KUNO_LAB_MTP_SWEEPS")) {
        None => false,
        Some(value) if value == "fused" => true,
        _ => return Err("MTP sweep policy requires canonical fused"),
    };
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
    let preserve_routed_bank = match values.remove(OsStr::new("KUNO_LAB_FROZEN_BANK_PLACEMENT")) {
        None => false,
        Some(v) if v == "preserve-public" && frozen_host_sources && warm_replay => true,
        _ => {
            return Err("bank placement requires canonical warm frozen public sources");
        }
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
    if frozen_host_sources
        && (!warm_replay
            || !request_reuse
            || !public_existing_fill
            || device_router
            || prepared_experts
            || live_member_reduction)
    {
        return Err("frozen sources require original public warm request scope");
    }
    if frozen_host_router && !frozen_host_sources {
        return Err("frozen device router requires immutable host source selection");
    }
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
    let raw_gu_math = values.remove(OsStr::new("KUNO_LAB_FROZEN_GU_MATH"));
    let frozen_gu_integer = crate::target_lab_build::frozen_gu_integer(
        raw_gu_math.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && fused_sweeps
            && draft_profile == Profile::SelfContainedQ8
            && !mtp_q8_head_broadcast
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let raw_gu_schedule = values.remove(OsStr::new("KUNO_LAB_FROZEN_GU_SCHEDULE"));
    let frozen_gu_stride_four = crate::target_lab_build::frozen_gu_stride_four(
        raw_gu_schedule.as_deref(),
        frozen_gu_integer,
    )?;
    let raw_down_schedule = values.remove(OsStr::new("KUNO_LAB_FROZEN_DOWN_SCHEDULE"));
    let frozen_down_stride_four = crate::target_lab_build::frozen_down_stride_four(
        raw_down_schedule.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && frozen_four_row_down
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
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let raw_mtp_device = values.remove(OsStr::new("KUNO_LAB_MTP_DEVICE"));
    let mtp_device = crate::target_lab_build::MtpDevice::parse(
        raw_mtp_device.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && fused_sweeps
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
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
        mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && draft_profile == Profile::SelfContainedQ8
            && !mtp_q8_head_broadcast
            && !mtp_english_vocab
            && frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let raw_down_math = values.remove(OsStr::new("KUNO_LAB_FROZEN_DOWN_MATH"));
    let frozen_down_integer = crate::target_lab_build::frozen_down_integer(
        raw_down_math.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction
            && fused_sweeps
            && draft_profile == Profile::SelfContainedQ8
            && !mtp_q8_head_broadcast
            && !frozen_four_row_down
            && !frozen_down_stride_four,
    )?;
    let raw_iq4_decoder = values.remove(OsStr::new("KUNO_LAB_TARGET_IQ4_DECODER"));
    let public_iq4_decoder = crate::target_lab_build::public_iq4_decoder(
        raw_iq4_decoder.as_deref(),
        mtp_device == crate::target_lab_build::MtpDevice::Intel
            && frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && fused_sweeps
            && draft_profile == Profile::SelfContainedQ8
            && !mtp_q8_head_broadcast
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction
            && !frozen_down_stride_four,
    )?;
    let raw_bank_fill = values.remove(OsStr::new("KUNO_LAB_FROZEN_BANK_FILL"));
    let bank_fill = crate::target_lab_build::public_host_bank_fill(
        raw_bank_fill.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && draft_profile == Profile::SelfContainedQ8
            && !mtp_q8_head_broadcast
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    if mtp_batch_repair && bank_fill != crate::executor::PublicHostBankFillPolicy::Original {
        return Err("MTP batch repair requires original public bank fill");
    }
    let raw_backlog = values.remove(OsStr::new("KUNO_LAB_TARGET_SUBMISSION_BACKLOG"));
    let target_submission_backlog = crate::target_lab_build::public_target_submission_backlog(
        raw_backlog.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && mtp_device == crate::target_lab_build::MtpDevice::Intel
            && fused_sweeps
            && draft_profile == Profile::SelfContainedQ8
            && !mtp_q8_head_broadcast
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let raw_ple_reads = values.remove(OsStr::new("KUNO_LAB_TARGET_PLE_READS"));
    let public_grouped_ple_reads = crate::target_lab_build::public_grouped_ple_reads(
        raw_ple_reads.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let raw_status_folds = values.remove(OsStr::new("KUNO_LAB_TARGET_STATUS_FOLDS"));
    let public_grouped_status_folds = crate::target_lab_build::public_grouped_status_folds(
        raw_status_folds.as_deref(),
        frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && !mtp_english_vocab
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let raw_lookahead = values.remove(OsStr::new("KUNO_LAB_TARGET_LOOKAHEAD"));
    let four_lookahead = crate::target_lab_build::public_four_lookahead(
        raw_lookahead.as_deref(),
        mtp_device == crate::target_lab_build::MtpDevice::Intel
            && frozen_gu_integer
            && frozen_down_integer
            && fused_sweeps
            && draft_profile == Profile::SelfContainedQ8
            && !mtp_q8_head_broadcast
            && !mtp_english_vocab
            && (!mtp_batch_repair || public_grouped_ple_reads)
            && !public_iq4_decoder
            && !three_gdn_streaming
            && bank_fill == crate::executor::PublicHostBankFillPolicy::Original
            && frozen_host_sources
            && frozen_host_router
            && frozen_partition_exit
            && verify3_dense
            && verify3_fused
            && request_reuse
            && chunk_io
            && inline_dispatch
            && public_existing_fill
            && warm_replay
            && !frozen_private_tables
            && !frozen_workgroup_tables
            && !frozen_shared_down_scales
            && !frozen_shared_gate_up_factors
            && !frozen_two_row_gate_up
            && !frozen_packed_host_staging
            && !frozen_three_members
            && !preserve_routed_bank
            && !prepared_experts
            && !live_member_reduction,
    )?;
    let integer_verify_head_scope = frozen_host_sources
        && frozen_host_router
        && frozen_partition_exit
        && verify3_dense
        && verify3_fused
        && request_reuse
        && chunk_io
        && inline_dispatch
        && public_existing_fill
        && warm_replay
        && !frozen_private_tables
        && !frozen_workgroup_tables
        && !frozen_shared_down_scales
        && !frozen_shared_gate_up_factors
        && !frozen_two_row_gate_up
        && !frozen_packed_host_staging
        && !frozen_three_members
        && !preserve_routed_bank
        && !prepared_experts
        && !live_member_reduction
        && mtp_device == crate::target_lab_build::MtpDevice::Intel
        && fused_sweeps
        && draft_profile == Profile::SelfContainedQ8
        && !mtp_q8_head_broadcast
        && !mtp_english_vocab
        && frozen_gu_integer
        && frozen_down_integer
        && public_grouped_ple_reads
        && !public_iq4_decoder
        && bank_fill == crate::executor::PublicHostBankFillPolicy::Original;
    let raw_head = values.remove(OsStr::new("KUNO_LAB_TARGET_HEAD"));
    let public_integer_verify_head = crate::target_lab_build::public_integer_verify_head(
        raw_head.as_deref(),
        integer_verify_head_scope,
    )?;
    let profile_first_warm_chunk = match values.remove(OsStr::new("KUNO_LAB_TARGET_PROFILE")) {
        None => false,
        Some(v) if v == "first-warm-chunk" && integer_verify_head_scope => true,
        _ => return Err("target profiling requires the fixed public warm integer scope"),
    };
    let mut path = |name: &str| {
        values
            .remove(OsStr::new(name))
            .map(PathBuf::from)
            .ok_or("speculative public path missing")
    };
    Ok(Inputs {
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
        profile_first_warm_chunk,
        bank_fill,
        target_submission_backlog,
        frozen_gu_stride_four,
        frozen_down_stride_four,
        three_gdn_streaming,
        public_grouped_ple_reads,
        public_grouped_status_folds,
        preserve_routed_bank,
        warm_replay,
        verify3_dense,
        verify3_fused,
        prepared_experts,
        public_existing_fill,
        chunk_io,
        request_reuse,
        live_member_reduction,
        device_router,
        directory: path("KUNO_TEST_Q3_DIR")?,
        mtp: path("KUNO_TEST_MTP")?,
        reference: path("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE")?,
        receipt: path("KUNO_PUBLIC_RECEIPT")?,
    })
}

fn first_warm_profile_round(selected: bool, replay: usize, completed_rounds: usize) -> bool {
    selected && replay == 1 && completed_rounds == 0
}

/// Pause even when verification fails. The closure touches only the target's
/// already configured profiler; it cannot enable a disabled logger.
struct TargetProfileGuard<F: Fn(bool)>(F);
impl<F: Fn(bool)> TargetProfileGuard<F> {
    fn new(set_paused: F) -> Self {
        set_paused(false);
        Self(set_paused)
    }
}
impl<F: Fn(bool)> Drop for TargetProfileGuard<F> {
    fn drop(&mut self) {
        (self.0)(true);
    }
}

fn target_profile_record(
    selected: bool,
    completed_rounds: u64,
    method: cubecl_common::profile::TimingMethod,
) -> std::result::Result<serde_json::Value, &'static str> {
    if completed_rounds != u64::from(selected)
        || (selected && method != cubecl_common::profile::TimingMethod::Device)
    {
        return Err("target profiling actual scope/timing method rejected");
    }
    Ok(serde_json::json!({
        "selected":selected,
        "scope":if selected {"first-warm-target-round-verify-only"} else {"disabled"},
        "completed_target_rounds":completed_rounds,
        "timing_method":method.to_string(),
        "diagnostic_only":selected,
        "normal_throughput_comparable":!selected,
    }))
}

/// All rows are real target outputs. A first mismatch contributes only the
/// target correction; later provisional tokens are discarded.
fn accepted_prefix(inputs: &[u32], predicted: &[u32]) -> std::result::Result<usize, &'static str> {
    if !(1..=3).contains(&inputs.len())
        || predicted.len() != inputs.len()
        || inputs.iter().chain(predicted).any(|&t| t >= 248320)
    {
        return Err("speculative acceptance geometry");
    }
    for row in 1..inputs.len() {
        if inputs[row] != predicted[row - 1] {
            return Ok(row);
        }
    }
    Ok(inputs.len())
}

async fn draft_client(
    fused_sweeps: bool,
    profile_first_warm_chunk: bool,
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
            tasks_max: 256,
            memory_config: crate::private_arena::required_memory_configuration(),
        },
    );
    let client = WgpuRuntime::client(&selected);
    if profile_first_warm_chunk {
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
) -> Result<()> {
    if input.frozen_host_sources != host_device.is_some() {
        return Err("public frozen sources require exact parent device selection".into());
    }
    if input.profile_first_warm_chunk {
        if client.properties().timing_method != cubecl_common::profile::TimingMethod::Device {
            return Err("target profiling requires actual device timestamps".into());
        }
        client.set_profiling_paused(true);
    }
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_speculative_failure panic=true")
    }));
    let expected = reference::Reference::load(&input.reference)?;
    let paths: Vec<_> = (1..=2)
        .map(|i| {
            input.directory.join(format!(
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    if file.file_len() != 83_617_662_656 {
        return Err("original DAS extent rejected".into());
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.dtype().gguf_id())),
    )
    .map_err(|_| "original DAS inventory rejected")?;
    let sidecar = GgufFile::open(&input.mtp, GgufLimits::default())?;
    input.draft_profile.validate_sidecar(&sidecar)?;
    // SAFETY: exclusive protected runner verifies immutable static artifacts.
    unsafe {
        file.enable_mapped_weights()?;
        sidecar.enable_mapped_weights()?;
    }
    let graph = Qwen4Exp::bind(&file)?;
    let draft_graph = input.draft_profile.bind(&sidecar, graph.bound_graph())?;
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
    let (draft_client, draft_limits, mtp_device_receipt) = cubecl::future::block_on(draft_client(
        input.fused_sweeps,
        input.profile_first_warm_chunk,
        input.mtp_device,
        client,
        host_device,
    ))?;
    draft_client.configure_transient_io(64 * 1024 * 1024)?;
    if draft_client.public_target_request_io_configured()? {
        return Err("target request configuration leaked to draft stream".into());
    }
    let draft_target_backlog_query_rejected =
        Some(draft_client.public_target_submission_backlog().is_err());
    if draft_target_backlog_query_rejected != Some(true) {
        return Err("speculative draft stream acquired target backlog policy".into());
    }
    let target_submission_backlog_receipt =
        crate::target_lab_build::public_target_submission_backlog_record(
            input.target_submission_backlog,
            actual_target_backlog,
            input.request_reuse,
            draft_target_backlog_query_rejected,
        )?;
    let target_retired = Arc::new(AtomicBool::new(false));
    let draft_retired = Arc::new(AtomicBool::new(false));
    let cancel = AtomicBool::new(false);
    let load_disk_before = process_disk_read_bytes()?;
    let loading = Instant::now();
    let mut draft = if input.mtp_english_vocab {
        crate::resident_mtp::ResidentMtp::new_public_english_with_fused_sweeps(
            &draft_graph,
            draft_client,
            &draft_limits,
            draft_retired.clone(),
        )?
    } else {
        input.draft_profile.resident(
            &draft_graph,
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
    let mut executor = GpuExecutor::new_with_profile(
        &file,
        client.clone(),
        8 * 1024 * 1024 * 1024,
        144,
        31 * 1024 * 1024 * 1024,
        OriginalPackedProfile::DaslabIq3,
    )?;
    executor.target_rows_reserved = true;
    executor.target_batch_router_reserved = input.device_router || input.frozen_host_router;
    executor.target_frozen_host_router_reserved = input.frozen_host_router;
    executor.target_frozen_host_staging_reserved = input.frozen_packed_host_staging;
    executor.target_frozen_gu_integer_reserved = input.frozen_gu_integer;
    executor.target_frozen_down_integer_reserved = input.frozen_down_integer;
    executor.target_verify3_dense = input.verify3_dense;
    executor.target_verify3_fused = input.verify3_fused;
    executor.token_io_checkpoint = input.chunk_io;
    if input.four_lookahead {
        executor.reserve_public_four_rows()?;
    }
    executor.enable_transient_resident(target_retired.clone())?;
    executor.fast_dense = true;
    executor.fast_dense_fault = true;
    executor.shared_kernels = true;
    executor.shared_gdn = true;
    let mut passes = Vec::with_capacity(if input.warm_replay { 2 } else { 1 });
    let mut warm_reset = None;
    let attempt = (|| -> Result<()> {
        // Keep one healthy admission. The optional one-shot reset retains its
        // ledger/epochs/public banks and audits every actual private owner.
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
        if input.request_reuse {
            config = config.reserve_public_target_request_io()?;
        }
        if input.live_member_reduction {
            config = config.with_live_member_reduction()?;
        }
        if input.public_existing_fill {
            config = config.with_public_existing_expert_fill()?;
        }
        if input.warm_replay {
            config = config.reserve_public_warm_replay()?;
        }
        if input.device_router {
            config = config.reserve_public_warm_batch_router()?;
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
        if input.public_grouped_ple_reads {
            config = config.reserve_public_grouped_ple_reads()?;
        }
        if input.public_grouped_status_folds {
            config = config.reserve_public_grouped_status_folds()?;
        }
        if input.four_lookahead {
            config = config.reserve_public_four_rows()?;
        }
        let prepared_host = std::cell::OnceCell::new();
        let mut host_attachment = None;
        let mut host_setup = None;
        let mut bank_fill_setup_experts = None;
        let mut target = ResidentModel::new_with_native_batch(
            graph.bound_graph(),
            client.clone(),
            reference::CONTEXT,
            &mut executor,
            config,
            true,
        )?;
        let runs = (|| -> Result<()> {
            for replay in 0..if input.warm_replay { 2 } else { 1 } {
                println!("public_speculative_pass_start replay={replay}");
                if input.device_router && replay == 1 {
                    target.activate_public_warm_batch_router(&executor, &cancel)?;
                }
                let replay_disk_before = if replay == 0 {
                    load_disk_before
                } else {
                    process_disk_read_bytes()?
                };
                let replay_prefill = Instant::now();
                let pass = (|| -> Result<serde_json::Value> {
                    let before_prefill_repair = draft.traffic();
                    let initial = target.prefill_chunk_with_carry(&mut executor, &[0], &cancel)?;
                    if initial.start() != 0
                        || initial.count() != 1
                        || initial.binding_id() != graph.bound_graph().binding_id()
                    {
                        return Err("speculative actual initial range rejected".into());
                    }
                    expected.verify(0, initial.logits(0)?, initial.hidden(0)?)?;
                    println!("public_speculative_initial_prefill_accepted replay={replay}");
                    let mut confirmed = greedy(initial.logits(0)?)?;
                    let mut carry = Zeroizing::new(initial.hidden(0)?.to_vec());
                    drop(initial);
                    if target.completed_target_rows_cursor(&executor)? != 1 || draft.position() != 0
                    {
                        return Err("speculative initial real cursor rejected".into());
                    }
                    let expected_bank_fill = if replay == 1 {
                        input.bank_fill
                    } else {
                        crate::executor::PublicHostBankFillPolicy::Original
                    };
                    let initial_bank_fill = target.native_frozen_source_bank_fill(&executor)?;
                    let initial_prefill_bank_fill_receipt =
                        crate::target_lab_build::public_host_bank_fill_record(
                            expected_bank_fill,
                            initial_bank_fill,
                            bank_fill_setup_experts,
                        )?;
                    let (initial_iq4_decoder, initial_iq4_dispatches) =
                        executor.public_target_iq4_decoder()?;
                    let initial_prefill_target_iq4_decoder_receipt =
                        crate::target_lab_build::public_iq4_decoder_record(
                            input.public_iq4_decoder && replay == 1,
                            initial_iq4_decoder,
                            initial_iq4_dispatches,
                            executor.target_verify3_dense_calls,
                        )?;
                    let initial_prefill_frozen_gu_integer =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_gu_integer(&executor)?
                        } else {
                            false
                        };
                    let (initial_gu_preparations, initial_gu_dispatches, initial_gu_audits) =
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
                    let initial_prefill_frozen_gu_integer_counters =
                        crate::target_lab_build::frozen_gu_integer_initial_record(
                            input.frozen_gu_integer && replay == 1,
                            initial_prefill_frozen_gu_integer,
                            initial_gu_preparations,
                            initial_gu_dispatches,
                            initial_gu_audits,
                        )?;
                    let initial_prefill_frozen_gu_stride_four =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_gu_stride_four(&executor)?
                        } else {
                            false
                        };
                    let initial_gu_stride_dispatches =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target
                                .native_frozen_source_counters(&executor)?
                                .gu_stride_four_dispatches
                        } else {
                            0
                        };
                    let initial_prefill_frozen_gu_stride_four_counters =
                        crate::target_lab_build::frozen_gu_stride_four_record(
                            input.frozen_gu_stride_four && replay == 1,
                            initial_prefill_frozen_gu_stride_four,
                            initial_prefill_frozen_gu_integer,
                            initial_gu_dispatches,
                            initial_gu_stride_dispatches,
                        )?;
                    let initial_prefill_frozen_down_stride_four =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_down_stride_four(&executor)?
                        } else {
                            false
                        };
                    let (
                        initial_down_layers,
                        initial_down_dispatches,
                        initial_down_source_dispatches,
                    ) = if executor.target_frozen_host_source_identity.is_some() {
                        let counts = target.native_frozen_source_counters(&executor)?;
                        (
                            counts.down_stride_four_layers,
                            counts.down_stride_four_dispatches,
                            counts.source_dispatches,
                        )
                    } else {
                        (0, 0, 0)
                    };
                    let initial_prefill_frozen_down_stride_four_counters =
                        crate::target_lab_build::frozen_down_stride_four_initial_record(
                            input.frozen_down_stride_four && replay == 1,
                            initial_prefill_frozen_down_stride_four,
                            initial_down_layers,
                            initial_down_dispatches,
                            initial_down_source_dispatches,
                        )?;
                    if target.native_live_member_reduction(&executor)?
                        != input.live_member_reduction
                    {
                        return Err("target initial live reduction geometry mismatch".into());
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
                            input.frozen_down_integer && replay == 1,
                            initial_prefill_frozen_down_integer,
                            initial_down_integer_preparations,
                            initial_down_integer_dispatches,
                            initial_down_integer_audits,
                            initial_down_integer_sources,
                        )?;
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
                            before_prefill_repair,
                            draft.traffic(),
                            0,
                            0,
                        )?;
                    let initial_prefill_mtp_fused_four_repair =
                        crate::target_lab_build::FourMtpRepairStatistics::default().record(
                            false,
                            initial_four_rows,
                            &crate::target_lab_build::LookaheadStatistics::default(),
                            before_prefill_repair,
                            draft.traffic(),
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
                    let selected_four_lookahead = input.four_lookahead && replay == 1;
                    if selected_four_lookahead {
                        target.set_public_four_rows(&mut executor, &cancel)?;
                    }
                    if input.three_gdn_streaming && replay == 1 {
                        target.set_public_three_gdn_streaming(&mut executor, &cancel)?;
                    }
                    if input.public_grouped_ple_reads && replay == 1 {
                        target.set_public_grouped_ple_reads(&executor, true, &cancel)?;
                    }
                    if input.public_grouped_status_folds && replay == 1 {
                        target.set_public_grouped_status_folds(&executor, true, &cancel)?;
                    }
                    let selected_integer_head = input.public_integer_verify_head && replay == 1;
                    if selected_integer_head {
                        target.activate_public_q6_integer_verify_head(&mut executor, &cancel)?;
                    }
                    let mut integer_head_stats =
                        crate::target_lab_build::IntegerVerifyHeadStatistics::default();
                    let load_ns = loading.elapsed().as_nanos() as u64;
                    let initial_prefill_ns = replay_prefill.elapsed().as_nanos() as u64;
                    let mut accepted_tokens = 0usize;
                    let mut proposed_tokens = 0usize;
                    let mut accepted_drafts = 0usize;
                    let mut rounds = Vec::new();
                    let mut profiled_target_rounds = 0u64;
                    let mut target_recomputations = 0u64;
                    let before_traffic = draft.traffic();
                    let selected_mtp_batch_repair = input.mtp_batch_repair && replay == 1;
                    let mut mtp_batch_repair_cells = 0u64;
                    let selected_four_batch_repair =
                        selected_four_lookahead && selected_mtp_batch_repair;
                    let mut four_batch_repair =
                        crate::target_lab_build::FourMtpRepairStatistics::default();
                    let before_uploads = executor.expert_uploads;
                    let before_hits = executor.expert_hits;
                    let before_gpu_routers = executor.gpu_router_calls;
                    let before_bytes = executor.uploaded_bytes;
                    let mut lookahead_stats =
                        crate::target_lab_build::LookaheadStatistics::default();
                    let before_four_dense = executor.target_verify4_dense_calls;
                    let before_four_fused = executor.target_verify4_fused_calls;
                    let before_dense = executor.target_verify3_dense_calls;
                    let before_fused = executor.target_verify3_fused_calls;
                    let load_disk_after = process_disk_read_bytes()?;
                    let before_weight_reads = executor.weight_read_ns;
                    let before_weight_uploads = executor.weight_upload_ns;
                    let before_phases = target.phase_ns.clone();
                    let (actual_direct, before_fills) =
                        target.native_public_fill_counters(&executor)?;
                    if actual_direct != input.public_existing_fill {
                        return Err("public fill frozen selection rejected".into());
                    }
                    let before_host_io = crate::target_lab_build::host_io_snapshot();
                    let start = Instant::now();
                    if input.request_reuse {
                        target.begin_public_target_request_io(&mut executor, &cancel)?;
                        if !target.public_target_request_io_active()? {
                            return Err("public target request scope did not activate".into());
                        }
                    }
                    while target.completed_target_rows_cursor(&executor)? < reference::POSITIONS {
                        let position = target.completed_target_rows_cursor(&executor)?;
                        let count = (reference::POSITIONS - position)
                            .min(if selected_four_lookahead { 4 } else { 3 });
                        if confirmed != expected.input_token(position)?
                            || draft.position() != position - 1
                        {
                            return Err("speculative actual input/cell alignment rejected".into());
                        }
                        let round_start = Instant::now();
                        let mut inputs = Zeroizing::new(vec![confirmed]);
                        let drafting = Instant::now();
                        if count > 1 {
                            let output = draft.step_draft(confirmed, &carry, &cancel)?;
                            if output.start() != position - 1 {
                                return Err("speculative first draft cell rejected".into());
                            }
                            inputs.push(output.greedy());
                            drop(output);
                            if count > 2 {
                                let output = draft.follow_draft(&cancel)?;
                                if output.start() != position {
                                    return Err("speculative second draft cell rejected".into());
                                }
                                inputs.push(output.greedy());
                                drop(output);
                            }
                        }
                        let mut third_follow_ns = 0;
                        if count == 4 {
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
                        let draft_ns = drafting.elapsed().as_nanos() as u64;
                        let verifying = Instant::now();
                        let mut round =
                            LatestTargetRound::begin(&mut target, &mut executor, &inputs, &cancel)?;
                        let profile_this_round = first_warm_profile_round(
                            input.profile_first_warm_chunk,
                            replay,
                            rounds.len(),
                        );
                        let profile_guard = profile_this_round.then(|| {
                            TargetProfileGuard::new(|paused| client.set_profiling_paused(paused))
                        });
                        let rows = round.verify(&cancel)?;
                        drop(profile_guard);
                        if profile_this_round {
                            profiled_target_rounds = profiled_target_rounds
                                .checked_add(1)
                                .ok_or("target profiling round count overflow")?;
                            // Flush the configured Basic summary while paused;
                            // neither vector checks nor repairs are profiled.
                            cubecl::future::block_on(client.sync())?;
                        }
                        if rows.start() != position
                            || rows.count() != count
                            || rows.binding_id() != graph.bound_graph().binding_id()
                        {
                            return Err("speculative verification identity rejected".into());
                        }
                        let mut predicted = Zeroizing::new(Vec::with_capacity(count));
                        for row in 0..count {
                            predicted.push(greedy(rows.logits(row)?)?);
                        }
                        let accepted = if count == 4 {
                            crate::target_lab_build::public_four_accepted_prefix(
                                &inputs, &predicted, 248320, None,
                            )?
                        } else {
                            accepted_prefix(&inputs, &predicted)?
                        };
                        let mut real_carries =
                            Zeroizing::new(Vec::with_capacity(accepted * reference::HIDDEN));
                        for row in 0..accepted {
                            expected.verify(
                                position + row,
                                rows.logits(row)?,
                                rows.hidden(row)?,
                            )?;
                            real_carries.extend_from_slice(rows.hidden(row)?);
                        }
                        let next_confirmed = predicted[accepted - 1];
                        drop(rows);
                        let verify_ns = verifying.elapsed().as_nanos() as u64;
                        let repair = Instant::now();
                        let before_recomputations = target_recomputations;
                        if let Some(recomputed) = round.finish(accepted, &cancel)? {
                            target_recomputations = target_recomputations
                                .checked_add(1)
                                .ok_or("grouped PLE actual repair count overflow")?;
                            if recomputed.start() != position || recomputed.count() != accepted {
                                return Err("speculative repaired range rejected".into());
                            }
                            for row in 0..accepted {
                                expected.verify(
                                    position + row,
                                    recomputed.logits(row)?,
                                    recomputed.hidden(row)?,
                                )?;
                            }
                            drop(recomputed);
                        }
                        if target.completed_target_rows_cursor(&executor)? != position + accepted {
                            return Err("speculative repaired actual cursor rejected".into());
                        }
                        if selected_mtp_batch_repair {
                            let repair_before = draft.traffic();
                            if selected_four_batch_repair && count == 4 {
                                draft.repair_accepted_cache_fused_four(
                                    position,
                                    &inputs[..accepted],
                                    &carry,
                                    &real_carries,
                                    &cancel,
                                )?;
                            } else {
                                draft.repair_accepted_cache(
                                    position,
                                    &inputs[..accepted],
                                    &carry,
                                    &real_carries,
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
                                    .ok_or("MTP batch repair appended cell count overflow")?;
                            }
                        } else if count > 1 {
                            // First cell used real target carry. Keep that exact KV;
                            // replace only subsequent provisional cells with reality.
                            draft.rewind(position)?;
                            for row in 1..accepted {
                                draft.append_kv(
                                    inputs[row],
                                    &real_carries
                                        [(row - 1) * reference::HIDDEN..row * reference::HIDDEN],
                                    &cancel,
                                )?;
                            }
                        } else {
                            draft.append_kv(confirmed, &carry, &cancel)?;
                        }
                        if draft.position() != position + accepted - 1 {
                            return Err("speculative corrected draft cursor rejected".into());
                        }
                        carry.copy_from_slice(
                            &real_carries
                                [(accepted - 1) * reference::HIDDEN..accepted * reference::HIDDEN],
                        );
                        confirmed = next_confirmed;
                        accepted_tokens += accepted;
                        proposed_tokens += count - 1;
                        accepted_drafts += accepted - 1;
                        let repair_ns = u64::try_from(repair.elapsed().as_nanos())?;
                        if selected_integer_head {
                            integer_head_stats.add(
                                count,
                                accepted,
                                target_recomputations != before_recomputations,
                            )?;
                        }
                        lookahead_stats.add(
                            selected_four_lookahead,
                            count,
                            accepted,
                            crate::target_lab_build::LookaheadPhase {
                                draft_ns,
                                third_follow_ns,
                                verify_ns,
                                repair_ns,
                            },
                        )?;
                        rounds.push(serde_json::json!({"target_start":position,"proposed_drafts":count-1,"accepted_drafts":accepted-1,"accepted_outputs":accepted,"draft_ns":draft_ns,"verify_ns_including_public_vector_checks":verify_ns,"repair_ns":repair_ns,"third_follow_ns":third_follow_ns,"total_ns":round_start.elapsed().as_nanos() as u64,"target_completed_cursor":position+accepted,"draft_completed_cell":draft.position()}));
                    }
                    if target.native_live_member_reduction(&executor)?
                        != input.live_member_reduction
                    {
                        return Err("target completed live reduction geometry mismatch".into());
                    }
                    let request_close = Instant::now();
                    let mut request_scope_receipt = None;
                    if input.request_reuse {
                        // Full128MiB owning I/O audit is part of throughput,
                        // before any public warm reset or final model close.
                        request_scope_receipt = Some(crate::target_lab_build::reuse_record(
                            target.end_public_target_request_io(&mut executor, &cancel)?,
                        ));
                        if target.public_target_request_io_active()? {
                            return Err("public target request scope did not close".into());
                        }
                    }
                    let request_close_ns = if input.request_reuse {
                        request_close.elapsed().as_nanos() as u64
                    } else {
                        0
                    };
                    let elapsed_ns = start.elapsed().as_nanos() as u64;
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
                    let four_lookahead_receipt = lookahead_stats.record_with_mtp_repair(
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
                    )?;
                    let host_io = crate::target_lab_build::host_io_delta(
                        before_host_io,
                        crate::target_lab_build::host_io_snapshot(),
                    )?;
                    if accepted_tokens != 12 {
                        return Err("speculative actual accepted token count rejected".into());
                    }
                    if !selected_four_lookahead
                        && input.verify3_dense
                        && executor.target_verify3_dense_calls == before_dense
                    {
                        return Err("target Verify3 actual dispatch evidence missing".into());
                    }
                    if !selected_four_lookahead
                        && input.verify3_fused
                        && executor.target_verify3_fused_calls == before_fused
                    {
                        return Err("target fused Verify3 actual dispatch evidence missing".into());
                    }
                    let host_phase_ns: BTreeMap<_, _> = target
                        .phase_ns
                        .iter()
                        .map(|(name, value)| {
                            (*name, value - before_phases.get(name).copied().unwrap_or(0))
                        })
                        .collect();
                    let timed_physical_read_bytes = process_disk_read_bytes()?
                        .checked_sub(load_disk_after)
                        .ok_or("public process read counter regressed")?;
                    let after = draft.traffic();
                    let mtp_fused_four_repair_receipt = four_batch_repair.record(
                        selected_four_batch_repair,
                        actual_four_rows,
                        &lookahead_stats,
                        before_traffic,
                        after,
                    )?;
                    let mtp_batch_repair_receipt = if selected_four_batch_repair {
                        mtp_fused_four_repair_receipt["original_batched_tails"].clone()
                    } else {
                        crate::target_lab_build::public_mtp_batch_repair_traffic_record(
                            selected_mtp_batch_repair,
                            before_traffic,
                            after,
                            if selected_mtp_batch_repair {
                                rounds.len() as u64
                            } else {
                                0
                            },
                            mtp_batch_repair_cells,
                        )?
                    };
                    let (actual_direct, after_fills) =
                        target.native_public_fill_counters(&executor)?;
                    let frozen_routes = if executor.target_frozen_host_source_identity.is_some() {
                        let counts = target.native_frozen_source_counters(&executor)?;
                        // Ranked filler preserves the cold residents, so this
                        // candidate can be all-hit. The unchanged original
                        // diagnostic still requires actual HOST and VRAM work.
                        crate::target_lab_build::public_host_bank_fill_tiers(
                            expected_bank_fill,
                            input.preserve_routed_bank,
                            counts.vram_routes,
                            counts.host_routes,
                        )?;
                        Some(serde_json::json!({"vram_routes":counts.vram_routes,
                            "host_routes":counts.host_routes,"source_dispatches":counts.source_dispatches,"staging_dispatches":counts.staging_dispatches,"staged_unique_host_groups":counts.staged_unique_host_groups,"staged_packed_bytes":counts.staged_packed_bytes,"gu_integer_preparations":counts.gu_integer_preparations,"gu_integer_dispatches":counts.gu_integer_dispatches,"gu_integer_audits":counts.gu_integer_audits,"gu_stride_four_dispatches":counts.gu_stride_four_dispatches,"down_stride_four_layers":counts.down_stride_four_layers,"down_stride_four_dispatches":counts.down_stride_four_dispatches}))
                    } else {
                        None
                    };
                    let frozen_gu_integer = if executor.target_frozen_host_source_identity.is_some()
                    {
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
                    let frozen_gu_integer_counters =
                        crate::target_lab_build::frozen_gu_integer_record(
                            input.frozen_gu_integer && replay == 1,
                            frozen_gu_integer,
                            gu_preparations,
                            gu_dispatches,
                            gu_audits,
                        )?;
                    let frozen_gu_stride_four =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_gu_stride_four(&executor)?
                        } else {
                            false
                        };
                    let gu_stride_dispatches =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target
                                .native_frozen_source_counters(&executor)?
                                .gu_stride_four_dispatches
                        } else {
                            0
                        };
                    let frozen_gu_stride_four_counters =
                        crate::target_lab_build::frozen_gu_stride_four_record(
                            input.frozen_gu_stride_four && replay == 1,
                            frozen_gu_stride_four,
                            frozen_gu_integer,
                            gu_dispatches,
                            gu_stride_dispatches,
                        )?;
                    let frozen_down_stride_four =
                        if executor.target_frozen_host_source_identity.is_some() {
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
                            input.frozen_down_stride_four && replay == 1,
                            frozen_down_stride_four,
                            down_layers,
                            down_dispatches,
                            down_source_dispatches,
                        )?;
                    let (public_grouped_ple_reads, grouped_ple_completed_chunks) =
                        target.completed_public_grouped_ple_reads(&executor)?;
                    let grouped_ple_reads_receipt =
                        crate::target_lab_build::public_grouped_ple_reads_record(
                            input.public_grouped_ple_reads && replay == 1,
                            public_grouped_ple_reads,
                            grouped_ple_completed_chunks,
                            u64::try_from(rounds.len())?,
                            target_recomputations,
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
                    let frozen_down_integer_counters =
                        crate::target_lab_build::frozen_down_integer_record(
                            input.frozen_down_integer && replay == 1,
                            frozen_down_integer,
                            down_integer_preparations,
                            down_integer_dispatches,
                            down_integer_audits,
                            down_integer_sources,
                        )?;
                    let (actual_iq4_decoder, iq4_dispatches) =
                        executor.public_target_iq4_decoder()?;
                    let target_iq4_decoder_receipt =
                        crate::target_lab_build::public_iq4_decoder_record(
                            input.public_iq4_decoder && replay == 1,
                            actual_iq4_decoder,
                            iq4_dispatches,
                            executor.target_verify3_dense_calls,
                        )?;
                    let (public_grouped_status_folds, completed_status_folds) =
                        target.completed_public_grouped_status_folds(&executor)?;
                    let grouped_status_folds_receipt =
                        crate::target_lab_build::public_grouped_status_folds_record(
                            input.public_grouped_status_folds && replay == 1,
                            public_grouped_status_folds,
                            completed_status_folds,
                            u64::try_from(rounds.len())?,
                            target_recomputations,
                        )?;
                    let (three_gdn_streaming, three_gdn_completed_chunks) =
                        target.completed_public_three_gdn_streaming(&executor)?;
                    if three_gdn_streaming != (input.three_gdn_streaming && replay == 1) {
                        return Err("speculative actual completed GDN policy mismatch".into());
                    }
                    let frozen_partition_exit =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_partition_exit(&executor)?
                        } else {
                            false
                        };
                    let frozen_private_tables =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_private_tables(&executor)?
                        } else {
                            false
                        };
                    let frozen_workgroup_tables =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_workgroup_tables(&executor)?
                        } else {
                            false
                        };
                    if frozen_workgroup_tables != (input.frozen_workgroup_tables && replay == 1) {
                        return Err("actual frozen workgroup tables policy mismatch".into());
                    }
                    let frozen_shared_down_scales =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_shared_down_scales(&executor)?
                        } else {
                            false
                        };
                    if frozen_shared_down_scales != (input.frozen_shared_down_scales && replay == 1)
                    {
                        return Err("actual frozen shared down scales policy mismatch".into());
                    }
                    let frozen_four_row_down =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_four_row_down(&executor)?
                        } else {
                            false
                        };
                    if frozen_four_row_down != (input.frozen_four_row_down && replay == 1) {
                        return Err("actual four-row down policy mismatch".into());
                    }
                    let frozen_two_row_gate_up =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_two_row_gate_up(&executor)?
                        } else {
                            false
                        };
                    let frozen_three_members =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_three_members(&executor)?
                        } else {
                            false
                        };
                    if frozen_three_members != (input.frozen_three_members && replay == 1) {
                        return Err("actual three-member policy mismatch".into());
                    }
                    let frozen_packed_host_staging =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_packed_staging(&executor)?
                        } else {
                            false
                        };
                    if frozen_packed_host_staging
                        != (input.frozen_packed_host_staging && replay == 1)
                    {
                        return Err("actual packed host staging policy mismatch".into());
                    }
                    if frozen_two_row_gate_up != (input.frozen_two_row_gate_up && replay == 1) {
                        return Err("actual two-row gate/up policy mismatch".into());
                    }
                    let mtp_english_vocab = draft.completed_public_english_vocab()?;
                    if mtp_english_vocab != input.mtp_english_vocab {
                        return Err("actual completed English vocabulary mismatch".into());
                    }
                    let mtp_english_vocab_receipt = if mtp_english_vocab {
                        Some(draft.public_english_vocab_receipt()?)
                    } else {
                        None
                    };
                    let mtp_q8_head_shared_scales = draft.completed_q8_head_shared_scales()?;
                    if mtp_q8_head_shared_scales != input.mtp_q8_head_shared_scales {
                        return Err("actual MTP shared head scales mismatch".into());
                    }
                    let mtp_q8_head_broadcast = draft.completed_q8_head_broadcast()?;
                    if mtp_q8_head_broadcast != input.mtp_q8_head_broadcast {
                        return Err("actual completed MTP Q8 head policy mismatch".into());
                    }
                    let frozen_shared_gate_up_factors =
                        if executor.target_frozen_host_source_identity.is_some() {
                            target.native_frozen_source_shared_gate_up_factors(&executor)?
                        } else {
                            false
                        };
                    if frozen_shared_gate_up_factors
                        != (input.frozen_shared_gate_up_factors && replay == 1)
                    {
                        return Err("frozen public gate/up factors policy mismatch".into());
                    }
                    if frozen_private_tables != (input.frozen_private_tables && replay == 1) {
                        return Err("actual frozen private table policy mismatch".into());
                    }
                    if frozen_partition_exit != (input.frozen_partition_exit && replay == 1) {
                        return Err("frozen source partition exit policy changed".into());
                    }
                    if actual_direct != input.public_existing_fill {
                        return Err("public fill frozen selection changed".into());
                    }
                    let completed_bank_fill = target.native_frozen_source_bank_fill(&executor)?;
                    let bank_fill_receipt = crate::target_lab_build::public_host_bank_fill_record(
                        expected_bank_fill,
                        completed_bank_fill,
                        bank_fill_setup_experts,
                    )?;
                    if bank_fill_receipt != initial_prefill_bank_fill_receipt {
                        return Err("public bank filler immutable setup receipt changed".into());
                    }
                    let placement_scope = if expected_bank_fill
                        == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
                    {
                        "preserved actual residents with fixed public profile-ranked empty-slot filler"
                    } else if input.preserve_routed_bank {
                        "preserved actual routed VRAM resident set"
                    } else {
                        "deterministic diagnostic VRAM placement"
                    };
                    let mut record = serde_json::json!({"replay":replay,"device_router":(input.device_router || input.frozen_host_router) && replay == 1,"frozen_host_gpu_router":input.frozen_host_router && replay == 1,"frozen_host_inactive_partition_exit":frozen_partition_exit,"frozen_host_private_tables":frozen_private_tables,"frozen_host_workgroup_tables":frozen_workgroup_tables,"frozen_host_shared_down_scales":frozen_shared_down_scales,"frozen_host_shared_gate_up_factors":frozen_shared_gate_up_factors,"mtp_q8_head_broadcast":mtp_q8_head_broadcast,"mtp_q8_head_shared_scales":mtp_q8_head_shared_scales,"mtp_english_vocab":mtp_english_vocab,"mtp_english_vocab_receipt":mtp_english_vocab_receipt,"frozen_host_four_row_down":frozen_four_row_down,"frozen_host_two_row_gate_up":frozen_two_row_gate_up,"frozen_packed_host_staging":frozen_packed_host_staging,"frozen_three_member_arithmetic":frozen_three_members,"frozen_gu_integer":frozen_gu_integer,"frozen_gu_integer_counters":frozen_gu_integer_counters,"frozen_down_integer":frozen_down_integer,"frozen_down_integer_counters":frozen_down_integer_counters,"frozen_gu_stride_four":frozen_gu_stride_four,"frozen_gu_stride_four_counters":frozen_gu_stride_four_counters,"frozen_down_stride_four":frozen_down_stride_four,"frozen_down_stride_four_counters":frozen_down_stride_four_counters,"target_grouped_ple_reads":public_grouped_ple_reads,"target_grouped_ple_reads_receipt":grouped_ple_reads_receipt,"target_grouped_status_folds":public_grouped_status_folds,"target_grouped_status_folds_receipt":grouped_status_folds_receipt,"initial_prefill_target_grouped_status_folds":initial_prefill_target_grouped_status_folds,"initial_prefill_target_grouped_status_folds_receipt":initial_prefill_target_grouped_status_folds_receipt,"target_three_gdn_streaming":three_gdn_streaming,"target_three_gdn_completed_chunks":three_gdn_completed_chunks,"frozen_bank_placement":if input.preserve_routed_bank && replay == 1 {"preserve-routed"} else {"cpu13-diagnostic"},"device_router_scope":if input.frozen_host_router {"authenticated actual512 VRAM/host sources; deferred strict original IDs and full vectors, experimental F32 probabilities"} else {"conditional actual all-hit warm13; strict original IDs and full vectors, experimental F32 probabilities"},"request_scope_closed_and_audited":input.request_reuse,"request_scope":request_scope_receipt,"request_close_ns_inside_timing":request_close_ns,"host_io":host_io,"model_weights_warm":replay==1,"initial_prefill_ns":initial_prefill_ns,"initial_prefill_frozen_gu_integer":initial_prefill_frozen_gu_integer,"initial_prefill_frozen_gu_integer_counters":initial_prefill_frozen_gu_integer_counters,"initial_prefill_frozen_down_integer":initial_prefill_frozen_down_integer,"initial_prefill_frozen_down_integer_counters":initial_prefill_frozen_down_integer_counters,"initial_prefill_frozen_gu_stride_four":initial_prefill_frozen_gu_stride_four,"initial_prefill_frozen_gu_stride_four_counters":initial_prefill_frozen_gu_stride_four_counters,"initial_prefill_frozen_down_stride_four":initial_prefill_frozen_down_stride_four,"initial_prefill_frozen_down_stride_four_counters":initial_prefill_frozen_down_stride_four_counters,"gpu_router_calls":executor.gpu_router_calls-before_gpu_routers,"verify3_dense_calls":executor.target_verify3_dense_calls-before_dense,"verify3_fused_calls":executor.target_verify3_fused_calls-before_fused,"target_host_phase_ns":host_phase_ns,"load_and_initial_prefill_physical_read_bytes":load_disk_after.checked_sub(replay_disk_before).ok_or("public process read counter regressed")?,"timed_physical_read_bytes":timed_physical_read_bytes,"target_fill_counters_before":fill_counter_record(before_fills),"target_fill_counters_after":fill_counter_record(after_fills),"frozen_source_routes":frozen_routes,"weight_read_ns":executor.weight_read_ns-before_weight_reads,"weight_upload_enqueue_ns":executor.weight_upload_ns-before_weight_uploads,"timing_scope":if input.frozen_host_router && replay == 1 {format!("original CPU13 routes and weights; {placement_scope} plus immutable direct host sources and deferred original GPU route audits, no demand expert uploads; full vectors and actual repair included")} else if input.frozen_host_sources && replay == 1 {format!("original CPU13 routes and weights; {placement_scope} plus immutable direct host sources, no demand expert uploads; full vectors and actual repair included")} else if input.device_router && replay == 1 {"conditional actual all-hit GPU routing; deferred original-route audit, public checks and actual repair included".to_string()} else {"original normal routing/demand uploads; public checks and actual repair included".to_string()},"load_and_initial_prefill_cumulative_ns":load_ns,"accepted_tokens":accepted_tokens,"elapsed_ns":elapsed_ns,"accepted_tokens_per_second":accepted_tokens as f64*1e9/elapsed_ns as f64,"proposed_drafts":proposed_tokens,"accepted_drafts":accepted_drafts,"rounds":rounds,"expert_uploads":executor.expert_uploads-before_uploads,"expert_hits":executor.expert_hits-before_hits,"uploaded_bytes":executor.uploaded_bytes-before_bytes,"draft_traffic":{"host_read_calls":after.host_read_calls-before_traffic.host_read_calls,"host_read_bytes":after.host_read_bytes-before_traffic.host_read_bytes,"explicit_dispatches":after.explicit_dispatches-before_traffic.explicit_dispatches,"private_upload_calls":after.private_upload_calls-before_traffic.private_upload_calls,"private_upload_bytes":after.private_upload_bytes-before_traffic.private_upload_bytes,"minimal_packet_reads":after.minimal_packet_reads-before_traffic.minimal_packet_reads,"followup_copy_calls":after.followup_copy_calls-before_traffic.followup_copy_calls}});
                    record["target_submission_backlog"] = target_submission_backlog_receipt.clone();
                    record["mtp_batch_repair_requested"] = input.mtp_batch_repair.into();
                    record["mtp_batch_repair"] = mtp_batch_repair_receipt;
                    record["draft_traffic"]["batched_repair_calls"] = if selected_four_batch_repair
                    {
                        mtp_fused_four_repair_receipt["aggregate_completed_calls"].clone()
                    } else {
                        record["mtp_batch_repair"]["completed_calls"].clone()
                    };
                    record["draft_traffic"]["batched_repair_cells"] = if selected_four_batch_repair
                    {
                        mtp_fused_four_repair_receipt["aggregate_appended_cells"].clone()
                    } else {
                        record["mtp_batch_repair"]["appended_cells"].clone()
                    };
                    record["draft_traffic"]["batched_repair_fault_gates"] =
                        if selected_four_batch_repair {
                            mtp_fused_four_repair_receipt["aggregate_completed_fault_gates"].clone()
                        } else {
                            record["mtp_batch_repair"]["completed_fault_gates"].clone()
                        };
                    record["mtp_fused_four_repair"] = mtp_fused_four_repair_receipt;
                    record["initial_prefill_mtp_batch_repair"] = initial_prefill_mtp_batch_repair;
                    record["initial_prefill_mtp_fused_four_repair"] =
                        initial_prefill_mtp_fused_four_repair;
                    record["frozen_bank_fill"] =
                        crate::target_lab_build::public_host_bank_fill_name(
                            completed_bank_fill.policy,
                        )
                        .into();
                    record["frozen_bank_fill_receipt"] = bank_fill_receipt;
                    record["target_integer_head_requested"] =
                        input.public_integer_verify_head.into();
                    record["target_profile"] = target_profile_record(
                        input.profile_first_warm_chunk && replay == 1,
                        profiled_target_rounds,
                        client.properties().timing_method,
                    )?;
                    record["target_integer_head_receipt"] = target_integer_head_receipt;
                    record["initial_prefill_target_integer_head_receipt"] =
                        initial_prefill_target_integer_head_receipt;
                    record["target_four_lookahead_receipt"] = four_lookahead_receipt;
                    record["initial_prefill_target_four_lookahead_receipt"] =
                        initial_four_lookahead_receipt;
                    record["initial_prefill_frozen_bank_fill"] =
                        crate::target_lab_build::public_host_bank_fill_name(
                            initial_bank_fill.policy,
                        )
                        .into();
                    record["initial_prefill_frozen_bank_fill_receipt"] =
                        initial_prefill_bank_fill_receipt;
                    if expected_bank_fill
                        == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
                    {
                        record["frozen_bank_placement"] = "preserve-routed-profile-filler".into();
                    }
                    record["target_iq4_decoder"] =
                        crate::target_lab_build::public_iq4_decoder_name(actual_iq4_decoder).into();
                    record["target_iq4_decoder_receipt"] = target_iq4_decoder_receipt;
                    record["initial_prefill_target_iq4_decoder"] =
                        crate::target_lab_build::public_iq4_decoder_name(initial_iq4_decoder)
                            .into();
                    record["initial_prefill_target_iq4_decoder_receipt"] =
                        initial_prefill_target_iq4_decoder_receipt;
                    Ok(record)
                })()?;
                println!("public_speculative_full_pass_accepted replay={replay}");
                passes.push(pass);
                if input.warm_replay && replay == 0 {
                    // All guarded outputs and the old host carry are dropped.
                    // The reset performs complete actual private/source audits.
                    if input.frozen_host_sources {
                        println!("public_frozen_host_prepare_start");
                        let preparation = Instant::now();
                        let (plan, fills) = if input.bank_fill
                            == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
                        {
                            target.prepare_public_host_sources_with_fill_policy(
                                &mut executor,
                                &cancel,
                                input.preserve_routed_bank,
                                input.bank_fill,
                            )?
                        } else {
                            target.prepare_public_host_sources(
                                &mut executor,
                                &cancel,
                                input.preserve_routed_bank,
                            )?
                        };
                        if input.bank_fill
                            == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
                        {
                            bank_fill_setup_experts = Some(fills.experts);
                        }
                        println!("public_frozen_host_prepare_complete");
                        prepared_host
                            .set(plan)
                            .map_err(|_| "public source plan already initialized")?;
                        host_setup =
                            Some(serde_json::json!({"bank_completion_experts":fills.experts,
                            "bank_completion_uploaded_bytes":fills.bytes,
                            "bank_completion_read_ns":fills.read_ns,
                            "bank_completion_upload_ns":fills.upload_ns,
                            "prepare_ns":preparation.elapsed().as_nanos()}));
                    }
                    println!("public_warm_reset_start");
                    let receipt = target.reset_public_warm_replay(&mut executor, &cancel)?;
                    println!("public_warm_reset_complete");
                    crate::target_lab_build::validate_frozen_integer_math_close(
                        input.frozen_gu_integer,
                        input.frozen_down_integer,
                        receipt.main.buffers,
                        receipt.main.bytes,
                    )?;
                    draft.rewind(0)?;
                    // The typed reset commits target cursor0; the existing
                    // completed-projection proof intentionally requires >0.
                    // Next pass proves start0/count1/cursor1 with real prefill.
                    if draft.position() != 0 {
                        return Err("public warm replay empty draft state rejected".into());
                    }
                    if input.frozen_host_sources {
                        println!("public_frozen_host_attach_start");
                        let attachment_start = Instant::now();
                        let prepared = prepared_host.get().ok_or("public host plan absent")?;
                        let reserved = prepared.budget().total_host_reserved_bytes;
                        // SAFETY: protected runner verified the full immutable
                        // original model hashes before opening these sources.
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
                        if let Some(setup) = host_setup.as_mut() {
                            setup["attach_ns"] =
                                u64::try_from(attachment_start.elapsed().as_nanos())?.into();
                            setup["separate_host_reserved_bytes"] = reserved.into();
                        }
                        if input.frozen_host_router {
                            target.activate_public_frozen_host_batch_router(&executor, &cancel)?;
                            println!("public_frozen_host_gpu_router_active");
                        }
                        println!("public_frozen_host_attach_complete");
                    }
                    if input.public_iq4_decoder {
                        executor.activate_public_target_iq4_decoder()?;
                    }
                    warm_reset = Some(
                        serde_json::json!({"main_buffers":receipt.main.buffers,"main_bytes":receipt.main.bytes,"source_buffers":receipt.source.buffers,"source_bytes":receipt.source.bytes,"admission_and_public_banks_retained":true,"transient_binding_info_caches_retained":false,"ledger_epoch_or_poison_reset":false}),
                    );
                }
            }
            Ok(())
        })();
        let attached_resident_bank_allocation = host_attachment
            .as_ref()
            .map(|owner| owner.resident_bank_allocation())
            .transpose();
        let close = target.close();
        let audit = target.public_owned_zero_audit_receipt();
        drop(target);
        let cleanup = (|| -> Result<_> {
            close?;
            let (main, sources) = audit.ok_or("speculative target full owning erasure rejected")?;
            let host_close = if let Some(owner) = host_attachment.take() {
                // SAFETY: actual target private close and full owning zero audit
                // succeeded; lexical model drop removed every SourceSet consumer.
                Some(unsafe { owner.close()? })
            } else {
                None
            };
            crate::target_lab_build::validate_frozen_integer_math_close(
                input.frozen_gu_integer,
                input.frozen_down_integer,
                main.buffers,
                main.bytes,
            )?;
            Ok((main, sources, host_close))
        })();
        if runs.is_err() {
            println!(
                "public_speculative_failed_cleanup complete={}",
                u8::from(cleanup.is_ok())
            );
        }
        runs?;
        let (main, sources, host_close) = cleanup?;
        let resident_bank_allocation =
            crate::target_lab_build::public_resident_bank_allocation_record(
                attached_resident_bank_allocation?,
                host_close
                    .as_ref()
                    .map(|closed| closed.resident_bank_allocation),
            )?;
        let record = passes
            .last_mut()
            .ok_or("public controller completion missing")?;
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
        Ok(())
    })();
    let target_close = executor.close_transient_resident();
    drop(executor);
    let draft_close = draft.close();
    drop(draft);
    let draft_close = draft_close.and_then(|closed| {
        input
            .draft_profile
            .validate_close_with_english(&closed, input.mtp_english_vocab)?;
        Ok(closed)
    });
    if attempt.is_err() {
        println!(
            "public_speculative_failed_device_cleanup target={} draft={}",
            u8::from(target_close.is_ok()),
            u8::from(draft_close.is_ok())
        );
    }
    attempt?;
    target_close?;
    let draft_close = draft_close?;
    if target_retired.load(Ordering::Acquire) || draft_retired.load(Ordering::Acquire) {
        return Err("speculative physical stream retired".into());
    }
    file.verify_file_unchanged()?;
    sidecar.verify_file_unchanged()?;
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    if client.public_target_request_io_configured()? != input.request_reuse {
        return Err("target request stream frozen configuration mismatch".into());
    }
    let final_target_backlog = if input.request_reuse {
        Some(client.public_target_submission_backlog()?)
    } else {
        if client.public_target_submission_backlog().is_ok() {
            return Err("ordinary target stream acquired fixed backlog policy after close".into());
        }
        None
    };
    let final_backlog_receipt = crate::target_lab_build::public_target_submission_backlog_record(
        input.target_submission_backlog,
        final_target_backlog,
        input.request_reuse,
        draft_target_backlog_query_rejected,
    )?;
    if final_backlog_receipt != target_submission_backlog_receipt {
        return Err("immutable target backlog policy changed before full close".into());
    }
    let mut value = serde_json::json!({"device_dispatch":crate::target_lab_build::name(input.inline_dispatch),"observed_inline_device_dispatch":crate::target_lab_build::actual_inline(),"scope":if input.mtp_device == crate::target_lab_build::MtpDevice::Intel {"fixed-public-DASLab13-real-independent-stream-greedy-speculation"}else{"fixed-public-DASLab13-real-dual-GPU-greedy-speculation"},"target_card":"B70","draft_card":if input.mtp_device == crate::target_lab_build::MtpDevice::Intel {"B70"}else{"RTX3060"},"mtp_device_requested":input.mtp_device.name(),"mtp_device":mtp_device_receipt,"intel_mtp_admission":intel_mtp_admission,"draft_profile":input.draft_profile.name(),"gpu_chronological_qsa_selector":input.gpu_qsa_selector,"qsa_extra_runtime_info_bytes":if input.gpu_qsa_selector{crate::resident_qsa::batch::EXTRA_INFO_BYTES_PER_CHUNK}else{0},"qsa_extra_host_reserved_bytes":if input.gpu_qsa_selector{crate::resident_qsa::batch::EXTRA_HOST_BYTES_PER_CHUNK}else{0},"draft_public_reserved_bytes":input.draft_profile.public_bytes_with_english(input.mtp_english_vocab)?,"frozen_gu_integer_requested":input.frozen_gu_integer,"frozen_down_math":if input.frozen_down_integer {"integer-public"}else{"original"},"frozen_down_integer_requested":input.frozen_down_integer,"frozen_down_integer_extra_main_bytes":if input.frozen_down_integer{65536}else{0},"frozen_down_integer_extra_host_bytes":if input.frozen_down_integer{65536}else{0},"frozen_down_integer_extra_info_bytes":if input.frozen_down_integer{196608}else{0},"frozen_down_integer_extra_runtime_bytes":if input.frozen_down_integer{462848}else{0},"frozen_gu_stride_four_requested":input.frozen_gu_stride_four,"frozen_down_schedule":if input.frozen_down_stride_four {"four-public"} else {"original"},"frozen_down_stride_four_requested":input.frozen_down_stride_four,"target_ple_reads":if input.public_grouped_ple_reads {"grouped-public"} else {"original"},"target_grouped_ple_reads_requested":input.public_grouped_ple_reads,"target_status_folds":if input.public_grouped_status_folds {"grouped-public"} else {"original"},"target_grouped_status_folds_requested":input.public_grouped_status_folds,"frozen_gu_integer_extra_main_bytes":if input.frozen_gu_integer {65536} else {0},"frozen_gu_integer_extra_host_bytes":if input.frozen_gu_integer {65536} else {0},"frozen_gu_integer_extra_info_bytes":if input.frozen_gu_integer {196608} else {0},"frozen_gu_integer_extra_runtime_bytes":if input.frozen_gu_integer {462848} else {0},"verify3_dense":input.verify3_dense,"verify3_logical_finite_fused":input.verify3_fused,"prepared_verify3_experts":input.prepared_experts,"live_member_reduction":input.live_member_reduction,"public_existing_fill":input.public_existing_fill,"chunk_io_checkpoint":input.chunk_io,"request_io_reuse":input.request_reuse,"request_reuse_metadata_reserved_bytes":if input.request_reuse {4*1024*1024}else{0},"request_binding_capacity":if input.request_reuse {4096}else{0},"fused_mtp_sweeps":input.fused_sweeps,"warm_replay":input.warm_replay,"warm_reset":warm_reset,"compiled_token0":0,"seed":424242,"temperature":0,"oracle_forcing":false,"expert_route_forcing":false,"passes":passes,"draft_close":{"private_buffers":draft_close.private.buffers,"private_bytes":draft_close.private.bytes,"public_bytes":draft_close.public_bytes,"main_reserved":draft_close.main_reserved},"private_or_production_accepted":false});
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
    value["target_profile_requested"] = if input.profile_first_warm_chunk {
        "first-warm-chunk"
    } else {
        "disabled"
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
    println!("public_speculative_complete all_accepted_vectors=true owners_erased=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    fn target_profile_env() -> Vec<(OsString, OsString)> {
        let mut base = intel_mtp_env();
        for (key, value) in [
            ("KUNO_LAB_TARGET_PROFILE", "first-warm-chunk"),
            ("KUNO_LAB_FROZEN_GU_MATH", "integer-public"),
            ("KUNO_LAB_FROZEN_DOWN_MATH", "integer-public"),
            ("KUNO_LAB_TARGET_PLE_READS", "grouped-public"),
        ] {
            base.push((key.into(), value.into()));
        }
        base
    }

    #[test]
    fn target_profile_canonical_default_and_three_four_head_repair_composition() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        assert!(
            !registration(&args, intel_mtp_env(), build)
                .unwrap()
                .profile_first_warm_chunk
        );
        for four in [false, true] {
            for head in [false, true] {
                for repair in [false, true] {
                    let mut selected = target_profile_env();
                    if four {
                        selected.push(("KUNO_LAB_TARGET_LOOKAHEAD".into(), "four-public".into()));
                    }
                    if head {
                        selected.push(("KUNO_LAB_TARGET_HEAD".into(), "integer-public".into()));
                    }
                    if repair {
                        selected.push(("KUNO_LAB_MTP_REPAIR".into(), "batch-public".into()));
                    }
                    let got = registration(&args, selected, build).unwrap();
                    assert!(got.profile_first_warm_chunk);
                    assert_eq!(got.four_lookahead, four);
                    assert_eq!(got.public_integer_verify_head, head);
                    assert_eq!(got.mtp_batch_repair, repair);
                }
            }
        }
    }

    #[test]
    fn target_profile_rejects_missing_scope_noncanonical_and_alternate_policies() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = target_profile_env();
        for key in [
            "KUNO_LAB_WARM_REPLAY",
            "KUNO_LAB_MTP_DEVICE",
            "KUNO_LAB_MTP_SWEEPS",
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_FILL",
            "KUNO_LAB_TARGET_DENSE",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_TARGET_SOURCES",
            "KUNO_LAB_TARGET_ROUTER",
            "KUNO_LAB_FROZEN_PARTITION_EXIT",
            "KUNO_LAB_FROZEN_GU_MATH",
            "KUNO_LAB_FROZEN_DOWN_MATH",
            "KUNO_LAB_TARGET_PLE_READS",
        ] {
            let mut bad = base.clone();
            bad.retain(|(k, _)| k != key);
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        for (key, value) in [
            ("KUNO_LAB_TARGET_PROFILE", "first"),
            ("KUNO_LAB_TARGET_PROFILE", "first-warm-chunk "),
            ("KUNO_LAB_TARGET_PROFILE", ""),
            ("KUNO_LAB_MTP_PROFILE", "strata-q2"),
            ("KUNO_LAB_MTP_Q8_HEAD", "broadcast-public"),
            ("KUNO_LAB_MTP_VOCAB", "english-public"),
            ("KUNO_LAB_TARGET_IQ4_DECODER", "packed-registers-public"),
            ("KUNO_LAB_FROZEN_BANK_FILL", "profile-ranked-public"),
            ("KUNO_LAB_FROZEN_BANK_PLACEMENT", "preserve-public"),
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
            bad.retain(|(k, _)| k != key);
            bad.push((key.into(), value.into()));
            assert!(registration(&args, bad, build).is_err(), "{key}");
        }
        let mut duplicate = base.clone();
        duplicate.push(("KUNO_LAB_TARGET_PROFILE".into(), "first-warm-chunk".into()));
        assert!(registration(&args, duplicate, build).is_err());
        assert!(registration(&args, base, crate::target_lab_build::Build::Threaded).is_err());
    }

    #[test]
    fn target_profile_selects_only_first_warm_verification_and_exact_device_receipt() {
        use cubecl_common::profile::TimingMethod::{Device, System};
        for selected in [false, true] {
            for replay in 0..3 {
                for rounds in 0..4 {
                    assert_eq!(
                        first_warm_profile_round(selected, replay, rounds),
                        selected && replay == 1 && rounds == 0
                    );
                }
            }
        }
        let cold = target_profile_record(false, 0, Device).unwrap();
        assert_eq!(cold["selected"], false);
        assert_eq!(cold["completed_target_rounds"], 0);
        let warm = target_profile_record(true, 1, Device).unwrap();
        assert_eq!(warm["timing_method"], "device");
        assert_eq!(warm["scope"], "first-warm-target-round-verify-only");
        assert_eq!(warm["normal_throughput_comparable"], false);
        assert!(target_profile_record(true, 0, Device).is_err());
        assert!(target_profile_record(true, 2, Device).is_err());
        assert!(target_profile_record(false, 1, Device).is_err());
        assert!(target_profile_record(true, 1, System).is_err());
        assert!(target_profile_record(false, 0, System).is_ok());
    }

    #[test]
    fn target_profile_guard_pauses_after_success_and_verification_error() {
        use std::cell::RefCell;
        for fail in [false, true] {
            let events = RefCell::new(Vec::new());
            let verify = || -> std::result::Result<(), &'static str> {
                let guard = TargetProfileGuard::new(|paused| events.borrow_mut().push(paused));
                if fail {
                    return Err("fixed public verification fault");
                }
                drop(guard);
                Ok(())
            };
            assert_eq!(verify().is_err(), fail);
            assert_eq!(*events.borrow(), [false, true]);
        }
    }

    #[test]
    fn integer_verify_head_default_off_and_original_three_four_composition() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            "KUNO_LAB_WARM_REPLAY",
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        ] {
            let mut invalid = base.clone();
            invalid.retain(|(k, _)| k != key);
            invalid.push((key.into(), value.into()));
            assert!(registration(&args, invalid, build).is_err(), "{key}");
        }
        // Removing either new composition selector retains its original scope.
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
    fn four_lookahead_registration_preserves_default_and_admits_original_composition() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            "KUNO_LAB_WARM_REPLAY",
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
    }
    #[test]
    fn profile_ranked_bank_filler_default_off_and_composes_original_public_policies() {
        use crate::executor::PublicHostBankFillPolicy as P;
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            "KUNO_LAB_WARM_REPLAY",
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let mut legacy_preserve = intel_mtp_env();
        legacy_preserve.retain(|(key, _)| {
            ![
                "KUNO_LAB_MTP_DEVICE",
                "KUNO_PUBLIC_MTP_INTEL_TOTAL_BYTES",
                "KUNO_PUBLIC_MTP_INTEL_USED_BYTES",
                "KUNO_PUBLIC_MTP_INTEL_DEVICE_ID",
            ]
            .iter()
            .any(|removed| key == removed)
        });
        legacy_preserve.push((
            "KUNO_LAB_FROZEN_BANK_PLACEMENT".into(),
            "preserve-public".into(),
        ));
        assert_eq!(
            registration(&args, legacy_preserve, build)
                .unwrap()
                .bank_fill,
            P::Original
        );
    }

    #[test]
    fn target_backlog_default_off_and_original_public_compositions() {
        use cubecl_runtime::server::PublicTargetSubmissionBacklog as P;
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let mut cold_only = intel_mtp_env();
        cold_only.retain(|(k, _)| k != "KUNO_LAB_WARM_REPLAY");
        cold_only.push((
            "KUNO_LAB_TARGET_SUBMISSION_BACKLOG".into(),
            "queued-2048-public".into(),
        ));
        assert!(registration(&args, cold_only, build).is_err());
        let mut preserve = intel_mtp_env();
        preserve.push((
            "KUNO_LAB_TARGET_SUBMISSION_BACKLOG".into(),
            "queued-2048-public".into(),
        ));
        preserve.push((
            "KUNO_LAB_FROZEN_BANK_PLACEMENT".into(),
            "preserve-public".into(),
        ));
        assert!(registration(&args, preserve, build).is_err());
    }

    #[test]
    fn grouped_status_registration_keeps_default_off_and_original_public_compositions() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
    fn grouped_status_cpu13_requires_its_real_single_warm_replay() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let mut invalid = frozen_down_stride_four_env();
        invalid.push((
            "KUNO_LAB_TARGET_STATUS_FOLDS".into(),
            "grouped-public".into(),
        ));
        invalid.retain(|(k, _)| k != "KUNO_LAB_WARM_REPLAY");
        assert!(registration(&args, invalid, crate::target_lab_build::Build::Inline).is_err());
    }
    #[test]
    fn grouped_ple_registration_keeps_default_off_and_original_public_compositions() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
    fn grouped_ple_cpu13_requires_its_real_single_warm_replay() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let mut invalid = frozen_down_stride_four_env();
        invalid.push(("KUNO_LAB_TARGET_PLE_READS".into(), "grouped-public".into()));
        invalid.retain(|(k, _)| k != "KUNO_LAB_WARM_REPLAY");
        assert!(registration(&args, invalid, crate::target_lab_build::Build::Inline).is_err());
    }
    #[test]
    fn grouped_status_registration_composes_independently_with_ple_and_down() {
        let args = ["lab", "intel", "daslab-speculative-probe"]
            .map(OsString::from)
            .to_vec();
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
                let got = registration(
                    &["lab", "intel", "daslab-speculative-probe"].map(OsString::from),
                    valid,
                    build,
                )
                .unwrap();
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
    use super::*;
    fn env() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "1"),
            ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
            ("KUNO_TEST_Q3_DIR", "/original"),
            ("KUNO_TEST_MTP", "/original/sidecar"),
            ("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "/public/carry"),
            ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .to_vec()
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
        base.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
        base
    }
    #[test]
    fn mtp_batch_repair_registration_keeps_original_default_and_composes_exact_scope() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
    fn mtp_batch_repair_cpu13_requires_real_warm_pass() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let mut invalid = intel_mtp_env();
        invalid.retain(|(key, _)| key != "KUNO_LAB_WARM_REPLAY");
        invalid.push(("KUNO_LAB_MTP_REPAIR".into(), "batch-public".into()));
        assert!(registration(&args, invalid, crate::target_lab_build::Build::Inline).is_err());
    }

    #[test]
    fn intel_mtp_controller_registration_retains_full144_original_target_and_explicit_profile() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            "KUNO_LAB_WARM_REPLAY",
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
        let mut preserve = base;
        preserve.push((
            "KUNO_LAB_FROZEN_BANK_PLACEMENT".into(),
            "preserve-public".into(),
        ));
        assert!(registration(&args, preserve, build).is_err());
    }

    #[test]
    fn iq4_decoder_registration_requires_fixed_intel_full_q8_and_keeps_old_defaults() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            "KUNO_LAB_WARM_REPLAY",
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
        base.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
        base
    }
    #[test]
    fn frozen_down_stride_four_registration_accepts_plain_integer_and_stride_gu() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let base = frozen_down_stride_four_env();
        assert!(
            !registration(&args, base.clone(), build)
                .unwrap()
                .frozen_down_stride_four
        );
        for (integer, stride) in [(false, false), (true, false), (true, true)] {
            let mut valid = base.clone();
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
    fn frozen_gu_stride_four_controller_registration_is_explicit_and_integer_only() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        base.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        base.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
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
            "KUNO_LAB_WARM_REPLAY",
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
            ("KUNO_LAB_FROZEN_BANK_PLACEMENT", "preserve-public"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        base.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
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
            "KUNO_LAB_WARM_REPLAY",
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
            ("KUNO_LAB_FROZEN_BANK_PLACEMENT", "preserve-public"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        base.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        fn args() -> Vec<OsString> {
            ["binary", "intel", "daslab-speculative-probe"]
                .map(Into::into)
                .to_vec()
        }
        let build = crate::target_lab_build::Build::Inline;
        let mut selected = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
            "KUNO_LAB_WARM_REPLAY",
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut base = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut selected = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_FROZEN_PARTITION_EXIT", "1"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut selected = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
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
    fn bank_placement_is_explicit_warm_frozen_only_and_rejects_malformed_flags() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let build = crate::target_lab_build::Build::Inline;
        let mut selected = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_FILL", "direct-public"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_TARGET_SOURCES", "frozen-host-public"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
        ] {
            selected.push((key.into(), value.into()));
        }
        assert!(
            !registration(&args, selected.clone(), build)
                .unwrap()
                .preserve_routed_bank
        );
        let flag = (
            "KUNO_LAB_FROZEN_BANK_PLACEMENT".into(),
            "preserve-public".into(),
        );
        let mut valid = selected.clone();
        valid.push(flag.clone());
        let registered = registration(&args, valid.clone(), build).unwrap();
        assert!(registered.preserve_routed_bank);
        // The measurement composes with the accepted exit candidate.
        valid.push(("KUNO_LAB_FROZEN_PARTITION_EXIT".into(), "1".into()));
        let composed = registration(&args, valid.clone(), build).unwrap();
        assert!(composed.preserve_routed_bank && composed.frozen_partition_exit);
        for raw in [
            "",
            "0",
            "true",
            "preserve",
            "preserve-public ",
            "diagnostic",
        ] {
            let mut invalid = selected.clone();
            invalid.push(("KUNO_LAB_FROZEN_BANK_PLACEMENT".into(), raw.into()));
            assert!(registration(&args, invalid, build).is_err());
        }
        // Without warm replay or without frozen sources the flag is rejected.
        for key in ["KUNO_LAB_TARGET_SOURCES", "KUNO_LAB_WARM_REPLAY"] {
            let mut invalid = valid.clone();
            invalid.retain(|(k, _)| k != key);
            invalid.push((
                "KUNO_LAB_FROZEN_BANK_PLACEMENT".into(),
                "preserve-public".into(),
            ));
            assert!(registration(&args, invalid, build).is_err());
        }
        // Duplicates remain rejected.
        let mut duplicate = valid.clone();
        duplicate.push(flag);
        assert!(registration(&args, duplicate, build).is_err());
    }
    #[test]
    fn qsa_gpu_selection_is_explicit_and_rejects_malformed_or_duplicate_policy() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
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
    fn draft_profile_is_explicit_and_keeps_existing_public_build_admission() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let threaded = crate::target_lab_build::Build::Threaded;
        assert_eq!(
            registration(&args, env(), threaded).unwrap().draft_profile,
            Profile::SelfContainedQ8
        );
        let mut q2 = env();
        // Same opaque path as Q8: admission is selected explicitly, never by name.
        q2.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        let selected = registration(&args, q2.clone(), threaded).unwrap();
        assert_eq!(selected.draft_profile, Profile::StrataQ2);
        assert_eq!(selected.mtp, PathBuf::from("/original/sidecar"));
        assert_eq!(selected.draft_profile.public_bytes(), 2_539_257_856);
        assert!(!selected.fused_sweeps);
        let mut fused = q2.clone();
        fused.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
        assert!(registration(&args, fused, threaded).unwrap().fused_sweeps);
        let mut inline = q2.clone();
        inline.push(("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()));
        assert_eq!(
            registration(
                &args,
                inline.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .draft_profile,
            Profile::StrataQ2
        );
        assert!(registration(&args, inline, threaded).is_err());
        assert!(registration(&args, q2.clone(), crate::target_lab_build::Build::Rejected).is_err());
        for key in [
            "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY",
            "KUNO_LAB_STATIC_CACHE_BYTES",
            "KUNO_TEST_Q3_DIR",
            "KUNO_TEST_MTP",
            "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE",
            "KUNO_PUBLIC_RECEIPT",
        ] {
            let missing = q2.iter().filter(|(k, _)| k != key).cloned();
            assert!(registration(&args, missing, threaded).is_err());
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
            assert!(registration(&args, malformed, threaded).is_err());
        }
        let mut repeated = q2.clone();
        repeated.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        assert!(registration(&args, repeated, threaded).is_err());
        for key in [
            "KUNO_LAB_TEST",
            "KUNO_TEST_OTHER",
            "KUNO_PUBLIC_MTP_REFERENCE",
        ] {
            let mut foreign = q2.clone();
            foreign.push((key.into(), "1".into()));
            assert!(registration(&args, foreign, threaded).is_err());
        }
    }
    #[test]
    fn device_router_requires_original_pure_inline_one_shot_warm_request() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let mut values = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
            ("KUNO_LAB_TARGET_ROUTER", "device-warm-public"),
        ] {
            values.push((key.into(), value.into()));
        }
        assert!(
            registration(
                &args,
                values.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .device_router
        );
        for missing in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_WARM_REPLAY",
        ] {
            let subset = values.iter().filter(|(k, _)| k != missing).cloned();
            assert!(registration(&args, subset, crate::target_lab_build::Build::Inline).is_err());
        }
        let mut integer = values.clone();
        integer.push(("KUNO_LAB_TARGET_EXPERTS".into(), "prepared-verify3".into()));
        assert!(registration(&args, integer, crate::target_lab_build::Build::Inline).is_err());
        assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
    }

    #[test]
    fn real_greedy_acceptance_keeps_only_prefix_and_target_correction() {
        assert_eq!(accepted_prefix(&[7, 8, 9], &[8, 9, 10]), Ok(3));
        assert_eq!(accepted_prefix(&[7, 8, 9], &[8, 11, 12]), Ok(2));
        assert_eq!(accepted_prefix(&[7, 8, 9], &[11, 9, 12]), Ok(1));
        assert_eq!(accepted_prefix(&[7, 8], &[8, 9]), Ok(2));
        assert_eq!(accepted_prefix(&[7], &[8]), Ok(1));
        for (x, y) in [
            (vec![], vec![]),
            (vec![1, 2, 3, 4], vec![1, 2, 3, 4]),
            (vec![1, 2], vec![1]),
            (vec![1, 248320], vec![1, 2]),
        ] {
            assert!(accepted_prefix(&x, &y).is_err());
        }
    }
    #[test]
    fn registration_is_exclusive_default_public_dual_gpu() {
        let args: Vec<OsString> = ["binary", "intel", "daslab-speculative-probe"]
            .map(Into::into)
            .to_vec();
        let env = || {
            [
                ("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "1"),
                ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
                ("KUNO_TEST_Q3_DIR", "/original"),
                ("KUNO_TEST_MTP", "/original/sidecar"),
                ("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "/public/carry"),
                ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
            ]
            .map(|(k, v)| (k.into(), v.into()))
            .to_vec()
        };
        assert!(registration(&args, env(), crate::target_lab_build::Build::Threaded).is_ok());
        let mut live = env();
        live.push(("KUNO_LAB_TARGET_MOE".into(), "live".into()));
        assert!(
            registration(
                &args,
                live.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .unwrap()
            .live_member_reduction
        );
        live.push(("KUNO_LAB_TARGET_EXPERTS".into(), "prepared-verify3".into()));
        assert!(registration(&args, live, crate::target_lab_build::Build::Threaded).is_err());
        for value in ["", "auto", "1", "live "] {
            let mut invalid = env();
            invalid.push(("KUNO_LAB_TARGET_MOE".into(), value.into()));
            assert!(
                registration(&args, invalid, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        let mut reusable = env();
        reusable.extend([
            ("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()),
            ("KUNO_LAB_TARGET_IO".into(), "chunk".into()),
            ("KUNO_LAB_TARGET_REUSE".into(), "request".into()),
        ]);
        assert!(
            registration(
                &args,
                reusable.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .request_reuse
        );
        assert!(
            registration(
                &args,
                reusable.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        for key in ["KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO"] {
            let mut missing = reusable.clone();
            missing.retain(|(k, _)| k != key);
            assert!(registration(&args, missing, crate::target_lab_build::Build::Inline).is_err());
        }
        for raw in ["", "auto", "1", "request ", "chunk"] {
            let mut wrong = reusable.clone();
            for (k, v) in &mut wrong {
                if k == "KUNO_LAB_TARGET_REUSE" {
                    *v = raw.into();
                }
            }
            assert!(registration(&args, wrong, crate::target_lab_build::Build::Inline).is_err());
        }
        let mut inline = env();
        inline.push(("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()));
        assert!(
            registration(
                &args,
                inline.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .inline_dispatch
        );
        assert!(
            registration(
                &args,
                inline.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        assert!(registration(&args, inline, crate::target_lab_build::Build::Rejected).is_err());
        assert!(registration(&args, env(), crate::target_lab_build::Build::Inline).is_err());
        for bad in ["", "auto", "threaded", "1", "inline "] {
            let mut value = env();
            value.push(("KUNO_LAB_DEVICE_DISPATCH".into(), bad.into()));
            assert!(registration(&args, value, crate::target_lab_build::Build::Inline).is_err());
        }

        let mut finite = env();
        finite.push(("KUNO_LAB_TARGET_DENSE".into(), "verify3-fused".into()));
        let selected =
            registration(&args, finite, crate::target_lab_build::Build::Threaded).unwrap();
        assert!(selected.verify3_dense && selected.verify3_fused);
        let mut warm = env();
        warm.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
        assert!(
            registration(&args, warm, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .warm_replay
        );
        for invalid in ["", "auto", "0", "1", "once ", "twice"] {
            let mut values = env();
            values.push(("KUNO_LAB_WARM_REPLAY".into(), invalid.into()));
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut fused = env();
        fused.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
        assert!(
            registration(&args, fused, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .fused_sweeps
        );
        for invalid in ["", "auto", "0", "1", "fused "] {
            let mut values = env();
            values.push(("KUNO_LAB_MTP_SWEEPS".into(), invalid.into()));
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_DENSE".into(), "verify3".into()));
        assert!(
            registration(&args, candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .verify3_dense
        );
        let mut direct = env();
        direct.push(("KUNO_LAB_TARGET_FILL".into(), "direct-public".into()));
        assert!(
            registration(&args, direct, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .public_existing_fill
        );
        for invalid in ["", "auto", "0", "1", "direct-public "] {
            let mut values = env();
            values.push(("KUNO_LAB_TARGET_FILL".into(), invalid.into()));
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_EXPERTS".into(), "prepared-verify3".into()));
        assert!(
            registration(&args, candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .prepared_experts
        );
        for invalid in ["", "auto", "0", "1", "prepared-verify3 "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_EXPERTS".into(), invalid.into()));
            assert!(
                registration(&args, candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_IO".into(), "chunk".into()));
        assert!(
            registration(&args, candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .chunk_io
        );
        for invalid in ["", "auto", "0", "1", "chunk "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_IO".into(), invalid.into()));
            assert!(
                registration(&args, candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        for invalid in ["", "auto", "0", "1", "verify3 "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_DENSE".into(), invalid.into()));
            assert!(
                registration(&args, candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        assert!(registration(&args, env(), crate::target_lab_build::Build::Rejected).is_err());
        for index in 0..6 {
            let mut values = env();
            values.remove(index);
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut values = env();
        values.push(("KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY".into(), "1".into()));
        assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        let mut wrong = args.clone();
        wrong[1] = "nvidia".into();
        assert!(registration(&wrong, env(), crate::target_lab_build::Build::Threaded).is_err());
    }
}
