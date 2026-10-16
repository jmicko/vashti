//! Actual layer-major MAX8 laboratory composition. It reuses the original
//! committed state/cache/banks and publishes only after whole-model acceptance.
//! Source-only until parent completes physical whole-model validation. No
//! default/API/catalog change, replay, fallback or private checkpoint export.
pub(crate) struct CompletedAdaptiveBoundary(());
#[cfg(test)]
mod bank_fill_policy_tests {
    use super::*;
    use crate::executor::PublicHostBankFillPolicy::{Original, StrataRankedFiller};
    #[test]
    fn ranked_fill_policy_never_reinterprets_legacy_preserve_or_default_scope() {
        for preserve in [false, true] {
            for scope in [false, true] {
                assert!(validate_bank_fill_selection(Original, preserve, scope).is_ok());
                assert_eq!(
                    validate_bank_fill_selection(StrataRankedFiller, preserve, scope).is_ok(),
                    !preserve && scope
                );
            }
        }
    }
}
fn validate_bank_fill_selection(
    policy: crate::executor::PublicHostBankFillPolicy,
    preserve: bool,
    original_scope: bool,
) -> Result<()> {
    if policy == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
        && (preserve || !original_scope)
    {
        return Err(Error::InvalidFormat("fixed public ranked bank fill scope"));
    }
    Ok(())
}

use super::*;
use crate::{
    executor::{GpuExecutor, NativeProjectionTile, batch_moe},
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, ResidentBatchProjection},
    resident_gdn::{BoundProjection, batch as gdn},
    resident_gr::batch as gr,
    resident_qsa::batch as qsa,
    token_waves,
};
use kuno_text::cpu::Matrix;
#[path = "resident_native_commit.rs"]
mod native_commit;
#[path = "resident_model_prompt.rs"]
pub(crate) mod prompt_full;
#[path = "resident_target_rows.rs"]
pub(crate) mod target_rows;
pub(crate) use target_rows::NativeTargetRows;
#[path = "resident_target_checkpoint.rs"]
mod latest_checkpoint;
#[path = "resident_target_round.rs"]
mod latest_round;
pub(crate) use latest_round::LatestTargetRound;
#[path = "resident_language_request_io.rs"]
pub(crate) mod public_language_request_io;
#[path = "resident_target_request_io.rs"]
pub(crate) mod public_target_request_io;
#[path = "resident_model_public_reset.rs"]
pub(crate) mod public_warm_replay;
pub(crate) const TARGET_ROWS_MAX: usize = target_rows::MAX_ROWS;
pub(crate) const TARGET_ROWS_RUNTIME_LIMIT: u64 = target_rows::RUNTIME_LIMIT;
pub(crate) const TARGET_ROWS_PRIVATE_MAIN: u64 = TOTAL_MAIN
    + target_rows::MAIN_BYTES
    + if crate::strata_intel_core::enabled() {
        crate::strata_global_moe::MAIN_BYTES
    } else {
        0
    };
#[allow(dead_code)] // Parent owns the exact executor construction-extent gate.
pub(crate) const TARGET_ROWS_PREPARED_VERIFY3_MAIN: u64 =
    TARGET_ROWS_PRIVATE_MAIN + batch_moe::prepared_verify3::MAIN_BYTES;

// The native INT8 QSA cache owns8MiB instead of32MiB per layer;
// retaining one extra pooled-key row raises its full owner from512KiB to1MiB.
// All12 QSA layers change together. Match the actual selected full roots,
// including padding and the separate unchanged GDN replay input owners.
pub(super) const OLD_MAIN: u64 = 785_186_816 + gdn::strata_replay::MAIN_BYTES
    - if cfg!(feature = "strata-intel-core") {
        12 * (24 * 1024 * 1024 - 512 * 1024)
    } else {
        0
    };
// Family requests already include the native GR increment. Both aggregate
// equalities below must describe that SAME full request vector; charging it
// only in TOTAL_MAIN made the two constructor checks mutually impossible.
const NATIVE_QSA_BATCH_MAIN: u64 = if cfg!(feature = "strata-intel-core") {
    // Original MAX8 KV candidate page shrinks; split attention's partial and
    // statistics planes replace the two full-score softmax pages together.
    22_052_864 - 17_825_792
} else {
    0
};
const NEW_MAIN: u64 = 33_062_912 + NATIVE_GR_MAIN + NATIVE_QSA_BATCH_MAIN;
const NATIVE_GDN_RUNTIME_INFO: u64 = if cfg!(feature = "strata-intel-core") {
    // runtime_info_reservation(..., true) retains both audited copies.
    2 * crate::strata_gdn::prompt::INFO_BYTES
} else {
    0
};
// A native prefix is formed before the request I/O checkpoint, so its
// runtime records coexist with the original verification window. Reserve the
// actual36 Conv/Replay/fault folds,12 suffix clear/folds and1 PLE History,
// conservatively retaining both audited copies of each2048-byte record.
const NATIVE_PREFIX_RUNTIME_INFO: u64 = if cfg!(feature = "strata-intel-core") {
    2 * (36 * 3 + 12 * 2 + 1) * 2048
} else {
    0
};
const TOTAL_MAIN: u64 = OLD_MAIN + NEW_MAIN;
const NATIVE_GR_MAIN: u64 = if cfg!(feature = "strata-intel-core") {
    crate::strata_gr::PRIVATE_INCREMENT
} else {
    0
};
const PUBLIC_MAIN: u64 = 25_165_824;
const CORE_CACHE_HOST: usize = if cfg!(feature = "strata-intel-core") {
    crate::strata_global_adapt::HOST_BYTES
        + crate::strata_sampler::request::HOST_BYTES
        + kuno_text::cpu_expert::native::HOST_BYTES
        + crate::strata_prompt::groups::HOST_BYTES
        + crate::strata_prompt::storage::HOST_BYTES
} else {
    0
};
const PROMPT_SOURCE_HOST: usize = if cfg!(feature = "strata-intel-core") {
    ple_batch::prompt_source::HOST_BYTES + prompt_full::HOST_BYTES
} else {
    0
};
const HOST_CAP: usize = 12 * 1024 * 1024 + CORE_CACHE_HOST + PROMPT_SOURCE_HOST;
pub(crate) const ORIGINAL_BANK_HOST: usize = if cfg!(feature = "strata-resident-240") {
    704 * 1024
} else {
    640 * 1024
};
const HOST_COMPONENTS: usize =
    10_551_296 + ORIGINAL_BANK_HOST + CORE_CACHE_HOST + PROMPT_SOURCE_HOST;
const META_CAP: usize = 256 * 1024 + CORE_CACHE_HOST;
const LAYERS: usize = 48;
const D: usize = 2560;
const HC: usize = 4;
const ROUTER: usize = 512;
const RESIDUAL_WORDS: usize = MAX_BATCH * HC * D;
const BLOCK_WORDS: usize = MAX_BATCH * D;
const HYBRID_MAIN: u64 = 163_840;
const HYBRID_HOST: usize = 1024 * 1024;
// Gate and down planes each prepare once per count1 layer. Retain their full
// INFO source/target/inspection charge through the token's owning I/O audit.
const HYBRID_PREPARE_INFO: u64 = 2 * LAYERS as u64 * 2048;
// Separate native gate/up/down chunks and activation replace two broad
// projections. Reserve all eight additional full INFO owners per layer;
// no old source/target/inspection allowance is spent.
const NATIVE_EXPERT_ROLE_INFO: u64 = if cfg!(feature = "strata-intel-core") {
    8 * LAYERS as u64 * 2048
} else {
    0
};
const HYBRID_RUNTIME: u64 = 1024 * 1024 + HYBRID_PREPARE_INFO + NATIVE_EXPERT_ROLE_INFO;
const DEFAULT_REUSE_METADATA: usize = 256 * 1024;

/// Complete separate runtime metadata reservation, added to the existing
/// application component plan. Only fixed public laboratory configurations can
/// select the larger ceilings; device owners and family metadata remain intact.
#[allow(dead_code)] // Original independent envelope remains directly tested.
pub(crate) fn host_reservation(hybrid: bool, reuse_metadata: usize) -> Result<(usize, usize)> {
    host_reservation_with_router(hybrid, reuse_metadata, false)
}
pub(crate) fn host_reservation_with_router(
    hybrid: bool,
    reuse_metadata: usize,
    router: bool,
) -> Result<(usize, usize)> {
    host_reservation_with_outputs(hybrid, reuse_metadata, router, false)
}
fn host_reservation_with_outputs(
    hybrid: bool,
    reuse_metadata: usize,
    router: bool,
    target_rows: bool,
) -> Result<(usize, usize)> {
    host_reservation_with_checkpoint(hybrid, reuse_metadata, router, target_rows, false)
}
/// Construction-only reservation for the explicit public target repair lab.
/// The output capacity is required; ordinary reservations retain their bounds.
#[allow(dead_code)] // Parent owns the protected registration/pre-device charge.
pub(crate) fn host_reservation_with_latest_checkpoint(
    reuse_metadata: usize,
) -> Result<(usize, usize)> {
    host_reservation_with_checkpoint(false, reuse_metadata, false, true, true)
}
fn host_reservation_with_checkpoint(
    hybrid: bool,
    reuse_metadata: usize,
    router: bool,
    target_rows: bool,
    checkpoint: bool,
) -> Result<(usize, usize)> {
    host_reservation_with_verify3(
        hybrid,
        reuse_metadata,
        router,
        target_rows,
        checkpoint,
        false,
    )
}
/// Parent pre-device charge for the explicit prepared expert candidate.
#[allow(dead_code)]
pub(crate) fn host_reservation_with_prepared_verify3(
    reuse_metadata: usize,
    checkpoint: bool,
) -> Result<(usize, usize)> {
    host_reservation_with_verify3(false, reuse_metadata, false, true, checkpoint, true)
}
fn host_reservation_with_verify3(
    hybrid: bool,
    reuse_metadata: usize,
    router: bool,
    target_rows: bool,
    checkpoint: bool,
    prepared_verify3: bool,
) -> Result<(usize, usize)> {
    if !matches!(reuse_metadata, DEFAULT_REUSE_METADATA | 1048576 | 4194304)
        || router && !hybrid
        || reuse_metadata > DEFAULT_REUSE_METADATA && !hybrid && !target_rows
        || target_rows && (hybrid || router)
        || checkpoint && !target_rows
        || prepared_verify3 && (!target_rows || hybrid || router)
    {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    let reserved = HOST_COMPONENTS
        .checked_add(if hybrid { HYBRID_HOST } else { 0 })
        .and_then(|n| n.checked_add(reuse_metadata))
        .and_then(|n| {
            n.checked_add(if router {
                batch_moe::RESIDENT_ROUTER_HOST
            } else {
                0
            })
        })
        .and_then(|n| {
            n.checked_add(if target_rows {
                target_rows::HOST_INCREMENT
            } else {
                0
            })
        })
        .and_then(|n| {
            n.checked_add(if checkpoint {
                latest_checkpoint::HOST_INCREMENT
            } else {
                0
            })
        })
        .and_then(|n| {
            n.checked_add(if prepared_verify3 {
                batch_moe::prepared_verify3::HOST_INCREMENT
            } else {
                0
            })
        })
        .ok_or(Error::ArithmeticOverflow)?;
    let limit = HOST_CAP
        .checked_add(reuse_metadata - DEFAULT_REUSE_METADATA)
        .and_then(|n| {
            n.checked_add(if router {
                batch_moe::RESIDENT_ROUTER_HOST
            } else {
                0
            })
        })
        .and_then(|n| {
            n.checked_add(if target_rows {
                target_rows::HOST_INCREMENT
            } else {
                0
            })
        })
        .ok_or(Error::ArithmeticOverflow)?;
    if reserved > limit {
        return Err(Error::LimitExceeded(
            "native host and runtime metadata plan",
        ));
    }
    Ok((reserved, limit))
}
// Two extra scratch clears per layer; replacement merge/fold plus the new
// owning packet fold have zero net INFO growth. Reserve128 rather than96.
// Count full INFO source/target/inspection ownership, both bridge upload pages
// with duplicate inspection, packet readback and duplicated slab padding. The
// prepared integer pair adds96 records separately; no interior-fence/cache
// credit or existing128-record clear allowance is spent on those prepares.
const HYBRID_IO_DERIVED: u64 = 128 * 2048
    + 4 * HYBRID_MAIN
    + 2 * 32768
    + 4 * 8192
    + HYBRID_PREPARE_INFO
    + NATIVE_EXPERT_ROLE_INFO;
const _: () = {
    assert!(HYBRID_IO_DERIVED <= HYBRID_RUNTIME);
};

fn checked_cpu_miss_total(current: u64, completed_chunk: u64) -> Result<Zeroizing<u64>> {
    if completed_chunk > (LAYERS * 10) as u64 {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    current
        .checked_add(completed_chunk)
        .map(Zeroizing::new)
        .ok_or(Error::ArithmeticOverflow)
}

fn error() -> Error {
    Error::InvalidFormat("native model chunk rejected")
}

/// Not Clone/Copy and constructible only after the model's completed boundary
/// validation below. Public policy setters borrow it or immediately consume it;
/// no caller can construct or retain a copy of this completed-boundary proof.
pub(crate) struct CompletedNativeProjectionBoundary {
    _private: (),
}

/// Public fixed-fixture policy, selected only between completed transactions.
/// Runtime token counts never select a shader specialization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum NativeDecodeKernels {
    #[default]
    Chunk,
    SingleMoe,
    SingleGdn,
    SingleAll,
    SingleQ4All,
    // Actual public target state; deliberately absent from the count1 parser.
    ThreeGdn,
    FourGdn,
}
impl NativeDecodeKernels {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "chunk" => Ok(Self::Chunk),
            "single-moe" => Ok(Self::SingleMoe),
            "single-gdn" => Ok(Self::SingleGdn),
            "single-all" => Ok(Self::SingleAll),
            "single-q4-all" => Ok(Self::SingleQ4All),
            _ => Err(error()),
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::SingleMoe => "single-moe",
            Self::SingleGdn => "single-gdn",
            Self::SingleAll => "single-all",
            Self::SingleQ4All => "single-q4-all",
            Self::ThreeGdn => "three-gdn-streaming-public",
            Self::FourGdn => "four-gdn-streaming-public",
        }
    }
    fn single_moe(self) -> bool {
        matches!(self, Self::SingleMoe | Self::SingleAll | Self::SingleQ4All)
    }
    fn recurrence(self) -> crate::gdn_chunk::RecurrencePolicy {
        if self == Self::FourGdn {
            crate::gdn_chunk::RecurrencePolicy::FourStreaming
        } else if self == Self::ThreeGdn {
            crate::gdn_chunk::RecurrencePolicy::ThreeStreaming
        } else if matches!(self, Self::SingleGdn | Self::SingleAll | Self::SingleQ4All) {
            crate::gdn_chunk::RecurrencePolicy::SingleStreaming
        } else {
            crate::gdn_chunk::RecurrencePolicy::Chunk
        }
    }
    fn validate_count(self, count: usize) -> Result<()> {
        let accepted = match self {
            Self::Chunk => (1..=MAX_BATCH).contains(&count),
            Self::ThreeGdn => (1..=3).contains(&count),
            Self::FourGdn => (1..=crate::strata_port::verify_capacity()).contains(&count),
            _ => count == 1,
        };
        if !accepted {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(())
    }
}
fn sum(pages: &[u64]) -> Result<u64> {
    pages.iter().try_fold(0_u64, |n, &p| {
        n.checked_add(p).ok_or(Error::ArithmeticOverflow)
    })
}

/// Constructed from the same already-selected public lab client/adapter limits.
/// No device is selected/created here; all component owners recheck that client.
#[derive(Clone)]
pub(crate) struct Config {
    client: ComputeClient<WgpuRuntime>,
    limits: wgpu::Limits,
    tile: usize,
    cpu_operator: Option<Arc<kuno_text::cpu_expert::ExpertOperator>>,
    cpu_gpu_overlap: bool,
    request_reuse_metadata: usize,
    resident_router_reserved: bool,
    target_rows_reserved: bool,
    public_four_rows_reserved: bool,
    latest_target_checkpoint_reserved: bool,
    prepared_verify3_experts_reserved: bool,
    public_existing_expert_fill: bool,
    public_warm_replay: bool,
    public_target_request_io: bool,
    public_language_request_io: Option<public_language_request_io::Geometry>,
    live_member_reduction: bool,
    public_warm_batch_router: bool,
    public_gpu_qsa_selector: bool,
    public_frozen_host_sources: bool,
    public_frozen_host_router: bool,
    public_frozen_host_staging: bool,
    public_frozen_gu_integer: bool,
    public_frozen_down_integer: bool,
    public_grouped_ple_reads_reserved: bool,
    public_grouped_status_folds_reserved: bool,
}
fn prepared_verify3_configuration(
    target_rows: bool,
    already_reserved: bool,
    hybrid: bool,
    router: bool,
    tile: usize,
) -> Result<()> {
    if !target_rows || already_reserved || hybrid || router || tile != 8 {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok(())
}
fn live_member_reduction_configuration(
    selected: bool,
    target_rows: bool,
    hybrid: bool,
    router: bool,
    prepared_verify3: bool,
    tile: usize,
) -> Result<()> {
    if selected && (!target_rows || hybrid || router || prepared_verify3 || tile != 8) {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok(())
}
fn gpu_qsa_selector_configuration(
    selected: bool,
    target_rows: bool,
    hybrid: bool,
    router: bool,
    tile: usize,
) -> Result<()> {
    if selected && (!target_rows || hybrid || router || tile != 8) {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)] // Independent construction policy axes.
fn frozen_host_source_configuration(
    selected: bool,
    target_rows: bool,
    hybrid: bool,
    router: bool,
    prepared: bool,
    batch_router: bool,
    live_members: bool,
    tile: usize,
) -> Result<()> {
    if selected
        && (!target_rows
            || hybrid
            || router
            || prepared
            || batch_router
            || live_members
            || tile != 8)
    {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok(())
}
fn frozen_staging_configuration(selected: bool, sources: bool, router: bool) -> Result<()> {
    if selected && (!sources || !router) {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok(())
}
fn frozen_gu_integer_configuration(
    selected: bool,
    sources: bool,
    router: bool,
    stage: bool,
) -> Result<()> {
    if selected && (!sources || !router || stage) {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok(())
}
impl Config {
    fn validate_public_frozen_down_integer(&self) -> Result<()> {
        if self.public_frozen_down_integer
            && (!self.target_rows_reserved
                || !self.latest_target_checkpoint_reserved
                || self.public_target_request_io == self.public_language_request_io.is_some()
                || !self.public_frozen_host_sources
                || !self.public_frozen_host_router
                || self.public_frozen_host_staging
                || self.cpu_operator.is_some()
                || self.resident_router_reserved
                || self.prepared_verify3_experts_reserved
                || self.live_member_reduction
                || self.tile != 8)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(())
    }
    fn validate_public_grouped_ple_reads(&self) -> Result<()> {
        if self.public_grouped_ple_reads_reserved
            && (!self.target_rows_reserved
                || !self.latest_target_checkpoint_reserved
                || self.public_target_request_io == self.public_language_request_io.is_some()
                || !self.public_frozen_host_sources
                || !self.public_frozen_host_router
                || self.cpu_operator.is_some()
                || self.resident_router_reserved
                || self.prepared_verify3_experts_reserved
                || self.live_member_reduction
                || self.tile != 8)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(())
    }
    fn validate_public_grouped_status_folds(&self) -> Result<()> {
        if self.public_grouped_status_folds_reserved
            && (!self.target_rows_reserved
                || !self.latest_target_checkpoint_reserved
                || self.public_target_request_io == self.public_language_request_io.is_some()
                || !self.public_frozen_host_sources
                || !self.public_frozen_host_router
                || self.cpu_operator.is_some()
                || self.resident_router_reserved
                || self.prepared_verify3_experts_reserved
                || self.live_member_reduction
                || self.tile != 8
                || crate::status_fold_batch::selected())
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(())
    }
    fn qsa_host_reservation(&self) -> Result<(usize, usize)> {
        self.validate_public_frozen_down_integer()?;
        self.validate_public_grouped_ple_reads()?;
        self.validate_public_grouped_status_folds()?;
        if self.public_frozen_host_router
            && (!self.public_frozen_host_sources
                || !self.target_rows_reserved
                || !(self.public_target_request_io || self.public_language_request_io.is_some())
                || self.public_warm_batch_router
                || self.cpu_operator.is_some()
                || self.resident_router_reserved
                || self.prepared_verify3_experts_reserved
                || self.live_member_reduction
                || self.tile != 8)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        frozen_staging_configuration(
            self.public_frozen_host_staging,
            self.public_frozen_host_sources,
            self.public_frozen_host_router,
        )?;
        frozen_gu_integer_configuration(
            self.public_frozen_gu_integer,
            self.public_frozen_host_sources,
            self.public_frozen_host_router,
            self.public_frozen_host_staging,
        )?;
        let (reserved, limit) = host_reservation_with_verify3(
            self.cpu_operator.is_some(),
            self.request_reuse_metadata,
            self.resident_router_reserved,
            self.target_rows_reserved,
            self.latest_target_checkpoint_reserved,
            self.prepared_verify3_experts_reserved,
        )?;
        let extra = if self.public_gpu_qsa_selector {
            usize::try_from(qsa::EXTRA_HOST_BYTES_PER_CHUNK)
                .map_err(|_| Error::ArithmeticOverflow)?
        } else {
            0
        };
        let extra = extra
            .checked_add(if self.public_frozen_host_sources {
                usize::try_from(batch_moe::frozen_sources::EXTRA_HOST_BYTES_PER_CHUNK)
                    .map_err(|_| Error::ArithmeticOverflow)?
            } else {
                0
            })
            .and_then(|n| {
                n.checked_add(if self.public_frozen_host_router {
                    usize::try_from(batch_moe::frozen_sources::ROUTER_HOST_BYTES).ok()?
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if self.public_frozen_gu_integer {
                    batch_moe::FROZEN_GU_INTEGER_HOST_BYTES
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if self.public_frozen_down_integer {
                    batch_moe::FROZEN_DOWN_INTEGER_HOST_BYTES
                } else {
                    0
                })
            })
            .ok_or(Error::ArithmeticOverflow)?;
        Ok((
            reserved
                .checked_add(extra)
                .ok_or(Error::ArithmeticOverflow)?,
            limit.checked_add(extra).ok_or(Error::ArithmeticOverflow)?,
        ))
    }
    pub(crate) fn new(
        client: &ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        tile: usize,
    ) -> Result<Self> {
        validate_limits(limits, tile)?;
        Ok(Self {
            client: client.clone(),
            limits: limits.clone(),
            tile,
            cpu_operator: None,
            cpu_gpu_overlap: false,
            request_reuse_metadata: DEFAULT_REUSE_METADATA,
            resident_router_reserved: false,
            target_rows_reserved: false,
            public_four_rows_reserved: false,
            latest_target_checkpoint_reserved: false,
            prepared_verify3_experts_reserved: false,
            public_existing_expert_fill: false,
            public_warm_replay: false,
            public_target_request_io: false,
            public_language_request_io: None,
            live_member_reduction: false,
            public_warm_batch_router: false,
            public_gpu_qsa_selector: false,
            public_frozen_host_sources: false,
            public_frozen_host_router: false,
            public_frozen_host_staging: false,
            public_frozen_gu_integer: false,
            public_frozen_down_integer: false,
            public_grouped_ple_reads_reserved: false,
            public_grouped_status_folds_reserved: false,
        })
    }
    /// Construction-fixed, one-shot empty PRIVATE-state replay after the
    /// successful compiled-public thirteen-row run. Public resident weights
    /// stay owned; no ordinary request/recovery/epoch-reset API is installed.
    #[allow(dead_code)] // Parent owns the exclusive protected public runner.
    pub(crate) fn reserve_public_warm_replay(mut self) -> Result<Self> {
        public_warm_replay::configuration(
            self.target_rows_reserved,
            self.latest_target_checkpoint_reserved,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.tile,
            self.public_warm_replay,
        )?;
        self.public_warm_replay = true;
        Ok(self)
    }
    pub(crate) fn with_public_existing_expert_fill(mut self) -> Result<Self> {
        prepared_verify3_configuration(
            self.target_rows_reserved,
            self.public_existing_expert_fill,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.tile,
        )?;
        self.public_existing_expert_fill = true;
        Ok(self)
    }
    /// Construction-fixed public target selector. Selected blocks are exactly
    /// chronological within the original2048-token budget; longer contexts
    /// retain the original guarded CPU selector. No new private MAIN owners.
    pub(crate) fn reserve_public_gpu_qsa_selector(mut self) -> Result<Self> {
        if self.public_gpu_qsa_selector {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        gpu_qsa_selector_configuration(
            true,
            self.target_rows_reserved,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.tile,
        )?;
        self.public_gpu_qsa_selector = true;
        Ok(self)
    }
    /// Precharge immutable public host sources before the ordinary model
    /// constructor. This grants no completion or physical import authority.
    pub(crate) fn reserve_public_frozen_host_sources(mut self) -> Result<Self> {
        if self.public_frozen_host_sources {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        frozen_host_source_configuration(
            true,
            self.target_rows_reserved,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.prepared_verify3_experts_reserved,
            self.public_warm_batch_router,
            self.live_member_reduction,
            self.tile,
        )?;
        self.public_frozen_host_sources = true;
        Ok(self)
    }
    /// Source-bound GPU routing uses the authenticated full512 source table.
    /// Construction grants no activation authority before complete attachment.
    pub(crate) fn reserve_public_frozen_host_batch_router(mut self) -> Result<Self> {
        if self.public_frozen_host_router
            || !self.public_frozen_host_sources
            || !self.target_rows_reserved
            || !(self.public_target_request_io || self.public_language_request_io.is_some())
            || self.public_warm_batch_router
            || self.cpu_operator.is_some()
            || self.resident_router_reserved
            || self.prepared_verify3_experts_reserved
            || self.live_member_reduction
            || self.tile != 8
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.public_frozen_host_router = true;
        Ok(self)
    }
    /// Bounded original packed host-weight copies; charge before device owners.
    pub(crate) fn reserve_public_frozen_host_staging(mut self) -> Result<Self> {
        if self.public_frozen_host_staging {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        frozen_staging_configuration(
            true,
            self.public_frozen_host_sources,
            self.public_frozen_host_router,
        )?;
        self.qsa_host_reservation()?;
        if self.public_frozen_gu_integer || self.public_frozen_down_integer {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.public_frozen_host_staging = true;
        Ok(self)
    }
    /// Explicit private preparation page for the immutable public GU candidate.
    /// The original target rows/router must be independently precharged first.
    #[allow(dead_code)] // Root owns fixed-public controller registration.
    pub(crate) fn reserve_public_frozen_gu_integer(mut self) -> Result<Self> {
        if self.public_frozen_gu_integer {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        frozen_gu_integer_configuration(
            true,
            self.public_frozen_host_sources,
            self.public_frozen_host_router,
            self.public_frozen_host_staging,
        )?;
        self.public_frozen_gu_integer = true;
        self.qsa_host_reservation()?;
        Ok(self)
    }
    /// Independent full route preparation owner for the alternate DOWN math.
    #[allow(dead_code)] // Fixed-public controller registration is separate.
    pub(crate) fn reserve_public_frozen_down_integer(mut self) -> Result<Self> {
        if self.public_frozen_down_integer {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.public_frozen_down_integer = true;
        self.qsa_host_reservation()?;
        Ok(self)
    }
    /// Construction-fixed original-F32 reduction of actual grouped members.
    /// The complete MAX8 roots, original ranks, INFO charges and all owning
    /// finite/status/padding proofs remain unchanged. Parent owns the fixed
    /// public DASLab registration and physical numerical/performance gates.
    #[allow(dead_code)]
    pub(crate) fn with_live_member_reduction(mut self) -> Result<Self> {
        if self.live_member_reduction {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        live_member_reduction_configuration(
            true,
            self.target_rows_reserved,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.prepared_verify3_experts_reserved,
            self.tile,
        )?;
        self.live_member_reduction = true;
        Ok(self)
    }
    pub(crate) fn reserve_public_warm_batch_router(mut self) -> Result<Self> {
        if self.public_warm_batch_router
            || !self.target_rows_reserved
            || !self.public_warm_replay
            || !self.public_target_request_io
            || self.cpu_operator.is_some()
            || self.resident_router_reserved
            || self.prepared_verify3_experts_reserved
            || self.public_frozen_host_sources
            || self.public_frozen_host_router
            || self.tile != 8
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.public_warm_batch_router = true;
        Ok(self)
    }
    pub(crate) fn reserve_resident_router(mut self) -> Result<Self> {
        if self.cpu_operator.is_none()
            || !self.cpu_gpu_overlap
            || self.tile != 8
            || self.resident_router_reserved
            || self.target_rows_reserved
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        host_reservation_with_router(true, self.request_reuse_metadata, true)?;
        self.resident_router_reserved = true;
        Ok(self)
    }
    /// Fixed-public construction-only output capacity. No hybrid owners or
    /// count1 policies are installed; the caller configures the matching
    /// protected runtime ceiling before constructing the whole native owner.
    #[allow(dead_code)] // Parent owns the dedicated fixed-public registration.
    pub(crate) fn reserve_target_rows(mut self) -> Result<Self> {
        if self.target_rows_reserved
            || self.cpu_operator.is_some()
            || self.resident_router_reserved
            || self.tile != 8
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        host_reservation_with_outputs(false, self.request_reuse_metadata, false, true)?;
        self.target_rows_reserved = true;
        Ok(self)
    }
    #[allow(dead_code)] // Controller is integrated separately after source acceptance.
    pub(crate) fn reserve_public_four_rows(mut self) -> Result<Self> {
        if self.public_four_rows_reserved
            || !self.target_rows_reserved
            || self.tile != 8
            || self.cpu_operator.is_some()
            || self.resident_router_reserved
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.public_four_rows_reserved = true;
        Ok(self)
    }
    pub(crate) fn reserve_request_reuse_metadata(mut self, bytes: usize) -> Result<Self> {
        host_reservation_with_verify3(
            self.cpu_operator.is_some(),
            bytes,
            self.resident_router_reserved,
            self.target_rows_reserved,
            self.latest_target_checkpoint_reserved,
            self.prepared_verify3_experts_reserved,
        )?;
        self.request_reuse_metadata = bytes;
        Ok(self)
    }
    /// Fixed PUBLIC target stream only. The backend is configured separately
    /// before construction; no process setting or draft stream is changed.
    #[allow(dead_code)] // Protected registration and GPU acceptance are parent-owned.
    pub(crate) fn reserve_public_target_request_io(mut self) -> Result<Self> {
        public_target_request_io::configuration(
            self.target_rows_reserved,
            self.public_target_request_io,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.tile,
        )?;
        public_target_request_io::check_runtime(&self.client)?;
        self = self.reserve_request_reuse_metadata(public_target_request_io::METADATA_BYTES)?;
        self.public_target_request_io = true;
        Ok(self)
    }
    pub(crate) fn reserve_public_language_request_io(
        mut self,
        prompt: usize,
        outputs: usize,
    ) -> Result<Self> {
        public_language_request_io::configuration(
            self.target_rows_reserved,
            self.latest_target_checkpoint_reserved,
            self.public_language_request_io.is_some() || self.public_target_request_io,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.public_warm_replay,
            self.tile,
        )?;
        let geometry = public_language_request_io::Geometry::new(prompt, outputs)?;
        public_language_request_io::check_runtime(&self.client)?;
        self = self.reserve_request_reuse_metadata(public_language_request_io::METADATA_BYTES)?;
        self.public_language_request_io = Some(geometry);
        Ok(self)
    }
    /// Permission only: the actual read policy stays off through prefill.
    /// Same PLE owners and 4MiB host/owning I/O guards; no reservation increase.
    #[allow(dead_code)] // Root owns public parser/controller registration.
    pub(crate) fn reserve_public_grouped_ple_reads(mut self) -> Result<Self> {
        if self.public_grouped_ple_reads_reserved {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.public_grouped_ple_reads_reserved = true;
        self.qsa_host_reservation()?;
        Ok(self)
    }
    /// Permission only: grouped status dispatch remains off through prefill.
    /// Original workspace owners, INFO envelope and full audits are unchanged.
    #[allow(dead_code)] // Root owns fixed-public controller registration.
    pub(crate) fn reserve_public_grouped_status_folds(mut self) -> Result<Self> {
        if self.public_grouped_status_folds_reserved {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.public_grouped_status_folds_reserved = true;
        self.qsa_host_reservation()?;
        Ok(self)
    }
    /// One latest-chunk host checkpoint plus exact QSA suffix-clear/fold INFO.
    /// This grants no new MAIN owner, arithmetic policy or private inference API.
    #[allow(dead_code)] // Parent owns the fixed-public target repair registration.
    pub(crate) fn reserve_latest_target_checkpoint(mut self) -> Result<Self> {
        latest_round::checkpoint_configuration(
            self.target_rows_reserved,
            self.latest_target_checkpoint_reserved,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.tile,
        )?;
        host_reservation_with_verify3(
            false,
            self.request_reuse_metadata,
            false,
            true,
            true,
            self.prepared_verify3_experts_reserved,
        )?;
        self.latest_target_checkpoint_reserved = true;
        Ok(self)
    }
    /// Construction-fixed alternate activation arithmetic for1..3 PUBLIC
    /// target rows. No global count1 integer flag or hybrid/router fallback.
    #[allow(dead_code)] // Protected flag/physical acceptance are parent-owned.
    pub(crate) fn reserve_prepared_verify3_experts(mut self) -> Result<Self> {
        if self.live_member_reduction {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        prepared_verify3_configuration(
            self.target_rows_reserved,
            self.prepared_verify3_experts_reserved,
            self.cpu_operator.is_some(),
            self.resident_router_reserved,
            self.tile,
        )?;
        batch_moe::prepared_verify3::validate_limits(&self.limits)?;
        host_reservation_with_prepared_verify3(
            self.request_reuse_metadata,
            self.latest_target_checkpoint_reserved,
        )?;
        self.prepared_verify3_experts_reserved = true;
        Ok(self)
    }
    /// Explicit fixed-public laboratory reservation. Ordinary constructors keep
    /// their original owner inventory; reservation alone never places a rank.
    pub(crate) fn reserve_count1_hybrid(self) -> Result<Self> {
        self.reserve_count1_hybrid_workers(false)
    }
    pub(crate) fn reserve_count1_hybrid_workers(self, pooled: bool) -> Result<Self> {
        self.reserve_count1_hybrid_execution(pooled, false)
    }
    pub(crate) fn reserve_count1_hybrid_execution(
        self,
        pooled: bool,
        overlap: bool,
    ) -> Result<Self> {
        self.reserve_count1_hybrid_arithmetic(pooled, overlap, false, false, false, 16)
    }
    pub(crate) fn reserve_count1_hybrid_arithmetic(
        mut self,
        pooled: bool,
        overlap: bool,
        q8: bool,
        residual: bool,
        prefetch: bool,
        threads: usize,
    ) -> Result<Self> {
        use kuno_text::cpu_expert::{Backend, ExpertOperator, WorkerPlan};
        if self.cpu_operator.is_some()
            || self.target_rows_reserved
            || self.tile != 8
            || ((overlap || q8) && !pooled)
            || (residual && !q8)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let plan = WorkerPlan::physical(threads)?;
        let backend = if cfg!(feature = "cpu-fma") {
            Backend::Avx2Fma
        } else {
            Backend::Avx2
        };
        let operator = if pooled {
            ExpertOperator::new_pooled(plan, backend)?
        } else {
            ExpertOperator::new(plan, backend)?
        };
        let operator = if cfg!(feature = "cpu-buffered-weights") {
            operator.with_buffered_weights()?
        } else {
            operator
        };
        let operator = if prefetch {
            operator.with_mapped_prefetch()
        } else {
            operator
        };
        let operator = operator.with_phase_diagnostics();
        self.cpu_operator = Some(Arc::new(if residual {
            operator.with_q8_residual()?
        } else if q8 {
            operator.with_q8_activations()?
        } else {
            operator
        }));
        self.cpu_gpu_overlap = overlap;
        Ok(self)
    }
}
fn validate_limits(l: &wgpu::Limits, tile: usize) -> Result<()> {
    if ![4, 8].contains(&tile)
        || l.max_compute_invocations_per_workgroup < 64
        || l.max_compute_workgroup_size_x < 64
        || l.max_compute_workgroups_per_dimension < 16384
        || l.max_compute_workgroup_storage_size < 8192
        || l.max_bind_groups < 1
        || l.max_storage_buffers_per_shader_stage < 11
        || l.max_bindings_per_bind_group < 12
        || l.max_uniform_buffer_binding_size < 512
        || l.max_storage_buffer_binding_size < 128 * 1024 * 1024
        || l.max_buffer_size < 128 * 1024 * 1024
        || l.min_storage_buffer_offset_alignment == 0
        || l.min_uniform_buffer_offset_alignment == 0
        || l.min_storage_buffer_offset_alignment > 512
        || l.min_uniform_buffer_offset_alignment > 512
    {
        return Err(Error::LimitExceeded("native model device inventory"));
    }
    Ok(())
}

pub(super) struct Plan {
    pub(super) requests: Vec<u64>,
    pub(super) config: Config,
    pub(super) runtime: u64,
}
impl Plan {
    pub(super) fn new(
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
        executor: &GpuExecutor<'_>,
        config: Config,
        old: &[u64],
        context: usize,
    ) -> Result<Self> {
        graph.verify_unchanged()?;
        if !std::ptr::eq(client.info(), config.client.info())
            || sum(old)? != OLD_MAIN
            || old.len() != 1286 + gdn::strata_replay::EXTRA_BUFFERS as usize
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        validate_limits(&config.limits, config.tile)?;
        if config.public_frozen_host_staging
            && (config.limits.max_compute_invocations_per_workgroup < 256
                || config.limits.max_compute_workgroup_size_x < 256)
        {
            return Err(Error::LimitExceeded(
                "frozen host staging copy device limits",
            ));
        }
        if config.public_frozen_gu_integer {
            crate::quant_matvec::frozen_gu_integer::validate_limits(&config.limits)?;
        }
        if config.public_frozen_down_integer {
            crate::quant_matvec::frozen_down_integer::validate_limits(&config.limits)?;
        }
        config.qsa_host_reservation()?;
        frozen_host_source_configuration(
            config.public_frozen_host_sources,
            config.target_rows_reserved,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.prepared_verify3_experts_reserved,
            config.public_warm_batch_router,
            config.live_member_reduction,
            config.tile,
        )?;
        gpu_qsa_selector_configuration(
            config.public_gpu_qsa_selector,
            config.target_rows_reserved,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.tile,
        )?;
        live_member_reduction_configuration(
            config.live_member_reduction,
            config.target_rows_reserved,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.prepared_verify3_experts_reserved,
            config.tile,
        )?;
        batch_moe::live_member_reduction_policy(
            config.live_member_reduction,
            executor.packed_profile,
            executor.resident_bank_capacity(),
            config.tile,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.prepared_verify3_experts_reserved,
        )?;
        if config.public_warm_batch_router {
            let (host, limit) = host_reservation_with_verify3(
                false,
                config.request_reuse_metadata,
                false,
                true,
                true,
                false,
            )?;
            let host = host
                .checked_add(batch_moe::batch_router::HOST_BYTES)
                .ok_or(Error::ArithmeticOverflow)?;
            let limit = limit
                .checked_add(batch_moe::batch_router::HOST_BYTES)
                .ok_or(Error::ArithmeticOverflow)?;
            if host > limit {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        }
        if config.public_four_rows_reserved != executor.target_four_rows_reserved
            || config.public_four_rows_reserved
                && (!config.target_rows_reserved
                    || !config.latest_target_checkpoint_reserved
                    || !config.public_frozen_host_sources
                    || !config.public_frozen_host_router
                    || !config.public_frozen_gu_integer
                    || !config.public_frozen_down_integer
                    || config.public_frozen_host_staging
                    || config.prepared_verify3_experts_reserved
                    || config.live_member_reduction)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if config.public_frozen_host_sources
            && (executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || executor.resident_bank_capacity() != crate::daslab_profile::BANK_SLOTS
                || executor.target_frozen_host_source_identity.is_some()
                || executor.resident_banks_frozen)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        executor.check_original_bank_metadata(LAYERS, ORIGINAL_BANK_HOST)?;
        let g = graph.geometry();
        if g.layers != LAYERS
            || g.residual.width != D
            || g.residual.streams != HC
            || g.experts != ROUTER
            || g.experts_used != 10
            || g.max_context != 4096
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let mut requests = Vec::new();
        for (pages, expected) in [
            (
                gr::Workspace::allocation_requests(graph)?.to_vec(),
                1_998_848 + NATIVE_GR_MAIN,
            ),
            (
                gdn::GdnBatchWorkspace::allocation_requests(graph)?.to_vec(),
                3_932_160,
            ),
            (
                qsa::QsaBatchWorkspace::allocation_requests(graph)?.to_vec(),
                17_825_792 + NATIVE_QSA_BATCH_MAIN,
            ),
            (
                executor
                    .routed_moe_batch_allocation_requests(graph, config.tile)?
                    .to_vec(),
                1_409_024,
            ),
            (
                ple_batch::PleBatchWorkspace::allocation_requests_with_profile(
                    graph,
                    executor.packed_profile,
                )?
                .to_vec(),
                6_258_688,
            ),
            (
                shared_batch::Workspace::allocation_requests(graph)?.to_vec(),
                262_144,
            ),
        ] {
            if sum(&pages)? != expected {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            requests.extend(pages);
        }
        for words in [
            RESIDUAL_WORDS,
            RESIDUAL_WORDS,
            BLOCK_WORDS,
            BLOCK_WORDS,
            MAX_BATCH * ROUTER,
            MAX_BATCH * ROUTER + 1,
        ] {
            requests.push(BufferLayout::new(words).map_err(u::fault)?.owned_bytes as u64);
        }
        let hybrid = config.cpu_operator.is_some();
        public_target_request_io::check_plan(&config, executor)?;
        if config.public_language_request_io.is_some() {
            public_language_request_io::configuration(
                config.target_rows_reserved,
                config.latest_target_checkpoint_reserved,
                config.public_target_request_io,
                hybrid,
                config.resident_router_reserved,
                config.public_warm_replay,
                config.tile,
            )?;
            public_language_request_io::check_runtime(client)?;
            if config.request_reuse_metadata != public_language_request_io::METADATA_BYTES
                || !executor.target_rows_reserved
                || !executor.token_io_checkpoint
                || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || executor.resident_bank_capacity() != crate::daslab_profile::BANK_SLOTS
                || executor.resident_runtime_io_limit() != public_language_request_io::RUNTIME_BYTES
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        }

        host_reservation_with_verify3(
            hybrid,
            config.request_reuse_metadata,
            config.resident_router_reserved,
            config.target_rows_reserved,
            config.latest_target_checkpoint_reserved,
            config.prepared_verify3_experts_reserved,
        )?;
        if config.public_warm_batch_router {
            let (host, limit) = host_reservation_with_verify3(
                false,
                config.request_reuse_metadata,
                false,
                true,
                true,
                false,
            )?;
            let host = host
                .checked_add(batch_moe::batch_router::HOST_BYTES)
                .ok_or(Error::ArithmeticOverflow)?;
            let limit = limit
                .checked_add(batch_moe::batch_router::HOST_BYTES)
                .ok_or(Error::ArithmeticOverflow)?;
            if host > limit {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        }
        if hybrid {
            requests.extend(batch_moe::Workspace::count1_hybrid_allocation_requests()?);
        }
        let router = config.resident_router_reserved;
        let target_rows = config.target_rows_reserved;
        if config.latest_target_checkpoint_reserved {
            latest_round::checkpoint_configuration(
                target_rows,
                false,
                hybrid,
                router,
                config.tile,
            )?;
        }
        executor.check_target_rows_policy()?;
        if target_rows != executor.target_rows_reserved
            || target_rows
                && (hybrid
                    || router
                    || config.tile != 8
                    || executor.native_projection_member_capacity() != 8
                    || executor.native_projection_tile() != 8)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if router != executor.resident_router_reserved
            || router
                && (!hybrid
                    || !executor.packed_profile.is_low_bit()
                    || executor.resident_bank_capacity() != crate::daslab_profile::BANK_SLOTS)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if router {
            requests.extend(batch_moe::Workspace::resident_router_allocation_requests());
        }
        if target_rows {
            requests.extend(target_rows::Owners::allocation_requests()?);
        }
        let prepared_verify3 = config.prepared_verify3_experts_reserved;
        if prepared_verify3 {
            if !target_rows
                || hybrid
                || router
                || config.tile != 8
                || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || executor.resident_bank_capacity() != crate::daslab_profile::BANK_SLOTS
                || executor.q3_integer_moe
                || executor.resident_banks_frozen
                || executor.adaptive.is_some()
                || executor.secondary.is_some()
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            batch_moe::prepared_verify3::validate_limits(&config.limits)?;
            requests.push(batch_moe::prepared_verify3::MAIN_BYTES);
        }
        if config.public_existing_expert_fill
            && (!target_rows
                || hybrid
                || router
                || config.tile != 8
                || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || executor.resident_bank_capacity() != crate::daslab_profile::BANK_SLOTS
                || executor.resident_banks_frozen
                || executor.adaptive.is_some()
                || executor.secondary.is_some()
                || executor.secondary_frequency.is_some())
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if config.public_warm_batch_router {
            if !config.public_warm_replay
                || !config.public_target_request_io
                || !target_rows
                || hybrid
                || router
                || prepared_verify3
                || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || executor.resident_bank_capacity() != crate::daslab_profile::BANK_SLOTS
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            requests.extend(batch_moe::Workspace::batch_router_allocation_requests());
        }
        if config.public_frozen_host_router {
            if !config.public_frozen_host_sources || config.public_warm_batch_router {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            requests.extend(batch_moe::Workspace::batch_router_allocation_requests());
        }
        if (config.public_warm_batch_router || config.public_frozen_host_router)
            != executor.target_batch_router_reserved
            || config.public_frozen_host_router != executor.target_frozen_host_router_reserved
            || config.public_frozen_host_staging != executor.target_frozen_host_staging_reserved
            || config.public_frozen_gu_integer != executor.target_frozen_gu_integer_reserved
            || config.public_frozen_down_integer != executor.target_frozen_down_integer_reserved
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if config.public_frozen_host_staging {
            requests.extend([batch_moe::FROZEN_STAGING_OWNER_BYTES; 3]);
        }
        if config.public_frozen_gu_integer {
            requests.push(batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES);
        }
        if crate::strata_intel_core::enabled() {
            requests.extend(crate::strata_global_moe::Workspace::requests_for_context(
                context,
            )?);
        }
        if config.public_frozen_down_integer {
            requests.push(batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES);
        }
        let extra = if config.public_frozen_host_router {
            batch_moe::frozen_sources::ROUTER_MAIN_BYTES
        } else if config.public_warm_batch_router {
            batch_moe::batch_router::MAIN_BYTES
        } else {
            0
        } + if config.public_frozen_host_staging {
            batch_moe::FROZEN_STAGING_MAIN_BYTES
        } else {
            0
        } + if config.public_frozen_gu_integer {
            batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES
        } else {
            0
        } + if config.public_frozen_down_integer {
            batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES
        } else {
            0
        } + if hybrid { HYBRID_MAIN } else { 0 }
            + if router {
                batch_moe::RESIDENT_ROUTER_MAIN
            } else {
                0
            }
            + if target_rows {
                target_rows::MAIN_BYTES
            } else {
                0
            }
            + if prepared_verify3 {
                batch_moe::prepared_verify3::MAIN_BYTES
            } else {
                0
            };
        let extra = extra
            + if crate::strata_intel_core::enabled() {
                crate::strata_global_moe::Workspace::main_bytes_for_context(context)?
            } else {
                0
            };
        if requests.len()
            != 90
                + if crate::strata_intel_core::enabled() {
                    crate::strata_global_moe::Workspace::requests_for_context(context)?.len()
                } else {
                    0
                }
                + usize::from(config.public_frozen_host_staging) * 3
                + usize::from(config.public_frozen_gu_integer)
                + usize::from(config.public_frozen_down_integer)
                + usize::from(hybrid) * 2
                + usize::from(router) * 2
                + usize::from(target_rows) * 2
                + usize::from(prepared_verify3)
                + usize::from(config.public_warm_batch_router || config.public_frozen_host_router)
                    * 2
            || sum(&requests)? != NEW_MAIN + extra
            || OLD_MAIN.checked_add(sum(&requests)?) != Some(TOTAL_MAIN + extra)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let head = graph.tensor("output.weight")?;
        let row = head.info().row_bytes();
        if row == 0 {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let tile_rows = (64_u64 * 1024 * 1024 / row).min(16384);
        if tile_rows == 0 {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let head_tiles = (g.vocabulary as u64).div_ceil(tile_rows);
        let runtime = runtime_bytes(head_tiles)?
            .checked_add(if hybrid { HYBRID_RUNTIME } else { 0 })
            .and_then(|n| {
                n.checked_add(if router {
                    batch_moe::resident_router_runtime(executor.q3_integer_moe)
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if target_rows {
                    target_rows::RUNTIME_INCREMENT
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if config.latest_target_checkpoint_reserved {
                    latest_checkpoint::RUNTIME_INCREMENT
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if prepared_verify3 {
                    batch_moe::prepared_verify3::RUNTIME_INCREMENT
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if config.public_frozen_host_router {
                    batch_moe::frozen_sources::ROUTER_RUNTIME_BYTES
                } else if config.public_warm_batch_router {
                    batch_moe::batch_router::RUNTIME_BYTES
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if config.public_gpu_qsa_selector {
                    qsa::EXTRA_INFO_BYTES_PER_CHUNK
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if config.public_frozen_host_sources {
                    batch_moe::frozen_sources::EXTRA_RUNTIME_BYTES_PER_CHUNK
                } else {
                    0
                })
            })
            .ok_or(Error::ArithmeticOverflow)?;
        let runtime = runtime
            .checked_add(if config.public_frozen_host_staging {
                batch_moe::FROZEN_STAGING_RUNTIME_BYTES
            } else {
                0
            })
            .and_then(|n| {
                n.checked_add(if config.public_frozen_gu_integer {
                    batch_moe::FROZEN_GU_INTEGER_RUNTIME_BYTES
                } else {
                    0
                })
            })
            .and_then(|n| {
                n.checked_add(if config.public_frozen_down_integer {
                    batch_moe::FROZEN_DOWN_INTEGER_RUNTIME_BYTES
                } else {
                    0
                })
            })
            .ok_or(Error::ArithmeticOverflow)?;
        if runtime
            .checked_add(executor.native_mtp_runtime())
            .is_none_or(|n| n > executor.resident_runtime_io_limit())
            || HOST_COMPONENTS + if hybrid { HYBRID_HOST } else { 0 } > HOST_CAP
            || metadata_bytes() > META_CAP
        {
            return Err(Error::LimitExceeded("native model aggregate private plan"));
        }
        // All component/API-visible reservations remain independent of MAIN.
        if gdn::GdnBatchWorkspace::runtime_info_reservation(graph, true)?
            != 4_718_592 + NATIVE_GDN_RUNTIME_INFO
            || qsa::QsaBatchWorkspace::runtime_io_reservation(graph, true)? != 7_864_320
            || batch_moe::Workspace::runtime_io_increment() != 524288
            || ops::GUARDED_ROUTER_CAPACITY_BYTES > 8192
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(Self {
            requests,
            config,
            runtime,
        })
    }
    pub(super) fn check_public(&self, graph: BoundGraph<'_, '_>, pages: u64) -> Result<()> {
        self.check_public_with_audit(graph, pages, false)
    }
    pub(super) fn check_public_with_audit(
        &self,
        graph: BoundGraph<'_, '_>,
        pages: u64,
        public_zero_audit: bool,
    ) -> Result<()> {
        target_rows::check_public_audit(self.config.target_rows_reserved, public_zero_audit)?;
        // prewarm() independently proves exact planned cache names/counts and
        // seals them; here the actual graph count/full-page SUM must match too.
        if graph.static_vectors().count() != 341 || pages != PUBLIC_MAIN {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let derived = (0..LAYERS).try_fold(0usize, |n, l| {
            Ok::<_, Error>(n + usize::from(graph.layer_kind(l)? == AttentionKind::Gdn))
        })?;
        if 341 + derived + 2 != 379 {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(())
    }
}
fn runtime_bytes(head_tiles: u64) -> Result<u64> {
    // Audited all-stage envelope, counting source+target+inspection separately.
    // Real forced router/interior-MoE fences drain before another source window.
    // Keep the helper's conservative embedding eight-record INFO envelope.
    // These nine actual outer/scalar clears are a separate one-time category.
    const OUTER_CLEAR_INFO: u64 = 9 * 2048;
    const WITHOUT_HEAD: u64 = 61_751_296
        + OUTER_CLEAR_INFO
        + if cfg!(feature = "strata-intel-core") {
            crate::strata_prompt::groups::RUNTIME_BYTES
                + prompt_full::RUNTIME_BYTES
                + gdn::strata_replay::PROMPT_INFO_BYTES
                + crate::strata_qsa::prompt::INFO_BYTES
                + NATIVE_GDN_RUNTIME_INFO
                + NATIVE_PREFIX_RUNTIME_INFO
        } else {
            0
        };
    let dispatches = head_tiles
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(Error::ArithmeticOverflow)?;
    let head = dispatches
        .checked_mul(2048)
        .and_then(|n| n.checked_add(2 * 1024 * 1024 + 65536))
        .ok_or(Error::ArithmeticOverflow)?;
    WITHOUT_HEAD
        .checked_add(head)
        .ok_or(Error::ArithmeticOverflow)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Purpose {
    Advance,
    LastLogits,
    AllRowsWithCarry,
}
pub(crate) enum PrefixPhase<'a> {
    Verified(&'a mut NativeTargetRows),
    /// Original verified R only; target candidate state is not yet published.
    Pending {
        source: target_rows::VerifiedCarryContext<'a>,
        accepted: usize,
    },
}
type PrefixSelection<'a> =
    Option<&'a mut dyn for<'rows> FnMut(PrefixPhase<'rows>) -> Result<usize>>;
enum ChunkOutput {
    AcceptedPrefix(usize),
    Legacy(Zeroizing<Vec<f32>>),
    #[allow(dead_code)] // Read by the additive parent-registered result method.
    TargetRows(NativeTargetRows),
}
struct Outer {
    residual: [BufferId; 2],
    block: [BufferId; 2],
    router: BufferId,
    packet: BufferId,
}
impl Zeroize for Outer {
    fn zeroize(&mut self) {
        self.residual.zeroize();
        self.block.zeroize();
        self.router.zeroize();
        self.packet.zeroize();
    }
}
fn decode_family_policy(
    tile: usize,
    recurrence: crate::gdn_chunk::RecurrencePolicy,
    q4_block: bool,
    dense_members: usize,
) -> Result<NativeDecodeKernels> {
    use crate::gdn_chunk::RecurrencePolicy::{
        Chunk, FourStreaming, SingleStreaming, ThreeStreaming,
    };
    Ok(match (tile, recurrence, q4_block, dense_members) {
        (4 | 8, Chunk, false, 8) => NativeDecodeKernels::Chunk,
        (1, Chunk, false, 8) => NativeDecodeKernels::SingleMoe,
        (4 | 8, SingleStreaming, false, 8) => NativeDecodeKernels::SingleGdn,
        (1, SingleStreaming, false, 1) => NativeDecodeKernels::SingleAll,
        (1, SingleStreaming, true, 1) => NativeDecodeKernels::SingleQ4All,
        (8, ThreeStreaming, false, 8) => NativeDecodeKernels::ThreeGdn,
        (8, FourStreaming, false, 8) => NativeDecodeKernels::FourGdn,
        _ => return Err(error()),
    })
}
#[derive(Clone, Copy)]
struct PublicThreeGdnScope {
    original_profile: bool,
    target_rows: bool,
    checkpoint: bool,
    public_request: bool,
    frozen_sources: bool,
    frozen_router: bool,
    partition_exit: bool,
    original_io: bool,
    original_dispatch: bool,
    original_families: bool,
}
impl PublicThreeGdnScope {
    fn validate(self) -> Result<()> {
        if !self.original_profile
            || !self.target_rows
            || !self.checkpoint
            || !self.public_request
            || !self.frozen_sources
            || !self.frozen_router
            || !self.partition_exit
            || !self.original_io
            || !self.original_dispatch
            || !self.original_families
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(())
    }
}
fn projection_policy_allows_tile(policy: NativeDecodeKernels, tile: usize) -> bool {
    match policy {
        NativeDecodeKernels::Chunk => true,
        NativeDecodeKernels::ThreeGdn | NativeDecodeKernels::FourGdn => tile == 8,
        _ => tile == 1,
    }
}
fn three_gdn_selection_allowed(
    policy: NativeDecodeKernels,
    accepted_prefill: bool,
    request_active: bool,
) -> bool {
    accepted_prefill
        && !request_active
        && matches!(
            policy,
            NativeDecodeKernels::Chunk | NativeDecodeKernels::ThreeGdn
        )
}
#[allow(dead_code)] // Warm-only activation is consumed by the separate public controller.
fn four_rows_selection_allowed(
    policy: NativeDecodeKernels,
    accepted_prefill: bool,
    request_active: bool,
) -> bool {
    accepted_prefill && !request_active && policy == NativeDecodeKernels::Chunk
}
fn q6_verify_head_selection_allowed(
    policy: NativeDecodeKernels,
    accepted_prefill: bool,
    request_active: bool,
) -> bool {
    accepted_prefill
        && !request_active
        && matches!(
            policy,
            NativeDecodeKernels::Chunk
                | NativeDecodeKernels::ThreeGdn
                | NativeDecodeKernels::FourGdn
        )
}
fn next_three_gdn_chunks(policy: NativeDecodeKernels, current: u64) -> Result<u64> {
    if policy == NativeDecodeKernels::ThreeGdn {
        current.checked_add(1).ok_or(Error::ArithmeticOverflow)
    } else {
        Ok(current)
    }
}
fn next_four_rows_chunks(
    policy: NativeDecodeKernels,
    count: usize,
    actual: (bool, u64, u64),
    current: u64,
) -> Result<u64> {
    if policy == NativeDecodeKernels::FourGdn && count == 4 {
        if !actual.0 || actual.1 == 0 || actual.1 != actual.2 {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        current.checked_add(1).ok_or(Error::ArithmeticOverflow)
    } else {
        Ok(current)
    }
}
fn grouped_status_states(states: [(bool, u64); 3]) -> Result<(bool, [u64; 3])> {
    if states.iter().any(|state| state.0 != states[0].0) {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok((states[0].0, states.map(|state| state.1)))
}
fn grouped_ple_selection_allowed(
    reserved: bool,
    selected: bool,
    requested: bool,
    accepted_prefill: bool,
    request_active: bool,
    policy: NativeDecodeKernels,
) -> bool {
    reserved
        && accepted_prefill
        && !request_active
        && (!selected || requested)
        && matches!(
            policy,
            NativeDecodeKernels::Chunk
                | NativeDecodeKernels::ThreeGdn
                | NativeDecodeKernels::FourGdn
        )
}
fn next_grouped_ple_chunks(selected: bool, actual: bool, current: u64) -> Result<u64> {
    if selected != actual {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    if actual {
        current.checked_add(1).ok_or(Error::ArithmeticOverflow)
    } else {
        Ok(current)
    }
}
fn grouped_ple_run_allowed(selected: bool, count: usize, purpose: Purpose) -> bool {
    !selected || (1..=3).contains(&count) && purpose == Purpose::AllRowsWithCarry
}
fn grouped_ple_run_allowed_selected(
    selected: bool,
    count: usize,
    purpose: Purpose,
    four: bool,
) -> bool {
    grouped_ple_run_allowed(selected, count, purpose)
        || four
            && (4..=crate::strata_port::verify_capacity()).contains(&count)
            && purpose == Purpose::AllRowsWithCarry
}
fn grouped_ple_completed_receipt(selected: bool, chunks: u64) -> Result<(bool, u64)> {
    if selected == (chunks == 0) {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    Ok((selected, chunks))
}
pub(super) struct Workspace {
    gr: gr::Workspace,
    gdn: gdn::GdnBatchWorkspace,
    qsa: qsa::QsaBatchWorkspace,
    moe: batch_moe::Workspace,
    ple: ple_batch::PleBatchWorkspace,
    shared: shared_batch::Workspace,
    outer: Zeroizing<Outer>,
    core_stages: Option<[BufferId; 3]>,
    core_prompt_stages: Option<[BufferId; 3]>,
    global_moe: Option<crate::strata_global_moe::Workspace>,
    poison: Arc<AtomicBool>,
    dense_member_capacity: usize,
    target_rows: Option<target_rows::Owners>,
    latest_target_checkpoint_reserved: bool,
    public_warm_replay: Option<public_warm_replay::Permit>,
    public_target_request_io: Option<public_target_request_io::Permit>,
    public_language_request_io: Option<public_language_request_io::Permit>,
    public_gpu_qsa_selector: bool,
    public_frozen_host_sources: bool,
    public_frozen_host_router: bool,
    three_gdn_completed_chunks: u64,
    public_four_rows_reserved: bool,
    four_rows_completed_chunks: u64,
    public_grouped_ple_reads_reserved: bool,
    public_grouped_status_folds_reserved: bool,
    grouped_ple_completed_chunks: u64,
    grouped_status_completed: crate::status_fold_batch::Completed,
}
impl Workspace {
    fn grouped_status_state(&self) -> Result<(bool, [u64; 3])> {
        let states = [
            self.gr.public_grouped_status_folds(),
            self.gdn.public_grouped_status_folds(),
            self.qsa.public_grouped_status_folds(),
        ];
        grouped_status_states(states)
    }
    fn decode_kernels(&self) -> Result<NativeDecodeKernels> {
        if crate::strata_intel_core::enabled() {
            if self.global_moe.is_none() || self.dense_member_capacity != 8 {
                return Err(error());
            }
            return match self.gdn.recurrence_policy() {
                crate::gdn_chunk::RecurrencePolicy::Chunk => Ok(NativeDecodeKernels::Chunk),
                crate::gdn_chunk::RecurrencePolicy::FourStreaming => {
                    Ok(NativeDecodeKernels::FourGdn)
                }
                _ => Err(error()),
            };
        }
        decode_family_policy(
            self.moe.projection_tile(),
            self.gdn.recurrence_policy(),
            self.moe.gate_policy() == crate::quant_matvec::grouped_moe::GatePolicy::Q4Block,
            self.dense_member_capacity,
        )
    }
    /// Called by every target/round/request path, including prefill. The new
    /// state remains restricted to its actual public frozen construction.
    fn validate_target_decode_policy(
        &self,
        executor: &GpuExecutor<'_>,
        admission: &crate::executor::ResidentAdmission,
    ) -> Result<()> {
        if crate::strata_intel_core::enabled() {
            return self.validate_global_scope(executor, admission);
        }
        if self.public_four_rows_reserved != executor.target_four_rows_reserved
            || executor.target_four_rows_active
                != (self.decode_kernels()? == NativeDecodeKernels::FourGdn)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if self.moe.frozen_host_source_down_integer_selected() {
            if !executor.target_frozen_down_integer_reserved
                || self.moe.frozen_down_integer_owner_id().is_none()
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            self.three_gdn_scope(executor, admission)?.validate()?;
        }
        if self.ple.public_grouped_reads() {
            if !self.public_grouped_ple_reads_reserved {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            self.three_gdn_scope(executor, admission)?.validate()?;
        }
        if self.grouped_status_state()?.0 {
            if !self.public_grouped_status_folds_reserved || crate::status_fold_batch::selected() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            self.three_gdn_scope(executor, admission)?.validate()?;
        }
        match self.decode_kernels()? {
            NativeDecodeKernels::Chunk => Ok(()),
            NativeDecodeKernels::ThreeGdn => self.three_gdn_scope(executor, admission)?.validate(),
            NativeDecodeKernels::FourGdn => {
                if !self.public_four_rows_reserved
                    || !executor.target_four_rows_active
                    || !executor.target_four_rows_reserved
                    || !self.moe.public_four_members_policy()
                {
                    return Err(Error::InvalidFormat(concat!(
                        "native model chunk rejected at source line ",
                        line!()
                    )));
                }
                self.three_gdn_scope(executor, admission)?.validate()
            }
            _ => Err(error()),
        }
    }
    fn validate_execution_scope(
        &self,
        executor: &GpuExecutor<'_>,
        admission: &crate::executor::ResidentAdmission,
    ) -> Result<()> {
        if crate::strata_intel_core::enabled() {
            self.validate_global_scope(executor, admission)
        } else {
            self.three_gdn_scope(executor, admission)?.validate()
        }
    }
    fn validate_global_scope(
        &self,
        executor: &GpuExecutor<'_>,
        admission: &crate::executor::ResidentAdmission,
    ) -> Result<()> {
        executor.check_resident_admission(admission)?;
        if !executor.global_target_scope()?
            || !executor.target_rows_reserved
            || self.target_rows.is_none()
            || !self.latest_target_checkpoint_reserved
            || !executor.token_io_checkpoint
            || !admission.deferred_token_io()
            || !admission.transient_io()
            || self.public_language_request_io.is_none()
            || self.public_target_request_io.is_some()
            || self.public_four_rows_reserved != executor.target_four_rows_reserved
            || executor.target_four_rows_active
                != (self.decode_kernels()? == NativeDecodeKernels::FourGdn)
            || executor.target_four_rows_active && !self.public_four_rows_reserved
            || executor.native_projection_tile() != 8
            || executor.native_projection_member_capacity() != 8
            || !executor.target_verify3_dense
            || !executor.target_verify3_fused
            || self.moe.cpu_miss_cap().is_some()
            || self.moe.cpu_gpu_overlap().is_some()
            || self.moe.resident_router_active()
            || self.moe.frozen_host_sources_enabled()
            || self.moe.prepared_verify3_owner_id().is_some()
            || self.moe.live_member_reduction_enabled()
        {
            return Err(Error::InvalidFormat("global native controller authority"));
        }
        self.global_moe.as_ref().ok_or(error())?.validate_idle()?;
        executor.check_target_rows_policy()
    }
    fn three_gdn_scope(
        &self,
        executor: &GpuExecutor<'_>,
        admission: &crate::executor::ResidentAdmission,
    ) -> Result<PublicThreeGdnScope> {
        Ok(PublicThreeGdnScope {
            original_profile: executor.packed_profile
                == crate::expert_banks::OriginalPackedProfile::DaslabIq3,
            target_rows: self.target_rows.is_some() && executor.target_rows_reserved,
            checkpoint: self.latest_target_checkpoint_reserved,
            public_request: self.public_target_request_io.is_some()
                != self.public_language_request_io.is_some(),
            frozen_sources: self.public_frozen_host_sources
                && self.moe.frozen_host_sources_enabled(),
            frozen_router: self.public_frozen_host_router
                && self.moe.frozen_host_batch_router_active()
                && executor.target_frozen_host_router_reserved
                && executor.target_batch_router_reserved
                && executor.target_frozen_host_source_identity.is_some()
                && executor.resident_banks_frozen,
            partition_exit: self.moe.frozen_host_sources_enabled()
                && self.moe.frozen_host_source_partition_exit()?,
            original_io: executor.token_io_checkpoint
                && admission.deferred_token_io()
                && admission.transient_io(),
            original_dispatch: executor.target_verify3_dense
                && executor.target_verify3_fused
                && self.moe.projection_tile() == 8
                && self.dense_member_capacity == 8
                && executor.native_projection_tile() == 8
                && executor.native_projection_member_capacity() == 8,
            original_families: self.moe.cpu_miss_cap().is_none()
                && self.moe.cpu_gpu_overlap().is_none()
                && !self.moe.resident_router_active()
                && self.moe.count1_hybrid_owner_ids().is_none()
                && self.moe.prepared_verify3_owner_id().is_none()
                && !self.moe.live_member_reduction_enabled(),
        })
    }
    pub(super) fn allocate(
        model: &mut ResidentModel<'_, '_>,
        executor: &GpuExecutor<'_>,
        sequence: SequenceKey,
        config: Config,
        allocation: &crate::executor::ModelConstructionAllocation<'_>,
    ) -> Result<Self> {
        allocation.check(&model.admission)?;
        frozen_host_source_configuration(
            config.public_frozen_host_sources,
            config.target_rows_reserved,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.prepared_verify3_experts_reserved,
            config.public_warm_batch_router,
            config.live_member_reduction,
            config.tile,
        )?;
        gpu_qsa_selector_configuration(
            config.public_gpu_qsa_selector,
            config.target_rows_reserved,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.tile,
        )?;
        config.qsa_host_reservation()?;
        live_member_reduction_configuration(
            config.live_member_reduction,
            config.target_rows_reserved,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.prepared_verify3_experts_reserved,
            config.tile,
        )?;
        batch_moe::live_member_reduction_policy(
            config.live_member_reduction,
            executor.packed_profile,
            executor.resident_bank_capacity(),
            config.tile,
            config.cpu_operator.is_some(),
            config.resident_router_reserved,
            config.prepared_verify3_experts_reserved,
        )?;
        if config.public_warm_batch_router {
            let (host, limit) = host_reservation_with_verify3(
                false,
                config.request_reuse_metadata,
                false,
                true,
                true,
                false,
            )?;
            let host = host
                .checked_add(batch_moe::batch_router::HOST_BYTES)
                .ok_or(Error::ArithmeticOverflow)?;
            let limit = limit
                .checked_add(batch_moe::batch_router::HOST_BYTES)
                .ok_or(Error::ArithmeticOverflow)?;
            if host > limit {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        }
        if config.live_member_reduction {
            executor.check_target_rows_policy()?;
        }
        let public_target_request_io = if config.public_target_request_io {
            Some(public_target_request_io::Permit::new(
                model.graph.binding_id(),
                executor.packed_profile,
                model.context,
                model.position,
                model.admission.transient_io(),
                model.arena.public_owned_zero_audit_enabled()
                    && model.io.public_owned_zero_audit_enabled(),
                config.public_warm_replay,
                model.admission.request_io_reuse_active(),
            )?)
        } else {
            None
        };
        let public_language_request_io = config
            .public_language_request_io
            .map(|geometry| {
                public_language_request_io::Permit::new(
                    model.graph.binding_id(),
                    executor.packed_profile,
                    geometry,
                    model.context,
                    model.position,
                    model.admission.transient_io(),
                    model.arena.public_owned_zero_audit_enabled()
                        && model.io.public_owned_zero_audit_enabled(),
                    model.admission.request_io_reuse_active(),
                )
            })
            .transpose()?;
        let public_warm_replay = if config.public_warm_replay {
            public_warm_replay::configuration(
                config.target_rows_reserved,
                config.latest_target_checkpoint_reserved,
                config.cpu_operator.is_some(),
                config.resident_router_reserved,
                config.tile,
                false,
            )?;
            Some(public_warm_replay::Permit::new(
                model.graph.binding_id(),
                executor.packed_profile,
                model.context,
                model.position,
                model.admission.transient_io(),
                model.arena.public_owned_zero_audit_enabled()
                    && model.io.public_owned_zero_audit_enabled(),
            )?)
        } else {
            None
        };
        let graph = model.graph;
        let poison = model.admission.sequence_poison();
        let cancel = AtomicBool::new(false); // Idle zero-state constructor only.
        model
            .admission
            .check_client(model.arena.client().map_err(u::fault)?)?;
        if !model.admission.transient_io()
            || !std::ptr::eq(
                config.client.info(),
                model.arena.client().map_err(u::fault)?.info(),
            )
            || model.public.full_page_bytes() != PUBLIC_MAIN
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let gr = gr::Workspace::allocate(graph, &mut model.arena)?;
        let gdn = gdn::GdnBatchWorkspace::allocate(
            graph,
            sequence,
            &mut model.arena,
            poison.clone(),
            &cancel,
        )?;
        let qsa =
            qsa::QsaBatchWorkspace::allocate(graph, sequence, &mut model.arena, poison.clone())?;
        let mut moe = batch_moe::Workspace::allocate_with_policies(
            executor,
            graph,
            sequence,
            &mut model.arena,
            model.admission.clone(),
            poison.clone(),
            &config.limits,
            config.tile,
            &cancel,
            config.public_existing_expert_fill,
            config.live_member_reduction,
        )?;
        if let Some(operator) = config.cpu_operator {
            moe.reserve_count1_hybrid(
                &mut model.arena,
                operator,
                config.cpu_gpu_overlap,
                allocation,
            )?;
        }
        if config.resident_router_reserved {
            moe.reserve_resident_router(&mut model.arena, allocation)?;
        }
        if config.public_warm_batch_router {
            moe.reserve_batch_router(&mut model.arena, allocation)?;
        }
        if config.public_frozen_host_router {
            moe.reserve_frozen_host_batch_router_with_four_rows(
                &mut model.arena,
                allocation,
                model.context,
                config.public_four_rows_reserved,
            )?;
        }
        if config.public_frozen_host_staging {
            moe.reserve_frozen_host_staging(&mut model.arena, allocation)?;
        }
        if config.public_frozen_gu_integer {
            moe.reserve_frozen_gu_integer(&mut model.arena, allocation)?;
        }
        if config.public_frozen_down_integer {
            moe.reserve_frozen_down_integer(&mut model.arena, allocation)?;
        }
        if config.prepared_verify3_experts_reserved {
            moe.reserve_prepared_verify3(&mut model.arena, allocation)?;
        }
        let shared = shared_batch::Workspace::allocate(graph, &mut model.arena)?;
        let global_moe = if crate::strata_intel_core::enabled() {
            Some(crate::strata_global_moe::Workspace::allocate(
                graph,
                &mut model.arena,
                model.context,
            )?)
        } else {
            None
        };
        let core_stages = global_moe.as_ref().map(|m| m.stage_ids());
        let core_prompt_stages = global_moe.as_ref().and_then(|m| m.prompt_stage_ids());
        let mut alloc = |n| model.arena.allocate_zeroed(n).map_err(u::fault);
        let outer = Zeroizing::new(Outer {
            residual: [alloc(RESIDUAL_WORDS)?, alloc(RESIDUAL_WORDS)?],
            block: [alloc(BLOCK_WORDS)?, alloc(BLOCK_WORDS)?],
            router: alloc(MAX_BATCH * ROUTER)?,
            packet: alloc(MAX_BATCH * ROUTER + 1)?,
        });
        let target_rows = if config.target_rows_reserved {
            Some(target_rows::Owners::allocate_with_four_rows(
                &mut model.arena,
                config.public_four_rows_reserved,
            )?)
        } else {
            None
        };
        let ple = ple_batch::PleBatchWorkspace::allocate_with_profile(
            model,
            sequence,
            poison.clone(),
            &cancel,
            executor.packed_profile,
        )?;
        let result = Self {
            gr,
            gdn,
            qsa,
            moe,
            ple,
            shared,
            outer,
            core_stages,
            core_prompt_stages,
            global_moe,
            poison,
            dense_member_capacity: 8,
            target_rows,
            latest_target_checkpoint_reserved: config.latest_target_checkpoint_reserved,
            public_warm_replay,
            public_target_request_io,
            public_language_request_io,
            public_gpu_qsa_selector: config.public_gpu_qsa_selector,
            public_frozen_host_sources: config.public_frozen_host_sources,
            public_frozen_host_router: config.public_frozen_host_router,
            three_gdn_completed_chunks: 0,
            public_four_rows_reserved: config.public_four_rows_reserved,
            four_rows_completed_chunks: 0,
            public_grouped_ple_reads_reserved: config.public_grouped_ple_reads_reserved,
            grouped_ple_completed_chunks: 0,
            public_grouped_status_folds_reserved: config.public_grouped_status_folds_reserved,
            grouped_status_completed: crate::status_fold_batch::Completed::default(),
        };
        result.prove_outer(model)?;
        Ok(result)
    }
    fn prove_outer(&self, model: &ResidentModel<'_, '_>) -> Result<()> {
        let mut handles = Vec::new(); // Public resource references, no tensor copies.
        for (id, words) in [
            (self.outer.residual[0], RESIDUAL_WORDS),
            (self.outer.residual[1], RESIDUAL_WORDS),
            (self.outer.block[0], BLOCK_WORDS),
            (self.outer.block[1], BLOCK_WORDS),
            (self.outer.router, MAX_BATCH * ROUTER),
            (self.outer.packet, MAX_BATCH * ROUTER + 1),
            (model.b.fault, 1),
            (model.b.logits, model.graph.geometry().vocabulary),
            (model.b.logit_packet, model.graph.geometry().vocabulary + 1),
        ] {
            handles.push(gr::full_private(&model.arena, id, words)?);
        }
        if let Some(ids) = self.moe.count1_hybrid_owner_ids() {
            for (id, words) in ids.into_iter().zip([10 * D, 4 + 80 * 3]) {
                handles.push(gr::full_private(&model.arena, id, words)?);
            }
        }
        if let Some(ids) = self.moe.resident_router_ids() {
            for (id, words) in ids
                .into_iter()
                .zip([48 * 512, batch_io::RESIDENT_AUDIT_WORDS])
            {
                handles.push(gr::full_private(&model.arena, id, words)?);
            }
        }
        if let Some(ids) = self.moe.batch_router_ids() {
            for (id, words) in ids.into_iter().zip([
                48 * 512,
                if self.public_four_rows_reserved {
                    crate::resident_batch_router_gpu::four::AUDIT_WORDS
                } else {
                    crate::resident_batch_router_gpu::AUDIT_WORDS
                },
            ]) {
                handles.push(gr::full_private(&model.arena, id, words)?);
            }
        }
        if let Some(ids) = self.moe.frozen_staging_owner_ids() {
            for id in ids {
                handles.push(gr::full_private(
                    &model.arena,
                    id,
                    (batch_moe::FROZEN_STAGING_OWNER_BYTES / 4) as usize,
                )?);
            }
        }
        if let Some(owner) = self.moe.frozen_gu_integer_owner_id() {
            handles.push(gr::full_private(
                &model.arena,
                owner,
                batch_moe::FROZEN_GU_INTEGER_SCRATCH_WORDS,
            )?);
        }
        if let Some(owner) = self.moe.frozen_down_integer_owner_id() {
            handles.push(gr::full_private(
                &model.arena,
                owner,
                batch_moe::FROZEN_DOWN_INTEGER_SCRATCH_WORDS,
            )?);
        }
        if let Some(owner) = self.moe.prepared_verify3_owner_id() {
            handles.push(gr::full_private(
                &model.arena,
                owner,
                batch_moe::prepared_verify3::SCRATCH_WORDS,
            )?);
        }
        if let Some(rows) = &self.target_rows {
            handles.extend(rows.prove(&model.arena)?);
        }
        gr::physical(&model.arena, &handles)?;
        model
            .admission
            .check_client(model.arena.client().map_err(u::fault)?)?;
        if !Arc::ptr_eq(&self.poison, &model.admission.sequence_poison())
            || self.poison.load(Ordering::Acquire)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(())
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        self.poison.store(true, Ordering::Release);
    }
}

/// Original armed stage transactions prepared before the captured graph.
/// Their private wrappers prevent commit before the corresponding math enqueue.
struct AttentionInvocations {
    gdn: Box<[Option<gdn::PreparedGdnBatch>; LAYERS]>,
    qsa: Box<[Option<qsa::PreparedQsaBatch>; LAYERS]>,
}
impl AttentionInvocations {
    fn new() -> Result<Self> {
        if metadata_bytes() > META_CAP {
            return Err(Error::LimitExceeded("native prepared attention metadata"));
        }
        Ok(Self {
            gdn: Box::new(std::array::from_fn(|_| None)),
            qsa: Box::new(std::array::from_fn(|_| None)),
        })
    }
    fn consumed(&self) -> Result<()> {
        if self.gdn.iter().any(Option::is_some) || self.qsa.iter().any(Option::is_some) {
            return Err(Error::InvalidFormat(
                "native attention preparation unconsumed",
            ));
        }
        Ok(())
    }
}
struct Pending {
    gdn: Box<[Option<gdn::PendingGdnBatch>; LAYERS]>,
    qsa: Box<[Option<qsa::PendingQsaBatch>; LAYERS]>,
    moe: Box<[Option<batch_moe::Pending>; LAYERS]>,
    ple: Option<ple_batch::PendingPleBatch>,
}
fn pending_bytes() -> usize {
    std::mem::size_of::<[Option<gdn::PendingGdnBatch>; LAYERS]>()
        + std::mem::size_of::<[Option<qsa::PendingQsaBatch>; LAYERS]>()
        + std::mem::size_of::<[Option<batch_moe::Pending>; LAYERS]>()
        + std::mem::size_of::<Option<ple_batch::PendingPleBatch>>()
}
fn metadata_bytes() -> usize {
    // Actual family anchors retain a subset of their pending metadata. Charge a
    // second complete pending representation for every real GDN/QSA/PLE anchor,
    // plus the controller/session and bounded stack/capability headroom. MoE's
    // retained dynamic candidates and plans have their separate 2 MiB budget.
    pending_bytes()
        + if crate::strata_intel_core::enabled() {
            std::mem::size_of::<[Option<gdn::PreparedGdnBatch>; LAYERS]>()
                + std::mem::size_of::<[Option<qsa::PreparedQsaBatch>; LAYERS]>()
        } else {
            0
        }
        + if crate::strata_intel_core::enabled() {
            crate::strata_global_adapt::HOST_BYTES
        } else {
            0
        }
        + 36 * std::mem::size_of::<gdn::PendingGdnBatch>()
        + 12 * std::mem::size_of::<qsa::PendingQsaBatch>()
        + std::mem::size_of::<ple_batch::PendingPleBatch>()
        + std::mem::size_of::<Workspace>()
        + std::mem::size_of::<batch_moe::Session>()
        + 8192
}
impl Pending {
    fn new() -> Result<Self> {
        if metadata_bytes() > META_CAP {
            return Err(Error::LimitExceeded("native model pending owner capacity"));
        }
        // Allocate zero/None before filling. Each opaque pending owns guarded
        // metadata and sticky-only Drop; no filled inline table is returned.
        Ok(Self {
            gdn: Box::new(std::array::from_fn(|_| None)),
            qsa: Box::new(std::array::from_fn(|_| None)),
            moe: Box::new(std::array::from_fn(|_| None)),
            ple: None,
        })
    }
}
struct Flight {
    poison: Arc<AtomicBool>,
    accepted: bool,
}
impl Drop for Flight {
    fn drop(&mut self) {
        if !self.accepted {
            self.poison.store(true, Ordering::Release);
        }
        self.accepted.zeroize();
    }
}

impl<'m, 'f> ResidentModel<'m, 'f> {
    /// # Safety
    /// Retain this original model/arena, its admission and the original draft
    /// on the sole worker through the lexical sampler scope's close/drop.
    /// No model reset/close/client swap is allowed during that scope.
    pub(crate) unsafe fn begin_sampling_request(
        &self,
        executor: &GpuExecutor<'f>,
        params: crate::strata_sampler::Params,
        prefix: &[u32],
        draft: Option<u64>,
    ) -> Result<crate::strata_sampler::request::Scope> {
        self.validate_native_request_boundary(executor)?;
        if self.position != prefix.len() {
            return Err(error());
        }
        let native = self.native_batch.as_ref().ok_or(error())?;
        let global = native.global_moe.as_ref().ok_or(error())?;
        global.validate_idle()?;
        unsafe {
            crate::strata_sampler::request::Scope::new(
                &self.arena,
                global.sampler_ids(),
                self.graph.binding_id(),
                draft,
                params,
                prefix,
                self.retired.clone(),
            )
        }
    }

    /// Exclusive fixed-public proposal/verification/accepted-prefix round.
    #[allow(dead_code)] // Parent owns the protected lab consumer.
    pub(crate) fn begin_latest_target_round<'a>(
        &'a mut self,
        executor: &'a mut GpuExecutor<'f>,
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> Result<LatestTargetRound<'a, 'm, 'f>> {
        LatestTargetRound::begin(self, executor, tokens, cancel)
    }
    /// Fixed-public laboratory experiment, called between completed native
    /// chunks. A fresh/scalar owner, pending work or missing full owning audit
    /// cannot change policy. This adds no launch, map, fence or device allocation.
    pub(crate) fn set_lab_native_projection_tile(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tile: usize,
    ) -> Result<usize> {
        let actual = self.native_decode_kernels()?;
        if !projection_policy_allows_tile(actual, tile)
            || self
                .native_batch
                .as_ref()
                .ok_or(error())?
                .moe
                .frozen_host_source_down_integer_selected()
                && tile != 8
            || self
                .native_batch
                .as_ref()
                .ok_or(error())?
                .grouped_status_state()?
                .0
                && tile != 8
            || self
                .native_batch
                .as_ref()
                .ok_or(error())?
                .ple
                .public_grouped_reads()
                && tile != 8
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let tile = NativeProjectionTile::new(tile)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        executor.set_native_projection_tile(tile, boundary)
    }

    /// # Safety
    /// Parent authenticated both complete original shards and holds the full
    /// protected process/RAM reservation. Same actual device as this model.
    pub(crate) unsafe fn prepare_global_sources_before_prefill(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prompt_host_grant: u64,
    ) -> Result<()> {
        self.validate_unstarted_native_boundary(executor)?;
        if self.position != 0 || !crate::strata_intel_core::enabled() {
            return Err(error());
        }
        let client = self.arena.client().map_err(u::fault)?;
        let budget = executor.native_expert_capacity()?;
        let plan = crate::strata_global_plan::Plan::from_graph(
            self.graph,
            budget,
            client.properties().memory.alignment.max(256),
        )?;
        let reserved = crate::public_host_expert_mirror::strata_global::Host::reservation(&plan)?;
        let stages = self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .global_moe
            .as_ref()
            .ok_or(error())?
            .stage_handles(&self.arena)?;
        let prompt_stages = self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .global_moe
            .as_ref()
            .ok_or(error())?
            .prompt_stage_handles(&self.arena)?;
        unsafe {
            executor.prepare_strata_global_sources(self.graph, plan, device, reserved)?;
        }
        executor
            .strata_global_sources
            .as_mut()
            .and_then(std::sync::Arc::get_mut)
            .ok_or(error())?
            .attach_stages(device, queue, stages)?;
        let sources = executor
            .strata_global_sources
            .as_mut()
            .and_then(std::sync::Arc::get_mut)
            .ok_or(error())?;
        if let Some(prompt_stages) = prompt_stages {
            if prompt_host_grant != 0 {
                return Err(error());
            }
            sources.attach_prompt_stage(prompt_stages[2].clone())?;
        } else {
            let (indices, extra_host) = sources.prompt_loan_quote(self.graph)?;
            if extra_host != prompt_host_grant {
                return Err(error());
            }
            // Separate exact caller-held RAM grant, never GPU pricing credit.
            unsafe {
                sources.prepare_prompt_loan(self.graph, indices, prompt_host_grant)?;
            }
        }
        self.validate_unstarted_native_boundary(executor)?;
        executor.activate_native_core_q6_head()
    }

    fn validate_native_projection_owners(&self, executor: &GpuExecutor<'f>) -> Result<()> {
        if self.poisoned
            || self.retired.load(Ordering::Acquire)
            || !self.pending_qsa.is_empty()
            || !self.pending_moe.is_empty()
            || self.ple_batch_pending.is_some()
            || !self.admission.transient_io()
            || !self.arena.public_owned_zero_audit_enabled()
            || !self.io.public_owned_zero_audit_enabled()
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.graph.verify_unchanged()?;
        executor.check_client(self.arena.client().map_err(u::fault)?)?;
        self.admission
            .check_client(self.io.client().map_err(u::fault)?)?;
        executor.check_resident_admission(&self.admission)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.prove_outer(self)?;
        native.gr.validate_all_consumers_queued()?;
        native.shared.validate_all_consumers_queued()?;
        if let Some(global) = &native.global_moe {
            global.validate_idle()?;
            match &executor.strata_global_sources {
                Some(sources) => sources.validate_execution_boundary(self.graph)?,
                None if self.position == 0 && !executor.target_four_rows_active => (),
                None => return Err(error()),
            }
        }
        Ok(())
    }

    fn completed_native_projection_boundary(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<CompletedNativeProjectionBoundary> {
        self.validate_native_projection_owners(executor)?;
        self.native_batch
            .as_ref()
            .ok_or(error())?
            .ple
            .validate_completed_projection_boundary(self)?;
        Ok(CompletedNativeProjectionBoundary { _private: () })
    }

    fn validate_unstarted_native_boundary(&self, executor: &GpuExecutor<'f>) -> Result<()> {
        self.validate_native_projection_owners(executor)?;
        if !crate::strata_intel_core::enabled()
            || self.position != 0
            || self.ple_history.tokens().iter().any(|&t| t != -1)
            || self.ple_conv.history().iter().any(|&v| v != 0.)
        {
            return Err(error());
        }
        self.native_batch
            .as_ref()
            .ok_or(error())?
            .ple
            .validate_unstarted_projection_boundary(self)
    }

    // No completed-policy token escapes this initial-or-committed request gate.
    fn validate_native_request_boundary(&self, executor: &GpuExecutor<'f>) -> Result<()> {
        if crate::strata_intel_core::enabled() && self.position == 0 {
            self.validate_unstarted_native_boundary(executor)
        } else {
            self.completed_native_projection_boundary(executor)
                .map(|_| ())
        }
    }

    pub(crate) fn adapt_global_after_window(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.completed_native_projection_boundary(executor)?;
        self.native_batch
            .as_ref()
            .ok_or(error())?
            .validate_global_scope(executor, &self.admission)?;
        self.native_batch
            .as_mut()
            .ok_or(error())?
            .global_moe
            .as_mut()
            .ok_or(error())?
            .adapt_after_finished(
                self.graph,
                &self.arena,
                &self.admission,
                executor
                    .strata_global_sources
                    .as_mut()
                    .and_then(std::sync::Arc::get_mut)
                    .ok_or(error())?,
                cancel,
            )
    }
    pub(crate) fn validate_global_request_geometry(
        &self,
        executor: &GpuExecutor<'f>,
        prompt: usize,
        outputs: usize,
    ) -> Result<()> {
        self.validate_native_request_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.validate_global_scope(executor, &self.admission)?;
        let permit = native.public_language_request_io.as_ref().ok_or(error())?;
        permit.identity(self.graph.binding_id(), self.context)?;
        permit.validate_geometry(prompt, outputs)
    }
    /// Actual committed cursor for the explicitly reserved target-row owner.
    /// This performs the existing complete native boundary proof; it does not
    /// infer completion from the returned result's start/count metadata.
    #[allow(dead_code)] // Parent owns the dedicated fixed-public registration.
    pub(crate) fn completed_target_rows_cursor(&self, executor: &GpuExecutor<'f>) -> Result<usize> {
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !executor.target_rows_reserved || native.target_rows.is_none() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        executor.check_target_rows_policy()?;
        self.validate_native_request_boundary(executor)?;
        Ok(self.position)
    }

    /// Every oracle operation requires the unchanged complete native boundary,
    /// exact compiled-public context, and accepted cap10 count1 execution.
    fn lab_oracle_boundary(
        &self,
        executor: &GpuExecutor<'f>,
        position: usize,
        fixture: crate::oracle_trace::Fixture,
    ) -> Result<CompletedNativeProjectionBoundary> {
        let boundary = self.completed_native_projection_boundary(executor)?;
        if self.position != position
            || self.context != fixture.context()
            || self.native_cpu_miss_cap()? != Some(10)
            || (fixture != crate::oracle_trace::Fixture::Q4Rainwater
                && self.native_decode_kernels()? != NativeDecodeKernels::SingleAll)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        executor.check_lab_oracle_fixture(fixture)?;
        Ok(boundary)
    }

    fn lab_oracle_attempt<T>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        operation: impl FnOnce(&mut Self, &mut GpuExecutor<'f>) -> Result<T>,
    ) -> Result<T> {
        let attempt =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(self, executor)));
        match attempt {
            Ok(Ok(value)) => Ok(value),
            _ => {
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }

    pub(crate) fn begin_lab_oracle_trace(&mut self, executor: &mut GpuExecutor<'f>) -> Result<()> {
        self.begin_lab_oracle_trace_for(executor, crate::oracle_trace::Fixture::Q4Rainwater)
    }

    #[allow(dead_code)] // Dedicated fixed-public Q3 orchestrator is registered separately.
    pub(crate) fn begin_lab_q3_oracle_trace(
        &mut self,
        executor: &mut GpuExecutor<'f>,
    ) -> Result<()> {
        self.begin_lab_oracle_trace_for(executor, crate::oracle_trace::Fixture::Q3Prefix23)
    }

    pub(crate) fn begin_lab_oracle_trace_for(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        fixture: crate::oracle_trace::Fixture,
    ) -> Result<()> {
        self.lab_oracle_attempt(executor, |model, executor| {
            let boundary = model.lab_oracle_boundary(executor, fixture.start(), fixture)?;
            executor.check_lab_oracle_policy(&boundary)?;
            executor.oracle_trace = Some(match fixture {
                crate::oracle_trace::Fixture::Q4Rainwater => {
                    crate::oracle_trace::Trace::training()?
                }
                crate::oracle_trace::Fixture::Q3Prefix23 => {
                    crate::oracle_trace::Trace::q3_prefix23_training()?
                }
                crate::oracle_trace::Fixture::DaslabPrefix13 => {
                    crate::oracle_trace::Trace::daslab_prefix13_training()?
                }
                crate::oracle_trace::Fixture::Daslab32 => {
                    crate::oracle_trace::Trace::daslab32_training()?
                }
            });
            Ok(())
        })
    }

    pub(crate) fn finish_lab_oracle_training(
        &mut self,
        executor: &mut GpuExecutor<'f>,
    ) -> Result<crate::oracle_trace::Trace> {
        self.lab_oracle_attempt(executor, |model, executor| {
            let trace = executor.oracle_trace.as_ref().ok_or(error())?;
            model.lab_oracle_boundary(executor, trace.start() + trace.decode(), trace.fixture())?;
            trace.validated_replay_copy()
        })
    }

    pub(crate) fn preload_lab_oracle(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        trace: &crate::oracle_trace::Trace,
        cancel: &AtomicBool,
    ) -> Result<crate::expert_banks::FillStats> {
        self.lab_oracle_attempt(executor, |model, executor| {
            u::cancel(cancel)?;
            let boundary = model.lab_oracle_boundary(executor, trace.start(), trace.fixture())?;
            let stats = executor.preload_lab_oracle(
                &boundary,
                trace,
                &model.arena,
                model.b.fault,
                cancel,
            )?;
            u::cancel(cancel)?;
            Ok(stats)
        })
    }

    pub(crate) fn begin_lab_oracle_replay(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        trace: crate::oracle_trace::Trace,
    ) -> Result<()> {
        self.lab_oracle_attempt(executor, |model, executor| {
            let boundary = model.lab_oracle_boundary(executor, trace.start(), trace.fixture())?;
            executor.check_lab_oracle_preloaded(&boundary, &trace)?;
            executor.oracle_trace = Some(trace);
            Ok(())
        })
    }

    pub(crate) fn begin_lab_resident_router(
        &mut self,
        executor: &mut GpuExecutor<'f>,
    ) -> Result<()> {
        self.lab_oracle_attempt(executor, |model, executor| {
            let trace = executor.oracle_trace.as_ref().ok_or(error())?;
            if !trace.is_replay() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            let boundary = model.lab_oracle_boundary(executor, trace.start(), trace.fixture())?;
            let native = model.native_batch.as_mut().ok_or(error())?;
            native
                .moe
                .enable_resident_router(executor, model.graph, &model.arena, &boundary)
        })
    }

    pub(crate) fn finish_lab_oracle_replay(
        &mut self,
        executor: &mut GpuExecutor<'f>,
    ) -> Result<()> {
        self.lab_oracle_attempt(executor, |model, executor| {
            let trace = executor.oracle_trace.as_ref().ok_or(error())?;
            model.lab_oracle_boundary(executor, trace.start() + trace.decode(), trace.fixture())?;
            if !trace.is_replay() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            trace.complete()
        })
    }

    /// Read the actual installed family policies rather than echoing an env
    /// selection. No private metadata or model state is exposed.
    pub(crate) fn native_decode_kernels(&self) -> Result<NativeDecodeKernels> {
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.decode_kernels()
    }
    pub(crate) fn advise_gpu_sources_not_needed(
        &self,
        executor: &GpuExecutor<'f>,
        pageout: bool,
    ) -> Result<u64> {
        let proof = self.completed_native_projection_boundary(executor)?;
        if self.native_cpu_miss_cap()? != Some(10) {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        executor.advise_resident_gpu_sources_not_needed(&proof, pageout)
    }

    #[allow(dead_code)] // Protected receipt exposure is owned separately.
    pub(crate) fn native_dense_member_capacity(&self) -> Result<usize> {
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.decode_kernels()?;
        Ok(native.dense_member_capacity)
    }

    pub(crate) fn native_cpu_miss_cap(&self) -> Result<Option<usize>> {
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_miss_cap())
    }

    pub(crate) fn native_cpu_workers_pooled(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_workers_pooled())
    }

    pub(crate) fn native_cpu_arithmetic_geometry(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<(usize, usize)>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_arithmetic_geometry())
    }
    pub(crate) fn native_cpu_phase_ns(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<[u64; 4]>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_phase_ns())
    }
    pub(crate) fn native_public_fill_counters(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, batch_moe::PublicFillCounters)> {
        self.completed_native_projection_boundary(executor)?;
        let moe = &self.native_batch.as_ref().ok_or(error())?.moe;
        Ok((
            moe.public_existing_fill_enabled(),
            moe.public_fill_counters(),
        ))
    }
    /// Actual construction-fixed public MoE policy at the original completed
    /// all-family boundary; no requested flag is credited as dispatch evidence.
    #[allow(dead_code)] // Parent owns the protected public receipt.
    pub(crate) fn native_live_member_reduction(&self, executor: &GpuExecutor<'f>) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if native.target_rows.is_none() || !executor.target_rows_reserved {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        executor.check_target_rows_policy()?;
        native
            .moe
            .validate_public_warm_boundary(executor, self.graph, &self.arena)?;
        Ok(native.moe.live_member_reduction_enabled())
    }

    pub(crate) fn native_cpu_buffered_weights(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<(bool, u64)>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_buffered_weights())
    }

    pub(crate) fn native_cpu_mapped_prefetch(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<(bool, u64)>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_mapped_prefetch())
    }
    pub(crate) fn native_cpu_q8_residual(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_q8_residual())
    }
    pub(crate) fn native_cpu_q8_activations(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_q8_activations())
    }
    pub(crate) fn native_deferred_router_reads(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, u64)> {
        self.completed_native_projection_boundary(executor)?;
        if self.native_cpu_miss_cap()?.is_none() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok((
            self.admission.deferred_token_io(),
            self.admission.deferred_router_reads(),
        ))
    }

    pub(crate) fn native_cpu_gpu_overlap(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_gpu_overlap())
    }

    /// Aggregate only at a healthy completed original-model boundary. The lab
    /// may publish it for the compiled public fixture; private requests may not.
    pub(crate) fn completed_cpu_misses(&self, executor: &GpuExecutor<'f>) -> Result<u64> {
        self.completed_native_projection_boundary(executor)?;
        Ok(*self.completed_cpu_misses)
    }

    pub(crate) fn shared_gdn_calls(&self) -> Result<u64> {
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .gdn
            .shared_vector_calls)
    }

    /// Fixed public operator-only borrow. Original complete native health and
    /// actual workspace proof precede the attachment's private source mapping.
    /// Shared model/executor borrows prevent reset/fill through the returned
    /// guard; no source pointer, bank or independently usable GPU handle escapes.
    pub(crate) fn borrow_public_host_cpu_layer<'borrow, 'reader, 'plan>(
        &'borrow self,
        attachment: &'borrow mut crate::public_host_expert_attachment::Attachment<'plan, 'm, 'f>,
        prepared: &'borrow crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        reader: &'reader kuno_text::gguf::PositionalReader<'f>,
        executor: &'borrow GpuExecutor<'f>,
        layer: usize,
        cancel: &AtomicBool,
    ) -> Result<crate::public_host_expert_attachment::CpuLayerBorrow<'borrow, 'reader, 'm, 'f>>
    {
        self.completed_native_projection_boundary(executor)?;
        attachment.borrow_cpu_layer(
            prepared,
            reader,
            executor,
            &self.native_batch.as_ref().ok_or(error())?.moe,
            layer,
            cancel,
        )
    }

    /// Public-only source preparation follows an actual committed whole-model
    /// boundary. Empty slots receive original weights; existing residents stay.
    pub(crate) fn prepare_public_host_sources(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
        preserve_routed_bank: bool,
    ) -> Result<(
        crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        crate::expert_banks::FillStats,
    )> {
        self.prepare_public_host_sources_with_fill_policy(
            executor,
            cancel,
            preserve_routed_bank,
            crate::executor::PublicHostBankFillPolicy::Original,
        )
    }
    #[allow(dead_code)] // Separate fixed-public controller owns opt-in registration.
    pub(crate) fn prepare_public_host_sources_with_fill_policy(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
        preserve_routed_bank: bool,
        policy: crate::executor::PublicHostBankFillPolicy,
    ) -> Result<(
        crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        crate::expert_banks::FillStats,
    )> {
        u::cancel(cancel)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources {
            return Err(Error::InvalidFormat(
                "public host prepare construction reserve",
            ));
        }
        if native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(
                "public host prepare source already attached",
            ));
        }
        let ranked = policy == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller;
        validate_bank_fill_selection(
            policy,
            preserve_routed_bank,
            native.public_frozen_host_router
                && native.target_rows.is_some()
                && native.latest_target_checkpoint_reserved
                && (native.public_target_request_io.is_some()
                    != native.public_language_request_io.is_some())
                && executor.target_frozen_host_router_reserved
                && executor.target_batch_router_reserved
                && executor.target_verify3_dense
                && executor.target_verify3_fused
                && executor.token_io_checkpoint
                && self.admission.deferred_token_io()
                && self.admission.transient_io()
                && native.moe.projection_tile() == 8
                && native.dense_member_capacity == 8
                && executor.native_projection_tile() == 8
                && executor.native_projection_member_capacity() == 8
                && executor.packed_profile == crate::expert_banks::OriginalPackedProfile::DaslabIq3,
        )?;

        if self.admission.request_io_reuse_active() {
            return Err(Error::InvalidFormat("public host prepare live request I/O"));
        }
        let boundary = self
            .completed_native_projection_boundary(executor)
            .map_err(|_| Error::InvalidFormat("public host prepare initial completed boundary"))?;
        native
            .moe
            .validate_public_warm_boundary(executor, self.graph, &self.arena)
            .map_err(|_| Error::InvalidFormat("public host prepare routed completed boundary"))?;
        let (fills, publication) = if ranked {
            let (fills, publication) = executor.complete_public_host_ranked_banks(
                &boundary,
                self.graph,
                &self.arena,
                self.b.fault,
                cancel,
            )?;
            (fills, Some(publication))
        } else {
            let fills = if self.context == 13
                && native.public_warm_replay.is_some()
                && !preserve_routed_bank
            {
                // Deterministic PHYSICAL placement exercises both actual tiers in
                // the complete CPU13 numerical gate. It never changes routing.
                executor.complete_public_host_validation_banks(
                    &boundary,
                    self.graph,
                    &self.arena,
                    self.b.fault,
                    cancel,
                )?
            } else {
                executor.complete_public_host_banks(
                    &boundary,
                    self.graph,
                    &self.arena,
                    self.b.fault,
                    cancel,
                )?
            };
            (fills, None)
        };
        let result = (|| {
            self.completed_native_projection_boundary(executor)
                .map_err(|_| {
                    Error::InvalidFormat("public host prepare filled completed boundary")
                })?;
            let prepared = if let Some(publication) = publication {
                crate::public_host_expert_attachment::PreparedPlan::new_with_fill_publication(
                    self.graph,
                    executor.public_host_model(),
                    executor,
                    65536,
                    crate::public_host_expert_plan::METADATA_HOST_CAP,
                    cancel,
                    Some(publication),
                )?
            } else {
                crate::public_host_expert_attachment::PreparedPlan::new(
                    self.graph,
                    executor.public_host_model(),
                    executor,
                    65536,
                    crate::public_host_expert_plan::METADATA_HOST_CAP,
                    cancel,
                )?
            };
            Ok((prepared, fills))
        })();
        if ranked && result.is_err() {
            self.poisoned = true;
        }
        result
    }
    /// The parent retains exact public hash/source proofs and PreparedPlan.
    /// No model/workspace borrow escapes into the imported-page lifetime.
    #[allow(clippy::too_many_arguments)] // Explicit immutable source policies belong to this attachment.
    pub(crate) unsafe fn attach_public_host_sources<'plan>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        prepared: &'plan crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        device: &wgpu::Device,
        reserved: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        cancel: &AtomicBool,
    ) -> Result<crate::public_host_expert_attachment::Attachment<'plan, 'm, 'f>> {
        // SAFETY: forwards the same caller contract with original math.
        unsafe {
            self.attach_public_host_sources_with_gu_math(
                executor,
                prepared,
                device,
                reserved,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)] // Explicit immutable source policies belong to this attachment.
    pub(crate) unsafe fn attach_public_host_sources_with_gu_math<'plan>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        prepared: &'plan crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        device: &wgpu::Device,
        reserved: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        cancel: &AtomicBool,
    ) -> Result<crate::public_host_expert_attachment::Attachment<'plan, 'm, 'f>> {
        // SAFETY: forwards the same verified original source contract; the
        // existing entry explicitly retains default integer scheduling.
        unsafe {
            self.attach_public_host_sources_with_gu_schedule(
                executor,
                prepared,
                device,
                reserved,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_public_host_sources_with_gu_schedule<'plan>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        prepared: &'plan crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        device: &wgpu::Device,
        reserved: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        cancel: &AtomicBool,
    ) -> Result<crate::public_host_expert_attachment::Attachment<'plan, 'm, 'f>> {
        // SAFETY: unchanged verified caller contract; default DOWN schedule.
        unsafe {
            self.attach_public_host_sources_with_schedules(
                executor,
                prepared,
                device,
                reserved,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                gu_stride_four,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_public_host_sources_with_schedules<'plan>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        prepared: &'plan crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        device: &wgpu::Device,
        reserved: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
        cancel: &AtomicBool,
    ) -> Result<crate::public_host_expert_attachment::Attachment<'plan, 'm, 'f>> {
        // Original entry retains its exact default arithmetic policy.
        unsafe {
            self.attach_public_host_sources_with_down_math(
                executor,
                prepared,
                device,
                reserved,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                gu_stride_four,
                down_stride_four,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_public_host_sources_with_down_math<'plan>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        prepared: &'plan crate::public_host_expert_attachment::PreparedPlan<'m, 'f>,
        device: &wgpu::Device,
        reserved: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
        down_integer: bool,
        cancel: &AtomicBool,
    ) -> Result<crate::public_host_expert_attachment::Attachment<'plan, 'm, 'f>> {
        u::cancel(cancel)?;
        batch_moe::frozen_sources::Sources::validate_down_integer_policy(
            down_integer,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            four_row_down,
            two_row_gate_up,
            packed_host_staging,
            three_members,
            down_stride_four,
        )?;
        batch_moe::frozen_sources::Sources::validate_gu_schedule_policy(
            gu_integer,
            gu_stride_four,
        )?;
        batch_moe::frozen_sources::Sources::validate_down_stride_four_policy(
            down_stride_four,
            inactive_partition_exit,
            four_row_down,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            packed_host_staging,
            three_members,
            two_row_gate_up,
        )?;
        if self.poisoned
            || self.retired.load(Ordering::Acquire)
            || self.admission.request_io_reuse_active()
            || !self.pending_qsa.is_empty()
            || !self.pending_moe.is_empty()
            || self.ple_batch_pending.is_some()
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources
            || native.moe.frozen_host_sources_enabled()
            || (gu_integer || down_stride_four || down_integer)
                && (native.target_rows.is_none()
                    || !native.latest_target_checkpoint_reserved
                    || native.public_target_request_io.is_some()
                        == native.public_language_request_io.is_some()
                    || (gu_integer && !executor.target_frozen_gu_integer_reserved)
                    || (down_integer && !executor.target_frozen_down_integer_reserved)
                    || !executor.target_verify3_dense
                    || !executor.target_verify3_fused
                    || !executor.token_io_checkpoint
                    || !self.admission.deferred_token_io()
                    || !self.admission.transient_io()
                    || native.dense_member_capacity != 8
                    || executor.native_projection_tile() != 8
                    || executor.native_projection_member_capacity() != 8)
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if down_integer {
            let accepted_prefill = match (
                &native.public_target_request_io,
                &native.public_language_request_io,
            ) {
                (Some(_), None) => {
                    self.context == 13
                        && self.position == 0
                        && native
                            .public_warm_replay
                            .as_ref()
                            .is_some_and(|p| p.reset_consumed())
                }
                (None, Some(permit)) => {
                    permit
                        .identity(self.graph.binding_id(), self.context)
                        .is_ok()
                        && permit.validate_begin(self.position, false).is_ok()
                }
                _ => false,
            };
            if !accepted_prefill || native.decode_kernels()? != NativeDecodeKernels::Chunk {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        }
        if self.position == 0 {
            if self.context != 13
                || native
                    .public_warm_replay
                    .as_ref()
                    .is_none_or(|permit| !permit.reset_consumed())
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        } else {
            self.completed_native_projection_boundary(executor)?;
        }
        native.prove_outer(self)?;
        native
            .moe
            .validate_public_warm_boundary(executor, self.graph, &self.arena)?;
        let four_rows = self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .public_four_rows_reserved;
        if four_rows {
            self.native_batch
                .as_mut()
                .ok_or(error())?
                .moe
                .set_public_four_members_policy(executor, self.graph, &self.arena, true)?;
        }
        // SAFETY: immutable actual original source/all-family proofs precede import.
        unsafe {
            prepared.attach_with_four_rows(
                device,
                self.arena.client().map_err(u::fault)?,
                executor,
                &mut self.native_batch.as_mut().ok_or(error())?.moe,
                &self.arena,
                reserved,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                gu_stride_four,
                down_stride_four,
                down_integer,
                four_rows,
                cancel,
            )
        }
    }
    #[allow(dead_code)] // Actual immutable setup receipt for separately owned public controller.
    pub(crate) fn native_frozen_source_bank_fill(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<crate::executor::CompletedPublicHostBankFill> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        native
            .moe
            .frozen_host_source_bank_fill(executor, self.graph)
    }

    pub(crate) fn native_frozen_source_counters(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<batch_moe::frozen_sources::Counters> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(native.moe.frozen_source_counters())
    }

    pub(crate) fn native_frozen_source_partition_exit(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_partition_exit()
    }

    pub(crate) fn native_frozen_source_private_tables(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_private_tables()
    }

    pub(crate) fn native_frozen_source_workgroup_tables(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_workgroup_tables()
    }

    pub(crate) fn native_frozen_source_shared_down_scales(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_shared_down_scales()
    }
    pub(crate) fn native_frozen_source_shared_gate_up_factors(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_shared_gate_up_factors()
    }
    pub(crate) fn native_frozen_source_four_row_down(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_four_row_down()
    }

    /// Completed original native boundary plus actual accepted source use.
    pub(crate) fn native_frozen_source_two_row_gate_up(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_two_row_gate_up()
    }

    pub(crate) fn native_frozen_source_three_members(
        &self,
        executor: &GpuExecutor<'_>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_three_members()
    }
    #[allow(dead_code)] // Root owns public receipt registration.
    pub(crate) fn native_frozen_source_gu_integer(
        &self,
        executor: &GpuExecutor<'_>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_gu_integer()
    }
    #[allow(dead_code)] // Root owns public receipt registration.
    pub(crate) fn native_frozen_source_down_integer(
        &self,
        executor: &GpuExecutor<'_>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if native.moe.frozen_host_source_down_integer_selected() {
            native.validate_target_decode_policy(executor, &self.admission)?;
        }
        native.moe.frozen_host_source_down_integer()
    }
    #[allow(dead_code)] // Root owns the public scheduling receipt/selector.
    pub(crate) fn native_frozen_source_gu_stride_four(
        &self,
        executor: &GpuExecutor<'_>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_gu_stride_four()
    }
    #[allow(dead_code)] // Root owns the public controller opt-in/receipt.
    pub(crate) fn native_frozen_source_down_stride_four(
        &self,
        executor: &GpuExecutor<'_>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.moe.frozen_host_source_down_stride_four()
    }
    pub(crate) fn native_frozen_source_packed_staging(
        &self,
        executor: &GpuExecutor<'_>,
    ) -> Result<bool> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let selected = native.moe.frozen_host_source_packed_staging()?;
        if selected && native.moe.frozen_source_counters().staging_dispatches == 0 {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok(selected)
    }
    pub(crate) fn activate_public_frozen_host_batch_router(
        &mut self,
        executor: &GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        if self.poisoned
            || self.retired.load(Ordering::Acquire)
            || self.admission.request_io_reuse_active()
            || !executor.target_frozen_host_router_reserved
            || !executor.target_batch_router_reserved
            || executor.target_frozen_host_source_identity.is_none()
            || !executor.resident_banks_frozen
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if self.position == 0 {
            if self.context != 13
                || self
                    .native_batch
                    .as_ref()
                    .and_then(|n| n.public_warm_replay.as_ref())
                    .is_none_or(|p| !p.reset_consumed())
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        } else {
            self.completed_native_projection_boundary(executor)?;
        }
        let native = self.native_batch.as_mut().ok_or(error())?;
        if !native.public_frozen_host_sources
            || !native.public_frozen_host_router
            || !native.moe.frozen_host_sources_enabled()
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native
            .moe
            .activate_frozen_host_batch_router(executor, self.graph, &self.arena)?;
        self.admission.checkpoint_public_router_activation()
    }

    pub(crate) fn activate_public_warm_batch_router(
        &mut self,
        executor: &GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        if self.context != 13
            || self.position != 0
            || !executor.target_batch_router_reserved
            || self
                .native_batch
                .as_ref()
                .and_then(|n| n.public_warm_replay.as_ref())
                .is_none_or(|p| !p.reset_consumed())
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.native_batch
            .as_mut()
            .ok_or(error())?
            .moe
            .activate_batch_router(executor, self.graph, &self.arena)?;
        self.admission.checkpoint_public_router_activation()
    }

    pub(crate) fn begin_lab_request_io_reuse(&self, executor: &GpuExecutor<'f>) -> Result<()> {
        self.completed_native_projection_boundary(executor)?;
        if self.native_cpu_miss_cap()? != Some(10) {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.admission.begin_request_io_reuse()
    }

    pub(crate) fn request_io_checkpoints(&self) -> u64 {
        self.admission.request_io_checkpoints()
    }

    pub(crate) fn end_lab_request_io_reuse(&self) -> Result<()> {
        self.admission.end_request_io_reuse()
    }

    /// Activate only after accepted original prefill and actual source import.
    /// This does not replace the original count1 head or reserve another owner.
    #[allow(dead_code)] // Dedicated controller is integrated separately.
    pub(crate) fn activate_public_q6_integer_verify_head(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        let accepted_prefill = match (
            &native.public_target_request_io,
            &native.public_language_request_io,
        ) {
            (Some(_), None) => {
                self.context == 13
                    && self.position == 1
                    && native
                        .public_warm_replay
                        .as_ref()
                        .is_some_and(|p| p.reset_consumed())
            }
            (None, Some(permit)) => {
                permit
                    .identity(self.graph.binding_id(), self.context)
                    .is_ok()
                    && permit.validate_begin(self.position, false).is_ok()
            }
            _ => false,
        };
        if !q6_verify_head_selection_allowed(
            native.decode_kernels()?,
            accepted_prefill,
            self.admission.request_io_reuse_active(),
        ) || !native.ple.public_grouped_reads()
            || !crate::strata_intel_core::enabled()
                && (!native.moe.frozen_host_sources_enabled()
                    || native.moe.frozen_gu_integer_owner_id().is_none()
                    || !native.moe.frozen_host_source_down_integer_selected())
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.activate_public_q6_integer_verify_head()?;
            u::cancel(cancel)
        }));
        match attempt {
            Ok(Ok(())) => Ok(()),
            _ => {
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }
    /// Actual enqueue counters are publishable only at the unchanged completed
    /// all-family boundary. Prefix rollback never undoes completed head work.
    #[allow(dead_code)] // Dedicated controller is integrated separately.
    pub(crate) fn completed_public_q6_integer_verify_head(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, u64, u64, u64)> {
        self.completed_native_projection_boundary(executor)?;
        executor.public_q6_integer_verify_head()
    }

    /// Explicit reserved count1 lab only. The original completed all-family
    /// boundary remains the sole authority; this performs no inference/commit.
    pub(crate) fn set_lab_count1_cpu_misses(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cap: usize,
        cancel: &AtomicBool,
    ) -> Result<usize> {
        u::cancel(cancel)?;
        if self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .frozen_host_source_down_integer_selected()
            || cap > 10
            || self
                .native_batch
                .as_ref()
                .ok_or(error())?
                .ple
                .public_grouped_reads()
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let boundary = self.completed_native_projection_boundary(executor)?;
        let attempt =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<usize> {
                let native = self.native_batch.as_mut().ok_or(error())?;
                let observed = native.moe.set_lab_count1_cpu_misses(
                    executor,
                    self.graph,
                    &self.arena,
                    cap,
                    &boundary,
                )?;
                u::cancel(cancel)?;
                Ok(observed)
            }));
        match attempt {
            Ok(Ok(observed)) => Ok(observed),
            _ => {
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }

    /// Explicit fixed-public Verify3 selection after accepted prefill and
    /// actual frozen source/router activation. No allocations or owner changes.
    pub(crate) fn set_public_three_gdn_streaming(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        executor.check_target_rows_policy()?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.validate_execution_scope(executor, &self.admission)?;
        let accepted_prefill = match (
            &native.public_target_request_io,
            &native.public_language_request_io,
        ) {
            (Some(_), None) => {
                self.context == 13
                    && self.position == 1
                    && native
                        .public_warm_replay
                        .as_ref()
                        .is_some_and(|p| p.reset_consumed())
            }
            (None, Some(permit)) => {
                permit
                    .identity(self.graph.binding_id(), self.context)
                    .is_ok()
                    && permit.validate_begin(self.position, false).is_ok()
            }
            _ => false,
        };
        if !three_gdn_selection_allowed(
            native.decode_kernels()?,
            accepted_prefill,
            self.admission.request_io_reuse_active(),
        ) {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let native = self.native_batch.as_mut().ok_or(error())?;
            native.gdn.set_lab_recurrence_policy(
                executor,
                &self.arena,
                crate::gdn_chunk::RecurrencePolicy::ThreeStreaming,
                &boundary,
                cancel,
            )?;
            if native.decode_kernels()? != NativeDecodeKernels::ThreeGdn {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            native.validate_target_decode_policy(executor, &self.admission)?;
            u::cancel(cancel)
        }));
        match attempt {
            Ok(Ok(())) => Ok(()),
            _ => {
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }
    /// Selection plus positive accepted whole-chunk execution at the original
    /// completed boundary; requested environment flags are not evidence.
    #[allow(dead_code)] // Controller is integrated separately after source acceptance.
    pub(crate) fn adapt_strata_cache(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        attachment: &mut crate::public_host_expert_attachment::Attachment<'_, '_, '_>,
        cancel: &AtomicBool,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        if !crate::strata_port::enabled() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        u::cancel(cancel)?;
        let _boundary = self.completed_native_projection_boundary(executor)?;
        self.native_batch
            .as_ref()
            .ok_or_else(error)?
            .target_rows
            .as_ref()
            .ok_or_else(error)?
            .check_no_live_output()?;
        // SAFETY: actual all-family completed boundary and released output
        // leases; adaptation joins the target queue before public cache writes.
        let swaps = unsafe { attachment.adapt_strata(queue)? };
        if swaps > 0 {
            self.native_batch
                .as_mut()
                .ok_or_else(error)?
                .moe
                .refresh_strata_table(executor, self.graph, &self.arena)?;
        }
        Ok(())
    }
    pub(crate) fn set_public_four_rows(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        executor.check_target_rows_policy()?;
        if !executor.target_four_rows_reserved || executor.target_four_rows_active {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_four_rows_reserved
            || !crate::strata_intel_core::enabled()
                && (!native.moe.public_four_members_policy()
                    || !native.moe.frozen_host_sources_enabled())
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        native.validate_execution_scope(executor, &self.admission)?;
        let accepted_prefill = match (
            &native.public_target_request_io,
            &native.public_language_request_io,
        ) {
            (Some(_), None) => {
                self.context == 13
                    && self.position == 1
                    && native
                        .public_warm_replay
                        .as_ref()
                        .is_some_and(|p| p.reset_consumed())
            }
            (None, Some(permit)) => {
                permit
                    .identity(self.graph.binding_id(), self.context)
                    .is_ok()
                    && permit.validate_begin(self.position, false).is_ok()
            }
            _ => false,
        };
        if !four_rows_selection_allowed(
            native.decode_kernels()?,
            accepted_prefill,
            self.admission.request_io_reuse_active(),
        ) {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let native = self.native_batch.as_mut().ok_or(error())?;
            native.gdn.set_lab_recurrence_policy(
                executor,
                &self.arena,
                crate::gdn_chunk::RecurrencePolicy::FourStreaming,
                &boundary,
                cancel,
            )?;
            if let Some(p) = native.public_target_request_io.as_mut() {
                p.enable_four_rows()?;
            }
            if let Some(p) = native.public_language_request_io.as_mut() {
                p.enable_four_rows()?;
            }
            native.ple.select_public_four_rows(&boundary)?;
            executor.target_four_rows_active = true;
            if native.decode_kernels()? != NativeDecodeKernels::FourGdn {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            native.validate_target_decode_policy(executor, &self.admission)?;
            u::cancel(cancel)
        }));
        match attempt {
            Ok(Ok(())) => Ok(()),
            _ => {
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }
    /// Selection plus positive accepted whole-chunk execution at the original
    /// completed boundary; requested environment flags are not evidence.
    #[allow(dead_code)] // Controller is integrated separately after source acceptance.
    pub(crate) fn completed_public_four_rows(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, u64)> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        let selected = native.decode_kernels()? == NativeDecodeKernels::FourGdn;
        if !selected {
            if executor.target_four_rows_active
                || native.four_rows_completed_chunks != 0
                || executor.target_verify4_dense_calls != 0
                || executor.target_verify4_fused_calls != 0
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            return Ok((false, 0));
        }
        native.validate_target_decode_policy(executor, &self.admission)?;
        if native.four_rows_completed_chunks == 0 {
            return Ok((false, 0));
        }
        if executor.target_verify4_dense_calls == 0
            || executor.target_verify4_dense_calls != executor.target_verify4_fused_calls
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok((true, native.four_rows_completed_chunks))
    }
    pub(crate) fn completed_public_three_gdn_streaming(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, u64)> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        let selected = native.decode_kernels()? == NativeDecodeKernels::ThreeGdn;
        if selected {
            native.validate_target_decode_policy(executor, &self.admission)?;
            if native.three_gdn_completed_chunks == 0 {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        } else if native.three_gdn_completed_chunks != 0 {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        Ok((selected, native.three_gdn_completed_chunks))
    }
    /// Separate public target read policy, fixed after accepted prefill and
    /// before the request begins. Never enables the global shared-GR flag.
    #[allow(dead_code)] // Root owns protected public selector registration.
    pub(crate) fn set_public_grouped_ple_reads(
        &mut self,
        executor: &GpuExecutor<'f>,
        selected: bool,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        executor.check_target_rows_policy()?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.validate_execution_scope(executor, &self.admission)?;
        let accepted_prefill = match (
            &native.public_target_request_io,
            &native.public_language_request_io,
        ) {
            (Some(_), None) => {
                self.context == 13
                    && self.position == 1
                    && native
                        .public_warm_replay
                        .as_ref()
                        .is_some_and(|p| p.reset_consumed())
            }
            (None, Some(permit)) => {
                permit
                    .identity(self.graph.binding_id(), self.context)
                    .is_ok()
                    && permit.validate_begin(self.position, false).is_ok()
            }
            _ => false,
        };
        if !grouped_ple_selection_allowed(
            native.public_grouped_ple_reads_reserved,
            native.ple.public_grouped_reads(),
            selected,
            accepted_prefill,
            self.admission.request_io_reuse_active(),
            native.decode_kernels()?,
        ) || native.grouped_ple_completed_chunks != 0
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if selected {
            self.native_batch
                .as_mut()
                .ok_or(error())?
                .ple
                .select_public_grouped_reads(self.position, &boundary)?;
        }
        Ok(())
    }
    /// Positive actual PLE read execution joined to whole-model acceptance.
    /// Requested policy and enqueue alone cannot produce this receipt.
    #[allow(dead_code)] // Root owns protected public completed receipt.
    pub(crate) fn completed_public_grouped_ple_reads(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, u64)> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        let selected = native.ple.public_grouped_reads();
        if selected {
            native.validate_target_decode_policy(executor, &self.admission)?;
        }
        grouped_ple_completed_receipt(selected, native.grouped_ple_completed_chunks)
    }
    /// Fixed-public target status policy selected only after real accepted
    /// prefill/warmreset and source/router attachment, before request I/O.
    #[allow(dead_code)] // Root owns protected controller registration.
    pub(crate) fn set_public_grouped_status_folds(
        &mut self,
        executor: &GpuExecutor<'f>,
        selected: bool,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        executor.check_target_rows_policy()?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.validate_execution_scope(executor, &self.admission)?;
        let accepted_prefill = match (
            &native.public_target_request_io,
            &native.public_language_request_io,
        ) {
            (Some(_), None) => {
                self.context == 13
                    && self.position == 1
                    && native
                        .public_warm_replay
                        .as_ref()
                        .is_some_and(|p| p.reset_consumed())
            }
            (None, Some(permit)) => {
                permit
                    .identity(self.graph.binding_id(), self.context)
                    .is_ok()
                    && permit.validate_begin(self.position, false).is_ok()
            }
            _ => false,
        };
        let (already_selected, calls) = native.grouped_status_state()?;
        if !grouped_ple_selection_allowed(
            native.public_grouped_status_folds_reserved,
            already_selected,
            selected,
            accepted_prefill,
            self.admission.request_io_reuse_active(),
            native.decode_kernels()?,
        ) || calls != [0; 3]
            || native.grouped_status_completed != crate::status_fold_batch::Completed::default()
            || crate::status_fold_batch::selected()
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        // Preflight all three actual workspaces BEFORE any immutable selection.
        native.gr.validate_public_grouped_status_selection()?;
        native
            .gdn
            .validate_public_grouped_status_selection(&self.arena, cancel)?;
        native
            .qsa
            .validate_public_grouped_status_selection(&self.arena, cancel)?;
        if selected {
            let native = self.native_batch.as_mut().ok_or(error())?;
            native.gr.enable_public_grouped_status_folds(&boundary);
            native.gdn.enable_public_grouped_status_folds(&boundary);
            native.qsa.enable_public_grouped_status_folds(&boundary);
        }
        Ok(())
    }
    /// Requested flags/enqueue alone earn no actual completed receipt.
    #[allow(dead_code)] // Root owns protected controller receipt registration.
    pub(crate) fn completed_public_grouped_status_folds(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, crate::status_fold_batch::Completed)> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        let (selected, actual) = native.grouped_status_state()?;
        if selected {
            native.validate_target_decode_policy(executor, &self.admission)?;
        }
        native.grouped_status_completed.receipt(selected, actual)
    }
    pub(crate) fn set_lab_native_decode_kernels(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        policy: NativeDecodeKernels,
        cancel: &AtomicBool,
    ) -> Result<NativeDecodeKernels> {
        u::cancel(cancel)?;
        if policy == NativeDecodeKernels::ThreeGdn
            || self
                .native_batch
                .as_ref()
                .ok_or(error())?
                .moe
                .frozen_host_source_down_integer_selected()
                && policy != NativeDecodeKernels::Chunk
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .grouped_status_state()?
            .0
            && policy != NativeDecodeKernels::Chunk
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .ple
            .public_grouped_reads()
            && policy != NativeDecodeKernels::Chunk
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let boundary = self.completed_native_projection_boundary(executor)?;
        let observed = self.native_decode_kernels()?;
        if self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .live_member_reduction_enabled()
            && policy != NativeDecodeKernels::Chunk
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        // Each fixture has one public continuation policy. Repeated same-policy
        // selection is allowed; a fresh owner is required for another sample.
        if observed != NativeDecodeKernels::Chunk && observed != policy
            || policy != NativeDecodeKernels::Chunk && executor.native_projection_tile() != 1
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let native = self.native_batch.as_mut().ok_or(error())?;
            native.gdn.set_lab_recurrence_policy(
                executor,
                &self.arena,
                policy.recurrence(),
                &boundary,
                cancel,
            )?;
            if policy == NativeDecodeKernels::SingleQ4All {
                native
                    .moe
                    .set_lab_single_q4_decode(executor, &self.arena, &boundary)?;
            } else if policy.single_moe() {
                native
                    .moe
                    .set_lab_single_member_decode(executor, &self.arena, &boundary)?;
            }
            if matches!(
                policy,
                NativeDecodeKernels::SingleAll | NativeDecodeKernels::SingleQ4All
            ) {
                native.dense_member_capacity =
                    executor.set_lab_native_single_dense_dispatch(&boundary)?;
            }
            if native.dense_member_capacity != executor.native_projection_member_capacity() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            let actual = self.native_decode_kernels()?;
            if actual != policy {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            u::cancel(cancel)?;
            Ok(actual)
        }));
        match attempt {
            Ok(Ok(actual)) => Ok(actual),
            _ => {
                // A failed multi-family policy publication cannot leave an
                // owner usable under a partially selected experiment.
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }

    pub(crate) fn new_with_native_batch(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut GpuExecutor<'f>,
        config: Config,
        public_zero_audit: bool,
    ) -> Result<Self> {
        Self::construct_native(
            graph,
            client,
            context,
            executor,
            public_zero_audit,
            Some(config),
        )
    }
    pub(crate) fn new_for_lab_fixture(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut GpuExecutor<'f>,
        public_zero_audit: bool,
        config: Option<Config>,
    ) -> Result<Self> {
        match config {
            Some(config) => Self::new_with_native_batch(
                graph,
                client,
                context,
                executor,
                config,
                public_zero_audit,
            ),
            None => Self::new_with_public_owned_zero_audit(
                graph,
                client,
                context,
                executor,
                public_zero_audit,
            ),
        }
    }
    pub(super) fn invalidate_target_carry(&self) -> Result<()> {
        if let Some(rows) = self
            .native_batch
            .as_ref()
            .and_then(|n| n.target_rows.as_ref())
        {
            rows.check_no_live_output()?;
            rows.invalidate_carry()?;
        }
        Ok(())
    }
    /// Real layer-major enqueue and whole-chunk acceptance. Opaque receipts
    /// cannot publish a token until ALL families and both required fences pass.
    pub(crate) fn prefill_chunk(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        purpose: Purpose,
        cancel: &AtomicBool,
    ) -> Result<Zeroizing<Vec<f32>>> {
        if purpose == Purpose::AllRowsWithCarry {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        self.execute_native_chunk(
            executor,
            tokens,
            purpose,
            None,
            cancel,
            |output| match output {
                ChunkOutput::Legacy(logits) => Ok(logits),
                ChunkOutput::TargetRows(_) | ChunkOutput::AcceptedPrefix(_) => Err(error()),
            },
        )
    }
    /// Additive fixed-public target-output foundation. It commits every
    /// supplied row; this is not accepted-prefix or speculative execution.
    #[allow(dead_code)] // Parent owns the dedicated fixed-public registration.
    pub(crate) fn prefill_chunk_with_carry(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> Result<NativeTargetRows> {
        self.execute_native_chunk(
            executor,
            tokens,
            Purpose::AllRowsWithCarry,
            None,
            cancel,
            |output| match output {
                ChunkOutput::TargetRows(rows) => Ok(rows),
                ChunkOutput::Legacy(_) | ChunkOutput::AcceptedPrefix(_) => Err(error()),
            },
        )
    }
    /// The selected native controller observes provisional verified rows while
    /// ALL original state transactions remain open. Only the chosen prefix is
    /// published. The observer cannot access the model or executor, and its
    /// host-row lease is dropped before candidate state consumers run. A second
    /// scoped phase grants verified R to the drafter while the original commit
    /// receipt remains owned; neither phase grants target state readiness.
    pub(crate) fn verify_native_prefix(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        cancel: &AtomicBool,
        select: &mut dyn for<'rows> FnMut(PrefixPhase<'rows>) -> Result<usize>,
    ) -> Result<usize> {
        if !crate::strata_intel_core::enabled() {
            return Err(error());
        }
        self.validate_native_request_boundary(executor)?;
        self.native_batch
            .as_ref()
            .ok_or(error())?
            .validate_target_decode_policy(executor, &self.admission)?;
        executor.check_target_rows_policy()?;
        self.execute_native_chunk(
            executor,
            tokens,
            Purpose::AllRowsWithCarry,
            Some(select),
            cancel,
            |output| match output {
                ChunkOutput::AcceptedPrefix(n) => Ok(n),
                _ => Err(error()),
            },
        )
    }
    /// Join every source worker before returning to decode or releasing the
    /// original model. The worker carries public source capabilities only;
    /// actual embedding rows live in bounded wiping request buffers.
    pub(crate) fn with_prompt_ple_source<T>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        prompt: &[u32],
        chunk: usize,
        cancel: &AtomicBool,
        body: impl FnOnce(&mut Self, &mut GpuExecutor<'f>) -> Result<T>,
    ) -> Result<T> {
        if self.position != 0 || self.poisoned || self.ple_batch_pending.is_some() {
            return Err(error());
        }
        let graph = self.graph;
        let old = Zeroizing::new([self.ple_history.tokens()[0], self.ple_history.tokens()[1]]);
        let profile = executor.packed_profile;
        ple_batch::prompt_source::run(graph, prompt, chunk, old, profile, cancel, || {
            body(self, executor)
        })
    }
    #[allow(clippy::too_many_arguments)] // Same transaction with an optional scoped prefix observer.
    fn execute_native_chunk<T>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        purpose: Purpose,
        select: PrefixSelection<'_>,
        cancel: &AtomicBool,
        accept: impl FnOnce(ChunkOutput) -> Result<T>,
    ) -> Result<T> {
        if let Some(rows) = &self.native_batch.as_ref().ok_or(error())?.target_rows {
            // No command/state mutation occurs while an admitted output-host
            // payload is still held. This guard also covers legacy purposes.
            rows.check_no_live_output()?;
            rows.invalidate_carry()?;
        }
        let mut native = self.native_batch.take().ok_or(error())?;
        let mut flight = Flight {
            poison: native.poison.clone(),
            accepted: false,
        };
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let selecting = select.is_some();
            if selecting {
                native
                    .global_moe
                    .as_mut()
                    .ok_or(error())?
                    .set_verify_usage(true)?;
            }
            let output = native.run(self, executor, tokens, purpose, select, cancel)?;
            if selecting {
                native
                    .global_moe
                    .as_mut()
                    .ok_or(error())?
                    .set_verify_usage(false)?;
            }
            accept(output)
        }));
        match attempt {
            Ok(Ok(logits)) => {
                flight.accepted = true;
                self.native_batch = Some(native);
                Ok(logits)
            }
            failure => {
                flight.poison.store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                self.native_batch = Some(native);
                let _ = self.close(); // Explicit close outside admission-held guards.
                let _ = executor.close_transient_resident(); // Abort/drain the actual staged bank registry.
                // Keep the original typed failure after the same terminal
                // cleanup. The fixed-public lab must be able to distinguish
                // a rejected capacity from an execution fault; panic payloads
                // remain suppressed and no request values are added.
                Err(match failure {
                    Ok(Err(cause)) => cause,
                    _ => error(),
                })
            }
        }
    }
}

type TokenPacket = (
    Zeroizing<[u32; MAX_BATCH]>,
    Zeroizing<[u32; MAX_BATCH]>,
    Zeroizing<usize>,
);
fn packet(start: usize, context: usize, tokens: &[u32]) -> Result<TokenPacket> {
    if !(1..=MAX_BATCH).contains(&tokens.len()) || context == 0 || context > 4096 {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    let end = start
        .checked_add(tokens.len())
        .ok_or(Error::ArithmeticOverflow)?;
    if end > context || tokens.iter().any(|&t| t as usize >= 248320) {
        return Err(Error::InvalidFormat(concat!(
            "native model chunk rejected at source line ",
            line!()
        )));
    }
    let mut ids = Zeroizing::new([0; MAX_BATCH]);
    let mut positions = Zeroizing::new([0; MAX_BATCH]);
    for (t, &id) in tokens.iter().enumerate() {
        ids[t] = id;
        positions[t] = (start + t) as u32;
    }
    Ok((ids, positions, Zeroizing::new(end)))
}

/// Used by real dispatch: a failed/cancelled actual layer cannot queue a later
/// layer. The callback contains the full GR/attention/router/MoE/shared work.
fn drive_layers(cancel: &AtomicBool, mut layer: impl FnMut(usize) -> Result<()>) -> Result<()> {
    for index in 0..LAYERS {
        u::cancel(cancel)?;
        layer(index)?;
    }
    u::cancel(cancel)
}
/// The observer receives only provisional rows, never model state access.
/// Dropping the original row lease precedes every accepted-prefix consumer.
fn select_verified_rows<R>(
    mut rows: R,
    count: usize,
    select: impl FnOnce(&mut R) -> Result<usize>,
) -> Result<usize> {
    if !(1..=MAX_BATCH).contains(&count) {
        return Err(error());
    }
    let keep = select(&mut rows)?;
    if !(1..=count).contains(&keep) {
        return Err(error());
    }
    Ok(keep)
}

/// Both actual completed proofs surround PLE's real owning diagnostic maps.
/// No guarded output/PLE permit escapes if a map, audit or second proof fails.
fn completed_proof<O, P>(
    first: impl FnOnce() -> Result<O>,
    ple: impl FnOnce() -> Result<P>,
    second: impl FnOnce() -> Result<()>,
) -> Result<(O, P)> {
    let output = first()?;
    let prepared = ple()?;
    second()?;
    Ok((output, prepared))
}

// Constructor hybrid owners exist before the completed count1 policy is
// installed. They alone cannot authorize finish_prepared during token0.
fn prepared_shared_overlap(
    resident_router: bool,
    cap: Option<usize>,
    overlap: Option<bool>,
) -> bool {
    !resident_router && cap.is_some() && overlap == Some(true)
}

impl Workspace {
    #[allow(clippy::too_many_arguments)] // Original execution owners plus scoped prefix selection.
    fn run_global(
        &mut self,
        m: &mut ResidentModel<'_, '_>,
        e: &mut GpuExecutor<'_>,
        tokens: &[u32],
        purpose: Purpose,
        select: PrefixSelection<'_>,
        cancel: &AtomicBool,
    ) -> Result<ChunkOutput> {
        if select.is_some() && purpose != Purpose::AllRowsWithCarry {
            return Err(error());
        }
        self.prove_outer(m)?;
        if !crate::strata_intel_core::enabled()
            || e.strata_global_sources.is_none()
            || m.poisoned
            || m.retired.load(Ordering::Acquire)
            || !m.pending_qsa.is_empty()
            || !m.pending_moe.is_empty()
            || m.ple_batch_pending.is_some()
            || m.layers.len() != LAYERS
            || tokens.is_empty()
            || tokens.len() > 8
        {
            return Err(Error::InvalidFormat("global native chunk boundary"));
        }
        self.global_moe.as_ref().ok_or(error())?.validate_idle()?;
        e.check_client(m.arena.client().map_err(u::fault)?)?;
        e.check_resident_admission(&m.admission)?;
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        let decode_policy = self.decode_kernels()?;
        let grouped_ple_reads = self.ple.public_grouped_reads();
        let (grouped_status_folds, grouped_status_before) = self.grouped_status_state()?;
        let start = Zeroizing::new(m.position);
        let count = Zeroizing::new(tokens.len());
        let (ids, positions, _end) = packet(*start, m.context, tokens)?;
        let mut pending = Pending::new()?;
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        let clear_ids = Zeroizing::new([
            self.outer.residual[0],
            self.outer.residual[1],
            self.outer.block[0],
            self.outer.block[1],
            self.outer.router,
            self.outer.packet,
            m.b.fault,
            m.b.logits,
            m.b.logit_packet,
        ]);
        u::zero_many(&m.arena, &*clear_ids)?;
        if let Some(rows) = &self.target_rows {
            u::zero_many(&m.arena, &rows.ids())?;
        }
        m.admission.shared_target_weights()?.gather(
            m.graph,
            &m.arena,
            m.b.logit_packet,
            m.b.fault,
            tokens,
            u::handle(&m.arena, self.outer.block[0])?,
            u::handle(&m.arena, self.outer.residual[0])?,
            false,
            cancel,
        )?;
        self.global_moe
            .as_mut()
            .ok_or(error())?
            .begin(m.graph, &m.arena, *count)?;
        // Layer0 performs attention+FFN, hence two residual swaps. The actual
        // layer1 PLE input/output roots are known before its math is queued.
        // Source/history uploads happen now, outside the retained native graph.
        let mut ple_invocation = Some(self.ple.prepare_invocation(
            m,
            self.outer.residual[0],
            self.outer.residual[1],
            &BatchSpans::new(*count, 0, 0, HC * D, 0, HC * D),
            *start,
            &ids,
            &positions,
            cancel,
        )?);
        // Arm each original state transaction before capture. All attention
        // inputs use the same authenticated GR MIXED scratch owner; each layer
        // writes its current values immediately before consuming the invocation.
        let mixed = self.gr.invocation_input_owner()?;
        let attention_spans = BatchSpans::new(*count, 0, 0, D, 0, D);
        let mut attention_invocations = AttentionInvocations::new()?;
        for (layer, stage) in m.layers.iter_mut().enumerate() {
            u::cancel(cancel)?;
            match stage {
                Layer::Gdn(stage, public) => {
                    attention_invocations.gdn[layer] = Some(stage.prepare_batch(
                        m.graph,
                        &m.arena,
                        public,
                        &mut self.gdn,
                        mixed,
                        self.outer.block[0],
                        &attention_spans,
                        *start as u64,
                        m.b.fault,
                        cancel,
                    )?);
                }
                Layer::Qsa(stage, public) => {
                    let mut selector =
                        qsa::BatchSelector::new(&m.admission, self.public_gpu_qsa_selector);
                    attention_invocations.qsa[layer] = Some(stage.prepare_batch(
                        m.graph,
                        &m.arena,
                        public,
                        &mut selector,
                        &mut self.qsa,
                        mixed,
                        self.outer.block[0],
                        &attention_spans,
                        *start as u64,
                        m.b.fault,
                        cancel,
                    )?);
                }
            }
        }
        // Prompt-final count1 uses the same implementation before the request
        // scope opens. Capture only the active verification shape, whose head
        // and target-row consumers are invariant for this count/context pair.
        // Input gather/clears and original PLE uploads precede capture. Any
        // remaining required internal submission has its own retained segment.
        let branch = if purpose == Purpose::AllRowsWithCarry
            && *start != 0
            && m.admission.request_io_reuse_active()
        {
            let mut state_role = None;
            for layer in &m.layers {
                if let Layer::Gdn(stage, _) = layer {
                    let role = stage.native_capture_state_role()?;
                    if state_role.is_some_and(|previous| previous != role) {
                        return Err(Error::InvalidFormat("native GDN capture role alignment"));
                    }
                    state_role = Some(role);
                }
            }
            Some(
                cubecl_runtime::server::NativeStrataBranch::target_window_with_state(
                    u8::try_from(*count).map_err(u::fault)?,
                    u16::try_from(m.context).map_err(u::fault)?,
                    state_role.ok_or(error())? | (self.ple.native_capture_history_role()? << 1),
                )
                .map_err(u::fault)?,
            )
        } else {
            None
        };
        // The optional laboratory observation/injection paths deliberately keep
        // their actual per-layer boundaries. Normal native inference replays a
        // sealed whole window with fresh controls and its real queue fence.
        let replay = if let Some(key) = branch {
            if !grouped_status_folds
                && !crate::status_fold_batch::selected()
                && m.numeric_boundary.is_unarmed()
                && m.lifecycle_hooks.is_unarmed()
                && m.cancel_after_attention.is_none()
            {
                // Retained pointer consumers still require the original
                // source publication/loan boundary, even without enqueue_data.
                e.strata_global_sources
                    .as_ref()
                    .ok_or(error())?
                    .validate_graph_ready(m.graph)?;
                super::target_replay::try_submit(
                    &m.arena,
                    key,
                    self.global_moe
                        .as_ref()
                        .ok_or(error())?
                        .invocation_controls(*start)?,
                    m.b.fault,
                )?
            } else {
                None
            }
        } else {
            None
        };
        if let Some(key) = branch.filter(|_| replay.is_none()) {
            m.arena
                .client()
                .map_err(u::fault)?
                .begin_native_strata_invocation(
                    key,
                    self.global_moe
                        .as_ref()
                        .ok_or(error())?
                        .invocation_controls(*start)?,
                )
                .map_err(u::fault)?;
        }
        // Layer0 has two swaps; layer1 adds the sole PLE swap; all remaining
        // attention/FFN pairs have even parity. The retained final owner is1.
        let mut current = Zeroizing::new(if replay.is_some() { 1usize } else { 0 });
        if let Some(completed) = &replay {
            u::cancel(cancel)?;
            self.gr.complete_native_replay(
                m.graph,
                &m.arena,
                self.outer.residual[*current],
                m.b.fault,
                completed,
            )?;
            self.shared
                .complete_native_replay(m.graph, &m.arena, m.b.fault, completed)?;
            pending.ple = Some(self.ple.complete_prepared_replay(
                m,
                ple_invocation.take().ok_or(error())?,
                completed,
                cancel,
            )?);
            for (layer, stage) in m.layers.iter_mut().enumerate() {
                u::cancel(cancel)?;
                match stage {
                    Layer::Gdn(stage, public) => {
                        pending.gdn[layer] = Some(stage.complete_prepared_replay(
                            m.graph,
                            &m.arena,
                            public,
                            &mut self.gdn,
                            attention_invocations.gdn[layer].take().ok_or(error())?,
                            completed,
                            cancel,
                        )?);
                    }
                    Layer::Qsa(stage, public) => {
                        let mut selector =
                            qsa::BatchSelector::new(&m.admission, self.public_gpu_qsa_selector);
                        pending.qsa[layer] = Some(stage.complete_prepared_replay(
                            m.graph,
                            &m.arena,
                            public,
                            &mut selector,
                            &mut self.qsa,
                            attention_invocations.qsa[layer].take().ok_or(error())?,
                            completed,
                            cancel,
                        )?);
                    }
                }
            }
            self.global_moe
                .as_mut()
                .ok_or(error())?
                .complete_native_replay(m.graph, &m.arena, m.b.fault, *start as u64, completed)?;
        } else {
            drive_layers(cancel, |layer| {
                let attention_start = Instant::now();
                let dense_q8 = if crate::strata_intel_core::enabled() && (1..=8).contains(&*count) {
                    self.core_stages.map(|ids| ids[0])
                } else {
                    None
                };
                if layer == 1 {
                    // PLE consumes the residual directly, so complete the pending
                    // FFN write before its observation/projection/history work.
                    self.gr.flush_pending(&m.arena, m.b.fault, cancel)?;
                    m.numeric_boundary.observe(
                        numeric_boundaries::Boundary::Input,
                        &m.arena,
                        &m.admission,
                        self.outer.residual[*current],
                        m.b.fault,
                        RESIDUAL_WORDS,
                        *start,
                        *count,
                        cancel,
                    )?;
                    let mut projector = ResidentBatchProjection {
                        executor: e,
                        fault: m.b.fault,
                        dense_q8,
                    };
                    let spans = BatchSpans::new(*count, 0, 0, HC * D, 0, HC * D);
                    pending.ple = Some(self.ple.enqueue_prepared(
                        m,
                        &mut projector,
                        self.outer.residual[*current],
                        self.outer.residual[1 - *current],
                        &spans,
                        ple_invocation.take().ok_or(error())?,
                        cancel,
                    )?);
                    m.numeric_boundary.observe(
                        numeric_boundaries::Boundary::Injected,
                        &m.arena,
                        &m.admission,
                        self.outer.residual[1 - *current],
                        m.b.fault,
                        RESIDUAL_WORDS,
                        *start,
                        *count,
                        cancel,
                    )?;
                    *current = 1 - *current;
                }
                let mut projector = ResidentBatchProjection {
                    executor: e,
                    fault: m.b.fault,
                    dense_q8,
                };
                let prepared = self.gr.read(
                    m.graph,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    gr::Kind::Attention(layer),
                    self.outer.residual[*current],
                    *count,
                    m.b.fault,
                    cancel,
                )?;
                let spans = BatchSpans::new(*count, 0, 0, D, 0, D);
                match &mut m.layers[layer] {
                    Layer::Gdn(stage, public) => {
                        pending.gdn[layer] = Some(stage.enqueue_prepared(
                            m.graph,
                            &m.arena,
                            public,
                            &mut projector,
                            &mut self.gdn,
                            prepared.mixed(),
                            self.outer.block[0],
                            &spans,
                            attention_invocations.gdn[layer].take().ok_or(error())?,
                            cancel,
                        )?)
                    }
                    Layer::Qsa(stage, public) => {
                        let mut selector =
                            qsa::BatchSelector::new(&m.admission, self.public_gpu_qsa_selector);
                        pending.qsa[layer] = Some(stage.enqueue_prepared(
                            m.graph,
                            &m.arena,
                            public,
                            &mut projector,
                            &mut selector,
                            &mut self.qsa,
                            prepared.mixed(),
                            self.outer.block[0],
                            &spans,
                            attention_invocations.qsa[layer].take().ok_or(error())?,
                            cancel,
                        )?);
                    }
                }
                prepared.combine(self.outer.block[0], self.outer.residual[1 - *current])?;
                *current = 1 - *current;
                // Public laboratory aggregates only. These wall-clock enqueue
                // intervals include any internal dependency waits; they are not
                // isolated GPU kernel times or additional completion proofs.
                *m.phase_ns
                    .entry("native_attention_and_gr_enqueue")
                    .or_default() += attention_start.elapsed().as_nanos();
                if m.cancel_after_attention == Some(layer) {
                    m.cancel_after_attention = None;
                    cancel.store(true, Ordering::Release);
                    u::cancel(cancel)?;
                }
                let router_enqueue_start = Instant::now();

                let prepared = self.gr.read(
                    m.graph,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    gr::Kind::Ffn(layer),
                    self.outer.residual[*current],
                    *count,
                    m.b.fault,
                    cancel,
                )?;
                u::zero_many(&m.arena, &[self.outer.router, self.outer.packet])?;
                let tensor = m
                    .graph
                    .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))?;
                let matrix = Matrix::new(tensor)?;
                if matrix.inputs() != D || matrix.outputs() != ROUTER || matrix.experts() != 1 {
                    return Err(Error::InvalidFormat(concat!(
                        "native model chunk rejected at source line ",
                        line!()
                    )));
                }
                projector.enqueue_batch(
                    BoundProjection { tensor, matrix },
                    &m.arena,
                    prepared.mixed(),
                    self.outer.router,
                    &BatchSpans::new(*count, 0, 0, D, 0, ROUTER),
                    cancel,
                )?;
                *m.phase_ns
                    .entry("native_ffn_gr_and_router_enqueue")
                    .or_default() += router_enqueue_start.elapsed().as_nanos();
                let sources = e.strata_global_sources.as_ref().ok_or(error())?.clone();
                self.global_moe.as_mut().ok_or(error())?.enqueue(
                    m.graph,
                    &m.arena,
                    &sources,
                    &m.admission,
                    layer,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.router,
                    m.b.fault,
                    cancel,
                )?;
                let shared_start = Instant::now();
                let mut projector = ResidentBatchProjection {
                    executor: e,
                    fault: m.b.fault,
                    dense_q8,
                };
                self.shared.prepare(
                    m.graph,
                    layer,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.block[1],
                    m.b.fault,
                    *count,
                    cancel,
                )?;
                // Intel's selected NO_HOST branch has no CPU jobs to join. Leave
                // all routed/shared work in the owning window's queued graph;
                // its normal completion/fault boundary observes the real results.
                self.global_moe.as_mut().ok_or(error())?.finish_layer(
                    &m.arena,
                    layer,
                    self.outer.block[0],
                    m.b.fault,
                )?;
                self.shared.finish_prepared(
                    m.graph,
                    layer,
                    &m.arena,
                    &mut m.public,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.block[1],
                    m.b.fault,
                    *count,
                    cancel,
                )?;
                *m.phase_ns
                    .entry("native_shared_and_no_host_enqueue")
                    .or_default() += shared_start.elapsed().as_nanos();
                prepared.combine(self.outer.block[1], self.outer.residual[1 - *current])?;
                *current = 1 - *current;
                *m.phase_ns
                    .entry("native_shared_and_ffn_combine_enqueue")
                    .or_default() += shared_start.elapsed().as_nanos();
                // Fixed-public lab hook: actual routed/shared consumers and HC
                // combine are already queued; no family has published its state.
                m.lifecycle_hooks.late_combine(layer)?;
                Ok(())
            })?;
            let final_head_start = Instant::now();
            if purpose == Purpose::LastLogits {
                let mut projector = ResidentBatchProjection {
                    executor: e,
                    fault: m.b.fault,
                    dense_q8: None,
                };
                self.gr
                    .read(
                        m.graph,
                        &m.arena,
                        &mut m.public,
                        &mut projector,
                        gr::Kind::Final,
                        self.outer.residual[*current],
                        *count,
                        m.b.fault,
                        cancel,
                    )?
                    .enqueue_final_head(m.graph, &mut projector, m.b.logits)?;
            } else if purpose == Purpose::AllRowsWithCarry {
                let rows = self.target_rows.as_ref().ok_or(error())?;
                let mut projector = ResidentBatchProjection {
                    executor: e,
                    fault: m.b.fault,
                    dense_q8: None,
                };
                self.gr
                    .read(
                        m.graph,
                        &m.arena,
                        &mut m.public,
                        &mut projector,
                        if *count >= 4 {
                            gr::Kind::FinalFourRows
                        } else {
                            gr::Kind::FinalAllRows
                        },
                        self.outer.residual[*current],
                        *count,
                        m.b.fault,
                        cancel,
                    )?
                    .enqueue_final_head_with_target_capacity(
                        m.graph,
                        &mut projector,
                        rows.logits(),
                        rows.four_rows_reserved(),
                    )?;
            }
            *m.phase_ns.entry("native_final_head_enqueue").or_default() +=
                final_head_start.elapsed().as_nanos();
            self.gr.flush_pending(&m.arena, m.b.fault, cancel)?;
            self.gr.validate_all_consumers_queued()?;
            self.shared.validate_all_consumers_queued()?;
        }
        // Capture real queued counters before PLE/family permits borrow their
        // disjoint workspaces. Completion and publication remain below.
        let mut grouped_status_expected = [1u64, 0, 0]; // FinalAllRows GR.
        for layer in m.layers.iter() {
            grouped_status_expected[0] += 2; // Attention and FFN GR.
            match layer {
                Layer::Gdn(_, _) => grouped_status_expected[1] += 1,
                Layer::Qsa(_, _) => grouped_status_expected[2] += 1,
            }
        }
        if crate::strata_intel_core::enabled() {
            // Native GR writers join finite faults directly; no status-fold
            // dispatch is claimed for the retired pointwise stages.
            grouped_status_expected[0] = 0;
        }
        let (still_selected, grouped_status_after) = self.grouped_status_state()?;
        if grouped_status_folds != still_selected {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        attention_invocations.consumed()?;
        if let Some(key) = branch.filter(|_| replay.is_none()) {
            // Join/seal the real native recording before any logical output or
            // state proof is published. This grants no fault/erasure credit;
            // completed_proof below retains both original observations.
            m.arena
                .client()
                .map_err(u::fault)?
                .end_native_strata_branch(key)
                .map_err(u::fault)?;
        }
        let final_proof_start = Instant::now();
        let mut output = if purpose == Purpose::LastLogits {
            ChunkOutput::Legacy(batch_io::read_last_logits(
                m.graph,
                &m.arena,
                &m.admission,
                m.b.logits,
                m.b.logit_packet,
                m.b.fault,
                cancel,
            )?)
        } else if purpose == Purpose::AllRowsWithCarry {
            ChunkOutput::TargetRows(self.target_rows.as_ref().ok_or(error())?.read(
                m.graph,
                &m.arena,
                &m.admission,
                self.outer.residual[*current],
                m.b.fault,
                *count,
                *start,
                cancel,
            )?)
        } else {
            batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)?;
            ChunkOutput::Legacy(Zeroizing::new(Vec::new()))
        };
        let selected_prefix = if let Some(select) = select {
            let keep = match output {
                ChunkOutput::TargetRows(rows) => {
                    select_verified_rows(rows, *count, |rows| select(PrefixPhase::Verified(rows)))?
                }
                _ => return Err(error()),
            };
            self.target_rows
                .as_ref()
                .ok_or(error())?
                .check_no_live_output()?;
            // Real retained commit uses original candidate roots and typed n.
            // Its original recording still needs the real join/fault/family proof.
            let commit = self
                .submit_native_prefix_commit(m, &pending, *count, keep, cancel)?
                .map(|submission| m.admission.retain_native_commit(submission))
                .transpose()?;
            let source = self
                .target_rows
                .as_ref()
                .ok_or(error())?
                .verified_carry_context(
                    &m.arena,
                    &m.retired,
                    m.graph.binding_id(),
                    &self.outer.residual,
                    [*start, *count],
                )?;
            if select(PrefixPhase::Pending {
                source,
                accepted: keep,
            })? != keep
            {
                return Err(error());
            }
            if let Some(commit) = commit {
                // The observer may run the original drafter, but may not
                // publish target state. Join its exact original submission.
                commit.join()?;
            }
            output = ChunkOutput::AcceptedPrefix(keep);
            Some(keep)
        } else {
            None
        };
        let (output, ple) = completed_proof(
            || Ok(output),
            || {
                self.ple.prepare_device_prefix_after_whole_model_fence(
                    ple_batch::Preparation {
                        graph: m.graph,
                        arena: &m.arena,
                        admission: &m.admission,
                        retired: &m.retired,
                        context: m.context,
                        position: *start,
                        owner_poisoned: &mut m.poisoned,
                        sequence_poison: &m.ple_batch_poison,
                        conv: &mut m.ple_conv,
                        history: &mut m.ple_history,
                        anchor: &mut m.ple_batch_pending,
                        fault: m.b.fault,
                        scalar_ids: [m.b.ple_emb, m.b.ple_key, m.b.ple_value],
                    },
                    pending.ple.take().ok_or(error())?,
                    selected_prefix.unwrap_or(*count),
                    cancel,
                )
            },
            || {
                // The real PLE preparation exists and the first whole-model
                // proof completed. Failure here must drop that real ticket.
                m.lifecycle_hooks.before_second_proof()?;
                batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)
            },
        )?;
        *m.phase_ns
            .entry("native_completed_final_proofs")
            .or_default() += final_proof_start.elapsed().as_nanos();
        let family_start = Instant::now();
        let mut ga = {
            let (first_stage, first_receipt) = match (&m.layers[0], pending.gdn[0].as_ref()) {
                (Layer::Gdn(s, _), Some(r)) => (s.as_ref(), r),
                _ => return Err(error()),
            };
            let mut refs = [(first_stage, first_receipt); 36];
            let mut at = 0;
            for layer in 0..LAYERS {
                if let Layer::Gdn(s, _) = &m.layers[layer] {
                    refs[at] = (s, pending.gdn[layer].as_ref().ok_or(error())?);
                    at += 1;
                }
            }
            if at != refs.len() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            self.gdn
                .accept_after_whole_model_chunk_acceptance(&m.arena, &refs, cancel)?
        };
        let mut qa = {
            let (first_stage, first_receipt) = match (&m.layers[3], pending.qsa[3].as_ref()) {
                (Layer::Qsa(s, _), Some(r)) => (s.as_ref(), r),
                _ => return Err(error()),
            };
            let mut refs = [(first_stage, first_receipt); 12];
            let mut at = 0;
            for layer in 0..LAYERS {
                if let Layer::Qsa(s, _) = &m.layers[layer] {
                    refs[at] = (s, pending.qsa[layer].as_ref().ok_or(error())?);
                    at += 1;
                }
            }
            if at != refs.len() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            self.qsa
                .accept_after_whole_model_chunk_acceptance(&m.arena, &refs, cancel)?
        };
        if let Some(keep) = selected_prefix {
            ga.select_prefix(keep)?;
            qa.select_prefix(keep)?;
        }
        self.global_moe
            .as_mut()
            .ok_or(error())?
            .accept(&m.arena, &m.admission, cancel)?;
        ple.preflight(cancel)?;
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        if self.poison.load(Ordering::Acquire)
            || m.retired.load(Ordering::Acquire)
            || m.position != *start
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        // Prepare the checked counter before ANY family publication. The
        // guarded proposal is discarded on all late failure/cancellation paths.
        let completed_cpu_misses = checked_cpu_miss_total(*m.completed_cpu_misses, 0)?;
        let three_gdn_completed_chunks =
            next_three_gdn_chunks(decode_policy, self.three_gdn_completed_chunks)?;
        let four_rows_completed_chunks =
            if decode_policy == NativeDecodeKernels::FourGdn && *count == 4 {
                self.four_rows_completed_chunks
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow)?
            } else {
                self.four_rows_completed_chunks
            };
        let grouped_ple_completed_chunks = next_grouped_ple_chunks(
            grouped_ple_reads,
            ple.public_grouped_read_completed(),
            self.grouped_ple_completed_chunks,
        )?;
        let grouped_status_completed = self.grouped_status_completed.propose(
            grouped_status_folds,
            grouped_status_before,
            grouped_status_after,
            grouped_status_expected,
        )?;
        *m.phase_ns.entry("native_all_family_preflight").or_default() +=
            family_start.elapsed().as_nanos();
        let commit_start = Instant::now();
        // Every actual-family permit has now preflighted. Remaining cancellation
        // or fallible commit failure terminally destroys the whole model; it
        // cannot publish output/cursor or resume a partially changed sequence.
        for (layer, owner) in m.layers.iter_mut().enumerate() {
            match owner {
                Layer::Gdn(stage, _) => {
                    stage.commit_after_whole_chunk_acceptance(
                        pending.gdn[layer].take().ok_or(error())?,
                        &m.arena,
                        &mut ga,
                        cancel,
                    )?;
                    // Inject only AFTER the first genuine family commit. The
                    // next real commit must reject and destroy this sequence.
                    m.lifecycle_hooks.after_gdn_commit(layer, cancel);
                }
                Layer::Qsa(stage, _) => stage.commit_after_whole_chunk_acceptance(
                    pending.qsa[layer].take().ok_or(error())?,
                    &m.arena,
                    &mut qa,
                    cancel,
                )?,
            }
        }
        ga.finish(&m.arena, cancel)?;
        qa.finish(&m.arena, cancel)?;
        self.global_moe.as_mut().ok_or(error())?.finish()?;
        ple.commit(cancel)?;
        *m.phase_ns.entry("native_all_family_commit").or_default() +=
            commit_start.elapsed().as_nanos();
        u::cancel(cancel)?;
        if self.poison.load(Ordering::Acquire) || m.retired.load(Ordering::Acquire) || m.poisoned {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        *m.completed_cpu_misses = *completed_cpu_misses;
        self.three_gdn_completed_chunks = three_gdn_completed_chunks;
        self.four_rows_completed_chunks = four_rows_completed_chunks;
        self.grouped_ple_completed_chunks = grouped_ple_completed_chunks;
        self.grouped_status_completed = grouped_status_completed;
        m.position = *start + selected_prefix.unwrap_or(*count);
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)] // Original execution owners plus scoped prefix selection.
    fn run(
        &mut self,
        m: &mut ResidentModel<'_, '_>,
        e: &mut GpuExecutor<'_>,
        tokens: &[u32],
        purpose: Purpose,
        select: PrefixSelection<'_>,
        cancel: &AtomicBool,
    ) -> Result<ChunkOutput> {
        if crate::strata_intel_core::enabled() {
            return self.run_global(m, e, tokens, purpose, select, cancel);
        }
        if select.is_some() {
            return Err(error());
        }
        self.prove_outer(m)?;
        let _core_stage_scope = crate::strata_stage::Scope::new(&m.arena, self.core_stages)?;
        // Immutable for this exclusive transaction; capture before real PLE
        // acceptance retains its disjoint mutable workspace borrow.
        let decode_policy = self.decode_kernels()?;
        let grouped_ple_reads = self.ple.public_grouped_reads();
        let (grouped_status_folds, grouped_status_before) = self.grouped_status_state()?;
        if !grouped_ple_run_allowed_selected(
            grouped_ple_reads,
            tokens.len(),
            purpose,
            decode_policy == NativeDecodeKernels::FourGdn && e.target_four_rows_active,
        ) || !grouped_ple_run_allowed_selected(
            grouped_status_folds,
            tokens.len(),
            purpose,
            decode_policy == NativeDecodeKernels::FourGdn && e.target_four_rows_active,
        ) || !grouped_ple_run_allowed_selected(
            self.moe.frozen_host_source_down_integer_selected(),
            tokens.len(),
            purpose,
            decode_policy == NativeDecodeKernels::FourGdn && e.target_four_rows_active,
        ) {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        e.check_client(m.arena.client().map_err(u::fault)?)?;
        e.check_resident_admission(&m.admission)?;
        if self.dense_member_capacity != e.native_projection_member_capacity() {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if e.target_rows_reserved != self.target_rows.is_some()
            || purpose == Purpose::AllRowsWithCarry && self.target_rows.is_none()
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if self.target_rows.is_some() {
            e.check_target_rows_policy()?;
            target_rows::coordinates_selected(tokens.len(), m.position, e.target_four_rows_active)?;
            self.validate_target_decode_policy(e, &m.admission)?;
            if self.moe.cpu_miss_cap().is_some()
                || self.moe.cpu_gpu_overlap().is_some()
                || self.moe.resident_router_active()
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
        }
        if let Some(permit) = &self.public_target_request_io {
            public_target_request_io::check_execution(self, m, e)?;
            permit.validate_run(
                m.graph.binding_id(),
                m.context,
                m.position,
                tokens.len(),
                m.admission.request_io_reuse_active(),
            )?;
        }
        if let Some(permit) = &self.public_language_request_io {
            public_target_request_io::check_execution(self, m, e)?;
            permit.validate_run(
                m.graph.binding_id(),
                m.context,
                m.position,
                tokens.len(),
                m.admission.request_io_reuse_active(),
            )?;
        }
        if self.moe.live_member_reduction_enabled() {
            live_member_reduction_configuration(
                true,
                self.target_rows.is_some(),
                self.moe.count1_hybrid_owner_ids().is_some(),
                self.moe.resident_router_ids().is_some() || e.resident_router_reserved,
                self.moe.prepared_verify3_owner_id().is_some(),
                self.moe.projection_tile(),
            )?;
            batch_moe::live_member_reduction_policy(
                true,
                e.packed_profile,
                e.resident_bank_capacity(),
                self.moe.projection_tile(),
                self.moe.count1_hybrid_owner_ids().is_some(),
                self.moe.resident_router_ids().is_some() || e.resident_router_reserved,
                self.moe.prepared_verify3_owner_id().is_some(),
            )?;
            self.validate_target_decode_policy(e, &m.admission)?;
        }
        if self.moe.prepared_verify3_owner_id().is_some() {
            if self.target_rows.is_none()
                || e.q3_integer_moe
                || e.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
            {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            batch_moe::prepared_verify3::validate_count(
                tokens.len(),
                1,
                tokens
                    .len()
                    .checked_mul(10)
                    .ok_or(Error::ArithmeticOverflow)?,
            )?;
        }
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        // Preflight both single-token families before clears, embeddings or
        // any other layer can enqueue work for this transaction.
        decode_policy.validate_count(tokens.len())?;
        if self.moe.cpu_miss_cap().is_some() && tokens.len() != 1 {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        if m.poisoned
            || m.retired.load(Ordering::Acquire)
            || !m.pending_qsa.is_empty()
            || !m.pending_moe.is_empty()
            || m.ple_batch_pending.is_some()
            || m.layers.len() != LAYERS
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let start = Zeroizing::new(m.position);
        let count = Zeroizing::new(tokens.len());
        let (ids, positions, end) = packet(*start, m.context, tokens)?;
        let mut pending = Pending::new()?;
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        let clear_ids = Zeroizing::new([
            self.outer.residual[0],
            self.outer.residual[1],
            self.outer.block[0],
            self.outer.block[1],
            self.outer.router,
            self.outer.packet,
            m.b.fault,
            m.b.logits,
            m.b.logit_packet,
        ]);
        u::zero_many(&m.arena, &*clear_ids)?;
        if let Some(rows) = &self.target_rows {
            u::zero_many(&m.arena, &rows.ids())?;
        }
        batch_io::enqueue_embedding(
            m.graph,
            &m.arena,
            &m.admission,
            self.outer.residual[0],
            batch_io::InputView {
                first: 0,
                stride: HC * D,
                words: RESIDUAL_WORDS,
            },
            *count,
            *start,
            m.context,
            &ids,
            &positions,
            cancel,
        )?;
        let mut session = e.begin_routed_moe_chunk(
            &mut self.moe,
            m.graph,
            &m.arena,
            *start as u64,
            *count,
            m.b.fault,
            cancel,
        )?;
        let mut current = Zeroizing::new(0usize);
        drive_layers(cancel, |layer| {
            let attention_start = Instant::now();
            let dense_q8 = if crate::strata_intel_core::enabled()
                && e.target_four_rows_active
                && *count == 4
            {
                self.core_stages.map(|ids| ids[0])
            } else {
                None
            };
            if layer == 1 {
                // PLE consumes the residual directly, so complete the pending
                // FFN write before its observation/projection/history work.
                self.gr.flush_pending(&m.arena, m.b.fault, cancel)?;
                m.numeric_boundary.observe(
                    numeric_boundaries::Boundary::Input,
                    &m.arena,
                    &m.admission,
                    self.outer.residual[*current],
                    m.b.fault,
                    RESIDUAL_WORDS,
                    *start,
                    *count,
                    cancel,
                )?;
                let mut projector = ResidentBatchProjection {
                    executor: e,
                    fault: m.b.fault,
                    dense_q8,
                };
                let spans = BatchSpans::new(*count, 0, 0, HC * D, 0, HC * D);
                pending.ple = Some(self.ple.enqueue(
                    m,
                    &mut projector,
                    self.outer.residual[*current],
                    self.outer.residual[1 - *current],
                    &spans,
                    *start,
                    &ids,
                    &positions,
                    cancel,
                )?);
                m.numeric_boundary.observe(
                    numeric_boundaries::Boundary::Injected,
                    &m.arena,
                    &m.admission,
                    self.outer.residual[1 - *current],
                    m.b.fault,
                    RESIDUAL_WORDS,
                    *start,
                    *count,
                    cancel,
                )?;
                *current = 1 - *current;
            }
            let mut projector = ResidentBatchProjection {
                executor: e,
                fault: m.b.fault,
                dense_q8,
            };
            let prepared = self.gr.read(
                m.graph,
                &m.arena,
                &mut m.public,
                &mut projector,
                gr::Kind::Attention(layer),
                self.outer.residual[*current],
                *count,
                m.b.fault,
                cancel,
            )?;
            let spans = BatchSpans::new(*count, 0, 0, D, 0, D);
            match &mut m.layers[layer] {
                Layer::Gdn(stage, public) => {
                    pending.gdn[layer] = Some(stage.enqueue_batch(
                        m.graph,
                        &m.arena,
                        public,
                        &mut projector,
                        &mut self.gdn,
                        prepared.mixed(),
                        self.outer.block[0],
                        &spans,
                        *start as u64,
                        m.b.fault,
                        cancel,
                    )?)
                }
                Layer::Qsa(stage, public) => {
                    let mut selector =
                        qsa::BatchSelector::new(&m.admission, self.public_gpu_qsa_selector);
                    pending.qsa[layer] = Some(stage.enqueue_batch(
                        m.graph,
                        &m.arena,
                        public,
                        &mut projector,
                        &mut selector,
                        &mut self.qsa,
                        prepared.mixed(),
                        self.outer.block[0],
                        &spans,
                        *start as u64,
                        m.b.fault,
                        cancel,
                    )?);
                }
            }
            prepared.combine(self.outer.block[0], self.outer.residual[1 - *current])?;
            *current = 1 - *current;
            // Public laboratory aggregates only. These wall-clock enqueue
            // intervals include any internal dependency waits; they are not
            // isolated GPU kernel times or additional completion proofs.
            *m.phase_ns
                .entry("native_attention_and_gr_enqueue")
                .or_default() += attention_start.elapsed().as_nanos();
            if m.cancel_after_attention == Some(layer) {
                m.cancel_after_attention = None;
                cancel.store(true, Ordering::Release);
                u::cancel(cancel)?;
            }
            let router_enqueue_start = Instant::now();

            let prepared = self.gr.read(
                m.graph,
                &m.arena,
                &mut m.public,
                &mut projector,
                gr::Kind::Ffn(layer),
                self.outer.residual[*current],
                *count,
                m.b.fault,
                cancel,
            )?;
            u::zero_many(&m.arena, &[self.outer.router, self.outer.packet])?;
            let tensor = m
                .graph
                .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))?;
            let matrix = Matrix::new(tensor)?;
            if matrix.inputs() != D || matrix.outputs() != ROUTER || matrix.experts() != 1 {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            projector.enqueue_batch(
                BoundProjection { tensor, matrix },
                &m.arena,
                prepared.mixed(),
                self.outer.router,
                &BatchSpans::new(*count, 0, 0, D, 0, ROUTER),
                cancel,
            )?;
            *m.phase_ns
                .entry("native_ffn_gr_and_router_enqueue")
                .or_default() += router_enqueue_start.elapsed().as_nanos();
            if self.moe.batch_router_active() {
                let expert_start = Instant::now();
                pending.moe[layer] = Some(if self.moe.frozen_host_batch_router_active() {
                    if !self.public_frozen_host_router || !self.public_frozen_host_sources {
                        return Err(Error::InvalidFormat(concat!(
                            "native model chunk rejected at source line ",
                            line!()
                        )));
                    }
                    e.enqueue_frozen_host_batch_router_moe(
                        m.graph,
                        &m.arena,
                        &mut self.moe,
                        &session,
                        layer,
                        prepared.mixed(),
                        self.outer.block[0],
                        self.outer.router,
                        &spans,
                        cancel,
                    )?
                } else {
                    if self.public_frozen_host_router {
                        return Err(Error::InvalidFormat(concat!(
                            "native model chunk rejected at source line ",
                            line!()
                        )));
                    }
                    e.enqueue_batch_router_moe(
                        m.graph,
                        &m.arena,
                        &mut self.moe,
                        &session,
                        layer,
                        prepared.mixed(),
                        self.outer.block[0],
                        self.outer.router,
                        &spans,
                        cancel,
                    )?
                });
                *m.phase_ns
                    .entry("native_device_router_and_expert_enqueue")
                    .or_default() += expert_start.elapsed().as_nanos();
            } else if self.moe.resident_router_active() {
                batch_io::enqueue_count1_router_input(
                    m.graph,
                    &m.arena,
                    &m.admission,
                    prepared.mixed(),
                    self.outer.router,
                    self.outer.packet,
                    m.b.fault,
                    batch_io::InputView {
                        first: 0,
                        stride: D,
                        words: BLOCK_WORDS,
                    },
                    batch_io::RouterView {
                        first: 0,
                        stride: ROUTER,
                        words: MAX_BATCH * ROUTER,
                    },
                    *count,
                    cancel,
                    layer,
                    *start as u64,
                    true,
                    true,
                )?;
                let expert_start = Instant::now();
                pending.moe[layer] = Some(e.enqueue_resident_router_moe(
                    m.graph,
                    &m.arena,
                    &mut self.moe,
                    &session,
                    layer,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.packet,
                    &spans,
                    cancel,
                )?);
                *m.phase_ns
                    .entry("native_route_and_expert_enqueue")
                    .or_default() += expert_start.elapsed().as_nanos();
            } else {
                let router_fence_start = Instant::now();
                let hybrid_packet = if self.moe.cpu_miss_cap().is_some() {
                    Some(batch_io::read_count1_router_input(
                        m.graph,
                        &m.arena,
                        &m.admission,
                        prepared.mixed(),
                        self.outer.router,
                        self.outer.packet,
                        m.b.fault,
                        batch_io::InputView {
                            first: 0,
                            stride: D,
                            words: BLOCK_WORDS,
                        },
                        batch_io::RouterView {
                            first: 0,
                            stride: ROUTER,
                            words: MAX_BATCH * ROUTER,
                        },
                        *count,
                        cancel,
                        layer,
                        *start as u64,
                        e.gpu_router,
                        e.shared_kernels,
                    )?)
                } else {
                    None
                };
                let ordinary_logits = if hybrid_packet.is_none() {
                    Some(batch_io::read_router(
                        m.graph,
                        &m.arena,
                        &m.admission,
                        self.outer.router,
                        self.outer.packet,
                        m.b.fault,
                        batch_io::RouterView {
                            first: 0,
                            stride: ROUTER,
                            words: MAX_BATCH * ROUTER,
                        },
                        *count,
                        cancel,
                    )?)
                } else {
                    None
                };
                let logits = match (&hybrid_packet, &ordinary_logits) {
                    (Some(packet), None) => packet.logits(),
                    (None, Some(logits)) => logits,
                    _ => return Err(error()),
                };
                *m.phase_ns.entry("native_router_fence").or_default() +=
                    router_fence_start.elapsed().as_nanos();
                let use_gpu_router = e.gpu_router && hybrid_packet.is_some();
                if use_gpu_router {
                    if hybrid_packet
                        .as_ref()
                        .and_then(|packet| packet.gpu_selection())
                        .is_none()
                    {
                        return Err(Error::InvalidFormat(concat!(
                            "native model chunk rejected at source line ",
                            line!()
                        )));
                    }
                    e.gpu_router_calls = e
                        .gpu_router_calls
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?;
                }
                let expert_start = Instant::now();
                let mut routes = token_waves::Input::new(*count, self.moe.capacity(), ROUTER)
                    .map_err(|_| error())?;
                for token in 0..*count {
                    let selection = if use_gpu_router {
                        None
                    } else {
                        Some(
                            ops::router_topk_guarded(logits.row(token)?, 10, true)
                                .map_err(u::fault)?,
                        )
                    };
                    let selected = match (&selection, &hybrid_packet) {
                        (Some(selection), _) => selection.as_slice(),
                        (None, Some(packet)) => packet.gpu_selection().ok_or(error())?.as_slice(),
                        _ => return Err(error()),
                    };
                    for (rank, &(expert, probability)) in selected.iter().enumerate() {
                        routes
                            .set(token, rank, expert, probability)
                            .map_err(|_| error())?;
                    }
                }
                let waves = token_waves::Plan::build(&routes, cancel).map_err(|_| error())?;
                let overlap_shared =
                    hybrid_packet.is_some() && self.moe.cpu_gpu_overlap() == Some(true);
                if overlap_shared {
                    let mut projector = ResidentBatchProjection {
                        executor: e,
                        fault: m.b.fault,
                        dense_q8: None,
                    };
                    self.shared.prepare(
                        m.graph,
                        layer,
                        &m.arena,
                        &mut m.public,
                        &mut projector,
                        prepared.mixed(),
                        self.outer.block[0],
                        self.outer.block[1],
                        m.b.fault,
                        *count,
                        cancel,
                    )?;
                }
                pending.moe[layer] = Some(if let Some(packet) = &hybrid_packet {
                    e.enqueue_routed_moe_count1_hybrid(
                        m.graph,
                        &m.arena,
                        &mut self.moe,
                        &session,
                        layer,
                        prepared.mixed(),
                        self.outer.block[0],
                        &spans,
                        &waves,
                        packet,
                        cancel,
                    )?
                } else if self.moe.frozen_host_sources_enabled() {
                    if !self.public_frozen_host_sources {
                        return Err(Error::InvalidFormat(concat!(
                            "native model chunk rejected at source line ",
                            line!()
                        )));
                    }
                    e.enqueue_routed_moe_frozen_host_batch(
                        m.graph,
                        &m.arena,
                        &mut self.moe,
                        &session,
                        layer,
                        prepared.mixed(),
                        self.outer.block[0],
                        &spans,
                        &waves,
                        cancel,
                    )?
                } else {
                    if e.target_frozen_host_source_identity.is_some() {
                        return Err(Error::InvalidFormat(concat!(
                            "native model chunk rejected at source line ",
                            line!()
                        )));
                    }
                    e.enqueue_routed_moe_batch(
                        m.graph,
                        &m.arena,
                        &mut self.moe,
                        &session,
                        layer,
                        prepared.mixed(),
                        self.outer.block[0],
                        &spans,
                        &waves,
                        cancel,
                    )?
                });
                e.check_original_bank_metadata(LAYERS, ORIGINAL_BANK_HOST)?;
                *m.phase_ns
                    .entry("native_route_and_expert_enqueue")
                    .or_default() += expert_start.elapsed().as_nanos();
            }
            let overlap_shared = prepared_shared_overlap(
                self.moe.resident_router_active(),
                self.moe.cpu_miss_cap(),
                self.moe.cpu_gpu_overlap(),
            );
            let shared_start = Instant::now();
            let mut projector = ResidentBatchProjection {
                executor: e,
                fault: m.b.fault,
                dense_q8,
            };
            if overlap_shared {
                self.shared.finish_prepared(
                    m.graph,
                    layer,
                    &m.arena,
                    &mut m.public,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.block[1],
                    m.b.fault,
                    *count,
                    cancel,
                )?;
                e.shared_overlap_calls = e
                    .shared_overlap_calls
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow)?;
            } else {
                self.shared.enqueue(
                    m.graph,
                    layer,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.block[1],
                    m.b.fault,
                    *count,
                    cancel,
                )?;
            }
            prepared.combine(self.outer.block[1], self.outer.residual[1 - *current])?;
            *current = 1 - *current;
            *m.phase_ns
                .entry("native_shared_and_ffn_combine_enqueue")
                .or_default() += shared_start.elapsed().as_nanos();
            // Fixed-public lab hook: actual routed/shared consumers and HC
            // combine are already queued; no family has published its state.
            m.lifecycle_hooks.late_combine(layer)?;
            Ok(())
        })?;
        let final_head_start = Instant::now();
        if purpose == Purpose::LastLogits {
            let mut projector = ResidentBatchProjection {
                executor: e,
                fault: m.b.fault,
                dense_q8: None,
            };
            self.gr
                .read(
                    m.graph,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    gr::Kind::Final,
                    self.outer.residual[*current],
                    *count,
                    m.b.fault,
                    cancel,
                )?
                .enqueue_final_head(m.graph, &mut projector, m.b.logits)?;
        } else if purpose == Purpose::AllRowsWithCarry {
            let rows = self.target_rows.as_ref().ok_or(error())?;
            let mut projector = ResidentBatchProjection {
                executor: e,
                fault: m.b.fault,
                dense_q8: None,
            };
            self.gr
                .read(
                    m.graph,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    if *count >= 4 {
                        gr::Kind::FinalFourRows
                    } else {
                        gr::Kind::FinalAllRows
                    },
                    self.outer.residual[*current],
                    *count,
                    m.b.fault,
                    cancel,
                )?
                .enqueue_final_head_with_target_capacity(
                    m.graph,
                    &mut projector,
                    rows.logits(),
                    rows.four_rows_reserved(),
                )?;
        }
        *m.phase_ns.entry("native_final_head_enqueue").or_default() +=
            final_head_start.elapsed().as_nanos();
        self.gr.flush_pending(&m.arena, m.b.fault, cancel)?;
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        // Capture real queued counters before PLE/family permits borrow their
        // disjoint workspaces. Completion and publication remain below.
        let mut grouped_status_expected = [1u64, 0, 0]; // FinalAllRows GR.
        for layer in m.layers.iter() {
            grouped_status_expected[0] += 2; // Attention and FFN GR.
            match layer {
                Layer::Gdn(_, _) => grouped_status_expected[1] += 1,
                Layer::Qsa(_, _) => grouped_status_expected[2] += 1,
            }
        }
        if crate::strata_intel_core::enabled() {
            // Native GR writers join finite faults directly; no status-fold
            // dispatch is claimed for the retired pointwise stages.
            grouped_status_expected[0] = 0;
        }
        let (still_selected, grouped_status_after) = self.grouped_status_state()?;
        if grouped_status_folds != still_selected {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        let final_proof_start = Instant::now();
        let (output, ple) = completed_proof(
            || {
                Ok(if purpose == Purpose::LastLogits {
                    ChunkOutput::Legacy(batch_io::read_last_logits(
                        m.graph,
                        &m.arena,
                        &m.admission,
                        m.b.logits,
                        m.b.logit_packet,
                        m.b.fault,
                        cancel,
                    )?)
                } else if purpose == Purpose::AllRowsWithCarry {
                    ChunkOutput::TargetRows(self.target_rows.as_ref().ok_or(error())?.read(
                        m.graph,
                        &m.arena,
                        &m.admission,
                        self.outer.residual[*current],
                        m.b.fault,
                        *count,
                        *start,
                        cancel,
                    )?)
                } else {
                    batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)?;
                    ChunkOutput::Legacy(Zeroizing::new(Vec::new()))
                })
            },
            || {
                self.ple.prepare_split_after_whole_model_fence(
                    ple_batch::Preparation {
                        graph: m.graph,
                        arena: &m.arena,
                        admission: &m.admission,
                        retired: &m.retired,
                        context: m.context,
                        position: *start,
                        owner_poisoned: &mut m.poisoned,
                        sequence_poison: &m.ple_batch_poison,
                        conv: &mut m.ple_conv,
                        history: &mut m.ple_history,
                        anchor: &mut m.ple_batch_pending,
                        fault: m.b.fault,
                        scalar_ids: [m.b.ple_emb, m.b.ple_key, m.b.ple_value],
                    },
                    pending.ple.take().ok_or(error())?,
                    cancel,
                )
            },
            || {
                // The real PLE preparation exists and the first whole-model
                // proof completed. Failure here must drop that real ticket.
                m.lifecycle_hooks.before_second_proof()?;
                batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)
            },
        )?;
        *m.phase_ns
            .entry("native_completed_final_proofs")
            .or_default() += final_proof_start.elapsed().as_nanos();
        let batch_router_audit_start = Instant::now();
        self.moe.resolve_batch_router_chunk(
            e,
            m.graph,
            &m.arena,
            &session,
            &mut pending.moe,
            cancel,
        )?;
        if self.moe.batch_router_active() {
            *m.phase_ns.entry("native_device_router_audit").or_default() +=
                batch_router_audit_start.elapsed().as_nanos();
        }
        let resident_router_audit_start = Instant::now();
        self.moe.resolve_resident_router_token(
            e,
            m.graph,
            &m.arena,
            &session,
            &mut pending.moe,
            cancel,
        )?;
        if self.moe.resident_router_active() {
            *m.phase_ns
                .entry("native_resident_router_audit")
                .or_default() += resident_router_audit_start.elapsed().as_nanos();
        }
        let family_start = Instant::now();
        let mut ga = {
            let (first_stage, first_receipt) = match (&m.layers[0], pending.gdn[0].as_ref()) {
                (Layer::Gdn(s, _), Some(r)) => (s.as_ref(), r),
                _ => return Err(error()),
            };
            let mut refs = [(first_stage, first_receipt); 36];
            let mut at = 0;
            for layer in 0..LAYERS {
                if let Layer::Gdn(s, _) = &m.layers[layer] {
                    refs[at] = (s, pending.gdn[layer].as_ref().ok_or(error())?);
                    at += 1;
                }
            }
            if at != refs.len() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            self.gdn
                .accept_after_whole_model_chunk_acceptance(&m.arena, &refs, cancel)?
        };
        let mut qa = {
            let (first_stage, first_receipt) = match (&m.layers[3], pending.qsa[3].as_ref()) {
                (Layer::Qsa(s, _), Some(r)) => (s.as_ref(), r),
                _ => return Err(error()),
            };
            let mut refs = [(first_stage, first_receipt); 12];
            let mut at = 0;
            for layer in 0..LAYERS {
                if let Layer::Qsa(s, _) = &m.layers[layer] {
                    refs[at] = (s, pending.qsa[layer].as_ref().ok_or(error())?);
                    at += 1;
                }
            }
            if at != refs.len() {
                return Err(Error::InvalidFormat(concat!(
                    "native model chunk rejected at source line ",
                    line!()
                )));
            }
            self.qsa
                .accept_after_whole_model_chunk_acceptance(&m.arena, &refs, cancel)?
        };
        let four_dispatch_receipt = (
            e.target_four_rows_active,
            e.target_verify4_dense_calls,
            e.target_verify4_fused_calls,
        );
        let mut ma = {
            let first = pending.moe[0].as_ref().ok_or(error())?;
            let mut refs = [first; LAYERS];
            for (i, r) in refs.iter_mut().enumerate() {
                *r = pending.moe[i].as_ref().ok_or(error())?;
            }
            self.moe.accept_after_whole_model_chunk_acceptance(
                e,
                &mut session,
                &m.arena,
                &refs,
                cancel,
            )?
        };
        ple.preflight(cancel)?;
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        if self.poison.load(Ordering::Acquire)
            || m.retired.load(Ordering::Acquire)
            || m.position != *start
        {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        // Prepare the checked counter before ANY family publication. The
        // guarded proposal is discarded on all late failure/cancellation paths.
        let completed_cpu_misses =
            checked_cpu_miss_total(*m.completed_cpu_misses, ma.completed_cpu_misses())?;
        let three_gdn_completed_chunks =
            next_three_gdn_chunks(decode_policy, self.three_gdn_completed_chunks)?;
        let four_rows_completed_chunks = next_four_rows_chunks(
            decode_policy,
            *count,
            four_dispatch_receipt,
            self.four_rows_completed_chunks,
        )?;
        let grouped_ple_completed_chunks = next_grouped_ple_chunks(
            grouped_ple_reads,
            ple.public_grouped_read_completed(),
            self.grouped_ple_completed_chunks,
        )?;
        let grouped_status_completed = self.grouped_status_completed.propose(
            grouped_status_folds,
            grouped_status_before,
            grouped_status_after,
            grouped_status_expected,
        )?;
        *m.phase_ns.entry("native_all_family_preflight").or_default() +=
            family_start.elapsed().as_nanos();
        let commit_start = Instant::now();
        // Every actual-family permit has now preflighted. Remaining cancellation
        // or fallible commit failure terminally destroys the whole model; it
        // cannot publish output/cursor or resume a partially changed sequence.
        for (layer, owner) in m.layers.iter_mut().enumerate() {
            match owner {
                Layer::Gdn(stage, _) => {
                    stage.commit_after_whole_chunk_acceptance(
                        pending.gdn[layer].take().ok_or(error())?,
                        &m.arena,
                        &mut ga,
                        cancel,
                    )?;
                    // Inject only AFTER the first genuine family commit. The
                    // next real commit must reject and destroy this sequence.
                    m.lifecycle_hooks.after_gdn_commit(layer, cancel);
                }
                Layer::Qsa(stage, _) => stage.commit_after_whole_chunk_acceptance(
                    pending.qsa[layer].take().ok_or(error())?,
                    &m.arena,
                    &mut qa,
                    cancel,
                )?,
            }
        }
        for receipt in pending.moe.iter_mut() {
            ma.commit(receipt.take().ok_or(error())?, &m.arena, cancel)?;
        }
        ga.finish(&m.arena, cancel)?;
        qa.finish(&m.arena, cancel)?;
        ma.finish(&m.arena, cancel)?;
        self.moe.finish_resident_router_token()?;
        ple.commit(cancel)?;
        if *count == 1 {
            e.promote_adaptive(&CompletedAdaptiveBoundary(()), &m.arena, m.b.fault, cancel)?;
        }
        *m.phase_ns.entry("native_all_family_commit").or_default() +=
            commit_start.elapsed().as_nanos();
        e.check_original_bank_metadata(LAYERS, ORIGINAL_BANK_HOST)?;
        u::cancel(cancel)?;
        if self.poison.load(Ordering::Acquire) || m.retired.load(Ordering::Acquire) || m.poisoned {
            return Err(Error::InvalidFormat(concat!(
                "native model chunk rejected at source line ",
                line!()
            )));
        }
        *m.completed_cpu_misses = *completed_cpu_misses;
        self.three_gdn_completed_chunks = three_gdn_completed_chunks;
        self.four_rows_completed_chunks = four_rows_completed_chunks;
        self.grouped_ple_completed_chunks = grouped_ple_completed_chunks;
        self.grouped_status_completed = grouped_status_completed;
        m.position = *end;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn scoped_verified_rows_release_original_lease_before_prefix_consumers() {
        use super::{MAX_BATCH, error, select_verified_rows};
        use std::cell::Cell;
        struct Rows<'a>(&'a Cell<bool>);
        impl Drop for Rows<'_> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        for count in 1..=MAX_BATCH {
            for keep in 0..=count + 1 {
                let dropped = Cell::new(false);
                let result = select_verified_rows(Rows(&dropped), count, |_| {
                    assert!(!dropped.get());
                    Ok(keep)
                });
                assert!(dropped.get());
                assert_eq!(result.ok(), (1..=count).contains(&keep).then_some(keep));
            }
        }
        let dropped = Cell::new(false);
        assert!(select_verified_rows(Rows(&dropped), 3, |_| Err(error())).is_err());
        assert!(dropped.get());
    }
    #[test]
    fn q6_verifier_activation_keeps_original_prefill_and_request_boundary() {
        use super::{NativeDecodeKernels as K, q6_verify_head_selection_allowed};
        for policy in [
            K::Chunk,
            K::ThreeGdn,
            K::FourGdn,
            K::SingleAll,
            K::SingleGdn,
            K::SingleMoe,
            K::SingleQ4All,
        ] {
            for accepted in [false, true] {
                for active in [false, true] {
                    assert_eq!(
                        q6_verify_head_selection_allowed(policy, accepted, active),
                        accepted
                            && !active
                            && matches!(policy, K::Chunk | K::ThreeGdn | K::FourGdn)
                    );
                }
            }
        }
    }
    #[test]
    fn four_rows_policy_and_actual_completion_preserve_original_prefill_and_repair_tails() {
        use NativeDecodeKernels::{Chunk, FourGdn, ThreeGdn};
        assert!(!four_rows_selection_allowed(Chunk, false, false));
        assert!(!four_rows_selection_allowed(Chunk, true, true));
        assert!(!four_rows_selection_allowed(ThreeGdn, true, false));
        assert!(four_rows_selection_allowed(Chunk, true, false));
        assert!(!three_gdn_selection_allowed(FourGdn, true, false));
        assert_eq!(
            next_four_rows_chunks(FourGdn, 4, (true, 5, 5), 2).unwrap(),
            3
        );
        for count in 1..=3 {
            assert_eq!(
                next_four_rows_chunks(FourGdn, count, (true, 5, 5), 3).unwrap(),
                3
            );
            assert!(grouped_ple_run_allowed_selected(
                true,
                count,
                Purpose::AllRowsWithCarry,
                true
            ));
        }
        assert_eq!(
            next_four_rows_chunks(Chunk, 3, (false, 0, 0), 0).unwrap(),
            0
        );
        for actual in [(false, 5, 5), (true, 0, 0), (true, 5, 4)] {
            assert!(next_four_rows_chunks(FourGdn, 4, actual, 0).is_err());
        }
        assert!(next_four_rows_chunks(FourGdn, 4, (true, 5, 5), u64::MAX).is_err());
        assert!(!grouped_ple_run_allowed_selected(
            true,
            4,
            Purpose::AllRowsWithCarry,
            false
        ));
        assert!(grouped_ple_run_allowed_selected(
            true,
            4,
            Purpose::AllRowsWithCarry,
            true
        ));
        assert!(!grouped_ple_run_allowed_selected(
            true,
            5,
            Purpose::AllRowsWithCarry,
            true
        ));
    }
    #[test]
    fn grouped_status_actual_workspace_states_reject_partial_selection() {
        for bits in 0..8 {
            let states = std::array::from_fn(|i| (bits & (1 << i) != 0, i as u64));
            let actual = grouped_status_states(states);
            if bits == 0 || bits == 7 {
                assert_eq!(actual.unwrap(), (bits == 7, [0, 1, 2]));
            } else {
                assert!(actual.is_err());
            }
        }
    }
    #[test]
    fn grouped_status_completion_joins_actual_folds_before_any_commit() {
        use crate::status_fold_batch::Completed;
        let zero = Completed::default();
        assert_eq!(zero.receipt(false, [0; 3]).unwrap(), (false, zero));
        let proposal = zero
            .propose(true, [0; 3], [97, 36, 12], [97, 36, 12])
            .unwrap();
        assert_eq!(zero, Completed::default()); // Proposal has not published.
        assert_eq!(
            proposal.receipt(true, [97, 36, 12]).unwrap(),
            (true, proposal)
        );
        for role in 0..3 {
            for delta in [-1i64, 1] {
                let mut actual = [97, 36, 12];
                actual[role] = (actual[role] as i64 + delta) as u64;
                assert!(zero.propose(true, [0; 3], actual, [97, 36, 12]).is_err());
            }
        }
        assert!(
            zero.propose(false, [0; 3], [97, 36, 12], [97, 36, 12])
                .is_err()
        );
        assert!(
            zero.propose(true, [1, 0, 0], [98, 36, 12], [97, 36, 12])
                .is_err()
        );
        assert!(
            zero.propose(true, [0; 3], [97, 36, 12], [97, 35, 13])
                .is_err()
        );
        assert!(zero.receipt(true, [0; 3]).is_err());
        assert!(proposal.receipt(false, [97, 36, 12]).is_err());
        assert!(proposal.receipt(true, [97, 36, 11]).is_err());
    }
    #[test]
    fn grouped_status_scope_stays_full_families_all_rows_and_immutable_before_io() {
        for count in 0..=9 {
            for purpose in [
                Purpose::Advance,
                Purpose::LastLogits,
                Purpose::AllRowsWithCarry,
            ] {
                assert_eq!(
                    grouped_ple_run_allowed(true, count, purpose),
                    (1..=3).contains(&count) && purpose == Purpose::AllRowsWithCarry
                );
            }
        }
        assert!(grouped_ple_selection_allowed(
            true,
            false,
            true,
            true,
            false,
            NativeDecodeKernels::Chunk
        ));
        assert!(grouped_ple_selection_allowed(
            true,
            true,
            true,
            true,
            false,
            NativeDecodeKernels::ThreeGdn
        ));
        for (reserved, accepted, active) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            assert!(!grouped_ple_selection_allowed(
                reserved,
                false,
                true,
                accepted,
                active,
                NativeDecodeKernels::Chunk
            ));
        }
        assert!(!grouped_ple_selection_allowed(
            true,
            true,
            false,
            true,
            false,
            NativeDecodeKernels::Chunk
        ));
    }
    #[test]
    fn grouped_ple_selection_requires_reserved_accepted_public_prefill_and_is_immutable() {
        use NativeDecodeKernels::{Chunk, SingleAll, SingleGdn, SingleMoe, SingleQ4All, ThreeGdn};
        for policy in [
            Chunk,
            ThreeGdn,
            SingleMoe,
            SingleGdn,
            SingleAll,
            SingleQ4All,
        ] {
            for reserved in [false, true] {
                for selected in [false, true] {
                    for requested in [false, true] {
                        for accepted_prefill in [false, true] {
                            for request_active in [false, true] {
                                assert_eq!(
                                    grouped_ple_selection_allowed(
                                        reserved,
                                        selected,
                                        requested,
                                        accepted_prefill,
                                        request_active,
                                        policy,
                                    ),
                                    reserved
                                        && accepted_prefill
                                        && !request_active
                                        && (!selected || requested)
                                        && matches!(policy, Chunk | ThreeGdn),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn grouped_ple_actual_target_run_stays_all_rows_one_to_three_without_count1_mutation() {
        for count in [0, 1, 2, 3, 4, 5, 8, 9, usize::MAX] {
            for purpose in [
                Purpose::Advance,
                Purpose::LastLogits,
                Purpose::AllRowsWithCarry,
            ] {
                assert_eq!(
                    grouped_ple_run_allowed(true, count, purpose),
                    (1..=3).contains(&count) && purpose == Purpose::AllRowsWithCarry,
                );
                assert!(grouped_ple_run_allowed(false, count, purpose));
            }
        }
    }
    #[test]
    fn grouped_ple_completion_joins_actual_dispatch_before_publication_and_never_false_credits() {
        assert_eq!(next_grouped_ple_chunks(false, false, 0).unwrap(), 0);
        assert_eq!(next_grouped_ple_chunks(true, true, 73).unwrap(), 74);
        assert!(next_grouped_ple_chunks(false, true, 0).is_err());
        assert!(next_grouped_ple_chunks(true, false, 0).is_err());
        assert!(next_grouped_ple_chunks(true, true, u64::MAX).is_err());
        let current = 73;
        let proposed = next_grouped_ple_chunks(true, true, current).unwrap();
        assert_eq!(proposed, 74);
        assert_eq!(current, 73); // Real commit alone publishes the proposal.
        assert_eq!(grouped_ple_completed_receipt(false, 0).unwrap(), (false, 0));
        assert_eq!(grouped_ple_completed_receipt(true, 74).unwrap(), (true, 74));
        assert!(grouped_ple_completed_receipt(true, 0).is_err());
        assert!(grouped_ple_completed_receipt(false, 74).is_err());
    }
    #[test]
    fn three_gdn_actual_policy_is_exact_and_cannot_enter_count1_geometry() {
        use crate::gdn_chunk::RecurrencePolicy::{Chunk, SingleStreaming, ThreeStreaming};
        assert_eq!(
            decode_family_policy(8, ThreeStreaming, false, 8).unwrap(),
            NativeDecodeKernels::ThreeGdn
        );
        assert_eq!(
            decode_family_policy(8, Chunk, false, 8).unwrap(),
            NativeDecodeKernels::Chunk
        );
        assert_eq!(
            decode_family_policy(8, SingleStreaming, false, 8).unwrap(),
            NativeDecodeKernels::SingleGdn
        );
        for tile in [1, 4, 8] {
            for dense in [1, 8] {
                for q4 in [false, true] {
                    assert_eq!(
                        decode_family_policy(tile, ThreeStreaming, q4, dense).is_ok(),
                        tile == 8 && dense == 8 && !q4
                    );
                }
            }
        }
        let policy = NativeDecodeKernels::ThreeGdn;
        assert_eq!(policy.recurrence(), ThreeStreaming);
        assert!(!policy.single_moe());
        for count in [0, 1, 2, 3, 4, 5, 8, 9, usize::MAX] {
            assert_eq!(
                policy.validate_count(count).is_ok(),
                (1..=3).contains(&count)
            );
        }
        for tile in [0, 1, 4, 8, usize::MAX] {
            assert_eq!(projection_policy_allows_tile(policy, tile), tile == 8);
        }
        // The normal count1 CLI cannot select this state.
        assert!(NativeDecodeKernels::parse(policy.label()).is_err());
        assert_eq!(NativeDecodeKernels::default(), NativeDecodeKernels::Chunk);
    }
    #[test]
    fn three_gdn_public_scope_rejects_each_missing_actual_owner_or_dispatch_gate() {
        let scope = PublicThreeGdnScope {
            original_profile: true,
            target_rows: true,
            checkpoint: true,
            public_request: true,
            frozen_sources: true,
            frozen_router: true,
            partition_exit: true,
            original_io: true,
            original_dispatch: true,
            original_families: true,
        };
        scope.validate().unwrap();
        for missing in 0..10 {
            let mut invalid = scope;
            match missing {
                0 => invalid.original_profile = false,
                1 => invalid.target_rows = false,
                2 => invalid.checkpoint = false,
                3 => invalid.public_request = false,
                4 => invalid.frozen_sources = false,
                5 => invalid.frozen_router = false,
                6 => invalid.partition_exit = false,
                7 => invalid.original_io = false,
                8 => invalid.original_dispatch = false,
                _ => invalid.original_families = false,
            }
            assert!(invalid.validate().is_err());
        }
    }
    #[test]
    fn three_gdn_selection_needs_accepted_prefill_before_request_and_no_count1_state() {
        for policy in [
            NativeDecodeKernels::Chunk,
            NativeDecodeKernels::ThreeGdn,
            NativeDecodeKernels::SingleMoe,
            NativeDecodeKernels::SingleGdn,
            NativeDecodeKernels::SingleAll,
            NativeDecodeKernels::SingleQ4All,
        ] {
            for prefill in [false, true] {
                for active in [false, true] {
                    assert_eq!(
                        three_gdn_selection_allowed(policy, prefill, active),
                        prefill
                            && !active
                            && matches!(
                                policy,
                                NativeDecodeKernels::Chunk | NativeDecodeKernels::ThreeGdn
                            )
                    );
                }
            }
        }
    }
    #[test]
    fn three_gdn_completion_counter_is_checked_before_family_publication() {
        assert_eq!(
            next_three_gdn_chunks(NativeDecodeKernels::ThreeGdn, 0).unwrap(),
            1
        );
        assert_eq!(
            next_three_gdn_chunks(NativeDecodeKernels::ThreeGdn, 9).unwrap(),
            10
        );
        assert!(next_three_gdn_chunks(NativeDecodeKernels::ThreeGdn, u64::MAX).is_err());
        assert_eq!(
            next_three_gdn_chunks(NativeDecodeKernels::Chunk, u64::MAX).unwrap(),
            u64::MAX
        );
        let current = 9;
        let candidate = next_three_gdn_chunks(NativeDecodeKernels::ThreeGdn, current).unwrap();
        assert_eq!(candidate, 10);
        // No mutable current is lent to preflight; late failure can abandon it.
        assert_eq!(current, 9);
    }
    #[test]
    fn frozen_down_integer_exact_owner_and_independent_gu_down_budget_increments() {
        use crate::quant_matvec::frozen_down_integer as i;
        let full = BufferLayout::new(i::SCRATCH_WORDS).unwrap();
        assert_eq!(full.logical_words, 16384);
        assert_eq!(full.owned_bytes, 65536);
        assert_eq!(
            full.owned_bytes as u64,
            batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES
        );
        assert_eq!(batch_moe::FROZEN_DOWN_INTEGER_HOST_BYTES, 65536);
        assert_eq!(batch_moe::FROZEN_DOWN_INTEGER_INFO_BYTES, 196608);
        assert_eq!(batch_moe::FROZEN_DOWN_INTEGER_RUNTIME_BYTES, 462848);
        assert_eq!(
            BufferLayout::new(i::LOGICAL_WORDS).unwrap().owned_bytes,
            65536
        );
        for gu in [false, true] {
            for down in [false, true] {
                let increments = u64::from(gu) * batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES
                    + u64::from(down) * batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES;
                assert_eq!(increments, u64::from(gu as u8 + down as u8) * 65536);
                let runtime = runtime_bytes(38).unwrap()
                    + u64::from(gu) * batch_moe::FROZEN_GU_INTEGER_RUNTIME_BYTES
                    + u64::from(down) * batch_moe::FROZEN_DOWN_INTEGER_RUNTIME_BYTES;
                assert!(runtime < TARGET_ROWS_RUNTIME_LIMIT);
            }
        }
        for count in 0..=8 {
            for purpose in [
                Purpose::Advance,
                Purpose::LastLogits,
                Purpose::AllRowsWithCarry,
            ] {
                assert_eq!(
                    grouped_ple_run_allowed(true, count, purpose),
                    (1..=3).contains(&count) && purpose == Purpose::AllRowsWithCarry
                );
            }
        }
    }
    #[test]
    fn frozen_gu_integer_full_page_and_independent_transient_charges_are_exact() {
        for sources in [false, true] {
            for router in [false, true] {
                for staging in [false, true] {
                    assert!(
                        frozen_gu_integer_configuration(false, sources, router, staging).is_ok()
                    );
                    assert_eq!(
                        frozen_gu_integer_configuration(true, sources, router, staging).is_ok(),
                        sources && router && !staging
                    );
                }
            }
        }
        let bytes =
            crate::private_arena::BufferLayout::new(batch_moe::FROZEN_GU_INTEGER_SCRATCH_WORDS)
                .unwrap();
        assert_eq!(bytes.logical_words, 16384);
        assert_eq!(
            bytes.owned_bytes as u64,
            batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES
        );
        assert_eq!(bytes.owned_bytes, 65536);
        assert_eq!(batch_moe::FROZEN_GU_INTEGER_INFO_BYTES, 196608);
        assert_eq!(batch_moe::FROZEN_GU_INTEGER_RUNTIME_BYTES, 462848);
        assert_eq!(batch_moe::FROZEN_GU_INTEGER_HOST_BYTES, 65536);
        assert_eq!(
            TARGET_ROWS_PRIVATE_MAIN
                + batch_moe::frozen_sources::ROUTER_MAIN_BYTES
                + batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES,
            828932096
        );
    }
    #[test]
    fn frozen_staging_budget_requires_both_original_sources_and_gpu_router() {
        for sources in [false, true] {
            for router in [false, true] {
                assert!(frozen_staging_configuration(false, sources, router).is_ok());
                assert_eq!(
                    frozen_staging_configuration(true, sources, router).is_ok(),
                    sources && router
                );
            }
        }
        assert_eq!(batch_moe::FROZEN_STAGING_OWNER_BYTES, 33_554_432);
        assert_eq!(batch_moe::FROZEN_STAGING_MAIN_BYTES, 100_663_296);
        assert_eq!(batch_moe::FROZEN_STAGING_RUNTIME_BYTES, 589_824);
        for bytes in [batch_moe::FROZEN_STAGING_OWNER_BYTES; 3] {
            assert_eq!(
                crate::private_arena::BufferLayout::new((bytes / 4) as usize)
                    .unwrap()
                    .owned_bytes as u64,
                bytes
            );
        }
    }

    use super::*;
    #[test]
    fn target_rows_charge_two_full_owners_and_independent_larger_io_and_host_caps() {
        assert_eq!(TARGET_ROWS_PRIVATE_MAIN, 826_638_336);
        assert_eq!(
            target_rows::Owners::allocation_requests().unwrap(),
            [4_194_304; 2]
        );
        assert_eq!(1286 + 90 + 2, 1378);
        for (tiles, expected) in [(16, 70_295_552), (38, 70_385_664)] {
            let selected = runtime_bytes(tiles).unwrap() + target_rows::RUNTIME_INCREMENT;
            assert_eq!(selected, expected);
            assert!(selected > crate::executor::RESIDENT_RUNTIME_IO_MAX);
            assert!(selected < TARGET_ROWS_RUNTIME_LIMIT);
        }
        for reuse in [DEFAULT_REUSE_METADATA, 1048576, 4194304] {
            let (reserved, limit) =
                host_reservation_with_outputs(false, reuse, false, true).unwrap();
            assert_eq!(reserved, HOST_COMPONENTS + reuse + 6 * 1024 * 1024);
            assert_eq!(limit, 18 * 1024 * 1024 + reuse - DEFAULT_REUSE_METADATA);
            assert_eq!(limit - reserved, 1_114_112);
        }
        assert_eq!(
            host_reservation(false, DEFAULT_REUSE_METADATA).unwrap(),
            (11_468_800, 12_582_912)
        );
        assert!(host_reservation_with_outputs(true, DEFAULT_REUSE_METADATA, false, true).is_err());
        assert!(host_reservation_with_outputs(false, DEFAULT_REUSE_METADATA, true, true).is_err());
    }
    #[test]
    fn constructor_overlap_owners_cannot_finish_an_unprepared_shared_projection() {
        for cap in [None, Some(0), Some(10)] {
            for overlap in [None, Some(false), Some(true)] {
                for resident in [false, true] {
                    let packet_present = cap.is_some();
                    let original = packet_present && overlap == Some(true);
                    assert_eq!(
                        prepared_shared_overlap(resident, cap, overlap),
                        !resident && original
                    );
                }
            }
        }
        assert!(!prepared_shared_overlap(false, None, Some(true)));
        assert!(prepared_shared_overlap(false, Some(10), Some(true)));
        assert!(!prepared_shared_overlap(true, Some(10), Some(true)));
    }
    #[test]
    fn resident_router_adds_full_separate_host_runtime_and_main_charges() {
        for reuse in [DEFAULT_REUSE_METADATA, 1048576, 4194304] {
            let old = host_reservation(true, reuse).unwrap();
            let new = host_reservation_with_router(true, reuse, true).unwrap();
            assert_eq!(
                new,
                (
                    old.0 + batch_moe::RESIDENT_ROUTER_HOST,
                    old.1 + batch_moe::RESIDENT_ROUTER_HOST
                )
            );
        }
        assert!(host_reservation_with_router(false, DEFAULT_REUSE_METADATA, true).is_err());
        for head in [16, 38] {
            let old = runtime_bytes(head).unwrap() + HYBRID_RUNTIME;
            assert!(old <= crate::executor::RESIDENT_RUNTIME_IO_MAX);
            for prepared in [false, true] {
                let selected = batch_moe::resident_router_runtime(prepared);
                assert_eq!(
                    selected,
                    9_281_536
                        + if prepared {
                            batch_moe::RESIDENT_ROUTER_INTEGER_CLEAR_INFO
                        } else {
                            0
                        }
                );
                let new = old + selected;
                assert!(new > crate::executor::RESIDENT_RUNTIME_IO_MAX);
                assert!(new < batch_moe::RESIDENT_ROUTER_RUNTIME_LIMIT);
            }
        }
    }

    #[test]
    fn reuse_metadata_has_a_separate_complete_and_explicit_host_charge() {
        for bytes in [DEFAULT_REUSE_METADATA, 1048576, 4194304] {
            let (reserved, limit) = host_reservation(true, bytes).unwrap();
            assert_eq!(reserved, HOST_COMPONENTS + HYBRID_HOST + bytes);
            assert_eq!(limit, HOST_CAP + bytes - DEFAULT_REUSE_METADATA);
            assert_eq!(limit - reserved, 65536);
        }
        assert!(host_reservation(false, DEFAULT_REUSE_METADATA).is_ok());
        for bytes in [0, 262145, 1048577, 4194305, usize::MAX] {
            assert!(host_reservation(true, bytes).is_err());
        }
        assert!(host_reservation(false, 1048576).is_err());
        assert!(host_reservation(false, 4194304).is_err());
    }
    #[test]
    fn completed_counter_preflight_rejects_overflow_and_impossible_chunk_without_publication() {
        assert_eq!(*checked_cpu_miss_total(7, 480).unwrap(), 487);
        assert_eq!(*checked_cpu_miss_total(u64::MAX, 0).unwrap(), u64::MAX);
        assert!(checked_cpu_miss_total(u64::MAX, 1).is_err());
        assert!(checked_cpu_miss_total(0, 481).is_err());
        // The current value is never borrowed mutably during preflight: a
        // prepared proposal can be abandoned after any family failure.
        let current = Zeroizing::new(7u64);
        drop(checked_cpu_miss_total(*current, 480).unwrap());
        assert_eq!(*current, 7);
    }
    #[test]
    fn public_decode_profiles_are_strict_and_preflight_single_counts() {
        for (label, profile) in [
            ("chunk", NativeDecodeKernels::Chunk),
            ("single-moe", NativeDecodeKernels::SingleMoe),
            ("single-gdn", NativeDecodeKernels::SingleGdn),
            ("single-all", NativeDecodeKernels::SingleAll),
            ("single-q4-all", NativeDecodeKernels::SingleQ4All),
        ] {
            assert_eq!(NativeDecodeKernels::parse(label).unwrap(), profile);
            assert_eq!(profile.label(), label);
            for count in [0, 1, 3, 5, 8, 9, usize::MAX] {
                assert_eq!(
                    profile.validate_count(count).is_ok(),
                    (1..=MAX_BATCH).contains(&count)
                        && (profile == NativeDecodeKernels::Chunk || count == 1)
                );
            }
        }
        for label in [
            "",
            "1",
            "SingleAll",
            " single-all",
            "single-all\n",
            "chunk ",
        ] {
            assert!(NativeDecodeKernels::parse(label).is_err());
        }
        assert_eq!(NativeDecodeKernels::default(), NativeDecodeKernels::Chunk);
    }
    #[test]
    fn prepared_attention_inventory_fits_admitted_metadata_without_extra_state_roots() {
        // Check the selected feature's real aggregate, independently of older
        // feature-off hard-coded GPU quotes in the historical test below.
        assert!(metadata_bytes() <= META_CAP);
        if crate::strata_intel_core::enabled() {
            let prepared = AttentionInvocations::new().unwrap();
            assert_eq!(prepared.gdn.len(), LAYERS);
            assert_eq!(prepared.qsa.len(), LAYERS);
            assert!(prepared.gdn.iter().all(Option::is_none));
            assert!(prepared.qsa.iter().all(Option::is_none));
            // Each entry is an opaque metadata ticket, never a new GPU owner.
            assert_eq!(
                std::mem::size_of::<gdn::PreparedGdnBatch>(),
                std::mem::size_of::<gdn::PendingGdnBatch>()
            );
            assert_eq!(
                std::mem::size_of::<qsa::PreparedQsaBatch>(),
                std::mem::size_of::<qsa::PendingQsaBatch>()
            );
        }
    }
    #[test]
    fn aggregate_plan_counts_original_owners_once_and_independent_runtime_head_tiles() {
        assert_eq!(TOTAL_MAIN, 818249728);
        assert_eq!(
            NEW_MAIN + PUBLIC_MAIN + 2 * 67108864 + 41943040 + OLD_MAIN,
            1019576320
        );
        assert_eq!(runtime_bytes(16).unwrap(), 64000000);
        assert_eq!(runtime_bytes(38).unwrap(), 64090112);
        assert!(runtime_bytes(u64::MAX).is_err());
        assert!(runtime_bytes(4096).unwrap() > crate::executor::RESIDENT_RUNTIME_IO_MAX);
        assert!(metadata_bytes() <= META_CAP);
        assert_eq!(HOST_COMPONENTS, 11206656);
        const { assert!(HOST_COMPONENTS <= HOST_CAP) };
    }
    #[test]
    fn explicit_count1_reservation_reuses_packet_and_stays_under_independent_caps() {
        assert_eq!(
            batch_moe::Workspace::count1_hybrid_allocation_requests().unwrap(),
            [131072, 32768]
        );
        assert_eq!(TOTAL_MAIN + HYBRID_MAIN, 818413568);
        assert_eq!(1286 + 90 + 2, 1378);
        assert_eq!(HOST_CAP - (HOST_COMPONENTS + HYBRID_HOST), 327680);
        assert_eq!(HYBRID_PREPARE_INFO, 96 * 2048);
        assert_eq!(HYBRID_PREPARE_INFO, 196608);
        assert_eq!(HYBRID_RUNTIME, 1245184);
        assert_eq!(runtime_bytes(16).unwrap() + HYBRID_RUNTIME, 65245184);
        assert_eq!(runtime_bytes(38).unwrap() + HYBRID_RUNTIME, 65335296);
        assert_eq!(HYBRID_IO_DERIVED, 1212416);
        assert_eq!(HYBRID_RUNTIME - HYBRID_IO_DERIVED, 32768);
        assert!(
            runtime_bytes(38).unwrap() + HYBRID_RUNTIME <= crate::executor::RESIDENT_RUNTIME_IO_MAX
        );
        assert_eq!(
            BufferLayout::new(MAX_BATCH * ROUTER + 1)
                .unwrap()
                .owned_bytes,
            32768
        );
        const { assert!(D + ROUTER <= MAX_BATCH * ROUTER) };
    }
    #[test]
    fn runtime_packet_covers_counts_boundaries_without_future_or_inactive_values() {
        for count in [1, 3, 5, 8] {
            for end in [4, 2048, 2049, 2052, 4096] {
                if count > end {
                    continue;
                }
                let input = vec![7; count];
                let (ids, p, actual) = packet(end - count, 4096, &input).unwrap();
                assert_eq!(*actual, end);
                for t in 0..8 {
                    if t < count {
                        assert_eq!(ids[t], 7);
                        assert_eq!(p[t] as usize, end - count + t);
                    } else {
                        assert_eq!(ids[t], 0);
                        assert_eq!(p[t], 0);
                    }
                }
            }
        }
        for (start, context, ids) in [
            (0, 4096, vec![]),
            (0, 4096, vec![1; 9]),
            (4096, 4096, vec![1]),
            (usize::MAX, 4096, vec![1]),
            (0, 4096, vec![248320]),
            (0, 4097, vec![1]),
        ] {
            assert!(packet(start, context, &ids).is_err());
        }
    }
    #[test]
    fn actual_inflight_guard_abandon_error_unwind_is_sticky_without_lock_callback() {
        for kind in 0..4 {
            let poison = Arc::new(AtomicBool::new(false));
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut flight = Flight {
                    poison: poison.clone(),
                    accepted: false,
                };
                match kind {
                    0 => {
                        flight.accepted = true;
                        Ok(())
                    }
                    1 => Err(error()),
                    2 => Ok(()),
                    _ => panic!("public injected chunk panic"),
                }
            }));
            assert_eq!(poison.load(Ordering::Acquire), kind != 0);
            assert_eq!(attempt.is_err(), kind == 3);
        }
    }
    #[test]
    fn production_layer_driver_is_complete_and_stops_before_next_actual_layer() {
        for failed in [None, Some(0), Some(3), Some(23), Some(47)] {
            let mut queued = Vec::new();
            let result = drive_layers(&AtomicBool::new(false), |layer| {
                queued.push(layer);
                if failed == Some(layer) {
                    Err(error())
                } else {
                    Ok(())
                }
            });
            assert_eq!(
                queued,
                (0..failed.map_or(48, |n| n + 1)).collect::<Vec<_>>()
            );
            assert_eq!(result.is_err(), failed.is_some());
        }
        let cancelled = AtomicBool::new(false);
        let mut queued = Vec::new();
        assert!(
            drive_layers(&cancelled, |layer| {
                queued.push(layer);
                if layer == 3 {
                    cancelled.store(true, Ordering::Release);
                }
                Ok(())
            })
            .is_err()
        );
        assert_eq!(queued, [0, 1, 2, 3]);
    }
    #[test]
    fn production_two_proofs_never_return_output_or_ple_permit_after_any_failure() {
        use std::cell::RefCell;
        struct DropProof<'a>(&'a RefCell<Vec<&'static str>>, &'static str);
        impl Drop for DropProof<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().push(self.1);
            }
        }
        for failure in [None, Some(0), Some(1), Some(2)] {
            let events = RefCell::new(Vec::new());
            let result = completed_proof(
                || {
                    events.borrow_mut().push("first model owning fence/audit");
                    if failure == Some(0) {
                        Err(error())
                    } else {
                        Ok(DropProof(&events, "output dropped"))
                    }
                },
                || {
                    events.borrow_mut().push("PLE owning maps/audit/prepare");
                    if failure == Some(1) {
                        Err(error())
                    } else {
                        Ok(DropProof(&events, "PLE permit dropped"))
                    }
                },
                || {
                    events.borrow_mut().push("second model fault/cleanup proof");
                    if failure == Some(2) {
                        Err(error())
                    } else {
                        Ok(())
                    }
                },
            );
            assert_eq!(result.is_err(), failure.is_some());
            match result {
                Ok((output, permit)) => {
                    assert_eq!(events.borrow().len(), 3);
                    events.borrow_mut().push("all family preflight then commit");
                    drop(permit);
                    drop(output);
                }
                Err(_) => assert!(
                    !events
                        .borrow()
                        .contains(&"all family preflight then commit")
                ),
            }
            let observed = events.borrow();
            assert_eq!(observed.first(), Some(&"first model owning fence/audit"));
            if failure == Some(0) {
                assert_eq!(observed.len(), 1);
            }
            if failure == Some(1) {
                assert_eq!(
                    &*observed,
                    &[
                        "first model owning fence/audit",
                        "PLE owning maps/audit/prepare",
                        "output dropped"
                    ]
                );
            }
            if failure == Some(2) {
                assert_eq!(
                    &*observed,
                    &[
                        "first model owning fence/audit",
                        "PLE owning maps/audit/prepare",
                        "second model fault/cleanup proof",
                        "PLE permit dropped",
                        "output dropped"
                    ]
                );
            }
        }
    }
    #[test]
    fn wholeflight_poison_survives_panic_inside_the_actual_second_proof() {
        let poison = Arc::new(AtomicBool::new(false));
        let reached_commit = AtomicBool::new(false);
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _flight = Flight {
                poison: poison.clone(),
                accepted: false,
            };
            completed_proof(
                || Ok(()),
                || Ok(()),
                || panic!("public injected second-fence panic"),
            )
            .unwrap();
            reached_commit.store(true, Ordering::Release);
        }));
        assert!(attempt.is_err());
        assert!(poison.load(Ordering::Acquire));
        assert!(!reached_commit.load(Ordering::Acquire));
    }
    #[test]
    fn actual_limit_inventory_rejects_binding_alignment_and_workspace_shortfalls() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 16384,
            max_storage_buffers_per_shader_stage: 11,
            max_bindings_per_bind_group: 12,
            max_compute_workgroup_storage_size: 8192,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_buffer_size: 128 * 1024 * 1024,
            ..Default::default()
        };
        for tile in [4, 8] {
            validate_limits(&limits, tile).unwrap();
        }
        assert!(validate_limits(&limits, 16).is_err());
        for kind in 0..6 {
            let mut malformed = limits.clone();
            match kind {
                0 => malformed.max_compute_workgroups_per_dimension = 4096,
                1 => malformed.max_storage_buffers_per_shader_stage = 10,
                2 => malformed.max_bindings_per_bind_group = 11,
                3 => malformed.max_compute_workgroup_storage_size = 8191,
                4 => malformed.min_storage_buffer_offset_alignment = 1024,
                _ => malformed.min_uniform_buffer_offset_alignment = 0,
            }
            assert!(validate_limits(&malformed, 8).is_err());
        }
    }
}

#[cfg(test)]
mod prepared_verify3_plan_tests {
    use super::*;
    #[test]
    fn explicit_prepared_source_policy_and_default_reservations_remain_separate() {
        prepared_verify3_configuration(true, false, false, false, 8).unwrap();
        for args in [
            (false, false, false, false, 8),
            (true, true, false, false, 8),
            (true, false, true, false, 8),
            (true, false, false, true, 8),
            (true, false, false, false, 1),
            (true, false, false, false, 4),
        ] {
            assert!(
                prepared_verify3_configuration(args.0, args.1, args.2, args.3, args.4).is_err()
            );
        }
        assert_eq!(TARGET_ROWS_PRIVATE_MAIN, 826_638_336);
        assert_eq!(TARGET_ROWS_PREPARED_VERIFY3_MAIN, 826_703_872);
        assert_eq!(
            host_reservation_with_latest_checkpoint(DEFAULT_REUSE_METADATA).unwrap(),
            (18_808_832, 18_874_368)
        );
        assert_eq!(
            host_reservation_with_prepared_verify3(DEFAULT_REUSE_METADATA, true).unwrap(),
            (18_874_368, 18_874_368)
        );
        for reuse in [DEFAULT_REUSE_METADATA, 1048576, 4194304] {
            let old = host_reservation_with_checkpoint(false, reuse, false, true, true).unwrap();
            let new = host_reservation_with_prepared_verify3(reuse, true).unwrap();
            assert_eq!(new, (old.0 + 65536, old.1));
        }
        assert_eq!(
            host_reservation(false, DEFAULT_REUSE_METADATA).unwrap(),
            (11_468_800, 12_582_912)
        );
        let runtime = runtime_bytes(16).unwrap()
            + target_rows::RUNTIME_INCREMENT
            + latest_checkpoint::RUNTIME_INCREMENT
            + batch_moe::prepared_verify3::RUNTIME_INCREMENT;
        assert_eq!(runtime, 71_004_160);
        assert!(runtime < TARGET_ROWS_RUNTIME_LIMIT);
    }
}

#[cfg(test)]
mod live_member_reduction_plan_tests {
    use super::*;

    #[test]
    fn live_member_configuration_is_target_only_and_excludes_prepared_arithmetic() {
        live_member_reduction_configuration(true, true, false, false, false, 8).unwrap();
        for (rows, hybrid, router, prepared, tile) in [
            (false, false, false, false, 8),
            (true, true, false, false, 8),
            (true, false, true, false, 8),
            (true, false, false, true, 8),
            (true, false, false, false, 4),
            (true, false, false, false, 1),
        ] {
            assert!(
                live_member_reduction_configuration(true, rows, hybrid, router, prepared, tile,)
                    .is_err()
            );
            // The additive gate must not change any old construction policy.
            live_member_reduction_configuration(false, rows, hybrid, router, prepared, tile)
                .unwrap();
        }
    }

    #[test]
    fn live_member_reduction_spends_no_existing_or_reuse_owner_credit() {
        // There is no alternate MAIN/runtime/host plan for live reductions.
        // Full owners and INFO records stay charged under the original plan.
        assert_eq!(TARGET_ROWS_PRIVATE_MAIN, 826_638_336);
        assert_eq!(
            runtime_bytes(16).unwrap() + target_rows::RUNTIME_INCREMENT,
            70_295_552
        );
        for reuse in [DEFAULT_REUSE_METADATA, 1048576, 4194304] {
            live_member_reduction_configuration(true, true, false, false, false, 8).unwrap();
            assert_eq!(
                host_reservation_with_outputs(false, reuse, false, true).unwrap(),
                (
                    HOST_COMPONENTS + reuse + target_rows::HOST_INCREMENT,
                    HOST_CAP + reuse - DEFAULT_REUSE_METADATA + target_rows::HOST_INCREMENT
                )
            );
            assert_eq!(
                host_reservation_with_checkpoint(false, reuse, false, true, true).unwrap(),
                (
                    HOST_COMPONENTS
                        + reuse
                        + target_rows::HOST_INCREMENT
                        + latest_checkpoint::HOST_INCREMENT,
                    HOST_CAP + reuse - DEFAULT_REUSE_METADATA + target_rows::HOST_INCREMENT
                )
            );
        }
    }
}

#[cfg(test)]
mod gpu_qsa_plan_tests {
    use super::*;
    #[test]
    fn selector_keeps_default_and_rejects_non_target_geometry() {
        gpu_qsa_selector_configuration(true, true, false, false, 8).unwrap();
        for (rows, hybrid, router, tile) in [
            (false, false, false, 8),
            (true, true, false, 8),
            (true, false, true, 8),
            (true, false, false, 1),
        ] {
            assert!(gpu_qsa_selector_configuration(true, rows, hybrid, router, tile).is_err());
            gpu_qsa_selector_configuration(false, rows, hybrid, router, tile).unwrap();
        }
        assert_eq!(qsa::EXTRA_INFO_BYTES_PER_CHUNK, 12 * 2 * 2048);
        assert_eq!(qsa::EXTRA_HOST_BYTES_PER_CHUNK, 4096);
        assert_eq!(TARGET_ROWS_PRIVATE_MAIN, 826_638_336);
    }
}

#[cfg(test)]
mod frozen_source_configuration_tests {
    use super::*;
    #[test]
    fn original_target_domain_and_all_incompatible_policies_are_checked() {
        frozen_host_source_configuration(false, false, true, true, true, true, true, 1).unwrap();
        frozen_host_source_configuration(true, true, false, false, false, false, false, 8).unwrap();
        for (target, hybrid, router, prepared, batch_router, live, tile) in [
            (false, false, false, false, false, false, 8),
            (true, true, false, false, false, false, 8),
            (true, false, true, false, false, false, 8),
            (true, false, false, true, false, false, 8),
            (true, false, false, false, true, false, 8),
            (true, false, false, false, false, true, 8),
            (true, false, false, false, false, false, 4),
        ] {
            assert!(
                frozen_host_source_configuration(
                    true,
                    target,
                    hybrid,
                    router,
                    prepared,
                    batch_router,
                    live,
                    tile
                )
                .is_err()
            );
        }
    }
}
