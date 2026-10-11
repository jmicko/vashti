//! Actual layer-major MAX8 laboratory composition. It reuses the original
//! committed state/cache/banks and publishes only after whole-model acceptance.
//! Source-only until parent completes physical whole-model validation. No
//! default/API/catalog change, replay, fallback or private checkpoint export.
pub(crate) struct CompletedAdaptiveBoundary(());
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
#[path = "resident_target_rows.rs"]
mod target_rows;
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
pub(crate) const TARGET_ROWS_PRIVATE_MAIN: u64 = TOTAL_MAIN + target_rows::MAIN_BYTES;
#[allow(dead_code)] // Parent owns the exact executor construction-extent gate.
pub(crate) const TARGET_ROWS_PREPARED_VERIFY3_MAIN: u64 =
    TARGET_ROWS_PRIVATE_MAIN + batch_moe::prepared_verify3::MAIN_BYTES;

const OLD_MAIN: u64 = 785_186_816;
const NEW_MAIN: u64 = 33_062_912;
const TOTAL_MAIN: u64 = OLD_MAIN + NEW_MAIN;
const PUBLIC_MAIN: u64 = 25_165_824;
const HOST_CAP: usize = 12 * 1024 * 1024;
pub(crate) const ORIGINAL_BANK_HOST: usize = 640 * 1024;
const HOST_COMPONENTS: usize = 10_551_296 + ORIGINAL_BANK_HOST;
const META_CAP: usize = 256 * 1024;
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
const HYBRID_RUNTIME: u64 = 1024 * 1024 + HYBRID_PREPARE_INFO;
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
        return Err(error());
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
const HYBRID_IO_DERIVED: u64 =
    128 * 2048 + 4 * HYBRID_MAIN + 2 * 32768 + 4 * 8192 + HYBRID_PREPARE_INFO;
const _: () = {
    assert!(HYBRID_IO_DERIVED <= HYBRID_RUNTIME);
};

fn checked_cpu_miss_total(current: u64, completed_chunk: u64) -> Result<Zeroizing<u64>> {
    if completed_chunk > (LAYERS * 10) as u64 {
        return Err(error());
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
        }
    }
    fn single_moe(self) -> bool {
        matches!(self, Self::SingleMoe | Self::SingleAll | Self::SingleQ4All)
    }
    fn recurrence(self) -> crate::gdn_chunk::RecurrencePolicy {
        if matches!(self, Self::SingleGdn | Self::SingleAll | Self::SingleQ4All) {
            crate::gdn_chunk::RecurrencePolicy::SingleStreaming
        } else {
            crate::gdn_chunk::RecurrencePolicy::Chunk
        }
    }
    fn validate_count(self, count: usize) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&count) || self != Self::Chunk && count != 1 {
            return Err(error());
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
}
fn prepared_verify3_configuration(
    target_rows: bool,
    already_reserved: bool,
    hybrid: bool,
    router: bool,
    tile: usize,
) -> Result<()> {
    if !target_rows || already_reserved || hybrid || router || tile != 8 {
        return Err(error());
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
        return Err(error());
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
        return Err(error());
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
        return Err(error());
    }
    Ok(())
}
fn frozen_staging_configuration(selected: bool, sources: bool, router: bool) -> Result<()> {
    if selected && (!sources || !router) {
        return Err(error());
    }
    Ok(())
}
impl Config {
    fn qsa_host_reservation(&self) -> Result<(usize, usize)> {
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
            return Err(error());
        }
        frozen_staging_configuration(
            self.public_frozen_host_staging,
            self.public_frozen_host_sources,
            self.public_frozen_host_router,
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
        }
        self.public_frozen_host_router = true;
        Ok(self)
    }
    /// Bounded original packed host-weight copies; charge before device owners.
    #[allow(dead_code)] // Default-off parent selection is wired in the staging leaf integration.
    pub(crate) fn reserve_public_frozen_host_staging(mut self) -> Result<Self> {
        if self.public_frozen_host_staging {
            return Err(error());
        }
        frozen_staging_configuration(
            true,
            self.public_frozen_host_sources,
            self.public_frozen_host_router,
        )?;
        self.qsa_host_reservation()?;
        self.public_frozen_host_staging = true;
        Ok(self)
    }
    /// Construction-fixed original-F32 reduction of actual grouped members.
    /// The complete MAX8 roots, original ranks, INFO charges and all owning
    /// finite/status/padding proofs remain unchanged. Parent owns the fixed
    /// public DASLab registration and physical numerical/performance gates.
    #[allow(dead_code)]
    pub(crate) fn with_live_member_reduction(mut self) -> Result<Self> {
        if self.live_member_reduction {
            return Err(error());
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
        }
        host_reservation_with_outputs(false, self.request_reuse_metadata, false, true)?;
        self.target_rows_reserved = true;
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
            return Err(error());
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
            return Err(error());
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
    ) -> Result<Self> {
        graph.verify_unchanged()?;
        if !std::ptr::eq(client.info(), config.client.info())
            || sum(old)? != OLD_MAIN
            || old.len() != 1286
        {
            return Err(error());
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
                return Err(error());
            }
        }
        if config.public_frozen_host_sources
            && (executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || executor.resident_bank_capacity() != 144
                || executor.target_frozen_host_source_identity.is_some()
                || executor.resident_banks_frozen)
        {
            return Err(error());
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
            return Err(error());
        }
        let mut requests = Vec::new();
        for (pages, expected) in [
            (
                gr::Workspace::allocation_requests(graph)?.to_vec(),
                1_998_848,
            ),
            (
                gdn::GdnBatchWorkspace::allocation_requests(graph)?.to_vec(),
                3_932_160,
            ),
            (
                qsa::QsaBatchWorkspace::allocation_requests(graph)?.to_vec(),
                17_825_792,
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
                return Err(error());
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
                || executor.resident_bank_capacity() != 144
                || executor.resident_runtime_io_limit() != public_language_request_io::RUNTIME_BYTES
            {
                return Err(error());
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
                return Err(error());
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
            return Err(error());
        }
        if router != executor.resident_router_reserved
            || router
                && (!hybrid
                    || !executor.packed_profile.is_low_bit()
                    || executor.resident_bank_capacity() != 144)
        {
            return Err(error());
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
                || executor.resident_bank_capacity() != 144
                || executor.q3_integer_moe
                || executor.resident_banks_frozen
                || executor.adaptive.is_some()
                || executor.secondary.is_some()
            {
                return Err(error());
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
                || executor.resident_bank_capacity() != 144
                || executor.resident_banks_frozen
                || executor.adaptive.is_some()
                || executor.secondary.is_some()
                || executor.secondary_frequency.is_some())
        {
            return Err(error());
        }
        if config.public_warm_batch_router {
            if !config.public_warm_replay
                || !config.public_target_request_io
                || !target_rows
                || hybrid
                || router
                || prepared_verify3
                || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || executor.resident_bank_capacity() != 144
            {
                return Err(error());
            }
            requests.extend(batch_moe::Workspace::batch_router_allocation_requests());
        }
        if config.public_frozen_host_router {
            if !config.public_frozen_host_sources || config.public_warm_batch_router {
                return Err(error());
            }
            requests.extend(batch_moe::Workspace::batch_router_allocation_requests());
        }
        if (config.public_warm_batch_router || config.public_frozen_host_router)
            != executor.target_batch_router_reserved
            || config.public_frozen_host_router != executor.target_frozen_host_router_reserved
        {
            return Err(error());
        }
        if config.public_frozen_host_staging {
            requests.extend([batch_moe::FROZEN_STAGING_OWNER_BYTES; 3]);
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
        if requests.len()
            != 90
                + usize::from(config.public_frozen_host_staging) * 3
                + usize::from(hybrid) * 2
                + usize::from(router) * 2
                + usize::from(target_rows) * 2
                + usize::from(prepared_verify3)
                + usize::from(config.public_warm_batch_router || config.public_frozen_host_router)
                    * 2
            || sum(&requests)? != NEW_MAIN + extra
            || OLD_MAIN.checked_add(sum(&requests)?) != Some(TOTAL_MAIN + extra)
        {
            return Err(error());
        }
        let head = graph.tensor("output.weight")?;
        let row = head.info().row_bytes();
        if row == 0 {
            return Err(error());
        }
        let tile_rows = (64_u64 * 1024 * 1024 / row).min(16384);
        if tile_rows == 0 {
            return Err(error());
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
            .ok_or(Error::ArithmeticOverflow)?;
        if runtime > executor.resident_runtime_io_limit()
            || HOST_COMPONENTS + if hybrid { HYBRID_HOST } else { 0 } > HOST_CAP
            || metadata_bytes() > META_CAP
        {
            return Err(Error::LimitExceeded("native model aggregate private plan"));
        }
        // All component/API-visible reservations remain independent of MAIN.
        if gdn::GdnBatchWorkspace::runtime_info_reservation(graph, true)? != 4_718_592
            || qsa::QsaBatchWorkspace::runtime_io_reservation(graph, true)? != 7_864_320
            || batch_moe::Workspace::runtime_io_increment() != 524288
            || ops::GUARDED_ROUTER_CAPACITY_BYTES > 8192
        {
            return Err(error());
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
            return Err(error());
        }
        let derived = (0..LAYERS).try_fold(0usize, |n, l| {
            Ok::<_, Error>(n + usize::from(graph.layer_kind(l)? == AttentionKind::Gdn))
        })?;
        if 341 + derived + 2 != 379 {
            return Err(error());
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
    const WITHOUT_HEAD: u64 = 61_751_296 + OUTER_CLEAR_INFO;
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
enum ChunkOutput {
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
pub(super) struct Workspace {
    gr: gr::Workspace,
    gdn: gdn::GdnBatchWorkspace,
    qsa: qsa::QsaBatchWorkspace,
    moe: batch_moe::Workspace,
    ple: ple_batch::PleBatchWorkspace,
    shared: shared_batch::Workspace,
    outer: Zeroizing<Outer>,
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
}
impl Workspace {
    fn decode_kernels(&self) -> Result<NativeDecodeKernels> {
        let single_moe = match self.moe.projection_tile() {
            1 => true,
            4 | 8 => false,
            _ => return Err(error()),
        };
        let single_gdn =
            self.gdn.recurrence_policy() == crate::gdn_chunk::RecurrencePolicy::SingleStreaming;
        let q4_block =
            self.moe.gate_policy() == crate::quant_matvec::grouped_moe::GatePolicy::Q4Block;
        Ok(
            match (single_moe, single_gdn, q4_block, self.dense_member_capacity) {
                (false, false, false, 8) => NativeDecodeKernels::Chunk,
                (true, false, false, 8) => NativeDecodeKernels::SingleMoe,
                (false, true, false, 8) => NativeDecodeKernels::SingleGdn,
                (true, true, false, 1) => NativeDecodeKernels::SingleAll,
                (true, true, true, 1) => NativeDecodeKernels::SingleQ4All,
                _ => return Err(error()),
            },
        )
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
                return Err(error());
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
            return Err(error());
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
            moe.reserve_frozen_host_batch_router(&mut model.arena, allocation, model.context)?;
        }
        if config.public_frozen_host_staging {
            moe.reserve_frozen_host_staging(&mut model.arena, allocation)?;
        }
        if config.prepared_verify3_experts_reserved {
            moe.reserve_prepared_verify3(&mut model.arena, allocation)?;
        }
        let shared = shared_batch::Workspace::allocate(graph, &mut model.arena)?;
        let mut alloc = |n| model.arena.allocate_zeroed(n).map_err(u::fault);
        let outer = Zeroizing::new(Outer {
            residual: [alloc(RESIDUAL_WORDS)?, alloc(RESIDUAL_WORDS)?],
            block: [alloc(BLOCK_WORDS)?, alloc(BLOCK_WORDS)?],
            router: alloc(MAX_BATCH * ROUTER)?,
            packet: alloc(MAX_BATCH * ROUTER + 1)?,
        });
        let target_rows = if config.target_rows_reserved {
            Some(target_rows::Owners::allocate(&mut model.arena)?)
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
            for (id, words) in ids
                .into_iter()
                .zip([48 * 512, crate::resident_batch_router_gpu::AUDIT_WORDS])
            {
                handles.push(gr::full_private(&model.arena, id, words)?);
            }
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
            return Err(error());
        }
        Ok(())
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        self.poison.store(true, Ordering::Release);
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
        if tile != 1 && self.native_decode_kernels()? != NativeDecodeKernels::Chunk {
            return Err(error());
        }
        let tile = NativeProjectionTile::new(tile)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        executor.set_native_projection_tile(tile, boundary)
    }

    fn completed_native_projection_boundary(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<CompletedNativeProjectionBoundary> {
        if self.poisoned
            || self.retired.load(Ordering::Acquire)
            || !self.pending_qsa.is_empty()
            || !self.pending_moe.is_empty()
            || self.ple_batch_pending.is_some()
            || !self.admission.transient_io()
            || !self.arena.public_owned_zero_audit_enabled()
            || !self.io.public_owned_zero_audit_enabled()
        {
            return Err(error());
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
        native.ple.validate_completed_projection_boundary(self)?;
        Ok(CompletedNativeProjectionBoundary { _private: () })
    }

    /// Actual committed cursor for the explicitly reserved target-row owner.
    /// This performs the existing complete native boundary proof; it does not
    /// infer completion from the returned result's start/count metadata.
    #[allow(dead_code)] // Parent owns the dedicated fixed-public registration.
    pub(crate) fn completed_target_rows_cursor(&self, executor: &GpuExecutor<'f>) -> Result<usize> {
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !executor.target_rows_reserved || native.target_rows.is_none() {
            return Err(error());
        }
        executor.check_target_rows_policy()?;
        self.completed_native_projection_boundary(executor)?;
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
            return Err(error());
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
                return Err(error());
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
                return Err(error());
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
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
        let fills =
            if self.context == 13 && native.public_warm_replay.is_some() && !preserve_routed_bank {
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
        self.completed_native_projection_boundary(executor)
            .map_err(|_| Error::InvalidFormat("public host prepare filled completed boundary"))?;
        let prepared = crate::public_host_expert_attachment::PreparedPlan::new(
            self.graph,
            executor.public_host_model(),
            executor,
            65536,
            crate::public_host_expert_plan::METADATA_HOST_CAP,
            cancel,
        )?;
        Ok((prepared, fills))
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
        cancel: &AtomicBool,
    ) -> Result<crate::public_host_expert_attachment::Attachment<'plan, 'm, 'f>> {
        u::cancel(cancel)?;
        if self.poisoned
            || self.retired.load(Ordering::Acquire)
            || self.admission.request_io_reuse_active()
            || !self.pending_qsa.is_empty()
            || !self.pending_moe.is_empty()
            || self.ple_batch_pending.is_some()
        {
            return Err(error());
        }
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || native.moe.frozen_host_sources_enabled() {
            return Err(error());
        }
        if self.position == 0 {
            if self.context != 13
                || native
                    .public_warm_replay
                    .as_ref()
                    .is_none_or(|permit| !permit.reset_consumed())
            {
                return Err(error());
            }
        } else {
            self.completed_native_projection_boundary(executor)?;
        }
        native.prove_outer(self)?;
        native
            .moe
            .validate_public_warm_boundary(executor, self.graph, &self.arena)?;
        // SAFETY: fixed-public caller verified original source hashes and
        // immutability. All actual bank/family/device checks above precede import.
        unsafe {
            prepared.attach(
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
                cancel,
            )
        }
    }
    pub(crate) fn native_frozen_source_counters(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<batch_moe::frozen_sources::Counters> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        if !native.public_frozen_host_sources || !native.moe.frozen_host_sources_enabled() {
            return Err(error());
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
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
            return Err(error());
        }
        native.moe.frozen_host_source_two_row_gate_up()
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
            return Err(error());
        }
        if self.position == 0 {
            if self.context != 13
                || self
                    .native_batch
                    .as_ref()
                    .and_then(|n| n.public_warm_replay.as_ref())
                    .is_none_or(|p| !p.reset_consumed())
            {
                return Err(error());
            }
        } else {
            self.completed_native_projection_boundary(executor)?;
        }
        let native = self.native_batch.as_mut().ok_or(error())?;
        if !native.public_frozen_host_sources
            || !native.public_frozen_host_router
            || !native.moe.frozen_host_sources_enabled()
        {
            return Err(error());
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
            return Err(error());
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
            return Err(error());
        }
        self.admission.begin_request_io_reuse()
    }

    pub(crate) fn request_io_checkpoints(&self) -> u64 {
        self.admission.request_io_checkpoints()
    }

    pub(crate) fn end_lab_request_io_reuse(&self) -> Result<()> {
        self.admission.end_request_io_reuse()
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
        if cap > 10 {
            return Err(error());
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

    pub(crate) fn set_lab_native_decode_kernels(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        policy: NativeDecodeKernels,
        cancel: &AtomicBool,
    ) -> Result<NativeDecodeKernels> {
        u::cancel(cancel)?;
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
            return Err(error());
        }
        // Each fixture has one public continuation policy. Repeated same-policy
        // selection is allowed; a fresh owner is required for another sample.
        if observed != NativeDecodeKernels::Chunk && observed != policy
            || policy != NativeDecodeKernels::Chunk && executor.native_projection_tile() != 1
        {
            return Err(error());
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
                return Err(error());
            }
            let actual = self.native_decode_kernels()?;
            if actual != policy {
                return Err(error());
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
            return Err(error());
        }
        self.execute_native_chunk(executor, tokens, purpose, cancel, |output| match output {
            ChunkOutput::Legacy(logits) => Ok(logits),
            ChunkOutput::TargetRows(_) => Err(error()),
        })
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
            cancel,
            |output| match output {
                ChunkOutput::TargetRows(rows) => Ok(rows),
                ChunkOutput::Legacy(_) => Err(error()),
            },
        )
    }
    fn execute_native_chunk<T>(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        purpose: Purpose,
        cancel: &AtomicBool,
        accept: impl FnOnce(ChunkOutput) -> Result<T>,
    ) -> Result<T> {
        if let Some(rows) = &self.native_batch.as_ref().ok_or(error())?.target_rows {
            // No command/state mutation occurs while an admitted output-host
            // payload is still held. This guard also covers legacy purposes.
            rows.check_no_live_output()?;
        }
        let mut native = self.native_batch.take().ok_or(error())?;
        let mut flight = Flight {
            poison: native.poison.clone(),
            accepted: false,
        };
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native
                .run(self, executor, tokens, purpose, cancel)
                .and_then(accept)
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
        return Err(error());
    }
    let end = start
        .checked_add(tokens.len())
        .ok_or(Error::ArithmeticOverflow)?;
    if end > context || tokens.iter().any(|&t| t as usize >= 248320) {
        return Err(error());
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
    fn run(
        &mut self,
        m: &mut ResidentModel<'_, '_>,
        e: &mut GpuExecutor<'_>,
        tokens: &[u32],
        purpose: Purpose,
        cancel: &AtomicBool,
    ) -> Result<ChunkOutput> {
        self.prove_outer(m)?;
        e.check_client(m.arena.client().map_err(u::fault)?)?;
        e.check_resident_admission(&m.admission)?;
        if self.dense_member_capacity != e.native_projection_member_capacity() {
            return Err(error());
        }
        if e.target_rows_reserved != self.target_rows.is_some()
            || purpose == Purpose::AllRowsWithCarry && self.target_rows.is_none()
        {
            return Err(error());
        }
        if self.target_rows.is_some() {
            e.check_target_rows_policy()?;
            target_rows::coordinates(tokens.len(), m.position)?;
            if self.decode_kernels()? != NativeDecodeKernels::Chunk
                || self.moe.cpu_miss_cap().is_some()
                || self.moe.cpu_gpu_overlap().is_some()
                || self.moe.resident_router_active()
            {
                return Err(error());
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
            if self.decode_kernels()? != NativeDecodeKernels::Chunk {
                return Err(error());
            }
        }
        if self.moe.prepared_verify3_owner_id().is_some() {
            if self.target_rows.is_none()
                || e.q3_integer_moe
                || e.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
            {
                return Err(error());
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
        self.decode_kernels()?.validate_count(tokens.len())?;
        if self.moe.cpu_miss_cap().is_some() && tokens.len() != 1 {
            return Err(error());
        }
        if m.poisoned
            || m.retired.load(Ordering::Acquire)
            || !m.pending_qsa.is_empty()
            || !m.pending_moe.is_empty()
            || m.ple_batch_pending.is_some()
            || m.layers.len() != LAYERS
        {
            return Err(error());
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
            if layer == 1 {
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
                return Err(error());
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
                        return Err(error());
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
                        return Err(error());
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
                        return Err(error());
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
                        return Err(error());
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
                        return Err(error());
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
            };
            self.gr
                .read(
                    m.graph,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    gr::Kind::FinalAllRows,
                    self.outer.residual[*current],
                    *count,
                    m.b.fault,
                    cancel,
                )?
                .enqueue_final_head(m.graph, &mut projector, rows.logits())?;
        }
        *m.phase_ns.entry("native_final_head_enqueue").or_default() +=
            final_head_start.elapsed().as_nanos();
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
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
                return Err(error());
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
                return Err(error());
            }
            self.qsa
                .accept_after_whole_model_chunk_acceptance(&m.arena, &refs, cancel)?
        };
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
            return Err(error());
        }
        // Prepare the checked counter before ANY family publication. The
        // guarded proposal is discarded on all late failure/cancellation paths.
        let completed_cpu_misses =
            checked_cpu_miss_total(*m.completed_cpu_misses, ma.completed_cpu_misses())?;
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
            return Err(error());
        }
        *m.completed_cpu_misses = *completed_cpu_misses;
        m.position = *end;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
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
