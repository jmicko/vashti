//! Fixed-public original expert lab. Caller supplies an inspected
//! file and an exclusively configured transient stream; no path reopen/default.
//! Reference arithmetic follows bank96_probe and core CPU's original 64 tree.
//! Numeric success is local operator evidence, never whole-model acceptance.
use crate::{
    bank_slots::{Plan, StagedSlots},
    cpu_gpu_expert_schedule::{self as schedule, Mode, Split, Times},
    expert_banks::{self, ExpertBanks},
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, PrivateArena},
    quant_matvec,
    resident_moe::{self, ResidentMoe},
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::transient::TransientIoReceipt;
use kuno_text::cpu::Matrix;
use kuno_text::cpu_expert::{Backend, ExpertOperator, PackedExperts, PackedProjection, WorkerPlan};
use kuno_text::model::{BoundGraph, GraphBindingId, RoutedWeights};
use kuno_text::{Error, GgufFile, Result, TensorType, TensorView};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
const D: usize = 2560;
const FF: usize = 640;
const N: usize = 10;
const MIB: u64 = 1024 * 1024;
pub const PRIVATE_BYTES: u64 = 884_736;
pub const BANK_BYTES: u64 = 134_217_728;
pub const SOURCE_BYTES: u64 = 41_943_040;
pub const IO_COMPONENT_BYTES: u64 = 8 * MIB;
pub const PHYSICAL_LIMIT: u64 = 256 * MIB;
pub const HOST_PRIVATE_LIMIT: usize = 4 * 1024 * 1024;
const HOST_BRIDGE_LIMIT: usize = 1024 * 1024;
const HOST_ORACLE_LIMIT: usize = 2 * 1024 * 1024;
const STATUS: usize = 580;
const PROBS: [[f32; N]; 2] = [
    [0.05, 0.07, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.10, 0.08],
    [0.0, -0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3],
];
fn fault() -> Error {
    Error::InvalidFormat("public CPU/GPU expert fixture rejected")
}
/// Numeric-only infrastructure classification. Never retain or format the
/// original error, selected ranks, private values or backend payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericStage {
    Config,
    Limits,
    Control,
    Workers,
    Geometry,
    OriginalBanks,
    StartupIo,
    StartupCapacity,
    HostBudget,
    Allocation,
    PhysicalOwners,
    PhysicalClient,
    PhysicalMetadata,
    PhysicalPrivateOwner,
    PhysicalReferenceOwner,
    PhysicalBankOwner,
    PhysicalResource,
    PhysicalPrivateExtent,
    PhysicalReferenceExtent,
    PhysicalBankOffset,
    PhysicalBankSize,
    PhysicalBankBacking,
    PhysicalAlias,
    FixtureUpload,
    Priming,
    Reference,
    CandidatePlan,
    Candidate,
    CpuPartials,
    Merge,
    NumericAcceptance,
    FinalAudit,
    FileConsistency,
    Metadata,
    SlotCommit,
    Sentinels,
    Erasure,
}
impl NumericStage {
    fn error(self) -> Error {
        Error::InvalidFormat(match self {
            Self::Config => "public CPU/GPU numeric config rejected",
            Self::Limits => "public CPU/GPU numeric limits rejected",
            Self::Control => "public CPU/GPU numeric control rejected",
            Self::Workers => "public CPU/GPU numeric workers rejected",
            Self::Geometry => "public CPU/GPU numeric geometry rejected",
            Self::OriginalBanks => "public CPU/GPU numeric original-banks rejected",
            Self::StartupIo => "public CPU/GPU numeric startup-io rejected",
            Self::StartupCapacity => "public CPU/GPU numeric startup-capacity rejected",
            Self::HostBudget => "public CPU/GPU numeric host-budget rejected",
            Self::Allocation => "public CPU/GPU numeric allocation rejected",
            Self::PhysicalOwners => "public CPU/GPU numeric physical-owners rejected",
            Self::PhysicalClient => "public CPU/GPU numeric physical-client rejected",
            Self::PhysicalMetadata => "public CPU/GPU numeric physical-metadata rejected",
            Self::PhysicalPrivateOwner => "public CPU/GPU numeric physical-private-owner rejected",
            Self::PhysicalReferenceOwner => {
                "public CPU/GPU numeric physical-reference-owner rejected"
            }
            Self::PhysicalBankOwner => "public CPU/GPU numeric physical-public-bank-owner rejected",
            Self::PhysicalResource => "public CPU/GPU numeric physical-resource rejected",
            Self::PhysicalPrivateExtent => {
                "public CPU/GPU numeric physical-private-extent rejected"
            }
            Self::PhysicalReferenceExtent => {
                "public CPU/GPU numeric physical-reference-extent rejected"
            }
            Self::PhysicalBankOffset => {
                "public CPU/GPU numeric physical-public-bank-offset rejected"
            }
            Self::PhysicalBankSize => "public CPU/GPU numeric physical-public-bank-size rejected",
            Self::PhysicalBankBacking => {
                "public CPU/GPU numeric physical-public-bank-backing rejected"
            }
            Self::PhysicalAlias => "public CPU/GPU numeric physical-alias rejected",
            Self::FixtureUpload => "public CPU/GPU numeric fixture-upload rejected",
            Self::Priming => "public CPU/GPU numeric priming rejected",
            Self::Reference => "public CPU/GPU numeric reference rejected",
            Self::CandidatePlan => "public CPU/GPU numeric candidate-plan rejected",
            Self::Candidate => "public CPU/GPU numeric candidate rejected",
            Self::CpuPartials => "public CPU/GPU numeric cpu-partials rejected",
            Self::Merge => "public CPU/GPU numeric merge rejected",
            Self::NumericAcceptance => "public CPU/GPU numeric acceptance rejected",
            Self::FinalAudit => "public CPU/GPU numeric final-audit rejected",
            Self::FileConsistency => "public CPU/GPU numeric file-consistency rejected",
            Self::Metadata => "public CPU/GPU numeric metadata rejected",
            Self::SlotCommit => "public CPU/GPU numeric slot-commit rejected",
            Self::Sentinels => "public CPU/GPU numeric sentinels rejected",
            Self::Erasure => "public CPU/GPU numeric erasure rejected",
        })
    }
}
fn classified_numeric_result<T>(
    numeric: bool,
    stage: NumericStage,
    result: Result<T>,
) -> Result<T> {
    result.map_err(|original| if numeric { stage.error() } else { original })
}
fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(fault())
    } else {
        Ok(())
    }
}
/// Public geometry/work ownership only. No user routes or prompt-derived vectors.
#[derive(Clone, Copy)]
pub struct Config {
    pub down: TensorType,
    pub workers: usize,
    pub backend: Backend,
    pub misses: usize,
    pub cpu_misses: usize,
    pub group: usize,
    pub vector: usize,
    pub mode: Mode,
}
impl Config {
    fn validate(self) -> Result<()> {
        if !matches!(self.down, TensorType::Q5_0 | TensorType::Q8_0)
            || ![1, 4, 8, 16].contains(&self.workers)
            || self.misses > N
            || self.cpu_misses > self.misses
            || self.group >= 32
            || self.vector > 1
        {
            return Err(fault());
        }
        Ok(())
    }
}
/// Deliberately has no public constructor: a real numerical run grants only
/// this exact fixture/client/file/config another explicitly requested timing run.
pub struct NumericPermit<'m> {
    client: ComputeClient<WgpuRuntime>,
    file: &'m GgufFile,
    config: Config,
    binding: GraphBindingId,
    samples: Arc<std::sync::atomic::AtomicUsize>,
    retired: Arc<AtomicBool>,
    thread: std::thread::ThreadId,
}
pub enum Purpose<'a, 'm> {
    Numeric,
    Calibration(&'a NumericPermit<'m>),
}
pub struct Report {
    pub layer: usize,
    pub down_dtype: u32,
    pub workers: usize,
    pub misses: usize,
    pub cpu_misses: usize,
    pub gpu_misses: usize,
    pub max_tree_error: f32,
    pub max_legacy_error: f32,
    pub max_core_error: f32,
    pub private_pages: usize,
    pub private_bytes: u64,
    pub bank_bytes: u64,
    pub source_precharge: u64,
    pub io_inspected_bytes: u64,
    pub zero_buffers: u64,
    pub zero_bytes: u64,
    pub timing: Option<Times>,
    pub bridge_and_merge: Option<std::time::Duration>,
    pub metadata_bytes: usize,
    pub snapshot_bytes: usize,
}
fn same_config(a: Config, b: Config) -> bool {
    a.down == b.down
        && a.workers == b.workers
        && a.backend == b.backend
        && a.misses == b.misses
        && a.cpu_misses == b.cpu_misses
        && a.group == b.group
        && a.vector == b.vector
        && a.mode == b.mode
}
fn limits(l: &wgpu::Limits) -> Result<()> {
    if l.max_compute_invocations_per_workgroup < 256
        || l.max_compute_workgroup_size_x < 256
        || l.max_compute_workgroups_per_dimension < 25600
        || l.max_compute_workgroup_storage_size < 512
        || l.max_storage_buffers_per_shader_stage < 7
        || l.max_bindings_per_bind_group < 7
        || l.max_bind_groups < 1
        || l.max_uniform_buffer_binding_size < 512
        || l.max_storage_buffer_binding_size < 128 * MIB
        || l.max_buffer_size < 128 * MIB
        || l.min_storage_buffer_offset_alignment == 0
        || l.min_storage_buffer_offset_alignment > 512
        || l.min_uniform_buffer_offset_alignment == 0
        || l.min_uniform_buffer_offset_alignment > 512
    {
        return Err(fault());
    }
    Ok(())
}
fn sample(c: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    c.memory_usage()
        .map(|m| m.bytes_reserved)
        .map_err(|_| fault())
}
fn observe(l: &mut MainAdmission, c: &ComputeClient<WgpuRuntime>) -> Result<()> {
    l.observe_after(sample(c)?).map_err(|_| fault())
}
fn audit(c: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<TransientIoReceipt> {
    let r = cubecl::future::block_on(c.sanitize_transient_io_audited()).map_err(|_| {
        retired.store(true, Ordering::Release);
        fault()
    })?;
    let n = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes));
    if n == Some(0)
        || n != Some(r.inspected_bytes)
        || r.inspected_buffers == 0
        || r.inspected_bytes > IO_COMPONENT_BYTES / 2
    {
        retired.store(true, Ordering::Release);
        return Err(fault());
    }
    Ok(r)
}
fn fence(
    arena: &PrivateArena,
    l: &mut MainAdmission,
    retired: &AtomicBool,
) -> Result<TransientIoReceipt> {
    let r = audit(arena.client().map_err(|_| fault())?, retired)?;
    // Completed consumers and dropped leases permit actual source reclamation.
    arena.client().map_err(|_| fault())?.memory_cleanup();
    l.completed_fence(sample(arena.client().map_err(|_| fault())?)?)
        .map_err(|_| fault())?;
    Ok(r)
}
fn guarded<T>(n: usize, limit: usize) -> Result<Zeroizing<Vec<T>>>
where
    T: Zeroize + Default + Clone,
{
    if n.checked_mul(size_of::<T>()).is_none_or(|n| n > limit) {
        return Err(fault());
    }
    let mut v = Zeroizing::new(Vec::new());
    v.try_reserve_exact(n).map_err(|_| fault())?;
    if v.capacity() != n {
        return Err(fault());
    }
    if v.capacity()
        .checked_mul(size_of::<T>())
        .is_none_or(|n| n > limit)
    {
        return Err(fault());
    }
    v.resize(n, T::default());
    Ok(v)
}
struct Lease<'a>(&'a mut [u8]);
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn validate_words(bytes: &[u8], logical: usize, owned: usize) -> Result<()> {
    let end = logical.checked_mul(4).ok_or_else(fault)?;
    if bytes.len() != owned || end > owned || bytes[end..].iter().any(|&b| b != 0) {
        return Err(fault());
    }
    Ok(())
}
fn read(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
    let b = arena.layout(id).map_err(|_| fault())?;
    let mut bytes = arena
        .client()
        .map_err(|_| fault())?
        .read_one(arena.handle(id).map_err(|_| fault())?.clone())
        .map_err(|_| fault())?;
    let lease = Lease(&mut bytes);
    validate_words(lease.0, b.logical_words, b.owned_bytes)?;
    let mut out = guarded(b.logical_words, HOST_ORACLE_LIMIT)?;
    for (i, chunk) in lease.0[..b.logical_words * 4].chunks_exact(4).enumerate() {
        out[i] = u32::from_le_bytes(chunk.try_into().map_err(|_| fault())?);
    }
    Ok(out)
}
fn floats(words: &[u32]) -> Result<Zeroizing<Vec<f32>>> {
    let mut v = guarded(words.len(), HOST_ORACLE_LIMIT)?;
    for (i, &bits) in words.iter().enumerate() {
        v[i] = f32::from_bits(bits);
        if !v[i].is_finite() {
            return Err(fault());
        }
    }
    Ok(v)
}
fn upload(arena: &PrivateArena, id: BufferId, words: &[u32]) -> Result<()> {
    let b = arena.layout(id).map_err(|_| fault())?;
    if words.len() != b.logical_words {
        return Err(fault());
    }
    let mut bytes = guarded(b.owned_bytes, HOST_BRIDGE_LIMIT)?;
    for (i, &word) in words.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    arena
        .client()
        .map_err(|_| fault())?
        .write_private(
            arena.handle(id).map_err(|_| fault())?.clone(),
            std::mem::take(&mut *bytes),
        )
        .map_err(|_| fault())
}
fn upload_f32(arena: &PrivateArena, id: BufferId, x: &[f32]) -> Result<()> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(fault());
    }
    let mut w = guarded(x.len(), HOST_BRIDGE_LIMIT)?;
    for (i, &v) in x.iter().enumerate() {
        w[i] = v.to_bits();
    }
    upload(arena, id, &w)
}
fn compare(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> Result<f32> {
    if a.len() != b.len() || a.is_empty() {
        return Err(fault());
    }
    let mut max = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        let d = (a - b).abs();
        if !a.is_finite() || !b.is_finite() || d > atol + rtol * b.abs() {
            return Err(fault());
        }
        max = max.max(d);
    }
    Ok(max)
}
fn bits(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() || a.iter().zip(b).any(|(&a, &b)| a.to_bits() != b.to_bits()) {
        return Err(fault());
    }
    Ok(())
}
fn dot64(w: &[f32], x: &[f32]) -> Result<f32> {
    if w.len() != x.len() {
        return Err(fault());
    }
    let mut sums = Zeroizing::new([0.0f32; 64]);
    for (lane, s) in sums.iter_mut().enumerate() {
        for c in (lane..x.len()).step_by(64) {
            *s += w[c] * x[c];
        }
    }
    for s in [32, 16, 8, 4, 2, 1] {
        for l in 0..s {
            sums[l] += sums[l + s];
        }
    }
    if !sums[0].is_finite() {
        return Err(fault());
    }
    Ok(sums[0])
}
fn project(
    t: TensorView<'_>,
    e: usize,
    x: &[f32],
    tree: bool,
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<f32>>> {
    let m = Matrix::new(t)?;
    if m.inputs() != x.len() || m.outputs() > D || e >= 512 {
        return Err(fault());
    }
    let mut out = guarded(m.outputs(), HOST_ORACLE_LIMIT)?;
    if !tree {
        m.apply(e, x, &mut out, cancel)?;
        return Ok(out);
    }
    let rb = t.info().row_bytes() as usize;
    let mut packed = vec![0; rb];
    if packed.capacity() > 8 * 1024 * 1024 {
        return Err(fault());
    }
    let mut decoded = guarded(x.len(), HOST_ORACLE_LIMIT)?;
    for (r, y) in out.iter_mut().enumerate() {
        check(cancel)?;
        t.read_range(((e * m.outputs() + r) * rb) as u64, &mut packed)?;
        t.info().dtype().decode(&packed, &mut decoded)?;
        if decoded.iter().any(|v| !v.is_finite()) {
            return Err(fault());
        }
        *y = dot64(&decoded, x)?;
    }
    Ok(out)
}
struct Oracle {
    act: Zeroizing<Vec<f32>>,
    part: Zeroizing<Vec<f32>>,
    out: Zeroizing<Vec<f32>>,
    core: Zeroizing<Vec<f32>>,
}
fn oracle(
    w: RoutedWeights<'_>,
    ids: &[usize; N],
    x: &[f32],
    p: &[f32; N],
    cancel: &AtomicBool,
) -> Result<Oracle> {
    let mut o = Oracle {
        act: guarded(N * FF, HOST_ORACLE_LIMIT)?,
        part: guarded(N * D, HOST_ORACLE_LIMIT)?,
        out: guarded(D, HOST_ORACLE_LIMIT)?,
        core: guarded(D, HOST_ORACLE_LIMIT)?,
    };
    for (rank, &e) in ids.iter().enumerate() {
        for tree in [true, false] {
            let g = project(w.gate, e, x, tree, cancel)?;
            let u = project(w.up, e, x, tree, cancel)?;
            let mut a = guarded(FF, HOST_ORACLE_LIMIT)?;
            for j in 0..FF {
                a[j] = kuno_text::ops::silu(g[j]) * u[j];
            }
            let d = project(w.down, e, &a, tree, cancel)?;
            if tree {
                o.act[rank * FF..][..FF].copy_from_slice(&a);
                o.part[rank * D..][..D].copy_from_slice(&d);
            }
            for j in 0..D {
                if tree {
                    o.out[j] += p[rank] * d[j];
                } else {
                    o.core[j] += p[rank] * d[j];
                }
            }
        }
    }
    Ok(o)
}
fn original(w: RoutedWeights<'_>, dtype: TensorType) -> Result<()> {
    for (t, shape, kind) in [
        (w.gate, [D as u64, FF as u64, 512], TensorType::Q4K),
        (w.up, [D as u64, FF as u64, 512], TensorType::Q4K),
        (w.down, [FF as u64, D as u64, 512], dtype),
    ] {
        if t.info().dimensions() != shape || t.info().dtype() != kind {
            return Err(fault());
        }
    }
    if ExpertBanks::allocation(w, 16).map_err(|_| fault())? as u64 != BANK_BYTES / 2 {
        return Err(fault());
    }
    Ok(())
}
fn sources(w: RoutedWeights<'_>, n: usize) -> Result<Zeroizing<Vec<u64>>> {
    if n > N {
        return Err(fault());
    }
    let mut v = guarded(n * 3, HOST_BRIDGE_LIMIT)?;
    for i in 0..n {
        for (j, t) in [w.gate, w.up, w.down].iter().enumerate() {
            v[i * 3 + j] = t
                .info()
                .row_bytes()
                .checked_mul(t.info().dimensions()[1])
                .ok_or_else(fault)?;
        }
    }
    Ok(v)
}
fn plan(b: &mut ExpertBanks, ids: &[usize]) -> Result<(Plan, usize)> {
    let mut staged = StagedSlots::begin(&b.slots).map_err(|_| fault())?;
    let p = staged.plan(&b.slots, ids).map_err(|_| fault())?;
    let bytes = b
        .slots
        .original_metadata_bytes()
        .map_err(|_| fault())?
        .checked_add(
            staged
                .retained_native_metadata_bytes(&p)
                .map_err(|_| fault())?,
        )
        .ok_or_else(fault)?;
    if bytes > HOST_BRIDGE_LIMIT / 4 {
        return Err(fault());
    }
    staged
        .validate_pending(&b.slots, &p, ids)
        .map_err(|_| fault())?;
    Ok((p, bytes))
}
fn fill(
    c: &ComputeClient<WgpuRuntime>,
    l: &mut MainAdmission,
    b: &ExpertBanks,
    w: RoutedWeights<'_>,
    p: &Plan,
    cancel: &AtomicBool,
) -> Result<u64> {
    observe(l, c)?;
    let n = l
        .precharge(&sources(w, p.uploads.len())?)
        .map_err(|_| fault())?;
    for &(e, s) in &p.uploads {
        for (bank, t) in [(b.gate(), w.gate), (b.up(), w.up), (b.down(), w.down)] {
            check(cancel)?;
            expert_banks::fill(c, bank, t, e, s, cancel).map_err(|_| fault())?;
        }
    }
    observe(l, c)?;
    Ok(n)
}
// No factoring or two-sum shortcut: exactly one partial per original rank.
// CubeCL0.10 portable finite predicate uses x!=x and finite magnitude.
#[allow(clippy::eq_op)]
#[cube(launch, launch_unchecked)]
fn rank_merge(
    gpu: &Array<f32>,
    cpu: &Array<f32>,
    map: &Array<u32>,
    prob: &Array<f32>,
    out: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    hidden: usize,
    gpu_count: usize,
    cpu_count: usize,
) {
    let j = ABSOLUTE_POS_X as usize;
    if j < hidden {
        let mut sum = 0.0f32;
        let mut valid = true;
        for rank in 0..10 {
            let owner = map[rank * 2];
            let index = map[rank * 2 + 1] as usize;
            let mut part = 0.0f32;
            if owner == 0 {
                if index < gpu_count {
                    part = gpu[index * hidden + j];
                } else {
                    valid = false;
                }
            } else if owner == 1 {
                if index < cpu_count {
                    part = cpu[index * hidden + j];
                } else {
                    valid = false;
                }
            } else {
                valid = false;
            }
            let p = prob[rank];
            if part != part || part.abs() > f32::MAX {
                valid = false;
            }
            if p != p || p.abs() > f32::MAX {
                valid = false;
            }
            sum += p * part;
        }
        if sum != sum || sum.abs() > f32::MAX {
            valid = false;
        }
        if valid {
            out[j] = sum;
        } else {
            out[j] = 0.0;
            fault[0].fetch_or(1);
        }
    }
}
// Own thirteen planned pages; unchanged ResidentMoe supplies the other five,
// and its output is this reference output page. No writable subview binding.
struct Buffers {
    ids: [BufferId; 13],
}
const X: usize = 0;
const P: usize = 1;
const Y: usize = 2;
const FAULT: usize = 3;
const MAP: usize = 4;
const CPU: usize = 5;
const SLOTS: usize = 6;
const ACT: usize = 7;
const PART: usize = 8;
const FLAGS: usize = 9;
const REF: usize = 10;
const RAWG: usize = 11;
const RAWU: usize = 12;
const WORDS: [usize; 13] = [
    D,
    N,
    D,
    1,
    N * 2,
    N * D,
    N,
    N * FF,
    N * D,
    STATUS,
    D,
    FF + 17,
    FF + 17,
];
impl Buffers {
    fn allocate(a: &mut PrivateArena) -> Result<Self> {
        let first = a.allocate_zeroed(WORDS[0]).map_err(|_| fault())?;
        let mut ids = [first; 13];
        for i in 1..13 {
            ids[i] = a.allocate_zeroed(WORDS[i]).map_err(|_| fault())?;
        }
        Ok(Self { ids })
    }
    fn h(&self, a: &PrivateArena, i: usize) -> Result<cubecl::server::Handle> {
        let id = self.ids[i];
        let b = a.layout(id).map_err(|_| fault())?;
        if b.logical_words != WORDS[i] {
            return Err(fault());
        }
        Ok(a.handle(id).map_err(|_| fault())?.clone())
    }
}
fn extent(offset: u64, size: u64, backing: u64, logical: u64, page: u64) -> Result<()> {
    if logical == 0 || page == 0 || offset != 0 || size != logical || backing != page {
        return Err(fault());
    }
    Ok(())
}
/// Classifies only a rejected fixed inventory extent, without retaining any
/// address/size. The full-span invariant in extent() remains unchanged.
fn physical_extent_stage(index: usize, offset: u64, size: u64, logical: u64) -> NumericStage {
    if index < 13 {
        NumericStage::PhysicalPrivateExtent
    } else if index < 18 {
        NumericStage::PhysicalReferenceExtent
    } else if offset != 0 {
        NumericStage::PhysicalBankOffset
    } else if logical == 0 || size != logical {
        NumericStage::PhysicalBankSize
    } else {
        NumericStage::PhysicalBankBacking
    }
}
fn permit_health(
    actual: &Arc<AtomicBool>,
    supplied: &Arc<AtomicBool>,
    thread: std::thread::ThreadId,
) -> Result<()> {
    if !Arc::ptr_eq(actual, supplied)
        || actual.load(Ordering::Acquire)
        || supplied.load(Ordering::Acquire)
        || thread != std::thread::current().id()
    {
        return Err(fault());
    }
    Ok(())
}
fn sample_ticket(samples: &std::sync::atomic::AtomicUsize) -> Result<()> {
    samples
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            if n < 16 { Some(n + 1) } else { None }
        })
        .map(|_| ())
        .map_err(|_| fault())
}
fn physical(
    a: &PrivateArena,
    b: &Buffers,
    banks: &ExpertBanks,
    other: &ExpertBanks,
    reference: &ResidentMoe,
    stage: &mut NumericStage,
) -> Result<()> {
    *stage = NumericStage::PhysicalClient;
    let c = a.client().map_err(|_| fault())?;
    banks.check_client(c)?;
    other.check_client(c)?;
    let stream = a.handle(b.ids[0]).map_err(|_| fault())?.stream;
    *stage = NumericStage::PhysicalMetadata;
    let mut handles = schedule::Metadata::new(24)?;
    for (i, &id) in b.ids.iter().enumerate() {
        *stage = NumericStage::PhysicalPrivateOwner;
        if b.ids[..i].contains(&id) {
            return Err(fault());
        }
        let h = b.h(a, i)?;
        let layout = a.layout(id).map_err(|_| fault())?;
        if h.stream != stream || h.size_in_used() != layout.owned_bytes as u64 {
            return Err(fault());
        }
        handles.push((h, layout.owned_bytes as u64, layout.owned_bytes as u64))?;
    }
    for (id, n) in reference.fixed_public_probe_owners() {
        *stage = NumericStage::PhysicalReferenceOwner;
        let l = a.layout(id).map_err(|_| fault())?;
        let h = a.handle(id).map_err(|_| fault())?.clone();
        if l.logical_words != n
            || h.stream != stream
            || h.size_in_used() != l.owned_bytes as u64
            || b.ids.contains(&id)
        {
            return Err(fault());
        }
        handles.push((h, l.owned_bytes as u64, l.owned_bytes as u64))?;
    }
    for banks in [banks, other] {
        for bank in [banks.gate(), banks.up(), banks.down()] {
            *stage = NumericStage::PhysicalBankOwner;
            let segment = bank.single()?;
            let n = (segment.words() * 4) as u64;
            let h = segment.handle().clone();
            if h.stream != stream || h.size_in_used() != n {
                return Err(fault());
            }
            handles.push((h, n, main_admission::page_bytes(n).map_err(|_| fault())?))?;
        }
    }
    *stage = NumericStage::PhysicalMetadata;
    if handles.len() != 24 {
        return Err(fault());
    }
    let mut resources = schedule::Metadata::new(24)?;
    for i in 0..handles.len() {
        *stage = NumericStage::PhysicalResource;
        resources.push(
            c.get_resource(handles.get(i)?.0.clone())
                .map_err(|_| fault())?,
        )?;
    }
    *stage = NumericStage::PhysicalMetadata;
    if handles
        .metadata_bytes()?
        .checked_add(resources.metadata_bytes()?)
        .is_none_or(|n| n > 32768)
    {
        return Err(fault());
    }
    for i in 0..resources.len() {
        *stage = NumericStage::PhysicalResource;
        let r = resources.get(i)?.resource();
        let h = handles.get(i)?;
        if let Err(error) = extent(r.offset, r.size, r.buffer.size(), h.1, h.2) {
            *stage = physical_extent_stage(i, r.offset, r.size, h.1);
            return Err(error);
        }
        for j in 0..i {
            if resources.get(j)?.resource().buffer == r.buffer {
                *stage = NumericStage::PhysicalAlias;
                return Err(fault());
            }
        }
    }

    Ok(())
}
struct Values {
    act: Zeroizing<Vec<f32>>,
    part: Zeroizing<Vec<f32>>,
    out: Zeroizing<Vec<f32>>,
}
#[allow(clippy::too_many_arguments)]
fn reference(
    g: BoundGraph<'_, '_>,
    a: &PrivateArena,
    l: &mut MainAdmission,
    retired: &AtomicBool,
    b: &mut ExpertBanks,
    m: &mut ResidentMoe,
    buffers: &Buffers,
    ids: &[usize; N],
    prob: &[f32; N],
    cancel: &AtomicBool,
) -> Result<(Values, usize)> {
    let (p, metadata) = plan(b, ids)?;
    fill(
        a.client().map_err(|_| fault())?,
        l,
        b,
        m.weights(g)?,
        &p,
        cancel,
    )?;
    let pending = m.enqueue(
        g,
        a,
        b,
        buffers.ids[X],
        buffers.ids[REF],
        &p.slots,
        prob,
        cancel,
        true,
        false,
    )?;
    let s = pending.status();
    let flags = read(a, s.buffer)?;
    if flags.len() != s.words || flags.iter().any(|&v| v != 1) {
        return Err(fault());
    }
    let [(act, _), (part, _)] = m.fixed_public_probe_outputs();
    let v = Values {
        act: floats(&read(a, act)?)?,
        part: floats(&read(a, part)?)?,
        out: floats(&read(a, buffers.ids[REF])?)?,
    };
    fence(a, l, retired)?;
    check(cancel)?;
    b.slots.commit(p).map_err(|_| fault())?;
    m.accept_after_whole_token_acceptance(pending)?;
    Ok((v, metadata))
}
#[allow(clippy::too_many_arguments)]
fn candidate_gpu(
    a: &PrivateArena,
    b: &Buffers,
    banks: &ExpertBanks,
    w: RoutedWeights<'_>,
    slots: &[usize],
    l: &mut MainAdmission,
    retired: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<Values> {
    check(cancel)?;
    let c = a.client().map_err(|_| fault())?;
    let mut s = Zeroizing::new([0u32; N]);
    for (i, &v) in slots.iter().enumerate() {
        if v >= 16 || slots[..i].contains(&v) {
            return Err(fault());
        }
        s[i] = v as u32;
    }
    upload(a, b.ids[SLOTS], &*s)?;
    if !slots.is_empty() {
        quant_matvec::moe_block::moe_gate_up::launch::<WgpuRuntime>(
            c,
            CubeCount::Static((N * FF) as u32, 1, 1),
            CubeDim::new_1d(64),
            unsafe {
                ArrayArg::from_raw_parts(
                    banks.gate().single()?.handle().clone(),
                    banks.gate().words(),
                )
            },
            unsafe {
                ArrayArg::from_raw_parts(banks.up().single()?.handle().clone(), banks.up().words())
            },
            unsafe { ArrayArg::from_raw_parts(b.h(a, X)?, D) },
            unsafe { ArrayArg::from_raw_parts(b.h(a, SLOTS)?, N) },
            unsafe { ArrayArg::from_raw_parts(b.h(a, ACT)?, N * FF) },
            slots.len(),
            D,
            FF,
            w.gate.info().dtype().gguf_id(),
            w.up.info().dtype().gguf_id(),
        );
        quant_matvec::moe::moe_down::launch::<WgpuRuntime>(
            c,
            CubeCount::Static((N * D) as u32, 1, 1),
            CubeDim::new_1d(64),
            unsafe {
                ArrayArg::from_raw_parts(
                    banks.down().single()?.handle().clone(),
                    banks.down().words(),
                )
            },
            unsafe { ArrayArg::from_raw_parts(b.h(a, ACT)?, N * FF) },
            unsafe { ArrayArg::from_raw_parts(b.h(a, SLOTS)?, N) },
            unsafe { ArrayArg::from_raw_parts(b.h(a, PART)?, N * D) },
            slots.len(),
            D,
            FF,
            w.down.info().dtype().gguf_id(),
        );
    }
    let v = Values {
        act: floats(&read(a, b.ids[ACT])?)?,
        part: floats(&read(a, b.ids[PART])?)?,
        out: guarded(D, HOST_ORACLE_LIMIT)?,
    };
    for tail in [&v.act[slots.len() * FF..], &v.part[slots.len() * D..]] {
        if tail.iter().any(|v| v.to_bits() != 0) {
            return Err(fault());
        }
    }
    fence(a, l, retired)?;
    check(cancel)?;
    Ok(v)
}
fn merge(
    a: &PrivateArena,
    b: &Buffers,
    nc: usize,
    ng: usize,
    l: &mut MainAdmission,
    retired: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<f32>>> {
    check(cancel)?;
    let c = a.client().map_err(|_| fault())?;
    rank_merge::launch::<WgpuRuntime>(
        c,
        CubeCount::Static(D.div_ceil(64) as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(b.h(a, PART)?, N * D) },
        unsafe { ArrayArg::from_raw_parts(b.h(a, CPU)?, N * D) },
        unsafe { ArrayArg::from_raw_parts(b.h(a, MAP)?, N * 2) },
        unsafe { ArrayArg::from_raw_parts(b.h(a, P)?, N) },
        unsafe { ArrayArg::from_raw_parts(b.h(a, Y)?, D) },
        unsafe { ArrayArg::from_raw_parts(b.h(a, FAULT)?, 1) },
        D,
        ng,
        nc,
    );
    let mut offset = 0;
    for (i, n) in [(X, D), (ACT, N * FF), (PART, N * D), (Y, D)] {
        resident_moe::finite_chunks::launch::<WgpuRuntime>(
            c,
            CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(b.h(a, i)?, n) },
            unsafe { ArrayArg::from_raw_parts(b.h(a, FLAGS)?, STATUS) },
            offset,
        );
        offset += n.div_ceil(64);
    }
    if offset != STATUS {
        return Err(fault());
    }
    let flags = read(a, b.ids[FLAGS])?;
    let error = read(a, b.ids[FAULT])?;
    let y = floats(&read(a, b.ids[Y])?)?;
    fence(a, l, retired)?;
    check(cancel)?;
    if flags.iter().any(|&v| v != 1) || error.as_slice() != [0] {
        return Err(fault());
    }
    Ok(y)
}
fn immutable(a: &PrivateArena, b: &Buffers, x: &[f32], p: &[f32], s: &Split) -> Result<()> {
    for (id, values) in [(b.ids[X], x), (b.ids[P], p)] {
        let w = read(a, id)?;
        if w.len() != values.len() || w.iter().zip(values).any(|(&a, &b)| a != b.to_bits()) {
            return Err(fault());
        }
    }
    if read(a, b.ids[MAP])?.as_slice() != s.map() {
        return Err(fault());
    }
    Ok(())
}
// Keep each original owner and completion guard explicit in this fixture-only seam.
#[allow(clippy::too_many_arguments)]
fn raw_projections(
    a: &PrivateArena,
    b: &Buffers,
    banks: &mut ExpertBanks,
    w: RoutedWeights<'_>,
    ids: &[usize; N],
    x: &[f32],
    op: &ExpertOperator,
    reader: &kuno_text::gguf::PositionalReader<'_>,
    oracle: &Oracle,
    l: &mut MainAdmission,
    retired: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<()> {
    let (p, _) = plan(banks, ids)?;
    if !p.uploads.is_empty() {
        return Err(fault());
    }
    let c = a.client().map_err(|_| fault())?;
    for (rank, &e) in ids.iter().enumerate() {
        for (t, bank, out) in [(w.gate, banks.gate(), RAWG), (w.up, banks.up(), RAWU)] {
            check(cancel)?;
            let cpu = op.project(&PackedProjection::new(reader, t)?, e, x, cancel)?;
            let expected = project(t, e, x, true, cancel)?;
            bits(cpu.values(), &expected)?;
            let slot = p.slots[rank];
            let bytes = bank.expert_bytes();
            let total = bank.words() * 4;
            let begin = slot.checked_mul(bytes).ok_or_else(fault)?;
            let end = total.checked_sub(begin + bytes).ok_or_else(fault)?;
            if !begin.is_multiple_of(512) || !end.is_multiple_of(512) {
                return Err(fault());
            }
            let h = bank
                .single()?
                .handle()
                .clone()
                .offset_start(begin as u64)
                .offset_end(end as u64);
            quant_matvec::cooperative::cooperative_matvec::launch::<WgpuRuntime>(
                c,
                CubeCount::Static(FF as u32, 1, 1),
                CubeDim::new_1d(64),
                unsafe { ArrayArg::from_raw_parts(h, bytes / 4) },
                unsafe { ArrayArg::from_raw_parts(b.h(a, X)?, D) },
                unsafe { ArrayArg::from_raw_parts(b.h(a, out)?, FF + 17) },
                FF,
                D,
                t.info().dtype().gguf_id(),
                17,
            );
            let got = read(a, b.ids[out])?;
            if got[..17].iter().any(|&v| v != 0) {
                return Err(fault());
            }
            compare(&floats(&got[17..])?, &expected, 1e-4, 3e-4)?;
            fence(a, l, retired)?;
        }
        let down = op.project(
            &PackedProjection::new(reader, w.down)?,
            e,
            &oracle.act[rank * FF..][..FF],
            cancel,
        )?;
        bits(down.values(), &oracle.part[rank * D..][..D])?;
    }
    Ok(())
}
// Per-run guarded snapshots only. Calibration does not execute these reads
// before its completed consumers; numeric mode keeps its original ordering.
struct ReferenceWindow<T>(Option<T>);
impl<T> ReferenceWindow<T> {
    fn before(calibrating: bool, reference: impl FnOnce() -> Result<T>) -> Result<Self> {
        if calibrating {
            Ok(Self(None))
        } else {
            Ok(Self(Some(reference()?)))
        }
    }
    fn after(self, reference: impl FnOnce() -> Result<T>) -> Result<T> {
        match self.0 {
            Some(value) => Ok(value),
            None => reference(),
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn verified_references(
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    ledger: &mut MainAdmission,
    retired: &AtomicBool,
    flat: &mut ExpertBanks,
    reference_moe: &mut ResidentMoe,
    buffers: &Buffers,
    weights: RoutedWeights<'_>,
    ids: &[usize; N],
    x: &[f32],
    prob: &[f32; N],
    op: &ExpertOperator,
    portable: &ExpertOperator,
    experts: &PackedExperts<'_, '_>,
    reader: &kuno_text::gguf::PositionalReader<'_>,
    cancel: &AtomicBool,
) -> Result<(Values, Oracle, usize)> {
    let (full, metadata) = reference(
        graph,
        arena,
        ledger,
        retired,
        flat,
        reference_moe,
        buffers,
        ids,
        prob,
        cancel,
    )?;
    let expected = oracle(weights, ids, x, prob, cancel)?;
    let all_cpu = portable.execute(experts, ids, x, cancel)?;
    bits(all_cpu.active(), &expected.part)?;
    drop(all_cpu);
    raw_projections(
        arena, buffers, flat, weights, ids, x, op, reader, &expected, ledger, retired, cancel,
    )?;
    Ok((full, expected, metadata))
}
fn snapshot_bytes(v: &[&Vec<f32>]) -> Result<usize> {
    let bytes = v.iter().try_fold(0usize, |n, v| {
        v.capacity()
            .checked_mul(4)
            .and_then(|b| n.checked_add(b))
            .ok_or_else(fault)
    })?;
    if bytes > HOST_ORACLE_LIMIT {
        return Err(fault());
    }
    Ok(bytes)
}
fn public_sentinels(
    c: &ComputeClient<WgpuRuntime>,
    banks: &ExpertBanks,
) -> Result<Vec<(cubecl::server::Handle, [u8; 16])>> {
    let mut out = Vec::with_capacity(6);
    for bank in [banks.gate(), banks.up(), banks.down()] {
        let n = bank.words() * 4;
        for first in [0, n - 16] {
            let h = bank
                .single()?
                .handle()
                .clone()
                .offset_start(first as u64)
                .offset_end((n - first - 16) as u64);
            let data = c.read_one(h.clone()).map_err(|_| fault())?;
            let sample: [u8; 16] = data[..].try_into().map_err(|_| fault())?;
            drop(data);
            out.push((h, sample));
        }
    }
    Ok(out)
}
fn allocate_bank(
    c: &ComputeClient<WgpuRuntime>,
    l: &mut MainAdmission,
    w: RoutedWeights<'_>,
) -> Result<ExpertBanks> {
    observe(l, c)?;
    l.precharge(&ExpertBanks::allocation_requests(w, 16)?)
        .map_err(|_| fault())?;
    let b = ExpertBanks::new_full_pages_for_cpu_gpu_lab(c, w)?;
    observe(l, c)?;
    Ok(b)
}
fn host_preflight() -> Result<()> {
    // Exact-capacity guards forbid hidden Vec over-reserve before filling.
    // Oracle retained eleven F32 snapshots + maximum owning-u32/f32 read pair,
    // two raw projection outputs and one decoded row remain below2MiB.
    let oracle = (2 * (N * FF + N * D + D) + N * FF + N * D + 3 * D) * 4
        + 2 * N * D * 4
        + 2 * FF * 4
        + D * 4;
    // Two simultaneous measured StagedSlots/Plan inventories are each capped
    // at256KiB; both committed states, full moved bridge+word encoding, input,
    // handle/resource headers and one scoped orchestrator fit the spare1MiB.
    // Separate4KiB reservation covers the guarded scoped handle/control header.
    let bridge =
        2 * (HOST_BRIDGE_LIMIT / 4) + 2 * 32 * 1024 + 131072 + N * D * 4 + D * 4 + 32768 + 4096;
    if oracle > HOST_ORACLE_LIMIT
        || bridge > HOST_BRIDGE_LIMIT
        || kuno_text::cpu_expert::MAX_PRIVATE_HOST_BYTES + HOST_ORACLE_LIMIT + HOST_BRIDGE_LIMIT
            > HOST_PRIVATE_LIMIT
    {
        return Err(fault());
    }
    Ok(())
}
struct Terminal<'a> {
    cancel: &'a AtomicBool,
    retired: &'a AtomicBool,
    armed: bool,
}
impl Drop for Terminal<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.cancel.store(true, Ordering::Release);
            self.retired.store(true, Ordering::Release);
        }
    }
}
fn startup(baseline: u64) -> Result<u64> {
    main_admission::check_capacity(
        PHYSICAL_LIMIT,
        &[baseline, BANK_BYTES, PRIVATE_BYTES, SOURCE_BYTES, 64 * MIB],
    )
    .map_err(|_| fault())
}
fn pages() -> Result<Vec<u64>> {
    let mut p = WORDS
        .iter()
        .map(|&n| main_admission::page_bytes((n * 4) as u64).map_err(|_| fault()))
        .collect::<Result<Vec<_>>>()?;
    p.extend([32768, 32768, 32768, 131072, 32768]);
    if p.len() != 18 || p.iter().sum::<u64>() != PRIVATE_BYTES {
        return Err(fault());
    }
    Ok(p)
}
/// This CALL reads only selected original tensor rows/components at execution.
/// Parent owns physical execution/CLI registration. Requires its required main
/// allocator and already-configured hard64MiB transient policy on this stream.
/// At most ONE explicit point; shared sealed permit bounds calibration to16.
/// It is a local one-card gate; parent must separately approve both-card parity.
/// Timings include full owning GPU verification/fences, never enqueue-only work.
/// Calibration defers independent references until after the timed consumers;
/// mandatory priming and prior samples remain explicit cache conditioning.
/// These are controlled operator costs, not cold-read or whole-model costs.
#[allow(clippy::too_many_arguments)]
pub fn run<'graph, 'm>(
    client: &ComputeClient<WgpuRuntime>,
    file: &'m GgufFile,
    graph: BoundGraph<'graph, 'm>,
    device_limits: &wgpu::Limits,
    retired: Arc<AtomicBool>,
    config: Config,
    purpose: Purpose<'_, 'm>,
    cancel: &AtomicBool,
) -> Result<(Report, NumericPermit<'m>)> {
    let numeric = matches!(&purpose, Purpose::Numeric);
    let mut stage = NumericStage::Config;
    let result = run_inner(
        client,
        file,
        graph,
        device_limits,
        retired,
        config,
        purpose,
        cancel,
        &mut stage,
    );
    classified_numeric_result(numeric, stage, result)
}
#[allow(clippy::too_many_arguments)]
fn run_inner<'graph, 'm>(
    client: &ComputeClient<WgpuRuntime>,
    file: &'m GgufFile,
    graph: BoundGraph<'graph, 'm>,
    device_limits: &wgpu::Limits,
    retired: Arc<AtomicBool>,
    config: Config,
    purpose: Purpose<'_, 'm>,
    cancel: &AtomicBool,
    stage: &mut NumericStage,
) -> Result<(Report, NumericPermit<'m>)> {
    config.validate()?;
    *stage = NumericStage::Limits;
    limits(device_limits)?;
    *stage = NumericStage::Control;
    check(cancel)?;
    if retired.load(Ordering::Acquire) {
        return Err(fault());
    }
    let (timed, samples) = match purpose {
        Purpose::Numeric => (false, Arc::new(std::sync::atomic::AtomicUsize::new(0))),
        Purpose::Calibration(p) => {
            permit_health(&p.retired, &retired, p.thread)?;
            if !std::ptr::eq(p.file, file)
                || !std::ptr::eq(p.client.info(), client.info())
                || p.binding != graph.binding_id()
                || !same_config(p.config, config)
            {
                return Err(fault());
            }
            sample_ticket(&p.samples)?;
            (true, p.samples.clone())
        }
    };
    // Hardware setup and CPU affinity proof are deliberately before allocations.
    *stage = NumericStage::Workers;
    let op = ExpertOperator::new(WorkerPlan::physical(config.workers)?, config.backend)?;
    let portable = ExpertOperator::new(WorkerPlan::portable(1)?, Backend::Portable)?;
    *stage = NumericStage::Geometry;
    let reader = file.positional_reader()?;
    graph.verify_unchanged()?;
    let geometry = graph.geometry();
    if geometry.layers != 48
        || geometry.residual.width != D
        || geometry.experts != 512
        || geometry.experts_used != N
        || geometry.expert_hidden != FF
    {
        return Err(fault());
    }
    let layer = (0..48)
        .find(|&layer| {
            file.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))
                .is_some_and(|t| t.info().dtype() == config.down)
        })
        .ok_or_else(fault)?;
    let weights = RoutedWeights {
        gate: graph.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
        up: graph.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
        down: graph.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
    };
    if [weights.gate, weights.up, weights.down]
        .iter()
        .any(|t| !t.belongs_to(file))
    {
        return Err(fault());
    }
    *stage = NumericStage::OriginalBanks;
    original(weights, config.down)?;
    let experts = PackedExperts::new(&reader, weights.gate, weights.up, weights.down)?;
    // Existing unrelated pending IO cannot be carried into this envelope.
    let mut terminal = Terminal {
        cancel,
        retired: &retired,
        armed: true,
    };
    *stage = NumericStage::StartupIo;
    cubecl::future::block_on(client.sync()).map_err(|_| fault())?;
    let initial =
        cubecl::future::block_on(client.sanitize_transient_io_audited()).map_err(|_| fault())?;
    if initial.inspected_bytes
        != initial
            .upload_bytes
            .checked_add(initial.uniform_bytes)
            .and_then(|v| v.checked_add(initial.readback_bytes))
            .ok_or_else(fault)?
    {
        retired.store(true, Ordering::Release);
        return Err(fault());
    }
    // Prior successful runner dropped its banks only after completed audit;
    // cleanup now releases those free MAIN pages before a fresh aggregate.
    client.memory_cleanup();
    *stage = NumericStage::StartupCapacity;
    let baseline = sample(client)?;
    startup(baseline)?;
    *stage = NumericStage::HostBudget;
    host_preflight()?;
    *stage = NumericStage::Allocation;
    let mut ledger =
        MainAdmission::new(PHYSICAL_LIMIT - 64 * MIB, baseline).map_err(|_| fault())?;
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), MIB as usize, retired.clone())
            .map_err(|_| fault())?;
    arena
        .enable_public_owned_zero_audit()
        .map_err(|_| fault())?;
    let mut candidate_owner = None;
    let mut reference_owner = None;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Report> {
        let banks = candidate_owner.insert(allocate_bank(client, &mut ledger, weights)?);
        let flat = reference_owner.insert(allocate_bank(client, &mut ledger, weights)?);
        ledger.precharge(&pages()?).map_err(|_| fault())?;
        let buffers = Buffers::allocate(&mut arena)?;
        let mut reference_moe = ResidentMoe::allocate(graph, layer, &mut arena)?;
        if arena.buffer_count() != 18 || arena.charged_bytes() as u64 != PRIVATE_BYTES {
            return Err(fault());
        }
        observe(&mut ledger, client)?;
        *stage = NumericStage::PhysicalOwners;
        physical(&arena, &buffers, banks, flat, &reference_moe, stage)?;
        *stage = NumericStage::FixtureUpload;
        let ids: Zeroizing<[usize; N]> =
            Zeroizing::new(std::array::from_fn(|r| config.group * N + r));
        let split = Split::new(&ids, config.misses, config.cpu_misses)?;
        let mut x = guarded(D, HOST_BRIDGE_LIMIT)?;
        for (i, v) in x.iter_mut().enumerate() {
            *v = if config.vector == 0 {
                ((i * 17 + i / 64 * 3) % 31) as f32 / 16.0 - 15.0 / 16.0
            } else {
                match i % 7 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => ((i * 11) % 23) as f32 / 16.0 - 11.0 / 16.0,
                }
            };
        }
        let prob = &PROBS[config.vector];
        upload_f32(&arena, buffers.ids[X], &x)?;
        upload_f32(&arena, buffers.ids[P], prob)?;
        upload(&arena, buffers.ids[MAP], split.map())?;
        // Prime precisely the selected hits, plus unrelated PUBLIC filler IDs. The
        // ten-hit source state is then measured through a real protected plan.
        let primed: Zeroizing<[usize; N]> = Zeroizing::new(std::array::from_fn(|r| {
            if r < N - config.misses {
                ids[r]
            } else {
                480 + r
            }
        }));
        *stage = NumericStage::Priming;
        let (_, prime_meta) = reference(
            graph,
            &arena,
            &mut ledger,
            &retired,
            banks,
            &mut reference_moe,
            &buffers,
            &primed,
            prob,
            cancel,
        )?;
        *stage = NumericStage::Reference;
        let references = ReferenceWindow::before(timed, || {
            verified_references(
                graph,
                &arena,
                &mut ledger,
                &retired,
                flat,
                &mut reference_moe,
                &buffers,
                weights,
                &ids,
                &x,
                prob,
                &op,
                &portable,
                &experts,
                &reader,
                cancel,
            )
        })?;
        let mut metadata = prime_meta;
        *stage = NumericStage::CandidatePlan;
        let candidate_plan = if split.gpu().is_empty() {
            None
        } else {
            let (p, n) = plan(banks, split.gpu())?;
            metadata = metadata.max(n);
            if p.uploads.len() != config.misses - config.cpu_misses {
                return Err(fault());
            }
            Some(p)
        };
        // Declared all-selected M is verified separately, even at G==0.
        let (all_plan, n) = plan(banks, &*ids)?;
        metadata = metadata.max(n);
        if all_plan.uploads.len() != config.misses {
            return Err(fault());
        }
        drop(all_plan);
        let mut charge = 0;
        *stage = NumericStage::Candidate;
        let (cpu, gpu, times) = schedule::pair(
            config.mode,
            cancel,
            !split.cpu().is_empty(),
            || op.execute(&experts, split.cpu(), &x, cancel),
            || {
                if let Some(p) = &candidate_plan {
                    charge = fill(client, &mut ledger, banks, weights, p, cancel)?;
                    candidate_gpu(
                        &arena,
                        &buffers,
                        banks,
                        weights,
                        &p.slots,
                        &mut ledger,
                        &retired,
                        cancel,
                    )
                } else {
                    candidate_gpu(
                        &arena,
                        &buffers,
                        banks,
                        weights,
                        &[],
                        &mut ledger,
                        &retired,
                        cancel,
                    )
                }
            },
        )?;
        check(cancel)?;
        *stage = NumericStage::CpuPartials;
        if let Some(cpu) = &cpu {
            if cpu.ranks() != split.cpu().len()
                || cpu.hidden() != D
                || cpu.owned_bytes() > MIB as usize
                || cpu.full()[split.cpu().len() * D..]
                    .iter()
                    .any(|v| v.to_bits() != 0)
            {
                return Err(fault());
            }
            if cpu.active().iter().any(|v| !v.is_finite()) {
                return Err(fault());
            }
        }
        *stage = NumericStage::Merge;
        let bridge_start = std::time::Instant::now();
        if let Some(cpu) = &cpu {
            upload_f32(&arena, buffers.ids[CPU], cpu.full())?;
        }
        let y = merge(
            &arena,
            &buffers,
            split.cpu().len(),
            split.gpu().len(),
            &mut ledger,
            &retired,
            cancel,
        )?;
        let bridge = bridge_start.elapsed();
        // Calibration reaches here only after both consumers and the original-
        // rank bridge completed owning maps/fault checks and audited cleanup.
        // Required real priming and prior samples can still condition caches.
        *stage = NumericStage::Reference;
        let (expected_full, expected, reference_meta) = references.after(|| {
            verified_references(
                graph,
                &arena,
                &mut ledger,
                &retired,
                flat,
                &mut reference_moe,
                &buffers,
                weights,
                &ids,
                &x,
                prob,
                &op,
                &portable,
                &experts,
                &reader,
                cancel,
            )
        })?;
        metadata = metadata.max(reference_meta);
        *stage = NumericStage::NumericAcceptance;
        let mut tree_error = compare(&expected_full.act, &expected.act, 1e-4, 3e-4)?
            .max(compare(&expected_full.part, &expected.part, 1e-4, 3e-4)?)
            .max(compare(&expected_full.out, &expected.out, 1e-4, 3e-4)?);
        let mut core_error = compare(&expected_full.out, &expected.core, 1e-3, 3e-3)?;
        if let Some(cpu) = &cpu {
            for (r, &id) in split.cpu().iter().enumerate() {
                let rank = ids.iter().position(|&e| e == id).ok_or_else(fault)?;
                bits(&cpu.active()[r * D..][..D], &expected.part[rank * D..][..D])?;
            }
        }
        let mut legacy_error = compare(&y, &expected_full.out, 1e-4, 3e-4)?;
        tree_error = tree_error.max(compare(&y, &expected.out, 1e-4, 3e-4)?);
        core_error = core_error.max(compare(&y, &expected.core, 1e-3, 3e-3)?);
        for (r, &id) in split.gpu().iter().enumerate() {
            let rank = ids.iter().position(|&e| e == id).ok_or_else(fault)?;
            tree_error = tree_error
                .max(compare(
                    &gpu.act[r * FF..][..FF],
                    &expected.act[rank * FF..][..FF],
                    1e-4,
                    3e-4,
                )?)
                .max(compare(
                    &gpu.part[r * D..][..D],
                    &expected.part[rank * D..][..D],
                    1e-4,
                    3e-4,
                )?);
            legacy_error = legacy_error.max(compare(
                &gpu.part[r * D..][..D],
                &expected_full.part[rank * D..][..D],
                1e-4,
                3e-4,
            )?);
        }
        immutable(&arena, &buffers, &x, prob, &split)?;
        *stage = NumericStage::FinalAudit;
        let last = fence(&arena, &mut ledger, &retired)?;
        check(cancel)?;
        *stage = NumericStage::FileConsistency;
        reader.verify_file_unchanged()?;
        // Publication is LOCAL fixture metadata only, after ALL arithmetic/maps and
        // actual cleanup. No model token/cursor or whole-family right is invoked.
        *stage = NumericStage::Metadata;
        let snapshots = snapshot_bytes(&[
            &expected.act,
            &expected.part,
            &expected.out,
            &expected.core,
            &expected_full.act,
            &expected_full.part,
            &expected_full.out,
            &gpu.act,
            &gpu.part,
            &gpu.out,
            &y,
        ])?;
        *stage = NumericStage::SlotCommit;
        if let Some(p) = candidate_plan {
            banks.slots.commit(p).map_err(|_| fault())?;
        }
        *stage = NumericStage::Sentinels;
        let mut sentinels = public_sentinels(client, banks)?;
        sentinels.extend(public_sentinels(client, flat)?);
        fence(&arena, &mut ledger, &retired)?;
        *stage = NumericStage::Erasure;
        arena.close().map_err(|_| fault())?;
        *stage = NumericStage::Sentinels;
        for (h, expected) in sentinels {
            let actual = client.read_one(h).map_err(|_| fault())?;
            if actual[..] != expected {
                return Err(fault());
            }
            drop(actual);
        }
        *stage = NumericStage::FinalAudit;
        audit(client, &retired)?;
        *stage = NumericStage::Erasure;
        let zero = arena.owned_zero_audit_receipt().ok_or_else(fault)?;
        if zero.buffers != 18 || zero.bytes != PRIVATE_BYTES {
            return Err(fault());
        }
        Ok(Report {
            layer,
            down_dtype: config.down.gguf_id(),
            workers: config.workers,
            misses: config.misses,
            cpu_misses: config.cpu_misses,
            gpu_misses: config.misses - config.cpu_misses,
            max_tree_error: tree_error,
            max_legacy_error: legacy_error,
            max_core_error: core_error,
            private_pages: 18,
            private_bytes: PRIVATE_BYTES,
            bank_bytes: BANK_BYTES,
            source_precharge: charge,
            io_inspected_bytes: last.inspected_bytes,
            zero_buffers: zero.buffers,
            zero_bytes: zero.bytes,
            timing: timed.then_some(times),
            bridge_and_merge: timed.then_some(bridge),
            metadata_bytes: metadata,
            snapshot_bytes: snapshots,
        })
    }));
    match attempt {
        Ok(Ok(report)) if !retired.load(Ordering::Acquire) && !cancel.load(Ordering::Acquire) => {
            *stage = NumericStage::FileConsistency;
            file.verify_file_unchanged()?;
            check(cancel)?;
            if retired.load(Ordering::Acquire) {
                return Err(fault());
            }
            terminal.armed = false;
            Ok((
                report,
                NumericPermit {
                    client: client.clone(),
                    file,
                    config,
                    binding: graph.binding_id(),
                    samples,
                    retired: retired.clone(),
                    thread: std::thread::current().id(),
                },
            ))
        }
        _ => {
            cancel.store(true, Ordering::Release);
            ledger.poison();
            retired.store(true, Ordering::Release);
            let _ = arena.close();
            Err(fault())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_stage_classification_discards_original_errors_without_changing_calibration() {
        let result = classified_numeric_result::<()>(
            true,
            NumericStage::Merge,
            Err(Error::InvalidFormat("synthetic backend detail")),
        );
        assert!(matches!(
            result,
            Err(Error::InvalidFormat(
                "public CPU/GPU numeric merge rejected"
            ))
        ));
        let calibration = classified_numeric_result::<()>(
            false,
            NumericStage::Merge,
            Err(Error::LimitExceeded("synthetic original limit")),
        );
        assert!(matches!(
            calibration,
            Err(Error::LimitExceeded("synthetic original limit"))
        ));
        assert_eq!(
            classified_numeric_result(true, NumericStage::Erasure, Ok(17usize)).unwrap(),
            17
        );
    }
    #[test]
    fn numeric_real_preflight_failures_have_distinct_fixed_stages() {
        let invalid = Config {
            down: TensorType::Q5_0,
            workers: 3,
            backend: Backend::Portable,
            misses: 10,
            cpu_misses: 5,
            group: 0,
            vector: 0,
            mode: Mode::Concurrent,
        };
        let config = classified_numeric_result(true, NumericStage::Config, invalid.validate());
        assert!(matches!(
            config,
            Err(Error::InvalidFormat(
                "public CPU/GPU numeric config rejected"
            ))
        ));
        let unsupported = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 6,
            ..Default::default()
        };
        let device = classified_numeric_result(true, NumericStage::Limits, limits(&unsupported));
        assert!(matches!(
            device,
            Err(Error::InvalidFormat(
                "public CPU/GPU numeric limits rejected"
            ))
        ));
        let capacity =
            classified_numeric_result(true, NumericStage::StartupCapacity, startup(PHYSICAL_LIMIT));
        assert!(matches!(
            capacity,
            Err(Error::InvalidFormat(
                "public CPU/GPU numeric startup-capacity rejected"
            ))
        ));
    }
    #[test]
    fn calibration_defers_exactly_one_reference_and_retains_error_gates() {
        use std::cell::RefCell;
        for calibrating in [false, true] {
            let calls = RefCell::new(Vec::new());
            let window = ReferenceWindow::before(calibrating, || {
                calls.borrow_mut().push("reference");
                Ok(Zeroizing::new(vec![7u32]))
            })
            .unwrap();
            assert_eq!(calls.borrow().is_empty(), calibrating);
            calls.borrow_mut().push("completed-consumers");
            let reference = window
                .after(|| {
                    calls.borrow_mut().push("reference");
                    Ok(Zeroizing::new(vec![7u32]))
                })
                .unwrap();
            assert_eq!(&*reference, &[7]);
            assert_eq!(
                &*calls.borrow(),
                if calibrating {
                    &["completed-consumers", "reference"]
                } else {
                    &["reference", "completed-consumers"]
                }
            );
        }
        let window =
            ReferenceWindow::before(true, || -> Result<()> { panic!("early read") }).unwrap();
        assert!(window.after(|| Err(fault())).is_err());
        assert!(ReferenceWindow::<()>::before(false, || Err(fault())).is_err());
    }
    fn merge_reference(
        gpu: &[f32],
        cpu: &[f32],
        map: &[u32; 20],
        prob: &[f32; 10],
        d: usize,
        ng: usize,
        nc: usize,
    ) -> Result<Zeroizing<Vec<f32>>> {
        if gpu.len() != 10 * d || cpu.len() != 10 * d || ng + nc != 10 {
            return Err(fault());
        }
        let mut y = guarded::<f32>(d, HOST_ORACLE_LIMIT)?;
        for j in 0..d {
            for r in 0..10 {
                let i = map[2 * r + 1] as usize;
                let v = match map[2 * r] {
                    0 if i < ng => gpu[i * d + j],
                    1 if i < nc => cpu[i * d + j],
                    _ => return Err(fault()),
                };
                if !v.is_finite() || !prob[r].is_finite() {
                    return Err(fault());
                }
                y[j] += prob[r] * v;
            }
            if !y[j].is_finite() {
                return Err(fault());
            }
        }
        Ok(y)
    }
    #[test]
    fn original_rank_merge_includes_zero_probability_and_all_split_endpoints() {
        let ids = std::array::from_fn(|r| r);
        for m in 0..=10 {
            for k in 0..=m {
                let s = Split::new(&ids, m, k).unwrap();
                let mut cpu = vec![0.0; 30];
                let mut gpu = vec![0.0; 30];
                for (r, &id) in s.cpu().iter().enumerate() {
                    for j in 0..3 {
                        cpu[r * 3 + j] = (id as f32 - 5.0) * (j as f32 + 0.5);
                    }
                }
                for (r, &id) in s.gpu().iter().enumerate() {
                    for j in 0..3 {
                        gpu[r * 3 + j] = (id as f32 - 5.0) * (j as f32 + 0.5);
                    }
                }
                let p = &PROBS[1];
                let y = merge_reference(&gpu, &cpu, s.map(), p, 3, 10 - k, k).unwrap();
                let mut expected = [0.0; 3];
                for (r, &prob) in p.iter().enumerate() {
                    for (j, v) in expected.iter_mut().enumerate() {
                        *v += prob * ((r as f32 - 5.0) * (j as f32 + 0.5));
                    }
                }
                bits(&y, &expected).unwrap();
                let mut map = *s.map();
                map[1] = 10;
                assert!(merge_reference(&gpu, &cpu, &map, p, 3, 10 - k, k).is_err());
                let mut invalid_gpu = gpu.clone();
                let mut invalid_cpu = cpu.clone();
                if s.map()[0] == 0 {
                    invalid_gpu[s.map()[1] as usize * 3] = f32::NAN;
                } else {
                    invalid_cpu[s.map()[1] as usize * 3] = f32::NAN;
                }
                assert!(
                    merge_reference(&invalid_gpu, &invalid_cpu, s.map(), p, 3, 10 - k, k).is_err()
                );
            }
        }
    }
    #[test]
    fn exact_owning_length_tail_and_guard_capacity_reject() {
        let mut v = vec![0; 32768];
        validate_words(&v, 10, 32768).unwrap();
        for bad in [32767, 32769] {
            assert!(validate_words(&vec![0; bad], 10, 32768).is_err());
        }
        v[40] = 1;
        assert!(validate_words(&v, 10, 32768).is_err());
        assert!(validate_words(&[], usize::MAX, 0).is_err());
        assert!(guarded::<f32>(10, 39).is_err());
        assert!(guarded::<f32>(usize::MAX, HOST_PRIVATE_LIMIT).is_err());
    }
    #[test]
    fn physical_sum_sources_metadata_and_exact_startup_limit() {
        assert_eq!(pages().unwrap().len(), 18);
        assert_eq!(pages().unwrap().iter().sum::<u64>(), 884736);
        let threshold = PHYSICAL_LIMIT - (BANK_BYTES + PRIVATE_BYTES + SOURCE_BYTES + 64 * MIB);
        assert_eq!(threshold, 24_281_088);
        assert!(startup(threshold).is_ok());
        assert!(startup(threshold + 1).is_err());
        assert!(startup(u64::MAX).is_err());
        assert_eq!(
            main_admission::page_bytes(921600).unwrap() * 2
                + main_admission::page_bytes(1740800).unwrap(),
            4 * MIB
        );
        let bank = crate::bank_slots::BankSlots::new(16, 512).unwrap();
        assert!(
            bank.original_metadata_bytes().unwrap()
                + StagedSlots::bounded_native_metadata_bytes(16).unwrap()
                < HOST_BRIDGE_LIMIT / 4
        );
        let inventory = 262144 + 884736 + 65536 + 128 * 2048 + 884736 + 2 * 8192;
        assert_eq!(inventory, 2375680);
        assert_eq!(inventory * 2, 4751360);
        assert!(inventory * 2 < IO_COMPONENT_BYTES);
    }
    #[test]
    fn unchanged_tolerances_reject_nonfinite_and_do_not_accept_bit_drift() {
        assert!(bits(&[0.0], &[-0.0]).is_err());
        assert!(compare(&[f32::NAN], &[0.0], 1e-4, 3e-4).is_err());
        assert!(compare(&[0.0002], &[0.0], 1e-4, 3e-4).is_err());
        assert!(compare(&[], &[], 1e-4, 3e-4).is_err());
    }
    #[test]
    fn original_row_decoder_and_independent_64_schedule_public_formats() {
        for kind in [TensorType::Q4K, TensorType::Q5_0, TensorType::Q8_0] {
            let columns = 256;
            let mut packed = vec![0; kind.row_bytes(columns as u64).unwrap() as usize];
            let block = match kind {
                TensorType::Q4K => 144,
                TensorType::Q5_0 => 22,
                _ => 34,
            };
            for (b, chunk) in packed.chunks_mut(block).enumerate() {
                chunk[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                if kind == TensorType::Q4K {
                    chunk[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                    for (i, v) in chunk[4..].iter_mut().enumerate() {
                        *v = ((i * 17 + b * 23) % 256) as u8;
                    }
                } else {
                    for (i, v) in chunk[2..].iter_mut().enumerate() {
                        *v = ((i * 17 + b * 23) % 256) as u8;
                    }
                }
            }
            let mut decoded = vec![0.0; columns];
            kind.decode(&packed, &mut decoded).unwrap();
            let x = (0..columns)
                .map(|i| (i as f32 - 128.0) / 128.0)
                .collect::<Vec<_>>();
            let expected =
                crate::quant_matvec::cooperative::reference_tree_dot(&decoded, &x).unwrap();
            assert_eq!(dot64(&decoded, &x).unwrap().to_bits(), expected.to_bits());
        }
    }
    #[test]
    fn actual_extent_host_envelope_and_bounded_sample_ticket() {
        host_preflight().unwrap();
        extent(0, 18022400, 33554432, 18022400, 33554432).unwrap();
        extent(0, 32768, 32768, 32768, 32768).unwrap();
        for (offset, size, page) in [
            (4, 18022400, 33554432),
            (0, 33554432, 33554432),
            (0, 18022400, 18022400),
        ] {
            assert!(extent(offset, size, page, 18022400, 33554432).is_err());
        }
        let count = std::sync::atomic::AtomicUsize::new(0);
        for _ in 0..16 {
            sample_ticket(&count).unwrap();
        }
        assert!(sample_ticket(&count).is_err());
        assert_eq!(count.load(Ordering::Acquire), 16);
    }
    #[test]
    fn rejected_physical_extent_preserves_checks_and_identifies_fixed_owner_class() {
        let logical = 14_745_600;
        let page = 16_777_216;
        // A logical-sized original allocation is still rejected; diagnostics
        // must not make it equivalent to the required complete physical page.
        assert!(extent(0, logical, logical, logical, page).is_err());
        for index in 0..24 {
            assert_eq!(
                physical_extent_stage(index, 0, logical, logical),
                if index < 13 {
                    NumericStage::PhysicalPrivateExtent
                } else if index < 18 {
                    NumericStage::PhysicalReferenceExtent
                } else {
                    NumericStage::PhysicalBankBacking
                }
            );
        }
        assert!(extent(0, logical, page, logical, page).is_ok());
        assert!(extent(4, logical, page, logical, page).is_err());
        assert_eq!(
            physical_extent_stage(18, 4, logical, logical),
            NumericStage::PhysicalBankOffset
        );
        assert!(extent(0, page, page, logical, page).is_err());
        assert_eq!(
            physical_extent_stage(18, 0, page, logical),
            NumericStage::PhysicalBankSize
        );
    }
    #[test]
    fn sealed_gate_binds_actual_retirement_arc_and_worker_thread() {
        let actual = Arc::new(AtomicBool::new(false));
        let cloned = actual.clone();
        let forged = Arc::new(AtomicBool::new(false));
        let thread = std::thread::current().id();
        permit_health(&actual, &cloned, thread).unwrap();
        assert!(permit_health(&actual, &forged, thread).is_err());
        let other = std::thread::spawn(|| std::thread::current().id())
            .join()
            .unwrap();
        assert!(permit_health(&actual, &cloned, other).is_err());
        actual.store(true, Ordering::Release);
        assert!(permit_health(&actual, &cloned, thread).is_err());
        assert!(permit_health(&actual, &forged, thread).is_err());
    }
    #[test]
    fn production_terminal_guard_poison_on_early_error_and_unwind() {
        for panic in [false, true] {
            let c = AtomicBool::new(false);
            let r = AtomicBool::new(false);
            let _ = std::panic::catch_unwind(|| -> Result<()> {
                let _g = Terminal {
                    cancel: &c,
                    retired: &r,
                    armed: true,
                };
                if panic {
                    panic!("fixed public early failure");
                }
                Err(fault())
            });
            assert!(c.load(Ordering::Acquire));
            assert!(r.load(Ordering::Acquire));
        }
        let c = AtomicBool::new(false);
        let r = AtomicBool::new(false);
        {
            let mut g = Terminal {
                cancel: &c,
                retired: &r,
                armed: true,
            };
            g.armed = false;
        }
        assert!(!c.load(Ordering::Acquire));
        assert!(!r.load(Ordering::Acquire));
    }
    #[test]
    fn rank_merge_checked_unchecked_wgsl_validates_branch_loads_and_atomic_fault() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let a = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            b.scope.register_type::<u32>(u);
            b.scope.register_type::<usize>(u);
            b.scope.register_type::<f32>(f);
            b.scope
                .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
            let gpu = b.input_array(Type::new(f)).into();
            let cpu = b.input_array(Type::new(f)).into();
            let map = b.input_array(Type::new(u)).into();
            let prob = b.input_array(Type::new(f)).into();
            let out = b.output_array(Type::new(f)).into();
            let fault = b.output_array(Type::new(a)).into();
            let d = b.scalar(u).into();
            let ng = b.scalar(u).into();
            let nc = b.scalar(u).into();
            rank_merge::expand(&mut b.scope, gpu, cpu, map, prob, out, fault, d, ng, nc);
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("rank_merge"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    u,
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            assert!(source.contains("atomicOr("));
            assert!(source.contains("scalars_u32"));
            assert!(source.contains("buffer_0_global"));
            assert!(source.contains("buffer_1_global"));
            assert!(!source.contains("fma("));
            let fun = &module.entry_points[0].function;
            fn root(
                f: &wgpu::naga::Function,
                e: wgpu::naga::Handle<wgpu::naga::Expression>,
            ) -> Option<wgpu::naga::Handle<wgpu::naga::GlobalVariable>> {
                match f.expressions[e] {
                    wgpu::naga::Expression::GlobalVariable(g) => Some(g),
                    wgpu::naga::Expression::Access { base, .. }
                    | wgpu::naga::Expression::AccessIndex { base, .. } => root(f, base),
                    _ => None,
                }
            }
            fn loads(
                module: &wgpu::naga::Module,
                f: &wgpu::naga::Function,
                body: &wgpu::naga::Block,
                depth: usize,
                n: &mut [usize; 2],
            ) {
                for statement in body {
                    match statement {
                        wgpu::naga::Statement::Emit(range) => {
                            for e in range.clone() {
                                if let wgpu::naga::Expression::Load { pointer } = f.expressions[e]
                                    && let Some(g) = root(f, pointer)
                                    && let Some(binding) = &module.global_variables[g].binding
                                    && binding.binding < 2
                                {
                                    assert!(
                                        depth >= 3,
                                        "rank partial load escaped row/owner/index guards"
                                    );
                                    n[binding.binding as usize] += 1;
                                }
                            }
                        }
                        wgpu::naga::Statement::If { accept, reject, .. } => {
                            loads(module, f, accept, depth + 1, n);
                            loads(module, f, reject, depth + 1, n);
                        }
                        wgpu::naga::Statement::Loop {
                            body, continuing, ..
                        } => {
                            loads(module, f, body, depth, n);
                            loads(module, f, continuing, depth, n);
                        }
                        wgpu::naga::Statement::Block(body) => loads(module, f, body, depth, n),
                        _ => {}
                    }
                }
            }
            let mut reads = [0; 2];
            loads(&module, fun, &fun.body, 0, &mut reads);
            assert!(reads.iter().all(|&n| n > 0));
        }
    }
}
