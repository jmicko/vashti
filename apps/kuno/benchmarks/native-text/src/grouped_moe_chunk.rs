//! Owning MAX8 routed-only operator experiment. No graph/state/cache publication.
//! The private staged ledger and immutable wave + actual pending plan remain
//! pinned until original fills, consumers, whole-owner/status/I/O checks finish.
//! Any partial mutation or abandoned attempt poisons the whole physical bank.
use crate::{
    bank_slots::{BankSlots, Plan as BankPlan, StagedSlots},
    expert_banks::{self, BankInstance, ExpertBanks},
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::grouped_moe as k,
    token_waves,
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::{
    TensorType,
    model::{BoundGraph, GraphBindingId, RoutedWeights},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const PRIVATE_CAP: usize = 2 * 1024 * 1024;
pub const MAIN_CAP: u64 = 1024 * 1024 * 1024;
pub const IO_CAP: u64 = 64 * 1024 * 1024;
pub const IO_CHECKPOINT_BOUND: u64 = 8 * 1024 * 1024;
pub const HOST_CAP: usize = 32 * 1024 * 1024;
pub const N: usize = 7;
pub const MAP: usize = 0;
pub const INPUT: usize = 1;
pub const ACT: usize = 2;
pub const PART: usize = 3;
pub const PROB: usize = 4;
pub const OUTPUT: usize = 5;
pub const STATUS: usize = 6;
pub struct Layout {
    pub geometry: k::Geometry,
    pub spans: k::Spans,
    pub words: [usize; N],
}
impl Layout {
    pub fn original(dtype: TensorType, tile: usize, capacity: usize) -> Result<Self> {
        let g = k::Geometry::new(
            [
                k::Projection::new(crate::quant_matvec::Q4_K, 2560, 640, 512)?,
                k::Projection::new(crate::quant_matvec::Q4_K, 2560, 640, 512)?,
                k::Projection::new(dtype.gguf_id(), 640, 2560, 512)?,
            ],
            tile,
            capacity,
        )?;
        let spans = k::Spans::new([5, 7, 2563, 11, 645, 13, 2567, 9, 17, 2565, 3]);
        let words = [
            5 + k::MAP_WORDS + 7,
            17 + 7 * 2565 + 2560 + 23,
            11 + 79 * 645 + 640 + 13,
            13 + 79 * 2567 + 2560 + 17,
            9 + k::MAX_ROUTES + 19,
            17 + 7 * 2565 + 2560 + 23,
            3 + g.status_words() + 29,
        ];
        let l = Self {
            geometry: g,
            spans,
            words,
        };
        l.validate()?;
        Ok(l)
    }
    pub fn validate(&self) -> Result<()> {
        self.spans.validate(self.geometry, self.words)?;
        let g = self.geometry;
        if g.hidden() != 2560
            || g.ff() != 640
            || g.projections()[0].dtype() != crate::quant_matvec::Q4_K
            || g.projections()[1].dtype() != crate::quant_matvec::Q4_K
            || !matches!(
                g.projections()[2].dtype(),
                crate::quant_matvec::Q5_0 | crate::quant_matvec::Q8_0
            )
            || self.page_bytes() != Ok(1_671_168)
            || g.scratch_bytes(true)? != self.page_bytes()?
        {
            return Err("grouped chunk exact private geometry".into());
        }
        // Exact fixture bounds: no guard can accidentally move a page class.
        let want = [32768, 131072, 262144, 1048576, 32768, 131072, 32768];
        for (words, bytes) in self.words.into_iter().zip(want) {
            if BufferLayout::new(words)?.owned_bytes != bytes {
                return Err("grouped chunk guard page class".into());
            }
        }
        Ok(())
    }
    pub fn page_bytes(&self) -> std::result::Result<usize, crate::private_arena::ArenaError> {
        self.words.into_iter().try_fold(0usize, |sum, w| {
            sum.checked_add(BufferLayout::new(w)?.owned_bytes)
                .ok_or(crate::private_arena::ArenaError::BudgetExceeded)
        })
    }
    pub fn validate_limits(&self, l: &wgpu::Limits) -> Result<()> {
        self.validate()?;
        if l.max_compute_invocations_per_workgroup < 64
            || l.max_compute_workgroup_size_x < 64
            || l.max_compute_workgroup_size_y < 1
            || l.max_compute_workgroup_size_z < 1
            || l.max_bind_groups < 1
            || l.max_uniform_buffer_binding_size < 512
            || l.max_compute_workgroup_storage_size < (2 * 64 * self.geometry.tile() * 4) as u32
            || l.max_storage_buffers_per_shader_stage < 7
            || l.max_bindings_per_bind_group < 7
            || l.min_storage_buffer_offset_alignment == 0
            || l.min_uniform_buffer_offset_alignment == 0
            || l.max_compute_workgroups_per_dimension < 2560
            || l.max_storage_buffer_binding_size < 128 * 1024 * 1024
            || l.max_buffer_size < 128 * 1024 * 1024
        {
            return Err("grouped chunk actual device limits".into());
        }
        Ok(())
    }
}
/// All original eighty ranks/bits remain guarded until whole operator acceptance.
pub struct Packet {
    pub experts: Zeroizing<[u32; k::MAX_ROUTES]>,
    pub probabilities: Zeroizing<[f32; k::MAX_ROUTES]>,
    pub count: usize,
}
impl Packet {
    pub fn from_plan(plan: &token_waves::Plan) -> Result<Self> {
        let mut p = Self {
            experts: Zeroizing::new([0; k::MAX_ROUTES]),
            probabilities: Zeroizing::new([0.; k::MAX_ROUTES]),
            count: plan.count(),
        };
        if !(1..=8).contains(&p.count) {
            return Err("grouped packet count".into());
        }
        for t in 0..p.count {
            let row = plan.token(t).ok_or("grouped packet token")?;
            if row.len() != 10 {
                return Err("grouped packet ranks".into());
            }
            for (r, route) in row.iter().enumerate() {
                p.experts[t * 10 + r] = route.expert();
                p.probabilities[t * 10 + r] = route.probability();
            }
        }
        Ok(p)
    }
}
impl Drop for Packet {
    fn drop(&mut self) {
        self.count.zeroize();
    }
}
/// Adapter pins the borrowed immutable wave AND the actual pending generation.
/// Full plan canonical validation precedes any victim fill, including every
/// window in a union larger than ten. It does not make a raw tuple trustworthy.
struct ProtectedWave<'a> {
    wave: token_waves::WaveView<'a>,
    pending: &'a BankPlan,
    staged: &'a StagedSlots,
    bank: &'a BankSlots,
    selected: Zeroizing<[usize; k::MAX_ROUTES]>,
    pairs: Zeroizing<[(u32, u32); k::MAX_GROUPS]>,
    len: usize,
}
impl<'a> ProtectedWave<'a> {
    fn new(
        wave: token_waves::WaveView<'a>,
        pending: &'a BankPlan,
        staged: &'a StagedSlots,
        bank: &'a BankSlots,
    ) -> Result<Self> {
        let len = wave.protected_experts().len();
        let mut p = Self {
            wave,
            pending,
            staged,
            bank,
            selected: Zeroizing::new([0; k::MAX_ROUTES]),
            pairs: Zeroizing::new([(0, 0); k::MAX_GROUPS]),
            len,
        };
        if len == 0 || len > k::MAX_GROUPS || len > bank.capacity() || pending.slots.len() != len {
            return Err("grouped pinned union bounds".into());
        }
        for (i, &e) in p.wave.protected_experts().iter().enumerate() {
            p.selected[i] = e as usize;
            p.pairs[i] = (e, pending.slots[i] as u32);
        }
        p.validate()?;
        Ok(p)
    }
    fn validate(&self) -> Result<()> {
        self.staged
            .validate_pending(self.bank, self.pending, &self.selected[..self.len])?;
        for (i, &e) in self.wave.protected_experts().iter().enumerate() {
            if self.selected[i] != e as usize || self.pairs[i] != (e, self.pending.slots[i] as u32)
            {
                return Err("grouped pinned canonical union".into());
            }
        }
        for t in self.wave.tokens() {
            let row = self.wave.token(t).ok_or("grouped pinned token")?;
            if row.len() != 10
                || row
                    .iter()
                    .any(|r| !self.wave.protected_experts().contains(&r.expert()))
            {
                return Err("grouped pinned incomplete union".into());
            }
        }
        Ok(())
    }
    fn map(&self, packet: &Packet) -> Result<k::GroupMap<'_>> {
        self.validate()?;
        let tokens = self.wave.tokens();
        Ok(k::GroupMap::new(
            packet.count,
            tokens.start,
            tokens.len(),
            &packet.experts[..packet.count * 10],
            &packet.probabilities[..packet.count * 10],
            &self.pairs[..self.len],
            self.bank.capacity(),
        )?)
    }
}
impl Drop for ProtectedWave<'_> {
    fn drop(&mut self) {
        self.len.zeroize();
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Ready,
    Running,
    AwaitingAcceptance,
    AcceptedOperator,
    Poisoned,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    Never,
    BeforePlan,
    FirstCopy,
    Enqueued,
    LateInput,
}
pub struct Snapshot(pub [Zeroizing<Vec<u32>>; N]);
struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
struct Read(cubecl::bytes::Bytes);
impl Drop for Read {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
pub fn write(
    arena: &PrivateArena,
    id: BufferId,
    words: &[u32],
    retired: &AtomicBool,
) -> Result<()> {
    let l = arena.layout(id)?;
    if words.len() != l.logical_words {
        return Err("grouped owning write extent".into());
    }
    let mut b = Upload(Vec::new());
    b.0.try_reserve_exact(l.owned_bytes)?;
    for &x in words {
        b.0.extend_from_slice(&x.to_le_bytes());
    }
    b.0.resize(l.owned_bytes, 0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        arena
            .client()?
            .write_private(arena.handle(id)?.clone(), std::mem::take(&mut b.0))
            .map_err(|_| "grouped owning write failed")?;
        Ok(())
    }));
    if !matches!(result, Ok(Ok(()))) {
        retired.store(true, Ordering::Release);
        return Err("grouped owning write failed; context retired".into());
    }
    Ok(())
}
pub fn read(
    arena: &PrivateArena,
    id: BufferId,
    retired: &AtomicBool,
) -> Result<Zeroizing<Vec<u32>>> {
    arena.layout(id)?;
    arena.client()?;
    let attempted =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_inner(arena, id)));
    match attempted {
        Ok(Ok(v)) => Ok(v),
        _ => {
            retired.store(true, Ordering::Release);
            Err("grouped owning read failed; context retired".into())
        }
    }
}
fn read_inner(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
    let l = arena.layout(id)?;
    let b = Read(
        arena
            .client()?
            .read_one(arena.handle(id)?.clone())
            .map_err(|_| "grouped completed owning read")?,
    );
    if b.0.len() != l.owned_bytes {
        return Err("grouped owning read physical extent".into());
    }
    let mut v = Zeroizing::new(Vec::new());
    v.try_reserve_exact(l.owned_words)?;
    for x in b.0.chunks_exact(4) {
        v.push(u32::from_le_bytes(x.try_into()?));
    }
    Ok(v)
}
pub fn audit(client: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<()> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io_audited())
    }));
    let r = match result {
        Ok(Ok(r)) => r,
        _ => {
            retired.store(true, Ordering::Release);
            return Err("grouped checked I/O cleanup failed; context retired".into());
        }
    };
    check_audit(
        r.upload_bytes,
        r.uniform_bytes,
        r.readback_bytes,
        r.inspected_bytes,
        r.inspected_buffers,
        retired,
    )
}
fn check_audit(
    upload: u64,
    uniform: u64,
    readback: u64,
    inspected: u64,
    buffers: u64,
    retired: &AtomicBool,
) -> Result<()> {
    let n = upload
        .checked_add(uniform)
        .and_then(|n| n.checked_add(readback));
    if n.is_none_or(|n| n == 0 || n > IO_CHECKPOINT_BOUND || n != inspected)
        || buffers == 0
        || retired.load(Ordering::Acquire)
    {
        retired.store(true, Ordering::Release);
        return Err("grouped malformed I/O audit; context retired".into());
    }
    Ok(())
}
pub fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    Ok(client
        .memory_usage()
        .map_err(|_| "grouped public memory telemetry")?
        .bytes_reserved)
}
pub fn source_requests(weights: RoutedWeights<'_>, misses: usize) -> Result<Vec<u64>> {
    if misses > 10 {
        return Err("grouped source window bounds".into());
    }
    let mut sizes = [0; 3];
    for (i, t) in [weights.gate, weights.up, weights.down]
        .into_iter()
        .enumerate()
    {
        sizes[i] = t
            .info()
            .row_bytes()
            .checked_mul(t.info().dimensions()[1])
            .ok_or("grouped source overflow")?;
        main_admission::page_bytes(sizes[i])?;
    }
    Ok((0..misses).flat_map(|_| sizes).collect())
}
fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err("grouped chunk cancelled".into())
    } else {
        Ok(())
    }
}
fn weights<'a>(graph: BoundGraph<'_, 'a>, layer: usize) -> Result<RoutedWeights<'a>> {
    if layer >= graph.geometry().layers {
        return Err("grouped original layer".into());
    }
    Ok(RoutedWeights {
        gate: graph.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
        up: graph.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
        down: graph.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
    })
}
pub fn original_weights<'a>(graph: BoundGraph<'_, 'a>, layer: usize) -> Result<RoutedWeights<'a>> {
    weights(graph, layer)
}
/// Constructor-sealed original artifact/physical bank generation. Priming may
/// use the actual bank ledger before it is consumed by the chunk owner; replacing
/// a physical bank underneath this seal is rejected by its immutable instance.
pub struct OriginalBank {
    bank: ExpertBanks,
    binding: GraphBindingId,
    layer: usize,
    instance: BankInstance,
}
impl OriginalBank {
    pub fn allocate(
        graph: BoundGraph<'_, '_>,
        layer: usize,
        client: &ComputeClient<WgpuRuntime>,
        ledger: &mut MainAdmission,
        capacity: usize,
    ) -> Result<Self> {
        graph.verify_unchanged()?;
        let w = weights(graph, layer)?;
        ledger.observe_before(sample(client)?)?;
        ledger.precharge(&ExpertBanks::allocation_requests(w, capacity)?)?;
        let bank = ExpertBanks::new(client, w, capacity)?;
        let instance = bank.instance();
        ledger.observe_after(sample(client)?)?;
        Ok(Self {
            bank,
            binding: graph.binding_id(),
            layer,
            instance,
        })
    }
    pub fn validate(
        &self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        client: &ComputeClient<WgpuRuntime>,
    ) -> Result<()> {
        if self.binding != graph.binding_id()
            || self.layer != layer
            || self.instance != self.bank.instance()
        {
            return Err("grouped original bank binding/generation".into());
        }
        graph.verify_unchanged()?;
        self.bank.check_client(client)?;
        self.bank.check_names(weights(graph, layer)?)?;
        Ok(())
    }
    pub fn banks(&self) -> &ExpertBanks {
        &self.bank
    }
    // Fixed diagnostic only, before chunk ownership. No model commit/fill API
    // is introduced here; the existing planner/fill/consumer owner still applies.
    pub fn fixture_bank_mut(&mut self) -> &mut ExpertBanks {
        &mut self.bank
    }
}
/// Owns exclusive banks through cleanup. No mutable bank/ledger escape and no
/// publication method: routed-only acceptance cannot stand in for full FFN/model.
pub struct Owner {
    layout: Layout,
    ids: [BufferId; N],
    banks: ExpertBanks,
    binding: GraphBindingId,
    layer: usize,
    staged: Option<StagedSlots>,
    phase: Phase,
    before: Snapshot,
    retired: Arc<AtomicBool>,
}
impl Owner {
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &mut PrivateArena,
        ledger: &mut MainAdmission,
        layout: Layout,
        input: &[u32],
        limits: &wgpu::Limits,
        original: OriginalBank,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        layout.validate_limits(limits)?;
        original.validate(graph, layer, arena.client()?)?;
        let banks = original.bank;
        let w = weights(graph, layer)?;
        for (t, p) in [w.gate, w.up, w.down]
            .into_iter()
            .zip(layout.geometry.projections())
        {
            if t.info().dtype().gguf_id() != p.dtype()
                || t.info().dimensions()
                    != [p.columns() as u64, p.rows() as u64, p.experts() as u64]
                || t.info().row_bytes() != p.row_bytes() as u64
            {
                return Err("grouped original packed metadata".into());
            }
        }
        if arena
            .charged_bytes()
            .checked_add(layout.page_bytes()?)
            .is_none_or(|n| n > arena.budget_bytes())
            || input.len() != layout.words[INPUT]
        {
            return Err("grouped private admission".into());
        }
        banks.check_client(arena.client()?)?;
        banks.check_names(w)?;
        if banks.slots.capacity() != layout.geometry.capacity() {
            return Err("grouped immutable supplied bank capacity".into());
        }
        ledger.observe_before(sample(arena.client()?)?)?;
        ledger.precharge(&layout.words.map(|n| {
            BufferLayout::new(n)
                .expect("prevalidated layout")
                .owned_bytes as u64
        }))?;
        let before = images(&layout, input)?;
        let mut ids = Vec::with_capacity(N);
        for i in 0..N {
            ids.push(arena.upload_u32(&before.0[i][..layout.words[i]])?);
        }
        let ids: [BufferId; N] = ids.try_into().map_err(|_| "grouped private owner count")?;
        let staged = StagedSlots::begin(&banks.slots)?;
        let owner = Self {
            layout,
            ids,
            banks,
            binding: graph.binding_id(),
            layer,
            staged: Some(staged),
            phase: Phase::Ready,
            before,
            retired,
        };
        owner.binding_proof(graph, arena)?;
        ledger.observe_after(sample(arena.client()?)?)?;
        Ok(owner)
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    /// Explicit malformed fixture only; no arbitrary handle/geometry setter.
    pub(crate) fn fixed_public_alias_output_to_input(&mut self) -> Result<()> {
        if self.phase != Phase::Ready || self.layout.words[INPUT] != self.layout.words[OUTPUT] {
            self.poison();
            return Err("grouped malformed alias fixture precondition".into());
        }
        self.ids[OUTPUT] = self.ids[INPUT];
        Ok(())
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn poison(&mut self) {
        self.phase = Phase::Poisoned;
    }
    pub fn banks(&self) -> &ExpertBanks {
        &self.banks
    }
    pub fn original_ledger_unchanged(&self) -> Result<()> {
        self.staged
            .as_ref()
            .ok_or("grouped missing staged ledger")?
            .validate_origin(&self.banks.slots)?;
        Ok(())
    }
    pub fn binding_proof(&self, graph: BoundGraph<'_, '_>, arena: &PrivateArena) -> Result<()> {
        self.layout.validate()?;
        if self.retired.load(Ordering::Acquire) {
            return Err("grouped context retired".into());
        }
        graph.verify_unchanged()?;
        if graph.binding_id() != self.binding
            || self.banks.hidden != 2560
            || self.banks.ff != 640
            || self.banks.experts != 512
            || self.banks.slots.capacity() != self.layout.geometry.capacity()
        {
            return Err("grouped model/bank identity".into());
        }
        let client = arena.client()?;
        self.banks.check_client(client)?;
        self.banks.check_names(weights(graph, self.layer)?)?;
        let public: Vec<_> = [self.banks.gate(), self.banks.up(), self.banks.down()]
            .into_iter()
            .flat_map(|b| b.segments())
            .collect();
        if public.len()
            != if self.layout.geometry.capacity() == 96 {
                6
            } else {
                3
            }
        {
            return Err("grouped bank segment count".into());
        }
        for i in 0..N {
            if self.ids[..i].contains(&self.ids[i])
                || arena.layout(self.ids[i])?.logical_words != self.layout.words[i]
                || arena.layout(self.ids[i])? != BufferLayout::new(self.layout.words[i])?
            {
                return Err("grouped full private owner identity/epoch".into());
            }
        }
        let private = self
            .ids
            .iter()
            .map(|&id| arena.handle(id))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let stream = private[0].stream;
        if private.iter().any(|h| h.stream != stream)
            || public.iter().any(|s| s.handle().stream != stream)
        {
            return Err("grouped cross-stream binding".into());
        }
        let resources = private
            .iter()
            .map(|h| client.get_resource((*h).clone()))
            .chain(
                public
                    .iter()
                    .map(|s| client.get_resource(s.handle().clone())),
            )
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| "grouped physical lookup")?;
        let pages = resources
            .iter()
            .enumerate()
            .map(|(i, r)| -> Result<_> {
                let v = r.resource();
                let logical = if i < N {
                    arena.layout(self.ids[i])?.owned_bytes as u64
                } else {
                    (public[i - N].words() * 4) as u64
                };
                let bound = if i < N {
                    logical
                } else {
                    main_admission::page_bytes(logical)?
                };
                Ok((
                    v.buffer.clone(),
                    v.offset,
                    v.size,
                    v.buffer.size(),
                    logical,
                    bound,
                    i < N,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        validate_physical_pages(&pages)?;
        for (bank, projection) in [self.banks.gate(), self.banks.up(), self.banks.down()]
            .into_iter()
            .zip(self.layout.geometry.projections())
        {
            if bank.expert_bytes() != projection.expert_bytes() {
                return Err("grouped immutable expert byte extent".into());
            }
        }
        for shard in 0..self.banks.gate().segments().len() {
            let p = [self.banks.gate(), self.banks.up(), self.banks.down()]
                .map(|b| &b.segments()[shard]);
            if p.iter()
                .any(|s| s.first_slot() != p[0].first_slot() || s.capacity() != p[0].capacity())
            {
                return Err("grouped public shard mismatch".into());
            }
            self.layout.geometry.validate_shard(
                p[0].first_slot(),
                p[0].capacity(),
                p.map(|s| s.words()),
            )?;
        }
        self.original_ledger_unchanged()?;
        Ok(())
    }
    fn arg(&self, arena: &PrivateArena, index: usize) -> Result<ArrayArg<WgpuRuntime>> {
        let id = *self.ids.get(index).ok_or("grouped argument index")?;
        Ok(unsafe {
            ArrayArg::from_raw_parts(arena.handle(id)?.clone(), arena.layout(id)?.owned_words)
        })
    }
    fn snapshot(&self, arena: &PrivateArena) -> Result<Snapshot> {
        let mut v = Vec::with_capacity(N);
        for id in self.ids {
            v.push(read(arena, id, &self.retired)?);
        }
        audit(arena.client()?, &self.retired)?;
        Ok(Snapshot(
            v.try_into().map_err(|_| "grouped snapshot count")?,
        ))
    }
    /// Synchronous fixed operator owner. No ordinary inference timeout/replay.
    /// The external comparator must accept every original intermediate/output.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        ledger: &mut MainAdmission,
        plan: &token_waves::Plan,
        cancel: &AtomicBool,
        failure: Failure,
        accept: impl FnOnce(&Layout, &Snapshot) -> Result<()>,
    ) -> Result<Snapshot> {
        if self.phase != Phase::Ready {
            self.poison();
            ledger.poison();
            return Err("grouped owner cannot retry".into());
        }
        self.phase = Phase::Running;
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.execute_inner(graph, arena, ledger, plan, cancel, failure, accept)
        }));
        match attempt {
            Ok(Ok(v)) => Ok(v),
            _ => {
                self.poison();
                ledger.poison();
                Err("grouped chunk rejected and entire owner poisoned".into())
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn execute_inner(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        ledger: &mut MainAdmission,
        plan: &token_waves::Plan,
        cancel: &AtomicBool,
        failure: Failure,
        accept: impl FnOnce(&Layout, &Snapshot) -> Result<()>,
    ) -> Result<Snapshot> {
        if failure == Failure::BeforePlan {
            cancel.store(true, Ordering::Release);
        }
        cancelled(cancel)?;
        self.binding_proof(graph, arena)?;
        let packet = Packet::from_plan(plan)?;
        if plan.wave_count() == 0 || plan.wave_count() > 8 {
            return Err("grouped chunk waves".into());
        }
        let w = weights(graph, self.layer)?;
        let mut expected_map = Zeroizing::new(vec![0u32; self.layout.words[MAP]]);
        let mut expected_prob = Zeroizing::new(vec![0u32; self.layout.words[PROB]]);
        let mut processed = 0;
        let mut prior: Option<Snapshot> = None;
        for i in 0..plan.wave_count() {
            cancelled(cancel)?;
            self.binding_proof(graph, arena)?;
            let wave = plan.wave(i).ok_or("grouped missing wave")?;
            if wave.tokens().start != processed {
                return Err("grouped noncontiguous wave".into());
            }
            let mut selected = Zeroizing::new([0usize; k::MAX_GROUPS]);
            let union = wave.protected_experts();
            if union.is_empty() || union.len() > selected.len() {
                return Err("grouped union size".into());
            }
            for (j, &e) in union.iter().enumerate() {
                selected[j] = e as usize;
            }
            let pending = self
                .staged
                .as_mut()
                .ok_or("grouped staged ledger")?
                .plan(&self.banks.slots, &selected[..union.len()])?;
            let pin = ProtectedWave::new(
                wave,
                &pending,
                self.staged.as_ref().ok_or("grouped staged ledger")?,
                &self.banks.slots,
            )?;
            pin.validate()?;
            for window in pending.uploads.chunks(10) {
                pin.validate()?;
                cancelled(cancel)?;
                ledger.observe_before(sample(arena.client()?)?)?;
                ledger.precharge(&source_requests(w, window.len())?)?;
                let mut copies = 0;
                for &(e, s) in window {
                    for (bank, t) in [
                        (self.banks.gate(), w.gate),
                        (self.banks.up(), w.up),
                        (self.banks.down(), w.down),
                    ] {
                        pin.validate()?;
                        expert_banks::fill(arena.client()?, bank, t, e, s, cancel)?;
                        copies += 1;
                        if failure == Failure::FirstCopy && copies == 1 {
                            cancel.store(true, Ordering::Release);
                        }
                    }
                }
                // Owning completion precedes source debt release. The entire
                // pinned union remains protected through every upload window.
                drop(read(arena, self.ids[STATUS], &self.retired)?);
                audit(arena.client()?, &self.retired)?;
                cancelled(cancel)?;
                ledger.completed_fence(sample(arena.client()?)?)?;
            }
            let map = pin.map(&packet)?;
            if map.count() != packet.count {
                return Err("grouped map original count".into());
            }
            map.validate(
                &packet.experts[..packet.count * 10],
                &packet.probabilities[..packet.count * 10],
                self.banks.slots.capacity(),
            )?;
            expected_map.fill(0);
            expected_prob.fill(0);
            expected_map
                [self.layout.spans.map_first()..self.layout.spans.map_first() + k::MAP_WORDS]
                .copy_from_slice(map.words());
            for (r, &p) in map.probabilities().iter().enumerate() {
                expected_prob[self.layout.spans.prob_first() + r] = p.to_bits();
            }
            write(arena, self.ids[MAP], &expected_map, &self.retired)?;
            write(arena, self.ids[PROB], &expected_prob, &self.retired)?;
            self.enqueue_wave(arena, map.groups())?;
            if failure == Failure::Enqueued {
                cancel.store(true, Ordering::Release);
            }
            let actual = self.snapshot(arena)?;
            let end = pin.wave.tokens().end;
            validate_wave(
                &self.layout,
                &actual,
                &self.before,
                &expected_map,
                &expected_prob,
                end,
                packet.count,
                false,
            )?;
            if let Some(previous) = &prior {
                validate_previous_immutable(&self.layout, previous, &actual, processed)?;
            }
            cancelled(cancel)?;
            pin.validate()?;
            drop(map);
            drop(pin);
            ledger.completed_fence(sample(arena.client()?)?)?;
            self.staged
                .as_mut()
                .ok_or("grouped staged ledger")?
                .advance_after_completed_consumers(
                    &self.banks.slots,
                    pending,
                    &selected[..union.len()],
                )?;
            processed = end;
            prior = Some(actual);
        }
        if processed != packet.count
            || self
                .staged
                .as_ref()
                .ok_or("grouped staged ledger")?
                .accepted_waves()
                != plan.wave_count()
        {
            return Err("grouped incomplete chunk".into());
        }
        self.enqueue_final(arena)?;
        let actual = self.snapshot(arena)?;
        validate_wave(
            &self.layout,
            &actual,
            &self.before,
            &expected_map,
            &expected_prob,
            processed,
            packet.count,
            true,
        )?;
        cancelled(cancel)?;
        self.binding_proof(graph, arena)?;
        ledger.completed_fence(sample(arena.client()?)?)?;
        validate_previous_immutable(
            &self.layout,
            prior
                .as_ref()
                .ok_or("grouped missing completed wave image")?,
            &actual,
            processed,
        )?;
        self.phase = Phase::AwaitingAcceptance;
        accept(&self.layout, &actual)?;
        cancelled(cancel)?;
        self.original_ledger_unchanged()?;
        self.phase = Phase::AcceptedOperator;
        Ok(actual)
    }
    fn enqueue_wave(&self, arena: &PrivateArena, groups: usize) -> Result<()> {
        let g = self.layout.geometry;
        let s = &self.layout.spans;
        let [gate, up, down] = g.projections();
        let client = arena.client()?;
        for shard in 0..self.banks.gate().segments().len() {
            let ga = &self.banks.gate().segments()[shard];
            let u = &self.banks.up().segments()[shard];
            let d = &self.banks.down().segments()[shard];
            unsafe {
                k::grouped_gate_up::launch_unchecked::<WgpuRuntime>(
                    client,
                    g.grid(groups, false)?,
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(ga.handle().clone(), ga.words()),
                    ArrayArg::from_raw_parts(u.handle().clone(), u.words()),
                    self.arg(arena, INPUT)?,
                    self.arg(arena, MAP)?,
                    self.arg(arena, ACT)?,
                    self.arg(arena, STATUS)?,
                    s.map_first(),
                    s.input_first(),
                    s.input_stride(),
                    s.act_first(),
                    s.act_stride(),
                    s.status_first(),
                    ga.first_slot(),
                    ga.capacity(),
                    g.hidden(),
                    g.ff(),
                    gate.dtype(),
                    up.dtype(),
                    gate.row_bytes(),
                    up.row_bytes(),
                    g.tile(),
                );
                k::grouped_down::launch_unchecked::<WgpuRuntime>(
                    client,
                    g.grid(groups, true)?,
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(d.handle().clone(), d.words()),
                    self.arg(arena, ACT)?,
                    self.arg(arena, MAP)?,
                    self.arg(arena, PART)?,
                    self.arg(arena, STATUS)?,
                    s.map_first(),
                    s.act_first(),
                    s.act_stride(),
                    s.partial_first(),
                    s.partial_stride(),
                    s.status_first(),
                    d.first_slot(),
                    d.capacity(),
                    g.hidden(),
                    g.ff(),
                    down.dtype(),
                    down.row_bytes(),
                    g.tile(),
                );
            }
        }
        Ok(())
    }
    fn enqueue_final(&self, arena: &PrivateArena) -> Result<()> {
        let g = self.layout.geometry;
        let s = &self.layout.spans;
        let c = arena.client()?;
        unsafe {
            k::grouped_reduce::launch_unchecked::<WgpuRuntime>(
                c,
                CubeCount::Static(g.hidden().div_ceil(64) as u32, 8, 1),
                CubeDim::new_1d(64),
                self.arg(arena, PART)?,
                self.arg(arena, PROB)?,
                self.arg(arena, MAP)?,
                self.arg(arena, STATUS)?,
                self.arg(arena, OUTPUT)?,
                s.map_first(),
                s.partial_first(),
                s.partial_stride(),
                s.prob_first(),
                s.status_first(),
                s.output_first(),
                s.output_stride(),
                g.hidden(),
            );
            k::grouped_finite::launch_unchecked::<WgpuRuntime>(
                c,
                CubeCount::Static(
                    g.hidden().div_ceil(64).max(g.ff().div_ceil(64)) as u32,
                    80,
                    4,
                ),
                CubeDim::new_1d(64),
                self.arg(arena, INPUT)?,
                self.arg(arena, ACT)?,
                self.arg(arena, PART)?,
                self.arg(arena, OUTPUT)?,
                self.arg(arena, MAP)?,
                self.arg(arena, STATUS)?,
                s.map_first(),
                s.input_first(),
                s.input_stride(),
                s.act_first(),
                s.act_stride(),
                s.partial_first(),
                s.partial_stride(),
                s.output_first(),
                s.output_stride(),
                s.status_first(),
                g.hidden(),
                g.ff(),
            );
        }
        Ok(())
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.poison();
        self.layer.zeroize();
    }
}
/// Full physical allocations are distinct. Original PUBLIC banks may own a
/// backing span between their exact logical bytes and the charged exclusive-bin
/// upper bound. Every PRIVATE page must equal its exact full owning span.
fn validate_physical_pages<I: Eq>(pages: &[(I, u64, u64, u64, u64, u64, bool)]) -> Result<()> {
    for (i, (identity, offset, size, physical, logical, bound, private)) in pages.iter().enumerate()
    {
        if *offset != 0
            || *size != *logical
            || *physical < *logical
            || *physical > *bound
            || (*private && *physical != *logical)
            || pages[..i].iter().any(|x| &x.0 == identity)
        {
            return Err("grouped full physical page or alias".into());
        }
    }
    Ok(())
}
pub fn images(l: &Layout, input: &[u32]) -> Result<Snapshot> {
    l.validate()?;
    if input.len() != l.words[INPUT] {
        return Err("grouped image input".into());
    }
    let mut v = Vec::with_capacity(N);
    for (i, &n) in l.words.iter().enumerate() {
        let page = BufferLayout::new(n)?;
        let mut x = Zeroizing::new(vec![0u32; page.owned_words]);
        if matches!(i, INPUT | ACT | PART | OUTPUT) {
            x[..n].fill(0x3f340000);
        }
        if i == INPUT {
            x[..n].copy_from_slice(input);
        }
        v.push(x);
    }
    Ok(Snapshot(
        v.try_into().map_err(|_| "grouped before image count")?,
    ))
}
fn matrix_indices(first: usize, stride: usize, width: usize, count: usize, index: usize) -> bool {
    if index < first {
        return false;
    }
    let r = (index - first) / stride;
    let c = (index - first) % stride;
    r < count && c < width
}
fn validate_previous_immutable(
    l: &Layout,
    previous: &Snapshot,
    actual: &Snapshot,
    completed: usize,
) -> Result<()> {
    if completed > 8 {
        return Err("grouped previous-wave count".into());
    }
    for (field, first, stride, width) in [
        (ACT, l.spans.act_first(), l.spans.act_stride(), 640),
        (
            PART,
            l.spans.partial_first(),
            l.spans.partial_stride(),
            2560,
        ),
    ] {
        if previous.0[field].len() != actual.0[field].len() {
            return Err("grouped previous-wave owning span".into());
        }
        for r in 0..completed * 10 {
            for c in 0..width {
                let at = first + r * stride + c;
                if previous.0[field][at] != actual.0[field][at] {
                    return Err("grouped later wave changed completed route".into());
                }
            }
        }
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub fn validate_wave(
    l: &Layout,
    actual: &Snapshot,
    before: &Snapshot,
    map: &[u32],
    prob: &[u32],
    completed: usize,
    count: usize,
    final_stage: bool,
) -> Result<()> {
    l.validate()?;
    if count == 0 || count > 8 || completed == 0 || completed > count {
        return Err("grouped wave accepted count".into());
    }
    let s = &l.spans;
    for i in 0..N {
        if actual.0[i].len() != before.0[i].len()
            || actual.0[i].len() != BufferLayout::new(l.words[i])?.owned_words
        {
            return Err("grouped owning snapshot size".into());
        }
        for (j, &bits) in actual.0[i].iter().enumerate() {
            let active = match i {
                ACT => matrix_indices(s.act_first(), s.act_stride(), 640, completed * 10, j),
                PART => matrix_indices(
                    s.partial_first(),
                    s.partial_stride(),
                    2560,
                    completed * 10,
                    j,
                ),
                OUTPUT => {
                    final_stage
                        && matrix_indices(s.output_first(), s.output_stride(), 2560, count, j)
                }
                _ => false,
            };
            if active {
                if !f32::from_bits(bits).is_finite() {
                    return Err("grouped nonfinite intermediate".into());
                }
            } else if i == MAP {
                let expected = if j < map.len() { map[j] } else { 0 };
                if bits != expected {
                    return Err("grouped map owning guard or mutation".into());
                }
            } else if i == PROB {
                let expected = if j < prob.len() { prob[j] } else { 0 };
                if bits != expected {
                    return Err("grouped probability owning guard or mutation".into());
                }
            } else if i == STATUS {
                if final_stage {
                    continue;
                }
                let p = s.status_first();
                let want = if j >= p && j < p + 80 {
                    u32::from(j - p < completed * 10)
                } else if j >= p + 80 && j < p + 160 {
                    u32::from(j - p - 80 < completed * 10)
                } else {
                    0
                };
                if bits != want {
                    return Err("grouped incomplete wave readiness/tail".into());
                }
            } else if bits != before.0[i][j] {
                return Err("grouped modified source gap inactive or physical tail".into());
            }
        }
    }
    if final_stage {
        k::validate_status(l.geometry, count, s.status_first(), &actual.0[STATUS])?;
    } else {
        for t in 0..completed {
            for c in 0..2560 {
                if !f32::from_bits(actual.0[INPUT][s.input_first() + t * s.input_stride() + c])
                    .is_finite()
                {
                    return Err("grouped nonfinite accepted input".into());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(count: usize, capacity: usize, overlap: bool) -> token_waves::Plan {
        let mut p = token_waves::Input::new(count, capacity, 512).unwrap();
        for t in 0..count {
            for r in 0..10 {
                p.set(
                    t,
                    r,
                    if overlap {
                        r as u32
                    } else {
                        (t * 10 + r) as u32
                    },
                    if r == 0 { 0. } else { 1. / 9. },
                )
                .unwrap();
            }
        }
        token_waves::Plan::build(&p, &AtomicBool::new(false)).unwrap()
    }
    #[test]
    fn actual_pending_wave_adapter_covers_all_eighty_routes_and_guarded_zero_probability_bits() {
        for cap in [16, 64, 72, 96] {
            for count in [1, 3, 5, 8] {
                for overlap in [true, false] {
                    let p = plan(count, cap, overlap);
                    let packet = Packet::from_plan(&p).unwrap();
                    let bank = BankSlots::new(cap, 512).unwrap();
                    let mut staged = StagedSlots::begin(&bank).unwrap();
                    let mut covered = Zeroizing::new([0u8; 80]);
                    for i in 0..p.wave_count() {
                        let wave = p.wave(i).unwrap();
                        let selected = Zeroizing::new(
                            wave.protected_experts()
                                .iter()
                                .map(|&e| e as usize)
                                .collect::<Vec<_>>(),
                        );
                        let pending = staged.plan(&bank, &selected).unwrap();
                        let pin = ProtectedWave::new(wave, &pending, &staged, &bank).unwrap();
                        let map = pin.map(&packet).unwrap();
                        assert_eq!(map.count(), count);
                        assert_eq!(map.probabilities()[0].to_bits(), 0);
                        for t in pin.wave.tokens() {
                            for r in 0..10 {
                                covered[t * 10 + r] += 1;
                            }
                        }
                        drop(map);
                        drop(pin);
                        staged
                            .advance_after_completed_consumers(&bank, pending, &selected)
                            .unwrap();
                        staged.validate_origin(&bank).unwrap();
                    }
                    assert_eq!(staged.accepted_waves(), p.wave_count());
                    assert!(covered[..count * 10].iter().all(|&n| n == 1));
                    assert!(covered[count * 10..].iter().all(|&n| n == 0));
                }
            }
        }
    }
    #[test]
    fn pinned_adapter_rejects_modified_mappings_cross_bank_stale_generation_and_incomplete_union() {
        let p = plan(3, 64, false);
        let packet = Packet::from_plan(&p).unwrap();
        let mut bank = BankSlots::new(64, 512).unwrap();
        let mut stage = StagedSlots::begin(&bank).unwrap();
        let ids = Zeroizing::new(
            p.wave(0)
                .unwrap()
                .protected_experts()
                .iter()
                .map(|&e| e as usize)
                .collect::<Vec<_>>(),
        );
        let mut pending = stage.plan(&bank, &ids).unwrap();
        pending.slots[0] = 63;
        assert!(ProtectedWave::new(p.wave(0).unwrap(), &pending, &stage, &bank).is_err());
        drop(pending);
        let foreign = BankSlots::new(64, 512).unwrap();
        let mut other = StagedSlots::begin(&foreign).unwrap();
        let wrong = other.plan(&foreign, &ids).unwrap();
        assert!(ProtectedWave::new(p.wave(0).unwrap(), &wrong, &stage, &bank).is_err());
        let incomplete = stage.plan(&bank, &ids[..10]).unwrap();
        assert!(ProtectedWave::new(p.wave(0).unwrap(), &incomplete, &stage, &bank).is_err());
        let pending = stage.plan(&bank, &ids).unwrap();
        let mut pin = ProtectedWave::new(p.wave(0).unwrap(), &pending, &stage, &bank).unwrap();
        pin.pairs[0].1 = 63;
        assert!(pin.validate().is_err());
        assert!(pin.map(&packet).is_err());
        drop(pin);
        let mutation = bank.plan(&[100]).unwrap();
        bank.commit(mutation).unwrap();
        assert!(ProtectedWave::new(p.wave(0).unwrap(), &pending, &stage, &bank).is_err());
    }
    #[test]
    fn full_fixture_geometry_and_real_adapter_limits_reject_before_allocation() {
        for cap in [16, 64, 72, 96] {
            for dtype in [TensorType::Q5_0, TensorType::Q8_0] {
                for tile in [4, 8] {
                    let mut l = Layout::original(dtype, tile, cap).unwrap();
                    assert_eq!(l.page_bytes().unwrap(), 1_671_168);
                    assert_eq!(l.geometry.scratch_bytes(true).unwrap(), 1_671_168);
                    let mut limits = wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 7,
                        max_storage_buffer_binding_size: 128 * 1024 * 1024,
                        max_buffer_size: 128 * 1024 * 1024,
                        ..Default::default()
                    };
                    l.validate_limits(&limits).unwrap();
                    limits.max_storage_buffers_per_shader_stage = 6;
                    assert!(l.validate_limits(&limits).is_err());
                    limits.max_storage_buffers_per_shader_stage = 7;
                    limits.max_compute_workgroups_per_dimension = 2559;
                    assert!(l.validate_limits(&limits).is_err());
                    limits.max_compute_workgroups_per_dimension = 65535;
                    limits.min_storage_buffer_offset_alignment = 0;
                    assert!(l.validate_limits(&limits).is_err());
                    l.words[OUTPUT] = 32769;
                    assert!(l.validate().is_err());
                }
            }
        }
        assert!(Layout::original(TensorType::F32, 8, 16).is_err());
        assert!(Layout::original(TensorType::Q5_0, 8, 73).is_err());
    }
    fn valid_snapshot(
        l: &Layout,
        count: usize,
        completed: usize,
        last: bool,
    ) -> (Snapshot, Snapshot, Zeroizing<Vec<u32>>, Zeroizing<Vec<u32>>) {
        let before = images(l, &vec![0x3f340000; l.words[INPUT]]).unwrap();
        let mut actual = Snapshot(std::array::from_fn(|i| {
            Zeroizing::new(before.0[i].to_vec())
        }));
        for (field, first, stride, width, n) in [
            (
                ACT,
                l.spans.act_first(),
                l.spans.act_stride(),
                640,
                completed * 10,
            ),
            (
                PART,
                l.spans.partial_first(),
                l.spans.partial_stride(),
                2560,
                completed * 10,
            ),
            (
                OUTPUT,
                l.spans.output_first(),
                l.spans.output_stride(),
                2560,
                if last { count } else { 0 },
            ),
        ] {
            for r in 0..n {
                for c in 0..width {
                    actual.0[field][first + r * stride + c] = 0.25f32.to_bits();
                }
            }
        }
        let p = l.spans.status_first();
        for r in 0..completed * 10 {
            actual.0[STATUS][p + r] = 1;
            actual.0[STATUS][p + 80 + r] = 1;
        }
        if last {
            let mut off = p + 160;
            for (length, active) in [
                (8 * 40, count * 40),
                (80 * 10, count * 10 * 10),
                (80 * 40, count * 10 * 40),
                (8 * 40, count * 40),
            ] {
                actual.0[STATUS][off..off + active].fill(1);
                off += length;
            }
        }
        let map = Zeroizing::new(vec![0; l.words[MAP]]);
        let prob = Zeroizing::new(vec![0; l.words[PROB]]);
        (actual, before, map, prob)
    }
    #[test]
    fn actual_owning_snapshot_guard_status_and_late_failure_acceptance() {
        let l = Layout::original(TensorType::Q8_0, 8, 16).unwrap();
        for count in [1, 3, 5, 8] {
            let (actual, before, map, prob) = valid_snapshot(&l, count, count, true);
            validate_wave(&l, &actual, &before, &map, &prob, count, count, true).unwrap();
        }
        let (mut actual, before, map, prob) = valid_snapshot(&l, 8, 7, false);
        validate_wave(&l, &actual, &before, &map, &prob, 7, 8, false).unwrap();
        actual.0[STATUS][l.spans.status_first() + 70] = 1;
        assert!(validate_wave(&l, &actual, &before, &map, &prob, 7, 8, false).is_err());
        actual.0[STATUS][l.spans.status_first() + 70] = 0;
        actual.0[ACT][l.spans.act_first() + 69 * l.spans.act_stride() + 639] = f32::NAN.to_bits();
        assert!(validate_wave(&l, &actual, &before, &map, &prob, 7, 8, false).is_err());
        let (mut a, b, m, p) = valid_snapshot(&l, 3, 3, true);
        a.0[PART][l.words[PART] + 1] = 1;
        assert!(validate_wave(&l, &a, &b, &m, &p, 3, 3, true).is_err());
        let (mut a, b, m, p) = valid_snapshot(&l, 3, 3, true);
        a.0[ACT][l.spans.act_first() + 30 * l.spans.act_stride()] = 0;
        assert!(validate_wave(&l, &a, &b, &m, &p, 3, 3, true).is_err());
        let (mut a, b, m, p) = valid_snapshot(&l, 3, 3, true);
        a.0[INPUT][l.spans.input_first()] = 0;
        assert!(validate_wave(&l, &a, &b, &m, &p, 3, 3, true).is_err());
        let (mut a, b, m, p) = valid_snapshot(&l, 3, 3, true);
        a.0[MAP][l.words[MAP] + 1] = 1;
        assert!(validate_wave(&l, &a, &b, &m, &p, 3, 3, true).is_err());
        let (mut a, b, m, p) = valid_snapshot(&l, 3, 3, true);
        a.0[STATUS][l.spans.status_first() + 159] = 1;
        assert!(validate_wave(&l, &a, &b, &m, &p, 3, 3, true).is_err());
    }
    #[test]
    fn full_physical_alias_padding_and_public_page_bound_proof() {
        let pages = [
            (1, 0, 32768, 32768, 32768, 32768, true),
            (2, 0, 9216000, 9216256, 9216000, 16777216, false),
        ];
        validate_physical_pages(&pages).unwrap();
        for broken in [
            (1, 0, 9216000, 9216256, 9216000, 16777216, false),
            (2, 256, 9216000, 9216256, 9216000, 16777216, false),
            (2, 0, 9216000, 16777472, 9216000, 16777216, false),
            (2, 0, 9216000, 9215744, 9216000, 16777216, false),
            (2, 0, 32768, 65536, 32768, 65536, true),
            (2, 0, 32764, 32768, 32768, 32768, true),
        ] {
            assert!(validate_physical_pages(&[pages[0], broken]).is_err());
        }
    }
    #[test]
    fn later_waves_and_final_reduce_cannot_rewrite_previously_completed_routes() {
        let l = Layout::original(TensorType::Q8_0, 8, 16).unwrap();
        let (prior, _, _, _) = valid_snapshot(&l, 8, 3, false);
        let (mut later, _, _, _) = valid_snapshot(&l, 8, 5, false);
        validate_previous_immutable(&l, &prior, &later, 3).unwrap();
        later.0[PART][l.spans.partial_first() + 29 * l.spans.partial_stride() + 2559] =
            0.26f32.to_bits();
        assert!(validate_previous_immutable(&l, &prior, &later, 3).is_err());
        let (last, _, _, _) = valid_snapshot(&l, 8, 8, false);
        let (final_image, _, _, _) = valid_snapshot(&l, 8, 8, true);
        validate_previous_immutable(&l, &last, &final_image, 8).unwrap();
    }
    #[test]
    fn malformed_audit_never_allows_clean_context_reuse() {
        for (u, un, r, n, b) in [
            (0, 0, 0, 0, 0),
            (1, 2, 3, 5, 1),
            (1, 2, 3, 6, 0),
            (u64::MAX, 1, 0, 0, 1),
            (IO_CHECKPOINT_BOUND, 1, 0, IO_CHECKPOINT_BOUND + 1, 1),
        ] {
            let retired = AtomicBool::new(false);
            assert!(check_audit(u, un, r, n, b, &retired).is_err());
            assert!(retired.load(Ordering::Acquire));
        }
        let fresh = AtomicBool::new(false);
        check_audit(1, 2, 3, 6, 1, &fresh).unwrap();
        assert!(!fresh.load(Ordering::Acquire));
        let retired = AtomicBool::new(true);
        assert!(check_audit(1, 2, 3, 6, 1, &retired).is_err());
    }
}
