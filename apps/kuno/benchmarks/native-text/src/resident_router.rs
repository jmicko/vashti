//! Sealed fixed-public all-resident dispatch authorization. No routing result is
//! accepted, counted or published until the batched owning audit resolves it.
use super::*;
use crate::resident_model::batch_io;

const TABLE_WORDS: usize = 48 * 512;
const TABLE_BYTES: usize = 128 * 1024;
pub(crate) const RESIDENT_ROUTER_MAIN: u64 =
    TABLE_BYTES as u64 + batch_io::RESIDENT_AUDIT_BYTES as u64;
pub(crate) const RESIDENT_ROUTER_HOST: usize = batch_io::RESIDENT_AUDIT_HOST;
// Count both complete owners' runtime source/target/inspection mirrors even
// though only the table is uploaded and the audit is mapped. Keep extra INFO
// and slab rollover separate; no reclaimed old bridge credit is assumed.
pub(crate) const RESIDENT_ROUTER_PREFLIGHT_INFO: u64 =
    crate::resident_router_gpu::EXTRA_INFO_BYTES_PER_TOKEN;
pub(crate) const RESIDENT_ROUTER_INTEGER_CLEAR_INFO: u64 = 48 * 2048;
pub(crate) const RESIDENT_ROUTER_RUNTIME: u64 =
    4 * RESIDENT_ROUTER_MAIN + (48 * 2 + 4) * 2048 + 65536 + RESIDENT_ROUTER_PREFLIGHT_INFO;
pub(crate) const RESIDENT_ROUTER_RUNTIME_LIMIT: u64 = 128 * 1024 * 1024;
pub(crate) fn resident_router_runtime(prepared_integer: bool) -> u64 {
    RESIDENT_ROUTER_RUNTIME
        + if prepared_integer {
            RESIDENT_ROUTER_INTEGER_CLEAR_INFO
        } else {
            0
        }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arithmetic {
    OriginalF32,
    PreparedTwoQ8,
}
impl Arithmetic {
    fn from_policy(prepared_integer: bool, requested_skip: bool) -> Result<Self> {
        if prepared_integer && !requested_skip {
            return Err(err(
                "resident prepared integer requires declared segment policy",
            ));
        }
        Ok(if prepared_integer {
            Self::PreparedTwoQ8
        } else {
            Self::OriginalF32
        })
    }
    fn validate(self, prepared_integer: bool, requested_skip: bool) -> Result<()> {
        if self != Self::from_policy(prepared_integer, requested_skip)? {
            return Err(err(
                "resident router arithmetic changed after authorization",
            ));
        }
        Ok(())
    }
}

struct FrozenBank {
    instance: expert_banks::BankInstance,
    pins: [Handle; 6],
}
struct Authorization {
    nonce: Zeroizing<u64>,
    binding: GraphBindingId,
    workspace: Zeroizing<u64>,
    fixture: crate::oracle_trace::Fixture,
    arithmetic: Arithmetic,
    requested_skip: bool,
    next: Zeroizing<u64>,
    table: Zeroizing<Vec<u32>>,
    banks: Vec<FrozenBank>,
}
pub(super) struct RouterOwners {
    ids: Zeroizing<[BufferId; 2]>,
    authorization: Option<Authorization>,
}

fn layer_table(source: &crate::bank_slots::BankSlots) -> Result<Zeroizing<[u32; 512]>> {
    if source.capacity() != 144 || source.experts() != 512 {
        return Err(err("resident router exact slot geometry"));
    }
    let mut table = Zeroizing::new([u32::MAX; 512]);
    let mut seen = Zeroizing::new([false; 144]);
    for (id, cell) in table.iter_mut().enumerate() {
        if let Some(slot) = source
            .resident_slot(id)
            .map_err(|_| err("resident router slot lookup"))?
        {
            if slot >= 144
                || seen[slot]
                || source
                    .resident_expert(slot)
                    .map_err(|_| err("resident router slot reverse lookup"))?
                    != Some(id)
            {
                return Err(err("resident router noncanonical mapping"));
            }
            seen[slot] = true;
            *cell = slot as u32;
        }
    }
    for (slot, &present) in seen.iter().enumerate() {
        if source
            .resident_expert(slot)
            .map_err(|_| err("resident router slot reverse lookup"))?
            .is_some()
            != present
        {
            return Err(err("resident router incomplete mapping"));
        }
    }
    Ok(table)
}

fn all_hit_proof(
    source: &crate::bank_slots::BankSlots,
    selected: &[(u32, f32); 10],
    table: &[u32],
    wires: &[u32],
) -> Result<(count1::Proof, StagedSlots, Last)> {
    if table.len() != 512 || wires.len() != k::MAP_WORDS + 80 + OWNER_WORDS {
        return Err(Error::InvalidBufferLength);
    }
    let ids = Zeroizing::new(selected.map(|v| v.0 as usize));
    let probabilities = Zeroizing::new(selected.map(|v| v.1));
    let (mut proof, staged, last, split) =
        count1::Proof::prepare_empty_slots(source, &ids, &probabilities, 10, 10)?;
    if proof.misses() != 0
        || proof.cpu_count() != 0
        || !split.cpu().is_empty()
        || split.gpu() != ids.as_slice()
        || !last.pending.uploads.is_empty()
        || last.pending.slots.len() != 10
    {
        return Err(err("resident router requires actual all-hit routes"));
    }
    let mut pairs = Zeroizing::new([(0u32, 0u32); 10]);
    for rank in 0..10 {
        let id = ids[rank];
        if id >= 512 || table[id] >= 144 || last.pending.slots[rank] != table[id] as usize {
            return Err(err("resident router actual slot changed"));
        }
        pairs[rank] = (id as u32, table[id]);
    }
    let map = hybrid_k::Map::new(
        &ids,
        &probabilities,
        &count1::Proof::native_map(&split),
        &*pairs,
        144,
    )
    .map_err(err)?;
    if wires[..k::MAP_WORDS] != *map.words()
        || wires[k::MAP_WORDS..k::MAP_WORDS + 80]
            .iter()
            .zip(map.probabilities())
            .any(|(&a, &b)| a != b.to_bits())
        || wires[k::MAP_WORDS + 80..] != *map.owner_words()
    {
        return Err(err("resident router consumed headers differ"));
    }
    proof.complete(None)?;
    proof.validate_source(source, &staged, &last, true)?;
    Ok((proof, staged, last))
}

impl Authorization {
    fn owned_bytes(&self) -> Result<usize> {
        self.table
            .capacity()
            .checked_mul(4)
            .and_then(|n| n.checked_add(self.banks.capacity() * std::mem::size_of::<FrozenBank>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(Error::ArithmeticOverflow)
    }
    fn check_layer(
        &self,
        e: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &Workspace,
        layer: usize,
    ) -> Result<()> {
        if layer >= 48
            || graph.binding_id() != self.binding
            || w.identity != *self.workspace
            || !e.resident_router_reserved
            || !e.packed_profile.is_low_bit()
            || e.bank_slots != 144
            || e.adaptive.is_some()
            || e.secondary.is_some()
            || e.secondary_frequency.is_some()
            || e.batch_experts
            || e.ordered_public_upload
            || e.word_down
            || e.fused_experts
            || e.integer_experts
            || e.q6_integer_head
            || e.q3_subgroup_moe
            || e.q3_shared_original_moe
            || e.q3_skip_empty_segments != self.requested_skip
            || e.q3_expert_oracle
            || e.q3_router_reference.is_some()
            || !e.q3_empty_slot_fill
            || !e.gpu_router
            || !e.shared_kernels
            || !e.fast_dense
            || !e.model.weights_mapped()
            || e.gpu_expert_limit != 10
            || w.cpu_miss_cap() != Some(10)
            || w.cpu_gpu_overlap() != Some(true)
            || w.cpu_q8_activations() != Some(false)
            || w.cpu_q8_residual() != Some(false)
            || w.layout.tile != 1
            || w.layout.gate_policy != k::GatePolicy::Generic
        {
            return Err(err("resident router immutable configured authorization"));
        }
        self.arithmetic
            .validate(e.q3_integer_moe, e.q3_skip_empty_segments)?;
        graph.verify_unchanged()?;
        e.model.verify_file_unchanged()?;
        e.check_lab_oracle_fixture(self.fixture)?;
        let weights = weights(graph, layer)?;
        let bank = e
            .banks
            .get(weights.gate.info().name())
            .ok_or(err("resident router original bank absent"))?;
        let original = self.banks.get(layer).ok_or(Error::OutOfBounds)?;
        if bank.instance() != original.instance || bank.profile() != e.packed_profile {
            return Err(err("resident router bank instance changed"));
        }
        bank.check_names(weights)?;
        bank.check_client(arena.client().map_err(u::fault)?)?;
        let current = layer_table(&bank.slots)?;
        if self.table[layer * 512..(layer + 1) * 512] != *current {
            return Err(err("resident router resident mapping changed"));
        }
        let handles: Vec<_> = [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .flat_map(|b| b.segments().iter().map(|p| p.handle().clone()))
            .collect();
        if handles.len() != original.pins.len() {
            return Err(err("resident router pin inventory changed"));
        }
        for (a, b) in handles.iter().zip(&original.pins) {
            if a.stream != b.stream
                || a.size() != b.size()
                || a.offset_start != b.offset_start
                || a.offset_end != b.offset_end
            {
                return Err(err("resident router original pin changed"));
            }
            let metadata = u::resources(
                arena.client().map_err(u::fault)?,
                vec![a.clone(), b.clone()],
            )?;
            if metadata.len() != 2
                || metadata[0].resource().buffer != metadata[1].resource().buffer
                || metadata[0].resource().offset != metadata[1].resource().offset
                || metadata[0].resource().size != metadata[1].resource().size
            {
                return Err(err("resident router physical pin changed"));
            }
        }
        Ok(())
    }
}

impl Workspace {
    pub(crate) fn resident_router_allocation_requests() -> [u64; 2] {
        [TABLE_BYTES as u64, batch_io::RESIDENT_AUDIT_BYTES as u64]
    }
    pub(crate) fn reserve_resident_router(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
    ) -> Result<()> {
        allocation.check(&self.admission)?;
        if self.resident_router.is_some() || self.hybrid.is_none() || *self.policy.chunk != 0 {
            return Err(err("resident router constructor reservation"));
        }
        self.resident_router = Some(RouterOwners {
            ids: Zeroizing::new([
                arena.allocate_zeroed(TABLE_WORDS).map_err(u::fault)?,
                arena
                    .allocate_zeroed(batch_io::RESIDENT_AUDIT_WORDS)
                    .map_err(u::fault)?,
            ]),
            authorization: None,
        });
        Ok(())
    }
    pub(crate) fn resident_router_ids(&self) -> Option<[BufferId; 2]> {
        self.resident_router.as_ref().map(|r| *r.ids)
    }
    pub(crate) fn resident_router_active(&self) -> bool {
        self.resident_router
            .as_ref()
            .is_some_and(|r| r.authorization.is_some())
    }
    pub(crate) fn enable_resident_router(
        &mut self,
        e: &mut GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<()> {
        self.healthy(arena)?;
        self.policy.completed_boundary()?;
        let owners = self
            .resident_router
            .as_ref()
            .ok_or(err("resident router constructor owners absent"))?;
        if owners.authorization.is_some() {
            return Err(err("resident router already active"));
        }
        let trace = e
            .oracle_trace
            .as_ref()
            .ok_or(err("resident router replay trace absent"))?;
        trace.validate_ready_replay()?;
        if !trace.is_replay()
            || trace.fixture() == crate::oracle_trace::Fixture::Q4Rainwater
            || e.banks.len() != 48
        {
            return Err(err("resident router fixed public replay only"));
        }
        let mut table = Zeroizing::new(Vec::new());
        table
            .try_reserve_exact(TABLE_BYTES / 4)
            .map_err(|_| Error::LimitExceeded("resident router table host"))?;
        if table.capacity() != TABLE_BYTES / 4 {
            return Err(err("resident router table host capacity"));
        }
        table.resize(TABLE_BYTES / 4, 0);
        let mut banks = Vec::new();
        banks
            .try_reserve_exact(48)
            .map_err(|_| Error::LimitExceeded("resident router bank pins"))?;
        if banks.capacity() != 48 {
            return Err(err("resident router bank pin capacity"));
        }
        for layer in 0..48 {
            let weights = weights(graph, layer)?;
            let bank = e
                .banks
                .get(weights.gate.info().name())
                .ok_or(err("resident router bank absent"))?;
            bank.check_names(weights)?;
            let current = layer_table(&bank.slots)?;
            table[layer * 512..(layer + 1) * 512].copy_from_slice(&*current);
            let handles: Vec<_> = [bank.gate(), bank.up(), bank.down()]
                .into_iter()
                .flat_map(|b| b.segments().iter().map(|p| p.handle().clone()))
                .collect();
            let pins = handles
                .try_into()
                .map_err(|_| err("resident router exact two bank segments"))?;
            banks.push(FrozenBank {
                instance: bank.instance(),
                pins,
            });
        }
        let authorization = Authorization {
            nonce: Zeroizing::new(next()?),
            binding: graph.binding_id(),
            workspace: Zeroizing::new(self.identity),
            fixture: trace.fixture(),
            arithmetic: Arithmetic::from_policy(e.q3_integer_moe, e.q3_skip_empty_segments)?,
            requested_skip: e.q3_skip_empty_segments,
            next: Zeroizing::new(trace.start() as u64),
            table,
            banks,
        };
        if authorization
            .owned_bytes()?
            .checked_add(batch_io::RESIDENT_AUDIT_BYTES + 48 * 32768 + 4096)
            .is_none_or(|n| n > RESIDENT_ROUTER_HOST)
        {
            return Err(Error::LimitExceeded("resident router complete host plan"));
        }
        for layer in 0..48 {
            authorization.check_layer(e, graph, arena, self, layer)?;
        }
        let handle = arena.handle(owners.ids[0]).map_err(u::fault)?.clone();
        let mut bytes = Zeroizing::new(Vec::new());
        bytes
            .try_reserve_exact(TABLE_BYTES)
            .map_err(|_| err("resident router upload host"))?;
        if bytes.capacity() != TABLE_BYTES {
            return Err(err("resident router upload capacity"));
        }
        for &word in &*authorization.table {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        arena
            .client()
            .map_err(u::fault)?
            .write_private(handle, std::mem::take(&mut *bytes))
            .map_err(u::fault)?;
        self.resident_router
            .as_mut()
            .ok_or(err("resident router owner disappeared"))?
            .authorization = Some(authorization);
        e.resident_banks_frozen = true;
        Ok(())
    }
    pub(crate) fn begin_resident_router_token(
        &self,
        e: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        start: u64,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let Some(owners) = &self.resident_router else {
            return Ok(());
        };
        let Some(a) = &owners.authorization else {
            return Ok(());
        };
        u::cancel(cancel)?;
        if !e.resident_banks_frozen
            || count != 1
            || start != *a.next
            || start < a.fixture.start() as u64
            || start >= a.fixture.prefix() as u64
            || e.oracle_trace
                .as_ref()
                .is_none_or(|t| !t.is_replay() || t.fixture() != a.fixture)
        {
            return Err(err("resident router token boundary"));
        }
        for layer in 0..48 {
            a.check_layer(e, graph, arena, self, layer)?;
        }
        u::zero(arena, owners.ids[1])
    }
    pub(crate) fn resolve_resident_router_token(
        &self,
        e: &mut GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        session: &Session,
        pending: &mut [Option<Pending>; 48],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !self.resident_router_active() {
            return Ok(());
        }
        let owners = self
            .resident_router
            .as_ref()
            .ok_or(err("resident router owner absent"))?;
        let a = owners
            .authorization
            .as_ref()
            .ok_or(err("resident router authorization absent"))?;
        self.check(e, session, arena)?;
        let mut inputs = Zeroizing::new([session.stamp.ids[0]; 48]);
        for (layer, p) in pending.iter().enumerate() {
            let p = p
                .as_ref()
                .ok_or(err("resident router missing layer receipt"))?;
            if p.deferred_router.as_ref().map(|v| **v) != Some(*a.nonce)
                || *p.layer != layer
                || *p.stamp != *session.stamp
                || !p.armed
                || p.count1.is_some()
                || p.staged.is_some()
                || p.last.is_some()
            {
                return Err(err("resident router unresolved owner identity"));
            }
            inputs[layer] = p.views.ids[0];
        }
        let audits = batch_io::read_resident_router_audit(
            graph,
            arena,
            &self.admission,
            owners.ids[1],
            &inputs,
            session.stamp.fault,
            session.stamp.start,
            cancel,
        )?;
        if audits.len() != 48 {
            return Err(Error::InvalidBufferLength);
        }
        for (layer, audit) in audits.iter().enumerate() {
            u::cancel(cancel)?;
            a.check_layer(e, graph, arena, self, layer)?;
            let p = pending[layer]
                .as_mut()
                .ok_or(err("resident router pending disappeared"))?;
            audit.input.validate_for(
                graph,
                p.views.ids[0],
                session.stamp.fault,
                layer,
                session.stamp.start,
                0,
                2560,
            )?;
            let bank = e
                .banks
                .get(&p.key)
                .ok_or(err("resident router bank disappeared"))?;
            let (proof, staged, last) = all_hit_proof(
                &bank.slots,
                audit
                    .input
                    .gpu_selection()
                    .ok_or(err("resident router audit selection absent"))?,
                &a.table[layer * 512..(layer + 1) * 512],
                &*audit.wires,
            )?;
            p.count1 = Some(proof);
            p.staged = Some(staged);
            p.last = Some(last);
            p.deferred_router.take();
            self.check_pending(e, session, arena, p)?;
        }
        e.expert_hits = e
            .expert_hits
            .checked_add(480)
            .ok_or(Error::ArithmeticOverflow)?;
        e.gpu_router_calls = e
            .gpu_router_calls
            .checked_add(48)
            .ok_or(Error::ArithmeticOverflow)?;
        e.moe_calls = e
            .moe_calls
            .checked_add(48)
            .ok_or(Error::ArithmeticOverflow)?;
        e.resident_router_audits = e
            .resident_router_audits
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(())
    }
    pub(crate) fn finish_resident_router_token(&mut self) -> Result<()> {
        if let Some(a) = self
            .resident_router
            .as_mut()
            .and_then(|r| r.authorization.as_mut())
        {
            self.policy.completed_boundary()?;
            *a.next = a.next.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        }
        Ok(())
    }
}

impl GpuExecutor<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_resident_router_moe(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        packet: BufferId,
        views: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        let result = guarded(&w.poison.clone(), || {
            u::cancel(cancel)?;
            w.check(self, s, arena)?;
            let owners = w
                .resident_router
                .as_ref()
                .ok_or(err("resident router owning reservation absent"))?;
            let a = owners
                .authorization
                .as_ref()
                .ok_or(err("resident router sealed authorization absent"))?;
            a.check_layer(self, graph, arena, w, layer)?;
            if s.stamp.count != 1
                || s.stamp.start != *a.next
                || views.count() != 1
                || views.input_first() != 0
                || views.input_stride() != 2560
                || views.output_first() != 0
                || views.output_stride() != 2560
                || views.packed_first() != 0
            {
                return Err(err("resident router exact count1 views"));
            }
            let weights = weights(graph, layer)?;
            let key = weights.gate.info().name().to_owned();
            let bank = self
                .banks
                .get(&key)
                .ok_or(err("resident router immutable bank absent"))?;
            let g = w.layout.geometries[layer];
            let ids = w.ids(input, output);
            let spans = k::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
            let mut words = Zeroizing::new([0usize; 7]);
            for (i, &id) in ids.iter().enumerate() {
                words[i] = u::words(arena, id)?;
            }
            spans.validate(g, *words).map_err(err)?;
            physical_with_packet(arena, bank, g, &ids, s.stamp.fault, w, Some(packet))?;
            let hybrid = w
                .hybrid
                .as_ref()
                .ok_or(err("resident router hybrid owner absent"))?;
            let hybrid_ids = *hybrid.ids;
            let route_ids = *owners.ids;
            let nonce = *a.nonce;
            w.policy.issue(layer)?;
            let clears = Zeroizing::new([
                w.ids[0],
                w.ids[1],
                w.ids[2],
                w.ids[3],
                w.ids[4],
                hybrid_ids[0],
                hybrid_ids[1],
            ]);
            u::zero_batch(arena, &*clears)?;
            let handle = |id| arena.handle(id).cloned().map_err(u::fault);
            crate::resident_router_gpu::launch(
                arena.client().map_err(u::fault)?,
                [
                    handle(packet)?,
                    handle(route_ids[0])?,
                    handle(ids[MAP])?,
                    handle(ids[PROB])?,
                    handle(hybrid_ids[1])?,
                    handle(route_ids[1])?,
                    handle(s.stamp.fault)?,
                ],
                layer,
                s.stamp.start,
            )?;
            match a.arithmetic {
                Arithmetic::OriginalF32 => launch_wave(arena, bank, g, &spans, &ids, 10)?,
                Arithmetic::PreparedTwoQ8 => {
                    count1_executor::launch_resident_prepared_pair(
                        arena,
                        bank,
                        g,
                        &spans,
                        &ids,
                        hybrid_ids[0],
                    )?;
                    // Every gate/down consumer was queued before launch_pair
                    // returned. The same stream now ends that private scratch
                    // phase with a full owning clear, before original rank
                    // merge/fold lends CPU_PART as its entirely zero CPU owner.
                    u::zero(arena, hybrid_ids[0])?;
                }
            }
            launch_finite(arena, g, &spans, &ids)?;
            hybrid_k::merge(
                arena,
                g,
                &ids,
                hybrid_ids[0],
                hybrid_ids[1],
                s.stamp.fault,
                &spans,
                1,
                0,
                2560,
                0,
            )?;
            launch_finite(arena, g, &spans, &ids)?;
            hybrid_k::fold(
                arena,
                g,
                &ids,
                hybrid_ids[0],
                hybrid_ids[1],
                s.stamp.fault,
                &spans,
                1,
                0,
                2560,
                0,
            )?;
            u::cancel(cancel)?;
            Ok(Pending {
                views: Views {
                    ids: Zeroizing::new([input, output]),
                    offsets: Zeroizing::new([0, 2560, 0, 2560]),
                },
                stamp: Zeroizing::new((*s.stamp).clone()),
                layer: Zeroizing::new(layer),
                key,
                instance: bank.instance(),
                pins: [bank.gate(), bank.up(), bank.down()]
                    .into_iter()
                    .flat_map(|b| b.segments().iter().map(|p| p.handle().clone()))
                    .collect(),
                staged: None,
                last: None,
                count1: None,
                deferred_router: Some(Zeroizing::new(nonce)),
                waves: Zeroizing::new(1),
                poison: w.poison.clone(),
                armed: true,
            })
        });
        if result.is_err() {
            self.poison_resident();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bank(ids: &[usize]) -> crate::bank_slots::BankSlots {
        let mut b = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        let p = b.plan(ids).unwrap();
        b.commit(p).unwrap();
        b
    }
    fn wires(
        source: &crate::bank_slots::BankSlots,
        selected: &[(u32, f32); 10],
    ) -> Zeroizing<Vec<u32>> {
        let ids = selected.map(|v| v.0 as usize);
        let probs = selected.map(|v| v.1);
        let pairs = ids.map(|id| (id as u32, source.resident_slot(id).unwrap().unwrap() as u32));
        let mut rank = [0u32; 20];
        for i in 0..10 {
            rank[i * 2 + 1] = i as u32;
        }
        let map = hybrid_k::Map::new(&ids, &probs, &rank, &pairs, 144).unwrap();
        Zeroizing::new(
            map.words()
                .iter()
                .copied()
                .chain(map.probabilities().iter().map(|v| v.to_bits()))
                .chain(map.owner_words().iter().copied())
                .collect(),
        )
    }
    #[test]
    fn audited_all_hit_routes_preserve_actual_slots_original_ranks_and_no_uploads() {
        let b = bank(&[511, 70, 5, 143, 10, 2, 99, 300, 45, 201]);
        let selected = [511, 5, 99, 70, 2, 300, 143, 201, 45, 10].map(|id| (id, 0.1));
        let t = layer_table(&b).unwrap();
        let wire = wires(&b, &selected);
        let (proof, staged, last) = all_hit_proof(&b, &selected, &*t, &wire).unwrap();
        assert_eq!(proof.ids(), &selected.map(|v| v.0 as usize));
        assert_eq!(proof.cpu_count(), 0);
        assert_eq!(proof.misses(), 0);
        assert!(last.pending.uploads.is_empty());
        proof.validate_source(&b, &staged, &last, true).unwrap();
    }
    #[test]
    fn original_all_hit_publication_changes_recency_without_changing_frozen_mapping() {
        let mut b = bank(&(0..144).collect::<Vec<_>>());
        let selected = [51, 9, 128, 1, 72, 143, 60, 121, 0, 99].map(|id| (id, 0.1));
        let original = layer_table(&b).unwrap();
        let wire = wires(&b, &selected);
        let (proof, mut staged, last) = all_hit_proof(&b, &selected, &*original, &wire).unwrap();
        proof.validate_source(&b, &staged, &last, true).unwrap();
        staged
            .advance_after_completed_consumers(&b, last.pending, &last.selected[..*last.len])
            .unwrap();
        staged
            .publish_after_whole_model_acceptance(&mut b, 1)
            .unwrap();
        assert_eq!(*layer_table(&b).unwrap(), *original);
        all_hit_proof(&b, &selected, &*original, &wire).unwrap();
        // A real replacement is distinguished from harmless all-hit recency.
        let replacement = b.plan(&[511]).unwrap();
        b.commit(replacement).unwrap();
        assert_ne!(*layer_table(&b).unwrap(), *original);
    }
    #[test]
    fn missing_changed_slots_consumed_headers_and_owner_readiness_cannot_gain_acceptance() {
        let b = bank(&(0..10).collect::<Vec<_>>());
        let selected = std::array::from_fn(|i| (i as u32, 0.1));
        let t = layer_table(&b).unwrap();
        let wire = wires(&b, &selected);
        for index in [4, 4 + 80, k::MAP_WORDS, k::MAP_WORDS + 80 + 6] {
            let mut changed = Zeroizing::new(wire.to_vec());
            changed[index] ^= 1;
            assert!(all_hit_proof(&b, &selected, &*t, &changed).is_err());
        }
        let mut changed_table = t;
        changed_table[0] = 99;
        assert!(all_hit_proof(&b, &selected, &*changed_table, &wire).is_err());
        let mut missing = selected;
        missing[9].0 = 511;
        assert!(all_hit_proof(&b, &missing, &*layer_table(&b).unwrap(), &wire).is_err());
    }
    #[test]
    fn full_public_pages_and_extra_runtime_are_independently_charged() {
        assert_eq!(TABLE_WORDS, crate::resident_router_gpu::TABLE_WORDS);
        assert_eq!(
            batch_io::RESIDENT_AUDIT_WORDS,
            crate::resident_router_gpu::AUDIT_WORDS
        );
        assert_eq!(
            BufferLayout::new(TABLE_WORDS).unwrap().owned_bytes,
            TABLE_BYTES
        );
        assert_eq!(
            BufferLayout::new(batch_io::RESIDENT_AUDIT_WORDS)
                .unwrap()
                .owned_bytes,
            batch_io::RESIDENT_AUDIT_BYTES
        );
        assert_eq!(
            Workspace::resident_router_allocation_requests()
                .iter()
                .sum::<u64>(),
            RESIDENT_ROUTER_MAIN
        );
        const { assert!(RESIDENT_ROUTER_RUNTIME > 4 * RESIDENT_ROUTER_MAIN) };
        const { assert!(RESIDENT_ROUTER_RUNTIME + 67108864 < RESIDENT_ROUTER_RUNTIME_LIMIT) };
        assert_eq!(crate::resident_router_gpu::EXTRA_INFO_CALLS_PER_TOKEN, 48);
        assert_eq!(RESIDENT_ROUTER_PREFLIGHT_INFO, 98_304);
        assert_eq!(RESIDENT_ROUTER_INTEGER_CLEAR_INFO, 98304);
        assert_eq!(resident_router_runtime(false), 9_281_536);
        assert_eq!(resident_router_runtime(true), 9_379_840);
    }
    #[test]
    fn prepared_arithmetic_and_required_segment_policy_are_immutable_after_authorization() {
        for requested in [false, true] {
            let f32 = Arithmetic::from_policy(false, requested).unwrap();
            assert_eq!(f32, Arithmetic::OriginalF32);
            assert!(f32.validate(false, requested).is_ok());
            assert!(f32.validate(true, requested).is_err());
        }
        assert!(Arithmetic::from_policy(true, false).is_err());
        let prepared = Arithmetic::from_policy(true, true).unwrap();
        assert_eq!(prepared, Arithmetic::PreparedTwoQ8);
        assert!(prepared.validate(true, true).is_ok());
        assert!(prepared.validate(false, true).is_err());
        assert!(prepared.validate(true, false).is_err());
    }
}
