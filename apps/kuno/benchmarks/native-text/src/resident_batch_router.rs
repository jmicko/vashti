//! Construction-reserved, public warm13 all-hit GPU routing experiment.
//! Actual routed IDs are checked against original CPU routing before any family
//! commit. Missing experts reject; this is not a general offload implementation.
use super::*;
use crate::resident_batch_router_gpu as gpu;

pub(crate) const MAIN_BYTES: u64 = gpu::NEW_TABLE_AUDIT_BYTES;
pub(crate) const HOST_BYTES: usize = 5 * 1024 * 1024;
pub(crate) const RUNTIME_BYTES: u64 = 4 * MAIN_BYTES + 48 * 4096 + 65536;

pub(super) struct Owners {
    ids: Zeroizing<[BufferId; 2]>,
    table: Option<Zeroizing<Vec<u32>>>,
    instances: Vec<expert_banks::BankInstance>,
}
fn reject() -> Error {
    err("public warm batch router rejected")
}
fn slot_table(slots: &crate::bank_slots::BankSlots) -> Result<Zeroizing<[u32; 512]>> {
    if slots.capacity() != 144 || slots.experts() != 512 {
        return Err(reject());
    }
    let mut out = Zeroizing::new([u32::MAX; 512]);
    let mut seen = [false; 144];
    for (id, cell) in out.iter_mut().enumerate() {
        if let Some(slot) = slots.resident_slot(id).map_err(|_| reject())? {
            if slot >= 144
                || seen[slot]
                || slots.resident_expert(slot).map_err(|_| reject())? != Some(id)
            {
                return Err(reject());
            }
            seen[slot] = true;
            *cell = slot as u32;
        }
    }
    Ok(out)
}

impl Workspace {
    pub(crate) fn batch_router_allocation_requests() -> [u64; 2] {
        [128 * 1024, 2 * 1024 * 1024]
    }
    pub(crate) fn reserve_batch_router(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
    ) -> Result<()> {
        allocation.check(&self.admission)?;
        if self.batch_router.is_some()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || self.prepared_verify3.is_some()
            || self.layout.tile != 8
            || self.layout.capacity != 144
            || *self.policy.chunk != 0
        {
            return Err(reject());
        }
        self.batch_router = Some(Owners {
            ids: Zeroizing::new([
                arena.allocate_zeroed(48 * 512).map_err(u::fault)?,
                arena.allocate_zeroed(gpu::AUDIT_WORDS).map_err(u::fault)?,
            ]),
            table: None,
            instances: Vec::new(),
        });
        Ok(())
    }
    pub(crate) fn batch_router_active(&self) -> bool {
        self.batch_router
            .as_ref()
            .is_some_and(|r| r.table.is_some())
    }
    pub(crate) fn batch_router_ids(&self) -> Option<[BufferId; 2]> {
        self.batch_router.as_ref().map(|r| *r.ids)
    }
    pub(crate) fn activate_batch_router(
        &mut self,
        e: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
    ) -> Result<()> {
        self.healthy(arena)?;
        self.policy.completed_boundary()?;
        let owners = self.batch_router.as_ref().ok_or(reject())?;
        if owners.table.is_some()
            || !e.target_rows_reserved
            || e.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || e.adaptive.is_some()
            || e.secondary.is_some()
            || e.resident_banks_frozen
            || e.batch_moe.is_some()
            || !arena.public_owned_zero_audit_enabled()
        {
            return Err(reject());
        }
        let mut table = Zeroizing::new(vec![0u32; 32768]);
        let mut instances = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = weights(graph, layer)?;
            let bank = e.banks.get(weights.gate.info().name()).ok_or(reject())?;
            bank.check_names(weights)?;
            bank.check_client(arena.client().map_err(u::fault)?)?;
            table[layer * 512..(layer + 1) * 512].copy_from_slice(&*slot_table(&bank.slots)?);
            instances.push(bank.instance());
        }
        let mut bytes = Zeroizing::new(Vec::with_capacity(128 * 1024));
        for word in table.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        arena
            .client()
            .map_err(u::fault)?
            .write_private(
                arena.handle(owners.ids[0]).map_err(u::fault)?.clone(),
                std::mem::take(&mut *bytes),
            )
            .map_err(u::fault)?;
        let owners = self.batch_router.as_mut().ok_or(reject())?;
        owners.instances = instances;
        owners.table = Some(table);
        Ok(())
    }
    pub(crate) fn begin_batch_router_chunk(
        &self,
        arena: &PrivateArena,
        start: u64,
        count: usize,
    ) -> Result<()> {
        if !self.batch_router_active() {
            return Ok(());
        }
        if !(1..=3).contains(&count) || start.checked_add(count as u64).is_none_or(|x| x > 13) {
            return Err(reject());
        }
        u::zero(arena, self.batch_router.as_ref().ok_or(reject())?.ids[1])
    }
    pub(crate) fn resolve_batch_router_chunk(
        &self,
        e: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        s: &Session,
        pending: &mut [Option<Pending>; 48],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !self.batch_router_active() {
            return Ok(());
        }
        let owners = self.batch_router.as_ref().ok_or(reject())?;
        let table = owners.table.as_ref().ok_or(reject())?;
        let handle = arena.handle(owners.ids[1]).map_err(u::fault)?.clone();
        // The complete mapped lease drops before the admission's mandatory
        // owning I/O audit. Every decoded route owner wipes on drop.
        let routes = crate::resident_model::batch_io::read_target_output(
            arena,
            &self.admission,
            handle,
            cancel,
            || Ok(()),
            |bytes| {
                // Reserve before decoding private route owners. Growing this
                // vector afterwards could leave un-wiped relocation copies.
                let mut routes = Vec::new();
                routes.try_reserve_exact(48).map_err(|_| reject())?;
                if routes.capacity() != 48 {
                    return Err(reject());
                }
                for layer in 0..48 {
                    routes.push(gpu::audit_frame(
                        bytes,
                        table,
                        layer,
                        s.stamp.start,
                        s.stamp.count,
                    )?);
                }
                Ok(routes)
            },
        )?;
        if routes.len() != 48 {
            return Err(reject());
        }
        for (layer, routes) in routes.iter().enumerate() {
            if routes.count() != s.stamp.count {
                return Err(reject());
            }
            let p = pending[layer].as_mut().ok_or(reject())?;
            if p.deferred_router.as_deref() != Some(&self.identity)
                || *p.stamp != *s.stamp
                || *p.layer != layer
                || !p.armed
                || p.staged.is_some()
                || p.last.is_some()
            {
                return Err(reject());
            }
            let bank = e.banks.get(&p.key).ok_or(reject())?;
            self.check_batch_router_bank(e, graph, arena, layer)?;
            let mut staged = StagedSlots::begin(&bank.slots).map_err(|_| reject())?;
            let mut selected = Zeroizing::new([0usize; 80]);
            for (i, &(id, _)) in routes.pairs().iter().enumerate() {
                selected[i] = id as usize;
            }
            let len = routes.groups();
            let plan = staged
                .plan(&bank.slots, &selected[..len])
                .map_err(|_| reject())?;
            if !plan.uploads.is_empty()
                || plan.slots.len() != len
                || plan
                    .slots
                    .iter()
                    .zip(routes.pairs())
                    .any(|(&slot, &(_, actual))| slot != actual as usize)
            {
                return Err(reject());
            }
            p.staged = Some(staged);
            p.last = Some(Last {
                pending: plan,
                selected,
                len: Zeroizing::new(len),
            });
            p.deferred_router = None;
        }
        Ok(())
    }
    fn check_batch_router_bank(
        &self,
        e: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        layer: usize,
    ) -> Result<()> {
        let owners = self.batch_router.as_ref().ok_or(reject())?;
        let table = owners.table.as_ref().ok_or(reject())?;
        let weights = weights(graph, layer)?;
        let bank = e.banks.get(weights.gate.info().name()).ok_or(reject())?;
        if layer >= 48
            || owners.instances.get(layer) != Some(&bank.instance())
            || table[layer * 512..(layer + 1) * 512] != *slot_table(&bank.slots)?
        {
            return Err(reject());
        }
        bank.check_names(weights)?;
        bank.check_client(arena.client().map_err(u::fault)?)?;
        Ok(())
    }
}

impl GpuExecutor<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_batch_router_moe(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        router: BufferId,
        views: &BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        guarded(&w.poison.clone(), || {
            u::cancel(cancel)?;
            w.check(self, s, arena)?;
            w.check_batch_router_bank(self, graph, arena, layer)?;
            if !(1..=3).contains(&views.count())
                || s.stamp.count != views.count()
                || views.input_first() != 0
                || views.input_stride() != 2560
                || views.output_first() != 0
                || views.output_stride() != 2560
                || views.packed_first() != 0
            {
                return Err(reject());
            }
            let bank = self
                .banks
                .get(weights(graph, layer)?.gate.info().name())
                .ok_or(reject())?;
            let ids = w.ids(input, output);
            let g = w.layout.geometries[layer];
            let spans = k::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
            let mut words = [0usize; 7];
            for (i, &id) in ids.iter().enumerate() {
                words[i] = u::words(arena, id)?;
            }
            spans.validate(g, words).map_err(err)?;
            physical_with_packet(arena, bank, g, &ids, s.stamp.fault, w, Some(router))?;
            let owners = w.batch_router.as_ref().ok_or(reject())?;
            let handle = |id| arena.handle(id).cloned().map_err(u::fault);
            // Router source is additionally included in complete physical
            // metadata checking with all private roots and immutable banks.
            let hs = [
                handle(router)?,
                handle(owners.ids[0])?,
                handle(ids[MAP])?,
                handle(ids[PROB])?,
                handle(owners.ids[1])?,
                handle(s.stamp.fault)?,
            ];
            w.policy.issue(layer)?;
            u::zero_many(arena, &*w.ids)?;
            gpu::launch(
                arena.client().map_err(u::fault)?,
                hs,
                layer,
                s.stamp.start,
                views.count(),
            )?;
            // Fixed maximum geometry is independent of private group count.
            // Existing kernels guard every group using the device MAP header.
            launch_wave(arena, bank, g, &spans, &ids, 30)?;
            launch_finite_for_count(arena, g, &spans, &ids, views.count())?;
            fold(
                arena,
                g,
                &ids,
                s.stamp.fault,
                views.count(),
                0,
                views.count(),
                false,
            )?;
            launch_reduce(arena, g, &spans, &ids, views.count())?;
            launch_finite_for_count(arena, g, &spans, &ids, views.count())?;
            fold(
                arena,
                g,
                &ids,
                s.stamp.fault,
                views.count(),
                0,
                views.count(),
                true,
            )?;
            self.expert_hits = self
                .expert_hits
                .checked_add((views.count() * 10) as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            self.gpu_router_calls = self
                .gpu_router_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            self.moe_calls = self
                .moe_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(Pending {
                views: Views {
                    ids: Zeroizing::new([input, output]),
                    offsets: Zeroizing::new([0, 2560, 0, 2560]),
                },
                stamp: Zeroizing::new((*s.stamp).clone()),
                layer: Zeroizing::new(layer),
                key: weights(graph, layer)?.gate.info().name().to_owned(),
                instance: bank.instance(),
                pins: [bank.gate(), bank.up(), bank.down()]
                    .into_iter()
                    .flat_map(|b| b.segments().iter().map(|p| p.handle().clone()))
                    .collect(),
                staged: None,
                last: None,
                count1: None,
                frozen_sources: None,
                deferred_router: Some(Zeroizing::new(w.identity)),
                waves: Zeroizing::new(1),
                poison: w.poison.clone(),
                armed: true,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_table_uses_actual_original_slots_and_marks_missing_experts() {
        let mut slots = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        let ids = [500, 2, 72, 71, 128];
        let plan = slots.plan(&ids).unwrap();
        let actual = plan.slots.clone();
        slots.commit(plan).unwrap();
        let table = slot_table(&slots).unwrap();
        for (&id, slot) in ids.iter().zip(actual) {
            assert_eq!(table[id], slot as u32);
        }
        assert_eq!(table.iter().filter(|&&x| x < 144).count(), ids.len());
        for id in [0, 1, 3, 511] {
            assert_eq!(table[id], u32::MAX);
        }
        assert!(slot_table(&crate::bank_slots::BankSlots::new(72, 512).unwrap()).is_err());
    }
}
