//! Construction-reserved fixed-public actual-score GPU routing experiment.
//! Ordinary warm13 addresses only real144 VRAM slots and rejects misses. The
//! explicit frozen-source variant addresses authenticated original512 sources
//! without extending BankSlots or publishing recency. Both variants require the
//! deferred original CPU ID/order/wire audit before any all-family commit.
use super::*;
use crate::resident_batch_router_gpu as gpu;

pub(crate) const MAIN_BYTES: u64 = gpu::NEW_TABLE_AUDIT_BYTES;
pub(crate) const HOST_BYTES: usize = 5 * 1024 * 1024;
pub(crate) const RUNTIME_BYTES: u64 = 4 * MAIN_BYTES + 48 * 4096 + 65536;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourcePolicy {
    Resident144,
    FrozenHost512 {
        identity: crate::public_host_expert_plan::PlanIdentity,
        context: usize,
    },
}
pub(super) struct Owners {
    // Construction context is immutable; activation captures the exact sealed
    // SourceSet identity once. Ordinary warm144 never accepts that source mode.
    configured_context: Option<usize>,
    four_reserved: bool,
    source_policy: Option<SourcePolicy>,
    ids: Zeroizing<[BufferId; 2]>,
    table: Option<Zeroizing<Vec<u32>>>,
    instances: Vec<expert_banks::BankInstance>,
}
fn reject() -> Error {
    err("public warm batch router rejected")
}
fn valid_window(context: usize, start: u64, count: usize) -> bool {
    (1..=4096).contains(&context)
        && (1..=3).contains(&count)
        && start
            .checked_add(count as u64)
            .is_some_and(|end| end <= context as u64)
}
fn slot_table(slots: &crate::bank_slots::BankSlots) -> Result<Zeroizing<[u32; 512]>> {
    if slots.capacity() != crate::daslab_profile::BANK_SLOTS || slots.experts() != 512 {
        return Err(reject());
    }
    let mut out = Zeroizing::new([u32::MAX; 512]);
    let mut seen = [false; crate::daslab_profile::BANK_SLOTS];
    for (id, cell) in out.iter_mut().enumerate() {
        if let Some(slot) = slots.resident_slot(id).map_err(|_| reject())? {
            if slot >= crate::daslab_profile::BANK_SLOTS
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
        self.reserve_batch_router_policy(arena, allocation, None, false)
    }
    pub(crate) fn reserve_frozen_host_batch_router(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
        context: usize,
    ) -> Result<()> {
        self.reserve_frozen_host_batch_router_with_four_rows(arena, allocation, context, false)
    }
    pub(crate) fn reserve_frozen_host_batch_router_with_four_rows(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
        context: usize,
        four_reserved: bool,
    ) -> Result<()> {
        if !(1..=4096).contains(&context) {
            return Err(reject());
        }
        self.reserve_batch_router_policy(arena, allocation, Some(context), four_reserved)
    }
    pub(super) fn four_member_router_reserved(&self) -> bool {
        self.batch_router
            .as_ref()
            .is_some_and(|r| r.four_reserved && r.configured_context.is_some())
    }
    fn reserve_batch_router_policy(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
        configured_context: Option<usize>,
        four_reserved: bool,
    ) -> Result<()> {
        allocation.check(&self.admission)?;
        if self.batch_router.is_some()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || self.prepared_verify3.is_some()
            || self.layout.tile != 8
            || self.layout.capacity != crate::daslab_profile::BANK_SLOTS
            || *self.policy.chunk != 0
        {
            return Err(reject());
        }
        self.batch_router = Some(Owners {
            configured_context,
            four_reserved,
            source_policy: None,
            ids: Zeroizing::new([
                arena.allocate_zeroed(48 * 512).map_err(u::fault)?,
                arena
                    .allocate_zeroed(if four_reserved {
                        gpu::four::AUDIT_WORDS
                    } else {
                        gpu::AUDIT_WORDS
                    })
                    .map_err(u::fault)?,
            ]),
            table: None,
            instances: Vec::new(),
        });
        Ok(())
    }
    pub(super) fn frozen_host_batch_router_reserved(&self) -> bool {
        self.batch_router
            .as_ref()
            .is_some_and(|r| r.configured_context.is_some())
    }
    pub(crate) fn frozen_host_batch_router_active(&self) -> bool {
        self.batch_router.as_ref().is_some_and(|r| {
            matches!(r.source_policy, Some(SourcePolicy::FrozenHost512 { .. })) && r.table.is_some()
        })
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
            || owners.configured_context.is_some()
            || owners.source_policy.is_some()
            || self.frozen_sources.is_some()
            || e.target_frozen_host_router_reserved
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
        owners.source_policy = Some(SourcePolicy::Resident144);
        owners.table = Some(table);
        Ok(())
    }
    pub(crate) fn activate_frozen_host_batch_router(
        &mut self,
        e: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
    ) -> Result<()> {
        self.validate_public_warm_boundary(e, graph, arena)?;
        let owners = self.batch_router.as_ref().ok_or(reject())?;
        let context = owners.configured_context.ok_or(reject())?;
        let sources = self.frozen_sources.as_ref().ok_or(reject())?;
        if owners.table.is_some()
            || owners.source_policy.is_some()
            || owners.four_reserved != sources.four_members()
            || owners.four_reserved != self.public_four_members_policy()
            || !e.target_rows_reserved
            || !e.target_batch_router_reserved
            || !e.target_frozen_host_router_reserved
            || !e.resident_banks_frozen
            || e.target_frozen_host_source_identity != Some(sources.plan_identity())
            || context > graph.geometry().max_context
            || e.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || e.batch_moe.is_some()
            || !arena.public_owned_zero_audit_enabled()
        {
            return Err(reject());
        }
        let mut table = Zeroizing::new(vec![0u32; gpu::TABLE_WORDS]);
        let mut instances = Vec::with_capacity(48);
        for layer in 0..48 {
            let weights = weights(graph, layer)?;
            let bank = e.banks.get(weights.gate.info().name()).ok_or(reject())?;
            bank.check_names(weights)?;
            sources.validate_source_layer(
                arena.client().map_err(u::fault)?,
                layer,
                bank,
                arena.handle(self.ids[0]).map_err(u::fault)?,
            )?;
            table[layer * 512..(layer + 1) * 512]
                .copy_from_slice(&*sources.slot_table(layer, bank)?);
            instances.push(bank.instance());
        }
        gpu::validate_frozen_source_table(&table)?;
        let mut bytes = Zeroizing::new(Vec::with_capacity(128 * 1024));
        for &word in table.iter() {
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
        let policy = SourcePolicy::FrozenHost512 {
            identity: sources.plan_identity(),
            context,
        };
        let owners = self.batch_router.as_mut().ok_or(reject())?;
        owners.instances = instances;
        owners.source_policy = Some(policy);
        owners.table = Some(table);
        Ok(())
    }
    pub(crate) fn refresh_strata_table(
        &mut self,
        e: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
    ) -> Result<()> {
        if !crate::strata_port::enabled() {
            return Err(reject());
        }
        self.healthy(arena)?;
        self.policy.completed_boundary()?;
        let sources = self.frozen_sources.as_ref().ok_or_else(reject)?;
        let owners = self.batch_router.as_ref().ok_or_else(reject)?;
        if !matches!(
            owners.source_policy,
            Some(SourcePolicy::FrozenHost512 { .. })
        ) || owners.table.is_none()
        {
            return Err(reject());
        }
        let mut table = Zeroizing::new(vec![0u32; gpu::TABLE_WORDS]);
        for layer in 0..48 {
            let bank = e
                .banks
                .get(weights(graph, layer)?.gate.info().name())
                .ok_or_else(reject)?;
            table[layer * 512..(layer + 1) * 512]
                .copy_from_slice(&*sources.slot_table(layer, bank)?);
        }
        gpu::validate_frozen_source_table(&table)?;
        let mut bytes = Zeroizing::new(Vec::with_capacity(128 * 1024));
        for &word in table.iter() {
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
        self.batch_router.as_mut().ok_or_else(reject)?.table = Some(table);
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
        let owners = self.batch_router.as_ref().ok_or(reject())?;
        let context = match owners.source_policy.ok_or(reject())? {
            SourcePolicy::Resident144 => 13,
            SourcePolicy::FrozenHost512 { context, .. } => context,
        };
        let four = owners.four_reserved
            && matches!(
                owners.source_policy,
                Some(SourcePolicy::FrozenHost512 { .. })
            );
        if !(valid_window(context, start, count)
            || four
                && count == 4
                && (1..=4096).contains(&context)
                && start
                    .checked_add(4)
                    .is_some_and(|end| end <= context as u64))
        {
            return Err(reject());
        }
        u::zero(arena, owners.ids[1])
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
                    routes.push(match owners.source_policy.ok_or(reject())? {
                        SourcePolicy::Resident144 => {
                            gpu::audit_frame(bytes, table, layer, s.stamp.start, s.stamp.count)?
                        }
                        SourcePolicy::FrozenHost512 { identity, .. } => {
                            let sources = self.frozen_sources.as_ref().ok_or(reject())?;
                            if identity != sources.plan_identity() {
                                return Err(reject());
                            }
                            let bank = e
                                .banks
                                .get(weights(graph, layer)?.gate.info().name())
                                .ok_or(reject())?;
                            gpu::audit_frame_frozen_sources(
                                bytes,
                                table,
                                layer,
                                s.stamp.start,
                                s.stamp.count,
                                sources.geometry(layer, bank)?,
                            )?
                        }
                    });
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
            if let Some(SourcePolicy::FrozenHost512 { identity, .. }) = owners.source_policy {
                let sources = self.frozen_sources.as_ref().ok_or(reject())?;
                let proof = p.frozen_sources.as_mut().ok_or(reject())?;
                if identity != sources.plan_identity() || !Arc::ptr_eq(sources, &proof.source) {
                    return Err(reject());
                }
                proof.resolve_audited(layer, bank, routes)?;
                p.deferred_router = None;
                continue;
            }
            if p.frozen_sources.is_some() {
                return Err(reject());
            }
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
    pub(super) fn check_batch_router_bank(
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
        if layer >= 48 || owners.instances.get(layer) != Some(&bank.instance()) {
            return Err(reject());
        }
        let expected = match owners.source_policy.ok_or(reject())? {
            SourcePolicy::Resident144 => slot_table(&bank.slots)?,
            SourcePolicy::FrozenHost512 { identity, context } => {
                let sources = self.frozen_sources.as_ref().ok_or(reject())?;
                if identity != sources.plan_identity()
                    || e.target_frozen_host_source_identity != Some(identity)
                    || !e.target_frozen_host_router_reserved
                    || !e.resident_banks_frozen
                    || context > graph.geometry().max_context
                {
                    return Err(reject());
                }
                sources.slot_table(layer, bank)?
            }
        };
        if table[layer * 512..(layer + 1) * 512] != *expected {
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
            if w.frozen_host_batch_router_active() {
                return Err(reject());
            }
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
    fn immutable_context_keeps_original_warm13_limit_and_bounds_source_count1_to3() {
        for context in [4, 13, 512, 2048, 4096] {
            for count in 1..=3 {
                assert!(valid_window(context, context as u64 - count as u64, count));
                assert!(!valid_window(
                    context,
                    context as u64 - count as u64 + 1,
                    count
                ));
            }
            assert!(!valid_window(context, 0, 0));
            assert!(!valid_window(context, 0, 4));
            assert!(!valid_window(context, u64::MAX, 1));
        }
        assert!(!valid_window(0, 0, 1));
        assert!(!valid_window(4097, 0, 1));
        assert!(!valid_window(13, 13, 1));
        assert!(valid_window(512, 13, 1));
    }
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
