//! Opaque, default-unused original three-row public operator enqueue.
//! Saved Sources authenticates physical roots; no Handles escape to the probe.
use super::*;
use crate::{
    executor::batch_moe::{Workspace, frozen_sources::Sources},
    public_host_expert_plan::{self as host, Role},
    quant_matvec::grouped_moe as k,
    resident_util as u,
};
use cubecl::{
    prelude::{ArrayArg, CubeDim},
    server::Handle,
};
use cubecl_runtime::public_external::PublicExternalHandle;
const SOURCES: usize = 15;
const PRIVATE: usize = 7;
const METADATA_CAP: usize = 8192;
const ORIGINAL_BANK_CAPACITY: usize = crate::daslab_profile::BANK_SLOTS;
const FIRST: [usize; 5] = [0, 72, 144, 288, 432];
const COUNTS: [usize; 5] = [72, 72, 144, 144, 80];
fn reject_overlap() -> Error {
    Error::InvalidFormat("public CPU overlap source owners rejected")
}
fn full(h: &Handle, logical: usize) -> Result<()> {
    if h.offset_start.unwrap_or(0) != 0
        || h.offset_end.unwrap_or(0) != 0
        || h.size() != logical as u64
        || h.size_in_used() != logical as u64
    {
        return Err(reject_overlap());
    }
    Ok(())
}
impl GpuExecutor<'_> {
    #[allow(clippy::too_many_arguments)] // Bounded same-owner source authority, no caller bytes.
    pub(crate) fn enqueue_public_host_cpu_overlap(
        &self,
        plan: &host::Plan<'_, '_>,
        graph: BoundGraph<'_, '_>,
        source: &Sources,
        workspace: &Workspace,
        layer: usize,
        host: [&PublicExternalHandle; 9],
        arena: &PrivateArena,
        s: &k::Spans,
        ids: [BufferId; 7],
        groups: usize,
        all: bool,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        self.check_public_host_cpu_borrow(plan, graph, arena.client().map_err(u::fault)?)?;
        if layer != 0
            || groups == 0
            || groups > 30
            || source.plan_identity() != plan.identity()
            || !workspace.frozen_host_sources_enabled()
            || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id))
        {
            return Err(reject_overlap());
        }
        let bank = self
            .banks
            .get(&format!("blk.{layer}.ffn_gate_exps.weight"))
            .ok_or_else(reject_overlap)?;
        let g = k::Geometry::new(
            [
                k::Projection::new(
                    graph
                        .tensor(bank.source_names()[0])?
                        .info()
                        .dtype()
                        .gguf_id(),
                    2560,
                    640,
                    512,
                )
                .map_err(Error::InvalidFormat)?,
                k::Projection::new(
                    graph
                        .tensor(bank.source_names()[1])?
                        .info()
                        .dtype()
                        .gguf_id(),
                    2560,
                    640,
                    512,
                )
                .map_err(Error::InvalidFormat)?,
                k::Projection::new(
                    graph
                        .tensor(bank.source_names()[2])?
                        .info()
                        .dtype()
                        .gguf_id(),
                    640,
                    2560,
                    512,
                )
                .map_err(Error::InvalidFormat)?,
            ],
            8,
            ORIGINAL_BANK_CAPACITY,
        )
        .and_then(k::Geometry::with_frozen_host_sources)
        .and_then(k::Geometry::with_inactive_partition_exit)
        .map_err(Error::InvalidFormat)?;
        let words = ids
            .map(|id| u::words(arena, id))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        s.validate(g, words.try_into().map_err(|_| reject_overlap())?)
            .map_err(Error::InvalidFormat)?;
        let expected = arena.handle(ids[0]).map_err(u::fault)?;
        source.validate_source_layer(&self.client, layer, bank, expected)?;
        let mut handles = Vec::with_capacity(SOURCES + PRIVATE);
        for (role, b) in [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .enumerate()
        {
            if b.segments().len() != 2 {
                return Err(reject_overlap());
            }
            for (page, seg) in b.segments().iter().enumerate() {
                if seg.first_slot() != FIRST[page] || seg.capacity() != COUNTS[page] {
                    return Err(reject_overlap());
                }
                full(seg.handle(), seg.words() * 4)?;
                handles.push(seg.handle().clone());
            }
            for page in 0..3 {
                let h = host[role * 3 + page].handle();
                let p = plan.page(layer, [Role::Gate, Role::Up, Role::Down][role], page)?;
                full(
                    &h,
                    usize::try_from(p.logical_bytes()).map_err(|_| reject_overlap())?,
                )?;
                handles.push(h);
            }
        }
        for id in ids {
            let h = arena.handle(id).map_err(u::fault)?.clone();
            full(&h, arena.layout(id).map_err(u::fault)?.owned_bytes)?;
            handles.push(h);
        }
        if handles.len() != SOURCES + PRIVATE
            || handles.capacity() != SOURCES + PRIVATE
            || handles.iter().any(|h| h.stream != expected.stream)
        {
            return Err(reject_overlap());
        }
        let handle_bytes = handles.capacity() * std::mem::size_of::<Handle>();
        let actual = u::resources(&self.client, handles)?;
        if actual.len() != SOURCES + PRIVATE
            || handle_bytes + actual.capacity() * std::mem::size_of_val(&actual[0]) > METADATA_CAP
        {
            return Err(reject_overlap());
        }
        for (i, a) in actual.iter().enumerate() {
            let r = a.resource();
            if r.offset != 0 || actual[..i].iter().any(|b| b.resource().buffer == r.buffer) {
                return Err(reject_overlap());
            }
            if i >= SOURCES {
                let n = arena
                    .layout(ids[i - SOURCES])
                    .map_err(u::fault)?
                    .owned_bytes as u64;
                if r.size != n || r.buffer.size() != n {
                    return Err(reject_overlap());
                }
            } else if i % 5 >= 2 {
                let n = plan
                    .page(layer, [Role::Gate, Role::Up, Role::Down][i / 5], i % 5 - 2)?
                    .logical_bytes();
                if r.size != n || r.buffer.size() != n {
                    return Err(reject_overlap());
                }
            }
        }
        let partitions = if all { 5 } else { 2 };
        let [gate, up, down] = g.projections();
        // All GU is ordered before DOWN, identical original64-lane arithmetic.
        for page in 0..partitions {
            u::cancel(cancel)?;
            let arg = |role: usize| -> Result<ArrayArg<WgpuRuntime>> {
                let h = if page < 2 {
                    [bank.gate(), bank.up(), bank.down()][role].segments()[page]
                        .handle()
                        .clone()
                } else {
                    host[role * 3 + page - 2].handle()
                };
                let words = if page < 2 {
                    [bank.gate(), bank.up(), bank.down()][role].segments()[page].words()
                } else {
                    usize::try_from(
                        plan.page(layer, [Role::Gate, Role::Up, Role::Down][role], page - 2)?
                            .logical_bytes()
                            / 4,
                    )
                    .map_err(|_| reject_overlap())?
                };
                // SAFETY: immutable captured Source roots/actual physical owners proved above.
                Ok(unsafe { ArrayArg::from_raw_parts(h, words) })
            };
            unsafe {
                k::grouped_gate_up_frozen::launch_unchecked::<WgpuRuntime>(
                    &self.client,
                    g.grid(groups, false).map_err(Error::InvalidFormat)?,
                    CubeDim::new_1d(64),
                    arg(0)?,
                    arg(1)?,
                    u::arg(arena, ids[1])?,
                    u::arg(arena, ids[0])?,
                    u::arg(arena, ids[2])?,
                    u::arg(arena, ids[6])?,
                    s.map_first(),
                    s.input_first(),
                    s.input_stride(),
                    s.act_first(),
                    s.act_stride(),
                    s.status_first(),
                    FIRST[page],
                    COUNTS[page],
                    g.hidden(),
                    g.ff(),
                    gate.dtype(),
                    up.dtype(),
                    gate.row_bytes(),
                    up.row_bytes(),
                    g.tile(),
                    false,
                );
            }
        }
        for page in 0..partitions {
            u::cancel(cancel)?;
            let (h, n) = if page < 2 {
                let seg = &bank.down().segments()[page];
                (seg.handle().clone(), seg.words())
            } else {
                (
                    host[6 + page - 2].handle(),
                    usize::try_from(plan.page(layer, Role::Down, page - 2)?.logical_bytes() / 4)
                        .map_err(|_| reject_overlap())?,
                )
            };
            unsafe {
                k::grouped_down_frozen::launch_unchecked::<WgpuRuntime>(
                    &self.client,
                    g.grid(groups, true).map_err(Error::InvalidFormat)?,
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(h, n),
                    u::arg(arena, ids[2])?,
                    u::arg(arena, ids[0])?,
                    u::arg(arena, ids[3])?,
                    u::arg(arena, ids[6])?,
                    s.map_first(),
                    s.act_first(),
                    s.act_stride(),
                    s.partial_first(),
                    s.partial_stride(),
                    s.status_first(),
                    FIRST[page],
                    COUNTS[page],
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_das_layer_zero_banks_convert_to_original_count_three_sources() {
        let types = crate::daslab_profile::routed_types(0).unwrap();
        assert_eq!(types, [18, 18, 20]);
        let projections = [
            k::Projection::new(types[0], 2560, 640, 512).unwrap(),
            k::Projection::new(types[1], 2560, 640, 512).unwrap(),
            k::Projection::new(types[2], 640, 2560, 512).unwrap(),
        ];
        assert_eq!(ORIGINAL_BANK_CAPACITY, 144);
        assert!(k::Geometry::new(projections, 8, 512).is_err());
        let banks = k::Geometry::new(projections, 8, ORIGINAL_BANK_CAPACITY).unwrap();
        assert_eq!(banks.capacity(), 144);
        assert!(!banks.frozen_host_sources());
        let source = banks
            .with_frozen_host_sources()
            .and_then(k::Geometry::with_inactive_partition_exit)
            .unwrap();
        assert_eq!(source.capacity(), 512);
        assert_eq!(source.projections(), projections);
        assert!(source.frozen_host_sources());
        assert!(source.inactive_partition_exit());
        assert_eq!(source.member_limit(), 3);
        assert!(source.validate_count(3).is_ok());
        assert!(source.validate_count(4).is_err());
        for (&first, &count) in FIRST.iter().zip(&COUNTS) {
            let words = projections.map(|p| p.segment_words(count).unwrap());
            assert!(source.validate_shard(first, count, words).is_ok());
        }
    }
}
