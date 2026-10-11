//! Authenticated scoped CPU reads of SAME original registered HOST pages.
//! No payload copy/read, bare source handle interface or normal-model policy.
use super::*;
use crate::{private_arena::BufferId, quant_matvec::grouped_moe as k};
use cubecl::prelude::{ArrayArg, CubeDim};
use cubecl::server::Handle;
use cubecl_runtime::storage::ManagedResource;
use kuno_text::{
    TensorView,
    cpu_expert::{PackedExperts, batch::ExpertView},
    gguf::PositionalReader,
};
use std::{cell::Cell, marker::PhantomData};
use zeroize::{Zeroize, Zeroizing};

const PAGE_COUNT: usize = 9;
const RESOURCE_COUNT: usize = PAGE_COUNT + 7;
const CHECK_HOST_BYTES: usize = 8192;
const FIRST: [usize; 3] = [144, 288, 432];
const PARTS: [usize; 3] = [144, 144, 80];

fn resource_peak(count: usize) -> Result<usize> {
    let bytes = count
        .checked_mul(size_of::<Handle>() + size_of::<ManagedResource<WgpuResource>>())
        .ok_or(Error::ArithmeticOverflow)?;
    if count > RESOURCE_COUNT || bytes > CHECK_HOST_BYTES {
        return Err(rejected());
    }
    Ok(bytes)
}
fn source_index(layer: usize, role: usize, page: usize) -> Result<usize> {
    if layer >= plan::LAYERS || role >= 3 || page >= 3 {
        return Err(rejected());
    }
    Ok(layer * PAGE_COUNT + role * 3 + page)
}
fn original_expert(range: &plan::Range<'_>, page: plan::Page) -> Result<usize> {
    if range.bytes() != page.expert_bytes()
        || range.bytes() == 0
        || !range.source_offset().is_multiple_of(range.bytes())
    {
        return Err(rejected());
    }
    let expert = usize::try_from(range.source_offset() / range.bytes())
        .map_err(|_| Error::ArithmeticOverflow)?;
    if expert >= plan::EXPERTS {
        return Err(rejected());
    }
    Ok(expert)
}
fn same_tensor(a: TensorView<'_>, b: TensorView<'_>, model: &GgufFile) -> bool {
    a.belongs_to(model) && b.belongs_to(model) && std::ptr::eq(a.info(), b.info())
}
fn check_full_handle(handle: &Handle, bytes: u64) -> Result<()> {
    if bytes == 0
        // ComputeClient::empty's contiguous owner represents zero offsets as
        // Some(0). Prove values, not their optional representation. Reject
        // forged offsets before size_in_used's unchecked subtraction.
        || handle.offset_start.unwrap_or(0) != 0
        || handle.offset_end.unwrap_or(0) != 0
        || handle.size() != bytes
        || handle.size_in_used() != bytes
    {
        return Err(Error::InvalidFormat(
            "public host CPU borrow full owner rejected",
        ));
    }
    Ok(())
}

/// Exclusive Attachment anchor, normal CPU metadata and nine private maps.
/// The retained model/workspace borrows forbid reset/fill while views live.
/// No Clone/Debug/serialization, raw source Handle or mutable byte interface.
pub(crate) struct CpuLayerBorrow<'borrow, 'reader, 'model, 'file> {
    prepared: &'borrow PreparedPlan<'model, 'file>,
    weights: PackedExperts<'reader, 'file>,
    maps: mirror::ReadOnlyLayer<'borrow>,
    owners: [&'borrow PublicExternalHandle; PAGE_COUNT],
    client: &'borrow ComputeClient<WgpuRuntime>,
    _executor: &'borrow GpuExecutor<'file>,
    _workspace: &'borrow batch_moe::Workspace,
    _exclusive: PhantomData<&'borrow mut ()>,
    layer: Zeroizing<usize>,
    geometry: k::Geometry,
    gpu_armed: Cell<bool>,
}

impl<'plan, 'model, 'file> Attachment<'plan, 'model, 'file> {
    #[allow(clippy::too_many_arguments)] // Existing actual source/boundary identities, no caller bytes.
    pub(crate) fn borrow_cpu_layer<'borrow, 'reader>(
        &'borrow mut self,
        prepared: &'borrow PreparedPlan<'model, 'file>,
        reader: &'reader PositionalReader<'file>,
        executor: &'borrow GpuExecutor<'file>,
        workspace: &'borrow batch_moe::Workspace,
        layer: usize,
        cancel: &AtomicBool,
    ) -> Result<CpuLayerBorrow<'borrow, 'reader, 'model, 'file>> {
        u::cancel(cancel)?;
        if self.phase != Phase::Attached
            || layer >= plan::LAYERS
            || self.owners.len() != mirror::PAGES
            || !reader.belongs_to(prepared.model)
            || !workspace.frozen_host_sources_enabled()
            || self.sources.as_ref().ok_or_else(rejected)?.plan_identity()
                != prepared.plan.identity()
            || self.budget != prepared.budget
        {
            return Err(rejected());
        }
        // This executor-owned helper reconstructs ACTUAL48 bank references.
        // Pre-attachment completed_banks intentionally rejects attached state.
        executor.check_public_host_cpu_borrow(&prepared.plan, prepared.graph, &self.client)?;
        if workspace.frozen_host_source_bank_fill(executor, prepared.graph)?
            != prepared.plan.bank_fill()?
        {
            return Err(rejected());
        }
        reader.verify_file_unchanged()?;
        let mirror = self.mirror.as_ref().ok_or_else(rejected)?;
        let first = self.owners[source_index(layer, 0, 0)?]
            .as_ref()
            .ok_or_else(rejected)?;
        let mut owners = [first; PAGE_COUNT];
        for (i, owner) in owners.iter_mut().enumerate() {
            *owner = self.owners[source_index(layer, i / 3, i % 3)?]
                .as_ref()
                .ok_or_else(rejected)?;
        }
        resource_peak(PAGE_COUNT)?; // before transient validation vectors
        let handles: Vec<_> = owners.iter().map(|owner| owner.handle()).collect();
        if handles.capacity() != PAGE_COUNT || handles.iter().any(|h| h.stream != handles[0].stream)
        {
            return Err(rejected());
        }
        let actual = u::resources(&self.client, handles)?;
        resource_peak(actual.capacity())?;
        if actual.len() != PAGE_COUNT {
            return Err(rejected());
        }
        for (i, (owner, actual)) in owners.iter().zip(&actual).enumerate() {
            let page = prepared.plan.page(layer, ROLES[i / 3], i % 3)?;
            let handle = owner.handle();
            check_full_handle(&handle, page.logical_bytes())?;
            mirror
                .check_readonly_page(
                    &prepared.plan,
                    layer,
                    ROLES[i / 3],
                    i % 3,
                    &handle.binding(),
                    actual.resource(),
                )
                .map_err(Error::InvalidFormat)?;
        }
        if actual.iter().enumerate().any(|(i, a)| {
            actual[..i]
                .iter()
                .any(|old| old.resource().buffer == a.resource().buffer)
        }) {
            return Err(rejected());
        }
        let tensors = [
            prepared.plan.range(layer, Role::Gate, 0)?.tensor(),
            prepared.plan.range(layer, Role::Up, 0)?.tensor(),
            prepared.plan.range(layer, Role::Down, 0)?.tensor(),
        ];
        let projections = tensors.map(|tensor| {
            let d = tensor.info().dimensions();
            if d.len() != 3 {
                return Err(rejected());
            }
            k::Projection::new(
                tensor.info().dtype().gguf_id(),
                usize::try_from(d[0]).map_err(|_| Error::ArithmeticOverflow)?,
                usize::try_from(d[1]).map_err(|_| Error::ArithmeticOverflow)?,
                usize::try_from(d[2]).map_err(|_| Error::ArithmeticOverflow)?,
            )
            .map_err(Error::InvalidFormat)
        });
        let [gate, up, down] = projections;
        let geometry = k::Geometry::new([gate?, up?, down?], 8, plan::VRAM_SLOTS)
            .and_then(k::Geometry::with_frozen_host_sources)
            .and_then(k::Geometry::with_inactive_partition_exit)
            .map_err(Error::InvalidFormat)?;
        let weights = PackedExperts::new(reader, tensors[0], tensors[1], tensors[2])?;
        // Both fixed guard/header constituents were precharged by construction.
        attachment_metadata(self.owners.capacity(), self.receipts.capacity())?;
        u::cancel(cancel)?;
        let maps = mirror
            .borrow_readonly_layer(layer)
            .map_err(Error::InvalidFormat)?;
        Ok(CpuLayerBorrow {
            prepared,
            weights,
            maps,
            owners,
            client: &self.client,
            _executor: executor,
            _workspace: workspace,
            _exclusive: PhantomData,
            layer: Zeroizing::new(layer),
            geometry,
            gpu_armed: Cell::new(false),
        })
    }
}

impl CpuLayerBorrow<'_, '_, '_, '_> {
    pub(crate) fn weights(&self) -> &PackedExperts<'_, '_> {
        &self.weights
    }
    pub(crate) fn geometry(&self) -> k::Geometry {
        self.geometry
    }
    pub(crate) fn placements(&self) -> Result<[mirror::Placement; PAGE_COUNT]> {
        self.maps.placements().map_err(Error::InvalidFormat)
    }
    pub(crate) fn expert_for_host_ordinal(&self, ordinal: usize) -> Result<usize> {
        let mut expert = None;
        for role in ROLES {
            let range = self.prepared.plan.range(*self.layer, role, ordinal)?;
            let page = self.prepared.plan.page(*self.layer, role, range.page())?;
            let id = original_expert(&range, page)?;
            if expert.is_some_and(|old| old != id) {
                return Err(rejected());
            }
            let address = self.prepared.plan.address(*self.layer, id)?;
            if address.tier() != plan::Tier::Host
                || address.virtual_slot() != plan::VRAM_SLOTS + ordinal
                || address.page() != range.page()
                || range.destination_offset() != address.local_slot() as u64 * page.expert_bytes()
            {
                return Err(rejected());
            }
            expert = Some(id);
        }
        expert.ok_or_else(rejected)
    }
    pub(crate) fn resident_expert(&self, slot: usize) -> Result<usize> {
        if slot >= plan::VRAM_SLOTS {
            return Err(rejected());
        }
        let mut found = None;
        for id in 0..plan::EXPERTS {
            let address = self.prepared.plan.address(*self.layer, id)?;
            if address.tier() == plan::Tier::Vram && address.virtual_slot() == slot {
                if found.is_some() {
                    return Err(rejected());
                }
                found = Some(id);
            }
        }
        found.ok_or_else(rejected)
    }
    pub(crate) fn expert(&self, id: usize) -> Result<ExpertView<'_>> {
        let address = self.prepared.plan.address(*self.layer, id)?;
        if address.tier() != plan::Tier::Host || address.virtual_slot() < plan::VRAM_SLOTS {
            return Err(rejected());
        }
        let ordinal = address.virtual_slot() - plan::VRAM_SLOTS;
        let mut spans: [&[u8]; 3] = [&[]; 3];
        for (i, role) in ROLES.into_iter().enumerate() {
            let range = self.prepared.plan.range(*self.layer, role, ordinal)?;
            let page = self.prepared.plan.page(*self.layer, role, range.page())?;
            let original = self.prepared.plan.range(*self.layer, role, 0)?.tensor();
            if original_expert(&range, page)? != id
                || range.page() != address.page()
                || !same_tensor(range.tensor(), original, self.prepared.model)
                || range.tensor().info().dtype().gguf_id() != page.dtype()
                || range.tensor().info().row_bytes() != page.row_bytes()
                || range.destination_offset() != address.local_slot() as u64 * page.expert_bytes()
            {
                return Err(rejected());
            }
            spans[i] = self
                .maps
                .span(
                    role,
                    range.page(),
                    range.destination_offset(),
                    range.bytes(),
                )
                .map_err(Error::InvalidFormat)?;
        }
        self.weights.borrow_expert(id, spans)
    }
    fn join(&self) -> Result<()> {
        cubecl_common::future::block_on(self.client.sync()).map_err(|_| backend_error())?;
        self.gpu_armed.set(false);
        Ok(())
    }

    /// Original, HOST-only partial control. Source handles are internal, private
    /// owners remain arena-bound, and success requires actual joined GPU work.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_original_gpu_partials(
        &self,
        arena: &PrivateArena,
        spans: &k::Spans,
        ids: [BufferId; 7],
        groups: usize,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        if !(1..=3).contains(&count)
            || groups == 0
            || groups > count * k::RANKS
            || !std::ptr::eq(arena.client().map_err(u::fault)?.info(), self.client.info())
        {
            return Err(rejected());
        }
        let mut words = Zeroizing::new([0usize; 7]);
        resource_peak(RESOURCE_COUNT)?;
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(RESOURCE_COUNT)
            .map_err(|_| rejected())?;
        if handles.capacity() != RESOURCE_COUNT {
            return Err(rejected());
        }
        for owner in self.owners {
            handles.push(owner.handle());
        }
        for (i, id) in ids.into_iter().enumerate() {
            if ids[..i].contains(&id) {
                return Err(rejected());
            }
            words[i] = u::words(arena, id)?;
            let full = arena.handle(id).map_err(u::fault)?.clone();
            check_full_handle(
                &full,
                arena.layout(id).map_err(u::fault)?.owned_bytes as u64,
            )?;
            if full.stream != handles[0].stream {
                return Err(rejected());
            }
            handles.push(full);
        }
        spans
            .validate(self.geometry, *words)
            .map_err(Error::InvalidFormat)?;
        let actual = u::resources(self.client, handles)?;
        resource_peak(actual.capacity())?;
        if actual.len() != RESOURCE_COUNT
            || actual.iter().enumerate().any(|(i, a)| {
                actual[..i]
                    .iter()
                    .any(|b| a.resource().buffer == b.resource().buffer)
            })
        {
            return Err(rejected());
        }
        for i in 0..PAGE_COUNT {
            let h = self.owners[i].handle();
            // Registration authority was checked before map. Recheck the same
            // physical page on this stream immediately before consumption.
            let page = self.prepared.plan.page(*self.layer, ROLES[i / 3], i % 3)?;
            check_full_handle(&h, page.logical_bytes())?;
            let r = actual[i].resource();
            self.maps
                .check_resource(ROLES[i / 3], i % 3, &h.binding(), r)
                .map_err(Error::InvalidFormat)?;
            if r.offset != 0 || r.size != page.logical_bytes() || r.buffer.size() != r.size {
                return Err(rejected());
            }
        }
        for (i, id) in ids.into_iter().enumerate() {
            let r = actual[PAGE_COUNT + i].resource();
            let bytes = arena.layout(id).map_err(u::fault)?.owned_bytes as u64;
            if r.offset != 0 || r.size != bytes || r.buffer.size() != bytes {
                return Err(rejected());
            }
        }
        self.gpu_armed.set(true); // before first enqueue; Drop joins on unwind
        let result = self.enqueue_original(arena, spans, &ids, groups, cancel);
        let joined = self.join(); // including partial enqueue/error
        result.and(joined)
    }
    fn enqueue_original(
        &self,
        arena: &PrivateArena,
        s: &k::Spans,
        ids: &[BufferId; 7],
        groups: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let [gate, up, down] = self.geometry.projections();
        for page in 0..3 {
            u::cancel(cancel)?;
            let ga = self.owners[page].handle();
            let uph = self.owners[3 + page].handle();
            let gp = self.prepared.plan.page(*self.layer, Role::Gate, page)?;
            let upp = self.prepared.plan.page(*self.layer, Role::Up, page)?;
            // SAFETY: complete actual typed owners, original shapes, same stream
            // and no aliases authenticated above; source bytes are immutable.
            unsafe {
                k::grouped_gate_up_frozen::launch_unchecked::<WgpuRuntime>(
                    self.client,
                    self.geometry
                        .grid(groups, false)
                        .map_err(Error::InvalidFormat)?,
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(ga, gp.logical_bytes() as usize / 4),
                    ArrayArg::from_raw_parts(uph, upp.logical_bytes() as usize / 4),
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
                    PARTS[page],
                    self.geometry.hidden(),
                    self.geometry.ff(),
                    gate.dtype(),
                    up.dtype(),
                    gate.row_bytes(),
                    up.row_bytes(),
                    self.geometry.tile(),
                    false,
                );
            }
        }
        for page in 0..3 {
            u::cancel(cancel)?;
            let downh = self.owners[6 + page].handle();
            let p = self.prepared.plan.page(*self.layer, Role::Down, page)?;
            // SAFETY: same original guarded source/private owner proof as GU.
            unsafe {
                k::grouped_down_frozen::launch_unchecked::<WgpuRuntime>(
                    self.client,
                    self.geometry
                        .grid(groups, true)
                        .map_err(Error::InvalidFormat)?,
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(downh, p.logical_bytes() as usize / 4),
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
                    PARTS[page],
                    self.geometry.hidden(),
                    self.geometry.ff(),
                    down.dtype(),
                    down.row_bytes(),
                    self.geometry.tile(),
                );
            }
        }
        Ok(())
    }
}
impl Drop for CpuLayerBorrow<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if self.gpu_armed.get() {
            let _ = self.join();
        }
        // Synchronous CPU leaf already joined. Source allocation ownership is
        // still Attachment's; readonly unmap is not a detach/free receipt.
        self.layer.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    use cubecl_runtime::{
        allocator::ContiguousMemoryLayoutPolicy,
        server::{MemoryLayoutDescriptor, MemoryLayoutPolicy},
    };
    #[test]
    fn full_owner_accepts_all_zero_offset_representations() {
        let owner = Handle::new(StreamId { value: 23 }, 32768);
        for start in [None, Some(0)] {
            for end in [None, Some(0)] {
                let mut full = owner.clone();
                full.offset_start = start;
                full.offset_end = end;
                check_full_handle(&full, 32768).unwrap();
                assert_eq!(full.size_in_used(), owner.size());
            }
        }
    }
    #[test]
    fn full_owner_checks_actual_empty_layout_and_rejects_subspans() {
        let stream = StreamId { value: 23 };
        let policy = ContiguousMemoryLayoutPolicy::new(256);
        // This is ComputeClient::empty's actual descriptor/apply path, without
        // creating a device. The private arena owns this complete layout.
        let descriptor = MemoryLayoutDescriptor::contiguous([32768].into(), 1);
        let (owner, mut layouts) = policy.apply(stream, &[descriptor]);
        let full = layouts.remove(0).memory;
        assert_eq!(full.offset_start, Some(0));
        assert_eq!(full.offset_end, Some(0));
        check_full_handle(&full, 32768).unwrap();
        assert_eq!(full.size(), owner.size());

        let padded = MemoryLayoutDescriptor::contiguous([257].into(), 1);
        let (owner, mut layouts) = policy.apply(stream, &[padded]);
        let subspan = layouts.remove(0).memory;
        assert_eq!(subspan.offset_end, Some(255));
        assert!(check_full_handle(&subspan, 257).is_err());
        assert!(check_full_handle(&subspan, owner.size()).is_err());

        let descriptors = [
            MemoryLayoutDescriptor::contiguous([256].into(), 1),
            MemoryLayoutDescriptor::contiguous([256].into(), 1),
        ];
        let (owner, layouts) = policy.apply(stream, &descriptors);
        for subspan in layouts {
            assert!(check_full_handle(&subspan.memory, 256).is_err());
            assert!(check_full_handle(&subspan.memory, owner.size()).is_err());
        }
    }
    #[test]
    fn full_owner_rejects_nonzero_overflowing_and_wrong_sized_handles() {
        let owner = Handle::new(StreamId { value: 23 }, 32768);
        assert!(check_full_handle(&owner, 0).is_err());
        assert!(check_full_handle(&owner, 32767).is_err());
        assert!(check_full_handle(&owner, 32769).is_err());
        for value in [1, 32768, 32769, u64::MAX] {
            let mut subspan = owner.clone();
            subspan.offset_start = Some(value);
            assert!(check_full_handle(&subspan, 32768).is_err());
            subspan.offset_start = None;
            subspan.offset_end = Some(value);
            assert!(check_full_handle(&subspan, 32768).is_err());
        }
    }
    #[test]
    fn coordinates_are_exact_and_checked_before_owner_access() {
        for layer in 0..48 {
            for role in 0..3 {
                for page in 0..3 {
                    assert_eq!(
                        source_index(layer, role, page).unwrap(),
                        layer * 9 + role * 3 + page
                    );
                }
            }
        }
        for values in [(48, 0, 0), (0, 3, 0), (0, 0, 3), (usize::MAX, 0, 0)] {
            assert!(source_index(values.0, values.1, values.2).is_err());
        }
    }
    #[test]
    fn transient_validation_and_fixed_guards_fit_existing_caps() {
        assert!(resource_peak(PAGE_COUNT).unwrap() <= CHECK_HOST_BYTES);
        assert!(resource_peak(RESOURCE_COUNT).unwrap() <= CHECK_HOST_BYTES);
        assert!(resource_peak(RESOURCE_COUNT + 1).is_err());
        assert!(resource_peak(usize::MAX).is_err());
        assert!(
            attachment_metadata(mirror::PAGES, mirror::PAGES).unwrap() <= ATTACHMENT_HOST_BYTES
        );
        assert_eq!(FIRST, [144, 288, 432]);
        assert_eq!(PARTS.iter().sum::<usize>(), plan::HOST_SLOTS);
    }
}
