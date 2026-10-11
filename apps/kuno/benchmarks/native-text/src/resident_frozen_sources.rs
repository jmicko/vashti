//! Explicit fixed-public frozen original expert sources. No fill, eviction,
//! recency update, source copy, GPU allocation or backing-release authority.
//! Parent retains the authenticated Mirror until consumers/pins are gone and
//! original joined external detach receipts have been accepted.
use super::*;
use crate::public_host_expert_mirror::PageResource;
use crate::public_host_expert_plan::{self as plan, Role};
use cubecl::wgpu::WgpuResource;
use cubecl_runtime::public_external::is_public_external_binding;

pub(crate) const EXTRA_INFO_BYTES_PER_CHUNK: u64 = 6 * 48 * 2048;
pub(crate) const SOURCE_SET_HOST_BYTES: u64 = 2 * 1024 * 1024;
pub(crate) const PHYSICAL_HOST_BYTES: u64 = 8 * 1024;
pub(crate) const EXTRA_HOST_BYTES: u64 = SOURCE_SET_HOST_BYTES + PHYSICAL_HOST_BYTES;
pub(crate) const EXTRA_RUNTIME_BYTES_PER_CHUNK: u64 = EXTRA_INFO_BYTES_PER_CHUNK;
pub(crate) const EXTRA_HOST_BYTES_PER_CHUNK: u64 = EXTRA_HOST_BYTES;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GateKernel {
    Original,
    Frozen,
    PrivateTables,
    WorkgroupTables,
    SharedFactors,
    PrivateFactors,
    WorkgroupFactors,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DownKernel {
    Original,
    Frozen,
    SharedScales,
}
// This selector is used by actual dispatch, so compound policy tests cover the
// runtime entrypoints rather than just the stored attachment flags.
fn kernel_policy(g: k::Geometry) -> (GateKernel, DownKernel) {
    let gate = if g.shared_gate_up_factors() {
        if g.workgroup_tables() { GateKernel::WorkgroupFactors }
        else if g.private_tables() { GateKernel::PrivateFactors }
        else { GateKernel::SharedFactors }
    } else if g.workgroup_tables() {
        GateKernel::WorkgroupTables
    } else if g.private_tables() {
        GateKernel::PrivateTables
    } else if g.inactive_partition_exit() {
        GateKernel::Frozen
    } else {
        GateKernel::Original
    };
    let down = if g.shared_down_scales() {
        DownKernel::SharedScales
    } else if g.inactive_partition_exit() {
        DownKernel::Frozen
    } else {
        DownKernel::Original
    };
    (gate, down)
}
const PARTS: [usize; 5] = [72, 72, 144, 144, 80];
const FIRST: [usize; 5] = [0, 72, 144, 288, 432];
const ROLES: [Role; 3] = [Role::Gate, Role::Up, Role::Down];
const PAGES: usize = 15;
const EXTENTS: usize = 8 + PAGES;
const ROUTER_EXTENTS: usize = EXTENTS + 3;
pub(crate) const ROUTER_MAIN_BYTES: u64 = super::batch_router::MAIN_BYTES;
pub(crate) const ROUTER_RUNTIME_BYTES: u64 = super::batch_router::RUNTIME_BYTES;
pub(crate) const ROUTER_HOST_BYTES: u64 = super::batch_router::HOST_BYTES as u64;
fn reject() -> Error {
    err("frozen public expert source rejected")
}
fn page_index(layer: usize, role: Role, page: usize) -> Result<usize> {
    let role = ROLES.iter().position(|&r| r == role).ok_or(reject())?;
    if layer >= 48 || page >= 3 {
        return Err(reject());
    }
    Ok(layer * 9 + role * 3 + page)
}
fn same_resource(a: &WgpuResource, b: &WgpuResource) -> bool {
    a.buffer == b.buffer && a.offset == b.offset && a.size == b.size
}
fn full_handle(h: &Handle, bytes: u64) -> Result<()> {
    if bytes == 0
        || h.size() != bytes
        || h.size_in_used() != bytes
        || h.offset_start.unwrap_or(0) != 0
        || h.offset_end.unwrap_or(0) != 0
    {
        return Err(reject());
    }
    Ok(())
}

/// Sealed witness created only from the authenticated materializer's typed
/// page. Shape/hash alone is never authority for an expert role or mapping.
pub(crate) struct RegisteredPage {
    plan_identity: plan::PlanIdentity,
    layer: Zeroizing<usize>,
    role: Role,
    page: Zeroizing<usize>,
    binding: GraphBindingId,
    handle: Handle,
    resource: WgpuResource,
    logical: u64,
    backing: u64,
}
impl RegisteredPage {
    pub(crate) fn from_mirror(
        client: &ComputeClient<WgpuRuntime>,
        plan: &plan::Plan<'_, '_>,
        witness: &PageResource,
        handle: Handle,
    ) -> Result<Self> {
        let (layer, role, page) = witness.coordinates();
        page_index(layer, role, page)?;
        if witness.binding() != plan.binding() || !witness.belongs_to(plan) {
            return Err(reject());
        }
        let expected = plan.page(layer, role, page)?;
        full_handle(&handle, expected.logical_bytes())?;
        if !is_public_external_binding(&handle.clone().binding()) {
            return Err(reject());
        }
        expected
            .check_actual_allocation(witness.backing_bytes(), witness.allocation_alignment())?;
        let metadata = u::resources(client, vec![handle.clone()])?;
        if metadata.len() != 1
            || !same_resource(metadata[0].resource(), witness.resource())
            || witness.resource().offset != 0
            || witness.resource().size != expected.logical_bytes()
            || witness.resource().buffer.size() != expected.logical_bytes()
        {
            return Err(reject());
        }
        Ok(Self {
            plan_identity: plan.identity(),
            layer: Zeroizing::new(layer),
            role,
            page: Zeroizing::new(page),
            binding: plan.binding(),
            handle,
            resource: witness.resource().clone(),
            logical: expected.logical_bytes(),
            backing: witness.backing_bytes(),
        })
    }
}
struct SourcePage {
    handle: Handle,
    resource: WgpuResource,
    words: usize,
}
struct Layer {
    instance: crate::expert_banks::BankInstance,
    origin: StagedSlots,
    by_expert: Zeroizing<Box<[u16]>>,
    geometry: k::Geometry,
    // role-major five exact full sources; never ordinary BankSlots(512).
    pages: Vec<SourcePage>,
}
pub(crate) struct Sources {
    binding: GraphBindingId,
    plan_identity: plan::PlanIdentity,
    layers: Vec<Layer>,
    metadata_bytes: usize,
}
fn initial_metadata(layers: usize, pages: usize, banks: usize) -> Result<usize> {
    let bytes = std::mem::size_of::<Sources>()
        .checked_add(2 * std::mem::size_of::<usize>() + std::mem::align_of::<Sources>() - 1)
        .and_then(|n| n.checked_add(layers.checked_mul(std::mem::size_of::<Layer>())?))
        .and_then(|n| n.checked_add(pages.checked_mul(std::mem::size_of::<RegisteredPage>())?))
        .and_then(|n| n.checked_add(banks.checked_mul(std::mem::size_of::<&ExpertBanks>())?))
        .ok_or(Error::ArithmeticOverflow)?;
    if bytes as u64 > SOURCE_SET_HOST_BYTES {
        return Err(reject());
    }
    Ok(bytes)
}
impl Sources {
    /// Parent completes all original consumers and freezes every actual full
    /// bank before this call. Only attach() at the checked completed boundary
    /// grants a native workspace authority to consume these immutable sources.
    pub(crate) fn new(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        plan: &plan::Plan<'_, '_>,
        pages: Vec<RegisteredPage>,
        reserved_host_bytes: u64,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
            false,
            false,
            false,
            false,
            false,
        )
    }
    /// Explicit matched-control candidate, frozen before any source descriptors
    /// are allocated or attached. All actual source/owner admission is shared.
    pub(crate) fn new_with_inactive_partition_exit(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        plan: &plan::Plan<'_, '_>,
        pages: Vec<RegisteredPage>,
        reserved_host_bytes: u64,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
            true,
            false,
            false,
            false,
            false,
        )
    }
    /// Public original codebooks use module-private pointer indexing rather
    /// than repeated by-value SPIR-V array spills. Explicit exit policy only.
    #[allow(dead_code)]
    pub(crate) fn new_with_private_tables(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        plan: &plan::Plan<'_, '_>,
        pages: Vec<RegisteredPage>,
        reserved_host_bytes: u64,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
            true,
            true,
            false,
            false,
            false,
        )
    }
    /// Literal-lane initialization of one shared immutable public IQ grid.
    #[allow(dead_code)]
    pub(crate) fn new_with_workgroup_tables(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        plan: &plan::Plan<'_, '_>,
        pages: Vec<RegisteredPage>,
        reserved_host_bytes: u64,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
            true,
            false,
            true,
            false,
            false,
        )
    }
    /// Explicit frozen-down IQ4_NL metadata candidate; exact original sources,
    /// admission and the exit control remain shared. No private table policy.
    #[allow(dead_code)]
    pub(crate) fn new_with_shared_down_scales(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        plan: &plan::Plan<'_, '_>,
        pages: Vec<RegisteredPage>,
        reserved_host_bytes: u64,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
            true,
            false,
            false,
            true,
            false,
        )
    }
    /// Explicit frozen IQ2/IQ3 subgroup-factor staging candidate.
    #[allow(dead_code)]
    pub(crate) fn new_with_shared_gate_up_factors(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        plan: &plan::Plan<'_, '_>,
        pages: Vec<RegisteredPage>,
        reserved_host_bytes: u64,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
            true,
            false,
            false,
            false,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_policy(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        plan: &plan::Plan<'_, '_>,
        pages: Vec<RegisteredPage>,
        reserved_host_bytes: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
    ) -> Result<Arc<Self>> {
        if (private_tables || workgroup_tables || shared_down_scales || shared_gate_up_factors)
            && !inactive_partition_exit
            || private_tables && workgroup_tables
            || reserved_host_bytes < EXTRA_HOST_BYTES
            || graph.binding_id() != plan.binding()
            || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
            || executor.bank_slots != 144
            || executor.batch_moe.is_some()
            || pages.len() != 48 * 9
        {
            return Err(reject());
        }
        graph.verify_unchanged()?;
        executor.model.verify_file_unchanged()?;
        let mut banks = Vec::with_capacity(48);
        for layer in 0..48 {
            let weight = weights(graph, layer)?;
            banks.push(
                executor
                    .banks
                    .get(weight.gate.info().name())
                    .ok_or(reject())?,
            );
        }
        let bank_array: [&ExpertBanks; 48] = banks.as_slice().try_into().map_err(|_| reject())?;
        plan.check_frozen(graph, executor.model, bank_array)?;
        let mut layers = Vec::with_capacity(48);
        let mut metadata_bytes =
            initial_metadata(layers.capacity(), pages.capacity(), banks.capacity())?;
        let mut actual_pages = pages.into_iter();
        for (layer, bank) in banks.into_iter().enumerate() {
            let original = weights(graph, layer)?;
            bank.check_names(original)?;
            bank.check_client(&executor.client)?;
            let projections = [original.gate, original.up, original.down].map(|t| {
                let d = t.info().dimensions();
                if d.len() != 3 {
                    return Err(reject());
                }
                k::Projection::new(
                    t.info().dtype().gguf_id(),
                    d[0] as usize,
                    d[1] as usize,
                    d[2] as usize,
                )
                .map_err(err)
            });
            let [gate, up, down] = projections;
            let mut geometry = k::Geometry::new([gate?, up?, down?], 8, 144)
                .and_then(k::Geometry::with_frozen_host_sources)
                .map_err(err)?;
            if inactive_partition_exit {
                geometry = geometry.with_inactive_partition_exit().map_err(err)?;
            }
            if private_tables {
                geometry = geometry.with_private_tables().map_err(err)?;
            }
            if workgroup_tables {
                geometry = geometry.with_workgroup_tables().map_err(err)?;
            }
            if shared_down_scales {
                geometry = geometry.with_shared_down_scales().map_err(err)?;
            }
            if shared_gate_up_factors {
                geometry = geometry.with_shared_gate_up_factors().map_err(err)?;
            }
            let layer_bound = StagedSlots::bounded_native_metadata_bytes(144)
                .map_err(|_| reject())?
                .checked_add(512 * std::mem::size_of::<u16>())
                .and_then(|n| n.checked_add(PAGES.checked_mul(std::mem::size_of::<SourcePage>())?))
                .ok_or(Error::ArithmeticOverflow)?;
            if metadata_bytes
                .checked_add(layer_bound)
                .is_none_or(|n| n as u64 > SOURCE_SET_HOST_BYTES)
            {
                return Err(reject());
            }
            let mut by_expert = Zeroizing::new(Box::new([u16::MAX; 512]) as Box<[u16]>);
            for expert in 0..512 {
                by_expert[expert] = u16::try_from(plan.address(layer, expert)?.virtual_slot())
                    .map_err(|_| reject())?;
            }
            check_mapping(&by_expert, &bank.slots)?;
            let mut origin = StagedSlots::begin(&bank.slots).map_err(|_| reject())?;
            let empty = origin.plan(&bank.slots, &[]).map_err(|_| reject())?;
            metadata_bytes = metadata_bytes
                .checked_add(
                    origin
                        .retained_native_metadata_bytes(&empty)
                        .map_err(|_| reject())?,
                )
                .and_then(|n| {
                    n.checked_add(by_expert.len().checked_mul(std::mem::size_of::<u16>())?)
                })
                .ok_or(Error::ArithmeticOverflow)?;
            drop(empty);
            let mut sources = Vec::with_capacity(PAGES);
            if metadata_bytes
                .checked_add(
                    sources
                        .capacity()
                        .checked_mul(std::mem::size_of::<SourcePage>())
                        .ok_or(Error::ArithmeticOverflow)?,
                )
                .is_none_or(|n| n as u64 > SOURCE_SET_HOST_BYTES)
            {
                return Err(reject());
            }
            let projections = geometry.projections();
            for (role_index, role) in ROLES.into_iter().enumerate() {
                let resident = [bank.gate(), bank.up(), bank.down()][role_index];
                if resident.segments().len() != 2 {
                    return Err(reject());
                }
                for segment in resident.segments() {
                    let handle = segment.handle().clone();
                    full_handle(&handle, segment.words() as u64 * 4)?;
                    if is_public_external_binding(&handle.clone().binding()) {
                        return Err(reject());
                    }
                    let metadata = u::resources(&executor.client, vec![handle.clone()])?;
                    if metadata.len() != 1 {
                        return Err(reject());
                    }
                    let resource = metadata[0].resource().clone();
                    check_page(
                        resource.offset,
                        resource.size,
                        resource.buffer.size(),
                        segment.words() as u64 * 4,
                        main_admission::page_bytes(segment.words() as u64 * 4)
                            .map_err(admission_error)?,
                    )?;
                    sources.push(SourcePage {
                        handle,
                        resource,
                        words: segment.words(),
                    });
                }
                for p in 0..3 {
                    let registered = actual_pages.next().ok_or(reject())?;
                    let expected = plan.page(layer, role, p)?;
                    if registered.plan_identity != plan.identity()
                        || *registered.layer != layer
                        || registered.role != role
                        || *registered.page != p
                        || registered.binding != graph.binding_id()
                        || registered.logical != expected.logical_bytes()
                        || registered.backing < registered.logical
                    {
                        return Err(reject());
                    }
                    sources.push(SourcePage {
                        handle: registered.handle,
                        resource: registered.resource,
                        words: usize::try_from(expected.logical_bytes() / 4)
                            .map_err(|_| reject())?,
                    });
                }
                for shard in 0..5 {
                    if sources[role_index * 5 + shard].words
                        != projections[role_index]
                            .segment_words(PARTS[shard])
                            .map_err(err)?
                    {
                        return Err(reject());
                    }
                }
            }
            for (i, source) in sources.iter().enumerate() {
                if source.handle.stream != sources[0].handle.stream
                    || sources[..i]
                        .iter()
                        .any(|p| p.resource.buffer == source.resource.buffer)
                    || layers.iter().any(|l: &Layer| {
                        l.pages
                            .iter()
                            .any(|p| p.resource.buffer == source.resource.buffer)
                    })
                {
                    return Err(reject());
                }
            }
            metadata_bytes = metadata_bytes
                .checked_add(
                    sources
                        .capacity()
                        .checked_mul(std::mem::size_of::<SourcePage>())
                        .ok_or(Error::ArithmeticOverflow)?,
                )
                .ok_or(Error::ArithmeticOverflow)?;
            layers.push(Layer {
                instance: bank.instance(),
                origin,
                by_expert,
                geometry,
                pages: sources,
            });
        }
        if actual_pages.next().is_some() || metadata_bytes as u64 > SOURCE_SET_HOST_BYTES {
            return Err(reject());
        }
        Ok(Arc::new(Self {
            binding: graph.binding_id(),
            plan_identity: plan.identity(),
            layers,
            metadata_bytes,
        }))
    }
    pub(crate) fn plan_identity(&self) -> plan::PlanIdentity {
        self.plan_identity
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }
    pub(crate) fn inactive_partition_exit(&self) -> bool {
        self.layers[0].geometry.inactive_partition_exit()
    }
    pub(crate) fn private_tables(&self) -> bool {
        self.layers[0].geometry.private_tables()
    }
    pub(crate) fn workgroup_tables(&self) -> bool {
        self.layers[0].geometry.workgroup_tables()
    }
    pub(crate) fn shared_down_scales(&self) -> bool {
        self.layers[0].geometry.shared_down_scales()
    }
    pub(crate) fn shared_gate_up_factors(&self) -> bool {
        self.layers[0].geometry.shared_gate_up_factors()
    }
    fn check_layer(&self, layer: usize, bank: &ExpertBanks) -> Result<&Layer> {
        let l = self.layers.get(layer).ok_or(reject())?;
        if l.instance != bank.instance()
            || bank.profile() != crate::expert_banks::OriginalPackedProfile::DaslabIq3
        {
            return Err(reject());
        }
        l.origin
            .validate_origin(&bank.slots)
            .map_err(|_| reject())?;
        check_mapping(&l.by_expert, &bank.slots)?;
        Ok(l)
    }
    pub(super) fn slot_table(
        &self,
        layer: usize,
        bank: &ExpertBanks,
    ) -> Result<Zeroizing<[u32; 512]>> {
        let l = self.check_layer(layer, bank)?;
        Ok(Zeroizing::new(std::array::from_fn(|i| {
            u32::from(l.by_expert[i])
        })))
    }
    pub(super) fn geometry(&self, layer: usize, bank: &ExpertBanks) -> Result<k::Geometry> {
        Ok(self.check_layer(layer, bank)?.geometry)
    }
    fn pins(&self, layer: usize, bank: &ExpertBanks) -> Result<Vec<Handle>> {
        Ok(self
            .check_layer(layer, bank)?
            .pages
            .iter()
            .map(|p| p.handle.clone())
            .collect())
    }
    pub(super) fn validate_source_layer(
        &self,
        client: &ComputeClient<WgpuRuntime>,
        layer: usize,
        bank: &ExpertBanks,
        expected_stream: &Handle,
    ) -> Result<()> {
        let l = self.check_layer(layer, bank)?;
        bank.check_client(client)?;
        if l.pages
            .iter()
            .any(|page| page.handle.stream != expected_stream.stream)
        {
            return Err(reject());
        }
        let handles: Vec<_> = l.pages.iter().map(|p| p.handle.clone()).collect();
        let handle_bytes = handles
            .capacity()
            .checked_mul(std::mem::size_of::<Handle>())
            .ok_or(Error::ArithmeticOverflow)?;
        let actual = u::resources(client, handles)?;
        if actual.len() != PAGES {
            return Err(reject());
        }
        let primary = handle_bytes
            .checked_add(
                actual
                    .capacity()
                    .checked_mul(std::mem::size_of_val(&actual[0]))
                    .ok_or(Error::ArithmeticOverflow)?,
            )
            .ok_or(Error::ArithmeticOverflow)?;
        if primary > PHYSICAL_HOST_BYTES as usize {
            return Err(reject());
        }
        for (i, (a, page)) in actual.iter().zip(&l.pages).enumerate() {
            let bytes = page.words as u64 * 4;
            let r = a.resource();
            full_handle(&page.handle, bytes)?;
            if !same_resource(r, &page.resource)
                || is_public_external_binding(&page.handle.clone().binding()) != (i % 5 >= 2)
                || actual[..i]
                    .iter()
                    .any(|old| old.resource().buffer == r.buffer)
            {
                return Err(reject());
            }
            check_page(
                r.offset,
                r.size,
                r.buffer.size(),
                bytes,
                if i % 5 >= 2 {
                    bytes
                } else {
                    main_admission::page_bytes(bytes).map_err(admission_error)?
                },
            )?;
        }
        for (role, b) in [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .enumerate()
        {
            let old_handles: Vec<_> = b.segments().iter().map(|s| s.handle().clone()).collect();
            let old_handle_bytes = old_handles
                .capacity()
                .checked_mul(std::mem::size_of::<Handle>())
                .ok_or(Error::ArithmeticOverflow)?;
            let old = u::resources(client, old_handles)?;
            if old.len() != 2 {
                return Err(reject());
            }
            let peak = primary
                .checked_add(old_handle_bytes)
                .and_then(|n| {
                    n.checked_add(old.capacity().checked_mul(std::mem::size_of_val(&old[0]))?)
                })
                .ok_or(Error::ArithmeticOverflow)?;
            if peak > PHYSICAL_HOST_BYTES as usize {
                return Err(reject());
            }
            if old.len() != 2
                || old
                    .iter()
                    .enumerate()
                    .any(|(s, m)| !same_resource(m.resource(), &l.pages[role * 5 + s].resource))
            {
                return Err(reject());
            }
        }
        Ok(())
    }
    pub(super) fn physical(
        &self,
        arena: &PrivateArena,
        layer: usize,
        bank: &ExpertBanks,
        ids: &[BufferId; 7],
        fault: BufferId,
        pending_pins: Option<&[Handle]>,
    ) -> Result<()> {
        self.physical_inner(arena, layer, bank, ids, fault, pending_pins, &[])
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn physical_with_router(
        &self,
        arena: &PrivateArena,
        layer: usize,
        bank: &ExpertBanks,
        ids: &[BufferId; 7],
        fault: BufferId,
        pending_pins: Option<&[Handle]>,
        router: &[BufferId; 3],
    ) -> Result<()> {
        self.physical_inner(arena, layer, bank, ids, fault, pending_pins, router)
    }
    #[allow(clippy::too_many_arguments)]
    fn physical_inner(
        &self,
        arena: &PrivateArena,
        layer: usize,
        bank: &ExpertBanks,
        ids: &[BufferId; 7],
        fault: BufferId,
        pending_pins: Option<&[Handle]>,
        router: &[BufferId],
    ) -> Result<()> {
        if !matches!(router.len(), 0 | 3) {
            return Err(reject());
        }
        let l = self.check_layer(layer, bank)?;
        let client = arena.client().map_err(u::fault)?;
        bank.check_client(client)?;
        let stream = arena.handle(fault).map_err(u::fault)?.stream;
        let private = 8 + router.len();
        let extent_count = private + PAGES;
        let mut handles = Vec::with_capacity(extent_count);
        let mut extents = Zeroizing::new([(0u64, 0u64); ROUTER_EXTENTS]);
        for (i, id) in ids
            .iter()
            .copied()
            .chain(std::iter::once(fault))
            .chain(router.iter().copied())
            .enumerate()
        {
            if ids[..i.min(ids.len())].contains(&id) {
                return Err(reject());
            }
            let h = arena.handle(id).map_err(u::fault)?;
            let bytes = arena.layout(id).map_err(u::fault)?.owned_bytes as u64;
            full_handle(h, bytes)?;
            if h.stream != stream || is_public_external_binding(&h.clone().binding()) {
                return Err(reject());
            }
            extents[i] = (bytes, bytes);
            handles.push(h.clone());
        }
        if pending_pins.is_some_and(|pins| pins.len() != PAGES) {
            return Err(reject());
        }
        for (i, page) in l.pages.iter().enumerate() {
            let handle = pending_pins.map_or(&page.handle, |pins| &pins[i]);
            let bytes = page.words as u64 * 4;
            full_handle(handle, bytes)?;
            if handle.stream != stream
                || is_public_external_binding(&handle.clone().binding()) != (i % 5 >= 2)
            {
                return Err(reject());
            }
            extents[private + i] = (
                bytes,
                if i % 5 >= 2 {
                    bytes
                } else {
                    main_admission::page_bytes(bytes).map_err(admission_error)?
                },
            );
            handles.push(handle.clone());
        }
        let handle_bytes = handles
            .capacity()
            .checked_mul(std::mem::size_of::<Handle>())
            .ok_or(Error::ArithmeticOverflow)?;
        let metadata = u::resources(client, handles)?;
        if metadata.len() != extent_count {
            return Err(reject());
        }
        let primary = handle_bytes
            .checked_add(
                metadata
                    .capacity()
                    .checked_mul(std::mem::size_of_val(&metadata[0]))
                    .ok_or(Error::ArithmeticOverflow)?,
            )
            .and_then(|n| n.checked_add(std::mem::size_of_val(&*extents)))
            .ok_or(Error::ArithmeticOverflow)?;
        if primary > PHYSICAL_HOST_BYTES as usize {
            return Err(reject());
        }
        for (i, r) in metadata.iter().enumerate() {
            let actual = r.resource();
            let (logical, bound) = extents[i];
            check_page(
                actual.offset,
                actual.size,
                actual.buffer.size(),
                logical,
                bound,
            )?;
            if metadata[..i]
                .iter()
                .any(|old| old.resource().buffer == actual.buffer)
            {
                return Err(reject());
            }
            if i >= private && !same_resource(actual, &l.pages[i - private].resource) {
                return Err(reject());
            }
        }
        // Retained registered pins must still resolve to the actual original
        // VRAM pages; matching stream/size alone is not source identity.
        for (role, b) in [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .enumerate()
        {
            let old_handles: Vec<_> = b.segments().iter().map(|s| s.handle().clone()).collect();
            let old_handle_bytes = old_handles
                .capacity()
                .checked_mul(std::mem::size_of::<Handle>())
                .ok_or(Error::ArithmeticOverflow)?;
            let old = u::resources(client, old_handles)?;
            if old.len() != 2 {
                return Err(reject());
            }
            let peak = primary
                .checked_add(old_handle_bytes)
                .and_then(|n| {
                    n.checked_add(old.capacity().checked_mul(std::mem::size_of_val(&old[0]))?)
                })
                .ok_or(Error::ArithmeticOverflow)?;
            if peak > PHYSICAL_HOST_BYTES as usize {
                return Err(reject());
            }
            if old.len() != 2
                || old
                    .iter()
                    .enumerate()
                    .any(|(s, m)| !same_resource(m.resource(), &l.pages[role * 5 + s].resource))
            {
                return Err(reject());
            }
        }
        for shard in 0..5 {
            l.geometry
                .validate_shard(
                    FIRST[shard],
                    PARTS[shard],
                    std::array::from_fn(|r| l.pages[r * 5 + shard].words),
                )
                .map_err(err)?;
        }
        Ok(())
    }
}
fn check_mapping(map: &[u16], bank: &crate::bank_slots::BankSlots) -> Result<()> {
    if map.len() != 512 || bank.capacity() != 144 || bank.experts() != 512 {
        return Err(reject());
    }
    let mut seen = Zeroizing::new([false; 512]);
    let mut next_host = 144;
    for (expert, &slot) in map.iter().enumerate() {
        let slot = slot as usize;
        if slot >= 512 || seen[slot] {
            return Err(reject());
        }
        seen[slot] = true;
        let resident = bank.resident_slot(expert).map_err(|_| reject())?;
        if slot < 144 {
            if resident != Some(slot)
                || bank.resident_expert(slot).map_err(|_| reject())? != Some(expert)
            {
                return Err(reject());
            }
        } else {
            if resident.is_some() || slot != next_host {
                return Err(reject());
            }
            next_host += 1;
        }
    }
    if next_host != 512 || !seen.iter().all(|&v| v) {
        return Err(reject());
    }
    Ok(())
}
pub(super) struct Proof {
    pub(super) source: Arc<Sources>,
    pairs: Zeroizing<[(u32, u32); 30]>,
    len: Zeroizing<usize>,
    count: Zeroizing<usize>,
    vram: u64,
    host: u64,
    pub(super) router_owners: Option<Zeroizing<[BufferId; 3]>>,
}
impl Drop for Proof {
    fn drop(&mut self) {
        self.vram.zeroize();
        self.host.zeroize();
    }
}
impl Proof {
    fn new(
        source: Arc<Sources>,
        layer: usize,
        bank: &ExpertBanks,
        routes: &token_waves::Plan,
    ) -> Result<Self> {
        let l = source.check_layer(layer, bank)?;
        if !(1..=3).contains(&routes.count()) || routes.wave_count() != 1 {
            return Err(reject());
        }
        let union = routes.wave(0).ok_or(reject())?;
        if union.tokens() != (0..routes.count())
            || union.protected_experts().is_empty()
            || union.protected_experts().len() > 30
        {
            return Err(reject());
        }
        let mut p = Self {
            source: source.clone(),
            pairs: Zeroizing::new([(0, 0); 30]),
            len: Zeroizing::new(union.protected_experts().len()),
            count: Zeroizing::new(routes.count()),
            vram: 0,
            host: 0,
            router_owners: None,
        };
        for (i, &expert) in union.protected_experts().iter().enumerate() {
            let slot = *l.by_expert.get(expert as usize).ok_or(reject())?;
            p.pairs[i] = (expert, u32::from(slot));
        }
        for token in 0..routes.count() {
            for route in routes.token(token).ok_or(reject())? {
                if l.by_expert[route.expert() as usize] < 144 {
                    p.vram += 1;
                } else {
                    p.host += 1;
                }
            }
        }
        p.validate(layer, bank)?;
        Ok(p)
    }
    pub(super) fn deferred(
        source: Arc<Sources>,
        count: usize,
        router_owners: [BufferId; 3],
    ) -> Result<Self> {
        if !(1..=3).contains(&count) {
            return Err(reject());
        }
        Ok(Self {
            source,
            pairs: Zeroizing::new([(0, 0); 30]),
            len: Zeroizing::new(0),
            count: Zeroizing::new(count),
            vram: 0,
            host: 0,
            router_owners: Some(Zeroizing::new(router_owners)),
        })
    }
    pub(super) fn resolve_audited(
        &mut self,
        layer: usize,
        bank: &ExpertBanks,
        routes: &crate::resident_batch_router_gpu::AuditedRoutes,
    ) -> Result<()> {
        if *self.len != 0
            || self.vram != 0
            || self.host != 0
            || self.router_owners.is_none()
            || routes.count() != *self.count
            || routes.groups() == 0
            || routes.groups() > 30
        {
            return Err(reject());
        }
        let l = self.source.check_layer(layer, bank)?;
        for (i, &(id, slot)) in routes.pairs().iter().enumerate() {
            if id >= 512
                || slot != u32::from(l.by_expert[id as usize])
                || routes.pairs()[..i]
                    .iter()
                    .any(|&(e, s)| e == id || s == slot)
            {
                return Err(reject());
            }
        }
        let mut vram = 0;
        let mut host = 0;
        for &id in &routes.experts()[..*self.count * 10] {
            if id >= 512 || !routes.pairs().iter().any(|&(e, _)| e == id) {
                return Err(reject());
            }
            if l.by_expert[id as usize] < 144 {
                vram += 1;
            } else {
                host += 1;
            }
        }
        self.pairs[..routes.groups()].copy_from_slice(routes.pairs());
        *self.len = routes.groups();
        self.vram = vram;
        self.host = host;
        self.validate(layer, bank)
    }
    pub(super) fn count(&self) -> usize {
        *self.count
    }
    pub(super) fn validate(&self, layer: usize, bank: &ExpertBanks) -> Result<()> {
        let l = self.source.check_layer(layer, bank)?;
        if *self.len == 0
            || *self.len > 30
            || !(1..=3).contains(&*self.count)
            || self.vram + self.host != (*self.count * 10) as u64
            || *self.len > *self.count * 10
        {
            return Err(reject());
        }
        for (i, &(expert, slot)) in self.pairs[..*self.len].iter().enumerate() {
            if expert >= 512
                || slot != u32::from(l.by_expert[expert as usize])
                || self.pairs[..i]
                    .iter()
                    .any(|&(e, s)| e == expert || s == slot)
            {
                return Err(reject());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counters {
    pub vram_routes: u64,
    pub host_routes: u64,
    pub source_dispatches: u64,
}
impl Counters {
    pub(super) fn add(self, p: &Proof) -> Result<Self> {
        Ok(Self {
            vram_routes: self
                .vram_routes
                .checked_add(p.vram)
                .ok_or(Error::ArithmeticOverflow)?,
            host_routes: self
                .host_routes
                .checked_add(p.host)
                .ok_or(Error::ArithmeticOverflow)?,
            source_dispatches: self
                .source_dispatches
                .checked_add(10)
                .ok_or(Error::ArithmeticOverflow)?,
        })
    }
}
impl Workspace {
    pub(crate) fn frozen_host_sources_enabled(&self) -> bool {
        self.frozen_sources.is_some()
    }
    pub(crate) fn frozen_host_source_partition_exit(&self) -> Result<bool> {
        Ok(self
            .frozen_sources
            .as_ref()
            .ok_or_else(reject)?
            .inactive_partition_exit())
    }
    #[allow(dead_code)]
    pub(crate) fn frozen_host_source_private_tables(&self) -> Result<bool> {
        Ok(self
            .frozen_sources
            .as_ref()
            .ok_or_else(reject)?
            .private_tables())
    }
    #[allow(dead_code)]
    pub(crate) fn frozen_host_source_workgroup_tables(&self) -> Result<bool> {
        Ok(self
            .frozen_sources
            .as_ref()
            .ok_or_else(reject)?
            .workgroup_tables())
    }
    pub(crate) fn frozen_host_source_shared_down_scales(&self) -> Result<bool> {
        Ok(self
            .frozen_sources
            .as_ref()
            .ok_or_else(reject)?
            .shared_down_scales())
    }
    #[allow(dead_code)]
    pub(crate) fn frozen_host_source_shared_gate_up_factors(&self) -> Result<bool> {
        Ok(self
            .frozen_sources
            .as_ref()
            .ok_or_else(reject)?
            .shared_gate_up_factors())
    }
    pub(crate) fn frozen_source_counters(&self) -> Counters {
        self.frozen_source_counters
    }
    /// Explicit post-warm attachment. Metadata reservation is separate from
    /// public backing/registry, original per-layer pending and GPU MAIN credit.
    pub(crate) fn attach_frozen_host_sources(
        &mut self,
        executor: &mut GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        sources: Arc<Sources>,
        reserved_host_bytes: u64,
    ) -> Result<()> {
        self.validate_public_warm_boundary(executor, graph, arena)?;
        if self.frozen_sources.is_some()
            || self.batch_router.is_some()
                && (!self.frozen_host_batch_router_reserved()
                    || self.batch_router_active()
                    || !executor.target_frozen_host_router_reserved
                    || !executor.target_batch_router_reserved)
            || self.prepared_verify3.is_some()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || executor.target_frozen_host_router_reserved
                != self.frozen_host_batch_router_reserved()
            || sources.binding != self.binding
            || sources.layers.len() != 48
            || self.layout.capacity != 144
            || self.layout.tile != 8
            || self.live_member_reduction_enabled()
            || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
            || executor.adaptive.is_some()
            || executor.secondary.is_some()
            || executor.secondary_frequency.is_some()
            || executor.resident_router_reserved
            || executor.batch_moe.is_some()
            || executor.target_frozen_host_source_identity.is_some()
            || !executor.target_rows_reserved
            || reserved_host_bytes < EXTRA_HOST_BYTES
            || sources.metadata_bytes as u64 > SOURCE_SET_HOST_BYTES
        {
            return Err(reject());
        }
        executor.check_target_rows_policy()?;
        for layer in 0..48 {
            let original = weights(graph, layer)?;
            let bank = executor
                .banks
                .get(original.gate.info().name())
                .ok_or(reject())?;
            bank.check_names(original)?;
            sources.validate_source_layer(
                arena.client().map_err(u::fault)?,
                layer,
                bank,
                arena.handle(self.ids[0]).map_err(u::fault)?,
            )?;
            let mut g = self.layout.geometries[layer]
                .with_frozen_host_sources()
                .map_err(err)?;
            if sources.inactive_partition_exit() {
                g = g.with_inactive_partition_exit().map_err(err)?;
            }
            if sources.private_tables() {
                g = g.with_private_tables().map_err(err)?;
            }
            if sources.workgroup_tables() {
                g = g.with_workgroup_tables().map_err(err)?;
            }
            if sources.shared_down_scales() {
                g = g.with_shared_down_scales().map_err(err)?;
            }
            if sources.shared_gate_up_factors() {
                g = g.with_shared_gate_up_factors().map_err(err)?;
            }
            if g != sources.geometry(layer, bank)? {
                return Err(reject());
            }
        }
        // Sticky constructor-equivalent freeze; original scalar bank mutation
        // already rejects this latch. No admission, ledger or epoch is reset.
        executor.target_frozen_host_source_identity = Some(sources.plan_identity());
        executor.resident_banks_frozen = true;
        self.frozen_sources = Some(sources);
        Ok(())
    }
}
impl GpuExecutor<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_routed_moe_frozen_host_batch(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        views: &BatchSpans,
        routes: &token_waves::Plan,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        let out = guarded(&w.poison.clone(), || {
            self.frozen_batch_checked(
                graph, arena, w, s, layer, input, output, views, routes, cancel,
            )
        });
        if out.is_err() {
            self.poison_resident();
        }
        out
    }
    /// Device routing over authenticated original VRAM+host sources. The
    /// copied actual-score audit must resolve this Pending before acceptance.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_frozen_host_batch_router_moe(
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
        let out = guarded(&w.poison.clone(), || {
            u::cancel(cancel)?;
            w.check(self, s, arena)?;
            if !w.frozen_host_batch_router_active()
                || !(1..=3).contains(&views.count())
                || views.count() != s.stamp.count
                || views.input_first() != 0
                || views.output_first() != 0
                || views.input_stride() != 2560
                || views.output_stride() != 2560
                || views.packed_first() != 0
            {
                return Err(reject());
            }
            w.check_batch_router_bank(self, graph, arena, layer)?;
            graph.verify_unchanged()?;
            self.model.verify_file_unchanged()?;
            let source = w.frozen_sources.as_ref().ok_or(reject())?.clone();
            let original = weights(graph, layer)?;
            let key = original.gate.info().name().to_owned();
            let bank = self.banks.get(&key).ok_or(reject())?;
            let g = source.geometry(layer, bank)?;
            let ids = w.ids(input, output);
            let spans = k::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
            let mut words = Zeroizing::new([0usize; 7]);
            for (i, &id) in ids.iter().enumerate() {
                words[i] = u::words(arena, id)?;
            }
            spans.validate(g, *words).map_err(err)?;
            let reserved = w.batch_router_ids().ok_or(reject())?;
            let router_owners = [router, reserved[0], reserved[1]];
            source.physical_with_router(
                arena,
                layer,
                bank,
                &ids,
                s.stamp.fault,
                None,
                &router_owners,
            )?;
            let handle = |id| arena.handle(id).cloned().map_err(u::fault);
            let handles = [
                handle(router)?,
                handle(reserved[0])?,
                handle(ids[MAP])?,
                handle(ids[PROB])?,
                handle(reserved[1])?,
                handle(s.stamp.fault)?,
            ];
            crate::resident_batch_router_gpu::validate_owners(
                &handles,
                layer,
                s.stamp.start,
                views.count(),
            )?;
            let proof = Proof::deferred(source.clone(), views.count(), router_owners)?;
            w.policy.issue(layer)?;
            u::zero_many(arena, &*w.ids)?;
            crate::resident_batch_router_gpu::launch_frozen_sources(
                arena.client().map_err(u::fault)?,
                handles,
                layer,
                s.stamp.start,
                views.count(),
            )?;
            // Fixed maximum union only schedules work. Each original kernel
            // consumes the actual GPU MAP group/rank header, never forced IDs.
            source.launch(arena, layer, bank, g, &spans, &ids, 30)?;
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
            u::cancel(cancel)?;
            w.check(self, s, arena)?;
            w.check_batch_router_bank(self, graph, arena, layer)?;
            let pins = source.pins(layer, bank)?;
            let metadata = std::mem::size_of::<Pending>()
                .checked_add(std::mem::size_of::<Proof>())
                .and_then(|n| {
                    n.checked_add(pins.capacity().checked_mul(std::mem::size_of::<Handle>())?)
                })
                .ok_or(Error::ArithmeticOverflow)?;
            if metadata > 32 * 1024 {
                return Err(Error::LimitExceeded("frozen router pending host envelope"));
            }
            let pending = Pending {
                views: Views {
                    ids: Zeroizing::new([input, output]),
                    offsets: Zeroizing::new([0, 2560, 0, 2560]),
                },
                stamp: Zeroizing::new((*s.stamp).clone()),
                layer: Zeroizing::new(layer),
                key,
                instance: bank.instance(),
                pins,
                staged: None,
                last: None,
                count1: None,
                deferred_router: Some(Zeroizing::new(w.identity)),
                frozen_sources: Some(proof),
                waves: Zeroizing::new(1),
                poison: w.poison.clone(),
                armed: true,
            };
            self.gpu_router_calls = self
                .gpu_router_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            self.moe_calls = self
                .moe_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            // Truthful source tiers are counted only after actual CPU audit and
            // all-family acceptance. No ordinary GPU hit/upload claim here.
            Ok(pending)
        });
        if out.is_err() {
            self.poison_resident();
        }
        out
    }
    #[allow(clippy::too_many_arguments)]
    fn frozen_batch_checked(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        views: &BatchSpans,
        routes: &token_waves::Plan,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        u::cancel(cancel)?;
        w.check(self, s, arena)?;
        graph.verify_unchanged()?;
        self.model.verify_file_unchanged()?;
        let source = w.frozen_sources.as_ref().ok_or(reject())?.clone();
        if !self.resident_banks_frozen
            || self.target_frozen_host_router_reserved
            || self.target_frozen_host_source_identity != Some(source.plan_identity())
            || graph.binding_id() != source.binding
            || w.binding != source.binding
            || !(1..=3).contains(&routes.count())
            || routes.count() != s.stamp.count
            || views.count() != routes.count()
            || routes.wave_count() != 1
            || views.packed_first() != 0
            || views.input_stride() < 2560
            || views.output_stride() < 2560
            || self.adaptive.is_some()
            || self.secondary.is_some()
            || self.secondary_frequency.is_some()
            || self.batch_experts
            || self.ordered_public_upload
            || self.word_down
            || w.hybrid.is_some()
            || w.resident_router.is_some()
            || w.batch_router.is_some()
            || w.prepared_verify3.is_some()
        {
            return Err(reject());
        }
        let original = weights(graph, layer)?;
        if [original.gate, original.up, original.down]
            .into_iter()
            .any(|t| !t.belongs_to(self.model))
        {
            return Err(reject());
        }
        let key = original.gate.info().name().to_owned();
        let bank = self.banks.get(&key).ok_or(reject())?;
        bank.check_names(original)?;
        let g = source.geometry(layer, bank)?;
        let proof = Proof::new(source.clone(), layer, bank, routes)?;
        let ids = w.ids(input, output);
        let spans = k::Spans::new([
            0,
            views.input_first(),
            views.input_stride(),
            0,
            g.ff(),
            0,
            g.hidden(),
            0,
            views.output_first(),
            views.output_stride(),
            0,
        ]);
        let mut words = Zeroizing::new([0usize; 7]);
        for (i, &id) in ids.iter().enumerate() {
            words[i] = u::words(arena, id)?;
        }
        spans.validate(g, *words).map_err(err)?;
        source.physical(arena, layer, bank, &ids, s.stamp.fault, None)?;
        let mut experts = Zeroizing::new([0u32; 30]);
        let mut probabilities = Zeroizing::new([0f32; 30]);
        for token in 0..routes.count() {
            let row = routes.token(token).ok_or(reject())?;
            if row.len() != 10 {
                return Err(reject());
            }
            for (rank, route) in row.iter().enumerate() {
                experts[token * 10 + rank] = route.expert();
                probabilities[token * 10 + rank] = route.probability();
            }
        }
        let map = k::GroupMap::new_frozen_sources(
            routes.count(),
            &experts[..routes.count() * 10],
            &probabilities[..routes.count() * 10],
            &proof.pairs[..*proof.len],
            g,
        )
        .map_err(err)?;
        map.validate_geometry(g).map_err(err)?;
        // Everything above is read-only. Original family issue/owning clear
        // precedes the first map upload and every source consumer.
        w.policy.issue(layer)?;
        u::zero_many(arena, &*w.ids)?;
        write_words(arena, ids[MAP], map.words(), &w.poison)?;
        let mut probs = Zeroizing::new([0u32; 80]);
        for (i, &p) in map.probabilities().iter().enumerate() {
            probs[i] = p.to_bits();
        }
        write_words(arena, ids[PROB], &*probs, &w.poison)?;
        source.physical(arena, layer, bank, &ids, s.stamp.fault, None)?;
        source.launch(arena, layer, bank, g, &spans, &ids, map.groups())?;
        launch_finite_for_count(arena, g, &spans, &ids, routes.count())?;
        fold(
            arena,
            g,
            &ids,
            s.stamp.fault,
            routes.count(),
            0,
            routes.count(),
            false,
        )?;
        launch_reduce(arena, g, &spans, &ids, routes.count())?;
        launch_finite_for_count(arena, g, &spans, &ids, routes.count())?;
        fold(
            arena,
            g,
            &ids,
            s.stamp.fault,
            routes.count(),
            0,
            routes.count(),
            true,
        )?;
        u::cancel(cancel)?;
        w.check(self, s, arena)?;
        proof.validate(layer, bank)?;
        drop(map);
        let pins = source.pins(layer, bank)?;
        let metadata = std::mem::size_of::<Pending>()
            + std::mem::size_of::<Proof>()
            + pins.capacity() * std::mem::size_of::<Handle>();
        if metadata > 32 * 1024 {
            return Err(Error::LimitExceeded("frozen source pending host envelope"));
        }
        let out = Pending {
            views: Views {
                ids: Zeroizing::new([input, output]),
                offsets: Zeroizing::new([
                    views.input_first(),
                    views.input_stride(),
                    views.output_first(),
                    views.output_stride(),
                ]),
            },
            stamp: Zeroizing::new((*s.stamp).clone()),
            layer: Zeroizing::new(layer),
            key,
            instance: bank.instance(),
            pins,
            staged: None,
            last: None,
            count1: None,
            deferred_router: None,
            frozen_sources: Some(proof),
            waves: Zeroizing::new(1),
            poison: w.poison.clone(),
            armed: true,
        };
        self.moe_calls = self
            .moe_calls
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        // Host accesses never become ordinary GPU cache-hit/upload evidence.
        Ok(out)
    }
}
impl Sources {
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        arena: &PrivateArena,
        layer: usize,
        bank: &ExpertBanks,
        g: k::Geometry,
        s: &k::Spans,
        ids: &[BufferId; 7],
        groups: usize,
    ) -> Result<()> {
        let l = self.check_layer(layer, bank)?;
        if g != l.geometry {
            return Err(reject());
        }
        let [gate, up, down] = g.projections();
        let client = arena.client().map_err(u::fault)?;
        for shard in 0..5 {
            let ga = &l.pages[shard];
            let up_page = &l.pages[5 + shard];
            let down_page = &l.pages[10 + shard];
            macro_rules! dispatch {
                ($gate:ident, $down:ident) => {
                    unsafe {
                        k::$gate::launch_unchecked::<WgpuRuntime>(
                            client,
                            g.grid(groups, false).map_err(err)?,
                            CubeDim::new_1d(64),
                            ArrayArg::from_raw_parts(ga.handle.clone(), ga.words),
                            ArrayArg::from_raw_parts(up_page.handle.clone(), up_page.words),
                            arg(arena, ids[INPUT])?,
                            arg(arena, ids[MAP])?,
                            arg(arena, ids[ACT])?,
                            arg(arena, ids[STATUS])?,
                            s.map_first(),
                            s.input_first(),
                            s.input_stride(),
                            s.act_first(),
                            s.act_stride(),
                            s.status_first(),
                            FIRST[shard],
                            PARTS[shard],
                            g.hidden(),
                            g.ff(),
                            gate.dtype(),
                            up.dtype(),
                            gate.row_bytes(),
                            up.row_bytes(),
                            g.tile(),
                            false,
                        );
                        k::$down::launch_unchecked::<WgpuRuntime>(
                            client,
                            g.grid(groups, true).map_err(err)?,
                            CubeDim::new_1d(64),
                            ArrayArg::from_raw_parts(down_page.handle.clone(), down_page.words),
                            arg(arena, ids[ACT])?,
                            arg(arena, ids[MAP])?,
                            arg(arena, ids[PART])?,
                            arg(arena, ids[STATUS])?,
                            s.map_first(),
                            s.act_first(),
                            s.act_stride(),
                            s.partial_first(),
                            s.partial_stride(),
                            s.status_first(),
                            FIRST[shard],
                            PARTS[shard],
                            g.hidden(),
                            g.ff(),
                            down.dtype(),
                            down.row_bytes(),
                            g.tile(),
                        );
                    }
                };
            }
            match kernel_policy(g) {
                (GateKernel::WorkgroupFactors, DownKernel::SharedScales) => dispatch!(
                    grouped_gate_up_frozen_workgroup_grid_shared_factors,
                    grouped_down_frozen_shared_scales
                ),
                (GateKernel::WorkgroupFactors, DownKernel::Frozen) => dispatch!(
                    grouped_gate_up_frozen_workgroup_grid_shared_factors, grouped_down_frozen
                ),
                (GateKernel::PrivateFactors, DownKernel::SharedScales) => dispatch!(
                    grouped_gate_up_frozen_private_grid_shared_factors,
                    grouped_down_frozen_shared_scales
                ),
                (GateKernel::PrivateFactors, DownKernel::Frozen) => dispatch!(
                    grouped_gate_up_frozen_private_grid_shared_factors, grouped_down_frozen
                ),
                (GateKernel::SharedFactors, DownKernel::SharedScales) => dispatch!(
                    grouped_gate_up_frozen_shared_factors, grouped_down_frozen_shared_scales
                ),
                (GateKernel::SharedFactors, DownKernel::Frozen) => dispatch!(
                    grouped_gate_up_frozen_shared_factors, grouped_down_frozen
                ),
                (GateKernel::WorkgroupTables, DownKernel::SharedScales) => dispatch!(
                    grouped_gate_up_frozen_workgroup_grid,
                    grouped_down_frozen_shared_scales
                ),
                (GateKernel::WorkgroupTables, DownKernel::Frozen) => {
                    dispatch!(grouped_gate_up_frozen_workgroup_grid, grouped_down_frozen)
                }
                (GateKernel::PrivateTables, DownKernel::SharedScales) => dispatch!(
                    grouped_gate_up_frozen_private_grid,
                    grouped_down_frozen_shared_scales
                ),
                (GateKernel::PrivateTables, DownKernel::Frozen) => {
                    dispatch!(grouped_gate_up_frozen_private_grid, grouped_down_frozen)
                }
                (GateKernel::Frozen, DownKernel::SharedScales) => {
                    dispatch!(grouped_gate_up_frozen, grouped_down_frozen_shared_scales)
                }
                (GateKernel::Frozen, DownKernel::Frozen) => {
                    dispatch!(grouped_gate_up_frozen, grouped_down_frozen)
                }
                (GateKernel::Original, DownKernel::Original) => {
                    dispatch!(grouped_gate_up, grouped_down)
                }
                _ => return Err(reject()),
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_entrypoints_preserve_compound_table_and_scale_policies() {
        for layer in 0..48 {
            let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
            let original = k::Geometry::new(
                [
                    k::Projection::new(gate, 2560, 640, 512).unwrap(),
                    k::Projection::new(up, 2560, 640, 512).unwrap(),
                    k::Projection::new(down, 640, 2560, 512).unwrap(),
                ],
                8,
                144,
            )
            .unwrap()
            .with_frozen_host_sources()
            .unwrap();
            assert_eq!(
                kernel_policy(original),
                (GateKernel::Original, DownKernel::Original)
            );
            let frozen = original.with_inactive_partition_exit().unwrap();
            for table in [
                GateKernel::Frozen,
                GateKernel::PrivateTables,
                GateKernel::WorkgroupTables,
            ] {
                let selected = match table {
                    GateKernel::Frozen => frozen,
                    GateKernel::PrivateTables => frozen.with_private_tables().unwrap(),
                    GateKernel::WorkgroupTables => frozen.with_workgroup_tables().unwrap(),
                    _ => unreachable!(),
                };
                let factor = match table {
                    GateKernel::Frozen => GateKernel::SharedFactors,
                    GateKernel::PrivateTables => GateKernel::PrivateFactors,
                    GateKernel::WorkgroupTables => GateKernel::WorkgroupFactors,
                    _ => unreachable!(),
                };
                let factors = selected.with_shared_gate_up_factors().unwrap();
                assert_eq!(kernel_policy(factors), (factor, DownKernel::Frozen));
                assert_eq!(kernel_policy(factors.with_shared_down_scales().unwrap()),
                    (factor, DownKernel::SharedScales));
                assert_eq!(kernel_policy(selected), (table, DownKernel::Frozen));
                assert_eq!(
                    kernel_policy(selected.with_shared_down_scales().unwrap()),
                    (table, DownKernel::SharedScales)
                );
            }
        }
    }
    fn full_bank() -> crate::bank_slots::BankSlots {
        let mut bank = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        for batch in 0..9 {
            let selected: Vec<_> = (batch * 16..batch * 16 + 16)
                .map(|i| (i * 3 + 11) % 512)
                .collect();
            let plan = bank.plan(&selected).unwrap();
            bank.commit(plan).unwrap();
        }
        bank
    }
    fn map(bank: &crate::bank_slots::BankSlots) -> Zeroizing<[u16; 512]> {
        let mut m = Zeroizing::new([u16::MAX; 512]);
        let mut host = 144;
        for expert in 0..512 {
            m[expert] = match bank.resident_slot(expert).unwrap() {
                Some(s) => s as u16,
                None => {
                    let s = host;
                    host += 1;
                    s
                }
            };
        }
        m
    }
    #[test]
    fn frozen_mapping_requires_complete_actual_bank_and_sorted_complement() {
        let bank = full_bank();
        let mut m = map(&bank);
        check_mapping(&*m, &bank).unwrap();
        let old = *m;
        m.swap(0, 1);
        assert!(check_mapping(&*m, &bank).is_err());
        *m = old;
        m[1] = m[0];
        assert!(check_mapping(&*m, &bank).is_err());
        *m = old;
        let missing: Vec<_> = (0..512)
            .filter(|&e| bank.resident_slot(e).unwrap().is_none())
            .collect();
        m.swap(missing[0], missing[1]);
        assert!(check_mapping(&*m, &bank).is_err());
        *m = old;
        let empty = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        assert!(check_mapping(&*m, &empty).is_err());
        assert!(check_mapping(&m[..511], &bank).is_err());
    }
    #[test]
    fn even_all_hit_recency_commit_invalidates_frozen_origin() {
        let mut bank = full_bank();
        let origin = StagedSlots::begin(&bank).unwrap();
        let before = map(&bank);
        let expert = bank.resident_expert(0).unwrap().unwrap();
        let plan = bank.plan(&[expert]).unwrap();
        assert!(plan.uploads.is_empty());
        bank.commit(plan).unwrap();
        assert_eq!(*before, *map(&bank));
        assert!(origin.validate_origin(&bank).is_err());
    }
    #[test]
    fn metadata_initial_peak_rejects_excess_capacity_and_overflow_before_fill() {
        let initial = initial_metadata(48, 432, 48).unwrap();
        let layer = StagedSlots::bounded_native_metadata_bytes(144).unwrap()
            + 512 * 2
            + PAGES * std::mem::size_of::<SourcePage>();
        assert!(initial + 48 * layer < SOURCE_SET_HOST_BYTES as usize);
        for (layers, pages, banks) in [
            (usize::MAX, 432, 48),
            (48, usize::MAX, 48),
            (48, 432, usize::MAX),
            (48, 1_000_000, 48),
        ] {
            assert!(initial_metadata(layers, pages, banks).is_err());
        }
    }
    #[test]
    fn truthful_increment_keeps_runtime_and_host_source_metadata_separate() {
        assert_eq!(EXTRA_INFO_BYTES_PER_CHUNK, 589_824);
        assert_eq!(EXTRA_HOST_BYTES, 2_105_344);
        assert_eq!(ROUTER_MAIN_BYTES, 2_228_224);
        assert_eq!(ROUTER_RUNTIME_BYTES, 9_175_040);
        assert_eq!(ROUTER_HOST_BYTES, 5 * 1024 * 1024);
        assert_eq!(PARTS.into_iter().sum::<usize>(), 512);
        let mut next = 0;
        for (first, slots) in FIRST.into_iter().zip(PARTS) {
            assert_eq!(first, next);
            next += slots;
        }
        for layer in 0..48 {
            for (role_index, role) in ROLES.into_iter().enumerate() {
                for p in 0..3 {
                    assert_eq!(
                        page_index(layer, role, p).unwrap(),
                        layer * 9 + role_index * 3 + p
                    );
                }
            }
        }
        assert!(page_index(48, Role::Gate, 0).is_err());
        assert!(page_index(0, Role::Down, 3).is_err());
    }
}
