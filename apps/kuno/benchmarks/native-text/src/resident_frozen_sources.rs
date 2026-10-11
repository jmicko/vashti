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
    StagedFrozen,
    ThreeMembers,
    StagedThreeMembers,
    StagedTwoRows,
    Original,
    Frozen,
    PrivateTables,
    WorkgroupTables,
    SharedFactors,
    PrivateFactors,
    WorkgroupFactors,
    TwoRows,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DownKernel {
    StagedFrozen,
    ThreeMembers,
    StagedThreeMembers,
    StagedFourRows,
    FourRowsThreeMembers,
    StagedFourRowsThreeMembers,
    Original,
    Frozen,
    SharedScales,
    FourRows,
    FourRowsSharedScales,
}
// This selector is used by actual dispatch, so compound policy tests cover the
// runtime entrypoints rather than just the stored attachment flags.
fn kernel_policy(g: k::Geometry) -> (GateKernel, DownKernel) {
    let gate = if g.three_member_arithmetic() {
        GateKernel::ThreeMembers
    } else if g.two_row_gate_up() {
        GateKernel::TwoRows
    } else if g.shared_gate_up_factors() {
        if g.workgroup_tables() {
            GateKernel::WorkgroupFactors
        } else if g.private_tables() {
            GateKernel::PrivateFactors
        } else {
            GateKernel::SharedFactors
        }
    } else if g.workgroup_tables() {
        GateKernel::WorkgroupTables
    } else if g.private_tables() {
        GateKernel::PrivateTables
    } else if g.inactive_partition_exit() {
        GateKernel::Frozen
    } else {
        GateKernel::Original
    };
    let down = if g.three_member_arithmetic() {
        if g.four_row_down() {
            DownKernel::FourRowsThreeMembers
        } else {
            DownKernel::ThreeMembers
        }
    } else if g.four_row_down() {
        if g.shared_down_scales() {
            DownKernel::FourRowsSharedScales
        } else {
            DownKernel::FourRows
        }
    } else if g.shared_down_scales() {
        DownKernel::SharedScales
    } else if g.inactive_partition_exit() {
        DownKernel::Frozen
    } else {
        DownKernel::Original
    };
    (gate, down)
}
fn source_kernel_policy(g: k::Geometry, staged_shard: bool) -> Result<(GateKernel, DownKernel)> {
    let (gate, down) = kernel_policy(g);
    if !staged_shard {
        return Ok((gate, down));
    }
    k::frozen_host_staging::Shape::new(g).map_err(err)?;
    let gate = match gate {
        GateKernel::ThreeMembers => GateKernel::StagedThreeMembers,
        GateKernel::Frozen => GateKernel::StagedFrozen,
        GateKernel::TwoRows => GateKernel::StagedTwoRows,
        _ => return Err(reject()),
    };
    let down = match down {
        DownKernel::ThreeMembers => DownKernel::StagedThreeMembers,
        DownKernel::FourRowsThreeMembers => DownKernel::StagedFourRowsThreeMembers,
        DownKernel::Frozen => DownKernel::StagedFrozen,
        DownKernel::FourRows => DownKernel::StagedFourRows,
        _ => return Err(reject()),
    };
    Ok((gate, down))
}
// Construction and attachment must apply the same arithmetic specialization.
// The public owner geometry remains tile8; only unreachable arithmetic cells
// are omitted for an authenticated one-to-three-member wave.
fn three_member_geometry(g: k::Geometry, enabled: bool) -> Result<k::Geometry> {
    if enabled {
        g.with_three_member_arithmetic().map_err(err)
    } else {
        Ok(g)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourcePhase {
    Interleaved,
    GateOnly,
    PrepareAndDown,
}
fn source_phases(down_integer: bool) -> &'static [SourcePhase] {
    if down_integer {
        &[SourcePhase::GateOnly, SourcePhase::PrepareAndDown]
    } else {
        &[SourcePhase::Interleaved]
    }
}
const PARTS: [usize; 5] = [72, 72, 144, 144, 80];
const FIRST: [usize; 5] = [0, 72, 144, 288, 432];
const ROLES: [Role; 3] = [Role::Gate, Role::Up, Role::Down];
const PAGES: usize = 15;
const EXTENTS: usize = 8 + PAGES;
const ROUTER_EXTENTS: usize = EXTENTS + 3 + 3 + 1;
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
/// Content-free PUBLIC resident-buffer measurement. This is not main-pool,
/// device-local heap, detach, erasure or capacity-admission authority. Source
/// handle clones/bindings do not allocate another physical buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResidentBankAllocation {
    pub layers: u64,
    pub source_bindings: u64,
    pub physical_buffers: u64,
    pub logical_bytes: u64,
    pub physical_buffer_bytes: u64,
    pub conservative_bin_bytes: u64,
    pub padding_bytes: u64,
    pub admission_headroom_bytes: u64,
}
const RESIDENT_SOURCE_PAGES: [usize; 6] = [0, 1, 5, 6, 10, 11];
impl ResidentBankAllocation {
    // Caller has proved that this is one distinct, original VRAM buffer. The
    // source constructor rejects duplicate buffers across ALL roles/layers.
    fn add_authenticated_buffer(&mut self, logical: u64, physical: u64) -> Result<()> {
        let bound = main_admission::page_bytes(logical).map_err(admission_error)?;
        if physical < logical || physical > bound {
            return Err(reject());
        }
        let add = |old: u64, next: u64| old.checked_add(next).ok_or(Error::ArithmeticOverflow);
        self.source_bindings = add(self.source_bindings, 1)?;
        self.physical_buffers = add(self.physical_buffers, 1)?;
        self.logical_bytes = add(self.logical_bytes, logical)?;
        self.physical_buffer_bytes = add(self.physical_buffer_bytes, physical)?;
        self.conservative_bin_bytes = add(self.conservative_bin_bytes, bound)?;
        Ok(())
    }
    fn complete(mut self, layers: usize) -> Result<Self> {
        let expected = crate::daslab_profile::bank_bytes_144();
        if layers != 48
            || self.source_bindings != 48 * RESIDENT_SOURCE_PAGES.len() as u64
            || self.physical_buffers != self.source_bindings
            || self.logical_bytes != expected.logical
            || self.conservative_bin_bytes != expected.physical
        {
            return Err(reject());
        }
        self.layers = layers as u64;
        self.padding_bytes = self
            .physical_buffer_bytes
            .checked_sub(self.logical_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        self.admission_headroom_bytes = self
            .conservative_bin_bytes
            .checked_sub(self.physical_buffer_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(self)
    }
}
#[cfg(test)]
mod resident_bank_allocation_tests {
    use super::*;
    fn original_buffers() -> Vec<u64> {
        // Independent inspected GGUF metadata; no model payload/device.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../text/receipts/daslab-iq3-artifact-20261009.json"
        ))
        .unwrap();
        fixture["tensor_inventory"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|tensor| {
                let name = tensor["name"].as_str().unwrap();
                name.ends_with("ffn_gate_exps.weight")
                    || name.ends_with("ffn_up_exps.weight")
                    || name.ends_with("ffn_down_exps.weight")
            })
            .flat_map(|tensor| {
                let bytes = tensor["tensor_bytes"].as_u64().unwrap();
                assert_eq!(bytes % 512, 0);
                [bytes / 512 * 72; 2]
            })
            .collect()
    }
    #[test]
    fn original_full_48_layers_count_unique_roots_not_handle_clones() {
        let mut actual = ResidentBankAllocation::default();
        let buffers = original_buffers();
        assert_eq!(buffers.len(), 288);
        for logical in buffers {
            actual.add_authenticated_buffer(logical, logical).unwrap();
        }
        let actual = actual.complete(48).unwrap();
        assert_eq!(actual.layers, 48);
        assert_eq!(actual.source_bindings, 288);
        assert_eq!(actual.physical_buffers, 288);
        assert_eq!(actual.logical_bytes, 14_144_716_800);
        assert_eq!(actual.physical_buffer_bytes, actual.logical_bytes);
        assert_eq!(actual.conservative_bin_bytes, 18_723_373_056);
        assert_eq!(actual.padding_bytes, 0);
        assert_eq!(actual.admission_headroom_bytes, 4_578_656_256);
    }
    #[test]
    fn actual_backing_padding_is_separate_from_unused_admission_bound() {
        let mut actual = ResidentBankAllocation::default();
        for logical in original_buffers() {
            let physical = logical.next_multiple_of(32 * 1024);
            actual.add_authenticated_buffer(logical, physical).unwrap();
        }
        let actual = actual.complete(48).unwrap();
        assert_eq!(actual.physical_buffer_bytes, 14_149_484_544);
        assert_eq!(actual.padding_bytes, 4_767_744);
        assert_eq!(actual.admission_headroom_bytes, 4_573_888_512);
        assert_eq!(
            actual.padding_bytes + actual.admission_headroom_bytes,
            actual.conservative_bin_bytes - actual.logical_bytes
        );
    }
    #[test]
    fn imported_host_partitions_never_enter_resident_telemetry() {
        let mut host = 0;
        let mut resident = 0;
        for role in 0..3 {
            for shard in 0..5 {
                let index = role * 5 + shard;
                if RESIDENT_SOURCE_PAGES.contains(&index) {
                    assert!(shard < 2);
                    resident += 1;
                } else {
                    assert!(shard >= 2);
                    host += 1;
                }
            }
        }
        assert_eq!(resident, 6);
        assert_eq!(host, 9);
    }
    #[test]
    fn incomplete_overcounted_or_wrong_profile_cannot_emit_snapshot() {
        for layers in [0, 47, 49, usize::MAX] {
            assert!(ResidentBankAllocation::default().complete(layers).is_err());
        }
        let mut actual = ResidentBankAllocation::default();
        let buffers = original_buffers();
        for &logical in &buffers {
            actual.add_authenticated_buffer(logical, logical).unwrap();
        }
        actual.complete(48).unwrap();
        for field in 0..4 {
            let mut wrong = actual;
            match field {
                0 => wrong.source_bindings -= 1,
                1 => wrong.physical_buffers += 1,
                2 => wrong.logical_bytes -= 4,
                _ => wrong.conservative_bin_bytes += 4,
            }
            assert!(wrong.complete(48).is_err());
        }
        actual
            .add_authenticated_buffer(buffers[0], buffers[0])
            .unwrap();
        assert!(actual.complete(48).is_err());
    }
    #[test]
    fn full_physical_extent_and_checked_sum_are_required() {
        let logical = 45_158_400;
        let bound = main_admission::page_bytes(logical).unwrap();
        for (logical, physical) in [
            (0, 0),
            (logical, logical - 4),
            (logical, bound + 4),
            (u64::MAX, u64::MAX),
        ] {
            assert!(
                ResidentBankAllocation::default()
                    .add_authenticated_buffer(logical, physical)
                    .is_err()
            );
        }
        let mut overflow = ResidentBankAllocation {
            physical_buffer_bytes: u64::MAX,
            ..ResidentBankAllocation::default()
        };
        assert!(overflow.add_authenticated_buffer(logical, logical).is_err());
        let mut upper = ResidentBankAllocation::default();
        for logical in original_buffers() {
            upper
                .add_authenticated_buffer(logical, main_admission::page_bytes(logical).unwrap())
                .unwrap();
        }
        let upper = upper.complete(48).unwrap();
        assert_eq!(upper.physical_buffer_bytes, upper.conservative_bin_bytes);
        assert_eq!(upper.admission_headroom_bytes, 0);
    }
}
pub(crate) struct Sources {
    binding: GraphBindingId,
    plan_identity: plan::PlanIdentity,
    bank_fill: crate::executor::CompletedPublicHostBankFill,
    layers: Vec<Layer>,
    metadata_bytes: usize,
    packed_host_staging: bool,
    gu_integer: bool,
    gu_stride_four: bool,
    down_stride_four: bool,
    down_integer: bool,
    four_members: bool,
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
fn validate_ranked_fill_source_policy(
    completed: crate::executor::CompletedPublicHostBankFill,
    partition_exit: bool,
    competing_modes: [bool; 7],
) -> Result<()> {
    completed.validate()?;
    if completed.policy == crate::executor::PublicHostBankFillPolicy::StrataRankedFiller
        && (!partition_exit || competing_modes.into_iter().any(|mode| mode))
    {
        return Err(reject());
    }
    Ok(())
}
#[cfg(test)]
mod ranked_fill_source_tests {
    use super::*;
    #[test]
    fn ranked_fill_requires_actual_publication_and_original_source_scope() {
        let selected = crate::executor::CompletedPublicHostBankFill {
            policy: crate::executor::PublicHostBankFillPolicy::StrataRankedFiller,
            completed_layers: 48,
            filler_experts: 1769,
            preserved_residents: 5143,
        };
        assert!(validate_ranked_fill_source_policy(selected, true, [false; 7]).is_ok());
        assert!(validate_ranked_fill_source_policy(selected, false, [false; 7]).is_err());
        for i in 0..7 {
            let mut modes = [false; 7];
            modes[i] = true;
            assert!(validate_ranked_fill_source_policy(selected, true, modes).is_err());
            assert!(validate_ranked_fill_source_policy(Default::default(), false, modes).is_ok());
        }
        let mut bad = selected;
        bad.completed_layers = 0;
        assert!(validate_ranked_fill_source_policy(bad, true, [false; 7]).is_err());
    }
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
            false,
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
            false,
            false,
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
            false,
            false,
            false,
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
            false,
            false,
            false,
            false,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_three_member_policy(
        selected: bool,
        exit: bool,
        private: bool,
        workgroup: bool,
        scales: bool,
        factors: bool,
        paired_gu: bool,
    ) -> Result<()> {
        if selected && (!exit || private || workgroup || scales || factors || paired_gu) {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn down_integer(&self) -> bool {
        self.down_integer
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_down_integer_policy(
        selected: bool,
        exit: bool,
        private: bool,
        workgroup: bool,
        scales: bool,
        factors: bool,
        four_rows: bool,
        two_gu: bool,
        staging: bool,
        three_members: bool,
        stride_four: bool,
    ) -> Result<()> {
        if selected
            && (!exit
                || private
                || workgroup
                || scales
                || factors
                || four_rows
                || two_gu
                || staging
                || three_members
                || stride_four)
        {
            return Err(reject());
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Independent immutable source-policy axes.
    pub(crate) fn validate_gu_integer_policy(
        selected: bool,
        exit: bool,
        private: bool,
        workgroup: bool,
        scales: bool,
        factors: bool,
        paired_gu: bool,
        staging: bool,
        three_members: bool,
    ) -> Result<()> {
        if selected
            && (!exit
                || private
                || workgroup
                || scales
                || factors
                || paired_gu
                || staging
                || three_members)
        {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn validate_gu_schedule_policy(
        gu_integer: bool,
        gu_stride_four: bool,
    ) -> Result<()> {
        if gu_stride_four && !gu_integer {
            return Err(reject());
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn validate_down_stride_four_policy(
        selected: bool,
        exit: bool,
        four_rows: bool,
        private: bool,
        workgroup: bool,
        scales: bool,
        factors: bool,
        staging: bool,
        three_members: bool,
        paired_gu: bool,
    ) -> Result<()> {
        if selected
            && (!exit
                || !four_rows
                || private
                || workgroup
                || scales
                || factors
                || staging
                || three_members
                || paired_gu)
        {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn validate_staging_policy(
        selected: bool,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
    ) -> Result<()> {
        if selected
            && (!inactive_partition_exit
                || private_tables
                || workgroup_tables
                || shared_down_scales
                || shared_gate_up_factors)
        {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn validate_kernel_policies(
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
    ) -> Result<()> {
        if ((private_tables
            || workgroup_tables
            || shared_down_scales
            || shared_gate_up_factors
            || four_row_down
            || two_row_gate_up)
            && !inactive_partition_exit)
            || private_tables && workgroup_tables
            || two_row_gate_up && (private_tables || workgroup_tables || shared_gate_up_factors)
        {
            return Err(reject());
        }
        Ok(())
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
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy_and_gu_math(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
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
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_policy_and_gu_math(
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
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy_and_gu_schedule(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
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
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_policy_and_gu_schedule(
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
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy_and_schedules(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
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
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_policy_and_schedules(
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
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
    ) -> Result<Arc<Self>> {
        // Original entry retains its exact default arithmetic policy.
        Self::new_with_policy_and_down_math(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
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
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_policy_and_down_math(
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
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
        down_integer: bool,
    ) -> Result<Arc<Self>> {
        Self::new_with_policy_and_four_rows(
            executor,
            graph,
            plan,
            pages,
            reserved_host_bytes,
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
            false,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_policy_and_four_rows(
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
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
        down_integer: bool,
        four_members: bool,
    ) -> Result<Arc<Self>> {
        if four_members != executor.target_four_rows_reserved
            || four_members
                && (!gu_integer
                    || !down_integer
                    || three_members
                    || down_stride_four
                    || packed_host_staging)
        {
            return Err(reject());
        }
        Self::validate_down_integer_policy(
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
        Self::validate_gu_schedule_policy(gu_integer, gu_stride_four)?;
        Self::validate_down_stride_four_policy(
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
        Self::validate_kernel_policies(
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            four_row_down,
            two_row_gate_up,
        )?;
        Self::validate_staging_policy(
            packed_host_staging,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
        )?;
        Self::validate_three_member_policy(
            three_members,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            two_row_gate_up,
        )?;
        Self::validate_gu_integer_policy(
            gu_integer,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            two_row_gate_up,
            packed_host_staging,
            three_members,
        )?;
        if down_integer != executor.target_frozen_down_integer_reserved
            || gu_integer != executor.target_frozen_gu_integer_reserved
            || (gu_integer || down_stride_four || down_integer)
                && (!executor.target_frozen_host_router_reserved
                    || !executor.target_batch_router_reserved
                    || !executor.target_rows_reserved
                    || !executor.target_verify3_dense
                    || !executor.target_verify3_fused
                    || !executor.token_io_checkpoint
                    || executor.native_projection_tile() != 8
                    || executor.native_projection_member_capacity() != 8)
        {
            return Err(reject());
        }
        if three_members && !executor.target_frozen_host_router_reserved {
            return Err(reject());
        }
        if reserved_host_bytes < EXTRA_HOST_BYTES
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
        let bank_fill = plan.bank_fill()?;
        validate_ranked_fill_source_policy(
            bank_fill,
            inactive_partition_exit,
            [
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                two_row_gate_up,
                packed_host_staging,
                three_members,
            ],
        )?;

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
            if four_row_down {
                geometry = geometry.with_four_row_down().map_err(err)?;
            }
            if two_row_gate_up {
                geometry = geometry.with_two_row_gate_up().map_err(err)?;
            }
            geometry = three_member_geometry(geometry, three_members)?;
            if four_members {
                geometry = geometry.with_four_members().map_err(err)?;
            }
            if gu_integer {
                crate::quant_matvec::frozen_gu_integer::validate_geometry(geometry)?;
            }
            if down_integer {
                crate::quant_matvec::frozen_down_integer::validate_geometry(geometry)?;
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
            if packed_host_staging {
                k::frozen_host_staging::Shape::new(geometry).map_err(err)?;
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
            bank_fill,
            layers,
            metadata_bytes,
            packed_host_staging,
            gu_integer,
            gu_stride_four,
            down_stride_four,
            down_integer,
            four_members,
        }))
    }
    pub(crate) fn four_members(&self) -> bool {
        self.four_members
    }
    fn count_limit(&self) -> usize {
        if self.four_members { 4 } else { 3 }
    }
    pub(crate) fn gu_integer(&self) -> bool {
        self.gu_integer
    }
    pub(crate) fn gu_stride_four(&self) -> bool {
        self.gu_stride_four
    }
    pub(crate) fn down_stride_four(&self) -> bool {
        self.down_stride_four
    }
    pub(crate) fn three_members(&self) -> bool {
        self.layers[0].geometry.three_member_arithmetic()
    }
    pub(crate) fn packed_host_staging(&self) -> bool {
        self.packed_host_staging
    }
    pub(crate) fn plan_identity(&self) -> plan::PlanIdentity {
        self.plan_identity
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }
    /// Reconstruct from the accepted immutable physical resources only. The
    /// six resident pages per layer are distinct roots, although BankSegment,
    /// SourcePage and kernel bindings can retain clones of those same roots.
    /// The remaining nine imported HOST pages are deliberately excluded. No
    /// get_resource/memory_usage/readback call or device submission is made.
    pub(crate) fn resident_bank_allocation(&self) -> Result<ResidentBankAllocation> {
        if self.layers.len() != 48 {
            return Err(reject());
        }
        let mut result = ResidentBankAllocation::default();
        for layer in &self.layers {
            if layer.pages.len() != PAGES {
                return Err(reject());
            }
            for index in RESIDENT_SOURCE_PAGES {
                let page = &layer.pages[index];
                let logical = (page.words as u64)
                    .checked_mul(4)
                    .ok_or(Error::ArithmeticOverflow)?;
                full_handle(&page.handle, logical)?;
                let r = &page.resource;
                check_page(
                    r.offset,
                    r.size,
                    r.buffer.size(),
                    logical,
                    main_admission::page_bytes(logical).map_err(admission_error)?,
                )?;
                result.add_authenticated_buffer(logical, r.buffer.size())?;
            }
        }
        result.complete(self.layers.len())
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
    pub(crate) fn four_row_down(&self) -> bool {
        self.layers[0].geometry.four_row_down()
    }
    pub(crate) fn two_row_gate_up(&self) -> bool {
        self.layers[0].geometry.two_row_gate_up()
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
        self.physical_inner(
            arena,
            layer,
            bank,
            ids,
            fault,
            pending_pins,
            &[],
            None,
            None,
            None,
        )
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
        staging: Option<&[BufferId; 3]>,
        gu_integer: Option<BufferId>,
        down_integer: Option<BufferId>,
    ) -> Result<()> {
        self.physical_inner(
            arena,
            layer,
            bank,
            ids,
            fault,
            pending_pins,
            router,
            staging,
            gu_integer,
            down_integer,
        )
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
        staging: Option<&[BufferId; 3]>,
        gu_integer: Option<BufferId>,
        down_integer: Option<BufferId>,
    ) -> Result<()> {
        if !matches!(router.len(), 0 | 3)
            || self.packed_host_staging != staging.is_some()
            || staging.is_some() && router.len() != 3
            || self.gu_integer != gu_integer.is_some()
            || gu_integer.is_some() && (router.len() != 3 || staging.is_some())
            || self.down_integer != down_integer.is_some()
            || down_integer.is_some() && (router.len() != 3 || staging.is_some())
            || self.down_stride_four && router.len() != 3
        {
            return Err(reject());
        }
        let l = self.check_layer(layer, bank)?;
        let client = arena.client().map_err(u::fault)?;
        bank.check_client(client)?;
        let stream = arena.handle(fault).map_err(u::fault)?.stream;
        let stage_ids = staging.map_or(&[][..], |ids| &ids[..]);
        let private = 8
            + router.len()
            + stage_ids.len()
            + usize::from(gu_integer.is_some())
            + usize::from(down_integer.is_some());
        let extent_count = private + PAGES;
        let mut handles = Vec::with_capacity(extent_count);
        let mut extents = Zeroizing::new([(0u64, 0u64); ROUTER_EXTENTS]);
        for (i, id) in ids
            .iter()
            .copied()
            .chain(std::iter::once(fault))
            .chain(router.iter().copied())
            .chain(stage_ids.iter().copied())
            .chain(gu_integer)
            .chain(down_integer)
            .enumerate()
        {
            if ids[..i.min(ids.len())].contains(&id) {
                return Err(reject());
            }
            let h = arena.handle(id).map_err(u::fault)?;
            let bytes = arena.layout(id).map_err(u::fault)?.owned_bytes as u64;
            full_handle(h, bytes)?;
            if i >= 8 + router.len()
                && i < 8 + router.len() + stage_ids.len()
                && bytes != FROZEN_STAGING_OWNER_BYTES
            {
                return Err(reject());
            }
            if gu_integer == Some(id)
                && (bytes != FROZEN_GU_INTEGER_MAIN_BYTES
                    || arena.layout(id).map_err(u::fault)?.logical_words
                        != FROZEN_GU_INTEGER_SCRATCH_WORDS)
            {
                return Err(reject());
            }
            if down_integer == Some(id)
                && (bytes != FROZEN_DOWN_INTEGER_MAIN_BYTES
                    || arena.layout(id).map_err(u::fault)?.logical_words
                        != FROZEN_DOWN_INTEGER_SCRATCH_WORDS)
            {
                return Err(reject());
            }
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
#[derive(Clone, Copy)]
struct GuIntegerDispatch {
    layer: usize,
    count: usize,
    dispatches: u64,
    stride_four_dispatches: u64,
}
impl GuIntegerDispatch {
    #[allow(dead_code)] // Original count1..3 private receipt fixture.
    fn validate(self, layer: usize, count: usize, stride_four: bool) -> Result<()> {
        self.validate_selected(layer, count, stride_four, false)
    }
    fn validate_selected(
        self,
        layer: usize,
        count: usize,
        stride_four: bool,
        four: bool,
    ) -> Result<()> {
        if self.layer != layer
            || layer >= 48
            || self.count != count
            || self.dispatches != 5
            || self.stride_four_dispatches != if stride_four { 5 } else { 0 }
            || !(1..=if four { 4 } else { 3 }).contains(&count)
        {
            return Err(reject());
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct DownStrideFourDispatch {
    layer: usize,
    count: usize,
    dispatches: u64,
}
impl DownStrideFourDispatch {
    fn validate(self, layer: usize, count: usize) -> Result<()> {
        if self.layer != layer
            || layer >= 48
            || self.count != count
            || !(1..=3).contains(&count)
            || self.dispatches != 5
        {
            return Err(reject());
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct DownIntegerDispatch {
    layer: usize,
    count: usize,
    preparations: u64,
    dispatches: u64,
    audits: u64,
}
impl DownIntegerDispatch {
    #[allow(dead_code)] // Original count1..3 private receipt fixture.
    fn validate(self, layer: usize, count: usize) -> Result<()> {
        self.validate_selected(layer, count, false)
    }
    fn validate_selected(self, layer: usize, count: usize, four: bool) -> Result<()> {
        if self.layer != layer
            || layer >= 48
            || self.count != count
            || !(1..=if four { 4 } else { 3 }).contains(&count)
            || self.preparations != 1
            || self.dispatches != 5
            || self.audits != 1
        {
            return Err(reject());
        }
        Ok(())
    }
}
struct SourceLaunch {
    gu_integer: Option<GuIntegerDispatch>,
    down_stride_four: Option<DownStrideFourDispatch>,
    down_integer: Option<DownIntegerDispatch>,
}
pub(super) struct Proof {
    pub(super) source: Arc<Sources>,
    pairs: Zeroizing<[(u32, u32); 40]>,
    len: Zeroizing<usize>,
    count: Zeroizing<usize>,
    vram: u64,
    host: u64,
    gu_integer_launch: Option<GuIntegerDispatch>,
    down_stride_four_launch: Option<DownStrideFourDispatch>,
    down_integer_launch: Option<DownIntegerDispatch>,
    pub(super) router_owners: Option<Zeroizing<[BufferId; 3]>>,
}
impl Drop for Proof {
    fn drop(&mut self) {
        self.vram.zeroize();
        self.host.zeroize();
        self.gu_integer_launch = None;
        self.down_stride_four_launch = None;
        self.down_integer_launch = None;
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
        if !(1..=source.count_limit()).contains(&routes.count()) || routes.wave_count() != 1 {
            return Err(reject());
        }
        let union = routes.wave(0).ok_or(reject())?;
        if union.tokens() != (0..routes.count())
            || union.protected_experts().is_empty()
            || union.protected_experts().len() > source.count_limit() * 10
        {
            return Err(reject());
        }
        let mut p = Self {
            source: source.clone(),
            pairs: Zeroizing::new([(0, 0); 40]),
            len: Zeroizing::new(union.protected_experts().len()),
            count: Zeroizing::new(routes.count()),
            vram: 0,
            host: 0,
            gu_integer_launch: None,
            down_stride_four_launch: None,
            down_integer_launch: None,
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
        if !(1..=source.count_limit()).contains(&count) {
            return Err(reject());
        }
        Ok(Self {
            source,
            pairs: Zeroizing::new([(0, 0); 40]),
            len: Zeroizing::new(0),
            count: Zeroizing::new(count),
            vram: 0,
            host: 0,
            gu_integer_launch: None,
            down_stride_four_launch: None,
            down_integer_launch: None,
            router_owners: Some(Zeroizing::new(router_owners)),
        })
    }
    fn record_gu_integer_launch(
        &mut self,
        layer: usize,
        receipt: Option<GuIntegerDispatch>,
    ) -> Result<()> {
        if self.gu_integer_launch.is_some() || self.source.gu_integer != receipt.is_some() {
            return Err(reject());
        }
        if let Some(receipt) = receipt {
            receipt.validate_selected(
                layer,
                *self.count,
                self.source.gu_stride_four,
                self.source.four_members,
            )?;
            self.gu_integer_launch = Some(receipt);
        }
        Ok(())
    }
    fn record_down_stride_four_launch(
        &mut self,
        layer: usize,
        receipt: Option<DownStrideFourDispatch>,
    ) -> Result<()> {
        if self.down_stride_four_launch.is_some()
            || self.source.down_stride_four != receipt.is_some()
        {
            return Err(reject());
        }
        if let Some(receipt) = receipt {
            receipt.validate(layer, *self.count)?;
            if self.router_owners.is_none() {
                return Err(reject());
            }
            self.down_stride_four_launch = Some(receipt);
        }
        Ok(())
    }
    fn record_down_integer_launch(
        &mut self,
        layer: usize,
        receipt: Option<DownIntegerDispatch>,
    ) -> Result<()> {
        if self.down_integer_launch.is_some() || self.source.down_integer != receipt.is_some() {
            return Err(reject());
        }
        if let Some(receipt) = receipt {
            receipt.validate_selected(layer, *self.count, self.source.four_members)?;
            if self.router_owners.is_none() {
                return Err(reject());
            }
            self.down_integer_launch = Some(receipt);
        }
        Ok(())
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
            || routes.groups() > self.source.count_limit() * 10
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
        if self.source.gu_integer != self.gu_integer_launch.is_some() {
            return Err(reject());
        }
        if let Some(receipt) = self.gu_integer_launch {
            receipt.validate_selected(
                layer,
                *self.count,
                self.source.gu_stride_four,
                self.source.four_members,
            )?;
        }
        if self.source.down_stride_four != self.down_stride_four_launch.is_some() {
            return Err(reject());
        }
        if let Some(receipt) = self.down_stride_four_launch {
            receipt.validate(layer, *self.count)?;
            if self.router_owners.is_none() {
                return Err(reject());
            }
        }
        if *self.len == 0
            || *self.len > self.source.count_limit() * 10
            || !(1..=self.source.count_limit()).contains(&*self.count)
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

fn gu_integer_completed_flag(
    selected: bool,
    preparations: u64,
    dispatches: u64,
    audits: u64,
) -> Result<bool> {
    if if selected {
        preparations == 0
            || preparations != audits
            || preparations.checked_mul(5) != Some(dispatches)
    } else {
        preparations != 0 || dispatches != 0 || audits != 0
    } {
        return Err(reject());
    }
    Ok(selected)
}
fn gu_stride_four_completed_flag(
    integer: bool,
    selected: bool,
    preparations: u64,
    dispatches: u64,
    audits: u64,
    stride_four_dispatches: u64,
) -> Result<bool> {
    Sources::validate_gu_schedule_policy(integer, selected)?;
    gu_integer_completed_flag(integer, preparations, dispatches, audits)?;
    if stride_four_dispatches != if selected { dispatches } else { 0 } {
        return Err(reject());
    }
    Ok(selected)
}
fn down_stride_four_completed_flag(
    selected: bool,
    layers: u64,
    dispatches: u64,
    source_dispatches: u64,
) -> Result<bool> {
    if if selected {
        layers == 0
            || layers.checked_mul(5) != Some(dispatches)
            || layers.checked_mul(10) != Some(source_dispatches)
    } else {
        layers != 0 || dispatches != 0
    } {
        return Err(reject());
    }
    Ok(selected)
}
fn down_integer_completed_flag(
    selected: bool,
    preparations: u64,
    dispatches: u64,
    audits: u64,
    source_dispatches: u64,
) -> Result<bool> {
    gu_integer_completed_flag(selected, preparations, dispatches, audits)?;
    if selected && preparations.checked_mul(10) != Some(source_dispatches) {
        return Err(reject());
    }
    Ok(selected)
}
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counters {
    pub vram_routes: u64,
    pub host_routes: u64,
    pub source_dispatches: u64,
    pub gu_integer_preparations: u64,
    pub gu_integer_dispatches: u64,
    pub gu_stride_four_dispatches: u64,
    pub gu_integer_audits: u64,
    pub down_stride_four_layers: u64,
    pub down_stride_four_dispatches: u64,
    pub down_integer_preparations: u64,
    pub down_integer_dispatches: u64,
    pub down_integer_audits: u64,
    pub staging_dispatches: u64,
    pub staged_unique_host_groups: u64,
    pub staged_packed_bytes: u64,
}
impl Counters {
    pub(super) fn add(self, p: &Proof, layer: usize) -> Result<Self> {
        let gu = match p.gu_integer_launch {
            Some(receipt) if p.source.gu_integer => {
                receipt.validate_selected(
                    layer,
                    *p.count,
                    p.source.gu_stride_four,
                    p.source.four_members,
                )?;
                receipt.dispatches
            }
            None if !p.source.gu_integer => 0,
            _ => return Err(reject()),
        };
        let stride_four = p.gu_integer_launch.map_or(0, |r| r.stride_four_dispatches);
        let down_stride_four = match p.down_stride_four_launch {
            Some(receipt) if p.source.down_stride_four && p.router_owners.is_some() => {
                receipt.validate(layer, *p.count)?;
                receipt.dispatches
            }
            None if !p.source.down_stride_four => 0,
            _ => return Err(reject()),
        };
        let down_integer = match p.down_integer_launch {
            Some(receipt) if p.source.down_integer && p.router_owners.is_some() => {
                receipt.validate_selected(layer, *p.count, p.source.four_members)?;
                receipt.dispatches
            }
            None if !p.source.down_integer => 0,
            _ => return Err(reject()),
        };
        let (groups, bytes) = if p.source.packed_host_staging {
            let geometry = p.source.layers.get(layer).ok_or_else(reject)?.geometry;
            let groups = p.pairs[..*p.len]
                .iter()
                .filter(|(_, slot)| *slot >= 144)
                .count() as u64;
            let stride = geometry
                .projections()
                .into_iter()
                .try_fold(0u64, |sum, projection| {
                    sum.checked_add(projection.expert_bytes() as u64)
                        .ok_or(Error::ArithmeticOverflow)
                })?;
            (
                groups,
                groups
                    .checked_mul(stride)
                    .ok_or(Error::ArithmeticOverflow)?,
            )
        } else {
            (0, 0)
        };
        Ok(Self {
            vram_routes: self
                .vram_routes
                .checked_add(p.vram)
                .ok_or(Error::ArithmeticOverflow)?,
            host_routes: self
                .host_routes
                .checked_add(p.host)
                .ok_or(Error::ArithmeticOverflow)?,
            gu_integer_preparations: self
                .gu_integer_preparations
                .checked_add(u64::from(gu != 0))
                .ok_or(Error::ArithmeticOverflow)?,
            gu_integer_dispatches: self
                .gu_integer_dispatches
                .checked_add(gu)
                .ok_or(Error::ArithmeticOverflow)?,
            gu_stride_four_dispatches: self
                .gu_stride_four_dispatches
                .checked_add(stride_four)
                .ok_or(Error::ArithmeticOverflow)?,
            gu_integer_audits: self
                .gu_integer_audits
                .checked_add(u64::from(gu != 0))
                .ok_or(Error::ArithmeticOverflow)?,
            down_stride_four_layers: self
                .down_stride_four_layers
                .checked_add(u64::from(down_stride_four != 0))
                .ok_or(Error::ArithmeticOverflow)?,
            down_stride_four_dispatches: self
                .down_stride_four_dispatches
                .checked_add(down_stride_four)
                .ok_or(Error::ArithmeticOverflow)?,
            down_integer_preparations: self
                .down_integer_preparations
                .checked_add(u64::from(down_integer != 0))
                .ok_or(Error::ArithmeticOverflow)?,
            down_integer_dispatches: self
                .down_integer_dispatches
                .checked_add(down_integer)
                .ok_or(Error::ArithmeticOverflow)?,
            down_integer_audits: self
                .down_integer_audits
                .checked_add(u64::from(down_integer != 0))
                .ok_or(Error::ArithmeticOverflow)?,
            source_dispatches: self
                .source_dispatches
                .checked_add(10)
                .ok_or(Error::ArithmeticOverflow)?,
            staging_dispatches: self
                .staging_dispatches
                .checked_add(if p.source.packed_host_staging { 6 } else { 0 })
                .ok_or(Error::ArithmeticOverflow)?,
            staged_unique_host_groups: self
                .staged_unique_host_groups
                .checked_add(groups)
                .ok_or(Error::ArithmeticOverflow)?,
            staged_packed_bytes: self
                .staged_packed_bytes
                .checked_add(bytes)
                .ok_or(Error::ArithmeticOverflow)?,
        })
    }
}
impl Workspace {
    pub(crate) fn frozen_host_sources_enabled(&self) -> bool {
        self.frozen_sources.is_some()
    }
    pub(crate) fn frozen_host_source_bank_fill(
        &self,
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
    ) -> Result<crate::executor::CompletedPublicHostBankFill> {
        let Some(source) = self.frozen_sources.as_ref() else {
            return Ok(Default::default());
        };
        if !executor.resident_banks_frozen
            || executor.target_frozen_host_source_identity != Some(source.plan_identity)
            || source.binding != graph.binding_id()
            || self.binding != source.binding
        {
            return Err(reject());
        }
        graph.verify_unchanged()?;
        executor.model.verify_file_unchanged()?;
        source.bank_fill.validate()?;
        for layer in 0..48 {
            let key = format!("blk.{layer}.ffn_gate_exps.weight");
            source.check_layer(layer, executor.banks.get(&key).ok_or_else(reject)?)?;
        }
        Ok(source.bank_fill)
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
    pub(crate) fn frozen_host_source_four_row_down(&self) -> Result<bool> {
        Ok(self
            .frozen_sources
            .as_ref()
            .ok_or_else(reject)?
            .four_row_down())
    }
    pub(crate) fn frozen_host_source_two_row_gate_up(&self) -> Result<bool> {
        let sources = self.frozen_sources.as_ref().ok_or_else(reject)?;
        let enabled = sources.two_row_gate_up();
        if enabled && self.frozen_source_counters.source_dispatches == 0 {
            return Err(reject());
        }
        Ok(enabled)
    }
    pub(crate) fn frozen_host_source_three_members(&self) -> Result<bool> {
        let source = self.frozen_sources.as_ref().ok_or_else(reject)?;
        if source.three_members() && self.frozen_source_counters.source_dispatches == 0 {
            return Err(reject());
        }
        Ok(source.three_members())
    }
    pub(crate) fn frozen_host_source_gu_integer(&self) -> Result<bool> {
        let sources = self.frozen_sources.as_ref().ok_or_else(reject)?;
        let counters = self.frozen_source_counters;
        gu_integer_completed_flag(
            sources.gu_integer(),
            counters.gu_integer_preparations,
            counters.gu_integer_dispatches,
            counters.gu_integer_audits,
        )
    }
    pub(crate) fn frozen_host_source_gu_stride_four(&self) -> Result<bool> {
        let sources = self.frozen_sources.as_ref().ok_or_else(reject)?;
        let counters = self.frozen_source_counters;
        gu_stride_four_completed_flag(
            sources.gu_integer(),
            sources.gu_stride_four(),
            counters.gu_integer_preparations,
            counters.gu_integer_dispatches,
            counters.gu_integer_audits,
            counters.gu_stride_four_dispatches,
        )
    }
    pub(crate) fn frozen_host_source_down_integer_selected(&self) -> bool {
        self.frozen_sources
            .as_ref()
            .is_some_and(|source| source.down_integer())
    }
    pub(crate) fn frozen_host_source_down_integer(&self) -> Result<bool> {
        let source = self.frozen_sources.as_ref().ok_or(reject())?;
        let c = self.frozen_source_counters;
        down_integer_completed_flag(
            source.down_integer(),
            c.down_integer_preparations,
            c.down_integer_dispatches,
            c.down_integer_audits,
            c.source_dispatches,
        )
    }
    pub(crate) fn frozen_host_source_down_stride_four(&self) -> Result<bool> {
        let source = self.frozen_sources.as_ref().ok_or_else(reject)?;
        let counters = self.frozen_source_counters;
        down_stride_four_completed_flag(
            source.down_stride_four(),
            counters.down_stride_four_layers,
            counters.down_stride_four_dispatches,
            counters.source_dispatches,
        )
    }
    pub(crate) fn frozen_host_source_packed_staging(&self) -> Result<bool> {
        Ok(self
            .frozen_sources
            .as_ref()
            .ok_or_else(reject)?
            .packed_host_staging())
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
        Sources::validate_down_integer_policy(
            sources.down_integer(),
            sources.inactive_partition_exit(),
            sources.private_tables(),
            sources.workgroup_tables(),
            sources.shared_down_scales(),
            sources.shared_gate_up_factors(),
            sources.four_row_down(),
            sources.two_row_gate_up(),
            sources.packed_host_staging(),
            sources.three_members(),
            sources.down_stride_four(),
        )?;
        Sources::validate_gu_schedule_policy(sources.gu_integer(), sources.gu_stride_four())?;
        Sources::validate_down_stride_four_policy(
            sources.down_stride_four(),
            sources.inactive_partition_exit(),
            sources.four_row_down(),
            sources.private_tables(),
            sources.workgroup_tables(),
            sources.shared_down_scales(),
            sources.shared_gate_up_factors(),
            sources.packed_host_staging(),
            sources.three_members(),
            sources.two_row_gate_up(),
        )?;
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
            || sources.packed_host_staging() != self.frozen_staging.is_some()
            || sources.down_integer() != self.frozen_down_integer.is_some()
            || sources.down_integer() != executor.target_frozen_down_integer_reserved
            || sources.down_integer() && !self.frozen_host_batch_router_reserved()
            || sources.gu_integer() != self.frozen_gu_integer.is_some()
            || sources.gu_integer() != executor.target_frozen_gu_integer_reserved
            || sources.gu_integer() && !self.frozen_host_batch_router_reserved()
            || sources.down_stride_four() && !self.frozen_host_batch_router_reserved()
            || sources.packed_host_staging() && !self.frozen_host_batch_router_reserved()
            || sources.three_members() && !self.frozen_host_batch_router_reserved()
            || sources.four_members() != self.public_four_members_policy()
            || sources.four_members() != executor.target_four_rows_reserved
            || sources.four_members() && !self.four_member_router_reserved()
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
            if sources.four_row_down() {
                g = g.with_four_row_down().map_err(err)?;
            }
            if sources.two_row_gate_up() {
                g = g.with_two_row_gate_up().map_err(err)?;
            }
            g = three_member_geometry(g, sources.three_members())?;
            if sources.four_members() {
                g = g.with_four_members().map_err(err)?;
            }
            if sources.gu_integer() {
                crate::quant_matvec::frozen_gu_integer::validate_geometry(g)?;
            }
            if sources.down_integer() {
                crate::quant_matvec::frozen_down_integer::validate_geometry(g)?;
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
                || !(1..=if w.public_four_members_policy() { 4 } else { 3 })
                    .contains(&views.count())
                || views.count() == 4
                    && (!self.target_four_rows_active || !w.four_member_router_reserved())
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
                w.frozen_staging_owner_ids().as_ref(),
                w.frozen_gu_integer_owner_id(),
                w.frozen_down_integer_owner_id(),
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
            let mut proof = Proof::deferred(source.clone(), views.count(), router_owners)?;
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
            let source_launch = source.launch(
                arena,
                layer,
                bank,
                g,
                &spans,
                &ids,
                if views.count() == 4 { 40 } else { 30 },
                w.frozen_staging_owner_ids().as_ref(),
                w.frozen_gu_integer_owner_id(),
                w.frozen_down_integer_owner_id(),
                views.count(),
                s.stamp.fault,
            )?;
            proof.record_gu_integer_launch(layer, source_launch.gu_integer)?;
            proof.record_down_stride_four_launch(layer, source_launch.down_stride_four)?;
            proof.record_down_integer_launch(layer, source_launch.down_integer)?;
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
            || !(1..=source.count_limit()).contains(&routes.count())
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
        source.launch(
            arena,
            layer,
            bank,
            g,
            &spans,
            &ids,
            map.groups(),
            None,
            None,
            None,
            routes.count(),
            s.stamp.fault,
        )?;
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
        staging: Option<&[BufferId; 3]>,
        gu_integer: Option<BufferId>,
        down_integer: Option<BufferId>,
        count: usize,
        fault: BufferId,
    ) -> Result<SourceLaunch> {
        let l = self.check_layer(layer, bank)?;
        if self.packed_host_staging != staging.is_some()
            || g != l.geometry
            || self.down_integer != down_integer.is_some()
            || self.down_integer
                && (groups != if count == 4 { 40 } else { 30 }
                    || staging.is_some()
                    || !(1..=self.count_limit()).contains(&count))
            || self.gu_integer != gu_integer.is_some()
            || self.gu_integer && (staging.is_some() || !(1..=self.count_limit()).contains(&count))
        {
            return Err(reject());
        }
        Self::validate_gu_schedule_policy(self.gu_integer, self.gu_stride_four)?;
        if self.down_stride_four && (groups != 30 || !(1..=3).contains(&count)) {
            return Err(reject());
        }
        let [gate, up, down] = g.projections();
        let client = arena.client().map_err(u::fault)?;
        let integer_handles = if let Some(scratch) = gu_integer {
            let handles = ids.map(|id| arena.handle(id).cloned().map_err(u::fault));
            let [a, b, c, d, e, f, h] = handles;
            let handles = [a?, b?, c?, d?, e?, f?, h?];
            let scratch = arena.handle(scratch).cloned().map_err(u::fault)?;
            (if count == 4 {
                crate::quant_matvec::frozen_gu_integer::four::prepare
            } else {
                crate::quant_matvec::frozen_gu_integer::prepare
            })(client, g, s, handles.clone(), scratch.clone(), count)?;
            Some((handles, scratch))
        } else {
            None
        };
        let down_integer_handles = if let Some(scratch) = down_integer {
            crate::quant_matvec::frozen_down_integer::validate_geometry(g)?;
            let handles = ids.map(|id| arena.handle(id).cloned().map_err(u::fault));
            let [a, b, c, d, e, f, h] = handles;
            Some((
                [a?, b?, c?, d?, e?, f?, h?],
                arena.handle(scratch).cloned().map_err(u::fault)?,
            ))
        } else {
            None
        };
        let mut down_integer_preparations = 0u64;
        let mut down_integer_dispatches = 0u64;
        let mut integer_dispatches = 0u64;
        let mut stride_four_dispatches = 0u64;
        let mut down_stride_four_dispatches = 0u64;
        // Original source schedule stays interleaved. Alternate DOWN needs
        // every real-route ACT completed before one route-indexed preparation.
        for &phase in source_phases(self.down_integer) {
            if phase == SourcePhase::PrepareAndDown {
                let (handles, scratch) = down_integer_handles.as_ref().ok_or_else(reject)?;
                (if count == 4 {
                    crate::quant_matvec::frozen_down_integer::four::prepare
                } else {
                    crate::quant_matvec::frozen_down_integer::prepare
                })(client, g, s, handles.clone(), scratch.clone(), count)?;
                down_integer_preparations = down_integer_preparations
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            for shard in 0..5 {
                let ga = &l.pages[shard];
                let up_page = &l.pages[5 + shard];
                let down_page = &l.pages[10 + shard];
                let staged_shard = self.packed_host_staging && shard >= 2;
                let (gate_handle, gate_words, up_handle, up_words, down_handle, down_words) =
                    if staged_shard {
                        let stage = staging.ok_or_else(reject)?;
                        let shape = k::frozen_host_staging::Shape::new(g).map_err(err)?;
                        // Original public sources remain pinned and unchanged. Each role
                        // copy precedes every matching consumer on this exact stream.
                        unsafe {
                            k::frozen_host_staging::stage_gate_up::launch_unchecked::<WgpuRuntime>(
                                client,
                                shape.copy_grid(false),
                                CubeDim::new_1d(256),
                                ArrayArg::from_raw_parts(ga.handle.clone(), ga.words),
                                ArrayArg::from_raw_parts(up_page.handle.clone(), up_page.words),
                                arg(arena, ids[MAP])?,
                                arg(arena, stage[0])?,
                                arg(arena, stage[1])?,
                                arg(arena, fault)?,
                                s.map_first(),
                                FIRST[shard],
                                PARTS[shard],
                                shape.gate_up_words(),
                            );
                            k::frozen_host_staging::stage_down::launch_unchecked::<WgpuRuntime>(
                                client,
                                shape.copy_grid(true),
                                CubeDim::new_1d(256),
                                ArrayArg::from_raw_parts(down_page.handle.clone(), down_page.words),
                                arg(arena, ids[MAP])?,
                                arg(arena, stage[2])?,
                                arg(arena, fault)?,
                                s.map_first(),
                                FIRST[shard],
                                PARTS[shard],
                                shape.down_words(),
                            );
                        }
                        (
                            arena.handle(stage[0]).map_err(u::fault)?.clone(),
                            (FROZEN_STAGING_OWNER_BYTES / 4) as usize,
                            arena.handle(stage[1]).map_err(u::fault)?.clone(),
                            (FROZEN_STAGING_OWNER_BYTES / 4) as usize,
                            arena.handle(stage[2]).map_err(u::fault)?.clone(),
                            (FROZEN_STAGING_OWNER_BYTES / 4) as usize,
                        )
                    } else {
                        (
                            ga.handle.clone(),
                            ga.words,
                            up_page.handle.clone(),
                            up_page.words,
                            down_page.handle.clone(),
                            down_page.words,
                        )
                    };
                macro_rules! dispatch_gate {
                    ($gate:ident) => {
                        unsafe {
                            k::$gate::launch_unchecked::<WgpuRuntime>(
                                client,
                                g.grid(groups, false).map_err(err)?,
                                CubeDim::new_1d(64),
                                ArrayArg::from_raw_parts(gate_handle.clone(), gate_words),
                                ArrayArg::from_raw_parts(up_handle.clone(), up_words),
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
                        }
                    };
                }
                macro_rules! dispatch_down {
                    ($down:ident) => {
                        unsafe {
                            k::$down::launch_unchecked::<WgpuRuntime>(
                                client,
                                g.grid(groups, true).map_err(err)?,
                                CubeDim::new_1d(64),
                                ArrayArg::from_raw_parts(down_handle.clone(), down_words),
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
                let (gate, down) = source_kernel_policy(g, staged_shard)?;
                if (gate == GateKernel::Original) != (down == DownKernel::Original) {
                    return Err(reject());
                }
                if phase != SourcePhase::PrepareAndDown {
                    if let Some((handles, scratch)) = &integer_handles {
                        if gate != GateKernel::Frozen || staged_shard {
                            return Err(reject());
                        }
                        let launch = match (count == 4, self.gu_stride_four) {
                            (true, true) => {
                                crate::quant_matvec::frozen_gu_integer::four::launch_group_stride4
                            }
                            (true, false) => crate::quant_matvec::frozen_gu_integer::four::launch,
                            (false, true) => {
                                crate::quant_matvec::frozen_gu_integer::launch_group_stride4
                            }
                            (false, false) => crate::quant_matvec::frozen_gu_integer::launch,
                        };
                        launch(
                            client,
                            g,
                            s,
                            [gate_handle.clone(), up_handle.clone()],
                            handles.clone(),
                            scratch.clone(),
                            FIRST[shard],
                            PARTS[shard],
                            groups,
                            count,
                        )?;
                        // Only an actual successful alternative leaf call creates
                        // stride-four credit; neither selection nor DOWN can credit it.
                        if self.gu_stride_four {
                            stride_four_dispatches = stride_four_dispatches
                                .checked_add(1)
                                .ok_or(Error::ArithmeticOverflow)?;
                        }
                        integer_dispatches = integer_dispatches
                            .checked_add(1)
                            .ok_or(Error::ArithmeticOverflow)?;
                    } else {
                        match gate {
                            GateKernel::ThreeMembers => {
                                dispatch_gate!(grouped_gate_up_frozen_three_members)
                            }
                            GateKernel::StagedThreeMembers => {
                                dispatch_gate!(grouped_gate_up_frozen_staged_three_members)
                            }
                            GateKernel::StagedTwoRows => {
                                dispatch_gate!(grouped_gate_up_frozen_staged_two_rows)
                            }
                            GateKernel::StagedFrozen => {
                                dispatch_gate!(grouped_gate_up_frozen_staged)
                            }
                            GateKernel::TwoRows => dispatch_gate!(grouped_gate_up_frozen_two_rows),
                            GateKernel::Original => dispatch_gate!(grouped_gate_up),
                            GateKernel::Frozen => dispatch_gate!(grouped_gate_up_frozen),
                            GateKernel::PrivateTables => {
                                dispatch_gate!(grouped_gate_up_frozen_private_grid)
                            }
                            GateKernel::WorkgroupTables => {
                                dispatch_gate!(grouped_gate_up_frozen_workgroup_grid)
                            }
                            GateKernel::SharedFactors => {
                                dispatch_gate!(grouped_gate_up_frozen_shared_factors)
                            }
                            GateKernel::PrivateFactors => {
                                dispatch_gate!(grouped_gate_up_frozen_private_grid_shared_factors)
                            }
                            GateKernel::WorkgroupFactors => {
                                dispatch_gate!(grouped_gate_up_frozen_workgroup_grid_shared_factors)
                            }
                        }
                    }
                }
                if self.down_integer {
                    if phase == SourcePhase::PrepareAndDown {
                        if down != DownKernel::Frozen || staged_shard {
                            return Err(reject());
                        }
                        let (handles, scratch) =
                            down_integer_handles.as_ref().ok_or_else(reject)?;
                        (if count == 4 {
                            crate::quant_matvec::frozen_down_integer::four::launch
                        } else {
                            crate::quant_matvec::frozen_down_integer::launch
                        })(
                            client,
                            g,
                            s,
                            down_handle.clone(),
                            handles.clone(),
                            scratch.clone(),
                            FIRST[shard],
                            PARTS[shard],
                            groups,
                            count,
                        )?;
                        down_integer_dispatches = down_integer_dispatches
                            .checked_add(1)
                            .ok_or(Error::ArithmeticOverflow)?;
                    }
                } else if self.down_stride_four {
                    if down != DownKernel::FourRows || staged_shard {
                        return Err(reject());
                    }
                    // Same original source handles/runtime addresses and INFO; only
                    // the distinct entry/grid changes. Credit follows enqueue.
                    unsafe {
                        k::grouped_down_frozen_four_rows_stride_four::launch_unchecked::<WgpuRuntime>(
                            client,
                            CubeCount::Static(640, 4, 1),
                            CubeDim::new_1d(64),
                            ArrayArg::from_raw_parts(down_handle.clone(), down_words),
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
                            g.projections()[2].dtype(),
                            g.projections()[2].row_bytes(),
                            g.tile(),
                        );
                    }
                    down_stride_four_dispatches = down_stride_four_dispatches
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?;
                } else {
                    match down {
                        DownKernel::ThreeMembers => {
                            dispatch_down!(grouped_down_frozen_three_members)
                        }
                        DownKernel::FourRowsThreeMembers => {
                            dispatch_down!(grouped_down_frozen_four_rows_three_members)
                        }
                        DownKernel::StagedThreeMembers => {
                            dispatch_down!(grouped_down_frozen_staged_three_members)
                        }
                        DownKernel::StagedFourRowsThreeMembers => {
                            dispatch_down!(grouped_down_frozen_staged_four_rows_three_members)
                        }
                        DownKernel::Original => dispatch_down!(grouped_down),
                        DownKernel::StagedFrozen => dispatch_down!(grouped_down_frozen_staged),
                        DownKernel::StagedFourRows => {
                            dispatch_down!(grouped_down_frozen_staged_four_rows)
                        }
                        DownKernel::Frozen => dispatch_down!(grouped_down_frozen),
                        DownKernel::SharedScales => {
                            dispatch_down!(grouped_down_frozen_shared_scales)
                        }
                        DownKernel::FourRows => dispatch_down!(grouped_down_frozen_four_rows),
                        DownKernel::FourRowsSharedScales => {
                            dispatch_down!(grouped_down_frozen_four_rows_shared_scales)
                        }
                    }
                }
            }
        }
        let gu_integer = if let Some((_, scratch)) = integer_handles {
            (if count == 4 {
                crate::quant_matvec::frozen_gu_integer::four::audit
            } else {
                crate::quant_matvec::frozen_gu_integer::audit
            })(
                client,
                scratch,
                arena.handle(fault).cloned().map_err(u::fault)?,
                count,
            )?;
            let receipt = GuIntegerDispatch {
                layer,
                count,
                dispatches: integer_dispatches,
                stride_four_dispatches,
            };
            receipt.validate_selected(layer, count, self.gu_stride_four, self.four_members)?;
            Some(receipt)
        } else {
            None
        };
        let down_stride_four = if self.down_stride_four {
            let receipt = DownStrideFourDispatch {
                layer,
                count,
                dispatches: down_stride_four_dispatches,
            };
            receipt.validate(layer, count)?;
            Some(receipt)
        } else {
            None
        };
        let down_integer = if let Some((_, scratch)) = down_integer_handles {
            (if count == 4 {
                crate::quant_matvec::frozen_down_integer::four::audit
            } else {
                crate::quant_matvec::frozen_down_integer::audit
            })(
                client,
                scratch,
                arena.handle(fault).cloned().map_err(u::fault)?,
                count,
            )?;
            let receipt = DownIntegerDispatch {
                layer,
                count,
                preparations: down_integer_preparations,
                dispatches: down_integer_dispatches,
                audits: 1,
            };
            receipt.validate_selected(layer, count, self.four_members)?;
            Some(receipt)
        } else {
            None
        };
        Ok(SourceLaunch {
            gu_integer,
            down_stride_four,
            down_integer,
        })
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn frozen_down_integer_policy_preserves_all_original_source_types_and_rejects_competitors() {
        for mask in 0usize..512 {
            let f: [bool; 9] = std::array::from_fn(|i| mask & (1 << i) != 0);
            for selected in [false, true] {
                for exit in [false, true] {
                    assert_eq!(
                        Sources::validate_down_integer_policy(
                            selected, exit, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]
                        )
                        .is_ok(),
                        !selected || exit && mask == 0
                    );
                }
            }
        }
        for layer in 0..48 {
            let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
            let g = k::Geometry::new(
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
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
            crate::quant_matvec::frozen_down_integer::validate_geometry(g).unwrap();
            crate::quant_matvec::frozen_gu_integer::validate_geometry(g).unwrap();
            assert_eq!(kernel_policy(g), (GateKernel::Frozen, DownKernel::Frozen));
            assert!(!g.four_row_down());
            assert_eq!(g.tile(), 8);
            for gu in [false, true] {
                for stride in [false, true] {
                    assert_eq!(
                        Sources::validate_gu_schedule_policy(gu, stride).is_ok(),
                        !stride || gu
                    );
                }
            }
        }
    }
    #[test]
    fn frozen_down_integer_actual_receipt_requires_prepare_five_down_and_full_audit() {
        for layer in 0..48 {
            for count in 1..=3 {
                let r = DownIntegerDispatch {
                    layer,
                    count,
                    preparations: 1,
                    dispatches: 5,
                    audits: 1,
                };
                r.validate(layer, count).unwrap();
                assert!(r.validate((layer + 1) % 48, count).is_err());
                assert!(r.validate(layer, count + 1).is_err());
                for n in [0, 2, u64::MAX] {
                    assert!(
                        DownIntegerDispatch {
                            preparations: n,
                            ..r
                        }
                        .validate(layer, count)
                        .is_err()
                    );
                    assert!(
                        DownIntegerDispatch { audits: n, ..r }
                            .validate(layer, count)
                            .is_err()
                    );
                }
                for n in [0, 1, 4, 6, 10, u64::MAX] {
                    assert!(
                        DownIntegerDispatch { dispatches: n, ..r }
                            .validate(layer, count)
                            .is_err()
                    );
                }
            }
        }
        for count in [0, 4, 8] {
            assert!(
                DownIntegerDispatch {
                    layer: 0,
                    count,
                    preparations: 1,
                    dispatches: 5,
                    audits: 1
                }
                .validate(0, count)
                .is_err()
            );
        }
        assert!(
            DownIntegerDispatch {
                layer: 48,
                count: 1,
                preparations: 1,
                dispatches: 5,
                audits: 1
            }
            .validate(48, 1)
            .is_err()
        );
        use crate::quant_matvec::frozen_down_integer as i;
        assert_eq!(
            (
                i::SCRATCH_WORDS,
                i::MAIN_BYTES,
                i::HOST_INCREMENT,
                i::INFO_INCREMENT,
                i::RUNTIME_INCREMENT
            ),
            (
                FROZEN_DOWN_INTEGER_SCRATCH_WORDS,
                FROZEN_DOWN_INTEGER_MAIN_BYTES,
                FROZEN_DOWN_INTEGER_HOST_BYTES,
                FROZEN_DOWN_INTEGER_INFO_BYTES,
                FROZEN_DOWN_INTEGER_RUNTIME_BYTES
            )
        );
        assert_eq!(i::LOGICAL_WORDS, 10864);
        assert_eq!(8 + 3 + 2 + PAGES, 28); // Two distinct integer roots fit original30 metadata bound.
        const {
            assert!(28 <= ROUTER_EXTENTS);
        };
    }
    #[test]
    fn frozen_down_integer_completed_receipt_never_credits_attempts_or_missing_source_shards() {
        assert!(!down_integer_completed_flag(false, 0, 0, 0, 480).unwrap());
        for n in [1u64, 48, 48 * 74] {
            assert!(down_integer_completed_flag(true, n, 5 * n, n, 10 * n).unwrap());
            for source in [0, 10 * n - 1, 10 * n + 1] {
                assert!(down_integer_completed_flag(true, n, 5 * n, n, source).is_err());
            }
        }
        for (p, d, a) in [
            (0, 0, 0),
            (1, 4, 1),
            (1, 6, 1),
            (1, 5, 0),
            (1, 5, 2),
            (u64::MAX, u64::MAX, u64::MAX),
        ] {
            assert!(down_integer_completed_flag(true, p, d, a, 10).is_err());
        }
        for (p, d, a) in [(1, 0, 0), (0, 5, 0), (0, 0, 1), (1, 5, 1)] {
            assert!(down_integer_completed_flag(false, p, d, a, 10).is_err());
        }
    }
    #[test]
    fn frozen_down_integer_real_schedule_finishes_every_gate_before_route_preparation() {
        fn events(integer: bool) -> Vec<(char, usize)> {
            let mut e = Vec::new();
            for &phase in source_phases(integer) {
                if phase == SourcePhase::PrepareAndDown {
                    e.push(('P', 0));
                }
                for shard in 0..5 {
                    if phase != SourcePhase::PrepareAndDown {
                        e.push(('G', shard));
                    }
                    if !integer || phase == SourcePhase::PrepareAndDown {
                        e.push(('D', shard));
                    }
                }
            }
            if integer {
                e.push(('A', 0));
            }
            e
        }
        assert_eq!(
            events(false),
            (0..5)
                .flat_map(|i| [('G', i), ('D', i)])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            events(true),
            vec![
                ('G', 0),
                ('G', 1),
                ('G', 2),
                ('G', 3),
                ('G', 4),
                ('P', 0),
                ('D', 0),
                ('D', 1),
                ('D', 2),
                ('D', 3),
                ('D', 4),
                ('A', 0)
            ]
        );
        assert_eq!(FIRST, [0, 72, 144, 288, 432]);
        assert_eq!(PARTS, [72, 72, 144, 144, 80]);
        assert_eq!(PARTS.iter().sum::<usize>(), 512);
    }

    use super::*;
    #[test]
    fn down_stride_four_policy_requires_four_rows_and_rejects_every_competing_source_mode() {
        for selected in [false, true] {
            for bits in 0u16..512 {
                assert_eq!(
                    Sources::validate_down_stride_four_policy(
                        selected,
                        bits & 1 != 0,
                        bits & 2 != 0,
                        bits & 4 != 0,
                        bits & 8 != 0,
                        bits & 16 != 0,
                        bits & 32 != 0,
                        bits & 64 != 0,
                        bits & 128 != 0,
                        bits & 256 != 0,
                    )
                    .is_ok(),
                    !selected || bits == 3
                );
            }
        }
        // GU arithmetic/scheduling remains an independent axis, including the
        // original plain GU path; DOWN has no prepared activation owner.
        for gu_integer in [false, true] {
            for gu_stride_four in [false, true] {
                assert_eq!(
                    Sources::validate_gu_schedule_policy(gu_integer, gu_stride_four).is_ok(),
                    !gu_stride_four || gu_integer
                );
                Sources::validate_down_stride_four_policy(
                    true, true, true, false, false, false, false, false, false, false,
                )
                .unwrap();
            }
        }
    }
    #[test]
    fn down_stride_four_completed_flag_requires_actual_positive_five_per_layer() {
        assert!(!down_stride_four_completed_flag(false, 0, 0, 480).unwrap());
        assert!(down_stride_four_completed_flag(true, 1, 5, 10).unwrap());
        assert!(down_stride_four_completed_flag(true, 48, 240, 480).unwrap());
        for (layers, down, source) in [
            (0, 0, 0),
            (0, 5, 10),
            (1, 4, 10),
            (1, 6, 10),
            (1, 5, 0),
            (1, 5, 9),
            (1, 5, 11),
            (u64::MAX, u64::MAX, u64::MAX),
        ] {
            assert!(down_stride_four_completed_flag(true, layers, down, source).is_err());
        }
        for (layers, down) in [(1, 0), (0, 5), (1, 5)] {
            assert!(down_stride_four_completed_flag(false, layers, down, 10).is_err());
        }
    }
    #[test]
    fn down_stride_four_actual_receipt_requires_exact_layer_count_and_all_five_sources() {
        for layer in 0..48 {
            for count in 1..=3 {
                let receipt = DownStrideFourDispatch {
                    layer,
                    count,
                    dispatches: 5,
                };
                receipt.validate(layer, count).unwrap();
                assert!(receipt.validate((layer + 1) % 48, count).is_err());
                assert!(receipt.validate(layer, count + 1).is_err());
                for dispatches in [0, 1, 4, 6, 10] {
                    assert!(
                        DownStrideFourDispatch {
                            dispatches,
                            ..receipt
                        }
                        .validate(layer, count)
                        .is_err()
                    );
                }
            }
        }
        for count in [0, 4, 8] {
            assert!(
                DownStrideFourDispatch {
                    layer: 0,
                    count,
                    dispatches: 5
                }
                .validate(0, count)
                .is_err()
            );
        }
        assert!(
            DownStrideFourDispatch {
                layer: 48,
                count: 1,
                dispatches: 5
            }
            .validate(48, 1)
            .is_err()
        );
    }
    #[test]
    fn frozen_gu_integer_policy_rejects_every_conflict_and_keeps_original_down_geometry() {
        for selected in [false, true] {
            for bits in 0u16..256 {
                let exit = bits & 1 != 0;
                let private = bits & 2 != 0;
                let workgroup = bits & 4 != 0;
                let scales = bits & 8 != 0;
                let factors = bits & 16 != 0;
                let paired = bits & 32 != 0;
                let staging = bits & 64 != 0;
                let members = bits & 128 != 0;
                assert_eq!(
                    Sources::validate_gu_integer_policy(
                        selected, exit, private, workgroup, scales, factors, paired, staging,
                        members
                    )
                    .is_ok(),
                    !selected || (exit && bits == 1)
                );
            }
        }
        // The actual 48-layer profile, not a synthetic cross-product of dtypes.
        for layer in 0..48 {
            let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
            let projections = [
                k::Projection::new(gate, 2560, 640, 512).unwrap(),
                k::Projection::new(up, 2560, 640, 512).unwrap(),
                k::Projection::new(down, 640, 2560, 512).unwrap(),
            ];
            let original = k::Geometry::new(projections, 8, 144)
                .unwrap()
                .with_frozen_host_sources()
                .unwrap()
                .with_inactive_partition_exit()
                .unwrap();
            for four_down in [false, true] {
                let g = if four_down {
                    original.with_four_row_down().unwrap()
                } else {
                    original
                };
                crate::quant_matvec::frozen_gu_integer::validate_geometry(g).unwrap();
                let (gate, down) = source_kernel_policy(g, false).unwrap();
                assert_eq!(gate, GateKernel::Frozen);
                assert_eq!(
                    down,
                    if four_down {
                        DownKernel::FourRows
                    } else {
                        DownKernel::Frozen
                    }
                );
                assert_eq!(g.tile(), 8);
                assert_eq!(g.projections(), projections);
                assert_eq!(g.member_capacity(), original.member_capacity());
                assert_eq!(g.capacity(), 512);
                assert_eq!(k::MAP_WORDS, 804);
            }
        }
    }

    #[test]
    fn frozen_gu_integer_completed_flag_requires_positive_specific_joined_counts() {
        assert!(!gu_integer_completed_flag(false, 0, 0, 0).unwrap());
        assert!(gu_integer_completed_flag(true, 1, 5, 1).unwrap());
        assert!(gu_integer_completed_flag(true, 48, 240, 48).unwrap());
        for (p, d, a) in [
            (0, 0, 0),
            (0, 5, 0),
            (1, 0, 1),
            (1, 4, 1),
            (1, 6, 1),
            (1, 5, 0),
            (1, 5, 2),
            (u64::MAX, u64::MAX, u64::MAX),
        ] {
            assert!(gu_integer_completed_flag(true, p, d, a).is_err());
        }
        for (p, d, a) in [(1, 0, 0), (0, 5, 0), (0, 0, 1), (1, 5, 1)] {
            assert!(gu_integer_completed_flag(false, p, d, a).is_err());
        }
    }
    #[test]
    fn frozen_gu_integer_actual_launch_receipt_requires_all_shards_exact_layer_and_count() {
        for layer in 0..48 {
            for count in 1..=3 {
                let receipt = GuIntegerDispatch {
                    layer,
                    count,
                    dispatches: 5,
                    stride_four_dispatches: 0,
                };
                receipt.validate(layer, count, false).unwrap();
                assert!(receipt.validate(layer, count, true).is_err());
                assert!(receipt.validate((layer + 1) % 48, count, false).is_err());
                assert!(receipt.validate(layer, count + 1, false).is_err());
                for dispatches in [0, 1, 4, 6, 10] {
                    assert!(
                        GuIntegerDispatch {
                            dispatches,
                            ..receipt
                        }
                        .validate(layer, count, false)
                        .is_err()
                    );
                }
            }
        }
        for count in [0, 4, 8] {
            assert!(
                GuIntegerDispatch {
                    layer: 0,
                    count,
                    dispatches: 5,
                    stride_four_dispatches: 0,
                }
                .validate(0, count, false)
                .is_err()
            );
        }
        assert!(
            GuIntegerDispatch {
                layer: 48,
                count: 1,
                dispatches: 5,
                stride_four_dispatches: 0,
            }
            .validate(48, 1, false)
            .is_err()
        );
        // Exact parent inventory matches the leaf rather than an untracked tail.
        use crate::quant_matvec::frozen_gu_integer as integer;
        assert_eq!(integer::SCRATCH_WORDS, FROZEN_GU_INTEGER_SCRATCH_WORDS);
        assert_eq!(integer::MAIN_BYTES, FROZEN_GU_INTEGER_MAIN_BYTES);
        assert_eq!(integer::HOST_INCREMENT, FROZEN_GU_INTEGER_HOST_BYTES);
        assert_eq!(integer::INFO_INCREMENT, FROZEN_GU_INTEGER_INFO_BYTES);
        assert_eq!(integer::RUNTIME_INCREMENT, FROZEN_GU_INTEGER_RUNTIME_BYTES);
        assert_eq!(integer::LOGICAL_WORDS, 4384);
        assert_eq!(ROUTER_EXTENTS, 30);
    }
    #[test]
    fn frozen_gu_integer_stride_four_requires_integer_and_actual_joined_leaf_receipts() {
        for integer in [false, true] {
            for selected in [false, true] {
                assert_eq!(
                    Sources::validate_gu_schedule_policy(integer, selected).is_ok(),
                    !selected || integer
                );
            }
        }
        assert!(!gu_stride_four_completed_flag(false, false, 0, 0, 0, 0).unwrap());
        assert!(!gu_stride_four_completed_flag(true, false, 48, 240, 48, 0).unwrap());
        assert!(gu_stride_four_completed_flag(true, true, 48, 240, 48, 240).unwrap());
        for (integer, selected, p, d, a, s) in [
            (false, true, 0, 0, 0, 0),
            (true, true, 0, 0, 0, 0),
            (true, true, 48, 240, 48, 0),
            (true, true, 48, 240, 48, 239),
            (true, true, 48, 240, 48, 241),
            (true, true, 48, 240, 47, 240),
            (true, false, 48, 240, 48, 240),
        ] {
            assert!(gu_stride_four_completed_flag(integer, selected, p, d, a, s).is_err());
        }
        for layer in 0..48 {
            for count in 1..=3 {
                let receipt = GuIntegerDispatch {
                    layer,
                    count,
                    dispatches: 5,
                    stride_four_dispatches: 5,
                };
                receipt.validate(layer, count, true).unwrap();
                assert!(receipt.validate(layer, count, false).is_err());
                for stride_four_dispatches in [0, 1, 4, 6, 10] {
                    assert!(
                        GuIntegerDispatch {
                            stride_four_dispatches,
                            ..receipt
                        }
                        .validate(layer, count, true)
                        .is_err()
                    );
                }
            }
        }
    }
    #[test]
    fn three_member_actual_entries_keep_original_tile_owners_and_stage_composition() {
        assert!(
            Sources::validate_three_member_policy(false, false, true, true, true, true, true)
                .is_ok()
        );
        assert!(
            Sources::validate_three_member_policy(true, false, false, false, false, false, false)
                .is_err()
        );
        for flags in [
            [true, false, false, false, false],
            [false, true, false, false, false],
            [false, false, true, false, false],
            [false, false, false, true, false],
            [false, false, false, false, true],
        ] {
            assert!(
                Sources::validate_three_member_policy(
                    true, true, flags[0], flags[1], flags[2], flags[3], flags[4]
                )
                .is_err()
            );
        }
        for layer in 0..48 {
            let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
            let base = k::Geometry::new(
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
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
            for four in [false, true] {
                let g = if four {
                    base.with_four_row_down().unwrap()
                } else {
                    base
                };
                assert_eq!(three_member_geometry(g, false).unwrap(), g);
                let g = three_member_geometry(g, true).unwrap();
                assert_eq!(g.tile(), 8);
                assert_eq!(g.arithmetic_members(), 3);
                assert_eq!(
                    source_kernel_policy(g, false).unwrap(),
                    (
                        GateKernel::ThreeMembers,
                        if four {
                            DownKernel::FourRowsThreeMembers
                        } else {
                            DownKernel::ThreeMembers
                        }
                    )
                );
                assert_eq!(
                    source_kernel_policy(g, true).unwrap(),
                    (
                        GateKernel::StagedThreeMembers,
                        if four {
                            DownKernel::StagedFourRowsThreeMembers
                        } else {
                            DownKernel::StagedThreeMembers
                        }
                    )
                );
                assert_eq!(
                    k::frozen_host_staging::Shape::new(g).unwrap().stage_words(),
                    [8388608; 3]
                );
            }
        }
    }
    #[test]
    fn packed_staging_source_scope_and_actual_host_dispatch_are_sealed() {
        assert!(Sources::validate_staging_policy(false, false, true, true, true, true).is_ok());
        assert!(Sources::validate_staging_policy(true, false, false, false, false, false).is_err());
        for flags in [
            [true, false, false, false],
            [false, true, false, false],
            [false, false, true, false],
            [false, false, false, true],
        ] {
            assert!(
                Sources::validate_staging_policy(
                    true, true, flags[0], flags[1], flags[2], flags[3]
                )
                .is_err()
            );
        }
        for layer in 0..48 {
            let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
            let base = k::Geometry::new(
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
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
            for two in [false, true] {
                for four in [false, true] {
                    let g = if two {
                        base.with_two_row_gate_up().unwrap()
                    } else {
                        base
                    };
                    let g = if four {
                        g.with_four_row_down().unwrap()
                    } else {
                        g
                    };
                    assert_eq!(source_kernel_policy(g, false).unwrap(), kernel_policy(g));
                    assert_eq!(
                        source_kernel_policy(g, true).unwrap(),
                        (
                            if two {
                                GateKernel::StagedTwoRows
                            } else {
                                GateKernel::StagedFrozen
                            },
                            if four {
                                DownKernel::StagedFourRows
                            } else {
                                DownKernel::StagedFrozen
                            },
                        )
                    );
                    let shape = k::frozen_host_staging::Shape::new(g).unwrap();
                    assert_eq!(
                        shape.stage_words().map(|w| w as u64 * 4),
                        [FROZEN_STAGING_OWNER_BYTES; 3]
                    );
                }
            }
            assert!(source_kernel_policy(base.with_private_tables().unwrap(), true).is_err());
            assert!(source_kernel_policy(base.with_shared_down_scales().unwrap(), true).is_err());
        }
    }
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
                assert_eq!(
                    kernel_policy(factors.with_shared_down_scales().unwrap()),
                    (factor, DownKernel::SharedScales)
                );
                for (geometry, expected_gate) in [(selected, table), (factors, factor)] {
                    let rows = geometry.with_four_row_down().unwrap();
                    assert_eq!(kernel_policy(rows), (expected_gate, DownKernel::FourRows));
                    assert_eq!(
                        kernel_policy(rows.with_shared_down_scales().unwrap()),
                        (expected_gate, DownKernel::FourRowsSharedScales)
                    );
                    assert!(matches!(
                        rows.grid(30, true).unwrap(),
                        CubeCount::Static(640, 30, 1)
                    ));
                    assert!(matches!(
                        geometry.grid(30, true).unwrap(),
                        CubeCount::Static(2560, 30, 1)
                    ));
                }
                assert_eq!(kernel_policy(selected), (table, DownKernel::Frozen));
                assert_eq!(
                    kernel_policy(selected.with_shared_down_scales().unwrap()),
                    (table, DownKernel::SharedScales)
                );
            }
        }
    }
    #[test]
    fn two_row_gate_up_attachment_flags_reject_incompatible_policies_before_import() {
        assert!(
            Sources::validate_kernel_policies(false, false, false, false, false, false, false)
                .is_ok()
        );
        assert!(
            Sources::validate_kernel_policies(false, false, false, false, false, false, true)
                .is_err()
        );
        for down_scales in [false, true] {
            for down_rows in [false, true] {
                assert!(
                    Sources::validate_kernel_policies(
                        true,
                        false,
                        false,
                        down_scales,
                        false,
                        down_rows,
                        true
                    )
                    .is_ok()
                );
                for (private, workgroup, factors) in [
                    (true, false, false),
                    (false, true, false),
                    (false, false, true),
                ] {
                    assert!(
                        Sources::validate_kernel_policies(
                            true,
                            private,
                            workgroup,
                            down_scales,
                            factors,
                            down_rows,
                            true
                        )
                        .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn two_row_gate_up_actual_dispatch_uses_paired_grid_and_independent_down_choices() {
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
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
            assert_eq!(
                kernel_policy(original),
                (GateKernel::Frozen, DownKernel::Frozen)
            );
            let paired = original.with_two_row_gate_up().unwrap();
            assert_eq!(
                kernel_policy(paired),
                (GateKernel::TwoRows, DownKernel::Frozen)
            );
            assert!(matches!(
                paired.grid(30, false).unwrap(),
                CubeCount::Static(320, 30, 1)
            ));
            assert!(paired.with_private_tables().is_err());
            assert!(paired.with_workgroup_tables().is_err());
            assert!(paired.with_shared_gate_up_factors().is_err());
            assert_eq!(
                kernel_policy(paired.with_shared_down_scales().unwrap()),
                (GateKernel::TwoRows, DownKernel::SharedScales)
            );
            let down_rows = paired.with_four_row_down().unwrap();
            assert_eq!(
                kernel_policy(down_rows),
                (GateKernel::TwoRows, DownKernel::FourRows)
            );
            assert_eq!(
                kernel_policy(down_rows.with_shared_down_scales().unwrap()),
                (GateKernel::TwoRows, DownKernel::FourRowsSharedScales)
            );
            assert_eq!(
                down_rows.with_shared_down_scales().unwrap(),
                original
                    .with_shared_down_scales()
                    .unwrap()
                    .with_four_row_down()
                    .unwrap()
                    .with_two_row_gate_up()
                    .unwrap()
            );
            assert!(matches!(
                down_rows.grid(30, true).unwrap(),
                CubeCount::Static(640, 30, 1)
            ));
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
