//! Candidate grouped all-ten routed MoE operators, not a graph transaction.
//!
//! Original PUBLIC banks stay tightly packed and unchanged. Every selection,
//! slot, group/member map, count, activation address and probability is runtime
//! private data. Geometry/dtypes and public tile policy alone specialize.
//! Tile1 is an explicit single-member decode policy, not a private count choice.
//! A caller must pin BOTH its immutable token wave and actual pending BankSlots
//! protection through completed use; a borrowed raw slot slice is not that proof.
//! No bank fill, commit, graph hook, CPU fallback or device action lives here.
//! Parent owns physical owner/alias/epoch/device-limit validation and acceptance.

use super::{
    IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, Q2_0, Q4_K, binary16, iq_gate_up_factor,
    iq_gate_up_value_with_factor, iq4_nl_scale, iq4_nl_value_with_scale, packed_value,
};
use cubecl::prelude::*;
use zeroize::Zeroize;

pub const MAX_BATCH: usize = 8;
pub const RANKS: usize = 10;
pub const MAX_ROUTES: usize = MAX_BATCH * RANKS;
pub const MAX_GROUPS: usize = MAX_ROUTES;
pub const LANES: usize = 64;
pub const MAP_WORDS: usize = 4 + MAX_GROUPS * 2 + MAX_GROUPS * MAX_BATCH;
const SLOTS: usize = 4;
const LENGTHS: usize = SLOTS + MAX_GROUPS;
const MEMBERS: usize = LENGTHS + MAX_GROUPS;
const MAX_BANK_BYTES: usize = 128 * 1024 * 1024;

#[path = "frozen_host_staging.rs"]
pub(crate) mod frozen_host_staging;

/// Request-derived reduction/coverage scratch has an owning guard BEFORE any
/// private fill, including early error and unwinding paths. Fixed public shape;
/// no Copy/Clone/Debug/Hash/serialization and no unguarded temporary Vec.
struct RouteScratch {
    probability_sum: f64,
    covered: [u8; MAX_ROUTES],
}
impl RouteScratch {
    fn empty() -> Self {
        Self {
            probability_sum: 0.,
            covered: [0; MAX_ROUTES],
        }
    }
}
impl Zeroize for RouteScratch {
    fn zeroize(&mut self) {
        self.probability_sum.zeroize();
        self.covered.zeroize();
        #[cfg(test)]
        {
            assert_eq!(self.probability_sum.to_bits(), 0);
            assert!(self.covered.iter().all(|&v| v == 0));
            SCRATCH_WIPES.with(|n| n.set(n.get() + 1));
        }
    }
}
#[cfg(test)]
thread_local! { static SCRATCH_WIPES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

/// Original immutable tensor metadata. It contains no selected expert/slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Projection {
    dtype: u32,
    columns: usize,
    rows: usize,
    experts: usize,
    row_bytes: usize,
    expert_bytes: usize,
}
impl Projection {
    pub fn new(
        dtype: u32,
        columns: usize,
        rows: usize,
        experts: usize,
    ) -> Result<Self, &'static str> {
        let (block, bytes) = super::block_layout(dtype).ok_or("grouped MoE dtype")?;
        if columns == 0 || rows == 0 || experts != 512 || !columns.is_multiple_of(block) {
            return Err("grouped MoE original block geometry");
        }
        let row_bytes = (columns / block)
            .checked_mul(bytes)
            .ok_or("grouped row overflow")?;
        let expert_bytes = rows
            .checked_mul(row_bytes)
            .ok_or("grouped expert overflow")?;
        super::validate_layout(dtype, rows, columns, expert_bytes, columns, rows)?;
        Ok(Self {
            dtype,
            columns,
            rows,
            experts,
            row_bytes,
            expert_bytes,
        })
    }
    pub fn dtype(self) -> u32 {
        self.dtype
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn experts(self) -> usize {
        self.experts
    }
    pub fn row_bytes(self) -> usize {
        self.row_bytes
    }
    pub fn expert_bytes(self) -> usize {
        self.expert_bytes
    }
    pub fn segment_words(self, capacity: usize) -> Result<usize, &'static str> {
        let bytes = self
            .expert_bytes
            .checked_mul(capacity)
            .ok_or("grouped bank overflow")?;
        if capacity == 0
            || !self.expert_bytes.is_multiple_of(4)
            || bytes > MAX_BANK_BYTES
            || bytes > u32::MAX as usize
        {
            return Err("grouped bank binding extent");
        }
        Ok(bytes.div_ceil(4))
    }
}

/// Public kernel policy; never stores the runtime group/route map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GatePolicy {
    #[default]
    Generic,
    Q4Block,
}

/// Immutable public reduction policy. Runtime group lengths never enter the
/// compilation key, dispatch geometry, INFO record or owning reservation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReductionPolicy {
    #[default]
    AllMembers,
    // Public lab consumers select this policy in a separate integration leaf.
    #[allow(dead_code)]
    LiveMembers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    projections: [Projection; 3],
    hidden: usize,
    ff: usize,
    tile: usize,
    capacity: usize,
    gate_policy: GatePolicy,
    reduction_policy: ReductionPolicy,
    secondary_partition: bool,
    frozen_host_sources: bool,
    inactive_partition_exit: bool,
    private_tables: bool,
    workgroup_tables: bool,
    shared_down_scales: bool,
    shared_gate_up_factors: bool,
    four_row_down: bool,
    two_row_gate_up: bool,
    three_member_arithmetic: bool,
    four_members: bool,
}
// Retain every old geometry hash payload when unselected. The explicit public
// fourth-member policy has an additional discriminator, never private values.
impl std::hash::Hash for Geometry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.projections.hash(state);
        self.hidden.hash(state);
        self.ff.hash(state);
        self.tile.hash(state);
        self.capacity.hash(state);
        self.gate_policy.hash(state);
        self.reduction_policy.hash(state);
        self.secondary_partition.hash(state);
        self.frozen_host_sources.hash(state);
        self.inactive_partition_exit.hash(state);
        self.private_tables.hash(state);
        self.workgroup_tables.hash(state);
        self.shared_down_scales.hash(state);
        self.shared_gate_up_factors.hash(state);
        self.four_row_down.hash(state);
        self.two_row_gate_up.hash(state);
        self.three_member_arithmetic.hash(state);
        if self.four_members {
            "frozen-four-members".hash(state);
        }
    }
}
impl Geometry {
    pub fn new(
        projections: [Projection; 3],
        tile: usize,
        capacity: usize,
    ) -> Result<Self, &'static str> {
        let [gate, up, down] = projections;
        if !matches!(tile, 1 | 4 | 8)
            || !crate::bank_slots::routed_capacity(capacity)
            || gate.columns != up.columns
            || gate.rows != up.rows
            || down.columns != gate.rows
            || down.rows != gate.columns
            || gate.columns > 4096
            || gate.rows > 4096
        {
            return Err("grouped MoE public geometry");
        }
        // Explicit segmented profiles retain original packed model metadata.
        // The accepted artifact families have equal gate/up dtypes; no Q4-only
        // arithmetic policy is selected for the nonlinear IQ family.
        if crate::bank_slots::segmented_capacity(capacity)
            && (!(gate.dtype == super::Q4_K
                && up.dtype == super::Q4_K
                && matches!(down.dtype, super::Q5_0 | super::Q8_0)
                || matches!(gate.dtype, super::IQ3_XXS | super::IQ4_XS)
                    && up.dtype == gate.dtype
                    && matches!(down.dtype, super::IQ4_NL | super::Q8_0)
                || matches!(gate.dtype, super::IQ3_XXS | super::IQ2_S | super::IQ3_S)
                    && up.dtype == gate.dtype
                    && matches!(down.dtype, super::IQ4_NL | super::Q2_0))
                || gate.columns != 2560
                || gate.rows != 640)
        {
            return Err("grouped segmented original metadata");
        }
        for projection in projections {
            for &count in crate::bank_slots::segmented_capacities(capacity).unwrap_or(&[capacity]) {
                if count != 0 {
                    projection.segment_words(count)?;
                }
            }
        }
        Ok(Self {
            projections,
            hidden: gate.columns,
            ff: gate.rows,
            tile,
            capacity,
            gate_policy: GatePolicy::Generic,
            reduction_policy: ReductionPolicy::AllMembers,
            secondary_partition: false,
            frozen_host_sources: false,
            inactive_partition_exit: false,
            private_tables: false,
            workgroup_tables: false,
            shared_down_scales: false,
            shared_gate_up_factors: false,
            four_row_down: false,
            two_row_gate_up: false,
            three_member_arithmetic: false,
            four_members: false,
        })
    }
    /// Original full512 physical-cell addressing, without BankSlots or fixed
    /// source partitions. Only the connected native global family consumes it.
    pub(crate) fn global_sources(projections: [Projection; 3]) -> Result<Self, &'static str> {
        let mut geometry = Self::new(projections, 8, crate::daslab_profile::BANK_SLOTS)?;
        if !crate::strata_intel_core::enabled()
            || projections.iter().any(|p| p.experts() != 512)
            || geometry.hidden() != 2560
            || geometry.ff() != 640
        {
            return Err("global native expert geometry");
        }
        geometry.capacity = 512;
        Ok(geometry)
    }
    /// Explicit source-only public profile. These virtual slots are NOT
    /// BankSlots capacity; the ordinary constructor continues to reject512.
    pub(crate) fn with_frozen_host_sources(mut self) -> Result<Self, &'static str> {
        let [gate, up, down] = self.projections;
        if self.capacity != crate::daslab_profile::BANK_SLOTS
            || self.tile != 8
            || self.secondary_partition
            || self.frozen_host_sources
            || self.gate_policy != GatePolicy::Generic
            || gate.columns != 2560
            || gate.rows != 640
            || up != gate
            || !matches!(
                gate.dtype,
                super::IQ3_XXS | super::IQ2_S | super::IQ3_S | super::IQ4_XS
            )
            || !matches!(down.dtype, super::IQ4_NL | super::Q2_0)
        {
            return Err("grouped frozen host source profile");
        }
        for projection in self.projections {
            for slots in crate::daslab_profile::SOURCE_PARTS {
                projection.segment_words(slots)?;
            }
        }
        self.capacity = 512;
        self.frozen_host_sources = true;
        Ok(self)
    }
    pub(crate) fn frozen_host_sources(self) -> bool {
        self.frozen_host_sources
    }
    /// Construction-fixed source-only candidate. The default frozen source
    /// geometry retains the original kernels, including their inactive trees.
    pub(crate) fn with_inactive_partition_exit(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || self.capacity != 512
            || self.tile != 8
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.inactive_partition_exit
        {
            return Err("grouped frozen inactive-partition policy");
        }
        self.inactive_partition_exit = true;
        Ok(self)
    }
    pub(crate) fn inactive_partition_exit(self) -> bool {
        self.inactive_partition_exit
    }
    /// Explicit public table-storage experiment. The original64 arithmetic,
    /// owners and bindings remain identical to the frozen exit control.
    pub(crate) fn with_private_tables(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.private_tables
            || self.workgroup_tables
            || self.two_row_gate_up
            || self.three_member_arithmetic
            || self.four_members
        {
            return Err("grouped frozen private-table policy");
        }
        self.private_tables = true;
        Ok(self)
    }
    pub(crate) fn private_tables(self) -> bool {
        self.private_tables
    }
    /// Explicit public workgroup codebooks; exact original64 math and owners.
    pub(crate) fn with_workgroup_tables(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.private_tables
            || self.workgroup_tables
            || self.two_row_gate_up
            || self.three_member_arithmetic
            || self.four_members
        {
            return Err("grouped frozen workgroup-table policy");
        }
        self.workgroup_tables = true;
        Ok(self)
    }
    pub(crate) fn workgroup_tables(self) -> bool {
        self.workgroup_tables
    }
    /// Explicit frozen IQ4_NL scale-sharing candidate. Q2_0 keeps its exact
    /// original decoder; no source binding, MAIN owner or INFO span is added.
    #[allow(dead_code)]
    pub(crate) fn with_shared_down_scales(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.shared_down_scales
            || self.three_member_arithmetic
            || self.four_members
        {
            return Err("grouped frozen shared-down-scale policy");
        }
        self.shared_down_scales = true;
        Ok(self)
    }
    pub(crate) fn shared_down_scales(self) -> bool {
        self.shared_down_scales
    }
    /// Original IQ2/IQ3 subgroup factors shared by the active frozen row.
    /// Public codebook placement and down scales remain independent policies.
    #[allow(dead_code)]
    pub(crate) fn with_shared_gate_up_factors(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.shared_gate_up_factors
            || self.two_row_gate_up
            || self.three_member_arithmetic
            || self.four_members
        {
            return Err("grouped frozen shared-gate-up-factor policy");
        }
        self.shared_gate_up_factors = true;
        Ok(self)
    }
    pub(crate) fn shared_gate_up_factors(self) -> bool {
        self.shared_gate_up_factors
    }
    /// Two complete GU output rows per64-thread cube. Each thread retains two
    /// original virtual lanes; table/factor microexperiments are incompatible.
    #[allow(dead_code)]
    pub(crate) fn with_two_row_gate_up(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.hidden != 2560
            || self.ff != 640
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.private_tables
            || self.workgroup_tables
            || self.shared_gate_up_factors
            || self.two_row_gate_up
            || !matches!(
                self.projections[0].dtype,
                IQ2_S | IQ3_XXS | IQ3_S | super::IQ4_XS
            )
            || self.projections[0] != self.projections[1]
            || self.three_member_arithmetic
            || self.four_members
        {
            return Err("grouped frozen two-row-gate-up policy");
        }
        self.two_row_gate_up = true;
        Ok(self)
    }
    #[allow(dead_code)]
    pub(crate) fn two_row_gate_up(self) -> bool {
        self.two_row_gate_up
    }
    /// Four output rows per64-thread cube, retaining the original64 virtual
    /// lane accumulators and exact tree. Storage owners and route maps do not
    /// change; the matching explicit down entrypoint must consume this grid.
    #[allow(dead_code)]
    pub(crate) fn with_four_row_down(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.hidden != 2560
            || self.ff != 640
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || !matches!(self.projections[2].dtype, super::IQ4_NL | super::Q2_0)
            || self.four_row_down
            || self.four_members
        {
            return Err("grouped frozen four-row-down policy");
        }
        self.four_row_down = true;
        Ok(self)
    }
    #[allow(dead_code)]
    pub(crate) fn four_row_down(self) -> bool {
        self.four_row_down
    }
    /// Fixed public arithmetic capacity for the frozen maximum-three-row path.
    /// MAP, tile8 coordinates, full MAX8 owners and MAX80 status stay unchanged.
    #[allow(dead_code)]
    pub(crate) fn with_three_member_arithmetic(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.hidden != 2560
            || self.ff != 640
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.private_tables
            || self.workgroup_tables
            || self.shared_gate_up_factors
            || self.shared_down_scales
            || self.two_row_gate_up
            || self.three_member_arithmetic
            || self.four_members
        {
            return Err("grouped frozen three-member arithmetic policy");
        }
        self.three_member_arithmetic = true;
        Ok(self)
    }
    #[allow(dead_code)]
    pub(crate) fn three_member_arithmetic(self) -> bool {
        self.three_member_arithmetic
    }
    /// Arithmetic scratch size only; never a route/member/storage capacity.
    #[allow(dead_code)]
    pub(crate) fn arithmetic_members(self) -> usize {
        if self.three_member_arithmetic {
            3
        } else {
            self.tile
        }
    }
    /// Per-dispatch shared scratch, not a MAIN allocation or owning charge.
    #[allow(dead_code)]
    pub(crate) fn down_shared_memory_bytes(self) -> usize {
        let rows = if self.four_row_down { 4 } else { 1 };
        LANES * self.arithmetic_members() * std::mem::size_of::<f32>()
            + if self.shared_down_scales && self.projections[2].dtype == super::IQ4_NL {
                rows * (self.ff / 32) * std::mem::size_of::<f32>()
            } else {
                0
            }
    }
    /// Narrow immutable32+16 partition; ordinary flat48 remains unchanged.
    pub(crate) fn stationary_secondary(projections: [Projection; 3]) -> Result<Self, &'static str> {
        let [gate, up, down] = projections;
        if gate.dtype != Q4_K
            || up != gate
            || gate.columns != 2560
            || gate.rows != 640
            || gate.experts != 512
            || !matches!(down.dtype, super::Q5_0 | super::Q8_0)
        {
            return Err("secondary original geometry");
        }
        let mut g = Self::new(projections, 1, 48)?;
        g.secondary_partition = true;
        Ok(g)
    }
    pub fn projections(self) -> [Projection; 3] {
        self.projections
    }
    pub fn hidden(self) -> usize {
        self.hidden
    }
    pub fn ff(self) -> usize {
        self.ff
    }
    pub fn tile(self) -> usize {
        self.tile
    }
    pub fn gate_policy(self) -> GatePolicy {
        self.gate_policy
    }
    #[allow(dead_code)]
    pub fn live_member_reduction(self) -> bool {
        self.reduction_policy == ReductionPolicy::LiveMembers
    }
    /// Original F32 tree on the actual guarded members only. A public wrapper
    /// must choose the matching *_live entrypoints; ordinary entrypoints retain
    /// their original complete-tile reduction. All barriers remain unconditional.
    #[allow(dead_code)]
    pub fn with_live_member_reduction(mut self) -> Result<Self, &'static str> {
        if !matches!(self.tile, 4 | 8)
            || self.gate_policy != GatePolicy::Generic
            || self.secondary_partition
            || self.inactive_partition_exit
            || self.four_members
        {
            return Err("grouped live-member public policy");
        }
        self.reduction_policy = ReductionPolicy::LiveMembers;
        Ok(self)
    }
    /// Explicit public decode specialization; never selected from a route count.
    pub fn with_q4_block(mut self) -> Result<Self, &'static str> {
        if self.tile != 1 || self.projections[0].dtype != Q4_K || self.projections[1].dtype != Q4_K
        {
            return Err("grouped Q4 block public policy");
        }
        self.gate_policy = GatePolicy::Q4Block;
        Ok(self)
    }
    /// Public launch capacity; MAX8 owning spans remain unchanged for tile1.
    pub fn member_capacity(self) -> usize {
        if self.tile == 1 { 1 } else { MAX_BATCH }
    }
    /// Fixed public four-member integer path; owning MAX8/MAP limits stay.
    #[allow(dead_code)] // Parent selection is independently guarded.
    pub(crate) fn with_four_members(mut self) -> Result<Self, &'static str> {
        if !self.frozen_host_sources
            || !self.inactive_partition_exit
            || self.capacity != 512
            || self.tile != 8
            || self.hidden != 2560
            || self.ff != 640
            || self.secondary_partition
            || self.gate_policy != GatePolicy::Generic
            || self.reduction_policy != ReductionPolicy::AllMembers
            || self.private_tables
            || self.workgroup_tables
            || self.shared_down_scales
            || self.shared_gate_up_factors
            || self.four_row_down
            || self.two_row_gate_up
            || self.three_member_arithmetic
            || self.four_members
        {
            return Err("grouped frozen four-member integer policy");
        }
        self.four_members = true;
        Ok(self)
    }
    #[allow(dead_code)]
    pub(crate) fn four_members(self) -> bool {
        self.four_members
    }
    /// Runtime frozen count bound only, never full owning/storage capacity.
    pub(crate) fn member_limit(self) -> usize {
        if self.frozen_host_sources {
            if self.four_members {
                crate::strata_port::verify_capacity()
            } else {
                3
            }
        } else {
            self.member_capacity()
        }
    }
    pub fn validate_count(self, count: usize) -> Result<(), &'static str> {
        if !(1..=self.member_limit()).contains(&count) {
            return Err("grouped public member policy count");
        }
        Ok(())
    }
    pub fn capacity(self) -> usize {
        self.capacity
    }
    /// Original tightly packed full shard views only; no hidden offsets or
    /// runtime-slot multiplication can exceed these validated original spans.
    pub fn validate_shard(
        self,
        first_slot: usize,
        slot_count: usize,
        binding_words: [usize; 3],
    ) -> Result<(), &'static str> {
        let flat = [self.capacity];
        let parts = if self.frozen_host_sources {
            &crate::daslab_profile::SOURCE_PARTS[..]
        } else if self.secondary_partition {
            &[32, 16][..]
        } else {
            crate::bank_slots::segmented_capacities(self.capacity).unwrap_or(&flat)
        };
        let mut start = 0;
        let valid = parts.iter().any(|&count| {
            let matches = first_slot == start && slot_count == count;
            start += count;
            matches
        });
        if !valid {
            return Err("grouped original shard geometry");
        }
        for (projection, words) in self.projections.into_iter().zip(binding_words) {
            if words != projection.segment_words(slot_count)? {
                return Err("grouped original shard binding");
            }
        }
        Ok(())
    }
    /// Runtime group count changes dispatch only, not public CubeDim/pipeline.
    /// The wrapper must additionally check actual adapter axes/binding limits.
    pub fn grid(self, groups: usize, down: bool) -> Result<CubeCount, &'static str> {
        if groups == 0 || groups > MAX_GROUPS || groups > self.capacity {
            return Err("grouped dispatch group count");
        }
        Ok(CubeCount::Static(
            if down {
                self.hidden / if self.four_row_down { 4 } else { 1 }
            } else {
                self.ff / if self.two_row_gate_up { 2 } else { 1 }
            } as u32,
            groups as u32,
            (self.member_capacity() / self.tile) as u32,
        ))
    }
    /// Explicit public member policy only; runtime count remains a guarded
    /// operand. Inactive reducer cubes never write storage output.
    pub fn reduce_grid(self) -> CubeCount {
        CubeCount::Static(
            self.hidden.div_ceil(LANES) as u32,
            self.member_capacity() as u32,
            1,
        )
    }
    /// Checked frozen chunk count changes dispatch only, never specialization
    /// or full owning spans. Omitted reducer cubes cannot write storage output.
    pub fn reduce_grid_for_count(self, count: usize) -> Result<CubeCount, &'static str> {
        self.validate_count(count)?;
        Ok(CubeCount::Static(
            self.hidden.div_ceil(LANES) as u32,
            count as u32,
            1,
        ))
    }
    /// All ten routes for every admitted public member, with unchanged full
    /// MAX80 status owners. Inactive finite cubes only touch workgroup scratch.
    pub fn finite_grid(self) -> CubeCount {
        CubeCount::Static(
            self.hidden.div_ceil(LANES).max(self.ff.div_ceil(LANES)) as u32,
            (self.member_capacity() * RANKS) as u32,
            4,
        )
    }
    /// Cover every active rank of the checked frozen chunk count. Omitted
    /// cubes only touched workgroup scratch; the separate full inactive/status/
    /// padding fold and unchanged MAX80 status owners remain mandatory.
    pub fn finite_grid_for_count(self, count: usize) -> Result<CubeCount, &'static str> {
        self.validate_count(count)?;
        Ok(CubeCount::Static(
            self.hidden.div_ceil(LANES).max(self.ff.div_ceil(LANES)) as u32,
            (count * RANKS) as u32,
            4,
        ))
    }
    pub fn status_words(self) -> usize {
        MAX_ROUTES * 2
            + MAX_BATCH * self.hidden.div_ceil(LANES)
            + MAX_ROUTES * self.ff.div_ceil(LANES)
            + MAX_ROUTES * self.hidden.div_ceil(LANES)
            + MAX_BATCH * self.hidden.div_ceil(LANES)
    }
    /// Full owner pages only; existing input/output may be supplied separately.
    pub fn scratch_bytes(self, include_input_output: bool) -> Result<usize, &'static str> {
        let mut words = vec![
            MAX_ROUTES * self.ff,
            MAX_ROUTES * self.hidden,
            MAP_WORDS,
            MAX_ROUTES,
            self.status_words(),
        ];
        if include_input_output {
            words.extend([MAX_BATCH * self.hidden; 2]);
        }
        words.into_iter().try_fold(0usize, |sum, words| {
            let page =
                crate::private_arena::BufferLayout::new(words).map_err(|_| "grouped owner page")?;
            sum.checked_add(page.owned_bytes)
                .ok_or("grouped owner sum overflow")
        })
    }
}

#[cfg(test)]
mod daslab_geometry_tests {
    use super::*;
    #[test]
    fn all_actual_pinned_daslab_routed_triples_admit_original72_plus72_views() {
        for layer in 0..48 {
            let types = crate::daslab_profile::routed_types(layer).unwrap();
            let gate = Projection::new(types[0], 2560, 640, 512).unwrap();
            let up = Projection::new(types[1], 2560, 640, 512).unwrap();
            let down = Projection::new(types[2], 640, 2560, 512).unwrap();
            let geometry = Geometry::new([gate, up, down], 1, 144).unwrap();
            assert_eq!(geometry.gate_policy, GatePolicy::Generic);
        }
    }
}

/// Request-private fixed map. The borrowed protected union must be produced
/// from the caller's pinned actual pending bank plan and immutable token wave.
/// No Clone/Copy/Debug/Hash/serialization; full fixed capacities wipe on Drop.
pub struct GroupMap<'a> {
    words: Box<[u32; MAP_WORDS]>,
    probabilities: Box<[f32; MAX_ROUTES]>,
    protected: &'a [(u32, u32)],
    frozen_host_sources: bool,
    four_members: bool,
}
fn source_map_capacity(frozen: bool, four_members: bool, capacity: usize, count: usize) -> bool {
    if frozen {
        capacity == 512
            && (1..=if four_members {
                crate::strata_port::verify_capacity()
            } else {
                3
            })
                .contains(&count)
    } else {
        !four_members && crate::bank_slots::routed_capacity(capacity)
    }
}
impl<'a> GroupMap<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        count: usize,
        wave_first: usize,
        wave_count: usize,
        experts: &[u32],
        probabilities: &[f32],
        protected: &'a [(u32, u32)],
        capacity: usize,
    ) -> Result<Self, &'static str> {
        Self::new_with_policy(
            count,
            wave_first,
            wave_count,
            experts,
            probabilities,
            protected,
            capacity,
            false,
            false,
        )
    }
    /// Source-aware map only after the frozen original source proof has bound
    /// each actual expert to its exact immutable virtual address.
    pub(crate) fn new_frozen_sources(
        count: usize,
        experts: &[u32],
        probabilities: &[f32],
        protected: &'a [(u32, u32)],
        geometry: Geometry,
    ) -> Result<Self, &'static str> {
        if !geometry.frozen_host_sources() {
            return Err("grouped frozen source map policy");
        }
        geometry.validate_count(count)?;
        Self::new_with_policy(
            count,
            0,
            count,
            experts,
            probabilities,
            protected,
            512,
            true,
            geometry.four_members(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn new_with_policy(
        count: usize,
        wave_first: usize,
        wave_count: usize,
        experts: &[u32],
        probabilities: &[f32],
        protected: &'a [(u32, u32)],
        capacity: usize,
        frozen_host_sources: bool,
        four_members: bool,
    ) -> Result<Self, &'static str> {
        let mut map = Self {
            words: Box::new([0; MAP_WORDS]),
            probabilities: Box::new([0.; MAX_ROUTES]),
            protected,
            frozen_host_sources,
            four_members,
        };
        if !(1..=MAX_BATCH).contains(&count)
            || wave_count == 0
            || wave_first
                .checked_add(wave_count)
                .is_none_or(|end| end > count)
            || experts.len() != count * RANKS
            || probabilities.len() != count * RANKS
            || !source_map_capacity(frozen_host_sources, four_members, capacity, count)
            || protected.is_empty()
            || protected.len() > capacity.min(MAX_GROUPS)
        {
            return Err("grouped private map bounds");
        }
        let mut scratch = zeroize::Zeroizing::new(RouteScratch::empty());
        for token in 0..count {
            let row = &experts[token * RANKS..(token + 1) * RANKS];
            let probs = &probabilities[token * RANKS..(token + 1) * RANKS];
            scratch.probability_sum.zeroize();
            for rank in 0..RANKS {
                if row[rank] >= 512
                    || row[..rank].contains(&row[rank])
                    || !probs[rank].is_finite()
                    || probs[rank] < 0.
                {
                    return Err("grouped original selection");
                }
                scratch.probability_sum += f64::from(probs[rank]);
                map.probabilities[token * RANKS + rank] = probs[rank];
            }
            if (scratch.probability_sum - 1.).abs() > 1e-5 {
                return Err("grouped original probabilities");
            }
        }
        map.words[0] = count as u32;
        map.words[1] = wave_first as u32;
        map.words[2] = wave_count as u32;
        map.words[3] = protected.len() as u32;
        for (group, &(expert, slot)) in protected.iter().enumerate() {
            if expert >= 512
                || slot as usize >= capacity
                || protected[..group]
                    .iter()
                    .any(|&(e, s)| e == expert || s == slot)
            {
                return Err("grouped protected expert slot");
            }
            map.words[SLOTS + group] = slot;
        }
        for token in wave_first..wave_first + wave_count {
            for rank in 0..RANKS {
                let route = token * RANKS + rank;
                let group = protected
                    .iter()
                    .position(|&(expert, _)| expert == experts[route])
                    .ok_or("grouped missing protected expert")?;
                let length = map.words[LENGTHS + group] as usize;
                if length >= MAX_BATCH {
                    return Err("grouped too many expert members");
                }
                map.words[MEMBERS + group * MAX_BATCH + length] = route as u32;
                map.words[LENGTHS + group] += 1;
            }
        }
        map.validate(experts, probabilities, capacity)?;
        Ok(map)
    }
    pub fn words(&self) -> &[u32; MAP_WORDS] {
        &self.words
    }
    pub fn probabilities(&self) -> &[f32; MAX_ROUTES] {
        &self.probabilities
    }
    pub fn count(&self) -> usize {
        self.words[0] as usize
    }
    pub fn groups(&self) -> usize {
        self.words[3] as usize
    }
    /// Check explicit public policy before any map upload or grouped launch.
    /// Canonical route/probability validation remains separately mandatory.
    pub fn validate_geometry(&self, geometry: Geometry) -> Result<(), &'static str> {
        geometry.validate_count(self.count())?;
        if geometry.frozen_host_sources != self.frozen_host_sources
            || geometry.four_members != self.four_members
        {
            return Err("grouped map source policy mismatch");
        }
        if geometry.tile() == 1
            && (self.words[1] != 0
                || self.words[2] != 1
                || self.groups() != RANKS
                || self.words[LENGTHS..LENGTHS + self.groups()]
                    .iter()
                    .any(|&length| length != 1))
        {
            return Err("grouped single-member wave policy");
        }
        Ok(())
    }
    pub fn validate(
        &self,
        experts: &[u32],
        probabilities: &[f32],
        capacity: usize,
    ) -> Result<(), &'static str> {
        let count = self.words[0] as usize;
        let first = self.words[1] as usize;
        let wave = self.words[2] as usize;
        let groups = self.words[3] as usize;
        if !(1..=MAX_BATCH).contains(&count)
            || wave == 0
            || first.checked_add(wave).is_none_or(|end| end > count)
            || !source_map_capacity(self.frozen_host_sources, self.four_members, capacity, count)
            || groups == 0
            || groups != self.protected.len()
            || groups > MAX_GROUPS.min(capacity)
            || experts.len() != count * RANKS
            || probabilities.len() != count * RANKS
        {
            return Err("grouped modified map header");
        }
        // Repeat original selection checks for acceptance, including routes
        // outside this wave. Neither validation path mutates or normalizes it.
        let mut scratch = zeroize::Zeroizing::new(RouteScratch::empty());
        for token in 0..count {
            let row = &experts[token * RANKS..(token + 1) * RANKS];
            let probs = &probabilities[token * RANKS..(token + 1) * RANKS];
            scratch.probability_sum.zeroize();
            for rank in 0..RANKS {
                if row[rank] >= 512
                    || row[..rank].contains(&row[rank])
                    || !probs[rank].is_finite()
                    || probs[rank] < 0.
                {
                    return Err("grouped modified original selection");
                }
                scratch.probability_sum += f64::from(probs[rank]);
            }
            if (scratch.probability_sum - 1.).abs() > 1e-5 {
                return Err("grouped modified original probabilities");
            }
        }
        // RouteScratch's guard wipes on success, early error and panic.
        for group in 0..MAX_GROUPS {
            let length = self.words[LENGTHS + group] as usize;
            if group >= groups {
                if self.words[SLOTS + group] != 0
                    || length != 0
                    || self.words[MEMBERS + group * MAX_BATCH..MEMBERS + (group + 1) * MAX_BATCH]
                        .iter()
                        .any(|&x| x != 0)
                {
                    return Err("grouped inactive map data");
                }
                continue;
            }
            let (expert, slot) = self.protected[group];
            if length == 0
                || length > wave
                || self.words[SLOTS + group] != slot
                || slot as usize >= capacity
            {
                return Err("grouped modified slot member geometry");
            }
            for member in 0..MAX_BATCH {
                let route = self.words[MEMBERS + group * MAX_BATCH + member] as usize;
                if member >= length {
                    if route != 0 {
                        return Err("grouped inactive member data");
                    }
                    continue;
                }
                if route >= count * RANKS
                    || route / RANKS < first
                    || route / RANKS >= first + wave
                    || experts[route] != expert
                    || scratch.covered[route] != 0
                {
                    return Err("grouped missing duplicate or crossed route");
                }
                for previous in 0..member {
                    if self.words[MEMBERS + group * MAX_BATCH + previous] as usize / RANKS
                        == route / RANKS
                    {
                        return Err("grouped repeated token in expert");
                    }
                }
                scratch.covered[route] = 1;
            }
        }
        for (route, &covered) in scratch.covered.iter().enumerate() {
            let expected = (first * RANKS..(first + wave) * RANKS).contains(&route) as u8;
            if covered != expected {
                return Err("grouped incomplete route coverage");
            }
            let probability = if route < probabilities.len() {
                probabilities[route]
            } else {
                0.
            };
            if self.probabilities[route].to_bits() != probability.to_bits() {
                return Err("grouped changed probability");
            }
        }
        Ok(())
    }
}
impl GroupMap<'_> {
    pub fn wipe(&mut self) {
        self.words.zeroize();
        self.probabilities.zeroize();
    }
}
impl Drop for GroupMap<'_> {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Whole-buffer runtime offsets/strides. Values wipe on Drop, never specialize.
pub struct Spans {
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    prob_first: usize,
    output_first: usize,
    output_stride: usize,
    status_first: usize,
}
impl Spans {
    pub fn new(mut values: [usize; 11]) -> Self {
        let spans = Self {
            map_first: values[0],
            input_first: values[1],
            input_stride: values[2],
            act_first: values[3],
            act_stride: values[4],
            partial_first: values[5],
            partial_stride: values[6],
            prob_first: values[7],
            output_first: values[8],
            output_stride: values[9],
            status_first: values[10],
        };
        values.zeroize();
        spans
    }
    pub fn map_first(&self) -> usize {
        self.map_first
    }
    pub fn input_first(&self) -> usize {
        self.input_first
    }
    pub fn input_stride(&self) -> usize {
        self.input_stride
    }
    pub fn act_first(&self) -> usize {
        self.act_first
    }
    pub fn act_stride(&self) -> usize {
        self.act_stride
    }
    pub fn partial_first(&self) -> usize {
        self.partial_first
    }
    pub fn partial_stride(&self) -> usize {
        self.partial_stride
    }
    pub fn prob_first(&self) -> usize {
        self.prob_first
    }
    pub fn output_first(&self) -> usize {
        self.output_first
    }
    pub fn output_stride(&self) -> usize {
        self.output_stride
    }
    pub fn status_first(&self) -> usize {
        self.status_first
    }
    /// Full MAX8/MAX80 spans, including inactive capacity, before ArrayArgs.
    pub fn validate(&self, geometry: Geometry, words: [usize; 7]) -> Result<(), &'static str> {
        let [map, input, act, partial, prob, output, status] = words;
        let end = |first: usize, stride: usize, count: usize, width: usize, bound: usize| {
            if stride < width {
                return Err("grouped overlapping rows");
            }
            let end = (count - 1)
                .checked_mul(stride)
                .and_then(|last| first.checked_add(last))
                .and_then(|last| last.checked_add(width))
                .ok_or("grouped runtime span overflow")?;
            if end > bound || end > u32::MAX as usize || bound > u32::MAX as usize {
                return Err("grouped runtime span extent");
            }
            Ok(())
        };
        end(self.map_first, MAP_WORDS, 1, MAP_WORDS, map)?;
        end(
            self.input_first,
            self.input_stride,
            MAX_BATCH,
            geometry.hidden,
            input,
        )?;
        end(
            self.act_first,
            self.act_stride,
            MAX_ROUTES,
            geometry.ff,
            act,
        )?;
        end(
            self.partial_first,
            self.partial_stride,
            MAX_ROUTES,
            geometry.hidden,
            partial,
        )?;
        end(self.prob_first, MAX_ROUTES, 1, MAX_ROUTES, prob)?;
        end(
            self.output_first,
            self.output_stride,
            MAX_BATCH,
            geometry.hidden,
            output,
        )?;
        end(
            self.status_first,
            geometry.status_words(),
            1,
            geometry.status_words(),
            status,
        )
    }
}
impl Spans {
    pub fn wipe(&mut self) {
        self.map_first.zeroize();
        self.input_first.zeroize();
        self.input_stride.zeroize();
        self.act_first.zeroize();
        self.act_stride.zeroize();
        self.partial_first.zeroize();
        self.partial_stride.zeroize();
        self.prob_first.zeroize();
        self.output_first.zeroize();
        self.output_stride.zeroize();
        self.status_first.zeroize();
    }
}

impl Drop for Spans {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Content-free operator acceptance helper, not state/bank commit. Parent
/// supplies its original active count and full owning readback, after checked
/// completion/cancellation/context/epoch validation and before owner erasure.
pub fn validate_status(
    geometry: Geometry,
    count: usize,
    first: usize,
    values: &[u32],
) -> Result<(), &'static str> {
    geometry.validate_count(count)?;
    if first
        .checked_add(geometry.status_words())
        .is_none_or(|end| end > values.len())
    {
        return Err("grouped status owning span");
    }
    let d = geometry.hidden.div_ceil(LANES);
    let f = geometry.ff.div_ceil(LANES);
    let regions = [
        (MAX_ROUTES, count * RANKS),
        (MAX_ROUTES, count * RANKS),
        (MAX_BATCH * d, count * d),
        (MAX_ROUTES * f, count * RANKS * f),
        (MAX_ROUTES * d, count * RANKS * d),
        (MAX_BATCH * d, count * d),
    ];
    if values[..first].iter().any(|&v| v != 0)
        || values[first + geometry.status_words()..]
            .iter()
            .any(|&v| v != 0)
    {
        return Err("grouped status private guard");
    }
    let mut offset = first;
    for (length, active) in regions {
        for (index, &value) in values[offset..offset + length].iter().enumerate() {
            if value != u32::from(index < active) {
                return Err("grouped unfinished nonfinite or inactive status");
            }
        }
        offset += length;
    }
    Ok(())
}

// Explicit nested guards avoid eager CubeCL lowering of underflowing
// subtraction; Range::contains is not a CubeCL scalar operation.
#[cube]
#[allow(clippy::collapsible_if, clippy::manual_range_contains)]
pub(super) fn group_active(
    map: &Array<u32>,
    first: usize,
    group: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] tile: usize,
) -> bool {
    let mut active = false;
    if first <= map.len() {
        if MAP_WORDS <= map.len() - first {
            let count = map[first] as usize;
            let wave_first = map[first + 1] as usize;
            let wave_count = map[first + 2] as usize;
            let groups = map[first + 3] as usize;
            if count >= 1
                && count <= MAX_BATCH
                && wave_count >= 1
                && wave_count <= count
                && wave_first <= count - wave_count
                && groups >= 1
                && groups <= MAX_GROUPS
                && group < groups
            {
                let length = map[first + LENGTHS + group] as usize;
                let slot = map[first + SLOTS + group] as usize;
                if length >= 1 && length <= wave_count && slot >= first_slot {
                    if slot - first_slot < slot_count {
                        active = true;
                        if comptime!(tile == 1) {
                            if count != 1 || wave_first != 0 || wave_count != 1 || length != 1 {
                                active = false;
                            }
                        }
                        for member in 0..length {
                            let route = map[first + MEMBERS + group * MAX_BATCH + member] as usize;
                            if route >= count * RANKS
                                || route / RANKS < wave_first
                                || route / RANKS >= wave_first + wave_count
                            {
                                active = false;
                            }
                        }
                    }
                }
            }
        }
    }
    active
}

/// Original frozen verification emits full waves of at most three tokens.
/// Public workgroup coordinates/header only: all active lanes reach every fence.
#[cube]
#[allow(clippy::collapsible_if, clippy::manual_range_contains)]
pub(super) fn frozen_three_wave(map: &Array<u32>, first: usize, first_member: usize) -> bool {
    let mut valid = false;
    if first_member == 0 && first <= map.len() {
        if MAP_WORDS <= map.len() - first {
            let count = map[first] as usize;
            let groups = map[first + 3] as usize;
            valid = count >= 1
                && count <= 3
                && map[first + 1] == 0
                && map[first + 2] as usize == count
                && groups >= 1
                && groups <= 30;
        }
    }
    valid
}

#[cube]
pub(super) fn grouped_silu(value: f32) -> f32 {
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    value * sigmoid
}

// Original Q4_K field reconstruction from moe_block. A block's header and
// scale words are loaded once per lane; only its four quant bytes vary.
#[cube]
fn q4_word_byte(word: u32, byte: usize) -> u32 {
    (word >> ((byte * 8) as u32)) & 255
}
#[cube]
fn q4_block_scale(scales0: u32, scales2: u32, group: usize) -> u32 {
    if group < 4 {
        q4_word_byte(scales0, group) & 63
    } else {
        (q4_word_byte(scales2, group - 4) & 15) | ((q4_word_byte(scales0, group - 4) >> 6) << 4)
    }
}
#[cube]
fn q4_block_minimum(scales1: u32, scales2: u32, group: usize) -> u32 {
    if group < 4 {
        q4_word_byte(scales1, group) & 63
    } else {
        (q4_word_byte(scales2, group - 4) >> 4) | ((q4_word_byte(scales1, group - 4) >> 6) << 4)
    }
}

/// Six storage buffers plus INFO. Public CubeDim64, grid=(FF,groups,8/tile).
/// Full spans/immutable shard packing and group validation must precede launch.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        false,
        false,
        false,
        tile,
    );
}

/// Authenticated frozen-source partition only; inactive groups do no writes.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        false,
        false,
        tile,
    );
}

/// Frozen full-wave arithmetic scratch for exactly three public member cells.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_three_members(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        false,
        false,
        3_usize,
    );
}

/// Exact original frozen arithmetic over previously copied HOST group words.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_staged(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        false,
        true,
        tile,
    );
}

/// Frozen full-wave arithmetic scratch for exactly three public member cells.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_staged_three_members(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        false,
        true,
        3_usize,
    );
}

/// Explicit immutable public IQ table-storage candidate. The exact entrypoint
/// name selects bounded module-private codebook lowering; all math is shared.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_private_grid(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        false,
        false,
        tile,
    );
}

/// Explicit immutable public IQ table-storage candidate. The exact entrypoint
/// name selects bounded workgroup codebook lowering; all math is shared.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_workgroup_grid(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        false,
        false,
        tile,
    );
}

/// Original frozen IQ subgroup factors; exact original code/sign and tree.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_shared_factors(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        true,
        false,
        tile,
    );
}

/// Original frozen IQ subgroup factors; exact original code/sign and tree.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_private_grid_shared_factors(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        true,
        false,
        tile,
    );
}

/// Original frozen IQ subgroup factors; exact original code/sign and tree.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_workgroup_grid_shared_factors(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
        true,
        true,
        false,
        tile,
    );
}

/// Explicit immutable live-member policy; original INFO/owners/grid unchanged.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn grouped_gate_up_live(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        true,
        false,
        false,
        false,
        tile,
    );
}

#[cube]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn grouped_gate_up_inner(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
    #[comptime] live_member_reduction: bool,
    #[comptime] inactive_partition_exit: bool,
    #[comptime] shared_gate_up_factors: bool,
    #[comptime] staged_host_weights: bool,
    #[comptime] arithmetic_members: usize,
) {
    let feature = CUBE_POS_X as usize;
    let group = CUBE_POS_Y as usize;
    let first_member = CUBE_POS_Z as usize * tile;
    let lane = UNIT_POS_X as usize;
    let mut live = 0_usize;
    let mut active =
        group_active(map, map_first, group, first_slot, slot_count, tile) && feature < ff;
    if comptime!(arithmetic_members == 3) {
        if comptime!(hidden != 2560 || ff != 640 || tile != 8 || q4_block) {
            terminate!();
        }
        active = active && frozen_three_wave(map, map_first, first_member);
    }
    if comptime!(staged_host_weights) {
        if comptime!(hidden != 2560 || ff != 640 || tile != 8 || q4_block) {
            terminate!();
        }
        active = active
            && frozen_host_staging::host_group_active(
                map, map_first, group, first_slot, slot_count,
            )
            && frozen_host_staging::stage_extent_valid(
                gate.len(),
                comptime!(ff * gate_row_bytes / 4),
            )
            && frozen_host_staging::stage_extent_valid(up.len(), comptime!(ff * up_row_bytes / 4));
    }
    if active {
        if first_member >= map[map_first + LENGTHS + group] as usize {
            active = false;
        }
        if comptime!(live_member_reduction) {
            if active {
                let remaining = map[map_first + LENGTHS + group] as usize - first_member;
                live = tile;
                if remaining < tile {
                    live = remaining;
                }
            }
        }
    }
    // The decision uses only workgroup coordinates and immutable MAP fields.
    // Inactive shards must not erase rows already produced by another shard.
    if comptime!(inactive_partition_exit) {
        if !active {
            terminate!();
        }
    }
    let mut gates = Array::<f32>::new(arithmetic_members);
    let mut ups = Array::<f32>::new(arithmetic_members);
    #[unroll]
    for member in 0..arithmetic_members {
        gates[member] = 0.;
        ups[member] = 0.;
    }
    if active {
        let slot = frozen_host_staging::weight_slot(
            map,
            map_first,
            group,
            first_slot,
            staged_host_weights,
        );
        let length = map[map_first + LENGTHS + group] as usize;
        let row = slot * ff + feature;
        let gate_offset = row * gate_row_bytes;
        let up_offset = row * up_row_bytes;
        if comptime!(q4_block) {
            // Geometry::with_q4_block proves tile1 and two whole-block Q4_K
            // rows. Slot/route/bases remain the validated runtime packet.
            let route = map[map_first + MEMBERS + group * MAX_BATCH] as usize;
            let input_base = input_first + route / RANKS * input_stride;
            let gate_row_word = row * gate_row_bytes / 4;
            let up_row_word = row * up_row_bytes / 4;
            let quarter_lane = lane % 32;
            let odd_group = lane / 32;
            for block in 0..comptime!(hidden / 256) {
                let gate_base = gate_row_word + block * 36;
                let up_base = up_row_word + block * 36;
                let gate_header = gate[gate_base];
                let up_header = up[up_base];
                let gate_d = binary16(gate_header & 65535);
                let gate_dmin = binary16(gate_header >> 16);
                let up_d = binary16(up_header & 65535);
                let up_dmin = binary16(up_header >> 16);
                let gate_scales0 = gate[gate_base + 1];
                let gate_scales1 = gate[gate_base + 2];
                let gate_scales2 = gate[gate_base + 3];
                let up_scales0 = up[up_base + 1];
                let up_scales1 = up[up_base + 2];
                let up_scales2 = up[up_base + 3];
                #[unroll]
                for pair in 0..4_usize {
                    let sub_group = pair * 2 + odd_group;
                    let quant_offset = 4 + pair * 8 + quarter_lane / 4;
                    let gate_byte = q4_word_byte(gate[gate_base + quant_offset], quarter_lane & 3);
                    let up_byte = q4_word_byte(up[up_base + quant_offset], quarter_lane & 3);
                    let gate_quant = (gate_byte >> ((odd_group * 4) as u32)) & 15;
                    let up_quant = (up_byte >> ((odd_group * 4) as u32)) & 15;
                    let gate_factor =
                        gate_d * q4_block_scale(gate_scales0, gate_scales2, sub_group) as f32;
                    let gate_bias =
                        gate_dmin * q4_block_minimum(gate_scales1, gate_scales2, sub_group) as f32;
                    let up_factor = up_d * q4_block_scale(up_scales0, up_scales2, sub_group) as f32;
                    let up_bias =
                        up_dmin * q4_block_minimum(up_scales1, up_scales2, sub_group) as f32;
                    let g = gate_factor * gate_quant as f32 - gate_bias;
                    let u = up_factor * up_quant as f32 - up_bias;
                    let column = block * 256 + pair * LANES + lane;
                    let x = input[input_base + column];
                    gates[0] += g * x;
                    ups[0] += u * x;
                }
            }
        } else if comptime!(
            shared_gate_up_factors
                && (gate_dtype == IQ2_S || gate_dtype == IQ3_XXS || gate_dtype == IQ3_S)
        ) {
            // Frozen geometry proves equal GU dtypes and2560 input columns.
            // Every active lane reaches the barrier; each factor is written
            // exactly once by its lane-strided index before any factor read.
            let factor_width = comptime!(if gate_dtype == IQ2_S { 16 } else { 32 });
            let factors_per_row = comptime!(hidden / factor_width);
            let mut factors = SharedMemory::<f32>::new(comptime!(factors_per_row * 2));
            for factor_tile in 0..comptime!((factors_per_row * 2).div_ceil(LANES)) {
                let index = factor_tile * LANES + lane;
                if index < comptime!(factors_per_row * 2) {
                    if index < comptime!(factors_per_row) {
                        factors[index] =
                            iq_gate_up_factor(gate, gate_offset, index * factor_width, gate_dtype);
                    } else {
                        factors[index] = iq_gate_up_factor(
                            up,
                            up_offset,
                            (index - factors_per_row) * factor_width,
                            up_dtype,
                        );
                    }
                }
            }
            sync_cube();
            for column_tile in 0..comptime!(hidden.div_ceil(LANES)) {
                let column = column_tile * LANES + lane;
                if column < hidden {
                    let factor_index = column / factor_width;
                    let g = iq_gate_up_value_with_factor(
                        gate,
                        gate_offset,
                        column,
                        factors[factor_index],
                        gate_dtype,
                    );
                    let u = iq_gate_up_value_with_factor(
                        up,
                        up_offset,
                        column,
                        factors[factors_per_row + factor_index],
                        up_dtype,
                    );
                    #[unroll]
                    for member in 0..arithmetic_members {
                        if first_member + member < length {
                            let route = map
                                [map_first + MEMBERS + group * MAX_BATCH + first_member + member]
                                as usize;
                            let x = input[input_first + route / RANKS * input_stride + column];
                            gates[member] += g * x;
                            ups[member] += u * x;
                        }
                    }
                }
            }
        } else {
            for column_tile in 0..comptime!(hidden.div_ceil(LANES)) {
                let column = column_tile * LANES + lane;
                if column < hidden {
                    // Decode each original weight only ONCE, reused across RHS.
                    let g = packed_value(gate, gate_offset, column, gate_dtype);
                    let u = packed_value(up, up_offset, column, up_dtype);
                    #[unroll]
                    for member in 0..arithmetic_members {
                        if first_member + member < length {
                            let route = map
                                [map_first + MEMBERS + group * MAX_BATCH + first_member + member]
                                as usize;
                            let x = input[input_first + route / RANKS * input_stride + column];
                            gates[member] += g * x;
                            ups[member] += u * x;
                        }
                    }
                }
            }
        }
    }
    let mut gate_sums = SharedMemory::<f32>::new(LANES * arithmetic_members);
    let mut up_sums = SharedMemory::<f32>::new(LANES * arithmetic_members);
    #[unroll]
    for member in 0..arithmetic_members {
        if comptime!(live_member_reduction) {
            if member < live {
                gate_sums[member * LANES + lane] = gates[member];
                up_sums[member * LANES + lane] = ups[member];
            }
        } else {
            gate_sums[member * LANES + lane] = gates[member];
            up_sums[member * LANES + lane] = ups[member];
        }
    }
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            #[unroll]
            for member in 0..arithmetic_members {
                if comptime!(live_member_reduction) {
                    if member < live {
                        gate_sums[member * LANES + lane] +=
                            gate_sums[member * LANES + lane + stride];
                        up_sums[member * LANES + lane] += up_sums[member * LANES + lane + stride];
                    }
                } else {
                    gate_sums[member * LANES + lane] += gate_sums[member * LANES + lane + stride];
                    up_sums[member * LANES + lane] += up_sums[member * LANES + lane + stride];
                }
            }
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && active {
        let length = map[map_first + LENGTHS + group] as usize;
        #[unroll]
        for member in 0..arithmetic_members {
            if first_member + member < length {
                let route =
                    map[map_first + MEMBERS + group * MAX_BATCH + first_member + member] as usize;
                activations[act_first + route * act_stride + feature] =
                    grouped_silu(gate_sums[member * LANES]) * up_sums[member * LANES];
                if feature == 0 {
                    status[status_first + route] = 1;
                }
            }
        }
    }
}

/// Two complete original GU rows per64-thread cube. The named entrypoint must
/// consume Geometry::two_row_gate_up's paired grid; original entrypoints remain
/// unchanged. Every source-exit predicate is workgroup-uniform.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_two_rows(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_two_rows_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        false,
    );
}

/// Exact original paired GU arithmetic over copied HOST group words.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_gate_up_frozen_staged_two_rows(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
) {
    grouped_gate_up_two_rows_inner(
        gate,
        up,
        input,
        map,
        activations,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        gate_dtype,
        up_dtype,
        gate_row_bytes,
        up_row_bytes,
        tile,
        q4_block,
        true,
    );
}

#[cube]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn grouped_gate_up_two_rows_inner(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    map: &Array<u32>,
    activations: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
    #[comptime] gate_row_bytes: usize,
    #[comptime] up_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] q4_block: bool,
    #[comptime] staged_host_weights: bool,
) {
    // Public generic-only admission has no Q4 block specialization.
    if comptime!(q4_block) {
        terminate!();
    }
    let first_feature = CUBE_POS_X as usize * 2;
    let group = CUBE_POS_Y as usize;
    let first_member = CUBE_POS_Z as usize * tile;
    let physical_lane = UNIT_POS_X as usize;
    let row = physical_lane / 32;
    let row_lane = physical_lane % 32;
    let feature = first_feature + row;
    // Frozen FF640 admits whole pairs. Never return on a per-row predicate:
    // both physical rows must reach every shared reduction fence together.
    let mut active =
        group_active(map, map_first, group, first_slot, slot_count, tile) && first_feature + 1 < ff;
    if comptime!(staged_host_weights) {
        if comptime!(hidden != 2560 || ff != 640 || tile != 8 || q4_block) {
            terminate!();
        }
        active = active
            && frozen_host_staging::host_group_active(
                map, map_first, group, first_slot, slot_count,
            )
            && frozen_host_staging::stage_extent_valid(
                gate.len(),
                comptime!(ff * gate_row_bytes / 4),
            )
            && frozen_host_staging::stage_extent_valid(up.len(), comptime!(ff * up_row_bytes / 4));
    }
    if active {
        if first_member >= map[map_first + LENGTHS + group] as usize {
            active = false;
        }
    }
    if !active {
        terminate!();
    }
    let slot =
        frozen_host_staging::weight_slot(map, map_first, group, first_slot, staged_host_weights);
    let length = map[map_first + LENGTHS + group] as usize;
    let source_row = slot * ff + feature;
    let gate_offset = source_row * gate_row_bytes;
    let up_offset = source_row * up_row_bytes;
    let mut gates = Array::<f32>::new(comptime!(2 * tile));
    let mut ups = Array::<f32>::new(comptime!(2 * tile));
    #[unroll]
    for virtual_lane in 0_usize..2 {
        #[unroll]
        for member in 0..tile {
            gates[virtual_lane * tile + member] = 0.;
            ups[virtual_lane * tile + member] = 0.;
        }
    }
    for column_tile in 0..comptime!(hidden.div_ceil(LANES)) {
        #[unroll]
        for virtual_lane in 0_usize..2 {
            let column = column_tile * LANES + row_lane + virtual_lane * 32;
            if column < hidden {
                let g = packed_value(gate, gate_offset, column, gate_dtype);
                let u = packed_value(up, up_offset, column, up_dtype);
                #[unroll]
                for member in 0..tile {
                    if first_member + member < length {
                        let route = map
                            [map_first + MEMBERS + group * MAX_BATCH + first_member + member]
                            as usize;
                        let x = input[input_first + route / RANKS * input_stride + column];
                        gates[virtual_lane * tile + member] += g * x;
                        ups[virtual_lane * tile + member] += u * x;
                    }
                }
            }
        }
    }
    let mut gate_sums = SharedMemory::<f32>::new(comptime!(2 * 32 * tile));
    let mut up_sums = SharedMemory::<f32>::new(comptime!(2 * 32 * tile));
    #[unroll]
    for member in 0..tile {
        let index = (row * tile + member) * 32 + row_lane;
        // Exactly the original stride32 operands and order. The remaining
        // original strides16,8,4,2,1 use the same shared F32 additions.
        gate_sums[index] = gates[member] + gates[tile + member];
        up_sums[index] = ups[member] + ups[tile + member];
    }
    sync_cube();
    let mut stride = 16_usize;
    while stride > 0 {
        if row_lane < stride {
            #[unroll]
            for member in 0..tile {
                let index = (row * tile + member) * 32 + row_lane;
                gate_sums[index] += gate_sums[index + stride];
                up_sums[index] += up_sums[index + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    if row_lane == 0 {
        #[unroll]
        for member in 0..tile {
            if first_member + member < length {
                let route =
                    map[map_first + MEMBERS + group * MAX_BATCH + first_member + member] as usize;
                let index = (row * tile + member) * 32;
                activations[act_first + route * act_stride + feature] =
                    grouped_silu(gate_sums[index]) * up_sums[index];
                if feature == 0 {
                    status[status_first + route] = 1;
                }
            }
        }
    }
}

/// Five storage buffers plus INFO. Same original64-lane tree for each route.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_inner(
        down,
        activations,
        map,
        partials,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        false,
        false,
        false,
        tile,
    );
}

/// Authenticated frozen-source partition only; inactive groups do no writes.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_inner(
        down,
        activations,
        map,
        partials,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        true,
        false,
        false,
        tile,
    );
}

/// Frozen full-wave arithmetic scratch for exactly three public member cells.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_three_members(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_inner(
        down,
        activations,
        map,
        partials,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        true,
        false,
        false,
        3_usize,
    );
}

/// Exact original frozen arithmetic over previously copied HOST group words.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_staged(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_inner(
        down,
        activations,
        map,
        partials,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        true,
        false,
        true,
        tile,
    );
}

/// Frozen full-wave arithmetic scratch for exactly three public member cells.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_staged_three_members(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_inner(
        down,
        activations,
        map,
        partials,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        true,
        false,
        true,
        3_usize,
    );
}

/// Original frozen-down arithmetic with twenty cooperatively decoded IQ4_NL
/// scales per active row. Q2_0 is compile-time identical to the exit control.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_shared_scales(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_inner(
        down,
        activations,
        map,
        partials,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        true,
        true,
        false,
        tile,
    );
}

/// Explicit immutable live-member policy; original INFO/owners/grid unchanged.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn grouped_down_live(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_inner(
        down,
        activations,
        map,
        partials,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        true,
        false,
        false,
        false,
        tile,
    );
}

#[cube]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn grouped_down_inner(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] live_member_reduction: bool,
    #[comptime] inactive_partition_exit: bool,
    #[comptime] shared_down_scales: bool,
    #[comptime] staged_host_weights: bool,
    #[comptime] arithmetic_members: usize,
) {
    let feature = CUBE_POS_X as usize;
    let group = CUBE_POS_Y as usize;
    let first_member = CUBE_POS_Z as usize * tile;
    let lane = UNIT_POS_X as usize;
    let mut live = 0_usize;
    let mut active =
        group_active(map, map_first, group, first_slot, slot_count, tile) && feature < hidden;
    if comptime!(arithmetic_members == 3) {
        if comptime!(hidden != 2560 || ff != 640 || tile != 8) {
            terminate!();
        }
        active = active && frozen_three_wave(map, map_first, first_member);
    }
    if comptime!(staged_host_weights) {
        if comptime!(hidden != 2560 || ff != 640 || tile != 8) {
            terminate!();
        }
        active = active
            && frozen_host_staging::host_group_active(
                map, map_first, group, first_slot, slot_count,
            )
            && frozen_host_staging::stage_extent_valid(
                down.len(),
                comptime!(hidden * down_row_bytes / 4),
            );
    }
    if active {
        if first_member >= map[map_first + LENGTHS + group] as usize {
            active = false;
        }
        if comptime!(live_member_reduction) {
            if active {
                let remaining = map[map_first + LENGTHS + group] as usize - first_member;
                live = tile;
                if remaining < tile {
                    live = remaining;
                }
            }
        }
    }
    // Same workgroup-uniform source guard and original no-write semantics.
    if comptime!(inactive_partition_exit) {
        if !active {
            terminate!();
        }
    }
    let mut partial = Array::<f32>::new(arithmetic_members);
    #[unroll]
    for member in 0..arithmetic_members {
        partial[member] = 0.;
    }
    if active {
        let slot = frozen_host_staging::weight_slot(
            map,
            map_first,
            group,
            first_slot,
            staged_host_weights,
        );
        let length = map[map_first + LENGTHS + group] as usize;
        let row_offset = (slot * hidden + feature) * down_row_bytes;
        if comptime!(shared_down_scales && down_dtype == IQ4_NL) {
            // active is workgroup-uniform. Frozen geometry proves FF=640,
            // so lanes0..19 initialize every scale before the uniform barrier.
            // Each header retains the original integer IEEE half conversion.
            let mut scales = SharedMemory::<f32>::new(comptime!(ff / 32));
            if lane < comptime!(ff / 32) {
                scales[lane] = iq4_nl_scale(down, row_offset, lane);
            }
            sync_cube();
            for column_tile in 0..comptime!(ff.div_ceil(LANES)) {
                let column = column_tile * LANES + lane;
                if column < ff {
                    let weight =
                        iq4_nl_value_with_scale(down, row_offset, column, scales[column / 32]);
                    #[unroll]
                    for member in 0..arithmetic_members {
                        if first_member + member < length {
                            let route = map
                                [map_first + MEMBERS + group * MAX_BATCH + first_member + member]
                                as usize;
                            partial[member] +=
                                weight * activations[act_first + route * act_stride + column];
                        }
                    }
                }
            }
        } else {
            for column_tile in 0..comptime!(ff.div_ceil(LANES)) {
                let column = column_tile * LANES + lane;
                if column < ff {
                    let weight = packed_value(down, row_offset, column, down_dtype);
                    #[unroll]
                    for member in 0..arithmetic_members {
                        if first_member + member < length {
                            let route = map
                                [map_first + MEMBERS + group * MAX_BATCH + first_member + member]
                                as usize;
                            partial[member] +=
                                weight * activations[act_first + route * act_stride + column];
                        }
                    }
                }
            }
        }
    }
    let mut sums = SharedMemory::<f32>::new(LANES * arithmetic_members);
    #[unroll]
    for member in 0..arithmetic_members {
        if comptime!(live_member_reduction) {
            if member < live {
                sums[member * LANES + lane] = partial[member];
            }
        } else {
            sums[member * LANES + lane] = partial[member];
        }
    }
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            #[unroll]
            for member in 0..arithmetic_members {
                if comptime!(live_member_reduction) {
                    if member < live {
                        sums[member * LANES + lane] += sums[member * LANES + lane + stride];
                    }
                } else {
                    sums[member * LANES + lane] += sums[member * LANES + lane + stride];
                }
            }
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && active {
        let length = map[map_first + LENGTHS + group] as usize;
        #[unroll]
        for member in 0..arithmetic_members {
            if first_member + member < length {
                let route =
                    map[map_first + MEMBERS + group * MAX_BATCH + first_member + member] as usize;
                partials[partial_first + route * partial_stride + feature] = sums[member * LANES];
                if feature == 0 {
                    status[status_first + MAX_ROUTES + route] = 1;
                }
            }
        }
    }
}

/// Four original down rows in one64-thread frozen cube. Each physical thread
/// retains four virtual original64-lane accumulators; no dot is regrouped.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn grouped_down_frozen_four_rows(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_four_rows_inner(
        down,
        activations,
        map,
        partials,
        status,
        CUBE_POS_Y as usize,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        false,
        tile,
    );
}

/// Frozen full-wave arithmetic scratch for exactly three public member cells.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_four_rows_three_members(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_four_rows_inner(
        down,
        activations,
        map,
        partials,
        status,
        CUBE_POS_Y as usize,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        false,
        3_usize,
    );
}

/// Exact original frozen arithmetic over previously copied HOST group words.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_staged_four_rows(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_four_rows_inner(
        down,
        activations,
        map,
        partials,
        status,
        CUBE_POS_Y as usize,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        true,
        tile,
    );
}

/// Frozen full-wave arithmetic scratch for exactly three public member cells.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_staged_four_rows_three_members(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_four_rows_inner(
        down,
        activations,
        map,
        partials,
        status,
        CUBE_POS_Y as usize,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        false,
        true,
        3_usize,
    );
}

/// The same four-row tree with exact original IQ4_NL scale staging. Q2_0
/// compiles to the identical four-row packed-value path without scale scratch.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn grouped_down_frozen_four_rows_shared_scales(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    grouped_down_four_rows_inner(
        down,
        activations,
        map,
        partials,
        status,
        CUBE_POS_Y as usize,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        hidden,
        ff,
        down_dtype,
        down_row_bytes,
        tile,
        true,
        false,
        tile,
    );
}

/// Uniform private MAP predicate used by the distinct bounded scheduling path.
/// It never reads source weights or allocates/initializes shared memory.
#[cube]
#[allow(clippy::collapsible_if)]
fn down_stride_four_group_active(
    map: &Array<u32>,
    map_first: usize,
    group: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] tile: usize,
) -> bool {
    let mut active = frozen_three_wave(map, map_first, 0_usize)
        && group_active(map, map_first, group, first_slot, slot_count, tile);
    if active {
        active = map[map_first + 3] as usize <= map[map_first] as usize * RANKS;
    }
    active
}

/// Explicit four-y-workgroup schedule over the original public30-group
/// envelope. Every valid group invokes the same original virtual64 arithmetic.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn grouped_down_frozen_four_rows_stride_four(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
) {
    if comptime!(
        hidden != 2560 || ff != 640 || tile != 8 || (down_dtype != IQ4_NL && down_dtype != Q2_0)
    ) {
        terminate!();
    }
    if CUBE_POS_X as usize >= comptime!(hidden / 4) || CUBE_POS_Y >= 4 || CUBE_POS_Z != 0 {
        terminate!();
    }
    // The scan and exit are identical for every physical lane. No source
    // payload read, reduction reset or barrier occurs before one valid group.
    let mut any_valid = false;
    let mut scan = CUBE_POS_Y as usize;
    while scan < 30 {
        if down_stride_four_group_active(map, map_first, scan, first_slot, slot_count, tile) {
            any_valid = true;
        }
        scan += 4;
    }
    if !any_valid {
        terminate!();
    }
    let mut group = CUBE_POS_Y as usize;
    while group < 30 {
        if down_stride_four_group_active(map, map_first, group, first_slot, slot_count, tile) {
            grouped_down_four_rows_inner(
                down,
                activations,
                map,
                partials,
                status,
                group,
                map_first,
                act_first,
                act_stride,
                partial_first,
                partial_stride,
                status_first,
                first_slot,
                slot_count,
                hidden,
                ff,
                down_dtype,
                down_row_bytes,
                tile,
                false,
                false,
                tile,
            );
            // Every lane finishes all shared reads and output before any lane
            // resets that same full shared owner for another original group.
            sync_cube();
        }
        group += 4;
    }
}

#[cube]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
fn grouped_down_four_rows_inner(
    down: &Array<u32>,
    activations: &Array<f32>,
    map: &Array<u32>,
    partials: &mut Array<f32>,
    status: &mut Array<u32>,
    group: usize,
    map_first: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    status_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] down_dtype: u32,
    #[comptime] down_row_bytes: usize,
    #[comptime] tile: usize,
    #[comptime] shared_down_scales: bool,
    #[comptime] staged_host_weights: bool,
    #[comptime] arithmetic_members: usize,
) {
    let first_feature = CUBE_POS_X as usize * 4;
    let first_member = CUBE_POS_Z as usize * tile;
    let lane = UNIT_POS_X as usize;
    let row = lane / 16;
    let row_lane = lane % 16;
    let feature = first_feature + row;
    // Only public cube coordinates and the shared group map enter the exit.
    // Frozen geometry fixes hidden=2560 and exactly four complete rows/cube.
    let mut active = group_active(map, map_first, group, first_slot, slot_count, tile)
        && first_feature + 3 < hidden;
    if comptime!(arithmetic_members == 3) {
        if comptime!(hidden != 2560 || ff != 640 || tile != 8) {
            terminate!();
        }
        active = active && frozen_three_wave(map, map_first, first_member);
    }
    if comptime!(staged_host_weights) {
        if comptime!(hidden != 2560 || ff != 640 || tile != 8) {
            terminate!();
        }
        active = active
            && frozen_host_staging::host_group_active(
                map, map_first, group, first_slot, slot_count,
            )
            && frozen_host_staging::stage_extent_valid(
                down.len(),
                comptime!(hidden * down_row_bytes / 4),
            );
    }
    if active {
        if first_member >= map[map_first + LENGTHS + group] as usize {
            active = false;
        }
    }
    if !active {
        terminate!();
    }
    let slot =
        frozen_host_staging::weight_slot(map, map_first, group, first_slot, staged_host_weights);
    let length = map[map_first + LENGTHS + group] as usize;
    let row_offset = (slot * hidden + feature) * down_row_bytes;
    let mut partial = Array::<f32>::new(comptime!(4 * arithmetic_members));
    #[unroll]
    for virtual_lane in 0_usize..4 {
        #[unroll]
        for member in 0..arithmetic_members {
            partial[virtual_lane * arithmetic_members + member] = 0.;
        }
    }
    if comptime!(shared_down_scales && down_dtype == IQ4_NL) {
        let mut scales = SharedMemory::<f32>::new(comptime!(4 * (ff / 32)));
        //64 physical lanes initialize all80 scales in two bounded rounds.
        // The source row for each scale is independent of the lane's dot row.
        #[unroll]
        for scale_tile in 0..comptime!((4 * (ff / 32)).div_ceil(LANES)) {
            let index = scale_tile * LANES + lane;
            if index < comptime!(4 * (ff / 32)) {
                let scale_row = index / comptime!(ff / 32);
                let block = index % comptime!(ff / 32);
                let offset = (slot * hidden + first_feature + scale_row) * down_row_bytes;
                scales[index] = iq4_nl_scale(down, offset, block);
            }
        }
        sync_cube();
        for column_tile in 0..comptime!(ff.div_ceil(LANES)) {
            #[unroll]
            for virtual_lane in 0_usize..4 {
                let column = column_tile * LANES + row_lane + virtual_lane * 16;
                if column < ff {
                    let weight = iq4_nl_value_with_scale(
                        down,
                        row_offset,
                        column,
                        scales[row * comptime!(ff / 32) + column / 32],
                    );
                    #[unroll]
                    for member in 0..arithmetic_members {
                        if first_member + member < length {
                            let route = map
                                [map_first + MEMBERS + group * MAX_BATCH + first_member + member]
                                as usize;
                            partial[virtual_lane * arithmetic_members + member] +=
                                weight * activations[act_first + route * act_stride + column];
                        }
                    }
                }
            }
        }
    } else {
        for column_tile in 0..comptime!(ff.div_ceil(LANES)) {
            #[unroll]
            for virtual_lane in 0_usize..4 {
                let column = column_tile * LANES + row_lane + virtual_lane * 16;
                if column < ff {
                    let weight = packed_value(down, row_offset, column, down_dtype);
                    #[unroll]
                    for member in 0..arithmetic_members {
                        if first_member + member < length {
                            let route = map
                                [map_first + MEMBERS + group * MAX_BATCH + first_member + member]
                                as usize;
                            partial[virtual_lane * arithmetic_members + member] +=
                                weight * activations[act_first + route * act_stride + column];
                        }
                    }
                }
            }
        }
    }
    let mut sums = SharedMemory::<f32>::new(comptime!(4 * 16 * arithmetic_members));
    #[unroll]
    for member in 0..arithmetic_members {
        // Original stride32 then stride16 additions, with identical operands
        // and parentheses, precede the remaining original strides8..1.
        let lower = partial[member] + partial[2 * arithmetic_members + member];
        let upper = partial[arithmetic_members + member] + partial[3 * arithmetic_members + member];
        sums[(row * arithmetic_members + member) * 16 + row_lane] = lower + upper;
    }
    sync_cube();
    let mut stride = 8_usize;
    while stride > 0 {
        if row_lane < stride {
            #[unroll]
            for member in 0..arithmetic_members {
                let index = (row * arithmetic_members + member) * 16 + row_lane;
                sums[index] += sums[index + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    if row_lane == 0 {
        #[unroll]
        for member in 0..arithmetic_members {
            if first_member + member < length {
                let route =
                    map[map_first + MEMBERS + group * MAX_BATCH + first_member + member] as usize;
                partials[partial_first + route * partial_stride + feature] =
                    sums[(row * arithmetic_members + member) * 16];
                if feature == 0 {
                    status[status_first + MAX_ROUTES + route] = 1;
                }
            }
        }
    }
}

/// Run once after every wave's completed consumers. Original rank0..9 order;
/// all readiness flags remain mandatory separate acceptance input.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::collapsible_if, clippy::manual_range_contains)]
pub fn grouped_reduce(
    partials: &Array<f32>,
    probabilities: &Array<f32>,
    map: &Array<u32>,
    status: &Array<u32>,
    output: &mut Array<f32>,
    map_first: usize,
    partial_first: usize,
    partial_stride: usize,
    prob_first: usize,
    status_first: usize,
    output_first: usize,
    output_stride: usize,
    #[comptime] hidden: usize,
) {
    let feature = CUBE_POS_X as usize * LANES + UNIT_POS_X as usize;
    let token = CUBE_POS_Y as usize;
    if map_first <= map.len() {
        if MAP_WORDS <= map.len() - map_first {
            let count = map[map_first] as usize;
            if count >= 1 && count <= MAX_BATCH && token < count && feature < hidden {
                let mut ready = true;
                for rank in 0..RANKS {
                    let route = token * RANKS + rank;
                    if status[status_first + route] != 1
                        || status[status_first + MAX_ROUTES + route] != 1
                    {
                        ready = false;
                    }
                }
                if ready {
                    let mut sum = 0.0_f32;
                    for rank in 0..RANKS {
                        let route = token * RANKS + rank;
                        sum += probabilities[prob_first + route]
                            * partials[partial_first + route * partial_stride + feature];
                    }
                    output[output_first + token * output_stride + feature] = sum;
                }
            }
        }
    }
}

/// Full active input/activation/partial/output finite sweeps. Public grid
/// =(max(ceilD/64,ceilFF/64),public_members*10,4); every barrier is unconditional.
/// Default public_members remains MAX8. Ready flags at0..159 are never
/// overwritten by a finite sweep or later wave; full status offsets stay MAX80.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::collapsible_if, clippy::manual_range_contains)]
pub fn grouped_finite(
    input: &Array<f32>,
    activations: &Array<f32>,
    partials: &Array<f32>,
    output: &Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    output_first: usize,
    output_stride: usize,
    status_first: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
) {
    let chunk = CUBE_POS_X as usize;
    let row = CUBE_POS_Y as usize;
    let stage = CUBE_POS_Z as usize;
    let lane = UNIT_POS_X as usize;
    let mut active = false;
    if map_first <= map.len() {
        if MAP_WORDS <= map.len() - map_first {
            let count = map[map_first] as usize;
            if count >= 1 && count <= MAX_BATCH {
                if stage == 0 || stage == 3 {
                    active = row < count && chunk < comptime!(hidden.div_ceil(LANES));
                } else if stage == 1 {
                    active = row < count * RANKS && chunk < comptime!(ff.div_ceil(LANES));
                } else if stage == 2 {
                    active = row < count * RANKS && chunk < comptime!(hidden.div_ceil(LANES));
                }
            }
        }
    }
    let feature = chunk * LANES + lane;
    let mut valid = true;
    if active {
        let mut value = 0.0_f32;
        if stage == 0 && feature < hidden {
            value = input[input_first + row * input_stride + feature];
        } else if stage == 1 && feature < ff {
            value = activations[act_first + row * act_stride + feature];
        } else if stage == 2 && feature < hidden {
            value = partials[partial_first + row * partial_stride + feature];
        } else if stage == 3 && feature < hidden {
            value = output[output_first + row * output_stride + feature];
        }
        valid = value.abs() <= f32::MAX;
    }
    let mut flags = SharedMemory::<u32>::new(LANES);
    // Explicit stores avoid CubeCL's known conditional-literal lowering trap.
    if valid {
        flags[lane] = 1;
    } else {
        flags[lane] = 0;
    }
    sync_cube();
    let mut stride = 32usize;
    while stride > 0 {
        if lane < stride {
            flags[lane] = flags[lane] & flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && active {
        let d = comptime!(hidden.div_ceil(LANES));
        let f = comptime!(ff.div_ceil(LANES));
        if stage == 0 {
            status[status_first + MAX_ROUTES * 2 + row * d + chunk] = flags[0];
        } else if stage == 1 {
            status[status_first + MAX_ROUTES * 2 + MAX_BATCH * d + row * f + chunk] = flags[0];
        } else if stage == 2 {
            status[status_first
                + MAX_ROUTES * 2
                + MAX_BATCH * d
                + MAX_ROUTES * f
                + row * d
                + chunk] = flags[0];
        } else if stage == 3 {
            status[status_first
                + MAX_ROUTES * 2
                + MAX_BATCH * d
                + MAX_ROUTES * f
                + MAX_ROUTES * d
                + row * d
                + chunk] = flags[0];
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn original160_shards_check_all_boundaries_and_exact_three_projection_bindings() {
        let projections = [
            Projection::new(super::super::Q4_K, 2560, 640, 512).unwrap(),
            Projection::new(super::super::Q4_K, 2560, 640, 512).unwrap(),
            Projection::new(super::super::Q8_0, 640, 2560, 512).unwrap(),
        ];
        let g = Geometry::new(projections, 8, 160).unwrap();
        for (first, count) in [(0, 72), (72, 72), (144, 16)] {
            let words = projections.map(|p| p.segment_words(count).unwrap());
            g.validate_shard(first, count, words).unwrap();
            let mut bad = words;
            bad[2] -= 1;
            assert!(g.validate_shard(first, count, bad).is_err());
        }
        for (first, count) in [(64, 64), (144, 15), (160, 16), (0, 160), (144, 72)] {
            let words = projections.map(|p| p.segment_words(count).unwrap_or(0));
            assert!(g.validate_shard(first, count, words).is_err());
        }
    }

    use super::super::{BF16, F16, F32, IQ3_XXS, IQ4_NL, IQ4_XS, Q4_K, Q5_0, Q6_K, Q8_0};
    use super::*;

    fn geometry(dtype: u32, hidden: usize, ff: usize, tile: usize, capacity: usize) -> Geometry {
        Geometry::new(
            [
                Projection::new(dtype, hidden, ff, 512).unwrap(),
                Projection::new(dtype, hidden, ff, 512).unwrap(),
                Projection::new(dtype, ff, hidden, 512).unwrap(),
            ],
            tile,
            capacity,
        )
        .unwrap()
    }
    fn selections(count: usize, unique: bool) -> (Vec<u32>, Vec<f32>) {
        let experts = (0..count * RANKS)
            .map(|route| {
                if unique {
                    route as u32
                } else {
                    // Same ten experts, irregular original rank association each token.
                    ((route % RANKS + route / RANKS * 3) % RANKS) as u32
                }
            })
            .collect();
        let probability = [0.05, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.1, 0.08];
        let probabilities = (0..count * RANKS)
            .map(|route| probability[route % RANKS])
            .collect();
        (experts, probabilities)
    }
    fn protected(experts: &[u32], first: usize, count: usize) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        for &expert in &experts[first * RANKS..(first + count) * RANKS] {
            if !result.iter().any(|&(e, _)| e == expert) {
                result.push((expert, result.len() as u32));
            }
        }
        result.reverse(); // Group order is unrelated to original token/rank.
        result
    }

    #[test]
    fn map_preserves_every_original_rank_probability_across_reuse_and_waves() {
        for count in [1, 3, 5, 8] {
            for unique in [false, true] {
                let (experts, probabilities) = selections(count, unique);
                for (capacity, width) in [(16, 1), (64, 6), (72, 7), (96, 8)] {
                    let mut first = 0;
                    let mut covered = [0u8; MAX_ROUTES];
                    while first < count {
                        let wave = width.min(count - first);
                        let union = protected(&experts, first, wave);
                        let map = GroupMap::new(
                            count,
                            first,
                            wave,
                            &experts,
                            &probabilities,
                            &union,
                            capacity,
                        )
                        .unwrap();
                        assert_eq!(map.count(), count);
                        assert_eq!(map.groups(), union.len());
                        assert_eq!(map.words().len(), MAP_WORDS);
                        assert_eq!(
                            map.probabilities()[..count * RANKS]
                                .iter()
                                .map(|p| p.to_bits())
                                .collect::<Vec<_>>(),
                            probabilities
                                .iter()
                                .map(|p| p.to_bits())
                                .collect::<Vec<_>>()
                        );
                        map.validate(&experts, &probabilities, capacity).unwrap();
                        for (group, &(expert, slot)) in union.iter().enumerate().take(map.groups())
                        {
                            for member in 0..map.words[LENGTHS + group] as usize {
                                let route =
                                    map.words[MEMBERS + group * MAX_BATCH + member] as usize;
                                covered[route] += 1;
                                assert_eq!(experts[route], expert);
                                assert_eq!(map.words[SLOTS + group], slot);
                            }
                        }
                        first += wave;
                    }
                    assert!(covered[..count * RANKS].iter().all(|&v| v == 1));
                    assert!(covered[count * RANKS..].iter().all(|&v| v == 0));
                }
            }
        }
        let (experts, probabilities) = selections(8, true);
        let union = protected(&experts, 0, 8);
        assert!(GroupMap::new(8, 0, 8, &experts, &probabilities, &union, 64).is_err());
        let (experts, probabilities) = selections(8, false);
        let slots = [0, 63, 64, 95, 1, 2, 3, 4, 5, 6];
        let union: Vec<_> = (0..10u32).zip(slots).collect();
        let map = GroupMap::new(8, 0, 8, &experts, &probabilities, &union, 96).unwrap();
        assert_eq!(&map.words[SLOTS..SLOTS + 4], &[0, 63, 64, 95]);
    }

    #[test]
    fn map_rejects_incomplete_crossed_duplicate_and_changed_original_data() {
        let (experts, probabilities) = selections(3, false);
        let union = protected(&experts, 1, 2);
        for (count, first, wave, cap) in [
            (0, 0, 1, 16),
            (9, 0, 1, 16),
            (3, usize::MAX, 2, 16),
            (3, 2, 2, 16),
            (3, 0, 0, 16),
            (3, 0, 3, 9),
            (3, 0, 3, 73),
        ] {
            assert!(
                GroupMap::new(count, first, wave, &experts, &probabilities, &union, cap).is_err()
            );
        }
        let mut bad = probabilities.clone();
        bad[0] = f32::NAN;
        assert!(GroupMap::new(3, 1, 2, &experts, &bad, &union, 16).is_err());
        bad[0] = -0.01;
        assert!(GroupMap::new(3, 1, 2, &experts, &bad, &union, 16).is_err());
        bad[0] = 1.;
        assert!(GroupMap::new(3, 1, 2, &experts, &bad, &union, 16).is_err());
        let mut wrong = experts.clone();
        wrong[1] = wrong[0];
        assert!(GroupMap::new(3, 1, 2, &wrong, &probabilities, &union, 16).is_err());
        let mut duplicate_slot = union.clone();
        duplicate_slot[1].1 = duplicate_slot[0].1;
        assert!(GroupMap::new(3, 1, 2, &experts, &probabilities, &duplicate_slot, 16).is_err());
        assert!(GroupMap::new(3, 1, 2, &experts, &probabilities, &union[..9], 16).is_err());
        for attack in 0..6 {
            let mut map = GroupMap::new(3, 1, 2, &experts, &probabilities, &union, 16).unwrap();
            match attack {
                0 => map.words[MEMBERS] = 0, // out of this wave
                1 => map.words[MEMBERS + 1] = map.words[MEMBERS],
                2 => map.words[LENGTHS] = 0,
                3 => map.words[SLOTS] = 15,
                4 => map.words[SLOTS + 79] = 1,
                _ => map.probabilities[0] += 0.001,
            }
            assert!(map.validate(&experts, &probabilities, 16).is_err());
        }
    }

    #[test]
    fn original_metadata_full_spans_and_exact_page_envelope_are_checked() {
        assert!(Projection::new(99, 256, 64, 512).is_err());
        assert!(Projection::new(Q4_K, 255, 64, 512).is_err());
        assert!(Projection::new(F32, usize::MAX, 64, 512).is_err());
        assert!(Projection::new(F32, 4, 4, 511).is_err());
        let unaligned = Projection::new(F16, 3, 1, 512).unwrap();
        assert!(unaligned.segment_words(16).is_err());
        // Existing banks reject per-expert bytes not divisible by four.
        assert!(
            Geometry::new(
                [
                    unaligned,
                    unaligned,
                    Projection::new(F16, 1, 3, 512).unwrap()
                ],
                8,
                16
            )
            .is_err()
        );
        for down in [Q5_0, Q8_0] {
            let p = [
                Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                Projection::new(down, 640, 2560, 512).unwrap(),
            ];
            let g = Geometry::new(p, 8, 96).unwrap();
            assert_eq!(
                (g.hidden(), g.ff(), g.tile(), g.capacity()),
                (2560, 640, 8, 96)
            );
            assert_eq!(g.projections(), p);
            assert_eq!(
                (p[0].dtype(), p[0].columns(), p[0].rows(), p[0].experts()),
                (Q4_K, 2560, 640, 512)
            );
            assert_eq!(p[0].row_bytes(), 1440);
            assert_eq!(p[0].expert_bytes(), 921600);
            assert_eq!(p[0].segment_words(64).unwrap(), 14745600);
            assert!(p[0].segment_words(0).is_err());
            assert!(p[0].segment_words(usize::MAX).is_err());
            assert_eq!(g.status_words(), 4800);
            assert_eq!(g.scratch_bytes(false).unwrap(), 1376 * 1024);
            assert_eq!(g.scratch_bytes(true).unwrap(), 1632 * 1024);
            assert!(matches!(
                g.grid(80, false).unwrap(),
                CubeCount::Static(640, 80, 1)
            ));
            assert!(matches!(
                g.grid(1, true).unwrap(),
                CubeCount::Static(2560, 1, 1)
            ));
            assert!(g.grid(0, false).is_err());
            assert!(g.grid(81, true).is_err());
            let spans = Spans::new([5, 7, 2563, 11, 643, 13, 2567, 3, 17, 2569, 19]);
            let words = [
                5 + MAP_WORDS,
                7 + 7 * 2563 + 2560,
                11 + 79 * 643 + 640,
                13 + 79 * 2567 + 2560,
                3 + 80,
                17 + 7 * 2569 + 2560,
                19 + 4800,
            ];
            spans.validate(g, words).unwrap();
            for field in 0..7 {
                let mut short = words;
                short[field] -= 1;
                assert!(spans.validate(g, short).is_err());
            }
            assert!(
                Spans::new([usize::MAX, 7, 2563, 11, 643, 13, 2567, 3, 17, 2569, 19])
                    .validate(g, words)
                    .is_err()
            );
            assert!(
                Spans::new([5, 7, 2559, 11, 643, 13, 2567, 3, 17, 2569, 19])
                    .validate(g, words)
                    .is_err()
            );
            assert!(Geometry::new(p, 2, 64).is_err());
            assert!(Geometry::new(p, 8, 73).is_err());
        }
        // Generic4096 arena ceiling is not blanket bank admission. Reject
        // Q4_K/Q8_0 bindings exceeding128MiB, never silently shrink them.
        let p = [
            Projection::new(Q4_K, 4096, 4096, 512).unwrap(),
            Projection::new(Q4_K, 4096, 4096, 512).unwrap(),
            Projection::new(Q8_0, 4096, 4096, 512).unwrap(),
        ];
        assert!(Geometry::new(p, 8, 16).is_err()); // Q4 binding151MiB: reject, do not shrink.
        let g = Geometry {
            projections: p,
            hidden: 4096,
            ff: 4096,
            tile: 8,
            capacity: 16,
            gate_policy: GatePolicy::Generic,
            reduction_policy: ReductionPolicy::AllMembers,
            secondary_partition: false,
            frozen_host_sources: false,
            inactive_partition_exit: false,
            private_tables: false,
            workgroup_tables: false,
            shared_down_scales: false,
            shared_gate_up_factors: false,
            four_row_down: false,
            two_row_gate_up: false,
            three_member_arithmetic: false,
            four_members: false,
        };
        assert_eq!(g.scratch_bytes(true).unwrap(), 4480 * 1024);
    }

    #[test]
    fn explicit_single_member_grid_rejects_other_counts_waves_and_members() {
        let single = geometry(F32, 256, 256, 1, 16);
        let original = geometry(F32, 256, 256, 8, 16);
        assert_eq!(single.member_capacity(), 1);
        assert_eq!(single.status_words(), original.status_words());
        assert_eq!(single.scratch_bytes(true), original.scratch_bytes(true));
        assert!(matches!(
            single.grid(10, false).unwrap(),
            CubeCount::Static(256, 10, 1)
        ));
        single.validate_count(1).unwrap();
        for count in [0, 2, 3, 5, 8, usize::MAX] {
            assert!(single.validate_count(count).is_err());
        }
        let (experts, p) = selections(1, true);
        let union = protected(&experts, 0, 1);
        let mut map = GroupMap::new(1, 0, 1, &experts, &p, &union, 16).unwrap();
        map.validate_geometry(single).unwrap();
        for (index, bad) in [(0, 3), (1, 1), (2, 2), (3, u32::MAX), (LENGTHS, 2)] {
            let old = map.words[index];
            map.words[index] = bad;
            assert!(map.validate_geometry(single).is_err());
            map.words[index] = old;
        }
    }

    #[test]
    fn finite_and_reduce_grids_use_only_public_members_and_keep_full_owners() {
        for down in [Q5_0, Q8_0] {
            let projections = [
                Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                Projection::new(down, 640, 2560, 512).unwrap(),
            ];
            for capacity in [16, 64, 96, 128] {
                let default = Geometry::new(projections, 8, capacity).unwrap();
                let single = Geometry::new(projections, 1, capacity).unwrap();
                let blocked = single.with_q4_block().unwrap();
                for tile in [4, 8] {
                    let g = Geometry::new(projections, tile, capacity).unwrap();
                    assert!(matches!(g.reduce_grid(), CubeCount::Static(40, 8, 1)));
                    assert!(matches!(g.finite_grid(), CubeCount::Static(40, 80, 4)));
                    for count in [1, 3, 5, 8] {
                        g.validate_count(count).unwrap();
                    }
                }
                let spans = Spans::new([5, 7, 2563, 11, 643, 13, 2567, 3, 17, 2569, 19]);
                let full_words = [
                    5 + MAP_WORDS,
                    7 + 7 * 2563 + 2560,
                    11 + 79 * 643 + 640,
                    13 + 79 * 2567 + 2560,
                    3 + MAX_ROUTES,
                    17 + 7 * 2569 + 2560,
                    19 + default.status_words(),
                ];
                for g in [default, single, blocked] {
                    spans.validate(g, full_words).unwrap();
                    assert_eq!(g.status_words(), 4800);
                    assert_eq!(g.scratch_bytes(true), default.scratch_bytes(true));
                    for field in 0..full_words.len() {
                        let mut short = full_words;
                        short[field] -= 1;
                        assert!(spans.validate(g, short).is_err());
                    }
                }
                for g in [single, blocked] {
                    assert!(matches!(g.reduce_grid(), CubeCount::Static(40, 1, 1)));
                    assert!(matches!(g.finite_grid(), CubeCount::Static(40, 10, 4)));
                    g.validate_count(1).unwrap();
                    for count in [0, 2, 3, 5, 8, usize::MAX] {
                        assert!(g.validate_count(count).is_err());
                    }
                }
            }
        }
    }

    #[test]
    fn count_aware_grids_validate_policy_and_keep_original_shape_and_owners() {
        for tile in [1, 4, 8] {
            let g = geometry(F32, 257, 129, tile, 16);
            let full = geometry(F32, 257, 129, 8, 16);
            assert_eq!(g.status_words(), full.status_words());
            assert_eq!(g.scratch_bytes(true), full.scratch_bytes(true));
            for count in 1..=g.member_capacity() {
                assert!(matches!(
                    g.reduce_grid_for_count(count).unwrap(),
                    CubeCount::Static(5, rows, 1) if rows == count as u32
                ));
                assert!(matches!(
                    g.finite_grid_for_count(count).unwrap(),
                    CubeCount::Static(5, rows, 4) if rows == (count * RANKS) as u32
                ));
            }
            for count in [0, g.member_capacity() + 1, 9, usize::MAX] {
                assert_eq!(
                    g.reduce_grid_for_count(count).unwrap_err(),
                    "grouped public member policy count"
                );
                assert_eq!(
                    g.finite_grid_for_count(count).unwrap_err(),
                    "grouped public member policy count"
                );
            }
        }
        let g = geometry(F32, 2560, 640, 8, 16);
        assert!(matches!(
            g.finite_grid_for_count(3).unwrap(),
            CubeCount::Static(40, 30, 4)
        ));
        assert!(matches!(
            g.reduce_grid_for_count(3).unwrap(),
            CubeCount::Static(40, 3, 1)
        ));
        // Same two sweeps and reducer dispatch; no owners/INFO are removed.
        assert_eq!((40 * 80 * 4 - 40 * 30 * 4) * 2 * 48, 768_000);
    }

    #[test]
    fn count_aware_grids_cover_all_active_storage_and_leave_full_inactive_domains() {
        for (hidden, ff) in [(257, 129), (129, 257), (2560, 640)] {
            for tile in [4, 8] {
                let g = geometry(F32, hidden, ff, tile, 16);
                let d = hidden.div_ceil(LANES);
                let f = ff.div_ceil(LANES);
                let starts = [
                    MAX_ROUTES * 2,
                    MAX_ROUTES * 2 + MAX_BATCH * d,
                    MAX_ROUTES * 2 + MAX_BATCH * d + MAX_ROUTES * f,
                    MAX_ROUTES * 2 + MAX_BATCH * d + MAX_ROUTES * f + MAX_ROUTES * d,
                ];
                for count in 1..=MAX_BATCH {
                    let mut expected_status = vec![0u8; g.status_words()];
                    for (stage, length) in
                        [count * d, count * RANKS * f, count * RANKS * d, count * d]
                            .into_iter()
                            .enumerate()
                    {
                        expected_status[starts[stage]..starts[stage] + length].fill(1);
                    }
                    // Independently expected full MAX8 output footprint. A
                    // count-aware grid must preserve every active feature and
                    // never touch ready flags, inactive rows or status gaps.
                    let mut expected_output = vec![0u8; MAX_BATCH * hidden];
                    expected_output[..count * hidden].fill(1);
                    for grid in [g.finite_grid(), g.finite_grid_for_count(count).unwrap()] {
                        let CubeCount::Static(chunks, rows, stages) = grid else {
                            panic!("fixed finite grid");
                        };
                        let mut writes = vec![0u8; g.status_words()];
                        for (stage, &start) in starts.iter().enumerate().take(stages as usize) {
                            let (active_rows, width) = match stage {
                                0 | 3 => (count, d),
                                1 => (count * RANKS, f),
                                2 => (count * RANKS, d),
                                _ => panic!("finite stage"),
                            };
                            for row in 0..rows as usize {
                                for chunk in 0..chunks as usize {
                                    if row < active_rows && chunk < width {
                                        writes[start + row * width + chunk] += 1;
                                    }
                                }
                            }
                        }
                        assert_eq!(writes, expected_status);
                    }
                    for grid in [g.reduce_grid(), g.reduce_grid_for_count(count).unwrap()] {
                        let CubeCount::Static(chunks, rows, depth) = grid else {
                            panic!("fixed reducer grid");
                        };
                        assert_eq!(depth, 1);
                        let mut writes = vec![0u8; MAX_BATCH * hidden];
                        for token in 0..rows as usize {
                            for feature in 0..chunks as usize * LANES {
                                if token < count && feature < hidden {
                                    writes[token * hidden + feature] += 1;
                                }
                            }
                        }
                        assert_eq!(writes, expected_output);
                    }
                }
            }
        }
    }

    #[test]
    fn removed_count1_cubes_have_no_output_or_status_storage_effect() {
        let default = geometry(F32, 2560, 640, 8, 16);
        let single = geometry(F32, 2560, 640, 1, 16);
        // Enumerate the unchanged shader's storage-write coordinates. The
        // expected active ranges are separate full-MAX80 status sections;
        // all ready flags and inactive locations retain their prior bits.
        let d = default.hidden().div_ceil(LANES);
        let f = default.ff().div_ceil(LANES);
        let starts = [
            MAX_ROUTES * 2,
            MAX_ROUTES * 2 + MAX_BATCH * d,
            MAX_ROUTES * 2 + MAX_BATCH * d + MAX_ROUTES * f,
            MAX_ROUTES * 2 + MAX_BATCH * d + MAX_ROUTES * f + MAX_ROUTES * d,
        ];
        let mut expected_status = vec![false; default.status_words()];
        for (stage, length) in [d, RANKS * f, RANKS * d, d].into_iter().enumerate() {
            expected_status[starts[stage]..starts[stage] + length].fill(true);
        }
        let mut expected_output = vec![false; MAX_BATCH * default.hidden()];
        expected_output[..default.hidden()].fill(true);
        for g in [default, single] {
            let CubeCount::Static(chunks, rows, stages) = g.finite_grid() else {
                panic!("fixed public finite grid");
            };
            let mut status_writes = vec![false; g.status_words()];
            for (stage, &start) in starts.iter().enumerate().take(stages as usize) {
                for row in 0..rows as usize {
                    for chunk in 0..chunks as usize {
                        let (active_rows, width) = match stage {
                            0 | 3 => (1, d),
                            1 => (RANKS, f),
                            2 => (RANKS, d),
                            _ => panic!("finite stage"),
                        };
                        if row < active_rows && chunk < width {
                            let index = start + row * width + chunk;
                            assert!(!status_writes[index]);
                            status_writes[index] = true;
                        }
                    }
                }
            }
            assert_eq!(status_writes, expected_status);
            let CubeCount::Static(chunks, rows, depth) = g.reduce_grid() else {
                panic!("fixed public reducer grid");
            };
            assert_eq!(depth, 1);
            let mut output_writes = vec![false; MAX_BATCH * g.hidden()];
            for token in 0..rows as usize {
                for feature in 0..chunks as usize * LANES {
                    if token < 1 && feature < g.hidden() {
                        output_writes[token * g.hidden() + feature] = true;
                    }
                }
            }
            assert_eq!(output_writes, expected_output);
        }
    }

    #[test]
    fn q4_block_is_explicit_public_tile1_policy_with_unchanged_owners() {
        assert_eq!(GatePolicy::default(), GatePolicy::Generic);
        for tile in [1, 4, 8] {
            for dtype in [
                F32, F16, BF16, Q4_K, Q5_0, Q6_K, Q8_0, IQ3_XXS, IQ4_NL, IQ4_XS,
            ] {
                let g = geometry(dtype, 256, 256, tile, 16);
                assert_eq!(g.gate_policy(), GatePolicy::Generic);
                let blocked = g.with_q4_block();
                assert_eq!(blocked.is_ok(), tile == 1 && dtype == Q4_K);
                if let Ok(b) = blocked {
                    assert_eq!(b.gate_policy(), GatePolicy::Q4Block);
                    assert_eq!(b.scratch_bytes(true), g.scratch_bytes(true));
                    assert_eq!(b.status_words(), g.status_words());
                    assert_eq!(b.projections(), g.projections());
                    b.validate_count(1).unwrap();
                    assert!(b.validate_count(2).is_err());
                }
            }
        }
        let q4 = Projection::new(Q4_K, 256, 256, 512).unwrap();
        let q8 = Projection::new(Q8_0, 256, 256, 512).unwrap();
        for p in [[q4, q8, q8], [q8, q4, q8]] {
            assert!(Geometry::new(p, 1, 16).unwrap().with_q4_block().is_err());
        }
    }

    fn q4_test_block(seed: usize) -> Vec<u8> {
        let mut packed = vec![0u8; 144];
        packed[..2].copy_from_slice(&0x2800u16.to_le_bytes());
        packed[2..4].copy_from_slice(&0x2400u16.to_le_bytes());
        for group in 0..8 {
            let scale = ((group * 19 + seed * 7) % 64) as u8;
            let minimum = ((group * 29 + seed * 11) % 64) as u8;
            if group < 4 {
                packed[4 + group] |= scale;
                packed[8 + group] |= minimum;
            } else {
                packed[8 + group] = (scale & 15) | ((minimum & 15) << 4);
                packed[group] |= (scale >> 4) << 6;
                packed[4 + group] |= (minimum >> 4) << 6;
            }
        }
        for column in 0..256 {
            let group = column / 32;
            packed[16 + (group / 2) * 32 + column % 32] |=
                (((column * 7 + group + seed * 3) % 16) as u8) << ((group % 2) * 4);
        }
        packed
    }
    fn q4_block_dot(words: &[u32], row: usize, x: &[f32]) -> f32 {
        let blocks = x.len() / 256;
        let mut sums = [0f32; LANES];
        for (lane, sum) in sums.iter_mut().enumerate() {
            for block in 0..blocks {
                let base = row * blocks * 36 + block * 36;
                let d = binary16(words[base] & 65535);
                let dmin = binary16(words[base] >> 16);
                let scales = [words[base + 1], words[base + 2], words[base + 3]];
                for pair in 0..4 {
                    let group = pair * 2 + lane / 32;
                    let byte = q4_word_byte(words[base + 4 + pair * 8 + (lane % 32) / 4], lane & 3);
                    let quant = (byte >> ((lane / 32) * 4)) & 15;
                    let factor = d * q4_block_scale(scales[0], scales[2], group) as f32;
                    let bias = dmin * q4_block_minimum(scales[1], scales[2], group) as f32;
                    let weight = factor * quant as f32 - bias;
                    let column = block * 256 + pair * LANES + lane;
                    *sum += weight * x[column];
                }
            }
        }
        for stride in [32, 16, 8, 4, 2, 1] {
            for lane in 0..stride {
                sums[lane] += sums[lane + stride];
            }
        }
        sums[0]
    }
    #[test]
    fn q4_block_fields_and_lane_order_match_independent_decoder_at_shard_edges() {
        use kuno_text::TensorType;
        for hidden in [256, 512] {
            let blocks = hidden / 256;
            let packed = (0..96 * blocks).flat_map(q4_test_block).collect::<Vec<_>>();
            let words = super::super::pack_le_words(&packed);
            let x = (0..hidden)
                .map(|i| ((i % 17) as f32 - 8.) / 16.)
                .collect::<Vec<_>>();
            for row in [0, 63, 64, 95, 1, 62, 65, 94, 31, 32] {
                let mut decoded = vec![0f32; hidden];
                TensorType::Q4K
                    .decode(
                        &packed[row * blocks * 144..(row + 1) * blocks * 144],
                        &mut decoded,
                    )
                    .unwrap();
                let mut lanes = [0f32; LANES];
                for (column, (&weight, &input)) in decoded.iter().zip(&x).enumerate() {
                    lanes[column % LANES] += weight * input;
                }
                for stride in [32, 16, 8, 4, 2, 1] {
                    for lane in 0..stride {
                        lanes[lane] += lanes[lane + stride];
                    }
                }
                let original = lanes[0];
                let candidate = q4_block_dot(&words, row, &x);
                assert_eq!(candidate.to_bits(), original.to_bits());
                let f64 = decoded
                    .iter()
                    .zip(&x)
                    .map(|(&w, &v)| f64::from(w) * f64::from(v))
                    .sum::<f64>() as f32;
                assert_eq!(candidate.to_bits(), f64.to_bits());
                assert_eq!(
                    grouped_silu(candidate).to_bits(),
                    kuno_text::ops::silu(original).to_bits()
                );
                // Every original column, including the bank's final word.
                for column in 0..hidden {
                    let mut basis = vec![0f32; hidden];
                    basis[column] = 1.;
                    assert_eq!(
                        q4_block_dot(&words, row, &basis).to_bits(),
                        decoded[column].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn missing_ready_or_late_nonfinite_and_every_inactive_tail_reject_acceptance() {
        let g = geometry(F32, 65, 7, 8, 16);
        for count in [1, 3, 5, 8] {
            let first = 5;
            let mut status = vec![0u32; first + g.status_words() + 13];
            let regions = [
                (80, count * 10),
                (80, count * 10),
                (16, count * 2),
                (80, count * 10),
                (160, count * 20),
                (16, count * 2),
            ];
            let mut offset = first;
            for (length, active) in regions {
                status[offset..offset + active].fill(1);
                offset += length;
            }
            validate_status(g, count, first, &status).unwrap();
            let mut base = first;
            for (length, active) in regions {
                let index = base + active - 1;
                status[index] = 0;
                assert!(validate_status(g, count, first, &status).is_err());
                status[index] = 1;
                if active < length {
                    status[base + active] = 1;
                    assert!(validate_status(g, count, first, &status).is_err());
                    status[base + active] = 0;
                }
                base += length;
            }
            for index in [0, status.len() - 1] {
                status[index] = 1;
                assert!(validate_status(g, count, first, &status).is_err());
                status[index] = 0;
            }
            assert!(validate_status(g, 0, first, &status).is_err());
            assert!(validate_status(g, 9, first, &status).is_err());
            assert!(validate_status(g, count, usize::MAX, &status).is_err());
        }
    }

    fn tree_dot(weights: &[f32], input: &[f32]) -> f32 {
        let mut lanes = [0f32; 64];
        for (column, (&w, &x)) in weights.iter().zip(input).enumerate() {
            lanes[column % 64] += w * x;
        }
        for stride in [32, 16, 8, 4, 2, 1] {
            for lane in 0..stride {
                lanes[lane] += lanes[lane + stride];
            }
        }
        lanes[0]
    }
    fn core_silu(x: f32) -> f32 {
        kuno_text::ops::silu(x)
    }

    #[test]
    fn grouped_dense_numeric_oracle_preserves_original_routes_trees_and_rank_sum() {
        let (hidden, ff) = (65, 7);
        for count in [1, 3, 5, 8] {
            let (experts, p) = selections(count, false);
            let union = protected(&experts, 0, count);
            let map = GroupMap::new(count, 0, count, &experts, &p, &union, 16).unwrap();
            let x: Vec<f32> = (0..count * hidden)
                .map(|i| ((i * 7 % 37) as i32 - 18) as f32 / 32.)
                .collect();
            let weight = |expert: usize, phase: usize, row: usize, column: usize| {
                ((expert * 11 + phase * 17 + row * 5 + column * 3) % 41) as f32 / 64. - 20. / 64.
            };
            let mut scalar_act = vec![0f32; count * 10 * ff];
            let mut scalar_parts = vec![0f32; count * 10 * hidden];
            for route in 0..count * 10 {
                let expert = experts[route] as usize;
                let input = &x[route / 10 * hidden..(route / 10 + 1) * hidden];
                for feature in 0..ff {
                    let gate: Vec<_> = (0..hidden)
                        .map(|col| weight(expert, 0, feature, col))
                        .collect();
                    let up: Vec<_> = (0..hidden)
                        .map(|col| weight(expert, 1, feature, col))
                        .collect();
                    scalar_act[route * ff + feature] =
                        core_silu(tree_dot(&gate, input)) * tree_dot(&up, input);
                }
                for feature in 0..hidden {
                    let down: Vec<_> = (0..ff).map(|col| weight(expert, 2, feature, col)).collect();
                    scalar_parts[route * hidden + feature] =
                        tree_dot(&down, &scalar_act[route * ff..(route + 1) * ff]);
                }
            }
            for tile in [1, 4, 8] {
                if tile == 1 && count != 1 {
                    continue;
                }
                let mut acts = vec![0f32; count * 10 * ff];
                let mut parts = vec![0f32; count * 10 * hidden];
                for (group, &(expert, _)) in union.iter().enumerate().take(map.groups()) {
                    let length = map.words[LENGTHS + group] as usize;
                    let expert = expert as usize;
                    for first in (0..if tile == 1 { 1 } else { 8 }).step_by(tile) {
                        for feature in 0..ff {
                            let mut gates = vec![[0f32; 64]; tile];
                            let mut ups = vec![[0f32; 64]; tile];
                            for column in 0..hidden {
                                let g = weight(expert, 0, feature, column);
                                let u = weight(expert, 1, feature, column);
                                for member in 0..tile {
                                    if first + member < length {
                                        let route = map.words[MEMBERS + group * 8 + first + member]
                                            as usize;
                                        let value = x[route / 10 * hidden + column];
                                        gates[member][column % 64] += g * value;
                                        ups[member][column % 64] += u * value;
                                    }
                                }
                            }
                            for stride in [32, 16, 8, 4, 2, 1] {
                                for lane in 0..stride {
                                    for member in 0..tile {
                                        gates[member][lane] += gates[member][lane + stride];
                                        ups[member][lane] += ups[member][lane + stride];
                                    }
                                }
                            }
                            for member in 0..tile {
                                if first + member < length {
                                    let route =
                                        map.words[MEMBERS + group * 8 + first + member] as usize;
                                    acts[route * ff + feature] =
                                        core_silu(gates[member][0]) * ups[member][0];
                                }
                            }
                        }
                        for feature in 0..hidden {
                            let mut sums = vec![[0f32; 64]; tile];
                            for column in 0..ff {
                                let w = weight(expert, 2, feature, column);
                                for (member, lanes) in sums.iter_mut().enumerate() {
                                    if first + member < length {
                                        let route = map.words[MEMBERS + group * 8 + first + member]
                                            as usize;
                                        lanes[column % 64] += w * acts[route * ff + column];
                                    }
                                }
                            }
                            for stride in [32, 16, 8, 4, 2, 1] {
                                for lane in 0..stride {
                                    for lanes in &mut sums {
                                        lanes[lane] += lanes[lane + stride];
                                    }
                                }
                            }
                            for (member, lanes) in sums.iter().enumerate() {
                                if first + member < length {
                                    let route =
                                        map.words[MEMBERS + group * 8 + first + member] as usize;
                                    parts[route * hidden + feature] = lanes[0];
                                }
                            }
                        }
                    }
                }
                assert_eq!(
                    acts.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    scalar_act.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    parts.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    scalar_parts.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                for token in 0..count {
                    for feature in 0..hidden {
                        let mut actual = 0f32;
                        let mut expected = 0f32;
                        for rank in 0..10 {
                            let route = token * 10 + rank;
                            actual += map.probabilities[route] * parts[route * hidden + feature];
                            expected += p[route] * scalar_parts[route * hidden + feature];
                        }
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn full_private_map_and_addressing_wipe_is_explicit_and_idempotent() {
        let (experts, p) = selections(8, false);
        let union = protected(&experts, 0, 8);
        let mut map = GroupMap::new(8, 0, 8, &experts, &p, &union, 64).unwrap();
        map.wipe();
        map.wipe();
        assert!(map.words().iter().all(|&v| v == 0));
        assert!(map.probabilities().iter().all(|v| v.to_bits() == 0));
        assert!(map.validate(&experts, &p, 64).is_err());
        let mut spans = Spans::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        spans.wipe();
        spans.wipe();
        assert_eq!(
            [
                spans.map_first(),
                spans.input_first(),
                spans.input_stride(),
                spans.act_first(),
                spans.act_stride(),
                spans.partial_first(),
                spans.partial_stride(),
                spans.prob_first(),
                spans.output_first(),
                spans.output_stride(),
                spans.status_first()
            ],
            [0; 11]
        );
        assert_eq!(union.len(), 10); // Borrowed bank protection remains caller-owned.
    }

    #[test]
    fn only_exact_full_original_shards_are_admitted_before_indexing() {
        for down in [Q5_0, Q8_0] {
            let p = [
                Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                Projection::new(down, 640, 2560, 512).unwrap(),
            ];
            for cap in [16, 64, 72, 96, 128] {
                let g = Geometry::new(p, 8, cap).unwrap();
                let shards = if crate::bank_slots::segmented_capacity(cap) {
                    vec![(0, 64), (64, cap - 64)]
                } else {
                    vec![(0, cap)]
                };
                for (first, count) in shards {
                    let words = p.map(|p| p.segment_words(count).unwrap());
                    g.validate_shard(first, count, words).unwrap();
                    for field in 0..3 {
                        let mut short = words;
                        short[field] -= 1;
                        assert!(g.validate_shard(first, count, short).is_err());
                    }
                    assert!(g.validate_shard(first + 1, count, words).is_err());
                    assert!(g.validate_shard(first, count - 1, words).is_err());
                    assert!(g.validate_shard(usize::MAX, count, words).is_err());
                }
            }
        }
    }

    #[test]
    fn unsloth_iq_shards_keep_exact_original_row_strides_and_generic_policy() {
        for (gate, row_bytes, expert_bytes) in [(IQ3_XXS, 980, 627200), (IQ4_XS, 1360, 870400)] {
            for down in [IQ4_NL, Q8_0] {
                let p = [
                    Projection::new(gate, 2560, 640, 512).unwrap(),
                    Projection::new(gate, 2560, 640, 512).unwrap(),
                    Projection::new(down, 640, 2560, 512).unwrap(),
                ];
                assert_eq!(p[0].row_bytes(), row_bytes);
                assert_eq!(p[0].expert_bytes(), expert_bytes);
                assert_eq!(p[1], p[0]);
                assert_eq!(p[2].row_bytes(), if down == IQ4_NL { 360 } else { 680 });
                for cap in [16, 64, 72, 96, 128, 144, 160] {
                    for tile in [1, 4, 8] {
                        let g = Geometry::new(p, tile, cap).unwrap();
                        assert_eq!(g.gate_policy(), GatePolicy::Generic);
                        assert!(g.with_q4_block().is_err());
                        assert!(Geometry::stationary_secondary(p).is_err());
                        let flat = [cap];
                        let shards = crate::bank_slots::segmented_capacities(cap).unwrap_or(&flat);
                        let mut first = 0;
                        for &slots in shards {
                            if slots == 0 {
                                continue;
                            }
                            let words =
                                p.map(|projection| projection.segment_words(slots).unwrap());
                            g.validate_shard(first, slots, words).unwrap();
                            for field in 0..3 {
                                let mut short = words;
                                short[field] -= 1;
                                assert!(g.validate_shard(first, slots, short).is_err());
                            }
                            first += slots;
                        }
                        assert_eq!(first, cap);
                    }
                }
                for cap in [96, 128, 144, 160] {
                    let mut wrong = p;
                    wrong[1] = Projection::new(
                        if gate == IQ3_XXS { IQ4_XS } else { IQ3_XXS },
                        2560,
                        640,
                        512,
                    )
                    .unwrap();
                    assert!(Geometry::new(wrong, 1, cap).is_err());
                }
            }
        }
    }

    #[test]
    fn offline_iq_original_gate_up_and_down_validate_for_count1_and_max8() {
        use cubecl::server::ExecutionMode;
        for gate in [IQ3_XXS, IQ4_XS] {
            for down in [IQ4_NL, Q8_0] {
                let p = [
                    Projection::new(gate, 2560, 640, 512).unwrap(),
                    Projection::new(gate, 2560, 640, 512).unwrap(),
                    Projection::new(down, 640, 2560, 512).unwrap(),
                ];
                for tile in [1, 4, 8] {
                    let g = Geometry::new(p, tile, 160).unwrap();
                    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                        validate_shader(&shader(0, g, mode), 8, true);
                        validate_shader(&shader(1, g, mode), 8, true);
                        validate_shader(&shader(2, g, mode), 7, false);
                        validate_shader(&shader(3, g, mode), 10, true);
                    }
                }
            }
        }
    }

    fn shader(kind: usize, g: Geometry, mode: cubecl::server::ExecutionMode) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope.register_type::<f32>(float);
        b.scope
            .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        match kind {
            0 => {
                let gate = b.input_array(Type::new(uint)).into();
                let up = b.input_array(Type::new(uint)).into();
                let input = b.input_array(Type::new(float)).into();
                let map = b.input_array(Type::new(uint)).into();
                let act = b.output_array(Type::new(float)).into();
                let status = b.output_array(Type::new(uint)).into();
                let map_first = b.scalar(uint).into();
                let input_first = b.scalar(uint).into();
                let input_stride = b.scalar(uint).into();
                let act_first = b.scalar(uint).into();
                let act_stride = b.scalar(uint).into();
                let status_first = b.scalar(uint).into();
                let first_slot = b.scalar(uint).into();
                let slot_count = b.scalar(uint).into();
                let expand = if g.workgroup_tables() {
                    grouped_gate_up_frozen_workgroup_grid::expand
                } else if g.private_tables() {
                    grouped_gate_up_frozen_private_grid::expand
                } else if g.inactive_partition_exit() {
                    grouped_gate_up_frozen::expand
                } else if g.live_member_reduction() {
                    grouped_gate_up_live::expand
                } else {
                    grouped_gate_up::expand
                };
                expand(
                    &mut b.scope,
                    gate,
                    up,
                    input,
                    map,
                    act,
                    status,
                    map_first,
                    input_first,
                    input_stride,
                    act_first,
                    act_stride,
                    status_first,
                    first_slot,
                    slot_count,
                    g.hidden,
                    g.ff,
                    g.projections[0].dtype,
                    g.projections[1].dtype,
                    g.projections[0].row_bytes,
                    g.projections[1].row_bytes,
                    g.tile,
                    g.gate_policy == GatePolicy::Q4Block,
                );
            }
            1 => {
                let down = b.input_array(Type::new(uint)).into();
                let act = b.input_array(Type::new(float)).into();
                let map = b.input_array(Type::new(uint)).into();
                let partial = b.output_array(Type::new(float)).into();
                let status = b.output_array(Type::new(uint)).into();
                let map_first = b.scalar(uint).into();
                let act_first = b.scalar(uint).into();
                let act_stride = b.scalar(uint).into();
                let partial_first = b.scalar(uint).into();
                let partial_stride = b.scalar(uint).into();
                let status_first = b.scalar(uint).into();
                let first_slot = b.scalar(uint).into();
                let slot_count = b.scalar(uint).into();
                let expand = if g.shared_down_scales() {
                    grouped_down_frozen_shared_scales::expand
                } else if g.inactive_partition_exit() {
                    grouped_down_frozen::expand
                } else if g.live_member_reduction() {
                    grouped_down_live::expand
                } else {
                    grouped_down::expand
                };
                expand(
                    &mut b.scope,
                    down,
                    act,
                    map,
                    partial,
                    status,
                    map_first,
                    act_first,
                    act_stride,
                    partial_first,
                    partial_stride,
                    status_first,
                    first_slot,
                    slot_count,
                    g.hidden,
                    g.ff,
                    g.projections[2].dtype,
                    g.projections[2].row_bytes,
                    g.tile,
                );
            }
            2 => {
                let partial = b.input_array(Type::new(float)).into();
                let prob = b.input_array(Type::new(float)).into();
                let map = b.input_array(Type::new(uint)).into();
                let status = b.input_array(Type::new(uint)).into();
                let output = b.output_array(Type::new(float)).into();
                let map_first = b.scalar(uint).into();
                let partial_first = b.scalar(uint).into();
                let partial_stride = b.scalar(uint).into();
                let prob_first = b.scalar(uint).into();
                let status_first = b.scalar(uint).into();
                let output_first = b.scalar(uint).into();
                let output_stride = b.scalar(uint).into();
                grouped_reduce::expand(
                    &mut b.scope,
                    partial,
                    prob,
                    map,
                    status,
                    output,
                    map_first,
                    partial_first,
                    partial_stride,
                    prob_first,
                    status_first,
                    output_first,
                    output_stride,
                    g.hidden,
                );
            }
            3 => {
                let input = b.input_array(Type::new(float)).into();
                let act = b.input_array(Type::new(float)).into();
                let partial = b.input_array(Type::new(float)).into();
                let output = b.input_array(Type::new(float)).into();
                let map = b.input_array(Type::new(uint)).into();
                let status = b.output_array(Type::new(uint)).into();
                let map_first = b.scalar(uint).into();
                let input_first = b.scalar(uint).into();
                let input_stride = b.scalar(uint).into();
                let act_first = b.scalar(uint).into();
                let act_stride = b.scalar(uint).into();
                let partial_first = b.scalar(uint).into();
                let partial_stride = b.scalar(uint).into();
                let output_first = b.scalar(uint).into();
                let output_stride = b.scalar(uint).into();
                let status_first = b.scalar(uint).into();
                grouped_finite::expand(
                    &mut b.scope,
                    input,
                    act,
                    partial,
                    output,
                    map,
                    status,
                    map_first,
                    input_first,
                    input_stride,
                    act_first,
                    act_stride,
                    partial_first,
                    partial_stride,
                    output_first,
                    output_stride,
                    status_first,
                    g.hidden,
                    g.ff,
                );
            }
            _ => unreachable!(),
        }
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name(if kind == 0 && g.workgroup_tables() {
                            "grouped_gate_up_frozen_workgroup_grid"
                        } else if kind == 0 && g.private_tables() {
                            "grouped_gate_up_frozen_private_grid"
                        } else {
                            "grouped_moe_operator"
                        }),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                uint,
            )
            .unwrap()
            .to_string()
    }

    fn validate_shader(source: &str, scalars: usize, barrier: bool) {
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert!(source.contains("@compute"));
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        let bindings = module
            .global_variables
            .iter()
            .filter(|(_, v)| v.binding.is_some())
            .count();
        assert!(
            bindings <= 7,
            "operator exceeds usual eight-storage-binding budget including INFO"
        );
        assert_eq!(source.contains("workgroupBarrier"), barrier);
        for scalar in 0..scalars {
            assert!(
                source.contains(&format!("scalars_u32[{scalar}]")),
                "missing runtime scalar{scalar}"
            );
        }
    }

    #[test]
    fn frozen_inactive_exit_validates_checked_and_unchecked_shaders() {
        use cubecl::server::ExecutionMode;
        let mut profiles = Vec::new();
        for layer in 0..48 {
            let types = crate::daslab_profile::routed_types(layer).unwrap();
            if !profiles.contains(&types) {
                profiles.push(types);
            }
        }
        for [gate, up, down] in profiles {
            let p = [
                Projection::new(gate, 2560, 640, 512).unwrap(),
                Projection::new(up, 2560, 640, 512).unwrap(),
                Projection::new(down, 640, 2560, 512).unwrap(),
            ];
            let ordinary = Geometry::new(p, 8, 144).unwrap();
            let old = ordinary.with_frozen_host_sources().unwrap();
            let new = old.with_inactive_partition_exit().unwrap();
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                for kind in [0, 1] {
                    let old_source = shader(kind, old, mode);
                    // Uninitialized declaration order can vary between CubeCL
                    // expansions; compare lowered operation/owner structure.
                    let ordinary_module =
                        wgpu::naga::front::wgsl::parse_str(&shader(kind, ordinary, mode)).unwrap();
                    let new_source = shader(kind, new, mode);
                    validate_shader(&new_source, 8, true);
                    let original = wgpu::naga::front::wgsl::parse_str(&old_source).unwrap();
                    let candidate = wgpu::naga::front::wgsl::parse_str(&new_source).unwrap();
                    let old_body = &original.entry_points[0].function.body;
                    let new_body = &candidate.entry_points[0].function.body;
                    // One handoff plus a loop barrier executed at six strides.
                    assert_eq!(unconditional_barriers(old_body, false), 2);
                    assert_eq!(unconditional_barriers(new_body, false), 2);
                    let old_events = memory_events(&original);
                    assert_eq!(old_events, memory_events(&ordinary_module));
                    assert_eq!(
                        arithmetic_operations(&original),
                        arithmetic_operations(&ordinary_module)
                    );
                    let new_events = memory_events(&candidate);
                    assert_eq!(
                        new_events.iter().filter(|&&x| x == 'r').count(),
                        old_events.iter().filter(|&&x| x == 'r').count() + 1
                    );
                    // This is an actual terminating path, before the first
                    // workgroup or output store and before any shared barrier.
                    let exit = new_events.iter().position(|&x| x == 'r').unwrap();
                    for event in ['s', 'w', 'b'] {
                        let first = new_events.iter().position(|&x| x == event).unwrap();
                        assert!(exit < first);
                        assert_eq!(
                            old_events.iter().filter(|&&x| x == event).count(),
                            new_events.iter().filter(|&&x| x == event).count()
                        );
                    }
                    assert_eq!(
                        arithmetic_operations(&original),
                        arithmetic_operations(&candidate)
                    );
                    let info = wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::empty(),
                    )
                    .validate(&candidate)
                    .unwrap();
                    let words = wgpu::naga::back::spv::write_vec(
                        &candidate,
                        &info,
                        &wgpu::naga::back::spv::Options {
                            lang_version: (1, 3),
                            zero_initialize_workgroup_memory:
                                wgpu::naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                            ..Default::default()
                        },
                        Some(&wgpu::naga::back::spv::PipelineOptions {
                            shader_stage: wgpu::naga::ShaderStage::Compute,
                            entry_point: candidate.entry_points[0].name.clone(),
                        }),
                    )
                    .unwrap();
                    assert!(!words.is_empty());
                }
            }
        }
    }

    #[test]
    fn private_table_actual_mixed_shaders_remove_function_array_spills() {
        use cubecl::server::ExecutionMode;
        use wgpu::naga::{AddressSpace, Expression, Literal};

        fn table_values(module: &wgpu::naga::Module, private: bool) -> Vec<Vec<u32>> {
            fn value(module: &wgpu::naga::Module, h: wgpu::naga::Handle<Expression>) -> u32 {
                match module.global_expressions[h] {
                    Expression::Literal(Literal::U32(v)) => v,
                    Expression::Literal(Literal::I32(v)) => v as u32,
                    Expression::Literal(Literal::AbstractInt(v)) => v as u32,
                    Expression::As { expr, .. } => value(module, expr),
                    ref e => panic!("unexpected public table literal: {e:?}"),
                }
            }
            let initializers: Vec<_> = if private {
                module
                    .global_variables
                    .iter()
                    .filter_map(|(_, var)| {
                        if var
                            .name
                            .as_deref()
                            .is_some_and(|n| n.starts_with("arrays_"))
                        {
                            assert_eq!(var.space, AddressSpace::Private);
                            Some(var.init.unwrap())
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                module
                    .constants
                    .iter()
                    .filter_map(|(_, c)| {
                        c.name
                            .as_deref()
                            .is_some_and(|n| n.starts_with("arrays_"))
                            .then_some(c.init)
                    })
                    .collect()
            };
            let mut tables: Vec<_> = initializers
                .into_iter()
                .map(|h| match &module.global_expressions[h] {
                    Expression::Compose { components, .. } => {
                        components.iter().map(|&v| value(module, v)).collect()
                    }
                    e => panic!("unexpected public table initializer: {e:?}"),
                })
                .collect();
            tables.sort();
            tables
        }

        // Inspect actual Naga SPIR-V. A dynamic value-array access spills its
        // entire initializer into a Function variable; pointer indexing must
        // eliminate both that variable and every whole-composite store to it.

        let mut profiles = Vec::new();
        for layer in 0..48 {
            let p = crate::daslab_profile::routed_types(layer).unwrap();
            if !profiles.contains(&p) {
                profiles.push(p);
            }
        }
        for [gate, up, down] in profiles {
            let old = Geometry::new(
                [
                    Projection::new(gate, 2560, 640, 512).unwrap(),
                    Projection::new(up, 2560, 640, 512).unwrap(),
                    Projection::new(down, 640, 2560, 512).unwrap(),
                ],
                8,
                144,
            )
            .unwrap()
            .with_frozen_host_sources()
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
            let new = old.with_private_tables().unwrap();
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let old_source = shader(0, old, mode);
                let new_source = shader(0, new, mode);
                validate_shader(&new_source, 8, true);
                assert!(new_source.contains("fn grouped_gate_up_frozen_private_grid"));
                let original = wgpu::naga::front::wgsl::parse_str(&old_source).unwrap();
                let candidate = wgpu::naga::front::wgsl::parse_str(&new_source).unwrap();
                assert_public_tables_immutable(&candidate);
                let old_tables = table_values(&original, false);
                let new_tables = table_values(&candidate, true);
                assert_eq!(old_tables, new_tables);
                for table in &new_tables {
                    let expected: Vec<u32> = match table.len() {
                        256 => kuno_text::quant::IQ3_XXS_GRID.to_vec(),
                        512 => kuno_text::quant::IQ3_S_GRID.to_vec(),
                        2048 => kuno_text::quant::IQ2_S_GRID
                            .iter()
                            .flat_map(|v| [*v as u32, (*v >> 32) as u32])
                            .collect(),
                        n => panic!("unexpected public table size {n}"),
                    };
                    assert_eq!(*table, expected);
                }
                assert_eq!(
                    arithmetic_operations(&original),
                    arithmetic_operations(&candidate)
                );
                assert_eq!(memory_events(&original), memory_events(&candidate));
                assert_eq!(
                    unconditional_barriers(&original.entry_points[0].function.body, false),
                    2
                );
                assert_eq!(
                    unconditional_barriers(&candidate.entry_points[0].function.body, false),
                    2
                );
                let before = spirv_tables(&original);
                let after = spirv_tables(&candidate);
                assert_eq!(after.0, 0);
                assert_eq!(after.2, 0);
                assert_eq!(after.1, new_tables.len());
                if !new_tables.is_empty() {
                    assert!(before.0 > 0 && before.2 >= before.0);
                } else {
                    assert_eq!(before, (0, 0, 0));
                }
                eprintln!(
                    "private_table gate={gate} up={up} mode={mode:?} public_words={:?} original_function_private_stores={before:?} candidate={after:?}",
                    new_tables.iter().map(Vec::len).collect::<Vec<_>>()
                );
                // Downward expansion and its compiler selection stay unchanged.
                let old_down = wgpu::naga::front::wgsl::parse_str(&shader(1, old, mode)).unwrap();
                let new_down = wgpu::naga::front::wgsl::parse_str(&shader(1, new, mode)).unwrap();
                assert_eq!(
                    table_values(&old_down, false),
                    table_values(&new_down, false)
                );
                assert_eq!(
                    arithmetic_operations(&old_down),
                    arithmetic_operations(&new_down)
                );
                assert_eq!(memory_events(&old_down), memory_events(&new_down));
            }
        }
    }

    fn assert_public_tables_immutable(module: &wgpu::naga::Module) {
        use wgpu::naga::{AddressSpace, Expression, Function, Statement};
        fn global(
            function: &Function,
            pointer: wgpu::naga::Handle<Expression>,
        ) -> Option<wgpu::naga::Handle<wgpu::naga::GlobalVariable>> {
            match function.expressions[pointer] {
                Expression::GlobalVariable(var) => Some(var),
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    global(function, base)
                }
                _ => None,
            }
        }
        fn visit(module: &wgpu::naga::Module, function: &Function, body: &wgpu::naga::Block) {
            for statement in body {
                match statement {
                    Statement::Store { pointer, .. } => {
                        if let Some(var) = global(function, *pointer) {
                            let var = &module.global_variables[var];
                            assert!(
                                !(var.space == AddressSpace::Private
                                    && var
                                        .name
                                        .as_deref()
                                        .is_some_and(|n| n.starts_with("arrays_"))),
                                "public private table or element was mutated"
                            );
                        }
                    }
                    Statement::Block(block) => visit(module, function, block),
                    Statement::If { accept, reject, .. } => {
                        visit(module, function, accept);
                        visit(module, function, reject);
                    }
                    Statement::Switch { cases, .. } => {
                        for case in cases {
                            visit(module, function, &case.body);
                        }
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => {
                        visit(module, function, body);
                        visit(module, function, continuing);
                    }
                    _ => {}
                }
            }
        }
        for entry in &module.entry_points {
            visit(module, &entry.function, &entry.function.body);
        }
        for (_, function) in module.functions.iter() {
            visit(module, function, &function.body);
        }
    }

    #[test]
    #[should_panic(expected = "public private table or element was mutated")]
    fn private_table_element_store_cannot_bypass_immutability_proof() {
        let module = wgpu::naga::front::wgsl::parse_str(
            "var<private> arrays_0: array<u32,256>;\n\
             @compute @workgroup_size(64) fn main(\
             @builtin(local_invocation_id) id: vec3<u32>) { arrays_0[id.x] = 1u; }",
        )
        .unwrap();
        assert_public_tables_immutable(&module);
    }

    fn runtime_spirv(module: &wgpu::naga::Module) -> Vec<u32> {
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(module)
        .unwrap();
        wgpu::naga::back::spv::write_vec(
            module,
            &info,
            &wgpu::naga::back::spv::Options {
                lang_version: (1, 3),
                // CubeCL backend/base.rs disables implicit workgroup zeroing.
                zero_initialize_workgroup_memory:
                    wgpu::naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                ..Default::default()
            },
            Some(&wgpu::naga::back::spv::PipelineOptions {
                shader_stage: wgpu::naga::ShaderStage::Compute,
                entry_point: module.entry_points[0].name.clone(),
            }),
        )
        .unwrap()
    }

    fn runtime_spirv_barriers(module: &wgpu::naga::Module) -> usize {
        let words = runtime_spirv(module);
        let mut cursor = 5;
        let mut barriers = 0;
        while cursor < words.len() {
            let len = (words[cursor] >> 16) as usize;
            assert!(len > 0);
            barriers += usize::from(words[cursor] & 0xffff == 224); // OpControlBarrier
            cursor += len;
        }
        assert_eq!(cursor, words.len());
        barriers
    }

    fn shared_pointer_global(
        function: &wgpu::naga::Function,
        pointer: wgpu::naga::Handle<wgpu::naga::Expression>,
    ) -> Option<wgpu::naga::Handle<wgpu::naga::GlobalVariable>> {
        use wgpu::naga::Expression;
        match function.expressions[pointer] {
            Expression::GlobalVariable(var) => Some(var),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                shared_pointer_global(function, base)
            }
            _ => None,
        }
    }

    fn is_physical_lane_x(
        function: &wgpu::naga::Function,
        expression: wgpu::naga::Handle<wgpu::naga::Expression>,
    ) -> bool {
        use wgpu::naga::{Binding, BuiltIn, Expression};
        match function.expressions[expression] {
            Expression::As { expr, .. } => is_physical_lane_x(function, expr),
            Expression::AccessIndex { base, index: 0 } => match function.expressions[base] {
                Expression::FunctionArgument(arg) => matches!(
                    function.arguments[arg as usize].binding,
                    Some(Binding::BuiltIn(BuiltIn::LocalInvocationId))
                ),
                _ => false,
            },
            _ => false,
        }
    }

    fn assert_shared_loads_follow_barrier(
        module: &wgpu::naga::Module,
        array: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
    ) {
        use wgpu::naga::{Expression, Function, Statement};
        fn visit(
            function: &Function,
            block: &wgpu::naga::Block,
            array: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
            mut ready: bool,
        ) -> bool {
            for statement in block {
                match statement {
                    Statement::Emit(range) => {
                        for handle in range.clone() {
                            if let Expression::Load { pointer } = function.expressions[handle] {
                                assert!(
                                    ready
                                        || shared_pointer_global(function, pointer) != Some(array),
                                    "shared array read before initialization barrier"
                                );
                            }
                        }
                    }
                    Statement::ControlBarrier(_) => ready = true,
                    Statement::Block(body) => ready = visit(function, body, array, ready),
                    Statement::If { accept, reject, .. } => {
                        visit(function, accept, array, ready);
                        visit(function, reject, array, ready);
                    }
                    Statement::Switch { cases, .. } => {
                        for case in cases {
                            visit(function, &case.body, array, ready);
                        }
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => {
                        visit(function, body, array, ready);
                        visit(function, continuing, array, ready);
                    }
                    _ => {}
                }
            }
            ready
        }
        let function = &module.entry_points[0].function;
        visit(function, &function.body, array, false);
        for (_, function) in module.functions.iter() {
            visit(function, &function.body, array, false);
        }
    }

    fn spirv_tables(module: &wgpu::naga::Module) -> (usize, usize, usize) {
        use std::collections::HashMap;
        let words = runtime_spirv(module);
        let mut instructions = Vec::new();
        let mut cursor = 5;
        while cursor < words.len() {
            let len = (words[cursor] >> 16) as usize;
            assert!(len > 0);
            instructions.push((words[cursor] & 0xffff, &words[cursor + 1..cursor + len]));
            cursor += len;
        }
        assert_eq!(cursor, words.len());
        let mut constants = HashMap::new();
        let mut arrays = HashMap::new();
        let mut pointers = HashMap::new();
        for &(opcode, operands) in &instructions {
            match opcode {
                43 if operands.len() == 3 => {
                    constants.insert(operands[1], operands[2]);
                }
                28 => {
                    arrays.insert(operands[0], operands[2]);
                }
                32 => {
                    pointers.insert(operands[0], (operands[1], operands[2]));
                }
                _ => {}
            }
        }
        let mut functions = Vec::new();
        let mut private = Vec::new();
        for &(opcode, operands) in &instructions {
            if opcode != 59 {
                continue;
            }
            if let Some(&(space, pointee)) = pointers.get(&operands[0])
                && arrays
                    .get(&pointee)
                    .and_then(|id| constants.get(id))
                    .is_some_and(|&n| matches!(n, 256 | 512 | 2048))
            {
                match space {
                    7 => functions.push(operands[1]),
                    6 => private.push(operands[1]),
                    _ => {}
                }
            }
        }
        let stores = instructions
            .iter()
            .filter(|(opcode, operands)| *opcode == 62 && functions.contains(&operands[0]))
            .count();
        assert!(
            !instructions
                .iter()
                .any(|(opcode, operands)| { *opcode == 62 && private.contains(&operands[0]) }),
            "public private table was mutated after its initializer"
        );
        (functions.len(), private.len(), stores)
    }

    fn assert_workgroup_tables_immutable_after_init(module: &wgpu::naga::Module) {
        use wgpu::naga::{Expression, Function, Statement};
        fn global(
            function: &Function,
            pointer: wgpu::naga::Handle<Expression>,
        ) -> Option<wgpu::naga::Handle<wgpu::naga::GlobalVariable>> {
            match function.expressions[pointer] {
                Expression::GlobalVariable(var) => Some(var),
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    global(function, base)
                }
                _ => None,
            }
        }
        fn visit(module: &wgpu::naga::Module, function: &Function, body: &wgpu::naga::Block) {
            for statement in body {
                match statement {
                    Statement::Store { pointer, .. } => {
                        if let Some(var) = global(function, *pointer) {
                            assert!(
                                !module.global_variables[var]
                                    .name
                                    .as_deref()
                                    .is_some_and(|n| n.starts_with("arrays_")),
                                "public workgroup table or element was mutated after init"
                            );
                        }
                    }
                    Statement::Block(b) => visit(module, function, b),
                    Statement::If { accept, reject, .. } => {
                        visit(module, function, accept);
                        visit(module, function, reject);
                    }
                    Statement::Switch { cases, .. } => {
                        for case in cases {
                            visit(module, function, &case.body);
                        }
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => {
                        visit(module, function, body);
                        visit(module, function, continuing);
                    }
                    _ => {}
                }
            }
        }
        for (array, variable) in module.global_variables.iter() {
            if variable
                .name
                .as_deref()
                .is_some_and(|n| n.starts_with("arrays_"))
            {
                assert_shared_loads_follow_barrier(module, array);
            }
        }
        let function = &module.entry_points[0].function;
        let barrier = function
            .body
            .iter()
            .position(|s| matches!(s, Statement::ControlBarrier(_)))
            .unwrap();
        for statement in function.body.iter().skip(barrier + 1) {
            visit(
                module,
                function,
                &wgpu::naga::Block::from_vec(vec![statement.clone()]),
            );
        }
        for (_, function) in module.functions.iter() {
            visit(module, function, &function.body);
        }
    }

    #[test]
    #[should_panic(expected = "public workgroup table or element was mutated after init")]
    fn workgroup_table_element_store_cannot_bypass_immutability_proof() {
        let module = wgpu::naga::front::wgsl::parse_str(
            "var<workgroup> arrays_0: array<u32,256>; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { workgroupBarrier(); if id.x==0u {arrays_0[id.x]=1u;} }"
        ).unwrap();
        assert_workgroup_tables_immutable_after_init(&module);
    }

    #[test]
    #[should_panic(expected = "shared array read before initialization barrier")]
    fn workgroup_table_read_before_barrier_cannot_bypass_initialization_proof() {
        let module = wgpu::naga::front::wgsl::parse_str(
            "var<workgroup> arrays_0:array<u32,256>; @group(0) @binding(0) var<storage,read_write> output:array<u32>; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { output[id.x]=arrays_0[id.x]; workgroupBarrier(); }"
        ).unwrap();
        assert_workgroup_tables_immutable_after_init(&module);
    }

    #[test]
    fn workgroup_table_actual_mixed_shaders_literal_init_and_unchanged_math() {
        use cubecl::server::ExecutionMode;
        use wgpu::naga::{
            AddressSpace, ArraySize, Expression, Literal, Statement, SwitchValue, TypeInner,
        };
        let mut profiles = Vec::new();
        for layer in 0..48 {
            let profile = crate::daslab_profile::routed_types(layer).unwrap();
            if !profiles.contains(&profile) {
                profiles.push(profile);
            }
        }
        for [gate, up, down] in profiles {
            assert_eq!(gate, up);
            let old = Geometry::new(
                [
                    Projection::new(gate, 2560, 640, 512).unwrap(),
                    Projection::new(up, 2560, 640, 512).unwrap(),
                    Projection::new(down, 640, 2560, 512).unwrap(),
                ],
                8,
                144,
            )
            .unwrap()
            .with_frozen_host_sources()
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
            let new = old.with_workgroup_tables().unwrap();
            let expected: Vec<u32> = match gate {
                IQ3_XXS => kuno_text::quant::IQ3_XXS_GRID.to_vec(),
                crate::quant_matvec::IQ3_S => kuno_text::quant::IQ3_S_GRID.to_vec(),
                crate::quant_matvec::IQ2_S => kuno_text::quant::IQ2_S_GRID
                    .iter()
                    .flat_map(|v| [*v as u32, (*v >> 32) as u32])
                    .collect(),
                _ => Vec::new(),
            };
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let source = shader(0, new, mode);
                validate_shader(&source, 8, true);
                let candidate = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                let original = wgpu::naga::front::wgsl::parse_str(&shader(0, old, mode)).unwrap();
                let grids: Vec<_> = candidate
                    .global_variables
                    .iter()
                    .filter(|(_, var)| {
                        var.name
                            .as_deref()
                            .is_some_and(|n| n.starts_with("arrays_"))
                    })
                    .collect();
                assert_eq!(grids.len(), usize::from(!expected.is_empty()));
                let shared_bytes: u32 = candidate
                    .global_variables
                    .iter()
                    .filter(|(_, var)| var.space == AddressSpace::WorkGroup)
                    .map(|(_, var)| match candidate.types[var.ty].inner {
                        TypeInner::Array {
                            size: ArraySize::Constant(n),
                            stride,
                            ..
                        } => n.get() * stride,
                        ref t => panic!("unexpected shared owner {t:?}"),
                    })
                    .sum();
                assert_eq!(shared_bytes, 4096 + expected.len() as u32 * 4);
                let mut events = memory_events(&candidate);
                if !expected.is_empty() {
                    let (grid, variable) = grids[0];
                    assert_eq!(variable.space, AddressSpace::WorkGroup);
                    assert!(variable.init.is_none());
                    let function = &candidate.entry_points[0].function;
                    let (switch_index, cases) = function
                        .body
                        .iter()
                        .enumerate()
                        .find_map(|(i, statement)| match statement {
                            Statement::Switch { selector, cases } => {
                                assert!(
                                    is_physical_lane_x(function, *selector),
                                    "physical lane selector required for table initializer"
                                );
                                Some((i, cases))
                            }
                            _ => None,
                        })
                        .expect("top-level literal-lane initializer");
                    assert_eq!(cases.len(), 65);
                    assert!(matches!(
                        function.body[switch_index + 1],
                        Statement::ControlBarrier(_)
                    ));
                    // Only the original uniform exit precedes initialization;
                    // every lane reaches the initializer barrier together.
                    assert!(
                        memory_events(&candidate)
                            .iter()
                            .position(|&e| e == 'r')
                            .unwrap()
                            < events.iter().position(|&e| e == 'w').unwrap()
                    );
                    let mut words = vec![None; expected.len()];
                    for case in cases {
                        assert!(!case.fall_through);
                        let lane = match case.value {
                            SwitchValue::U32(v) => v as usize,
                            SwitchValue::Default => {
                                assert!(case.body.is_empty());
                                continue;
                            }
                            _ => panic!("public unsigned lane literal required"),
                        };
                        assert!(lane < 64);
                        for statement in &case.body {
                            if matches!(statement, Statement::Emit(_)) {
                                continue;
                            }
                            let Statement::Store { pointer, value } = *statement else {
                                panic!("only public literal stores in initializer");
                            };
                            let (base, index) = match function.expressions[pointer] {
                                Expression::AccessIndex { base, index } => (base, index as usize),
                                Expression::Access { base, index } => {
                                    let Expression::Literal(Literal::U32(index)) =
                                        function.expressions[index]
                                    else {
                                        panic!("literal table index required");
                                    };
                                    (base, index as usize)
                                }
                                _ => panic!("direct table element required"),
                            };
                            assert!(
                                matches!(function.expressions[base], Expression::GlobalVariable(v) if v == grid)
                            );
                            let Expression::Literal(Literal::U32(word)) =
                                function.expressions[value]
                            else {
                                panic!("literal public word required");
                            };
                            assert_eq!(index % 64, lane);
                            assert_eq!(word, expected[index]);
                            assert!(words[index].replace(word).is_none());
                        }
                    }
                    assert_eq!(
                        words,
                        expected.iter().copied().map(Some).collect::<Vec<_>>()
                    );
                    assert_workgroup_tables_immutable_after_init(&candidate);
                    let first = events.iter().position(|&e| e == 'w').unwrap();
                    assert!(
                        events[first..first + expected.len()]
                            .iter()
                            .all(|&e| e == 'w')
                    );
                    assert_eq!(events[first + expected.len()], 'b');
                    events.drain(first..first + expected.len() + 1);
                }
                assert_eq!(events, memory_events(&original));
                // Literal-index initialization adds no runtime arithmetic.
                assert_eq!(
                    arithmetic_operations(&candidate),
                    arithmetic_operations(&original)
                );
                assert_eq!(
                    unconditional_barriers(&candidate.entry_points[0].function.body, false),
                    2 + usize::from(!expected.is_empty())
                );
                assert_eq!(spirv_tables(&candidate), (0, 0, 0));
                assert_eq!(
                    runtime_spirv_barriers(&candidate),
                    2 + usize::from(!expected.is_empty())
                );
                for kind in [1, 2, 3] {
                    let a = wgpu::naga::front::wgsl::parse_str(&shader(kind, old, mode)).unwrap();
                    let b = wgpu::naga::front::wgsl::parse_str(&shader(kind, new, mode)).unwrap();
                    assert_eq!(memory_events(&a), memory_events(&b));
                    assert_eq!(arithmetic_operations(&a), arithmetic_operations(&b));
                }
                eprintln!(
                    "workgroup_table gate={gate} mode={mode:?} words={} shared_bytes={shared_bytes} no_function_or_private_tables=true",
                    expected.len()
                );
            }
        }
    }

    fn arithmetic_operations(module: &wgpu::naga::Module) -> Vec<String> {
        let mut operations = Vec::new();
        for (_, expression) in module.entry_points[0].function.expressions.iter() {
            match expression {
                wgpu::naga::Expression::Binary { op, .. } => operations.push(format!("{op:?}")),
                wgpu::naga::Expression::Math { fun, .. } => operations.push(format!("{fun:?}")),
                _ => {}
            }
        }
        operations.sort_unstable();
        operations
    }

    fn memory_events(module: &wgpu::naga::Module) -> Vec<char> {
        use wgpu::naga::{AddressSpace, Expression, Function, Statement};
        fn global(
            function: &Function,
            pointer: wgpu::naga::Handle<Expression>,
        ) -> Option<wgpu::naga::Handle<wgpu::naga::GlobalVariable>> {
            match function.expressions[pointer] {
                Expression::GlobalVariable(var) => Some(var),
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    global(function, base)
                }
                _ => None,
            }
        }
        fn visit(module: &wgpu::naga::Module, body: &wgpu::naga::Block, out: &mut Vec<char>) {
            let function = &module.entry_points[0].function;
            for statement in body {
                match statement {
                    Statement::Return { .. } => out.push('r'),
                    Statement::ControlBarrier(_) => out.push('b'),
                    Statement::Store { pointer, .. } => {
                        if let Some(var) = global(function, *pointer) {
                            match module.global_variables[var].space {
                                AddressSpace::Storage { .. } => out.push('s'),
                                AddressSpace::WorkGroup => out.push('w'),
                                _ => {}
                            }
                        }
                    }
                    Statement::Block(block) => visit(module, block, out),
                    Statement::If { accept, reject, .. } => {
                        visit(module, accept, out);
                        visit(module, reject, out);
                    }
                    Statement::Switch { cases, .. } => {
                        for case in cases {
                            visit(module, &case.body, out);
                        }
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => {
                        visit(module, body, out);
                        visit(module, continuing, out);
                    }
                    _ => {}
                }
            }
        }
        let mut events = Vec::new();
        visit(module, &module.entry_points[0].function.body, &mut events);
        events
    }

    fn unconditional_barriers(block: &wgpu::naga::Block, conditional: bool) -> usize {
        use wgpu::naga::Statement;
        let mut barriers = 0;
        for statement in block {
            match statement {
                Statement::ControlBarrier(_) => {
                    assert!(
                        !conditional,
                        "workgroup barrier nested under a runtime predicate"
                    );
                    barriers += 1;
                }
                Statement::Block(block) => barriers += unconditional_barriers(block, conditional),
                Statement::If { accept, reject, .. } => {
                    barriers += unconditional_barriers(accept, true);
                    barriers += unconditional_barriers(reject, true);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        barriers += unconditional_barriers(&case.body, true);
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    barriers += unconditional_barriers(body, conditional);
                    barriers += unconditional_barriers(continuing, conditional);
                }
                _ => {}
            }
        }
        barriers
    }

    #[test]
    fn live_member_policy_preserves_every_owner_grid_and_public_scalar() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let key = |value: Geometry| {
            let mut hash = DefaultHasher::new();
            value.hash(&mut hash);
            hash.finish()
        };
        let axes = |grid: CubeCount| match grid {
            CubeCount::Static(x, y, z) => (x, y, z),
            _ => panic!("public grouped geometry must retain static dispatch axes"),
        };
        for tile in [4, 8] {
            let original = geometry(F32, 65, 7, tile, 16);
            let live = original.with_live_member_reduction().unwrap();
            assert!(!original.live_member_reduction());
            assert!(live.live_member_reduction());
            assert_ne!(key(original), key(live));
            assert_eq!(original.projections(), live.projections());
            assert_eq!(original.member_capacity(), live.member_capacity());
            assert_eq!(original.status_words(), live.status_words());
            for include_io in [false, true] {
                assert_eq!(
                    original.scratch_bytes(include_io),
                    live.scratch_bytes(include_io)
                );
            }
            for count in [1, 2, 3, 5, 8] {
                let (selected, probabilities) = selections(count, false);
                let slots = protected(&selected, 0, count);
                let map =
                    GroupMap::new(count, 0, count, &selected, &probabilities, &slots, 16).unwrap();
                map.validate_geometry(live).unwrap();
                for down in [false, true] {
                    assert_eq!(
                        axes(original.grid(map.groups(), down).unwrap()),
                        axes(live.grid(map.groups(), down).unwrap())
                    );
                }
                assert_eq!(
                    axes(original.finite_grid_for_count(count).unwrap()),
                    axes(live.finite_grid_for_count(count).unwrap())
                );
                assert_eq!(
                    axes(original.reduce_grid_for_count(count).unwrap()),
                    axes(live.reduce_grid_for_count(count).unwrap())
                );
                let mut bad =
                    GroupMap::new(count, 0, count, &selected, &probabilities, &slots, 16).unwrap();
                bad.words[LENGTHS] = (count + 1) as u32;
                // Geometry admission intentionally excludes private map data;
                // the unchanged canonical route proof rejects this malformed
                // length before either public kernel policy may be launched.
                assert!(bad.validate(&selected, &probabilities, 16).is_err());
            }
        }
        assert!(
            geometry(F32, 65, 7, 1, 16)
                .with_live_member_reduction()
                .is_err()
        );
        assert!(
            geometry(Q4_K, 256, 256, 1, 16)
                .with_q4_block()
                .unwrap()
                .with_live_member_reduction()
                .is_err()
        );
    }

    #[test]
    fn live_member_original64_tree_skips_uninitialized_planes_without_changing_active_bits() {
        for tile in [4, 8] {
            for length in [1, 2, 3, 5, 8] {
                for first in (0..MAX_BATCH).step_by(tile) {
                    // A group validated by the original packet proof provides
                    // its length. An absent/out-of-segment/feature-disabled or
                    // exhausted group has no initialized reduction plane.
                    for active in [false, true] {
                        let live = if active && first < length {
                            tile.min(length - first)
                        } else {
                            0
                        };
                        let poison = f32::from_bits(0x7fc0_1234);
                        let mut optimized = [[poison; LANES]; MAX_BATCH];
                        let mut original = [[0f32; LANES]; MAX_BATCH];
                        let mut reference = [0f32; MAX_BATCH];
                        for member in 0..live {
                            let row = std::array::from_fn::<f32, LANES, _>(|lane| {
                                match (lane + member + first) % 7 {
                                    0 => 1.0e12,
                                    1 => -1.0e12,
                                    2 => -0.0,
                                    3 => 0.000_123,
                                    4 => -0.75,
                                    5 => 17.25,
                                    _ => -0.000_031,
                                }
                            });
                            original[member] = row;
                            optimized[member] = row;
                            reference[member] = tree_dot(&row, &[1.; LANES]);
                        }
                        let before = optimized;
                        let mut adds = 0;
                        for stride in [32, 16, 8, 4, 2, 1] {
                            for lane in 0..stride {
                                for member in 0..tile {
                                    original[member][lane] += original[member][lane + stride];
                                    if member < live {
                                        optimized[member][lane] += optimized[member][lane + stride];
                                        adds += 1;
                                    }
                                }
                            }
                        }
                        assert_eq!(adds, live * 63);
                        for member in 0..live {
                            assert_eq!(original[member][0].to_bits(), reference[member].to_bits());
                            assert_eq!(optimized[member][0].to_bits(), reference[member].to_bits());
                        }
                        for member in live..MAX_BATCH {
                            assert_eq!(
                                optimized[member].map(f32::to_bits),
                                before[member].map(f32::to_bits)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn live_member_checked_and_unchecked_shaders_keep_uniform_barriers_and_original_bindings() {
        use cubecl::server::ExecutionMode;
        for tile in [4, 8] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                for (gate, down) in [
                    (IQ3_XXS, IQ4_NL),
                    (IQ3_XXS, Q8_0),
                    (IQ4_XS, IQ4_NL),
                    (IQ4_XS, Q8_0),
                    (super::super::IQ2_S, IQ4_NL),
                    (super::super::IQ2_S, super::super::Q2_0),
                    (super::super::IQ3_S, IQ4_NL),
                    (super::super::IQ3_S, super::super::Q2_0),
                    (IQ3_XXS, super::super::Q2_0),
                ] {
                    let original = Geometry::new(
                        [
                            Projection::new(gate, 2560, 640, 512).unwrap(),
                            Projection::new(gate, 2560, 640, 512).unwrap(),
                            Projection::new(down, 640, 2560, 512).unwrap(),
                        ],
                        tile,
                        144,
                    )
                    .unwrap();
                    let live = original.with_live_member_reduction().unwrap();
                    for kind in [0, 1] {
                        let source = shader(kind, live, mode);
                        validate_shader(&source, 8, true);
                        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                        let barriers: usize = module
                            .functions
                            .iter()
                            .map(|(_, f)| unconditional_barriers(&f.body, false))
                            .sum::<usize>()
                            + unconditional_barriers(&module.entry_points[0].function.body, false);
                        assert_eq!(barriers, 2); // Initial handoff plus unconditional tree loop.
                        let old = shader(kind, original, mode);
                        assert_ne!(
                            canonical_declaration_order(&source),
                            canonical_declaration_order(&old)
                        );
                        let old_module = wgpu::naga::front::wgsl::parse_str(&old).unwrap();
                        let bindings = |m: &wgpu::naga::Module| {
                            m.global_variables
                                .iter()
                                .filter_map(|(_, v)| v.binding)
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(bindings(&module), bindings(&old_module));
                    }
                }
            }
        }
    }

    #[test]
    fn offline_shaders_all_original_decoders_reuse_weights_across_rhs() {
        use cubecl::server::ExecutionMode;
        for dtype in [
            F32, F16, BF16, Q4_K, Q5_0, Q6_K, Q8_0, IQ3_XXS, IQ4_NL, IQ4_XS,
        ] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let a = geometry(dtype, 256, 256, 4, 16);
                let b = geometry(dtype, 256, 256, 8, 16);
                let single = geometry(dtype, 256, 256, 1, 16);
                for kind in [0, 1] {
                    let one = shader(kind, single, mode);
                    let four = shader(kind, a, mode);
                    let eight = shader(kind, b, mode);
                    validate_shader(&one, 8, true);
                    validate_shader(&four, 8, true);
                    validate_shader(&eight, 8, true);
                    let weights = if kind == 0 { 2 } else { 1 };
                    for buffer in 0..weights {
                        let pattern = format!("buffer_{buffer}_global[");
                        assert!(four.matches(&pattern).count() > 0);
                        assert_eq!(
                            one.matches(&pattern).count(),
                            four.matches(&pattern).count()
                        );
                        assert_eq!(
                            four.matches(&pattern).count(),
                            eight.matches(&pattern).count(),
                            "weight decoding must not multiply with tile"
                        );
                    }
                    let input_buffer = if kind == 0 { 2 } else { 1 };
                    let pattern = format!("buffer_{input_buffer}_global[");
                    assert_eq!(
                        four.matches(&pattern).count(),
                        4 * one.matches(&pattern).count()
                    );
                    assert_eq!(
                        eight.matches(&pattern).count(),
                        2 * four.matches(&pattern).count()
                    );
                }
            }
        }
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let g = geometry(F32, 65, 7, 4, 16);
            validate_shader(&shader(2, g, mode), 7, false);
            validate_shader(&shader(3, g, mode), 10, true);
            for tile in [1, 4, 8] {
                for down in [Q5_0, Q8_0] {
                    let p = [
                        Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                        Projection::new(Q4_K, 2560, 640, 512).unwrap(),
                        Projection::new(down, 640, 2560, 512).unwrap(),
                    ];
                    let g = Geometry::new(p, tile, 96).unwrap();
                    validate_shader(&shader(0, g, mode), 8, true);
                    validate_shader(&shader(1, g, mode), 8, true);
                }
            }
            let mixed = Geometry::new(
                [
                    Projection::new(Q4_K, 256, 256, 512).unwrap(),
                    Projection::new(Q6_K, 256, 256, 512).unwrap(),
                    Projection::new(F32, 256, 256, 512).unwrap(),
                ],
                4,
                16,
            )
            .unwrap();
            validate_shader(&shader(0, mixed, mode), 8, true);
        }
    }
    #[test]
    fn offline_q4_block_headers_are_reused_without_new_bindings_or_runtime_keys() {
        use cubecl::server::ExecutionMode;
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let generic = geometry(Q4_K, 256, 256, 1, 16);
            let blocked = generic.with_q4_block().unwrap();
            let source = shader(0, blocked, mode);
            validate_shader(&source, 8, true);
            for buffer in 0..2 {
                // Four header/scale word loads and four quant word loads.
                // The block loop is runtime; the four pairs are unrolled.
                assert_eq!(
                    source.matches(&format!("buffer_{buffer}_global[")).count(),
                    8
                );
            }
            // CubeCL may enumerate uninitialized mutable declarations in a
            // different order. Compare exact shader text after sorting only
            // contiguous declarations at the same indentation; no operation,
            // initializer, binding or scope boundary may move.
            for kind in 1..4 {
                assert_eq!(
                    canonical_declaration_order(&shader(kind, blocked, mode)),
                    canonical_declaration_order(&shader(kind, generic, mode))
                );
            }
        }
    }

    fn canonical_declaration_order(source: &str) -> String {
        let mut lines = Vec::new();
        let mut declarations = Vec::new();
        let mut indentation = None;
        for line in source.lines() {
            let trimmed = line.trim_start();
            let prefix = &line[..line.len() - trimmed.len()];
            let uninitialized = trimmed.starts_with("var l_mut_")
                && trimmed.ends_with(';')
                && trimmed.contains(':')
                && !trimmed.contains(['=', '{', '}']);
            if !uninitialized || indentation.is_some_and(|previous| previous != prefix) {
                declarations.sort_unstable();
                lines.append(&mut declarations);
                indentation = None;
            }
            if uninitialized {
                indentation = Some(prefix);
                declarations.push(line);
            } else {
                lines.push(line);
            }
        }
        declarations.sort_unstable();
        lines.append(&mut declarations);
        let mut canonical = lines.join("\n");
        if source.ends_with('\n') {
            canonical.push('\n');
        }
        canonical
    }

    #[test]
    fn declaration_normalization_preserves_initializers_operations_and_scope() {
        let original =
            "fn main() {\n    var l_mut_8: f32;\n    var l_mut_1: bool;\n    let value = 1.;\n}\n";
        let reordered =
            "fn main() {\n    var l_mut_1: bool;\n    var l_mut_8: f32;\n    let value = 1.;\n}\n";
        assert_eq!(
            canonical_declaration_order(original),
            canonical_declaration_order(reordered)
        );
        for changed in [
            "fn main() {\n    var l_mut_1: bool;\n    var l_mut_8: f32 = 0.;\n    let value = 1.;\n}\n",
            "fn main() {\n    var l_mut_1: bool;\n    let value = 1.;\n    var l_mut_8: f32;\n}\n",
            "fn main() {\n    var l_mut_1: bool;\n    if true {\n        var l_mut_8: f32;\n    }\n    let value = 1.;\n}\n",
            "fn main() {\n    var l_mut_1: bool;\n    var l_mut_8: f32;\n    let value = 2.;\n}\n",
        ] {
            assert_ne!(
                canonical_declaration_order(original),
                canonical_declaration_order(changed)
            );
        }
        let initialized = "var l_mut_8: f32 = 0.;\nvar l_mut_1: f32;\n";
        let moved = "var l_mut_1: f32;\nvar l_mut_8: f32 = 0.;\n";
        assert_ne!(
            canonical_declaration_order(initialized),
            canonical_declaration_order(moved)
        );
    }

    fn packed_fixture(dtype: u32, rows: usize, columns: usize, phase: usize) -> Vec<u8> {
        if matches!(dtype, IQ3_XXS | IQ4_NL | IQ4_XS) {
            return super::super::iq_fixtures::packed(dtype, rows, columns, phase);
        }
        let (elements, bytes) = super::super::block_layout(dtype).unwrap();
        let mut result = vec![0u8; rows * columns / elements * bytes];
        for (index, block) in result.chunks_exact_mut(bytes).enumerate() {
            match dtype {
                F32 => {
                    let v = ((index * 7 + phase * 3) % 29) as f32 / 32. - 14. / 32.;
                    block.copy_from_slice(&v.to_le_bytes());
                }
                F16 => {
                    let bits = [0x3000u16, 0xb400, 0x3800, 0xb800, 0x3400, 0];
                    block.copy_from_slice(&bits[(index + phase) % bits.len()].to_le_bytes());
                }
                BF16 => {
                    let bits = [0x3e00u16, 0xbe80, 0x3f00, 0xbf00, 0x3e80, 0];
                    block.copy_from_slice(&bits[(index + phase) % bits.len()].to_le_bytes());
                }
                Q6_K => {
                    for (i, b) in block[..208].iter_mut().enumerate() {
                        *b = (index * 29 + i * 37 + phase * 23 + 7) as u8;
                    }
                    block[208..].copy_from_slice(&0x2400u16.to_le_bytes());
                }
                Q4_K => {
                    block[..2].copy_from_slice(&0x2000u16.to_le_bytes());
                    block[2..4].copy_from_slice(&0x1800u16.to_le_bytes());
                    for (i, b) in block[4..].iter_mut().enumerate() {
                        *b = (index * 13 + i * 31 + phase * 47 + 23) as u8;
                    }
                }
                Q5_0 | Q8_0 => {
                    block[..2].copy_from_slice(&0x2400u16.to_le_bytes());
                    for (i, b) in block[2..].iter_mut().enumerate() {
                        *b = (index * 17 + i * 43 + phase * 29 + 11) as u8;
                    }
                }
                _ => unreachable!(),
            }
        }
        result
    }

    /// CPU grouped projection schedule: independent core decoded weights;
    /// source loads counted outside the member loop, runtime map/scatter and
    /// full owning guards preserved. It is not a GPU acceptance substitute.
    // Fixture mirrors full runtime span arguments without hiding addressing.
    #[allow(clippy::too_many_arguments)]
    fn grouped_projection(
        map: &GroupMap<'_>,
        tile: usize,
        rows: usize,
        columns: usize,
        decoded: &[f32],
        input: &[f32],
        input_first: usize,
        input_stride: usize,
        output: &mut [f32],
        output_first: usize,
        output_stride: usize,
        route_input: bool,
    ) -> usize {
        let mut loads = 0;
        for group in 0..map.groups() {
            let length = map.words[LENGTHS + group] as usize;
            let expert = map.protected[group].0 as usize;
            for member_first in (0..length).step_by(tile) {
                for feature in 0..rows {
                    let mut sums = [[0f32; LANES]; MAX_BATCH];
                    for lane in 0..LANES {
                        for column in (lane..columns).step_by(LANES) {
                            let weight = decoded[(expert * rows + feature) * columns + column];
                            loads += 1;
                            for (member, lanes) in sums[..tile].iter_mut().enumerate() {
                                if member_first + member < length {
                                    let route = map.words
                                        [MEMBERS + group * MAX_BATCH + member_first + member]
                                        as usize;
                                    let source = if route_input { route } else { route / RANKS };
                                    lanes[lane] += weight
                                        * input[input_first + source * input_stride + column];
                                }
                            }
                        }
                    }
                    for stride in [32, 16, 8, 4, 2, 1] {
                        for lanes in &mut sums[..tile] {
                            for lane in 0..stride {
                                lanes[lane] += lanes[lane + stride];
                            }
                        }
                    }
                    for (member, lanes) in sums[..tile].iter().enumerate() {
                        if member_first + member < length {
                            let route = map.words
                                [MEMBERS + group * MAX_BATCH + member_first + member]
                                as usize;
                            output[output_first + route * output_stride + feature] = lanes[0];
                        }
                    }
                }
            }
        }
        loads
    }

    #[test]
    fn frozen_partition_boundaries_preserve_rank_bits_after_full_stale_reset() {
        use kuno_text::TensorType;
        let projections = [
            Projection::new(IQ3_XXS, 2560, 640, 512).unwrap(),
            Projection::new(IQ3_XXS, 2560, 640, 512).unwrap(),
            Projection::new(IQ4_NL, 640, 2560, 512).unwrap(),
        ];
        let geometry = Geometry::new(projections, 8, 144)
            .unwrap()
            .with_frozen_host_sources()
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
        let virtual_slots = [0u32, 71, 72, 143, 144, 287, 288, 431, 432, 511];
        let protected: Vec<_> = virtual_slots
            .iter()
            .enumerate()
            .map(|(expert, &slot)| (expert as u32, slot))
            .collect();
        let (rows, columns, first, stride) = (7, 256, 5, 13);
        for dtype in [IQ3_XXS, IQ4_NL, IQ4_XS] {
            let packed = packed_fixture(dtype, 10 * rows, columns, 3);
            let mut decoded = vec![0f32; 10 * rows * columns];
            TensorType::from_gguf_id(dtype)
                .unwrap()
                .decode(&packed, &mut decoded)
                .unwrap();
            let mut actual = vec![f32::NAN; first + MAX_ROUTES * stride + 11];
            let mut writes = [0u8; MAX_ROUTES];
            // A smaller second chunk must not inherit prior route/tail rows.
            for (epoch, count) in [3usize, 1, 2, 1].into_iter().enumerate() {
                actual.fill(0.); // existing complete owner reset, ONCE/layer
                writes.fill(0);
                let experts: Vec<_> = (0..count)
                    .flat_map(|t| (0..10).map(move |r| ((r + t + epoch) % 10) as u32))
                    .collect();
                let probabilities = vec![0.1; count * 10];
                let map = GroupMap::new_frozen_sources(
                    count,
                    &experts,
                    &probabilities,
                    &protected,
                    geometry,
                )
                .unwrap();
                let input: Vec<_> = (0..MAX_BATCH * columns)
                    .map(|i| ((i * 7 + epoch * 11) % 31) as f32 / 32. - 0.5)
                    .collect();
                let mut reference = vec![0.; actual.len()];
                grouped_projection(
                    &map,
                    8,
                    rows,
                    columns,
                    &decoded,
                    &input,
                    0,
                    columns,
                    &mut reference,
                    first,
                    stride,
                    false,
                );
                // Independent source membership uses the sealed partition
                // intervals, not a replica of group_active's shader predicates.
                for (lower, upper) in [(0u32, 72u32), (72, 144), (144, 288), (288, 432), (432, 512)]
                {
                    let prior = actual.clone();
                    for group in 0..30 {
                        if group >= map.groups() {
                            continue;
                        }
                        let slot = map.words[SLOTS + group];
                        if !(lower..upper).contains(&slot) {
                            continue;
                        }
                        let expert = map.protected[group].0 as usize;
                        for member in 0..map.words[LENGTHS + group] as usize {
                            let route = map.words[MEMBERS + group * MAX_BATCH + member] as usize;
                            writes[route] += 1;
                            for row in 0..rows {
                                let weight = &decoded[(expert * rows + row) * columns..][..columns];
                                let x = &input[route / RANKS * columns..][..columns];
                                actual[first + route * stride + row] = tree_dot(weight, x);
                            }
                        }
                    }
                    for route in 0..count * RANKS {
                        let slot = virtual_slots[experts[route] as usize];
                        if !(lower..upper).contains(&slot) {
                            assert_eq!(
                                actual[first + route * stride..][..rows]
                                    .iter()
                                    .map(|x| x.to_bits())
                                    .collect::<Vec<_>>(),
                                prior[first + route * stride..][..rows]
                                    .iter()
                                    .map(|x| x.to_bits())
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                for (route, &written) in writes.iter().enumerate() {
                    assert_eq!(written, u8::from(route < count * RANKS));
                }
                // Leave finite stale rows throughout the complete allocation;
                // the next chunk's once-only reset must erase even inactive
                // MAX8 rows, prefix/gaps and the full suffix before any shard.
                actual.fill(17.25);
                writes.fill(9);
            }
        }
    }

    #[test]
    fn original_q4_gate_and_both_down_formats_have_independent_numeric_and_guard_oracles() {
        use kuno_text::TensorType;
        let (hidden, ff) = (256, 32);
        let sentinel = -12345f32;
        for down in [Q5_0, Q8_0] {
            let mut decoded = Vec::new();
            for (phase, dtype, rows, columns) in [
                (0, Q4_K, ff, hidden),
                (1, Q4_K, ff, hidden),
                (2, down, hidden, ff),
            ] {
                let packed = packed_fixture(dtype, 10 * rows, columns, phase);
                let mut values = vec![0f32; 10 * rows * columns];
                TensorType::from_gguf_id(dtype)
                    .unwrap()
                    .decode(&packed, &mut values)
                    .unwrap();
                decoded.push(values);
            }
            for count in [1, 3, 5, 8] {
                let (experts, probabilities) = selections(count, false);
                let union = protected(&experts, 0, count);
                let map =
                    GroupMap::new(count, 0, count, &experts, &probabilities, &union, 16).unwrap();
                let (xf, xs, af, as_, pf, ps, of, os) =
                    (7, hidden + 3, 11, ff + 5, 13, hidden + 7, 17, hidden + 9);
                let mut input = vec![sentinel; xf + MAX_BATCH * xs + 5];
                for token in 0..count {
                    for column in 0..hidden {
                        input[xf + token * xs + column] =
                            ((column * 3 + token * 7) % 19) as f32 / 128. - 9. / 128.;
                    }
                }
                let mut scalar_act = vec![sentinel; af + MAX_ROUTES * as_ + 3];
                let mut scalar_parts = vec![sentinel; pf + MAX_ROUTES * ps + 3];
                for (route, &expert) in experts.iter().enumerate() {
                    let expert = expert as usize;
                    let x = &input[xf + route / RANKS * xs..][..hidden];
                    for feature in 0..ff {
                        let gate = &decoded[0][(expert * ff + feature) * hidden..][..hidden];
                        let up = &decoded[1][(expert * ff + feature) * hidden..][..hidden];
                        let gd = tree_dot(gate, x);
                        let ud = tree_dot(up, x);
                        // Independent f64 projection expectation also catches
                        // shared schedule errors in the exact64-tree oracle.
                        for (weights, value) in [(gate, gd), (up, ud)] {
                            let expected = weights
                                .iter()
                                .zip(x)
                                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                                .sum::<f64>();
                            let magnitude = weights
                                .iter()
                                .zip(x)
                                .map(|(&w, &x)| (f64::from(w) * f64::from(x)).abs())
                                .sum::<f64>();
                            assert!(
                                (f64::from(value) - expected).abs()
                                    <= 1e-6 + magnitude * f64::from(f32::EPSILON) * 16.
                            );
                        }
                        scalar_act[af + route * as_ + feature] = core_silu(gd) * ud;
                    }
                    let act = &scalar_act[af + route * as_..][..ff];
                    for feature in 0..hidden {
                        let weights = &decoded[2][(expert * hidden + feature) * ff..][..ff];
                        scalar_parts[pf + route * ps + feature] = tree_dot(weights, act);
                    }
                }
                for tile in [4, 8] {
                    let mut gates = vec![sentinel; scalar_act.len()];
                    let mut ups = vec![sentinel; scalar_act.len()];
                    let mut acts = vec![sentinel; scalar_act.len()];
                    let mut parts = vec![sentinel; scalar_parts.len()];
                    let gate_loads = grouped_projection(
                        &map,
                        tile,
                        ff,
                        hidden,
                        &decoded[0],
                        &input,
                        xf,
                        xs,
                        &mut gates,
                        af,
                        as_,
                        false,
                    );
                    let up_loads = grouped_projection(
                        &map,
                        tile,
                        ff,
                        hidden,
                        &decoded[1],
                        &input,
                        xf,
                        xs,
                        &mut ups,
                        af,
                        as_,
                        false,
                    );
                    for route in 0..count * RANKS {
                        for feature in 0..ff {
                            let i = af + route * as_ + feature;
                            acts[i] = core_silu(gates[i]) * ups[i];
                        }
                    }
                    let down_loads = grouped_projection(
                        &map,
                        tile,
                        hidden,
                        ff,
                        &decoded[2],
                        &acts,
                        af,
                        as_,
                        &mut parts,
                        pf,
                        ps,
                        true,
                    );
                    assert_eq!(gate_loads, 10 * ff * hidden * count.div_ceil(tile));
                    assert_eq!(gate_loads, up_loads);
                    assert_eq!(down_loads, gate_loads);
                    assert_eq!(
                        acts.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        scalar_act.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        parts.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        scalar_parts.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    let mut out = vec![sentinel; of + MAX_BATCH * os + 3];
                    let mut expected_out = out.clone();
                    for token in 0..count {
                        for feature in 0..hidden {
                            let mut actual = 0f32;
                            let mut expected = 0f32;
                            for rank in 0..RANKS {
                                let route = token * RANKS + rank;
                                actual +=
                                    map.probabilities[route] * parts[pf + route * ps + feature];
                                expected +=
                                    probabilities[route] * scalar_parts[pf + route * ps + feature];
                            }
                            out[of + token * os + feature] = actual;
                            expected_out[of + token * os + feature] = expected;
                        }
                    }
                    assert_eq!(
                        out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected_out.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[test]
    fn private_fields_are_runtime_not_compilation_arguments_or_geometry_keys() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        // Pin the actual vendored LaunchArg contract. Runtime scalar values
        // have the unit compilation argument; array metadata carries only a
        // fixed alias mode, never lengths, maps, handles, IDs or probabilities.
        let _: <usize as LaunchArg>::CompilationArg = ();
        let _: <u32 as LaunchArg>::CompilationArg = ();
        let _: <f32 as LaunchArg>::CompilationArg = ();
        let g = geometry(F32, 65, 7, 8, 16);
        let hash = |g: Geometry| {
            let mut h = DefaultHasher::new();
            g.hash(&mut h);
            h.finish()
        };
        let expected = hash(g);
        for count in [1, 3, 5, 8] {
            let (e, p) = selections(count, false);
            let union = protected(&e, 0, count);
            let map = GroupMap::new(count, 0, count, &e, &p, &union, 16).unwrap();
            let spans = Spans::new([
                count,
                count * 3,
                68,
                count * 5,
                11,
                count * 7,
                74,
                count * 11,
                count * 13,
                79,
                count * 17,
            ]);
            assert_eq!(hash(g), expected);
            assert!(matches!(
                g.grid(map.groups(), false).unwrap(),
                CubeCount::Static(7, 10, 1)
            ));
            assert_eq!(spans.map_first, count);
        }
        // Compiler annotations are a secondary guard alongside real offline
        // IR/runtime scalar checks and the independent numeric tests above.
        let source = include_str!("grouped_moe.rs");
        for (name, public_args) in [
            ("grouped_gate_up", 8),
            ("grouped_gate_up_live", 8),
            ("grouped_gate_up_frozen", 8),
            ("grouped_down", 5),
            ("grouped_down_live", 5),
            ("grouped_down_frozen", 5),
            ("grouped_reduce", 1),
            ("grouped_finite", 2),
        ] {
            let signature = source
                .split(&format!("pub fn {name}("))
                .nth(1)
                .unwrap()
                .split(") {")
                .next()
                .unwrap();
            assert_eq!(signature.matches("#[comptime]").count(), public_args);
            let runtime = signature.split("#[comptime]").next().unwrap();
            assert!(runtime.contains("map_first: usize"));
            assert!(runtime.contains("status_first: usize"));
        }
    }
    #[test]
    fn all_ten_original_projection_formats_reuse_decoded_values_with_independent_expectations() {
        use kuno_text::TensorType;
        let rows = 4;
        for dtype in [
            F32, F16, BF16, Q4_K, Q5_0, Q6_K, Q8_0, IQ3_XXS, IQ4_NL, IQ4_XS,
        ] {
            let block = super::super::block_layout(dtype).unwrap().0;
            let columns = if block == 1 { 138 } else { block * 3 };
            let original = Projection::new(dtype, columns, rows, 512).unwrap();
            original.segment_words(16).unwrap();
            let packed = packed_fixture(dtype, 10 * rows, columns, 3);
            let mut decoded = vec![0f32; 10 * rows * columns];
            TensorType::from_gguf_id(dtype)
                .unwrap()
                .decode(&packed, &mut decoded)
                .unwrap();
            for count in [1, 3, 5, 8] {
                let (e, p) = selections(count, false);
                let union = protected(&e, 0, count);
                let map = GroupMap::new(count, 0, count, &e, &p, &union, 16).unwrap();
                let (first, stride, out_first, out_stride) = (5, columns + 11, 7, rows + 5);
                let mut x = vec![12345f32; first + MAX_BATCH * stride + 13];
                for token in 0..count {
                    for column in 0..columns {
                        x[first + token * stride + column] =
                            ((column * 3 + token * 7) % 9) as f32 / 8. - 0.5;
                    }
                }
                for tile in [4, 8] {
                    let mut out = vec![-12345f32; out_first + MAX_ROUTES * out_stride + 13];
                    let loads = grouped_projection(
                        &map, tile, rows, columns, &decoded, &x, first, stride, &mut out,
                        out_first, out_stride, false,
                    );
                    assert_eq!(loads, 10 * rows * columns * count.div_ceil(tile));
                    for (index, &actual) in out.iter().enumerate() {
                        let route = index.saturating_sub(out_first) / out_stride;
                        let row = index.saturating_sub(out_first) % out_stride;
                        if index < out_first || route >= count * RANKS || row >= rows {
                            assert_eq!(actual, -12345f32);
                            continue;
                        }
                        let weights =
                            &decoded[(e[route] as usize * rows + row) * columns..][..columns];
                        let input = &x[first + route / RANKS * stride..][..columns];
                        assert_eq!(actual.to_bits(), tree_dot(weights, input).to_bits());
                        let expected = weights
                            .iter()
                            .zip(input)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum::<f64>();
                        let magnitude = weights
                            .iter()
                            .zip(input)
                            .map(|(&w, &x)| (f64::from(w) * f64::from(x)).abs())
                            .sum::<f64>();
                        assert!(
                            (f64::from(actual) - expected).abs()
                                <= 1e-6 + magnitude * f64::from(f32::EPSILON) * 16.
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn probability_and_coverage_scratch_guard_wipes_error_success_and_unwind_paths() {
        let before = SCRATCH_WIPES.with(|n| n.get());
        let (experts, p) = selections(3, false);
        let union = protected(&experts, 0, 3);
        let mut bad = p.clone();
        bad[0] = f32::NAN;
        assert!(GroupMap::new(3, 0, 3, &experts, &bad, &union, 16).is_err());
        assert_eq!(SCRATCH_WIPES.with(|n| n.get()), before + 1);
        let map = GroupMap::new(3, 0, 3, &experts, &p, &union, 16).unwrap();
        assert_eq!(SCRATCH_WIPES.with(|n| n.get()), before + 3);
        map.validate(&experts, &p, 16).unwrap();
        assert_eq!(SCRATCH_WIPES.with(|n| n.get()), before + 4);
        let failed = std::panic::catch_unwind(|| {
            let mut scratch = zeroize::Zeroizing::new(RouteScratch::empty());
            scratch.probability_sum = 0.75;
            scratch.covered.fill(0x55);
            panic!("fixed-public guarded scratch unwind fixture");
        });
        assert!(failed.is_err());
        assert_eq!(SCRATCH_WIPES.with(|n| n.get()), before + 5);
    }
    include!("iq4_down_scale_tests.rs");
    include!("iq_gate_up_factor_tests.rs");
    include!("down_four_row_tests.rs");
    include!("down_stride_four_tests.rs");
    include!("gate_up_two_row_tests.rs");
    include!("three_member_arithmetic_tests.rs");
}
#[cfg(test)]
mod frozen_source_tests {
    use super::*;
    fn geometry(layer: usize) -> Geometry {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
        let p = [
            Projection::new(gate, 2560, 640, 512).unwrap(),
            Projection::new(up, 2560, 640, 512).unwrap(),
            Projection::new(down, 640, 2560, 512).unwrap(),
        ];
        Geometry::new(p, 8, 144).unwrap()
    }
    #[test]
    fn private_table_policy_requires_frozen_exit_and_preserves_owners() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let hash = |g: Geometry| {
            let mut h = DefaultHasher::new();
            g.hash(&mut h);
            h.finish()
        };
        for layer in 0..48 {
            let ordinary = geometry(layer);
            assert!(ordinary.with_private_tables().is_err());
            let frozen = ordinary.with_frozen_host_sources().unwrap();
            assert!(frozen.with_private_tables().is_err());
            let old = frozen.with_inactive_partition_exit().unwrap();
            let new = old.with_private_tables().unwrap();
            assert!(!old.private_tables());
            assert!(new.private_tables());
            assert!(new.with_private_tables().is_err());
            assert!(new.with_live_member_reduction().is_err());
            assert_ne!(hash(old), hash(new));
            assert_eq!(old.projections(), new.projections());
            assert_eq!(old.member_capacity(), new.member_capacity());
            assert_eq!(old.status_words(), new.status_words());
            for io in [false, true] {
                assert_eq!(old.scratch_bytes(io), new.scratch_bytes(io));
            }
            for groups in [1, 10, 20, 30] {
                for down in [false, true] {
                    assert_eq!(
                        format!("{:?}", old.grid(groups, down).unwrap()),
                        format!("{:?}", new.grid(groups, down).unwrap())
                    );
                }
            }
        }
    }
    #[test]
    fn workgroup_table_policy_requires_frozen_exit_and_preserves_owners() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let hash = |g: Geometry| {
            let mut h = DefaultHasher::new();
            g.hash(&mut h);
            h.finish()
        };
        for layer in 0..48 {
            let ordinary = geometry(layer);
            assert!(ordinary.with_workgroup_tables().is_err());
            let frozen = ordinary.with_frozen_host_sources().unwrap();
            assert!(frozen.with_workgroup_tables().is_err());
            let old = frozen.with_inactive_partition_exit().unwrap();
            let new = old.with_workgroup_tables().unwrap();
            assert!(!old.workgroup_tables());
            assert!(new.workgroup_tables());
            assert!(new.with_workgroup_tables().is_err());
            assert!(new.with_private_tables().is_err());
            assert!(
                old.with_private_tables()
                    .unwrap()
                    .with_workgroup_tables()
                    .is_err()
            );
            assert!(new.with_live_member_reduction().is_err());
            assert_ne!(hash(old), hash(new));
            assert_eq!(old.projections(), new.projections());
            assert_eq!(old.member_capacity(), new.member_capacity());
            assert_eq!(old.status_words(), new.status_words());
            for io in [false, true] {
                assert_eq!(old.scratch_bytes(io), new.scratch_bytes(io));
            }
            for groups in [1, 10, 20, 30] {
                for down in [false, true] {
                    assert_eq!(
                        format!("{:?}", old.grid(groups, down).unwrap()),
                        format!("{:?}", new.grid(groups, down).unwrap())
                    );
                }
            }
        }
    }
    #[test]
    fn virtual_five_page_geometry_is_separate_from_ordinary_bank_capacity() {
        for layer in 0..48 {
            let ordinary = geometry(layer);
            assert!(!ordinary.frozen_host_sources());
            assert!(Geometry::new(ordinary.projections(), 8, 512).is_err());
            let frozen = ordinary.with_frozen_host_sources().unwrap();
            assert_eq!(frozen.capacity(), 512);
            let mut start = 0;
            for slots in crate::daslab_profile::SOURCE_PARTS {
                let words = frozen
                    .projections()
                    .map(|p| p.segment_words(slots).unwrap());
                frozen.validate_shard(start, slots, words).unwrap();
                assert!(frozen.validate_shard(start + 1, slots, words).is_err());
                let mut bad = words;
                bad[1] += 1;
                assert!(frozen.validate_shard(start, slots, bad).is_err());
                start += slots;
            }
            assert_eq!(start, 512);
            for count in 1..=3 {
                frozen.validate_count(count).unwrap();
            }
            assert!(frozen.validate_count(4).is_err());
            assert!(frozen.with_frozen_host_sources().is_err());
        }
        let p = [
            Projection::new(Q4_K, 2560, 640, 512).unwrap(),
            Projection::new(Q4_K, 2560, 640, 512).unwrap(),
            Projection::new(crate::quant_matvec::Q8_0, 640, 2560, 512).unwrap(),
        ];
        assert!(
            Geometry::new(p, 8, 144)
                .unwrap()
                .with_frozen_host_sources()
                .is_err()
        );
        assert!(
            Geometry::new(geometry(0).projections(), 4, 144)
                .unwrap()
                .with_frozen_host_sources()
                .is_err()
        );
    }
    #[test]
    fn inactive_partition_policy_is_frozen_only_and_preserves_exact_owner_geometry() {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let hash = |g: Geometry| {
            let mut h = DefaultHasher::new();
            g.hash(&mut h);
            h.finish()
        };
        let axes = |grid: CubeCount| match grid {
            CubeCount::Static(x, y, z) => (x, y, z),
            _ => panic!("expected original static grid"),
        };
        for layer in 0..48 {
            let ordinary = geometry(layer);
            assert!(ordinary.with_inactive_partition_exit().is_err());
            let old = ordinary.with_frozen_host_sources().unwrap();
            let new = old.with_inactive_partition_exit().unwrap();
            assert!(!ordinary.inactive_partition_exit());
            assert!(!old.inactive_partition_exit());
            assert!(new.inactive_partition_exit());
            assert!(new.with_inactive_partition_exit().is_err());
            assert!(new.with_live_member_reduction().is_err());
            assert!(
                old.with_live_member_reduction()
                    .unwrap()
                    .with_inactive_partition_exit()
                    .is_err()
            );
            assert_ne!(hash(old), hash(new));
            assert_eq!(old.projections(), new.projections());
            assert_eq!(old.member_capacity(), new.member_capacity());
            assert_eq!(old.status_words(), new.status_words());
            for io in [false, true] {
                assert_eq!(old.scratch_bytes(io), new.scratch_bytes(io));
            }
            for groups in [1, 10, 20, 30] {
                for down in [false, true] {
                    assert_eq!(
                        axes(old.grid(groups, down).unwrap()),
                        axes(new.grid(groups, down).unwrap())
                    );
                }
            }
            for count in 1..=3 {
                assert_eq!(
                    axes(old.finite_grid_for_count(count).unwrap()),
                    axes(new.finite_grid_for_count(count).unwrap())
                );
                assert_eq!(
                    axes(old.reduce_grid_for_count(count).unwrap()),
                    axes(new.reduce_grid_for_count(count).unwrap())
                );
            }
        }
    }
    #[test]
    fn virtual_source_map_preserves_actual_ranks_probability_bits_and_boundaries() {
        let g = geometry(0).with_frozen_host_sources().unwrap();
        let slots = [0u32, 71, 72, 143, 144, 287, 288, 431, 432, 511];
        let pairs: Vec<_> = slots
            .into_iter()
            .enumerate()
            .map(|(e, s)| (e as u32, s))
            .collect();
        for count in 1..=3 {
            let experts: Vec<_> = (0..count)
                .flat_map(|t| (0..10).map(move |r| ((r + t) % 10) as u32))
                .collect();
            let probabilities = vec![0.1f32; count * 10];
            assert!(GroupMap::new(count, 0, count, &experts, &probabilities, &pairs, 512).is_err());
            let map =
                GroupMap::new_frozen_sources(count, &experts, &probabilities, &pairs, g).unwrap();
            map.validate_geometry(g).unwrap();
            map.validate(&experts, &probabilities, 512).unwrap();
            assert!(map.validate_geometry(geometry(0)).is_err());
            assert!(map.validate(&experts, &probabilities, 144).is_err());
            for (group, &(_, slot)) in pairs.iter().enumerate() {
                assert_eq!(map.words()[SLOTS + group], slot);
                assert_eq!(map.words()[LENGTHS + group], count as u32);
                for t in 0..count {
                    let rank = (group + 10 - t) % 10;
                    assert_eq!(map.words()[MEMBERS + group * 8 + t], (t * 10 + rank) as u32);
                }
            }
            for (r, &p) in probabilities.iter().enumerate() {
                assert_eq!(map.probabilities()[r].to_bits(), p.to_bits());
            }
        }
        let experts: Vec<_> = (0..10).collect();
        let p = vec![0.1; 10];
        assert!(GroupMap::new_frozen_sources(1, &experts, &p, &pairs, geometry(0)).is_err());
        let mut bad = pairs.clone();
        bad[9].1 = 512;
        assert!(GroupMap::new_frozen_sources(1, &experts, &p, &bad, g).is_err());
        bad = pairs.clone();
        bad[9].1 = bad[0].1;
        assert!(GroupMap::new_frozen_sources(1, &experts, &p, &bad, g).is_err());
        assert!(GroupMap::new_frozen_sources(1, &experts, &p, &pairs[..9], g).is_err());
    }
    #[test]
    fn inactive_exit_does_not_admit_malformed_original_map_or_ready_status() {
        let old = geometry(0).with_frozen_host_sources().unwrap();
        let new = old.with_inactive_partition_exit().unwrap();
        let experts: Vec<_> = (0..30).map(|r| (r % 10) as u32).collect();
        let probabilities = vec![0.1; 30];
        let pairs: Vec<_> = [0u32, 71, 72, 143, 144, 287, 288, 431, 432, 511]
            .into_iter()
            .enumerate()
            .map(|(expert, slot)| (expert as u32, slot))
            .collect();
        for policy in [old, new] {
            for (index, bad) in [
                (0, 0),
                (3, 81),
                (SLOTS, 512),
                (LENGTHS, 0),
                (LENGTHS, 4),
                (MEMBERS, 30),
            ] {
                let mut map =
                    GroupMap::new_frozen_sources(3, &experts, &probabilities, &pairs, policy)
                        .unwrap();
                map.words[index] = bad;
                assert!(map.validate(&experts, &probabilities, 512).is_err());
            }
            let status = vec![0u32; policy.status_words()];
            assert!(validate_status(policy, 3, 0, &status).is_err());
        }
    }
}

#[cfg(test)]
#[path = "grouped_moe_four_member_tests.rs"]
mod four_member_tests;
