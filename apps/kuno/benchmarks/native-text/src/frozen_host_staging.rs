//! Exact transient copies of selected original frozen HOST expert words.
//!
//! This module owns no allocation, source identity, readiness or route fence.
//! The parent reserves/proves/erases three fixed PrivateArena owners and seals
//! router -> copies -> consumers -> next overwrite on the same client/stream.
//! Original source pages/pins and deferred MAP proofs remain authoritative.

#![allow(dead_code)] // Source-only leaf; the parent wires guarded dispatch.

use super::{GatePolicy, Geometry, MAP_WORDS, SLOTS, group_active};
use cubecl::prelude::*;

pub(crate) const OWNER_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const OWNER_WORDS: usize = OWNER_BYTES / 4;
pub(crate) const STAGE_GROUPS: usize = 30;
pub(crate) const COPY_LANES: usize = 256;
pub(crate) const COPY_WORDS_PER_CUBE: usize = 4 * COPY_LANES;

/// Public packed strides only; never contains a selected slot or route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    gate_up_words: usize,
    down_words: usize,
}
impl Shape {
    pub(crate) fn new(g: Geometry) -> Result<Self, &'static str> {
        if !g.frozen_host_sources()
            || !g.inactive_partition_exit()
            || g.capacity() != 512
            || g.hidden() != 2560
            || g.ff() != 640
            || g.tile() != 8
            || g.secondary_partition
            || g.gate_policy() != GatePolicy::Generic
            || g.live_member_reduction()
            || g.private_tables()
            || g.workgroup_tables()
            || g.shared_gate_up_factors()
            || g.shared_down_scales()
        {
            return Err("frozen host stage policy");
        }
        let [gate, up, down] = g.projections();
        if gate != up
            || !gate.expert_bytes().is_multiple_of(4)
            || !down.expert_bytes().is_multiple_of(4)
        {
            return Err("frozen host stage packed stride");
        }
        let shape = Self {
            gate_up_words: gate.expert_bytes() / 4,
            down_words: down.expert_bytes() / 4,
        };
        for words in shape.payload_words() {
            if words == 0 || words > OWNER_WORDS {
                return Err("frozen host stage owner extent");
            }
        }
        Ok(shape)
    }
    pub(crate) fn gate_up_words(self) -> usize {
        self.gate_up_words
    }
    pub(crate) fn down_words(self) -> usize {
        self.down_words
    }
    /// Owning MAIN extents, including every inactive group and physical tail.
    pub(crate) fn stage_words(self) -> [usize; 3] {
        [OWNER_WORDS; 3]
    }
    /// Logical packed payload capacity, distinct from physical owner charges.
    pub(crate) fn payload_words(self) -> [usize; 3] {
        [
            STAGE_GROUPS * self.gate_up_words,
            STAGE_GROUPS * self.gate_up_words,
            STAGE_GROUPS * self.down_words,
        ]
    }
    pub(crate) fn copy_grid(self, down: bool) -> CubeCount {
        let words = if down {
            self.down_words
        } else {
            self.gate_up_words
        };
        CubeCount::Static(
            words.div_ceil(COPY_WORDS_PER_CUBE) as u32,
            STAGE_GROUPS as u32,
            1,
        )
    }
}

/// Runtime guard keeps the ORIGINAL shard/slot/member proof before using a
/// group-position scratch address. Frozen three-token full waves only.
#[cube]
#[allow(clippy::collapsible_if, clippy::manual_range_contains)]
pub(super) fn host_group_active(
    map: &Array<u32>,
    first: usize,
    group: usize,
    first_slot: usize,
    slot_count: usize,
) -> bool {
    let mut active = false;
    if ((first_slot == 144 || first_slot == 288) && slot_count == 144)
        || (first_slot == 432 && slot_count == 80)
    {
        if first <= map.len() {
            if MAP_WORDS <= map.len() - first {
                let count = map[first] as usize;
                let groups = map[first + 3] as usize;
                if count >= 1
                    && count <= 3
                    && map[first + 1] == 0
                    && map[first + 2] as usize == count
                    && groups >= 1
                    && groups <= STAGE_GROUPS
                    && group < STAGE_GROUPS
                {
                    active = group_active(map, first, group, first_slot, slot_count, 8_usize);
                }
            }
        }
    }
    active
}

#[cube]
pub(super) fn stage_extent_valid(words: usize, #[comptime] expert_words: usize) -> bool {
    comptime!(expert_words > 0 && expert_words <= OWNER_WORDS / STAGE_GROUPS)
        && words >= comptime!(STAGE_GROUPS * expert_words)
        && words <= OWNER_WORDS
}

/// Called only inside the corresponding original active shard/member guard.
#[cube]
pub(super) fn weight_slot(
    map: &Array<u32>,
    first: usize,
    group: usize,
    first_slot: usize,
    #[comptime] staged: bool,
) -> usize {
    if comptime!(staged) {
        group
    } else {
        map[first + SLOTS + group] as usize - first_slot
    }
}

/// Two exact packed roles, six storage bindings including the existing fault.
/// Each stripe accesses 256 consecutive words across the workgroup. Nothing is
/// decoded, transposed, regrouped, persisted or promoted to a source identity.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::len_zero)] // CubeCL Array has no is_empty.
pub(crate) fn stage_gate_up(
    gate: &Array<u32>,
    up: &Array<u32>,
    map: &Array<u32>,
    stage_gate: &mut Array<u32>,
    stage_up: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    map_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] expert_words: usize,
) {
    if comptime!(expert_words == 0 || expert_words > OWNER_WORDS / STAGE_GROUPS) {
        terminate!();
    }
    let group = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if !host_group_active(map, map_first, group, first_slot, slot_count) {
        terminate!();
    }
    if gate.len() != slot_count * expert_words
        || up.len() != slot_count * expert_words
        || !stage_extent_valid(stage_gate.len(), expert_words)
        || !stage_extent_valid(stage_up.len(), expert_words)
    {
        if lane == 0 && fault.len() > 0 {
            fault[0].fetch_or(4);
        }
        terminate!();
    }
    let source_slot = map[map_first + SLOTS + group] as usize - first_slot;
    #[unroll]
    for stripe in 0_usize..4 {
        let word = CUBE_POS_X as usize * COPY_WORDS_PER_CUBE + lane + stripe * COPY_LANES;
        if word < expert_words {
            let source = source_slot * expert_words + word;
            let destination = group * expert_words + word;
            stage_gate[destination] = gate[source];
            stage_up[destination] = up[source];
        }
    }
}

/// One original packed down role; the same exact group/shard/tail guards.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::len_zero)] // CubeCL Array has no is_empty.
pub(crate) fn stage_down(
    down: &Array<u32>,
    map: &Array<u32>,
    stage_down: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    map_first: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] expert_words: usize,
) {
    if comptime!(expert_words == 0 || expert_words > OWNER_WORDS / STAGE_GROUPS) {
        terminate!();
    }
    let group = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if !host_group_active(map, map_first, group, first_slot, slot_count) {
        terminate!();
    }
    if down.len() != slot_count * expert_words
        || !stage_extent_valid(stage_down.len(), expert_words)
    {
        if lane == 0 && fault.len() > 0 {
            fault[0].fetch_or(4);
        }
        terminate!();
    }
    let source_slot = map[map_first + SLOTS + group] as usize - first_slot;
    #[unroll]
    for stripe in 0_usize..4 {
        let word = CUBE_POS_X as usize * COPY_WORDS_PER_CUBE + lane + stripe * COPY_LANES;
        if word < expert_words {
            stage_down[group * expert_words + word] = down[source_slot * expert_words + word];
        }
    }
}

#[cfg(test)]
#[path = "frozen_host_staging_tests.rs"]
mod tests;
