//! Candidate grouped all-ten routed MoE operators, not a graph transaction.
//!
//! Original PUBLIC banks stay tightly packed and unchanged. Every selection,
//! slot, group/member map, count, activation address and probability is runtime
//! private data. Geometry/dtypes and tile4/8 alone may specialize pipelines.
//! A caller must pin BOTH its immutable token wave and actual pending BankSlots
//! protection through completed use; a borrowed raw slot slice is not that proof.
//! No bank fill, commit, graph hook, CPU fallback or device action lives here.
//! Parent owns physical owner/alias/epoch/device-limit validation and acceptance.

use super::packed_value;
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Geometry {
    projections: [Projection; 3],
    hidden: usize,
    ff: usize,
    tile: usize,
    capacity: usize,
}
impl Geometry {
    pub fn new(
        projections: [Projection; 3],
        tile: usize,
        capacity: usize,
    ) -> Result<Self, &'static str> {
        let [gate, up, down] = projections;
        if !matches!(tile, 4 | 8)
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
        // Existing96 contract restricts original format/shape. Do not expand it.
        if capacity == 96
            && (gate.dtype != super::Q4_K
                || up.dtype != super::Q4_K
                || !matches!(down.dtype, super::Q5_0 | super::Q8_0)
                || gate.columns != 2560
                || gate.rows != 640)
        {
            return Err("grouped segmented original metadata");
        }
        for projection in projections {
            for count in if capacity == 96 {
                [64, 32]
            } else {
                [capacity, 0]
            } {
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
        })
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
        let valid = if self.capacity == 96 {
            (first_slot == 0 && slot_count == 64) || (first_slot == 64 && slot_count == 32)
        } else {
            first_slot == 0 && slot_count == self.capacity
        };
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
            if down { self.hidden } else { self.ff } as u32,
            groups as u32,
            (MAX_BATCH / self.tile) as u32,
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

/// Request-private fixed map. The borrowed protected union must be produced
/// from the caller's pinned actual pending bank plan and immutable token wave.
/// No Clone/Copy/Debug/Hash/serialization; full fixed capacities wipe on Drop.
pub struct GroupMap<'a> {
    words: Box<[u32; MAP_WORDS]>,
    probabilities: Box<[f32; MAX_ROUTES]>,
    protected: &'a [(u32, u32)],
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
        let mut map = Self {
            words: Box::new([0; MAP_WORDS]),
            probabilities: Box::new([0.; MAX_ROUTES]),
            protected,
        };
        if !(1..=MAX_BATCH).contains(&count)
            || wave_count == 0
            || wave_first
                .checked_add(wave_count)
                .is_none_or(|end| end > count)
            || experts.len() != count * RANKS
            || probabilities.len() != count * RANKS
            || !crate::bank_slots::routed_capacity(capacity)
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
            || !crate::bank_slots::routed_capacity(capacity)
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
    if !(1..=MAX_BATCH).contains(&count)
        || first
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
fn group_active(
    map: &Array<u32>,
    first: usize,
    group: usize,
    first_slot: usize,
    slot_count: usize,
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

#[cube]
fn grouped_silu(value: f32) -> f32 {
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    value * sigmoid
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
) {
    let feature = CUBE_POS_X as usize;
    let group = CUBE_POS_Y as usize;
    let first_member = CUBE_POS_Z as usize * tile;
    let lane = UNIT_POS_X as usize;
    let mut active = group_active(map, map_first, group, first_slot, slot_count) && feature < ff;
    if active {
        if first_member >= map[map_first + LENGTHS + group] as usize {
            active = false;
        }
    }
    let mut gates = Array::<f32>::new(tile);
    let mut ups = Array::<f32>::new(tile);
    #[unroll]
    for member in 0..tile {
        gates[member] = 0.;
        ups[member] = 0.;
    }
    if active {
        let slot = map[map_first + SLOTS + group] as usize - first_slot;
        let length = map[map_first + LENGTHS + group] as usize;
        let row = slot * ff + feature;
        let gate_offset = row * gate_row_bytes;
        let up_offset = row * up_row_bytes;
        for column_tile in 0..comptime!(hidden.div_ceil(LANES)) {
            let column = column_tile * LANES + lane;
            if column < hidden {
                // Decode each original weight only ONCE, reused across RHS.
                let g = packed_value(gate, gate_offset, column, gate_dtype);
                let u = packed_value(up, up_offset, column, up_dtype);
                #[unroll]
                for member in 0..tile {
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
    let mut gate_sums = SharedMemory::<f32>::new(LANES * tile);
    let mut up_sums = SharedMemory::<f32>::new(LANES * tile);
    #[unroll]
    for member in 0..tile {
        gate_sums[member * LANES + lane] = gates[member];
        up_sums[member * LANES + lane] = ups[member];
    }
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            #[unroll]
            for member in 0..tile {
                gate_sums[member * LANES + lane] += gate_sums[member * LANES + lane + stride];
                up_sums[member * LANES + lane] += up_sums[member * LANES + lane + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && active {
        let length = map[map_first + LENGTHS + group] as usize;
        #[unroll]
        for member in 0..tile {
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
    let feature = CUBE_POS_X as usize;
    let group = CUBE_POS_Y as usize;
    let first_member = CUBE_POS_Z as usize * tile;
    let lane = UNIT_POS_X as usize;
    let mut active =
        group_active(map, map_first, group, first_slot, slot_count) && feature < hidden;
    if active {
        if first_member >= map[map_first + LENGTHS + group] as usize {
            active = false;
        }
    }
    let mut partial = Array::<f32>::new(tile);
    #[unroll]
    for member in 0..tile {
        partial[member] = 0.;
    }
    if active {
        let slot = map[map_first + SLOTS + group] as usize - first_slot;
        let length = map[map_first + LENGTHS + group] as usize;
        let row_offset = (slot * hidden + feature) * down_row_bytes;
        for column_tile in 0..comptime!(ff.div_ceil(LANES)) {
            let column = column_tile * LANES + lane;
            if column < ff {
                let weight = packed_value(down, row_offset, column, down_dtype);
                #[unroll]
                for member in 0..tile {
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
    let mut sums = SharedMemory::<f32>::new(LANES * tile);
    #[unroll]
    for member in 0..tile {
        sums[member * LANES + lane] = partial[member];
    }
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            #[unroll]
            for member in 0..tile {
                sums[member * LANES + lane] += sums[member * LANES + lane + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && active {
        let length = map[map_first + LENGTHS + group] as usize;
        #[unroll]
        for member in 0..tile {
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
/// =(max(ceilD/64,ceilFF/64),MAX80,4); every barrier is unconditional. Ready
/// flags at0..159 are never overwritten by a finite sweep or later wave.
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
    use super::super::{BF16, F16, F32, Q4_K, Q5_0, Q6_K, Q8_0};
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
        };
        assert_eq!(g.scratch_bytes(true).unwrap(), 4480 * 1024);
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
            for tile in [4, 8] {
                let mut acts = vec![0f32; count * 10 * ff];
                let mut parts = vec![0f32; count * 10 * hidden];
                for (group, &(expert, _)) in union.iter().enumerate().take(map.groups()) {
                    let length = map.words[LENGTHS + group] as usize;
                    let expert = expert as usize;
                    for first in (0..8).step_by(tile) {
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
            for cap in [16, 64, 72, 96] {
                let g = Geometry::new(p, 8, cap).unwrap();
                let shards = if cap == 96 {
                    vec![(0, 64), (64, 32)]
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
                grouped_gate_up::expand(
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
                grouped_down::expand(
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
                        .kernel_name("grouped_moe_operator"),
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
    fn offline_shaders_all_original_decoders_reuse_weights_across_rhs() {
        use cubecl::server::ExecutionMode;
        for dtype in [F32, F16, BF16, Q4_K, Q5_0, Q6_K, Q8_0] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let a = geometry(dtype, 256, 256, 4, 16);
                let b = geometry(dtype, 256, 256, 8, 16);
                for kind in [0, 1] {
                    let four = shader(kind, a, mode);
                    let eight = shader(kind, b, mode);
                    validate_shader(&four, 8, true);
                    validate_shader(&eight, 8, true);
                    let weights = if kind == 0 { 2 } else { 1 };
                    for buffer in 0..weights {
                        let pattern = format!("buffer_{buffer}_global[");
                        assert!(four.matches(&pattern).count() > 0);
                        assert_eq!(
                            four.matches(&pattern).count(),
                            eight.matches(&pattern).count(),
                            "weight decoding must not multiply with tile"
                        );
                    }
                    let input_buffer = if kind == 0 { 2 } else { 1 };
                    let pattern = format!("buffer_{input_buffer}_global[");
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
            for tile in [4, 8] {
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

    fn packed_fixture(dtype: u32, rows: usize, columns: usize, phase: usize) -> Vec<u8> {
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
            ("grouped_gate_up", 7),
            ("grouped_down", 5),
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
    fn all_seven_original_projection_formats_reuse_decoded_values_with_independent_expectations() {
        use kuno_text::TensorType;
        let rows = 4;
        for dtype in [F32, F16, BF16, Q4_K, Q5_0, Q6_K, Q8_0] {
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
}
