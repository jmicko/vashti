//! Private count1 adapter; never grants cache/model publication or a fallback.
//! The coordinator proves actual misses/original bank pins, joins CPU workers,
//! uploads CPU before owner readiness, and retains all-family cleanup authority.
//! Original grouped gate/down kernels consume the subset map unchanged.
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::grouped_moe::{self as k, MAX_BATCH, MAX_ROUTES, RANKS, Spans},
    resident_util as u,
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::cpu_expert::Partials;
use zeroize::Zeroizing;

pub(super) const CPU_WORDS: usize = RANKS * 2560;
pub(super) const CPU_OWNED_WORDS: usize = 32768;
pub(super) const OWNER_WORDS: usize = 4 + MAX_ROUTES * 3;
pub(super) const OWNER_OWNED_WORDS: usize = 8192;
const SLOTS: usize = 4;
const LENGTHS: usize = 4 + MAX_ROUTES;
const MEMBERS: usize = 4 + MAX_ROUTES * 2;

fn rejected() -> &'static str {
    "native count1 hybrid map rejected"
}

/// Opaque guarded wire views. GPU indices are ORIGINAL ranks; CPU indices are
/// compact ordinals. Actual-miss/source publication proofs stay with Pending.
pub(super) struct Map {
    words: Zeroizing<[u32; k::MAP_WORDS]>,
    probabilities: Zeroizing<[f32; MAX_ROUTES]>,
    owners: Zeroizing<[u32; OWNER_WORDS]>,
    cpu_count: Zeroizing<usize>,
    gpu_count: Zeroizing<usize>,
}
impl Map {
    pub(super) fn new(
        ids: &[usize; RANKS],
        probabilities: &[f32; RANKS],
        split: &[u32; RANKS * 2],
        gpu_pairs: &[(u32, u32)],
        capacity: usize,
    ) -> Result<Self, &'static str> {
        if !crate::bank_slots::routed_capacity(capacity) || gpu_pairs.len() > RANKS {
            return Err(rejected());
        }
        let mut out = Self {
            words: Zeroizing::new([0; k::MAP_WORDS]),
            probabilities: Zeroizing::new([0.; MAX_ROUTES]),
            owners: Zeroizing::new([0; OWNER_WORDS]),
            cpu_count: Zeroizing::new(0),
            gpu_count: Zeroizing::new(gpu_pairs.len()),
        };
        let mut sum = Zeroizing::new(0f64);
        let mut seen = Zeroizing::new([false; RANKS]);
        for rank in 0..RANKS {
            if ids[rank] >= 512
                || ids[..rank].contains(&ids[rank])
                || !probabilities[rank].is_finite()
                || probabilities[rank] < 0.
            {
                return Err(rejected());
            }
            *sum += f64::from(probabilities[rank]);
            out.probabilities[rank] = probabilities[rank];
            let owner = split[rank * 2];
            let index = split[rank * 2 + 1] as usize;
            out.owners[4 + rank * 3] = owner;
            out.owners[5 + rank * 3] = split[rank * 2 + 1];
            if owner == 1 {
                if index != *out.cpu_count
                    || gpu_pairs.iter().any(|&(id, _)| id as usize == ids[rank])
                {
                    return Err(rejected());
                }
                *out.cpu_count += 1;
            } else if owner == 0 {
                if index != rank {
                    return Err(rejected());
                }
                let group = gpu_pairs
                    .iter()
                    .position(|&(id, _)| id as usize == ids[rank])
                    .ok_or_else(rejected)?;
                if seen[group] {
                    return Err(rejected());
                }
                seen[group] = true;
                out.words[LENGTHS + group] = 1;
                out.words[MEMBERS + group * MAX_BATCH] = rank as u32;
            } else {
                return Err(rejected());
            }
        }
        if (*sum - 1.).abs() > 1e-5
            || *out.cpu_count + *out.gpu_count != RANKS
            || seen[..gpu_pairs.len()].iter().any(|v| !*v)
        {
            return Err(rejected());
        }
        for (group, &(id, slot)) in gpu_pairs.iter().enumerate() {
            if id >= 512
                || slot as usize >= capacity
                || gpu_pairs[..group]
                    .iter()
                    .any(|&(e, s)| e == id || s == slot)
            {
                return Err(rejected());
            }
            out.words[SLOTS + group] = slot;
        }
        out.words[..4].copy_from_slice(&[1, 0, 1, *out.gpu_count as u32]);
        out.owners[..4].copy_from_slice(&[1, *out.cpu_count as u32, *out.gpu_count as u32, 0]);
        Ok(out)
    }
    pub(super) fn words(&self) -> &[u32; k::MAP_WORDS] {
        &self.words
    }
    pub(super) fn probabilities(&self) -> &[f32; MAX_ROUTES] {
        &self.probabilities
    }
    pub(super) fn owner_words(&self) -> &[u32; OWNER_WORDS] {
        &self.owners
    }
    pub(super) fn cpu_count(&self) -> usize {
        *self.cpu_count
    }
    pub(super) fn gpu_count(&self) -> usize {
        *self.gpu_count
    }
    /// All wires/counts are inline guarded arrays: there are no heap owners or
    /// spare capacities hidden behind this charge.
    pub(super) fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
    /// Only a core Partials owner can grant CPU readiness. Its private
    /// constructor follows scoped joins; no caller receives a readiness setter.
    /// Core owns and wipes its entire allocation. This checks its exposed
    /// logical ten-rank suffix; GPU fold separately inspects the uploaded full
    /// CPU owning page, including its additional allocator padding.
    pub(super) fn complete_cpu(
        &mut self,
        partials: Option<&Partials>,
        hidden: usize,
    ) -> Result<(), &'static str> {
        if hidden != 2560 {
            return Err(rejected());
        }
        if *self.cpu_count == 0 {
            return if partials.is_none() {
                Ok(())
            } else {
                Err(rejected())
            };
        }
        let partials = partials.ok_or_else(rejected)?;
        if partials.ranks() != *self.cpu_count
            || partials.hidden() != hidden
            || partials.full().len() != RANKS * hidden
            || partials.owned_bytes() == 0
            || partials.owned_bytes() > kuno_text::cpu_expert::MAX_PRIVATE_HOST_BYTES
            || partials.active().iter().any(|value| !value.is_finite())
            || partials.full()[*self.cpu_count * hidden..]
                .iter()
                .any(|value| value.to_bits() != 0)
            || (0..RANKS).any(|rank| self.owners[6 + rank * 3] != 0)
        {
            return Err(rejected());
        }
        for rank in 0..RANKS {
            if self.owners[4 + rank * 3] == 1 {
                self.owners[6 + rank * 3] = 1;
            }
        }
        Ok(())
    }
}

#[cube]
#[allow(clippy::collapsible_if)]
fn owner_header(owners: &Array<u32>, first: usize, count: usize) -> bool {
    let mut valid = false;
    if count == 1 && first <= owners.len() {
        if OWNER_WORDS <= owners.len() - first {
            let cpu = owners[first + 1] as usize;
            let gpu = owners[first + 2] as usize;
            if cpu <= RANKS {
                valid = owners[first] == 1 && owners[first + 3] == 0 && gpu == RANKS - cpu;
            }
        }
    }
    valid
}

/// Exactly one original rank-order multiply/add. No factoring, subset-order
/// sum, zero-probability skip or fake GPU readiness is permitted.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub(super) fn hybrid_merge(
    gpu: &Array<f32>,
    cpu: &Array<f32>,
    owners: &Array<u32>,
    probabilities: &Array<f32>,
    status: &Array<u32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    partial_first: usize,
    partial_stride: usize,
    cpu_first: usize,
    cpu_stride: usize,
    owner_first: usize,
    prob_first: usize,
    status_first: usize,
    output_first: usize,
    output_stride: usize,
    #[comptime] hidden: usize,
) {
    let feature = CUBE_POS_X as usize * k::LANES + UNIT_POS_X as usize;
    let token = CUBE_POS_Y as usize;
    if token == 0 && feature < hidden {
        let mut valid = owner_header(owners, owner_first, count);
        let mut sum = 0f32;
        if valid {
            let cpu_count = owners[owner_first + 1] as usize;
            for rank in 0..RANKS {
                let owner = owners[owner_first + 4 + rank * 3];
                let index = owners[owner_first + 5 + rank * 3] as usize;
                let ready = owners[owner_first + 6 + rank * 3];
                let mut value = 0f32;
                let mut accepted = false;
                if owner == 0 {
                    if index == rank
                        && ready == 0
                        && status[status_first + rank] == 1
                        && status[status_first + MAX_ROUTES + rank] == 1
                    {
                        value = gpu[partial_first + rank * partial_stride + feature];
                        accepted = true;
                    }
                } else if owner == 1 {
                    if index < cpu_count && ready == 1 {
                        value = cpu[cpu_first + index * cpu_stride + feature];
                        accepted = true;
                    }
                }
                let probability = probabilities[prob_first + rank];
                if !accepted
                    || !(value.abs() <= f32::MAX)
                    || !(probability.abs() <= f32::MAX)
                    || probability < 0.
                {
                    valid = false;
                }
                sum += probability * value;
            }
        }
        if !(sum.abs() <= f32::MAX) {
            valid = false;
        }
        if valid {
            output[output_first + token * output_stride + feature] = sum;
        } else {
            output[output_first + token * output_stride + feature] = 0.;
            fault[0].fetch_or(4);
        }
    }
}

#[cube]
fn active_coordinate(i: usize, first: usize, stride: usize, count: usize, width: usize) -> bool {
    let mut active = false;
    if stride >= width && stride > 0 && i >= first {
        let relative = i - first;
        active = relative / stride < count && relative % stride < width;
    }
    active
}

/// Full API-visible views, including all inactive MAX8/MAX80 capacity, row
/// gaps and owning padding. It only ORs the existing whole-model fault bit4.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub(super) fn hybrid_fold(
    input: &Array<u32>,
    act: &Array<u32>,
    gpu: &Array<u32>,
    cpu: &Array<u32>,
    map: &Array<u32>,
    probabilities: &Array<u32>,
    status: &Array<u32>,
    owners: &Array<u32>,
    output: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    map_first: usize,
    input_first: usize,
    input_stride: usize,
    act_first: usize,
    act_stride: usize,
    partial_first: usize,
    partial_stride: usize,
    cpu_first: usize,
    cpu_stride: usize,
    prob_first: usize,
    status_first: usize,
    owner_first: usize,
    output_first: usize,
    output_stride: usize,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
    #[comptime] capacity: usize,
) {
    let i = ABSOLUTE_POS as usize;
    let mut valid = owner_header(owners, owner_first, count);
    let mut cpu_count = 0usize;
    let mut gpu_count = 0usize;
    if valid {
        cpu_count = owners[owner_first + 1] as usize;
        gpu_count = owners[owner_first + 2] as usize;
        // Global bijection checks run once, while every API word still has
        // its own parallel finite/zero inspection below.
        if i == 0 {
            let mut cpu_seen = 0usize;
            for rank in 0..RANKS {
                let owner = owners[owner_first + 4 + rank * 3];
                let index = owners[owner_first + 5 + rank * 3] as usize;
                let ready = owners[owner_first + 6 + rank * 3];
                if owner == 0 {
                    if index != rank || ready != 0 {
                        valid = false;
                    }
                } else if owner == 1 {
                    if index != cpu_seen || ready != 1 {
                        valid = false;
                    }
                    cpu_seen += 1;
                } else {
                    valid = false;
                }
            }
            if cpu_seen != cpu_count {
                valid = false;
            }
        }
    }
    if i < input.len() {
        if active_coordinate(i, input_first, input_stride, 1, hidden) {
            if !(f32::from_bits(input[i]).abs() <= f32::MAX) {
                valid = false;
            }
        } else if input[i] != 0 {
            valid = false;
        }
    }
    if i < output.len() {
        if active_coordinate(i, output_first, output_stride, 1, hidden) {
            if !(f32::from_bits(output[i]).abs() <= f32::MAX) {
                valid = false;
            }
        } else if output[i] != 0 {
            valid = false;
        }
    }
    if i < act.len() {
        let mut active = false;
        if active_coordinate(i, act_first, act_stride, RANKS, ff)
            && owner_header(owners, owner_first, count)
        {
            let rank = (i - act_first) / act_stride;
            active = owners[owner_first + 4 + rank * 3] == 0;
        }
        if active {
            if !(f32::from_bits(act[i]).abs() <= f32::MAX) {
                valid = false;
            }
        } else if act[i] != 0 {
            valid = false;
        }
    }
    if i < gpu.len() {
        let mut active = false;
        if active_coordinate(i, partial_first, partial_stride, RANKS, hidden)
            && owner_header(owners, owner_first, count)
        {
            let rank = (i - partial_first) / partial_stride;
            active = owners[owner_first + 4 + rank * 3] == 0;
        }
        if active {
            if !(f32::from_bits(gpu[i]).abs() <= f32::MAX) {
                valid = false;
            }
        } else if gpu[i] != 0 {
            valid = false;
        }
    }
    if i < cpu.len() {
        if active_coordinate(i, cpu_first, cpu_stride, cpu_count, hidden) {
            if !(f32::from_bits(cpu[i]).abs() <= f32::MAX) {
                valid = false;
            }
        } else if cpu[i] != 0 {
            valid = false;
        }
    }
    if i < probabilities.len() {
        if i >= prob_first && i - prob_first < RANKS {
            let p = f32::from_bits(probabilities[i]);
            if !(p.abs() <= f32::MAX) || p < 0. {
                valid = false;
            }
        } else if probabilities[i] != 0 {
            valid = false;
        }
    }
    if i < owners.len() {
        if i < owner_first || i - owner_first >= 4 + RANKS * 3 {
            if owners[i] != 0 {
                valid = false;
            }
        }
    }
    if map_first > map.len() {
        valid = false;
    } else if k::MAP_WORDS > map.len() - map_first {
        valid = false;
    } else {
        if i == 0 {
            if map[map_first] != 1
                || map[map_first + 1] != 0
                || map[map_first + 2] != 1
                || map[map_first + 3] as usize != gpu_count
            {
                valid = false;
            }
        }
        if i < map.len() {
            let mut active = false;
            if i >= map_first {
                let at = i - map_first;
                if at < 4 {
                    active = true;
                } else if at >= SLOTS && at < LENGTHS {
                    let group = at - SLOTS;
                    if group < gpu_count {
                        active = true;
                        if map[i] as usize >= capacity {
                            valid = false;
                        }
                    }
                } else if at >= LENGTHS && at < MEMBERS {
                    if at - LENGTHS < gpu_count {
                        active = true;
                        if map[i] != 1 {
                            valid = false;
                        }
                    }
                } else if at >= MEMBERS && at < k::MAP_WORDS {
                    let group = (at - MEMBERS) / MAX_BATCH;
                    if group < gpu_count && (at - MEMBERS) % MAX_BATCH == 0 {
                        active = true;
                        let rank = map[i] as usize;
                        if rank >= RANKS {
                            valid = false;
                        } else if owner_header(owners, owner_first, count) {
                            if owners[owner_first + 4 + rank * 3] != 0 {
                                valid = false;
                            }
                        } else {
                            valid = false;
                        }
                    }
                }
            }
            if !active && map[i] != 0 {
                valid = false;
            }
        }
        // Coverage and unique real slots are acceptance data, not pipeline keys.
        if i == 0 {
            for group in 0..RANKS {
                if group < gpu_count {
                    let slot = map[map_first + SLOTS + group];
                    let rank = map[map_first + MEMBERS + group * MAX_BATCH];
                    for earlier in 0..group {
                        if map[map_first + SLOTS + earlier] == slot
                            || map[map_first + MEMBERS + earlier * MAX_BATCH] == rank
                        {
                            valid = false;
                        }
                    }
                }
            }
        }
    }
    if i < status.len() {
        let mut expect_one = false;
        if i >= status_first {
            let at = i - status_first;
            let d = comptime!(hidden.div_ceil(k::LANES));
            let f = comptime!(ff.div_ceil(k::LANES));
            let a = comptime!(MAX_ROUTES * 2 + MAX_BATCH * hidden.div_ceil(k::LANES));
            let p = comptime!(
                MAX_ROUTES * 2
                    + MAX_BATCH * hidden.div_ceil(k::LANES)
                    + MAX_ROUTES * ff.div_ceil(k::LANES)
            );
            let o = comptime!(
                MAX_ROUTES * 2
                    + MAX_BATCH * hidden.div_ceil(k::LANES)
                    + MAX_ROUTES * ff.div_ceil(k::LANES)
                    + MAX_ROUTES * hidden.div_ceil(k::LANES)
            );
            let end = comptime!(
                MAX_ROUTES * 2
                    + MAX_BATCH * 2 * hidden.div_ceil(k::LANES)
                    + MAX_ROUTES * ff.div_ceil(k::LANES)
                    + MAX_ROUTES * hidden.div_ceil(k::LANES)
            );
            if at < MAX_ROUTES * 2 {
                let rank = at % MAX_ROUTES;
                if rank < RANKS && owner_header(owners, owner_first, count) {
                    if owners[owner_first + 4 + rank * 3] == 0 {
                        expect_one = true;
                    }
                }
            } else if at < a {
                if at - MAX_ROUTES * 2 < d {
                    expect_one = true;
                }
            } else if at < p {
                if at - a < RANKS * f {
                    expect_one = true;
                }
            } else if at < o {
                if at - p < RANKS * d {
                    expect_one = true;
                }
            } else if at < end {
                if at - o < d {
                    expect_one = true;
                }
            }
        }
        // Explicit comparisons avoid conditional-literal lowering in CubeCL.
        if expect_one {
            if status[i] != 1 {
                valid = false;
            }
        } else {
            if status[i] != 0 {
                valid = false;
            }
        }
    }
    if !valid {
        fault[0].fetch_or(4);
    }
}

fn launch_error() -> kuno_text::Error {
    kuno_text::Error::InvalidFormat("native count1 hybrid owning launch rejected")
}

/// Pure launch preflight, shared with the host fixtures. Logical lengths keep
/// their original meanings; scans and ArrayArgs cover the entire owning pages.
#[allow(clippy::too_many_arguments)]
fn validate_layouts(
    geometry: k::Geometry,
    spans: &Spans,
    count: usize,
    cpu_first: usize,
    cpu_stride: usize,
    owner_first: usize,
    layouts: &[BufferLayout; 10],
) -> kuno_text::Result<[usize; 10]> {
    // No count0/MAX8 call may enqueue any consumer or mutate private state.
    if count != 1
        || geometry.hidden() != 2560
        || geometry.ff() != 640
        || !matches!(geometry.tile(), 1 | 8)
        || cpu_first != 0
        || cpu_stride != geometry.hidden()
        || owner_first != 0
    {
        return Err(launch_error());
    }
    for layout in layouts {
        if *layout != BufferLayout::new(layout.logical_words).map_err(u::fault)? {
            return Err(launch_error());
        }
    }
    if layouts[7] != BufferLayout::new(CPU_WORDS).map_err(u::fault)?
        || layouts[7].owned_words != CPU_OWNED_WORDS
        || layouts[8] != BufferLayout::new(OWNER_WORDS).map_err(u::fault)?
        || layouts[8].owned_words != OWNER_OWNED_WORDS
        || layouts[9] != BufferLayout::new(1).map_err(u::fault)?
    {
        return Err(launch_error());
    }
    let words = layouts.map(|layout| layout.owned_words);
    spans
        .validate(geometry, std::array::from_fn(|i| words[i]))
        .map_err(|_| launch_error())?;
    Ok(words)
}

#[allow(clippy::too_many_arguments)]
fn owning_views(
    arena: &PrivateArena,
    geometry: k::Geometry,
    ids: &[BufferId; 7],
    cpu: BufferId,
    owner: BufferId,
    fault: BufferId,
    spans: &Spans,
    count: usize,
    cpu_first: usize,
    cpu_stride: usize,
    owner_first: usize,
) -> kuno_text::Result<[usize; 10]> {
    if count != 1 {
        return Err(launch_error());
    }
    let mut all = Zeroizing::new([fault; 10]);
    all[..7].copy_from_slice(ids);
    all[7] = cpu;
    all[8] = owner;
    let mut layouts = [BufferLayout::new(1).map_err(u::fault)?; 10];
    for (i, &id) in all.iter().enumerate() {
        if all[..i].contains(&id) {
            return Err(launch_error());
        }
        layouts[i] = arena.layout(id).map_err(u::fault)?;
    }
    let words = validate_layouts(
        geometry,
        spans,
        count,
        cpu_first,
        cpu_stride,
        owner_first,
        &layouts,
    )?;
    let client = arena.client().map_err(u::fault)?;
    let stream = arena.handle(fault).map_err(u::fault)?.stream;
    // Fixed public cardinality, not a private-sized Vec. Clones remain scoped
    // to this validation; the arena owns every buffer throughout both launches.
    let mut resources: [Option<wgpu::Buffer>; 10] = std::array::from_fn(|_| None);
    for (i, &id) in all.iter().enumerate() {
        let handle = arena.handle(id).map_err(u::fault)?;
        let bytes = layouts[i].owned_bytes as u64;
        if handle.stream != stream
            || handle.offset_start.unwrap_or(0) != 0
            || handle.offset_end.unwrap_or(0) != 0
            || handle.size() != bytes
            || handle.size_in_used() != bytes
        {
            return Err(launch_error());
        }
        let resource = client.get_resource(handle.clone()).map_err(u::fault)?;
        let view = resource.resource();
        if view.offset != 0
            || view.size != bytes
            || view.buffer.size() != bytes
            || resources[..i]
                .iter()
                .flatten()
                .any(|buffer| buffer == &view.buffer)
        {
            return Err(launch_error());
        }
        resources[i] = Some(view.buffer.clone());
    }
    Ok(words)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn merge(
    arena: &PrivateArena,
    geometry: k::Geometry,
    ids: &[BufferId; 7],
    cpu: BufferId,
    owner: BufferId,
    fault: BufferId,
    spans: &Spans,
    count: usize,
    cpu_first: usize,
    cpu_stride: usize,
    owner_first: usize,
) -> kuno_text::Result<()> {
    owning_views(
        arena,
        geometry,
        ids,
        cpu,
        owner,
        fault,
        spans,
        count,
        cpu_first,
        cpu_stride,
        owner_first,
    )?;
    let client = arena.client().map_err(u::fault)?;
    // SAFETY: all ten complete private pages have matching stream, origin,
    // physical extent, nonaliasing and full MAX8/MAX80 span proofs above. The
    // fixed public grid never depends on CPU/GPU selections; only runtime
    // scalar/view data choose original-rank or compact partials.
    unsafe {
        hybrid_merge::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(
                geometry.hidden().div_ceil(k::LANES) as u32,
                MAX_BATCH as u32,
                1,
            ),
            CubeDim::new_1d(k::LANES as u32),
            super::arg(arena, ids[3])?,
            super::arg(arena, cpu)?,
            super::arg(arena, owner)?,
            super::arg(arena, ids[4])?,
            super::arg(arena, ids[6])?,
            super::arg(arena, ids[5])?,
            super::arg(arena, fault)?,
            count,
            spans.partial_first(),
            spans.partial_stride(),
            cpu_first,
            cpu_stride,
            owner_first,
            spans.prob_first(),
            spans.status_first(),
            spans.output_first(),
            spans.output_stride(),
            geometry.hidden(),
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fold(
    arena: &PrivateArena,
    geometry: k::Geometry,
    ids: &[BufferId; 7],
    cpu: BufferId,
    owner: BufferId,
    fault: BufferId,
    spans: &Spans,
    count: usize,
    cpu_first: usize,
    cpu_stride: usize,
    owner_first: usize,
) -> kuno_text::Result<()> {
    let words = owning_views(
        arena,
        geometry,
        ids,
        cpu,
        owner,
        fault,
        spans,
        count,
        cpu_first,
        cpu_stride,
        owner_first,
    )?;
    let client = arena.client().map_err(u::fault)?;
    // SAFETY: the same full-owner proof as merge, including input/output and
    // both new owners. Every API word of all nine inputs is inspected; the
    // fault is only ORed and remains under whole-model completion authority.
    unsafe {
        hybrid_fold::launch_unchecked::<WgpuRuntime>(
            client,
            u::grid(*words[..9].iter().max().ok_or_else(launch_error)?)?,
            CubeDim::new_1d(k::LANES as u32),
            super::arg(arena, ids[1])?,
            super::arg(arena, ids[2])?,
            super::arg(arena, ids[3])?,
            super::arg(arena, cpu)?,
            super::arg(arena, ids[0])?,
            super::arg(arena, ids[4])?,
            super::arg(arena, ids[6])?,
            super::arg(arena, owner)?,
            super::arg(arena, ids[5])?,
            super::arg(arena, fault)?,
            count,
            spans.map_first(),
            spans.input_first(),
            spans.input_stride(),
            spans.act_first(),
            spans.act_stride(),
            spans.partial_first(),
            spans.partial_stride(),
            cpu_first,
            cpu_stride,
            spans.prob_first(),
            spans.status_first(),
            owner_first,
            spans.output_first(),
            spans.output_stride(),
            geometry.hidden(),
            geometry.ff(),
            geometry.capacity(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> [usize; RANKS] {
        [37, 5, 411, 83, 7, 132, 19, 509, 208, 64]
    }
    fn probabilities() -> [f32; RANKS] {
        [0.1; RANKS]
    }
    fn split(cpu_mask: u16) -> [u32; RANKS * 2] {
        let mut split = [0; RANKS * 2];
        let mut cpu = 0;
        for rank in 0..RANKS {
            if cpu_mask & (1 << rank) != 0 {
                split[rank * 2] = 1;
                split[rank * 2 + 1] = cpu;
                cpu += 1;
            } else {
                split[rank * 2 + 1] = rank as u32;
            }
        }
        split
    }
    fn pairs(cpu_mask: u16) -> Vec<(u32, u32)> {
        // Canonical GPU pair order deliberately differs from original rank
        // order. The encoded member resolves the ID, never the pair ordinal.
        ids()
            .into_iter()
            .enumerate()
            .rev()
            .filter(|&(rank, _)| cpu_mask & (1 << rank) == 0)
            .enumerate()
            .map(|(slot, (_, id))| (id as u32, slot as u32))
            .collect()
    }
    fn geometry(tile: usize, capacity: usize, down: u32) -> k::Geometry {
        let p = [
            k::Projection::new(crate::quant_matvec::Q4_K, 2560, 640, 512).unwrap(),
            k::Projection::new(crate::quant_matvec::Q4_K, 2560, 640, 512).unwrap(),
            k::Projection::new(down, 640, 2560, 512).unwrap(),
        ];
        k::Geometry::new(p, tile, capacity).unwrap()
    }
    fn spans() -> Spans {
        Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0])
    }
    fn layouts() -> [BufferLayout; 10] {
        [
            k::MAP_WORDS,
            MAX_BATCH * 2560,
            MAX_ROUTES * 640,
            MAX_ROUTES * 2560,
            MAX_ROUTES,
            MAX_BATCH * 2560,
            4800,
            CPU_WORDS,
            OWNER_WORDS,
            1,
        ]
        .map(|n| BufferLayout::new(n).unwrap())
    }

    #[test]
    fn map_covers_all_original_ranks_and_keeps_full_inactive_envelopes_zero() {
        for capacity in [16, 64, 72, 96] {
            for mask in [0, 0b0101010101, 0b1000000001, 0x3ff] {
                let pairs = pairs(mask);
                let map =
                    Map::new(&ids(), &probabilities(), &split(mask), &pairs, capacity).unwrap();
                let cpu = mask.count_ones() as usize;
                assert_eq!(map.cpu_count(), cpu);
                assert_eq!(map.gpu_count(), RANKS - cpu);
                assert_eq!(&map.words()[..4], &[1, 0, 1, (RANKS - cpu) as u32]);
                assert_eq!(
                    &map.owner_words()[..4],
                    &[1, cpu as u32, (RANKS - cpu) as u32, 0]
                );
                assert_eq!(map.owned_bytes(), std::mem::size_of::<Map>());
                let mut next_cpu = 0;
                for (rank, &id) in ids().iter().enumerate() {
                    let record = &map.owner_words()[4 + rank * 3..7 + rank * 3];
                    if mask & (1 << rank) != 0 {
                        assert_eq!(record, &[1, next_cpu, 0]);
                        next_cpu += 1;
                    } else {
                        assert_eq!(record, &[0, rank as u32, 0]);
                        let group = pairs.iter().position(|&(e, _)| e == id as u32).unwrap();
                        assert_eq!(map.words()[SLOTS + group], pairs[group].1);
                        assert_eq!(map.words()[LENGTHS + group], 1);
                        assert_eq!(map.words()[MEMBERS + group * MAX_BATCH], rank as u32);
                    }
                }
                for group in 0..MAX_ROUTES {
                    if group >= pairs.len() {
                        assert_eq!(map.words()[SLOTS + group], 0);
                        assert_eq!(map.words()[LENGTHS + group], 0);
                        assert!(
                            map.words()
                                [MEMBERS + group * MAX_BATCH..MEMBERS + (group + 1) * MAX_BATCH]
                                .iter()
                                .all(|&v| v == 0)
                        );
                    } else {
                        assert!(
                            map.words()[MEMBERS + group * MAX_BATCH + 1
                                ..MEMBERS + (group + 1) * MAX_BATCH]
                                .iter()
                                .all(|&v| v == 0)
                        );
                    }
                }
                assert!(map.owner_words()[4 + RANKS * 3..].iter().all(|&v| v == 0));
                assert!(
                    map.probabilities()[RANKS..]
                        .iter()
                        .all(|v| v.to_bits() == 0)
                );
            }
        }
    }

    #[test]
    fn zero_probability_keeps_rank_id_membership_and_exact_signed_bits() {
        let mut p = probabilities();
        p[0] = -0.;
        p[9] = 0.2;
        let mask = 1 << 0;
        let mut map = Map::new(&ids(), &p, &split(mask), &pairs(mask), 16).unwrap();
        assert_eq!(map.probabilities()[0].to_bits(), (-0f32).to_bits());
        assert_eq!(&map.owner_words()[4..7], &[1, 0, 0]);
        assert!(map.complete_cpu(None, 2560).is_err());
        assert!(
            map.owner_words()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i >= 6 && (*i - 6) % 3 == 0)
                .all(|(_, &v)| v == 0)
        );
        let mut invalid = ids();
        invalid[0] = 512;
        assert!(Map::new(&invalid, &p, &split(mask), &pairs(mask), 16).is_err());
        invalid = ids();
        invalid[0] = invalid[1];
        assert!(Map::new(&invalid, &p, &split(mask), &pairs(mask), 16).is_err());
    }

    #[test]
    fn malformed_owner_bijections_and_noncanonical_gpu_pairs_are_rejected() {
        let mask = 0b0101010101;
        let original = split(mask);
        let gpu = pairs(mask);
        for rank in 0..RANKS {
            let mut bad = original;
            bad[rank * 2] = 2;
            assert!(Map::new(&ids(), &probabilities(), &bad, &gpu, 16).is_err());
            bad = original;
            bad[rank * 2 + 1] = 10;
            assert!(Map::new(&ids(), &probabilities(), &bad, &gpu, 16).is_err());
        }
        let mut compact_gpu = split(0);
        compact_gpu[5 * 2 + 1] = 4;
        assert!(Map::new(&ids(), &probabilities(), &compact_gpu, &pairs(0), 16).is_err());
        for problem in 0..6 {
            let mut bad = gpu.clone();
            match problem {
                0 => bad[0].0 = 512,
                1 => bad[0].1 = 16,
                2 => bad[1].0 = bad[0].0,
                3 => bad[1].1 = bad[0].1,
                4 => bad[0].0 = ids()[0] as u32, // a CPU-owned expert
                _ => {
                    bad.pop();
                }
            }
            assert!(Map::new(&ids(), &probabilities(), &original, &bad, 16).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, -0.1, 0.2] {
            let mut p = probabilities();
            p[9] = value;
            assert!(Map::new(&ids(), &p, &original, &gpu, 16).is_err());
        }
        assert!(Map::new(&ids(), &probabilities(), &original, &gpu, 15).is_err());
    }

    #[test]
    fn cpu_readiness_has_no_none_shortcut_even_for_zero_probability() {
        for mask in [0, 1, 0b0101010101, 0x3ff] {
            let mut map =
                Map::new(&ids(), &probabilities(), &split(mask), &pairs(mask), 16).unwrap();
            let before = *map.owner_words();
            assert!(map.complete_cpu(None, 1280).is_err());
            assert_eq!(map.complete_cpu(None, 2560).is_ok(), mask == 0);
            assert_eq!(map.owner_words(), &before);
        }
        // Actual successful CPU readiness requires a core-created Partials
        // owner and will be exercised by the parent numeric/device fixture.
    }

    #[test]
    fn original_rank_oracle_catches_subset_reordering_and_compact_gpu_indices() {
        let mask = 0b0101010101;
        let map = Map::new(&ids(), &probabilities(), &split(mask), &pairs(mask), 16).unwrap();
        let original = [1e20f32, -1e20, 30., 0., 0., 0., 0., 0., 0., 0.];
        let mut cpu = [0.; RANKS];
        let mut gpu = [0.; RANKS];
        for rank in 0..RANKS {
            let triple = &map.owner_words()[4 + rank * 3..7 + rank * 3];
            if triple[0] == 1 {
                cpu[triple[1] as usize] = original[rank];
            } else {
                gpu[rank] = original[rank];
            }
        }
        let mut actual = 0f32;
        for rank in 0..RANKS {
            let triple = &map.owner_words()[4 + rank * 3..7 + rank * 3];
            let value = if triple[0] == 1 {
                cpu[triple[1] as usize]
            } else {
                gpu[rank]
            };
            actual += map.probabilities()[rank] * value;
        }
        assert_eq!(actual, 3.);
        let subset_order = original
            .iter()
            .enumerate()
            .filter(|(r, _)| mask & (1 << r) != 0)
            .chain(
                original
                    .iter()
                    .enumerate()
                    .filter(|(r, _)| mask & (1 << r) == 0),
            )
            .fold(0f32, |sum, (_, &value)| sum + 0.1 * value);
        assert_ne!(subset_order, actual);
    }

    #[test]
    fn preflight_rejects_count_changes_hidden_extents_and_partial_owning_layouts() {
        let original = layouts();
        for down in [crate::quant_matvec::Q5_0, crate::quant_matvec::Q8_0] {
            for tile in [1, 8] {
                for capacity in [16, 64, 72, 96] {
                    let g = geometry(tile, capacity, down);
                    assert!(validate_layouts(g, &spans(), 1, 0, 2560, 0, &original).is_ok());
                    for count in [0, 2, 8, usize::MAX] {
                        assert!(
                            validate_layouts(g, &spans(), count, 0, 2560, 0, &original).is_err()
                        );
                    }
                    for [first, stride, owner] in [[1, 2560, 0], [0, 2559, 0], [0, 2560, 1]] {
                        assert!(
                            validate_layouts(g, &spans(), 1, first, stride, owner, &original)
                                .is_err()
                        );
                    }
                    for index in 0..10 {
                        let mut short = original;
                        short[index].owned_words -= 1;
                        assert!(validate_layouts(g, &spans(), 1, 0, 2560, 0, &short).is_err());
                    }
                    for index in [7, 8, 9] {
                        let mut changed = original;
                        changed[index] =
                            BufferLayout::new(changed[index].logical_words + 1).unwrap();
                        assert!(validate_layouts(g, &spans(), 1, 0, 2560, 0, &changed).is_err());
                    }
                    let bad_spans =
                        Spans::new([0, usize::MAX, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
                    assert!(validate_layouts(g, &bad_spans, 1, 0, 2560, 0, &original).is_err());
                }
            }
        }
        assert!(
            validate_layouts(
                geometry(4, 16, crate::quant_matvec::Q5_0),
                &spans(),
                1,
                0,
                2560,
                0,
                &original
            )
            .is_err()
        );
        let p = [
            k::Projection::new(crate::quant_matvec::Q4_K, 1280, 320, 512).unwrap(),
            k::Projection::new(crate::quant_matvec::Q4_K, 1280, 320, 512).unwrap(),
            k::Projection::new(crate::quant_matvec::Q8_0, 320, 1280, 512).unwrap(),
        ];
        assert!(
            validate_layouts(
                k::Geometry::new(p, 1, 16).unwrap(),
                &spans(),
                1,
                0,
                2560,
                0,
                &original
            )
            .is_err()
        );
    }

    fn shader(fold: bool, mode: cubecl::server::ExecutionMode) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let atom = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope.register_type::<f32>(float);
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        if fold {
            let input = b.input_array(Type::new(uint)).into();
            let act = b.input_array(Type::new(uint)).into();
            let gpu = b.input_array(Type::new(uint)).into();
            let cpu = b.input_array(Type::new(uint)).into();
            let map = b.input_array(Type::new(uint)).into();
            let prob = b.input_array(Type::new(uint)).into();
            let status = b.input_array(Type::new(uint)).into();
            let owners = b.input_array(Type::new(uint)).into();
            let output = b.input_array(Type::new(uint)).into();
            let fault = b.output_array(Type::new(atom)).into();
            let count = b.scalar(uint).into();
            let map_first = b.scalar(uint).into();
            let input_first = b.scalar(uint).into();
            let input_stride = b.scalar(uint).into();
            let act_first = b.scalar(uint).into();
            let act_stride = b.scalar(uint).into();
            let partial_first = b.scalar(uint).into();
            let partial_stride = b.scalar(uint).into();
            let cpu_first = b.scalar(uint).into();
            let cpu_stride = b.scalar(uint).into();
            let prob_first = b.scalar(uint).into();
            let status_first = b.scalar(uint).into();
            let owner_first = b.scalar(uint).into();
            let output_first = b.scalar(uint).into();
            let output_stride = b.scalar(uint).into();
            hybrid_fold::expand(
                &mut b.scope,
                input,
                act,
                gpu,
                cpu,
                map,
                prob,
                status,
                owners,
                output,
                fault,
                count,
                map_first,
                input_first,
                input_stride,
                act_first,
                act_stride,
                partial_first,
                partial_stride,
                cpu_first,
                cpu_stride,
                prob_first,
                status_first,
                owner_first,
                output_first,
                output_stride,
                2560,
                640,
                96,
            );
        } else {
            let gpu = b.input_array(Type::new(float)).into();
            let cpu = b.input_array(Type::new(float)).into();
            let owners = b.input_array(Type::new(uint)).into();
            let prob = b.input_array(Type::new(float)).into();
            let status = b.input_array(Type::new(uint)).into();
            let output = b.output_array(Type::new(float)).into();
            let fault = b.output_array(Type::new(atom)).into();
            let count = b.scalar(uint).into();
            let partial_first = b.scalar(uint).into();
            let partial_stride = b.scalar(uint).into();
            let cpu_first = b.scalar(uint).into();
            let cpu_stride = b.scalar(uint).into();
            let owner_first = b.scalar(uint).into();
            let prob_first = b.scalar(uint).into();
            let status_first = b.scalar(uint).into();
            let output_first = b.scalar(uint).into();
            let output_stride = b.scalar(uint).into();
            hybrid_merge::expand(
                &mut b.scope,
                gpu,
                cpu,
                owners,
                prob,
                status,
                output,
                fault,
                count,
                partial_first,
                partial_stride,
                cpu_first,
                cpu_stride,
                owner_first,
                prob_first,
                status_first,
                output_first,
                output_stride,
                2560,
            );
        }
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("count1_hybrid_ranks"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                uint,
            )
            .unwrap()
            .to_string()
    }

    #[test]
    fn offline_shaders_preserve_runtime_coordinates_full_scans_and_atomic_fault() {
        use cubecl::server::ExecutionMode;
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            for fold in [false, true] {
                let source = shader(fold, mode);
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                assert_eq!(module.entry_points.len(), 1);
                assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
                let storage = module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| matches!(v.space, wgpu::naga::AddressSpace::Storage { .. }))
                    .count();
                assert!(storage >= if fold { 10 } else { 7 });
                assert!(storage <= if fold { 11 } else { 8 });
                for scalar in 0..if fold { 15 } else { 10 } {
                    assert!(
                        source.contains(&format!("scalars_u32[{scalar}]")),
                        "missing runtime scalar{scalar}"
                    );
                }
                assert!(source.contains("atomicOr"));
                assert!(!source.contains("workgroupBarrier"));
                assert!(!source.contains("array<f32, 10>"));
                assert!(!source.contains("array<u32, 80>"));
            }
        }
    }
}
