//! Fixed-public arithmetic leaves for the original Q8 MTP sidecar.
//!
//! This is not an executor, model/session owner, device initializer or API.
//! The parent must bind all34 original tensors, keep all512 routed experts
//! resident, prove full private/public bindings disjoint on its sealed stream,
//! charge every dispatch/owner/INFO record and join every fault before committing
//! a draft position. The stored indexer tensors are deliberately never executed.
//! Draft tokens are provisional until the target verifier accepts them.
//!
//! Equations follow kuno_text::mtp (llama.cpp PR28243 at
//! 6fcaa16f4b360649933a54d1f91ad40ed35c0e11, MIT). Packed Q8 decoding and
//! the original64-lane dot/reduction are reused verbatim. Dense GQA uses the
//! accepted qsa_gpu f32 lane equations with identity, full-visible addressing;
//! no QSA selection, pooling, indexer or activation quantization is introduced.
//! The scalar reference uses f64 for attention/norm reductions: complete-vector
//! numeric gates remain required, and no bit-identical CPU claim follows.
#![allow(dead_code)] // Parent resident workspace connection is a separate gate.

use crate::quant_matvec::{Q8_0, fast};
use cubecl::prelude::*;
use kuno_text::{Error, Result};

pub const D: usize = 2560;
pub const HC: usize = 4;
pub const WIDE: usize = D * HC;
pub const LOW_RANK: usize = 320;
pub const FF: usize = 640;
pub const EXPERTS: usize = 512;
pub const RANKS: usize = 10;
pub const QH: usize = 24;
pub const KVH: usize = 2;
pub const HD: usize = 256;
pub const QUERY: usize = QH * HD;
pub const KV: usize = KVH * HD;
pub const CONTEXT: usize = if crate::strata_intel_core::enabled() {
    4096
} else {
    2048
};
pub const CACHE_WORDS: usize = CONTEXT * KV;
pub const CACHE_STRIDE: usize = if crate::strata_intel_core::enabled() {
    // Two256-value heads:64 packed code words and two packed FP16-scale words.
    KVH * (HD / 4 + HD / 64 / 2)
} else {
    KV
};
pub const CACHE_STORAGE_WORDS: usize = CONTEXT * CACHE_STRIDE;
pub const ATTENTION_WORDS: usize = QH * CONTEXT;
pub const CONDITION_WORDS: usize = HC * D * 2;
pub const ACTIVATION_WORDS: usize = RANKS * FF;
pub const PARTIAL_WORDS: usize = RANKS * D;
pub const READINESS_WORDS: usize = RANKS * 2;
pub const VOCAB: usize = 248320;
pub const ROW_BYTES: usize = D / 32 * 34;
pub const DOWN_ROW_BYTES: usize = FF / 32 * 34;
pub const EXPERT_BYTES: usize = FF * ROW_BYTES;
pub const SHARDS: usize = 8;
pub const SHARD_CAPACITY: usize = 72;
pub const ROUTE_FIRST: usize = crate::router_gpu::FIRST;
pub const ROUTE_END: usize = crate::router_gpu::END;
pub const LANES: usize = 64;

/// Public expert geometry only. Runtime IDs/probabilities never key a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub first: usize,
    pub experts: usize,
}
impl Shard {
    pub fn new(index: usize) -> Result<Self> {
        if index >= SHARDS {
            return Err(Error::OutOfBounds);
        }
        let first = index * SHARD_CAPACITY;
        Ok(Self {
            first,
            experts: (EXPERTS - first).min(SHARD_CAPACITY),
        })
    }
    pub fn validate(self) -> Result<()> {
        if !self.first.is_multiple_of(SHARD_CAPACITY)
            || Self::new(self.first / SHARD_CAPACITY)? != self
        {
            return Err(Error::InvalidFormat("MTP original expert shard"));
        }
        Ok(())
    }
    pub fn packed_bytes(self) -> Result<usize> {
        self.validate()?;
        Ok(self.experts * EXPERT_BYTES)
    }
    pub fn packed_words(self) -> Result<usize> {
        Ok(self.packed_bytes()? / 4)
    }
    pub fn owned_bytes(self) -> Result<u64> {
        crate::main_admission::page_bytes(self.packed_bytes()? as u64)
            .map_err(|_| Error::InvalidBufferLength)
    }
}

/// Exact full512 expert owners: three original families, seven72-expert128MiB
/// pages and one8-expert16MiB page each. No dynamic cache or expert substitution.
pub fn expert_owned_bytes() -> Result<u64> {
    (0..SHARDS).try_fold(0_u64, |sum, index| {
        sum.checked_add(3 * Shard::new(index)?.owned_bytes()?)
            .ok_or(Error::ArithmeticOverflow)
    })
}

/// Parent must check position/count before any unchecked launch. Public cache
/// shape is fixed; the actual position enters charged INFO, never shader IDs.
pub fn validate_position(position: usize) -> Result<usize> {
    if position >= CONTEXT {
        return Err(Error::OutOfBounds);
    }
    Ok(position + 1)
}
pub fn validate_rewind(old_position: usize, new_position: usize) -> Result<()> {
    if old_position > CONTEXT || new_position > old_position {
        return Err(Error::OutOfBounds);
    }
    Ok(())
}

pub fn validate_limits(limits: &wgpu::Limits) -> Result<()> {
    if limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroups_per_dimension < (QH * CONTEXT) as u32
        || limits.max_compute_workgroup_storage_size < 2048
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_storage_buffer_binding_size < 128 * 1024 * 1024
        || limits.max_buffer_size < 128 * 1024 * 1024
    {
        return Err(Error::LimitExceeded("MTP original resident kernel limits"));
    }
    Ok(())
}

#[cube]
#[allow(clippy::manual_range_contains)]
fn finite(value: f32) -> bool {
    value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32
}
#[cube]
#[allow(unused_assignments)]
fn sigmoid(value: f32) -> f32 {
    let mut result = 0.0_f32;
    if value >= 0.0 {
        result = 1.0 / (1.0 + (-value).exp());
    } else {
        let e = value.exp();
        result = e / (1.0 + e);
    }
    result
}

/// One fixed original embedding tile; enqueue every public tile in order.
/// Only the selected tile stores. The parent preclears embedding and binds the
/// token's existing full owner. Last tile is2560 rows; others16384 rows.
#[cube(launch, launch_unchecked)]
pub fn embedding_tile(
    weights: &Array<u32>,
    token: &Array<u32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] first: usize,
    #[comptime] rows: usize,
) {
    let column = ABSOLUTE_POS;
    let selected = token[0] as usize;
    if column < D {
        if selected >= VOCAB {
            fault[0].fetch_or(1);
        } else if selected >= first && selected - first < rows {
            let value = crate::quant_matvec::packed_value(
                weights,
                (selected - first) * ROW_BYTES,
                column,
                Q8_0,
            );
            if finite(value) {
                output[column] = value;
            } else {
                fault[0].fetch_or(1);
            }
        }
    }
}

/// Four EH input rows, [normalized embedding; normalized preceding HC stream].
/// Launch original packed_matmul Q8 with count4/tile4, stride5120 -> stride2560.
#[cube(launch, launch_unchecked)]
#[allow(unused_assignments)] // CubeCL runtime branch requires mutable initialization.
pub fn condition_concat(
    embedding: &Array<f32>,
    hidden: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let i = ABSOLUTE_POS;
    if i < HC * D * 2 {
        let stream = i / (D * 2);
        let column = i % (D * 2);
        let mut value = 0.0_f32;
        if column < D {
            value = embedding[column];
        } else {
            value = hidden[stream * D + column - D];
        }
        if finite(value) {
            output[i] = value;
        } else {
            fault[0].fetch_or(1);
        }
    }
}

/// K has already received exact partial64 NeoX. The proposed suffix is private;
/// any fault poisons/reset-erases the complete draft cache before reuse.
#[cube(launch, launch_unchecked)]
pub fn append_kv(
    key: &Array<f32>,
    value: &Array<f32>,
    keys: &mut Array<f32>,
    values: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    position: usize,
) {
    let i = ABSOLUTE_POS;
    if i < KV {
        if position >= CONTEXT {
            fault[0].fetch_or(1);
        } else {
            let k = key[i];
            let v = value[i];
            if finite(k) && finite(v) {
                keys[position * KV + i] = k;
                values[position * KV + i] = v;
            } else {
                fault[0].fetch_or(1);
            }
        }
    }
}

/// Complete logical suffix erase, including unused2K capacity. Owners are
/// exactly4MiB each, so there is no untracked cache padding. Parent binds a
/// completed speculative fence before repairing a prefix; target state is not
/// changed here. A rejected rewind performs no cache store.
#[cube(launch, launch_unchecked)]
pub fn rewind_kv(
    keys: &mut Array<f32>,
    values: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    old_position: usize,
    new_position: usize,
) {
    let i = ABSOLUTE_POS;
    if old_position > CONTEXT || new_position > old_position {
        if i == 0 {
            fault[0].fetch_or(1);
        }
    } else if i < CACHE_WORDS && i >= new_position * KV {
        keys[i] = 0.0;
        values[i] = 0.0;
    }
}

/// One64-lane cube per actual(head,visible token), compact head-major output.
/// Every token0..visible is attended; KV head=h/12, score=dot/16.
#[cube(launch, launch_unchecked)]
pub fn dense_logits(
    query: &Array<f32>,
    keys: &Array<f32>,
    logits: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    visible: usize,
) {
    let task = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    // Parent validates visible before division; also guard malformed device
    // dispatch values, without an out-of-range load or output publication.
    let mut head = 24_usize;
    let mut token = 2048_usize;
    if visible > 0 && visible <= CONTEXT {
        head = task / visible;
        token = task % visible;
    }
    let mut sum = 0.0_f32;
    let mut good = 1_u32;
    if head < QH && token < visible && token < CONTEXT {
        for tile in 0..4_usize {
            let d = tile * LANES + lane;
            let q = query[head * HD + d];
            let k = keys[token * KV + (head / 12) * HD + d];
            if finite(q) && finite(k) {
                sum += q * k;
            } else {
                good = 0;
            }
        }
    } else {
        good = 0;
    }
    let mut sums = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    sums[lane] = sum;
    flags[lane] = good;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            sums[lane] += sums[lane + stride];
            flags[lane] &= flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        let score = sums[0] / 16.0;
        if flags[0] == 1 && finite(score) {
            logits[task] = score;
        } else {
            fault[0].fetch_or(1);
        }
    }
}

/// Reuse qsa_gpu::attention_softmax directly (head-major compact count=visible)
/// and fold its24 statuses. It has no sparse-selection or cache dependency.
/// Norm/partial NeoX also reuse qsa_gpu::rms_rope plus the full4096-position
/// public tables, and split_query_gate preserves per-head Q/gate interleave.
#[allow(unused_imports)] // Reexports are the parent workspace's arithmetic seam.
pub use crate::qsa_gpu::{attention_softmax, rms_rope, rope_tables, split_query_gate};
/// HC4 helpers retain down/4 INSIDE SiLU and mean mix /4, with injections2sigmoid.
#[allow(unused_imports)]
pub use crate::residual_gpu::{activate_down, combine, grouped_rms_norm, mix};

#[cube(launch, launch_unchecked)]
#[allow(clippy::manual_range_contains)] // CubeCL has no runtime RangeInclusive::contains.
pub fn dense_values_gated(
    values: &Array<f32>,
    probabilities: &Array<f32>,
    gate: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    visible: usize,
) {
    let i = ABSOLUTE_POS;
    if i < QUERY {
        let head = i / HD;
        let dimension = i % HD;
        let mut sum = 0.0_f32;
        let mut good = visible > 0 && visible <= CONTEXT;
        if good {
            for token in 0..visible {
                let p = probabilities[head * visible + token];
                let value = values[token * KV + (head / 12) * HD + dimension];
                if finite(p) && p >= 0.0 && p <= 1.0 && finite(value) {
                    sum += p * value;
                } else {
                    good = false;
                }
            }
        }
        let g = gate[i];
        let value = sum * sigmoid(g);
        if good && finite(g) && finite(value) {
            output[i] = value;
        } else {
            fault[0].fetch_or(1);
        }
    }
}

/// Validate the actual original paired GPU top10 before any expert consumer.
/// Sum/order/unique-ID checks never replace/reorder/substitute a selected expert.
#[cube(launch, launch_unchecked)]
#[allow(clippy::manual_range_contains)]
pub fn routes_preflight(packet: &Array<u32>, fault: &mut Array<Atomic<u32>>) {
    if UNIT_POS_X == 0 {
        let mut sum = 0.0_f32;
        let mut good = true;
        for rank in 0..RANKS {
            let id = packet[ROUTE_FIRST + rank * 2];
            let p = f32::reinterpret(packet[ROUTE_FIRST + rank * 2 + 1]);
            if id >= EXPERTS as u32 || !finite(p) || p < 0.0 || p > 1.0 {
                good = false;
            }
            for previous in 0..rank {
                if packet[ROUTE_FIRST + previous * 2] == id {
                    good = false;
                }
            }
            sum += p;
        }
        if !good || !finite(sum) || (sum - 1.0).abs() > 0.00001 {
            fault[0].fetch_or(1);
        }
    }
}

/// Fixed original72×7+8 sharding. All eight gate/up dispatches precede all eight
/// down dispatches. Each original rank is covered exactly once, by its actual
/// expert's public shard; private IDs/ranks never affect a shader cache key.
#[cube(launch, launch_unchecked)]
pub fn q8_gate_up(
    gate: &Array<u32>,
    up: &Array<u32>,
    input: &Array<f32>,
    packet: &Array<u32>,
    activation: &mut Array<f32>,
    readiness: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] first: usize,
    #[comptime] experts: usize,
) {
    let rank = CUBE_POS_Y as usize;
    let row = CUBE_POS_X as usize * fast::ROWS_PER_CUBE + fast::local_row();
    let mut id = 512_usize;
    if rank < RANKS {
        id = packet[ROUTE_FIRST + rank * 2] as usize;
    }
    let mut local = 0_usize;
    let selected = rank < RANKS && id >= first && id - first < experts && id < EXPERTS;
    if selected {
        local = id - first;
    }
    let valid = selected && row < FF && fault[0].load() == 0;
    let base = (local * FF + row) * ROW_BYTES;
    let g = fast::dot(gate, input, base, 0, valid, D, Q8_0);
    let u = fast::dot(up, input, base, 0, valid, D, Q8_0);
    if fast::lane() == 0 && valid {
        let value = g * sigmoid(g) * u;
        if finite(value) {
            activation[rank * FF + row] = value;
            if row == 0 {
                readiness[rank] = 1;
            }
        } else {
            fault[0].fetch_or(1);
        }
    }
}

#[cube(launch, launch_unchecked)]
pub fn q8_down(
    down: &Array<u32>,
    activation: &Array<f32>,
    packet: &Array<u32>,
    partials: &mut Array<f32>,
    readiness: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] first: usize,
    #[comptime] experts: usize,
) {
    let rank = CUBE_POS_Y as usize;
    let row = CUBE_POS_X as usize * fast::ROWS_PER_CUBE + fast::local_row();
    let mut id = 512_usize;
    if rank < RANKS {
        id = packet[ROUTE_FIRST + rank * 2] as usize;
    }
    let mut local = 0_usize;
    let selected = rank < RANKS && id >= first && id - first < experts && id < EXPERTS;
    if selected {
        local = id - first;
    }
    let mut valid = selected && row < D && fault[0].load() == 0;
    if rank < RANKS {
        valid = valid && readiness[rank] == 1;
    }
    let base = (local * D + row) * DOWN_ROW_BYTES;
    let value = fast::dot(down, activation, base, rank * FF, valid, FF, Q8_0);
    if fast::lane() == 0 && valid {
        if finite(value) {
            partials[rank * D + row] = value;
            if row == 0 {
                readiness[RANKS + rank] = 1;
            }
        } else {
            fault[0].fetch_or(1);
        }
    }
}

/// Shared expert uses three ordinary Q8 projections, this unchanged SiLU×up,
/// and the original sequential f32 scalar gate (not a new reduction schedule).
#[cube(launch, launch_unchecked)]
pub fn shared_activate(
    gate: &Array<f32>,
    up: &Array<f32>,
    activation: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let i = ABSOLUTE_POS;
    if i < FF {
        let g = gate[i];
        let u = up[i];
        let value = g * sigmoid(g) * u;
        if finite(g) && finite(u) && finite(value) {
            activation[i] = value;
        } else {
            fault[0].fetch_or(1);
        }
    }
}
#[cube(launch, launch_unchecked)]
pub fn shared_gate(
    weight: &Array<f32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
) {
    if UNIT_POS_X == 0 {
        let mut sum = 0.0_f32;
        let mut good = true;
        for i in 0..D {
            let w = weight[i];
            let x = input[i];
            good = good && finite(w) && finite(x);
            sum += w * x;
        }
        if good && finite(sum) {
            output[0] = sum;
        } else {
            fault[0].fetch_or(1);
        }
    }
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::manual_range_contains)]
pub fn routed_shared_merge(
    partials: &Array<f32>,
    shared: &Array<f32>,
    scalar: &Array<f32>,
    packet: &Array<u32>,
    readiness: &Array<u32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let i = ABSOLUTE_POS;
    if i < D {
        let mut sum = 0.0_f32;
        let mut good = fault[0].load() == 0;
        for rank in 0..RANKS {
            let p = f32::reinterpret(packet[ROUTE_FIRST + rank * 2 + 1]);
            let v = partials[rank * D + i];
            good = good
                && readiness[rank] == 1
                && readiness[RANKS + rank] == 1
                && finite(p)
                && p >= 0.0
                && p <= 1.0
                && finite(v);
            sum += p * v;
        }
        let s = scalar[0];
        let v = shared[i];
        sum += sigmoid(s) * v;
        if good && finite(s) && finite(v) && finite(sum) {
            output[i] = sum;
        } else {
            fault[0].fetch_or(1);
        }
    }
}

/// Independent exact greedy over every248320 logit, including ties/end rows.
/// Stage1 is61 public4096-row cubes; stage2 consumes their complete triples.
pub const GREEDY_CHUNK: usize = 4096;
pub const GREEDY_GROUPS: usize = VOCAB.div_ceil(GREEDY_CHUNK);
pub const GREEDY_WORDS: usize = GREEDY_GROUPS * 3;
#[cube(launch, launch_unchecked)]
pub fn greedy_chunks(
    logits: &Array<f32>,
    candidates: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let group = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut best = 0.0_f32;
    best -= 3.4028235e38_f32;
    let mut id = 4294967295_u32;
    let mut good = 1_u32;
    for tile in 0..GREEDY_CHUNK / LANES {
        let i = group * GREEDY_CHUNK + tile * LANES + lane;
        if i < VOCAB {
            let v = logits[i];
            if !finite(v) {
                good = 0;
            } else if v > best || v == best && (i as u32) < id {
                best = v;
                id = i as u32;
            }
        }
    }
    let mut values = SharedMemory::<f32>::new(LANES);
    let mut ids = SharedMemory::<u32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    values[lane] = best;
    ids[lane] = id;
    flags[lane] = good;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            let v = values[lane + stride];
            let j = ids[lane + stride];
            if v > values[lane] || v == values[lane] && j < ids[lane] {
                values[lane] = v;
                ids[lane] = j;
            }
            flags[lane] &= flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && group < GREEDY_GROUPS {
        candidates[group * 3] = ids[0];
        candidates[group * 3 + 1] = values[0].to_bits();
        candidates[group * 3 + 2] = flags[0];
        if flags[0] != 1 {
            fault[0].fetch_or(1);
        }
    }
}
#[cube(launch, launch_unchecked)]
pub fn greedy_finish(
    candidates: &Array<u32>,
    token: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let lane = UNIT_POS_X as usize;
    let mut value = 0.0_f32;
    value -= 3.4028235e38_f32;
    let mut id = 4294967295_u32;
    let mut good = 1_u32;
    if lane < GREEDY_GROUPS {
        id = candidates[lane * 3];
        value = f32::reinterpret(candidates[lane * 3 + 1]);
        if candidates[lane * 3 + 2] != 1 || id >= VOCAB as u32 || !finite(value) {
            good = 0;
        }
    }
    let mut values = SharedMemory::<f32>::new(LANES);
    let mut ids = SharedMemory::<u32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    values[lane] = value;
    ids[lane] = id;
    flags[lane] = good;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            let v = values[lane + stride];
            let j = ids[lane + stride];
            if v > values[lane] || v == values[lane] && j < ids[lane] {
                values[lane] = v;
                ids[lane] = j;
            }
            flags[lane] &= flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        if flags[0] == 1 && fault[0].load() == 0 {
            token[0] = ids[0];
        } else {
            fault[0].fetch_or(1);
        }
    }
}

#[cfg(test)]
#[path = "mtp_resident_kernel_tests.rs"]
mod tests;
