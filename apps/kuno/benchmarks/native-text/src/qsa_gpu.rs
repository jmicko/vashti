//! Stateless Qwen3.8 QSA Rust/CubeCL prototype, oracle core ops/model at
//! 2467ac7554eaa18586905344c7e47447930831f9. No device work or hidden state.
//!
//! Owner: keep old cache readonly and candidate/output regions disjoint from
//! ALL inputs/outputs; validate real buffer regions, not just Rust handles.
//! Reset status to zero before every attempt and require successful completion
//! plus every stage status before committing the candidate. Cancellation,
//! bounds/finite/status/device failure poisons the WHOLE token. Never retain
//! prompt-derived caches, positions, scores or selection outside its request.
//!
//! Cache is token-major [K512,V512,RAW indexer key128]; the current token is a
//! distinct 1152-float candidate. No full-history copy is needed. Position and
//! visible/selected counts are runtime parameters. Static gamma is supplied
//! already converted/folded; do not add one or requantize it. Public sin/cos
//! tables cover all4096positions/32pairs and use the core's Rust f32 angle and
//! sin_cos; shader trig approximations must not change large-angle RoPE.
//!
//! Dispatch 64 lanes. Split/pack/output use ceil(elements/64) cubes; RMS/pool/
//! score use one cube/group; logits use 24*selected cubes; softmax uses24 cubes.
//! Skip pool/pooled-RMS/score entirely when complete_blocks==0. Score readback
//! and CPU qsa_select_blocks intentionally precede GPU attention initially.
//! Full budget is 512 COMPLETE blocks plus ALL 0..3 tail tokens (max2051).
//!
//! Core uses f64 reductions and rotation multiply/add, with f32 stage casts.
//! Here f32 tree reductions, exp and contraction/subnormal behavior
//! can differ. Near-cutoff scores may select different blocks: numerical and
//! full-graph device acceptance is required, not established by these tests.
//! Intermediate overflow fails even when a f64 oracle could cancel it.

use cubecl::prelude::*;
use kuno_text::ops;

pub const LANES: usize = 64;
pub const QH: usize = 24;
pub const KVH: usize = 2;
pub const HD: usize = 256;
pub const IH: usize = 4;
pub const ID: usize = 128;
pub const QUERY: usize = QH * HD;
pub const KV: usize = KVH * HD;
pub const INDEX_QUERY: usize = IH * ID;
pub const TOKEN: usize = KV * 2 + ID;
pub const BLOCK: usize = 4;
pub const BUDGET: usize = 2048;
pub const MAX_CONTEXT: usize = 4096;
pub const MAX_POSITION: u32 = (MAX_CONTEXT - 1) as u32;
pub const ROPE_PAIRS: usize = 32;
pub const ROPE_TABLE_WORDS: usize = MAX_CONTEXT * ROPE_PAIRS;
pub const EPSILON: f32 = 1e-6;
pub const THETA: f32 = 10_000_000.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct QsaLayout {
    pub visible: usize,
    pub previous: usize,
    pub complete: usize,
    pub selected: usize,
    pub old_elements: usize,
    pub logits_elements: usize,
}
impl QsaLayout {
    pub fn new(visible: usize) -> Result<Self, &'static str> {
        if !(1..=MAX_CONTEXT).contains(&visible) {
            return Err("QSA visible outside1..4096");
        }
        let complete = visible / BLOCK;
        let selected = complete.min(BUDGET / BLOCK) * BLOCK + visible % BLOCK;
        Ok(Self {
            visible,
            previous: visible - 1,
            complete,
            selected,
            old_elements: (visible - 1) * TOKEN,
            logits_elements: QH * selected,
        })
    }
    pub fn validate(self) -> Result<(), &'static str> {
        if Self::new(self.visible)? != self {
            return Err("forged QSA layout");
        }
        Ok(())
    }
}

pub fn rope_frequencies() -> [f32; 32] {
    std::array::from_fn(|i| THETA.powf(-(i as f32) / 32.0))
}
struct RopeTables {
    sin: Box<[f32]>,
    cos: Box<[f32]>,
}
/// One immutable PUBLIC table pair shared by every layer/request. GPU upload
/// remains once per bound public cache/context under fixed sin/cos names.
/// Separate arrays fit the cache's existing131072-float upload bound each.
pub fn rope_tables() -> (&'static [f32], &'static [f32]) {
    static TABLES: std::sync::OnceLock<RopeTables> = std::sync::OnceLock::new();
    let table = TABLES.get_or_init(|| {
        let frequencies = rope_frequencies();
        let mut sin = Vec::with_capacity(ROPE_TABLE_WORDS);
        let mut cos = Vec::with_capacity(ROPE_TABLE_WORDS);
        for position in 0..MAX_CONTEXT {
            for frequency in frequencies {
                let angle = position as f32 * frequency;
                let (s, c) = angle.sin_cos();
                sin.push(s);
                cos.push(c);
            }
        }
        RopeTables {
            sin: sin.into_boxed_slice(),
            cos: cos.into_boxed_slice(),
        }
    });
    (&table.sin, &table.cos)
}
pub fn validate_positions(layout: QsaLayout, positions: &[u32]) -> Result<(), &'static str> {
    layout.validate()?;
    if positions.len() != layout.visible
        || positions.iter().any(|&p| p > MAX_POSITION)
        || positions.windows(2).any(|p| p[0] >= p[1])
    {
        return Err("invalid QSA positions");
    }
    Ok(())
}
pub fn validate_cache_lengths(
    layout: QsaLayout,
    old: usize,
    candidate: usize,
) -> Result<(), &'static str> {
    layout.validate()?;
    if old != layout.old_elements || candidate != TOKEN {
        return Err("invalid QSA cache lengths");
    }
    Ok(())
}
/// Returns the public-layout shared-gamma specialization, never request data.
// Mirrors the norm dispatch's individual binding lengths and runtime scalars.
#[allow(clippy::too_many_arguments)]
pub fn validate_norm_lengths(
    input: usize,
    gamma: usize,
    sin_table: usize,
    cos_table: usize,
    positions: usize,
    output: usize,
    status: usize,
    groups: usize,
    width: usize,
    block_positions: bool,
    position: u32,
    epsilon: f32,
) -> Result<bool, &'static str> {
    if ![ID, HD].contains(&width)
        || groups == 0
        || groups > MAX_CONTEXT / BLOCK
        || input != groups * width
        || output != input
        || status != groups
        || sin_table != ROPE_TABLE_WORDS
        || cos_table != ROPE_TABLE_WORDS
        || !(gamma == width || gamma == input)
        || position > MAX_POSITION
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || (block_positions && positions < (groups - 1) * BLOCK + 1)
    {
        return Err("invalid QSA RMS/RoPE lengths or scalars");
    }
    Ok(gamma == width)
}
pub fn validate_selected(layout: QsaLayout, ids: &[u32]) -> Result<(), &'static str> {
    layout.validate()?;
    if ids.len() != layout.selected
        || ids.iter().any(|&i| i as usize >= layout.visible)
        || ids.windows(2).any(|v| v[0] >= v[1])
    {
        return Err("invalid QSA selection");
    }
    let full = ids.len() - layout.visible % BLOCK;
    for chunk in ids[..full].chunks_exact(BLOCK) {
        if !(chunk[0] as usize).is_multiple_of(BLOCK)
            || chunk
                .iter()
                .enumerate()
                .any(|(j, &v)| v != chunk[0] + j as u32)
            || chunk[3] as usize >= layout.complete * BLOCK
        {
            return Err("partial complete QSA block");
        }
    }
    for (j, &id) in ids[full..].iter().enumerate() {
        if id as usize != layout.complete * BLOCK + j {
            return Err("incorrect QSA tail");
        }
    }
    Ok(())
}
/// Scores/status remain transient. Exact deterministic core selection including
/// signed-zero ties, chronological block expansion and incomplete tail.
pub fn select_blocks(
    layout: QsaLayout,
    scores: &[f32],
    status: &[u32],
) -> Result<Vec<u32>, &'static str> {
    layout.validate()?;
    if scores.len() != layout.complete
        || status.len() != layout.complete
        || status.iter().any(|&s| s != 1)
    {
        return Err("failed QSA score stage");
    }
    let ids = ops::qsa_select_blocks(scores, layout.visible, BLOCK, BUDGET)
        .map_err(|_| "invalid QSA scores")?
        .into_iter()
        .map(|i| i as u32)
        .collect::<Vec<_>>();
    validate_selected(layout, &ids)?;
    Ok(ids)
}
/// Stage order: split, query RMS, key RMS, index-query RMS, pack, pool,
/// pooled RMS, score, logits, softmax, output. Empty pool stages are valid.
pub fn validate_token_status(layout: QsaLayout, stages: [&[u32]; 11]) -> Result<(), &'static str> {
    layout.validate()?;
    let sizes = [
        QUERY * 2,
        QH,
        KVH,
        IH,
        TOKEN,
        layout.complete,
        layout.complete,
        layout.complete,
        layout.logits_elements,
        QH,
        QUERY,
    ];
    if stages
        .iter()
        .zip(sizes)
        .any(|(s, n)| s.len() != n || s.iter().any(|&v| v != 1))
    {
        return Err("QSA token stage failed");
    }
    Ok(())
}
/// Typed logical element counts, not allocated byte capacities. These helpers
/// also prove indices for unchecked launches when grids are exactly documented.
// Each count corresponds to a separate split/pack dispatch binding.
#[allow(clippy::too_many_arguments)]
pub fn validate_projection_lengths(
    raw: usize,
    packed: usize,
    split_status: usize,
    key: usize,
    value: usize,
    raw_index_key: usize,
    token: usize,
    token_status: usize,
) -> Result<(), &'static str> {
    if [raw, packed, split_status] != [QUERY * 2; 3]
        || [key, value, raw_index_key, token, token_status] != [KV, KV, ID, TOKEN, TOKEN]
    {
        return Err("invalid QSA projection lengths");
    }
    Ok(())
}
// Each count corresponds to a separate pool/score dispatch binding.
#[allow(clippy::too_many_arguments)]
pub fn validate_pool_score_lengths(
    layout: QsaLayout,
    old: usize,
    token: usize,
    means: usize,
    pool_status: usize,
    pooled: usize,
    query: usize,
    scores: usize,
    score_status: usize,
) -> Result<(), &'static str> {
    validate_cache_lengths(layout, old, token)?;
    if [means, pooled] != [layout.complete * ID; 2]
        || query != INDEX_QUERY
        || [pool_status, scores, score_status] != [layout.complete; 3]
    {
        return Err("invalid QSA pool/score lengths");
    }
    Ok(())
}
// Each count corresponds to a separate attention dispatch binding.
#[allow(clippy::too_many_arguments)]
pub fn validate_attention_lengths(
    layout: QsaLayout,
    query: usize,
    old: usize,
    token: usize,
    selected: usize,
    logits: usize,
    logits_status: usize,
    probabilities: usize,
    softmax_status: usize,
    gate: usize,
    output: usize,
    output_status: usize,
) -> Result<(), &'static str> {
    validate_cache_lengths(layout, old, token)?;
    if [query, gate, output, output_status] != [QUERY; 4]
        || selected != layout.selected
        || [logits, logits_status, probabilities] != [layout.logits_elements; 3]
        || softmax_status != QH
    {
        return Err("invalid QSA attention lengths");
    }
    Ok(())
}

#[cube]
// Portable CubeCL comparisons; Range::contains/is_finite are host-only APIs.
#[allow(clippy::eq_op, clippy::manual_range_contains)]
fn finite(x: f32) -> bool {
    x == x && x <= f32::MAX && x >= -f32::MAX
}
#[cube]
fn logistic(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
#[cube]
fn cached(
    old: &Array<f32>,
    current: &Array<f32>,
    token: usize,
    previous: usize,
    offset: usize,
) -> f32 {
    if token < previous {
        old[token * TOKEN + offset]
    } else {
        current[offset]
    }
}

#[cube(launch)]
pub fn split_query_gate(raw: &Array<f32>, packed: &mut Array<f32>, status: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < QUERY * 2 {
        let mut d = i;
        let mut extra = 0_usize;
        if i >= QUERY {
            d = i - QUERY;
            extra = HD;
        }
        let x = raw[(d / HD) * (HD * 2) + d % HD + extra];
        packed[i] = 0.0;
        status[i] = 0;
        if finite(x) {
            packed[i] = x;
            status[i] = 1;
        }
    }
}

#[cube(launch)]
// CubeCL needs an initialized mutable local for the runtime three-way branch.
// Its native Rust body overwrites that initializer on every branch.
#[allow(unused_assignments)]
pub fn pack_token(
    key: &Array<f32>,
    value: &Array<f32>,
    raw_index_key: &Array<f32>,
    token: &mut Array<f32>,
    status: &mut Array<u32>,
) {
    let i = ABSOLUTE_POS;
    if i < TOKEN {
        let mut x = 0.0_f32;
        if i < KV {
            x = key[i];
        } else if i < KV * 2 {
            x = value[i - KV];
        } else {
            x = raw_index_key[i - KV * 2];
        }
        token[i] = 0.0;
        status[i] = 0;
        if finite(x) {
            token[i] = x;
            status[i] = 1;
        }
    }
}

/// Generic shared/per-value gamma RMS, then partial NeoX first64. For pooled
/// index keys set block_positions=true; positions[group*4] is first block pos.
/// Query/key/index-query use false and the runtime scalar current position.
#[cube(launch)]
pub fn rms_rope(
    input: &Array<f32>,
    gamma: &Array<f32>,
    sin_table: &Array<f32>,
    cos_table: &Array<f32>,
    positions: &Array<u32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    groups: usize,
    position: u32,
    epsilon: f32,
    #[comptime] width: usize,
    #[comptime] shared_gamma: bool,
    #[comptime] block_positions: bool,
) {
    let group = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut maximum = 1.0_f32;
    let mut lane_flag = 1_u32;
    let mut p = position;
    if block_positions && group < groups {
        p = positions[group * BLOCK];
    }
    if group >= groups || p > MAX_POSITION || !finite(epsilon) || epsilon <= 0.0 {
        lane_flag = 0;
    }
    if group < groups {
        for tile in 0..width / LANES {
            let d = tile * LANES + lane;
            let i = group * width + d;
            let mut gi = i;
            if shared_gamma {
                gi = d;
            }
            let x = input[i];
            if finite(x) && finite(gamma[gi]) {
                if x.abs() > maximum {
                    maximum = x.abs();
                }
            } else {
                lane_flag = 0;
            }
        }
    }
    // Explicit mutable flags avoid CubeCL0.10 constant-branch-expression bug.
    let mut flags = SharedMemory::<u32>::new(LANES);
    let mut scales = SharedMemory::<f32>::new(LANES);
    flags[lane] = lane_flag;
    scales[lane] = maximum;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            flags[lane] = flags[lane] & flags[lane + stride];
            if scales[lane + stride] > scales[lane] {
                scales[lane] = scales[lane + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    let scale = scales[0];
    let mut sum = 0.0_f32;
    if group < groups && flags[0] == 1 {
        for tile in 0..width / LANES {
            let x = input[group * width + tile * LANES + lane] / scale;
            sum += x * x;
        }
    }
    let mut sums = SharedMemory::<f32>::new(LANES);
    sums[lane] = sum;
    sync_cube();
    stride = 32;
    while stride > 0 {
        if lane < stride {
            sums[lane] = sums[lane] + sums[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    let denom = (sums[0] / width as f32 + (epsilon / scale) / scale).sqrt();
    let mut out_flag = 1_u32;
    if group < groups {
        for tile in 0..width / LANES {
            let d = tile * LANES + lane;
            let i = group * width + d;
            let mut gi = i;
            if shared_gamma {
                gi = d;
            }
            let mut y = ((input[i] / scale) / denom) * gamma[gi];
            let mut table_good = true;
            // Guard the runtime table index even after an invalid position
            // poisoned flags: unchecked launches must not load past a table.
            if d < 64 && p <= MAX_POSITION {
                let mut pair = d + 32;
                let mut frequency_index = d;
                if d >= 32 {
                    pair = d - 32;
                    frequency_index = d - 32;
                }
                let mut gpair = group * width + pair;
                if shared_gamma {
                    gpair = pair;
                }
                let mate = ((input[group * width + pair] / scale) / denom) * gamma[gpair];
                let table_index = p as usize * ROPE_PAIRS + frequency_index;
                let s = sin_table[table_index];
                let c = cos_table[table_index];
                if d < 32 {
                    y = y * c - mate * s;
                } else {
                    y = y * c + mate * s;
                }
                table_good = finite(s) && finite(c);
            }
            output[i] = 0.0;
            if flags[0] == 1 && finite(denom) && denom > 0.0 && table_good && finite(y) {
                output[i] = y;
            } else {
                out_flag = 0;
            }
        }
    }
    // Do not overwrite input flags until every lane has finished using them.
    let mut output_flags = SharedMemory::<u32>::new(LANES);
    output_flags[lane] = out_flag;
    sync_cube();
    stride = 32;
    while stride > 0 {
        if lane < stride {
            output_flags[lane] = output_flags[lane] & output_flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && group < groups {
        status[group] = output_flags[0];
    }
}

#[cube(launch)]
pub fn pool_indexer_raw(
    old: &Array<f32>,
    current: &Array<f32>,
    means: &mut Array<f32>,
    status: &mut Array<u32>,
    previous: usize,
    complete: usize,
) {
    let block = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut flag = 1_u32;
    if block < complete {
        for tile in 0..2 {
            let d = tile * LANES + lane;
            let mut sum = 0.0_f32;
            for j in 0..BLOCK {
                let t = block * BLOCK + j;
                if t <= previous {
                    let x = cached(old, current, t, previous, KV * 2 + d);
                    if finite(x) {
                        sum += x;
                    } else {
                        flag = 0;
                    }
                } else {
                    flag = 0;
                }
            }
            let y = sum / 4.0;
            means[block * ID + d] = 0.0;
            if finite(y) {
                means[block * ID + d] = y;
            } else {
                flag = 0;
            }
        }
    }
    let mut flags = SharedMemory::<u32>::new(LANES);
    flags[lane] = flag;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            flags[lane] = flags[lane] & flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && block < complete {
        status[block] = flags[0];
    }
}

#[cube(launch)]
pub fn score_blocks(
    pooled: &Array<f32>,
    query: &Array<f32>,
    scores: &mut Array<f32>,
    status: &mut Array<u32>,
    complete: usize,
) {
    let block = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut sums = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    let mut total: f32 = 0.0_f32;
    let mut valid: u32 = 1_u32;
    for head in 0..IH {
        let mut sum = 0.0_f32;
        let mut flag = 1_u32;
        if block < complete {
            for tile in 0..2 {
                let d = tile * LANES + lane;
                let x = pooled[block * ID + d];
                let q = query[head * ID + d];
                if finite(x) && finite(q) {
                    sum += x * q;
                } else {
                    flag = 0;
                }
            }
        }
        if !finite(sum) {
            flag = 0;
        }
        sums[lane] = sum;
        flags[lane] = flag;
        sync_cube();
        let mut stride = 32_usize;
        while stride > 0 {
            if lane < stride {
                sums[lane] = sums[lane] + sums[lane + stride];
                flags[lane] = flags[lane] & flags[lane + stride];
            }
            sync_cube();
            stride /= 2;
        }
        if lane == 0 {
            if sums[0] > 0.0 {
                total += sums[0];
            }
            if flags[0] != 1 || !finite(sums[0]) {
                valid = 0;
            }
        }
        // Lane0 must finish consuming the reduction before next head overwrites.
        sync_cube();
    }
    if lane == 0 && block < complete {
        let score = total / (ID as f32).sqrt();
        scores[block] = 0.0;
        status[block] = 0;
        if valid == 1 && finite(score) && score >= 0.0 {
            scores[block] = score;
            status[block] = 1;
        }
    }
}

#[cube(launch)]
pub fn attention_logits(
    query: &Array<f32>,
    old: &Array<f32>,
    current: &Array<f32>,
    selected: &Array<u32>,
    logits: &mut Array<f32>,
    status: &mut Array<u32>,
    previous: usize,
    count: usize,
) {
    let task = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let head = task / count;
    let rank = task % count;
    let token = selected[rank] as usize;
    let mut sum = 0.0_f32;
    let mut flag = 1_u32;
    if head < QH && token <= previous {
        for tile in 0..4 {
            let d = tile * LANES + lane;
            let q = query[head * HD + d];
            let k = cached(old, current, token, previous, (head / 12) * HD + d);
            if finite(q) && finite(k) {
                sum += q * k;
            } else {
                flag = 0;
            }
        }
    } else {
        flag = 0;
    }
    if !finite(sum) {
        flag = 0;
    }
    let mut sums = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    sums[lane] = sum;
    flags[lane] = flag;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            sums[lane] = sums[lane] + sums[lane + stride];
            flags[lane] = flags[lane] & flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && head < QH {
        let y = sums[0] / 16.0;
        logits[task] = 0.0;
        status[task] = 0;
        if flags[0] == 1 && finite(y) {
            logits[task] = y;
            status[task] = 1;
        }
    }
}

#[cube(launch)]
// CubeCL cannot lower RangeInclusive::contains for runtime shader values.
#[allow(clippy::manual_range_contains)]
pub fn attention_softmax(
    logits: &Array<f32>,
    probability: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
) {
    let head = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    // Layout guarantees count>=1; avoid a macro constant-only negative unary.
    let mut maximum = logits[head * count];
    let mut flag = 1_u32;
    if !finite(maximum) {
        flag = 0;
    }
    let mut rank = lane;
    while rank < count {
        let x = logits[head * count + rank];
        if finite(x) {
            if x > maximum {
                maximum = x;
            }
        } else {
            flag = 0;
        }
        rank += LANES;
    }
    let mut maxima = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    maxima[lane] = maximum;
    flags[lane] = flag;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            if maxima[lane + stride] > maxima[lane] {
                maxima[lane] = maxima[lane + stride];
            }
            flags[lane] = flags[lane] & flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    let max = maxima[0];
    let mut sum = 0.0_f32;
    rank = lane;
    while rank < count {
        sum += (logits[head * count + rank] - max).exp();
        rank += LANES;
    }
    let mut sums = SharedMemory::<f32>::new(LANES);
    sums[lane] = sum;
    sync_cube();
    stride = 32;
    while stride > 0 {
        if lane < stride {
            sums[lane] = sums[lane] + sums[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    let denominator = sums[0];
    let mut output_flag = 1_u32;
    rank = lane;
    while rank < count {
        let i = head * count + rank;
        let y = (logits[i] - max).exp() / denominator;
        probability[i] = 0.0;
        if flags[0] == 1
            && finite(denominator)
            && denominator > 0.0
            && finite(y)
            && y >= 0.0
            && y <= 1.0
        {
            probability[i] = y;
        } else {
            output_flag = 0;
        }
        rank += LANES;
    }
    let mut out_flags = SharedMemory::<u32>::new(LANES);
    out_flags[lane] = output_flag;
    sync_cube();
    stride = 32;
    while stride > 0 {
        if lane < stride {
            out_flags[lane] = out_flags[lane] & out_flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        status[head] = out_flags[0];
    }
}

#[cube(launch)]
// CubeCL cannot lower RangeInclusive::contains for runtime shader values.
#[allow(clippy::manual_range_contains)]
pub fn attention_values_gated(
    old: &Array<f32>,
    current: &Array<f32>,
    probability: &Array<f32>,
    selected: &Array<u32>,
    raw_gate: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    previous: usize,
    count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < QUERY {
        let head = i / HD;
        let dim = i % HD;
        let mut sum = 0.0_f32;
        let mut valid = true;
        for rank in 0..count {
            let t = selected[rank] as usize;
            let p = probability[head * count + rank];
            if t <= previous {
                let v = cached(old, current, t, previous, KV + (head / 12) * HD + dim);
                if finite(v) && finite(p) && p >= 0.0 && p <= 1.0 {
                    sum += p * v;
                } else {
                    valid = false;
                }
            } else {
                valid = false;
            }
        }
        let gate = raw_gate[i];
        let y = sum * logistic(gate);
        output[i] = 0.0;
        status[i] = 0;
        if valid && finite(gate) && finite(sum) && finite(y) {
            output[i] = y;
            status[i] = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32], abs: f32, rel: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.is_finite() && y.is_finite() && (x - y).abs() <= abs + rel * y.abs(),
                "at{i}: actual{x} expected{y}"
            );
        }
    }
    fn tree(mut lanes: [f32; LANES]) -> f32 {
        let mut stride = 32;
        while stride > 0 {
            for i in 0..stride {
                lanes[i] += lanes[i + stride];
            }
            stride /= 2;
        }
        lanes[0]
    }
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut lanes = [0.0; LANES];
        for (lane, partial) in lanes.iter_mut().enumerate() {
            for d in (lane..a.len()).step_by(LANES) {
                *partial += a[d] * b[d];
            }
        }
        tree(lanes)
    }
    fn norm_rope(input: &[f32], gamma: &[f32], width: usize, positions: &[u32]) -> Vec<f32> {
        let (sin_table, cos_table) = rope_tables();
        let mut out = vec![0.0; input.len()];
        for (h, head) in input.chunks_exact(width).enumerate() {
            let scale = head.iter().fold(1.0_f32, |s, x| s.max(x.abs()));
            let mut lanes = [0.0; LANES];
            for (lane, partial) in lanes.iter_mut().enumerate() {
                for d in (lane..width).step_by(LANES) {
                    let x = head[d] / scale;
                    *partial += x * x;
                }
            }
            let denom = (tree(lanes) / width as f32 + (EPSILON / scale) / scale).sqrt();
            let normalized = (0..width)
                .map(|d| {
                    ((head[d] / scale) / denom)
                        * gamma[if gamma.len() == width {
                            d
                        } else {
                            h * width + d
                        }]
                })
                .collect::<Vec<_>>();
            for d in 0..width {
                let mut y = normalized[d];
                if d < 64 {
                    let (index, pair) = if d < 32 {
                        (d, d + 32)
                    } else {
                        (d - 32, d - 32)
                    };
                    let table_index = positions[h] as usize * ROPE_PAIRS + index;
                    let s = sin_table[table_index];
                    let c = cos_table[table_index];
                    y = if d < 32 {
                        y * c - normalized[pair] * s
                    } else {
                        y * c + normalized[pair] * s
                    };
                }
                out[h * width + d] = y;
            }
        }
        out
    }
    fn oracle_norm_rope(input: &[f32], gamma: &[f32], width: usize, positions: &[u32]) -> Vec<f32> {
        input
            .chunks_exact(width)
            .enumerate()
            .flat_map(|(h, head)| {
                let g = if gamma.len() == width {
                    gamma
                } else {
                    &gamma[h * width..(h + 1) * width]
                };
                let mut n = ops::rms_norm(head, g, width, EPSILON).unwrap();
                ops::rope_neox_heads(&mut n, width, 64, positions[h], THETA).unwrap();
                n
            })
            .collect()
    }
    type CacheFixture = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    fn public_cache(visible: usize) -> CacheFixture {
        let mut packed = vec![0.0; visible * TOKEN];
        let mut keys = vec![];
        let mut values = vec![];
        let mut raw = vec![];
        for t in 0..visible {
            let k = (0..KV)
                .map(|d| (((d * 3 + t * 7) % 29) as f32 - 14.0) / 32.0)
                .collect::<Vec<_>>();
            let v = (0..KV)
                .map(|d| (((d * 5 + t * 11) % 37) as f32 - 18.0) / 16.0)
                .collect::<Vec<_>>();
            let r = (0..ID)
                .map(|d| (((d * 7 + t * t * 13) % 41) as f32 - 20.0) / 16.0)
                .collect::<Vec<_>>();
            packed[t * TOKEN..t * TOKEN + KV].copy_from_slice(&k);
            packed[t * TOKEN + KV..t * TOKEN + KV * 2].copy_from_slice(&v);
            packed[t * TOKEN + KV * 2..(t + 1) * TOKEN].copy_from_slice(&r);
            keys.extend(k);
            values.extend(v);
            raw.extend(r);
        }
        let candidate = packed.split_off((visible - 1) * TOKEN);
        (packed, candidate, keys, values, raw)
    }
    fn get(old: &[f32], current: &[f32], token: usize, previous: usize, offset: usize) -> f32 {
        if token < previous {
            old[token * TOKEN + offset]
        } else {
            current[offset]
        }
    }
    fn emulate_scores(
        layout: QsaLayout,
        old: &[f32],
        current: &[f32],
        iq: &[f32],
        gamma: &[f32],
        pos: &[u32],
    ) -> Vec<f32> {
        let mut means = vec![0.0; layout.complete * ID];
        for b in 0..layout.complete {
            for d in 0..ID {
                let mut sum = 0.0;
                for j in 0..BLOCK {
                    sum += get(old, current, b * BLOCK + j, layout.previous, KV * 2 + d);
                }
                means[b * ID + d] = sum / 4.0;
            }
        }
        let first = pos
            .iter()
            .step_by(BLOCK)
            .copied()
            .take(layout.complete)
            .collect::<Vec<_>>();
        let pooled = norm_rope(&means, gamma, ID, &first);
        pooled
            .chunks_exact(ID)
            .map(|p| {
                let mut score = 0.0;
                for q in iq.chunks_exact(ID) {
                    score += dot(q, p).max(0.0);
                }
                score / (ID as f32).sqrt()
            })
            .collect()
    }
    fn emulate_attend(
        layout: QsaLayout,
        q: &[f32],
        old: &[f32],
        current: &[f32],
        ids: &[u32],
        gate: &[f32],
    ) -> Vec<f32> {
        validate_selected(layout, ids).unwrap();
        let count = ids.len();
        let mut output = vec![0.0; QUERY];
        for h in 0..QH {
            let logits = ids
                .iter()
                .map(|&t| {
                    let k = (0..HD)
                        .map(|d| get(old, current, t as usize, layout.previous, (h / 12) * HD + d))
                        .collect::<Vec<_>>();
                    dot(&q[h * HD..(h + 1) * HD], &k) / 16.0
                })
                .collect::<Vec<_>>();
            let max = logits.iter().copied().fold(logits[0], f32::max);
            let exp = logits.iter().map(|&x| (x - max).exp()).collect::<Vec<_>>();
            let mut lanes = [0.0; LANES];
            for (lane, partial) in lanes.iter_mut().enumerate() {
                for r in (lane..count).step_by(LANES) {
                    *partial += exp[r];
                }
            }
            let denom = tree(lanes);
            let p = exp.iter().map(|&x| x / denom).collect::<Vec<_>>();
            for d in 0..HD {
                let mut sum = 0.0;
                for r in 0..count {
                    sum += p[r]
                        * get(
                            old,
                            current,
                            ids[r] as usize,
                            layout.previous,
                            KV + (h / 12) * HD + d,
                        );
                }
                output[h * HD + d] = sum * logistic(gate[h * HD + d]);
            }
        }
        output
    }

    #[test]
    fn split_interleaved_head_boundaries_and_sigmoid() {
        let raw = (0..QUERY * 2).map(|i| i as f32).collect::<Vec<_>>();
        let packed = (0..QUERY * 2)
            .map(|i| {
                let d = i % QUERY;
                raw[(d / HD) * HD * 2 + d % HD + if i >= QUERY { HD } else { 0 }]
            })
            .collect::<Vec<_>>();
        for h in 0..QH {
            assert_eq!(packed[h * HD], (h * HD * 2) as f32);
            assert_eq!(packed[QUERY + h * HD], (h * HD * 2 + HD) as f32);
            assert_eq!(packed[h * HD + 255], (h * HD * 2 + 255) as f32);
        }
        for x in [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0] {
            assert_eq!(logistic(x).to_bits(), ops::sigmoid(x).to_bits());
        }
    }
    #[test]
    fn mixed_gamma_rms_partial_neox_matches_core_at_runtime_positions() {
        for width in [ID, HD] {
            let positions = [0, 1, 7, 4095];
            let x = (0..width * 4)
                .map(|i| ((i % 37) as f32 - 18.0) / 16.0)
                .collect::<Vec<_>>();
            for gamma in [
                (0..width)
                    .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
                    .collect::<Vec<_>>(),
                (0..width * 4)
                    .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
                    .collect::<Vec<_>>(),
            ] {
                let a = norm_rope(&x, &gamma, width, &positions);
                let b = oracle_norm_rope(&x, &gamma, width, &positions);
                close(&a, &b, 5e-7, 4e-6);
                let plain = norm_rope(&x, &gamma, width, &[0; 4]);
                for h in 0..4 {
                    assert_eq!(
                        &a[h * width + 64..(h + 1) * width],
                        &plain[h * width + 64..(h + 1) * width]
                    );
                }
            }
        }
    }
    #[test]
    fn raw_key_pool_then_rms_first_stored_position_and_head_rectification() {
        let gamma = (0..ID)
            .map(|i| 0.5 + (i % 7) as f32 / 16.0)
            .collect::<Vec<_>>();
        let iq = (0..INDEX_QUERY)
            .map(|i| ((i % 19) as f32 - 9.0) / 16.0)
            .collect::<Vec<_>>();
        for visible in [4, 5, 8, 9] {
            let l = QsaLayout::new(visible).unwrap();
            let (old, current, _, _, raw) = public_cache(visible);
            let preserved = old.clone();
            let pos = (0..visible)
                .map(|i| (i * i + 3 * i) as u32)
                .collect::<Vec<_>>();
            let actual = emulate_scores(l, &old, &current, &iq, &gamma, &pos);
            let expected =
                ops::qsa_block_scores(&raw, &iq, &gamma, &pos, visible, ops::QsaShape::qwen38())
                    .unwrap();
            close(&actual, &expected, 2e-6, 1e-5);
            assert_eq!(old, preserved);
        }
        let mut pooled = vec![0.0; ID];
        pooled[0] = 1.0;
        let dots = [2.0, -3.0, 1.0, -4.0].map(|x| {
            let mut q = vec![0.0; ID];
            q[0] = x;
            dot(&q, &pooled)
        });
        assert_eq!(dots.iter().map(|x| x.max(0.0)).sum::<f32>(), 3.0);
        assert_eq!(dots.iter().sum::<f32>().max(0.0), 0.0);
    }
    #[test]
    fn exact_full_block_budget_signed_zero_ties_and_tail() {
        for visible in [1, 3, 4, 5, 2048, 2051, 2052, 4095, 4096] {
            let l = QsaLayout::new(visible).unwrap();
            let scores = (0..l.complete)
                .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let ids = select_blocks(l, &scores, &vec![1; l.complete]).unwrap();
            validate_selected(l, &ids).unwrap();
            let full = l.complete.min(512) * 4;
            assert_eq!(&ids[..full], &(0..full as u32).collect::<Vec<_>>());
            assert_eq!(
                &ids[full..],
                &(l.complete as u32 * 4..visible as u32).collect::<Vec<_>>()
            );
        }
        let l = QsaLayout::new(4096).unwrap();
        let scores = (0..l.complete).map(|i| i as f32).collect::<Vec<_>>();
        assert_eq!(
            select_blocks(l, &scores, &vec![1; l.complete]).unwrap(),
            (2048..4096).collect::<Vec<_>>()
        );
        let l = QsaLayout::new(2051).unwrap();
        assert_eq!(l.selected, 2051);
    }
    #[test]
    fn attention_uses_h_div_12_and_all_current_tail_values_before_sigmoid_gate() {
        let l = QsaLayout::new(2).unwrap();
        let (mut old, mut current, mut keys, mut values, _) = public_cache(2);
        keys.fill(0.0);
        old[..KV].fill(0.0);
        current[..KV].fill(0.0);
        for t in 0..2 {
            for h in 0..KVH {
                for d in 0..HD {
                    let x = if h == 0 {
                        1.0 + 2.0 * t as f32
                    } else {
                        10.0 + 20.0 * t as f32
                    };
                    values[t * KV + h * HD + d] = x;
                    if t == 0 {
                        old[KV + h * HD + d] = x
                    } else {
                        current[KV + h * HD + d] = x
                    }
                }
            }
        }
        let q = vec![0.0; QUERY];
        let gate = vec![0.0; QUERY];
        let ids = select_blocks(l, &[], &[]).unwrap();
        let out = emulate_attend(l, &q, &old, &current, &ids, &gate);
        assert!(out[..12 * HD].iter().all(|&x| x == 1.0));
        assert!(out[12 * HD..].iter().all(|&x| x == 10.0));
        let expected = ops::qsa_attend(&q, &keys, &values, &[0, 1], 2, ops::QsaShape::qwen38())
            .unwrap()
            .iter()
            .map(|x| x * 0.5)
            .collect::<Vec<_>>();
        assert_eq!(out, expected);
        for visible in [3, 5, 9] {
            let l = QsaLayout::new(visible).unwrap();
            let (old, current, keys, values, _) = public_cache(visible);
            let preserved = old.clone();
            let q = (0..QUERY)
                .map(|i| ((i % 23) as f32 - 11.0) / 32.0)
                .collect::<Vec<_>>();
            let gate = (0..QUERY)
                .map(|i| ((i % 13) as f32 - 6.0) / 4.0)
                .collect::<Vec<_>>();
            let ids = select_blocks(l, &vec![0.0; l.complete], &vec![1; l.complete]).unwrap();
            let selected = ids.iter().map(|&i| i as usize).collect::<Vec<_>>();
            let expected = ops::qsa_attend(
                &q,
                &keys,
                &values,
                &selected,
                visible,
                ops::QsaShape::qwen38(),
            )
            .unwrap()
            .iter()
            .zip(&gate)
            .map(|(&x, &g)| x * ops::sigmoid(g))
            .collect::<Vec<_>>();
            close(
                &emulate_attend(l, &q, &old, &current, &ids, &gate),
                &expected,
                6e-7,
                2e-5,
            );
            assert_eq!(old, preserved);
        }
    }
    #[test]
    fn contracts_reject_bad_lengths_positions_selection_and_any_stage_failure() {
        let l = QsaLayout::new(5).unwrap();
        assert!(QsaLayout::new(0).is_err());
        assert!(QsaLayout::new(4097).is_err());
        validate_positions(l, &[0, 1, 4, 9, 16]).unwrap();
        for p in [[0, 1, 1, 9, 16], [0, 1, 4, 9, MAX_POSITION + 1]] {
            assert!(validate_positions(l, &p).is_err());
        }
        validate_cache_lengths(l, l.old_elements, TOKEN).unwrap();
        assert!(validate_cache_lengths(l, l.old_elements - 1, TOKEN).is_err());
        validate_norm_lengths(
            QUERY,
            HD,
            ROPE_TABLE_WORDS,
            ROPE_TABLE_WORDS,
            0,
            QUERY,
            QH,
            QH,
            HD,
            false,
            4095,
            EPSILON,
        )
        .unwrap();
        assert!(
            validate_norm_lengths(
                QUERY,
                HD - 1,
                ROPE_TABLE_WORDS,
                ROPE_TABLE_WORDS,
                0,
                QUERY,
                QH,
                QH,
                HD,
                false,
                0,
                EPSILON
            )
            .is_err()
        );
        assert!(
            validate_norm_lengths(
                ID * 2,
                ID,
                ROPE_TABLE_WORDS,
                ROPE_TABLE_WORDS,
                4,
                ID * 2,
                2,
                2,
                ID,
                true,
                0,
                EPSILON
            )
            .is_err()
        );
        for ids in [[0, 1, 1, 3, 4], [0, 1, 2, 4, 3], [0, 1, 2, 3, 5]] {
            assert!(validate_selected(l, &ids).is_err());
        }
        assert!(select_blocks(l, &[f32::NAN], &[1]).is_err());
        assert!(select_blocks(l, &[-1.0], &[1]).is_err());
        assert!(select_blocks(l, &[0.0], &[0]).is_err());
        let sizes = [
            QUERY * 2,
            QH,
            KVH,
            IH,
            TOKEN,
            l.complete,
            l.complete,
            l.complete,
            l.logits_elements,
            QH,
            QUERY,
        ];
        let mut stages = sizes.map(|n| vec![1; n]);
        validate_token_status(l, std::array::from_fn(|i| stages[i].as_slice())).unwrap();
        for i in 0..11 {
            stages[i][0] = 0;
            assert!(
                validate_token_status(l, std::array::from_fn(|j| stages[j].as_slice())).is_err()
            );
            stages[i][0] = 1;
        }
        let forged = QsaLayout { selected: 4, ..l };
        assert!(forged.validate().is_err());
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!finite(x));
        }
        assert!(!finite(dot(&[f32::MAX; 128], &[2.0; 128])));
    }

    // Regression for CubeCL0.10: `flags[lane] = if valid {1} else {0}`
    // silently lowered to empty branches then unconditional0. Do not use it.
    // This deliberately starts0 and conditionally sets1, mirroring GDN's fix.
    #[cube(launch)]
    fn flag_regression(input: &Array<u32>, output: &mut Array<u32>) {
        let lane = UNIT_POS_X as usize;
        let mut flag = 0_u32;
        if input[lane] == 1 {
            flag = 1_u32;
        }
        let mut flags = SharedMemory::<u32>::new(LANES);
        flags[lane] = flag;
        sync_cube();
        output[lane] = flags[lane];
    }
    #[test]
    fn public_tables_are_bit_exact_core_angles_all_pairs_and_reused() {
        let (sin, cos) = rope_tables();
        let (same_sin, same_cos) = rope_tables();
        assert!(std::ptr::eq(sin, same_sin) && std::ptr::eq(cos, same_cos));
        assert_eq!(sin.len(), 131072);
        assert_eq!(cos.len(), 131072);
        for position in 0..MAX_CONTEXT {
            for pair in 0..ROPE_PAIRS {
                // Independent direct core formula, including its f32 angle.
                let angle = position as f32 * THETA.powf(-(pair as f32) / 32.0);
                let (s, c) = angle.sin_cos();
                let i = position * ROPE_PAIRS + pair;
                assert_eq!(sin[i].to_bits(), s.to_bits());
                assert_eq!(cos[i].to_bits(), c.to_bits());
            }
        }
        assert!(
            validate_norm_lengths(
                QUERY,
                HD,
                ROPE_TABLE_WORDS,
                ROPE_TABLE_WORDS,
                0,
                QUERY,
                QH,
                QH,
                HD,
                false,
                4096,
                EPSILON
            )
            .is_err()
        );
        assert!(
            validate_norm_lengths(
                QUERY,
                HD,
                ROPE_TABLE_WORDS - 1,
                ROPE_TABLE_WORDS,
                0,
                QUERY,
                QH,
                QH,
                HD,
                false,
                0,
                EPSILON
            )
            .is_err()
        );
        assert!(
            validate_norm_lengths(
                QUERY,
                HD,
                ROPE_TABLE_WORDS,
                ROPE_TABLE_WORDS - 1,
                0,
                QUERY,
                QH,
                QH,
                HD,
                false,
                0,
                EPSILON
            )
            .is_err()
        );
    }
    #[test]
    fn all_operators_generate_checked_unchecked_wgsl_and_success_branches() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        for variant in 0..12 {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let mut b = KernelBuilder::default();
                b.scope.register_type::<u32>(uint);
                b.scope.register_type::<usize>(uint);
                b.scope.register_type::<f32>(float);
                b.scope
                    .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                match variant {
                    0 => {
                        let x = b.input_array(Type::new(float)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        split_query_gate::expand(&mut b.scope, x, out, s);
                    }
                    1 => {
                        let k = b.input_array(Type::new(float)).into();
                        let v = b.input_array(Type::new(float)).into();
                        let ik = b.input_array(Type::new(float)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        pack_token::expand(&mut b.scope, k, v, ik, out, s);
                    }
                    2..=5 => {
                        let x = b.input_array(Type::new(float)).into();
                        let g = b.input_array(Type::new(float)).into();
                        let sin = b.input_array(Type::new(float)).into();
                        let cos = b.input_array(Type::new(float)).into();
                        let p = b.input_array(Type::new(uint)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        let groups = b.scalar(uint).into();
                        let pos = b.scalar(uint).into();
                        let eps = b.scalar(float).into();
                        let width = if variant < 4 { HD } else { ID };
                        let shared = variant != 3;
                        let block_pos = variant == 5;
                        rms_rope::expand(
                            &mut b.scope,
                            x,
                            g,
                            sin,
                            cos,
                            p,
                            out,
                            s,
                            groups,
                            pos,
                            eps,
                            width,
                            shared,
                            block_pos,
                        );
                    }
                    6 => {
                        let old = b.input_array(Type::new(float)).into();
                        let current = b.input_array(Type::new(float)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        let previous = b.scalar(uint).into();
                        let complete = b.scalar(uint).into();
                        pool_indexer_raw::expand(
                            &mut b.scope,
                            old,
                            current,
                            out,
                            s,
                            previous,
                            complete,
                        );
                    }
                    7 => {
                        let p = b.input_array(Type::new(float)).into();
                        let q = b.input_array(Type::new(float)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        let complete = b.scalar(uint).into();
                        score_blocks::expand(&mut b.scope, p, q, out, s, complete);
                    }
                    8 => {
                        let q = b.input_array(Type::new(float)).into();
                        let old = b.input_array(Type::new(float)).into();
                        let current = b.input_array(Type::new(float)).into();
                        let ids = b.input_array(Type::new(uint)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        let previous = b.scalar(uint).into();
                        let count = b.scalar(uint).into();
                        attention_logits::expand(
                            &mut b.scope,
                            q,
                            old,
                            current,
                            ids,
                            out,
                            s,
                            previous,
                            count,
                        );
                    }
                    9 => {
                        let x = b.input_array(Type::new(float)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        let count = b.scalar(uint).into();
                        attention_softmax::expand(&mut b.scope, x, out, s, count);
                    }
                    10 => {
                        let old = b.input_array(Type::new(float)).into();
                        let current = b.input_array(Type::new(float)).into();
                        let p = b.input_array(Type::new(float)).into();
                        let ids = b.input_array(Type::new(uint)).into();
                        let gate = b.input_array(Type::new(float)).into();
                        let out = b.output_array(Type::new(float)).into();
                        let s = b.output_array(Type::new(uint)).into();
                        let previous = b.scalar(uint).into();
                        let count = b.scalar(uint).into();
                        attention_values_gated::expand(
                            &mut b.scope,
                            old,
                            current,
                            p,
                            ids,
                            gate,
                            out,
                            s,
                            previous,
                            count,
                        );
                    }
                    _ => {
                        let x = b.input_array(Type::new(uint)).into();
                        let out = b.output_array(Type::new(uint)).into();
                        flag_regression::expand(&mut b.scope, x, out);
                    }
                }
                let settings = KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(64))
                    .kernel_name("qsa_offline");
                let wgsl = WgslCompiler::default()
                    .compile(
                        b.build(settings),
                        &WgpuCompilationOptions::default(),
                        mode,
                        uint,
                    )
                    .unwrap()
                    .to_string();
                assert!(wgsl.contains("@compute"), "variant{variant}");
                if variant >= 2 && variant != 10 {
                    let prefix = wgsl.split("workgroupBarrier();").next().unwrap();
                    assert!(
                        prefix.contains("u32(1)"),
                        "success flag vanished variant{variant}: {prefix}"
                    );
                }
                if variant == 11 {
                    let prefix = wgsl.split("workgroupBarrier();").next().unwrap();
                    // The original broken ternary generated `if ... {} else {}`
                    // and ONLY `shared_memory_0[lane] = u32(0);` in this prefix.
                    assert!(
                        prefix.contains("= u32(1);"),
                        "conditional success assignment vanished: {prefix}"
                    );
                    assert!(prefix.contains("= u32(0);"));
                    let one = prefix.rfind("= u32(1);").unwrap();
                    let store = prefix.rfind("shared_memory_").expect("shared flag store");
                    assert!(
                        prefix[..one].rfind("if ").is_some() && one < store,
                        "success must be assigned inside a branch before shared store: {prefix}"
                    );
                }
                if (2..=5).contains(&variant) {
                    assert!(
                        !wgsl.contains("sin(") && !wgsl.contains("cos("),
                        "RoPE must only index public tables"
                    );
                    assert!(
                        wgsl.contains("u32(4095)"),
                        "runtime position must be bounded before table loads"
                    );
                    assert!(
                        wgsl.contains("info.scalars_u32"),
                        "positions/groups must remain runtime"
                    );
                }
                // Optional diagnostic is static shader source only, no request data.
                if std::env::var_os("KUNO_QSA_DUMP_STATIC_WGSL").is_some() {
                    std::fs::write(format!("/tmp/kuno-qsa-{variant}-{mode:?}.wgsl"), wgsl).unwrap();
                }
            }
        }
    }
}
