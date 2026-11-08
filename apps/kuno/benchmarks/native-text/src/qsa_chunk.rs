//! MAX8 causal QSA candidate after projections; fixed public Qwen geometry.
//! All addresses/counts are runtime. Committed KV/raw IK remains read-only.
//! Each plane sees committed+t+1 tokens and only its complete raw-key groups.
//! No sequence/cache commit is provided. Accept every stage for the WHOLE chunk
//! only after ordinary completion, cancellation and checked owning cleanup.
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    qsa_gpu as qsa,
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::Zeroizing;
pub const MAX_BATCH: usize = 8;
pub const N: usize = 36;
pub const RAWQ: usize = 0;
pub const RAWK: usize = 1;
pub const VALUE: usize = 2;
pub const RAWIQ: usize = 3;
pub const RAWIK: usize = 4;
pub const OLD: usize = 5;
pub const POS: usize = 6;
pub const GQ: usize = 7;
pub const GK: usize = 8;
pub const GI: usize = 9;
pub const GP: usize = 10;
pub const SIN: usize = 11;
pub const COS: usize = 12;
pub const PACK: usize = 13;
pub const Q: usize = 14;
pub const K: usize = 15;
pub const IQ: usize = 16;
pub const CAND: usize = 17;
pub const MEANS: usize = 18;
pub const POOL: usize = 19;
pub const SCORE: usize = 20;
pub const SELECT: usize = 21;
pub const LOGIT: usize = 22;
pub const PROB: usize = 23;
pub const OUT: usize = 24;
pub const SPLIT_S: usize = 25;
pub const Q_S: usize = 26;
pub const K_S: usize = 27;
pub const IQ_S: usize = 28;
pub const CAND_S: usize = 29;
pub const POOL_S: usize = 30;
pub const PN_S: usize = 31;
pub const SCORE_S: usize = 32;
pub const LOGIT_S: usize = 33;
pub const SOFT_S: usize = 34;
pub const OUT_S: usize = 35;
const LANES: usize = 64;
fn fault() -> Error {
    Error::InvalidFormat("QSA chunk invalid layout or result")
}
#[derive(Clone, Copy)]
pub struct Span {
    pub base: usize,
    pub stride: usize,
    pub words: usize,
}
impl Span {
    pub fn validate(self, rows: usize, width: usize) -> Result<()> {
        let end = rows
            .checked_sub(1)
            .and_then(|n| n.checked_mul(self.stride))
            .and_then(|n| n.checked_add(self.base))
            .and_then(|n| n.checked_add(width))
            .ok_or(Error::ArithmeticOverflow)?;
        if width == 0
            || self.base > u32::MAX as usize
            || self.stride > u32::MAX as usize
            || self.stride < width
            || self.words > u32::MAX as usize
            || end > self.words
        {
            return Err(fault());
        }
        BufferLayout::new(self.words).map_err(|_| fault())?;
        Ok(())
    }
    pub fn at(self, row: usize) -> usize {
        self.base + row * self.stride
    }
}
#[derive(Clone, Copy)]
pub struct Spans {
    pub count: usize,
    pub previous: usize,
    pub capacity: usize,
    pub p: [Span; N],
}
impl Spans {
    pub fn widths(self) -> [usize; N] {
        let blocks = self.capacity / 4;
        let sel = qsa::BUDGET + 3;
        [
            qsa::QUERY * 2,
            qsa::KV,
            qsa::KV,
            qsa::INDEX_QUERY,
            qsa::ID,
            qsa::TOKEN,
            1,
            qsa::HD,
            qsa::HD,
            qsa::ID,
            qsa::ID,
            qsa::ROPE_PAIRS,
            qsa::ROPE_PAIRS,
            qsa::QUERY * 2,
            qsa::QUERY,
            qsa::KV,
            qsa::INDEX_QUERY,
            qsa::TOKEN,
            blocks * qsa::ID,
            blocks * qsa::ID,
            blocks,
            sel,
            qsa::QH * sel,
            qsa::QH * sel,
            qsa::QUERY,
            qsa::QUERY * 2,
            qsa::QH,
            qsa::KVH,
            qsa::IH,
            qsa::TOKEN,
            blocks,
            blocks,
            blocks,
            qsa::QH * sel,
            qsa::QH,
            qsa::QUERY,
        ]
    }
    pub fn rows(self) -> [usize; N] {
        let mut r = [MAX_BATCH; N];
        r[OLD] = self.capacity;
        r[POS] = self.capacity;
        r[GQ] = qsa::QH;
        r[GK] = qsa::KVH;
        r[GI] = qsa::IH;
        r[GP] = 1;
        r[SIN] = qsa::MAX_CONTEXT;
        r[COS] = qsa::MAX_CONTEXT;
        r
    }
    pub fn validate(self) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&self.count)
            || !(4..=qsa::MAX_CONTEXT).contains(&self.capacity)
            || self
                .previous
                .checked_add(self.count)
                .is_none_or(|n| n > self.capacity)
        {
            return Err(fault());
        }
        for ((p, r), w) in self.p.into_iter().zip(self.rows()).zip(self.widths()) {
            p.validate(r, w)?;
        }
        Ok(())
    }
    pub fn layout(self, t: usize) -> Result<qsa::QsaLayout> {
        if t >= self.count {
            return Err(fault());
        }
        qsa::QsaLayout::new(self.previous + t + 1).map_err(|_| fault())
    }
    pub fn validate_positions(self, positions: &[u32]) -> Result<()> {
        self.validate()?;
        if positions.len() != self.p[POS].words {
            return Err(fault());
        }
        let mut last = None;
        for t in 0..self.previous + self.count {
            let p = positions[self.p[POS].at(t)];
            if p > qsa::MAX_POSITION || last.is_some_and(|x| x >= p) {
                return Err(fault());
            }
            last = Some(p);
        }
        Ok(())
    }
}
pub struct Buffers(pub [BufferId; N]);
pub fn validate_pages(arena: &PrivateArena, ids: &[BufferId], words: &[usize]) -> Result<()> {
    if ids.is_empty() || ids.len() != words.len() {
        return Err(fault());
    }
    let client = arena.client().map_err(|_| fault())?;
    for (i, &id) in ids.iter().enumerate() {
        let h = arena.handle(id).map_err(|_| fault())?;
        if ids[..i].contains(&id)
            || arena.layout(id).map_err(|_| fault())?.logical_words != words[i]
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.stream != arena.handle(ids[0]).map_err(|_| fault())?.stream
        {
            return Err(fault());
        }
    }
    let mut resources = Vec::new();
    resources
        .try_reserve_exact(ids.len())
        .map_err(|_| fault())?;
    for &id in ids {
        resources.push(
            client
                .get_resource(arena.handle(id).map_err(|_| fault())?.clone())
                .map_err(|_| fault())?,
        );
    }
    for (i, r) in resources.iter().enumerate() {
        let h = arena.handle(ids[i]).map_err(|_| fault())?;
        let v = r.resource();
        if v.offset != 0
            || v.size != h.size()
            || v.buffer.size() != h.size()
            || resources[..i]
                .iter()
                .any(|x| x.resource().buffer == v.buffer)
        {
            return Err(fault());
        }
    }
    Ok(())
}
pub fn validate_bindings(arena: &PrivateArena, s: Spans, b: &Buffers) -> Result<()> {
    s.validate()?;
    validate_pages(arena, &b.0, &s.p.map(|p| p.words))
}

fn cancelled(c: &AtomicBool) -> Result<()> {
    if c.load(Ordering::Acquire) {
        Err(Error::InvalidFormat("QSA chunk cancelled"))
    } else {
        Ok(())
    }
}
fn arg(arena: &PrivateArena, b: &Buffers, i: usize) -> Result<ArrayArg<WgpuRuntime>> {
    let l = arena.layout(b.0[i]).map_err(|_| fault())?;
    Ok(unsafe {
        ArrayArg::from_raw_parts(
            arena.handle(b.0[i]).map_err(|_| fault())?.clone(),
            l.owned_words,
        )
    })
}
/// Selection is derived from each completed score plane by the original core
/// stable selector. Guard starts before allocation; no IDs or scores logged.
pub fn select(
    s: Spans,
    scores: &[f32],
    status: &[u32],
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<u32>>> {
    s.validate()?;
    cancelled(cancel)?;
    for (i, n) in [(SCORE, scores.len()), (SCORE_S, status.len())] {
        if n != BufferLayout::new(s.p[i].words)
            .map_err(|_| fault())?
            .owned_words
        {
            return Err(fault());
        }
    }
    let mut ids = Zeroizing::new(vec![0; s.p[SELECT].words]);
    for t in 0..s.count {
        let l = s.layout(t)?;
        let values = &scores[s.p[SCORE].at(t)..][..l.complete];
        let flags = &status[s.p[SCORE_S].at(t)..][..l.complete];
        let selection = guarded_select(l, values, flags)?;
        ids[s.p[SELECT].at(t)..][..l.selected].copy_from_slice(&selection);
    }
    cancelled(cancel)?;
    Ok(ids)
}
pub fn validate_selected(s: Spans, ids: &[u32]) -> Result<()> {
    s.validate()?;
    if ids.len() != s.p[SELECT].words {
        return Err(fault());
    }
    for t in 0..s.count {
        let l = s.layout(t)?;
        qsa::validate_selected(l, &ids[s.p[SELECT].at(t)..][..l.selected]).map_err(|_| fault())?;
    }
    Ok(())
}
/// All11 stage planes, plus active output and candidate finiteness. Full owning
/// downloads are mandatory; success carries no commit authority.
pub fn validate_download(
    s: Spans,
    data: [&[f32]; 2],
    status: [&[u32]; 11],
    cancel: &AtomicBool,
) -> Result<()> {
    s.validate()?;
    cancelled(cancel)?;
    for (i, v) in [CAND, OUT].into_iter().zip(data) {
        if v.len()
            != BufferLayout::new(s.p[i].words)
                .map_err(|_| fault())?
                .owned_words
        {
            return Err(fault());
        }
    }
    for (i, v) in (SPLIT_S..N).zip(status) {
        if v.len()
            != BufferLayout::new(s.p[i].words)
                .map_err(|_| fault())?
                .owned_words
        {
            return Err(fault());
        }
    }
    for t in 0..s.count {
        let l = s.layout(t)?;
        let sizes = [
            qsa::QUERY * 2,
            qsa::QH,
            qsa::KVH,
            qsa::IH,
            qsa::TOKEN,
            l.complete,
            l.complete,
            l.complete,
            l.logits_elements,
            qsa::QH,
            qsa::QUERY,
        ];
        for ((i, v), n) in (SPLIT_S..N).zip(status).zip(sizes) {
            if v[s.p[i].at(t)..][..n].iter().any(|&x| x != 1) {
                return Err(fault());
            }
        }
        for (i, v, w) in [(CAND, data[0], qsa::TOKEN), (OUT, data[1], qsa::QUERY)] {
            if v[s.p[i].at(t)..][..w].iter().any(|x| !x.is_finite()) {
                return Err(fault());
            }
        }
    }
    cancelled(cancel)
}

use qsa::{BLOCK, HD, ID, IH, KV, MAX_POSITION, QH, QUERY, TOKEN};

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
    ob: usize,
    os: usize,
    cb: usize,
    cs: usize,
) -> f32 {
    if token < previous {
        old[ob + token * os + offset]
    } else {
        current[cb + (token - previous) * cs + offset]
    }
}

#[cube(launch)]
pub fn split_batch(
    raw: &Array<f32>,
    packed: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    rb: usize,
    rs: usize,
    pb: usize,
    ps: usize,
    sb: usize,
    ss: usize,
) {
    let token = CUBE_POS_Y as usize;
    if token < count {
        // Y is the token plane; element coordinates must stay in X.
        let i = ABSOLUTE_POS_X as usize;
        if i < QUERY * 2 {
            let mut d = i;
            let mut extra = 0_usize;
            if i >= QUERY {
                d = i - QUERY;
                extra = HD;
            }
            let x = raw[rb + token * rs + (d / HD) * (HD * 2) + d % HD + extra];
            packed[pb + token * ps + i] = 0.0;
            status[sb + token * ss + i] = 0;
            if finite(x) {
                packed[pb + token * ps + i] = x;
                status[sb + token * ss + i] = 1;
            }
        }
    }
}

#[cube(launch)]
// CubeCL needs an initialized mutable local for the runtime three-way branch.
// Its native Rust body overwrites that initializer on every branch.
#[allow(unused_assignments)]
pub fn pack_batch(
    key: &Array<f32>,
    value: &Array<f32>,
    raw_index_key: &Array<f32>,
    token: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    kb: usize,
    ks: usize,
    vb: usize,
    vs: usize,
    rb: usize,
    rs: usize,
    cb: usize,
    cs: usize,
    sb: usize,
    ss: usize,
) {
    let t = CUBE_POS_Y as usize;
    if t < count {
        // Y is the token plane; element coordinates must stay in X.
        let i = ABSOLUTE_POS_X as usize;
        if i < TOKEN {
            let mut x = 0.0_f32;
            if i < KV {
                x = key[kb + t * ks + i];
            } else if i < KV * 2 {
                x = value[vb + t * vs + i - KV];
            } else {
                x = raw_index_key[rb + t * rs + i - KV * 2];
            }
            token[cb + t * cs + i] = 0.0;
            status[sb + t * ss + i] = 0;
            if finite(x) {
                token[cb + t * cs + i] = x;
                status[sb + t * ss + i] = 1;
            }
        }
    }
}

#[cube(launch)]
pub fn norm_batch(
    input: &Array<f32>,
    gamma: &Array<f32>,
    sin_table: &Array<f32>,
    cos_table: &Array<f32>,
    positions: &Array<u32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    previous: usize,
    heads: usize,
    ib: usize,
    is_: usize,
    gb: usize,
    gs: usize,
    sinb: usize,
    sins: usize,
    cosb: usize,
    coss: usize,
    pb: usize,
    ps: usize,
    ob: usize,
    os: usize,
    sb: usize,
    ss: usize,
    block_positions: u32,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    if token < count {
        let group = CUBE_POS_X as usize;
        let lane = UNIT_POS_X as usize;
        let mut maximum = 1.0_f32;
        let mut lane_flag = 1_u32;
        let mut groups = heads;
        let mut p = positions[pb + (previous + token) * ps];
        if block_positions == 1 {
            groups = (previous + token + 1) / BLOCK;
        }
        if block_positions == 1 && group < groups {
            p = positions[pb + group * BLOCK * ps];
        }
        if group >= groups || p > MAX_POSITION || !finite(qsa::EPSILON) || qsa::EPSILON <= 0.0 {
            lane_flag = 0;
        }
        if group < groups {
            for tile in 0..width / LANES {
                let d = tile * LANES + lane;
                let i = group * width + d;
                let gi = gb + group * gs + d;
                let x = input[ib + token * is_ + i];
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
                let x = input[ib + token * is_ + group * width + tile * LANES + lane] / scale;
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
        let denom = (sums[0] / width as f32 + (qsa::EPSILON / scale) / scale).sqrt();
        let mut out_flag = 1_u32;
        if group < groups {
            for tile in 0..width / LANES {
                let d = tile * LANES + lane;
                let i = group * width + d;
                let gi = gb + group * gs + d;
                let mut y = ((input[ib + token * is_ + i] / scale) / denom) * gamma[gi];
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
                    let gpair = gb + group * gs + pair;
                    let mate = ((input[ib + token * is_ + group * width + pair] / scale) / denom)
                        * gamma[gpair];
                    let si = sinb + p as usize * sins + frequency_index;
                    let ci = cosb + p as usize * coss + frequency_index;
                    let s = sin_table[si];
                    let c = cos_table[ci];
                    if d < 32 {
                        y = y * c - mate * s;
                    } else {
                        y = y * c + mate * s;
                    }
                    table_good = finite(s) && finite(c);
                }
                output[ob + token * os + i] = 0.0;
                if flags[0] == 1 && finite(denom) && denom > 0.0 && table_good && finite(y) {
                    output[ob + token * os + i] = y;
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
            status[sb + token * ss + group] = output_flags[0];
        }
    }
}

#[cube(launch)]
pub fn pool_batch(
    old: &Array<f32>,
    current: &Array<f32>,
    means: &mut Array<f32>,
    status: &mut Array<u32>,
    previous: usize,
    count: usize,
    ob: usize,
    os: usize,
    cb: usize,
    cs: usize,
    mb: usize,
    ms: usize,
    sb: usize,
    ss: usize,
) {
    let token = CUBE_POS_Y as usize;
    let complete = (previous + token + 1) / BLOCK;
    if token < count {
        let block = CUBE_POS_X as usize;
        let lane = UNIT_POS_X as usize;
        let mut flag = 1_u32;
        if block < complete {
            for tile in 0..2 {
                let d = tile * LANES + lane;
                let mut sum = 0.0_f32;
                for j in 0..BLOCK {
                    let t = block * BLOCK + j;
                    if t < previous + token + 1 {
                        let x = cached(old, current, t, previous, KV * 2 + d, ob, os, cb, cs);
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
                means[mb + token * ms + block * ID + d] = 0.0;
                if finite(y) {
                    means[mb + token * ms + block * ID + d] = y;
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
            status[sb + token * ss + block] = flags[0];
        }
    }
}

#[cube(launch)]
pub fn score_batch(
    pooled: &Array<f32>,
    query: &Array<f32>,
    scores: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    previous: usize,
    pb: usize,
    ps: usize,
    qb: usize,
    qs: usize,
    ob: usize,
    os: usize,
    sb: usize,
    ss: usize,
) {
    let token = CUBE_POS_Y as usize;
    let complete = (previous + token + 1) / BLOCK;
    if token < count {
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
                    let x = pooled[pb + token * ps + block * ID + d];
                    let q = query[qb + token * qs + head * ID + d];
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
            scores[ob + token * os + block] = 0.0;
            status[sb + token * ss + block] = 0;
            if valid == 1 && finite(score) && score >= 0.0 {
                scores[ob + token * os + block] = score;
                status[sb + token * ss + block] = 1;
            }
        }
    }
}

#[cube(launch)]
pub fn logits_batch(
    query: &Array<f32>,
    old: &Array<f32>,
    current: &Array<f32>,
    selected: &Array<u32>,
    logits: &mut Array<f32>,
    status: &mut Array<u32>,
    previous: usize,
    tokens: usize,
    qb: usize,
    qs: usize,
    ob: usize,
    os: usize,
    cb: usize,
    cs: usize,
    ib: usize,
    is_: usize,
    lb: usize,
    ls: usize,
    sb: usize,
    ss: usize,
) {
    let plane = CUBE_POS_Y as usize;
    let count = ((previous + plane + 1) / BLOCK).min(qsa::BUDGET / BLOCK) * BLOCK
        + (previous + plane + 1) % BLOCK;
    let count_tokens = tokens;
    if plane < count_tokens {
        let task = CUBE_POS_X as usize;
        let lane = UNIT_POS_X as usize;
        let head = task / (qsa::BUDGET + 3);
        let rank = task % (qsa::BUDGET + 3);
        if rank < count {
            let token = selected[ib + plane * is_ + rank] as usize;
            let mut sum = 0.0_f32;
            let mut flag = 1_u32;
            if head < QH && token < previous + plane + 1 {
                for tile in 0..4 {
                    let d = tile * LANES + lane;
                    let q = query[qb + plane * qs + head * HD + d];
                    let k = cached(
                        old,
                        current,
                        token,
                        previous,
                        (head / 12) * HD + d,
                        ob,
                        os,
                        cb,
                        cs,
                    );
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
                logits[lb + plane * ls + head * count + rank] = 0.0;
                status[sb + plane * ss + head * count + rank] = 0;
                if flags[0] == 1 && finite(y) {
                    logits[lb + plane * ls + head * count + rank] = y;
                    status[sb + plane * ss + head * count + rank] = 1;
                }
            }
        }
    }
}

#[cube(launch)]
// CubeCL cannot lower RangeInclusive::contains for runtime shader values.
#[allow(clippy::manual_range_contains)]
pub fn softmax_batch(
    logits: &Array<f32>,
    probability: &mut Array<f32>,
    status: &mut Array<u32>,
    tokens: usize,
    previous: usize,
    lb: usize,
    ls: usize,
    pb: usize,
    ps: usize,
    sb: usize,
    ss: usize,
) {
    let token = CUBE_POS_Y as usize;
    let count = ((previous + token + 1) / BLOCK).min(qsa::BUDGET / BLOCK) * BLOCK
        + (previous + token + 1) % BLOCK;
    if token < tokens {
        let head = CUBE_POS_X as usize;
        let lane = UNIT_POS_X as usize;
        // Layout guarantees count>=1; avoid a macro constant-only negative unary.
        let mut maximum = logits[lb + token * ls + head * count];
        let mut flag = 1_u32;
        if !finite(maximum) {
            flag = 0;
        }
        let mut rank = lane;
        while rank < count {
            let x = logits[lb + token * ls + head * count + rank];
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
            sum += (logits[lb + token * ls + head * count + rank] - max).exp();
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
            let y = (logits[lb + token * ls + i] - max).exp() / denominator;
            probability[pb + token * ps + i] = 0.0;
            if flags[0] == 1
                && finite(denominator)
                && denominator > 0.0
                && finite(y)
                && y >= 0.0
                && y <= 1.0
            {
                probability[pb + token * ps + i] = y;
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
            status[sb + token * ss + head] = out_flags[0];
        }
    }
}

#[cube(launch)]
// CubeCL cannot lower RangeInclusive::contains for runtime shader values.
#[allow(clippy::manual_range_contains)]
pub fn values_batch(
    old: &Array<f32>,
    current: &Array<f32>,
    probability: &Array<f32>,
    selected: &Array<u32>,
    raw_gate: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    previous: usize,
    tokens: usize,
    ob: usize,
    os: usize,
    cb: usize,
    cs: usize,
    pb: usize,
    ps: usize,
    ib: usize,
    is_: usize,
    gb: usize,
    gs: usize,
    yb: usize,
    ys: usize,
    sb: usize,
    ss: usize,
) {
    let token = CUBE_POS_Y as usize;
    let count = ((previous + token + 1) / BLOCK).min(qsa::BUDGET / BLOCK) * BLOCK
        + (previous + token + 1) % BLOCK;
    if token < tokens {
        // Y is the token plane; element coordinates must stay in X.
        let i = ABSOLUTE_POS_X as usize;
        if i < QUERY {
            let head = i / HD;
            let dim = i % HD;
            let mut sum = 0.0_f32;
            let mut valid = true;
            for rank in 0..count {
                let t = selected[ib + token * is_ + rank] as usize;
                let p = probability[pb + token * ps + head * count + rank];
                if t < previous + token + 1 {
                    let v = cached(
                        old,
                        current,
                        t,
                        previous,
                        KV + (head / 12) * HD + dim,
                        ob,
                        os,
                        cb,
                        cs,
                    );
                    if finite(v) && finite(p) && p >= 0.0 && p <= 1.0 {
                        sum += p * v;
                    } else {
                        valid = false;
                    }
                } else {
                    valid = false;
                }
            }
            let gate = raw_gate[gb + token * gs + QUERY + i];
            let y = sum * logistic(gate);
            output[yb + token * ys + i] = 0.0;
            status[sb + token * ss + i] = 0;
            if valid && finite(gate) && finite(sum) && finite(y) {
                output[yb + token * ys + i] = y;
                status[sb + token * ss + i] = 1;
            }
        }
    }
}

pub struct Prepared {
    spans: Spans,
    buffers: Buffers,
}
fn grid(n: usize, t: usize) -> CubeCount {
    CubeCount::Static(n as u32, t as u32, 1)
}
pub fn enqueue_prepare(
    arena: &PrivateArena,
    s: Spans,
    b: Buffers,
    positions: &[u32],
    cancel: &AtomicBool,
) -> Result<Prepared> {
    s.validate_positions(positions)?;
    cancelled(cancel)?;
    validate_bindings(arena, s, &b)?;
    let a = |i| arg(arena, &b, i);
    let p = s.p;
    let c = arena.client().map_err(|_| fault())?;
    let d = CubeDim::new_1d(64);
    split_batch::launch::<WgpuRuntime>(
        c,
        grid((qsa::QUERY * 2).div_ceil(64), MAX_BATCH),
        d,
        a(RAWQ)?,
        a(PACK)?,
        a(SPLIT_S)?,
        s.count,
        p[RAWQ].base,
        p[RAWQ].stride,
        p[PACK].base,
        p[PACK].stride,
        p[SPLIT_S].base,
        p[SPLIT_S].stride,
    );
    for (input, output, status, gamma, heads, width, block) in [
        (PACK, Q, Q_S, GQ, qsa::QH, qsa::HD, 0),
        (RAWK, K, K_S, GK, qsa::KVH, qsa::HD, 0),
        (RAWIQ, IQ, IQ_S, GI, qsa::IH, qsa::ID, 0),
    ] {
        cancelled(cancel)?;
        norm_batch::launch::<WgpuRuntime>(
            c,
            grid(heads, MAX_BATCH),
            d,
            a(input)?,
            a(gamma)?,
            a(SIN)?,
            a(COS)?,
            a(POS)?,
            a(output)?,
            a(status)?,
            s.count,
            s.previous,
            heads,
            p[input].base,
            p[input].stride,
            p[gamma].base,
            p[gamma].stride,
            p[SIN].base,
            p[SIN].stride,
            p[COS].base,
            p[COS].stride,
            p[POS].base,
            p[POS].stride,
            p[output].base,
            p[output].stride,
            p[status].base,
            p[status].stride,
            block,
            width,
        );
    }
    cancelled(cancel)?;
    pack_batch::launch::<WgpuRuntime>(
        c,
        grid(qsa::TOKEN.div_ceil(64), MAX_BATCH),
        d,
        a(K)?,
        a(VALUE)?,
        a(RAWIK)?,
        a(CAND)?,
        a(CAND_S)?,
        s.count,
        p[K].base,
        p[K].stride,
        p[VALUE].base,
        p[VALUE].stride,
        p[RAWIK].base,
        p[RAWIK].stride,
        p[CAND].base,
        p[CAND].stride,
        p[CAND_S].base,
        p[CAND_S].stride,
    );
    let blocks = s.capacity / 4;
    pool_batch::launch::<WgpuRuntime>(
        c,
        grid(blocks, MAX_BATCH),
        d,
        a(OLD)?,
        a(CAND)?,
        a(MEANS)?,
        a(POOL_S)?,
        s.previous,
        s.count,
        p[OLD].base,
        p[OLD].stride,
        p[CAND].base,
        p[CAND].stride,
        p[MEANS].base,
        p[MEANS].stride,
        p[POOL_S].base,
        p[POOL_S].stride,
    );
    norm_batch::launch::<WgpuRuntime>(
        c,
        grid(blocks, MAX_BATCH),
        d,
        a(MEANS)?,
        a(GP)?,
        a(SIN)?,
        a(COS)?,
        a(POS)?,
        a(POOL)?,
        a(PN_S)?,
        s.count,
        s.previous,
        blocks,
        p[MEANS].base,
        p[MEANS].stride,
        p[GP].base,
        0,
        p[SIN].base,
        p[SIN].stride,
        p[COS].base,
        p[COS].stride,
        p[POS].base,
        p[POS].stride,
        p[POOL].base,
        p[POOL].stride,
        p[PN_S].base,
        p[PN_S].stride,
        1,
        qsa::ID,
    );
    score_batch::launch::<WgpuRuntime>(
        c,
        grid(blocks, MAX_BATCH),
        d,
        a(POOL)?,
        a(IQ)?,
        a(SCORE)?,
        a(SCORE_S)?,
        s.count,
        s.previous,
        p[POOL].base,
        p[POOL].stride,
        p[IQ].base,
        p[IQ].stride,
        p[SCORE].base,
        p[SCORE].stride,
        p[SCORE_S].base,
        p[SCORE_S].stride,
    );
    cancelled(cancel)?;
    Ok(Prepared {
        spans: s,
        buffers: b,
    })
}
struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.0);
    }
}
/// Move a full guarded upload into the already owned selection destination.
/// No allocation of a destination or ordinary/public queue-write fallback.
pub fn enqueue_attention(
    arena: &PrivateArena,
    ready: &Prepared,
    ids: Zeroizing<Vec<u32>>,
    cancel: &AtomicBool,
) -> Result<()> {
    let s = ready.spans;
    let b = &ready.buffers;
    validate_selected(s, &ids)?;
    cancelled(cancel)?;
    validate_bindings(arena, s, b)?;
    let layout = arena.layout(b.0[SELECT]).map_err(|_| fault())?;
    let mut upload = Upload(Vec::new());
    upload
        .0
        .try_reserve_exact(layout.owned_bytes)
        .map_err(|_| fault())?;
    for &x in ids.iter() {
        upload.0.extend_from_slice(&x.to_le_bytes());
    }
    upload.0.resize(layout.owned_bytes, 0);
    let c = arena.client().map_err(|_| fault())?;
    c.write_private(
        arena.handle(b.0[SELECT]).map_err(|_| fault())?.clone(),
        std::mem::take(&mut upload.0),
    )
    .map_err(|_| fault())?;
    drop(ids);
    let a = |i| arg(arena, b, i);
    let p = s.p;
    let d = CubeDim::new_1d(64);
    cancelled(cancel)?;
    logits_batch::launch::<WgpuRuntime>(
        c,
        grid(qsa::QH * (qsa::BUDGET + 3), MAX_BATCH),
        d,
        a(Q)?,
        a(OLD)?,
        a(CAND)?,
        a(SELECT)?,
        a(LOGIT)?,
        a(LOGIT_S)?,
        s.previous,
        s.count,
        p[Q].base,
        p[Q].stride,
        p[OLD].base,
        p[OLD].stride,
        p[CAND].base,
        p[CAND].stride,
        p[SELECT].base,
        p[SELECT].stride,
        p[LOGIT].base,
        p[LOGIT].stride,
        p[LOGIT_S].base,
        p[LOGIT_S].stride,
    );
    cancelled(cancel)?;
    softmax_batch::launch::<WgpuRuntime>(
        c,
        grid(qsa::QH, MAX_BATCH),
        d,
        a(LOGIT)?,
        a(PROB)?,
        a(SOFT_S)?,
        s.count,
        s.previous,
        p[LOGIT].base,
        p[LOGIT].stride,
        p[PROB].base,
        p[PROB].stride,
        p[SOFT_S].base,
        p[SOFT_S].stride,
    );
    cancelled(cancel)?;
    values_batch::launch::<WgpuRuntime>(
        c,
        grid(qsa::QUERY.div_ceil(64), MAX_BATCH),
        d,
        a(OLD)?,
        a(CAND)?,
        a(PROB)?,
        a(SELECT)?,
        a(PACK)?,
        a(OUT)?,
        a(OUT_S)?,
        s.previous,
        s.count,
        p[OLD].base,
        p[OLD].stride,
        p[CAND].base,
        p[CAND].stride,
        p[PROB].base,
        p[PROB].stride,
        p[SELECT].base,
        p[SELECT].stride,
        p[PACK].base,
        p[PACK].stride,
        p[OUT].base,
        p[OUT].stride,
        p[OUT_S].base,
        p[OUT_S].stride,
    );
    cancelled(cancel)
}

pub(crate) fn guarded_select(
    l: qsa::QsaLayout,
    scores: &[f32],
    status: &[u32],
) -> Result<Zeroizing<Vec<u32>>> {
    l.validate().map_err(|_| fault())?;
    if scores.len() != l.complete
        || status.len() != l.complete
        || status.iter().any(|&x| x != 1)
        || scores.iter().any(|x| !x.is_finite() || *x < 0.)
    {
        return Err(fault());
    }
    let mut blocks = Zeroizing::new(Vec::new());
    blocks.try_reserve_exact(l.complete).map_err(|_| fault())?;
    blocks.extend(0..l.complete);
    blocks.sort_unstable_by(|&a, &b| {
        if scores[a] == scores[b] {
            a.cmp(&b)
        } else {
            scores[b].total_cmp(&scores[a])
        }
    });
    blocks.truncate(qsa::BUDGET / 4);
    blocks.sort_unstable();
    let mut ids = Zeroizing::new(Vec::new());
    ids.try_reserve_exact(l.selected).map_err(|_| fault())?;
    for &b in blocks.iter() {
        for j in 0..4 {
            ids.push((b * 4 + j) as u32);
        }
    }
    for t in l.complete * 4..l.visible {
        ids.push(t as u32);
    }
    qsa::validate_selected(l, &ids).map_err(|_| fault())?;
    Ok(ids)
}
