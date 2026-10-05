//! Fixed public MAX8 causal QSA acceptance only; no model payload or commit API.
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    qsa_chunk::{self as c, Buffers, N, Span, Spans},
    qsa_gpu as qsa,
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::ops;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CAPACITY: usize = 2064;
const MAIN_BUDGET: usize = 64 * 1024 * 1024;
const IO_BUDGET: u64 = 64 * 1024 * 1024;
const GAP: f32 = -173.25;
const STATUS_GAP: u32 = 0x53aa_9267;
const INFO_BYTES: usize = 512;
const PREP_DISPATCHES: usize = 8;
const ATTENTION_DISPATCHES: usize = 3;
pub(crate) fn fixture_spans(previous: usize, count: usize) -> Result<Spans> {
    let mut s = Spans {
        count,
        previous,
        capacity: CAPACITY,
        p: [Span {
            base: 0,
            stride: 1,
            words: 1,
        }; N],
    };
    let rows = s.rows();
    for (i, w) in s.widths().into_iter().enumerate() {
        let stride = if i == c::POS { 2 } else { w + 5 + i % 7 };
        s.p[i] = Span {
            base: 3 + i % 11,
            stride,
            words: 3 + i % 11 + rows[i] * stride + 17,
        };
    }
    s.validate()?;
    Ok(s)
}
fn reference_words(s: Spans) -> [usize; N] {
    let mut r = [1; N];
    for (i, w) in s.widths().into_iter().enumerate() {
        r[i] = match i {
            c::OLD => s.capacity * qsa::TOKEN,
            c::POS => s.capacity,
            c::GQ => qsa::QUERY,
            c::GK => qsa::KV,
            c::GI => qsa::INDEX_QUERY,
            c::GP => qsa::ID,
            c::SIN | c::COS => qsa::ROPE_TABLE_WORDS,
            _ => w,
        };
    }
    r
}
fn owned_plan(s: Spans) -> Result<usize> {
    let full =
        s.p.into_iter()
            .map(|x| x.words)
            .chain(reference_words(s))
            .try_fold(0usize, |n, w| {
                n.checked_add(BufferLayout::new(w)?.owned_bytes).ok_or_else(
                    || -> Box<dyn std::error::Error> { "QSA chunk main plan overflow".into() },
                )
            })?;
    if full > MAIN_BUDGET {
        return Err("QSA chunk main budget exceeded".into());
    }
    Ok(full)
}
fn io_bound(s: Spans) -> Result<usize> {
    let sources = (c::RAWQ..=c::OLD).try_fold(0usize, |n, i| {
        n.checked_add(BufferLayout::new(s.p[i].words)?.owned_bytes)
            .ok_or_else(|| -> Box<dyn std::error::Error> { "QSA chunk IO plan overflow".into() })
    })?;
    let reads = BufferLayout::new(s.p[c::SCORE].words)?.owned_bytes
        + BufferLayout::new(s.p[c::SCORE_S].words)?.owned_bytes;
    let primary = sources
        .checked_add(reads)
        .and_then(|n| n.checked_add(PREP_DISPATCHES * 2 * INFO_BYTES))
        .ok_or_else(|| -> Box<dyn std::error::Error> { "QSA chunk IO plan overflow".into() })?;
    let attention = BufferLayout::new(s.p[c::SELECT].words)?.owned_bytes
        + BufferLayout::new(s.p[c::OUT_S].words)?.owned_bytes
        + ATTENTION_DISPATCHES * 2 * INFO_BYTES;
    let diagnostic =
        s.p.iter()
            .map(|p| p.words)
            .chain(reference_words(s))
            .try_fold(0usize, |largest, w| -> Result<_> {
                Ok(largest.max(BufferLayout::new(w)?.owned_bytes))
            })?;
    let peak = primary
        .max(attention)
        .max(diagnostic)
        .checked_mul(2)
        .ok_or_else(|| -> Box<dyn std::error::Error> { "QSA chunk IO plan overflow".into() })?;
    if peak > IO_BUDGET as usize {
        return Err("QSA chunk IO budget exceeded".into());
    }
    Ok(peak)
}
struct Lease(cubecl::bytes::Bytes);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn read(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
) -> Result<Zeroizing<Vec<u32>>> {
    let lease = Lease(
        client
            .read_one(arena.handle(id)?.clone())
            .map_err(|_| "QSA chunk read failed")?,
    );
    if lease.0.len() != arena.layout(id)?.owned_bytes {
        return Err("QSA chunk read span mismatch".into());
    }
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(lease.0.len() / 4)
        .map_err(|_| "QSA chunk read allocation failed")?;
    for b in lease.0.chunks_exact(4) {
        out.push(u32::from_le_bytes(b.try_into().unwrap()));
    }
    Ok(out)
}
fn floats(words: &[u32]) -> Zeroizing<Vec<f32>> {
    let mut out = Zeroizing::new(Vec::new());
    out.reserve_exact(words.len());
    for &x in words {
        out.push(f32::from_bits(x));
    }
    out
}
fn audit(client: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<()> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io_audited())
    }));
    let r = match r {
        Ok(Ok(r)) => r,
        _ => {
            retired.store(true, Ordering::Release);
            return Err("QSA chunk runtime cleanup failed".into());
        }
    };
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("QSA chunk audit overflow")?;
    if full == 0
        || r.inspected_bytes != full
        || r.inspected_buffers == 0
        || retired.load(Ordering::Acquire)
    {
        retired.store(true, Ordering::Release);
        return Err("QSA chunk runtime inspection incomplete".into());
    }
    Ok(())
}
struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn write(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
    words: &[u32],
) -> Result<()> {
    let l = arena.layout(id)?;
    if words.len() != l.logical_words {
        return Err("QSA chunk upload span mismatch".into());
    }
    let mut u = Upload(Vec::new());
    u.0.try_reserve_exact(l.owned_bytes)?;
    for &x in words {
        u.0.extend_from_slice(&x.to_le_bytes());
    }
    u.0.resize(l.owned_bytes, 0);
    client
        .write_private(arena.handle(id)?.clone(), std::mem::take(&mut u.0))
        .map_err(|_| "QSA chunk owned upload failed")?;
    Ok(())
}
fn raw_key(t: usize) -> Zeroizing<Vec<f32>> {
    let mut v = Zeroizing::new(vec![0.; qsa::KV]);
    for (d, x) in v.iter_mut().enumerate() {
        *x = (((d * 3 + t * 7) % 29) as f32 - 14.) / 32.;
    }
    v
}
fn raw_value(t: usize) -> Zeroizing<Vec<f32>> {
    let mut v = Zeroizing::new(vec![0.; qsa::KV]);
    for (d, x) in v.iter_mut().enumerate() {
        *x = (((d * 5 + t * 11) % 37) as f32 - 18.) / 16.;
    }
    v
}
fn raw_index(t: usize) -> Zeroizing<Vec<f32>> {
    let mut r = Zeroizing::new(vec![0.; qsa::ID]);
    r[64] = ((t % 4) as f32 + 1.) / 8.;
    if t / 4 >= 8 {
        r[0] = 0.5 + (t % 4) as f32 / 4.;
    }
    r
}
fn normalized(raw: &[f32], gamma: &[f32], width: usize, p: u32) -> Result<Zeroizing<Vec<f32>>> {
    let mut x = Zeroizing::new(ops::rms_norm(raw, gamma, width, qsa::EPSILON)?);
    ops::rope_neox_heads(&mut x, width, 64, p, qsa::THETA)?;
    Ok(x)
}
struct Fixture {
    spans: Spans,
    words: [Zeroizing<Vec<u32>>; N],
    cache: Zeroizing<Vec<u32>>,
    positions: Zeroizing<Vec<u32>>,
    gamma: [Zeroizing<Vec<f32>>; 4],
}
fn fixture(previous: usize, count: usize) -> Result<Fixture> {
    let s = fixture_spans(previous, count)?;
    owned_plan(s)?;
    io_bound(s)?;
    let widths = s.widths();
    let mut f = Fixture {
        spans: s,
        words: std::array::from_fn(|i| {
            Zeroizing::new(vec![
                if i >= c::SPLIT_S {
                    STATUS_GAP
                } else {
                    GAP.to_bits()
                };
                s.p[i].words
            ])
        }),
        cache: Zeroizing::new(vec![0; s.capacity * qsa::TOKEN]),
        positions: Zeroizing::new(vec![0; s.capacity]),
        gamma: std::array::from_fn(|i| {
            let w = if i < 2 { qsa::HD } else { qsa::ID };
            let mut v = Zeroizing::new(vec![0.; w]);
            for (d, x) in v.iter_mut().enumerate() {
                *x = 0.625 + ((d + i * 3) % 7) as f32 / 16.;
            }
            v
        }),
    };
    for t in 0..s.capacity {
        f.positions[t] = t as u32;
        f.words[c::POS][s.p[c::POS].at(t)] = t as u32;
    }
    for g in 0..4 {
        let i = c::GQ + g;
        for h in 0..s.rows()[i] {
            for d in 0..widths[i] {
                f.words[i][s.p[i].at(h) + d] = f.gamma[g][d].to_bits();
            }
        }
    }
    let (sin, cos) = qsa::rope_tables();
    for (i, table) in [(c::SIN, sin), (c::COS, cos)] {
        for t in 0..qsa::MAX_CONTEXT {
            for d in 0..qsa::ROPE_PAIRS {
                f.words[i][s.p[i].at(t) + d] = table[t * qsa::ROPE_PAIRS + d].to_bits();
            }
        }
    }
    for t in 0..previous {
        let k = normalized(&raw_key(t), &f.gamma[1], qsa::HD, t as u32)?;
        let v = raw_value(t);
        let ik = raw_index(t);
        for (j, &x) in k.iter().chain(v.iter()).chain(ik.iter()).enumerate() {
            f.words[c::OLD][s.p[c::OLD].at(t) + j] = x.to_bits();
            f.cache[t * qsa::TOKEN + j] = x.to_bits();
        }
    }
    for t in 0..count {
        let absolute = previous + t;
        let k = raw_key(absolute);
        let v = raw_value(absolute);
        let ik = raw_index(absolute);
        for (i, values) in [(c::RAWK, &*k), (c::VALUE, &*v), (c::RAWIK, &*ik)] {
            for (d, &x) in values.iter().enumerate() {
                f.words[i][s.p[i].at(t) + d] = x.to_bits();
            }
        }
        for d in 0..qsa::QUERY * 2 {
            f.words[c::RAWQ][s.p[c::RAWQ].at(t) + d] =
                (((d * 7 + absolute * 3) % 37) as f32 / 16. - 18. / 16.).to_bits();
        }
        for d in 0..qsa::INDEX_QUERY {
            let head = d / qsa::ID;
            let dim = d % qsa::ID;
            let x: f32 = if (head < 2 && dim == 0) || (head >= 2 && dim == 32) {
                if head.is_multiple_of(2) { 1. } else { -1. }
            } else {
                0.
            };
            f.words[c::RAWIQ][s.p[c::RAWIQ].at(t) + d] = x.to_bits();
        }
    }
    s.validate_positions(&f.words[c::POS])?;
    Ok(f)
}
fn alloc(arena: &mut PrivateArena, words: &[Zeroizing<Vec<u32>>; N]) -> Result<Buffers> {
    let mut ids = Vec::with_capacity(N);
    for w in words {
        ids.push(arena.upload_u32(w)?);
    }
    Ok(Buffers(
        ids.try_into().map_err(|_| "QSA chunk allocation count")?,
    ))
}
fn ref_data(f: &Fixture) -> [Zeroizing<Vec<u32>>; N] {
    let widths = reference_words(f.spans);
    let mut r = std::array::from_fn(|i| {
        Zeroizing::new(vec![
            if i >= c::SPLIT_S {
                STATUS_GAP
            } else {
                GAP.to_bits()
            };
            widths[i]
        ])
    });
    r[c::OLD].copy_from_slice(&f.cache);
    r[c::POS].copy_from_slice(&f.positions);
    for g in 0..4 {
        let i = c::GQ + g;
        for (d, x) in r[i].iter_mut().enumerate() {
            *x = f.gamma[g][d % f.gamma[g].len()].to_bits();
        }
    }
    let (sin, cos) = qsa::rope_tables();
    for (i, v) in [(c::SIN, sin), (c::COS, cos)] {
        for (d, &x) in v.iter().enumerate() {
            r[i][d] = x.to_bits();
        }
    }
    r
}
fn scalar_prepare(arena: &PrivateArena, b: &Buffers, previous: usize) -> Result<()> {
    let client = arena.client()?;
    let a = |i| -> Result<ArrayArg<WgpuRuntime>> {
        Ok(unsafe {
            ArrayArg::from_raw_parts(
                arena.handle(b.0[i])?.clone(),
                arena.layout(b.0[i])?.owned_words,
            )
        })
    };
    let d = CubeDim::new_1d(64);
    let grid = |n: usize| CubeCount::Static(n as u32, 1, 1);
    let l = qsa::QsaLayout::new(previous + 1)?;
    qsa::split_query_gate::launch::<WgpuRuntime>(
        client,
        grid((qsa::QUERY * 2).div_ceil(64)),
        d,
        a(c::RAWQ)?,
        a(c::PACK)?,
        a(c::SPLIT_S)?,
    );
    // Original query norm sees just first QUERY packed words; gate follows.
    for (input, out, st, gamma, groups, width) in [
        (c::PACK, c::Q, c::Q_S, c::GQ, qsa::QH, qsa::HD),
        (c::RAWK, c::K, c::K_S, c::GK, qsa::KVH, qsa::HD),
        (c::RAWIQ, c::IQ, c::IQ_S, c::GI, qsa::IH, qsa::ID),
    ] {
        qsa::rms_rope::launch::<WgpuRuntime>(
            client,
            grid(groups),
            d,
            a(input)?,
            a(gamma)?,
            a(c::SIN)?,
            a(c::COS)?,
            a(c::POS)?,
            a(out)?,
            a(st)?,
            groups,
            previous as u32,
            qsa::EPSILON,
            width,
            false,
            false,
        );
    }
    qsa::pack_token::launch::<WgpuRuntime>(
        client,
        grid(qsa::TOKEN.div_ceil(64)),
        d,
        a(c::K)?,
        a(c::VALUE)?,
        a(c::RAWIK)?,
        a(c::CAND)?,
        a(c::CAND_S)?,
    );
    if l.complete > 0 {
        qsa::pool_indexer_raw::launch::<WgpuRuntime>(
            client,
            grid(l.complete),
            d,
            a(c::OLD)?,
            a(c::CAND)?,
            a(c::MEANS)?,
            a(c::POOL_S)?,
            previous,
            l.complete,
        );
        qsa::rms_rope::launch::<WgpuRuntime>(
            client,
            grid(l.complete),
            d,
            a(c::MEANS)?,
            a(c::GP)?,
            a(c::SIN)?,
            a(c::COS)?,
            a(c::POS)?,
            a(c::POOL)?,
            a(c::PN_S)?,
            l.complete,
            previous as u32,
            qsa::EPSILON,
            qsa::ID,
            true,
            true,
        );
        qsa::score_blocks::launch::<WgpuRuntime>(
            client,
            grid(l.complete),
            d,
            a(c::POOL)?,
            a(c::IQ)?,
            a(c::SCORE)?,
            a(c::SCORE_S)?,
            l.complete,
        );
    }
    Ok(())
}
fn scalar_attention(arena: &PrivateArena, b: &Buffers, previous: usize) -> Result<()> {
    let client = arena.client()?;
    let a = |i| -> Result<ArrayArg<WgpuRuntime>> {
        Ok(unsafe {
            ArrayArg::from_raw_parts(
                arena.handle(b.0[i])?.clone(),
                arena.layout(b.0[i])?.owned_words,
            )
        })
    };
    let d = CubeDim::new_1d(64);
    let grid = |n: usize| CubeCount::Static(n as u32, 1, 1);
    let l = qsa::QsaLayout::new(previous + 1)?;
    qsa::attention_logits::launch::<WgpuRuntime>(
        client,
        grid(l.logits_elements),
        d,
        a(c::Q)?,
        a(c::OLD)?,
        a(c::CAND)?,
        a(c::SELECT)?,
        a(c::LOGIT)?,
        a(c::LOGIT_S)?,
        previous,
        l.selected,
    );
    qsa::attention_softmax::launch::<WgpuRuntime>(
        client,
        grid(qsa::QH),
        d,
        a(c::LOGIT)?,
        a(c::PROB)?,
        a(c::SOFT_S)?,
        l.selected,
    );
    let h = arena
        .handle(b.0[c::PACK])?
        .clone()
        .offset_start((qsa::QUERY * 4) as u64)
        .offset_end((arena.layout(b.0[c::PACK])?.owned_bytes - qsa::QUERY * 8) as u64);
    let gate = unsafe { ArrayArg::from_raw_parts(h, qsa::QUERY) };
    qsa::attention_values_gated::launch::<WgpuRuntime>(
        client,
        grid(qsa::QUERY.div_ceil(64)),
        d,
        a(c::OLD)?,
        a(c::CAND)?,
        a(c::PROB)?,
        a(c::SELECT)?,
        gate,
        a(c::OUT)?,
        a(c::OUT_S)?,
        previous,
        l.selected,
    );
    Ok(())
}
fn compare(a: &[f32], e: &[f32]) -> Result<f32> {
    if a.len() != e.len() {
        return Err("QSA chunk parity extent".into());
    }
    let mut maximum = 0f32;
    for (&x, &y) in a.iter().zip(e) {
        let error = (x - y).abs();
        if !x.is_finite() || !y.is_finite() || error > 5e-5 + 5e-4 * y.abs() {
            return Err("QSA chunk numeric parity failed".into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn plane(words: &[u32], span: Span, t: usize, width: usize) -> Zeroizing<Vec<f32>> {
    floats(&words[span.at(t)..][..width])
}
fn unchanged(before: &[u32], after: &[u32], changed: impl Fn(usize) -> bool) -> Result<()> {
    for (i, &x) in after.iter().enumerate() {
        if !changed(i) && x != before.get(i).copied().unwrap_or(0) {
            return Err("QSA chunk readonly/gap/padding changed".into());
        }
    }
    Ok(())
}
type CoreOracle = (
    Zeroizing<Vec<f32>>,
    Zeroizing<Vec<f32>>,
    Zeroizing<Vec<u32>>,
);
fn core_token(f: &Fixture, t: usize) -> Result<CoreOracle> {
    let s = f.spans;
    let visible = s.previous + t + 1;
    let raw = plane(&f.words[c::RAWQ], s.p[c::RAWQ], t, qsa::QUERY * 2);
    let mut query_raw = Zeroizing::new(Vec::new());
    let mut gate = Zeroizing::new(Vec::new());
    query_raw.try_reserve_exact(qsa::QUERY)?;
    gate.try_reserve_exact(qsa::QUERY)?;
    for h in raw.chunks_exact(qsa::HD * 2) {
        query_raw.extend_from_slice(&h[..qsa::HD]);
        gate.extend_from_slice(&h[qsa::HD..]);
    }
    let query = normalized(&query_raw, &f.gamma[0], qsa::HD, (visible - 1) as u32)?;
    let iq = normalized(
        &plane(&f.words[c::RAWIQ], s.p[c::RAWIQ], t, qsa::INDEX_QUERY),
        &f.gamma[2],
        qsa::ID,
        (visible - 1) as u32,
    )?;
    let mut keys = Zeroizing::new(vec![0.; visible * qsa::KV]);
    let mut values = Zeroizing::new(vec![0.; visible * qsa::KV]);
    let mut index = Zeroizing::new(vec![0.; visible * qsa::ID]);
    for a in 0..visible {
        let k = normalized(&raw_key(a), &f.gamma[1], qsa::HD, a as u32)?;
        keys[a * qsa::KV..][..qsa::KV].copy_from_slice(&k);
        values[a * qsa::KV..][..qsa::KV].copy_from_slice(&raw_value(a));
        index[a * qsa::ID..][..qsa::ID].copy_from_slice(&raw_index(a));
    }
    let scores = Zeroizing::new(ops::qsa_block_scores(
        &index,
        &iq,
        &f.gamma[3],
        &f.positions[..visible],
        visible,
        ops::QsaShape::qwen38(),
    )?);
    let selected = Zeroizing::new(ops::qsa_select_blocks(&scores, visible, 4, qsa::BUDGET)?);
    let mut out = Zeroizing::new(ops::qsa_attend(
        &query,
        &keys,
        &values,
        &selected,
        visible,
        ops::QsaShape::qwen38(),
    )?);
    for (x, &g) in out.iter_mut().zip(gate.iter()) {
        *x *= ops::sigmoid(g);
    }
    let mut ids = Zeroizing::new(Vec::new());
    ids.try_reserve_exact(selected.len())?;
    for &x in selected.iter() {
        ids.push(x as u32);
    }
    Ok((out, scores, ids))
}
fn changed(s: Spans, i: usize, index: usize) -> bool {
    if i == c::SELECT {
        return true;
    }
    if i < c::PACK {
        return false;
    }
    let p = s.p[i];
    if index < p.base {
        return false;
    }
    let t = (index - p.base) / p.stride;
    if t >= s.count {
        return false;
    }
    let local = (index - p.base) % p.stride;
    let l = s.layout(t).unwrap();
    let width = match i {
        c::MEANS | c::POOL => l.complete * qsa::ID,
        c::SCORE | c::POOL_S | c::PN_S | c::SCORE_S => l.complete,
        c::LOGIT | c::PROB | c::LOGIT_S => l.logits_elements,
        _ => s.widths()[i],
    };
    local < width
}
// Fixed-public diagnostics expose stage labels only, never payload/status words.
fn diagnose_acceptance(s: Spans, data: [&[f32]; 2], status: [&[u32]; 11]) -> Result<()> {
    const LABELS: [&str; 11] = [
        "QSA chunk split-query-gate status",
        "QSA chunk query RMS/RoPE status",
        "QSA chunk key RMS/RoPE status",
        "QSA chunk index-query RMS/RoPE status",
        "QSA chunk candidate-pack status",
        "QSA chunk raw-pool status",
        "QSA chunk pooled RMS/RoPE status",
        "QSA chunk block-score status",
        "QSA chunk attention-logits status",
        "QSA chunk attention-softmax status",
        "QSA chunk gated-values status",
    ];
    for (i, v) in status.iter().enumerate() {
        if v.len() != BufferLayout::new(s.p[c::SPLIT_S + i].words)?.owned_words {
            return Err("QSA chunk status owning span".into());
        }
    }
    for (i, v) in [c::CAND, c::OUT].into_iter().zip(data) {
        if v.len() != BufferLayout::new(s.p[i].words)?.owned_words {
            return Err("QSA chunk data owning span".into());
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
        for (i, &n) in sizes.iter().enumerate() {
            if status[i][s.p[c::SPLIT_S + i].at(t)..][..n]
                .iter()
                .any(|&x| x != 1)
            {
                return Err(LABELS[i].into());
            }
        }
    }
    for (i, v, width, label) in [
        (
            c::CAND,
            data[0],
            qsa::TOKEN,
            "QSA chunk candidate nonfinite",
        ),
        (c::OUT, data[1], qsa::QUERY, "QSA chunk output nonfinite"),
    ] {
        for t in 0..s.count {
            if v[s.p[i].at(t)..][..width].iter().any(|x| !x.is_finite()) {
                return Err(label.into());
            }
        }
    }
    Ok(())
}
fn case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    previous: usize,
    count: usize,
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    println!("public_qsa_chunk_case previous={previous} count={count}");
    let f = fixture(previous, count).map_err(|_| "QSA chunk public fixture preflight")?;
    let s = f.spans;
    let b = alloc(arena, &f.words).map_err(|_| "QSA chunk owning allocation")?;
    let mut rw = ref_data(&f);
    let single = alloc(arena, &rw).map_err(|_| "QSA chunk scalar owning allocation")?;
    if arena.charged_bytes() != owned_plan(s)? {
        return Err("QSA chunk full owner charge mismatch".into());
    }
    let ids = b.0.into_iter().chain(single.0);
    let all_ids = ids.collect::<Vec<_>>();
    let all_words =
        s.p.map(|p| p.words)
            .into_iter()
            .chain(reference_words(s))
            .collect::<Vec<_>>();
    c::validate_pages(arena, &all_ids, &all_words)
        .map_err(|_| "QSA chunk combined full-page binding")?;
    let cancel = AtomicBool::new(true);
    if c::enqueue_prepare(arena, s, Buffers(b.0), &f.words[c::POS], &cancel).is_ok() {
        return Err("QSA chunk accepted cancelled enqueue".into());
    }
    cancel.store(false, Ordering::Release);
    let mut alias = b.0;
    alias[c::CAND] = alias[c::OLD];
    if c::validate_bindings(arena, s, &Buffers(alias)).is_ok() {
        return Err("QSA chunk accepted alias".into());
    }
    // These six retained full source stages include the entire nonzero committed
    // prefix and remain live until the completed score/status read returns.
    for i in c::RAWQ..=c::OLD {
        write(client, arena, b.0[i], &f.words[i])?;
    }
    let prepared = c::enqueue_prepare(arena, s, Buffers(b.0), &f.words[c::POS], &cancel)
        .map_err(|_| "QSA chunk preparation enqueue")?;
    let scores = read(client, arena, b.0[c::SCORE])?;
    let flags = read(client, arena, b.0[c::SCORE_S])?;
    audit(client, retired)?; // both leases have dropped; inference already completed.
    let ids = c::select(s, &floats(&scores), &flags, &cancel)
        .map_err(|_| "QSA chunk completed-score selection")?;
    let mut expected_ids = Zeroizing::new(vec![0; s.p[c::SELECT].words]);
    expected_ids.copy_from_slice(&ids);
    c::enqueue_attention(arena, &prepared, ids, &cancel)
        .map_err(|_| "QSA chunk selected upload/attention enqueue")?;
    // OUT_S is a completed whole-attention fence, before diagnostic page reads.
    let fence = read(client, arena, b.0[c::OUT_S])?;
    audit(client, retired)?;
    let mut actual: [Zeroizing<Vec<u32>>; N] = std::array::from_fn(|_| Zeroizing::new(Vec::new()));
    for (i, v) in actual.iter_mut().enumerate() {
        *v = read(client, arena, b.0[i])?;
        audit(client, retired)?;
        unchanged(&f.words[i], v, |j| changed(s, i, j))?;
    }
    if *fence != *actual[c::OUT_S] {
        return Err("QSA chunk fence changed".into());
    }
    let output = floats(&actual[c::OUT]);
    let candidate = floats(&actual[c::CAND]);
    let status = std::array::from_fn(|i| actual[c::SPLIT_S + i].as_slice());
    diagnose_acceptance(s, [&candidate, &output], status)?;
    c::validate_download(s, [&candidate, &output], status, &cancel)
        .map_err(|_| "QSA chunk canonical acceptance after diagnostics")?;
    cancel.store(true, Ordering::Release);
    if c::validate_download(s, [&candidate, &output], status, &cancel).is_ok() {
        return Err("QSA chunk accepted cancelled completion".into());
    }
    cancel.store(false, Ordering::Release);
    let mut max_core = 0f32;
    let mut max_single = 0f32;
    for t in 0..count {
        let absolute = previous + t;
        let l = s.layout(t)?;
        let (core, core_scores, core_ids) = core_token(&f, t)?;
        let chunk_output = &output[s.p[c::OUT].at(t)..][..qsa::QUERY];
        max_core = max_core.max(compare(chunk_output, &core)?).max(compare(
            &floats(&actual[c::SCORE][s.p[c::SCORE].at(t)..][..l.complete]),
            &core_scores,
        )?);
        if expected_ids[s.p[c::SELECT].at(t)..][..l.selected] != *core_ids {
            return Err("QSA chunk core selection mismatch".into());
        }
        for (i, values) in rw.iter_mut().enumerate().take(c::RAWIK + 1) {
            let width = s.widths()[i];
            values.copy_from_slice(&f.words[i][s.p[i].at(t)..][..width]);
            write(client, arena, single.0[i], values)?;
        }
        write(client, arena, single.0[c::OLD], &rw[c::OLD])?;
        scalar_prepare(arena, &single, absolute)?;
        let rs = read(client, arena, single.0[c::SCORE])?;
        let rf = read(client, arena, single.0[c::SCORE_S])?;
        audit(client, retired)?;
        let selected = c::guarded_select(l, &floats(&rs[..l.complete]), &rf[..l.complete])?;
        if selected.as_slice() != &expected_ids[s.p[c::SELECT].at(t)..][..l.selected] {
            return Err("QSA chunk scalar selection mismatch".into());
        }
        rw[c::SELECT].fill(0);
        rw[c::SELECT][..l.selected].copy_from_slice(&selected);
        write(client, arena, single.0[c::SELECT], &rw[c::SELECT])?;
        scalar_attention(arena, &single, absolute)?;
        let ro = read(client, arena, single.0[c::OUT])?;
        audit(client, retired)?;
        max_single = max_single.max(compare(chunk_output, &floats(&ro[..qsa::QUERY]))?);
        let rc = read(client, arena, single.0[c::CAND])?;
        audit(client, retired)?;
        max_single = max_single.max(compare(
            &candidate[s.p[c::CAND].at(t)..][..qsa::TOKEN],
            &floats(&rc[..qsa::TOKEN]),
        )?);
        let mut ss: [Zeroizing<Vec<u32>>; 11] = std::array::from_fn(|_| Zeroizing::new(Vec::new()));
        for (i, v) in ss.iter_mut().enumerate() {
            *v = read(client, arena, single.0[c::SPLIT_S + i])?;
            audit(client, retired)?;
        }
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
        qsa::validate_token_status(l, std::array::from_fn(|i| &ss[i][..sizes[i]]))?;
        // Reference sequence advances only after all original single-token statuses
        // and its completed readback have passed; true chunk OLD stays immutable.
        rw[c::OLD][absolute * qsa::TOKEN..][..qsa::TOKEN].copy_from_slice(&rc[..qsa::TOKEN]);
    }
    // A late raw key NaN cannot enter any earlier causal pool/attention plane.
    // It invalidates the whole candidate transaction, without a partial commit.
    if previous == 4 && count == 8 {
        let mut bad = Zeroizing::new(vec![0; f.words[c::RAWIK].len()]);
        bad.copy_from_slice(&f.words[c::RAWIK]);
        bad[s.p[c::RAWIK].at(count - 1) + 64] = f32::NAN.to_bits();
        write(client, arena, b.0[c::RAWIK], &bad)?;
        let ready = c::enqueue_prepare(arena, s, Buffers(b.0), &f.words[c::POS], &cancel)?;
        let sc = read(client, arena, b.0[c::SCORE])?;
        let st = read(client, arena, b.0[c::SCORE_S])?;
        audit(client, retired)?;
        let ids = c::select(s, &floats(&sc), &st, &cancel)?;
        c::enqueue_attention(arena, &ready, ids, &cancel)?;
        let later = read(client, arena, b.0[c::OUT])?;
        audit(client, retired)?;
        for t in 0..count - 1 {
            let at = s.p[c::OUT].at(t);
            if later[at..][..qsa::QUERY] != actual[c::OUT][at..][..qsa::QUERY] {
                return Err("QSA chunk future column affected earlier output".into());
            }
        }
        let later_means = read(client, arena, b.0[c::MEANS])?;
        audit(client, retired)?;
        for t in 0..count - 1 {
            let at = s.p[c::MEANS].at(t);
            let width = s.layout(t)?.complete * qsa::ID;
            if later_means[at..][..width] != actual[c::MEANS][at..][..width] {
                return Err("QSA chunk future raw key affected earlier pooled group".into());
            }
        }
        let last = s.p[c::MEANS].at(count - 1) + (s.layout(count - 1)?.complete - 1) * qsa::ID + 64;
        if later_means[last] == actual[c::MEANS][last] {
            return Err("QSA chunk late group diagnostic did not change".into());
        }
        let failed = read(client, arena, b.0[c::CAND_S])?;
        audit(client, retired)?;
        let mut statuses = status;
        statuses[c::CAND_S - c::SPLIT_S] = &failed;
        if c::validate_download(s, [&candidate, &output], statuses, &cancel).is_ok() {
            return Err("QSA chunk accepted late nonfinite candidate".into());
        }
        let old = read(client, arena, b.0[c::OLD])?;
        audit(client, retired)?;
        unchanged(&f.words[c::OLD], &old, |_| false)?;
    }
    let full = arena.charged_bytes();
    arena.reset()?;
    if b.0
        .iter()
        .chain(single.0.iter())
        .any(|&id| arena.handle(id).is_ok())
        || retired.load(Ordering::Acquire)
    {
        return Err("QSA chunk reset/retirement failure".into());
    }
    Ok(
        serde_json::json!({"previous":previous,"count":count,"capacity":CAPACITY,"max_batch":c::MAX_BATCH,"main_full_span_bytes":full,"runtime_first_audit_peak_bound":io_bound(s)?,"max_abs_core_error":max_core,"max_abs_single_error":max_single,"all_token_candidates_compared":true,"late_nonfinite_completed_group_rejected":previous==4&&count==8,"committed_cache_immutable":true,"future_groups_excluded":true,"inactive_gaps_padding_preserved":true,"full_main_zero_inspection":true,"cancel_before_enqueue_after_completion_rejected":true}),
    )
}
pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), MAIN_BUDGET, IO_BUDGET, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut cases = Vec::with_capacity(24);
        for previous in [0, 3, 4, 2047, 2049, 2051] {
            for count in [1, 3, 5, 8] {
                cases.push(case(client, &mut arena, previous, count, &retired)?);
            }
        }
        Ok(cases)
    }));
    let closed = arena.close();
    if closed.is_err() || retired.load(Ordering::Acquire) {
        return Err("QSA chunk cleanup/context failed".into());
    }
    let cases = result.map_err(|_| "QSA chunk probe panic")??;
    let full = arena
        .owned_zero_audit_receipt()
        .ok_or("QSA chunk main inspection missing")?;
    if full.buffers != (24 * N * 2) as u64
        || full.bytes != (24 * owned_plan(fixture_spans(0, 1)?)?) as u64
    {
        return Err("QSA chunk main inspection counts invalid".into());
    }
    let receipt = serde_json::json!({"scope":"fixed-public-synthetic-operator-test-only","fixture":"qsa-chunk-probe","cases":cases,"max_batch":8,"main_budget_bytes":MAIN_BUDGET,"runtime_io_budget_bytes":IO_BUDGET,"main_zero_buffers":full.buffers,"main_zero_bytes":full.bytes,"explicit_owned_cleanup_accepted":true,"context_retired":false,"whole_graph_batched_prefill":false,"production_private_worker_accepted":false,"proof_scope":"API-visible owning pages and tracked IO, excluding opaque driver/register/spill storage"});
    println!(
        "public_qsa_chunk_acceptance cases=24 main_buffers={} main_bytes={} context_retired=false",
        full.buffers, full.bytes
    );
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostics_preserve_acceptance_without_exposing_status_values() {
        let s = fixture_spans(0, 1).unwrap();
        let (data, mut flags) = fake_accepted(s);
        diagnose_acceptance(
            s,
            [&data[0], &data[1]],
            std::array::from_fn(|i| flags[i].as_slice()),
        )
        .unwrap();
        flags[1][s.p[c::Q_S].at(0)] = 0;
        let error = diagnose_acceptance(
            s,
            [&data[0], &data[1]],
            std::array::from_fn(|i| flags[i].as_slice()),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "QSA chunk query RMS/RoPE status");
    }
    #[test]
    fn exact_fixed_pages_and_two_independent_io_bounds() {
        let mut plan = None;
        for previous in [0, 3, 4, 2047, 2049, 2051] {
            for count in [1, 3, 5, 8] {
                let s = fixture_spans(previous, count).unwrap();
                let bytes = owned_plan(s).unwrap();
                let peak = io_bound(s).unwrap();
                if let Some((p, b, k)) = plan {
                    assert_eq!(s.p.map(|x| x.words), p);
                    assert_eq!(bytes, b);
                    assert_eq!(peak, k);
                } else {
                    plan = Some((s.p.map(|x| x.words), bytes, peak));
                }
                assert_eq!(bytes, 57_475_072);
                assert!(bytes <= MAIN_BUDGET);
                assert!(peak <= IO_BUDGET as usize);
            }
        }
        let s = fixture_spans(2051, 8).unwrap();
        let source = (0..=c::OLD)
            .map(|i| BufferLayout::new(s.p[i].words).unwrap().owned_bytes)
            .sum::<usize>();
        assert_eq!(source, 17_432_576);
        assert_eq!(io_bound(s).unwrap(), 35_012_608);
        assert_eq!(
            s.p.map(|p| BufferLayout::new(p.words).unwrap().owned_bytes)
                .iter()
                .sum::<usize>(),
            36_798_464
        );
        for p in s.p {
            let page = BufferLayout::new(p.words).unwrap().owned_bytes;
            assert!(page * 2 <= 32 * 1024 * 1024);
        }
        assert_eq!(PREP_DISPATCHES + ATTENTION_DISPATCHES, 11);
    }
    #[test]
    fn rejects_malformed_full_max8_spans_count_capacity_arithmetic_positions() {
        let good = fixture_spans(2047, 3).unwrap();
        for count in [0, 9, usize::MAX] {
            let mut s = good;
            s.count = count;
            assert!(s.validate().is_err());
        }
        for field in 0..N {
            let mut s = good;
            s.p[field].words =
                s.p[field].base + (s.rows()[field] - 1) * s.p[field].stride + s.widths()[field] - 1;
            assert!(s.validate().is_err());
            let mut s = good;
            s.p[field].stride = s.widths()[field] - 1;
            assert!(s.validate().is_err());
            let mut s = good;
            s.p[field].stride = usize::MAX;
            assert!(s.validate().is_err());
            let mut s = good;
            s.p[field].base = usize::MAX;
            assert!(s.validate().is_err());
        }
        let mut s = good;
        s.previous = usize::MAX;
        assert!(s.validate().is_err());
        let mut s = good;
        s.capacity = qsa::MAX_CONTEXT + 1;
        assert!(s.validate().is_err());
        let f = fixture(3, 3).unwrap();
        let mut positions = Zeroizing::new(vec![0; f.words[c::POS].len()]);
        positions.copy_from_slice(&f.words[c::POS]);
        positions[f.spans.p[c::POS].at(4)] = positions[f.spans.p[c::POS].at(3)];
        assert!(f.spans.validate_positions(&positions).is_err());
        positions[f.spans.p[c::POS].at(4)] = qsa::MAX_POSITION + 1;
        assert!(f.spans.validate_positions(&positions).is_err());
    }
    #[test]
    fn guarded_selection_matches_original_core_at_budget_signed_zero_ties_and_tail() {
        for visible in [1, 3, 4, 5, 7, 8, 2048, 2049, 2052, 2059, 4096] {
            let l = qsa::QsaLayout::new(visible).unwrap();
            let mut scores = Zeroizing::new(vec![0.; l.complete]);
            for (i, x) in scores.iter_mut().enumerate() {
                *x = if i % 11 == 0 {
                    -0.
                } else {
                    ((i * 13) % 31) as f32
                };
            }
            let ids = c::guarded_select(l, &scores, &vec![1; l.complete]).unwrap();
            let original = ops::qsa_select_blocks(&scores, visible, 4, qsa::BUDGET).unwrap();
            assert_eq!(
                ids.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                original
            );
            qsa::validate_selected(l, &ids).unwrap();
        }
    }
    fn fake_accepted(s: Spans) -> ([Vec<f32>; 2], [Vec<u32>; 11]) {
        let mut data = [
            vec![0.; BufferLayout::new(s.p[c::CAND].words).unwrap().owned_words],
            vec![0.; BufferLayout::new(s.p[c::OUT].words).unwrap().owned_words],
        ];
        let mut flags = std::array::from_fn(|i| {
            vec![
                STATUS_GAP;
                BufferLayout::new(s.p[c::SPLIT_S + i].words)
                    .unwrap()
                    .owned_words
            ]
        });
        for t in 0..s.count {
            let l = s.layout(t).unwrap();
            let lengths = [
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
            for (i, &n) in lengths.iter().enumerate() {
                flags[i][s.p[c::SPLIT_S + i].at(t)..][..n].fill(1);
            }
            data[0][s.p[c::CAND].at(t)..][..qsa::TOKEN].fill(1.);
            data[1][s.p[c::OUT].at(t)..][..qsa::QUERY].fill(2.);
        }
        (data, flags)
    }
    #[test]
    fn no_partial_status_nonfinite_download_surplus_or_cancel_acceptance() {
        let s = fixture_spans(2049, 8).unwrap();
        let (data, flags) = fake_accepted(s);
        let no = AtomicBool::new(false);
        let check = |data: &[Vec<f32>; 2], flags: &[Vec<u32>; 11], cancel: &AtomicBool| {
            c::validate_download(
                s,
                [&data[0], &data[1]],
                std::array::from_fn(|i| flags[i].as_slice()),
                cancel,
            )
        };
        check(&data, &flags, &no).unwrap();
        for stage in 0..11 {
            let mut bad = flags.clone();
            bad[stage][s.p[c::SPLIT_S + stage].at(7)] = 0;
            assert!(check(&data, &bad, &no).is_err());
        }
        let mut bad = data.clone();
        bad[0][s.p[c::CAND].at(7)] = f32::NAN;
        assert!(check(&bad, &flags, &no).is_err());
        let mut bad = data.clone();
        bad[1].push(0.);
        assert!(check(&bad, &flags, &no).is_err());
        let mut bad = data.clone();
        bad[1].pop();
        assert!(check(&bad, &flags, &no).is_err());
        assert!(check(&data, &flags, &AtomicBool::new(true)).is_err());
    }
    fn causal_means(f: &Fixture, t: usize) -> Vec<f32> {
        let s = f.spans;
        let complete = (s.previous + t + 1) / 4;
        let mut out = vec![0.; complete * qsa::ID];
        for b in 0..complete {
            for d in 0..qsa::ID {
                let mut sum = 0f32;
                for j in 0..4 {
                    let token = b * 4 + j;
                    let word = if token < s.previous {
                        f.words[c::OLD][s.p[c::OLD].at(token) + qsa::KV * 2 + d]
                    } else {
                        f.words[c::RAWIK][s.p[c::RAWIK].at(token - s.previous) + d]
                    };
                    sum += f32::from_bits(word);
                }
                out[b * qsa::ID + d] = sum / 4.;
            }
        }
        out
    }
    #[test]
    fn later_completed_groups_and_future_raw_keys_invisible_to_earlier_planes() {
        for previous in [0, 3, 4, 2047, 2049, 2051] {
            let mut f = fixture(previous, 8).unwrap();
            let old = f.words[c::OLD].clone();
            let before = (0..8).map(|t| causal_means(&f, t)).collect::<Vec<_>>();
            for (t, means) in before.iter().enumerate() {
                assert_eq!(means.len(), (previous + t + 1) / 4 * qsa::ID);
            }
            for d in 0..qsa::ID {
                f.words[c::RAWIK][f.spans.p[c::RAWIK].at(7) + d] = f32::NAN.to_bits();
            }
            for (t, means) in before.iter().enumerate().take(7) {
                assert_eq!(causal_means(&f, t), *means);
            }
            assert_eq!(*f.words[c::OLD], *old);
        }
    }
    #[test]
    fn exact_dispatch_inventory_and_packed_info_bound() {
        let source = include_str!("qsa_chunk.rs");
        let specs = [
            ("split_batch", 3, 7),
            ("pack_batch", 5, 11),
            ("norm_batch", 7, 18),
            ("pool_batch", 4, 10),
            ("score_batch", 4, 10),
            ("logits_batch", 6, 14),
            ("softmax_batch", 3, 8),
            ("values_batch", 7, 16),
        ];
        for (name, arrays, scalars) in specs {
            let sig = source
                .split(&format!("pub fn {name}("))
                .nth(1)
                .unwrap()
                .split(") {")
                .next()
                .unwrap();
            let runtime = sig.split("#[comptime]").next().unwrap();
            assert_eq!(runtime.matches("Array<").count(), arrays);
            assert_eq!(
                runtime.matches(": usize").count() + runtime.matches(": u32").count(),
                scalars
            );
            let mut info = cubecl::InfoBuilder::default();
            let address = cubecl::ir::AddressType::U64;
            for _ in 0..arrays {
                info.metadata.register_array(4_194_304, 4_194_304, address);
            }
            for _ in 0..scalars {
                info.scalars.push(1u64);
            }
            assert!(info.finish(address).data.len() * 8 <= INFO_BYTES);
        }
        let prep = source
            .split("pub fn enqueue_prepare(")
            .nth(1)
            .unwrap()
            .split("struct Upload")
            .next()
            .unwrap();
        assert_eq!(prep.matches("::launch").count(), 6);
        let attention = source
            .split("pub fn enqueue_attention(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn guarded_select")
            .next()
            .unwrap();
        assert_eq!(attention.matches("::launch").count(), 3);
        assert!(source.contains("token < previous + plane + 1"));
        assert!(source.contains("t < previous + token + 1"));
    }
    fn shader(name: &str, width: usize, mode: cubecl::server::ExecutionMode) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope.register_type::<f32>(float);
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        match name {
            "split_batch" => {
                let v_raw = b.input_array(Type::new(float)).into();
                let v_packed = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_count = b.scalar(uint).into();
                let v_rb = b.scalar(uint).into();
                let v_rs = b.scalar(uint).into();
                let v_pb = b.scalar(uint).into();
                let v_ps = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                c::split_batch::expand(
                    &mut b.scope,
                    v_raw,
                    v_packed,
                    v_status,
                    v_count,
                    v_rb,
                    v_rs,
                    v_pb,
                    v_ps,
                    v_sb,
                    v_ss,
                );
            }

            "pack_batch" => {
                let v_key = b.input_array(Type::new(float)).into();
                let v_value = b.input_array(Type::new(float)).into();
                let v_raw_index_key = b.input_array(Type::new(float)).into();
                let v_token = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_count = b.scalar(uint).into();
                let v_kb = b.scalar(uint).into();
                let v_ks = b.scalar(uint).into();
                let v_vb = b.scalar(uint).into();
                let v_vs = b.scalar(uint).into();
                let v_rb = b.scalar(uint).into();
                let v_rs = b.scalar(uint).into();
                let v_cb = b.scalar(uint).into();
                let v_cs = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                c::pack_batch::expand(
                    &mut b.scope,
                    v_key,
                    v_value,
                    v_raw_index_key,
                    v_token,
                    v_status,
                    v_count,
                    v_kb,
                    v_ks,
                    v_vb,
                    v_vs,
                    v_rb,
                    v_rs,
                    v_cb,
                    v_cs,
                    v_sb,
                    v_ss,
                );
            }

            "norm_batch" => {
                let v_input = b.input_array(Type::new(float)).into();
                let v_gamma = b.input_array(Type::new(float)).into();
                let v_sin_table = b.input_array(Type::new(float)).into();
                let v_cos_table = b.input_array(Type::new(float)).into();
                let v_positions = b.input_array(Type::new(uint)).into();
                let v_output = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_count = b.scalar(uint).into();
                let v_previous = b.scalar(uint).into();
                let v_heads = b.scalar(uint).into();
                let v_ib = b.scalar(uint).into();
                let v_is_ = b.scalar(uint).into();
                let v_gb = b.scalar(uint).into();
                let v_gs = b.scalar(uint).into();
                let v_sinb = b.scalar(uint).into();
                let v_sins = b.scalar(uint).into();
                let v_cosb = b.scalar(uint).into();
                let v_coss = b.scalar(uint).into();
                let v_pb = b.scalar(uint).into();
                let v_ps = b.scalar(uint).into();
                let v_ob = b.scalar(uint).into();
                let v_os = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                let v_block_positions = b.scalar(uint).into();
                c::norm_batch::expand(
                    &mut b.scope,
                    v_input,
                    v_gamma,
                    v_sin_table,
                    v_cos_table,
                    v_positions,
                    v_output,
                    v_status,
                    v_count,
                    v_previous,
                    v_heads,
                    v_ib,
                    v_is_,
                    v_gb,
                    v_gs,
                    v_sinb,
                    v_sins,
                    v_cosb,
                    v_coss,
                    v_pb,
                    v_ps,
                    v_ob,
                    v_os,
                    v_sb,
                    v_ss,
                    v_block_positions,
                    width,
                );
            }

            "pool_batch" => {
                let v_old = b.input_array(Type::new(float)).into();
                let v_current = b.input_array(Type::new(float)).into();
                let v_means = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_previous = b.scalar(uint).into();
                let v_count = b.scalar(uint).into();
                let v_ob = b.scalar(uint).into();
                let v_os = b.scalar(uint).into();
                let v_cb = b.scalar(uint).into();
                let v_cs = b.scalar(uint).into();
                let v_mb = b.scalar(uint).into();
                let v_ms = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                c::pool_batch::expand(
                    &mut b.scope,
                    v_old,
                    v_current,
                    v_means,
                    v_status,
                    v_previous,
                    v_count,
                    v_ob,
                    v_os,
                    v_cb,
                    v_cs,
                    v_mb,
                    v_ms,
                    v_sb,
                    v_ss,
                );
            }

            "score_batch" => {
                let v_pooled = b.input_array(Type::new(float)).into();
                let v_query = b.input_array(Type::new(float)).into();
                let v_scores = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_count = b.scalar(uint).into();
                let v_previous = b.scalar(uint).into();
                let v_pb = b.scalar(uint).into();
                let v_ps = b.scalar(uint).into();
                let v_qb = b.scalar(uint).into();
                let v_qs = b.scalar(uint).into();
                let v_ob = b.scalar(uint).into();
                let v_os = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                c::score_batch::expand(
                    &mut b.scope,
                    v_pooled,
                    v_query,
                    v_scores,
                    v_status,
                    v_count,
                    v_previous,
                    v_pb,
                    v_ps,
                    v_qb,
                    v_qs,
                    v_ob,
                    v_os,
                    v_sb,
                    v_ss,
                );
            }

            "logits_batch" => {
                let v_query = b.input_array(Type::new(float)).into();
                let v_old = b.input_array(Type::new(float)).into();
                let v_current = b.input_array(Type::new(float)).into();
                let v_selected = b.input_array(Type::new(uint)).into();
                let v_logits = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_previous = b.scalar(uint).into();
                let v_tokens = b.scalar(uint).into();
                let v_qb = b.scalar(uint).into();
                let v_qs = b.scalar(uint).into();
                let v_ob = b.scalar(uint).into();
                let v_os = b.scalar(uint).into();
                let v_cb = b.scalar(uint).into();
                let v_cs = b.scalar(uint).into();
                let v_ib = b.scalar(uint).into();
                let v_is_ = b.scalar(uint).into();
                let v_lb = b.scalar(uint).into();
                let v_ls = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                c::logits_batch::expand(
                    &mut b.scope,
                    v_query,
                    v_old,
                    v_current,
                    v_selected,
                    v_logits,
                    v_status,
                    v_previous,
                    v_tokens,
                    v_qb,
                    v_qs,
                    v_ob,
                    v_os,
                    v_cb,
                    v_cs,
                    v_ib,
                    v_is_,
                    v_lb,
                    v_ls,
                    v_sb,
                    v_ss,
                );
            }

            "softmax_batch" => {
                let v_logits = b.input_array(Type::new(float)).into();
                let v_probability = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_tokens = b.scalar(uint).into();
                let v_previous = b.scalar(uint).into();
                let v_lb = b.scalar(uint).into();
                let v_ls = b.scalar(uint).into();
                let v_pb = b.scalar(uint).into();
                let v_ps = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                c::softmax_batch::expand(
                    &mut b.scope,
                    v_logits,
                    v_probability,
                    v_status,
                    v_tokens,
                    v_previous,
                    v_lb,
                    v_ls,
                    v_pb,
                    v_ps,
                    v_sb,
                    v_ss,
                );
            }

            "values_batch" => {
                let v_old = b.input_array(Type::new(float)).into();
                let v_current = b.input_array(Type::new(float)).into();
                let v_probability = b.input_array(Type::new(float)).into();
                let v_selected = b.input_array(Type::new(uint)).into();
                let v_raw_gate = b.input_array(Type::new(float)).into();
                let v_output = b.output_array(Type::new(float)).into();
                let v_status = b.output_array(Type::new(uint)).into();
                let v_previous = b.scalar(uint).into();
                let v_tokens = b.scalar(uint).into();
                let v_ob = b.scalar(uint).into();
                let v_os = b.scalar(uint).into();
                let v_cb = b.scalar(uint).into();
                let v_cs = b.scalar(uint).into();
                let v_pb = b.scalar(uint).into();
                let v_ps = b.scalar(uint).into();
                let v_ib = b.scalar(uint).into();
                let v_is_ = b.scalar(uint).into();
                let v_gb = b.scalar(uint).into();
                let v_gs = b.scalar(uint).into();
                let v_yb = b.scalar(uint).into();
                let v_ys = b.scalar(uint).into();
                let v_sb = b.scalar(uint).into();
                let v_ss = b.scalar(uint).into();
                c::values_batch::expand(
                    &mut b.scope,
                    v_old,
                    v_current,
                    v_probability,
                    v_selected,
                    v_raw_gate,
                    v_output,
                    v_status,
                    v_previous,
                    v_tokens,
                    v_ob,
                    v_os,
                    v_cb,
                    v_cs,
                    v_pb,
                    v_ps,
                    v_ib,
                    v_is_,
                    v_gb,
                    v_gs,
                    v_yb,
                    v_ys,
                    v_sb,
                    v_ss,
                );
            }

            _ => panic!("unknown public kernel"),
        };
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name(name),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                uint,
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn all_offline_checked_unchecked_causal_kernels_parse_and_validate() {
        for name in [
            "split_batch",
            "pack_batch",
            "norm_batch",
            "pool_batch",
            "score_batch",
            "logits_batch",
            "softmax_batch",
            "values_batch",
        ] {
            for width in if name == "norm_batch" {
                vec![128, 256]
            } else {
                vec![128]
            } {
                for mode in [
                    cubecl::server::ExecutionMode::Checked,
                    cubecl::server::ExecutionMode::Unchecked,
                ] {
                    let wgsl = shader(name, width, mode);
                    assert!(wgsl.contains("scalars_u32"));
                    assert!(!wgsl.contains("fma("));
                    let module = wgpu::naga::front::wgsl::parse_str(&wgsl).unwrap();
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::empty(),
                    )
                    .validate(&module)
                    .unwrap();
                }
            }
        }
    }
    #[test]
    fn pointwise_shader_indices_exclude_token_plane_y() {
        for name in ["split_batch", "pack_batch", "values_batch"] {
            for mode in [
                cubecl::server::ExecutionMode::Checked,
                cubecl::server::ExecutionMode::Unchecked,
            ] {
                let wgsl = shader(name, 128, mode);
                assert!(wgsl.contains("global_id.x"));
                assert!(wgsl.contains("workgroup_id.y"));
                assert!(
                    !wgsl.contains("num_workgroups"),
                    "flattened grid index returned in {name}"
                );
            }
        }
        for count in [1, 3, 5, 8] {
            for width in [qsa::QUERY * 2, qsa::TOKEN, qsa::QUERY] {
                let mut visited = vec![0u8; c::MAX_BATCH * width];
                for y in 0..c::MAX_BATCH {
                    for x in 0..width.div_ceil(64) {
                        for lane in 0..64 {
                            let i = x * 64 + lane;
                            if y < count && i < width {
                                visited[y * width + i] += 1;
                            }
                        }
                    }
                }
                for (index, &visits) in visited.iter().enumerate() {
                    assert_eq!(visits, if index / width < count { 1 } else { 0 });
                }
            }
        }
    }
}
