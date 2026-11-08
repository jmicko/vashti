//! Fixed-public original layer3 QSA composition acceptance; no graph commit.
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    qsa_chunk::{self as c},
    qsa_gpu as qsa,
    qsa_layer_batch::{self as l, Buffers, Geometry, N, PublicLayer, Result},
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::{GgufFile, GgufLimits, model::Qwen4Exp, ops};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
const GAP: f32 = -173.25;
const FLAG: u32 = 0x53aa9267;
const HOST_CAP: usize = 256 * 1024 * 1024;
type Images = [Zeroizing<Vec<u32>>; N];
type CoreToken = (Vec<(usize, Zeroizing<Vec<f32>>)>, Zeroizing<Vec<u32>>);
struct Lease(cubecl::bytes::Bytes);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn read(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
    let bytes = Lease(
        arena
            .client()?
            .read_one(arena.handle(id)?.clone())
            .map_err(|_| "QSA layer owning read failed")?,
    );
    decode_read_packet(&bytes.0, arena.layout(id)?)
}
fn decode_read_packet(bytes: &[u8], layout: BufferLayout) -> Result<Zeroizing<Vec<u32>>> {
    if bytes.len() != layout.owned_bytes {
        return Err("QSA layer malformed owning read".into());
    }
    // Guard exists before reserve/push; never make an unguarded Vec copy.
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(bytes.len() / 4)?;
    for b in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes(b.try_into()?));
    }
    Ok(out)
}
fn audit(arena: &PrivateArena, retired: &AtomicBool) -> Result<()> {
    let client = arena.client()?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io_audited())
    }));
    let r = match result {
        Ok(Ok(r)) => r,
        _ => {
            retired.store(true, Ordering::Release);
            return Err("QSA layer IO cleanup failed".into());
        }
    };
    check_audit(
        r.upload_bytes,
        r.uniform_bytes,
        r.readback_bytes,
        r.inspected_bytes,
        r.inspected_buffers,
        retired,
    )
}
fn check_audit(
    upload: u64,
    uniform: u64,
    readback: u64,
    inspected: u64,
    buffers: u64,
    retired: &AtomicBool,
) -> Result<()> {
    let full = upload
        .checked_add(uniform)
        .and_then(|n| n.checked_add(readback));
    if full.is_none_or(|n| n == 0 || n != inspected)
        || buffers == 0
        || retired.load(Ordering::Acquire)
    {
        retired.store(true, Ordering::Release);
        return Err("QSA layer incomplete IO audit".into());
    }
    Ok(())
}
struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn write(arena: &PrivateArena, id: BufferId, words: &[u32]) -> Result<()> {
    let layout = arena.layout(id)?;
    if words.len() != layout.logical_words {
        return Err("QSA layer exact upload extent".into());
    }
    let mut upload = Upload(Vec::new());
    upload.0.try_reserve_exact(layout.owned_bytes)?;
    for &x in words {
        upload.0.extend_from_slice(&x.to_le_bytes());
    }
    upload.0.resize(layout.owned_bytes, 0);
    arena
        .client()?
        .write_private(arena.handle(id)?.clone(), std::mem::take(&mut upload.0))
        .map_err(|_| "QSA layer owning write failed")?;
    Ok(())
}
fn scalar_prepare(arena: &PrivateArena, b: &c::Buffers, previous: usize) -> Result<()> {
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
fn scalar_attention(arena: &PrivateArena, b: &c::Buffers, previous: usize) -> Result<()> {
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

fn floats(words: &[u32]) -> Zeroizing<Vec<f32>> {
    let mut out = Zeroizing::new(Vec::new());
    out.reserve_exact(words.len());
    out.extend(words.iter().map(|&x| f32::from_bits(x)));
    out
}
fn mixed_input(t: usize) -> Zeroizing<Vec<f32>> {
    let mut x = Zeroizing::new(Vec::new());
    x.reserve_exact(l::D);
    x.extend((0..l::D).map(|d| (((d * 7 + t * 11 + 3) % 37) as f32 - 18.) / 512.));
    x
}
fn host_preflight() -> Result<usize> {
    let a = l::Admission::new(l::CAPACITY)?;
    let b = Geometry::new(0, 8, l::CAPACITY, false)?;
    let s = Geometry::new(0, 1, l::CAPACITY, true)?;
    let before =
        b.p.iter()
            .chain(s.p.iter())
            .try_fold(0usize, |n, p| -> Result<_> {
                Ok(
                    n.checked_add(p.words.checked_mul(4).ok_or("QSA host overflow")?)
                        .ok_or("QSA host overflow")?,
                )
            })?;
    // Owning images, snapshots, scalar upload scratch, prefix/current CPU
    // key/value/rawIK triples, stage references, full decode lease and Bytes.
    // Original decoded matrix pages have their separate public256MiB plan.
    let working = 4 * l::CAPACITY * qsa::TOKEN * 4
        + 4 * qsa::QH * (qsa::BUDGET + 3) * 4
        + 8 * (qsa::QUERY * 2 + qsa::KV * 2 + qsa::INDEX_QUERY + qsa::ID) * 4
        + 2 * 16 * 1024 * 1024
        + 4 * qsa::ROPE_TABLE_WORDS * 4
        + 1024 * 1024;
    let full = before
        .checked_add(a.main)
        .and_then(|n| n.checked_add(working))
        .ok_or("QSA host overflow")?;
    if full > HOST_CAP {
        return Err("QSA private host preflight".into());
    }
    Ok(full)
}
struct Fixture {
    g: Geometry,
    before: Images,
    cache: Zeroizing<Vec<u32>>,
    positions: Zeroizing<Vec<u32>>,
}
fn initial_images(g: Geometry, public: &PublicLayer) -> Result<Fixture> {
    g.validate()?;
    host_preflight()?;
    let mut before: Images = std::array::from_fn(|_| Zeroizing::new(Vec::new()));
    for (i, v) in before.iter_mut().enumerate() {
        v.try_reserve_exact(g.p[i].words)?;
        v.resize(
            g.p[i].words,
            if i == c::SELECT {
                0
            } else if (c::SPLIT_S..c::N).contains(&i) || i >= l::PROJECTION_STATUS {
                FLAG
            } else {
                GAP.to_bits()
            },
        );
    }
    let mut cache = Zeroizing::new(Vec::new());
    cache.try_reserve_exact(g.qsa.capacity * qsa::TOKEN)?;
    cache.resize(g.qsa.capacity * qsa::TOKEN, GAP.to_bits());
    let mut positions = Zeroizing::new(Vec::new());
    positions.try_reserve_exact(g.qsa.capacity)?;
    positions.extend(0..g.qsa.capacity as u32);
    for t in 0..g.qsa.capacity {
        before[c::POS][g.p[c::POS].at(t)] = t as u32;
    }
    for (a, gamma) in public.gamma.iter().enumerate() {
        let i = c::GQ + a;
        for h in 0..g.qsa.rows()[i] {
            for (d, &x) in gamma.iter().enumerate() {
                before[i][g.p[i].at(h) + d] = x.to_bits();
            }
        }
    }
    let (sin, cos) = qsa::rope_tables();
    for (i, table) in [(c::SIN, sin), (c::COS, cos)] {
        for t in 0..qsa::MAX_CONTEXT {
            for d in 0..qsa::ROPE_PAIRS {
                before[i][g.p[i].at(t) + d] = table[t * qsa::ROPE_PAIRS + d].to_bits();
            }
        }
    }
    for t in 0..g.qsa.previous {
        let k = normalized(&raw_key(t), &public.gamma[1], qsa::HD, t as u32)?;
        let v = raw_value(t);
        let ik = raw_index(t);
        for (d, &x) in k.iter().chain(v.iter()).chain(ik.iter()).enumerate() {
            before[c::OLD][g.p[c::OLD].at(t) + d] = x.to_bits();
            cache[t * qsa::TOKEN + d] = x.to_bits();
        }
    }
    for t in 0..g.qsa.count {
        for (d, &x) in mixed_input(t).iter().enumerate() {
            before[l::INPUT][g.p[l::INPUT].at(t) + d] = x.to_bits();
        }
    }
    if !g.scalar {
        g.qsa.validate_positions(&before[c::POS])?;
    }
    Ok(Fixture {
        g,
        before,
        cache,
        positions,
    })
}
fn allocate(arena: &mut PrivateArena, f: &Fixture) -> Result<Buffers> {
    let mut ids = Vec::new();
    ids.try_reserve_exact(N)?;
    for v in &f.before {
        ids.push(arena.upload_u32(v)?);
    }
    Ok(Buffers(
        ids.try_into().map_err(|_| "QSA full layer buffer count")?,
    ))
}
fn snapshot(arena: &PrivateArena, b: &Buffers, retired: &AtomicBool) -> Result<Images> {
    // Last projection finite status is an ordered whole-layer completed fence.
    let fence = read(arena, b.0[l::PROJECTION_STATUS + 5])?;
    audit(arena, retired)?;
    let mut out: Images = std::array::from_fn(|_| Zeroizing::new(Vec::new()));
    for (i, v) in out.iter_mut().enumerate() {
        *v = read(arena, b.0[i])?;
        audit(arena, retired)?;
    }
    if *out[l::PROJECTION_STATUS + 5] != *fence {
        return Err("QSA final fence changed".into());
    }
    Ok(out)
}
fn compare(a: &[f32], e: &[f32], token: usize, stage: usize, reference: &str) -> Result<f32> {
    if a.len() != e.len() {
        return Err("QSA full layer parity extent".into());
    }
    let mut maximum = 0f32;
    for (d, (&x, &y)) in a.iter().zip(e).enumerate() {
        let error = (x - y).abs();
        let tolerance = 5e-5 + 5e-4 * y.abs();
        if !x.is_finite() || !y.is_finite() || error > tolerance {
            // Fixed-public mode only. Do not reuse numeric formatting for a
            // production private layer, request, route or worker failure.
            return Err(format!("fixed-public QSA layer parity reference={reference} token={token} stage={stage} index={d} actual={x} expected={y} abs={error} tolerance={tolerance}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn field(words: &[u32], g: Geometry, i: usize, t: usize) -> Result<Zeroizing<Vec<f32>>> {
    let width = g.active_width(i, t)?;
    Ok(floats(&words[g.p[i].at(t)..][..width]))
}
struct CoreSequence {
    keys: Zeroizing<Vec<f32>>,
    values: Zeroizing<Vec<f32>>,
    index: Zeroizing<Vec<f32>>,
}
impl CoreSequence {
    fn new(f: &Fixture) -> Result<Self> {
        let mut keys = Zeroizing::new(Vec::new());
        let mut values = Zeroizing::new(Vec::new());
        let mut index = Zeroizing::new(Vec::new());
        keys.try_reserve_exact(f.g.qsa.capacity * qsa::KV)?;
        values.try_reserve_exact(f.g.qsa.capacity * qsa::KV)?;
        index.try_reserve_exact(f.g.qsa.capacity * qsa::ID)?;
        for token in f.cache.chunks_exact(qsa::TOKEN).take(f.g.qsa.previous) {
            keys.extend(token[..qsa::KV].iter().map(|&x| f32::from_bits(x)));
            values.extend(
                token[qsa::KV..2 * qsa::KV]
                    .iter()
                    .map(|&x| f32::from_bits(x)),
            );
            index.extend(token[2 * qsa::KV..].iter().map(|&x| f32::from_bits(x)));
        }
        Ok(Self {
            keys,
            values,
            index,
        })
    }
    fn token(&mut self, public: &PublicLayer, f: &Fixture, t: usize) -> Result<CoreToken> {
        let visible = f.g.qsa.previous + t + 1;
        let x = mixed_input(t);
        let mut projected = Vec::new();
        projected.try_reserve_exact(5)?;
        for m in public.matrices.iter().take(5) {
            projected.push(m.cpu(&x)?);
        }
        let mut packed = Zeroizing::new(Vec::new());
        packed.try_reserve_exact(2 * qsa::QUERY)?;
        let mut gate = Zeroizing::new(Vec::new());
        gate.try_reserve_exact(qsa::QUERY)?;
        for h in projected[0].chunks_exact(2 * qsa::HD) {
            packed.extend_from_slice(&h[..qsa::HD]);
            gate.extend_from_slice(&h[qsa::HD..]);
        }
        packed.extend_from_slice(&gate);
        let query = normalized(
            &packed[..qsa::QUERY],
            &public.gamma[0],
            qsa::HD,
            (visible - 1) as u32,
        )?;
        let key = normalized(
            &projected[1],
            &public.gamma[1],
            qsa::HD,
            (visible - 1) as u32,
        )?;
        let iq = normalized(
            &projected[3],
            &public.gamma[2],
            qsa::ID,
            (visible - 1) as u32,
        )?;
        let mut candidate = Zeroizing::new(Vec::new());
        candidate.try_reserve_exact(qsa::TOKEN)?;
        candidate.extend_from_slice(&key);
        candidate.extend_from_slice(&projected[2]);
        candidate.extend_from_slice(&projected[4]);
        self.keys.extend_from_slice(&key);
        self.values.extend_from_slice(&projected[2]);
        self.index.extend_from_slice(&projected[4]);
        let score = Zeroizing::new(ops::qsa_block_scores(
            &self.index,
            &iq,
            &public.gamma[3],
            &f.positions[..visible],
            visible,
            ops::QsaShape::qwen38(),
        )?);
        let selected = Zeroizing::new(ops::qsa_select_blocks(&score, visible, 4, qsa::BUDGET)?);
        let mut ids = Zeroizing::new(Vec::new());
        ids.try_reserve_exact(selected.len())?;
        ids.extend(selected.iter().map(|&i| i as u32));
        let mut out = Zeroizing::new(ops::qsa_attend(
            &query,
            &self.keys,
            &self.values,
            &selected,
            visible,
            ops::QsaShape::qwen38(),
        )?);
        for (v, &z) in out.iter_mut().zip(gate.iter()) {
            *v *= ops::sigmoid(z);
        }
        let final_output = public.matrices[5].cpu(&out)?;
        let mut fields = Vec::new();
        fields.try_reserve_exact(13)?;
        for (i, v) in l::PROJECTED.into_iter().take(5).zip(projected) {
            fields.push((i, v));
        }
        fields.extend([
            (c::PACK, packed),
            (c::Q, query),
            (c::K, key),
            (c::IQ, iq),
            (c::CAND, candidate),
            (c::SCORE, score),
            (c::OUT, out),
            (l::OUTPUT, final_output),
        ]);
        Ok((fields, ids))
    }
}
fn scalar_status(g: Geometry, data: &Images) -> Result<()> {
    let l = g.qsa.layout(0)?;
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
    qsa::validate_token_status(
        l,
        std::array::from_fn(|i| &data[c::SPLIT_S + i][..sizes[i]]),
    )?;
    for (i, (rows, _)) in l::matrix_shapes().into_iter().enumerate() {
        if data[l::PROJECTION_STATUS + i][..rows]
            .iter()
            .any(|&x| x != 1)
        {
            return Err("QSA scalar projection finite flag".into());
        }
    }
    Ok(())
}
fn read_selection(
    arena: &PrivateArena,
    b: &Buffers,
    g: Geometry,
    retired: &AtomicBool,
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<u32>>> {
    let scores = read(arena, b.0[c::SCORE])?;
    let flags = read(arena, b.0[c::SCORE_S])?;
    audit(arena, retired)?;
    Ok(c::select(g.qsa, &floats(&scores), &flags, cancel)?)
}
fn correct_case(
    arena: &mut PrivateArena,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    public: &PublicLayer,
    previous: usize,
    count: usize,
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    println!("public_qsa_layer_case previous={previous} count={count}");
    let f = initial_images(Geometry::new(previous, count, l::CAPACITY, false)?, public)?;
    let mut rf = initial_images(Geometry::new(previous, 1, l::CAPACITY, true)?, public)?;
    let b = allocate(arena, &f)?;
    let single = allocate(arena, &rf)?;
    if arena.charged_bytes() != l::Admission::new(l::CAPACITY)?.main {
        return Err("QSA full page charge mismatch".into());
    }
    let ids = b.0.into_iter().chain(single.0).collect::<Vec<_>>();
    let words =
        f.g.p
            .map(|p| p.words)
            .into_iter()
            .chain(rf.g.p.map(|p| p.words))
            .collect::<Vec<_>>();
    c::validate_pages(arena, &ids, &words)?;
    l::binding_proof(arena, rf.g, &single, public)?;
    let mut layer = l::LayerBatch::new(graph, f.g, Buffers(b.0))?;
    let cancel = AtomicBool::new(false);
    layer.prepare(graph, arena, public, &f.before[c::POS], &cancel)?;
    let selected = read_selection(arena, &b, f.g, retired, &cancel)?;
    let mut expected_ids = Zeroizing::new(Vec::new());
    expected_ids.try_reserve_exact(selected.len())?;
    expected_ids.extend_from_slice(&selected);
    layer.attention(graph, arena, public, selected, &cancel)?;
    let data = snapshot(arena, &b, retired)?;
    l::validate_snapshot(f.g, &data, &f.before, &cancel)?;
    let mut core = CoreSequence::new(&f)?;
    let mut max_core = 0f32;
    let mut max_scalar = 0f32;
    for t in 0..count {
        let absolute = previous + t;
        let visible = f.g.qsa.layout(t)?;
        let (expected, core_ids) = core.token(public, &f, t)?;
        for (i, values) in expected {
            max_core = max_core.max(compare(
                &field(&data[i], f.g, i, t)?,
                &values,
                t,
                i,
                "core",
            )?);
        }
        let at = f.g.p[c::SELECT].at(t);
        if expected_ids[at..][..visible.selected] != *core_ids {
            return Err("QSA layer core selector mismatch".into());
        }
        rf.g.qsa.previous = absolute;
        rf.before[l::INPUT].clear();
        rf.before[l::INPUT].extend(mixed_input(t).iter().map(|x| x.to_bits()));
        write(arena, single.0[l::INPUT], &rf.before[l::INPUT])?;
        write(arena, single.0[c::OLD], &rf.before[c::OLD])?;
        // Reset the complete scalar status planes: a stale successful bit from
        // the prior comparator token must never stand in for current work.
        for i in (c::SPLIT_S..c::N).chain(l::PROJECTION_STATUS..N) {
            rf.before[i].fill(FLAG);
            write(arena, single.0[i], &rf.before[i])?;
        }
        for which in 0..5 {
            l::project(arena, public, rf.g, &single, which)?;
        }
        scalar_prepare(arena, &single.chunk(), absolute)?;
        let rs = read(arena, single.0[c::SCORE])?;
        let flags = read(arena, single.0[c::SCORE_S])?;
        audit(arena, retired)?;
        let ids = c::guarded_select(
            visible,
            &floats(&rs[..visible.complete]),
            &flags[..visible.complete],
        )?;
        if ids.as_slice() != &expected_ids[at..][..visible.selected] {
            return Err("QSA layer original scalar selector mismatch".into());
        }
        rf.before[c::SELECT].fill(0);
        rf.before[c::SELECT][..visible.selected].copy_from_slice(&ids);
        write(arena, single.0[c::SELECT], &rf.before[c::SELECT])?;
        scalar_attention(arena, &single.chunk(), absolute)?;
        l::project(arena, public, rf.g, &single, 5)?;
        let rd = snapshot(arena, &single, retired)?;
        scalar_status(rf.g, &rd)?;
        l::validate_snapshot(rf.g, &rd, &rf.before, &cancel)?;
        for i in (0..=c::RAWIK).chain(c::PACK..=c::OUT).chain([l::OUTPUT]) {
            if i == c::SELECT {
                continue;
            }
            max_scalar = max_scalar.max(compare(
                &field(&data[i], f.g, i, t)?,
                &field(&rd[i], rf.g, i, 0)?,
                t,
                i,
                "original-single-token-GPU",
            )?);
        }
        // Advance the separate fixed-public comparator only after all statuses
        // and completed read/erase audits. Chunk committed OLD stays readonly.
        for i in (0..=c::RAWIK).chain(c::PACK..c::N).chain([l::OUTPUT]) {
            rf.before[i].copy_from_slice(&rd[i][..rf.g.p[i].words]);
        }
        rf.before[c::OLD][absolute * qsa::TOKEN..][..qsa::TOKEN]
            .copy_from_slice(&rd[c::CAND][..qsa::TOKEN]);
    }
    layer.accept(
        graph,
        arena,
        &data,
        &f.before,
        &cancel,
        retired.load(Ordering::Acquire),
    )?;
    if layer.phase() != l::Phase::Accepted {
        return Err("QSA full layer candidate not accepted".into());
    }
    let bytes = arena.charged_bytes();
    arena.reset()?;
    if ids.iter().any(|&id| arena.handle(id).is_ok()) || retired.load(Ordering::Acquire) {
        return Err("QSA stale IDs/retirement after reset".into());
    }
    Ok(
        serde_json::json!({"previous":previous,"count":count,"capacity":l::CAPACITY,"main_full_page_bytes":bytes,"max_abs_core_error":max_core,"max_abs_original_scalar_error":max_scalar,"all_six_original_packed_projections":true,"all_17_stage_statuses":true,"readonly_committed_cache":true,"all_intermediates_compared":true,"inactive_gaps_owned_padding_preserved":true,"full_candidate_only_no_commit":true,"checked_main_io_erasure":true}),
    )
}
fn failure_case(
    arena: &mut PrivateArena,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    public: &PublicLayer,
    kind: usize,
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    let mut f = initial_images(Geometry::new(4, 8, l::CAPACITY, false)?, public)?;
    if kind == 0 {
        f.before[l::INPUT][f.g.p[l::INPUT].at(7)] = f32::NAN.to_bits();
    }
    let b = allocate(arena, &f)?;
    if kind == 3 {
        let mut alias = b.0;
        alias[c::RAWK] = alias[c::VALUE];
        if l::binding_proof(arena, f.g, &Buffers(alias), public).is_ok() {
            return Err("QSA physical alias accepted".into());
        }
        arena.reset()?;
        return Ok(
            serde_json::json!({"kind":"same-size-actual-physical-alias","rejected_before_dispatch":true,"candidate_committed":false}),
        );
    }
    let mut layer = l::LayerBatch::new(graph, f.g, Buffers(b.0))?;
    let cancel = AtomicBool::new(false);
    layer.prepare(graph, arena, public, &f.before[c::POS], &cancel)?;
    let selected = read_selection(arena, &b, f.g, retired, &cancel);
    if kind == 0 && selected.is_err() {
        // Selector fence has completed and audit drained sources. A failed
        // preparation cannot be promoted or retried even with empty IDs.
        if layer
            .attention(graph, arena, public, Zeroizing::new(Vec::new()), &cancel)
            .is_ok()
        {
            return Err("QSA nonfinite selector promoted".into());
        }
        if layer.phase() != l::Phase::Poisoned {
            return Err("QSA failed selection not poisoned".into());
        }
        arena.reset()?;
        return Ok(
            serde_json::json!({"kind":"late-nonfinite-input","rejected":true,"candidate_committed":false}),
        );
    }
    layer.attention(graph, arena, public, selected?, &cancel)?;
    let data = snapshot(arena, &b, retired)?;
    if kind == 1 {
        cancel.store(true, Ordering::Release);
    }
    if kind == 2 {
        arena.reset()?;
    }
    if layer
        .accept(
            graph,
            arena,
            &data,
            &f.before,
            &cancel,
            retired.load(Ordering::Acquire),
        )
        .is_ok()
        || layer.phase() != l::Phase::Poisoned
    {
        return Err("QSA bad/cancel/stale acceptance".into());
    }
    if layer
        .prepare(
            graph,
            arena,
            public,
            &f.before[c::POS],
            &AtomicBool::new(false),
        )
        .is_ok()
    {
        return Err("QSA poisoned retry accepted".into());
    }
    if kind != 2 {
        arena.reset()?;
    }
    Ok(
        serde_json::json!({"kind":(["late-nonfinite-input","completed-cancel","stale-arena-epoch"][kind]),"rejected":true,"poisoned_retry_rejected":true,"candidate_committed":false}),
    )
}
fn retire_failed_attempt<T>(attempt: &std::thread::Result<Result<T>>, retired: &AtomicBool) {
    if !matches!(attempt, Ok(Ok(_))) {
        retired.store(true, Ordering::Release);
    }
}
fn finish<T>(attempt: std::thread::Result<Result<T>>, clean: bool, retired: bool) -> Result<T> {
    if !clean || retired {
        return Err("QSA layer cleanup failed/retired".into());
    }
    attempt.map_err(|_| "QSA layer probe panic")?
}
pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let a = l::Admission::new(l::CAPACITY)?;
    let host = host_preflight()?;
    let path = std::env::var_os("KUNO_TEST_GGUF")
        .ok_or("original GGUF required for fixed-public QSA layer probe")?;
    let file = GgufFile::open(path, GgufLimits::default()).map_err(|_| "QSA original GGUF open")?;
    let model = Qwen4Exp::bind(&file).map_err(|_| "QSA original graph bind")?;
    let graph = model.bound_graph();
    let public = PublicLayer::load(client, graph)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), l::MAIN_CAP, l::IO_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut cases = Vec::new();
        cases.try_reserve_exact(24)?;
        for previous in [0, 3, 4, 2047, 2049, 2051] {
            for count in [1, 3, 5, 8] {
                cases.push(correct_case(
                    &mut arena, graph, &public, previous, count, &retired,
                )?);
            }
        }
        let mut failures = Vec::new();
        failures.try_reserve_exact(4)?;
        for kind in 0..4 {
            failures.push(failure_case(&mut arena, graph, &public, kind, &retired)?);
        }
        Ok((cases, failures))
    }));
    retire_failed_attempt(&attempt, &retired);
    let close = arena.close();
    let (cases, failures) = finish(attempt, close.is_ok(), retired.load(Ordering::Acquire))?;
    let erased = arena
        .owned_zero_audit_receipt()
        .ok_or("QSA main owning audit missing")?;
    if erased.bytes == 0 || erased.buffers == 0 {
        return Err("QSA empty main erasure receipt".into());
    }
    file.verify_file_unchanged()
        .map_err(|_| "QSA original file changed")?;
    let report = serde_json::json!({"scope":"fixed-public-original-QSA-layer3-only","fixture":"qsa-layer-batch-probe","cases":cases,"failure_cases":failures,"max_batch":8,"capacity":l::CAPACITY,"main_cap_bytes":l::MAIN_CAP,"main_dual_full_pages":a.main,"runtime_io_cap_bytes":l::IO_CAP,"runtime_audited_peak_bound":a.io,"host_scratch_upper_bound":host,"conservative_public_full_pages":public.public_bytes,"main_erased_bytes":erased.bytes,"main_erased_buffers":erased.buffers,"context_retired":false,"state_or_cursor_commit":false,"whole_graph_or_prefill_accepted":false,"timing_or_default_selection":false,"production_private_worker_or_driver_register_spill_erasure_accepted":false});
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    println!(
        "public_qsa_layer_batch_acceptance cases=24 failures=4 main_bytes={} main_buffers={} context_retired=false",
        erased.bytes, erased.buffers
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_full_read_and_failure_cleanup_never_publish_success() {
        let p = BufferLayout::new(3).unwrap();
        let bytes = Zeroizing::new(vec![0u8; p.owned_bytes]);
        assert_eq!(decode_read_packet(&bytes, p).unwrap().len(), p.owned_words);
        for n in [0, 3, p.owned_bytes - 1] {
            assert!(decode_read_packet(&bytes[..n], p).is_err());
        }
        let mut surplus = bytes;
        surplus.push(0);
        assert!(decode_read_packet(&surplus, p).is_err());
        assert_eq!(finish(Ok(Ok(7usize)), true, false).unwrap(), 7);
        for (clean, retired) in [(false, false), (true, true)] {
            assert!(finish(Ok(Ok(7usize)), clean, retired).is_err());
        }
        assert!(finish::<usize>(Ok(Err("injected read failure".into())), true, false).is_err());
        assert!(finish::<usize>(Err(Box::new("injected panic")), true, false).is_err());
    }
    #[test]
    fn unexpected_dispatch_read_or_panic_retires_before_best_effort_close() {
        let retired = AtomicBool::new(false);
        retire_failed_attempt(&Ok(Ok(7usize)), &retired);
        assert!(!retired.load(Ordering::Acquire));
        for failure in [
            Ok(Err("injected read/dispatch error".into())),
            Err(Box::new("injected panic") as Box<dyn std::any::Any + Send>),
        ] {
            let retired = AtomicBool::new(false);
            retire_failed_attempt::<usize>(&failure, &retired);
            assert!(retired.load(Ordering::Acquire));
            assert!(finish(failure, true, retired.load(Ordering::Acquire)).is_err());
        }
    }
    #[test]
    fn malformed_or_overflowing_audit_retires_before_success() {
        for (u, v, r, i, b) in [
            (u64::MAX, 1, 0, 0, 1),
            (0, 0, 0, 0, 1),
            (32, 16, 64, 111, 3),
            (32, 16, 64, 112, 0),
        ] {
            let retired = AtomicBool::new(false);
            assert!(check_audit(u, v, r, i, b, &retired).is_err());
            assert!(retired.load(Ordering::Acquire));
        }
    }
    #[test]
    fn host_plan_guards_before_fixture_allocations_and_mixed_columns_differ() {
        assert!(host_preflight().unwrap() < HOST_CAP);
        assert_ne!(*mixed_input(0), *mixed_input(7));
        assert_eq!(mixed_input(0).len(), l::D);
        let a = l::Admission::new(l::CAPACITY).unwrap();
        eprintln!(
            "public_qsa_layer_plan main={} io={} host={}",
            a.main,
            a.io,
            host_preflight().unwrap()
        );
    }
}
