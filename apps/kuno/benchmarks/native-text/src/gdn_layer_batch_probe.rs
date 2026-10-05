//! Fixed-public original-layer0 GDN transaction probe; no model graph hook.
use crate::{
    gdn_gpu, gdn_layer_batch as l, gdn_pointwise as scalar,
    private_arena::{BufferId, BufferLayout, PrivateArena},
    residual_batch_gpu::copy_span,
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::{
    GgufFile, GgufLimits,
    model::Qwen4Exp,
    ops::{self, CausalConvState, GdnState},
};
use l::{Buffers, Geometry, N, PublicLayer, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
const GAP: f32 = -173.25;
const FLAG: u32 = 0x53aa9267;
const HOST_CAP: usize = 128 * 1024 * 1024;
const STAGES: [usize; 12] = [
    l::RAW,
    l::Z,
    l::ALPHA,
    l::BETA,
    l::ACT,
    l::Q,
    l::K,
    l::V,
    l::CONTROL,
    l::RECURRENT,
    l::GATED,
    l::OUT,
];
fn mixed_input(t: usize) -> Zeroizing<Vec<f32>> {
    let mut x = Zeroizing::new(Vec::new());
    x.reserve_exact(l::D);
    x.extend((0..l::D).map(|d| (((d * 7 + t * 11 + 3) % 37) as f32 - 18.) / 512.));
    x
}
fn initial_history() -> Zeroizing<Vec<f32>> {
    let mut v = Zeroizing::new(Vec::new());
    v.reserve_exact(3 * 10240);
    v.extend((0..3 * 10240).map(|i| ((i * 5 % 23) as f32 - 11.) / 512.));
    v
}
fn initial_state() -> Zeroizing<Vec<f32>> {
    let mut v = Zeroizing::new(Vec::new());
    v.reserve_exact(48 * 128 * 128);
    v.extend(
        (0..48 * 128 * 128).map(|i| ((i * 7 + i / 128 * 3) % 31) as f32 / 1024. - 15. / 1024.),
    );
    v
}
fn initial_images(g: Geometry) -> Result<[Zeroizing<Vec<u32>>; N]> {
    g.validate(if g.scalar { 1 } else { 8 })?;
    let mut images = std::array::from_fn(|_| Zeroizing::new(Vec::new()));
    for (i, v) in images.iter_mut().enumerate() {
        v.try_reserve_exact(g.words(i))?;
        v.resize(
            g.words(i),
            if i >= l::STATUS { FLAG } else { GAP.to_bits() },
        );
    }
    for t in 0..if g.scalar { 1 } else { 8 } {
        let x = mixed_input(t);
        let at = g.span[l::INPUT].at(t);
        for (j, &x) in x.iter().enumerate() {
            images[l::INPUT][at + j] = x.to_bits();
        }
    }
    let history = initial_history();
    for c in 0..10240 {
        for tap in 0..3 {
            images[l::OLD_HISTORY][g.old_history.at(c) + tap] = history[c * 3 + tap].to_bits();
        }
    }
    let state = initial_state();
    for h in 0..48 {
        for k in 0..128 {
            for d in 0..128 {
                images[l::OLD_STATE][g.old_state.at(h, k, d)] =
                    state[(h * 128 + k) * 128 + d].to_bits();
            }
        }
    }
    Ok(images)
}
struct Set {
    g: Geometry,
    b: Buffers,
    before: [Zeroizing<Vec<u32>>; N],
}
fn allocate(arena: &mut PrivateArena, g: Geometry) -> Result<Set> {
    let before = initial_images(g)?;
    let mut ids = Vec::new();
    ids.try_reserve_exact(N)?;
    for v in &before {
        ids.push(arena.upload_u32(v)?);
    }
    Ok(Set {
        g,
        b: Buffers(ids.try_into().map_err(|_| "GDN layer buffer count")?),
        before,
    })
}
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
            .map_err(|_| "GDN layer owning read failed")?,
    );
    decode_read_packet(&bytes.0, arena.layout(id)?)
}
fn decode_read_packet(bytes: &[u8], layout: BufferLayout) -> Result<Zeroizing<Vec<u32>>> {
    if bytes.len() != layout.owned_bytes {
        return Err("GDN layer malformed owning read".into());
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
            return Err("GDN layer IO cleanup failed".into());
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
        return Err("GDN layer incomplete IO audit".into());
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
        return Err("GDN layer exact upload extent".into());
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
        .map_err(|_| "GDN layer owning write failed")?;
    Ok(())
}
fn snapshot(
    arena: &PrivateArena,
    set: &Set,
    retired: &AtomicBool,
) -> Result<[Zeroizing<Vec<u32>>; N]> {
    // Last output projection status is an ordered COMPLETED whole-layer fence.
    let fence = read(arena, set.b.0[26])?;
    audit(arena, retired)?;
    let mut out = std::array::from_fn(|_| Zeroizing::new(Vec::new()));
    for (i, v) in out.iter_mut().enumerate() {
        *v = read(arena, set.b.0[i])?;
        audit(arena, retired)?;
    }
    if *out[26] != *fence {
        return Err("GDN layer fence changed".into());
    }
    Ok(out)
}
fn compare(a: f32, b: f32) -> Result<f32> {
    let error = (a - b).abs();
    if !a.is_finite() || !b.is_finite() || error > 1e-4 + 3e-4 * b.abs() {
        return Err("public full GDN layer numeric parity".into());
    }
    Ok(error)
}
fn public_compare(
    a: f32,
    b: f32,
    token: usize,
    stage: usize,
    index: usize,
    reference: &str,
) -> Result<f32> {
    // This helper is confined to the early FIXED-PUBLIC synthetic-input probe;
    // it never formats production request state or arbitrary caller inputs.
    compare(a,b).map_err(|_|format!("fixed-public GDN layer parity reference={reference} token={token} stage={stage} index={index} actual={a} expected={b} abs={} tolerance={}",(a-b).abs(),1e-4+3e-4*b.abs()).into())
}
fn field(words: &[u32], g: Geometry, i: usize, t: usize) -> Zeroizing<Vec<f32>> {
    let s = g.span[i];
    let mut v = Zeroizing::new(Vec::new());
    v.reserve_exact(s.width);
    v.extend(
        words[s.at(t)..][..s.width]
            .iter()
            .map(|&x| f32::from_bits(x)),
    );
    v
}
fn compact_history(words: &[u32], g: Geometry, next: bool) -> Zeroizing<Vec<f32>> {
    let s = if next { g.next_history } else { g.old_history };
    let mut out = Zeroizing::new(Vec::new());
    out.reserve_exact(3 * s.channels);
    for c in 0..s.channels {
        for tap in 0..3 {
            out.push(f32::from_bits(words[s.at(c) + tap]));
        }
    }
    out
}
fn compact_state(words: &[u32], g: Geometry, next: bool) -> Zeroizing<Vec<f32>> {
    let s = if next { g.next_state } else { g.old_state };
    let shape = g.pointwise.shape;
    let mut out = Zeroizing::new(Vec::new());
    out.reserve_exact(shape.value_heads * shape.head_dim * shape.head_dim);
    for h in 0..shape.value_heads {
        for k in 0..shape.head_dim {
            for d in 0..shape.head_dim {
                out.push(f32::from_bits(words[s.at(h, k, d)]));
            }
        }
    }
    out
}
struct TokenOracle {
    stage: [Zeroizing<Vec<f32>>; 12],
    history: Zeroizing<Vec<f32>>,
    state: Zeroizing<Vec<f32>>,
}
fn cpu_oracle(public: &PublicLayer) -> Result<Vec<TokenOracle>> {
    let mut conv = CausalConvState::new(10240, 4, 1)?;
    let history = initial_history();
    for tap in 0..3 {
        let mut warm = Zeroizing::new(Vec::new());
        warm.reserve_exact(10240);
        warm.extend((0..10240).map(|c| history[c * 3 + tap]));
        drop(Zeroizing::new(conv.step(&warm, &public.conv_cpu)?));
    }
    if conv.history() != *history {
        return Err("GDN original history order".into());
    }
    let mut recurrence = GdnState::new_bounded(ops::GdnShape::qwen38(), 8 * 1024 * 1024)?;
    recurrence.set_values(&initial_state())?;
    let mut out = Vec::new();
    out.try_reserve_exact(8)?;
    for t in 0..8 {
        let x = mixed_input(t);
        let raw = public.matrices[0].cpu(&x)?;
        let z = public.matrices[1].cpu(&x)?;
        let alpha = public.matrices[2].cpu(&x)?;
        let beta = public.matrices[3].cpu(&x)?;
        let mut act = Zeroizing::new(conv.step(&raw, &public.conv_cpu)?);
        for v in act.iter_mut() {
            *v = ops::silu(*v);
        }
        let mut q = Zeroizing::new(ops::l2_norm_heads(&act[..2048], 128, 1e-6)?);
        for v in q.iter_mut() {
            *v /= (128f32).sqrt();
        }
        let k = Zeroizing::new(ops::l2_norm_heads(&act[2048..4096], 128, 1e-6)?);
        let mut v = Zeroizing::new(Vec::new());
        v.try_reserve_exact(6144)?;
        v.extend_from_slice(&act[4096..]);
        let mut controls = Zeroizing::new(ops::gdn_log_decay(
            &alpha,
            &public.params_cpu[..48],
            &public.params_cpu[48..],
        )?);
        controls.extend(beta.iter().map(|&b| ops::sigmoid(b)));
        let recurrent =
            Zeroizing::new(recurrence.step(&q, &k, &v, &controls[..48], &controls[48..])?);
        let gated = Zeroizing::new(ops::gdn_output_gate(
            &recurrent,
            &z,
            &public.gamma_cpu,
            128,
            1e-6,
        )?);
        let result = public.matrices[4].cpu(&gated)?;
        let mut history = Zeroizing::new(Vec::new());
        history.try_reserve_exact(conv.history().len())?;
        history.extend_from_slice(conv.history());
        let mut state = Zeroizing::new(Vec::new());
        state.try_reserve_exact(recurrence.values().len())?;
        state.extend_from_slice(recurrence.values());
        out.push(TokenOracle {
            stage: [
                raw, z, alpha, beta, act, q, k, v, controls, recurrent, gated, result,
            ],
            history,
            state,
        });
    }
    conv.reset();
    recurrence.reset();
    Ok(out)
}
fn cpu_check(
    data: &[Zeroizing<Vec<u32>>; N],
    g: Geometry,
    count: usize,
    oracle: &[TokenOracle],
    offset: usize,
) -> Result<f32> {
    let mut max = 0f32;
    for t in 0..count {
        for (j, i) in STAGES.into_iter().enumerate() {
            let actual = field(&data[i], g, i, t);
            let expected = &oracle[offset + t].stage[j];
            if actual.len() != expected.len() {
                return Err("GDN oracle stage extent".into());
            }
            for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
                max = max.max(public_compare(a, e, offset + t, i, index, "core")?);
            }
        }
    }
    for (stage, actual, expected) in [
        (
            l::NEXT_HISTORY,
            compact_history(&data[l::NEXT_HISTORY], g, true),
            &oracle[offset + count - 1].history,
        ),
        (
            l::NEXT_STATE,
            compact_state(&data[l::NEXT_STATE], g, true),
            &oracle[offset + count - 1].state,
        ),
    ] {
        for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            max = max.max(public_compare(
                a,
                e,
                offset + count - 1,
                stage,
                index,
                "core",
            )?);
        }
    }
    Ok(max)
}
fn scalar_step(
    arena: &PrivateArena,
    public: &PublicLayer,
    set: &Set,
    flat: BufferId,
) -> Result<()> {
    let g = set.g;
    let b = &set.b;
    let client = arena.client()?;
    let dim = CubeDim::new_1d(64);
    for which in 0..4 {
        l::project(arena, public, g, b, which, 1)?;
    }
    let arg = |i| l::arg(arena, b, i);
    let pubarg = |h: &cubecl::server::Handle, n| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };
    scalar::conv_silu::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(160, 1, 1),
        dim,
        arg(l::RAW)?,
        pubarg(&public.conv, 40960),
        arg(l::OLD_HISTORY)?,
        arg(l::NEXT_HISTORY)?,
        arg(l::ACT)?,
        arg(17)?,
        10240,
    );
    scalar::prepare_qkv::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(80, 1, 1),
        dim,
        arg(l::ACT)?,
        unsafe {
            ArrayArg::from_raw_parts(arena.handle(flat)?.clone(), arena.layout(flat)?.owned_words)
        },
        arg(18)?,
        1e-6,
        16,
        48,
    );
    for (i, first, width) in [(l::Q, 0, 2048), (l::K, 2048, 2048), (l::V, 4096, 6144)] {
        copy_span::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(width.div_ceil(64) as u32, 8, 1),
            dim,
            unsafe {
                ArrayArg::from_raw_parts(
                    arena.handle(flat)?.clone(),
                    arena.layout(flat)?.owned_words,
                )
            },
            arg(i)?,
            1,
            first,
            10240,
            0,
            width,
            width,
        );
    }
    scalar::prepare_controls::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(1, 1, 1),
        dim,
        arg(l::ALPHA)?,
        arg(l::BETA)?,
        pubarg(&public.params, 96),
        arg(l::CONTROL)?,
        arg(19)?,
        48,
    );
    gdn_gpu::gdn_step::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(
            gdn_gpu::GdnLayout::new(g.pointwise.shape)?.workgroups_x,
            1,
            1,
        ),
        dim,
        arg(l::Q)?,
        arg(l::K)?,
        arg(l::V)?,
        arg(l::CONTROL)?,
        arg(l::OLD_STATE)?,
        arg(l::NEXT_STATE)?,
        arg(l::RECURRENT)?,
        arg(20)?,
        128,
        16,
        48,
    );
    scalar::output_gate::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(48, 1, 1),
        dim,
        arg(l::RECURRENT)?,
        arg(l::Z)?,
        pubarg(&public.gamma, 128),
        arg(l::GATED)?,
        arg(21)?,
        1e-6,
        48,
        true,
    );
    l::project(arena, public, g, b, 4, 1)
}
fn all_bindings(
    arena: &PrivateArena,
    batch: &Set,
    single: &Set,
    flat: BufferId,
    public: &PublicLayer,
) -> Result<()> {
    l::binding_proof(arena, batch.g, &batch.b, public)?;
    l::binding_proof(arena, single.g, &single.b, public)?;
    let ids: Vec<_> = batch
        .b
        .0
        .into_iter()
        .chain(single.b.0)
        .chain([flat])
        .collect();
    let client = arena.client()?;
    let resources = ids
        .iter()
        .map(|id| -> Result<_> { Ok(client.get_resource(arena.handle(*id)?.clone())?) })
        .collect::<Result<Vec<_>>>()?;
    for (i, r) in resources.iter().enumerate() {
        let v = r.resource();
        if v.offset != 0
            || v.size != v.buffer.size()
            || v.size != arena.layout(ids[i])?.owned_bytes as u64
            || resources[..i]
                .iter()
                .any(|x| x.resource().buffer == v.buffer)
        {
            return Err("GDN batch/scalar/flat owner alias".into());
        }
    }
    Ok(())
}
fn scalar_advance_images(
    single: &mut Set,
    actual: &[Zeroizing<Vec<u32>>; N],
    t: usize,
) -> Result<()> {
    for (old, next) in [
        (l::OLD_HISTORY, l::NEXT_HISTORY),
        (l::OLD_STATE, l::NEXT_STATE),
    ] {
        if single.before[old].len() != single.g.words(next) {
            return Err("GDN scalar state image extent".into());
        }
        single.before[old].copy_from_slice(&actual[next][..single.g.words(next)]);
    }
    if t < 8 {
        for (x, v) in single.before[l::INPUT]
            .iter_mut()
            .zip(mixed_input(t).iter())
        {
            *x = v.to_bits();
        }
    }
    Ok(())
}
fn correct_case(
    arena: &mut PrivateArena,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    public: &PublicLayer,
    oracle: &[TokenOracle],
    count: usize,
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    let admission = l::Admission::new()?;
    l::validate_context(3, count)?;
    let batch = allocate(arena, Geometry::new(false)?)?;
    let mut single = allocate(arena, Geometry::new(true)?)?;
    let flat = arena.upload_u32(&Zeroizing::new(vec![GAP.to_bits(); 10240]))?;
    if arena.charged_bytes() != admission.main {
        return Err("GDN layer full page charge differs".into());
    }
    all_bindings(arena, &batch, &single, flat, public)?;
    let cancel = AtomicBool::new(false);
    let mut owner = l::LayerBatch::new(graph, batch.g, Buffers(batch.b.0), 3, count)?;
    owner.enqueue(graph, arena, public, &cancel)?;
    let actual = snapshot(arena, &batch, retired)?;
    l::validate_snapshot(batch.g, count, &actual, &batch.before, &cancel)?;
    let max_core = cpu_check(&actual, batch.g, count, oracle, 0)?;
    let mut max_scalar = 0f32;
    for t in 0..count {
        for i in [l::INPUT, l::OLD_HISTORY, l::OLD_STATE] {
            write(arena, single.b.0[i], &single.before[i])?;
        }
        scalar_step(arena, public, &single, flat)?;
        let scalar_data = snapshot(arena, &single, retired)?;
        l::validate_snapshot(single.g, 1, &scalar_data, &single.before, &cancel)?;
        cpu_check(&scalar_data, single.g, 1, oracle, t)?;
        for i in STAGES {
            let a = field(&actual[i], batch.g, i, t);
            let b = field(&scalar_data[i], single.g, i, 0);
            for (index, (&a, &b)) in a.iter().zip(b.iter()).enumerate() {
                max_scalar =
                    max_scalar.max(public_compare(a, b, t, i, index, "original-scalar-GPU")?);
            }
        }
        if t + 1 == count {
            for (stage, a, b) in [
                (
                    l::NEXT_HISTORY,
                    compact_history(&actual[l::NEXT_HISTORY], batch.g, true),
                    compact_history(&scalar_data[l::NEXT_HISTORY], single.g, true),
                ),
                (
                    l::NEXT_STATE,
                    compact_state(&actual[l::NEXT_STATE], batch.g, true),
                    compact_state(&scalar_data[l::NEXT_STATE], single.g, true),
                ),
            ] {
                for (index, (&a, &b)) in a.iter().zip(b.iter()).enumerate() {
                    max_scalar = max_scalar.max(public_compare(
                        a,
                        b,
                        t,
                        stage,
                        index,
                        "original-scalar-GPU",
                    )?);
                }
            }
        }
        scalar_advance_images(&mut single, &scalar_data, t + 1)?;
    }
    owner.accept(
        graph,
        arena,
        &actual,
        &batch.before,
        &cancel,
        retired.load(Ordering::Acquire),
    )?;
    arena.reset()?;
    if batch
        .b
        .0
        .iter()
        .chain(single.b.0.iter())
        .chain([&flat])
        .any(|&id| arena.handle(id).is_ok())
        || retired.load(Ordering::Acquire)
    {
        return Err("GDN layer stale epoch/retirement after reset".into());
    }
    Ok(
        serde_json::json!({"count":count,"position":3,"layer":0,"main_full_pages":admission.main,"io_peak_bound":admission.io,"max_core_abs_error":max_core,"max_scalar_abs_error":max_scalar,"all_active_stages_checked":true,"all_ten_statuses_checked":true,"initial_state_history_nonzero_immutable":true,"final_candidate_state_history_compared":true,"candidate_only_no_commit":true}),
    )
}
fn failure_case(
    arena: &mut PrivateArena,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    public: &PublicLayer,
    kind: usize,
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    l::Admission::new()?;
    l::validate_context(3, 8)?;
    let mut set = allocate(arena, Geometry::new(false)?)?;
    let cancel = AtomicBool::new(false);
    if kind == 0 {
        set.before[l::INPUT][set.g.span[l::INPUT].at(7)] = f32::NAN.to_bits();
        write(arena, set.b.0[l::INPUT], &set.before[l::INPUT])?;
    }
    if kind == 3 {
        // Equal logical extents make the physical duplicate proof, rather
        // than only a role-length mismatch, reject this real owner alias.
        let mut g = set.g;
        g.span[l::OUT] = g.span[l::INPUT];
        let mut b = set.b.0;
        b[l::OUT] = b[l::INPUT];
        let mut owner = l::LayerBatch::new(graph, g, Buffers(b), 3, 8)?;
        if owner.enqueue(graph, arena, public, &cancel).is_ok()
            || owner.phase() != l::Phase::Poisoned
        {
            return Err("GDN writable/source alias not terminal".into());
        }
        for i in [l::INPUT, l::OLD_HISTORY, l::OLD_STATE] {
            let source = read(arena, set.b.0[i])?;
            audit(arena, retired)?;
            for (j, &bits) in source.iter().enumerate() {
                if bits != set.before[i].get(j).copied().unwrap_or(0) {
                    return Err("GDN alias rejection changed old owner".into());
                }
            }
        }
        arena.reset()?;
        return Ok(
            serde_json::json!({"kind":"physical_source_output_alias","whole_chunk_rejected":true,"old_state_history_input_immutable":true,"retry_poisoned":true,"candidate_pages_erased":true}),
        );
    }
    let mut owner = l::LayerBatch::new(graph, set.g, Buffers(set.b.0), 3, 8)?;
    owner.enqueue(graph, arena, public, &cancel)?;
    let actual = snapshot(arena, &set, retired)?;
    if kind == 1 {
        cancel.store(true, Ordering::Release);
    }
    if kind == 2 {
        arena.reset()?;
    }
    if owner
        .accept(
            graph,
            arena,
            &actual,
            &set.before,
            &cancel,
            retired.load(Ordering::Acquire),
        )
        .is_ok()
        || owner.phase() != l::Phase::Poisoned
    {
        return Err("GDN failed candidate accepted/retryable".into());
    }
    for i in [l::INPUT, l::OLD_HISTORY, l::OLD_STATE] {
        for (j, &bits) in actual[i].iter().enumerate() {
            if bits != set.before[i].get(j).copied().unwrap_or(0) {
                return Err("GDN failed candidate changed old source".into());
            }
        }
    }
    if kind == 0
        && !l::STATUS_FIELDS.into_iter().any(|i| {
            (0..8).any(|t| actual[i][set.g.span[i].at(t)..][..set.g.span[i].width].contains(&0))
        })
    {
        return Err("GDN late failure has no failed stage status".into());
    }
    if kind != 2 {
        arena.reset()?;
    }
    if retired.load(Ordering::Acquire) {
        return Err("GDN failed candidate cleanup retired context".into());
    }
    Ok(
        serde_json::json!({"kind":(["late_input_nan","cancel_after_completed_fence","stale_epoch_after_completed_fence"][kind]),"whole_chunk_rejected":true,"old_state_history_input_immutable":true,"retry_poisoned":true,"candidate_pages_erased":true}),
    )
}
fn finish<T>(attempt: std::thread::Result<Result<T>>, clean: bool, retired: bool) -> Result<T> {
    if !clean || retired {
        return Err("GDN full layer cleanup failed/retired".into());
    }
    attempt.map_err(|_| "GDN full layer probe panicked")?
}
fn host_preflight() -> Result<usize> {
    let b = Geometry::new(false)?;
    let s = Geometry::new(true)?;
    let mut before = 0usize;
    let mut largest = 0usize;
    for g in [b, s] {
        for i in 0..N {
            before = before
                .checked_add(g.words(i).checked_mul(4).ok_or("GDN host overflow")?)
                .ok_or("GDN host overflow")?;
            largest = largest.max(BufferLayout::new(g.words(i))?.owned_bytes);
        }
    }
    let state = gdn_gpu::GdnLayout::new(b.pointwise.shape)?.state_elements * 4;
    let history = b.pointwise.channels * 3 * 4;
    let stages = STAGES.into_iter().try_fold(0usize, |n, i| {
        n.checked_add(b.span[i].width * 4)
            .ok_or("GDN host stage overflow")
    })?;
    let oracle = (stages + state + history)
        .checked_mul(l::MAX_BATCH)
        .ok_or("GDN oracle overflow")?;
    // All retained before-images, both full-page snapshots and eight oracle
    // checkpoints, plus: one owning Bytes read+fence, simultaneous two compact
    // state/history comparisons, core-oracle working state/history/stages,
    // scalar prepared-QKV owner image, and fixed Vec/container metadata reserve.
    let working = 2 * largest
        + 2 * (state + history)
        + state
        + history
        + 2 * stages
        + BufferLayout::new(b.pointwise.channels)?.owned_bytes
        + 64 * 1024;
    let full = before
        .checked_add(b.footprint()?)
        .and_then(|n| n.checked_add(s.footprint().ok()?))
        .and_then(|n| n.checked_add(oracle))
        .and_then(|n| n.checked_add(working))
        .ok_or("GDN aggregate host overflow")?;
    if full > HOST_CAP {
        return Err("GDN private host scratch admission".into());
    }
    Ok(full)
}
pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    l::Admission::new()?;
    // Host-state/result snapshots and all eight independent core checkpoints;
    // public decoded original weights have their separate256MiB preflight.
    let host_bound = host_preflight()?;
    let path = std::env::var_os("KUNO_TEST_GGUF")
        .ok_or("existing original GGUF required for public GDN layer probe")?;
    let file = GgufFile::open(path, GgufLimits::default()).map_err(|_| "GDN public GGUF open")?;
    let model = Qwen4Exp::bind(&file).map_err(|_| "GDN original graph bind")?;
    let graph = model.bound_graph();
    let public = PublicLayer::load(client, graph)?;
    let oracle = cpu_oracle(&public)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), l::MAIN_CAP, l::IO_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut cases = Vec::new();
        for count in [1, 3, 5, 8] {
            cases.push(correct_case(
                &mut arena, graph, &public, &oracle, count, &retired,
            )?);
        }
        let mut failures = Vec::new();
        for kind in 0..4 {
            failures.push(failure_case(&mut arena, graph, &public, kind, &retired)?);
        }
        Ok((cases, failures))
    }));
    let close = arena.close();
    let (cases, failures) = finish(attempt, close.is_ok(), retired.load(Ordering::Acquire))?;
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("GDN full layer owning zero audit missing")?;
    if zero.bytes == 0 || zero.buffers == 0 {
        return Err("GDN full layer empty zero audit".into());
    }
    file.verify_file_unchanged()
        .map_err(|_| "GDN original file changed")?;
    let report = serde_json::json!({"scope":"fixed-public-original-GDN-layer0-only","fixture":"gdn-layer-batch-probe","cases":cases,"failure_cases":failures,"max_batch":8,"original_packed_weight_pages":public.public_bytes,"all_five_original_matrices":true,"main_cap_bytes":l::MAIN_CAP,"runtime_io_cap_bytes":l::IO_CAP,"host_scratch_upper_bound":host_bound,"main_erased_bytes":zero.bytes,"main_erased_buffers":zero.buffers,"context_retired":false,"state_or_cursor_commit":false,"whole_graph_or_prefill_accepted":false,"performance_or_production_private_worker_accepted":false,"private_register_spill_driver_erasure_accepted":false});
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    println!(
        "public_gdn_layer_batch_acceptance cases=4 failures=4 main_bytes={} main_buffers={} context_retired=false",
        zero.bytes, zero.buffers
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_or_surplus_owning_reads_reject_before_result_allocation() {
        let layout = BufferLayout::new(3).unwrap();
        let mut bytes = Zeroizing::new(vec![0u8; layout.owned_bytes]);
        for (i, b) in bytes.chunks_exact_mut(4).enumerate() {
            b.copy_from_slice(&(i as u32).to_le_bytes());
        }
        let words = decode_read_packet(&bytes, layout).unwrap();
        assert_eq!(words.len(), layout.owned_words);
        assert_eq!(
            words[layout.owned_words - 1],
            (layout.owned_words - 1) as u32
        );
        for len in [0, 3, layout.owned_bytes - 1] {
            assert!(decode_read_packet(&bytes[..len], layout).is_err());
        }
        bytes.push(0);
        assert!(decode_read_packet(&bytes, layout).is_err());
    }
    #[test]
    fn cleanup_error_retirement_read_error_and_panic_never_publish_success() {
        assert_eq!(finish(Ok(Ok(7usize)), true, false).unwrap(), 7);
        for (clean, retired) in [(false, false), (false, true), (true, true)] {
            assert!(finish(Ok(Ok(7usize)), clean, retired).is_err());
        }
        assert!(
            finish::<usize>(
                Ok(Err("fixed public read/validation failure".into())),
                true,
                false
            )
            .is_err()
        );
        assert!(finish::<usize>(Err(Box::new("fixed public panic")), true, false).is_err());
    }
    #[test]
    fn malformed_audit_counts_and_overflow_retire_before_any_success() {
        let retired = AtomicBool::new(false);
        check_audit(32, 16, 64, 112, 3, &retired).unwrap();
        assert!(!retired.load(Ordering::Acquire));
        for (u, v, r, i, b) in [
            (u64::MAX, 1, 0, 0, 1),
            (u64::MAX, 0, 1, 0, 1),
            (0, 0, 0, 0, 1),
            (32, 16, 64, 111, 3),
            (32, 16, 64, 112, 0),
        ] {
            let retired = AtomicBool::new(false);
            assert!(check_audit(u, v, r, i, b, &retired).is_err());
            assert!(retired.load(Ordering::Acquire));
            assert!(finish(Ok(Ok(7usize)), true, retired.load(Ordering::Acquire)).is_err());
        }
    }
    #[test]
    fn full_page_host_and_info_preflight_includes_all_owners_and_audit_duplicates() {
        let a = l::Admission::new().unwrap();
        let host = host_preflight().unwrap();
        eprintln!(
            "fixed_public_gdn_layer_budget main={} io={} host={}",
            a.main, a.io, host
        );
        assert!(host <= HOST_CAP);
        // Current signatures: max8 raw Array records (recurrence), max21
        // runtime scalars (chunk bases/strides/count);21 scalars
        // and widerU64 addressing. <=64 dispatches between completed reads,
        // metadata source+target and full audit duplicate headroom included.
        for address in [cubecl::ir::AddressType::U32, cubecl::ir::AddressType::U64] {
            let mut info = cubecl::InfoBuilder::default();
            for _ in 0..8 {
                info.metadata.register_array(4_194_304, 4_194_304, address);
            }
            for _ in 0..21 {
                info.scalars.push(1_u64);
            }
            assert!(info.finish(address).data.len() * 8 <= 512);
        }
        let pages = (0..N)
            .map(|i| {
                BufferLayout::new(Geometry::new(false).unwrap().words(i))
                    .unwrap()
                    .owned_bytes
            })
            .sum::<usize>();
        assert!(pages < a.main);
        assert!(a.io < l::IO_CAP as usize);
    }
    #[test]
    fn fixed_fixture_nonzero_history_and_state_are_repeatable_and_guarded() {
        let history = initial_history();
        let state = initial_state();
        assert!(history.iter().any(|&v| v != 0.) && state.iter().any(|&v| v != 0.));
        assert_eq!(history.len(), 3 * 10240);
        assert_eq!(state.len(), 48 * 128 * 128);
        let g = Geometry::new(false).unwrap();
        let images = initial_images(g).unwrap();
        assert_eq!(
            *compact_history(&images[l::OLD_HISTORY], g, false),
            *history
        );
        assert_eq!(*compact_state(&images[l::OLD_STATE], g, false), *state);
        for t in [0, 3, 7] {
            assert_eq!(*field(&images[l::INPUT], g, l::INPUT, t), *mixed_input(t));
        }
    }
}
