//! Fixed PUBLIC original layer1 PLE candidate acceptance, MAX8 only.
//! Original scalar GPU projections + unchanged CPU projected tail are the
//! graph oracle. The additional count1 GPU tail is NEW topology, separately
//! labeled. No committed history, whole graph, performance or private API.
use crate::{
    ple_chunk::{self as c, *},
    private_arena::{BufferId, BufferLayout, OwnedZeroAudit, PrivateArena},
    quant_matvec::cooperative,
    residual_batch_gpu::finite_span,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use cubecl_runtime::transient::TransientIoReceipt;
use kuno_text::{
    GgufFile, GgufLimits,
    model::Qwen4Exp,
    ops::{self, CausalConvState, GatedResidualShape},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
const GAP: u32 = (-173.25f32).to_bits();
const STATUS_GAP: u32 = 0x53aa9267;
const HOST_CAP: usize = 64 * 1024 * 1024;
const HOST_PRIVATE_BOUND: usize = 48 * 1024 * 1024;
type Image = [Zeroizing<Vec<u32>>; N];
struct Bytes(cubecl::bytes::Bytes);
impl Drop for Bytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn words(n: usize, fill: u32) -> Result<Zeroizing<Vec<u32>>> {
    let mut v = Zeroizing::new(Vec::new());
    v.try_reserve_exact(n)
        .map_err(|_| "PLE guarded host allocation")?;
    if v.capacity() != n {
        return Err("PLE exact host snapshot capacity".into());
    }
    v.resize(n, fill);
    Ok(v)
}
fn empty_image() -> Image {
    std::array::from_fn(|_| Zeroizing::new(Vec::new()))
}
fn image(g: Geometry, count: usize) -> Result<Image> {
    g.validate(count)?;
    let mut v = empty_image();
    for (i, x) in v.iter_mut().enumerate() {
        *x = words(
            BufferLayout::new(g.words(i))?.owned_words,
            if i >= STATUS { STATUS_GAP } else { GAP },
        )?;
        for (j, val) in x.iter_mut().enumerate() {
            if g.active(i, count, j) {
                *val = if i >= STATUS { 0 } else { 0f32.to_bits() };
            }
        }
    }
    for t in 0..count {
        for d in 0..g.channels() {
            v[RESIDUAL][g.spans[RESIDUAL].at(t) + d] =
                (((d * 7 + t * 13 + 11) % 97) as f32 / 53. - 48. / 53.).to_bits();
        }
    }
    for d in 0..g.channels() {
        for j in 0..HISTORY {
            v[OLD][g.old.at(d) + j] =
                (((d * 5 + j * 17 + 3) % 43) as f32 / 997. - 21. / 997.).to_bits();
        }
    }
    Ok(v)
}
fn allocate(arena: &mut PrivateArena, g: Geometry) -> Result<Buffers> {
    let mut ids = Vec::new();
    ids.try_reserve_exact(N)?;
    for i in 0..N {
        ids.push(arena.allocate_zeroed(g.words(i))?);
    }
    Ok(Buffers(
        ids.try_into().map_err(|_| "PLE fixed owner count")?,
    ))
}
fn write_image(arena: &PrivateArena, b: &Buffers, v: &Image) -> Result<()> {
    for (i, x) in v.iter().enumerate() {
        let layout = arena.layout(b.0[i])?;
        if x.len() != layout.owned_words {
            return Err("PLE full owned upload extent".into());
        }
        let mut bytes = Zeroizing::new(Vec::new());
        bytes.try_reserve_exact(layout.owned_bytes)?;
        if bytes.capacity() != layout.owned_bytes {
            return Err("PLE exact host upload capacity".into());
        }
        bytes.resize(layout.owned_bytes, 0);
        for (out, &bits) in bytes.chunks_exact_mut(4).zip(x.iter()) {
            out.copy_from_slice(&bits.to_le_bytes());
        }
        arena
            .client()?
            .write_private(arena.handle(b.0[i])?.clone(), std::mem::take(&mut *bytes))
            .map_err(|_| "PLE owned source write")?;
    }
    Ok(())
}
struct ScrubSlice<'a>(&'a mut [u8]);
impl Drop for ScrubSlice<'_> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn decode_read(bytes: &mut [u8], expected: usize) -> Result<Zeroizing<Vec<u32>>> {
    let guard = ScrubSlice(bytes);
    if expected == 0 || !expected.is_multiple_of(4) || guard.0.len() != expected {
        return Err("PLE exact owning read extent".into());
    }
    let mut out = words(expected / 4, 0)?;
    for (x, b) in out.iter_mut().zip(guard.0.chunks_exact(4)) {
        *x = u32::from_le_bytes(b.try_into().map_err(|_| "PLE owning word")?);
    }
    Ok(out)
}
fn read(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
    let mut bytes = Bytes(
        arena
            .client()?
            .read_one(arena.handle(id)?.clone())
            .map_err(|_| "PLE owning read")?,
    );
    let expected = arena.layout(id)?.owned_bytes;
    // Host slice scrubs on every decode failure/unwind; Bytes then drops its
    // GPU lease BEFORE any caller can start an audited sanitizer.
    decode_read(&mut bytes.0, expected)
}
fn audit_receipt(r: &TransientIoReceipt, retired: &AtomicBool) -> Result<()> {
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes));
    if full.is_none_or(|n| n == 0 || n != r.inspected_bytes)
        || r.inspected_buffers == 0
        || retired.load(Ordering::Acquire)
    {
        retired.store(true, Ordering::Release);
        return Err("PLE full runtime zero inspection receipt".into());
    }
    Ok(())
}
fn audit(arena: &PrivateArena, retired: &AtomicBool) -> Result<()> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<TransientIoReceipt> {
            cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())
                .map_err(|_| "PLE runtime audit".into())
        },
    ));
    match r {
        Ok(Ok(receipt)) => audit_receipt(&receipt, retired),
        _ => {
            retired.store(true, Ordering::Release);
            Err("PLE bounded runtime cleanup failure".into())
        }
    }
}
fn snapshot(arena: &PrivateArena, b: &Buffers, retired: &AtomicBool) -> Result<Image> {
    // The first ordinary map completes ALL queued candidate work before any
    // bounded cleanup. No inference deadline is introduced by this fixture.
    drop(read(arena, b.0[21])?);
    audit(arena, retired)?;
    let mut out = empty_image();
    for (i, x) in out.iter_mut().enumerate() {
        *x = read(arena, b.0[i])?;
        audit(arena, retired)?;
    }
    Ok(out)
}
fn prove_all(
    arena: &PrivateArena,
    sets: &[(Geometry, &Buffers)],
    marker: BufferId,
    p: &PublicLayer,
) -> Result<()> {
    let mut handles: Vec<Handle> = Vec::new();
    handles.try_reserve_exact(N * 2 + 7)?;
    for &(g, b) in sets {
        g.validate(8)?;
        for (i, &id) in b.0.iter().enumerate() {
            if arena.layout(id)?.logical_words != g.words(i) {
                return Err("PLE full owner epoch/layout".into());
            }
            handles.push(arena.handle(id)?.clone());
        }
    }
    handles.push(arena.handle(marker)?.clone());
    handles.extend(p.matrices.iter().map(|m| m.handle.clone()));
    handles.extend(p.gamma.iter().cloned());
    handles.push(p.conv.clone());
    let resources = handles
        .iter()
        .map(|h| {
            arena
                .client()?
                .get_resource(h.clone())
                .map_err(|_| "PLE actual physical resource".into())
        })
        .collect::<Result<Vec<_>>>()?;
    let mut full = 0u64;
    for (i, r) in resources.iter().enumerate() {
        let v = r.resource();
        let h = &handles[i];
        if h.stream != handles[0].stream
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || v.offset != 0
            || v.size != h.size()
            || v.buffer.size() != v.size
            || resources[..i]
                .iter()
                .any(|x| x.resource().buffer == v.buffer)
        {
            return Err("PLE same-client stream full disjoint pages".into());
        }
        full = full
            .checked_add(v.size)
            .ok_or("PLE combined physical overflow")?;
    }
    let expected = arena
        .charged_bytes()
        .checked_add(p.physical_bytes)
        .ok_or("PLE combined page sum")?;
    if full != expected as u64
        || arena.charged_bytes() > MAIN_CAP
        || p.physical_bytes > PUBLIC_CAP
        || expected > MAIN_CAP + PUBLIC_CAP
    {
        return Err("PLE combined MAIN/PUBLIC physical budget".into());
    }
    Ok(())
}
fn set_embedding(before: &mut Image, source: &[u8]) -> Result<()> {
    if source.len() != before[EMB].len() * 4 {
        return Err("PLE original row before-image span".into());
    }
    for (v, b) in before[EMB].iter_mut().zip(source.chunks_exact(4)) {
        *v = u32::from_le_bytes(b.try_into().map_err(|_| "PLE row word")?);
    }
    Ok(())
}
fn scalar_arg(
    arena: &PrivateArena,
    g: Geometry,
    b: &Buffers,
    i: usize,
) -> Result<ArrayArg<WgpuRuntime>> {
    let s = g.spans[i];
    let full = arena.layout(b.0[i])?.owned_words;
    let suffix = full
        .checked_sub(s.base)
        .and_then(|n| n.checked_sub(s.width))
        .ok_or("PLE scalar view extent")?;
    Ok(unsafe {
        ArrayArg::from_raw_parts(
            arena
                .handle(b.0[i])?
                .clone()
                .offset_start((s.base * 4) as u64)
                .offset_end((suffix * 4) as u64),
            s.width,
        )
    })
}
fn scalar_project(
    arena: &PrivateArena,
    g: Geometry,
    b: &Buffers,
    p: &PublicLayer,
    which: usize,
) -> Result<()> {
    let m = &p.matrices[which];
    let (out, status) = if which == 0 { (KEY, 13) } else { (VALUE, 14) };
    cooperative::cooperative_matvec::launch::<WgpuRuntime>(
        arena.client()?,
        CubeCount::Static(m.geometry.rows() as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(m.handle.clone(), m.words) },
        scalar_arg(arena, g, b, EMB)?,
        scalar_arg(arena, g, b, out)?,
        m.geometry.rows(),
        m.geometry.columns(),
        m.geometry.dtype(),
        0,
    );
    let y = g.spans[out];
    let f = g.spans[status];
    finite_span::launch::<WgpuRuntime>(
        arena.client()?,
        CubeCount::Static(m.geometry.rows().div_ceil(64) as u32, 8, 1),
        CubeDim::new_1d(64),
        arg(arena, b, out)?,
        arg(arena, b, status)?,
        1,
        y.base,
        y.stride,
        f.base,
        f.stride,
        m.geometry.rows(),
    );
    Ok(())
}
fn floats_at(v: &[u32], base: usize, len: usize) -> Result<Zeroizing<Vec<f32>>> {
    let end = base.checked_add(len).ok_or("PLE oracle slice overflow")?;
    let src = v.get(base..end).ok_or("PLE oracle slice")?;
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(len)?;
    if out.capacity() != len {
        return Err("PLE exact host oracle capacity".into());
    }
    out.extend(src.iter().map(|&x| f32::from_bits(x)));
    Ok(out)
}
fn compare_bits(actual: &[u32], expected: &[u32], abs: f32, rel: f32) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("PLE numeric extent".into());
    }
    let mut max = 0f32;
    for (&a, &e) in actual.iter().zip(expected) {
        let a = f32::from_bits(a);
        let e = f32::from_bits(e);
        let d = (a - e).abs();
        if !a.is_finite() || !e.is_finite() || d > abs + rel * e.abs() {
            return Err("PLE numeric parity".into());
        }
        max = max.max(d);
    }
    Ok(max)
}
fn compare_floats(actual: &[u32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("PLE core extent".into());
    }
    let mut max = 0f32;
    for (&a, &e) in actual.iter().zip(expected) {
        let a = f32::from_bits(a);
        let d = (a - e).abs();
        if !a.is_finite() || !e.is_finite() || d > 6e-4 + 8e-4 * e.abs() {
            return Err("PLE original core projected-tail parity".into());
        }
        max = max.max(d);
    }
    Ok(max)
}
fn core_state(g: Geometry, before: &Image) -> Result<CausalConvState> {
    let mut state = CausalConvState::new_bounded(g.channels(), 4, 3, HOST_CAP)?;
    let mut input = Zeroizing::new(Vec::new());
    input.try_reserve_exact(g.channels())?;
    if input.capacity() != g.channels() {
        return Err("PLE exact core input capacity".into());
    }
    input.resize(g.channels(), 0.);
    let mut weights = Zeroizing::new(Vec::new());
    weights.try_reserve_exact(g.channels() * 4)?;
    if weights.capacity() != g.channels() * 4 {
        return Err("PLE exact core weights capacity".into());
    }
    weights.resize(g.channels() * 4, 0.);
    // Independent scalar core API installs the public nonzero old history.
    for j in 0..HISTORY {
        for d in 0..g.channels() {
            input[d] = f32::from_bits(before[OLD][g.old.at(d) + j]);
        }
        drop(Zeroizing::new(state.step(&input, &weights)?));
    }
    Ok(state)
}
#[allow(clippy::too_many_arguments)]
fn run_case(
    arena: &mut PrivateArena,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    p: &PublicLayer,
    retired: &AtomicBool,
    shape: GatedResidualShape,
    count: usize,
    old_tokens: [i32; 2],
    odd: bool,
) -> Result<serde_json::Value> {
    let g = Geometry::new(shape, false)?;
    let sg = Geometry::new(shape, true)?;
    let b = allocate(arena, g)?;
    let s = allocate(arena, sg)?;
    let marker = arena.allocate_zeroed(1)?;
    prove_all(arena, &[(g, &b), (sg, &s)], marker, p)?;
    let mut before = image(g, count)?;
    let tokens = Zeroizing::new([248044, 0, 17, 248046, 7, 248044, 23, 5]);
    let positions = Zeroizing::new(std::array::from_fn(|t| 9 + t as u32));
    let old_tokens = Zeroizing::new(old_tokens);
    let cancel = AtomicBool::new(false);
    let mut candidate_tokens = Zeroizing::new([old_tokens[0], old_tokens[1]]);
    let mut owner = None;
    if odd {
        for t in 0..count {
            for d in 0..g.channels() {
                before[KEY][g.spans[KEY].at(t) + d] =
                    (((d * 11 + t * 3) % 31) as f32 / 17. - 15. / 17.).to_bits();
                before[13][g.spans[13].at(t) + d] = 1;
            }
            for d in 0..g.shape.width {
                before[VALUE][g.spans[VALUE].at(t) + d] =
                    (((d * 3 + t * 11) % 29) as f32 / 31. - 14. / 31.).to_bits();
                before[14][g.spans[14].at(t) + d] = 1;
            }
        }
        write_image(arena, &b, &before)?;
        c::enqueue_tail(
            arena,
            g,
            &b,
            &p.gamma,
            p.gamma_cpu.each_ref().map(|x| x.len()),
            &p.conv,
            p.conv_cpu.len(),
            count,
            &cancel,
        )?;
    } else {
        let rows = RowBatch::prepare(
            graph,
            g,
            count,
            9,
            &tokens,
            &positions,
            &old_tokens,
            &cancel,
        )?;
        let mut history = ops::PleTokenHistory::new(graph.ple_hash());
        for &token in old_tokens.iter() {
            if token >= 0 {
                drop(Zeroizing::new(
                    graph.ple_hash().rows_and_advance(token, &mut history)?,
                ));
            }
        }
        for t in 0..count {
            let expected = Zeroizing::new(
                graph
                    .ple_hash()
                    .rows_and_advance(tokens[t] as i32, &mut history)?,
            );
            if expected.as_slice() != rows.rows()[t] {
                return Err("PLE original hash oracle mismatch".into());
            }
        }
        set_embedding(&mut before, rows.bytes())?;
        candidate_tokens.copy_from_slice(rows.next_tokens());
        drop(rows);
        write_image(arena, &b, &before)?;
        let mut chunk = Chunk::new(graph, g, Buffers(b.0), 9, count)?;
        chunk.enqueue(graph, arena, p, &tokens, &positions, &old_tokens, &cancel)?;
        owner = Some(chunk);
    }
    let actual = snapshot(arena, &b, retired)?;
    if let Some(o) = owner.as_mut() {
        o.accept(
            graph,
            arena,
            &actual,
            &before,
            &cancel,
            retired.load(Ordering::Acquire),
        )?;
        if o.phase() != Phase::Accepted {
            return Err("PLE candidate phase".into());
        }
        if o.candidate_tokens()? != &*candidate_tokens {
            return Err("PLE candidate token history".into());
        }
    } else {
        validate_snapshot(g, count, &actual, &before, &cancel)?;
    }
    let mut cpu = core_state(g, &before)?;
    let mut prior: Option<Image> = None;
    let mut max_scalar = 0f32;
    let mut max_core = 0f32;
    let mut max_projection = 0f32;
    for t in 0..count {
        let mut sbefore = image(sg, 1)?;
        for i in [EMB, RESIDUAL, KEY, VALUE] {
            let width = g.spans[i].width;
            sbefore[i][sg.spans[i].at(0)..sg.spans[i].at(0) + width]
                .copy_from_slice(&before[i][g.spans[i].at(t)..g.spans[i].at(t) + width]);
        }
        if let Some(previous) = prior.as_ref() {
            for d in 0..g.channels() {
                sbefore[OLD][sg.old.at(d)..sg.old.at(d) + 9]
                    .copy_from_slice(&previous[NEXT][sg.next.at(d)..sg.next.at(d) + 9]);
            }
        }
        if odd {
            for i in [13, 14] {
                let base = sg.spans[i].base;
                let width = sg.spans[i].width;
                sbefore[i][base..base + width].fill(1);
            }
        }
        write_image(arena, &s, &sbefore)?;
        if !odd {
            scalar_project(arena, sg, &s, p, 0)?;
            scalar_project(arena, sg, &s, p, 1)?;
        }
        c::enqueue_tail(
            arena,
            sg,
            &s,
            &p.gamma,
            p.gamma_cpu.each_ref().map(|x| x.len()),
            &p.conv,
            p.conv_cpu.len(),
            1,
            &cancel,
        )?;
        let single = snapshot(arena, &s, retired)?;
        validate_snapshot(sg, 1, &single, &sbefore, &cancel)?;
        for i in [KEY, VALUE] {
            let a = g.spans[i];
            let e = sg.spans[i];
            max_projection = max_projection.max(compare_bits(
                &actual[i][a.at(t)..a.at(t) + a.width],
                &single[i][e.base..e.base + e.width],
                1e-4,
                2e-4,
            )?);
        }
        for i in [KEY_NORM, QUERY_NORM, GATES, GATED, NORMALIZED, CONV, OUTPUT] {
            let a = g.spans[i];
            let e = sg.spans[i];
            max_scalar = max_scalar.max(compare_bits(
                &actual[i][a.at(t)..a.at(t) + a.width],
                &single[i][e.base..e.base + e.width],
                3e-4,
                5e-4,
            )?);
        }
        let key = floats_at(&single[KEY], sg.spans[KEY].base, g.channels())?;
        let value = floats_at(&single[VALUE], sg.spans[VALUE].base, g.shape.width)?;
        let r = floats_at(&before[RESIDUAL], g.spans[RESIDUAL].at(t), g.channels())?;
        // Existing core oracle internally allocates; fixed public probe only.
        // Guard returned vectors immediately. Production private adapter remains
        // a separate unaccepted boundary; no oracle data reaches a receipt.
        let oracle = ops::ple_inject_projected(
            &key,
            &value,
            &r,
            &p.gamma_cpu[0][..g.channels()],
            &p.gamma_cpu[1][..g.channels()],
            &p.gamma_cpu[2][..g.channels()],
            &p.conv_cpu[..g.channels() * 4],
            &mut cpu,
            shape,
            1e-6,
        )?;
        let output = Zeroizing::new(oracle.residual);
        let gates = Zeroizing::new(oracle.gates);
        let norm = Zeroizing::new(oracle.normalized);
        for (i, e) in [(OUTPUT, &*output), (GATES, &*gates), (NORMALIZED, &*norm)] {
            let span = g.spans[i];
            max_core = max_core.max(compare_floats(
                &actual[i][span.at(t)..span.at(t) + span.width],
                e,
            )?);
        }
        prior = Some(single);
    }
    let last = prior.as_ref().ok_or("PLE scalar history missing")?;
    for d in 0..g.channels() {
        max_scalar = max_scalar.max(compare_bits(
            &actual[NEXT][g.next.at(d)..g.next.at(d) + 9],
            &last[NEXT][sg.next.at(d)..sg.next.at(d) + 9],
            3e-4,
            5e-4,
        )?);
        max_core = max_core.max(compare_floats(
            &actual[NEXT][g.next.at(d)..g.next.at(d) + 9],
            &cpu.history()[d * 9..d * 9 + 9],
        )?);
    }
    cancel.store(true, Ordering::Release);
    if validate_snapshot(g, count, &actual, &before, &cancel).is_ok() {
        return Err("PLE post-completion cancellation accepted".into());
    }
    drop(prior);
    drop(actual);
    drop(before);
    drop(owner);
    let stale = b.0[0];
    arena.reset()?;
    if arena.layout(stale).is_ok() {
        return Err("PLE stale reset epoch accepted".into());
    }
    Ok(
        serde_json::json!({"count":count,"streams":shape.streams,"width":shape.width,"boundary":if odd{"projected-tail-only"}else{"original-layer1-pre-GR"},"projection_max_abs":max_projection,"new_count1_gpu_tail_max_abs":max_scalar,"original_core_tail_max_abs":max_core,"all_padding_readonly_history_and_candidate_status_accepted":true}),
    )
}
fn reject_cases(
    arena: &mut PrivateArena,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    p: &PublicLayer,
    retired: &AtomicBool,
) -> Result<()> {
    // Late nonfinite and completed cancellation reject the entire candidate.
    for kind in 0..2 {
        let g = Geometry::new(graph.geometry().residual, false)?;
        let b = allocate(arena, g)?;
        c::binding_proof(arena, g, &b, p)?;
        let count = 8;
        let tokens = Zeroizing::new([248044, 0, 17, 248046, 7, 248044, 23, 5]);
        let positions = Zeroizing::new(std::array::from_fn(|t| 9 + t as u32));
        let old = Zeroizing::new([3, 7]);
        let cancel = AtomicBool::new(false);
        let mut before = image(g, count)?;
        let rows = RowBatch::prepare(graph, g, count, 9, &tokens, &positions, &old, &cancel)?;
        set_embedding(&mut before, rows.bytes())?;
        drop(rows);
        if kind == 0 {
            before[RESIDUAL][g.spans[RESIDUAL].at(7) + g.channels() - 1] = f32::NAN.to_bits();
        }
        write_image(arena, &b, &before)?;
        let mut owner = Chunk::new(graph, g, Buffers(b.0), 9, count)?;
        owner.enqueue(graph, arena, p, &tokens, &positions, &old, &cancel)?;
        let actual = snapshot(arena, &b, retired)?;
        if kind == 1 {
            cancel.store(true, Ordering::Release);
        }
        if owner
            .accept(
                graph,
                arena,
                &actual,
                &before,
                &cancel,
                retired.load(Ordering::Acquire),
            )
            .is_ok()
            || owner.phase() != Phase::Poisoned
            || owner.candidate_tokens().is_ok()
        {
            return Err("PLE negative candidate falsely accepted".into());
        }
        drop(owner);
        drop(actual);
        drop(before);
        arena.reset()?;
    }
    // Cancellation and aliases preflight before the fallible private row read
    // or write; all original full owners remain in the arena for checked erase.
    let g = Geometry::new(graph.geometry().residual, false)?;
    let b = allocate(arena, g)?;
    let tokens = Zeroizing::new([3; 8]);
    let positions = Zeroizing::new(std::array::from_fn(|t| 9 + t as u32));
    let old = Zeroizing::new([-1, -1]);
    let mut owner = Chunk::new(graph, g, Buffers(b.0), 9, 1)?;
    if owner
        .enqueue(
            graph,
            arena,
            p,
            &tokens,
            &positions,
            &old,
            &AtomicBool::new(true),
        )
        .is_ok()
        || owner.phase() != Phase::Poisoned
    {
        return Err("PLE preenqueue cancellation accepted".into());
    }
    drop(owner);
    let mut aliases = b.0;
    aliases[KEY] = aliases[RESIDUAL];
    let mut owner = Chunk::new(graph, g, Buffers(aliases), 9, 1)?;
    if owner
        .enqueue(
            graph,
            arena,
            p,
            &tokens,
            &positions,
            &old,
            &AtomicBool::new(false),
        )
        .is_ok()
        || owner.phase() != Phase::Poisoned
    {
        return Err("PLE aliased whole pages accepted".into());
    }
    drop(owner);
    arena.reset()?;
    Ok(())
}
fn host_bound() -> Result<usize> {
    let g = Geometry::new(GatedResidualShape::new(4, 2560)?, false)?;
    // Five full images (batch before/actual + single before/prior/current),
    // two largest upload/read host spans, one embedding source, two history
    // images, sixteen channel-sized oracle temporaries, padded immutable CPU
    // vectors and64KiB fixed arrays/receipt metadata. Existing scalar core
    // allocations are fixed PUBLIC oracle only, not a private API claim.
    let value = 5 * g.footprint()?
        + 2 * 524_288
        + 131_072
        + 2 * 368_640
        + 16 * 10_240 * 4
        + 458_752
        + 65_536;
    if value > HOST_PRIVATE_BOUND || HOST_PRIVATE_BOUND > HOST_CAP {
        return Err("PLE simultaneous host capacity admission".into());
    }
    Ok(value)
}
fn fixture_io_bound() -> Result<usize> {
    let g = Geometry::new(GatedResidualShape::new(4, 2560)?, false)?;
    let sg = Geometry::new(g.shape, true)?;
    let upload = g.footprint()?.max(sg.footprint()?);
    let row = BufferLayout::new(g.words(EMB))?.owned_bytes;
    let largest = (0..N).try_fold(0usize, |n, i| {
        Ok::<_, Box<dyn std::error::Error>>(
            n.max(BufferLayout::new(g.words(i))?.owned_bytes)
                .max(BufferLayout::new(sg.words(i))?.owned_bytes),
        )
    })?;
    let bound = upload
        .checked_add(row)
        .and_then(|n| n.checked_add(largest))
        .and_then(|n| n.checked_add(64 * 2 * 512))
        .and_then(|n| n.checked_mul(2))
        .ok_or("PLE full diagnostic audit bound overflow")?;
    if bound as u64 > IO_CAP {
        return Err("PLE full diagnostic runtime budget".into());
    }
    Ok(bound)
}
fn finish<T>(attempt: std::thread::Result<Result<T>>, clean: bool, retired: bool) -> Result<T> {
    // Preserve the static cause after best-effort close, but only success may
    // pass the complete explicit-cleanup/unretired gate.
    match attempt {
        Ok(Err(e)) => Err(e),
        Err(_) => Err("PLE public probe panic; no receipt".into()),
        Ok(Ok(value)) => {
            if !clean || retired {
                return Err("PLE explicit whole-owner cleanup/context rejected".into());
            }
            Ok(value)
        }
    }
}
pub fn run(client: &ComputeClient<WgpuRuntime>, storage_alignment: usize) -> Result<()> {
    if storage_alignment == 0
        || !256usize.is_multiple_of(storage_alignment)
        || HOST_PRIVATE_BOUND > HOST_CAP
    {
        return Err("PLE alignment/host admission".into());
    }
    let (main, operator_io) = admission()?;
    // Full-page sentinel diagnostic uploads are wider than normal sources.
    let fixture_io = fixture_io_bound()?;
    let host_peak = host_bound()?;
    let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("PLE existing original GGUF required")?;
    let file = GgufFile::open(path, GgufLimits::default()).map_err(|_| "PLE original GGUF open")?;
    let model = Qwen4Exp::bind(&file).map_err(|_| "PLE original graph bind")?;
    let graph = model.bound_graph();
    let public = PublicLayer::load(client, graph)?; // only original public bytes, BEFORE configure
    if main
        .checked_add(public.physical_bytes)
        .is_none_or(|n| n > MAIN_CAP + PUBLIC_CAP)
    {
        return Err("PLE full physical admission".into());
    }
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena = PrivateArena::new_transient(client.clone(), MAIN_CAP, IO_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut reports = Vec::new();
        reports.try_reserve_exact(8)?;
        reject_cases(&mut arena, graph, &public, &retired)?;
        for odd in [false, true] {
            for count in [1, 3, 5, 8] {
                eprintln!(
                    "public_ple_case kind={} count={count}",
                    if odd {
                        "projected-odd"
                    } else {
                        "original-layer1"
                    }
                );
                reports.push(run_case(
                    &mut arena,
                    graph,
                    &public,
                    &retired,
                    if odd {
                        GatedResidualShape::new(3, 97)?
                    } else {
                        graph.geometry().residual
                    },
                    count,
                    if count == 1 {
                        [-1, -1]
                    } else {
                        [248046, 248044]
                    },
                    odd,
                )?);
            }
        }
        Ok::<_, Box<dyn std::error::Error>>(reports)
    }));
    if !matches!(&attempt, Ok(Ok(_))) {
        retired.store(true, Ordering::Release);
    }
    // A terminal unexpected failure still attempts every known main erase;
    // retired audit-mode failure quarantines full handles through process exit.
    let close = arena.close();
    let reports = finish(attempt, close.is_ok(), retired.load(Ordering::Acquire))?;
    let OwnedZeroAudit { buffers, bytes } = arena
        .owned_zero_audit_receipt()
        .ok_or("PLE full main zero receipt")?;
    if file.verify_file_unchanged().is_err() {
        retired.store(true, Ordering::Release);
        return Err("PLE original GGUF changed".into());
    }
    let receipt = serde_json::json!({"scope":"fixed-public-PLE-candidate-only","fixture":"ple-chunk-probe","cases":reports,"max_batch":8,"main_peak_bytes":main,"public_weight_bytes":public.physical_bytes,"runtime_operator_audit_bound_bytes":operator_io,"runtime_fixture_audit_bound_bytes":fixture_io,"runtime_budget_bytes":IO_CAP,"private_host_bound_bytes":host_peak,"private_host_cap_bytes":HOST_CAP,"main_zero_buffers":buffers,"main_zero_bytes":bytes,"explicit_owned_cleanup_accepted":true,"context_retired":false,"late_nonfinite_completed_cancel_preenqueue_cancel_alias_rejected":true,"original_scalar_gpu_projection_comparison":true,"original_resident_tail_is_cpu":true,"count1_gpu_tail_is_new_topology":true,"whole_graph_batched_prefill":false,"production_private_worker_accepted":false,"proof_scope":"API-visible owning pages and tracked IO; excludes opaque driver/register/spill/process-abort copies"});
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
    }
    Ok(())
}
#[cfg(test)]
pub(crate) fn test_core_history(channels: usize) -> Result<CausalConvState> {
    let mut state = CausalConvState::new(channels, 4, 3)?;
    for j in 0..9 {
        let input = (0..channels)
            .map(|c| (c * 11 + j * 7) as f32 / 101. - 0.25)
            .collect::<Vec<_>>();
        state.step(&input, &vec![0.; channels * 4])?;
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_diagnostic_and_host_caps_include_audit_duplicate_and_all_images() {
        assert_eq!(fixture_io_bound().unwrap(), 17_367_040);
        assert_eq!(host_bound().unwrap(), 42_909_696);
        assert!(host_bound().unwrap() < HOST_PRIVATE_BOUND);
    }
    #[test]
    fn owning_lease_decode_scrubs_success_partial_surplus_and_unwind() {
        let mut good = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            &*decode_read(&mut good, 8).unwrap(),
            &[0x04030201, 0x08070605]
        );
        assert_eq!(good, [0; 8]);
        for (len, expected) in [(0, 0), (7, 8), (8, 4), (8, 9), (4, 8)] {
            let mut bytes = vec![0xa5; len];
            assert!(decode_read(&mut bytes, expected).is_err());
            assert!(bytes.iter().all(|&b| b == 0));
        }
        let mut bytes = vec![0xa5; 13];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = ScrubSlice(&mut bytes);
            panic!("fixed lease unwind");
        }));
        assert!(result.is_err());
        assert!(bytes.iter().all(|&b| b == 0));
    }
    #[test]
    fn cleanup_is_required_before_numeric_receipt() {
        assert_eq!(finish(Ok(Ok(7)), true, false).unwrap(), 7);
        for (clean, retired) in [(false, false), (true, true), (false, true)] {
            assert!(finish(Ok(Ok(7)), clean, retired).is_err());
        }
        assert!(finish::<u8>(Err(Box::new("panic")), true, false).is_err());
    }
    #[test]
    fn malformed_zero_audit_retires() {
        for (upload, inspected, buffers) in [(1, 0, 1), (1, 1, 0), (0, 0, 1), (u64::MAX, 0, 1)] {
            let retired = AtomicBool::new(false);
            let r = TransientIoReceipt {
                upload_bytes: upload,
                uniform_bytes: 1,
                readback_bytes: 0,
                inspected_bytes: inspected,
                inspected_buffers: buffers,
                epoch: 1,
            };
            assert!(audit_receipt(&r, &retired).is_err());
            assert!(retired.load(Ordering::Acquire));
        }
    }
}
