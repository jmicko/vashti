//! Fixed-public MAX_BATCH=8 GDN recurrence acceptance. No model or input API.
//! Same physical client throughout; counts/errors only. Every input/source,
//! candidate and GPU comparison image belongs to an audited transient arena.
use crate::{
    gdn_chunk::{
        self as chunk, ChunkBuffers, ChunkSpans, MAX_BATCH, RecurrencePolicy, StateSpan, TokenSpan,
    },
    gdn_gpu::{self, GdnLayout},
    private_arena::{BufferId, BufferLayout, PrivateArena},
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::ops::{self, GdnShape, GdnState};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAIN_BUDGET: usize = 32 * 1024 * 1024;
const IO_BUDGET: u64 = 64 * 1024 * 1024;
const GAP: f32 = -173.25;
const STATUS_GAP: u32 = 0x53aa_9267;

struct Bytes(cubecl::bytes::Bytes);
impl Drop for Bytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn floats(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
) -> Result<Zeroizing<Vec<f32>>> {
    let bytes = Bytes(
        client
            .read_one(arena.handle(id)?.clone())
            .map_err(|_| "GDN chunk read failed")?,
    );
    if bytes.0.len() != arena.layout(id)?.owned_bytes {
        return Err("GDN chunk read span mismatch".into());
    }
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(bytes.0.len() / 4)
        .map_err(|_| "GDN chunk host read allocation failed")?;
    for b in bytes.0.chunks_exact(4) {
        values.push(f32::from_le_bytes(b.try_into().unwrap()));
    }
    Ok(values)
}
fn statuses(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
) -> Result<Zeroizing<Vec<u32>>> {
    let bytes = Bytes(
        client
            .read_one(arena.handle(id)?.clone())
            .map_err(|_| "GDN chunk status read failed")?,
    );
    if bytes.0.len() != arena.layout(id)?.owned_bytes {
        return Err("GDN chunk status span mismatch".into());
    }
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(bytes.0.len() / 4)
        .map_err(|_| "GDN chunk host read allocation failed")?;
    for b in bytes.0.chunks_exact(4) {
        values.push(u32::from_le_bytes(b.try_into().unwrap()));
    }
    Ok(values)
}
fn audited(client: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<serde_json::Value> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io_audited())
    }));
    let r = match result {
        Ok(Ok(r)) => r,
        _ => {
            retired.store(true, Ordering::Release);
            return Err("GDN chunk runtime cleanup failed".into());
        }
    };
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("GDN chunk audit overflow")?;
    if r.inspected_bytes != full
        || full == 0
        || r.inspected_buffers == 0
        || retired.load(Ordering::Acquire)
    {
        retired.store(true, Ordering::Release);
        return Err("GDN chunk runtime audit incomplete".into());
    }
    Ok(
        serde_json::json!({"upload_bytes":r.upload_bytes,"uniform_bytes":r.uniform_bytes,
        "readback_bytes":r.readback_bytes,"inspected_bytes":r.inspected_bytes,"inspected_buffers":r.inspected_buffers}),
    )
}
fn token_span(width: usize, base: usize, gap: usize) -> TokenSpan {
    let stride = width + gap;
    TokenSpan {
        base,
        stride,
        words: base + MAX_BATCH * stride + 13,
    }
}
fn state_span(s: GdnShape, base: usize, column_stride: usize) -> StateSpan {
    let key_stride = s.head_dim * column_stride + 5;
    let head_stride = s.head_dim * key_stride + 7;
    StateSpan {
        base,
        column_stride,
        key_stride,
        head_stride,
        words: base + s.value_heads * head_stride + 19,
    }
}
pub(crate) fn fixture_spans(shape: GdnShape, count: usize) -> Result<ChunkSpans> {
    let l = GdnLayout::new(shape)?;
    let spans = ChunkSpans {
        count,
        query: token_span(l.qk_elements, 3, 7),
        key: token_span(l.qk_elements, 11, 5),
        value: token_span(l.value_elements, 7, 11),
        controls: token_span(l.control_elements, 9, 3),
        old: state_span(shape, 5, 2),
        next: state_span(shape, 13, 1),
        output: token_span(l.value_elements, 5, 13),
        status: token_span(l.value_elements, 17, 9),
    };
    spans.validate(l)?;
    owned_plan(l, spans)?;
    Ok(spans)
}
fn owned_plan(l: GdnLayout, spans: ChunkSpans) -> Result<usize> {
    let reference = [
        l.qk_elements,
        l.qk_elements,
        l.value_elements,
        l.control_elements,
        l.state_elements,
        l.state_elements,
        l.value_elements,
        l.value_elements,
    ];
    let full = spans
        .words()
        .into_iter()
        .chain(reference)
        .try_fold(0_usize, |n, w| {
            n.checked_add(BufferLayout::new(w)?.owned_bytes).ok_or_else(
                || -> Box<dyn std::error::Error> { "GDN chunk full span plan overflow".into() },
            )
        })?;
    if full > MAIN_BUDGET {
        return Err("GDN chunk full span plan exceeds budget".into());
    }
    Ok(full)
}
struct Fixture {
    spans: ChunkSpans,
    query: Zeroizing<Vec<f32>>,
    key: Zeroizing<Vec<f32>>,
    value: Zeroizing<Vec<f32>>,
    controls: Zeroizing<Vec<f32>>,
    old: Zeroizing<Vec<f32>>,
    initial: Zeroizing<Vec<f32>>,
}
fn fixture(shape: GdnShape, count: usize) -> Result<Fixture> {
    let s = fixture_spans(shape, count)?;
    let l = GdnLayout::new(shape)?;
    let mut f = Fixture {
        spans: s,
        query: Zeroizing::new(vec![GAP; s.query.words]),
        key: Zeroizing::new(vec![GAP; s.key.words]),
        value: Zeroizing::new(vec![GAP; s.value.words]),
        controls: Zeroizing::new(vec![GAP; s.controls.words]),
        old: Zeroizing::new(vec![GAP; s.old.words]),
        initial: Zeroizing::new(vec![0.; l.state_elements]),
    };
    for (i, v) in f.initial.iter_mut().enumerate() {
        *v = ((i * 13 + 7) % 31) as f32 / 997. - 15. / 997.;
    }
    for h in 0..shape.value_heads {
        for i in 0..shape.head_dim {
            for j in 0..shape.head_dim {
                f.old[s.old.at(h, i, j)] = f.initial[(h * shape.head_dim + i) * shape.head_dim + j];
            }
        }
    }
    for t in 0..count {
        let raw = Zeroizing::new(
            (0..l.qk_elements)
                .map(|i| ((i * 3 + t * 7) % 23) as f32 - 11.)
                .collect::<Vec<_>>(),
        );
        let norm = Zeroizing::new(ops::l2_norm_heads(&raw, shape.head_dim, 1e-6)?);
        for i in 0..l.qk_elements {
            f.key[s.key.at(t) + i] = norm[i];
            f.query[s.query.at(t) + i] = norm[i] / (shape.head_dim as f32).sqrt();
        }
        for i in 0..l.value_elements {
            f.value[s.value.at(t) + i] = ((i * 5 + t * 11) % 29) as f32 / 31. - 14. / 31.;
        }
        for h in 0..shape.value_heads {
            f.controls[s.controls.at(t) + h] = -0.01 - ((h + t * 3) % 19) as f32 / 23.;
            f.controls[s.controls.at(t) + shape.value_heads + h] =
                0.1 + ((h * 7 + t) % 17) as f32 / 20.;
        }
    }
    Ok(f)
}
fn compare(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("GDN chunk parity extent".into());
    }
    let mut max = 0_f32;
    for (&a, &e) in actual.iter().zip(expected) {
        let err = (a - e).abs();
        if !a.is_finite() || !e.is_finite() || err > 3e-5 + 3e-4 * e.abs() {
            return Err("GDN chunk parity failed".into());
        }
        max = max.max(err);
    }
    Ok(max)
}
fn packed_state(shape: GdnShape, span: StateSpan, values: &[f32]) -> Zeroizing<Vec<f32>> {
    let mut out = Zeroizing::new(Vec::with_capacity(
        shape.value_heads * shape.head_dim * shape.head_dim,
    ));
    for h in 0..shape.value_heads {
        for i in 0..shape.head_dim {
            for j in 0..shape.head_dim {
                out.push(values[span.at(h, i, j)]);
            }
        }
    }
    out
}
fn untouched_floats(
    before: &[f32],
    after: &[f32],
    mut changed: impl FnMut(usize) -> bool,
) -> Result<()> {
    if after.len() < before.len() {
        return Err("GDN chunk preservation span".into());
    }
    for (i, &v) in after.iter().enumerate() {
        if !changed(i) && v.to_bits() != before.get(i).copied().unwrap_or(0.).to_bits() {
            return Err("GDN chunk gap or padding changed".into());
        }
    }
    Ok(())
}
fn active_token(span: TokenSpan, width: usize, count: usize, i: usize) -> bool {
    i >= span.base && (i - span.base) / span.stride < count && (i - span.base) % span.stride < width
}
fn active_state(shape: GdnShape, s: StateSpan, i: usize) -> bool {
    if i < s.base {
        return false;
    }
    let h = (i - s.base) / s.head_stride;
    let rem = (i - s.base) % s.head_stride;
    let key = rem / s.key_stride;
    let col = (rem % s.key_stride) / s.column_stride;
    h < shape.value_heads
        && key < shape.head_dim
        && col < shape.head_dim
        && (rem % s.key_stride).is_multiple_of(s.column_stride)
}
fn write(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
    values: &[f32],
) -> Result<()> {
    if values.len() != arena.layout(id)?.logical_words {
        return Err("GDN chunk write extent".into());
    }
    let mut bytes = Zeroizing::new(vec![0_u8; arena.layout(id)?.owned_bytes]);
    for (b, v) in bytes.chunks_exact_mut(4).zip(values) {
        b.copy_from_slice(&v.to_le_bytes());
    }
    client
        .write_private(arena.handle(id)?.clone(), std::mem::take(&mut *bytes))
        .map_err(|_| "GDN chunk owned write failed")?;
    Ok(())
}
fn single_enqueue(
    arena: &PrivateArena,
    l: GdnLayout,
    b: &ChunkBuffers,
    cancel: &AtomicBool,
) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        return Err("GDN chunk comparison cancelled".into());
    }
    let words = [
        l.qk_elements,
        l.qk_elements,
        l.value_elements,
        l.control_elements,
        l.state_elements,
        l.state_elements,
        l.value_elements,
        l.value_elements,
    ];
    chunk::validate_bindings(arena, b, words)?;
    let arg = |i: usize| -> Result<ArrayArg<WgpuRuntime>> {
        Ok(unsafe { ArrayArg::from_raw_parts(arena.handle(b.0[i])?.clone(), words[i]) })
    };
    gdn_gpu::gdn_step::launch::<WgpuRuntime>(
        arena.client()?,
        CubeCount::Static(l.workgroups_x, 1, 1),
        CubeDim::new_1d(64),
        arg(0)?,
        arg(1)?,
        arg(2)?,
        arg(3)?,
        arg(4)?,
        arg(5)?,
        arg(6)?,
        arg(7)?,
        l.shape.head_dim,
        l.shape.key_heads,
        l.shape.value_heads,
    );
    Ok(())
}
fn case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    shape: GdnShape,
    count: usize,
    retired: &AtomicBool,
    policy: RecurrencePolicy,
) -> Result<serde_json::Value> {
    policy.validate_count(count)?;
    let f = fixture(shape, count)?;
    let spans = f.spans;
    let l = GdnLayout::new(shape)?;
    let out_init = Zeroizing::new(vec![GAP; spans.output.words]);
    let next_init = Zeroizing::new(vec![GAP; spans.next.words]);
    let status_init = Zeroizing::new(vec![STATUS_GAP; spans.status.words]);
    let b = ChunkBuffers([
        arena.upload_f32(&f.query)?,
        arena.upload_f32(&f.key)?,
        arena.upload_f32(&f.value)?,
        arena.upload_f32(&f.controls)?,
        arena.upload_f32(&f.old)?,
        arena.upload_f32(&next_init)?,
        arena.upload_f32(&out_init)?,
        arena.upload_u32(&status_init)?,
    ]);
    let cancel = AtomicBool::new(true);
    if chunk::enqueue_with_policy(arena, shape, spans, &b, &cancel, policy).is_ok() {
        return Err("GDN chunk accepted pre-enqueue cancellation".into());
    }
    cancel.store(false, Ordering::Release);
    if policy == RecurrencePolicy::SingleStreaming {
        let mut forbidden = spans;
        forbidden.count = 3;
        if chunk::enqueue_with_policy(arena, shape, forbidden, &b, &cancel, policy).is_ok() {
            return Err("GDN streaming accepted a nonsingle runtime count".into());
        }
    }
    let mut alias = ChunkBuffers(b.0);
    alias.0[5] = b.0[4];
    if chunk::enqueue_with_policy(arena, shape, spans, &alias, &cancel, policy).is_ok() {
        return Err("GDN chunk accepted old/next alias".into());
    }
    // Retain a complete owned upload of every source through the chunk read:
    // the audit covers payload stages as well as packed runtime INFO.
    for (i, data) in [&*f.query, &*f.key, &*f.value, &*f.controls, &*f.old]
        .into_iter()
        .enumerate()
    {
        write(client, arena, b.0[i], data)?;
    }
    chunk::enqueue_with_policy(arena, shape, spans, &b, &cancel, policy)?;
    let next = floats(client, arena, b.0[5])?;
    let output = floats(client, arena, b.0[6])?;
    let status = statuses(client, arena, b.0[7])?;
    // Every guarded read lease has dropped before this cleanup boundary.
    let io = audited(client, retired)?;
    if io["upload_bytes"].as_u64().unwrap_or(0) == 0
        || io["uniform_bytes"].as_u64().unwrap_or(0) == 0
        || io["readback_bytes"].as_u64().unwrap_or(0) == 0
    {
        return Err("GDN chunk audit missed an I/O kind".into());
    }
    chunk::validate_download(shape, spans, &next, &output, &status, &cancel)?;
    cancel.store(true, Ordering::Release);
    if chunk::validate_download(shape, spans, &next, &output, &status, &cancel).is_ok() {
        return Err("GDN chunk accepted post-fence cancellation".into());
    }
    cancel.store(false, Ordering::Release);
    untouched_floats(&next_init, &next, |i| active_state(shape, spans.next, i))?;
    untouched_floats(&out_init, &output, |i| {
        active_token(spans.output, l.value_elements, count, i)
    })?;
    for (i, &v) in status.iter().enumerate() {
        if !active_token(spans.status, l.value_elements, count, i)
            && v != status_init.get(i).copied().unwrap_or(0)
        {
            return Err("GDN chunk status gaps changed".into());
        }
    }
    let old = floats(client, arena, b.0[4])?;
    untouched_floats(&f.old, &old, |_| false)?;
    audited(client, retired)?;
    if policy == RecurrencePolicy::SingleStreaming {
        // Exercise direct-old finite rejection on the actual device too. The
        // failed column must be zero and the deliberately invalid old image
        // must remain bitwise immutable through maps and checked cleanup.
        let mut invalid_old = Zeroizing::new(f.old.to_vec());
        invalid_old[spans.old.at(0, 0, 0)] = f32::NAN;
        write(client, arena, b.0[3], &f.controls)?;
        write(client, arena, b.0[4], &invalid_old)?;
        chunk::enqueue_with_policy(arena, shape, spans, &b, &cancel, policy)?;
        let failed_next = floats(client, arena, b.0[5])?;
        let failed_out = floats(client, arena, b.0[6])?;
        let failed_status = statuses(client, arena, b.0[7])?;
        if chunk::validate_download(
            shape,
            spans,
            &failed_next,
            &failed_out,
            &failed_status,
            &cancel,
        )
        .is_ok()
            || failed_status[spans.status.base] != 0
            || failed_out[spans.output.base].to_bits() != 0
            || (0..shape.head_dim).any(|i| failed_next[spans.next.at(0, i, 0)].to_bits() != 0)
        {
            return Err(
                "GDN streaming accepted nonfinite old state or retained a candidate".into(),
            );
        }
        let old = floats(client, arena, b.0[4])?;
        untouched_floats(&invalid_old, &old, |_| false)?;
        audited(client, retired)?;
        write(client, arena, b.0[4], &f.old)?;
    }
    let mut oracle = GdnState::new(shape)?;
    oracle.set_values(&f.initial)?;
    let mut single = ChunkBuffers([
        arena.allocate_zeroed(l.qk_elements)?,
        arena.allocate_zeroed(l.qk_elements)?,
        arena.allocate_zeroed(l.value_elements)?,
        arena.allocate_zeroed(l.control_elements)?,
        arena.upload_f32(&f.initial)?,
        arena.allocate_zeroed(l.state_elements)?,
        arena.allocate_zeroed(l.value_elements)?,
        arena.allocate_zeroed(l.value_elements)?,
    ]);
    let mut max_core = 0_f32;
    let mut max_single = 0_f32;
    let mut ref_final = Zeroizing::new(Vec::new());
    for t in 0..count {
        let q = &f.query[spans.query.at(t)..][..l.qk_elements];
        let k = &f.key[spans.key.at(t)..][..l.qk_elements];
        let v = &f.value[spans.value.at(t)..][..l.value_elements];
        let ctl = &f.controls[spans.controls.at(t)..][..l.control_elements];
        let expected = Zeroizing::new(oracle.step(
            q,
            k,
            v,
            &ctl[..shape.value_heads],
            &ctl[shape.value_heads..],
        )?);
        for (i, data) in [q, k, v, ctl].into_iter().enumerate() {
            write(client, arena, single.0[i], data)?;
        }
        single_enqueue(arena, l, &single, &cancel)?;
        let candidate = floats(client, arena, single.0[5])?;
        let out = floats(client, arena, single.0[6])?;
        let st = statuses(client, arena, single.0[7])?;
        gdn_gpu::validate_download(
            l,
            &candidate[..l.state_elements],
            &out[..l.value_elements],
            &st[..l.value_elements],
        )?;
        audited(client, retired)?;
        let chunk_out = &output[spans.output.at(t)..][..l.value_elements];
        max_core = max_core
            .max(compare(chunk_out, &expected)?)
            .max(compare(&candidate[..l.state_elements], oracle.values())?);
        max_single = max_single.max(compare(chunk_out, &out[..l.value_elements])?);
        ref_final.clear();
        ref_final.extend_from_slice(&candidate[..l.state_elements]);
        single.0.swap(4, 5);
    }
    let final_chunk = packed_state(shape, spans.next, &next);
    max_core = max_core.max(compare(&final_chunk, oracle.values())?);
    max_single = max_single.max(compare(&final_chunk, &ref_final)?);
    oracle.reset();
    let mut max_chunk = 0_f32;
    if policy == RecurrencePolicy::SingleStreaming {
        // Independently execute the unchanged count1 chunk on the same immutable
        // old state, without changing any layout or allocating another owner.
        chunk::enqueue(arena, shape, spans, &b, &cancel)?;
        let ordinary_next = floats(client, arena, b.0[5])?;
        let ordinary_out = floats(client, arena, b.0[6])?;
        let ordinary_status = statuses(client, arena, b.0[7])?;
        chunk::validate_download(
            shape,
            spans,
            &ordinary_next,
            &ordinary_out,
            &ordinary_status,
            &cancel,
        )?;
        max_chunk = compare(
            &packed_state(shape, spans.next, &ordinary_next),
            &final_chunk,
        )?
        .max(compare(&ordinary_out, &output)?);
        audited(client, retired)?;
    }
    // A finite invalid control must reject the WHOLE chunk, including earlier
    // per-token statuses. Old state still remains bitwise immutable.
    let mut invalid = Zeroizing::new(f.controls.to_vec());
    invalid[spans.controls.at(count - 1)] = 0.1;
    write(client, arena, b.0[3], &invalid)?;
    chunk::enqueue_with_policy(arena, shape, spans, &b, &cancel, policy)?;
    let failed_next = floats(client, arena, b.0[5])?;
    let failed_out = floats(client, arena, b.0[6])?;
    let failed_status = statuses(client, arena, b.0[7])?;
    if chunk::validate_download(
        shape,
        spans,
        &failed_next,
        &failed_out,
        &failed_status,
        &cancel,
    )
    .is_ok()
    {
        return Err("GDN chunk accepted invalid controls".into());
    }
    for j in 0..shape.head_dim {
        for t in 0..count {
            if failed_status[spans.status.at(t) + j] != 0
                || failed_out[spans.output.at(t) + j].to_bits() != 0
            {
                return Err("GDN invalid control retained a candidate output".into());
            }
        }
        for i in 0..shape.head_dim {
            if failed_next[spans.next.at(0, i, j)].to_bits() != 0 {
                return Err("GDN invalid control retained a candidate state".into());
            }
        }
    }
    let old = floats(client, arena, b.0[4])?;
    untouched_floats(&f.old, &old, |_| false)?;
    audited(client, retired)?;
    let full = arena.charged_bytes();
    if full != owned_plan(l, spans)? {
        return Err("GDN chunk owner plan mismatch".into());
    }
    arena.reset()?; // Audit every FULL owning page before release, including all gaps.
    if b.0.iter().any(|&id| arena.handle(id).is_ok()) || retired.load(Ordering::Acquire) {
        return Err("GDN chunk reset ownership failure".into());
    }
    Ok(
        serde_json::json!({"head_dim":shape.head_dim,"key_heads":shape.key_heads,"value_heads":shape.value_heads,"count":count,
        "recurrence_policy":policy.label(),"max_abs_chunk_error":max_chunk,
        "non_single_policy_count_rejected":policy==RecurrencePolicy::SingleStreaming,
        "nonfinite_old_state_rejected":policy==RecurrencePolicy::SingleStreaming,
        "max_batch":MAX_BATCH,"max_abs_single_error":max_single,"max_abs_core_error":max_core,"main_full_span_bytes":full,
        "runtime_spans":io,"immutable_old":true,"inactive_gaps_padding_preserved":true,"finite_control_failure_rejected":true,
        "cancel_before_enqueue_and_after_fence_rejected":true,"old_next_alias_rejected":true,"full_main_zero_audit":true}),
    )
}

pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), MAIN_BUDGET, IO_BUDGET, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut reports = Vec::with_capacity(10);
        for shape in [GdnShape::new(7, 2, 6)?, GdnShape::qwen38()] {
            for count in [1, 3, 5, 8] {
                reports.push(case(
                    client,
                    &mut arena,
                    shape,
                    count,
                    &retired,
                    RecurrencePolicy::Chunk,
                )?);
            }
            reports.push(case(
                client,
                &mut arena,
                shape,
                1,
                &retired,
                RecurrencePolicy::SingleStreaming,
            )?);
        }
        Ok(reports)
    }));
    let closed = arena.close(); // Always attempt bounded cleanup on error/panic.
    if closed.is_err() || retired.load(Ordering::Acquire) {
        return Err("GDN chunk full cleanup failed/context retired".into());
    }
    let reports = result.map_err(|_| "GDN chunk probe panicked")??;
    let full = arena
        .owned_zero_audit_receipt()
        .ok_or("GDN chunk full main audit missing")?;
    if full.buffers == 0 || full.bytes == 0 {
        return Err("GDN chunk empty audit".into());
    }
    let receipt = serde_json::json!({"scope":"fixed-public-synthetic-operator-test-only","fixture":"gdn-chunk-probe","case_count":reports.len(),
        "max_batch":MAX_BATCH,"main_budget_bytes":MAIN_BUDGET,"runtime_io_budget_bytes":IO_BUDGET,"cases":reports,
        "main_zero_buffers":full.buffers,"main_zero_bytes":full.bytes,"explicit_owned_cleanup_accepted":true,"context_retired":false,
        "production_private_worker_accepted":false,"whole_graph_batched_prefill":false,
        "proof_scope":"owned API-visible full pages and I/O; not opaque driver allocations or private register/spill storage"});
    println!(
        "public_gdn_chunk_acceptance cases=10 main_buffers={} main_bytes={} context_retired=false",
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
    fn fixed_eight_token_pages_and_exact_main_plan_ignore_active_count() {
        for shape in [GdnShape::new(7, 2, 6).unwrap(), GdnShape::qwen38()] {
            let mut expected = None;
            for count in [1, 3, 5, 8] {
                let spans = fixture_spans(shape, count).unwrap();
                let full = owned_plan(GdnLayout::new(shape).unwrap(), spans).unwrap();
                assert!(full <= MAIN_BUDGET);
                if let Some((words, bytes)) = expected {
                    assert_eq!(spans.words(), words);
                    assert_eq!(full, bytes);
                }
                expected = Some((spans.words(), full));
            }
        }
    }
    #[test]
    fn chunk_matches_every_original_f32_step_and_independent_core() {
        for shape in [GdnShape::new(7, 2, 6).unwrap(), GdnShape::qwen38()] {
            for count in [1, 3, 5, 8] {
                let f = fixture(shape, count).unwrap();
                let spans = f.spans;
                let l = GdnLayout::new(shape).unwrap();
                let old_copy = f.old.clone();
                let (next, out, status) = chunk::lane_reference(
                    shape,
                    spans,
                    [&f.query, &f.key, &f.value, &f.controls, &f.old],
                )
                .unwrap();
                if count == 1 {
                    let streaming = chunk::streaming_lane_reference(
                        shape,
                        spans,
                        [&f.query, &f.key, &f.value, &f.controls, &f.old],
                    )
                    .unwrap();
                    assert_eq!(streaming, (next.clone(), out.clone(), status.clone()));
                }
                chunk::validate_download(
                    shape,
                    spans,
                    &next,
                    &out,
                    &status,
                    &AtomicBool::new(false),
                )
                .unwrap();
                let mut original = f.initial.to_vec();
                let mut oracle = GdnState::new(shape).unwrap();
                oracle.set_values(&original).unwrap();
                for t in 0..count {
                    let q = &f.query[spans.query.at(t)..][..l.qk_elements];
                    let k = &f.key[spans.key.at(t)..][..l.qk_elements];
                    let v = &f.value[spans.value.at(t)..][..l.value_elements];
                    let c = &f.controls[spans.controls.at(t)..][..l.control_elements];
                    let single = gdn_gpu::lane_reference(
                        shape,
                        q,
                        k,
                        v,
                        &original,
                        &c[..shape.value_heads],
                        &c[shape.value_heads..],
                    )
                    .unwrap();
                    assert_eq!(
                        &out[spans.output.at(t)..][..l.value_elements],
                        single.output
                    );
                    original = single.state;
                    let core = oracle
                        .step(q, k, v, &c[..shape.value_heads], &c[shape.value_heads..])
                        .unwrap();
                    compare(&out[spans.output.at(t)..][..l.value_elements], &core).unwrap();
                }
                let final_state = packed_state(shape, spans.next, &next);
                assert_eq!(*final_state, original);
                compare(&final_state, oracle.values()).unwrap();
                assert_eq!(*f.old, *old_copy);
                untouched_floats(&vec![GAP; spans.output.words], &out, |i| {
                    active_token(spans.output, l.value_elements, count, i)
                })
                .unwrap();
                untouched_floats(&vec![GAP; spans.next.words], &next, |i| {
                    active_state(shape, spans.next, i)
                })
                .unwrap();
                for (i, &word) in status.iter().enumerate() {
                    if !active_token(spans.status, l.value_elements, count, i) {
                        assert_eq!(
                            word,
                            if i < spans.status.words {
                                STATUS_GAP
                            } else {
                                0
                            }
                        );
                    }
                }
                oracle.reset();
            }
        }
    }
    #[test]
    fn malformed_count_full_capacity_strides_and_overflow_reject_before_dispatch() {
        let shape = GdnShape::new(7, 2, 6).unwrap();
        let l = GdnLayout::new(shape).unwrap();
        let valid = fixture_spans(shape, 1).unwrap();
        let mut bad = valid;
        bad.count = 0;
        assert!(bad.validate(l).is_err());
        bad.count = 9;
        assert!(bad.validate(l).is_err());
        bad = valid;
        bad.query.words = bad.query.base + 7 * bad.query.stride + l.qk_elements - 1;
        assert!(bad.validate(l).is_err());
        bad = valid;
        bad.key.stride = l.qk_elements - 1;
        assert!(bad.validate(l).is_err());
        bad = valid;
        bad.old.column_stride = 0;
        assert!(bad.validate(l).is_err());
        bad = valid;
        bad.old.key_stride = 1;
        assert!(bad.validate(l).is_err());
        bad = valid;
        bad.next.head_stride = 1;
        assert!(bad.validate(l).is_err());
        bad = valid;
        bad.old.base = usize::MAX;
        assert!(bad.validate(l).is_err());
        bad = valid;
        bad.controls.stride = usize::MAX;
        assert!(bad.validate(l).is_err());
        let mut modified = l;
        modified.state_elements += 1;
        assert!(valid.validate(modified).is_err());
        assert!(fixture_spans(GdnShape::new(129, 1, 1).unwrap(), 1).is_err());
        let empty = GdnShape {
            head_dim: 0,
            key_heads: 1,
            value_heads: 1,
        };
        assert!(valid.old.validate(empty).is_err());
    }
    #[test]
    fn public_streaming_policy_rejects_non_single_counts_and_retains_default_chunk() {
        assert_eq!(RecurrencePolicy::default(), RecurrencePolicy::Chunk);
        for count in [0, 1, 3, 5, 8, 9, usize::MAX] {
            assert_eq!(
                RecurrencePolicy::Chunk.validate_count(count).is_ok(),
                (1..=8).contains(&count)
            );
            assert_eq!(
                RecurrencePolicy::SingleStreaming
                    .validate_count(count)
                    .is_ok(),
                count == 1
            );
            if count != 1 && (1..=8).contains(&count) {
                let f = fixture(GdnShape::new(7, 2, 6).unwrap(), count).unwrap();
                assert!(
                    chunk::streaming_lane_reference(
                        GdnShape::new(7, 2, 6).unwrap(),
                        f.spans,
                        [&f.query, &f.key, &f.value, &f.controls, &f.old]
                    )
                    .is_err()
                );
            }
        }
    }
    #[test]
    fn streaming_bad_controls_old_prediction_and_readout_zero_candidates_without_mutating_old() {
        let shape = GdnShape::new(7, 2, 6).unwrap();
        for which in 0..8 {
            let mut f = fixture(shape, 1).unwrap();
            let s = f.spans;
            match which {
                0 => f.controls[s.controls.base] = 0.1,
                1 => f.controls[s.controls.base + shape.value_heads] = 1.1,
                2 => f.controls[s.controls.base] = f32::NAN,
                3 => f.old[s.old.at(0, 0, 0)] = f32::NAN,
                4 => f.old[s.old.at(0, 0, 0)] = f32::INFINITY,
                5 => f.value[s.value.base] = f32::INFINITY,
                6 => {
                    for i in 0..shape.head_dim {
                        f.old[s.old.at(0, i, 0)] = f32::MAX;
                        f.key[s.key.base + i] = 1.;
                    }
                }
                _ => {
                    f.query[s.query.base] = f32::MAX;
                    f.query[s.query.base + 1] = f32::MAX;
                    f.old[s.old.at(0, 0, 0)] = f32::MAX / 4.;
                    f.old[s.old.at(0, 1, 0)] = f32::MAX / 4.;
                    f.controls[s.controls.base] = 0.;
                    f.controls[s.controls.base + shape.value_heads] = 0.;
                    for i in 0..shape.head_dim {
                        f.key[s.key.base + i] = 0.;
                    }
                }
            }
            let old_bits = f.old.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let inputs: [&[f32]; 5] = [&f.query, &f.key, &f.value, &f.controls, &f.old];
            let streaming = chunk::streaming_lane_reference(shape, s, inputs).unwrap();
            let ordinary = chunk::lane_reference(shape, s, inputs).unwrap();
            assert_eq!(streaming, ordinary);
            let (next, out, status) = streaming;
            assert_eq!(status[s.status.base], 0);
            assert_eq!(out[s.output.base].to_bits(), 0);
            for i in 0..shape.head_dim {
                assert_eq!(next[s.next.at(0, i, 0)].to_bits(), 0);
            }
            assert!(
                chunk::validate_download(shape, s, &next, &out, &status, &AtomicBool::new(false))
                    .is_err()
            );
            assert_eq!(
                old_bits,
                f.old.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn invalid_last_controls_or_nonfinite_intermediate_reject_whole_chunk_and_preserve_old() {
        let shape = GdnShape::new(7, 2, 6).unwrap();
        for which in 0..5 {
            let mut f = fixture(shape, 5).unwrap();
            let old = f.old.clone();
            match which {
                0 => f.controls[f.spans.controls.at(4)] = 0.1,
                1 => f.controls[f.spans.controls.at(4) + shape.value_heads] = 1.1,
                2 => f.controls[f.spans.controls.at(3)] = f32::NAN,
                3 => f.value[f.spans.value.at(2)] = f32::INFINITY,
                _ => {
                    for i in 0..shape.head_dim {
                        f.old[f.spans.old.at(0, i, 0)] = f32::MAX;
                        f.key[f.spans.key.at(0) + i] = 1. / (shape.head_dim as f32).sqrt();
                    }
                }
            }
            let authoritative = f.old.clone();
            let (next, out, status) = chunk::lane_reference(
                shape,
                f.spans,
                [&f.query, &f.key, &f.value, &f.controls, &f.old],
            )
            .unwrap();
            {
                assert!(
                    chunk::validate_download(
                        shape,
                        f.spans,
                        &next,
                        &out,
                        &status,
                        &AtomicBool::new(false)
                    )
                    .is_err()
                );
                for t in 0..f.spans.count {
                    assert_eq!(status[f.spans.status.at(t)], 0);
                }
            }
            assert_eq!(*f.old, *authoritative);
            assert_eq!(old.len(), f.old.len());
        }
    }
    #[test]
    fn no_partial_download_or_cancelled_chunk_can_be_accepted() {
        let shape = GdnShape::new(7, 2, 6).unwrap();
        let f = fixture(shape, 3).unwrap();
        let (next, out, status) = chunk::lane_reference(
            shape,
            f.spans,
            [&f.query, &f.key, &f.value, &f.controls, &f.old],
        )
        .unwrap();
        let no = AtomicBool::new(false);
        assert!(
            chunk::validate_download(shape, f.spans, &next[..next.len() - 1], &out, &status, &no)
                .is_err()
        );
        let mut surplus = next.clone();
        surplus.push(0.);
        assert!(chunk::validate_download(shape, f.spans, &surplus, &out, &status, &no).is_err());
        let mut bad = status.clone();
        bad[f.spans.status.at(2) + 41] = 0;
        assert!(chunk::validate_download(shape, f.spans, &next, &out, &bad, &no).is_err());
        let mut bad = out.clone();
        bad[f.spans.output.at(2) + 41] = f32::NAN;
        assert!(chunk::validate_download(shape, f.spans, &next, &bad, &status, &no).is_err());
        let mut bad = next.clone();
        bad[f.spans.next.at(5, 6, 6)] = f32::INFINITY;
        assert!(chunk::validate_download(shape, f.spans, &bad, &out, &status, &no).is_err());
        assert!(
            chunk::validate_download(shape, f.spans, &next, &out, &status, &AtomicBool::new(true))
                .is_err()
        );
    }
    #[test]
    fn all_addressing_and_active_count_are_runtime_and_info_fits_bounded_page() {
        let source = include_str!("gdn_chunk.rs");
        let sig = source
            .split("pub fn gdn_chunk(")
            .nth(1)
            .unwrap()
            .split(") {")
            .next()
            .unwrap();
        let runtime = sig.split("#[comptime]").next().unwrap();
        for name in [
            "count",
            "q_base",
            "q_stride",
            "k_base",
            "k_stride",
            "v_base",
            "v_stride",
            "c_base",
            "c_stride",
            "old_base",
            "old_head",
            "old_key",
            "old_column",
            "next_base",
            "next_head",
            "next_key",
            "next_column",
            "out_base",
            "out_stride",
            "status_base",
            "status_stride",
        ] {
            assert!(runtime.contains(&format!("{name}: usize")));
        }
        assert!(sig.contains("#[comptime] head_dim"));
        assert!(sig.contains("#[comptime] key_heads"));
        assert!(sig.contains("#[comptime] value_heads"));
        let address = cubecl::ir::AddressType::U64;
        let mut info = cubecl::InfoBuilder::default();
        for _ in 0..8 {
            info.metadata.register_array(2_097_152, 2_097_152, address);
        }
        for _ in 0..21 {
            info.scalars.push(1_u64);
        }
        assert!(info.finish(address).data.len() * 8 <= 512);
    }
    fn shader(
        shape: GdnShape,
        mode: cubecl::server::ExecutionMode,
        policy: RecurrencePolicy,
    ) -> String {
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
        let q = b.input_array(Type::new(float)).into();
        let k = b.input_array(Type::new(float)).into();
        let v = b.input_array(Type::new(float)).into();
        let c = b.input_array(Type::new(float)).into();
        let old = b.input_array(Type::new(float)).into();
        let next = b.output_array(Type::new(float)).into();
        let out = b.output_array(Type::new(float)).into();
        let status = b.output_array(Type::new(uint)).into();
        let scalars = (0..21)
            .map(|_| b.scalar(uint).into())
            .collect::<Vec<cubecl::frontend::NativeExpand<usize>>>();
        macro_rules! expand {
            ($kernel:ident) => {
                chunk::$kernel::expand(
                    &mut b.scope,
                    q,
                    k,
                    v,
                    c,
                    old,
                    next,
                    out,
                    status,
                    scalars[0].clone(),
                    scalars[1].clone(),
                    scalars[2].clone(),
                    scalars[3].clone(),
                    scalars[4].clone(),
                    scalars[5].clone(),
                    scalars[6].clone(),
                    scalars[7].clone(),
                    scalars[8].clone(),
                    scalars[9].clone(),
                    scalars[10].clone(),
                    scalars[11].clone(),
                    scalars[12].clone(),
                    scalars[13].clone(),
                    scalars[14].clone(),
                    scalars[15].clone(),
                    scalars[16].clone(),
                    scalars[17].clone(),
                    scalars[18].clone(),
                    scalars[19].clone(),
                    scalars[20].clone(),
                    shape.head_dim,
                    shape.key_heads,
                    shape.value_heads,
                )
            };
        }
        match policy {
            RecurrencePolicy::Chunk => expand!(gdn_chunk),
            RecurrencePolicy::SingleStreaming => expand!(gdn_single_streaming),
            RecurrencePolicy::ThreeStreaming => expand!(gdn_three_streaming),
            RecurrencePolicy::FourStreaming => expand!(gdn_four_streaming),
        }
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name(match policy {
                            RecurrencePolicy::Chunk => "gdn_chunk",
                            RecurrencePolicy::SingleStreaming => "gdn_single_streaming",
                            RecurrencePolicy::ThreeStreaming => "gdn_three_streaming",
                            RecurrencePolicy::FourStreaming => "gdn_four_streaming",
                        }),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                uint,
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn offline_checked_and_unchecked_wgsl_use_private_column_and_runtime_count() {
        for shape in [GdnShape::new(7, 2, 6).unwrap(), GdnShape::qwen38()] {
            for mode in [
                cubecl::server::ExecutionMode::Checked,
                cubecl::server::ExecutionMode::Unchecked,
            ] {
                let wgsl = shader(shape, mode, RecurrencePolicy::Chunk);
                assert!(wgsl.contains("exp("));
                assert!(wgsl.contains("array<f32,"));
                assert!(!wgsl.contains("workgroupBarrier"));
                assert!(!wgsl.contains("fma("));
                assert!(!wgsl.contains("array<f64"));
                for scalar in 0..21 {
                    assert!(wgsl.contains(&format!("scalars_u32[{scalar}]")));
                }
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
    #[test]
    fn offline_streaming_wgsl_uses_runtime_spans_without_private_column_array() {
        for shape in [GdnShape::new(7, 2, 6).unwrap(), GdnShape::qwen38()] {
            for mode in [
                cubecl::server::ExecutionMode::Checked,
                cubecl::server::ExecutionMode::Unchecked,
            ] {
                let wgsl = shader(shape, mode, RecurrencePolicy::SingleStreaming);
                assert!(wgsl.contains("exp("));
                assert!(!wgsl.contains("array<f32,"));
                assert!(!wgsl.contains("workgroupBarrier"));
                assert!(!wgsl.contains("fma("));
                assert!(!wgsl.contains("array<f64"));
                for scalar in 0..21 {
                    assert!(wgsl.contains(&format!("scalars_u32[{scalar}]")));
                }
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
    #[test]
    fn four_streaming_exact_count4_uses_original_tail_kernel_and_ordered_reference() {
        for count in 1..=3 {
            assert_eq!(
                RecurrencePolicy::FourStreaming
                    .execution_policy(count)
                    .unwrap(),
                RecurrencePolicy::Chunk
            );
        }
        assert_eq!(
            RecurrencePolicy::FourStreaming.execution_policy(4).unwrap(),
            RecurrencePolicy::FourStreaming
        );
        assert!(RecurrencePolicy::FourStreaming.execution_policy(5).is_err());
        for shape in [GdnShape::new(7, 2, 6).unwrap(), GdnShape::qwen38()] {
            let f = fixture(shape, 4).unwrap();
            let inputs = [
                &f.query[..],
                &f.key[..],
                &f.value[..],
                &f.controls[..],
                &f.old[..],
            ];
            let original = chunk::lane_reference(shape, f.spans, inputs).unwrap();
            let four = chunk::four_streaming_lane_reference(shape, f.spans, inputs).unwrap();
            three_streaming_assert_exact(&original, &four);
            for mode in [
                cubecl::server::ExecutionMode::Checked,
                cubecl::server::ExecutionMode::Unchecked,
            ] {
                let wgsl = shader(shape, mode, RecurrencePolicy::FourStreaming);
                assert!(!wgsl.contains("array<f32,"));
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
    include!("gdn_three_streaming_tests.rs");
}
