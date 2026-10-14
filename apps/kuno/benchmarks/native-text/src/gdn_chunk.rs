//! Bounded consecutive-token GDN recurrence candidate, not a model graph.
//! Q/K are already normalized and Q scaled. GGUF head pairing stays h%K.
//! Only public GdnShape specializes; count and EVERY base/stride are runtime.
//! Default Chunk retains a private state column and repeats gdn_step's exact f32
//! decay/prediction/delta/update/readout order. Old state is immutable. No state
//! commit is provided here: every active output/status AND final state must
//! pass acceptance after completion, cancellation and checked owner cleanup.
use crate::{
    gdn_gpu::GdnLayout,
    private_arena::{BufferId, BufferLayout, PrivateArena},
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::{Error, Result, ops::GdnShape};
use std::sync::atomic::{AtomicBool, Ordering};

pub const MAX_BATCH: usize = 8;
pub const MAX_HEAD_DIM: usize = 128;
const LANES: usize = 64;
/// Explicit public laboratory policy. Never infer a specialization from a
/// private token count. Each streaming policy admits its fixed public ceiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RecurrencePolicy {
    #[default]
    Chunk,
    SingleStreaming,
    ThreeStreaming,
    FourStreaming,
}
impl RecurrencePolicy {
    pub(crate) fn validate_count(self, count: usize) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&count)
            || self == Self::SingleStreaming && count != 1
            || self == Self::ThreeStreaming && count > 3
            || self == Self::FourStreaming && count > crate::strata_port::verify_capacity()
        {
            return Err(fault());
        }
        Ok(())
    }
    pub(crate) fn execution_policy(self, count: usize) -> Result<Self> {
        self.validate_count(count)?;
        Ok(
            if self == Self::FourStreaming && count != 4 && !crate::strata_intel_core::enabled() {
                Self::Chunk
            } else {
                self
            },
        )
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::SingleStreaming => "single-streaming",
            Self::ThreeStreaming => "three-streaming",
            Self::FourStreaming => "four-streaming",
        }
    }
}
fn fault() -> Error {
    Error::InvalidFormat("GDN chunk invalid layout or result")
}

#[cfg(test)]
#[path = "gdn_retained_metadata_tests.rs"]
mod retained_metadata_tests;

#[derive(Clone, Copy)]
pub struct TokenSpan {
    pub base: usize,
    pub stride: usize,
    pub words: usize,
}
impl TokenSpan {
    pub fn validate(self, width: usize) -> Result<()> {
        if width == 0
            || self.stride < width
            || self.words > u32::MAX as usize
            || self
                .base
                .checked_add(
                    (MAX_BATCH - 1)
                        .checked_mul(self.stride)
                        .ok_or(Error::ArithmeticOverflow)?,
                )
                .and_then(|n| n.checked_add(width))
                .is_none_or(|n| n > self.words)
        {
            return Err(fault());
        }
        Ok(())
    }
    pub fn at(self, token: usize) -> usize {
        self.base + token * self.stride
    }
}
#[derive(Clone, Copy)]
pub struct StateSpan {
    pub base: usize,
    pub head_stride: usize,
    pub key_stride: usize,
    pub column_stride: usize,
    pub words: usize,
}
impl StateSpan {
    pub fn validate(self, shape: GdnShape) -> Result<()> {
        GdnShape::new(shape.head_dim, shape.key_heads, shape.value_heads).map_err(|_| fault())?;
        let row = (shape.head_dim - 1)
            .checked_mul(self.column_stride)
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::ArithmeticOverflow)?;
        let head = (shape.head_dim - 1)
            .checked_mul(self.key_stride)
            .and_then(|n| n.checked_add(row))
            .ok_or(Error::ArithmeticOverflow)?;
        let end = (shape.value_heads - 1)
            .checked_mul(self.head_stride)
            .and_then(|n| n.checked_add(head))
            .and_then(|n| n.checked_add(self.base))
            .ok_or(Error::ArithmeticOverflow)?;
        if self.column_stride == 0
            || self.head_stride > u32::MAX as usize
            || self.key_stride > u32::MAX as usize
            || self.column_stride > u32::MAX as usize
            || self.key_stride < row
            || self.head_stride < head
            || self.words > u32::MAX as usize
            || end > self.words
        {
            return Err(fault());
        }
        Ok(())
    }
    pub fn at(self, head: usize, key: usize, column: usize) -> usize {
        self.base + head * self.head_stride + key * self.key_stride + column * self.column_stride
    }
}

#[cfg(test)]
pub(crate) fn lane_reference(
    shape: GdnShape,
    spans: ChunkSpans,
    inputs: [&[f32]; 5],
) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    let l = GdnLayout::new(shape).map_err(|_| fault())?;
    spans.validate(l)?;
    let [q, k, v, c, old] = inputs;
    if inputs
        .iter()
        .zip(spans.words())
        .take(5)
        .any(|(s, n)| s.len() != n)
    {
        return Err(fault());
    }
    let mut next = vec![
        0.;
        BufferLayout::new(spans.next.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    next[..spans.next.words].fill(-173.25);
    let mut out = vec![
        0.;
        BufferLayout::new(spans.output.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    out[..spans.output.words].fill(-173.25);
    let mut status = vec![
        0;
        BufferLayout::new(spans.status.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    status[..spans.status.words].fill(0x53aa_9267);
    for lane in 0..l.value_elements {
        let h = lane / shape.head_dim;
        let j = lane % shape.head_dim;
        let kh = h % shape.key_heads;
        let mut column = (0..shape.head_dim)
            .map(|i| old[spans.old.at(h, i, j)])
            .collect::<Vec<_>>();
        let mut valid = column.iter().all(|x| x.is_finite());
        for token in 0..spans.count {
            let qi = spans.query.at(token) + kh * shape.head_dim;
            let ki = spans.key.at(token) + kh * shape.head_dim;
            let ci = spans.controls.at(token);
            let log = c[ci + h];
            let beta = c[ci + shape.value_heads + h];
            let decay = log.exp();
            valid = valid
                && log.is_finite()
                && log <= 0.
                && beta.is_finite()
                && (0.0..=1.0).contains(&beta);
            let mut predicted = 0_f32;
            for i in 0..shape.head_dim {
                let decayed = column[i] * decay;
                predicted += decayed * k[ki + i];
            }
            let delta = (v[spans.value.at(token) + lane] - predicted) * beta;
            valid = valid && predicted.is_finite() && delta.is_finite();
            status[spans.status.at(token) + lane] = 0;
            out[spans.output.at(token) + lane] = 0.;
            if valid {
                let mut readout = 0_f32;
                for i in 0..shape.head_dim {
                    let decayed = column[i] * decay;
                    let updated = decayed + k[ki + i] * delta;
                    valid = valid && updated.is_finite();
                    column[i] = updated;
                    readout += updated * q[qi + i];
                }
                valid = valid && readout.is_finite();
                if valid {
                    out[spans.output.at(token) + lane] = readout;
                    status[spans.status.at(token) + lane] = 1;
                }
            }
        }
        for i in 0..shape.head_dim {
            next[spans.next.at(h, i, j)] = if valid { column[i] } else { 0. };
        }
        if !valid {
            for token in 0..spans.count {
                out[spans.output.at(token) + lane] = 0.;
                status[spans.status.at(token) + lane] = 0;
            }
        }
    }
    Ok((next, out, status))
}
#[derive(Clone, Copy)]
pub struct ChunkSpans {
    pub count: usize,
    pub query: TokenSpan,
    pub key: TokenSpan,
    pub value: TokenSpan,
    pub controls: TokenSpan,
    pub old: StateSpan,
    pub next: StateSpan,
    pub output: TokenSpan,
    pub status: TokenSpan,
}
impl ChunkSpans {
    pub fn validate(self, layout: GdnLayout) -> Result<()> {
        if layout != GdnLayout::new(layout.shape).map_err(|_| fault())?
            || layout.shape.head_dim > MAX_HEAD_DIM
            || !(1..=MAX_BATCH).contains(&self.count)
        {
            return Err(fault());
        }
        for (s, n) in [
            (self.query, layout.qk_elements),
            (self.key, layout.qk_elements),
            (self.value, layout.value_elements),
            (self.controls, layout.control_elements),
            (self.output, layout.value_elements),
            (self.status, layout.value_elements),
        ] {
            s.validate(n)?;
        }
        self.old.validate(layout.shape)?;
        self.next.validate(layout.shape)?;
        for n in self.words() {
            BufferLayout::new(n).map_err(|_| fault())?;
        }
        Ok(())
    }
    pub fn words(self) -> [usize; 8] {
        [
            self.query.words,
            self.key.words,
            self.value.words,
            self.controls.words,
            self.old.words,
            self.next.words,
            self.output.words,
            self.status.words,
        ]
    }
}
pub struct ChunkBuffers(pub [BufferId; 8]);

/// Reject semantic or physical alias/hidden padding before dispatch. All eight
/// handles remain owned by this same arena/stream through explicit cleanup.
pub fn validate_bindings(
    arena: &PrivateArena,
    buffers: &ChunkBuffers,
    words: [usize; 8],
) -> Result<()> {
    let client = arena.client().map_err(|_| fault())?;
    for (index, (&id, n)) in buffers.0.iter().zip(words).enumerate() {
        if buffers.0[..index].contains(&id)
            || arena.layout(id).map_err(|_| fault())?.logical_words != n
        {
            return Err(fault());
        }
        let h = arena.handle(id).map_err(|_| fault())?;
        if h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.stream != arena.handle(buffers.0[0]).map_err(|_| fault())?.stream
        {
            return Err(fault());
        }
    }
    // Eight full owner headers only; no tensor bytes or new retained owner.
    // Include the client/server conversion's headers conservatively within
    // the existing 64 KiB/stage runtime reservation, alongside its INFO bound.
    const HEADER_BYTES: usize = 8
        * (std::mem::size_of::<cubecl::server::Handle>()
            + std::mem::size_of::<cubecl::server::Binding>()
            + 2 * std::mem::size_of::<
                cubecl_runtime::storage::ManagedResource<cubecl::wgpu::WgpuResource>,
            >())
        + 6 * std::mem::size_of::<Vec<usize>>();
    const { assert!(HEADER_BYTES <= 4096) };
    const { assert!(37 * 2 * 512 + HEADER_BYTES <= 64 * 2 * 512) };
    let mut handles = Vec::with_capacity(buffers.0.len());
    if handles.capacity() != 8 {
        return Err(fault());
    }
    for &id in &buffers.0 {
        handles.push(arena.handle(id).map_err(|_| fault())?.clone());
    }
    // Default retains ordered get_resource calls. The selected existing
    // metadata policy inspects the same full roots without submitting work.
    let resources = crate::resident_util::resources(client, handles).map_err(|_| fault())?;
    if resources.len() != 8 || resources.capacity() != 8 {
        return Err(fault());
    }
    for (index, r) in resources.iter().enumerate() {
        let h = arena.handle(buffers.0[index]).map_err(|_| fault())?;
        let region = r.resource();
        if region.offset != 0 || region.size != h.size() || region.buffer.size() != h.size() {
            return Err(fault());
        }
        if resources[..index]
            .iter()
            .any(|old| old.resource().buffer == region.buffer)
        {
            return Err(fault());
        }
    }
    Ok(())
}

pub fn enqueue(
    arena: &PrivateArena,
    shape: GdnShape,
    spans: ChunkSpans,
    buffers: &ChunkBuffers,
    cancel: &AtomicBool,
) -> Result<()> {
    enqueue_with_policy(
        arena,
        shape,
        spans,
        buffers,
        cancel,
        RecurrencePolicy::Chunk,
    )
}

pub(crate) fn enqueue_with_policy(
    arena: &PrivateArena,
    shape: GdnShape,
    spans: ChunkSpans,
    buffers: &ChunkBuffers,
    cancel: &AtomicBool,
    policy: RecurrencePolicy,
) -> Result<()> {
    let layout = GdnLayout::new(shape).map_err(|_| fault())?;
    spans.validate(layout)?;
    policy.validate_count(spans.count)?;
    if cancel.load(Ordering::Acquire) {
        return Err(Error::InvalidFormat("GDN chunk cancelled"));
    }
    validate_bindings(arena, buffers, spans.words())?;
    let client = arena.client().map_err(|_| fault())?;
    let arg = |i: usize| -> Result<ArrayArg<WgpuRuntime>> {
        let layout = arena.layout(buffers.0[i]).map_err(|_| fault())?;
        // SAFETY: full same-stream resource and distinct ownership just checked.
        Ok(unsafe {
            ArrayArg::from_raw_parts(
                arena.handle(buffers.0[i]).map_err(|_| fault())?.clone(),
                layout.owned_words,
            )
        })
    };
    if cancel.load(Ordering::Acquire) {
        return Err(Error::InvalidFormat("GDN chunk cancelled"));
    }
    macro_rules! launch {
        ($kernel:ident) => {
            $kernel::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.workgroups_x, 1, 1),
                CubeDim::new_1d(LANES as u32),
                arg(0)?,
                arg(1)?,
                arg(2)?,
                arg(3)?,
                arg(4)?,
                arg(5)?,
                arg(6)?,
                arg(7)?,
                spans.count,
                spans.query.base,
                spans.query.stride,
                spans.key.base,
                spans.key.stride,
                spans.value.base,
                spans.value.stride,
                spans.controls.base,
                spans.controls.stride,
                spans.old.base,
                spans.old.head_stride,
                spans.old.key_stride,
                spans.old.column_stride,
                spans.next.base,
                spans.next.head_stride,
                spans.next.key_stride,
                spans.next.column_stride,
                spans.output.base,
                spans.output.stride,
                spans.status.base,
                spans.status.stride,
                shape.head_dim,
                shape.key_heads,
                shape.value_heads,
            )
        };
    }
    match policy.execution_policy(spans.count)? {
        RecurrencePolicy::Chunk => launch!(gdn_chunk),
        RecurrencePolicy::SingleStreaming => launch!(gdn_single_streaming),
        RecurrencePolicy::ThreeStreaming => launch!(gdn_three_streaming),
        RecurrencePolicy::FourStreaming => launch!(gdn_four_streaming),
    }
    Ok(())
}

pub(crate) fn enqueue_shared_count1(
    arena: &PrivateArena,
    shape: GdnShape,
    spans: ChunkSpans,
    buffers: &ChunkBuffers,
    cancel: &AtomicBool,
) -> Result<()> {
    let layout = GdnLayout::new(shape).map_err(|_| fault())?;
    spans.validate(layout)?;
    RecurrencePolicy::SingleStreaming.validate_count(spans.count)?;
    if cancel.load(Ordering::Acquire) {
        return Err(fault());
    }
    validate_bindings(arena, buffers, spans.words())?;
    let mut handles = Vec::with_capacity(8);
    for id in buffers.0 {
        handles.push(arena.handle(id).map_err(|_| fault())?.clone());
    }
    crate::next_shared::gdn(
        arena.client().map_err(|_| fault())?,
        shape,
        spans,
        handles.try_into().map_err(|_| fault())?,
    )
}

pub fn validate_download(
    shape: GdnShape,
    spans: ChunkSpans,
    next: &[f32],
    output: &[f32],
    status: &[u32],
    cancel: &AtomicBool,
) -> Result<()> {
    let l = GdnLayout::new(shape).map_err(|_| fault())?;
    spans.validate(l)?;
    if cancel.load(Ordering::Acquire) {
        return Err(Error::InvalidFormat("GDN chunk cancelled"));
    }
    if next.len()
        != BufferLayout::new(spans.next.words)
            .map_err(|_| fault())?
            .owned_words
        || output.len()
            != BufferLayout::new(spans.output.words)
                .map_err(|_| fault())?
                .owned_words
        || status.len()
            != BufferLayout::new(spans.status.words)
                .map_err(|_| fault())?
                .owned_words
    {
        return Err(fault());
    }
    for t in 0..spans.count {
        if output[spans.output.at(t)..][..l.value_elements]
            .iter()
            .any(|x| !x.is_finite())
            || status[spans.status.at(t)..][..l.value_elements]
                .iter()
                .any(|&s| s != 1)
        {
            return Err(fault());
        }
    }
    for h in 0..shape.value_heads {
        for i in 0..shape.head_dim {
            for j in 0..shape.head_dim {
                if !next[spans.next.at(h, i, j)].is_finite() {
                    return Err(fault());
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::manual_range_contains)]
#[cube]
fn finite(v: f32) -> bool {
    v <= 3.4028235e38_f32 && v >= -3.4028235e38_f32
}

// Rust Range::contains does not lower through the shader DSL.
#[allow(clippy::manual_range_contains)]
#[cube(launch)]
pub fn gdn_chunk(
    query: &Array<f32>,
    key: &Array<f32>,
    value: &Array<f32>,
    controls: &Array<f32>,
    old: &Array<f32>,
    next: &mut Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    q_base: usize,
    q_stride: usize,
    k_base: usize,
    k_stride: usize,
    v_base: usize,
    v_stride: usize,
    c_base: usize,
    c_stride: usize,
    old_base: usize,
    old_head: usize,
    old_key: usize,
    old_column: usize,
    next_base: usize,
    next_head: usize,
    next_key: usize,
    next_column: usize,
    out_base: usize,
    out_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] head_dim: usize,
    #[comptime] key_heads: usize,
    #[comptime] value_heads: usize,
) {
    let lane = ABSOLUTE_POS;
    if lane < value_heads * head_dim {
        let h = lane / head_dim;
        let j = lane % head_dim;
        let kh = h % key_heads;
        let mut column = Array::<f32>::new(head_dim);
        let mut valid = true;
        for i in 0..head_dim {
            column[i] = old[old_base + h * old_head + i * old_key + j * old_column];
            valid = valid && finite(column[i]);
        }
        for token in 0..count {
            let qi = q_base + token * q_stride + kh * head_dim;
            let ki = k_base + token * k_stride + kh * head_dim;
            let ci = c_base + token * c_stride;
            let log_decay = controls[ci + h];
            let beta = controls[ci + value_heads + h];
            let decay = log_decay.exp();
            valid = valid
                && finite(log_decay)
                && log_decay <= 0.0_f32
                && finite(beta)
                && beta >= 0.0_f32
                && beta <= 1.0_f32;
            status[status_base + token * status_stride + lane] = 0;
            output[out_base + token * out_stride + lane] = 0.0_f32;
            let mut predicted = 0.0_f32;
            for i in 0..head_dim {
                let decayed = column[i] * decay;
                predicted += decayed * key[ki + i];
            }
            let delta = (value[v_base + token * v_stride + lane] - predicted) * beta;
            valid = valid && finite(predicted) && finite(delta);
            if valid {
                let mut readout = 0.0_f32;
                for i in 0..head_dim {
                    let decayed = column[i] * decay;
                    let updated = decayed + key[ki + i] * delta;
                    valid = valid && finite(updated);
                    column[i] = updated;
                    readout += updated * query[qi + i];
                }
                valid = valid && finite(readout);
                if valid {
                    output[out_base + token * out_stride + lane] = readout;
                    status[status_base + token * status_stride + lane] = 1;
                }
            }
        }
        for i in 0..head_dim {
            let index = next_base + h * next_head + i * next_key + j * next_column;
            if valid {
                next[index] = column[i];
            } else {
                next[index] = 0.0_f32;
            }
        }
        if !valid {
            // Transactional chunk: invalidate earlier token candidates too.
            for token in 0..count {
                status[status_base + token * status_stride + lane] = 0;
                output[out_base + token * out_stride + lane] = 0.0_f32;
            }
        }
    }
}

/// Public count1 candidate: two direct immutable-old passes retain the original
/// per-column f32 arithmetic, without a head_dim-sized private lane array.
/// The host rejects count!=1 before launch. Runtime bases/strides and complete
/// MAX8 owning handles are identical to the ordinary chunk kernel.
#[allow(clippy::manual_range_contains)]
#[cube(launch)]
pub fn gdn_single_streaming(
    query: &Array<f32>,
    key: &Array<f32>,
    value: &Array<f32>,
    controls: &Array<f32>,
    old: &Array<f32>,
    next: &mut Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    q_base: usize,
    q_stride: usize,
    k_base: usize,
    k_stride: usize,
    v_base: usize,
    v_stride: usize,
    c_base: usize,
    c_stride: usize,
    old_base: usize,
    old_head: usize,
    old_key: usize,
    old_column: usize,
    next_base: usize,
    next_head: usize,
    next_key: usize,
    next_column: usize,
    out_base: usize,
    out_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] head_dim: usize,
    #[comptime] key_heads: usize,
    #[comptime] value_heads: usize,
) {
    let lane = ABSOLUTE_POS;
    if lane < value_heads * head_dim {
        let h = lane / head_dim;
        let j = lane % head_dim;
        let kh = h % key_heads;
        for token in 0..count {
            let qi = q_base + token * q_stride + kh * head_dim;
            let ki = k_base + token * k_stride + kh * head_dim;
            let ci = c_base + token * c_stride;
            let log_decay = controls[ci + h];
            let beta = controls[ci + value_heads + h];
            let decay = log_decay.exp();
            let mut valid = count == 1
                && finite(log_decay)
                && log_decay <= 0.0_f32
                && finite(beta)
                && beta >= 0.0_f32
                && beta <= 1.0_f32;
            status[status_base + token * status_stride + lane] = 0;
            output[out_base + token * out_stride + lane] = 0.0_f32;
            let mut predicted = 0.0_f32;
            for i in 0..head_dim {
                let old_value = old[old_base + h * old_head + i * old_key + j * old_column];
                valid = valid && finite(old_value);
                next[next_base + h * next_head + i * next_key + j * next_column] = 0.0_f32;
                let decayed = old_value * decay;
                predicted += decayed * key[ki + i];
            }
            let delta = (value[v_base + token * v_stride + lane] - predicted) * beta;
            valid = valid && finite(predicted) && finite(delta);
            if valid {
                let mut readout = 0.0_f32;
                for i in 0..head_dim {
                    let decayed =
                        old[old_base + h * old_head + i * old_key + j * old_column] * decay;
                    let updated = decayed + key[ki + i] * delta;
                    valid = valid && finite(updated);
                    next[next_base + h * next_head + i * next_key + j * next_column] = updated;
                    readout += updated * query[qi + i];
                }
                valid = valid && finite(readout);
                if valid {
                    output[out_base + token * out_stride + lane] = readout;
                    status[status_base + token * status_stride + lane] = 1;
                } else {
                    for i in 0..head_dim {
                        next[next_base + h * next_head + i * next_key + j * next_column] = 0.0_f32;
                    }
                }
            }
        }
    }
}

/// Explicit fixed-public count1..3 candidate. Each lane owns one state column.
/// Complete prediction precedes in-place update; later tokens consume that
/// lane's candidate column without a head_dim-sized private array. All address
/// spans/count remain runtime values, and old state is never written.
#[allow(clippy::manual_range_contains)]
#[cube(launch, launch_unchecked)]
pub fn gdn_three_streaming(
    query: &Array<f32>,
    key: &Array<f32>,
    value: &Array<f32>,
    controls: &Array<f32>,
    old: &Array<f32>,
    next: &mut Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    q_base: usize,
    q_stride: usize,
    k_base: usize,
    k_stride: usize,
    v_base: usize,
    v_stride: usize,
    c_base: usize,
    c_stride: usize,
    old_base: usize,
    old_head: usize,
    old_key: usize,
    old_column: usize,
    next_base: usize,
    next_head: usize,
    next_key: usize,
    next_column: usize,
    out_base: usize,
    out_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] head_dim: usize,
    #[comptime] key_heads: usize,
    #[comptime] value_heads: usize,
) {
    let lane = ABSOLUTE_POS;
    // Public policy admission repeats the bound before any unchecked access.
    if count >= 1 && count <= 3 && lane < value_heads * head_dim {
        let h = lane / head_dim;
        let j = lane % head_dim;
        let kh = h % key_heads;
        let mut valid = true;
        let mut next_written = false;
        for token in 0..count {
            let qi = q_base + token * q_stride + kh * head_dim;
            let ki = k_base + token * k_stride + kh * head_dim;
            let ci = c_base + token * c_stride;
            let log_decay = controls[ci + h];
            let beta = controls[ci + value_heads + h];
            let decay = log_decay.exp();
            valid = valid
                && finite(log_decay)
                && log_decay <= 0.0_f32
                && finite(beta)
                && beta >= 0.0_f32
                && beta <= 1.0_f32;
            status[status_base + token * status_stride + lane] = 0;
            output[out_base + token * out_stride + lane] = 0.0_f32;
            let mut predicted = 0.0_f32;
            for i in 0..head_dim {
                let previous = if token == 0 || !next_written {
                    old[old_base + h * old_head + i * old_key + j * old_column]
                } else {
                    next[next_base + h * next_head + i * next_key + j * next_column]
                };
                valid = valid && finite(previous);
                let decayed = previous * decay;
                predicted += decayed * key[ki + i];
            }
            let delta = (value[v_base + token * v_stride + lane] - predicted) * beta;
            valid = valid && finite(predicted) && finite(delta);
            if valid {
                let mut readout = 0.0_f32;
                for i in 0..head_dim {
                    let previous = if token == 0 || !next_written {
                        old[old_base + h * old_head + i * old_key + j * old_column]
                    } else {
                        next[next_base + h * next_head + i * next_key + j * next_column]
                    };
                    let decayed = previous * decay;
                    let updated = decayed + key[ki + i] * delta;
                    valid = valid && finite(updated);
                    next[next_base + h * next_head + i * next_key + j * next_column] = updated;
                    readout += updated * query[qi + i];
                }
                // The complete column is now initialized, even if a late
                // update/readout failed. Keep original invalid-column arithmetic
                // until the final whole-column erasure below.
                next_written = true;
                valid = valid && finite(readout);
                if valid {
                    output[out_base + token * out_stride + lane] = readout;
                    status[status_base + token * status_stride + lane] = 1;
                }
            }
        }
        if !valid {
            for i in 0..head_dim {
                next[next_base + h * next_head + i * next_key + j * next_column] = 0.0_f32;
            }
            // A later failed token invalidates every earlier candidate row.
            for token in 0..count {
                status[status_base + token * status_stride + lane] = 0;
                output[out_base + token * out_stride + lane] = 0.0_f32;
            }
        }
    }
}
#[allow(clippy::manual_range_contains)] // Cube expansion retains original scalar comparisons.
#[cube(launch, launch_unchecked)]
pub fn gdn_four_streaming(
    query: &Array<f32>,
    key: &Array<f32>,
    value: &Array<f32>,
    controls: &Array<f32>,
    old: &Array<f32>,
    next: &mut Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    q_base: usize,
    q_stride: usize,
    k_base: usize,
    k_stride: usize,
    v_base: usize,
    v_stride: usize,
    c_base: usize,
    c_stride: usize,
    old_base: usize,
    old_head: usize,
    old_key: usize,
    old_column: usize,
    next_base: usize,
    next_head: usize,
    next_key: usize,
    next_column: usize,
    out_base: usize,
    out_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] head_dim: usize,
    #[comptime] key_heads: usize,
    #[comptime] value_heads: usize,
) {
    let lane = ABSOLUTE_POS;
    // Public policy admission repeats the bound before any unchecked access.
    if count >= 1 && count <= 8 && lane < value_heads * head_dim {
        let h = lane / head_dim;
        let j = lane % head_dim;
        let kh = h % key_heads;
        let mut valid = true;
        let mut next_written = false;
        for token in 0..count {
            let qi = q_base + token * q_stride + kh * head_dim;
            let ki = k_base + token * k_stride + kh * head_dim;
            let ci = c_base + token * c_stride;
            let log_decay = controls[ci + h];
            let beta = controls[ci + value_heads + h];
            let decay = log_decay.exp();
            valid = valid
                && finite(log_decay)
                && log_decay <= 0.0_f32
                && finite(beta)
                && beta >= 0.0_f32
                && beta <= 1.0_f32;
            status[status_base + token * status_stride + lane] = 0;
            output[out_base + token * out_stride + lane] = 0.0_f32;
            let mut predicted = 0.0_f32;
            for i in 0..head_dim {
                let previous = if token == 0 || !next_written {
                    old[old_base + h * old_head + i * old_key + j * old_column]
                } else {
                    next[next_base + h * next_head + i * next_key + j * next_column]
                };
                valid = valid && finite(previous);
                let decayed = previous * decay;
                predicted += decayed * key[ki + i];
            }
            let delta = (value[v_base + token * v_stride + lane] - predicted) * beta;
            valid = valid && finite(predicted) && finite(delta);
            if valid {
                let mut readout = 0.0_f32;
                for i in 0..head_dim {
                    let previous = if token == 0 || !next_written {
                        old[old_base + h * old_head + i * old_key + j * old_column]
                    } else {
                        next[next_base + h * next_head + i * next_key + j * next_column]
                    };
                    let decayed = previous * decay;
                    let updated = decayed + key[ki + i] * delta;
                    valid = valid && finite(updated);
                    next[next_base + h * next_head + i * next_key + j * next_column] = updated;
                    readout += updated * query[qi + i];
                }
                // The complete column is now initialized, even if a late
                // update/readout failed. Keep original invalid-column arithmetic
                // until the final whole-column erasure below.
                next_written = true;
                valid = valid && finite(readout);
                if valid {
                    output[out_base + token * out_stride + lane] = readout;
                    status[status_base + token * status_stride + lane] = 1;
                }
            }
        }
        if !valid {
            for i in 0..head_dim {
                next[next_base + h * next_head + i * next_key + j * next_column] = 0.0_f32;
            }
            // A later failed token invalidates every earlier candidate row.
            for token in 0..count {
                status[status_base + token * status_stride + lane] = 0;
                output[out_base + token * out_stride + lane] = 0.0_f32;
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn streaming_lane_reference(
    shape: GdnShape,
    spans: ChunkSpans,
    inputs: [&[f32]; 5],
) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    let l = GdnLayout::new(shape).map_err(|_| fault())?;
    spans.validate(l)?;
    RecurrencePolicy::SingleStreaming.validate_count(spans.count)?;
    if inputs.iter().zip(spans.words()).any(|(s, n)| s.len() != n) {
        return Err(fault());
    }
    let [q, k, v, c, old] = inputs;
    let mut next = vec![
        0.;
        BufferLayout::new(spans.next.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    next[..spans.next.words].fill(-173.25);
    let mut out = vec![
        0.;
        BufferLayout::new(spans.output.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    out[..spans.output.words].fill(-173.25);
    let mut status = vec![
        0;
        BufferLayout::new(spans.status.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    status[..spans.status.words].fill(0x53aa_9267);
    for lane in 0..l.value_elements {
        let h = lane / shape.head_dim;
        let j = lane % shape.head_dim;
        let kh = h % shape.key_heads;
        let qi = spans.query.base + kh * shape.head_dim;
        let ki = spans.key.base + kh * shape.head_dim;
        let log = c[spans.controls.base + h];
        let beta = c[spans.controls.base + shape.value_heads + h];
        let decay = log.exp();
        let mut valid =
            log.is_finite() && log <= 0. && beta.is_finite() && (0.0..=1.0).contains(&beta);
        status[spans.status.base + lane] = 0;
        out[spans.output.base + lane] = 0.;
        let mut predicted = 0_f32;
        for i in 0..shape.head_dim {
            let old_value = old[spans.old.at(h, i, j)];
            valid = valid && old_value.is_finite();
            next[spans.next.at(h, i, j)] = 0.;
            let decayed = old_value * decay;
            predicted += decayed * k[ki + i];
        }
        let delta = (v[spans.value.base + lane] - predicted) * beta;
        valid = valid && predicted.is_finite() && delta.is_finite();
        if valid {
            let mut readout = 0_f32;
            for i in 0..shape.head_dim {
                let decayed = old[spans.old.at(h, i, j)] * decay;
                let updated = decayed + k[ki + i] * delta;
                valid = valid && updated.is_finite();
                next[spans.next.at(h, i, j)] = updated;
                readout += updated * q[qi + i];
            }
            valid = valid && readout.is_finite();
            if valid {
                out[spans.output.base + lane] = readout;
                status[spans.status.base + lane] = 1;
            } else {
                for i in 0..shape.head_dim {
                    next[spans.next.at(h, i, j)] = 0.;
                }
            }
        }
    }
    Ok((next, out, status))
}

#[cfg(test)]
pub(crate) fn three_streaming_lane_reference(
    shape: GdnShape,
    spans: ChunkSpans,
    inputs: [&[f32]; 5],
) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    let l = GdnLayout::new(shape).map_err(|_| fault())?;
    spans.validate(l)?;
    RecurrencePolicy::ThreeStreaming.validate_count(spans.count)?;
    if inputs.iter().zip(spans.words()).any(|(s, n)| s.len() != n) {
        return Err(fault());
    }
    let [q, k, v, c, old] = inputs;
    let mut next = vec![
        0.;
        BufferLayout::new(spans.next.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    next[..spans.next.words].fill(-173.25);
    let mut out = vec![
        0.;
        BufferLayout::new(spans.output.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    out[..spans.output.words].fill(-173.25);
    let mut status = vec![
        0;
        BufferLayout::new(spans.status.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    status[..spans.status.words].fill(0x53aa_9267);
    for lane in 0..l.value_elements {
        let h = lane / shape.head_dim;
        let j = lane % shape.head_dim;
        let kh = h % shape.key_heads;
        let mut valid = true;
        let mut next_written = false;
        for token in 0..spans.count {
            let qi = spans.query.at(token) + kh * shape.head_dim;
            let ki = spans.key.at(token) + kh * shape.head_dim;
            let ci = spans.controls.at(token);
            let log = c[ci + h];
            let beta = c[ci + shape.value_heads + h];
            let decay = log.exp();
            valid = valid
                && log.is_finite()
                && log <= 0.
                && beta.is_finite()
                && (0.0..=1.0).contains(&beta);
            status[spans.status.at(token) + lane] = 0;
            out[spans.output.at(token) + lane] = 0.;
            let mut predicted = 0_f32;
            for i in 0..shape.head_dim {
                let previous = if token == 0 || !next_written {
                    old[spans.old.at(h, i, j)]
                } else {
                    next[spans.next.at(h, i, j)]
                };
                valid = valid && previous.is_finite();
                let decayed = previous * decay;
                predicted += decayed * k[ki + i];
            }
            let delta = (v[spans.value.at(token) + lane] - predicted) * beta;
            valid = valid && predicted.is_finite() && delta.is_finite();
            if valid {
                let mut readout = 0_f32;
                for i in 0..shape.head_dim {
                    let previous = if token == 0 || !next_written {
                        old[spans.old.at(h, i, j)]
                    } else {
                        next[spans.next.at(h, i, j)]
                    };
                    let decayed = previous * decay;
                    let updated = decayed + k[ki + i] * delta;
                    valid = valid && updated.is_finite();
                    next[spans.next.at(h, i, j)] = updated;
                    readout += updated * q[qi + i];
                }
                next_written = true;
                valid = valid && readout.is_finite();
                if valid {
                    out[spans.output.at(token) + lane] = readout;
                    status[spans.status.at(token) + lane] = 1;
                }
            }
        }
        if !valid {
            for i in 0..shape.head_dim {
                next[spans.next.at(h, i, j)] = 0.;
            }
            for token in 0..spans.count {
                status[spans.status.at(token) + lane] = 0;
                out[spans.output.at(token) + lane] = 0.;
            }
        }
    }
    Ok((next, out, status))
}

#[cfg(test)]
pub(crate) fn four_streaming_lane_reference(
    shape: GdnShape,
    spans: ChunkSpans,
    inputs: [&[f32]; 5],
) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    let l = GdnLayout::new(shape).map_err(|_| fault())?;
    spans.validate(l)?;
    RecurrencePolicy::FourStreaming.validate_count(spans.count)?;
    if spans.count != 4 {
        return Err(fault());
    }
    if inputs.iter().zip(spans.words()).any(|(s, n)| s.len() != n) {
        return Err(fault());
    }
    let [q, k, v, c, old] = inputs;
    let mut next = vec![
        0.;
        BufferLayout::new(spans.next.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    next[..spans.next.words].fill(-173.25);
    let mut out = vec![
        0.;
        BufferLayout::new(spans.output.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    out[..spans.output.words].fill(-173.25);
    let mut status = vec![
        0;
        BufferLayout::new(spans.status.words)
            .map_err(|_| fault())?
            .owned_words
    ];
    status[..spans.status.words].fill(0x53aa_9267);
    for lane in 0..l.value_elements {
        let h = lane / shape.head_dim;
        let j = lane % shape.head_dim;
        let kh = h % shape.key_heads;
        let mut valid = true;
        let mut next_written = false;
        for token in 0..spans.count {
            let qi = spans.query.at(token) + kh * shape.head_dim;
            let ki = spans.key.at(token) + kh * shape.head_dim;
            let ci = spans.controls.at(token);
            let log = c[ci + h];
            let beta = c[ci + shape.value_heads + h];
            let decay = log.exp();
            valid = valid
                && log.is_finite()
                && log <= 0.
                && beta.is_finite()
                && (0.0..=1.0).contains(&beta);
            status[spans.status.at(token) + lane] = 0;
            out[spans.output.at(token) + lane] = 0.;
            let mut predicted = 0_f32;
            for i in 0..shape.head_dim {
                let previous = if token == 0 || !next_written {
                    old[spans.old.at(h, i, j)]
                } else {
                    next[spans.next.at(h, i, j)]
                };
                valid = valid && previous.is_finite();
                let decayed = previous * decay;
                predicted += decayed * k[ki + i];
            }
            let delta = (v[spans.value.at(token) + lane] - predicted) * beta;
            valid = valid && predicted.is_finite() && delta.is_finite();
            if valid {
                let mut readout = 0_f32;
                for i in 0..shape.head_dim {
                    let previous = if token == 0 || !next_written {
                        old[spans.old.at(h, i, j)]
                    } else {
                        next[spans.next.at(h, i, j)]
                    };
                    let decayed = previous * decay;
                    let updated = decayed + k[ki + i] * delta;
                    valid = valid && updated.is_finite();
                    next[spans.next.at(h, i, j)] = updated;
                    readout += updated * q[qi + i];
                }
                next_written = true;
                valid = valid && readout.is_finite();
                if valid {
                    out[spans.output.at(token) + lane] = readout;
                    status[spans.status.at(token) + lane] = 1;
                }
            }
        }
        if !valid {
            for i in 0..shape.head_dim {
                next[spans.next.at(h, i, j)] = 0.;
            }
            for token in 0..spans.count {
                status[spans.status.at(token) + lane] = 0;
                out[spans.output.at(token) + lane] = 0.;
            }
        }
    }
    Ok((next, out, status))
}
