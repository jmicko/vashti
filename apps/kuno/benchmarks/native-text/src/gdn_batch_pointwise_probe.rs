//! Fixed-public GDN pointwise chunk acceptance; no graph/request/state commit.
use crate::{
    gdn_batch_pointwise as b, gdn_pointwise as scalar,
    private_arena::{BufferId, BufferLayout, PrivateArena},
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::ops::{self, CausalConvState, GdnShape};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAIN_CAP: usize = 16 * 1024 * 1024;
const IO_CAP: u64 = 64 * 1024 * 1024;
const PUBLIC_CAP: usize = 4 * 1024 * 1024;
const GAP: f32 = -173.25;
const FLAG_GAP: u32 = 0x53aa9267;
const RAW: usize = 0;
const ALPHA: usize = 1;
const BETA: usize = 2;
const RECURRENT: usize = 3;
const Z: usize = 4;
const OLD: usize = 5;
const NEXT: usize = 6;
const ACT: usize = 7;
const QUERY: usize = 8;
const KEY: usize = 9;
const VALUE: usize = 10;
const CONTROL: usize = 11;
const GATED: usize = 12;
const SCONV: usize = 13;
const SQKV: usize = 14;
const SCONTROL: usize = 15;
const SGATE: usize = 16;
const FLAT: usize = 17;
const N: usize = 18;
#[derive(Clone, Copy)]
struct Spans {
    token: [b::Span; N],
    old: b::History,
    next: b::History,
}
fn widths(l: b::Layout) -> [usize; N] {
    let h = l.shape.value_heads;
    [
        l.channels,
        h,
        h,
        l.values,
        l.values,
        3 * l.channels,
        3 * l.channels,
        l.channels,
        l.qk,
        l.qk,
        l.values,
        2 * h,
        l.values,
        l.channels,
        l.channels,
        h,
        l.values,
        l.channels,
    ]
}
impl Spans {
    fn new(l: b::Layout, reference: bool) -> Result<Self> {
        if l != b::Layout::new(l.shape)? {
            return Err("pointwise mutated constructor layout".into());
        }
        let widths = widths(l);
        let mut v = Vec::new();
        for (i, w) in widths.into_iter().enumerate() {
            v.push(b::Span::new(
                w,
                if reference { 0 } else { 3 + i % 11 },
                if reference { 0 } else { 3 + i % 7 },
            )?);
        }
        let old = b::History::new(
            l.channels,
            if reference { 0 } else { 7 },
            if reference { 3 } else { 5 },
        )?;
        let next = b::History::new(
            l.channels,
            if reference { 0 } else { 11 },
            if reference { 3 } else { 7 },
        )?;
        Ok(Self {
            token: v.try_into().map_err(|_| "pointwise span count")?,
            old,
            next,
        })
    }
    fn words(self, i: usize) -> usize {
        if i == OLD {
            self.old.words
        } else if i == NEXT {
            self.next.words
        } else {
            self.token[i].words
        }
    }
    fn active(self, i: usize, count: usize, index: usize) -> bool {
        if i == OLD {
            self.old.active(index)
        } else if i == NEXT {
            self.next.active(index)
        } else {
            self.token[i].active(count, index)
        }
    }
    fn validate(self, count: usize) -> Result<()> {
        for i in 0..N {
            if i == OLD {
                self.old.validate(self.old.words)?;
            } else if i == NEXT {
                self.next.validate(self.next.words)?;
            } else {
                self.token[i].validate(count, self.words(i))?;
            }
        }
        Ok(())
    }
    fn validate_layout(self, l: b::Layout, count: usize) -> Result<()> {
        if l != b::Layout::new(l.shape)?
            || self.old.channels != l.channels
            || self.next.channels != l.channels
        {
            return Err("pointwise mutated layout/history".into());
        }
        self.validate(count)?;
        for (i, width) in widths(l).into_iter().enumerate() {
            if i != OLD && i != NEXT && self.token[i].width != width {
                return Err("pointwise span width disagrees with layout".into());
            }
        }
        Ok(())
    }
    fn footprint(self) -> Result<usize> {
        (0..N).try_fold(0usize, |sum, i| {
            sum.checked_add(BufferLayout::new(self.words(i))?.owned_bytes)
                .ok_or_else(|| "pointwise page sum overflow".into())
        })
    }
}
// Each raw-Array INFO packet is <=512 bytes (tested with the actual
// packer), and the owned runtime charges both its uniform and upload page.
// A 64-dispatch window also covers all 36 arena erases before reset/close.
const INFO_WINDOW: usize = 64 * 2 * 512;
fn pages(s: Spans, indices: impl IntoIterator<Item = usize>) -> Result<usize> {
    indices.into_iter().try_fold(0usize, |sum, i| {
        sum.checked_add(BufferLayout::new(s.words(i))?.owned_bytes)
            .ok_or_else(|| "pointwise I/O sum overflow".into())
    })
}
fn probe_plan(l: b::Layout) -> Result<(usize, usize, usize)> {
    let batch = Spans::new(l, false)?;
    let scalar = Spans::new(l, true)?;
    let main = batch
        .footprint()?
        .checked_add(scalar.footprint()?)
        .ok_or("pointwise constructor overflow")?;
    // Conservative: count every constructor upload even though each ordinary
    // arena initialization already checkpoints it. Before the first explicit
    // audit: two complete validators, six result reads, both histories, and
    // five immutable input reads. Cancellation adds the second validator.
    let validator = pages(batch, STATUSES.into_iter().chain(OUTPUTS).chain([NEXT]))?;
    let additional = pages(
        batch,
        OUTPUTS
            .into_iter()
            .chain([NEXT, OLD, RAW, ALPHA, BETA, RECURRENT, Z]),
    )?;
    let first_io = main
        .checked_add(validator.checked_mul(2).ok_or("pointwise I/O overflow")?)
        .and_then(|n| n.checked_add(additional))
        .and_then(|n| n.checked_add(INFO_WINDOW))
        .ok_or("pointwise I/O overflow")?;
    // Per scalar token: five owning writes and every possible result/status
    // page, including FLAT plus separate Q/K/V as a conservative union. The
    // final token additionally rereads the batch's complete candidate history.
    let scalar_io = pages(
        scalar,
        [RAW, ALPHA, BETA, RECURRENT, Z]
            .into_iter()
            .chain(STATUSES)
            .chain(OUTPUTS)
            .chain([FLAT, NEXT]),
    )?
    .checked_add(pages(batch, [NEXT])?)
    .and_then(|n| n.checked_add(INFO_WINDOW))
    .ok_or("pointwise scalar I/O overflow")?;
    if main > MAIN_CAP || first_io as u64 > IO_CAP || scalar_io as u64 > IO_CAP {
        return Err("pointwise aggregate admission".into());
    }
    Ok((main, first_io, scalar_io))
}
struct Public {
    l: b::Layout,
    weights: Handle,
    params: Handle,
    bad_params: Handle,
    gamma: [Handle; 2],
    weights_cpu: Zeroizing<Vec<f32>>,
    params_cpu: Zeroizing<Vec<f32>>,
    gamma_cpu: [Zeroizing<Vec<f32>>; 2],
}
type PublicFixtureValues = (
    Zeroizing<Vec<f32>>,
    Zeroizing<Vec<f32>>,
    [Zeroizing<Vec<f32>>; 2],
);
fn public_values(l: b::Layout) -> PublicFixtureValues {
    let weights = Zeroizing::new(
        (0..4 * l.channels)
            .map(|i| [-0.125, 0.25, 0.5, -0.0625][i % 4] + (i / 4 % 7) as f32 / 128.)
            .collect(),
    );
    let h = l.shape.value_heads;
    let mut params = Zeroizing::new(vec![0.; 2 * h]);
    for i in 0..h {
        params[i] = (i % 5) as f32 / 8. - 0.25;
        params[h + i] = -0.25 - (i % 11) as f32 / 16.;
    }
    let gamma = std::array::from_fn(|which| {
        Zeroizing::new(
            (0..if which == 0 {
                l.shape.head_dim
            } else {
                l.values
            })
                .map(|i| 0.5 + (i % 17) as f32 / 32.)
                .collect(),
        )
    });
    (weights, params, gamma)
}
fn public_handle(client: &ComputeClient<WgpuRuntime>, values: &[f32]) -> Result<Handle> {
    let words = BufferLayout::new(values.len())?.owned_words;
    let mut padded = Zeroizing::new(vec![0.; words]);
    padded[..values.len()].copy_from_slice(values);
    Ok(client.create_from_slice(f32::as_bytes(&padded)))
}
fn public_plan() -> Result<usize> {
    let mut sum = 0usize;
    for shape in shapes()? {
        let l = b::Layout::new(shape)?;
        for words in [
            4 * l.channels,
            2 * shape.value_heads,
            2 * shape.value_heads,
            shape.head_dim,
            l.values,
        ] {
            sum = sum
                .checked_add(BufferLayout::new(words)?.owned_bytes)
                .ok_or("pointwise public page overflow")?;
        }
    }
    if sum > PUBLIC_CAP {
        return Err("pointwise public page admission".into());
    }
    Ok(sum)
}
fn public_load(client: &ComputeClient<WgpuRuntime>) -> Result<Vec<Public>> {
    public_plan()?;
    let mut out = Vec::new();
    for shape in shapes()? {
        let l = b::Layout::new(shape)?;
        let (weights_cpu, params_cpu, gamma_cpu) = public_values(l);
        let mut bad = Zeroizing::new(params_cpu.to_vec());
        bad[shape.value_heads] = 0.0;
        out.push(Public {
            l,
            weights: public_handle(client, &weights_cpu)?,
            params: public_handle(client, &params_cpu)?,
            bad_params: public_handle(client, &bad)?,
            gamma: [
                public_handle(client, &gamma_cpu[0])?,
                public_handle(client, &gamma_cpu[1])?,
            ],
            weights_cpu,
            params_cpu,
            gamma_cpu,
        });
    }
    Ok(out)
}
fn shapes() -> Result<[GdnShape; 3]> {
    Ok([
        GdnShape::new(7, 2, 6)?,
        GdnShape::new(128, 1, 3)?,
        GdnShape::qwen38(),
    ])
}
fn input(width: usize, seed: usize) -> Zeroizing<Vec<f32>> {
    Zeroizing::new(
        (0..b::MAX_BATCH * width)
            .map(|i| (((i * 7 + i / width * 11 + seed) % 37) as f32 - 18.) / 16.)
            .collect(),
    )
}
fn initial(l: b::Layout) -> Zeroizing<Vec<f32>> {
    Zeroizing::new(
        (0..3 * l.channels)
            .map(|i| ((i * 5 % 23) as f32 - 11.) / 32.)
            .collect(),
    )
}
struct Set {
    ids: [BufferId; N],
    s: Spans,
}
fn allocate(arena: &mut PrivateArena, p: &Public, reference: bool) -> Result<Set> {
    let s = Spans::new(p.l, reference)?;
    let mut ids = Vec::new();
    let old = initial(p.l);
    for i in 0..N {
        let id = if (SCONV..=SGATE).contains(&i) {
            arena.upload_u32(&Zeroizing::new(vec![FLAG_GAP; s.words(i)]))?
        } else {
            let mut v = Zeroizing::new(vec![GAP; s.words(i)]);
            if i <= Z {
                let span = s.token[i];
                let mut x = input(span.width, i * 3);
                if i == BETA || i == Z {
                    for (j, x) in x.iter_mut().enumerate() {
                        *x = [-80., -8., -1., 0., 1., 8., 80.][j % 7];
                    }
                }
                for t in 0..b::MAX_BATCH {
                    v[span.at(t)..][..span.width]
                        .copy_from_slice(&x[t * span.width..][..span.width]);
                }
            }
            if i == OLD {
                for c in 0..p.l.channels {
                    v[s.old.at(c)..][..3].copy_from_slice(&old[c * 3..][..3]);
                }
            }
            arena.upload_f32(&v)?
        };
        ids.push(id);
    }
    Ok(Set {
        ids: ids.try_into().map_err(|_| "pointwise buffer count")?,
        s,
    })
}
fn arg(arena: &PrivateArena, set: &Set, i: usize) -> Result<ArrayArg<WgpuRuntime>> {
    let layout = arena.layout(set.ids[i])?;
    if layout.logical_words != set.s.words(i) {
        return Err("pointwise full binding geometry".into());
    }
    Ok(unsafe { ArrayArg::from_raw_parts(arena.handle(set.ids[i])?.clone(), layout.owned_words) })
}
fn public_arg(handle: &Handle, words: usize) -> ArrayArg<WgpuRuntime> {
    unsafe { ArrayArg::from_raw_parts(handle.clone(), words) }
}
fn original_arg(
    arena: &PrivateArena,
    set: &Set,
    i: usize,
    width: usize,
) -> Result<ArrayArg<WgpuRuntime>> {
    if width > arena.layout(set.ids[i])?.logical_words
        || arena.handle(set.ids[i])?.offset_start.unwrap_or(0) != 0
    {
        return Err("pointwise scalar prefix extent".into());
    }
    Ok(unsafe { ArrayArg::from_raw_parts(arena.handle(set.ids[i])?.clone(), width) })
}
fn validate_bindings(arena: &PrivateArena, p: &Public, sets: &[&Set]) -> Result<()> {
    let client = arena.client()?;
    let mut handles = Vec::new();
    for set in sets {
        set.s.validate_layout(p.l, b::MAX_BATCH)?;
        for (i, id) in set.ids.iter().enumerate() {
            if arena.layout(*id)?.logical_words != set.s.words(i) {
                return Err("pointwise logical owner mismatch".into());
            }
            let h = arena.handle(*id)?.clone();
            if h.offset_start.unwrap_or(0) != 0 || h.offset_end.unwrap_or(0) != 0 {
                return Err("pointwise trimmed owner".into());
            }
            handles.push(h);
        }
    }
    let private = handles.len();
    handles.extend([
        p.weights.clone(),
        p.params.clone(),
        p.bad_params.clone(),
        p.gamma[0].clone(),
        p.gamma[1].clone(),
    ]);
    let resources = handles
        .iter()
        .map(|h| client.get_resource(h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| "pointwise physical binding")?;
    for (i, r) in resources.iter().enumerate() {
        let v = r.resource();
        let h = &handles[i];
        if h.stream != handles[0].stream {
            return Err("pointwise stream mismatch".into());
        }
        if i < private && (v.offset != 0 || v.size != h.size() || v.buffer.size() != h.size()) {
            return Err("pointwise physical full owner mismatch".into());
        }
        let end = v
            .offset
            .checked_add(v.size)
            .ok_or("pointwise physical extent overflow")?;
        if end > v.buffer.size() {
            return Err("pointwise resource outside backing".into());
        }
        for other in resources.iter().skip(i + 1) {
            let z = other.resource();
            let ze = z
                .offset
                .checked_add(z.size)
                .ok_or("pointwise physical extent overflow")?;
            if v.buffer == z.buffer && v.offset < ze && z.offset < end {
                return Err("pointwise physical alias".into());
            }
        }
    }
    Ok(())
}
fn enqueue(
    arena: &PrivateArena,
    p: &Public,
    set: &Set,
    count: usize,
    gamma: usize,
    bad_params: bool,
    cancel: &AtomicBool,
) -> Result<()> {
    set.s.validate_layout(p.l, count)?;
    if gamma > 1 || cancel.load(Ordering::Acquire) {
        return Err("pointwise cancelled or gamma geometry".into());
    }
    validate_bindings(arena, p, &[set])?;
    let c = arena.client()?;
    let s = set.s;
    let l = p.l;
    let span = |i| s.token[i];
    let dim = CubeDim::new_1d(64);
    let grid = |width: usize| CubeCount::Static(width.div_ceil(64) as u32, b::MAX_BATCH as u32, 1);
    b::conv_silu::launch::<WgpuRuntime>(
        c,
        CubeCount::Static(l.channels.div_ceil(64) as u32, 1, 1),
        dim,
        arg(arena, set, RAW)?,
        public_arg(&p.weights, 4 * l.channels),
        arg(arena, set, OLD)?,
        arg(arena, set, NEXT)?,
        arg(arena, set, ACT)?,
        arg(arena, set, SCONV)?,
        count,
        span(RAW).base,
        span(RAW).stride,
        s.old.base,
        s.old.stride,
        s.next.base,
        s.next.stride,
        span(ACT).base,
        span(ACT).stride,
        span(SCONV).base,
        span(SCONV).stride,
        l.channels,
    );
    b::prepare_qkv::launch::<WgpuRuntime>(
        c,
        CubeCount::Static(
            (2 * l.shape.key_heads + l.shape.value_heads) as u32,
            b::MAX_BATCH as u32,
            1,
        ),
        dim,
        arg(arena, set, ACT)?,
        arg(arena, set, QUERY)?,
        arg(arena, set, KEY)?,
        arg(arena, set, VALUE)?,
        arg(arena, set, SQKV)?,
        b::EPSILON,
        count,
        span(ACT).base,
        span(ACT).stride,
        span(QUERY).base,
        span(QUERY).stride,
        span(KEY).base,
        span(KEY).stride,
        span(VALUE).base,
        span(VALUE).stride,
        span(SQKV).base,
        span(SQKV).stride,
        l.shape.head_dim,
        l.shape.key_heads,
        l.shape.value_heads,
    );
    b::prepare_controls::launch::<WgpuRuntime>(
        c,
        grid(l.shape.value_heads),
        dim,
        arg(arena, set, ALPHA)?,
        arg(arena, set, BETA)?,
        public_arg(
            if bad_params { &p.bad_params } else { &p.params },
            2 * l.shape.value_heads,
        ),
        arg(arena, set, CONTROL)?,
        arg(arena, set, SCONTROL)?,
        count,
        span(ALPHA).base,
        span(ALPHA).stride,
        span(BETA).base,
        span(BETA).stride,
        span(CONTROL).base,
        span(CONTROL).stride,
        span(SCONTROL).base,
        span(SCONTROL).stride,
        l.shape.value_heads,
    );
    b::output_gate::launch::<WgpuRuntime>(
        c,
        CubeCount::Static(l.shape.value_heads as u32, b::MAX_BATCH as u32, 1),
        dim,
        arg(arena, set, RECURRENT)?,
        arg(arena, set, Z)?,
        public_arg(&p.gamma[gamma], p.gamma_cpu[gamma].len()),
        arg(arena, set, GATED)?,
        arg(arena, set, SGATE)?,
        b::EPSILON,
        count,
        span(RECURRENT).base,
        span(RECURRENT).stride,
        span(Z).base,
        span(Z).stride,
        span(GATED).base,
        span(GATED).stride,
        span(SGATE).base,
        span(SGATE).stride,
        l.shape.head_dim,
        l.shape.value_heads,
        gamma == 0,
    );
    Ok(())
}
struct ScrubBytes(cubecl::bytes::Bytes);
impl Drop for ScrubBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn read(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
    let layout = arena.layout(id)?;
    let bytes = ScrubBytes(
        arena
            .client()?
            .read_one(arena.handle(id)?.clone())
            .map_err(|_| "pointwise completed read")?,
    );
    if bytes.0.len() != layout.owned_bytes {
        return Err("pointwise full read extent".into());
    }
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(layout.owned_words)?;
    for x in bytes.0.chunks_exact(4) {
        out.push(u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
    }
    Ok(out)
}
fn audit(arena: &PrivateArena) -> Result<()> {
    let r = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())
        .map_err(|_| "pointwise checked I/O cleanup")?;
    let total = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("pointwise I/O audit overflow")?;
    if total == 0 || r.inspected_bytes != total || r.inspected_buffers == 0 {
        return Err("pointwise I/O audit missing complete spans".into());
    }
    Ok(())
}
fn write(arena: &PrivateArena, id: BufferId, logical: &[f32]) -> Result<()> {
    let layout = arena.layout(id)?;
    if logical.len() != layout.logical_words {
        return Err("pointwise private write full logical extent".into());
    }
    let mut bytes = Zeroizing::new(vec![0u8; layout.owned_bytes]);
    bytes[..logical.len() * 4].copy_from_slice(f32::as_bytes(logical));
    arena
        .client()?
        .write_private(arena.handle(id)?.clone(), std::mem::take(&mut *bytes))
        .map_err(|_| "pointwise owned write failed")?;
    Ok(())
}
fn compare(actual: f32, reference: f32) -> Result<f32> {
    let e = (actual - reference).abs();
    if !actual.is_finite()
        || !reference.is_finite()
        || !e.is_finite()
        || e > 1e-4 + 3e-4 * reference.abs()
    {
        return Err("public pointwise numerical parity".into());
    }
    Ok(e)
}
fn checked_span(s: Spans, i: usize, actual: &[u32], count: usize, expected: &[f32]) -> Result<f32> {
    s.validate(count)?;
    let words = s.words(i);
    if actual.len() != BufferLayout::new(words)?.owned_words {
        return Err("pointwise exact output page extent".into());
    }
    let width = if i == NEXT || i == OLD {
        3 * s.old.channels
    } else {
        s.token[i].width
    };
    let tokens = if i == NEXT || i == OLD { 1 } else { count };
    if expected.len() != width * tokens {
        return Err("pointwise reference extent".into());
    }
    let mut max = 0f32;
    for (index, &bits) in actual.iter().enumerate() {
        if s.active(i, count, index) {
            let offset = if i == NEXT || i == OLD {
                let h = if i == OLD { s.old } else { s.next };
                let r = index - h.base;
                (r / h.stride) * 3 + r % h.stride
            } else {
                let span = s.token[i];
                let r = index - span.base;
                (r / span.stride) * width + r % span.stride
            };
            max = max.max(compare(f32::from_bits(bits), expected[offset])?);
        } else {
            let want = if index < words { GAP.to_bits() } else { 0 };
            if bits != want {
                return Err("pointwise inactive/gap/page padding changed".into());
            }
        }
    }
    Ok(max)
}
fn checked(set: &Set, i: usize, actual: &[u32], count: usize, expected: &[f32]) -> Result<f32> {
    checked_span(set.s, i, actual, count, expected)
}
fn readonly_bits(s: Spans, i: usize, words: &[u32], count: usize, expected: &[f32]) -> Result<()> {
    checked_span(s, i, words, count, expected)?;
    for (index, &bits) in words.iter().enumerate() {
        if s.active(i, count, index) {
            let offset = if i == OLD || i == NEXT {
                let h = if i == OLD { s.old } else { s.next };
                let r = index - h.base;
                r / h.stride * 3 + r % h.stride
            } else {
                let span = s.token[i];
                let r = index - span.base;
                r / span.stride * span.width + r % span.stride
            };
            if bits != expected[offset].to_bits() {
                return Err("pointwise raw/read-only bits changed".into());
            }
        }
    }
    Ok(())
}
fn flags_span(s: Spans, i: usize, actual: &[u32], count: usize, success: bool) -> Result<bool> {
    s.validate(count)?;
    if actual.len() != BufferLayout::new(s.words(i))?.owned_words {
        return Err("pointwise exact status page extent".into());
    }
    let mut failure = false;
    for (index, &v) in actual.iter().enumerate() {
        if s.active(i, count, index) {
            if v > 1 || success && v != 1 {
                return Err("pointwise active finite/status failed".into());
            }
            failure |= v == 0;
        } else {
            let want = if index < s.words(i) { FLAG_GAP } else { 0 };
            if v != want {
                return Err("pointwise inactive status/padding changed".into());
            }
        }
    }
    Ok(failure)
}
fn flags(set: &Set, i: usize, actual: &[u32], count: usize, success: bool) -> Result<bool> {
    flags_span(set.s, i, actual, count, success)
}
struct Oracle {
    tokens: Vec<[Zeroizing<Vec<f32>>; 6]>,
    history: Zeroizing<Vec<f32>>,
}
fn oracle_values(
    l: b::Layout,
    weights: &[f32],
    params: &[f32],
    gamma: &[f32],
    count: usize,
) -> Result<Oracle> {
    if l != b::Layout::new(l.shape)? || !(1..=b::MAX_BATCH).contains(&count) {
        return Err("pointwise CPU layout/count".into());
    }
    let raw = input(l.channels, 0);
    let alpha = input(l.shape.value_heads, 3);
    let mut beta = input(l.shape.value_heads, 6);
    let recurrent = input(l.values, 9);
    let mut z = input(l.values, 12);
    for (j, v) in beta.iter_mut().enumerate() {
        *v = [-80., -8., -1., 0., 1., 8., 80.][j % 7];
    }
    for (j, v) in z.iter_mut().enumerate() {
        *v = [-80., -8., -1., 0., 1., 8., 80.][j % 7];
    }
    let initial = initial(l);
    let mut conv = CausalConvState::new(l.channels, 4, 1)?;
    for tap in 0..3 {
        let warm: Zeroizing<Vec<_>> =
            Zeroizing::new((0..l.channels).map(|c| initial[c * 3 + tap]).collect());
        conv.step(&warm, weights)?;
    }
    if conv.history() != *initial {
        return Err("pointwise original raw history order".into());
    }
    let mut tokens = Vec::new();
    let h = l.shape.value_heads;
    let w = l.shape.head_dim;
    for t in 0..count {
        let mut act = Zeroizing::new(conv.step(&raw[t * l.channels..][..l.channels], weights)?);
        for v in act.iter_mut() {
            *v = ops::silu(*v);
        }
        let mut q = Zeroizing::new(ops::l2_norm_heads(&act[..l.qk], w, b::EPSILON)?);
        for v in q.iter_mut() {
            *v /= (w as f32).sqrt();
        }
        let k = Zeroizing::new(ops::l2_norm_heads(&act[l.qk..2 * l.qk], w, b::EPSILON)?);
        let v = Zeroizing::new(act[2 * l.qk..].to_vec());
        let mut ctl = Zeroizing::new(ops::gdn_log_decay(
            &alpha[t * h..][..h],
            &params[..h],
            &params[h..],
        )?);
        ctl.extend(beta[t * h..][..h].iter().map(|&v| ops::sigmoid(v)));
        let gated = Zeroizing::new(ops::gdn_output_gate(
            &recurrent[t * l.values..][..l.values],
            &z[t * l.values..][..l.values],
            gamma,
            w,
            b::EPSILON,
        )?);
        tokens.push([act, q, k, v, ctl, gated]);
    }
    let history = Zeroizing::new(conv.history().to_vec());
    conv.reset();
    Ok(Oracle { tokens, history })
}
fn oracle(p: &Public, count: usize, gamma: usize) -> Result<Oracle> {
    oracle_values(
        p.l,
        &p.weights_cpu,
        &p.params_cpu,
        &p.gamma_cpu[gamma],
        count,
    )
}
fn expectation(o: &Oracle, stage: usize) -> Zeroizing<Vec<f32>> {
    let mut v = Zeroizing::new(Vec::new());
    for token in &o.tokens {
        v.extend_from_slice(&token[stage]);
    }
    v
}
const OUTPUTS: [usize; 6] = [ACT, QUERY, KEY, VALUE, CONTROL, GATED];
const STATUSES: [usize; 4] = [SCONV, SQKV, SCONTROL, SGATE];
fn candidate_padding(s: Spans, i: usize, actual: &[u32], count: usize) -> Result<()> {
    s.validate(count)?;
    if actual.len() != BufferLayout::new(s.words(i))?.owned_words {
        return Err("pointwise exact candidate page extent".into());
    }
    for (index, &bits) in actual.iter().enumerate() {
        if s.active(i, count, index) {
            if !f32::from_bits(bits).is_finite() {
                return Err("pointwise nonfinite candidate".into());
            }
        } else {
            let want = if index < s.words(i) { GAP.to_bits() } else { 0 };
            if bits != want {
                return Err("pointwise candidate inactive/gap/page padding changed".into());
            }
        }
    }
    Ok(())
}
fn validate_values(
    l: b::Layout,
    s: Spans,
    count: usize,
    outputs: &[&[u32]; 7],
    status: &[&[u32]; 4],
    cancel: &AtomicBool,
) -> Result<()> {
    s.validate_layout(l, count)?;
    if cancel.load(Ordering::Acquire) {
        return Err("pointwise post-fence cancellation".into());
    }
    for (j, i) in STATUSES.into_iter().enumerate() {
        flags_span(s, i, status[j], count, true)?;
    }
    for (j, i) in OUTPUTS.into_iter().chain([NEXT]).enumerate() {
        candidate_padding(s, i, outputs[j], count)?;
    }
    for t in 0..count {
        let span = s.token[CONTROL];
        for h in 0..l.shape.value_heads {
            let decay = f32::from_bits(outputs[4][span.at(t) + h]);
            let beta = f32::from_bits(outputs[4][span.at(t) + l.shape.value_heads + h]);
            if decay > 0.0 || !(0.0..=1.0).contains(&beta) {
                return Err("pointwise candidate control range".into());
            }
        }
    }
    Ok(())
}
fn validate_chunk(
    arena: &PrivateArena,
    p: &Public,
    set: &Set,
    count: usize,
    cancel: &AtomicBool,
) -> Result<()> {
    let status = STATUSES
        .into_iter()
        .map(|i| read(arena, set.ids[i]))
        .collect::<Result<Vec<_>>>()?;
    let output = OUTPUTS
        .into_iter()
        .chain([NEXT])
        .map(|i| read(arena, set.ids[i]))
        .collect::<Result<Vec<_>>>()?;
    validate_values(
        p.l,
        set.s,
        count,
        &std::array::from_fn(|i| output[i].as_slice()),
        &std::array::from_fn(|i| status[i].as_slice()),
        cancel,
    )
}
fn scalar_step(arena: &PrivateArena, p: &Public, set: &Set, t: usize, gamma: usize) -> Result<()> {
    let l = p.l;
    for i in [RAW, ALPHA, BETA, RECURRENT, Z] {
        let mut v = Zeroizing::new(vec![GAP; set.s.words(i)]);
        let mut all = input(set.s.token[i].width, i * 3);
        if i == BETA || i == Z {
            for (j, x) in all.iter_mut().enumerate() {
                *x = [-80., -8., -1., 0., 1., 8., 80.][j % 7];
            }
        }
        v[..set.s.token[i].width]
            .copy_from_slice(&all[t * set.s.token[i].width..][..set.s.token[i].width]);
        write(arena, set.ids[i], &v)?;
    }
    let c = arena.client()?;
    let dim = CubeDim::new_1d(64);
    scalar::conv_silu::launch::<WgpuRuntime>(
        c,
        CubeCount::Static(l.channels.div_ceil(64) as u32, 1, 1),
        dim,
        original_arg(arena, set, RAW, l.channels)?,
        public_arg(&p.weights, 4 * l.channels),
        original_arg(arena, set, OLD, 3 * l.channels)?,
        original_arg(arena, set, NEXT, 3 * l.channels)?,
        original_arg(arena, set, ACT, l.channels)?,
        original_arg(arena, set, SCONV, l.channels)?,
        l.channels,
    );
    if l.shape.head_dim == 128 {
        scalar::prepare_qkv::launch::<WgpuRuntime>(
            c,
            CubeCount::Static((2 * l.shape.key_heads + l.shape.value_heads) as u32, 1, 1),
            dim,
            original_arg(arena, set, ACT, l.channels)?,
            original_arg(arena, set, FLAT, l.channels)?,
            original_arg(arena, set, SQKV, l.channels)?,
            b::EPSILON,
            l.shape.key_heads,
            l.shape.value_heads,
        );
    } else {
        let s = set.s.token;
        b::prepare_qkv::launch::<WgpuRuntime>(
            c,
            CubeCount::Static(
                (2 * l.shape.key_heads + l.shape.value_heads) as u32,
                b::MAX_BATCH as u32,
                1,
            ),
            dim,
            arg(arena, set, ACT)?,
            arg(arena, set, QUERY)?,
            arg(arena, set, KEY)?,
            arg(arena, set, VALUE)?,
            arg(arena, set, SQKV)?,
            b::EPSILON,
            1,
            s[ACT].base,
            s[ACT].stride,
            s[QUERY].base,
            s[QUERY].stride,
            s[KEY].base,
            s[KEY].stride,
            s[VALUE].base,
            s[VALUE].stride,
            s[SQKV].base,
            s[SQKV].stride,
            l.shape.head_dim,
            l.shape.key_heads,
            l.shape.value_heads,
        );
    }
    scalar::prepare_controls::launch::<WgpuRuntime>(
        c,
        CubeCount::Static(l.shape.value_heads.div_ceil(64) as u32, 1, 1),
        dim,
        original_arg(arena, set, ALPHA, l.shape.value_heads)?,
        original_arg(arena, set, BETA, l.shape.value_heads)?,
        public_arg(&p.params, 2 * l.shape.value_heads),
        original_arg(arena, set, CONTROL, 2 * l.shape.value_heads)?,
        original_arg(arena, set, SCONTROL, l.shape.value_heads)?,
        l.shape.value_heads,
    );
    if l.shape.head_dim == 128 {
        scalar::output_gate::launch::<WgpuRuntime>(
            c,
            CubeCount::Static(l.shape.value_heads as u32, 1, 1),
            dim,
            original_arg(arena, set, RECURRENT, l.values)?,
            original_arg(arena, set, Z, l.values)?,
            public_arg(&p.gamma[gamma], p.gamma_cpu[gamma].len()),
            original_arg(arena, set, GATED, l.values)?,
            original_arg(arena, set, SGATE, l.values)?,
            b::EPSILON,
            l.shape.value_heads,
            gamma == 0,
        );
    } else {
        let s = set.s.token;
        b::output_gate::launch::<WgpuRuntime>(
            c,
            CubeCount::Static(l.shape.value_heads as u32, b::MAX_BATCH as u32, 1),
            dim,
            arg(arena, set, RECURRENT)?,
            arg(arena, set, Z)?,
            public_arg(&p.gamma[gamma], p.gamma_cpu[gamma].len()),
            arg(arena, set, GATED)?,
            arg(arena, set, SGATE)?,
            b::EPSILON,
            1,
            s[RECURRENT].base,
            s[RECURRENT].stride,
            s[Z].base,
            s[Z].stride,
            s[GATED].base,
            s[GATED].stride,
            s[SGATE].base,
            s[SGATE].stride,
            l.shape.head_dim,
            l.shape.value_heads,
            gamma == 0,
        );
    }
    Ok(())
}
fn case(
    arena: &mut PrivateArena,
    p: &Public,
    count: usize,
    gamma: usize,
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    let (planned, initial_io_bound, scalar_io_bound) = probe_plan(p.l)?;
    let batch = allocate(arena, p, false)?;
    let mut single = allocate(arena, p, true)?;
    if arena.charged_bytes() != planned {
        return Err("pointwise page charge mismatch".into());
    }
    validate_bindings(arena, p, &[&batch, &single])?;
    let cancel = AtomicBool::new(true);
    if enqueue(arena, p, &batch, count, gamma, false, &cancel).is_ok() {
        return Err("pointwise cancellation ignored".into());
    }
    cancel.store(false, Ordering::Release);
    let mut alias = Set {
        ids: batch.ids,
        s: batch.s,
    };
    alias.ids[NEXT] = alias.ids[OLD];
    if enqueue(arena, p, &alias, count, gamma, false, &cancel).is_ok() {
        return Err("pointwise immutable history alias accepted".into());
    }
    let expected = oracle(p, count, gamma)?;
    enqueue(arena, p, &batch, count, gamma, false, &cancel)?;
    validate_chunk(arena, p, &batch, count, &cancel)?;
    let mut cpu_error = 0f32;
    let mut gpu_error = 0f32;
    let mut actual = Vec::new();
    for (j, i) in OUTPUTS.into_iter().enumerate() {
        let words = read(arena, batch.ids[i])?;
        cpu_error = cpu_error.max(checked(
            &batch,
            i,
            &words,
            count,
            &expectation(&expected, j),
        )?);
        actual.push(words);
    }
    let next = read(arena, batch.ids[NEXT])?;
    cpu_error = cpu_error.max(checked(&batch, NEXT, &next, count, &expected.history)?);
    readonly_bits(batch.s, NEXT, &next, count, &expected.history)?;
    readonly_bits(
        batch.s,
        OLD,
        &read(arena, batch.ids[OLD])?,
        count,
        &initial(p.l),
    )?;
    for i in [RAW, ALPHA, BETA, RECURRENT, Z] {
        let mut x = input(batch.s.token[i].width, i * 3);
        if i == BETA || i == Z {
            for (j, v) in x.iter_mut().enumerate() {
                *v = [-80., -8., -1., 0., 1., 8., 80.][j % 7];
            }
        }
        readonly_bits(batch.s, i, &read(arena, batch.ids[i])?, b::MAX_BATCH, &x)?;
    }
    cancel.store(true, Ordering::Release);
    if validate_chunk(arena, p, &batch, count, &cancel).is_ok() {
        return Err("pointwise post-fence cancellation accepted".into());
    }
    cancel.store(false, Ordering::Release);
    audit(arena)?;
    for t in 0..count {
        scalar_step(arena, p, &single, t, gamma)?;
        for i in STATUSES {
            flags(&single, i, &read(arena, single.ids[i])?, 1, true)?;
        }
        for (j, i) in OUTPUTS.into_iter().enumerate() {
            if p.l.shape.head_dim == 128 && (i == QUERY || i == KEY || i == VALUE) {
                continue;
            }
            let words = read(arena, single.ids[i])?;
            gpu_error = gpu_error.max(checked(&single, i, &words, 1, &expected.tokens[t][j])?);
            let span = batch.s.token[i];
            for d in 0..span.width {
                gpu_error = gpu_error.max(compare(
                    f32::from_bits(words[d]),
                    f32::from_bits(actual[j][span.at(t) + d]),
                )?);
            }
        }
        if p.l.shape.head_dim == 128 {
            let flat = read(arena, single.ids[FLAT])?;
            let mut expected_flat = Zeroizing::new(Vec::new());
            for j in 1..=3 {
                expected_flat.extend_from_slice(&expected.tokens[t][j]);
            }
            gpu_error = gpu_error.max(checked(&single, FLAT, &flat, 1, &expected_flat)?);
            for (j, i) in [(1, QUERY), (2, KEY), (3, VALUE)] {
                let offset = if i == QUERY {
                    0
                } else if i == KEY {
                    p.l.qk
                } else {
                    2 * p.l.qk
                };
                for d in 0..batch.s.token[i].width {
                    gpu_error = gpu_error.max(compare(
                        f32::from_bits(flat[offset + d]),
                        f32::from_bits(actual[j][batch.s.token[i].at(t) + d]),
                    )?);
                }
            }
        }
        let candidate = read(arena, single.ids[NEXT])?;
        if t + 1 == count {
            gpu_error = gpu_error.max(checked(&single, NEXT, &candidate, 1, &expected.history)?);
            let final_batch = read(arena, batch.ids[NEXT])?;
            for c in 0..p.l.channels {
                for tap in 0..3 {
                    if candidate[c * 3 + tap] != final_batch[batch.s.next.at(c) + tap] {
                        return Err("pointwise final raw history bits differ".into());
                    }
                }
            }
        }
        audit(arena)?;
        single.ids.swap(OLD, NEXT);
    }
    arena.reset()?;
    if batch
        .ids
        .iter()
        .chain(&single.ids)
        .any(|&id| arena.handle(id).is_ok())
        || retired.load(Ordering::Acquire)
    {
        return Err("pointwise reset epoch/retirement failed".into());
    }
    Ok(
        serde_json::json!({"head_dim":p.l.shape.head_dim,"key_heads":p.l.shape.key_heads,"value_heads":p.l.shape.value_heads,"count":count,"shared_gamma":gamma==0,"max_cpu_abs_error":cpu_error,"max_single_gpu_abs_error":gpu_error,"unchanged_scalar_qkv_closing_supported":p.l.shape.head_dim==128,"raw_history_bit_exact":true,"all_active_columns_and_padding_checked":true,"main_owned_bytes":planned,"initial_io_upper_bound_bytes":initial_io_bound,"scalar_io_upper_bound_bytes":scalar_io_bound}),
    )
}
fn failure_case(
    arena: &mut PrivateArena,
    p: &Public,
    kind: usize,
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    let count = 8;
    probe_plan(p.l)?;
    let planned = Spans::new(p.l, false)?.footprint()?;
    if planned > MAIN_CAP {
        return Err("pointwise fault admission".into());
    }
    let set = allocate(arena, p, false)?;
    if arena.charged_bytes() != planned {
        return Err("pointwise fault page plan".into());
    }
    if kind < 3 {
        let i = [RAW, ALPHA, Z][kind];
        let mut values = Zeroizing::new(vec![GAP; set.s.words(i)]);
        let mut x = input(set.s.token[i].width, i * 3);
        if i == Z {
            for (j, v) in x.iter_mut().enumerate() {
                *v = [-80., -8., -1., 0., 1., 8., 80.][j % 7];
            }
        }
        for t in 0..8 {
            values[set.s.token[i].at(t)..][..set.s.token[i].width]
                .copy_from_slice(&x[t * set.s.token[i].width..][..set.s.token[i].width]);
        }
        values[set.s.token[i].at(7)] = f32::NAN;
        write(arena, set.ids[i], &values)?;
    }
    enqueue(arena, p, &set, count, 1, kind == 3, &AtomicBool::new(false))?;
    if validate_chunk(arena, p, &set, count, &AtomicBool::new(false)).is_ok() {
        return Err("pointwise invalid/nonfinite whole chunk accepted".into());
    }
    let mut rejected = false;
    for i in STATUSES {
        rejected |= flags(&set, i, &read(arena, set.ids[i])?, count, false)?;
    }
    if !rejected {
        return Err("pointwise failure has no failed status".into());
    }
    for i in OUTPUTS.into_iter().chain([NEXT]) {
        candidate_padding(set.s, i, &read(arena, set.ids[i])?, count)?;
    }
    readonly_bits(
        set.s,
        OLD,
        &read(arena, set.ids[OLD])?,
        count,
        &initial(p.l),
    )?;
    if kind == 0 {
        let status = read(arena, set.ids[SCONV])?;
        for t in 0..count {
            if status[set.s.token[SCONV].at(t)] != 0 {
                return Err("pointwise late raw failure retained earlier success".into());
            }
        }
        let next = read(arena, set.ids[NEXT])?;
        for tap in 0..3 {
            if next[set.s.next.at(0) + tap] != 0.0f32.to_bits() {
                return Err("pointwise failed raw candidate retained history".into());
            }
        }
    }
    audit(arena)?;
    arena.reset()?;
    if set.ids.iter().any(|&id| arena.handle(id).is_ok()) || retired.load(Ordering::Acquire) {
        return Err("pointwise failed candidate cleanup rejected".into());
    }
    Ok(
        serde_json::json!({"head_dim":p.l.shape.head_dim,"key_heads":p.l.shape.key_heads,"value_heads":p.l.shape.value_heads,"late_token_failure":kind<3,"kind":(["raw_nan","alpha_nan","closing_z_nan","stored_a_zero"][kind]),"whole_chunk_rejected":true,"committed_raw_history_immutable":true,"candidate_pages_erased":true}),
    )
}
fn accept<T>(attempt: std::thread::Result<Result<T>>, closed: bool, retired: bool) -> Result<T> {
    if !closed || retired {
        return Err("pointwise cleanup failed/context retired".into());
    }
    attempt.map_err(|_| "pointwise probe panicked")?
}
pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    for shape in shapes()? {
        probe_plan(b::Layout::new(shape)?)?;
    }
    let public = public_load(client)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena = PrivateArena::new_transient(client.clone(), MAIN_CAP, IO_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut correct = Vec::new();
        let mut failures = Vec::new();
        for p in &public {
            for count in [1, 3, 5, 8] {
                for gamma in [0, 1] {
                    correct.push(case(&mut arena, p, count, gamma, &retired)?);
                }
            }
            for kind in 0..4 {
                failures.push(failure_case(&mut arena, p, kind, &retired)?);
            }
        }
        Ok((correct, failures))
    }));
    let close = arena.close();
    let (correct, failures) = accept(attempt, close.is_ok(), retired.load(Ordering::Acquire))?;
    let receipt = arena
        .owned_zero_audit_receipt()
        .ok_or("pointwise owning zero audit absent")?;
    if receipt.buffers == 0 || receipt.bytes == 0 {
        return Err("pointwise empty owning cleanup audit".into());
    }
    let output = serde_json::json!({"scope":"fixed-public-pointwise-operator-only","fixture":"gdn-batch-pointwise-probe","case_count":correct.len(),"cases":correct,"failure_cases":failures,"main_cap_bytes":MAIN_CAP,"runtime_io_cap_bytes":IO_CAP,"public_page_bytes":public_plan()?,"main_zero_buffers":receipt.buffers,"main_zero_bytes":receipt.bytes,"context_retired":false,"whole_graph_or_batched_prefill_accepted":false,"private_register_or_driver_erasure_accepted":false});
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;
    }
    println!(
        "public_gdn_batch_pointwise_acceptance cases=24 failures=12 main_buffers={} main_bytes={} context_retired=false",
        receipt.buffers, receipt.bytes
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn values(s: Spans, i: usize, count: usize, expected: &[f32]) -> Vec<u32> {
        let mut words = vec![0; BufferLayout::new(s.words(i)).unwrap().owned_words];
        words[..s.words(i)].fill(GAP.to_bits());
        let width = if i == NEXT || i == OLD {
            3 * s.old.channels
        } else {
            s.token[i].width
        };
        for (index, v) in words.iter_mut().enumerate() {
            if s.active(i, count, index) {
                let offset = if i == NEXT || i == OLD {
                    let h = if i == OLD { s.old } else { s.next };
                    let r = index - h.base;
                    r / h.stride * 3 + r % h.stride
                } else {
                    let span = s.token[i];
                    let r = index - span.base;
                    r / span.stride * width + r % span.stride
                };
                *v = expected[offset].to_bits();
            }
        }
        words
    }
    fn flags_image(s: Spans, i: usize, count: usize) -> Vec<u32> {
        let mut words = vec![0; BufferLayout::new(s.words(i)).unwrap().owned_words];
        words[..s.words(i)].fill(FLAG_GAP);
        for (index, v) in words.iter_mut().enumerate() {
            if s.active(i, count, index) {
                *v = 1;
            }
        }
        words
    }
    fn fixtures(
        l: b::Layout,
        s: Spans,
        count: usize,
        gamma: usize,
    ) -> ([Vec<u32>; 7], [Vec<u32>; 4]) {
        let (w, p, g) = public_values(l);
        let o = oracle_values(l, &w, &p, &g[gamma], count).unwrap();
        let output = std::array::from_fn(|j| {
            if j == 6 {
                values(s, NEXT, count, &o.history)
            } else {
                values(s, OUTPUTS[j], count, &expectation(&o, j))
            }
        });
        let status = std::array::from_fn(|j| flags_image(s, STATUSES[j], count));
        (output, status)
    }
    fn validate(
        l: b::Layout,
        s: Spans,
        count: usize,
        out: &[Vec<u32>; 7],
        flags: &[Vec<u32>; 4],
        cancel: bool,
    ) -> Result<()> {
        validate_values(
            l,
            s,
            count,
            &std::array::from_fn(|i| out[i].as_slice()),
            &std::array::from_fn(|i| flags[i].as_slice()),
            &AtomicBool::new(cancel),
        )
    }
    #[test]
    fn full_max8_layout_pages_reject_mutation_overlap_overflow_and_partial_capacity() {
        for shape in shapes().unwrap() {
            let l = b::Layout::new(shape).unwrap();
            for reference in [false, true] {
                let s = Spans::new(l, reference).unwrap();
                for count in [1, 3, 5, 8] {
                    s.validate_layout(l, count).unwrap();
                    assert!(s.footprint().unwrap() < MAIN_CAP / 2);
                }
                assert!(s.validate_layout(l, 0).is_err());
                assert!(s.validate_layout(l, 9).is_err());
                let mut bad = s;
                bad.token[QUERY].width += 1;
                assert!(bad.validate_layout(l, 1).is_err());
                let mut bad = s;
                bad.token[KEY].stride = bad.token[KEY].width - 1;
                assert!(bad.validate_layout(l, 1).is_err());
                let mut bad = s;
                bad.old.stride = 2;
                assert!(bad.validate_layout(l, 1).is_err());
                let mut bad = s;
                bad.token[VALUE].words = bad.token[VALUE].base + bad.token[VALUE].width;
                assert!(bad.validate_layout(l, 1).is_err());
                let mut corrupt = l;
                corrupt.channels += 1;
                assert!(Spans::new(corrupt, reference).is_err());
                assert!(s.validate_layout(corrupt, 1).is_err());
            }
        }
        assert!(b::Span::new(usize::MAX, 1, 1).is_err());
        assert!(b::History::new(usize::MAX, 1, 3).is_err());
        assert!(b::Layout::new(GdnShape::new(129, 1, 1).unwrap()).is_err());
        assert!(public_plan().unwrap() < PUBLIC_CAP);
    }
    #[test]
    fn exact_owning_download_checks_every_active_stage_tail_padding_and_cancel() {
        for shape in shapes().unwrap() {
            let l = b::Layout::new(shape).unwrap();
            let s = Spans::new(l, false).unwrap();
            for count in [1, 3, 5, 8] {
                for gamma in [0, 1] {
                    let (out, status) = fixtures(l, s, count, gamma);
                    validate(l, s, count, &out, &status, false).unwrap();
                    assert!(validate(l, s, count, &out, &status, true).is_err());
                    for j in 0..7 {
                        let i = if j == 6 { NEXT } else { OUTPUTS[j] };
                        for surplus in [false, true] {
                            let mut wrong = out.clone();
                            if surplus {
                                wrong[j].push(0);
                            } else {
                                wrong[j].pop();
                            }
                            assert!(validate(l, s, count, &wrong, &status, false).is_err());
                        }
                        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                            let mut wrong = out.clone();
                            let at = if i == NEXT {
                                s.next.at(l.channels - 1) + 2
                            } else {
                                s.token[i].at(count - 1) + s.token[i].width - 1
                            };
                            wrong[j][at] = value.to_bits();
                            assert!(validate(l, s, count, &wrong, &status, false).is_err());
                        }
                        for at in [0, s.words(i) - 1, out[j].len() - 1] {
                            let mut wrong = out.clone();
                            wrong[j][at] ^= 1;
                            assert!(validate(l, s, count, &wrong, &status, false).is_err());
                        }
                    }
                    for j in 0..4 {
                        let mut wrong = status.clone();
                        wrong[j][s.token[STATUSES[j]].at(count - 1)] = 0;
                        assert!(validate(l, s, count, &out, &wrong, false).is_err());
                        wrong[j][s.token[STATUSES[j]].at(count - 1)] = 2;
                        assert!(validate(l, s, count, &out, &wrong, false).is_err());
                        let mut wrong = status.clone();
                        wrong[j].push(0);
                        assert!(validate(l, s, count, &out, &wrong, false).is_err());
                    }
                    let mut wrong = out.clone();
                    wrong[4][s.token[CONTROL].at(count - 1)] = 0.1f32.to_bits();
                    assert!(validate(l, s, count, &wrong, &status, false).is_err());
                    let mut wrong = out.clone();
                    wrong[4][s.token[CONTROL].at(0) + l.shape.value_heads] = 1.1f32.to_bits();
                    assert!(validate(l, s, count, &wrong, &status, false).is_err());
                }
            }
        }
    }
    fn logistic(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }
    fn squares(head: &[f32]) -> f32 {
        let mut s = [0f32; 64];
        for (lane, partial) in s.iter_mut().enumerate() {
            let x = *head.get(lane).unwrap_or(&0.);
            let y = *head.get(lane + 64).unwrap_or(&0.);
            *partial = x * x + y * y;
        }
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                s[lane] += s[lane + stride];
            }
            stride /= 2;
        }
        s[0]
    }
    #[test]
    fn independent_raw_history_lags_pair_tree_q_scale_and_mixed_gamma_match_core() {
        for shape in shapes().unwrap() {
            let l = b::Layout::new(shape).unwrap();
            let (w, p, g) = public_values(l);
            for count in [1, 3, 5, 8] {
                for gamma in [0, 1] {
                    let expected = oracle_values(l, &w, &p, &g[gamma], count).unwrap();
                    let mut history = initial(l);
                    let raw = input(l.channels, 0);
                    for t in 0..count {
                        let mut act = vec![0f32; l.channels];
                        for c in 0..l.channels {
                            let mut sum = 0f32;
                            for tap in 0..4 {
                                let x = if tap == 3 {
                                    raw[t * l.channels + c]
                                } else {
                                    history[c * 3 + tap]
                                };
                                sum += x * w[c * 4 + tap];
                            }
                            act[c] = sum * logistic(sum);
                            history[c * 3] = history[c * 3 + 1];
                            history[c * 3 + 1] = history[c * 3 + 2];
                            history[c * 3 + 2] = raw[t * l.channels + c];
                            compare(act[c], expected.tokens[t][0][c]).unwrap();
                        }
                        for head in 0..2 * shape.key_heads {
                            let v = &act[head * shape.head_dim..][..shape.head_dim];
                            let inverse = 1.0 / (squares(v) + b::EPSILON).sqrt();
                            for (d, &x) in v.iter().enumerate() {
                                let mut y = x * inverse;
                                if head < shape.key_heads {
                                    y /= (shape.head_dim as f32).sqrt();
                                }
                                let j = if head < shape.key_heads { 1 } else { 2 };
                                let index = (head % shape.key_heads) * shape.head_dim + d;
                                compare(y, expected.tokens[t][j][index]).unwrap();
                            }
                        }
                        let rec = input(l.values, 9);
                        for h in 0..shape.value_heads {
                            let head = &rec[t * l.values + h * shape.head_dim..][..shape.head_dim];
                            let inv =
                                1.0 / (squares(head) / shape.head_dim as f32 + b::EPSILON).sqrt();
                            for (d, &x) in head.iter().enumerate() {
                                let i = h * shape.head_dim + d;
                                let z = [-80., -8., -1., 0., 1., 8., 80.][(t * l.values + i) % 7];
                                let weight = g[gamma][if gamma == 0 { d } else { i }];
                                compare(x * inv * weight * logistic(z), expected.tokens[t][5][i])
                                    .unwrap();
                            }
                        }
                    }
                    assert_eq!(&*history, &*expected.history);
                    assert_ne!(&*history, &*initial(l));
                }
            }
        }
    }
    #[test]
    fn cleanup_failure_retirement_dispatch_error_and_panic_never_accept() {
        assert_eq!(accept(Ok(Ok(3)), true, false).unwrap(), 3);
        for (closed, retired) in [(false, false), (true, true), (false, true)] {
            assert!(accept(Ok(Ok(3)), closed, retired).is_err());
        }
        assert!(accept::<usize>(Ok(Err("read/status failed".into())), true, false).is_err());
        assert!(accept::<usize>(Err(Box::new("panic")), true, false).is_err());
    }
    #[test]
    fn full_page_and_uniform_inventory_fits_before_any_allocation() {
        let expected_main = [1_179_648, 1_179_648, 9_699_328];
        for (shape, expected) in shapes().unwrap().into_iter().zip(expected_main) {
            let l = b::Layout::new(shape).unwrap();
            let (main, first_io, scalar_io) = probe_plan(l).unwrap();
            assert_eq!(main, expected);
            assert!(first_io < 20 * 1024 * 1024);
            assert!(scalar_io < 6 * 1024 * 1024);
            assert!(main <= MAIN_CAP);
            assert!((first_io as u64).max(scalar_io as u64) < IO_CAP);
        }
        // All four pointwise and unchanged scalar signatures have <=6 raw
        // Array bindings and <=13 runtime scalar values. Use wider addressing
        // and scalar storage than the current native u32 kernels require.
        for address in [cubecl::ir::AddressType::U32, cubecl::ir::AddressType::U64] {
            let mut info = cubecl::InfoBuilder::default();
            for _ in 0..6 {
                info.metadata.register_array(131_072, 131_072, address);
            }
            for _ in 0..13 {
                info.scalars.push(1_u64);
            }
            let bytes = info.finish(address).data.len() * 8;
            assert!(bytes <= 512);
            assert!(64 * 2 * bytes.max(256).next_power_of_two() <= INFO_WINDOW);
        }
    }
    #[test]
    fn immutable_raw_bits_reject_even_one_ulp_changes_inside_numeric_tolerance() {
        let l = b::Layout::new(GdnShape::new(7, 2, 6).unwrap()).unwrap();
        let s = Spans::new(l, false).unwrap();
        let expected = initial(l);
        let mut words = values(s, OLD, 8, &expected);
        readonly_bits(s, OLD, &words, 8, &expected).unwrap();
        words[s.old.at(0)] ^= 1;
        assert!(checked_span(s, OLD, &words, 8, &expected).is_ok());
        assert!(readonly_bits(s, OLD, &words, 8, &expected).is_err());
    }
}
