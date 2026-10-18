//! Fixed public batched GR operator fixture. No request or causal graph path.
//! MAX_BATCH8 complete private allocations, borrowed original public weights,
//! unchanged scalar kernels as a GPU reference, independent core CPU oracle.
use crate::quant_matvec::{
    cooperative,
    matmul::{BatchSpans, MAX_BATCH, MatrixGeometry, packed_matmul},
};
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    residual_batch_gpu as b, residual_gpu as scalar,
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::transient::TransientIoReceipt;
use kuno_text::model::{BoundGraph, GraphBindingId, Qwen4Exp};
use kuno_text::ops::{self, GatedResidualShape};
use kuno_text::{GgufFile, GgufLimits, TensorView};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAIN_BUDGET: usize = 16 * 1024 * 1024;
const IO_BUDGET: u64 = 64 * 1024 * 1024;
const PUBLIC_MAX: usize = 128 * 1024 * 1024;
const GAP: f32 = -12345.;
const STATUS_GAP: u32 = 0xe1234567;
const TILE: usize = 8;
#[path = "strata_dense_numeric_probe.rs"]
mod dense;
#[path = "strata_shared_numeric_probe.rs"]
mod shared_numeric;

pub(crate) fn read_native_public(
    arena: &PrivateArena,
    id: BufferId,
) -> Result<Zeroizing<Vec<u32>>> {
    native::read_id(arena, id)
}
#[path = "strata_gr_numeric_probe.rs"]
mod native;

struct Matrix {
    geometry: MatrixGeometry,
    handle: Handle,
    words: usize,
    decoded: Zeroizing<Vec<f32>>,
}
impl Matrix {
    fn load(
        client: &ComputeClient<WgpuRuntime>,
        tensor: TensorView<'_>,
        rows: usize,
        columns: usize,
    ) -> Result<Self> {
        if tensor.info().dimensions() != [columns as u64, rows as u64] {
            return Err("public GR matrix geometry".into());
        }
        let geometry = MatrixGeometry::new(tensor.info().dtype().gguf_id(), rows, columns, TILE)?;
        if geometry.total_bytes() > 32 * 1024 * 1024 {
            return Err("public GR matrix exceeds bound".into());
        }
        let mut packed = Zeroizing::new(vec![0_u8; geometry.total_bytes().div_ceil(4) * 4]);
        tensor
            .read_range(0, &mut packed[..geometry.total_bytes()])
            .map_err(|_| "public GR weight read failed")?;
        let mut decoded = Zeroizing::new(vec![
            0.;
            rows.checked_mul(columns)
                .ok_or("public GR size overflow")?
        ]);
        tensor
            .info()
            .dtype()
            .decode(&packed[..geometry.total_bytes()], &mut decoded)
            .map_err(|_| "public GR weight decode failed")?;
        if decoded.iter().any(|v| !v.is_finite()) {
            return Err("public GR weight nonfinite".into());
        }
        let words = packed.len() / 4;
        let handle = client.create_from_slice(&packed); // Original PUBLIC weights only.
        Ok(Self {
            geometry,
            handle,
            words,
            decoded,
        })
    }
    fn cpu(&self, x: &[f32]) -> Result<Zeroizing<Vec<f32>>> {
        if x.len() != self.geometry.columns() {
            return Err("public GR CPU matrix input".into());
        }
        let y = self
            .decoded
            .chunks_exact(x.len())
            .map(|row| {
                row.iter()
                    .zip(x)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum::<f64>() as f32
            })
            .collect::<Vec<_>>();
        if y.iter().any(|v| !v.is_finite()) {
            return Err("public GR CPU matrix nonfinite".into());
        }
        Ok(Zeroizing::new(y))
    }
}
struct PublicStage {
    binding: GraphBindingId,
    base: &'static str,
    shape: GatedResidualShape,
    low_rank: usize,
    epsilon: f32,
    gamma: Handle,
    gamma_cpu: Zeroizing<Vec<f32>>,
    down: Matrix,
    up: Matrix,
    injection: Option<Matrix>,
}
fn preflight_public(graph: BoundGraph<'_, '_>) -> Result<usize> {
    let g = graph.geometry();
    let r = g
        .residual
        .streams
        .checked_mul(g.residual.width)
        .ok_or("public GR geometry overflow")?;
    let mut bytes = 0usize;
    for (base, inject) in [
        ("blk.0.hc_attn", true),
        ("blk.0.hc_ffn", true),
        ("output_hc", false),
    ] {
        let norms = graph.static_vector(&format!("{base}_norm.weight"))?;
        if norms.len() != r || norms.iter().any(|v| !v.is_finite()) {
            return Err("public GR norm geometry".into());
        }
        bytes = bytes
            .checked_add((r * 4).max(32768).next_power_of_two())
            .ok_or("public GR budget overflow")?;
        for (name, rows, columns) in [
            ("down", g.low_rank, r),
            ("up", r, g.low_rank),
            ("inject", g.residual.streams, r),
        ] {
            if name == "inject" && !inject {
                continue;
            }
            let tensor = graph.tensor(&format!("{base}_{name}.weight"))?;
            if tensor.info().dimensions() != [columns as u64, rows as u64] {
                return Err("public GR matrix geometry".into());
            }
            let m = MatrixGeometry::new(tensor.info().dtype().gguf_id(), rows, columns, TILE)?;
            if m.total_bytes() > 32 * 1024 * 1024 {
                return Err("public GR matrix exceeds bound".into());
            }
            bytes = bytes
                .checked_add(m.total_bytes().max(32768).next_power_of_two())
                .ok_or("public GR budget overflow")?;
        }
    }
    if bytes > PUBLIC_MAX {
        return Err("public GR aggregate admission".into());
    }
    Ok(bytes)
}

fn load(
    client: &ComputeClient<WgpuRuntime>,
    graph: BoundGraph<'_, '_>,
) -> Result<Vec<PublicStage>> {
    let g = graph.geometry();
    if g.residual != GatedResidualShape::new(4, 2560)? || g.low_rank != 320 || g.epsilon != 1e-6 {
        return Err("public GR fixture requires admitted geometry".into());
    }
    let r = g.residual.streams * g.residual.width;
    preflight_public(graph)?;
    let mut out = Vec::new();
    let mut charge = 0usize;
    for (base, inject) in [
        ("blk.0.hc_attn", true),
        ("blk.0.hc_ffn", true),
        ("output_hc", false),
    ] {
        let gamma_cpu = Zeroizing::new(
            graph
                .static_vector(&format!("{base}_norm.weight"))?
                .to_vec(),
        );
        if gamma_cpu.len() != r || gamma_cpu.iter().any(|x| !x.is_finite()) {
            return Err("public GR norm geometry".into());
        }
        let down = Matrix::load(
            client,
            graph.tensor(&format!("{base}_down.weight"))?,
            g.low_rank,
            r,
        )?;
        let up = Matrix::load(
            client,
            graph.tensor(&format!("{base}_up.weight"))?,
            r,
            g.low_rank,
        )?;
        let injection = inject
            .then(|| {
                Matrix::load(
                    client,
                    graph.tensor(&format!("{base}_inject.weight"))?,
                    g.residual.streams,
                    r,
                )
            })
            .transpose()?;
        for m in [&down, &up].into_iter().chain(injection.iter()) {
            charge = charge
                .checked_add(m.geometry.total_bytes().max(32768).next_power_of_two())
                .ok_or("public GR charge overflow")?;
        }
        charge = charge
            .checked_add((r * 4).max(32768).next_power_of_two())
            .ok_or("public GR charge overflow")?;
        if charge > PUBLIC_MAX {
            return Err("public GR cache exceeds bound".into());
        }
        let gamma = client.create_from_slice(f32::as_bytes(&gamma_cpu));
        out.push(PublicStage {
            binding: graph.binding_id(),
            base,
            shape: g.residual,
            low_rank: g.low_rank,
            epsilon: g.epsilon,
            gamma,
            gamma_cpu,
            down,
            up,
            injection,
        });
    }
    Ok(out)
}
#[derive(Clone, Copy)]
struct Buffer {
    id: BufferId,
    span: b::Span,
}
const INPUT: usize = 0;
const NORM: usize = 1;
const DOWN: usize = 2;
const LOW: usize = 3;
const UP: usize = 4;
const MIX: usize = 5;
const INJECT: usize = 6;
const OUTPUT: usize = 7;
const BLOCK: usize = 8;
const SNORM: usize = 9;
const SDOWN: usize = 10;
const SLOW: usize = 11;
const SUP: usize = 12;
const SMIX: usize = 13;
const SINJECT: usize = 14;
const SOUTPUT: usize = 15;
const N: usize = 16;
fn widths(stage: &PublicStage) -> [usize; N] {
    let r = stage.shape.streams * stage.shape.width;
    [
        r,
        r,
        stage.low_rank,
        stage.low_rank,
        r,
        stage.shape.width,
        stage.shape.streams,
        r,
        stage.shape.width,
        stage.shape.streams,
        stage.low_rank,
        stage.low_rank,
        r,
        stage.shape.width,
        stage.shape.streams,
        r,
    ]
}
fn footprint(widths: [usize; N], aligned: bool) -> Result<usize> {
    widths.into_iter().try_fold(0usize, |sum, n| {
        let span = fixture_span(n, aligned)?;
        sum.checked_add(BufferLayout::new(span.words)?.owned_bytes)
            .ok_or_else(|| "public GR footprint overflow".into())
    })
}
fn fixture(width: usize, seed: usize) -> Zeroizing<Vec<f32>> {
    Zeroizing::new(
        (0..MAX_BATCH * width)
            .map(|i| (((i * 7 + i / width * 13 + seed) % 43) as f32 - 21.) / 16.)
            .collect(),
    )
}
fn fixture_span(width: usize, aligned: bool) -> Result<b::Span> {
    Ok(if aligned {
        b::Span::aligned_fixture(width)?
    } else {
        b::Span::fixture(width)?
    })
}
fn allocate(
    arena: &mut PrivateArena,
    stage: &PublicStage,
    seed: usize,
    aligned: bool,
) -> Result<[Buffer; N]> {
    let w = widths(stage);
    let x = fixture(w[INPUT], seed);
    let block = fixture(w[BLOCK], seed + 3);
    let mut buffers = Vec::with_capacity(N);
    for (i, width) in w.into_iter().enumerate() {
        let span = fixture_span(width, aligned)?;
        let id = if i >= SNORM {
            arena.upload_u32(&Zeroizing::new(vec![STATUS_GAP; span.words]))?
        } else {
            let mut v = Zeroizing::new(vec![GAP; span.words]);
            if i == INPUT || i == BLOCK {
                let src = if i == INPUT { &x } else { &block };
                for token in 0..MAX_BATCH {
                    v[span.index(token, 0)..][..width]
                        .copy_from_slice(&src[token * width..][..width]);
                }
            }
            arena.upload_f32(&v)?
        };
        buffers.push(Buffer { id, span });
    }
    buffers
        .try_into()
        .map_err(|_| "public GR buffer count".into())
}
fn arg(arena: &PrivateArena, v: Buffer) -> Result<ArrayArg<WgpuRuntime>> {
    v.span
        .validate(MAX_BATCH, arena.layout(v.id)?.owned_words)?;
    Ok(unsafe {
        ArrayArg::from_raw_parts(arena.handle(v.id)?.clone(), arena.layout(v.id)?.owned_words)
    })
}
fn scalar_region(
    span: b::Span,
    width: usize,
    token: usize,
    owned_words: usize,
) -> Result<(usize, usize)> {
    span.validate(MAX_BATCH, owned_words)?;
    if span.width != width || token >= MAX_BATCH {
        return Err("public scalar GR shape".into());
    }
    let first = span.index(token, 0);
    let suffix = owned_words
        .checked_sub(first)
        .and_then(|n| n.checked_sub(width))
        .ok_or("public scalar GR extent")?;
    Ok((first, suffix))
}
fn scalar_arg_at(
    arena: &PrivateArena,
    v: Buffer,
    width: usize,
    token: usize,
) -> Result<ArrayArg<WgpuRuntime>> {
    let (first, suffix) = scalar_region(v.span, width, token, arena.layout(v.id)?.owned_words)?;
    // Original full owner retained; expose only one checked logical region.
    let h = arena
        .handle(v.id)?
        .clone()
        .offset_start((first * 4) as u64)
        .offset_end((suffix * 4) as u64);
    Ok(unsafe { ArrayArg::from_raw_parts(h, width) })
}
fn binding_proof(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    stage: &PublicStage,
    sets: &[[Buffer; N]; 2],
) -> Result<()> {
    let private: Vec<_> = sets
        .iter()
        .flatten()
        .map(|v| arena.handle(v.id).cloned())
        .collect::<std::result::Result<_, _>>()?;
    let public: Vec<_> = [&stage.down, &stage.up]
        .into_iter()
        .chain(stage.injection.iter())
        .map(|m| m.handle.clone())
        .chain([stage.gamma.clone()])
        .collect();
    let all = private.iter().chain(public.iter()).collect::<Vec<_>>();
    let physical = all
        .iter()
        .map(|h| client.get_resource((*h).clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| "public GR resource proof")?;
    for (index, r) in physical.iter().enumerate() {
        let h = all[index];
        if h.stream != private[0].stream {
            return Err("public GR binding stream".into());
        }
        let v = r.resource();
        let end = v
            .offset
            .checked_add(v.size)
            .ok_or("public GR resource overflow")?;
        if end > v.buffer.size()
            || (index < private.len()
                && (v.offset != 0 || end != v.buffer.size() || v.size != h.size()))
        {
            return Err("public GR private physical extent".into());
        }
        for other in physical.iter().skip(index + 1) {
            let other = other.resource();
            let other_end = other
                .offset
                .checked_add(other.size)
                .ok_or("public GR resource overflow")?;
            if other.buffer == v.buffer && other.offset < end && v.offset < other_end {
                return Err("public GR physical alias".into());
            }
        }
    }
    Ok(())
}
fn finite(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    x: Buffer,
    s: Buffer,
    count: usize,
) -> Result<()> {
    finite_at(client, arena, x, s, count, 0)
}
fn finite_at(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    x: Buffer,
    s: Buffer,
    count: usize,
    token: usize,
) -> Result<()> {
    if count == 0 || token.checked_add(count).is_none_or(|n| n > MAX_BATCH) {
        return Err("public GR finite token range".into());
    }
    if x.span.width != s.span.width {
        return Err("public GR finite shape".into());
    }
    b::finite_span::launch::<WgpuRuntime>(
        client,
        x.span.grid(),
        CubeDim::new_1d(64),
        arg(arena, x)?,
        arg(arena, s)?,
        count,
        x.span.index(token, 0),
        x.span.stride,
        s.span.index(token, 0),
        s.span.stride,
        x.span.width,
    );
    Ok(())
}
fn project_batch(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    m: &Matrix,
    x: Buffer,
    output_and_status: (Buffer, Buffer),
    count: usize,
    tile: usize,
) -> Result<()> {
    let (y, s) = output_and_status;
    let g = MatrixGeometry::new(
        m.geometry.dtype(),
        m.geometry.rows(),
        m.geometry.columns(),
        tile,
    )?;
    let spans = BatchSpans::new(
        count,
        0,
        x.span.first,
        x.span.stride,
        y.span.first,
        y.span.stride,
    );
    g.validate_spans(
        &spans,
        m.words,
        arena.layout(x.id)?.owned_words,
        arena.layout(y.id)?.owned_words,
    )?;
    packed_matmul::launch::<WgpuRuntime>(
        client,
        g.grid(),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(m.handle.clone(), m.words) },
        arg(arena, x)?,
        arg(arena, y)?,
        count,
        0,
        x.span.first,
        x.span.stride,
        y.span.first,
        y.span.stride,
        g.rows(),
        g.columns(),
        g.dtype(),
        tile,
    );
    finite(client, arena, y, s, count)
}
fn batch(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    p: &PublicStage,
    v: &[Buffer; N],
    count: usize,
    tile: usize,
) -> Result<()> {
    for x in v {
        x.span.validate(count, arena.layout(x.id)?.owned_words)?;
    }
    let (hc, d, lr) = (p.shape.streams, p.shape.width, p.low_rank);
    b::grouped_rms_norm::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(hc as u32, MAX_BATCH as u32, 1),
        CubeDim::new_1d(64),
        arg(arena, v[INPUT])?,
        unsafe { ArrayArg::from_raw_parts(p.gamma.clone(), hc * d) },
        arg(arena, v[NORM])?,
        arg(arena, v[SNORM])?,
        p.epsilon,
        count,
        v[INPUT].span.first,
        v[INPUT].span.stride,
        v[NORM].span.first,
        v[NORM].span.stride,
        v[SNORM].span.first,
        v[SNORM].span.stride,
        hc,
        d,
    );
    project_batch(
        client,
        arena,
        &p.down,
        v[NORM],
        (v[DOWN], v[SDOWN]),
        count,
        tile,
    )?;
    b::activate_down::launch::<WgpuRuntime>(
        client,
        v[LOW].span.grid(),
        CubeDim::new_1d(64),
        arg(arena, v[DOWN])?,
        arg(arena, v[LOW])?,
        arg(arena, v[SLOW])?,
        count,
        v[DOWN].span.first,
        v[DOWN].span.stride,
        v[LOW].span.first,
        v[LOW].span.stride,
        v[SLOW].span.first,
        v[SLOW].span.stride,
        hc,
        lr,
    );
    project_batch(client, arena, &p.up, v[LOW], (v[UP], v[SUP]), count, tile)?;
    b::mix::launch::<WgpuRuntime>(
        client,
        v[MIX].span.grid(),
        CubeDim::new_1d(64),
        arg(arena, v[NORM])?,
        arg(arena, v[UP])?,
        arg(arena, v[MIX])?,
        arg(arena, v[SMIX])?,
        count,
        v[NORM].span.first,
        v[NORM].span.stride,
        v[UP].span.first,
        v[UP].span.stride,
        v[MIX].span.first,
        v[MIX].span.stride,
        v[SMIX].span.first,
        v[SMIX].span.stride,
        hc,
        d,
    );
    if let Some(m) = &p.injection {
        project_batch(
            client,
            arena,
            m,
            v[NORM],
            (v[INJECT], v[SINJECT]),
            count,
            tile,
        )?;
        b::combine::launch::<WgpuRuntime>(
            client,
            v[OUTPUT].span.grid(),
            CubeDim::new_1d(64),
            arg(arena, v[INPUT])?,
            arg(arena, v[BLOCK])?,
            arg(arena, v[INJECT])?,
            arg(arena, v[OUTPUT])?,
            arg(arena, v[SOUTPUT])?,
            count,
            v[INPUT].span.first,
            v[INPUT].span.stride,
            v[BLOCK].span.first,
            v[BLOCK].span.stride,
            v[INJECT].span.first,
            v[INJECT].span.stride,
            v[OUTPUT].span.first,
            v[OUTPUT].span.stride,
            v[SOUTPUT].span.first,
            v[SOUTPUT].span.stride,
            hc,
            d,
        );
    }
    Ok(())
}
fn project_scalar(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    m: &Matrix,
    x: Buffer,
    y: Buffer,
    s: Buffer,
    token: usize,
) -> Result<()> {
    cooperative::cooperative_matvec::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(m.geometry.rows() as u32, 1, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(m.handle.clone(), m.words) },
        scalar_arg_at(arena, x, m.geometry.columns(), token)?,
        scalar_arg_at(arena, y, m.geometry.rows(), token)?,
        m.geometry.rows(),
        m.geometry.columns(),
        m.geometry.dtype(),
        0,
    );
    finite_at(client, arena, y, s, 1, token)
}
fn single(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    p: &PublicStage,
    v: &[Buffer; N],
    source: &[Buffer; N],
    token: usize,
) -> Result<()> {
    for id in [INPUT, BLOCK] {
        let x = source[id];
        let y = v[id];
        b::copy_span::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(x.span.width.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            arg(arena, x)?,
            arg(arena, y)?,
            1,
            x.span.index(token, 0),
            x.span.stride,
            y.span.first,
            y.span.stride,
            x.span.width,
        );
    }
    single_compute(client, arena, p, v, 0)
}
fn single_compute(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    p: &PublicStage,
    v: &[Buffer; N],
    token: usize,
) -> Result<()> {
    let (hc, d, lr) = (p.shape.streams, p.shape.width, p.low_rank);
    scalar::grouped_rms_norm::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(hc as u32, 1, 1),
        CubeDim::new_1d(64),
        scalar_arg_at(arena, v[INPUT], hc * d, token)?,
        unsafe { ArrayArg::from_raw_parts(p.gamma.clone(), hc * d) },
        scalar_arg_at(arena, v[NORM], hc * d, token)?,
        scalar_arg_at(arena, v[SNORM], hc, token)?,
        p.epsilon,
        hc,
        d,
    );
    project_scalar(client, arena, &p.down, v[NORM], v[DOWN], v[SDOWN], token)?;
    scalar::activate_down::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(lr.div_ceil(64) as u32, 1, 1),
        CubeDim::new_1d(64),
        scalar_arg_at(arena, v[DOWN], lr, token)?,
        scalar_arg_at(arena, v[LOW], lr, token)?,
        scalar_arg_at(arena, v[SLOW], lr, token)?,
        hc,
        lr,
    );
    project_scalar(client, arena, &p.up, v[LOW], v[UP], v[SUP], token)?;
    scalar::mix::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(d.div_ceil(64) as u32, 1, 1),
        CubeDim::new_1d(64),
        scalar_arg_at(arena, v[NORM], hc * d, token)?,
        scalar_arg_at(arena, v[UP], hc * d, token)?,
        scalar_arg_at(arena, v[MIX], d, token)?,
        scalar_arg_at(arena, v[SMIX], d, token)?,
        hc,
        d,
    );
    if let Some(m) = &p.injection {
        project_scalar(client, arena, m, v[NORM], v[INJECT], v[SINJECT], token)?;
        scalar::combine::launch::<WgpuRuntime>(
            client,
            CubeCount::Static((hc * d).div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            scalar_arg_at(arena, v[INPUT], hc * d, token)?,
            scalar_arg_at(arena, v[BLOCK], d, token)?,
            scalar_arg_at(arena, v[INJECT], hc, token)?,
            scalar_arg_at(arena, v[OUTPUT], hc * d, token)?,
            scalar_arg_at(arena, v[SOUTPUT], hc * d, token)?,
            hc,
            d,
        );
    }
    Ok(())
}
fn oracle(p: &PublicStage, seed: usize) -> Result<Vec<[Zeroizing<Vec<f32>>; 7]>> {
    let w = p.shape.streams * p.shape.width;
    let input = fixture(w, seed);
    let block = fixture(p.shape.width, seed + 3);
    (0..MAX_BATCH)
        .map(|t| {
            let x = &input[t * w..][..w];
            let norm = Zeroizing::new(ops::rms_norm(x, &p.gamma_cpu, p.shape.width, p.epsilon)?);
            let down = p.down.cpu(&norm)?;
            let low = Zeroizing::new(p.shape.activate_down(&down)?);
            let up = p.up.cpu(&low)?;
            let mix = Zeroizing::new(p.shape.mix(&norm, &up)?);
            let (inj, out) = if let Some(m) = &p.injection {
                let inj = m.cpu(&norm)?;
                let out = Zeroizing::new(p.shape.combine(
                    x,
                    &block[t * p.shape.width..][..p.shape.width],
                    &inj,
                )?);
                (inj, out)
            } else {
                (Zeroizing::new(Vec::new()), Zeroizing::new(Vec::new()))
            };
            Ok([norm, down, low, up, mix, inj, out])
        })
        .collect()
}
fn audit(client: &ComputeClient<WgpuRuntime>) -> Result<TransientIoReceipt> {
    let r = cubecl::future::block_on(client.sanitize_transient_io_audited())
        .map_err(|_| "public GR I/O audit failed")?;
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("public GR audit count overflow")?;
    if full == 0 || r.inspected_bytes != full || r.inspected_buffers == 0 {
        return Err("public GR incomplete owned I/O audit".into());
    }
    Ok(r)
}
fn read(arena: &PrivateArena, v: Buffer) -> Result<Zeroizing<Vec<u32>>> {
    let expected_bytes = arena.layout(v.id)?.owned_bytes;
    let mut bytes = arena
        .client()?
        .read_one(arena.handle(v.id)?.clone())
        .map_err(|_| "public GR completed read failed")?;
    let result = decode_read(&mut bytes, expected_bytes);
    drop(bytes);
    result
}
// Guard the mapped lease before allocating/parsing a host result, including
// malformed extents and allocation failures. The returned host owner also wipes.
struct ScrubSlice<'a>(&'a mut [u8]);
impl Drop for ScrubSlice<'_> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn decode_read(bytes: &mut [u8], expected_bytes: usize) -> Result<Zeroizing<Vec<u32>>> {
    let guarded = ScrubSlice(bytes);
    if expected_bytes == 0 || !expected_bytes.is_multiple_of(4) || guarded.0.len() != expected_bytes
    {
        return Err("public GR full read extent".into());
    }
    let mut words = Zeroizing::new(Vec::new());
    words.try_reserve_exact(expected_bytes / 4)?;
    for x in guarded.0.chunks_exact(4) {
        words.push(u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
    }
    Ok(words)
}
fn check_span(
    span: b::Span,
    actual: &[u32],
    count: usize,
    expected: &[Zeroizing<Vec<f32>>],
) -> Result<f32> {
    span.validate(MAX_BATCH, span.words)?;
    if count > MAX_BATCH
        || expected.len() != count
        || expected.iter().any(|x| x.len() != span.width)
        || actual.len() != BufferLayout::new(span.words)?.owned_words
    {
        return Err("public GR output extent".into());
    }
    let mut max = 0_f32;
    for (i, &bits) in actual.iter().enumerate() {
        let relative = i.saturating_sub(span.first);
        let token = relative / span.stride;
        let d = relative % span.stride;
        if i >= span.first && token < count && d < span.width {
            let r = expected[token][d];
            let x = f32::from_bits(bits);
            let e = (x - r).abs();
            if !x.is_finite() || !e.is_finite() || e > 1e-4 + 3e-4 * r.abs() {
                return Err("public GR numerical parity".into());
            }
            max = max.max(e);
        } else {
            let want = if i < span.words { GAP.to_bits() } else { 0 };
            if bits != want {
                return Err("public GR inactive/gap/full-span padding changed".into());
            }
        }
    }
    Ok(max)
}
fn check(v: Buffer, actual: &[u32], count: usize, expected: &[Zeroizing<Vec<f32>>]) -> Result<f32> {
    check_span(v.span, actual, count, expected)
}
fn check_flag_span(span: b::Span, actual: &[u32], count: usize) -> Result<()> {
    span.validate(MAX_BATCH, span.words)?;
    if count > MAX_BATCH || actual.len() != BufferLayout::new(span.words)?.owned_words {
        return Err("public GR flag extent".into());
    }
    for (i, &bits) in actual.iter().enumerate() {
        let r = i.saturating_sub(span.first);
        let active = i >= span.first && r / span.stride < count && r % span.stride < span.width;
        let want = if active {
            1
        } else if i < span.words {
            STATUS_GAP
        } else {
            0
        };
        if bits != want {
            return Err("public GR finite/status or inactive padding".into());
        }
    }
    Ok(())
}
fn check_flags(v: Buffer, actual: &[u32], count: usize) -> Result<()> {
    check_flag_span(v.span, actual, count)
}
fn used(injection: bool) -> usize {
    if injection { 7 } else { 5 }
}
fn stages() -> [(usize, usize); 7] {
    [
        (NORM, SNORM),
        (DOWN, SDOWN),
        (LOW, SLOW),
        (UP, SUP),
        (MIX, SMIX),
        (INJECT, SINJECT),
        (OUTPUT, SOUTPUT),
    ]
}
fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    graph: BoundGraph<'_, '_>,
    public: &[PublicStage],
    retired: &AtomicBool,
) -> Result<Vec<serde_json::Value>> {
    let mut reports = Vec::new();
    for (stage_index, p) in public.iter().enumerate() {
        if p.binding != graph.binding_id() {
            return Err("public GR binding changed".into());
        }
        let expected = oracle(p, stage_index)?;
        for count in [1, 3, 5, 8] {
            for tile in [4, 8] {
                let planned = footprint(widths(p), false)?
                    .checked_add(footprint(widths(p), true)?)
                    .ok_or("public GR footprint overflow")?;
                if planned > MAIN_BUDGET {
                    return Err("public GR constructor admission".into());
                }
                let sets = [
                    allocate(arena, p, stage_index, false)?,
                    allocate(arena, p, stage_index, true)?,
                ];
                if arena.charged_bytes() != planned {
                    return Err("public GR physical page plan".into());
                }
                binding_proof(client, arena, p, &sets)?;
                batch(client, arena, p, &sets[0], count, tile)?;
                let mut max = 0_f32;
                for (j, (out, status)) in stages()
                    .into_iter()
                    .take(used(p.injection.is_some()))
                    .enumerate()
                {
                    let e: Vec<_> = expected
                        .iter()
                        .take(count)
                        .map(|row| Zeroizing::new(row[j].to_vec()))
                        .collect();
                    max = max.max(check(sets[0][out], &read(arena, sets[0][out])?, count, &e)?);
                    check_flags(sets[0][status], &read(arena, sets[0][status])?, count)?;
                }
                audit(client)?;
                // Every active column also executes the unchanged scalar GPU graph.
                let mut scalar_error = 0_f32;
                for token in 0..count {
                    single(client, arena, p, &sets[1], &sets[0], token)?;
                    for (j, (out, status)) in stages()
                        .into_iter()
                        .take(used(p.injection.is_some()))
                        .enumerate()
                    {
                        let expected_one = [Zeroizing::new(expected[token][j].to_vec())];
                        scalar_error = scalar_error.max(check(
                            sets[1][out],
                            &read(arena, sets[1][out])?,
                            1,
                            &expected_one,
                        )?);
                        check_flags(sets[1][status], &read(arena, sets[1][status])?, 1)?;
                        // Compare against the actual batched GPU column, not only CPU.
                        let all = read(arena, sets[0][out])?;
                        let gpu = [Zeroizing::new(
                            all[sets[0][out].span.index(token, 0)..][..sets[0][out].span.width]
                                .iter()
                                .map(|&x| f32::from_bits(x))
                                .collect(),
                        )];
                        scalar_error = scalar_error.max(check(
                            sets[1][out],
                            &read(arena, sets[1][out])?,
                            1,
                            &gpu,
                        )?);
                    }
                    audit(client)?;
                }
                if p.injection.is_none() {
                    for v in &sets {
                        for id in [INJECT, OUTPUT] {
                            check(v[id], &read(arena, v[id])?, 0, &[])?;
                        }
                        for id in [SINJECT, SOUTPUT] {
                            check_flags(v[id], &read(arena, v[id])?, 0)?;
                        }
                    }
                    audit(client)?;
                }
                // Original input and block are read-only, including each inactive column.
                for id in [INPUT, BLOCK] {
                    let width = sets[0][id].span.width;
                    let values = fixture(width, stage_index + if id == BLOCK { 3 } else { 0 });
                    let e: Vec<_> = (0..MAX_BATCH)
                        .map(|t| Zeroizing::new(values[t * width..][..width].to_vec()))
                        .collect();
                    check(sets[0][id], &read(arena, sets[0][id])?, MAX_BATCH, &e)?;
                }
                audit(client)?;
                arena.reset()?;
                if sets.iter().flatten().any(|x| arena.handle(x.id).is_ok())
                    || retired.load(Ordering::Acquire)
                {
                    return Err("public GR reset lifetime proof".into());
                }
                reports.push(serde_json::json!({"base":p.base,"count":count,"tile_tokens":tile,"max_cpu_abs_error":max,"max_scalar_gpu_abs_error":scalar_error,"full_private_main_bytes":planned,"all_active_columns_checked":true,"tails_and_padding_preserved":true,"no_injection":p.injection.is_none()}));
                println!(
                    "public_gr_batch base={} count={count} tile={tile} cpu_abs={max} scalar_gpu_abs={scalar_error}",
                    p.base
                );
            }
        }
    }
    Ok(reports)
}
// Cleanup is performed by the caller before this acceptance gate. A successful
// computation cannot produce an acceptance receipt if owning cleanup failed.
fn finish_probe<T>(
    attempt: std::thread::Result<Result<T>>,
    cleanup_ok: bool,
    retired: bool,
) -> Result<T> {
    if !cleanup_ok || retired {
        return Err("public GR owner cleanup/context retired".into());
    }
    attempt.map_err(|_| "public GR probe panicked")?
}
fn validate_storage_alignment(alignment: usize) -> Result<()> {
    if alignment == 0 || !256usize.is_multiple_of(alignment) {
        return Err("public GR scalar device storage alignment".into());
    }
    Ok(())
}
pub fn run(client: &ComputeClient<WgpuRuntime>, storage_alignment: usize) -> Result<()> {
    validate_storage_alignment(storage_alignment)?;
    let path = std::env::var_os("KUNO_TEST_GGUF")
        .ok_or("existing original GGUF required for public GR probe")?;
    // The selected native engine consumes both original DASLab shards. The
    // single-file loader deliberately rejects a partial split; retain that
    // protection and the original descriptors rather than merging payloads.
    let file = if crate::strata_intel_core::enabled() {
        let first = std::path::Path::new(&path);
        if first.file_name().and_then(|name| name.to_str())
            != Some("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
        {
            return Err("public native GR requires original DAS first shard".into());
        }
        let directory = first.parent().ok_or("public native GR shard directory")?;
        let paths: Vec<_> = (1..=2)
            .map(|i| {
                directory.join(format!(
                    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
                ))
            })
            .collect();
        let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
        if file.file_len() != 83_617_662_656 {
            return Err("public native GR original DAS extent".into());
        }
        crate::daslab_profile::validate_inventory(
            file.tensors()
                .iter()
                .map(|(name, t)| (name.as_str(), t.dtype().gguf_id())),
        )
        .map_err(|_| "public native GR original DAS inventory")?;
        file
    } else {
        GgufFile::open(path, GgufLimits::default())?
    };
    let model = Qwen4Exp::bind(&file).map_err(|_| "public GR model binding failed")?;
    let graph = model.bound_graph();
    let public = load(client, graph)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), MAIN_BUDGET, IO_BUDGET, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if crate::strata_intel_core::enabled() {
            let mut reports = native::cases(client, &mut arena, &public)?;
            reports.extend(dense::cases(client, &mut arena, &file)?);
            reports.extend(shared_numeric::cases(client, &mut arena, &file)?);
            #[cfg(feature = "strata-intel-core")]
            reports.extend(crate::strata_matrix::public_numeric(
                client, &mut arena, &file,
            )?);
            Ok(reports)
        } else {
            cases(client, &mut arena, graph, &public, &retired)
        }
    }));
    let close = arena.close();
    let reports = finish_probe(attempt, close.is_ok(), retired.load(Ordering::Acquire))?;
    let cleanup = arena
        .owned_zero_audit_receipt()
        .ok_or("public GR owning zero audit missing")?;
    file.verify_file_unchanged()
        .map_err(|_| "public GR original GGUF changed")?;
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let r = serde_json::json!({"scope":"fixed-public-GR-operator-only","fixture":"gr-batch-probe","case_count":reports.len(),"cases":reports,"borrowed_original_weights":true,"private_uploads_owned":true,"main_cleanup_buffers":cleanup.buffers,"main_cleanup_bytes":cleanup.bytes,"context_retired":false,"full_graph_or_prefill_accepted":false});
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&r)?.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outputs(span: b::Span, count: usize) -> (Vec<u32>, Vec<Zeroizing<Vec<f32>>>) {
        let mut words = vec![0; BufferLayout::new(span.words).unwrap().owned_words];
        words[..span.words].fill(GAP.to_bits());
        let expected = (0..count)
            .map(|t| {
                Zeroizing::new(
                    (0..span.width)
                        .map(|i| (t * 17 + i) as f32 / 32.)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        for (t, row) in expected.iter().enumerate() {
            for (i, x) in row.iter().enumerate() {
                words[span.index(t, i)] = x.to_bits();
            }
        }
        (words, expected)
    }
    fn flags(span: b::Span, count: usize) -> Vec<u32> {
        let mut words = vec![0; BufferLayout::new(span.words).unwrap().owned_words];
        words[..span.words].fill(STATUS_GAP);
        for t in 0..count {
            words[span.index(t, 0)..][..span.width].fill(1);
        }
        words
    }
    #[test]
    fn full_read_lease_is_wiped_on_success_and_every_malformed_extent() {
        let mut bytes = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let decoded = decode_read(&mut bytes, 8).unwrap();
        assert_eq!(&*decoded, &[0x04030201, 0x08070605]);
        assert!(bytes.iter().all(|&x| x == 0));
        for (size, expected) in [(0, 0), (7, 8), (8, 7), (4, 8), (8, 4)] {
            let mut bytes = vec![0xa5; size];
            assert!(decode_read(&mut bytes, expected).is_err());
            assert!(bytes.iter().all(|&x| x == 0));
        }
        let mut bytes = vec![0xa5; 8];
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = ScrubSlice(&mut bytes);
            panic!("fixed public guard fixture");
        }));
        assert!(panicked.is_err());
        assert!(bytes.iter().all(|&x| x == 0));
    }
    #[test]
    fn active_columns_finite_status_and_entire_owned_padding_are_checked() {
        for width in [4, 65, 2560, 10240] {
            let span = b::Span::fixture(width).unwrap();
            for count in [1, 3, 5, 8] {
                let (words, expected) = outputs(span, count);
                assert_eq!(check_span(span, &words, count, &expected).unwrap(), 0.);
                let statuses = flags(span, count);
                check_flag_span(span, &statuses, count).unwrap();
                for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e10] {
                    let mut wrong = words.clone();
                    wrong[span.index(count - 1, width - 1)] = value.to_bits();
                    assert!(check_span(span, &wrong, count, &expected).is_err());
                }
                for i in [0, span.index(0, width), span.words - 1, words.len() - 1] {
                    let mut wrong = words.clone();
                    wrong[i] ^= 1;
                    assert!(check_span(span, &wrong, count, &expected).is_err());
                    let mut wrong = statuses.clone();
                    wrong[i] ^= 1;
                    assert!(check_flag_span(span, &wrong, count).is_err());
                }
                for value in [0, 2, STATUS_GAP] {
                    let mut wrong = statuses.clone();
                    wrong[span.index(count - 1, width - 1)] = value;
                    assert!(check_flag_span(span, &wrong, count).is_err());
                }
                if count < MAX_BATCH {
                    let mut wrong = words.clone();
                    wrong[span.index(count, 0)] = 0;
                    assert!(check_span(span, &wrong, count, &expected).is_err());
                }
            }
            let (unused, expected) = outputs(span, 0);
            check_span(span, &unused, 0, &expected).unwrap();
            check_flag_span(span, &flags(span, 0), 0).unwrap();
        }
    }
    #[test]
    fn malformed_shapes_cannot_bypass_tail_or_status_checks() {
        let span = b::Span::fixture(4).unwrap();
        let (words, expected) = outputs(span, 1);
        assert!(check_span(span, &words[..span.words], 1, &expected).is_err());
        assert!(check_span(span, &words, 9, &expected).is_err());
        assert!(check_span(span, &words, 0, &expected).is_err());
        assert!(check_span(span, &words, 1, &[Zeroizing::new(vec![0.; 3])]).is_err());
        let bad = b::Span { stride: 0, ..span };
        assert!(check_span(bad, &words, 1, &expected).is_err());
        assert!(check_flag_span(bad, &flags(span, 1), 1).is_err());
        assert!(check_flag_span(span, &flags(span, 1), 9).is_err());
        let bad = b::Span {
            words: span.index(7, 0),
            ..span
        };
        assert!(check_span(bad, &words, 1, &expected).is_err());
    }
    #[test]
    fn acceptance_requires_completed_cleanup_and_unretired_owner() {
        assert_eq!(finish_probe(Ok(Ok(7)), true, false).unwrap(), 7);
        for (cleanup, retired) in [(false, false), (true, true), (false, true)] {
            assert!(finish_probe(Ok(Ok(7)), cleanup, retired).is_err());
        }
        assert!(
            finish_probe::<usize>(Ok(Err("dispatch/read/finite failed".into())), true, false)
                .is_err()
        );
        assert!(finish_probe::<usize>(Err(Box::new("panic")), true, false).is_err());
    }
    #[test]
    fn scalar_views_are_aligned_keep_full_owners_and_reject_unsupported_devices() {
        for alignment in [1, 16, 64, 128, 256] {
            validate_storage_alignment(alignment).unwrap();
        }
        for alignment in [0, 3, 512, 1024] {
            assert!(validate_storage_alignment(alignment).is_err());
        }
        let span = b::Span::aligned_fixture(65).unwrap();
        let words = BufferLayout::new(span.words).unwrap().owned_words;
        for token in 0..MAX_BATCH {
            let (first, suffix) = scalar_region(span, span.width, token, words).unwrap();
            assert_eq!(first, span.index(token, 0));
            assert_eq!(first * 4 % 256, 0);
            assert_eq!(words - first - suffix, span.width);
            if token > 0 {
                assert!(first >= span.index(token - 1, 0) + span.width);
            }
        }
        assert!(scalar_region(span, span.width, MAX_BATCH, words).is_err());
        assert!(scalar_region(span, span.width - 1, 0, words).is_err());
        assert!(scalar_region(span, span.width, 0, span.words - 1).is_err());
    }
}
