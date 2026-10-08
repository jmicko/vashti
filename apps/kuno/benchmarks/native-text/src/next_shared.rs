//! Fixed-public count1 port of the accepted resident arithmetic. Uses the
//! original CubeCL stream/owners; no second device, inference API or fallback.
//! Public geometry alone identifies pipelines. Offsets enter the existing
//! transient uniform ledger and never become compilation/cache keys.
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim, CubeElement};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, GgufFile, Result};
use zeroize::{Zeroize, Zeroizing};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MatrixSpec {
    rows: usize,
    columns: usize,
    dtype: u32,
    row_bytes: usize,
}
impl MatrixSpec {
    pub(crate) fn new(rows: usize, columns: usize, dtype: u32) -> Result<Self> {
        let (block, bytes) = crate::quant_matvec::block_layout(dtype)
            .ok_or(Error::InvalidFormat("shared packed dtype"))?;
        if rows == 0
            || rows > 16384
            || columns == 0
            || columns > 16384
            || !columns.is_multiple_of(block)
            || !columns.is_multiple_of(4)
        {
            return Err(Error::InvalidFormat("shared count1 public geometry"));
        }
        let row_bytes = columns / block * bytes;
        if rows * row_bytes > 64 * 1024 * 1024 {
            return Err(Error::LimitExceeded("shared count1 public weight tile"));
        }
        Ok(Self {
            rows,
            columns,
            dtype,
            row_bytes,
        })
    }
    fn source(self) -> String {
        // Reuse the exact arithmetic that passed original Next slabs. Only its
        // binding ABI changes: separate input/output owners and a16-byte INFO.
        let common = include_str!("ornith/common.wgsl")
            .split_once("fn sigmoid")
            .expect("shared packed helper boundary")
            .1;
        let body = include_str!("ornith/matrix_vector.wgsl")
            .replace("if ROUTED {base+=(route[rank]*ROWS)*ROW_BYTES;}", "")
            .replace("if RANKED_INPUT {input+=rank*COLUMNS;}", "")
            .replace("arena4[", "input4[")
            .replace("input4[out/4u][out%4u]=total;", "output[out]=total;");
        format!(
            "const DTYPE:u32={}u; const ROWS:u32={}u; const COLUMNS:u32={}u; const ROW_BYTES:u32={}u;\n\
             struct Params {{ a:u32, b:u32, c:u32, d:u32 }}\n\
             @group(0) @binding(0) var<storage,read> weight:array<u32>;\n\
             @group(0) @binding(1) var<storage,read> input4:array<vec4<f32>>;\n\
             @group(0) @binding(2) var<storage,read_write> output:array<f32>;\n\
             @group(0) @binding(3) var<uniform> p:Params;\nfn sigmoid{common}\n{body}",
            self.dtype, self.rows, self.columns, self.row_bytes
        )
    }
    fn source_fault(self, tail: bool) -> String {
        let mut source = self.source()
            .replace("@group(0) @binding(3) var<uniform> p:Params;",
                "@group(0) @binding(4) var<uniform> p:Params;\n@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;")
            .replace("@builtin(subgroup_size) width:u32)", "@builtin(subgroup_size) width:u32,@builtin(num_workgroups) dispatched:vec3<u32>)")
            .replace("output[out]=total;", "output[out]=total; if !(abs(total)<=3.4028234663852886e38) {atomicOr(&fault[0],1u);}");
        if tail {
            let end = source.rfind('}').expect("shared matrix entrypoint end");
            // Independent of subgroup width: every lane covers a distinct
            // owner index. Never read any active row written by another group.
            source.insert_str(end, "\nlet index=group.x*128u+sg*width+lane;\nlet end=p.d+p.c;\nfor(var i=index;i<arrayLength(&output);i+=128u*dispatched.x) {\nif i<p.d || i>=end {if !(abs(output[i])<=3.4028234663852886e38) {atomicOr(&fault[0],1u);}}\n}\n");
        }
        source
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matrix(MatrixSpec),
    MatrixFault(MatrixSpec, bool),
    Router,
    Gdn([usize; 3]),
}
impl Kernel {
    fn dimension(self) -> CubeDim {
        CubeDim::new_1d(match self {
            Self::Router => 256,
            Self::Gdn(_) => 32,
            Self::Matrix(_) | Self::MatrixFault(_, _) => 128,
        })
    }
    fn source(self) -> String {
        match self {
            Self::Matrix(spec) => spec.source(),
            Self::MatrixFault(spec, tail) => spec.source_fault(tail),
            Self::Gdn([head, key, value]) => format!(
                "const HEAD_DIM:u32={head}u; const KEY_HEADS:u32={key}u; const VALUE_HEADS:u32={value}u;\n{}",
                include_str!("next_gdn.wgsl")
            ),
            Self::Router => {
                let body = include_str!("ornith/router_parallel.wgsl")
                    .replace("arena[p.a+e]", "bitcast<f32>(logits[p.a+e])")
                    .replace("arena[p.a+e+256u]", "bitcast<f32>(logits[p.a+e+256u])");
                format!(
                    "const EXPERTS:u32=512u; const RANKS:u32=10u; const ROUTE_FIRST:u32={}u; const PAIRED:bool=true;\n\
                     struct Params {{ a:u32, b:u32, c:u32, d:u32 }}\n\
                     @group(0) @binding(0) var<storage,read> logits:array<u32>;\n\
                     @group(0) @binding(1) var<storage,read_write> route:array<u32>;\n\
                     @group(0) @binding(2) var<uniform> p:Params;\n{body}",
                    crate::router_gpu::FIRST
                )
            }
        }
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Matrix(spec) | Self::MatrixFault(spec, _) => match spec.dtype {
                12 => "NextDenseQ4K",
                14 => "NextDenseQ6K",
                6 | 8 => "NextDenseQ5Q8",
                _ => "NextDenseFloat",
            },
            Self::Router => "NextRouter",
            Self::Gdn(_) => "NextGdn",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(self.dimension())
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
    fn compile(
        &self,
        _compiler: &mut AutoCompiler,
        _options: &cubecl::WgpuCompilationOptions,
        _mode: ExecutionMode,
        _address: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: self.dimension(),
            debug_info: None,
        })
    }
}

/// Parent has already proved model origin, complete owners, count1 and the
/// whole-token finite/commit/erase gates. This narrows rather than replaces it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matrix(
    client: &ComputeClient<WgpuRuntime>,
    spec: MatrixSpec,
    weights: Handle,
    input: Handle,
    output: Handle,
    input_words: usize,
    output_words: usize,
    input_first: usize,
    output_first: usize,
) -> Result<()> {
    matrix_inner(
        client,
        spec,
        weights,
        input,
        output,
        input_words,
        output_words,
        input_first,
        output_first,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matrix_fault(
    client: &ComputeClient<WgpuRuntime>,
    spec: MatrixSpec,
    weights: Handle,
    input: Handle,
    output: Handle,
    input_words: usize,
    output_words: usize,
    input_first: usize,
    output_first: usize,
    fault: Handle,
    matrix_rows: usize,
    matrix_first: usize,
    tail: bool,
) -> Result<()> {
    matrix_inner(
        client,
        spec,
        weights,
        input,
        output,
        input_words,
        output_words,
        input_first,
        output_first,
        Some((fault, matrix_rows, matrix_first, tail)),
    )
}

#[allow(clippy::too_many_arguments)]
fn matrix_inner(
    client: &ComputeClient<WgpuRuntime>,
    spec: MatrixSpec,
    weights: Handle,
    input: Handle,
    output: Handle,
    input_words: usize,
    output_words: usize,
    input_first: usize,
    output_first: usize,
    fault: Option<(Handle, usize, usize, bool)>,
) -> Result<()> {
    if !input_words.is_multiple_of(4)
        || !input_first.is_multiple_of(4)
        || input_first
            .checked_add(spec.columns)
            .is_none_or(|n| n > input_words)
        || output_first
            .checked_add(spec.rows)
            .is_none_or(|n| n > output_words)
        || input_first > u32::MAX as usize
        || output_first > u32::MAX as usize
        || weights.size_in_used() < (spec.rows * spec.row_bytes).div_ceil(4) as u64 * 4
        || input.size_in_used() < input_words as u64 * 4
        || output.size_in_used() < output_words as u64 * 4
        || weights.stream != input.stream
        || input.stream != output.stream
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut grid = spec.rows.div_ceil(4);
    let (kernel, second) = if let Some((handle, matrix_rows, matrix_first, tail)) = &fault {
        if handle.stream != input.stream
            || output.size_in_used() != output_words as u64 * 4
            || handle.size_in_used() != 4
            || *matrix_rows == 0
            || *matrix_first > output_first
            || matrix_first
                .checked_add(*matrix_rows)
                .is_none_or(|n| n > output_words)
            || output_first
                .checked_add(spec.rows)
                .is_none_or(|n| n > *matrix_first + *matrix_rows)
            || *matrix_rows > u32::MAX as usize
            || *matrix_first > u32::MAX as usize
        {
            return Err(Error::InvalidBufferLength);
        }
        if *tail {
            grid = grid.max(output_words.div_ceil(128));
        }
        (
            Kernel::MatrixFault(spec, *tail),
            *matrix_rows as u64 | ((*matrix_first as u64) << 32),
        )
    } else {
        (Kernel::Matrix(spec), 0)
    };
    if grid > 65535 {
        return Err(Error::LimitExceeded("shared matrix fault grid"));
    }
    let info = MetadataBindingInfo::custom(vec![
        input_first as u64 | ((output_first as u64) << 32),
        second,
    ]);
    let mut args = KernelArguments::new()
        .with_buffer(weights.binding())
        .with_buffer(input.binding())
        .with_buffer(output.binding())
        .with_info(info);
    if let Some((handle, _, _, _)) = fault {
        args = args.with_buffer(handle.binding());
    }
    // SAFETY: original supported packed geometry/alignment, disjoint arena
    // owners and all runtime spans proved above. Tail work reads only outside
    // the complete matrix span; every writer validates its own finite result.
    unsafe {
        client.launch_unchecked(Box::new(kernel), CubeCount::Static(grid as u32, 1, 1), args);
    }

    Ok(())
}

pub(crate) fn router(
    client: &ComputeClient<WgpuRuntime>,
    logits: Handle,
    packet: Handle,
    first: usize,
) -> Result<()> {
    if first
        .checked_add(512)
        .is_none_or(|n| n > logits.size_in_used() as usize / 4)
        || packet.size_in_used() < crate::router_gpu::END as u64 * 4
        || logits.stream != packet.stream
        || first > u32::MAX as usize
    {
        return Err(Error::InvalidBufferLength);
    }
    let args = KernelArguments::new()
        .with_buffer(logits.binding())
        .with_buffer(packet.binding())
        .with_info(MetadataBindingInfo::custom(vec![first as u64, 0]));
    // SAFETY: single256-lane cube caches exactly512 finite-checked original
    // logits and writes only the original paired top10 owning packet interval.
    // The original router bridge still rejects faults/nonfinite inputs.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Router), CubeCount::Static(1, 1, 1), args);
    }
    Ok(())
}

pub(crate) fn probe(client: &ComputeClient<WgpuRuntime>, file: &GgufFile) -> Result<()> {
    let names = [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.6.ffn_down_exps.weight",
        "blk.0.ffn_gate_inp.weight",
        "output.weight",
    ];
    for name in names {
        let t = file
            .tensor(name)
            .ok_or(Error::InvalidFormat("shared original probe tensor"))?;
        let columns = t.info().dimensions()[0] as usize;
        let spec = MatrixSpec::new(7, columns, t.info().dtype().gguf_id())?;
        let mut bytes = vec![0u8; spec.rows * spec.row_bytes];
        t.read_range((3 * spec.row_bytes) as u64, &mut bytes)?;
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        let weights = client.create_from_slice(&bytes);
        let mut x = Zeroizing::new(vec![0f32; columns + 8]);
        let initial = vec![1f32; spec.rows + 19];
        for (i, v) in x[4..4 + columns].iter_mut().enumerate() {
            *v = ((i * 37 % 97) as f32 - 48.) / 97.;
        }
        let input = client.create_from_slice(f32::as_bytes(&x));
        let output = client.create_from_slice(f32::as_bytes(&initial));
        matrix(
            client,
            spec,
            weights,
            input,
            output.clone(),
            x.len(),
            initial.len(),
            4,
            3,
        )?;
        let mut got = client
            .read_one(output)
            .map_err(|_| Error::InvalidFormat("shared public probe read"))?;
        let actual = <f32 as cubecl::prelude::CubeElement>::from_bytes(&got);
        let mut row = Zeroizing::new(vec![0f32; columns]);
        let mut failed = actual.len() != initial.len();
        if !failed {
            failed |=
                actual[..3] != initial[..3] || actual[3 + spec.rows..] != initial[3 + spec.rows..];
            for r in 0..spec.rows {
                t.decode_row((r + 3) as u64, &mut row)?;
                let expected = row
                    .iter()
                    .zip(&x[4..4 + columns])
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32;
                failed |= !actual[r + 3].is_finite()
                    || (actual[r + 3] - expected).abs() > 5e-4 + 5e-5 * expected.abs();
            }
        }
        got.zeroize();
        if failed {
            return Err(Error::InvalidFormat(
                "shared original row/owning numeric probe",
            ));
        }
    }
    crate::router_gpu::probe_shared(client)
        .map_err(|_| Error::InvalidFormat("shared original router probe"))?;
    println!("next_shared_original_rows cases=6 numeric_and_owning_checks=true");
    Ok(())
}

fn gdn_geometry(
    shape: kuno_text::ops::GdnShape,
    spans: crate::gdn_chunk::ChunkSpans,
) -> Result<()> {
    let layout = crate::gdn_gpu::GdnLayout::new(shape)
        .map_err(|_| Error::InvalidFormat("shared GDN geometry"))?;
    spans.validate(layout)?;
    if spans.count != 1
        || shape.head_dim != 128
        || spans.old.column_stride != 1
        || spans.next.column_stride != 1
        || [
            spans.old.base,
            spans.old.head_stride,
            spans.old.key_stride,
            spans.next.base,
            spans.next.head_stride,
            spans.next.key_stride,
            spans.value.base,
            spans.output.base,
        ]
        .iter()
        .any(|n| !n.is_multiple_of(4))
        || spans.words().iter().any(|n| !n.is_multiple_of(4))
    {
        return Err(Error::InvalidFormat("shared count1 GDN vector spans"));
    }
    Ok(())
}

/// Original parent proves eight distinct complete owners before this call.
pub(crate) fn gdn(
    client: &ComputeClient<WgpuRuntime>,
    shape: kuno_text::ops::GdnShape,
    spans: crate::gdn_chunk::ChunkSpans,
    handles: [Handle; 8],
) -> Result<()> {
    gdn_geometry(shape, spans)?;
    if handles
        .iter()
        .zip(spans.words())
        .any(|(h, n)| h.stream != handles[0].stream || h.size_in_used() < n as u64 * 4)
    {
        return Err(Error::InvalidBufferLength);
    }
    let words = [
        spans.query.base,
        spans.key.base,
        spans.value.base,
        spans.controls.base,
        spans.old.base,
        spans.old.head_stride,
        spans.old.key_stride,
        spans.old.column_stride,
        spans.next.base,
        spans.next.head_stride,
        spans.next.key_stride,
        spans.next.column_stride,
        spans.output.base,
        spans.status.base,
        0,
        0,
    ];
    let info = words
        .chunks_exact(2)
        .map(|pair| pair[0] as u64 | ((pair[1] as u64) << 32))
        .collect();
    let mut args = KernelArguments::new();
    for handle in handles {
        args = args.with_buffer(handle.binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(info));
    // SAFETY: count1, head128, complete runtime span proofs, contiguous aligned
    // vec4 columns and distinct original owners. One cube owns one full head;
    // all query/key loads and state/input/output/status stores are inside them.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Gdn([
                shape.head_dim,
                shape.key_heads,
                shape.value_heads,
            ])),
            CubeCount::Static(shape.value_heads as u32, 1, 1),
            args,
        );
    }
    Ok(())
}

pub(crate) fn probe_gdn(
    client: &ComputeClient<WgpuRuntime>,
    shape: kuno_text::ops::GdnShape,
) -> Result<()> {
    use crate::gdn_chunk::{ChunkSpans, MAX_BATCH, StateSpan, TokenSpan};
    let layout = crate::gdn_gpu::GdnLayout::new(shape)
        .map_err(|_| Error::InvalidFormat("shared public GDN shape"))?;
    let token = |n| TokenSpan {
        base: 4,
        stride: n + 4,
        words: 4 + MAX_BATCH * (n + 4) + 4,
    };
    let state = StateSpan {
        base: 4,
        head_stride: shape.head_dim * shape.head_dim,
        key_stride: shape.head_dim,
        column_stride: 1,
        words: layout.state_elements + 8,
    };
    let spans = ChunkSpans {
        count: 1,
        query: token(layout.qk_elements),
        key: token(layout.qk_elements),
        value: token(layout.value_elements),
        controls: token(layout.control_elements),
        old: state,
        next: state,
        output: token(layout.value_elements),
        status: token(layout.value_elements),
    };
    gdn_geometry(shape, spans)?;
    for case in 0..3 {
        let mut inputs: [Zeroizing<Vec<f32>>; 5] =
            std::array::from_fn(|i| Zeroizing::new(vec![3.5; spans.words()[i]]));
        for (i, (view, width)) in [
            (spans.query, layout.qk_elements),
            (spans.key, layout.qk_elements),
            (spans.value, layout.value_elements),
            (spans.controls, layout.control_elements),
        ]
        .into_iter()
        .enumerate()
        {
            for j in 0..width {
                inputs[i][view.base + j] = ((j * 17 % 47) as f32 - 23.) / 211.;
            }
        }
        for h in 0..shape.value_heads {
            inputs[3][spans.controls.base + h] = -0.03;
            inputs[3][spans.controls.base + shape.value_heads + h] =
                if case == 1 { 0.0 } else { 0.41 };
        }
        for i in 0..layout.state_elements {
            inputs[4][state.base + i] = ((i * 31 % 113) as f32 - 56.) / 97.;
        }
        if case == 2 {
            inputs[4][state.base + 1] = f32::NAN;
            inputs[3][spans.controls.base + shape.value_heads + 1] = f32::INFINITY;
        }
        let handles: [Handle; 5] =
            std::array::from_fn(|i| client.create_from_slice(f32::as_bytes(&inputs[i])));
        let initial_next = Zeroizing::new(vec![1.0f32; spans.next.words]);
        let initial_out = Zeroizing::new(vec![1.0f32; spans.output.words]);
        let initial_status = Zeroizing::new(vec![0x51515151u32; spans.status.words]);
        let make_outputs = || {
            [
                client.create_from_slice(f32::as_bytes(&initial_next)),
                client.create_from_slice(f32::as_bytes(&initial_out)),
                client.create_from_slice(u32::as_bytes(&initial_status)),
            ]
        };
        let reference = make_outputs();
        let vector = make_outputs();
        // Independent original checked scalar implementation, including invalid
        // column zeroing. Nonzero prefix/tails prove the complete write interval.
        unsafe {
            crate::gdn_chunk::gdn_single_streaming::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.workgroups_x, 1, 1),
                CubeDim::new_1d(64),
                cubecl::prelude::ArrayArg::from_raw_parts(handles[0].clone(), spans.query.words),
                cubecl::prelude::ArrayArg::from_raw_parts(handles[1].clone(), spans.key.words),
                cubecl::prelude::ArrayArg::from_raw_parts(handles[2].clone(), spans.value.words),
                cubecl::prelude::ArrayArg::from_raw_parts(handles[3].clone(), spans.controls.words),
                cubecl::prelude::ArrayArg::from_raw_parts(handles[4].clone(), spans.old.words),
                cubecl::prelude::ArrayArg::from_raw_parts(reference[0].clone(), spans.next.words),
                cubecl::prelude::ArrayArg::from_raw_parts(reference[1].clone(), spans.output.words),
                cubecl::prelude::ArrayArg::from_raw_parts(reference[2].clone(), spans.status.words),
                1,
                spans.query.base,
                spans.query.stride,
                spans.key.base,
                spans.key.stride,
                spans.value.base,
                spans.value.stride,
                spans.controls.base,
                spans.controls.stride,
                state.base,
                state.head_stride,
                state.key_stride,
                1,
                state.base,
                state.head_stride,
                state.key_stride,
                1,
                spans.output.base,
                spans.output.stride,
                spans.status.base,
                spans.status.stride,
                shape.head_dim,
                shape.key_heads,
                shape.value_heads,
            );
        }
        let all = std::array::from_fn(|i| {
            if i < 5 {
                handles[i].clone()
            } else {
                vector[i - 5].clone()
            }
        });
        gdn(client, shape, spans, all)?;
        for i in 0..3 {
            let mut expected = client
                .read_one(reference[i].clone())
                .map_err(|_| Error::InvalidFormat("shared GDN reference read"))?;
            let mut actual = client
                .read_one(vector[i].clone())
                .map_err(|_| Error::InvalidFormat("shared GDN vector read"))?;
            let failed = if i == 2 {
                expected != actual
            } else {
                f32::from_bytes(&expected)
                    .iter()
                    .zip(f32::from_bytes(&actual))
                    .any(|(&a, &b)| {
                        !a.is_finite() || !b.is_finite() || (a - b).abs() > 5e-5 + 1e-4 * a.abs()
                    })
                    || expected.len() != actual.len()
            };
            expected.zeroize();
            actual.zeroize();
            if failed {
                return Err(Error::InvalidFormat("shared GDN numeric/owning probe"));
            }
        }
        let mut old = client
            .read_one(handles[4].clone())
            .map_err(|_| Error::InvalidFormat("shared GDN old owner read"))?;
        let changed = &old[..] != f32::as_bytes(&inputs[4]);
        old.zeroize();
        if changed {
            return Err(Error::InvalidFormat(
                "shared GDN changed immutable old owner",
            ));
        }
    }
    println!("next_shared_gdn cases=3 numeric_and_owning_checks=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_tail_grid_reads_every_inactive_owner_index_and_no_active_writer() {
        for width in [16usize, 32] {
            for (rows, first, total, words) in [
                (7usize, 3usize, 7usize, 26usize),
                (11, 7, 75, 512),
                (1, 0, 1, 8192),
            ] {
                let grid = rows.div_ceil(4).max(words.div_ceil(128));
                let mut seen = vec![0usize; words];
                for group in 0..grid {
                    for sg in 0..128 / width {
                        for lane in 0..width {
                            for i in (group * 128 + sg * width + lane..words).step_by(grid * 128) {
                                if i < first || i >= first + total {
                                    seen[i] += 1;
                                }
                            }
                        }
                    }
                }
                for (i, n) in seen.into_iter().enumerate() {
                    assert_eq!(n, usize::from(i < first || i >= first + total));
                }
            }
        }
        let spec = MatrixSpec::new(7, 256, 12).unwrap();
        let source = spec.source_fault(true);
        assert!(source.contains("@binding(3) var<storage,read_write> fault"));
        assert!(source.contains("@binding(4) var<uniform> p"));
        assert!(
            source.contains("num_workgroups")
                && !spec.source_fault(false).contains("let index=group.x")
        );
    }

    #[test]
    fn all_original_formats_and_router_validate_without_private_pipeline_keys() {
        for dtype in [0, 1, 30, 6, 8, 12, 14] {
            let spec = MatrixSpec::new(7, 256, dtype).unwrap();
            for k in [
                Kernel::Matrix(spec),
                Kernel::MatrixFault(spec, false),
                Kernel::MatrixFault(spec, true),
            ] {
                let module = wgpu::naga::front::wgsl::parse_str(&k.source()).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::SUBGROUP
                        | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
                )
                .validate(&module)
                .unwrap();
            }
        }
        let module = wgpu::naga::front::wgsl::parse_str(&Kernel::Router.source()).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::SUBGROUP,
        )
        .validate(&module)
        .unwrap();
        let module =
            wgpu::naga::front::wgsl::parse_str(&Kernel::Gdn([128, 16, 32]).source()).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert!(MatrixSpec::new(0, 256, 12).is_err());
        assert!(MatrixSpec::new(7, 257, 12).is_err());
        assert!(MatrixSpec::new(7, 256, 999).is_err());
        assert!(MatrixSpec::new(16385, 256, 12).is_err());
    }
}
