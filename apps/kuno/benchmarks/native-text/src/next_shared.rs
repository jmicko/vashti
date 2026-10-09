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
    /// Exact decoder set in ornith/matrix_vector.wgsl. The generic packed
    /// decoder supports additional types, but that does not add shader support.
    pub(crate) const fn supports_dtype(dtype: u32) -> bool {
        matches!(dtype, 0 | 1 | 6 | 8 | 12 | 14 | 30)
    }
    pub(crate) fn new(rows: usize, columns: usize, dtype: u32) -> Result<Self> {
        if !Self::supports_dtype(dtype) {
            return Err(Error::InvalidFormat("shared packed dtype"));
        }
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
        let mut body = include_str!("ornith/matrix_vector.wgsl")
            .replace("if ROUTED {base+=(route[rank]*ROWS)*ROW_BYTES;}", "")
            .replace("if RANKED_INPUT {input+=rank*COLUMNS;}", "")
            .replace("arena4[", "input4[")
            .replace("input4[out/4u][out%4u]=total;", "output[out]=total;");
        if self.dtype == 12 && self.columns <= 4096 {
            // Same exact dequant/dot/reduction schedule, sharing only unchanged
            // F32 input values among rows. The public 16KiB bound fits the
            // baseline native device limit. Skip input loads in groups without
            // an active row, including extra full-owner finite-scan groups.
            // num_subgroups describes the whole workgroup; keep the original
            // barrier unconditional, independent of subgroup uniformity analysis.
            body = body
                .replace("input4[", "cached_input[")
                .replace("(input+", "(")
                .replace("@builtin(workgroup_id) group:vec3<u32>,", "@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(num_subgroups) subgroups:u32,")
                .replace("    let row=", "    if group.x*subgroups<ROWS {\n        for(var i=local;i<COLUMNS/4u;i+=128u) {cached_input[i]=input4[p.a/4u+i];}\n    }\n    workgroupBarrier();\n    let row=");
            body = format!(
                "var<workgroup> cached_input:array<vec4<f32>,{}>;\n{body}",
                self.columns / 4
            );
        }
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
    GrNorm(usize),
}
impl Kernel {
    fn dimension(self) -> CubeDim {
        CubeDim::new_1d(match self {
            Self::Router => 256,
            Self::Gdn(_) => 128,
            Self::GrNorm(_) => 64,
            Self::Matrix(_) | Self::MatrixFault(_, _) => 128,
        })
    }
    fn source(self) -> String {
        match self {
            Self::GrNorm(width) => format!(
                "const WIDTH:u32={width}u;\n{}",
                include_str!("next_gr_norm.wgsl")
            ),
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
            Self::GrNorm(_) => "NextGrRms",
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
    fn partitioned_gdn_writes_every_state_column_once() {
        let mut seen = vec![0u32; 128 * 128];
        for lane in 0..128 {
            let column = (lane % 32) * 4;
            let start = (lane / 32) * 32;
            for row in start..start + 32 {
                for c in column..column + 4 {
                    seen[row * 128 + c] += 1;
                }
            }
        }
        assert!(seen.into_iter().all(|n| n == 1));
        assert_eq!(Kernel::Gdn([128, 16, 32]).dimension(), CubeDim::new_1d(128));
        // Fixed shared arrays: q/k, partial vec4 sums/masks, final masks.
        assert_eq!(2 * 128 * 4 + 2 * 128 * 16 + 32 * 16, 5632);
    }
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
    fn q4_input_sharing_skips_only_groups_without_active_rows() {
        // Both native subgroup sizes admitted by the protected device owner.
        // Include partial groups and grids enlarged for full owning tail checks.
        for width in [16usize, 32] {
            let subgroups = 128 / width;
            for rows in [1usize, 7, 8, 9, 31, 640, 2560] {
                for words in [rows, rows * 8, 8192usize.max(rows)] {
                    let grid = rows.div_ceil(4).max(words.div_ceil(128));
                    for group in 0..grid {
                        let load = group * subgroups < rows;
                        assert_eq!(
                            load,
                            (0..subgroups).any(|sg| group * (128 / width) + sg < rows)
                        );
                        if load {
                            for columns in [256usize, 2560, 4096] {
                                let mut loaded = vec![0usize; columns / 4];
                                for local in 0..128 {
                                    for i in (local..columns / 4).step_by(128) {
                                        loaded[i] += 1;
                                    }
                                }
                                assert!(loaded.into_iter().all(|n| n == 1));
                            }
                        }
                    }
                }
            }
        }
        for columns in [256, 2560, 4096, 8192] {
            let spec = MatrixSpec::new(7, columns, 12).unwrap();
            for source in [
                spec.source(),
                spec.source_fault(false),
                spec.source_fault(true),
            ] {
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::SUBGROUP
                        | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
                )
                .validate(&module)
                .unwrap();
                if columns <= 4096 {
                    assert!(source.contains("@builtin(num_subgroups) subgroups:u32"));
                    assert!(source.contains("if group.x*subgroups<ROWS {"));
                    // The guard ends before the only barrier. The later full
                    // owner tail scan remains outside the guarded preload.
                    assert!(source.contains("}\n    }\n    workgroupBarrier();\n    let row="));
                    assert_eq!(source.matches("workgroupBarrier()").count(), 1);
                    if let Some(tail) = source.find("let index=group.x") {
                        assert!(tail > source.find("workgroupBarrier()").unwrap());
                    }
                } else {
                    assert!(!source.contains("cached_input"));
                    assert!(!source.contains("num_subgroups"));
                }
            }
        }
    }

    #[test]
    fn shared_matrix_dispatch_rejects_types_without_a_shader_decoder() {
        for dtype in [0, 1, 30, 6, 8, 12, 14] {
            assert!(MatrixSpec::supports_dtype(dtype));
            assert!(MatrixSpec::new(7, 256, dtype).is_ok());
        }
        // These remain valid generic packed formats. Their presence in
        // block_layout must never opt them into the shader's F32 else branch.
        for dtype in [13, 18, 20, 21, 22, 23, 42] {
            assert!(crate::quant_matvec::block_layout(dtype).is_some());
            assert!(!MatrixSpec::supports_dtype(dtype));
            assert!(MatrixSpec::new(7, 256, dtype).is_err());
        }
        assert!(!MatrixSpec::supports_dtype(u32::MAX));
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

static GR_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn gr_calls() -> u64 {
    GR_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}
static GR_SELECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(crate) fn select_gr(selected: bool) {
    GR_SELECTED.store(selected, std::sync::atomic::Ordering::Release);
}
pub(crate) fn gr_selected() -> bool {
    GR_SELECTED.load(std::sync::atomic::Ordering::Acquire)
}
/// Parent proves actual full owners, physical aliases and model origin. This
/// seam independently narrows the runtime logical views without new owners.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gr_norm(
    client: &ComputeClient<WgpuRuntime>,
    handles: [Handle; 4],
    count: usize,
    streams: usize,
    width: usize,
    input_first: usize,
    epsilon: f32,
) -> Result<()> {
    if !(1..=8).contains(&count)
        || streams == 0
        || streams > 32
        || width == 0
        || width > 16384
        || !width.is_multiple_of(4)
        || !input_first.is_multiple_of(4)
        || input_first > u32::MAX as usize
    {
        return Err(Error::InvalidBufferLength);
    }
    let row = streams
        .checked_mul(width)
        .ok_or(Error::ArithmeticOverflow)?;
    let input_end = count
        .checked_mul(row)
        .and_then(|n| n.checked_add(input_first))
        .ok_or(Error::ArithmeticOverflow)?;
    let lengths = [input_end, row, count * row, count * streams];
    for (h, n) in handles.iter().zip(lengths) {
        if h.stream != handles[0].stream || h.size_in_used() < n as u64 * 4 {
            return Err(Error::InvalidBufferLength);
        }
    }
    let info = MetadataBindingInfo::custom(vec![
        count as u64 | ((input_first as u64) << 32),
        row as u64,
        row as u64,
        streams as u64 | ((epsilon.to_bits() as u64) << 32),
    ]);
    let [input, gamma, output, status] = handles;
    let args = KernelArguments::new()
        .with_buffer(input.binding())
        .with_buffer(gamma.binding())
        .with_buffer(output.binding())
        .with_buffer(status.binding())
        .with_info(info);
    // SAFETY: parent actual full-owner proof and checked aligned logical spans;
    // disjoint workgroups write one original active stream and status each.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::GrNorm(width)),
            CubeCount::Static(streams as u32, count as u32, 1),
            args,
        );
    }
    GR_CALLS
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |n| n.checked_add(1),
        )
        .map_err(|_| Error::ArithmeticOverflow)?;
    Ok(())
}

struct GrScrub(cubecl::bytes::Bytes);
impl Drop for GrScrub {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
pub(crate) fn probe_gr(
    client: &ComputeClient<WgpuRuntime>,
    retired: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use crate::{private_arena::PrivateArena, resident_util as u};
    let mut a = PrivateArena::new_transient_on_configured(client.clone(), 2 * 1024 * 1024, retired)
        .map_err(u::fault)?;
    const WIDTH: usize = 2560;
    const STREAMS: usize = 4;
    const ROW: usize = WIDTH * STREAMS;
    let input = a.allocate_zeroed(8 * ROW + 8).map_err(u::fault)?;
    let gamma = a.allocate_zeroed(ROW).map_err(u::fault)?;
    let output = a.allocate_zeroed(8 * ROW + 8).map_err(u::fault)?;
    let status = a.allocate_zeroed(8 * STREAMS + 8).map_err(u::fault)?;
    for count in [1, 3, 8] {
        for case in 0..6 {
            let mut x =
                Zeroizing::new(vec![-19f32; a.layout(input).map_err(u::fault)?.owned_words]);
            let mut g = Zeroizing::new(vec![1f32; a.layout(gamma).map_err(u::fault)?.owned_words]);
            for i in 0..count * ROW {
                x[4 + i] = (((i * 37 % 97) as f32) - 48.) / 97.;
                if case == 5 {
                    x[4 + i] *= 1e30;
                }
            }
            for i in 0..ROW {
                g[i] = 0.5 + (i % 13) as f32 / 13.;
            }
            if case == 1 {
                x[4] = f32::NAN;
            }
            if case == 2 {
                g[WIDTH + WIDTH - 1] = f32::INFINITY;
            }
            let epsilon = match case {
                3 => 0.,
                4 => f32::NAN,
                _ => 1e-6,
            };
            let mut initial = Zeroizing::new(vec![
                3.5f32;
                a.layout(output).map_err(u::fault)?.owned_words
            ]);
            let flags = Zeroizing::new(vec![9u32; a.layout(status).map_err(u::fault)?.owned_words]);
            for (id, bytes) in [
                (input, f32::as_bytes(&x)),
                (gamma, f32::as_bytes(&g)),
                (output, f32::as_bytes(&initial)),
                (status, u32::as_bytes(&flags)),
            ] {
                client
                    .write_private(a.handle(id).map_err(u::fault)?.clone(), bytes.to_vec())
                    .map_err(u::fault)?;
            }
            let owners = [input, gamma, output, status]
                .map(|id| a.handle(id).map_err(u::fault).cloned())
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            crate::resident_gr::batch::physical(&a, &owners)?;
            gr_norm(
                client,
                [
                    u::handle(&a, input)?,
                    u::handle(&a, gamma)?,
                    u::handle(&a, output)?,
                    u::handle(&a, status)?,
                ],
                count,
                STREAMS,
                WIDTH,
                4,
                epsilon,
            )?;
            let got = GrScrub(
                client
                    .read_one(a.handle(output).map_err(u::fault)?.clone())
                    .map_err(u::fault)?,
            );
            let flags_got = GrScrub(
                client
                    .read_one(a.handle(status).map_err(u::fault)?.clone())
                    .map_err(u::fault)?,
            );
            let check = (|| -> Result<()> {
                let values = f32::from_bytes(&got.0);
                let flags = u32::from_bytes(&flags_got.0);
                if values.len() != initial.len()
                    || flags.len() != a.layout(status).map_err(u::fault)?.owned_words
                {
                    return Err(Error::InvalidBufferLength);
                }
                for token in 0..count {
                    for stream in 0..STREAMS {
                        let base = token * ROW + stream * WIDTH;
                        let valid = epsilon.is_finite()
                            && epsilon > 0.
                            && x[4 + base..4 + base + WIDTH].iter().all(|v| v.is_finite())
                            && g[stream * WIDTH..(stream + 1) * WIDTH]
                                .iter()
                                .all(|v| v.is_finite());
                        if flags[token * STREAMS + stream] != u32::from(valid) {
                            return Err(Error::InvalidFormat("shared GR status probe"));
                        }
                        let sum = x[4 + base..4 + base + WIDTH]
                            .iter()
                            .map(|&v| (v as f64) * (v as f64))
                            .sum::<f64>();
                        let divisor = (sum / WIDTH as f64 + epsilon as f64).sqrt();
                        for j in 0..WIDTH {
                            let expected = if valid {
                                (x[4 + base + j] as f64 / divisor * g[stream * WIDTH + j] as f64)
                                    as f32
                            } else {
                                0.
                            };
                            let v = values[base + j];
                            if !v.is_finite() || (v - expected).abs() > 2e-5 + 2e-5 * expected.abs()
                            {
                                return Err(Error::InvalidFormat("shared GR numeric probe"));
                            }
                        }
                    }
                }
                if values[count * ROW..].iter().any(|&v| v != 3.5)
                    || flags[count * STREAMS..].iter().any(|&v| v != 9)
                {
                    return Err(Error::InvalidFormat("shared GR inactive probe"));
                }
                Ok(())
            })();
            drop(got);
            drop(flags_got);
            initial.zeroize();
            check?;
            cubecl::future::block_on(client.sanitize_transient_io_audited()).map_err(u::fault)?;
        }
    }
    a.close().map_err(u::fault)?;
    println!(
        "next_shared_gr_rms counts=1,3,8 cases=6 numeric_finite_status_and_owning_checks=true"
    );
    Ok(())
}
#[cfg(test)]
mod gr_tests {
    use super::*;
    #[test]
    fn subgroup_gr_shader_and_runtime_uniform_validate_offline() {
        let source = Kernel::GrNorm(2560).source();
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert!(source.contains("workgroupBarrier()"));
        assert!(source.contains("output_goods"));
        assert!(!source.contains("arrayLength"));
    }
}
