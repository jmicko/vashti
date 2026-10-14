//! Original packed dense matrices with Strata's single Q8_1 input planes.
//! Scratch is explicitly borrowed from the target's authentic private arena.
//! Its parent controls the reuse interval; no allocation or new weight copy.
use crate::quant_matvec::matmul::BatchSpans;
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::{
    prelude::{ComputeClient, CubeCount, CubeDim},
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
use kuno_text::{Error, Result};
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Kernel {
    dtype: u32,
    rows: usize,
    columns: usize,
    row_bytes: usize,
    members: usize,
    indexed: bool,
}
impl Kernel {
    fn source(self) -> String {
        let block = crate::quant_matvec::block_layout(self.dtype).unwrap().1;
        let mut source = format!(
            "// STRATA_DENSE_PACKED\nrequires packed_4x8_integer_dot_product;\nconst MEMBERS:u32={}u;const DTYPE:u32={}u;const ROWS:u32={}u;const COLUMNS:u32={}u;const ROW_BYTES:u32={}u;const BLOCK_BYTES:u32={block}u;const GROUP_VALUES:u32={}u;const PACKED_WORDS:u32={}u;const PLANE_WORDS:u32={}u;\nstruct Params {{output_first:u32,output_stride:u32,stride:u32,count:u32}}\n@group(0) @binding(0) var<storage,read> weight:array<u32>;\n@group(0) @binding(1) var<storage,read> planes:array<u32>;\n@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(4) var<uniform> p:Params;\n{}",
            self.members,
            self.dtype,
            self.rows,
            self.columns,
            self.row_bytes,
            if self.dtype == 14 { 16 } else { 32 },
            self.columns / 4,
            self.columns / 2 + self.columns / 16,
            include_str!("strata_dense.wgsl")
        );
        if self.indexed {
            source = source.replace(
                "count:u32}",
                "count:u32,source_first:u32,source_rows:u32,subset_first:u32,reserved:u32}",
            );
            source=source.replace("@group(0) @binding(4) var<uniform> p:Params;","@group(0) @binding(4) var<storage,read> row_ids:array<u32>;\n@group(0) @binding(5) var<uniform> p:Params;");
            source=source.replace("arrayLength(&weight)*4u<ROWS*ROW_BYTES","arrayLength(&weight)*4u<p.source_rows*ROW_BYTES || p.subset_first>arrayLength(&row_ids) || ROWS>arrayLength(&row_ids)-p.subset_first");
            source=source.replace("let row=g.x*4u+sg;var sum=vec4<f32>(0.0);",r#"let ordinal=g.x*4u+sg;var row=0u;var valid_row=ordinal<ROWS;
            if valid_row {let selected=row_ids[p.subset_first+ordinal];if selected<p.source_first || selected-p.source_first>=p.source_rows {valid_row=false;if lane==0u {atomicOr(&fault[0],8u);}}else{row=selected-p.source_first;}}
            var sum=vec4<f32>(0.0);"#);
            source = source.replace("if row<ROWS", "if valid_row");
            // Bound writes against the complete original source page interval.
            source = source
                .replace("p.output_stride<ROWS", "p.output_stride<p.source_rows")
                .replace(
                    "ROWS>arrayLength(&output)-p.output_first",
                    "p.source_rows>arrayLength(&output)-p.output_first",
                )
                .replace("-p.output_first-ROWS", "-p.output_first-p.source_rows");
        }
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "StrataPackedDenseQ8"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self)
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_info: None,
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(128),
        })
    }
}
pub(crate) fn supports(dtype: u32, columns: usize, scratch_bytes: u64) -> bool {
    supports_count(dtype, columns, scratch_bytes, 4)
}
pub(crate) fn supports_count(dtype: u32, columns: usize, scratch_bytes: u64, count: usize) -> bool {
    crate::strata_intel_core::enabled()
        && matches!(dtype, 8 | 12 | 13 | 14 | 20 | 23)
        && columns > 0
        && columns <= 16384
        && columns.is_multiple_of(256)
        && (1..=4).contains(&count)
        && (count * (columns / 2 + columns / 16) * 4) as u64 <= scratch_bytes
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
    client: &ComputeClient<WgpuRuntime>,
    columns: usize,
    input: Handle,
    planes: Handle,
    fault: Handle,
    spans: &BatchSpans,
) -> Result<()> {
    let spec = crate::q6_integer_matvec::Spec::new(1, columns)?;
    crate::q6_integer_matvec::batch::prepare(
        client,
        spec,
        input,
        planes,
        fault,
        crate::q6_integer_matvec::batch::PrepareSpans {
            count: spans.count(),
            input_first: spans.input_first(),
            input_stride: spans.input_stride(),
            planes_first: 0,
            planes_stride: spec.plane_words(),
            fault_first: 0,
        },
    )
}
static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn calls() -> u64 {
    CALLS.load(std::sync::atomic::Ordering::Relaxed)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn project(
    client: &ComputeClient<WgpuRuntime>,
    dtype: u32,
    rows: usize,
    columns: usize,
    weight: Handle,
    planes: Handle,
    output: Handle,
    fault: Handle,
    output_first: usize,
    output_stride: usize,
    count: usize,
) -> Result<()> {
    let (block_values, block_bytes) =
        crate::quant_matvec::block_layout(dtype).ok_or(Error::InvalidBufferLength)?;
    let row_bytes = columns / block_values * block_bytes;
    let layout =
        crate::quant_matvec::validate_layout(dtype, rows, columns, rows * row_bytes, columns, rows)
            .map_err(|_| Error::InvalidBufferLength)?;
    let stride = columns / 2 + columns / 16;
    if !supports_count(dtype, columns, planes.size_in_used(), count)
        || rows > 16384
        || fault.size_in_used() != 4
        || weight.size_in_used() < layout.total_bytes as u64
        || ![&weight, &planes, &output, &fault]
            .iter()
            .all(|h| h.stream == planes.stream)
        || output_stride < rows
        || output_first
            .checked_add((count - 1) * output_stride)
            .and_then(|n| n.checked_add(rows))
            .is_none_or(|n| n as u64 * 4 > output.size_in_used())
    {
        return Err(Error::InvalidBufferLength);
    }
    let info = [
        output_first as u32,
        output_stride as u32,
        stride as u32,
        count as u32,
    ];
    let args = KernelArguments::new()
        .with_buffer(weight.binding())
        .with_buffer(planes.binding())
        .with_buffer(output.binding())
        .with_buffer(fault.binding())
        .with_info(MetadataBindingInfo::custom(
            info.chunks_exact(2)
                .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                .collect(),
        ));
    // SAFETY: the caller checked distinct authentic arena roots and live original
    // packed tensor tiles; checked byte extents cover every active output/plane.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel {
                dtype,
                rows,
                columns,
                row_bytes: layout.row_bytes,
                members: if count == 1 { 1 } else { 4 },
                indexed: false,
            }),
            CubeCount::Static(rows.div_ceil(4) as u32, 1, 1),
            args,
        );
    }
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut result = [1, 4]
        .into_iter()
        .flat_map(|members| {
            [8, 12, 13, 14, 20, 23]
                .into_iter()
                .map(move |dtype| (members, dtype))
        })
        .map(|(members, dtype)| {
            Kernel {
                dtype,
                members,
                indexed: false,
                rows: 32,
                columns: 2560,
                row_bytes: {
                    let (values, bytes) = crate::quant_matvec::block_layout(dtype).unwrap();
                    2560 / values * bytes
                },
            }
            .compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &cubecl::WgpuCompilationOptions::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    result.push(
        Kernel {
            dtype: 8,
            rows: 32,
            columns: 2560,
            row_bytes: 2720,
            members: 1,
            indexed: true,
        }
        .compile(
            &mut AutoCompiler::Wgsl(Default::default()),
            &Default::default(),
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap(),
    );
    result
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn project_indexed(
    client: &ComputeClient<WgpuRuntime>,
    source_first: usize,
    source_rows: usize,
    subset_first: usize,
    subset_rows: usize,
    columns: usize,
    weight: Handle,
    planes: Handle,
    output: Handle,
    fault: Handle,
    ids: Handle,
) -> Result<()> {
    let row_bytes = columns / 32 * 34;
    if !supports_count(8, columns, planes.size_in_used(), 1)
        || source_rows > 16384
        || subset_rows == 0
        || subset_rows > source_rows
        || fault.size_in_used() != 4
        || weight.size_in_used() < (source_rows * row_bytes) as u64
        || subset_first
            .checked_add(subset_rows)
            .is_none_or(|n| n as u64 * 4 > ids.size_in_used())
        || source_first
            .checked_add(source_rows)
            .is_none_or(|n| n as u64 * 4 > output.size_in_used())
        || [&weight, &planes, &output, &fault, &ids]
            .iter()
            .any(|h| h.stream != planes.stream)
    {
        return Err(Error::InvalidBufferLength);
    }
    let stride = columns / 2 + columns / 16;
    let info = [
        source_first as u32,
        output.size_in_used() as u32 / 4,
        stride as u32,
        1,
        source_first as u32,
        source_rows as u32,
        subset_first as u32,
        0,
    ];
    let args = KernelArguments::new()
        .with_buffer(weight.binding())
        .with_buffer(planes.binding())
        .with_buffer(output.binding())
        .with_buffer(fault.binding())
        .with_buffer(ids.binding())
        .with_info(MetadataBindingInfo::custom(
            info.chunks_exact(2)
                .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                .collect(),
        ));
    // SAFETY: the authenticated public vocabulary proof and complete page/input/
    // output roots are checked by the owning MTP caller; shader checks every ID.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel {
                dtype: 8,
                rows: subset_rows,
                columns,
                row_bytes,
                members: 1,
                indexed: true,
            }),
            CubeCount::Static(subset_rows.div_ceil(4) as u32, 1, 1),
            args,
        );
    }
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
