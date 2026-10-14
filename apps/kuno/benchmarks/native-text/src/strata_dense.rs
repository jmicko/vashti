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
}
impl Kernel {
    fn source(self) -> String {
        let block = crate::quant_matvec::block_layout(self.dtype).unwrap().1;
        format!(
            "// STRATA_DENSE_PACKED\nrequires packed_4x8_integer_dot_product;\nconst DTYPE:u32={}u;const ROWS:u32={}u;const COLUMNS:u32={}u;const ROW_BYTES:u32={}u;const BLOCK_BYTES:u32={block}u;const GROUP_VALUES:u32={}u;const PACKED_WORDS:u32={}u;const PLANE_WORDS:u32={}u;\nstruct Params {{output_first:u32,output_stride:u32,stride:u32,count:u32}}\n@group(0) @binding(0) var<storage,read> weight:array<u32>;\n@group(0) @binding(1) var<storage,read> planes:array<u32>;\n@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(4) var<uniform> p:Params;\n{}",
            self.dtype,
            self.rows,
            self.columns,
            self.row_bytes,
            if self.dtype == 14 { 16 } else { 32 },
            self.columns / 4,
            self.columns / 2 + self.columns / 16,
            include_str!("strata_dense.wgsl")
        )
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
    crate::strata_intel_core::enabled()
        && matches!(dtype, 8 | 12 | 13 | 14 | 20 | 23)
        && columns > 0
        && columns <= 16384
        && columns.is_multiple_of(256)
        && (4 * (columns / 2 + columns / 16) * 4) as u64 <= scratch_bytes
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
            count: 4,
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
) -> Result<()> {
    let (block_values, block_bytes) =
        crate::quant_matvec::block_layout(dtype).ok_or(Error::InvalidBufferLength)?;
    let row_bytes = columns / block_values * block_bytes;
    let layout =
        crate::quant_matvec::validate_layout(dtype, rows, columns, rows * row_bytes, columns, rows)
            .map_err(|_| Error::InvalidBufferLength)?;
    let stride = columns / 2 + columns / 16;
    if !supports(dtype, columns, planes.size_in_used())
        || rows > 16384
        || fault.size_in_used() != 4
        || weight.size_in_used() < layout.total_bytes as u64
        || ![&weight, &planes, &output, &fault]
            .iter()
            .all(|h| h.stream == planes.stream)
        || output_stride < rows
        || output_first
            .checked_add(3 * output_stride)
            .and_then(|n| n.checked_add(rows))
            .is_none_or(|n| n as u64 * 4 > output.size_in_used())
    {
        return Err(Error::InvalidBufferLength);
    }
    let info = [output_first as u32, output_stride as u32, stride as u32, 4];
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
            }),
            CubeCount::Static(rows.div_ceil(4) as u32, 1, 1),
            args,
        );
    }
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [8, 12, 13, 14, 20, 23]
        .into_iter()
        .map(|dtype| {
            Kernel {
                dtype,
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
        .collect()
}
