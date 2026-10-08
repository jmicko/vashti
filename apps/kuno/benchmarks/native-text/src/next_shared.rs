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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matrix(MatrixSpec),
    Router,
}
impl Kernel {
    fn dimension(self) -> CubeDim {
        CubeDim::new_1d(if matches!(self, Self::Router) {
            256
        } else {
            128
        })
    }
    fn source(self) -> String {
        match self {
            Self::Matrix(spec) => spec.source(),
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
    let info =
        MetadataBindingInfo::custom(vec![input_first as u64 | ((output_first as u64) << 32), 0]);
    let args = KernelArguments::new()
        .with_buffer(weights.binding())
        .with_buffer(input.binding())
        .with_buffer(output.binding())
        .with_info(info);
    // SAFETY: exact supported packing, aligned complete vector reads and all
    // writes proved above; parent owns disjoint buffers and original live lease.
    // Public grid covers every row even on16-lane subgroups; row guard rejects
    // extra cubes. INFO's existing Drop and stream audit erase offsets.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Matrix(spec)),
            CubeCount::Static(spec.rows.div_ceil(4) as u32, 1, 1),
            args,
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_original_formats_and_router_validate_without_private_pipeline_keys() {
        for dtype in [0, 1, 30, 6, 8, 12, 14] {
            let k = Kernel::Matrix(MatrixSpec::new(7, 256, dtype).unwrap());
            let module = wgpu::naga::front::wgsl::parse_str(&k.source()).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::SUBGROUP
                    | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
            )
            .validate(&module)
            .unwrap();
        }
        let module = wgpu::naga::front::wgsl::parse_str(&Kernel::Router.source()).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::SUBGROUP,
        )
        .validate(&module)
        .unwrap();
        assert!(MatrixSpec::new(0, 256, 12).is_err());
        assert!(MatrixSpec::new(7, 257, 12).is_err());
        assert!(MatrixSpec::new(7, 256, 999).is_err());
        assert!(MatrixSpec::new(16385, 256, 12).is_err());
    }
}
