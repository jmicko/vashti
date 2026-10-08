//! Packed vector/subgroup experts on the original protected count1 stream.
//! Public geometry alone specializes shaders; all routes/offsets use the
//! original owning map and transient INFO. No new scratch or weight copies.
use crate::quant_matvec::grouped_moe::{Geometry, Spans};
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Kernel {
    down: bool,
    dtype: u32,
    row_bytes: usize,
    first_slot: usize,
    slots: usize,
}
impl Kernel {
    fn source(self) -> String {
        // Exactly the original packed-vector row implementation already
        // independently checked against original GGUF rows. Each projection
        // gets its own immutable weight binding and helper namespace.
        let common = include_str!("ornith/common.wgsl")
            .split_once("fn sigmoid")
            .unwrap()
            .1;
        let vector = include_str!("ornith/matrix_vector.wgsl");
        let helpers = vector.split_once("@compute").unwrap().0;
        let arithmetic = vector
            .split_once("    var sum=0.0;")
            .unwrap()
            .1
            .split_once("    let total=subgroupAdd(sum);")
            .unwrap()
            .0;
        let project = |name: &str, rows: usize, columns: usize, dtype: u32, row_bytes: usize| {
            let mut source = format!(
                "fn sigmoid{common}\n{helpers}\nfn project_{name}(row:u32,base:u32,input:u32,lane:u32,width:u32)->f32 {{\nvar sum=0.0;{arithmetic}\nreturn subgroupAdd(sum);\n}}\n"
            );
            source = source
                .replace("weight[", &format!("{name}["))
                .replace("arena4[", "input4[")
                .replace("DTYPE", &format!("{dtype}u"))
                .replace("ROWS", &format!("{rows}u"))
                .replace("COLUMNS", &format!("{columns}u"))
                .replace("ROW_BYTES", &format!("{row_bytes}u"));
            for helper in [
                "sigmoid",
                "silu",
                "wb",
                "wh",
                "signed",
                "word_byte",
                "q4_scale",
                "q4_minimum",
                "packed",
                "bytes4",
                "word_at",
            ] {
                source = source.replace(&format!("{helper}("), &format!("{name}_{helper}("));
            }
            source
        };
        let header = format!(
            "const FIRST_SLOT:u32={}u;const SLOTS:u32={}u;\n\
            struct Params {{a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}}\n\
            @group(0) @binding(3) var<storage,read> map:array<u32>;\n\
            @group(0) @binding(5) var<storage,read_write> status:array<u32>;\n\
            @group(0) @binding(6) var<uniform> p:Params;\n\
            fn valid_group(group:u32)->bool {{\n\
              if map[p.a]!=1u || map[p.a+1u]!=0u || map[p.a+2u]!=1u {{return false;}}\n\
              let groups=map[p.a+3u];if groups==0u || groups>10u || group>=groups {{return false;}}\n\
              let slot=map[p.a+4u+group];\n\
              return map[p.a+84u+group]==1u && slot>=FIRST_SLOT && slot-FIRST_SLOT<SLOTS && map[p.a+164u+group*8u]<10u;\n\
            }}\n",
            self.first_slot, self.slots
        );
        if self.down {
            let header = header
                .replace(
                    "@binding(3) var<storage,read> map",
                    "@binding(2) var<storage,read> map",
                )
                .replace(
                    "@binding(5) var<storage,read_write> status",
                    "@binding(4) var<storage,read_write> status",
                )
                .replace("@binding(6) var<uniform>", "@binding(5) var<uniform>");
            format!(
                "{header}\n\
                @group(0) @binding(0) var<storage,read> down:array<u32>;\n\
                @group(0) @binding(1) var<storage,read> input4:array<vec4<f32>>;\n\
                @group(0) @binding(3) var<storage,read_write> output:array<f32>;\n{}\n\
                @compute @workgroup_size(128)\n\
                fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,\n\
                    @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {{\n\
                    let row=g.x*(128u/width)+sg;let valid=valid_group(g.y);\n\
                    var slot=0u;var route=0u;if valid {{slot=map[p.a+4u+g.y]-FIRST_SLOT;route=map[p.a+164u+g.y*8u];}}\n\
                    var sum=0.0;if valid {{sum=project_down(row,(slot*2560u+row)*{}u,p.b+route*p.c,lane,width);}}\n\
                    if valid && row<2560u && lane==0u {{output[p.d+route*p.e+row]=sum;if row==0u {{status[p.f+80u+route]=1u;}}}}\n\
                }}",
                project("down", 2560, 640, self.dtype, self.row_bytes),
                self.row_bytes
            )
        } else {
            format!(
                "{header}\n\
                @group(0) @binding(0) var<storage,read> gate:array<u32>;\n\
                @group(0) @binding(1) var<storage,read> up:array<u32>;\n\
                @group(0) @binding(2) var<storage,read> input4:array<vec4<f32>>;\n\
                @group(0) @binding(4) var<storage,read_write> output:array<f32>;\n{}\n{}\n\
                @compute @workgroup_size(128)\n\
                fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(subgroup_id) sg:u32,\n\
                    @builtin(subgroup_invocation_id) lane:u32,@builtin(subgroup_size) width:u32) {{\n\
                    let row=g.x*(128u/width)+sg;let valid=valid_group(g.y);\n\
                    var slot=0u;var route=0u;if valid {{slot=map[p.a+4u+g.y]-FIRST_SLOT;route=map[p.a+164u+g.y*8u];}}\n\
                    var a=0.0;var b=0.0;if valid {{let base=(slot*640u+row)*1440u;\n\
                        a=project_gate(row,base,p.b,lane,width);b=project_up(row,base,p.b,lane,width);}}\n\
                    if valid && row<640u && lane==0u {{\n\
                        var sig=0.0;if a>=0.0 {{sig=1.0/(1.0+exp(-a));}}else{{let ex=exp(a);sig=ex/(1.0+ex);}}\n\
                        output[p.c+route*p.d+row]=a*sig*b;if row==0u {{status[p.e+route]=1u;}}\n\
                    }}\n\
                }}",
                project("gate", 640, 2560, 12, 1440),
                project("up", 640, 2560, 12, 1440)
            )
        }
    }
}
impl KernelMetadata for Kernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(128))
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
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(128),
            debug_info: None,
        })
    }
}
// Parent validates full original allocation identity, extent, aliasing, bank
// origin and transaction before this narrowed count1 vector seam.
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    g: Geometry,
    s: &Spans,
    banks: [Handle; 3],
    owners: [Handle; 7],
    first_slot: usize,
    slots: usize,
) -> Result<()> {
    let [gate, up, down] = g.projections();
    if g.hidden() != 2560
        || g.ff() != 640
        || g.tile() != 1
        || gate.dtype() != 12
        || up.dtype() != 12
        || !matches!(down.dtype(), 6 | 8)
        || first_slot.checked_add(slots).is_none_or(|n| n > 96)
        || slots == 0
    {
        return Err(Error::InvalidFormat("shared count1 expert geometry"));
    }
    let packed_words = [
        gate.segment_words(slots),
        up.segment_words(slots),
        down.segment_words(slots),
    ];
    let [a, b, c] = packed_words;
    g.validate_shard(
        first_slot,
        slots,
        [
            a.map_err(Error::InvalidFormat)?,
            b.map_err(Error::InvalidFormat)?,
            c.map_err(Error::InvalidFormat)?,
        ],
    )
    .map_err(Error::InvalidFormat)?;
    let words = owners.each_ref().map(|h| h.size_in_used() as usize / 4);
    s.validate(g, words).map_err(Error::InvalidFormat)?;
    if owners
        .iter()
        .chain(banks.iter())
        .any(|h| h.stream != owners[0].stream)
        || [s.input_first(), s.act_first(), s.act_stride()]
            .iter()
            .any(|v| !v.is_multiple_of(4))
        || owners.iter().any(|h| !h.size_in_used().is_multiple_of(16))
    {
        return Err(Error::InvalidBufferLength);
    }
    for (projection, bank) in [gate, up, down].iter().zip(&banks) {
        if bank.size_in_used()
            != projection
                .segment_words(slots)
                .map_err(Error::InvalidFormat)? as u64
                * 4
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let infos = [
        [
            s.map_first(),
            s.input_first(),
            s.act_first(),
            s.act_stride(),
            s.status_first(),
            0,
            0,
            0,
        ],
        [
            s.map_first(),
            s.act_first(),
            s.act_stride(),
            s.partial_first(),
            s.partial_stride(),
            s.status_first(),
            0,
            0,
        ],
    ];
    for (is_down, info) in infos.into_iter().enumerate() {
        if info.iter().any(|&v| v > u32::MAX as usize) {
            return Err(Error::InvalidBufferLength);
        }
        let mut args = KernelArguments::new();
        let indices = if is_down == 0 {
            vec![
                banks[0].clone(),
                banks[1].clone(),
                owners[1].clone(),
                owners[0].clone(),
                owners[2].clone(),
                owners[6].clone(),
            ]
        } else {
            vec![
                banks[2].clone(),
                owners[2].clone(),
                owners[0].clone(),
                owners[3].clone(),
                owners[6].clone(),
            ]
        };
        for h in indices {
            args = args.with_buffer(h.binding());
        }
        args = args.with_info(MetadataBindingInfo::custom(
            info.chunks_exact(2)
                .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
                .collect(),
        ));
        // SAFETY: original whole-owner/shard proofs plus exact count1 shapes;
        // validated runtime map bounds routes/slots before all weight accesses.
        unsafe {
            client.launch_unchecked(
                Box::new(Kernel {
                    down: is_down != 0,
                    dtype: down.dtype(),
                    row_bytes: down.row_bytes(),
                    first_slot,
                    slots,
                }),
                CubeCount::Static(if is_down == 0 { 160 } else { 640 }, 10, 1),
                args,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_vector_experts_validate() {
        for (down, dtype, row_bytes) in [(false, 6, 440), (true, 6, 440), (true, 8, 680)] {
            for (first_slot, slots) in [(0, 64), (64, 32)] {
                let source = Kernel {
                    down,
                    dtype,
                    row_bytes,
                    first_slot,
                    slots,
                }
                .source();
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::SUBGROUP
                        | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
                )
                .validate(&module)
                .unwrap();
            }
        }
    }
}
