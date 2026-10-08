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
    fused: bool,
    dtype: u32,
    row_bytes: usize,
    first_slot: usize,
    slots: usize,
    split_slots: usize,
}
fn projection(name: &str, rows: usize, columns: usize, dtype: u32, row_bytes: usize) -> String {
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
}

impl Kernel {
    fn source(self) -> String {
        // Exactly the original packed-vector row implementation already
        // independently checked against original GGUF rows. Each projection
        // gets its own immutable weight binding and helper namespace.
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
        let source = if self.down {
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
                projection("down", 2560, 640, self.dtype, self.row_bytes),
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
                projection("gate", 640, 2560, 12, 1440),
                projection("up", 640, 2560, 12, 1440)
            )
        };
        if self.split_slots == 0 {
            return if self.fused {
                fused_source(source, self.down)
            } else {
                source
            };
        }
        let split = self.split_slots;
        let source = format!("const SPLIT_SLOTS:u32={split}u;\n{source}");
        let source = if self.down {
            source.replace("@binding(5) var<uniform>", "@binding(6) var<uniform>")
                .replace("@compute @workgroup_size(128)", &format!(
                    "@group(0) @binding(5) var<storage,read> down_second:array<u32>;\n{}\n@compute @workgroup_size(128)",
                    projection("down_second", 2560, 640, self.dtype, self.row_bytes)))
                .replace("sum=project_down(row,(slot*2560u+row)*", "if slot>=SPLIT_SLOTS {sum=project_down_second(row,((slot-SPLIT_SLOTS)*2560u+row)*")
                .replace("u,p.b+route*p.c,lane,width);}", &format!(
                    "u,p.b+route*p.c,lane,width);}}else{{sum=project_down(row,(slot*2560u+row)*{}u,p.b+route*p.c,lane,width);}}}}", self.row_bytes))
        } else {
            source.replace("@binding(6) var<uniform>", "@binding(8) var<uniform>")
                .replace("@compute @workgroup_size(128)", &format!(
                    "@group(0) @binding(6) var<storage,read> gate_second:array<u32>;\n@group(0) @binding(7) var<storage,read> up_second:array<u32>;\n{}\n{}\n@compute @workgroup_size(128)",
                    projection("gate_second", 640, 2560, 12, 1440),
                    projection("up_second", 640, 2560, 12, 1440)))
                .replace("let base=(slot*640u+row)*1440u;", "if slot>=SPLIT_SLOTS {let base=((slot-SPLIT_SLOTS)*640u+row)*1440u;a=project_gate_second(row,base,p.b,lane,width);b=project_up_second(row,base,p.b,lane,width);}else{let base=(slot*640u+row)*1440u;")
                .replace("b=project_up(row,base,p.b,lane,width);}", "b=project_up(row,base,p.b,lane,width);}}")
        };
        if self.fused {
            fused_source(source, self.down)
        } else {
            source
        }
    }
}

/// Original per-projection arithmetic, interleaved to expose independent
/// gate/up instruction streams. Private input is workgroup-local only.
fn fused_projection(suffix: &str) -> String {
    format!(
        r#"
fn project_fused{suffix}(row:u32,base:u32,lane:u32,width:u32)->vec2<f32> {{
    var gate_sum=0.0; var up_sum=0.0;
    if row<640u {{
        for(var block=0u;block<10u;block++) {{
            let b=base/4u+block*36u;
            let gf=unpack2x16float(gate{suffix}[b]);
            let uf=unpack2x16float(up{suffix}[b]);
            let ga=gate{suffix}[b+1u];let gz=gate{suffix}[b+2u];let gs=gate{suffix}[b+3u];
            let ua=up{suffix}[b+1u];let uz=up{suffix}[b+2u];let us=up{suffix}[b+3u];
            for(var word=lane;word<32u;word+=width) {{
                let pair=word/8u;let i=(word%8u)*4u;
                let gq=gate{suffix}_bytes4(gate{suffix}[b+4u+word]);
                let uq=up{suffix}_bytes4(up{suffix}[b+4u+word]);
                let gl=vec4<f32>(gq&vec4<u32>(15u))*gf.x*gate{suffix}_q4_scale(ga,gs,pair*2u)
                    -vec4<f32>(gf.y*gate{suffix}_q4_minimum(gz,gs,pair*2u));
                let gh=vec4<f32>(gq>>vec4<u32>(4u))*gf.x*gate{suffix}_q4_scale(ga,gs,pair*2u+1u)
                    -vec4<f32>(gf.y*gate{suffix}_q4_minimum(gz,gs,pair*2u+1u));
                let ul=vec4<f32>(uq&vec4<u32>(15u))*uf.x*up{suffix}_q4_scale(ua,us,pair*2u)
                    -vec4<f32>(uf.y*up{suffix}_q4_minimum(uz,us,pair*2u));
                let uh=vec4<f32>(uq>>vec4<u32>(4u))*uf.x*up{suffix}_q4_scale(ua,us,pair*2u+1u)
                    -vec4<f32>(uf.y*up{suffix}_q4_minimum(uz,us,pair*2u+1u));
                let col=(block*256u+pair*64u+i)/4u;
                let low=cached_input[col];let high=cached_input[col+8u];
                gate_sum+=dot(gl,low)+dot(gh,high);
                up_sum+=dot(ul,low)+dot(uh,high);
            }}
        }}
    }}
    return vec2<f32>(subgroupAdd(gate_sum),subgroupAdd(up_sum));
}}
"#
    )
}
fn fused_source(mut source: String, down: bool) -> String {
    let columns = if down { 640 } else { 2560 };
    let loader = if down { "p.b+route*p.c" } else { "p.b" };
    // Only private input reads inside projection helpers become local reads.
    // Copy the vector once for all row subgroups, then use the original offsets
    // relative to that vector. Inactive groups participate in the same barrier.
    if down {
        source = source
            .replace("input4[", "cached_input[")
            .replace("(input+", "(");
    } else {
        source = source
            .replace("a=project_gate(row,base,p.b,lane,width);b=project_up(row,base,p.b,lane,width);",
                "let pair=project_fused(row,base,lane,width);a=pair.x;b=pair.y;")
            .replace("a=project_gate_second(row,base,p.b,lane,width);b=project_up_second(row,base,p.b,lane,width);",
                "let pair=project_fused_second(row,base,lane,width);a=pair.x;b=pair.y;");
        source.push_str(&fused_projection(""));
        if source.contains("var<storage,read> gate_second") {
            source.push_str(&fused_projection("_second"));
        }
    }
    source = source
        .replace("@builtin(workgroup_id) g:vec3<u32>,", "@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) local:u32,")
        .replace("var sum=0.0;if valid", &format!(
            "if valid {{for(var i=local;i<{}u;i+=128u) {{cached_input[i]=input4[({loader})/4u+i];}}}}\nworkgroupBarrier();\nvar sum=0.0;if valid", columns / 4))
        .replace("var a=0.0;var b=0.0;if valid", &format!(
            "if valid {{for(var i=local;i<{}u;i+=128u) {{cached_input[i]=input4[({loader})/4u+i];}}}}\nworkgroupBarrier();\nvar a=0.0;var b=0.0;if valid", columns / 4));
    format!(
        "var<workgroup> cached_input:array<vec4<f32>,{}>;\n{source}",
        columns / 4
    )
}

impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        if !self.down {
            "NextMoeGateUp"
        } else if self.dtype == 6 {
            "NextMoeDownQ5"
        } else {
            "NextMoeDownQ8"
        }
    }
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    g: Geometry,
    s: &Spans,
    banks: [Handle; 3],
    owners: [Handle; 7],
    first_slot: usize,
    slots: usize,
    fused: bool,
) -> Result<()> {
    launch_banks(
        client, g, s, banks, None, owners, first_slot, slots, 0, fused,
    )
}

pub(crate) fn launch_pair(
    client: &ComputeClient<WgpuRuntime>,
    g: Geometry,
    s: &Spans,
    banks: [[Handle; 3]; 2],
    owners: [Handle; 7],
    capacities: [usize; 2],
    fused: bool,
) -> Result<()> {
    let [first, second] = banks;
    let total = capacities[0]
        .checked_add(capacities[1])
        .ok_or(Error::ArithmeticOverflow)?;
    if capacities.contains(&0) || total != g.capacity() {
        return Err(Error::InvalidFormat("shared paired original bank capacity"));
    }
    launch_banks(
        client,
        g,
        s,
        first,
        Some(second),
        owners,
        0,
        total,
        capacities[0],
        fused,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_banks(
    client: &ComputeClient<WgpuRuntime>,
    g: Geometry,
    s: &Spans,
    banks: [Handle; 3],
    second: Option<[Handle; 3]>,
    owners: [Handle; 7],
    first_slot: usize,
    slots: usize,
    split_slots: usize,
    fused: bool,
) -> Result<()> {
    let [gate, up, down] = g.projections();
    if g.hidden() != 2560
        || g.ff() != 640
        || g.tile() != 1
        || gate.dtype() != 12
        || up.dtype() != 12
        || !matches!(down.dtype(), 6 | 8)
        || first_slot
            .checked_add(slots)
            .is_none_or(|n| n > g.capacity())
        || slots == 0
    {
        return Err(Error::InvalidFormat("shared count1 expert geometry"));
    }
    if (split_slots == 0) != second.is_none() || split_slots >= slots && split_slots != 0 {
        return Err(Error::InvalidFormat("shared original bank partition"));
    }
    let first_capacity = if split_slots == 0 { slots } else { split_slots };
    for (offset, capacity, buffers) in std::iter::once((first_slot, first_capacity, &banks)).chain(
        second
            .as_ref()
            .map(|b| (first_slot + split_slots, slots - split_slots, b)),
    ) {
        let mut packed_words = [0usize; 3];
        for (i, (projection, buffer)) in [gate, up, down].iter().zip(buffers).enumerate() {
            packed_words[i] = projection
                .segment_words(capacity)
                .map_err(Error::InvalidFormat)?;
            if buffer.size_in_used() != packed_words[i] as u64 * 4
                || buffer.stream != owners[0].stream
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        g.validate_shard(offset, capacity, packed_words)
            .map_err(Error::InvalidFormat)?;
    }
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
        let mut indices = if is_down == 0 {
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
        if let Some(second) = &second {
            if is_down == 0 {
                indices.extend([second[0].clone(), second[1].clone()]);
            } else {
                indices.push(second[2].clone());
            }
        }
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
                    fused,
                    dtype: down.dtype(),
                    row_bytes: down.row_bytes(),
                    first_slot,
                    slots,
                    split_slots,
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
            for (first_slot, slots, split_slots) in [
                (0, 64, 0),
                (64, 32, 0),
                (0, 96, 64),
                (0, 128, 64),
                (0, 144, 72),
            ] {
                for fused in [false, true] {
                    let source = Kernel {
                        fused,
                        down,
                        dtype,
                        row_bytes,
                        first_slot,
                        slots,
                        split_slots,
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
}
