//! Original64 F32 MoE projections with workgroup-owned public IQ codebooks.
//! No activation quantization, reordered reduction, FMA instruction or expanded
//! weights. Physical acceptance remains separate from host/source validation.
//! Requires enabled subgroup support with16/32 subgroup sizes and128 lanes.
//! The caller retains all original source/pin/epoch/owning/audited-close proofs.
//! Packing provenance is in iq_moe_shared_original-NOTICE.txt.
use super::{IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, Q8_0, grouped_moe as g};
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

const PROFILES: [(u32, u32); 9] = [
    (IQ3_XXS, IQ4_NL),
    (IQ3_XXS, Q8_0),
    (IQ4_XS, Q8_0),
    (IQ3_XXS, Q2_0),
    (IQ2_S, IQ4_NL),
    (IQ2_S, Q2_0),
    (IQ3_S, IQ4_NL),
    (IQ3_S, Q2_0),
    (IQ4_XS, IQ4_NL),
];
pub(crate) const MAX_SHARED_BYTES: usize = 19_460;
pub(crate) fn workgroup_bytes(dtype: u32) -> usize {
    match dtype {
        IQ3_XXS => 11_268 + 1024,
        IQ3_S => 11_268 + 2048,
        IQ2_S => MAX_SHARED_BYTES,
        IQ4_XS => 11_268,
        IQ4_NL | Q8_0 | Q2_0 => 3076,
        _ => 0,
    }
}
fn rejected() -> Error {
    Error::InvalidFormat("original F32 shared-IQ count1 geometry")
}
pub(crate) fn validate_geometry(geometry: g::Geometry, first: usize, slots: usize) -> Result<()> {
    let [gate, up, down] = geometry.projections();
    if geometry.hidden() != 2560
        || geometry.ff() != 640
        || geometry.tile() != 1
        || geometry.capacity() != 144
        || geometry.gate_policy() != g::GatePolicy::Generic
        || up.dtype() != gate.dtype()
        || !PROFILES.contains(&(gate.dtype(), down.dtype()))
        || !matches!((first, slots), (0 | 72, 72))
    {
        return Err(rejected());
    }
    geometry.validate_count(1).map_err(|_| rejected())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Kernel {
    down: bool,
    dtype: u32,
    row_bytes: usize,
    first: usize,
    slots: usize,
}

fn projection(name: &str, dtype: u32) -> String {
    let decode = match dtype {
        IQ3_XXS => {
            r#"
    let base=row+(column/256u)*98u;let within=column%256u;
    let group=within/32u;let octet=(within%32u)/8u;let lane=within%8u;
    let word=NAME_word(base+66u+group*4u);
    let first=NAME_half(base)*(0.5+f32(word>>28u));let scale=first*0.5;
    let sign_code=(word>>(octet*7u))&127u;
    var parity=sign_code^(sign_code>>4u);parity^=parity>>2u;parity^=parity>>1u;
    let signs=sign_code|((parity&1u)<<7u);
    let index=NAME_byte(base+2u+group*8u+octet*2u+lane/4u);
    let magnitude=(iq_grid[index]>>((lane%4u)*8u))&255u;
    var sign=1.0;if ((signs>>lane)&1u)!=0u {sign=-1.0;}
    let scaled=scale*f32(magnitude);return scaled*sign;
"#
        }
        IQ3_S => {
            r#"
    let base=row+(column/256u)*110u;let within=column%256u;
    let group=within/32u;let octet=(within%32u)/8u;let lane=within%8u;
    let quartet=octet*2u+lane/4u;
    let low=NAME_byte(base+2u+group*8u+quartet);
    let high=(NAME_byte(base+66u+group)>>quartet)&1u;
    let magnitude=(iq_grid[low|(high<<8u)]>>((lane%4u)*8u))&255u;
    let scales=NAME_byte(base+106u+group/2u);
    let scale=(scales>>((group%2u)*4u))&15u;
    let factor=NAME_half(base)*f32(1u+2u*scale);
    var value=factor*f32(magnitude);
    let signs=NAME_byte(base+74u+group*4u+octet);
    if ((signs>>lane)&1u)!=0u {value*= -1.0;}
    return value;
"#
        }
        IQ2_S => {
            r#"
    let base=row+(column/256u)*82u;let within=column%256u;
    let group=within/32u;let octet=(within%32u)/8u;let lane=within%8u;
    let low=NAME_byte(base+2u+group*4u+octet);
    let high=(NAME_byte(base+66u+group)>>(octet*2u))&3u;
    let grid=iq_grid[(low|(high<<8u))*2u+lane/4u];
    let magnitude=(grid>>((lane%4u)*8u))&255u;
    let scales=NAME_byte(base+74u+group);
    var scale=scales&15u;if octet>=2u {scale=scales>>4u;}
    let first=NAME_half(base)*(0.5+f32(scale));let factor=first*0.25;
    var value=factor*f32(magnitude);
    let signs=NAME_byte(base+34u+group*4u+octet);
    if ((signs>>lane)&1u)!=0u {value*= -1.0;}
    return value;
"#
        }
        IQ4_XS => {
            r#"
    let base=row+(column/256u)*136u;let within=column%256u;
    let group=within/32u;let lane=within%32u;
    let low=(NAME_byte(base+4u+group/2u)>>((group%2u)*4u))&15u;
    let high=(NAME_half_bits(base+2u)>>(group*2u))&3u;
    let subscale=i32(low|(high<<4u))-32;
    let scale=NAME_half(base)*f32(subscale);
    let codes=NAME_byte(base+8u+group*16u+lane%16u);
    var code=codes&15u;if lane>=16u {code=codes>>4u;}
    return scale*iq4_value(code);
"#
        }
        IQ4_NL => {
            r#"
    let base=row+(column/32u)*18u;let lane=column%32u;
    let codes=NAME_byte(base+2u+lane%16u);
    var code=codes&15u;if lane>=16u {code=codes>>4u;}
    return NAME_half(base)*iq4_value(code);
"#
        }
        Q8_0 => {
            r#"
    let base=row+(column/32u)*34u;
    return NAME_half(base)*f32(signed_byte(NAME_byte(base+2u+column%32u)));
"#
        }
        Q2_0 => {
            r#"
    let base=row+(column/64u)*18u;let lane=column%64u;
    let codes=NAME_byte(base+2u+lane/4u);
    let code=i32((codes>>((lane%4u)*2u))&3u)-1;
    return NAME_half(base)*f32(code);
"#
        }
        _ => unreachable!("validated original packed format"),
    };
    format!(
        r#"
fn NAME_byte(at:u32)->u32 {{return (NAME[at/4u]>>((at%4u)*8u))&255u;}}
fn NAME_word(at:u32)->u32 {{
    let shift=(at%4u)*8u;let low=NAME[at/4u];
    if shift==0u {{return low;}}
    return (low>>shift)|(NAME[at/4u+1u]<<(32u-shift));
}}
fn NAME_half_bits(at:u32)->u32 {{return NAME_byte(at)|(NAME_byte(at+1u)<<8u);}}
fn NAME_half(at:u32)->f32 {{return binary16(NAME_half_bits(at));}}
fn NAME_weight(row:u32,column:u32)->f32 {{{decode}}}
fn NAME_project(base:u32,enabled:bool,row:u32,lane:u32,sub_lane:u32,width:u32,scratch:u32)->f32 {{
    var low=0.0;var high=0.0;
    if enabled {{
        for(var tile=0u;tile<COLUMNS/64u;tile++) {{
            let column=tile*64u+lane;
            let lower=NAME_weight(base,column)*cached_input[column];low+=lower;
            let upper=NAME_weight(base,column+32u)*cached_input[column+32u];high+=upper;
        }}
    }}
    return reduce_original64(low,high,row,lane,sub_lane,width,scratch);
}}
"#
    )
    .replace("NAME", name)
}

impl Kernel {
    fn source(self) -> String {
        let columns = if self.down { 640 } else { 2560 };
        let rows = if self.down { 2560 } else { 640 };
        let (bindings, helpers, input_offset, epilogue) = if self.down {
            (
                "@group(0) @binding(0) var<storage,read> down:array<u32>;\n@group(0) @binding(1) var<storage,read> input:array<f32>;\n@group(0) @binding(2) var<storage,read> map:array<u32>;\n@group(0) @binding(3) var<storage,read_write> output:array<f32>;\n@group(0) @binding(4) var<storage,read_write> status:array<u32>;\n@group(0) @binding(5) var<uniform> p:Params;",
                projection("down", self.dtype),
                "route*p.c",
                "let value=down_project((slot*ROWS+row)*ROW_BYTES,ready,local_row,lane,sub_lane,width,0u);\nif ready && lane==0u {output[p.d+route*p.e+row]=value;if row==0u {status[p.f+80u+route]=1u;}}",
            )
        } else {
            (
                "@group(0) @binding(0) var<storage,read> gate:array<u32>;\n@group(0) @binding(1) var<storage,read> up:array<u32>;\n@group(0) @binding(2) var<storage,read> input:array<f32>;\n@group(0) @binding(3) var<storage,read> map:array<u32>;\n@group(0) @binding(4) var<storage,read_write> output:array<f32>;\n@group(0) @binding(5) var<storage,read_write> status:array<u32>;\n@group(0) @binding(6) var<uniform> p:Params;",
                format!(
                    "{}\n{}",
                    projection("gate", self.dtype),
                    projection("up", self.dtype)
                ),
                "(route/10u)*p.c",
                "let base=(slot*ROWS+row)*ROW_BYTES;\nlet a=gate_project(base,ready,local_row,lane,sub_lane,width,0u);\nlet b=up_project(base,ready,local_row,lane,sub_lane,width,128u);\nif ready && lane==0u {var sigmoid=0.0;if a>=0.0 {sigmoid=1.0/(1.0+exp(-a));}else{let exponential=exp(a);sigmoid=exponential/(1.0+exponential); }\nlet activation=a*sigmoid;output[p.d+route*p.e+row]=activation*b;if row==0u {status[p.f+route]=1u;}}",
            )
        };
        use std::fmt::Write;
        let words = match self.dtype {
            IQ3_XXS => kuno_text::quant::IQ3_XXS_GRID.to_vec(),
            IQ3_S => kuno_text::quant::IQ3_S_GRID.to_vec(),
            IQ2_S => kuno_text::quant::IQ2_S_GRID
                .into_iter()
                .flat_map(|word| [word as u32, (word >> 32) as u32])
                .collect(),
            _ => Vec::new(),
        };
        let mut grid = String::new();
        let mut initialize = String::new();
        if !words.is_empty() {
            writeln!(
                grid,
                "var<workgroup> iq_grid:array<u32,{}>;\nfn iq_literal(i:u32)->u32 {{switch i {{",
                words.len()
            )
            .unwrap();
            for (i, word) in words.iter().enumerate() {
                writeln!(grid, "case {i}u: {{return 0x{word:08x}u;}}").unwrap();
            }
            grid.push_str("default: {return 0u;}\n}}\n");
            write!(
                initialize,
                "for(var i=local;i<{}u;i+=128u) {{iq_grid[i]=iq_literal(i);}}",
                words.len()
            )
            .unwrap();
        }
        let mut source = include_str!("iq_moe_shared_original.wgsl")
            .replace("BINDINGS_VALUE", bindings)
            .replace("PROJECTION_HELPERS_VALUE", &helpers)
            .replace("INPUT_OFFSET_VALUE", input_offset)
            .replace(
                "INPUT_READY_VALUE",
                if self.down {
                    "if valid && status[p.f+route]!=1u {valid=false;}"
                } else {
                    ""
                },
            )
            .replace("EPILOGUE_VALUE", epilogue)
            .replace("GRID_DECLARATION_VALUE", &grid)
            .replace("GRID_INITIALIZE_VALUE", &initialize);
        for (key, value) in [
            ("COLUMNS_VALUE", columns.to_string()),
            ("ROWS_VALUE", rows.to_string()),
            ("ROW_BYTES_VALUE", self.row_bytes.to_string()),
            ("FIRST_SLOT_VALUE", self.first.to_string()),
            ("SLOTS_VALUE", self.slots.to_string()),
            (
                "REDUCTION_CELLS_VALUE",
                if self.down { 128 } else { 256 }.to_string(),
            ),
        ] {
            source = source.replace(key, &value);
        }
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        if self.down {
            "IQMoeSharedOriginalDown"
        } else {
            "IQMoeSharedOriginalGateUp"
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

/// Caller retains full physical disjointness/source/epoch/pending wave proofs.
/// No allocation, bank conversion, route change or publication lives here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 3],
    owners: [Handle; 7],
    first: usize,
    slots: usize,
    groups: usize,
) -> Result<()> {
    validate_geometry(geometry, first, slots)?;
    if !(1..=10).contains(&groups) {
        return Err(Error::InvalidBufferLength);
    }
    let projections = geometry.projections();
    let mut bank_words = [0usize; 3];
    for (i, (p, b)) in projections.iter().zip(&banks).enumerate() {
        bank_words[i] = p.segment_words(slots).map_err(Error::InvalidFormat)?;
        if b.size_in_used() != bank_words[i] as u64 * 4 {
            return Err(Error::InvalidBufferLength);
        }
    }
    geometry
        .validate_shard(first, slots, bank_words)
        .map_err(Error::InvalidFormat)?;
    let words = owners.each_ref().map(|h| h.size_in_used() as usize / 4);
    spans
        .validate(geometry, words)
        .map_err(Error::InvalidFormat)?;
    if banks
        .iter()
        .chain(&owners)
        .any(|h| h.stream != owners[0].stream || !h.size_in_used().is_multiple_of(4))
    {
        return Err(Error::InvalidBufferLength);
    }
    let infos = [
        [
            spans.map_first(),
            spans.input_first(),
            spans.input_stride(),
            spans.act_first(),
            spans.act_stride(),
            spans.status_first(),
            0,
            0,
        ],
        [
            spans.map_first(),
            spans.act_first(),
            spans.act_stride(),
            spans.partial_first(),
            spans.partial_stride(),
            spans.status_first(),
            0,
            0,
        ],
    ];
    for (down, info) in infos.into_iter().enumerate() {
        if info.iter().any(|&v| v > u32::MAX as usize) {
            return Err(Error::InvalidBufferLength);
        }
        let mut args = KernelArguments::new();
        if down == 0 {
            for h in [
                &banks[0], &banks[1], &owners[1], &owners[0], &owners[2], &owners[6],
            ] {
                args = args.with_buffer(h.clone().binding());
            }
        } else {
            for h in [&banks[2], &owners[2], &owners[0], &owners[3], &owners[6]] {
                args = args.with_buffer(h.clone().binding());
            }
        }
        args = args.with_info(MetadataBindingInfo::custom(
            info.chunks_exact(2)
                .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                .collect(),
        ));
        let p = projections[if down == 0 { 0 } else { 2 }];
        // SAFETY: exact original count1 geometry/shards/full owners above;
        // complete runtime map guard precedes every source/output access.
        unsafe {
            client.launch_unchecked(
                Box::new(Kernel {
                    down: down != 0,
                    dtype: p.dtype(),
                    row_bytes: p.row_bytes(),
                    first,
                    slots,
                }),
                CubeCount::Static(
                    (if down == 0 { 640usize } else { 2560 }).div_ceil(4) as u32,
                    groups as u32,
                    1,
                ),
                args,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::{TensorType, quant};

    fn geometry(gate: u32, down: u32) -> g::Geometry {
        g::Geometry::new(
            [
                g::Projection::new(gate, 2560, 640, 512).unwrap(),
                g::Projection::new(gate, 2560, 640, 512).unwrap(),
                g::Projection::new(down, 640, 2560, 512).unwrap(),
            ],
            1,
            144,
        )
        .unwrap()
    }

    #[test]
    fn exact_nine_profiles_cover_all48_das_layers_without_general_admission() {
        for (gate, down) in PROFILES {
            let g = geometry(gate, down);
            for first in [0, 72] {
                validate_geometry(g, first, 72).unwrap();
            }
            for (first, slots) in [(1, 72), (0, 71), (0, 144), (72, 73)] {
                assert!(validate_geometry(g, first, slots).is_err());
            }
            assert!(workgroup_bytes(gate) <= MAX_SHARED_BYTES);
            assert_eq!(workgroup_bytes(down), 3076);
        }
        for layer in 0..48 {
            let gate =
                crate::daslab_profile::expected_dtype(&format!("blk.{layer}.ffn_gate_exps.weight"))
                    .unwrap();
            let down =
                crate::daslab_profile::expected_dtype(&format!("blk.{layer}.ffn_down_exps.weight"))
                    .unwrap();
            validate_geometry(geometry(gate, down), 0, 72).unwrap();
        }
        assert!(validate_geometry(geometry(super::super::Q4_K, Q8_0), 0, 72).is_err());
        assert!(!PROFILES.contains(&(IQ3_S, Q8_0)));
        assert_eq!(workgroup_bytes(super::super::Q4_K), 0);
    }

    fn half_host(bits: u16) -> f32 {
        let bits = u32::from(bits);
        let sign = (bits & 32768) << 16;
        let exponent = (bits >> 10) & 31;
        let mut mantissa = bits & 1023;
        let result = if exponent == 0 {
            if mantissa == 0 {
                sign
            } else {
                let mut adjusted = 113;
                while mantissa & 1024 == 0 {
                    mantissa <<= 1;
                    adjusted -= 1;
                }
                sign | (adjusted << 23) | ((mantissa & 1023) << 13)
            }
        } else if exponent == 31 {
            sign | 2139095040 | (mantissa << 13)
        } else {
            sign | ((exponent + 112) << 23) | (mantissa << 13)
        };
        f32::from_bits(result)
    }
    fn weight_host(bytes: &[u8], row: usize, column: usize, dtype: u32) -> f32 {
        // The shader's original unaligned u32 byte extraction is modeled here,
        // not a call to a Rust quantized-row decoder.
        let byte = |at: usize| {
            let first = at / 4 * 4;
            (u32::from_le_bytes(bytes[first..first + 4].try_into().unwrap()) >> ((at % 4) * 8))
                & 255
        };
        let half = |at| half_host((byte(at) | (byte(at + 1) << 8)) as u16);
        let signed = |q: u32| if q >= 128 { q as i32 - 256 } else { q as i32 };
        match dtype {
            IQ3_XXS => {
                let base = row + column / 256 * 98;
                let within = column % 256;
                let group = within / 32;
                let octet = (within % 32) / 8;
                let lane = within % 8;
                let at = base + 66 + group * 4;
                let word =
                    byte(at) | (byte(at + 1) << 8) | (byte(at + 2) << 16) | (byte(at + 3) << 24);
                let scale = half(base) * (0.5 + (word >> 28) as f32) * 0.5;
                let code = (word >> (octet * 7)) & 127;
                let mut parity = code ^ (code >> 4);
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                let signs = code | ((parity & 1) << 7);
                let index = byte(base + 2 + group * 8 + octet * 2 + lane / 4) as usize;
                let magnitude = (quant::IQ3_XXS_GRID[index] >> ((lane % 4) * 8)) & 255;
                let sign = if (signs >> lane) & 1 != 0 { -1.0 } else { 1.0 };
                (scale * magnitude as f32) * sign
            }
            IQ3_S => {
                let base = row + column / 256 * 110;
                let within = column % 256;
                let group = within / 32;
                let octet = (within % 32) / 8;
                let lane = within % 8;
                let quartet = octet * 2 + lane / 4;
                let index = byte(base + 2 + group * 8 + quartet)
                    | (((byte(base + 66 + group) >> quartet) & 1) << 8);
                let magnitude = (quant::IQ3_S_GRID[index as usize] >> ((lane % 4) * 8)) & 255;
                let scale = (byte(base + 106 + group / 2) >> ((group % 2) * 4)) & 15;
                let mut value = (half(base) * (1 + 2 * scale) as f32) * magnitude as f32;
                if (byte(base + 74 + group * 4 + octet) >> lane) & 1 != 0 {
                    value *= -1.0;
                }
                value
            }
            IQ2_S => {
                let base = row + column / 256 * 82;
                let within = column % 256;
                let group = within / 32;
                let octet = (within % 32) / 8;
                let lane = within % 8;
                let index = byte(base + 2 + group * 4 + octet)
                    | (((byte(base + 66 + group) >> (2 * octet)) & 3) << 8);
                let word = quant::IQ2_S_GRID[index as usize];
                let grid = if lane < 4 {
                    word as u32
                } else {
                    (word >> 32) as u32
                };
                let magnitude = (grid >> ((lane % 4) * 8)) & 255;
                let scale = (byte(base + 74 + group) >> (octet / 2 * 4)) & 15;
                let mut value = ((half(base) * (0.5 + scale as f32)) * 0.25) * magnitude as f32;
                if (byte(base + 34 + group * 4 + octet) >> lane) & 1 != 0 {
                    value *= -1.0;
                }
                value
            }
            IQ4_NL | IQ4_XS => {
                let (base, offset, scale) = if dtype == IQ4_NL {
                    let base = row + column / 32 * 18;
                    (base, 2, half(base))
                } else {
                    let base = row + column / 256 * 136;
                    let group = column % 256 / 32;
                    let low = (byte(base + 4 + group / 2) >> ((group % 2) * 4)) & 15;
                    let high = ((byte(base + 2) | (byte(base + 3) << 8)) >> (group * 2)) & 3;
                    (
                        base,
                        8 + group * 16,
                        half(base) * ((low | (high << 4)) as i32 - 32) as f32,
                    )
                };
                let lane = column % 32;
                let codes = byte(base + offset + lane % 16);
                let code = if lane < 16 { codes & 15 } else { codes >> 4 };
                let mut word = 0xbfad9881u32;
                if code >= 4 {
                    word = 0xf6eaddcf;
                }
                if code >= 8 {
                    word = 0x26190d01;
                }
                if code >= 12 {
                    word = 0x71594535;
                }
                scale * signed((word >> ((code & 3) * 8)) & 255) as f32
            }
            Q8_0 => {
                let base = row + column / 32 * 34;
                half(base) * signed(byte(base + 2 + column % 32)) as f32
            }
            Q2_0 => {
                let base = row + column / 64 * 18;
                let lane = column % 64;
                let code = ((byte(base + 2 + lane / 4) >> ((lane % 4) * 2)) & 3) as i32 - 1;
                half(base) * code as f32
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn all_original_half_bit_patterns_and23712_pinned_c_values_keep_round_points() {
        for bits in 0..=u16::MAX {
            assert_eq!(half_host(bits).to_bits(), quant::f16_to_f32(bits).to_bits());
        }
        let mut count = 0;
        for dtype in [IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, IQ4_NL, Q2_0] {
            let (records, _) = super::super::iq_fixtures::records(dtype);
            let kind = TensorType::from_gguf_id(dtype).unwrap();
            let elements = kind.block_elements() as usize;
            let bytes = kind.block_bytes() as usize;
            for record in records.chunks_exact(bytes + elements * 4) {
                for prefix in [0, 3] {
                    let mut wire = vec![0x53u8; (prefix + bytes).div_ceil(4) * 4];
                    wire[prefix..prefix + bytes].copy_from_slice(&record[..bytes]);
                    for column in 0..elements {
                        let expected = f32::from_le_bytes(
                            record[bytes + column * 4..bytes + column * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let actual = weight_host(&wire, prefix, column, dtype);
                        if expected.is_finite() {
                            assert_eq!(actual.to_bits(), expected.to_bits());
                        } else {
                            assert!(!actual.is_finite());
                        }
                    }
                }
                count += elements;
            }
        }
        assert_eq!(count, 23_712);
    }

    fn original64(weights: &[f32], input: &[f32]) -> f32 {
        let mut lanes = [0f32; 64];
        for (lane, sum) in lanes.iter_mut().enumerate() {
            for column in (lane..input.len()).step_by(64) {
                *sum += weights[column] * input[column];
            }
        }
        for stride in [32, 16, 8, 4, 2, 1] {
            for lane in 0..stride {
                lanes[lane] += lanes[lane + stride];
            }
        }
        lanes[0]
    }
    fn shared_schedule(weights: &[f32], input: &[f32], width: usize) -> f32 {
        assert!(matches!(width, 16 | 32));
        let mut low = [0f32; 32];
        let mut high = [0f32; 32];
        for lane in 0..32 {
            for tile in 0..input.len() / 64 {
                let column = tile * 64 + lane;
                let lower = weights[column] * input[column];
                low[lane] += lower;
                let upper = weights[column + 32] * input[column + 32];
                high[lane] += upper;
            }
        }
        let mut value = std::array::from_fn::<_, 32, _>(|lane| low[lane] + high[lane]);
        let cells = value;
        if width == 16 {
            for lane in 0..16 {
                value[lane] += cells[lane + 16];
            }
        }
        let mut stride = width / 2;
        while stride > 0 {
            let before = value;
            for lane in 0..32 {
                if lane % width < stride {
                    value[lane] += before[lane + stride];
                }
            }
            stride /= 2;
        }
        value[0]
    }

    #[test]
    fn both_subgroup_widths_keep_exact_original64_f32_row_vectors_and_no_fma() {
        for dtype in [IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS, IQ4_NL, Q2_0] {
            for columns in [640usize, 2560] {
                // Whole256 blocks apply to the three large IQ formats.
                if !columns.is_multiple_of(
                    TensorType::from_gguf_id(dtype).unwrap().block_elements() as usize,
                ) {
                    continue;
                }
                for phase in [0, 3] {
                    let wire = super::super::iq_fixtures::packed(dtype, 4, columns, phase);
                    let expected = super::super::iq_fixtures::oracle(dtype, 4, columns, phase);
                    let row_bytes = TensorType::from_gguf_id(dtype)
                        .unwrap()
                        .row_bytes(columns as u64)
                        .unwrap() as usize;
                    let mut padded = wire.clone();
                    padded.resize(wire.len().div_ceil(4) * 4, 0);
                    let input = (0..columns)
                        .map(|i| ((i * 31 % 137) as f32 - 68.0) / 128.0)
                        .collect::<Vec<_>>();
                    for row in 0..4 {
                        let decoded = (0..columns)
                            .map(|col| weight_host(&padded, row * row_bytes, col, dtype))
                            .collect::<Vec<_>>();
                        for width in [16, 32] {
                            assert_eq!(
                                shared_schedule(&decoded, &input, width).to_bits(),
                                original64(&expected[row * columns..(row + 1) * columns], &input)
                                    .to_bits()
                            );
                        }
                    }
                }
            }
        }
        let mut weights = vec![0.0; 128];
        let mut input = vec![0.0; 128];
        weights[0] = -f32::from_bits(0x3f800002);
        input[0] = 1.0;
        weights[64] = f32::from_bits(0x3f800001);
        input[64] = f32::from_bits(0x3f800001);
        assert_eq!(
            shared_schedule(&weights, &input, 16).to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(
            shared_schedule(&weights, &input, 32).to_bits(),
            0.0f32.to_bits()
        );
        assert_ne!(
            weights[64].mul_add(input[64], weights[0]).to_bits(),
            0.0f32.to_bits()
        );
    }

    #[test]
    fn q8_rows_and_lane_numbering_match_existing_cpu_format_and_both_wave_sizes() {
        for columns in [640usize, 2560] {
            let mut wire = Vec::new();
            for block in 0..columns / 32 {
                wire.extend_from_slice(&[0x00, 0x24]);
                for index in 0..32 {
                    wire.push((block * 29 + index * 43 + 11) as u8);
                }
            }
            let mut decoded = vec![0.0; columns];
            TensorType::Q8_0.decode(&wire, &mut decoded).unwrap();
            let mut padded = wire.clone();
            padded.resize(wire.len().div_ceil(4) * 4, 0);
            let actual = (0..columns)
                .map(|column| weight_host(&padded, 0, column, Q8_0))
                .collect::<Vec<_>>();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let input = (0..columns)
                .map(|i| ((i * 11 % 29) as f32 - 14.0) / 256.0)
                .collect::<Vec<_>>();
            for width in [16, 32] {
                assert_eq!(
                    shared_schedule(&actual, &input, width).to_bits(),
                    original64(&decoded, &input).to_bits()
                );
            }
        }
        for width in [16, 32] {
            let mut visits = [[0u8; 32]; 4];
            for sg in 0..128 / width {
                for sub_lane in 0..width {
                    let row = sg / (32 / width);
                    let lane = (sg % (32 / width)) * width + sub_lane;
                    visits[row][lane] += 1;
                }
            }
            assert!(visits.iter().flatten().all(|&count| count == 1));
        }
    }

    #[test]
    fn all14_shaders_keep_workgroup_codebooks_original_tree_and_no_function_array_or_fma() {
        for (down, dtype, columns) in [
            (false, IQ3_XXS, 2560),
            (false, IQ3_S, 2560),
            (false, IQ2_S, 2560),
            (false, IQ4_XS, 2560),
            (true, IQ4_NL, 640),
            (true, Q8_0, 640),
            (true, Q2_0, 640),
        ] {
            for first in [0, 72] {
                let source = Kernel {
                    down,
                    dtype,
                    row_bytes: TensorType::from_gguf_id(dtype)
                        .unwrap()
                        .row_bytes(columns)
                        .unwrap() as usize,
                    first,
                    slots: 72,
                }
                .source();
                assert!(source.contains("let lower=") && source.contains("low+=lower;"));
                assert!(source.contains("let upper=") && source.contains("high+=upper;"));
                assert!(source.contains("var value=low+high;"));
                assert!(source.contains("if lane<16u {value+=reduction_cells"));
                assert!(source.contains("let other=subgroupShuffleDown(value,stride);"));
                assert!(
                    !source.contains("subgroupAdd(")
                        && !source.contains("fma(")
                        && !source.contains("q8_code")
                );
                assert!(source.contains("if ready && lane==0u"));
                if down {
                    assert!(source.contains("status[p.f+route]!=1u"));
                } else {
                    assert!(source.contains("sub_lane,width,128u"));
                }
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                let info = wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::SUBGROUP,
                )
                .validate(&module)
                .unwrap();
                assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
                let shared = module
                    .global_variables
                    .iter()
                    .filter(|(_, global)| global.space == wgpu::naga::AddressSpace::WorkGroup)
                    .map(|(_, global)| match module.types[global.ty].inner {
                        wgpu::naga::TypeInner::Array {
                            base,
                            size: wgpu::naga::ArraySize::Constant(count),
                            ..
                        } => {
                            assert!(matches!(
                                module.types[base].inner,
                                wgpu::naga::TypeInner::Scalar(_)
                            ));
                            count.get() * 4
                        }
                        wgpu::naga::TypeInner::Atomic(_) => 4,
                        _ => panic!("only bounded scalar workgroup owners"),
                    })
                    .sum::<u32>();
                assert_eq!(shared, workgroup_bytes(dtype) as u32);
                let grids = module
                    .global_variables
                    .iter()
                    .filter(|(_, global)| global.name.as_deref() == Some("iq_grid"))
                    .collect::<Vec<_>>();
                assert_eq!(
                    grids.len(),
                    usize::from(matches!(dtype, IQ3_XXS | IQ3_S | IQ2_S))
                );
                assert!(
                    grids
                        .iter()
                        .all(|(_, global)| global.space == wgpu::naga::AddressSpace::WorkGroup)
                );
                let spv = wgpu::naga::back::spv::write_vec(
                    &module,
                    &info,
                    &wgpu::naga::back::spv::Options {
                        lang_version: (1, 3),
                        ..Default::default()
                    },
                    Some(&wgpu::naga::back::spv::PipelineOptions {
                        shader_stage: wgpu::naga::ShaderStage::Compute,
                        entry_point: "main".into(),
                    }),
                )
                .unwrap();
                let mut arrays = std::collections::HashSet::new();
                let mut pointers = std::collections::HashMap::new();
                let mut variables = Vec::new();
                let mut mul = 0;
                let mut add = 0;
                let mut at = 5;
                while at < spv.len() {
                    let length = (spv[at] >> 16) as usize;
                    let opcode = spv[at] & 65535;
                    assert!(length > 0 && at + length <= spv.len());
                    let args = &spv[at + 1..at + length];
                    match opcode {
                        28 | 29 => {
                            arrays.insert(args[0]);
                        }
                        32 => {
                            pointers.insert(args[0], args[2]);
                        }
                        59 => variables.push((args[0], args[2])),
                        133 => mul += 1,
                        129 => add += 1,
                        12 => assert_ne!(args[3], 50, "GLSL.std.450 Fma is forbidden"),
                        _ => {}
                    }
                    at += length;
                }
                assert!(mul > 0 && add > 0);
                assert!(
                    variables
                        .iter()
                        .filter(|(_, storage)| *storage == 7)
                        .all(|(ty, _)| !arrays.contains(pointers.get(ty).unwrap()))
                );
            }
        }
    }
}
