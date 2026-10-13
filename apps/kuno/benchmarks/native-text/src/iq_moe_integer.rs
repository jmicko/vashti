//! Explicit original-packed IQ integer-dot candidate, not original64 parity.
//! Two residual Q8 activation planes stay workgroup-local. Immutable packed
//! banks/maps/full owners retain the existing protected count1 caller proofs.
//! Pinned arithmetic/layout provenance is in iq_moe_integer-NOTICE.txt.
use super::grouped_moe as g;
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

#[path = "iq_moe_integer_oracle.rs"]
pub(crate) mod oracle;
#[path = "iq_moe_integer_prepared.rs"]
pub(crate) mod prepared;

pub(crate) const MAX_SHARED_BYTES: usize = 24_196;
pub(crate) fn workgroup_bytes(dtype: u32) -> usize {
    match dtype {
        super::IQ3_XXS => 17_028,
        super::IQ3_S => 18_052,
        super::IQ2_S => MAX_SHARED_BYTES,
        super::IQ4_XS => 16_004,
        super::IQ4_NL | super::Q8_0 | super::Q2_0 => 4004,
        _ => 0,
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Kernel {
    down: bool,
    dtype: u32,
    row_bytes: usize,
    first: usize,
    slots: usize,
}

pub(crate) fn original_grid_words(dtype: u32) -> Vec<u32> {
    match dtype {
        super::IQ3_XXS => kuno_text::quant::IQ3_XXS_GRID.to_vec(),
        super::IQ3_S => kuno_text::quant::IQ3_S_GRID.to_vec(),
        super::IQ2_S => kuno_text::quant::IQ2_S_GRID
            .into_iter()
            .flat_map(|word| [word as u32, (word >> 32) as u32])
            .collect(),
        _ => Vec::new(),
    }
}

/// Each lane writes its immutable cells directly, once. A lookup switch with
/// one case per word repeatedly diverges for every initialization iteration;
/// the largest original table has2,048 words/16 iterations per lane. Keep the
/// same full workgroup table and barriers without any runtime literal lookup,
/// Function-address array copy, persistent buffer or private allocation.
pub(crate) fn grid_source(words: &[u32]) -> (String, String) {
    if words.is_empty() {
        return (String::new(), String::new());
    }
    use std::fmt::Write;
    let declaration = format!("var<workgroup> iq3_grid:array<u32,{}>;", words.len());
    let mut initialize = String::from("switch local {\n");
    for lane in 0..128 {
        writeln!(initialize, "case {lane}u: {{").unwrap();
        for index in (lane..words.len()).step_by(128) {
            writeln!(initialize, "iq3_grid[{index}u]=0x{:08x}u;", words[index]).unwrap();
        }
        initialize.push_str("}\n");
    }
    initialize.push_str("default: {}\n}\n");
    (declaration, initialize)
}

pub(crate) fn projection(name: &str, dtype: u32) -> String {
    let factor = match dtype {
        super::IQ3_XXS => {
            "let base=row+(block/8u)*98u;let group=block%8u;let word=NAME_word(base+66u+group*4u);return NAME_half(base)*(0.5+f32(word>>28u))*0.5;"
        }
        super::IQ3_S => {
            "let base=row+(block/8u)*110u;let group=block%8u;let scale=(NAME_byte(base+106u+group/2u)>>((group%2u)*4u))&15u;return NAME_half(base)*f32(1u+2u*scale);"
        }
        super::IQ2_S => {
            "let base=row+(block/8u)*82u;let group=block%8u;let scale=(NAME_byte(base+74u+group)>>(half*4u))&15u;return NAME_half(base)*(0.5+f32(scale))*0.25;"
        }
        super::IQ4_XS => {
            "let base=row+(block/8u)*136u;let group=block%8u;let lo=(NAME_byte(base+4u+group/2u)>>((group%2u)*4u))&15u;let hi=(NAME_half_bits(base+2u)>>(group*2u))&3u;return NAME_half(base)*f32(i32(lo|(hi<<4u))-32);"
        }
        super::IQ4_NL => "return NAME_half(row+block*18u);",
        super::Q8_0 => "return NAME_half(row+block*34u);",
        super::Q2_0 => "return NAME_half(row+(block/2u)*18u);",
        _ => unreachable!("validated packed integer format"),
    };
    let packed = match dtype {
        super::IQ3_XXS => {
            "let base=row+(block/8u)*98u;let group=block%8u;let octet=quartet/2u;let sign_word=NAME_word(base+66u+group*4u);let sign_code=(sign_word>>(octet*7u))&127u;let signs=sign_code|((countOneBits(sign_code)&1u)<<7u);let index=NAME_byte(base+2u+group*8u+quartet);return iq3_signed_word(iq3_grid[index],signs>>((quartet%2u)*4u));"
        }
        super::IQ3_S => {
            "let base=row+(block/8u)*110u;let group=block%8u;let index=NAME_byte(base+2u+group*8u+quartet)|(((NAME_byte(base+66u+group)>>quartet)&1u)<<8u);let signs=NAME_byte(base+74u+group*4u+quartet/2u);return iq3_signed_word(iq3_grid[index],signs>>((quartet%2u)*4u));"
        }
        super::IQ2_S => {
            "let base=row+(block/8u)*82u;let group=block%8u;let octet=quartet/2u;let index=NAME_byte(base+2u+group*4u+octet)|(((NAME_byte(base+66u+group)>>(octet*2u))&3u)<<8u);let signs=NAME_byte(base+34u+group*4u+octet);return iq3_signed_word(iq3_grid[index*2u+quartet%2u],signs>>((quartet%2u)*4u));"
        }
        super::IQ4_XS => {
            "let base=row+(block/8u)*136u+8u+(block%8u)*16u;return iq4_word(NAME_word(base+(quartet%4u)*4u),quartet>=4u);"
        }
        super::IQ4_NL => {
            "return iq4_word(NAME_word(row+block*18u+2u+(quartet%4u)*4u),quartet>=4u);"
        }
        super::Q8_0 => "return NAME_word(row+block*34u+2u+quartet*4u);",
        super::Q2_0 => "return q2_word(NAME_byte(row+(block/2u)*18u+2u+(block%2u)*8u+quartet));",
        _ => unreachable!("validated packed integer format"),
    };
    let accumulation = if dtype == super::IQ2_S {
        // IQ2_S owns distinct16-element subscales inside each32-element
        // activation plane. Never apply one half's factor to all32 codes.
        r#"for(var half=0u;half<2u;half++) {
            var first=0;var second=0;
            for(var quartet=half*4u;quartet<half*4u+4u;quartet++) {
                let q=NAME_packed(row,block,quartet);let at=block*8u+quartet;
                first+=dot4I8Packed(q,packed_first[at]);
                second+=dot4I8Packed(q,packed_second[at]);
            }
            let pair=scale_first[block]*f32(first)+scale_second[block]*f32(second);
            sum+=NAME_factor(row,block,half)*pair;
        }"#
    } else {
        r#"var first=0;var second=0;
        for(var quartet=0u;quartet<8u;quartet++) {
            let q=NAME_packed(row,block,quartet);let at=block*8u+quartet;
            first+=dot4I8Packed(q,packed_first[at]);
            second+=dot4I8Packed(q,packed_second[at]);
        }
        let pair=scale_first[block]*f32(first)+scale_second[block]*f32(second);
        sum+=NAME_factor(row,block,0u)*pair;"#
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
fn NAME_half(at:u32)->f32 {{return unpack2x16float(NAME_half_bits(at)).x;}}
fn NAME_factor(row:u32,block:u32,half:u32)->f32 {{{factor}}}
fn NAME_packed(row:u32,block:u32,quartet:u32)->u32 {{{packed}}}
fn NAME_project(row:u32,lane:u32,width:u32)->f32 {{
    var sum=0.0;
    for(var block=lane;block<BLOCKS_VALUEu;block+=width) {{
        // Exact bounded i32 sums precede explicit F32 scale/add round points.
        // This alternate reduction/activation policy is independently tested.
        {accumulation}
    }}
    return subgroupAdd(sum);
}}
"#
    )
    .replace("NAME", name)
}

impl Kernel {
    fn source(self) -> String {
        self.source_variant(false)
    }
    fn source_variant(self, prepared: bool) -> String {
        let columns = if self.down { 640 } else { 2560 };
        let rows = if self.down { 2560 } else { 640 };
        let (bindings, helpers, input_offset, epilogue) = if self.down {
            (
                "@group(0) @binding(0) var<storage,read> down:array<u32>;\n@group(0) @binding(1) var<storage,read> input:array<f32>;\n@group(0) @binding(2) var<storage,read> map:array<u32>;\n@group(0) @binding(3) var<storage,read_write> output:array<f32>;\n@group(0) @binding(4) var<storage,read_write> status:array<u32>;\n@group(0) @binding(5) var<uniform> p:Params;",
                projection("down", self.dtype),
                "route*p.c",
                "var sum=0.0;if valid && row<ROWS {sum=down_project((slot*ROWS+row)*ROW_BYTES,lane,width);}\nif valid && row<ROWS && lane==0u {output[p.d+route*p.e+row]=sum;if row==0u {status[p.f+80u+route]=1u;}}",
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
                "var a=0.0;var b=0.0;if valid && row<ROWS {let base=(slot*ROWS+row)*ROW_BYTES;a=gate_project(base,lane,width);b=up_project(base,lane,width);}\nif valid && row<ROWS && lane==0u {var sig=0.0;if a>=0.0 {sig=1.0/(1.0+exp(-a));}else{let e=exp(a);sig=e/(1.0+e);}\noutput[p.d+route*p.e+row]=a*sig*b;if row==0u {status[p.f+route]=1u;}}",
            )
        };
        let bindings = if prepared {
            // Custom CubeTask repr=None uses auto-reflected wgpu layouts.
            // Do not supply obsolete input bindings: prepared projection reads
            // only planes, and an entirely unused global is absent from its
            // entrypoint layout. All remaining resources are densely numbered.
            if self.down {
                "@group(0) @binding(0) var<storage,read> down:array<u32>;\n@group(0) @binding(1) var<storage,read> map:array<u32>;\n@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n@group(0) @binding(3) var<storage,read_write> status:array<u32>;\n@group(0) @binding(4) var<storage,read> planes:array<u32>;\n@group(0) @binding(5) var<uniform> p:Params;"
            } else {
                "@group(0) @binding(0) var<storage,read> gate:array<u32>;\n@group(0) @binding(1) var<storage,read> up:array<u32>;\n@group(0) @binding(2) var<storage,read> map:array<u32>;\n@group(0) @binding(3) var<storage,read_write> output:array<f32>;\n@group(0) @binding(4) var<storage,read_write> status:array<u32>;\n@group(0) @binding(5) var<storage,read> planes:array<u32>;\n@group(0) @binding(6) var<uniform> p:Params;"
            }.to_string()
        } else {
            bindings.to_string()
        };
        let (grid, initialize) = grid_source(&original_grid_words(self.dtype));
        let decode_helpers = include_str!("iq_moe_integer.wgsl")
            .split_once("fn iq4_byte")
            .unwrap()
            .1
            .split_once("// Original byte-addressed")
            .unwrap()
            .0;
        let mut source = (if prepared {
            include_str!("iq_moe_integer_prepared_project.wgsl")
        } else {
            include_str!("iq_moe_integer.wgsl")
        })
        .replace("BINDINGS_VALUE", &bindings)
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
        .replace(
            "EPILOGUE_VALUE",
            &epilogue.replace("valid && row<ROWS", "ready && row<ROWS"),
        )
        .replace("GRID_DECLARATION_VALUE", &grid)
        .replace("GRID_INITIALIZE_VALUE", &initialize)
        .replace(
            "DECODE_HELPERS_VALUE",
            &format!("fn iq4_byte{decode_helpers}"),
        )
        .replace(
            "PREPARED_DOWN_VALUE",
            if self.down { "true" } else { "false" },
        );
        for (key, value) in [
            ("COLUMNS_VALUE", columns.to_string()),
            ("ROWS_VALUE", rows.to_string()),
            ("ROW_BYTES_VALUE", self.row_bytes.to_string()),
            ("DTYPE_VALUE", self.dtype.to_string()),
            ("FIRST_SLOT_VALUE", self.first.to_string()),
            ("SLOTS_VALUE", self.slots.to_string()),
            ("PACKED_VALUE", (columns / 4).to_string()),
            ("BLOCKS_VALUE", (columns / 32).to_string()),
        ] {
            source = source.replace(key, &value);
        }
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        if self.down {
            "IQMoeIntegerDown"
        } else {
            "IQMoeIntegerGateUp"
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
pub(crate) fn validate_geometry(geometry: g::Geometry, first: usize, slots: usize) -> Result<()> {
    let [gate, up, down] = geometry.projections();
    if geometry.hidden() != 2560
        || geometry.ff() != 640
        || geometry.tile() != 1
        || geometry.capacity() != crate::daslab_profile::BANK_SLOTS
        || geometry.gate_policy() != g::GatePolicy::Generic
        || !matches!(
            gate.dtype(),
            super::IQ3_XXS | super::IQ3_S | super::IQ2_S | super::IQ4_XS
        )
        || up.dtype() != gate.dtype()
        || !matches!(down.dtype(), super::IQ4_NL | super::Q8_0 | super::Q2_0)
        || !matches!((first, slots), (0 | 72, 72))
    {
        return Err(Error::InvalidFormat(
            "fixed original-packed IQ integer geometry",
        ));
    }
    geometry.validate_count(1).map_err(Error::InvalidFormat)
}
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

    fn compile_public_source(source: &str) -> (wgpu::naga::Module, Vec<u32>) {
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::SUBGROUP
                | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)
        .unwrap();
        let spirv = wgpu::naga::back::spv::write_vec(
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
        (module, spirv)
    }

    fn switches(spirv: &[u32]) -> Vec<usize> {
        let mut cases = Vec::new();
        let mut at = 5;
        while at < spirv.len() {
            let length = (spirv[at] >> 16) as usize;
            let opcode = spirv[at] & 65535;
            assert!(length > 0 && at + length <= spirv.len());
            // OpSwitch on the original32-bit selector owns two fixed operands
            // (selector/default), then literal/target pairs for every case.
            if opcode == 251 {
                assert!(length >= 3 && (length - 3).is_multiple_of(2));
                cases.push((length - 3) / 2);
            }
            at += length;
        }
        cases
    }

    fn constant_grid_stores(
        block: &wgpu::naga::Block,
        expressions: &wgpu::naga::Arena<wgpu::naga::Expression>,
        grid: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
        stores: &mut Vec<(u32, u32)>,
    ) {
        use wgpu::naga::{Expression as E, Literal as L, Statement as S};
        for statement in block {
            match statement {
                S::Store { pointer, value } => {
                    if let E::AccessIndex { base, index } = expressions[*pointer]
                        && matches!(expressions[base], E::GlobalVariable(v) if v==grid)
                    {
                        let E::Literal(L::U32(bits)) = expressions[*value] else {
                            panic!("grid store must contain only its literal original word");
                        };
                        stores.push((index, bits));
                    }
                }
                S::Block(inner) => constant_grid_stores(inner, expressions, grid, stores),
                S::If { accept, reject, .. } => {
                    constant_grid_stores(accept, expressions, grid, stores);
                    constant_grid_stores(reject, expressions, grid, stores);
                }
                S::Switch { cases, .. } => {
                    for case in cases {
                        constant_grid_stores(&case.body, expressions, grid, stores);
                    }
                }
                S::Loop {
                    body, continuing, ..
                } => {
                    constant_grid_stores(body, expressions, grid, stores);
                    constant_grid_stores(continuing, expressions, grid, stores);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn grid_initializer_has_exact_original_bits_once_and_bounded_lane_switch() {
        use std::fmt::Write;
        for dtype in [
            super::super::IQ3_XXS,
            super::super::IQ3_S,
            super::super::IQ2_S,
        ] {
            let words = original_grid_words(dtype);
            let kernel = Kernel {
                down: false,
                dtype,
                row_bytes: kuno_text::TensorType::from_gguf_id(dtype)
                    .unwrap()
                    .row_bytes(2560)
                    .unwrap() as usize,
                first: 0,
                slots: 72,
            };
            let source = kernel.source();
            assert!(!source.contains("iq3_literal"));
            assert_eq!(source.matches("workgroupBarrier();").count(), 2);
            let (module, spirv) = compile_public_source(&source);
            assert_eq!(switches(&spirv), [128]);
            let grid = module
                .global_variables
                .iter()
                .find(|(_, variable)| variable.name.as_deref() == Some("iq3_grid"))
                .unwrap()
                .0;
            let entry = &module.entry_points[0].function;
            let lane_cases = entry
                .body
                .iter()
                .find_map(|statement| match statement {
                    wgpu::naga::Statement::Switch { cases, .. } => Some(cases),
                    _ => None,
                })
                .unwrap();
            let mut lanes = Vec::new();
            for case in lane_cases {
                assert!(!case.fall_through);
                let mut owned = Vec::new();
                constant_grid_stores(&case.body, &entry.expressions, grid, &mut owned);
                match case.value {
                    wgpu::naga::SwitchValue::U32(lane) => {
                        assert!(lane < 128);
                        lanes.push(lane);
                        assert_eq!(owned.len(), words.len() / 128);
                        assert!(owned.iter().all(|&(index, _)| index % 128 == lane));
                    }
                    wgpu::naga::SwitchValue::Default => assert!(owned.is_empty()),
                    _ => panic!("only original local-invocation lanes are permitted"),
                }
            }
            lanes.sort_unstable();
            assert_eq!(lanes, (0..128).collect::<Vec<_>>());
            let mut stores = Vec::new();
            constant_grid_stores(&entry.body, &entry.expressions, grid, &mut stores);
            stores.sort_by_key(|&(index, _)| index);
            assert_eq!(stores.len(), words.len());
            for (index, (&expected, &(actual_index, bits))) in words.iter().zip(&stores).enumerate()
            {
                assert_eq!(actual_index as usize, index);
                assert_eq!(bits, expected);
            }
            // Compile the exact prior literal-lookup scheme through the same
            // backend. This proves removal of the large switch in generated
            // SPIRV, not merely in its WGSL spelling; hardware speed is separate.
            let (declaration, initialize) = grid_source(&words);
            let mut legacy = format!("{declaration}\nfn iq3_literal(i:u32)->u32 {{switch i {{\n");
            for (index, &word) in words.iter().enumerate() {
                writeln!(legacy, "case {index}u: {{return 0x{word:08x}u;}}").unwrap();
            }
            legacy.push_str("default: {return 0u;}\n}}\n");
            let prior = source.replace(&declaration, &legacy).replace(
                &initialize,
                &format!(
                    "for(var i=local;i<{}u;i+=128u) {{iq3_grid[i]=iq3_literal(i);}}",
                    words.len()
                ),
            );
            let (_, previous_spirv) = compile_public_source(&prior);
            assert_eq!(switches(&previous_spirv), [words.len()]);
            println!(
                "public_iq_grid_lowering dtype={dtype} words={} previous_cases={} current_cases=128 exact_constant_stores=1",
                words.len(),
                words.len()
            );
        }
    }

    #[test]
    fn q2_four_code_bytes_preserve_all256_patterns_without_borrow() {
        for byte in 0..=255u32 {
            let packed = (byte & 3)
                | (((byte >> 2) & 3) << 8)
                | (((byte >> 4) & 3) << 16)
                | (((byte >> 6) & 3) << 24);
            let signed = ((packed + 0x7f7f7f7f) ^ 0x80808080).to_le_bytes();
            for (lane, &value) in signed.iter().enumerate() {
                assert_eq!(value as i8, ((byte >> (2 * lane)) & 3) as i8 - 1);
            }
        }
    }
    #[test]
    fn all48_actual_das_layers_have_exact_original_integer_projection_geometry() {
        for layer in 0..48 {
            let dtype = |name| {
                crate::daslab_profile::expected_dtype(&format!("blk.{layer}.{name}.weight"))
                    .unwrap()
            };
            let gate = dtype("ffn_gate_exps");
            let up = dtype("ffn_up_exps");
            let down = dtype("ffn_down_exps");
            let geometry = g::Geometry::new(
                [
                    g::Projection::new(gate, 2560, 640, 512).unwrap(),
                    g::Projection::new(up, 2560, 640, 512).unwrap(),
                    g::Projection::new(down, 640, 2560, 512).unwrap(),
                ],
                1,
                144,
            )
            .unwrap();
            for first in [0, 72] {
                validate_geometry(geometry, first, 72).unwrap();
            }
            assert!(validate_geometry(geometry, 1, 72).is_err());
            assert!(validate_geometry(geometry, 0, 73).is_err());
            assert!(workgroup_bytes(gate) <= MAX_SHARED_BYTES);
        }
    }
    #[test]
    fn mixed_iq_integer_shaders_use_shared_grid_not_function_array_copies() {
        for (down, dtype, columns) in [
            (false, super::super::IQ3_XXS, 2560),
            (false, super::super::IQ4_XS, 2560),
            (false, super::super::IQ3_S, 2560),
            (false, super::super::IQ2_S, 2560),
            (true, super::super::IQ4_NL, 640),
            (true, super::super::Q8_0, 640),
            (true, super::super::Q2_0, 640),
        ] {
            for first in [0, 72] {
                let row_bytes = kuno_text::TensorType::from_gguf_id(dtype)
                    .unwrap()
                    .row_bytes(columns)
                    .unwrap() as usize;
                let source = Kernel {
                    down,
                    dtype,
                    row_bytes,
                    first,
                    slots: 72,
                }
                .source();
                assert!(!source.contains("_VALUE"));
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                let info = wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::SUBGROUP
                        | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
                )
                .validate(&module)
                .unwrap();
                assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
                assert!(
                    source.contains("dot4I8Packed")
                        && source.contains("prepare_q8(local)")
                        && source.contains("subgroupAdd")
                );
                assert!(module.constants.iter().all(|(_, c)| !matches!(
                    module.types[c.ty].inner,
                    wgpu::naga::TypeInner::Array { .. }
                )));
                let grids = module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| v.name.as_deref() == Some("iq3_grid"))
                    .collect::<Vec<_>>();
                assert_eq!(
                    grids.len(),
                    usize::from(matches!(
                        dtype,
                        super::super::IQ3_XXS | super::super::IQ3_S | super::super::IQ2_S
                    ))
                );
                if let Some((_, grid)) = grids.first() {
                    assert_eq!(grid.space, wgpu::naga::AddressSpace::WorkGroup);
                }
                let shared = module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| v.space == wgpu::naga::AddressSpace::WorkGroup)
                    .map(|(_, v)| match module.types[v.ty].inner {
                        wgpu::naga::TypeInner::Array {
                            size: wgpu::naga::ArraySize::Constant(n),
                            stride,
                            ..
                        } => n.get() * stride,
                        wgpu::naga::TypeInner::Atomic(_) => 4,
                        _ => panic!("only declared bounded workgroup owners are permitted"),
                    })
                    .sum::<u32>();
                assert_eq!(shared, workgroup_bytes(dtype) as u32);
                let spirv = wgpu::naga::back::spv::write_vec(
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
                // Inspect actual backend lowering: no Function-address array
                // copy of the grid/input/planes, and native signed packed dot
                // instructions exist when the backend permits their capability.
                let mut arrays = std::collections::HashSet::new();
                let mut pointers = std::collections::HashMap::new();
                let mut variables = Vec::new();
                let mut dots = 0;
                let mut at = 5;
                while at < spirv.len() {
                    let length = (spirv[at] >> 16) as usize;
                    let opcode = spirv[at] & 65535;
                    assert!(length > 0 && at + length <= spirv.len());
                    let args = &spirv[at + 1..at + length];
                    match opcode {
                        28 | 29 => {
                            arrays.insert(args[0]);
                        }
                        32 => {
                            pointers.insert(args[0], args[2]);
                        }
                        59 => variables.push((args[0], args[2])),
                        4450 => dots += 1,
                        _ => {}
                    }
                    at += length;
                }
                assert!(dots > 0);
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
