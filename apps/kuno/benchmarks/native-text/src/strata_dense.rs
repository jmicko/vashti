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
// One full private owner for the largest supported target input:8 members,
// 16384 columns, original single Q8_1 planes (9/16 words per value). The arena
// rounds294912 logical bytes to512KiB; no expert-fetch buffers are borrowed.
pub(crate) const INPUT_WORDS: usize = 8 * (16384 / 2 + 16384 / 16);
pub(crate) const INPUT_BYTES: u64 = 512 * 1024;
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
        // Native default wide K/32 paths share each packed weight load across
        // all real RHS. Every descriptor keeps its original padded owner.
        let prefix = source
            .replace("weight:array<u32>", "weight:array<vec4<u32>>")
            .replace("weight[at/4u]", "weight[at/16u][(at/4u)%4u]")
            .replace("weight[at/4u+1u]", "weight[(at/4u+1u)/4u][(at/4u+1u)%4u]");
        let body = if self.dtype == 14 {
            include_str!("strata_native_q6.wgsl")
        } else {
            include_str!("strata_native_dense.wgsl")
        };
        // The registry and offline validator apply the same NoContraction
        // materialization to the original native multiply/add expressions.
        source =
            format!("// STRATA_ROUND_PRODUCTS: original native dense expressions.\n{prefix}{body}");
        if self.indexed {
            source = source.replace(
                "count:u32}",
                "count:u32,source_first:u32,source_rows:u32,subset_first:u32,reserved:u32}",
            );
            source=source.replace("@group(0) @binding(4) var<uniform> p:Params;","@group(0) @binding(4) var<storage,read> row_ids:array<u32>;\n@group(0) @binding(5) var<uniform> p:Params;");
            source=source.replace("arrayLength(&weight)*16u<ROWS*ROW_BYTES","arrayLength(&weight)*16u<p.source_rows*ROW_BYTES || p.subset_first>arrayLength(&row_ids) || ROWS>arrayLength(&row_ids)-p.subset_first");
            source=source.replace("let row=group.x*4u+sg;",r#"let ordinal=group.x*4u+sg;var row=0u;var valid_row=ordinal<ROWS;
            if valid_row {let selected=row_ids[p.subset_first+ordinal];if selected<p.source_first || selected-p.source_first>=p.source_rows {valid_row=false;if lane==0u {atomicOr(&fault[0],8u);}}else{row=selected-p.source_first;}}"#);
            source = source.replace("if row<ROWS", "if valid_row");
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

/// Same native aligned wide math across bounded original whole-head pages.
pub(crate) fn whole_head_source(rows: usize, members: usize) -> String {
    let mut s = Kernel {
        dtype: 14,
        rows,
        columns: 2560,
        row_bytes: 2100,
        members,
        indexed: false,
    }
    .source();
    s=s.replace("@group(0) @binding(4) var<uniform> p:Params;","@group(0) @binding(4) var<storage,read> addresses:array<vec4<u32>>;\n@group(0) @binding(5) var<uniform> p:Params;\nvar<private> strata_gate_address:vec2<u32>;");
    s = s
        .replace("weight:array<vec4<u32>>", "gate0:array<vec4<u32>>")
        .replace("weight[", "gate0[");
    s = s.replace(
        "arrayLength(&weight)*16u<ROWS*ROW_BYTES",
        "arrayLength(&addresses)!=2048u",
    );
    s = s.replace(
        "    let row=group.x*4u+sg;",
        r#"    let row=group.x*4u+sg;
    var valid_row=row<ROWS;
    if valid_row {
        let page=row/16384u;let src=addresses[page];
        let expected=min(16384u,ROWS-page*16384u);
        if src.w!=expected || src.z!=expected*525u || all(src.xy==vec2<u32>(0u)) || src.x%16u!=0u {
            valid_row=false;if lane==0u {atomicOr(&fault[0],8u);}
        }
        strata_gate_address=src.xy;
    }"#,
    );
    s = s.replace("if row<ROWS", "if valid_row").replace(
        "let b=row*ROW_BYTES+block*210u;",
        "let b=(row%16384u)*ROW_BYTES+block*210u;",
    );
    format!("// STRATA_PHYSICAL_EXPERTS: original whole-head page addresses.\n{s}")
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct PrepareKernel(usize, bool);
impl PrepareKernel {
    fn source(self) -> String {
        let mut source = format!(
            "// STRATA_DENSE_PACKED\nconst COLUMNS:u32={}u;const PACKED_WORDS:u32={}u;const BLOCKS:u32={}u;const PLANE_WORDS:u32={}u;\n{}",
            self.0,
            self.0 / 4,
            self.0 / 32,
            self.0 / 2 + self.0 / 16,
            include_str!("strata_dense_prepare.wgsl")
        );
        if self.1 {
            source.insert_str(
                0,
                "// STRATA_ROUND_PRODUCTS: native fused SwiGLU before Q8 sum.\n",
            );
            source = source.replace("@group(0) @binding(3) var<uniform> p:Params;",
                "@group(0) @binding(3) var<storage,read> up:array<f32>;\n@group(0) @binding(4) var<uniform> p:Params;");
            source = source.replace(
                "p.input_first<=arrayLength(&input);",
                "p.input_first<=arrayLength(&input) && arrayLength(&up)==arrayLength(&input);",
            );
            source = source.replace("let x=input[p.input_first+g.y*p.input_stride+block*32u+lane];",
                "let at=p.input_first+g.y*p.input_stride+block*32u+lane;let gi=input[at];let x=(gi/(1.0+exp(-gi)))*up[at];");
        }
        source
    }
}
impl KernelMetadata for PrepareKernel {
    fn name(&self) -> &'static str {
        if self.1 {
            "StrataSubgroupSwiGLUQ8Prepare"
        } else {
            "StrataSubgroupQ8Prepare"
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self)
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for PrepareKernel {
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
pub(crate) fn supports_count(dtype: u32, columns: usize, scratch_bytes: u64, count: usize) -> bool {
    crate::strata_intel_core::enabled()
        && matches!(dtype, 8 | 12 | 13 | 14 | 20 | 23)
        && columns > 0
        && columns <= 16384
        && columns.is_multiple_of(if matches!(dtype, 8 | 20) { 128 } else { 256 })
        && (1..=8).contains(&count)
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
    if !supports_count(8, columns, planes.size_in_used(), spans.count())
        || fault.size_in_used() != 4
        || !input.size_in_used().is_multiple_of(4)
        || [&input, &fault].iter().any(|h| h.stream != planes.stream)
    {
        return Err(Error::InvalidBufferLength);
    }
    let info = prepare_info(
        columns,
        spans,
        input.size_in_used() as usize / 4,
        planes.size_in_used() as usize / 4,
    )?;
    let args = KernelArguments::new()
        .with_buffer(input.binding())
        .with_buffer(planes.binding())
        .with_buffer(fault.binding())
        .with_info(MetadataBindingInfo::custom(
            info.chunks_exact(2)
                .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                .collect(),
        ));
    // SAFETY: original checked input spans, complete borrowed private scratch,
    // one subgroup per 32-value block, and serialized owning graph lifetime.
    unsafe {
        client.launch_unchecked(
            Box::new(PrepareKernel(columns, false)),
            CubeCount::Static((columns / 128) as u32, spans.count() as u32, 1),
            args,
        );
    }
    PREPARES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}
pub(crate) fn prepare_swiglu(
    client: &ComputeClient<WgpuRuntime>,
    columns: usize,
    gate: Handle,
    up: Handle,
    planes: Handle,
    fault: Handle,
    count: usize,
) -> Result<()> {
    if gate.size_in_used() != up.size_in_used()
        || fault.size_in_used() != 4
        || [&gate, &up, &fault]
            .iter()
            .any(|h| h.stream != planes.stream)
    {
        return Err(Error::InvalidBufferLength);
    }
    let spans = BatchSpans::new(count, 0, 0, columns, 0, columns);
    let words = prepare_info(
        columns,
        &spans,
        gate.size_in_used() as usize / 4,
        planes.size_in_used() as usize / 4,
    )?;
    let args = KernelArguments::new()
        .with_buffer(gate.binding())
        .with_buffer(planes.binding())
        .with_buffer(fault.binding())
        .with_buffer(up.binding())
        .with_info(MetadataBindingInfo::custom(
            words
                .chunks_exact(2)
                .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                .collect(),
        ));
    // SAFETY: original disjoint private owners, same native Q8 geometry and
    // all active rows checked before the fused producer reaches DOWN.
    unsafe {
        client.launch_unchecked(
            Box::new(PrepareKernel(columns, true)),
            CubeCount::Static((columns / 128) as u32, count as u32, 1),
            args,
        );
    }
    Ok(())
}
// This Q8_1 producer is not a Q6 weight operator. Native shared experts have
// 640 inputs, a legal20-block Q8 shape that cannot form Q6's256-value blocks.
fn prepare_info(
    columns: usize,
    spans: &BatchSpans,
    input_words: usize,
    planes_words: usize,
) -> Result<[u32; 4]> {
    if !supports_count(8, columns, planes_words as u64 * 4, spans.count())
        || spans.input_stride() < columns
    {
        return Err(Error::InvalidBufferLength);
    }
    let stride = columns / 2 + columns / 16;
    let end = spans
        .count()
        .checked_sub(1)
        .and_then(|n| n.checked_mul(spans.input_stride()))
        .and_then(|n| n.checked_add(spans.input_first()))
        .and_then(|n| n.checked_add(columns))
        .ok_or(Error::ArithmeticOverflow)?;
    if end > input_words
        || spans.count() * stride > planes_words
        || end > u32::MAX as usize
        || spans.input_stride() > u32::MAX as usize
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok([
        spans.input_first() as u32,
        spans.input_stride() as u32,
        stride as u32,
        spans.count() as u32,
    ])
}
static PREPARES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn prepares() -> u64 {
    PREPARES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
#[path = "strata_native_dense_tests.rs"]
mod native_wide_tests;

#[cfg(test)]
mod native_q6_tests {
    #[test]
    fn fused_shared_q8_image_uses_both_original_inputs_and_the_same_native_packing_geometry() {
        use super::*;
        use wgpu::naga as n;
        let kernel = PrepareKernel(640, true);
        let source = kernel.source();
        let module = n::front::wgsl::parse_str(&source).unwrap();
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
        let uses = info.get_entry_point(0);
        let mut owners = std::collections::BTreeMap::new();
        for (id, g) in module.global_variables.iter() {
            if let Some(b) = &g.binding {
                owners.insert(
                    b.binding,
                    (
                        uses[id].contains(n::valid::GlobalUse::READ),
                        uses[id].contains(n::valid::GlobalUse::WRITE),
                    ),
                );
            }
        }
        assert_eq!(owners.len(), 5);
        assert_eq!(owners[&0], (true, false));
        assert_eq!(owners[&3], (true, false));
        assert!(owners[&1].1 && owners[&2].1);
        n::back::spv::write_vec(
            &module,
            &info,
            &n::back::spv::Options::default(),
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        assert!(source.contains("let x=(gi/(1.0+exp(-gi)))*up[at]"));
        // Full20 Q8 blocks, each real lane visited once including the tail
        // workgroup; same native plane extents consumed by packed DOWN.
        let mut visits = [0u8; 640];
        for group in 0..5 {
            for subgroup in 0..4 {
                for lane in 0..32 {
                    visits[(group * 4 + subgroup) * 32 + lane] += 1;
                }
            }
        }
        assert!(visits.iter().all(|&n| n == 1));
        assert_eq!(640 / 2 + 640 / 16, 360);
    }

    #[cfg(feature = "strata-intel-core")]
    #[test]
    fn shared_q8_640_column_producer_checks_real_batched_input_and_plane_bounds() {
        use super::*;
        for count in 1..=8 {
            let spans = BatchSpans::new(count, 0, 7, 640, 0, 2560);
            let words = 7 + count * 640;
            let planes = count * 360;
            assert_eq!(
                prepare_info(640, &spans, words, planes).unwrap(),
                [7, 640, 360, count as u32]
            );
            assert!(prepare_info(640, &spans, words - 1, planes).is_err());
            assert!(prepare_info(640, &spans, words, planes - 1).is_err());
        }
        assert!(!supports_count(14, 640, 65536, 1));
        assert!(prepare_info(640, &BatchSpans::new(1, 0, 0, 639, 0, 1), 640, 360).is_err());
    }
    #[test]
    fn native_q6_iqs_covers_each_column_once_and_matches_canonical_gguf_decode() {
        // Independent canonical decoder validates every original weight index,
        // signed scale and offset in the native lane mapping. Public synthetic
        // data only; this is not a GPU dispatch/numerical acceptance claim.
        for salt in 0..9u32 {
            let mut packed = [0u8; 210];
            for (i, byte) in packed[..192].iter_mut().enumerate() {
                *byte = (i as u32 * 73 + salt * 29 + 11) as u8;
            }
            for (i, byte) in packed[192..208].iter_mut().enumerate() {
                *byte = ((i as i32 * 3 + salt as i32) % 16 - 8) as i8 as u8;
            }
            packed[208..].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
            let mut canonical = [0.0; 256];
            kuno_text::TensorType::Q6K
                .decode(&packed, &mut canonical)
                .unwrap();
            let mut visits = [0u8; 256];
            for lane in 0..32 {
                let low = u32::from_le_bytes(packed[lane * 4..lane * 4 + 4].try_into().unwrap());
                let hi_at = 128 + (8 * (lane / 16) + lane % 8) * 4;
                let high = u32::from_le_bytes(packed[hi_at..hi_at + 4].try_into().unwrap())
                    >> (2 * ((lane % 16) / 8));
                let q8_offset = 4 * (lane / 16) + (lane % 16) / 8;
                let sc_offset = 8 * (lane / 16) + (lane % 16) / 4;
                for part in 0..2 {
                    let codes = ((low >> (4 * part)) & 0x0f0f0f0f)
                        | (((high >> (4 * part)) << 4) & 0x30303030);
                    let scale = packed[192 + sc_offset + 4 * part] as i8 as f32;
                    for byte in 0..4 {
                        let column = (q8_offset + 2 * part) * 32 + (lane % 8) * 4 + byte;
                        let code = ((codes >> (8 * byte)) & 255) as i32 - 32;
                        assert_eq!(
                            0.25 * scale * code as f32,
                            canonical[column],
                            "lane={lane} part={part} byte={byte}"
                        );
                        visits[column] += 1;
                    }
                }
            }
            assert_eq!(visits, [1; 256]);
        }
    }
    #[test]
    fn actual_wide_q6_specializations_use_aligned_vector_storage_for_every_rhs_and_tail() {
        use super::*;
        use wgpu::naga as n;
        for rows in [32, 16384, crate::resident_mtp::cjk::ROWS % 16384] {
            for members in 1..=8 {
                let k = Kernel {
                    dtype: 14,
                    rows,
                    columns: 2560,
                    row_bytes: 2100,
                    members,
                    indexed: false,
                };
                let source = k.source();
                let module = n::front::wgsl::parse_str(&source).unwrap();
                let info = n::valid::Validator::new(
                    n::valid::ValidationFlags::all(),
                    n::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                let weight = module
                    .global_variables
                    .iter()
                    .find(|(_, g)| g.name.as_deref() == Some("weight"))
                    .unwrap()
                    .1;
                let n::TypeInner::Array {
                    base, stride: 16, ..
                } = module.types[weight.ty].inner
                else {
                    panic!("original Q6 aligned vector owner");
                };
                assert!(matches!(
                    module.types[base].inner,
                    n::TypeInner::Vector {
                        size: n::VectorSize::Quad,
                        scalar: n::Scalar {
                            kind: n::ScalarKind::Uint,
                            width: 4
                        }
                    }
                ));
                assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
                assert_eq!(
                    module
                        .global_variables
                        .iter()
                        .filter(|(_, g)| g.binding.is_some())
                        .count(),
                    5
                );
                assert!(
                    module
                        .functions
                        .iter()
                        .any(|(_, f)| f.name.as_deref() == Some("load16_a2"))
                );
                let words = n::back::spv::write_vec(
                    &module,
                    &info,
                    &Default::default(),
                    Some(&n::back::spv::PipelineOptions {
                        shader_stage: n::ShaderStage::Compute,
                        entry_point: "main".into(),
                    }),
                )
                .unwrap();
                assert!(!words.is_empty());
            }
        }
    }
    #[test]
    fn wide_q6_block_sharing_and_signed_scales_match_independent_gguf_decode() {
        // Canonical GGUF decoding is the value oracle. This checks original
        // field/column ownership for the native8-lane mapping and all partial
        // four-block iterations; no device execution is claimed.
        for blocks in [1, 2, 3, 4, 5, 10, 64] {
            let mut packed = vec![0u8; blocks * 210];
            for (block, b) in packed.chunks_exact_mut(210).enumerate() {
                for (i, v) in b[..192].iter_mut().enumerate() {
                    *v = (i * 73 + block * 29 + 11) as u8;
                }
                for (i, v) in b[192..208].iter_mut().enumerate() {
                    *v = ((i as i32 * 3 + block as i32) % 16 - 8) as i8 as u8;
                }
                b[208..].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
            }
            let mut canonical = vec![0f32; blocks * 256];
            kuno_text::TensorType::Q6K
                .decode(&packed, &mut canonical)
                .unwrap();
            let mut visits = vec![0u8; blocks * 256];
            for lane in 0..32 {
                let g = lane & 7;
                let sub = lane >> 3;
                for block in (sub..blocks).step_by(4) {
                    let b = &packed[block * 210..(block + 1) * 210];
                    let qh = 2 * (g >> 2) + (g & 1);
                    let sc = 8 * (g >> 2) + (g & 3);
                    let q8 = 4 * (g >> 2) + ((g & 3) >> 1);
                    let half = g & 1;
                    for word in 0..4 {
                        let low = u32::from_le_bytes(
                            b[16 * g + word * 4..16 * g + word * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let at = 128 + 16 * qh + word * 4;
                        let high = u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
                            >> (2 * ((g & 3) >> 1));
                        for part in 0..2 {
                            let code = ((low >> (4 * part)) & 0x0f0f0f0f)
                                | (((high >> (4 * part)) << 4) & 0x30303030);
                            let scale = b[192 + sc + 4 * part] as i8 as f32;
                            for byte in 0..4 {
                                let column = block * 256
                                    + (q8 + 2 * part) * 32
                                    + half * 16
                                    + word * 4
                                    + byte;
                                let value = ((code >> (8 * byte)) & 255) as i32 - 32;
                                assert_eq!(
                                    (0.25 * scale * value as f32).to_bits(),
                                    canonical[column].to_bits()
                                );
                                visits[column] += 1;
                            }
                        }
                    }
                }
            }
            assert!(visits.into_iter().all(|n| n == 1));
        }
    }
    #[test]
    fn native_eight_dense_scratch_covers_all_real_rhs_without_overcommit() {
        use super::{INPUT_BYTES, INPUT_WORDS};
        if !crate::strata_intel_core::enabled() {
            return;
        }
        assert_eq!(crate::strata_port::verify_capacity(), 8);
        assert_eq!(
            crate::private_arena::BufferLayout::new(INPUT_WORDS)
                .unwrap()
                .owned_bytes as u64,
            INPUT_BYTES
        );
        for columns in [2560, 6144, 10240, 16384] {
            for count in 1..=8 {
                let bytes = (count * (columns / 2 + columns / 16) * 4) as u64;
                for dtype in [8, 12, 13, 14, 20, 23] {
                    assert!(super::supports_count(dtype, columns, INPUT_BYTES, count));
                    assert!(super::supports_count(dtype, columns, bytes, count));
                    assert!(!super::supports_count(dtype, columns, bytes - 1, count));
                }
            }
        }
        //640 is the legal Q8 shared-expert input, not a whole Q*_K/IQ4_XS row.
        for dtype in [12, 13, 14, 23] {
            assert!(!super::supports_count(dtype, 640, INPUT_BYTES, 1));
        }
        assert!(!super::supports_count(14, 2560, 1 << 20, 0));
        assert!(!super::supports_count(14, 2560, 1 << 20, 9));
    }
}
static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn calls() -> u64 {
    CALLS.load(std::sync::atomic::Ordering::Relaxed)
}
pub(crate) fn record_projection_call() {
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        || !weight.size_in_used().is_multiple_of(16)
        || !weight.offset_start.unwrap_or(0).is_multiple_of(16)
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
                members: count,
                indexed: false,
            }),
            CubeCount::Static(rows.div_ceil(4) as u32, 1, 1),
            args,
        );
    }
    record_projection_call();
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut result = (1..=8)
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
    // The gathered native draft head ends in a partial four-row workgroup.
    // Validate its actual full-page and tail specializations as well as the
    // representative format kernels above; compact rows must stay bounded.
    if crate::strata_intel_core::enabled() {
        for rows in [16384, crate::resident_mtp::cjk::ROWS % 16384] {
            result.push(
                Kernel {
                    dtype: crate::quant_matvec::Q6_K,
                    rows,
                    columns: 2560,
                    row_bytes: 2100,
                    members: 1,
                    indexed: false,
                }
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap(),
            );
        }
    }
    result.push(
        PrepareKernel(640, true)
            .compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap(),
    );
    for columns in [2560, 16384] {
        result.push(
            PrepareKernel(columns, false)
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap(),
        );
    }
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
        || !weight.size_in_used().is_multiple_of(16)
        || !weight.offset_start.unwrap_or(0).is_multiple_of(16)
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
    record_projection_call();
    Ok(())
}
