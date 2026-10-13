//! Unused true-batched original-Q6/two-residual-Q8 projection leaf.
//! Parent authenticates full final-head roots/epoch and retains the scratch
//! lease through all original tiles, completion, full clear and erasure.
#![allow(dead_code)] // The separate parent owns runtime policy/connection.
use super::{Spec, span, stream, wide_weight_view, words};
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrepareSpans {
    pub count: usize,
    pub input_first: usize,
    pub input_stride: usize,
    pub planes_first: usize,
    pub planes_stride: usize,
    pub fault_first: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectSpans {
    pub count: usize,
    pub packed_first: usize,
    pub planes_first: usize,
    pub planes_stride: usize,
    pub output_first: usize,
    pub output_stride: usize,
    pub fault_first: usize,
    pub matrix_first: usize,
    pub matrix_rows: usize,
}
fn rows_span(count: usize, first: usize, stride: usize, width: usize, extent: usize) -> bool {
    (1..=4).contains(&count)
        && stride >= width
        && stride <= u32::MAX as usize
        && (count - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(width))
            .is_some_and(|length| span(first, length, extent))
}
impl PrepareSpans {
    pub(crate) fn validate(self, spec: Spec, extents: [usize; 3]) -> Result<()> {
        if !rows_span(
            self.count,
            self.input_first,
            self.input_stride,
            spec.columns(),
            extents[0],
        ) || !rows_span(
            self.count,
            self.planes_first,
            self.planes_stride,
            spec.plane_words(),
            extents[1],
        ) || !span(self.fault_first, 1, extents[2])
        {
            return Err(Error::InvalidBufferLength);
        }
        Ok(())
    }
}
impl ProjectSpans {
    pub(crate) fn validate(
        self,
        spec: Spec,
        weight_bytes: usize,
        extents: [usize; 3],
    ) -> Result<()> {
        if !span(self.packed_first, spec.weight_bytes(), weight_bytes)
            || !rows_span(
                self.count,
                self.planes_first,
                self.planes_stride,
                spec.plane_words(),
                extents[0],
            )
            || !rows_span(
                self.count,
                self.output_first,
                self.output_stride,
                spec.rows(),
                extents[1],
            )
            || !rows_span(
                self.count,
                self.matrix_first,
                self.output_stride,
                self.matrix_rows,
                extents[1],
            )
            || !span(self.fault_first, 1, extents[2])
            || self.matrix_rows == 0
            || self.output_first < self.matrix_first
            || self
                .output_first
                .checked_sub(self.matrix_first)
                .and_then(|n| n.checked_add(spec.rows()))
                .is_none_or(|end| end > self.matrix_rows)
        {
            return Err(Error::InvalidBufferLength);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Prepare(usize),
    ProjectThree(Spec, bool, bool),
    ProjectFour(Spec, bool, bool),
    ProjectPairedThree(Spec, bool, bool),
    ProjectPairedFour(Spec, bool, bool),
}
fn section(source: &str, begin: &str, end: &str, keep: bool) -> String {
    let (before, rest) = source.split_once(begin).unwrap();
    let (body, after) = rest.split_once(end).unwrap();
    format!("{before}{}{after}", if keep { body } else { "" })
}
const OLD_FINITE: &str =
    "fn finite(x:f32)->bool {return x<=3.4028234663852886e38 && x>=-3.4028234663852886e38;}";
const BIT_FINITE: &str =
    "fn finite(x:f32)->bool {return (bitcast<u32>(x)&0x7f800000u)!=0x7f800000u;}";
fn weight_helpers(wide: bool) -> String {
    let (_, project) = include_str!("q6_integer_matvec.wgsl")
        .split_once("// PROJECT_STAGE")
        .unwrap();
    let (project, wide_helpers) = project.split_once("// WIDE_WEIGHT_HELPERS").unwrap();
    let helpers = project.split_once("@compute").unwrap().0;
    if wide {
        let (before, rest) = helpers.split_once("// WEIGHT_HELPERS_BEGIN").unwrap();
        let (_, after) = rest.split_once("// WEIGHT_HELPERS_END").unwrap();
        format!("{before}\n{wide_helpers}\n{after}").replace(OLD_FINITE, BIT_FINITE)
    } else {
        helpers.replace(OLD_FINITE, BIT_FINITE)
    }
}
impl Kernel {
    fn source(self) -> String {
        let (prepare_guard, stages) = include_str!("q6_integer_batch.wgsl")
            .split_once("// PROJECT_BATCH_STAGE")
            .unwrap();
        if let Self::Prepare(columns) = self {
            if crate::strata_port::enabled() {
                return format!(
                    "requires packed_4x8_integer_dot_product;\nconst COLUMNS:u32={columns}u;const PACKED_WORDS:u32={}u;const BLOCKS:u32={}u;\nstruct Params {{a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}}\n@group(0) @binding(0) var<storage,read> input:array<f32>;\n@group(0) @binding(1) var<storage,read_write> planes:array<u32>;\n@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(3) var<uniform> p:Params;\n{}",
                    columns / 4,
                    columns / 32,
                    include_str!("strata-q6-prepare.wgsl")
                        .replace("// STRATA_BATCH_PREPARE_GUARD", prepare_guard)
                );
            }
            // Use the exact original quantizer/body. Only the guarded base
            // address gains its independent real member offset.
            return super::Kernel::Prepare(columns)
                .source()
                .replace(
                    "let block=id.x;if block>=COLUMNS/32u {return;}",
                    prepare_guard,
                )
                .replace("input[p.a+", "input[input_first+")
                .replace("planes[p.b+", "planes[planes_first+")
                .replace(OLD_FINITE, BIT_FINITE)
                .replace(
                    "!(abs(x)<3.4028234663852886e38)",
                    "(bitcast<u32>(x)&0x7fffffffu)>=0x7f7fffffu",
                );
        }
        let (spec, tail, wide, four, paired) = match self {
            Self::ProjectThree(spec, tail, wide) => (spec, tail, wide, false, false),
            Self::ProjectFour(spec, tail, wide) => (spec, tail, wide, true, false),
            Self::ProjectPairedThree(spec, tail, wide) => (spec, tail, wide, false, true),
            Self::ProjectPairedFour(spec, tail, wide) => (spec, tail, wide, true, true),
            Self::Prepare(_) => unreachable!(),
        };
        let (project, paired_stage) = stages.split_once("// PROJECT_PAIRED_STAGE").unwrap();
        let mut body = project.to_owned();
        if paired {
            let (helpers, paired_loop) = paired_stage.split_once("// PAIRED_LOOP").unwrap();
            let paired_loop = paired_loop.strip_prefix('\n').unwrap();
            let (before, rest) = body
                .split_once("        for(var group=lane;group<COLUMNS/16u;group+=width) {")
                .unwrap();
            let (_, after) = rest
                .split_once("    }\n    let total=subgroupAdd(sum);")
                .unwrap();
            let payload = if wide {
                "fn pair_words_at(at:u32)->vec4<u32> {return words_at(at);}\n"
            } else {
                "fn pair_words_at(at:u32)->vec4<u32> {return vec4<u32>(word_at(at),word_at(at+4u),word_at(at+8u),word_at(at+12u));}\n"
            };
            body = format!(
                "{payload}{helpers}{before}{paired_loop}    }}\n    let total=subgroupAdd(sum);{after}"
            );
        }
        for (begin, end) in [
            ("// FOUR_SUM_BEGIN", "// FOUR_SUM_END"),
            ("// FOUR_FINITE_BEGIN", "// FOUR_FINITE_END"),
            ("// FOUR_OUTPUT_BEGIN", "// FOUR_OUTPUT_END"),
        ] {
            body = section(&body, begin, end, four);
        }
        body = body.replace("VECTOR", if four { "vec4" } else { "vec3" });
        if crate::strata_port::enabled() {
            let first = body.find("fn member_pair(").expect("Q6 member helper");
            let end = body[first..].find("\n@compute").expect("Q6 helper end") + first;
            body.replace_range(
                first..end,
                r#"fn member_pair(codes:vec4<u32>,group:u32,member:u32)->f32 {
                let first=p.b+member*p.h;var a=0;
                for(var word=0u;word<4u;word++) {
                    a+=dot4I8Packed(codes[word],planes[first+group*4u+word]);
                }
                let d=bitcast<f32>(planes[first+2u*PACKED_WORDS+group/2u]);
                return d*f32(a);
            }"#,
            );
            body = body.replace("if !finite(factor0) || !finite(factor1) || !finite(x0) || !finite(x1) {atomicOr(&fault[p.d],1u);}", "")
                .replace("if !finite(factor0) || !finite(factor1) || !finite(y0) || !finite(y1) {atomicOr(&fault[p.d],1u);}", "")
                .replace("if !finite(factor0) || !finite(factor1) || !finite(z0) || !finite(z1) {atomicOr(&fault[p.d],1u);}", "")
                .replace("if !finite(factor0) || !finite(factor1) || !finite(w0) || !finite(w1) {atomicOr(&fault[p.d],1u);}", "");
            for axis in ["x", "y", "z", "w"] {
                body = body.replace(&format!("sum.{axis}+=factor0*{axis}0;sum.{axis}+=factor1*{axis}1;"),
                    &format!("sum.{axis}=fma(factor0,{axis}0,sum.{axis});sum.{axis}=fma(factor1,{axis}1,sum.{axis});"));
            }
        }

        let weight = if wide { "vec4<u32>" } else { "u32" };
        format!(
            "requires packed_4x8_integer_dot_product;\nconst COLUMNS:u32={}u;const PACKED_WORDS:u32={}u;const BLOCKS:u32={}u;const PLANE_WORDS:u32={}u;\nconst ROWS:u32={}u;const ROW_BYTES:u32={}u;const AUDIT_TAIL:bool={tail};const FOUR:bool={four};const MEMBERS:u32={}u;const WEIGHT_WORD_BYTES:u32={}u;\nstruct Params {{a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32,i:u32,j:u32,k:u32,l:u32}}\n@group(0) @binding(0) var<storage,read> weight:array<{weight}>;\n@group(0) @binding(1) var<storage,read> planes:array<u32>;\n@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(4) var<uniform> p:Params;\n{}\n{body}",
            spec.columns(),
            spec.columns() / 4,
            spec.columns() / 32,
            spec.plane_words(),
            spec.rows(),
            spec.row_bytes(),
            if four { 4 } else { 3 },
            if wide { 16 } else { 4 },
            weight_helpers(wide),
        )
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare(_) => "Q6ResidualQ8BatchPrepare",
            Self::ProjectThree(_, _, _) => "Q6IntegerBatchThree",
            Self::ProjectFour(_, _, _) => "Q6IntegerBatchFour",
            Self::ProjectPairedThree(_, _, _) => "Q6IntegerBatchPairedThree",
            Self::ProjectPairedFour(_, _, _) => "Q6IntegerBatchPairedFour",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((*self, crate::strata_port::enabled()))
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
fn project_info(spans: ProjectSpans) -> MetadataBindingInfo {
    let values = [
        spans.packed_first,
        spans.planes_first,
        spans.output_first,
        spans.fault_first,
        spans.matrix_first,
        spans.matrix_rows,
        spans.count,
        spans.planes_stride,
        spans.output_stride,
        0,
        0,
        0,
    ];
    MetadataBindingInfo::custom(
        values
            .chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    )
}
/// One original block quantizer per actual member/block. No allocation occurs.
pub(crate) fn prepare(
    client: &ComputeClient<WgpuRuntime>,
    spec: Spec,
    input: Handle,
    planes: Handle,
    fault: Handle,
    spans: PrepareSpans,
) -> Result<()> {
    stream(&[&input, &planes, &fault])?;
    spans.validate(spec, [words(&input)?, words(&planes)?, words(&fault)?])?;
    let args = KernelArguments::new()
        .with_buffer(input.binding())
        .with_buffer(planes.binding())
        .with_buffer(fault.binding())
        .with_info(super::info(&[
            spans.input_first,
            spans.planes_first,
            spans.fault_first,
            spans.input_stride,
            spans.planes_stride,
            0,
            spans.count,
            0,
        ]));
    // SAFETY: exact checked complete member intervals; parent freezes distinct
    // full original same-stream owners and retains this prepared epoch.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Prepare(spec.columns())),
            CubeCount::Static(
                (spec.columns() / 32).div_ceil(128) as u32,
                spans.count as u32,
                1,
            ),
            args,
        );
    }
    Ok(())
}
/// Decode each original weight group once and reuse it across real members.
/// Tail audit excludes all full active matrix intervals, never concurrent tiles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project(
    client: &ComputeClient<WgpuRuntime>,
    spec: Spec,
    weights: Handle,
    planes: Handle,
    output: Handle,
    fault: Handle,
    spans: ProjectSpans,
    audit_tail: bool,
) -> Result<()> {
    stream(&[&weights, &planes, &output, &fault])?;
    let weight_words = words(&weights)?;
    let output_words = words(&output)?;
    spans.validate(
        spec,
        weight_words * 4,
        [words(&planes)?, output_words, words(&fault)?],
    )?;
    let mut groups = spec.rows().div_ceil(4);
    if audit_tail {
        groups = groups.max(output_words.div_ceil(128));
    }
    if groups > 65535 {
        return Err(Error::LimitExceeded("Q6 integer batch output grid"));
    }
    let wide = wide_weight_view(weight_words);
    let kernel = if spans.count == 4 {
        Kernel::ProjectPairedFour(spec, audit_tail, wide)
    } else {
        Kernel::ProjectPairedThree(spec, audit_tail, wide)
    };
    let args = KernelArguments::new()
        .with_buffer(weights.binding())
        .with_buffer(planes.binding())
        .with_buffer(output.binding())
        .with_buffer(fault.binding())
        .with_info(project_info(spans));
    // SAFETY: checked original packed bytes and disjoint complete parent roots.
    // No selected RHS reads another member's prepared plane or output interval.
    unsafe {
        client.launch_unchecked(
            Box::new(kernel),
            CubeCount::Static(groups as u32, 1, 1),
            args,
        );
    }
    Ok(())
}

/// Complete port head shader set, including guarded preparation and real tails.
pub(crate) fn strata_compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut kernels = vec![Kernel::Prepare(2560)];
    for rows in [16384, 2560] {
        let spec = Spec::new(rows, 2560).unwrap();
        for wide in [false, true] {
            for tail in [false, true] {
                kernels.push(Kernel::ProjectPairedThree(spec, tail, wide));
                kernels.push(Kernel::ProjectPairedFour(spec, tail, wide));
            }
        }
    }
    kernels
        .into_iter()
        .map(|kernel| {
            kernel
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap()
        })
        .collect()
}

#[cfg(test)]
#[path = "q6_integer_batch_tests.rs"]
mod tests;
