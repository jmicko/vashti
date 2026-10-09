//! Opt-in original-Q6_K/two-residual-Q8 dense projection candidate.
//! Caller owns disjoint full private input/planes/output/fault pages, source
//! identity, stream/epoch, precharge, owning completion and erasure. No buffer
//! allocation or publication happens here. Offsets are runtime INFO, never keys.
//! This changes activation/reduction arithmetic; default F32 remains untouched.
//! Layout/provenance and independent synthetic C fixtures: q6_integer_fixtures.
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

#[path = "q6_integer_oracle.rs"]
pub(crate) mod oracle;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    rows: usize,
    columns: usize,
    row_bytes: usize,
}
impl Spec {
    pub(crate) fn new(rows: usize, columns: usize) -> Result<Self> {
        if rows == 0
            || rows > 16384
            || columns == 0
            || columns > 16384
            || !columns.is_multiple_of(256)
        {
            return Err(Error::InvalidFormat("Q6 integer public geometry"));
        }
        let row_bytes = columns / 256 * 210;
        if rows * row_bytes > 64 * 1024 * 1024 {
            return Err(Error::LimitExceeded("Q6 integer original tile"));
        }
        Ok(Self {
            rows,
            columns,
            row_bytes,
        })
    }
    pub(crate) fn rows(self) -> usize {
        self.rows
    }
    pub(crate) fn columns(self) -> usize {
        self.columns
    }
    pub(crate) fn row_bytes(self) -> usize {
        self.row_bytes
    }
    pub(crate) fn weight_bytes(self) -> usize {
        self.rows * self.row_bytes
    }
    pub(crate) fn plane_words(self) -> usize {
        self.columns / 2 + self.columns / 16
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrepareSpans {
    pub input_first: usize,
    pub planes_first: usize,
    pub fault_first: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectSpans {
    pub packed_first: usize,
    pub planes_first: usize,
    pub output_first: usize,
    pub fault_first: usize,
    pub matrix_first: usize,
    pub matrix_rows: usize,
}
fn span(first: usize, length: usize, extent: usize) -> bool {
    first <= u32::MAX as usize
        && length <= u32::MAX as usize
        && extent <= u32::MAX as usize
        && first.checked_add(length).is_some_and(|end| end <= extent)
}
impl PrepareSpans {
    pub(crate) fn validate(self, spec: Spec, words: [usize; 3]) -> Result<()> {
        if !span(self.input_first, spec.columns, words[0])
            || !span(self.planes_first, spec.plane_words(), words[1])
            || !span(self.fault_first, 1, words[2])
        {
            return Err(Error::InvalidBufferLength);
        }
        Ok(())
    }
}
impl ProjectSpans {
    pub(crate) fn validate(self, spec: Spec, weight_bytes: usize, words: [usize; 3]) -> Result<()> {
        if !span(self.packed_first, spec.weight_bytes(), weight_bytes)
            || !span(self.planes_first, spec.plane_words(), words[0])
            || !span(self.output_first, spec.rows, words[1])
            || !span(self.fault_first, 1, words[2])
            || !span(self.matrix_first, self.matrix_rows, words[1])
            || self.matrix_rows == 0
            || self.output_first < self.matrix_first
            || self
                .output_first
                .checked_add(spec.rows)
                .is_none_or(|end| end > self.matrix_first + self.matrix_rows)
        {
            return Err(Error::InvalidBufferLength);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Prepare(usize),
    Project(Spec, bool, bool),
}
// Array<vec4<u32>> must expose the complete byte extent, including metadata at
// the end of an original block. Do not enlarge a source view to select it.
fn wide_weight_view(weight_words: usize) -> bool {
    weight_words != 0 && weight_words.is_multiple_of(4)
}
impl Kernel {
    fn source(self) -> String {
        let (prepare, project) = include_str!("q6_integer_matvec.wgsl")
            .split_once("// PROJECT_STAGE")
            .unwrap();
        let (columns, constants, bindings, body) = match self {
            Self::Prepare(columns) => (
                columns,
                String::new(),
                "@group(0) @binding(0) var<storage,read> input:array<f32>;\n@group(0) @binding(1) var<storage,read_write> planes:array<u32>;\n@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(3) var<uniform> p:Params;",
                prepare.to_owned(),
            ),
            Self::Project(spec, tail, wide) => {
                let (body, helpers) = project.split_once("// WIDE_WEIGHT_HELPERS").unwrap();
                let body = if wide {
                    let (before, rest) = body.split_once("// WEIGHT_HELPERS_BEGIN").unwrap();
                    let (_, after) = rest.split_once("// WEIGHT_HELPERS_END").unwrap();
                    format!("{before}\n{helpers}\n{after}")
                } else {
                    body.to_owned()
                };
                (
                    spec.columns,
                    format!(
                        "const ROWS:u32={}u;const ROW_BYTES:u32={}u;const AUDIT_TAIL:bool={tail};\n",
                        spec.rows, spec.row_bytes
                    ),
                    if wide {
                        "@group(0) @binding(0) var<storage,read> weight:array<vec4<u32>>;\n@group(0) @binding(1) var<storage,read> planes:array<u32>;\n@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(4) var<uniform> p:Params;"
                    } else {
                        "@group(0) @binding(0) var<storage,read> weight:array<u32>;\n@group(0) @binding(1) var<storage,read> planes:array<u32>;\n@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(4) var<uniform> p:Params;"
                    },
                    body,
                )
            }
        };
        format!(
            "requires packed_4x8_integer_dot_product;\nconst COLUMNS:u32={columns}u;const PACKED_WORDS:u32={}u;const BLOCKS:u32={}u;\n{constants}struct Params {{a:u32,b:u32,c:u32,d:u32,e:u32,f:u32,g:u32,h:u32}}\n{bindings}\n{body}",
            columns / 4,
            columns / 32
        )
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare(_) => "Q6ResidualQ8Prepare",
            Self::Project(_, _, _) => "Q6IntegerProject",
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
fn words(handle: &Handle) -> Result<usize> {
    if !handle.size_in_used().is_multiple_of(4) || handle.size_in_used() / 4 > u32::MAX as u64 {
        return Err(Error::InvalidBufferLength);
    }
    Ok((handle.size_in_used() / 4) as usize)
}
fn stream(handles: &[&Handle]) -> Result<()> {
    if handles.iter().any(|h| h.stream != handles[0].stream) {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
fn info(words: &[usize]) -> MetadataBindingInfo {
    let mut values = [0u32; 8];
    for (dst, &src) in values.iter_mut().zip(words) {
        *dst = src as u32;
    }
    MetadataBindingInfo::custom(
        values
            .chunks_exact(2)
            .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
            .collect(),
    )
}
/// Prepare once for a frozen F32 vector, then reuse on same-stream head tiles.
/// All three handles are complete disjoint caller-owned private pages.
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
        .with_info(info(&[
            spans.input_first,
            spans.planes_first,
            spans.fault_first,
        ]));
    // SAFETY: fixed block-complete geometry, checked full array intervals,
    // caller's disjoint same-stream owner proofs; all scratch words initialized.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Prepare(spec.columns)),
            CubeCount::Static((spec.columns / 32).div_ceil(128) as u32, 1, 1),
            args,
        );
    }
    Ok(())
}
/// Fault is never cleared. Every active output is finite-checked; optional tail
/// scan excludes the whole matrix interval so other row writers cannot race it.
/// Caller must retain the prepared planes from this vector, not an older epoch.
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
    let mut groups = spec.rows.div_ceil(4);
    if audit_tail {
        groups = groups.max(output_words.div_ceil(128));
    }
    if groups > 65535 {
        return Err(Error::LimitExceeded("Q6 integer output grid"));
    }
    let args = KernelArguments::new()
        .with_buffer(weights.binding())
        .with_buffer(planes.binding())
        .with_buffer(output.binding())
        .with_buffer(fault.binding())
        .with_info(info(&[
            spans.packed_first,
            spans.planes_first,
            spans.output_first,
            spans.fault_first,
            spans.matrix_first,
            spans.matrix_rows,
        ]));
    // SAFETY: checked original byte-addressed Q6 ranges and disjoint full caller
    // owners. Subgroups16..32 cover at least four rows; inactive rows read none.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Project(
                spec,
                audit_tail,
                wide_weight_view(weight_words),
            )),
            CubeCount::Static(groups as u32, 1, 1),
            args,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "q6_integer_load_tests.rs"]
mod load_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn head_tiles_and_original_unaligned_rows_have_exact_checked_extents() {
        let spec = Spec::new(16384, 2560).unwrap();
        assert_eq!(spec.row_bytes(), 2100);
        assert_eq!(spec.weight_bytes(), 34_406_400);
        assert_eq!(spec.plane_words() * 4, 5760);
        assert_eq!(
            Spec::new(2560, 2560).unwrap().weight_bytes() + 15 * spec.weight_bytes(),
            521_472_000
        );
        for (r, c) in [
            (0, 2560),
            (16385, 2560),
            (1, 0),
            (1, 255),
            (1, 16385),
            (16384, 16384),
        ] {
            assert!(Spec::new(r, c).is_err());
        }
        let tiny = Spec::new(7, 256).unwrap();
        assert_eq!(tiny.row_bytes(), 210);
        ProjectSpans {
            packed_first: 3,
            planes_first: 5,
            output_first: 7,
            fault_first: 2,
            matrix_first: 7,
            matrix_rows: 7,
        }
        .validate(tiny, 3 + 1470, [5 + tiny.plane_words(), 14, 3])
        .unwrap();
    }
    #[test]
    fn spans_reject_missing_last_word_matrix_escape_and_overflow() {
        let spec = Spec::new(7, 2560).unwrap();
        let p = PrepareSpans {
            input_first: 3,
            planes_first: 5,
            fault_first: 2,
        };
        p.validate(spec, [2563, 1445, 3]).unwrap();
        for words in [[2562, 1445, 3], [2563, 1444, 3], [2563, 1445, 2]] {
            assert!(p.validate(spec, words).is_err());
        }
        assert!(
            PrepareSpans {
                input_first: usize::MAX,
                ..p
            }
            .validate(spec, [usize::MAX; 3])
            .is_err()
        );
        let p = ProjectSpans {
            packed_first: 3,
            planes_first: 5,
            output_first: 7,
            fault_first: 2,
            matrix_first: 7,
            matrix_rows: 7,
        };
        p.validate(spec, 14703, [1445, 14, 3]).unwrap();
        for (bytes, words) in [
            (14702, [1445, 14, 3]),
            (14703, [1444, 14, 3]),
            (14703, [1445, 13, 3]),
            (14703, [1445, 14, 2]),
        ] {
            assert!(p.validate(spec, bytes, words).is_err());
        }
        assert!(
            ProjectSpans {
                matrix_rows: 6,
                ..p
            }
            .validate(spec, 14703, [1445, 14, 3])
            .is_err()
        );
        assert!(
            ProjectSpans {
                matrix_first: 8,
                ..p
            }
            .validate(spec, 14703, [1445, 15, 3])
            .is_err()
        );
        assert!(
            ProjectSpans {
                matrix_first: usize::MAX,
                ..p
            }
            .validate(spec, 14703, [1445, usize::MAX, 3])
            .is_err()
        );
    }
    #[test]
    fn emitted_wgsl_preserves_raw_q6_offsets_integer_ops_and_runtime_info() {
        for columns in [256, 512, 2560, 16384] {
            let spec = Spec::new(7, columns).unwrap();
            for kernel in [
                Kernel::Prepare(columns),
                Kernel::Project(spec, false, false),
                Kernel::Project(spec, true, false),
                Kernel::Project(spec, false, true),
                Kernel::Project(spec, true, true),
            ] {
                let source = kernel.source();
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                assert!(source.contains("var<uniform> p:Params"));
                match kernel {
                    Kernel::Prepare(_) => {
                        assert!(source.contains("finite(reconstructed)"));
                        assert!(source.contains("atomicOr(&fault[p.c],1u)"));
                        assert!(source.contains("floor(value)"));
                        assert!(!source.contains("subgroupAdd"));
                    }
                    Kernel::Project(_, _, wide) => {
                        assert!(source.contains("dot4I8Packed"));
                        assert!(source.contains("210u"));
                        assert!(source.contains("base+192u+sub"));
                        assert!(source.contains("base+208u"));
                        assert!(source.contains("p.e+p.f"));
                        assert!(source.contains("atomicLoad(&fault[p.d])==0u"));
                        assert!(source.contains("let codes=q6_words(base,sub)"));
                        assert_eq!(source.contains("weight:array<vec4<u32>>"), wide);
                        assert_eq!(source.contains("let low=weight[at/16u]"), wide);
                    }
                }
            }
        }
    }
}
