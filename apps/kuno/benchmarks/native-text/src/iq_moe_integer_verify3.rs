//! Construction-fixed PUBLIC Verify3 expert candidate. Original packed banks,
//! rank mapping and completion gates are retained. This uses the separately
//! declared two-residual-Q8 activation arithmetic, NOT original-F32 parity.
//! A full independent 64KiB private owner holds3 gate and30 down frames; no
//! CPU planes/readbacks, persistent inputs or per-request generated kernels.
//! Packed/quantizer provenance is unchanged iq_moe_integer-NOTICE.txt.
use crate::quant_matvec::{self as q, grouped_moe as g, iq_moe_integer as integer};
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

pub(crate) const MAX_COUNT: usize = 3;
pub(crate) const HEADER_WORDS: usize = 64;
pub(crate) const GATE_WORDS: usize = 1440;
pub(crate) const DOWN_WORDS: usize = 360;
pub(crate) const DOWN_FIRST: usize = HEADER_WORDS + MAX_COUNT * GATE_WORDS;
pub(crate) const SCRATCH_WORDS: usize = DOWN_FIRST + MAX_COUNT * 10 * DOWN_WORDS;
pub(crate) const OWNED_WORDS: usize = 16384;
pub(crate) const MAIN_BYTES: u64 = 65536;
pub(crate) const HOST_INCREMENT: usize = 65536;
pub(crate) const MAX_SHARED_BYTES: u32 = 3 * 1440 * 4 + 2048 * 4;
// Existing four bank projections are replaced by four. Additional complete
// INFO ownership: full clear, two preparations, full new-owner audit per layer.
pub(crate) const INFO_INCREMENT: u64 = 4 * 48 * 2048;
// Separately conservative startup/close INFO and full new owner inspection
// source+target+owning-padding duplication; no interior-fence/reuse credit.
pub(crate) const RUNTIME_INCREMENT: u64 = INFO_INCREMENT + 2 * 2048 + 4 * MAIN_BYTES;

fn rejected() -> Error {
    Error::InvalidFormat("public Verify3 prepared experts")
}
fn template(section: &str) -> &'static str {
    let source = include_str!("iq_moe_integer_verify3.wgsl");
    source
        .split_once(&format!("// VERIFY3_{section}_BEGIN\n"))
        .unwrap()
        .1
        .split_once(&format!("// VERIFY3_{section}_END"))
        .unwrap()
        .0
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Prepare(bool),
    Project {
        down: bool,
        dtype: u32,
        row_bytes: usize,
        first: usize,
    },
    Audit,
}

// Expand each scalar equation component-wise. This changes neither the packed
// decoder nor integer additions/F32 rounds/subgroup reduction for any column.
fn projection3(name: &str, dtype: u32) -> String {
    let source = integer::projection(name, dtype);
    source.replace("->f32 {\n    var sum=0.0;", "->vec3<f32> {\n    var sum=vec3<f32>(0.0);")
        .replace("var first=0;var second=0;", "var first=vec3<i32>(0);var second=vec3<i32>(0);")
        .replace("first+=dot4I8Packed(q,packed_first[at]);", "first+=vec3<i32>(dot4I8Packed(q,packed_first[at]),dot4I8Packed(q,packed_first[PACKED_VALUEu+at]),dot4I8Packed(q,packed_first[2u*PACKED_VALUEu+at]));")
        .replace("second+=dot4I8Packed(q,packed_second[at]);", "second+=vec3<i32>(dot4I8Packed(q,packed_second[at]),dot4I8Packed(q,packed_second[PACKED_VALUEu+at]),dot4I8Packed(q,packed_second[2u*PACKED_VALUEu+at]));")
        .replace("let pair=scale_first[block]*f32(first)+scale_second[block]*f32(second);", "let pair=vec3<f32>(scale_first[block],scale_first[BLOCKS_VALUEu+block],scale_first[2u*BLOCKS_VALUEu+block])*vec3<f32>(first)+vec3<f32>(scale_second[block],scale_second[BLOCKS_VALUEu+block],scale_second[2u*BLOCKS_VALUEu+block])*vec3<f32>(second);")
}
impl Kernel {
    fn source(self) -> String {
        match self {
            Self::Audit => template("AUDIT").to_owned(),
            Self::Prepare(down) => {
                let columns = if down { 640 } else { 2560 };
                let bindings = if down {
                    "@group(0) @binding(0) var<storage,read> input:array<f32>;\n@group(0) @binding(1) var<storage,read> status:array<u32>;\n@group(0) @binding(2) var<storage,read_write> planes:array<u32>;\n@group(0) @binding(3) var<uniform> p:Params;"
                } else {
                    "@group(0) @binding(0) var<storage,read> input:array<f32>;\n@group(0) @binding(1) var<storage,read_write> planes:array<u32>;\n@group(0) @binding(2) var<uniform> p:Params;"
                };
                let frame = if down {
                    "if group.y>=p.h*10u {return;}\nframe=4384u+group.y*360u;flag=3u+group.y;offset=group.y*p.c;\nif p.e>arrayLength(&status) || group.y>=arrayLength(&status)-p.e {valid=false;}\nif valid && status[p.e+group.y]!=1u {valid=false;}"
                } else {
                    "if group.y>=p.h {return;}\nframe=64u+group.y*1440u;flag=group.y;offset=group.y*p.c;"
                };
                let quantizer = include_str!("iq_moe_integer.wgsl")
                    .split_once("// Q8_PREPARE_BEGIN\n")
                    .unwrap()
                    .1
                    .split_once("// Q8_PREPARE_END")
                    .unwrap()
                    .0;
                template("PREPARE")
                    .replace("PREP_BINDINGS_VALUE", bindings)
                    .replace("FRAME_SELECT_VALUE", frame)
                    .replace("Q8_PREPARE_VALUE", quantizer)
                    .replace("COLUMNS_VALUE", &columns.to_string())
                    .replace("PACKED_VALUE", &(columns / 4).to_string())
                    .replace("BLOCKS_VALUE", &(columns / 32).to_string())
            }
            Self::Project {
                down,
                dtype,
                row_bytes,
                first,
            } => {
                let columns = if down { 640 } else { 2560 };
                let rows = if down { 2560 } else { 640 };
                let (grid, initialize) = integer::grid_source(&integer::original_grid_words(dtype));
                let decode = include_str!("iq_moe_integer.wgsl")
                    .split_once("fn iq4_byte")
                    .unwrap()
                    .1
                    .split_once("// Original byte-addressed")
                    .unwrap()
                    .0;
                let (bindings, helpers, ready, frame, epilogue) = if down {
                    (
                        "@group(0) @binding(0) var<storage,read> down:array<u32>;\n@group(0) @binding(1) var<storage,read> map:array<u32>;\n@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n@group(0) @binding(3) var<storage,read_write> status:array<u32>;\n@group(0) @binding(4) var<storage,read> planes:array<u32>;\n@group(0) @binding(5) var<uniform> p:Params;",
                        projection3("down", dtype),
                        "if planes[3u+route]!=1u || status[p.f+route]!=1u {ready=false;}",
                        "let frame=4384u+route*360u;",
                        "var total=vec3<f32>(0.0);if ready && row<ROWS {total=down_project((slot*ROWS+row)*ROW_BYTES,lane,width);}\nif ready && row<ROWS && lane==0u {for(var j=0u;j<length;j++) {let route=map[p.a+164u+group.y*8u+j];output[p.d+route*p.e+row]=total[j];if row==0u {status[p.f+80u+route]=1u;}}}",
                    )
                } else {
                    (
                        "@group(0) @binding(0) var<storage,read> gate:array<u32>;\n@group(0) @binding(1) var<storage,read> up:array<u32>;\n@group(0) @binding(2) var<storage,read> map:array<u32>;\n@group(0) @binding(3) var<storage,read_write> output:array<f32>;\n@group(0) @binding(4) var<storage,read_write> status:array<u32>;\n@group(0) @binding(5) var<storage,read> planes:array<u32>;\n@group(0) @binding(6) var<uniform> p:Params;",
                        format!(
                            "{}\n{}",
                            projection3("gate", dtype),
                            projection3("up", dtype)
                        ),
                        "if planes[route/10u]!=1u {ready=false;}",
                        "let frame=64u+(route/10u)*1440u;",
                        "var a=vec3<f32>(0.0);var b=vec3<f32>(0.0);if ready && row<ROWS {let base=(slot*ROWS+row)*ROW_BYTES;a=gate_project(base,lane,width);b=up_project(base,lane,width);}\nif ready && row<ROWS && lane==0u {for(var j=0u;j<length;j++) {let route=map[p.a+164u+group.y*8u+j];var sig=0.0;if a[j]>=0.0 {sig=1.0/(1.0+exp(-a[j]));}else{let e=exp(a[j]);sig=e/(1.0+e);}\noutput[p.d+route*p.e+row]=a[j]*sig*b[j];if row==0u {status[p.f+route]=1u;}}}",
                    )
                };
                let mut source = template("PROJECT")
                    .replace("BINDINGS_VALUE", bindings)
                    .replace("PROJECTION_HELPERS_VALUE", &helpers)
                    .replace("READY_CHECK_VALUE", ready)
                    .replace("FRAME_VALUE", frame)
                    .replace("EPILOGUE_VALUE", epilogue)
                    .replace("GRID_DECLARATION_VALUE", &grid)
                    .replace("GRID_INITIALIZE_VALUE", &initialize)
                    .replace("DECODE_HELPERS_VALUE", &format!("fn iq4_byte{decode}"));
                // Replace longest placeholder names first.
                for (key, value) in [
                    ("THREE_PACKED_VALUE", 3 * columns / 4),
                    ("THREE_BLOCKS_VALUE", 3 * columns / 32),
                    ("COLUMNS_VALUE", columns),
                    ("ROWS_VALUE", rows),
                    ("ROW_BYTES_VALUE", row_bytes),
                    ("FIRST_SLOT_VALUE", first),
                    ("PACKED_VALUE", columns / 4),
                    ("BLOCKS_VALUE", columns / 32),
                ] {
                    source = source.replace(key, &value.to_string());
                }
                source
            }
        }
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "PublicVerify3PreparedIQ"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(if matches!(self, Self::Audit) {
                256
            } else {
                128
            }))
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
            cube_dim: CubeDim::new_1d(if matches!(self, Self::Audit) {
                256
            } else {
                128
            }),
            debug_info: None,
        })
    }
}

pub(crate) fn validate_limits(limits: &wgpu::Limits) -> Result<()> {
    if limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_storage_size < MAX_SHARED_BYTES
        || limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_uniform_buffer_binding_size < 32
        || limits.max_compute_workgroups_per_dimension < 640
        || limits.max_storage_buffer_binding_size < 64 * 1024 * 1024
        || limits.max_buffer_size < 64 * 1024 * 1024
    {
        return Err(Error::LimitExceeded(
            "Verify3 prepared expert device policy",
        ));
    }
    Ok(())
}
/// Parent must check actual adapter SUBGROUP support/min/max before creation.
#[allow(dead_code)]
pub(crate) fn validate_subgroups(enabled: bool, min: u32, max: u32) -> Result<()> {
    if !enabled || !matches!(min, 16 | 32) || !matches!(max, 16 | 32) || min > max {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) fn validate_geometry(geometry: g::Geometry) -> Result<()> {
    let [gate, up, down] = geometry.projections();
    if geometry.tile() != 8
        || geometry.capacity() != 144
        || geometry.hidden() != 2560
        || geometry.ff() != 640
        || gate != up
        || !matches!(gate.dtype(), q::IQ3_XXS | q::IQ3_S | q::IQ2_S | q::IQ4_XS)
        || !matches!(down.dtype(), q::IQ4_NL | q::Q2_0 | q::Q8_0)
        || gate.experts() != 512
        || down.experts() != 512
        || geometry.gate_policy() != g::GatePolicy::Generic
    {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) fn validate_count(count: usize, waves: usize, groups: usize) -> Result<()> {
    if !(1..=3).contains(&count) || waves != 1 || !(1..=count * 10).contains(&groups) {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) fn validate_layout(
    geometry: g::Geometry,
    spans: &g::Spans,
    owner_words: [usize; 7],
    scratch_words: usize,
    count: usize,
    groups: usize,
) -> Result<()> {
    validate_geometry(geometry)?;
    validate_count(count, 1, groups)?;
    spans
        .validate(geometry, owner_words)
        .map_err(Error::InvalidFormat)?;
    if scratch_words != OWNED_WORDS
        || [
            spans.map_first(),
            spans.input_first(),
            spans.input_stride(),
            spans.act_first(),
            spans.act_stride(),
            spans.partial_first(),
            spans.partial_stride(),
            spans.status_first(),
        ]
        .into_iter()
        .any(|v| v > u32::MAX as usize)
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
fn info(values: [usize; 8]) -> MetadataBindingInfo {
    MetadataBindingInfo::custom(
        values
            .chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    )
}
struct Phase<'a> {
    client: &'a ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &'a g::Spans,
    banks: [[Handle; 3]; 2],
    owners: [Handle; 7],
    scratch: Handle,
    fault: Handle,
    count: usize,
    groups: usize,
}
impl Phase<'_> {
    fn prepare(&self, down: bool) {
        let s = self.spans;
        let parameters = info([
            0,
            if down { s.act_first() } else { s.input_first() },
            if down {
                s.act_stride()
            } else {
                s.input_stride()
            },
            0,
            s.status_first(),
            0,
            0,
            self.count,
        ]);
        let handles: &[&Handle] = if down {
            &[&self.owners[2], &self.owners[6], &self.scratch]
        } else {
            &[&self.owners[1], &self.scratch]
        };
        let mut args = KernelArguments::new();
        for h in handles {
            args = args.with_buffer((*h).clone().binding());
        }
        args = args.with_info(parameters);
        // SAFETY: complete owners/shards and all runtime spans proved before enqueue.
        unsafe {
            self.client.launch_unchecked(
                Box::new(Kernel::Prepare(down)),
                CubeCount::Static(
                    1,
                    if down {
                        (self.count * 10) as u32
                    } else {
                        self.count as u32
                    },
                    1,
                ),
                args,
            );
        }
    }
    fn project(&self, down: bool) {
        let s = self.spans;
        let p = self.geometry.projections()[if down { 2 } else { 0 }];
        for segment in 0..2 {
            let bank = &self.banks[segment];
            let handles: &[&Handle] = if down {
                &[
                    &bank[2],
                    &self.owners[0],
                    &self.owners[3],
                    &self.owners[6],
                    &self.scratch,
                ]
            } else {
                &[
                    &bank[0],
                    &bank[1],
                    &self.owners[0],
                    &self.owners[2],
                    &self.owners[6],
                    &self.scratch,
                ]
            };
            let mut args = KernelArguments::new();
            for h in handles {
                args = args.with_buffer((*h).clone().binding());
            }
            args = args.with_info(info([
                s.map_first(),
                0,
                0,
                if down {
                    s.partial_first()
                } else {
                    s.act_first()
                },
                if down {
                    s.partial_stride()
                } else {
                    s.act_stride()
                },
                s.status_first(),
                0,
                self.count,
            ]));
            // SAFETY: original pending single-wave rank map and physical disjointness
            // are caller-owned; both72-slot bank consumers precede down preparation.
            unsafe {
                self.client.launch_unchecked(
                    Box::new(Kernel::Project {
                        down,
                        dtype: p.dtype(),
                        row_bytes: p.row_bytes(),
                        first: segment * 72,
                    }),
                    CubeCount::Static((if down { 2560 } else { 640 }) / 4, self.groups as u32, 1),
                    args,
                );
            }
        }
    }
    fn audit(self) {
        let args = KernelArguments::new()
            .with_buffer(self.scratch.binding())
            .with_buffer(self.fault.binding())
            .with_info(MetadataBindingInfo::custom(vec![self.count as u64, 0]));
        // SAFETY: scans exactly the full16,384-word owner; only ORs sticky fault.
        unsafe {
            self.client.launch_unchecked(
                Box::new(Kernel::Audit),
                CubeCount::Static(64, 1, 1),
                args,
            );
        }
    }
}
/// Parent supplies the full new owner already cleared on the same stream,
/// original validated single-wave map and physically disjoint complete roots.
/// No map/fault/status reset, allocations or acceptance publication here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_pair(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [[Handle; 3]; 2],
    owners: [Handle; 7],
    scratch: Handle,
    fault: Handle,
    count: usize,
    groups: usize,
) -> Result<()> {
    validate_layout(
        geometry,
        spans,
        owners.each_ref().map(|h| h.size_in_used() as usize / 4),
        scratch.size_in_used() as usize / 4,
        count,
        groups,
    )?;
    for (segment, bank) in banks.iter().enumerate() {
        let mut words = [0; 3];
        for (i, (p, h)) in geometry.projections().iter().zip(bank).enumerate() {
            words[i] = p.segment_words(72).map_err(Error::InvalidFormat)?;
            if h.size_in_used() != (words[i] * 4) as u64 {
                return Err(rejected());
            }
        }
        geometry
            .validate_shard(segment * 72, 72, words)
            .map_err(Error::InvalidFormat)?;
    }
    if fault.size_in_used() != 32768
        || banks
            .iter()
            .flatten()
            .chain(&owners)
            .chain([&scratch, &fault])
            .any(|h| {
                h.stream != owners[0].stream
                    || !h.size_in_used().is_multiple_of(4)
                    || h.offset_start.unwrap_or(0) != 0
                    || h.offset_end.unwrap_or(0) != 0
                    || h.size() != h.size_in_used()
            })
    {
        return Err(rejected());
    }
    let phase = Phase {
        client,
        geometry,
        spans,
        banks,
        owners,
        scratch,
        fault,
        count,
        groups,
    };
    phase.prepare(false);
    phase.project(false);
    phase.prepare(true);
    phase.project(true);
    phase.audit();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::private_arena::BufferLayout;
    use crate::quant_matvec::iq_fixtures;
    use integer::oracle;
    use kuno_text::TensorType;
    use zeroize::Zeroizing;

    fn compile(kernel: Kernel) -> wgpu::naga::Module {
        let source = kernel.source();
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::SUBGROUP
                | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)
        .unwrap();
        let mut bindings = Vec::new();
        for (handle, global) in module.global_variables.iter() {
            if let Some(binding) = &global.binding {
                assert_eq!(binding.group, 0);
                assert!(
                    !info.get_entry_point(0)[handle].is_empty(),
                    "inactive reflected binding {}",
                    binding.binding
                );
                bindings.push(binding.binding);
            }
        }
        bindings.sort_unstable();
        assert_eq!(bindings, (0..bindings.len() as u32).collect::<Vec<_>>());
        let words = wgpu::naga::back::spv::write_vec(
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
        assert!(!words.is_empty());
        for (_, function) in module.functions.iter() {
            for (_, variable) in function.local_variables.iter() {
                assert!(
                    !matches!(
                        module.types[variable.ty].inner,
                        wgpu::naga::TypeInner::Array { .. }
                    ),
                    "no Function array spills"
                );
            }
        }
        for entry in &module.entry_points {
            for (_, variable) in entry.function.local_variables.iter() {
                assert!(!matches!(
                    module.types[variable.ty].inner,
                    wgpu::naga::TypeInner::Array { .. }
                ));
            }
        }
        assert!(!source.contains("fma("));
        module
    }
    fn geometry(gate: u32, down: u32) -> g::Geometry {
        g::Geometry::new(
            [
                g::Projection::new(gate, 2560, 640, 512).unwrap(),
                g::Projection::new(gate, 2560, 640, 512).unwrap(),
                g::Projection::new(down, 640, 2560, 512).unwrap(),
            ],
            8,
            144,
        )
        .unwrap()
    }
    #[test]
    fn all_seven_packed_roles_and_two_segments_compile_with_exact_three_column_shared_bounds() {
        for (down, dtype) in [
            (false, q::IQ3_XXS),
            (false, q::IQ3_S),
            (false, q::IQ2_S),
            (false, q::IQ4_XS),
            (true, q::IQ4_NL),
            (true, q::Q2_0),
            (true, q::Q8_0),
        ] {
            let columns = if down { 640 } else { 2560 };
            let rows = if down { 2560 } else { 640 };
            let projection = g::Projection::new(dtype, columns, rows, 512).unwrap();
            for first in [0, 72] {
                let kernel = Kernel::Project {
                    down,
                    dtype,
                    row_bytes: projection.row_bytes(),
                    first,
                };
                let source = kernel.source();
                let module = compile(kernel);
                assert!(source.contains("->vec3<f32>") && source.contains("subgroupAdd(sum)"));
                assert!(!source.contains("prepare_q8") && !source.contains("cached_input"));
                assert_eq!(
                    source.matches("@binding(").count(),
                    if down { 6 } else { 7 }
                );
                assert!(source.contains("2u*"));
                if dtype == q::IQ2_S {
                    assert!(source.contains("half<2u") && source.contains("half*4u+4u"));
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
                        _ => panic!("only bounded prepared caches"),
                    })
                    .sum::<u32>();
                assert_eq!(
                    shared,
                    3 * (2 * (columns / 4) + 2 * (columns / 32)) as u32 * 4
                        + integer::original_grid_words(dtype).len() as u32 * 4
                );
                assert!(shared <= MAX_SHARED_BYTES);
            }
        }
    }
    #[test]
    fn preparation_keeps_verbatim_accepted_quantizer_and_never_resets_header_on_failure() {
        let accepted = include_str!("iq_moe_integer.wgsl")
            .split_once("// Q8_PREPARE_BEGIN\n")
            .unwrap()
            .1
            .split_once("// Q8_PREPARE_END")
            .unwrap()
            .0;
        for down in [false, true] {
            let kernel = Kernel::Prepare(down);
            let source = kernel.source();
            compile(kernel);
            assert!(
                source.contains(&accepted.replace("BLOCKS_VALUE", if down { "20" } else { "80" }))
            );
            assert_eq!(source.matches("prepare_q8(local)").count(), 1);
            assert!(
                !source.contains("planes[flag]=0u") && !source.contains("planes[p.d+local]=0u")
            );
            assert!(source.contains("valid && atomicLoad(&quant_fault)==0u"));
            assert_eq!(
                source.matches("@binding(").count(),
                if down { 4 } else { 3 }
            );
        }
        compile(Kernel::Audit);
    }
    #[test]
    fn complete_plane_and_runtime_reservations_are_separate_and_exact() {
        assert_eq!(SCRATCH_WORDS, 15184);
        assert_eq!(SCRATCH_WORDS * 4, 60736);
        assert_eq!(
            BufferLayout::new(SCRATCH_WORDS).unwrap().owned_bytes as u64,
            MAIN_BYTES
        );
        let mut covered = vec![0u8; OWNED_WORDS];
        for (start, length) in (0..3)
            .map(|column| (HEADER_WORDS + column * GATE_WORDS, GATE_WORDS))
            .chain((0..30).map(|route| (DOWN_FIRST + route * DOWN_WORDS, DOWN_WORDS)))
        {
            for cell in &mut covered[start..start + length] {
                *cell += 1;
            }
        }
        assert!(covered[HEADER_WORDS..SCRATCH_WORDS].iter().all(|&n| n == 1));
        assert!(
            covered[..HEADER_WORDS]
                .iter()
                .chain(&covered[SCRATCH_WORDS..])
                .all(|&n| n == 0)
        );
        assert_eq!(INFO_INCREMENT, 393216);
        assert_eq!(RUNTIME_INCREMENT, 659456);
        assert_eq!(MAX_SHARED_BYTES, 25472);
    }
    #[test]
    fn admission_rejects_count_waves_source_and_incomplete_owner_before_any_enqueue() {
        let g = geometry(q::IQ2_S, q::Q2_0);
        let spans = g::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
        let words = [
            g::MAP_WORDS,
            8 * 2560,
            80 * 640,
            80 * 2560,
            80,
            8 * 2560,
            g.status_words(),
        ];
        for count in 1..=3 {
            validate_layout(g, &spans, words, OWNED_WORDS, count, count * 10).unwrap();
        }
        for (count, waves, groups) in [
            (0, 1, 1),
            (4, 1, 30),
            (usize::MAX, 1, 1),
            (1, 0, 10),
            (1, 2, 10),
            (3, 1, 31),
            (3, 1, 0),
        ] {
            assert!(validate_count(count, waves, groups).is_err());
        }
        assert!(validate_layout(g, &spans, words, OWNED_WORDS - 1, 3, 30).is_err());
        assert!(validate_layout(g, &spans, words, OWNED_WORDS + 1, 3, 30).is_err());
        for i in 0..7 {
            let mut short = words;
            short[i] = 0;
            assert!(validate_layout(g, &spans, short, OWNED_WORDS, 3, 30).is_err());
        }
        let old = g::Geometry::new(g.projections(), 1, 144).unwrap();
        assert!(validate_geometry(old).is_err());
        let old = geometry(q::Q4_K, q::Q8_0);
        assert!(validate_geometry(old).is_err());
    }
    #[test]
    fn group_members_select_actual_input_and_original_rank_frames_once() {
        for count in 1..=3 {
            let experts = (0..count).flat_map(|_| 0u32..10).collect::<Vec<_>>();
            let probabilities = vec![0.1; count * 10];
            let slots = (0..10).map(|id| (id, id + 71)).collect::<Vec<_>>();
            let map =
                g::GroupMap::new(count, 0, count, &experts, &probabilities, &slots, 144).unwrap();
            let words = map.words();
            assert_eq!(map.groups(), 10);
            let mut seen = [0u8; 30];
            for group in 0..map.groups() {
                let length = words[84 + group] as usize;
                assert_eq!(length, count);
                let mut columns = [false; 3];
                for j in 0..length {
                    let rank = words[164 + group * 8 + j] as usize;
                    assert_eq!(experts[rank], group as u32);
                    assert!(!columns[rank / 10]);
                    columns[rank / 10] = true;
                    seen[rank] += 1;
                    let gate = HEADER_WORDS + (rank / 10) * GATE_WORDS;
                    let down = DOWN_FIRST + rank * DOWN_WORDS;
                    assert!(gate + GATE_WORDS <= DOWN_FIRST && down + DOWN_WORDS <= SCRATCH_WORDS);
                }
            }
            assert!(seen[..count * 10].iter().all(|&n| n == 1));
            assert!(seen[count * 10..].iter().all(|&n| n == 0));
        }
    }
    fn audit_host(words: &[u32], count: usize) -> bool {
        if words.len() != OWNED_WORDS || !(1..=3).contains(&count) {
            return false;
        }
        words.iter().enumerate().all(|(at, &word)| {
            if at < HEADER_WORDS {
                word == u32::from(at < count || (3..3 + count * 10).contains(&at))
            } else if at < DOWN_FIRST {
                let index = at - HEADER_WORDS;
                if index / GATE_WORDS >= count {
                    word == 0
                } else if index % GATE_WORDS >= 1280 {
                    let x = f32::from_bits(word);
                    x.is_finite() && x >= 0.
                } else {
                    true
                }
            } else if at < SCRATCH_WORDS {
                let index = at - DOWN_FIRST;
                if index / DOWN_WORDS >= count * 10 {
                    word == 0
                } else if index % DOWN_WORDS >= 320 {
                    let x = f32::from_bits(word);
                    x.is_finite() && x >= 0.
                } else {
                    true
                }
            } else {
                word == 0
            }
        })
    }
    #[test]
    fn independent_plane_fixture_audits_every_active_flag_inactive_frame_scale_and_tail() {
        for count in 1..=3 {
            let mut words = Zeroizing::new(vec![0u32; OWNED_WORDS]);
            for column in 0..count {
                let input = (0..2560)
                    .map(|i| ((i * 17 + column * 3) % 257) as f32 / 123. - 1.)
                    .collect::<Vec<_>>();
                let packed = oracle::Planes::new(&input).unwrap().packed_words().unwrap();
                words[HEADER_WORDS + column * GATE_WORDS..HEADER_WORDS + (column + 1) * GATE_WORDS]
                    .copy_from_slice(&packed);
                words[column] = 1;
            }
            for rank in 0..count * 10 {
                let input = (0..640)
                    .map(|i| ((i * 13 + rank * 7) % 211) as f32 / 117. - 1.)
                    .collect::<Vec<_>>();
                let packed = oracle::Planes::new(&input).unwrap().packed_words().unwrap();
                words[DOWN_FIRST + rank * DOWN_WORDS..DOWN_FIRST + (rank + 1) * DOWN_WORDS]
                    .copy_from_slice(&packed);
                words[3 + rank] = 1;
            }
            assert!(audit_host(&words, count));
            for at in [
                0,
                3,
                33,
                HEADER_WORDS + 1280,
                DOWN_FIRST + 320,
                SCRATCH_WORDS,
                OWNED_WORDS - 1,
            ] {
                let old = words[at];
                words[at] = if old == 0 { 1 } else { u32::MAX };
                assert!(!audit_host(&words, count));
                words[at] = old;
            }
            if count < 3 {
                for at in [
                    HEADER_WORDS + count * GATE_WORDS,
                    DOWN_FIRST + count * 10 * DOWN_WORDS,
                ] {
                    words[at] = 1;
                    assert!(!audit_host(&words, count));
                    words[at] = 0;
                }
            }
        }
    }
    // Three components independently reproduce the accepted integer operator
    // equation from public C-backed weights and packed activation oracle words.
    fn packed_dot3(
        dtype: TensorType,
        wire: &[u8],
        packed: [&[u32]; 3],
        columns: usize,
        width: usize,
    ) -> [f32; 3] {
        let mut lanes = vec![[0f32; 3]; width];
        let blocks = columns / 32;
        let planes = columns / 4;
        for (lane, sums) in lanes.iter_mut().enumerate() {
            for block in (lane..blocks).step_by(width) {
                for half in 0..if dtype == TensorType::Iq2S { 2 } else { 1 } {
                    let (codes, factor) = if dtype == TensorType::Iq2S {
                        let (codes, f) =
                            oracle::unpack_iq2_s16(wire, columns, block * 2 + half).unwrap();
                        (codes.to_vec(), f)
                    } else {
                        let (codes, f) = oracle::unpack32(dtype, wire, columns, block).unwrap();
                        (codes.to_vec(), f)
                    };
                    let mut first = [0i32; 3];
                    let mut second = [0i32; 3];
                    for (offset, &code) in codes.iter().enumerate() {
                        let column = block * 32 + half * 16 + offset;
                        let word = column / 4;
                        let shift = (column % 4) * 8;
                        for component in 0..3 {
                            first[component] += i32::from(code)
                                * i32::from((packed[component][word] >> shift) as u8 as i8);
                            second[component] += i32::from(code)
                                * i32::from(
                                    (packed[component][planes + word] >> shift) as u8 as i8,
                                );
                        }
                    }
                    for component in 0..3 {
                        let a = f32::from_bits(packed[component][2 * planes + block]);
                        let b = f32::from_bits(packed[component][2 * planes + blocks + block]);
                        let pair = a * first[component] as f32 + b * second[component] as f32;
                        sums[component] += factor * pair;
                    }
                }
            }
        }
        let mut stride = width / 2;
        while stride > 0 {
            for lane in 0..stride {
                let upper = lanes[lane + stride];
                for (sum, right) in lanes[lane].iter_mut().zip(upper) {
                    *sum += right;
                }
            }
            stride /= 2;
        }
        lanes[0]
    }
    #[test]
    fn three_column_integer_equations_match_independent_scalar_oracle_all_formats_and_widths() {
        for dtype in [
            TensorType::Iq3Xxs,
            TensorType::Iq3S,
            TensorType::Iq2S,
            TensorType::Iq4Xs,
            TensorType::Iq4Nl,
            TensorType::Q2_0,
            TensorType::Q8_0,
        ] {
            let columns = if matches!(
                dtype,
                TensorType::Iq4Nl | TensorType::Q2_0 | TensorType::Q8_0
            ) {
                640
            } else {
                2560
            };
            let wire = if dtype == TensorType::Q8_0 {
                // Fixed public signed-byte Q8 control (other six formats use
                // independent SHA-pinned C fixtures, without Rust decoding).
                (0..columns / 32)
                    .flat_map(|block| {
                        let mut bytes = 0x3400u16.to_le_bytes().to_vec();
                        bytes.extend((0..32).map(|lane| ((block * 17 + lane * 11) % 255) as u8));
                        bytes
                    })
                    .collect::<Vec<_>>()
            } else {
                iq_fixtures::packed(dtype.gguf_id(), 1, columns, 3)
            };
            let inputs: (Vec<f32>, Vec<f32>, Vec<f32>) = (
                (0..columns)
                    .map(|i| ((i * 17) % 311) as f32 / 193. - 0.7)
                    .collect(),
                (0..columns)
                    .map(|i| ((i * 31) % 113) as f32 / 61. - 0.9)
                    .collect(),
                (0..columns)
                    .map(|i| ((i * 11) % 283) as f32 / 151. - 0.8)
                    .collect(),
            );
            let planes = [
                oracle::Planes::new(&inputs.0).unwrap(),
                oracle::Planes::new(&inputs.1).unwrap(),
                oracle::Planes::new(&inputs.2).unwrap(),
            ];
            let packed = planes.each_ref().map(|p| p.packed_words().unwrap());
            for width in [16, 32] {
                let actual = packed_dot3(
                    dtype,
                    &wire,
                    packed.each_ref().map(|p| p.as_slice()),
                    columns,
                    width,
                );
                for component in 0..3 {
                    let expected = oracle::project(dtype, &wire, &planes[component]).unwrap();
                    assert!(
                        (actual[component] - expected).abs() <= 1e-4 + 3e-4 * expected.abs(),
                        "dtype={} component={component} width={width}",
                        dtype.gguf_id()
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    #[test]
    fn verify3_device_limits_and_actual_subgroup_range_are_checked_independently() {
        let limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_storage_size: MAX_SHARED_BYTES,
            max_storage_buffers_per_shader_stage: 6,
            max_bindings_per_bind_group: 7,
            max_uniform_buffer_binding_size: 32,
            max_compute_workgroups_per_dimension: 640,
            max_storage_buffer_binding_size: 64 * 1024 * 1024,
            max_buffer_size: 64 * 1024 * 1024,
            ..Default::default()
        };
        validate_limits(&limits).unwrap();
        for (min, max) in [(16, 16), (16, 32), (32, 32)] {
            validate_subgroups(true, min, max).unwrap();
        }
        for (enabled, min, max) in [
            (false, 16, 32),
            (true, 8, 32),
            (true, 16, 64),
            (true, 32, 16),
        ] {
            assert!(validate_subgroups(enabled, min, max).is_err());
        }
        for field in 0..9 {
            let mut short = limits.clone();
            match field {
                0 => short.max_compute_invocations_per_workgroup -= 1,
                1 => short.max_compute_workgroup_size_x -= 1,
                2 => short.max_compute_workgroup_storage_size -= 1,
                3 => short.max_storage_buffers_per_shader_stage -= 1,
                4 => short.max_bindings_per_bind_group -= 1,
                5 => short.max_uniform_buffer_binding_size -= 1,
                6 => short.max_compute_workgroups_per_dimension -= 1,
                7 => short.max_storage_buffer_binding_size -= 1,
                _ => short.max_buffer_size -= 1,
            }
            assert!(validate_limits(&short).is_err());
        }
    }
}
