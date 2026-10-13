//! Default-off frozen DOWN two-residual-Q8/packed-dot arithmetic leaf.
//! Original packed IQ4_NL/Q2_0 weights, rank MAP, activation/status and complete
//! owners remain caller-owned. This is alternate math, not original64 parity.
//! Quantizer/packing provenance and MIT notices: iq_moe_integer-NOTICE.txt.
#![allow(dead_code)] // Separate parent and physical acceptance are required.

use super::grouped_moe as g;
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

pub(crate) const MAX_COUNT: usize = 3;
pub(crate) const MAX_ROUTES: usize = MAX_COUNT * 10;
pub(crate) const HEADER_WORDS: usize = 64;
pub(crate) const FRAME_WORDS: usize = 360;
pub(crate) const LOGICAL_WORDS: usize = HEADER_WORDS + MAX_ROUTES * FRAME_WORDS;
pub(crate) const SCRATCH_WORDS: usize = 16384;
pub(crate) const OWNED_WORDS: usize = SCRATCH_WORDS;
pub(crate) const MAIN_BYTES: u64 = (SCRATCH_WORDS * 4) as u64;
pub(crate) const HOST_INCREMENT: usize = MAIN_BYTES as usize;
pub(crate) const MAX_PREPARE_SHARED_BYTES: u32 = 4004;
pub(crate) const MAX_PROJECT_SHARED_BYTES: u32 = 768;
pub(crate) const MAX_SHARED_BYTES: u32 = MAX_PREPARE_SHARED_BYTES;
pub(crate) const INFO_INCREMENT: u64 = 2 * 48 * 2048;
pub(crate) const RUNTIME_INCREMENT: u64 = INFO_INCREMENT + 2 * 2048 + 4 * MAIN_BYTES;

fn rejected() -> Error {
    Error::InvalidFormat("public frozen DOWN integer eight-lane candidate")
}
fn template(section: &str) -> &'static str {
    include_str!("frozen_down_integer.wgsl")
        .split_once(&format!("// FROZEN_DOWN_{section}_BEGIN\n"))
        .unwrap()
        .1
        .split_once(&format!("// FROZEN_DOWN_{section}_END"))
        .unwrap()
        .0
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Prepare,
    Project { dtype: u32, row_bytes: usize },
    ProjectCompact { dtype: u32, row_bytes: usize },
    Audit,
}

// Load original encoded words once per32 activation block. Natural weight order
// and original half factors are retained, including both halves of Q2_0's64.
fn decode(dtype: u32) -> String {
    let (prepare, packed) = match dtype {
        super::IQ4_NL => (
            "let base=row+block*18u;let payload=base+2u;let low=down_load8(payload);let high=down_load8(payload+8u);return DownEncoded(vec4<u32>(low,high),down_half(base));",
            "var codes=e.codes.x;switch quartet%4u {case 1u:{codes=e.codes.y;}case 2u:{codes=e.codes.z;}case 3u:{codes=e.codes.w;}default:{}}return iq4_word(codes,quartet>=4u);",
        ),
        super::Q2_0 => (
            "let base=row+(block/2u)*18u;let payload=base+2u+(block%2u)*8u;return DownEncoded(vec4<u32>(down_load8(payload),0u,0u),down_half(base));",
            "let codes=select(e.codes.x,e.codes.y,quartet>=4u);return q2_word((codes>>((quartet%4u)*8u))&255u);",
        ),
        _ => unreachable!("validated frozen integer DOWN format"),
    };
    format!(
        r#"
struct DownEncoded {{codes:vec4<u32>,d:f32}}
fn down_load8(at:u32)->vec2<u32> {{
    let low=down[at/4u];let middle=down[at/4u+1u];
    if at%4u==0u {{return vec2<u32>(low,middle);}}
    let high=down[at/4u+2u];
    return vec2<u32>((low>>16u)|(middle<<16u),(middle>>16u)|(high<<16u));
}}
fn down_half(at:u32)->f32 {{return unpack2x16float((down[at/4u]>>((at%4u)*8u))&65535u).x;}}
fn down_prepare(row:u32,block:u32)->DownEncoded {{{prepare}}}
fn down_factor(e:DownEncoded)->f32 {{return e.d;}}
fn down_packed(e:DownEncoded,quartet:u32)->u32 {{{packed}}}
"#
    )
}
fn members(body: &str) -> String {
    let mut source = String::new();
    for (index, axis) in ["x", "y", "z"].into_iter().enumerate() {
        if index != 0 {
            source.push_str(&format!("if length>{index}u {{\n"));
        }
        source.push_str(
            &body
                .replace("MEMBER_INDEX", &index.to_string())
                .replace("AXIS", axis),
        );
        if index != 0 {
            source.push_str("}\n");
        }
    }
    source
}
impl Kernel {
    fn source(self) -> String {
        match self {
            Self::Prepare => {
                let quantizer = include_str!("iq_moe_integer.wgsl")
                    .split_once("// Q8_PREPARE_BEGIN\n")
                    .unwrap()
                    .1
                    .split_once("// Q8_PREPARE_END")
                    .unwrap()
                    .0;
                template("PREPARE")
                    .replace("Q8_PREPARE_VALUE", quantizer)
                    .replace("BLOCKS_VALUE", "20")
            }
            Self::Audit => template("AUDIT").to_owned(),
            Self::Project { dtype, row_bytes } | Self::ProjectCompact { dtype, row_bytes } => {
                let helpers = include_str!("iq_moe_integer.wgsl")
                    .split_once("fn iq4_byte")
                    .unwrap()
                    .1
                    .split_once("// Original byte-addressed")
                    .unwrap()
                    .0;
                let group_index = if matches!(self, Self::Project { .. }) {
                    "group.y"
                } else {
                    "routed_group"
                };
                let section = if matches!(self, Self::Project { .. }) {
                    template("PROJECT")
                } else {
                    template("PROJECT_COMPACT")
                };
                section
                    .replace("ROW_BYTES_VALUE", &row_bytes.to_string())
                    .replace("DECODE_HELPERS_VALUE", &format!("fn iq4_byte{helpers}"))
                    .replace("PROJECTION_HELPERS_VALUE", &decode(dtype))
                    .replace("FRAME_MEMBERS_VALUE", &members(&format!("frames.AXIS=64u+map[p.a+164u+{group_index}*8u+MEMBER_INDEXu]*360u;\n")))
                    .replace("DOT_MEMBERS_VALUE", &members("{first.AXIS+=dot4I8Packed(q,planes[frames.AXIS+at]);second.AXIS+=dot4I8Packed(q,planes[frames.AXIS+160u+at]);}\n"))
                    .replace("SCALE_MEMBERS_VALUE", &members("{let a=bitcast<f32>(planes[frames.AXIS+320u+block]);let b=bitcast<f32>(planes[frames.AXIS+340u+block]);let pair=a*f32(first.AXIS)+b*f32(second.AXIS);values.AXIS+=d*pair;}\n"))
                    .replace("OUTPUT_MEMBERS_VALUE", &members(&format!("{{let route=map[p.a+164u+{group_index}*8u+MEMBER_INDEXu];output[p.d+route*p.e+row]=sums[local*3u+MEMBER_INDEXu];if row==0u {{status[p.f+80u+route]=1u;}}}}\n")))
            }
        }
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare => "FrozenDownIntegerPrepare",
            Self::Project { .. } => "FrozenDownIntegerEightLanes",
            Self::ProjectCompact { .. } => "FrozenDownIntegerEightLanesCompact",
            Self::Audit => "FrozenDownIntegerAudit",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((*self, crate::strata_port::enabled()))
            .cube_dim(crate::strata_port::dimension(
                CubeDim::new_1d(64),
                matches!(self, Self::Project { .. } | Self::ProjectCompact { .. }),
            ))
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
            source: crate::strata_port::shader(self.source(), true, 3),
            repr: None,
            cube_dim: crate::strata_port::dimension(
                CubeDim::new_1d(64),
                matches!(self, Self::Project { .. } | Self::ProjectCompact { .. }),
            ),
            debug_info: None,
        })
    }
}

pub(crate) fn validate_count(count: usize) -> Result<()> {
    if !(1..=MAX_COUNT).contains(&count) {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) fn validate_geometry(geometry: g::Geometry) -> Result<()> {
    let [gate, up, down] = geometry.projections();
    if !geometry.frozen_host_sources()
        || !geometry.inactive_partition_exit()
        || geometry.hidden() != 2560
        || geometry.ff() != 640
        || geometry.tile() != 8
        || geometry.capacity() != 512
        || geometry.gate_policy() != g::GatePolicy::Generic
        || geometry.live_member_reduction()
        || geometry.private_tables()
        || geometry.workgroup_tables()
        || geometry.shared_gate_up_factors()
        || geometry.shared_down_scales()
        || geometry.two_row_gate_up()
        || geometry.three_member_arithmetic()
        || gate != up
        || !matches!(
            gate.dtype(),
            super::IQ2_S | super::IQ3_XXS | super::IQ3_S | super::IQ4_XS
        )
        || !matches!(down.dtype(), super::IQ4_NL | super::Q2_0)
    {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) fn validate_limits(limits: &wgpu::Limits) -> Result<()> {
    if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_storage_size < MAX_SHARED_BYTES
        || limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_uniform_buffer_binding_size < 32
        || limits.max_compute_workgroups_per_dimension < 320
        || limits.max_storage_buffer_binding_size < MAIN_BYTES
        || limits.max_buffer_size < MAIN_BYTES
    {
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
    validate_count(count)?;
    spans
        .validate(geometry, owner_words)
        .map_err(Error::InvalidFormat)?;
    if scratch_words != SCRATCH_WORDS || !(1..=MAX_ROUTES).contains(&groups) {
        return Err(rejected());
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
fn full_handles<'a>(
    handles: impl IntoIterator<Item = &'a Handle>,
    stream: cubecl_common::stream_id::StreamId,
) -> Result<()> {
    if handles.into_iter().any(|h| {
        h.stream != stream
            || !h.size_in_used().is_multiple_of(4)
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != h.size_in_used()
    }) {
        return Err(rejected());
    }
    Ok(())
}
/// Caller authenticates all physical roots, immutable count/source identity and
/// queues ALL five original GU consumers first. Every30 route frame is reset.
pub(crate) fn prepare(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    owners: [Handle; 7],
    scratch: Handle,
    count: usize,
) -> Result<()> {
    validate_layout(
        geometry,
        spans,
        owners.each_ref().map(|h| h.size_in_used() as usize / 4),
        scratch.size_in_used() as usize / 4,
        count,
        1,
    )?;
    full_handles(owners.iter().chain([&scratch]), owners[0].stream)?;
    let args = KernelArguments::new()
        .with_buffer(owners[2].clone().binding())
        .with_buffer(owners[6].clone().binding())
        .with_buffer(scratch.binding())
        .with_info(info([
            spans.act_first(),
            spans.act_stride(),
            spans.status_first(),
            0,
            0,
            0,
            0,
            count,
        ]));
    // SAFETY: parent supplies physical owner proofs; all30 reset spans/flags and
    // the single header/tail writer are disjoint, including inactive routes.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Prepare), CubeCount::Static(1, 30, 1), args);
    }
    Ok(())
}
/// Replace one DOWN dispatch for an original frozen source partition only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    down: Handle,
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
) -> Result<()> {
    validate_layout(
        geometry,
        spans,
        owners.each_ref().map(|h| h.size_in_used() as usize / 4),
        scratch.size_in_used() as usize / 4,
        count,
        groups,
    )?;
    if !crate::daslab_profile::SOURCE_FIRST
        .into_iter()
        .zip(crate::daslab_profile::SOURCE_PARTS)
        .any(|part| part == (first_slot, slot_count))
    {
        return Err(rejected());
    }
    let projections = geometry.projections();
    let segments = projections.map(|p| p.segment_words(slot_count));
    let [a, b, c] = segments;
    geometry
        .validate_shard(
            first_slot,
            slot_count,
            [
                a.map_err(Error::InvalidFormat)?,
                b.map_err(Error::InvalidFormat)?,
                c.map_err(Error::InvalidFormat)?,
            ],
        )
        .map_err(Error::InvalidFormat)?;
    let projection = projections[2];
    if down.size_in_used()
        != projection
            .segment_words(slot_count)
            .map_err(Error::InvalidFormat)? as u64
            * 4
    {
        return Err(rejected());
    }
    full_handles(
        [&down].into_iter().chain(&owners).chain([&scratch]),
        owners[0].stream,
    )?;
    let mut args = KernelArguments::new();
    for h in [&down, &owners[0], &owners[3], &owners[6], &scratch] {
        args = args.with_buffer(h.clone().binding());
    }
    args = args.with_info(info([
        spans.map_first(),
        first_slot,
        slot_count,
        spans.partial_first(),
        spans.partial_stride(),
        spans.status_first(),
        groups,
        count,
    ]));
    // SAFETY: original shard/private spans are bounded; parent retains original
    // source/root/MAP epoch identity. Shader exits uniformly before payloads.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Project {
                dtype: projection.dtype(),
                row_bytes: projection.row_bytes(),
            }),
            CubeCount::Static(320, groups as u32, 1),
            args,
        );
    }
    Ok(())
}
/// Compact variant for one original frozen source partition. The immutable GU
/// shard record's DOWN dispatch triplet drives an indirect launch; the DOWN
/// arithmetic, member order and payload checks stay the original ones.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_compact(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    down: Handle,
    owners: [Handle; 7],
    scratch: Handle,
    gu_scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
) -> Result<()> {
    validate_layout(
        geometry,
        spans,
        owners.each_ref().map(|h| h.size_in_used() as usize / 4),
        scratch.size_in_used() as usize / 4,
        count,
        groups,
    )?;
    if !(1..=MAX_ROUTES).contains(&groups) {
        return Err(rejected());
    }
    let shard = crate::quant_matvec::frozen_gu_integer::four::shard_index(first_slot, slot_count)?;
    let projections = geometry.projections();
    let segments = projections.map(|p| p.segment_words(slot_count));
    let [a, b, c] = segments;
    geometry
        .validate_shard(
            first_slot,
            slot_count,
            [
                a.map_err(Error::InvalidFormat)?,
                b.map_err(Error::InvalidFormat)?,
                c.map_err(Error::InvalidFormat)?,
            ],
        )
        .map_err(Error::InvalidFormat)?;
    let projection = projections[2];
    if down.size_in_used()
        != projection
            .segment_words(slot_count)
            .map_err(Error::InvalidFormat)? as u64
            * 4
    {
        return Err(rejected());
    }
    // The authenticated full GU owner supplies this shard's record only.
    let dispatch = crate::quant_matvec::frozen_gu_integer::four::record_dispatch_view(
        &gu_scratch,
        shard,
        true,
    )?;
    full_handles(
        [&down]
            .into_iter()
            .chain(&owners)
            .chain([&scratch, &gu_scratch]),
        owners[0].stream,
    )?;
    let mut args = KernelArguments::new();
    for h in [
        &down,
        &owners[0],
        &owners[3],
        &owners[6],
        &scratch,
        &gu_scratch,
    ] {
        args = args.with_buffer(h.clone().binding());
    }
    args = args.with_info(info([
        spans.map_first(),
        first_slot,
        slot_count,
        spans.partial_first(),
        spans.partial_stride(),
        spans.status_first(),
        shard,
        count,
    ]));
    // SAFETY: the record bounds every list access and the shard dispatch
    // uniformly; parent retains original source/root/MAP epoch identity.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::ProjectCompact {
                dtype: projection.dtype(),
                row_bytes: projection.row_bytes(),
            }),
            CubeCount::Dynamic(dispatch),
            args,
        );
    }
    Ok(())
}
/// Complete64KiB readiness/scale/inactive/header/tail audit after all consumers.
pub(crate) fn audit(
    client: &ComputeClient<WgpuRuntime>,
    scratch: Handle,
    fault: Handle,
    count: usize,
) -> Result<()> {
    validate_count(count)?;
    if scratch.size_in_used() != MAIN_BYTES || fault.size_in_used() != 32768 {
        return Err(rejected());
    }
    full_handles([&scratch, &fault], scratch.stream)?;
    let args = KernelArguments::new()
        .with_buffer(scratch.binding())
        .with_buffer(fault.binding())
        .with_info(info([count, 0, 0, 0, 0, 0, 0, 0]));
    // SAFETY: same full, authenticated owners; the sticky fault is never reset.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Audit), CubeCount::Static(256, 1, 1), args);
    }
    Ok(())
}

#[cfg(test)]
#[path = "frozen_down_integer_tests.rs"]
mod tests;

#[path = "frozen_down_integer_four.rs"]
pub(crate) mod four;

/// Whole-port shader set for the final compiler acceptance command. No GPU.
pub(crate) fn strata_compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut result = Vec::new();
    let mut emit = |kernel: &dyn CubeTask<AutoCompiler>| {
        result.push(
            kernel
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap(),
        );
    };
    for dtype in [super::IQ4_NL, super::Q2_0] {
        let row_bytes = kuno_text::TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(640)
            .unwrap() as usize;
        emit(&Kernel::ProjectCompact { dtype, row_bytes });
        emit(&four::Kernel::ProjectCompact { dtype, row_bytes });
    }
    emit(&Kernel::Prepare);
    emit(&Kernel::Audit);
    emit(&four::Kernel::Prepare);
    emit(&four::Kernel::Audit);
    result
}

pub(crate) fn strata_fused_source(dtype: u32, row_bytes: usize, count: usize) -> String {
    let kernel = if count == 4 {
        four::Kernel::Project { dtype, row_bytes }.source()
    } else {
        Kernel::Project { dtype, row_bytes }.source()
    };
    crate::strata_port::shader(kernel, true, if count == 4 { 4 } else { 3 })
}
