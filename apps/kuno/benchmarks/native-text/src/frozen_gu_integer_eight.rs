// Separate exactcount4 leaf. All original count1..3 entries remain unchanged.
//! Frozen GU two-residual-Q8 math plus a fixed-public primary-plane probe.
//! Original public banks, private rank MAP, DOWN and all owning spans remain
//! caller-owned. This is alternate arithmetic, never original64 F32 parity.
//! Packing/quantizer provenance and MIT notices: iq_moe_integer-NOTICE.txt.
#![allow(dead_code)] // Parent wiring is a separate integration commit.

use super::{decode, g, grid_source};

#[path = "frozen_compact_shuffle.rs"]
pub(crate) mod shuffle;
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

// Fixed-public laboratory build selection. Not a Kuno inference/API default.
// Rejected by unchanged full CPU13 in v195; retain only as an unused probe.
pub(crate) const FIRST_PLANE_PUBLIC_PROBE: bool = false;
pub(crate) const MAX_COUNT: usize = 8;
pub(crate) const MAX_ROUTES: usize = MAX_COUNT * 10;
pub(crate) const GROUP_STRIDE_FOUR: u32 = 4;
pub(crate) const HEADER_WORDS: usize = 64;
pub(crate) const FRAME_WORDS: usize = 1440;
pub(crate) const LOGICAL_WORDS: usize = HEADER_WORDS + MAX_COUNT * FRAME_WORDS;
// Allocate this exact logical extent, not LOGICAL_WORDS (which rounds to32KiB).
pub(crate) const SCRATCH_WORDS: usize = 16384;
pub(crate) const MAIN_BYTES: u64 = (SCRATCH_WORDS * 4) as u64;
pub(crate) const HOST_INCREMENT: usize = MAIN_BYTES as usize;
pub(crate) const MAX_PROJECT_SHARED_BYTES: u32 = 8192 + 2048;
pub(crate) const MAX_PREPARE_SHARED_BYTES: u32 = 16004;
pub(crate) const MAX_SHARED_BYTES: u32 = MAX_PREPARE_SHARED_BYTES;
// Prepare and audit are additional to the original five GU/five DOWN calls.
pub(crate) const INFO_INCREMENT: u64 = 2 * 48 * 2048;
pub(crate) const RUNTIME_INCREMENT: u64 = INFO_INCREMENT + 2 * 2048 + 4 * MAIN_BYTES;
// Original frozen source partitions; the compact shard records follow them.
pub(crate) const FIRST: [usize; 5] = crate::daslab_profile::SOURCE_FIRST;
pub(crate) const PARTS: [usize; 5] = crate::daslab_profile::SOURCE_PARTS;
pub(crate) const RECORD_BASE_WORD: usize = 12288;
pub(crate) const RECORD_WORDS: usize = 88;
pub(crate) const RECORD_GROUP_IDS: usize = 80;
pub(crate) const RECORD_DISPATCH_BYTES: u64 = 12;
pub(crate) const RECORD_BYTES: u64 = (RECORD_WORDS * 4) as u64;

fn rejected() -> Error {
    Error::InvalidFormat("public frozen GU integer eight-lane candidate")
}
fn template(section: &str) -> String {
    let source = include_str!("frozen_gu_integer_eight.wgsl")
        .split_once(&format!("// FROZEN_GU_{section}_BEGIN\n"))
        .unwrap()
        .1
        .split_once(&format!("// FROZEN_GU_{section}_END"))
        .unwrap()
        .0;
    crate::daslab_profile::source_geometry_shader(source)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Kernel {
    Prepare,
    Project { dtype: u32, row_bytes: usize },
    ProjectStride4 { dtype: u32, row_bytes: usize },
    Audit,
    PrepareCompact,
    ProjectCompact { dtype: u32, row_bytes: usize },
    AuditCompact,
    ProjectCompactFirstPlane { dtype: u32, row_bytes: usize },
    ProjectCompactShuffle { dtype: u32, row_bytes: usize },
}

fn members(body: &str) -> String {
    let mut source = String::new();
    for index in 0..8 {
        if index != 0 {
            source.push_str(&format!("if length>{index}u {{\n"));
        }
        source.push_str(
            &body
                .replace("MEMBER_INDEX", &index.to_string())
                .replace("AXIS", &index.to_string()),
        );
        if index != 0 {
            source.push_str("}\n");
        }
    }
    source
}

impl Kernel {
    pub(super) fn source(self) -> String {
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
                    .replace("BLOCKS_VALUE", "80")
            }
            Self::Audit => template("AUDIT").to_owned(),
            Self::PrepareCompact => {
                let quantizer = include_str!("iq_moe_integer.wgsl")
                    .split_once("// Q8_PREPARE_BEGIN\n")
                    .unwrap()
                    .1
                    .split_once("// Q8_PREPARE_END")
                    .unwrap()
                    .0;
                template("PREPARE_COMPACT")
                    .replace("Q8_PREPARE_VALUE", quantizer)
                    .replace("BLOCKS_VALUE", "80")
            }
            Self::AuditCompact => template("COMPACT_AUDIT").to_owned(),
            Self::ProjectCompactShuffle { dtype, row_bytes } => {
                shuffle::gate_up(Self::ProjectCompact { dtype, row_bytes }.source())
            }
            Self::Project { dtype, row_bytes }
            | Self::ProjectStride4 { dtype, row_bytes }
            | Self::ProjectCompact { dtype, row_bytes }
            | Self::ProjectCompactFirstPlane { dtype, row_bytes } => {
                let stride_four = !matches!(self, Self::Project { .. });
                let first_plane = matches!(self, Self::ProjectCompactFirstPlane { .. });
                // Schedules share all decoding, guards and reduction. Only the
                // explicit first-plane candidate omits residual dot/scale terms.
                let project = template("PROJECT");
                let source = if matches!(
                    self,
                    Self::ProjectCompact { .. } | Self::ProjectCompactFirstPlane { .. }
                ) {
                    let prefix = project
                        .split_once("@compute @workgroup_size(64)")
                        .unwrap()
                        .0;
                    format!(
                        "{prefix}{}",
                        template("PROJECT_COMPACT_MAIN")
                            .replace("PROJECT_BODY_VALUE", &template("PROJECT_BODY"))
                    )
                    // p.g is this launch's shard index in the compact variant.
                    .replace("p.g!=0u", "p.g>4u")
                } else if stride_four {
                    let prefix = project
                        .split_once("@compute @workgroup_size(64)")
                        .unwrap()
                        .0;
                    format!(
                        "{prefix}{}",
                        template("PROJECT_STRIDE4_MAIN")
                            .replace("PROJECT_BODY_VALUE", &template("PROJECT_BODY"))
                    )
                    .replace("p.g!=0u", "p.g<1u || p.g>80u")
                    .replace("group>=groups", "group>=groups || group>=p.g")
                } else {
                    project.to_owned()
                };
                let group_index = if matches!(self, Self::Project { .. }) {
                    "group.y"
                } else {
                    "routed_group"
                };
                let (grid, initialize) = grid_source(dtype);
                let helpers = include_str!("iq_moe_integer.wgsl")
                    .split_once("fn iq4_byte")
                    .unwrap()
                    .1
                    .split_once("// Original byte-addressed")
                    .unwrap()
                    .0;
                let halves = if dtype == crate::quant_matvec::IQ2_S {
                    2
                } else {
                    1
                };
                let quartets = if halves == 2 { 4 } else { 8 };
                let dots = if first_plane {
                    "{let first=planes[frames[AXIS]+at];gf[AXIS]+=dot4I8Packed(gq,first);uf[AXIS]+=dot4I8Packed(uq,first);}\n"
                } else {
                    "{let first=planes[frames[AXIS]+at];let second=planes[frames[AXIS]+640u+at];gf[AXIS]+=dot4I8Packed(gq,first);gs[AXIS]+=dot4I8Packed(gq,second);uf[AXIS]+=dot4I8Packed(uq,first);us[AXIS]+=dot4I8Packed(uq,second);}\n"
                };
                let scales = if first_plane {
                    "{let a=bitcast<f32>(planes[frames[AXIS]+1280u+block]);let gp=a*f32(gf[AXIS]);let up=a*f32(uf[AXIS]);gates[AXIS]+=gd*gp;ups[AXIS]+=ud*up;}\n"
                } else {
                    "{let a=bitcast<f32>(planes[frames[AXIS]+1280u+block]);let b=bitcast<f32>(planes[frames[AXIS]+1360u+block]);let gp=a*f32(gf[AXIS])+b*f32(gs[AXIS]);let up=a*f32(uf[AXIS])+b*f32(us[AXIS]);gates[AXIS]+=gd*gp;ups[AXIS]+=ud*up;}\n"
                };
                source
                    .replace("GRID_DECLARATION_VALUE", &grid)
                    .replace("GRID_INITIALIZE_VALUE", &initialize)
                    .replace("DECODE_HELPERS_VALUE", &format!("fn iq4_byte{helpers}"))
                    .replace("PROJECTION_HELPERS_VALUE", &format!("{}\n{}", decode("gate", dtype), decode("up", dtype)))
                    .replace("ROW_BYTES_VALUE", &row_bytes.to_string())
                    .replace("HALVES_VALUE", &halves.to_string())
                    .replace("QUARTETS_VALUE", &quartets.to_string())
                    .replace("FRAME_MEMBERS_VALUE", &members(&format!("frames[AXIS]=64u+(map[p.a+164u+{group_index}*8u+MEMBER_INDEXu]/10u)*1440u;\n")))
                    .replace("DOT_MEMBERS_VALUE", &members(dots))
                    .replace("SCALE_MEMBERS_VALUE", &members(scales))
                    .replace("OUTPUT_MEMBERS_VALUE", &members(&format!("{{let route=map[p.a+164u+{group_index}*8u+MEMBER_INDEXu];output[p.d+route*p.e+row]=silu(gate_sums[local*8u+MEMBER_INDEXu])*up_sums[local*8u+MEMBER_INDEXu];if row==0u {{status[p.f+route]=1u;}}}}\n")))
            }
        }
    }
    fn dim(self) -> CubeDim {
        let port_project = matches!(
            self,
            Self::Project { .. } | Self::ProjectStride4 { .. } | Self::ProjectCompact { .. }
        );
        crate::strata_port::dimension(
            CubeDim::new_1d(
                if matches!(
                    self,
                    Self::Project { .. }
                        | Self::ProjectStride4 { .. }
                        | Self::ProjectCompact { .. }
                        | Self::ProjectCompactFirstPlane { .. }
                        | Self::ProjectCompactShuffle { .. }
                ) {
                    64
                } else {
                    128
                },
            ),
            port_project,
        )
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare => "FrozenGuIntegerEightPrepare",
            Self::Project { .. } => "FrozenGuIntegerEightEightLanes",
            Self::ProjectStride4 { .. } => "FrozenGuIntegerEightEightLanesStride4",
            Self::Audit => "FrozenGuIntegerEightAudit",
            Self::PrepareCompact => "FrozenGuIntegerEightPrepareCompact",
            Self::ProjectCompact { .. } => "FrozenGuIntegerEightEightLanesCompact",
            Self::ProjectCompactFirstPlane { .. } => {
                "FrozenGuIntegerEightEightLanesCompactFirstPlane"
            }
            Self::AuditCompact => "FrozenGuIntegerEightAuditCompact",
            Self::ProjectCompactShuffle { .. } => "FrozenGuIntegerEightCompactShuffle",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((
                *self,
                crate::strata_port::enabled(),
                crate::strata_port::wide_experts(),
                crate::strata_port::expert_lanes(),
            ))
            .cube_dim(self.dim())
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
            source: crate::strata_port::wide_weights(
                crate::strata_port::shader(self.source(), false, 8),
                false,
            ),
            repr: None,
            cube_dim: self.dim(),
            debug_info: None,
        })
    }
}

pub(crate) fn validate_count(count: usize) -> Result<()> {
    if count != MAX_COUNT {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) fn validate_geometry(geometry: g::Geometry) -> Result<()> {
    if !geometry.four_members() {
        return Err(rejected());
    }
    super::validate_geometry(geometry)
}
pub(crate) fn validate_limits(limits: &wgpu::Limits) -> Result<()> {
    if limits.max_compute_invocations_per_workgroup < 128
        || limits.max_compute_workgroup_size_x < 128
        || limits.max_compute_workgroup_storage_size < MAX_SHARED_BYTES
        || limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_uniform_buffer_binding_size < 32
        || limits.max_compute_workgroups_per_dimension < 128
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

/// Clears the complete64KiB owner itself, then prepares all four public frames.
/// Parent must authenticate complete physical roots, same client, immutable input
/// transaction and reservation before enqueue; a Handle alone is not that proof.
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
        .with_buffer(owners[1].clone().binding())
        .with_buffer(scratch.binding())
        .with_info(info([
            0,
            spans.input_first(),
            spans.input_stride(),
            0,
            0,
            0,
            0,
            count,
        ]));
    // SAFETY: complete input/owner spans are checked above; fixed four frames
    // are disjoint, including inactive frame stores and the full owning tail.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Prepare), CubeCount::Static(1, 8, 1), args);
    }
    Ok(())
}

/// One original frozen public source partition, replacing only its GU dispatch.
/// Consumers use runtime count/MAP/routes; pipeline keys contain only dtype/row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 2],
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
) -> Result<()> {
    launch_with_schedule(
        client, geometry, spans, banks, owners, scratch, first_slot, slot_count, groups, count,
        false,
    )
}

/// Explicit alternative scheduling: four workgroups per row tile uniformly
/// stride over the same bounded original groups. No arithmetic or owner change.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_group_stride4(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 2],
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
) -> Result<()> {
    launch_with_schedule(
        client, geometry, spans, banks, owners, scratch, first_slot, slot_count, groups, count,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_with_schedule(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 2],
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
    stride_four: bool,
) -> Result<()> {
    validate_layout(
        geometry,
        spans,
        owners.each_ref().map(|h| h.size_in_used() as usize / 4),
        scratch.size_in_used() as usize / 4,
        count,
        groups,
    )?;
    let [gate, up, down] = geometry.projections();
    geometry
        .validate_shard(
            first_slot,
            slot_count,
            [
                gate.segment_words(slot_count)
                    .map_err(Error::InvalidFormat)?,
                up.segment_words(slot_count).map_err(Error::InvalidFormat)?,
                down.segment_words(slot_count)
                    .map_err(Error::InvalidFormat)?,
            ],
        )
        .map_err(Error::InvalidFormat)?;
    for (projection, bank) in [gate, up].into_iter().zip(&banks) {
        if bank.size_in_used()
            != projection
                .segment_words(slot_count)
                .map_err(Error::InvalidFormat)? as u64
                * 4
        {
            return Err(rejected());
        }
    }
    full_handles(
        banks.iter().chain(&owners).chain([&scratch]),
        owners[0].stream,
    )?;
    let mut args = KernelArguments::new();
    for h in [
        &banks[0], &banks[1], &owners[0], &owners[2], &owners[6], &scratch,
    ] {
        args = args.with_buffer(h.clone().binding());
    }
    args = args.with_info(info([
        spans.map_first(),
        first_slot,
        slot_count,
        spans.act_first(),
        spans.act_stride(),
        spans.status_first(),
        if stride_four { groups } else { 0 },
        count,
    ]));
    // SAFETY: parent retains frozen source identity/pending rank protection;
    // shader exits uniformly for inactive partitions before all payload reads.
    unsafe {
        let kernel = if stride_four {
            Kernel::ProjectStride4 {
                dtype: gate.dtype(),
                row_bytes: gate.row_bytes(),
            }
        } else {
            Kernel::Project {
                dtype: gate.dtype(),
                row_bytes: gate.row_bytes(),
            }
        };
        let grid_y = if stride_four {
            GROUP_STRIDE_FOUR
        } else {
            groups as u32
        };
        client.launch_unchecked(Box::new(kernel), CubeCount::Static(80, grid_y, 1), args);
    }
    Ok(())
}

/// Full owner readiness/scale/inactive/header/tail audit, ORing only sticky fault.
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
    // SAFETY: reads the entire authenticated owner; never resets a fault word.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Audit), CubeCount::Static(128, 1, 1), args);
    }
    Ok(())
}

/// Original partition index of one (first_slot, slot_count) shard pair.
pub(crate) fn shard_index(first_slot: usize, slot_count: usize) -> Result<usize> {
    let Some(shard) = FIRST.iter().position(|&first| first == first_slot) else {
        return Err(rejected());
    };
    if PARTS[shard] != slot_count {
        return Err(rejected());
    }
    Ok(shard)
}

/// Checked 12-byte indirect dispatch view derived only from an authenticated
/// full64KiB owner. GU views bytes+8, DOWN views bytes+20 of the record.
pub(crate) fn record_dispatch_view(
    scratch: &Handle,
    shard: usize,
    down: bool,
) -> Result<cubecl_runtime::server::Binding> {
    // CubeCL's contiguous allocator represents a complete owner with Some(0)
    // trims. Require zero numeric trims, not absent Option values. Check trims
    // before size_in_used(), whose subtraction assumes a valid handle.
    if shard >= FIRST.len()
        || scratch.size() != MAIN_BYTES
        || scratch.offset_start.unwrap_or(0) != 0
        || scratch.offset_end.unwrap_or(0) != 0
        || scratch.size_in_used() != MAIN_BYTES
    {
        return Err(rejected());
    }
    let offset =
        RECORD_BASE_WORD as u64 * 4 + RECORD_BYTES * shard as u64 + if down { 20 } else { 8 };
    if offset
        .checked_add(RECORD_DISPATCH_BYTES)
        .is_none_or(|end| end > MAIN_BYTES)
    {
        return Err(rejected());
    }
    Ok(scratch
        .clone()
        .offset_start(offset)
        .offset_end(MAIN_BYTES - offset - RECORD_DISPATCH_BYTES)
        .binding())
}

/// Compact variant: identical four-frame preparation plus the five disjoint
/// shard records derived from the immutable MAP. Parent retains all original
/// physical root/owner/transaction authentication.
pub(crate) fn prepare_compact(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    owners: [Handle; 7],
    scratch: Handle,
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
    if !(1..=RECORD_GROUP_IDS).contains(&groups) {
        return Err(rejected());
    }
    full_handles(owners.iter().chain([&scratch]), owners[0].stream)?;
    let args = KernelArguments::new()
        .with_buffer(owners[1].clone().binding())
        .with_buffer(scratch.binding())
        .with_buffer(owners[0].clone().binding())
        .with_info(info([
            0,
            spans.input_first(),
            spans.input_stride(),
            spans.map_first(),
            groups,
            0,
            0,
            count,
        ]));
    // SAFETY: complete input/owner spans are checked above; fixed four frames
    // are disjoint, and token0's five record lanes write disjoint tail words.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::PrepareCompact),
            CubeCount::Static(1, 8, 1),
            args,
        );
    }
    Ok(())
}

/// Compact variant for one original frozen source partition. The immutable
/// shard record's GU dispatch triplet drives an indirect launch; group
/// selection, member order and arithmetic stay the original stride-four body.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_compact(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 2],
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
) -> Result<()> {
    launch_compact_inner(
        client,
        geometry,
        spans,
        banks,
        owners,
        scratch,
        first_slot,
        slot_count,
        groups,
        count,
        CompactProjection::Original,
    )
}

/// Separately selected fixed-public experiment: original compact count4 GU with only
/// the primary Q8 activation plane. Prepare/audit still own both full planes.
/// The original CPU13 vector/carry/greedy gate must accept before timing credit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_compact_first_plane(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 2],
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
) -> Result<()> {
    launch_compact_inner(
        client,
        geometry,
        spans,
        banks,
        owners,
        scratch,
        first_slot,
        slot_count,
        groups,
        count,
        CompactProjection::FirstPlane,
    )
}

/// Count4 exact-tree shuffle candidate. Caller enables subgroup support with
/// admitted16/32 widths; the shader independently guards its actual width.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_compact_shuffle(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 2],
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
) -> Result<()> {
    launch_compact_inner(
        client,
        geometry,
        spans,
        banks,
        owners,
        scratch,
        first_slot,
        slot_count,
        groups,
        count,
        CompactProjection::Shuffle,
    )
}

#[derive(Clone, Copy)]
enum CompactProjection {
    Original,
    FirstPlane,
    Shuffle,
}

#[allow(clippy::too_many_arguments)]
fn launch_compact_inner(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [Handle; 2],
    owners: [Handle; 7],
    scratch: Handle,
    first_slot: usize,
    slot_count: usize,
    groups: usize,
    count: usize,
    projection: CompactProjection,
) -> Result<()> {
    let shard = shard_index(first_slot, slot_count)?;
    validate_layout(
        geometry,
        spans,
        owners.each_ref().map(|h| h.size_in_used() as usize / 4),
        scratch.size_in_used() as usize / 4,
        count,
        groups,
    )?;
    let [gate, up, down] = geometry.projections();
    geometry
        .validate_shard(
            first_slot,
            slot_count,
            [
                gate.segment_words(slot_count)
                    .map_err(Error::InvalidFormat)?,
                up.segment_words(slot_count).map_err(Error::InvalidFormat)?,
                down.segment_words(slot_count)
                    .map_err(Error::InvalidFormat)?,
            ],
        )
        .map_err(Error::InvalidFormat)?;
    for (projection, bank) in [gate, up].into_iter().zip(&banks) {
        if bank.size_in_used()
            != projection
                .segment_words(slot_count)
                .map_err(Error::InvalidFormat)? as u64
                * 4
        {
            return Err(rejected());
        }
    }
    full_handles(
        banks.iter().chain(&owners).chain([&scratch]),
        owners[0].stream,
    )?;
    let dispatch = record_dispatch_view(&scratch, shard, false)?;
    let mut args = KernelArguments::new();
    for h in [
        &banks[0], &banks[1], &owners[0], &owners[2], &owners[6], &scratch,
    ] {
        args = args.with_buffer(h.clone().binding());
    }
    args = args.with_info(info([
        spans.map_first(),
        first_slot,
        slot_count,
        spans.act_first(),
        spans.act_stride(),
        spans.status_first(),
        shard,
        count,
    ]));
    // SAFETY: parent retains frozen source identity/pending rank protection;
    // the record bounds every list access and the shard dispatch uniformly.
    unsafe {
        client.launch_unchecked(
            Box::new(match projection {
                CompactProjection::FirstPlane => Kernel::ProjectCompactFirstPlane {
                    dtype: gate.dtype(),
                    row_bytes: gate.row_bytes(),
                },
                CompactProjection::Shuffle => Kernel::ProjectCompactShuffle {
                    dtype: gate.dtype(),
                    row_bytes: gate.row_bytes(),
                },
                CompactProjection::Original => Kernel::ProjectCompact {
                    dtype: gate.dtype(),
                    row_bytes: gate.row_bytes(),
                },
            }),
            CubeCount::Dynamic(dispatch),
            args,
        );
    }
    Ok(())
}

/// Replacement audit for the compact path: validates the whole owner, the five
/// reconstructed shard records, then erases each disjoint record. A failed
/// validation keeps the records and stickies the original fault.
pub(crate) fn audit_compact(
    client: &ComputeClient<WgpuRuntime>,
    scratch: Handle,
    fault: Handle,
    map: Handle,
    count: usize,
    groups: usize,
    map_first: usize,
) -> Result<()> {
    validate_count(count)?;
    if scratch.size_in_used() != MAIN_BYTES || fault.size_in_used() != 32768 {
        return Err(rejected());
    }
    if !(1..=RECORD_GROUP_IDS).contains(&groups) {
        return Err(rejected());
    }
    full_handles([&scratch, &fault, &map], scratch.stream)?;
    let args = KernelArguments::new()
        .with_buffer(scratch.binding())
        .with_buffer(fault.binding())
        .with_buffer(map.binding())
        .with_info(info([count, map_first, groups, 0, 0, 0, 0, 0]));
    // SAFETY: reads/erases the authenticated owner after all consumers; the
    // fault word is never reset.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::AuditCompact),
            CubeCount::Static(129, 1, 1),
            args,
        );
    }
    Ok(())
}

pub(crate) fn compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut result = Vec::new();
    for dtype in [
        crate::quant_matvec::IQ2_S,
        crate::quant_matvec::IQ3_XXS,
        crate::quant_matvec::IQ3_S,
        crate::quant_matvec::IQ4_XS,
    ] {
        let row_bytes = kuno_text::TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(2560)
            .unwrap() as usize;
        result.push(
            Kernel::ProjectCompact { dtype, row_bytes }
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap(),
        );
    }
    for kernel in [Kernel::PrepareCompact, Kernel::AuditCompact] {
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
    }
    result
}
