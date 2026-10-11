//! Explicit frozen GU-only two-residual-Q8/packed-dot candidate.
//! Original public banks, private rank MAP, DOWN and all owning spans remain
//! caller-owned. This is alternate arithmetic, never original64 F32 parity.
//! Packing/quantizer provenance and MIT notices: iq_moe_integer-NOTICE.txt.
#![allow(dead_code)] // Parent wiring is a separate integration commit.

use super::{grouped_moe as g, iq_moe_integer as integer};
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

pub(crate) const MAX_COUNT: usize = 3;
pub(crate) const GROUP_STRIDE_FOUR: u32 = 4;
pub(crate) const HEADER_WORDS: usize = 64;
pub(crate) const FRAME_WORDS: usize = 1440;
pub(crate) const LOGICAL_WORDS: usize = HEADER_WORDS + MAX_COUNT * FRAME_WORDS;
// Allocate this exact logical extent, not LOGICAL_WORDS (which rounds to32KiB).
pub(crate) const SCRATCH_WORDS: usize = 16384;
pub(crate) const MAIN_BYTES: u64 = (SCRATCH_WORDS * 4) as u64;
pub(crate) const HOST_INCREMENT: usize = MAIN_BYTES as usize;
pub(crate) const MAX_PROJECT_SHARED_BYTES: u32 = 8192 + 1536;
pub(crate) const MAX_PREPARE_SHARED_BYTES: u32 = 16004;
pub(crate) const MAX_SHARED_BYTES: u32 = MAX_PREPARE_SHARED_BYTES;
// Prepare and audit are additional to the original five GU/five DOWN calls.
pub(crate) const INFO_INCREMENT: u64 = 2 * 48 * 2048;
pub(crate) const RUNTIME_INCREMENT: u64 = INFO_INCREMENT + 2 * 2048 + 4 * MAIN_BYTES;

fn rejected() -> Error {
    Error::InvalidFormat("public frozen GU integer eight-lane candidate")
}
fn template(section: &str) -> &'static str {
    include_str!("frozen_gu_integer.wgsl")
        .split_once(&format!("// FROZEN_GU_{section}_BEGIN\n"))
        .unwrap()
        .1
        .split_once(&format!("// FROZEN_GU_{section}_END"))
        .unwrap()
        .0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Prepare,
    Project { dtype: u32, row_bytes: usize },
    ProjectStride4 { dtype: u32, row_bytes: usize },
    Audit,
}

fn grid_source(dtype: u32) -> (String, String) {
    use std::fmt::Write;
    let words = integer::original_grid_words(dtype);
    if words.is_empty() {
        return (String::new(), String::new());
    }
    let declaration = format!("var<workgroup> iq3_grid:array<u32,{}>;", words.len());
    let mut initialize = String::from("switch local {\n");
    for lane in 0..64 {
        writeln!(initialize, "case {lane}u: {{").unwrap();
        for index in (lane..words.len()).step_by(64) {
            writeln!(initialize, "iq3_grid[{index}u]=0x{:08x}u;", words[index]).unwrap();
        }
        initialize.push_str("}\n");
    }
    initialize.push_str("default: {}\n}\n");
    (declaration, initialize)
}

// The original byte layout and factor round points stay unchanged. Cache just
// the encoded32-value payload/metadata, then expand each quartet immediately;
// holding all eight decoded words for both projections raises register demand.
// Every header/payload address below is2-byte aligned in the frozen profiles.
fn decode(name: &str, dtype: u32) -> String {
    let (prepare, factor, packed) = match dtype {
        super::IQ3_XXS => (
            "let base=row+(block/8u)*98u;let group=block%8u;let codes=NAME_load8(base+2u+group*8u);return NAME_Encoded(vec4<u32>(codes,0u,0u),NAME_word(base+66u+group*4u),0u,0u,NAME_half(base));",
            "return e.d*(0.5+f32(e.signs>>28u))*0.5;",
            "let octet=quartet/2u;let sign_code=(e.signs>>(octet*7u))&127u;let signs=sign_code|((countOneBits(sign_code)&1u)<<7u);let codes=select(e.codes.x,e.codes.y,quartet>=4u);let index=(codes>>((quartet%4u)*8u))&255u;return iq3_signed_word(iq3_grid[index],signs>>((quartet%2u)*4u));",
        ),
        super::IQ3_S => (
            "let base=row+(block/8u)*110u;let group=block%8u;let codes=NAME_load8(base+2u+group*8u);let scale=(NAME_byte(base+106u+group/2u)>>((group%2u)*4u))&15u;return NAME_Encoded(vec4<u32>(codes,0u,0u),NAME_word(base+74u+group*4u),NAME_byte(base+66u+group),scale,NAME_half(base));",
            "return e.d*f32(1u+2u*e.scale);",
            "let codes=select(e.codes.x,e.codes.y,quartet>=4u);let index=((codes>>((quartet%4u)*8u))&255u)|(((e.high>>quartet)&1u)<<8u);let signs=(e.signs>>((quartet/2u)*8u))&255u;return iq3_signed_word(iq3_grid[index],signs>>((quartet%2u)*4u));",
        ),
        super::IQ2_S => (
            "let base=row+(block/8u)*82u;let group=block%8u;return NAME_Encoded(vec4<u32>(NAME_word(base+2u+group*4u),0u,0u,0u),NAME_word(base+34u+group*4u),NAME_byte(base+66u+group),NAME_byte(base+74u+group),NAME_half(base));",
            "let scale=(e.scale>>(half*4u))&15u;return e.d*(0.5+f32(scale))*0.25;",
            "let octet=quartet/2u;let index=((e.codes.x>>(octet*8u))&255u)|(((e.high>>(octet*2u))&3u)<<8u);let signs=(e.signs>>(octet*8u))&255u;return iq3_signed_word(iq3_grid[index*2u+quartet%2u],signs>>((quartet%2u)*4u));",
        ),
        super::IQ4_XS => (
            "let base=row+(block/8u)*136u;let group=block%8u;let at=base+8u+group*16u;let header=NAME[base/4u];let lo=(NAME_byte(base+4u+group/2u)>>((group%2u)*4u))&15u;let hi=(header>>(16u+group*2u))&3u;return NAME_Encoded(vec4<u32>(NAME[at/4u],NAME[at/4u+1u],NAME[at/4u+2u],NAME[at/4u+3u]),0u,0u,lo|(hi<<4u),unpack2x16float(header&65535u).x);",
            "return e.d*f32(i32(e.scale)-32);",
            "var codes=e.codes.x;switch quartet%4u {case 1u:{codes=e.codes.y;}case 2u:{codes=e.codes.z;}case 3u:{codes=e.codes.w;}default:{}}return iq4_word(codes,quartet>=4u);",
        ),
        _ => unreachable!("validated frozen integer GU format"),
    };
    format!(
        r#"
struct NAME_Encoded {{codes:vec4<u32>,signs:u32,high:u32,scale:u32,d:f32}}
fn NAME_byte(at:u32)->u32 {{return (NAME[at/4u]>>((at%4u)*8u))&255u;}}
fn NAME_word(at:u32)->u32 {{
    let shift=(at%4u)*8u;let low=NAME[at/4u];
    if shift==0u {{return low;}}
    return (low>>shift)|(NAME[at/4u+1u]<<(32u-shift));
}}
fn NAME_load8(at:u32)->vec2<u32> {{
    let low=NAME[at/4u];let middle=NAME[at/4u+1u];
    if at%4u==0u {{return vec2<u32>(low,middle);}}
    let high=NAME[at/4u+2u];
    return vec2<u32>((low>>16u)|(middle<<16u),(middle>>16u)|(high<<16u));
}}
fn NAME_half(at:u32)->f32 {{return unpack2x16float((NAME[at/4u]>>((at%4u)*8u))&65535u).x;}}
fn NAME_prepare(row:u32,block:u32)->NAME_Encoded {{{prepare}}}
fn NAME_factor(e:NAME_Encoded,half:u32)->f32 {{{factor}}}
fn NAME_packed(e:NAME_Encoded,quartet:u32)->u32 {{{packed}}}
"#
    )
    .replace("NAME", name)
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
                    .replace("BLOCKS_VALUE", "80")
            }
            Self::Audit => template("AUDIT").to_owned(),
            Self::Project { dtype, row_bytes } | Self::ProjectStride4 { dtype, row_bytes } => {
                let stride_four = matches!(self, Self::ProjectStride4 { .. });
                // Both schedules use one arithmetic body. This keeps decoder,
                // member order and the fixed lane tree identical by construction.
                let project = template("PROJECT");
                let source = if stride_four {
                    let prefix = project
                        .split_once("@compute @workgroup_size(64)")
                        .unwrap()
                        .0;
                    format!(
                        "{prefix}{}",
                        template("PROJECT_STRIDE4_MAIN")
                            .replace("PROJECT_BODY_VALUE", template("PROJECT_BODY"))
                    )
                    .replace("p.g!=0u", "p.g<1u || p.g>80u")
                    .replace("group>=groups", "group>=groups || group>=p.g")
                } else {
                    project.to_owned()
                };
                let group_index = if stride_four {
                    "routed_group"
                } else {
                    "group.y"
                };
                let (grid, initialize) = grid_source(dtype);
                let helpers = include_str!("iq_moe_integer.wgsl")
                    .split_once("fn iq4_byte")
                    .unwrap()
                    .1
                    .split_once("// Original byte-addressed")
                    .unwrap()
                    .0;
                let halves = if dtype == super::IQ2_S { 2 } else { 1 };
                let quartets = if halves == 2 { 4 } else { 8 };
                source
                    .replace("GRID_DECLARATION_VALUE", &grid)
                    .replace("GRID_INITIALIZE_VALUE", &initialize)
                    .replace("DECODE_HELPERS_VALUE", &format!("fn iq4_byte{helpers}"))
                    .replace("PROJECTION_HELPERS_VALUE", &format!("{}\n{}", decode("gate", dtype), decode("up", dtype)))
                    .replace("ROW_BYTES_VALUE", &row_bytes.to_string())
                    .replace("HALVES_VALUE", &halves.to_string())
                    .replace("QUARTETS_VALUE", &quartets.to_string())
                    .replace("FRAME_MEMBERS_VALUE", &members(&format!("frames.AXIS=64u+(map[p.a+164u+{group_index}*8u+MEMBER_INDEXu]/10u)*1440u;\n")))
                    .replace("DOT_MEMBERS_VALUE", &members("{let first=planes[frames.AXIS+at];let second=planes[frames.AXIS+640u+at];gf.AXIS+=dot4I8Packed(gq,first);gs.AXIS+=dot4I8Packed(gq,second);uf.AXIS+=dot4I8Packed(uq,first);us.AXIS+=dot4I8Packed(uq,second);}\n"))
                    .replace("SCALE_MEMBERS_VALUE", &members("{let a=bitcast<f32>(planes[frames.AXIS+1280u+block]);let b=bitcast<f32>(planes[frames.AXIS+1360u+block]);let gp=a*f32(gf.AXIS)+b*f32(gs.AXIS);let up=a*f32(uf.AXIS)+b*f32(us.AXIS);gates.AXIS+=gd*gp;ups.AXIS+=ud*up;}\n"))
                    .replace("OUTPUT_MEMBERS_VALUE", &members(&format!("{{let route=map[p.a+164u+{group_index}*8u+MEMBER_INDEXu];output[p.d+route*p.e+row]=silu(gate_sums[local*3u+MEMBER_INDEXu])*up_sums[local*3u+MEMBER_INDEXu];if row==0u {{status[p.f+route]=1u;}}}}\n")))
            }
        }
    }
    fn dim(self) -> CubeDim {
        CubeDim::new_1d(
            if matches!(self, Self::Project { .. } | Self::ProjectStride4 { .. }) {
                64
            } else {
                128
            },
        )
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare => "FrozenGuIntegerPrepare",
            Self::Project { .. } => "FrozenGuIntegerEightLanes",
            Self::ProjectStride4 { .. } => "FrozenGuIntegerEightLanesStride4",
            Self::Audit => "FrozenGuIntegerAudit",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self).cube_dim(self.dim())
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
            cube_dim: self.dim(),
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
    if scratch_words != SCRATCH_WORDS || !(1..=g::MAX_GROUPS).contains(&groups) {
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

/// Clears the complete64KiB owner itself, then prepares all three public frames.
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
    // SAFETY: complete input/owner spans are checked above; fixed three frames
    // are disjoint, including inactive frame stores and the full owning tail.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Prepare), CubeCount::Static(1, 3, 1), args);
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

#[cfg(test)]
#[path = "frozen_gu_integer_tests.rs"]
mod tests;
