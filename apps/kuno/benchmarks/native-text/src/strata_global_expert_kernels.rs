//! MIT Strata fb58e0 native expert row_entries over the owning global cache.
//! Original packed VRAM cells or registered HOST aliases are read directly in
//! one 4/2/1 row-entry loop per role, matching the Intel NO_HOST device plan.
//! Request state is supplied solely by PrivateArena. The older explicit fetch
//! kernel remains available outside this selected decode path.
use super::*;
use crate::strata_global_sources::Sources;

pub(crate) const STAGE_BYTES: u64 = 128 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Fetch,
    Role {
        role: u8,
        dtype: u32,
        row_bytes: usize,
        direct_planes: bool,
    },
    PrepareGu,
    PrepareDown,
}
impl Op {
    fn source(self) -> String {
        match self {
            Self::Fetch => include_str!("strata_global_fetch.wgsl").into(),
            Self::Role {
                role,
                dtype,
                row_bytes,
                direct_planes,
            } => role_source(role, dtype, row_bytes, direct_planes),
            Self::PrepareGu => native_q8::source(false),
            Self::PrepareDown => native_q8::source(true),
        }
    }
    fn dim(self) -> CubeDim {
        CubeDim::new_1d(256)
    }
}
impl KernelMetadata for Op {
    fn name(&self) -> &'static str {
        match self {
            Self::Fetch => "StrataGlobalFetchAndRebase",
            Self::Role { role: 1, .. } => "StrataGlobalNativeGate",
            Self::Role { role: 2, .. } => "StrataGlobalNativeUp",
            Self::Role { role: 4, .. } => "StrataGlobalNativeGateUp",
            Self::Role { .. } => "StrataGlobalNativeDown",
            Self::PrepareGu => "StrataGlobalPrepareGu",
            Self::PrepareDown => "StrataGlobalSwiGluQ8Down",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self).cube_dim(self.dim())
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Op {
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
            cube_dim: self.dim(),
        })
    }
}
fn params(values: [usize; 8]) -> MetadataBindingInfo {
    MetadataBindingInfo::custom(
        values
            .chunks_exact(2)
            .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
            .collect(),
    )
}
fn full(handles: impl Iterator<Item = Handle>) -> Result<()> {
    let mut stream = None;
    for h in handles {
        if h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != h.size_in_used()
            || stream.is_some_and(|s| s != h.stream)
        {
            return Err(Error::InvalidBufferLength);
        }
        stream = Some(h.stream);
    }
    Ok(())
}
fn role_source(role: u8, dtype: u32, row_bytes: usize, direct_planes: bool) -> String {
    if role == 4 {
        return joint_gu_source(dtype, row_bytes, direct_planes);
    }
    let down = role == 3;
    let mut base = if down {
        crate::quant_matvec::frozen_down_integer::strata_fused_source(dtype, row_bytes, 8)
    } else {
        crate::quant_matvec::frozen_gu_integer::strata_fused_source(dtype, row_bytes, 8)
    };
    base = base.replace("p.h<5u", "p.h<1u");
    let names: &[&str] = if down { &["down"] } else { &["gate", "up"] };
    let first = base
        .find("@group(0) @binding(0)")
        .expect("native source descriptors");
    let end = first
        + base[first..].find("var<uniform> p:Params;").unwrap()
        + "var<uniform> p:Params;".len();
    let mut descriptors = String::from(
        "// STRATA_PHYSICAL_EXPERTS\n// STRATA_ROUND_PRODUCTS: stable separate expert products across storage modes.\n",
    );
    let mut index = 0;
    for name in names {
        descriptors.push_str(&format!(
            "@group(0) @binding({index}) var<storage,read> {name}0:array<vec4<u32>>;\n"
        ));
        index += 1;
    }
    for (name, access, ty) in [
        ("map", "read", "u32"),
        ("output", "read_write", "f32"),
        ("status", "read_write", "u32"),
        ("planes", "read", "u32"),
    ] {
        descriptors.push_str(&format!(
            "@group(0) @binding({index}) var<storage,{access}> {name}:array<{ty}>;\n"
        ));
        index += 1;
    }
    let grid_binding = index;
    if !down {
        index += 1;
    }
    descriptors.push_str(&format!(
        "@group(0) @binding({index}) var<storage,read> addresses:array<vec4<u32>>;\n"
    ));
    index += 1;
    descriptors.push_str(&format!(
        "@group(0) @binding({index}) var<uniform> p:Params;\nvar<private> strata_first:u32;\n"
    ));
    for name in names {
        descriptors.push_str(&format!("var<private> strata_{name}_address:vec2<u32>;\n"));
    }
    base.replace_range(first..end, &descriptors);
    let old = if down {
        "    if arrayLength(&down)!=p.c*2560u*(ROW_BYTES/4u) {return false;}"
    } else {
        "    if arrayLength(&gate)!=p.c*640u*(ROW_BYTES/4u) || arrayLength(&up)!=arrayLength(&gate) {return false;}"
    };
    assert!(base.contains(old));
    base = base.replace(old, "    if arrayLength(&addresses)!=2048u {return false;}");
    let old = "let slot=map[p.a+4u+group];if slot<p.b || slot-p.b>=p.c {return false;}";
    assert!(base.contains(old));
    let mut choose = String::from(
        "let slot=map[p.a+4u+group];if slot>=512u || p.b!=0u || p.c!=512u {return false;}strata_first=slot;\n",
    );
    for name in names {
        let role = match *name {
            "gate" => 0,
            "up" => 1,
            _ => 2,
        };
        let rows = if down { 2560 } else { 640 };
        // Both tiers are actual immutable original-weight allocations. HOST
        // already has a registered device address; do not redirect it through
        // a stage or require a stamp from a fetch that does not execute.
        choose.push_str(&format!("{{let src=addresses[{role}u*512u+slot];let words={rows}u*(ROW_BYTES/4u);if src.z!=words || src.w>1u || (src.x==0u && src.y==0u) || (src.x&15u)!=0u {{return false;}}strata_{name}_address=src.xy;}}\n"));
    }
    base = base.replace(old, &choose);
    for name in names {
        base = replace_reads(base, name);
        base.push_str(&format!(
            "\nfn strata_{name}(index:u32)->u32 {{return {name}0[index/4u][index%4u];}}\n"
        ));
    }
    if !down {
        base = crate::strata_intel_core::shared_table(base, grid_binding, dtype);
    }
    let mut bodies = String::new();
    let mut prefix = None;
    let signature = "fn main(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) local:u32,\n        @builtin(subgroup_id) sg:u32, @builtin(subgroup_invocation_id) sub_lane:u32,\n        @builtin(subgroup_size) subgroup_width:u32, @builtin(num_subgroups) subgroups:u32)";
    for width in [4, 2, 1] {
        let mut variant = roles::source(base.clone(), down, dtype, width, 8, role, direct_planes);
        if dtype == crate::quant_matvec::IQ4_XS && width == 1 {
            variant = variant.replace(
                "let member_base=(length/2u)*2u;if group.z!=0u || length%2u==0u {return;}",
                "let member_base=group.z;if member_base>=length {return;}",
            );
        }
        let first = variant.find("@compute @workgroup_size(256)").unwrap();
        let fn_start = variant[first..].find("fn main").unwrap() + first;
        let end = crate::strata_port::function_end(&variant, fn_start);
        if prefix.is_none() {
            prefix = Some(format!("{}{}", &variant[..first], &variant[end..]));
        }
        let body = &variant[fn_start..end];
        assert!(body.contains(signature));
        bodies.push_str(&body.replace(signature,&format!("fn strata_chunk{width}(group:vec3<u32>,local:u32,sg:u32,sub_lane:u32,subgroup_width:u32,subgroups:u32)")));
        bodies.push('\n');
    }
    let loop_body = if dtype == crate::quant_matvec::IQ4_XS {
        "for(var entry=0u;entry<length;entry++) {strata_chunk1(vec3<u32>(group.xy,entry),local,sg,sub_lane,subgroup_width,subgroups);}"
    } else {
        "for(var at=0u;at<length/4u;at++) {strata_chunk4(vec3<u32>(group.xy,at),local,sg,sub_lane,subgroup_width,subgroups);}if length%4u>=2u {strata_chunk2(group,local,sg,sub_lane,subgroup_width,subgroups);}if length%2u!=0u {strata_chunk1(group,local,sg,sub_lane,subgroup_width,subgroups);}"
    };
    // Pinned row_entries: all dependent cardinalities are in the SAME dispatch.
    // Direct immutable input reads have no shared tile to overwrite; chunks
    // publish distinct route rows and need no inter-chunk workgroup barrier.
    let loop_body = if direct_planes {
        loop_body.to_owned()
    } else {
        loop_body.replace(";}", ";workgroupBarrier();}")
    };
    format!(
        "{}\n{bodies}\n@compute @workgroup_size(256)\n{signature} {{\nif group.z!=0u || subgroup_width!=32u || subgroups!=8u || !valid_group(group.y) {{return;}}\nlet length=map[p.a+84u+group.y];\n{loop_body}\n}}",
        prefix.unwrap()
    )
}

// Same original native_gu_kernel row-domain split: the lower half of grid.x
// writes gate and the upper half writes up. No lane computes both projections;
// the one dispatch owns both disjoint output planes before SwiGLU consumes them.
fn joint_gu_source(dtype: u32, row_bytes: usize, direct_planes: bool) -> String {
    let gate = role_source(1, dtype, row_bytes, direct_planes);
    let up = role_source(2, dtype, row_bytes, direct_planes);
    let split = |s: &str| s.find("fn strata_chunk4(").expect("actual GU chunk body");
    let (g, u) = (split(&gate), split(&up));
    assert_eq!(&gate[..g], &up[..u]);
    let entry = |s: &str, name: &str| {
        s.replace("@compute @workgroup_size(256)\n", "")
            .replace("fn main(", &format!("fn strata_{name}_entry("))
            .replace("strata_chunk", &format!("strata_{name}_chunk"))
    };
    let half = 640 / (256 / crate::strata_port::expert_lanes());
    format!(
        "{}\n{}\n{}\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) sub_lane:u32,@builtin(subgroup_size) subgroup_width:u32,@builtin(num_subgroups) subgroups:u32) {{\nif group.x<{half}u {{strata_gate_entry(group,local,sg,sub_lane,subgroup_width,subgroups);}} else if group.x<{}u {{strata_up_entry(vec3<u32>(group.x-{half}u,group.yz),local,sg,sub_lane,subgroup_width,subgroups);}}\n}}",
        &gate[..g],
        entry(&gate[g..], "gate"),
        entry(&up[u..], "up"),
        half * 2
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fetch(
    client: &ComputeClient<WgpuRuntime>,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    sources: &Sources,
    layer: usize,
    geometry: g::Geometry,
    map: &Handle,
    status: &Handle,
    stages: &[Handle; 3],
    count: usize,
) -> Result<()> {
    if !(1..=8).contains(&count) {
        return Err(Error::InvalidBufferLength);
    }
    full(stages.iter().cloned().chain([map.clone(), status.clone()]))?;
    for (role, stage) in stages.iter().enumerate() {
        fetch_role(
            client, graph, sources, layer, geometry, map, status, stage, count, role,
        )?;
    }
    Ok(())
}
/// Explicit staging helper for consumers that request it. The selected Intel
/// NO_HOST decode path reads source aliases directly and never invokes this.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fetch_role(
    client: &ComputeClient<WgpuRuntime>,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    sources: &Sources,
    layer: usize,
    geometry: g::Geometry,
    map: &Handle,
    status: &Handle,
    stage: &Handle,
    count: usize,
    role: usize,
) -> Result<()> {
    let projection = geometry
        .projections()
        .get(role)
        .copied()
        .ok_or(Error::InvalidBufferLength)?;
    let bytes = projection.expert_bytes();
    if !(1..=8).contains(&count)
        || !bytes.is_multiple_of(16)
        || bytes as u64 * 80 > STAGE_BYTES
        || stage.size_in_used() != STAGE_BYTES
    {
        return Err(Error::InvalidBufferLength);
    }
    full([stage.clone(), map.clone(), status.clone()].into_iter())?;
    let args = KernelArguments::new()
        .with_buffer(sources.prototype(graph, layer, role)?.binding())
        .with_buffer(map.clone().binding())
        .with_buffer(stage.clone().binding())
        .with_buffer(status.clone().binding());
    let args =
        sources
            .bind(layer, args)?
            .with_info(params([0, count, bytes / 16, role, 0, 0, 0, 0]));
    // SAFETY: authentic original source extents, grouped disjoint raw slices;
    // the graph orders this fetch before the consuming dequantization.
    unsafe {
        client.launch_unchecked(
            Box::new(Op::Fetch),
            CubeCount::Static((bytes / 16).div_ceil(256) as u32, 80, 1),
            args,
        );
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn role(
    client: &ComputeClient<WgpuRuntime>,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    sources: &Sources,
    layer: usize,
    geometry: g::Geometry,
    spans: &g::Spans,
    owners: &[Handle; 7],
    planes: &Handle,
    count: usize,
    role: u8,
) -> Result<()> {
    if !(1..=8).contains(&count) || !(1..=4).contains(&role) {
        return Err(Error::InvalidBufferLength);
    }
    let down = role == 3;
    let mut args = KernelArguments::new();
    for r in if down { 2..3 } else { 0..2 } {
        args = args.with_buffer(sources.prototype(graph, layer, r)?.binding());
    }
    for h in [&owners[0], &owners[3], &owners[6], planes] {
        args = args.with_buffer(h.clone().binding());
    }
    if !down {
        args = crate::strata_intel_core::bind(args, owners[0].stream)?;
    }
    full(owners.iter().chain([planes]).cloned())?;
    let args = sources.bind(layer, args)?.with_info(params([
        spans.map_first(),
        0,
        512,
        spans.partial_first(),
        spans.partial_stride(),
        spans.status_first(),
        if down { 80 } else { 0 },
        count,
    ]));
    let projection = geometry.projections()[if down { 2 } else { 0 }];
    // SAFETY: one actual group source is chosen uniformly before shared loads;
    // all original role, map and private row extents are checked in WGSL.
    unsafe {
        client.launch_unchecked(
            Box::new(Op::Role {
                role,
                dtype: projection.dtype(),
                row_bytes: projection.row_bytes(),
                direct_planes: true,
            }),
            CubeCount::Static(
                (projection.rows() * if role == 4 { 2 } else { 1 }
                    / (256 / crate::strata_port::expert_lanes())) as u32,
                80,
                1,
            ),
            args,
        );
    }
    Ok(())
}
pub(crate) fn prepare_gu(
    client: &ComputeClient<WgpuRuntime>,
    spans: &g::Spans,
    owners: &[Handle; 7],
    scratch: &Handle,
    count: usize,
) -> Result<()> {
    let frames = native_q8::GU.frames(count)?;
    if scratch.size_in_used() != (native_q8::GU.words * 4) as u64 {
        return Err(Error::InvalidBufferLength);
    }
    full(owners.iter().chain([scratch]).cloned())?;
    let args = KernelArguments::new()
        .with_buffer(owners[1].clone().binding())
        .with_buffer(scratch.clone().binding())
        .with_info(params([
            count,
            spans.input_first(),
            spans.input_stride(),
            0,
            0,
            0,
            0,
            0,
        ]));
    // SAFETY: original complete input/scratch owners; native producer validates
    // active rows and clears every inactive/header/padding span without aliasing.
    unsafe {
        client.launch_unchecked(
            Box::new(Op::PrepareGu),
            CubeCount::Static(1, frames as u32, 1),
            args,
        );
    }
    Ok(())
}
pub(crate) fn prepare_down(
    client: &ComputeClient<WgpuRuntime>,
    spans: &g::Spans,
    owners: &[Handle; 7],
    scratch: &Handle,
    count: usize,
) -> Result<()> {
    let frames = native_q8::DOWN.frames(count)?;
    if scratch.size_in_used() != (native_q8::DOWN.words * 4) as u64 {
        return Err(Error::InvalidBufferLength);
    }
    full(owners.iter().chain([scratch]).cloned())?;
    let args = KernelArguments::new()
        .with_buffer(owners[3].clone().binding())
        .with_buffer(owners[6].clone().binding())
        .with_buffer(scratch.clone().binding())
        .with_info(params([
            count,
            spans.partial_first(),
            spans.partial_stride(),
            spans.status_first(),
            0,
            0,
            0,
            0,
        ]));
    // SAFETY: same actual PART/status/planes owners as both original passes.
    // The fused native producer consumes both GU stamps, stores the rounded
    // product, packs Q8_1 and clears dead GU storage before DOWN overwrites it.
    unsafe {
        client.launch_unchecked(
            Box::new(Op::PrepareDown),
            CubeCount::Static(1, frames as u32, 1),
            args,
        );
    }
    Ok(())
}
pub(crate) fn compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let mut ops = vec![Op::Fetch, Op::PrepareGu, Op::PrepareDown];
    for role in [1, 2, 3, 4] {
        for dtype in if role == 3 {
            vec![crate::quant_matvec::IQ4_NL, crate::quant_matvec::Q2_0]
        } else {
            vec![
                crate::quant_matvec::IQ2_S,
                crate::quant_matvec::IQ3_S,
                crate::quant_matvec::IQ3_XXS,
                crate::quant_matvec::IQ4_XS,
            ]
        } {
            let columns = if role == 3 { 640 } else { 2560 };
            let row_bytes = kuno_text::TensorType::from_gguf_id(dtype)
                .unwrap()
                .row_bytes(columns)
                .unwrap() as usize;
            ops.push(Op::Role {
                role,
                dtype,
                row_bytes,
                direct_planes: true,
            });
        }
    }
    ops.into_iter()
        .map(|op| {
            op.compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap()
        })
        .collect()
}

#[cfg(all(test, feature = "strata-intel-core"))]
mod direct_source_tests {
    use super::*;
    #[test]
    fn every_actual_global_role_validates_without_activation_workgroup_storage() {
        use wgpu::naga::{AddressSpace, valid};
        let mut roles = 0;
        for compiled in compiled_sources() {
            let joint = compiled.source.contains("fn strata_gate_chunk4(");
            if !joint && !compiled.source.contains("fn strata_chunk4(") {
                continue;
            }
            roles += 1;
            assert!(compiled.source.contains("STRATA_ROUND_PRODUCTS"));
            let module = wgpu::naga::front::wgsl::parse_str(&compiled.source).unwrap();
            valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
                .validate(&module)
                .unwrap();
            assert!(
                module
                    .global_variables
                    .iter()
                    .all(|(_, g)| g.space != AddressSpace::WorkGroup)
            );
            assert!(!compiled.source.contains("workgroupBarrier()"));
            assert!(!compiled.source.contains("strata_input["));
            let names: &[&str] = if joint {
                &[
                    "strata_input_word",
                    "strata_gate_chunk4",
                    "strata_gate_chunk2",
                    "strata_gate_chunk1",
                    "strata_up_chunk4",
                    "strata_up_chunk2",
                    "strata_up_chunk1",
                    "strata_gate_entry",
                    "strata_up_entry",
                ]
            } else {
                &[
                    "strata_input_word",
                    "strata_chunk4",
                    "strata_chunk2",
                    "strata_chunk1",
                ]
            };
            for &name in names {
                assert_eq!(
                    module
                        .functions
                        .iter()
                        .filter(|(_, f)| f.name.as_deref() == Some(name))
                        .count(),
                    1
                );
            }
        }
        assert_eq!(roles, 14); // Separate and joint GU/four formats, two DOWN formats.
    }
    #[test]
    fn actual_role_and_explicit_fetch_schemas_bind_original_addresses_without_a_dummy_stage_table()
    {
        let sources = compiled_sources();
        assert_eq!(sources.len(), 17);
        for compiled in sources {
            assert!(!compiled.source.contains("stage_addresses"));
            let module = wgpu::naga::front::wgsl::parse_str(&compiled.source).unwrap();
            let addresses = module
                .global_variables
                .iter()
                .filter(|(_, g)| g.name.as_deref() == Some("addresses"))
                .collect::<Vec<_>>();
            let physical = compiled.source.contains("STRATA_PHYSICAL_EXPERTS");
            assert_eq!(addresses.len(), usize::from(physical));
            let mut bindings = module
                .global_variables
                .iter()
                .filter_map(|(_, g)| g.binding.as_ref().map(|b| b.binding))
                .collect::<Vec<_>>();
            bindings.sort_unstable();
            assert_eq!(bindings, (0..bindings.len() as u32).collect::<Vec<_>>());
        }
    }
}

#[path = "strata_expert_input_probe.rs"]
mod input_probe;
pub(crate) use input_probe::cases as input_numeric_cases;
