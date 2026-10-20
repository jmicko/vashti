//! MIT Strata Intel prefill.cpp's native FP16 expert arithmetic, connected to
//! original global Sources, the real router map and the parent private owners.
//! An eight-slot transfer ring and paired FP16 products are connected. Full
//! chunk/queue scheduling remains part of the complete prefill port.
use super::*;
use crate::{quant_matvec as q, strata_global_sources::Sources};
use kuno_text::model::BoundGraph;
#[path = "strata_prompt_expert_numeric_probe.rs"]
mod public_numeric;
pub(crate) use public_numeric::cases as public_dequant_numeric;

#[cfg(all(test, feature = "strata-native-expert-bulk"))]
#[path = "strata_expert_bulk_budget_tests.rs"]
mod bulk_budget_tests;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Op {
    Gather,
    Dequant { dtype: u32, down: bool },
    Gemm,
    Activate,
}
impl Op {
    fn lanes(self) -> u32 {
        if self == Self::Gemm { 64 } else { 256 }
    }
    fn source(self) -> String {
        let all = include_str!("strata_prompt_experts.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut s = groups::shader_prefix() + section("COMMON");
        if let Self::Dequant { dtype, down } = self {
            let helpers = include_str!("iq_moe_integer.wgsl")
                .split_once("fn iq4_byte")
                .unwrap()
                .1
                .split_once("// Original byte-addressed")
                .unwrap()
                .0;
            s.push_str(&format!("fn iq4_byte{helpers}"));
            let (decode, value) = if down {
                (q::frozen_down_integer::decode(dtype).replace("down","gate"),"let e=gate_prepare(row*p.row_bytes,col/32u);let first=(col%32u)/4u;return decoded8(e.d,gate_packed(e,first),gate_packed(e,first+1u));".to_owned())
            } else {
                let decode = format!(
                    "{}\n{}",
                    q::frozen_gu_integer::decode("gate", dtype),
                    q::frozen_gu_integer::decode("up", dtype)
                );
                let value = "let first=(col%32u)/4u;if (row%2u)==0u {let e=gate_prepare((row/2u)*p.row_bytes,col/32u);return decoded8(gate_factor(e,(col%32u)/16u),gate_packed(e,first),gate_packed(e,first+1u));}let e=up_prepare((row/2u)*p.row_bytes,col/32u);return decoded8(up_factor(e,(col%32u)/16u),up_packed(e,first),up_packed(e,first+1u));";
                (decode, value.into())
            };
            let offset = match dtype {
                q::IQ3_XXS => 0,
                q::IQ3_S => 256,
                q::IQ2_S => 768,
                _ => 0,
            };
            let decode = decode.replace("iq3_grid[", &format!("strata_grid[{offset}u+"));
            let decode = crate::strata_expert_fused::replace_reads(decode, "gate");
            s.push_str(&crate::strata_expert_fused::replace_reads(decode, "up"));
            s.push_str("struct Decoded8 {a:vec4<f32>,b:vec4<f32>}\nfn codes4(c:u32)->vec4<f32> {return vec4<f32>(signed_byte(c&255u),signed_byte((c>>8u)&255u),signed_byte((c>>16u)&255u),signed_byte(c>>24u));}\nfn decoded8(d:f32,a:u32,b:u32)->Decoded8 {return Decoded8(vec4<f32>(d)*codes4(a),vec4<f32>(d)*codes4(b));}\n");
            s.push_str(&format!(
                "fn value8(row:u32,col:u32)->Decoded8 {{{value}}}\n"
            ));
            let mut bindings = String::from(
                "// STRATA_PHYSICAL_EXPERTS\n@group(0) @binding(0) var<storage,read> gate0:array<vec4<u32>>;\n",
            );
            let mut index = 1;
            if !down {
                bindings
                    .push_str("@group(0) @binding(1) var<storage,read> up0:array<vec4<u32>>;\n");
                index += 1;
            }
            for (name, access, ty) in [
                ("map", "read", "u32"),
                ("raw", "read", "u32"),
                ("packed", "read_write", "vec4<u32>"),
                ("fault", "read_write", "atomic<u32>"),
                ("strata_grid", "read", "u32"),
                ("addresses", "read", "vec4<u32>"),
            ] {
                bindings.push_str(&format!(
                    "@group(0) @binding({index}) var<storage,{access}> {name}:array<{ty}>;\n"
                ));
                index += 1;
            }
            bindings.push_str(&format!("const DOWN_IMAGE:bool={down};\n@group(0) @binding({index}) var<uniform> p:Params;\nvar<private> strata_gate_address:vec2<u32>;\nvar<private> host_gate:bool;\n"));
            if !down {
                bindings.push_str("var<private> strata_up_address:vec2<u32>;\nvar<private> host_up:bool;\nfn strata_up(index:u32)->u32 {if host_up {return raw[p.r0+p.r1+index];}return up0[index/4u][index%4u];}\n");
            }
            let choose_up = if down {
                ""
            } else {
                "let up=addresses[512u+map[4u+p.group]];if up.z!=src.z||up.w>1u {atomicOr(&fault[0],128u);return;}strata_up_address=up.xy;host_up=up.w==1u;"
            };
            let code = section("DEQUANT")
                .replace("BINDINGS", &bindings)
                .replace("CHOOSE_UP", choose_up);
            s.push_str(&code);
            return s;
        }
        s.push_str(section(match self {
            Self::Gather => "GATHER",
            Self::Dequant { .. } => "DEQUANT",
            Self::Gemm => "GEMM",
            Self::Activate => "ACTIVATE",
        }));
        s
    }
}
impl KernelMetadata for Op {
    fn native_release_after(&self) -> bool {
        matches!(self, Self::Dequant { down: true, .. })
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Gather => "StrataPromptExpertGather",
            Self::Dequant { .. } => "StrataPromptExpertF16Image",
            Self::Gemm => "StrataPromptExpertGemm",
            Self::Activate => "StrataPromptExpertF16Swiglu",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((*self, "prompt-owned-routes-v4-vector8"))
            .cube_dim(CubeDim::new_1d(self.lanes()))
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
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(self.lanes()),
            debug_info: None,
        })
    }
}
fn dispatch(
    client: &ComputeClient<WgpuRuntime>,
    op: Op,
    grid: [usize; 3],
    mut args: KernelArguments,
    words: [u32; 16],
) -> Result<()> {
    if grid.contains(&0) || grid.iter().any(|&v| v > 65535) || !active() {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    args = args.with_info(MetadataBindingInfo::custom(
        words
            .chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    // SAFETY: parent authenticates the original geometry and distinct private
    // owners. GPU map guards select real rows; stages are reused in queue order.
    unsafe {
        client.launch_unchecked(
            Box::new(op),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}

const RAW_SLOTS: usize = 8;
const DEQUANT_SLOTS: usize = 2;
const GU_WORDS: usize = 1280 * 2560 / 2;
const DOWN_WORDS: usize = 2560 * 640 / 2;
const DQ_WORDS: usize = GU_WORDS + DOWN_WORDS;

/// Actual descriptor view of one raw transfer slot. All source role spans are
/// original-format bytes; no scheduler-only narrowed declaration is accepted.
fn raw_slot(root: &Handle, slot: usize, roles: [usize; 3]) -> Result<Handle> {
    let root_bytes = allocation::whole_root_bytes(root)?;
    let bytes = roles
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(n))
        .ok_or(Error::ArithmeticOverflow)?;
    if slot >= RAW_SLOTS
        || roles[0] != roles[1]
        || roles.iter().any(|&n| n == 0 || !n.is_multiple_of(256))
        || bytes
            .checked_mul(RAW_SLOTS)
            .is_none_or(|n| n as u64 > allocation::RAW_RING_BYTES)
        || root.offset_start.unwrap_or(0) != 0
        || root.offset_end.unwrap_or(0) != 0
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let first = slot * bytes;
    let end = first + bytes;
    Ok(root
        .clone()
        .offset_start(first as u64)
        .offset_end(root_bytes - end as u64))
}

/// Bound only the execution grid. The descriptors and shader Params retain
/// the complete private prompt/map extents; no ownership check is narrowed.
fn member_dispatches(members: usize, count: usize) -> Result<[usize; 2]> {
    if !(1..=groups::ROUTE_CAPACITY).contains(&count) || !(1..=count).contains(&members) {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    Ok([
        (members * 1280).div_ceil(256),
        (members * 320).div_ceil(256),
    ])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue(
    client: &ComputeClient<WgpuRuntime>,
    graph: BoundGraph<'_, '_>,
    sources: &Sources,
    layer: usize,
    geometry: q::grouped_moe::Geometry,
    owners: &[Handle; 7],
    fault: &Handle,
    count: usize,
    map: &[u32],
) -> Result<()> {
    #[cfg(feature = "strata-native-expert-bulk")]
    {
        // SAFETY: the unchanged inner path performs only the original six
        // expert operators and checked Sources HOST copies using this client.
        // Original parent owners/Scope enclose capture, enqueue and completion.
        let outcome = unsafe {
            client.with_native_strata_expert_batch(|batch| {
                enqueue_original(
                    batch, graph, sources, layer, geometry, owners, fault, count, map,
                )
            })
        }
        .map_err(crate::resident_util::fault)?;
        match outcome {
            Ok(((), receipt)) => {
                // Existing public-only profile sink; ordinary private workers
                // retain no request usage counters or extra diagnostics.
                crate::strata_prefill_profile::count("expert_native_bulk_batches", 1);
                crate::strata_prefill_profile::count(
                    "expert_native_bulk_launches",
                    receipt.launches as u64,
                );
                crate::strata_prefill_profile::count(
                    "expert_native_bulk_copy_segments",
                    receipt.copy_segments as u64,
                );
                for (name, value) in [
                    ("expert_native_bulk_logical_phases", receipt.logical_phases),
                    ("expert_native_bulk_matrix_pieces", receipt.matrix_pieces),
                    ("expert_native_bulk_wide_prefixes", receipt.wide_prefixes),
                    ("expert_native_bulk_wide_blocks", receipt.wide_blocks),
                    ("expert_native_bulk_tail_pieces", receipt.tail_pieces),
                ] {
                    crate::strata_prefill_profile::count(name, value as u64);
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
    #[cfg(not(feature = "strata-native-expert-bulk"))]
    enqueue_original(
        client, graph, sources, layer, geometry, owners, fault, count, map,
    )
}

#[allow(clippy::too_many_arguments)]
fn enqueue_original(
    client: &ComputeClient<WgpuRuntime>,
    graph: BoundGraph<'_, '_>,
    sources: &Sources,
    layer: usize,
    geometry: q::grouped_moe::Geometry,
    owners: &[Handle; 7],
    fault: &Handle,
    count: usize,
    map: &[u32],
) -> Result<()> {
    sources.validate_prompt_ready(graph)?;
    let stage = stages()?;
    if !(1..=super::groups::ROUTE_CAPACITY).contains(&count)
        || fault.size_in_used() != 4
        || owners
            .iter()
            .chain(stage.iter())
            .chain([fault])
            .any(|h| h.stream != owners[0].stream)
        || owners[0].size_in_used() != (groups::MAP_WORDS * 4) as u64
        || owners[1].size_in_used() < (count * 2560 * 4) as u64
        || owners[2].size_in_used() != groups::GU_BYTES
        || owners[3].size_in_used() != groups::PART_BYTES
        || owners[6].size_in_used() != (groups::STATUS_WORDS * 4) as u64
        || (DEQUANT_SLOTS * DQ_WORDS * 4) as u64 > stage[1].size_in_used()
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let mixed_image = super::mixed_input_image(&owners[1], count, 0, 2560, false);
    let gather_input = mixed_image.as_ref().unwrap_or(&owners[1]);
    let projections = geometry.projections();
    for (role, p) in projections.iter().enumerate() {
        if p.experts() != 512
            || p.columns() != if role == 2 { 640 } else { 2560 }
            || p.rows() != if role == 2 { 2560 } else { 640 }
            || !matches!(
                (role, p.dtype()),
                (0 | 1, q::IQ2_S | q::IQ3_S | q::IQ3_XXS | q::IQ4_XS) | (2, q::IQ4_NL | q::Q2_0)
            )
        {
            return Err(crate::strata_prompt::invalid_buffer_length());
        }
    }
    if projections[0].dtype() != projections[1].dtype() {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let role_bytes = projections.map(|p| p.expert_bytes());
    let raw_stride = role_bytes.iter().sum::<usize>();
    if (RAW_SLOTS * raw_stride) as u64 > allocation::RAW_RING_BYTES {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    // Bind current immutable PUBLIC prototypes once for this exact layer and
    // checked Sources borrow. Actual per-slot source/copy and per-kernel full
    // dependency/loan gates remain in the original Sources methods.
    let layer_sources = sources.prompt_layer(graph, layer)?;
    let mut raw_slots = Vec::with_capacity(RAW_SLOTS);
    for slot in 0..RAW_SLOTS {
        raw_slots.push(raw_slot(&stage[2], slot, role_bytes)?);
    }
    // Native Intel prefill groups on the host once per layer. This is a bounded
    // routing-only readback, not decode's CPU expert/input contribution.
    let groups = super::groups::validate(map, count)?;
    crate::strata_prefill_profile::expert_member_geometry(map, groups);
    crate::strata_prefill_profile::count("expert_group_count", groups as u64);
    crate::strata_prefill_profile::count("expert_fp16_image_bytes", (groups * DQ_WORDS * 4) as u64);
    let mut issued = 0;
    for group in 0..groups {
        let mut timing = crate::strata_prefill_profile::CpuTimer::new();
        while issued < groups && issued < group + RAW_SLOTS {
            layer_sources.stage_group_on(
                client,
                map[4 + issued] as usize,
                role_bytes,
                &stage[2],
                (issued % RAW_SLOTS) * raw_stride,
            )?;
            issued += 1;
        }
        timing.mark("expert_cpu_stage_sources");
        // The route map was fully validated above. Each expert consumes only
        // its routed members, not every token in the prompt. Keep p.count as
        // the full prompt geometry for all shader ownership/route checks.
        let members = map[516 + group] as usize;
        let [gather_groups, activate_groups] = member_dispatches(members, count)?;
        let dq = (group % DEQUANT_SLOTS) * DQ_WORDS;
        let base = [
            count as u32,
            2560,
            1280,
            projections[0].row_bytes() as u32,
            group as u32,
            0,
            0, // The actual raw descriptor starts at this slot, not the root.
            (role_bytes[0] / 4) as u32,
            ((role_bytes[0] + role_bytes[1]) / 4) as u32,
            dq as u32,
            (dq + GU_WORDS) as u32,
            (raw_stride / 4) as u32,
            mixed_image.is_some() as u32,
            0,
            0,
            0,
        ];
        let args = KernelArguments::new().with_buffers(vec![
            owners[0].clone().binding(),
            gather_input.clone().binding(),
            stage[0].clone().binding(),
            fault.clone().binding(),
        ]);
        dispatch(client, Op::Gather, [gather_groups, 1, 1], args, base)?;
        timing.mark("expert_cpu_gather");
        // Both FP16 images are queued before their coupled expert products.
        for down in [false, true] {
            let mut words = base;
            let p = projections[if down { 2 } else { 0 }];
            if down {
                words[1] = 640;
                words[2] = 2560;
                words[3] = p.row_bytes() as u32;
                words[5] = 2;
            }
            let mut args = KernelArguments::new()
                .with_buffer(layer_sources.prototype(if down { 2 } else { 0 })?.binding());
            if !down {
                args = args.with_buffer(layer_sources.prototype(1)?.binding());
            }
            args = args.with_buffers(vec![
                owners[0].clone().binding(),
                raw_slots[group % RAW_SLOTS].clone().binding(),
                stage[1].clone().binding(),
                fault.clone().binding(),
            ]);
            args = crate::strata_intel_core::bind(args, owners[0].stream)?;
            args = layer_sources.bind_image(args)?;
            dispatch(
                client,
                Op::Dequant {
                    dtype: p.dtype(),
                    down,
                },
                [
                    (words[1] as usize * words[2] as usize / 8).div_ceil(256),
                    1,
                    1,
                ],
                args,
                words,
            )?;
        }
        timing.mark("expert_cpu_dequant");
        crate::strata_matrix::expert(
            client,
            [
                stage[0].clone(),
                stage[1].clone(),
                owners[2].clone(),
                owners[0].clone(),
                owners[6].clone(),
                fault.clone(),
            ],
            map[516 + group] as usize,
            2560,
            1280,
            dq * 2,
            group,
            false,
        )?;
        timing.mark("expert_cpu_gu_product");
        let args = KernelArguments::new().with_buffers(vec![
            owners[0].clone().binding(),
            owners[2].clone().binding(),
            stage[0].clone().binding(),
            owners[6].clone().binding(),
            fault.clone().binding(),
        ]);
        dispatch(client, Op::Activate, [activate_groups, 1, 1], args, base)?;
        timing.mark("expert_cpu_activate");
        crate::strata_matrix::expert(
            client,
            [
                stage[0].clone(),
                stage[1].clone(),
                owners[3].clone(),
                owners[0].clone(),
                owners[6].clone(),
                fault.clone(),
            ],
            map[516 + group] as usize,
            640,
            2560,
            (dq + GU_WORDS) * 2,
            group,
            true,
        )?;
        timing.mark("expert_cpu_down_product");
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Op::Gather,
        Op::Dequant {
            dtype: q::IQ2_S,
            down: false,
        },
        Op::Dequant {
            dtype: q::IQ3_S,
            down: false,
        },
        Op::Dequant {
            dtype: q::IQ3_XXS,
            down: false,
        },
        Op::Dequant {
            dtype: q::IQ4_XS,
            down: false,
        },
        Op::Dequant {
            dtype: q::IQ4_NL,
            down: true,
        },
        Op::Dequant {
            dtype: q::Q2_0,
            down: true,
        },
        Op::Gemm,
        Op::Activate,
    ]
    .into_iter()
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
mod bulk_metadata_tests {
    use super::*;
    use cubecl_runtime::{
        kernel::{CubeKernel, KernelDefinition, KernelTask},
        server::native_expert_batch::native_strata_expert_operator,
    };

    fn boxed(op: Op) -> Box<dyn CubeTask<AutoCompiler>> {
        Box::new(op)
    }

    #[test]
    fn original_boxed_producer_metadata_passes_the_strict_six_operator_gate() {
        for gu in [q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS] {
            for down in [q::IQ4_NL, q::Q2_0] {
                for matrix in crate::strata_matrix::expert_metadata_test_names() {
                    // These are the actual custom producers, behind the same
                    // Box<dyn CubeTask> boundary consumed by WgpuServer.
                    let gather = boxed(Op::Gather);
                    let gu = boxed(Op::Dequant { dtype: gu, down: false });
                    let down = boxed(Op::Dequant { dtype: down, down: true });
                    let activate = boxed(Op::Activate);
                    let sequence = [
                        (gather.name(), gather.native_release_after()),
                        (gu.name(), gu.native_release_after()),
                        (down.name(), down.native_release_after()),
                        (matrix, false),
                        (activate.name(), activate.native_release_after()),
                        (matrix, false),
                    ];
                    assert_eq!(sequence.map(|(_, release)| release), [false, false, true, false, false, false]);
                    for (index, (name, release)) in sequence.into_iter().enumerate() {
                        assert!(native_strata_expert_operator(name, release, index));
                        assert!(native_strata_expert_operator(name, release, index + 6));
                        assert!(!native_strata_expert_operator(name, !release, index));
                        assert!(!native_strata_expert_operator("StrataPromptExpertGemm", release, index));
                        let wrong = sequence[(index + 1) % 6];
                        assert!(!native_strata_expert_operator(wrong.0, wrong.1, index));
                    }
                    let id = down.id().native_release_boundary(down.native_release_after());
                    assert_eq!(id, down.as_ref().id().native_release_boundary(true));
                    assert_ne!(id, down.id().native_release_boundary(false));
                }
            }
        }
    }

    /// Exercise the real KernelTask adapter too, with original Op metadata.
    /// Defining/compiling a GPU kernel is intentionally outside this CPU test.
    struct SourceMetadata(Op);
    impl KernelMetadata for SourceMetadata {
        fn name(&self) -> &'static str { self.0.name() }
        fn native_release_after(&self) -> bool { self.0.native_release_after() }
        fn id(&self) -> KernelId { self.0.id() }
        fn address_type(&self) -> StorageType { self.0.address_type() }
    }
    impl CubeKernel for SourceMetadata {
        fn define(&self) -> KernelDefinition {
            panic!("metadata forwarding must not compile a kernel")
        }
    }
    #[test]
    fn real_kernel_task_and_box_preserve_original_release_identity() {
        for op in [
            Op::Gather,
            Op::Dequant { dtype: q::IQ3_S, down: false },
            Op::Dequant { dtype: q::IQ4_NL, down: true },
            Op::Activate,
        ] {
            let task = KernelTask::<AutoCompiler, _>::new(SourceMetadata(op));
            assert_eq!(task.name(), op.name());
            assert_eq!(task.native_release_after(), op.native_release_after());
            let boxed: Box<dyn CubeTask<AutoCompiler>> = Box::new(task);
            assert_eq!(boxed.name(), op.name());
            assert_eq!(boxed.native_release_after(), op.native_release_after());
            assert_eq!(boxed.id().native_release_boundary(boxed.native_release_after()),
                op.id().native_release_boundary(op.native_release_after()));
        }
    }
}

#[cfg(test)]
mod vector_dequant_tests {
    use super::*;
    use kuno_text::quant::TensorType;

    #[test]
    fn eight_value_images_preserve_original_half_bits_and_subscale_boundaries() {
        for (kind, columns) in [
            (TensorType::Iq3Xxs, 2560usize),
            (TensorType::Iq3S, 2560),
            (TensorType::Iq2S, 2560),
            (TensorType::Iq4Xs, 2560),
            (TensorType::Iq4Nl, 640),
            (TensorType::Q2_0, 640),
        ] {
            for salt in [0u8, 37, 113, 255] {
                let mut wire: Vec<u8> = (0..kind.row_bytes(columns as u64).unwrap() as usize)
                    .map(|i| (i as u8).wrapping_mul(73).wrapping_add(salt))
                    .collect();
                for block in wire.chunks_exact_mut(kind.block_bytes() as usize) {
                    block[..2].copy_from_slice(&half::f16::from_f32(0.015625).to_le_bytes());
                }
                let mut reference = vec![0.; columns];
                kind.decode(&wire, &mut reference).unwrap();
                let mut packed = vec![0u32; columns / 2];
                for column in (0..columns).step_by(8) {
                    // An eight-value run must never cross the original32-code
                    // record or either IQ2_S16-value scale. The shader shares
                    // one prepared record and two adjacent packed quartets.
                    assert_eq!(column / 32, (column + 7) / 32);
                    assert_eq!(column / 16, (column + 7) / 16);
                    let (codes, d): (Vec<i8>, f32) = if kind == TensorType::Iq2S {
                        let (codes, d) =
                            q::iq_moe_integer::oracle::unpack_iq2_s16(&wire, columns, column / 16)
                                .unwrap();
                        (codes.to_vec(), d)
                    } else {
                        let (codes, d) =
                            q::iq_moe_integer::oracle::unpack32(kind, &wire, columns, column / 32)
                                .unwrap();
                        (codes.to_vec(), d)
                    };
                    let local = column % codes.len();
                    for pair in 0..4 {
                        let first = half::f16::from_f32(d * f32::from(codes[local + pair * 2]));
                        let second =
                            half::f16::from_f32(d * f32::from(codes[local + pair * 2 + 1]));
                        packed[column / 2 + pair] =
                            u32::from(first.to_bits()) | u32::from(second.to_bits()) << 16;
                    }
                }
                for (column, expected) in reference.into_iter().enumerate() {
                    assert_eq!(
                        (packed[column / 2] >> (16 * (column % 2))) as u16,
                        half::f16::from_f32(expected).to_bits(),
                        "kind={kind:?} salt={salt} column={column}"
                    );
                }
            }
        }
    }

    #[test]
    fn actual_dequant_sources_store_original_aligned_vectors_with_full_fault_checks() {
        for (down, types) in [
            (false, &[q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS][..]),
            (true, &[q::IQ4_NL, q::Q2_0][..]),
        ] {
            for &dtype in types {
                let source = Op::Dequant { dtype, down }.source();
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                let variable = module
                    .global_variables
                    .iter()
                    .find(|(_, variable)| variable.name.as_deref() == Some("packed"))
                    .unwrap()
                    .1;
                assert!(matches!(
                    module.types[variable.ty].inner,
                    wgpu::naga::TypeInner::Array { stride: 16, .. }
                ));
                assert!(source.contains("fn value8("));
                assert!(!source.contains("fn value("));
                assert!(source.contains("packed[first+i]=select(vec4<u32>(0u),bits,good)"));
                assert!(source.contains("all(abs(values.a)<=maximum)"));
                assert!(source.contains("all(abs(hi)<=maximum)"));
                assert!(source.contains("atomicOr(&fault[0],128u)"));
            }
        }
        for slot in 0..DEQUANT_SLOTS {
            for (first, words) in [
                (slot * DQ_WORDS, GU_WORDS),
                (slot * DQ_WORDS + GU_WORDS, DOWN_WORDS),
            ] {
                assert!(first.is_multiple_of(4) && words.is_multiple_of(4));
                assert!(first + words <= DEQUANT_SLOTS * DQ_WORDS);
            }
        }
    }
}

#[cfg(test)]
mod copy_contract_tests {
    use cubecl_common::stream_id::StreamId;
    use cubecl_runtime::public_external::{
        PublicExternalBudget, PublicExternalHostSource, PublicExternalPool, PublicHostCopy,
        validate_host_copies,
    };
    use cubecl_runtime::server::Handle;

    #[test]
    fn staging_contract_rejects_private_sources_and_unowned_views() {
        // Metadata fixtures only: no device, buffers or content.
        // The backend must independently resolve the live registry entry.
        let stream = StreamId { value: 71 };
        let mut pool = PublicExternalPool::new(
            PublicExternalBudget {
                backing_bytes: 32768,
                buffers: 1,
                metadata_bytes: 131072,
            },
            stream,
        )
        .unwrap();
        // Abstract metadata resource only, never sent to a GPU backend. Its
        // device/initialization attestations are deliberately fixture values.
        let original = unsafe {
            PublicExternalHostSource::from_initialized_public_host(0u8, 32768, 32768, 2, 0, [1; 32])
        }
        .unwrap();
        let source = pool.register(original).unwrap().handle();
        let destination = Handle::new(stream, 32768);
        let make = || PublicHostCopy {
            source: source.clone().binding(),
            source_first: 64,
            destination: destination.clone().binding(),
            destination_first: 0,
            bytes: 64,
        };
        assert!(validate_host_copies(&[make()], stream).is_ok());
        let mut private = make();
        private.source = destination.clone().binding();
        assert!(validate_host_copies(&[private], stream).is_err());
        let mut readonly = make();
        readonly.destination = source.clone().binding();
        assert!(validate_host_copies(&[readonly], stream).is_err());
        let mut other_stream = make();
        other_stream.destination.stream = StreamId { value: 72 };
        assert!(validate_host_copies(&[other_stream], stream).is_err());
        let mut sliced = make();
        sliced.source.offset_start = Some(4);
        assert!(validate_host_copies(&[sliced], stream).is_err());
        let mut outside = make();
        outside.source_first = u64::MAX - 3;
        assert!(validate_host_copies(&[outside], stream).is_err());
        let mut outside = make();
        outside.destination_first = 32768;
        assert!(validate_host_copies(&[outside], stream).is_err());
    }
}

#[cfg(test)]
mod raw_slot_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;

    #[test]
    fn raw_slot_uses_actual_whole_parent_end_for_non_power_of_two_backing() {
        let size = allocation::ROOT_BYTES - 256;
        let root = Handle::new(StreamId { value: 92 }, size);
        let roles = [1024, 1024, 2048];
        for slot in 0..RAW_SLOTS {
            let view = raw_slot(&root, slot, roles).unwrap();
            assert_eq!(view.size(), size);
            assert_eq!(view.offset_start.unwrap(), (slot * 4096) as u64);
            assert_eq!(view.offset_end.unwrap(), size - ((slot + 1) * 4096) as u64);
            assert_eq!(view.size_in_used(), 4096);
        }
        assert!(raw_slot(&root.offset_end(256), 0, roles).is_err());
    }
    #[test]
    fn exact_views_cover_original_gu_up_down_for_every_supported_quant_pair() {
        let root = Handle::new(StreamId { value: 91 }, allocation::RAW_RING_BYTES);
        let mut old_guard_would_exceed_exact_slot = false;
        for gu in [q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS] {
            for down in [q::IQ4_NL, q::Q2_0] {
                let (gv, gb) = q::block_layout(gu).unwrap();
                let (dv, db) = q::block_layout(down).unwrap();
                let gu_bytes = 640 * (2560 / gv) * gb;
                let down_bytes = 2560 * (640 / dv) * db;
                let total = gu_bytes * 2 + down_bytes;
                old_guard_would_exceed_exact_slot |= gu_bytes > down_bytes;
                let mut prior_end = 0;
                for slot in 0..RAW_SLOTS {
                    let view = raw_slot(&root, slot, [gu_bytes, gu_bytes, down_bytes]).unwrap();
                    let first = view.offset_start.unwrap();
                    let end = allocation::RAW_RING_BYTES - view.offset_end.unwrap();
                    assert_eq!(first, prior_end);
                    assert_eq!(view.size_in_used(), total as u64);
                    assert!(end <= allocation::RAW_RING_BYTES);
                    assert!(first.is_multiple_of(256));
                    for (offset, bytes) in [
                        (0, gu_bytes),
                        (gu_bytes, gu_bytes),
                        (gu_bytes * 2, down_bytes),
                    ] {
                        assert!(first + offset as u64 + bytes as u64 <= end);
                    }
                    prior_end = end;
                }
            }
        }
        assert!(old_guard_would_exceed_exact_slot);
    }
    #[test]
    fn release_cache_identity_preserves_different_original_quant_formats() {
        let iq4 = Op::Dequant {
            dtype: q::IQ4_NL,
            down: true,
        };
        let q2 = Op::Dequant {
            dtype: q::Q2_0,
            down: true,
        };
        assert!(iq4.native_release_after() && q2.native_release_after());
        let a = iq4
            .id()
            .native_release_boundary(true)
            .public_external_read_mask(3);
        let b = q2
            .id()
            .native_release_boundary(true)
            .public_external_read_mask(3);
        assert_ne!(a.stable_hash(), b.stable_hash());
        assert!(a != b);
        let ordinary = Op::Gather.id();
        assert_eq!(
            ordinary.stable_hash(),
            ordinary
                .clone()
                .native_release_boundary(false)
                .stable_hash()
        );
        assert!(ordinary.clone().native_release_boundary(false) == ordinary);
    }
    #[test]
    fn slot_descriptors_reject_partial_roots_bad_geometry_and_raw_ring_overflow() {
        let root = Handle::new(StreamId { value: 91 }, allocation::RAW_RING_BYTES);
        let roles = [1024, 1024, 2048];
        assert!(raw_slot(&root, RAW_SLOTS, roles).is_err());
        assert!(raw_slot(&root.clone().offset_end(256), 0, roles).is_err());
        assert!(raw_slot(&root, 0, [1024, 512, 2048]).is_err());
        assert!(raw_slot(&root, 0, [1025, 1025, 2048]).is_err());
        assert!(raw_slot(&root, 0, [0, 0, 2048]).is_err());
        assert!(raw_slot(&root, 0, [4 * 1024 * 1024; 3]).is_err());
    }
}

#[cfg(test)]
mod member_dispatch_tests {
    use super::*;
    #[test]
    fn routed_member_grids_cover_exact_work_and_preserve_full_prompt_bounds() {
        for count in [1usize, 35, 256, 2048, 4096] {
            for members in [1, count.div_ceil(2), count] {
                let [gather, activate] = member_dispatches(members, count).unwrap();
                for (grid, words) in [(gather, members * 1280), (activate, members * 320)] {
                    assert!(grid * 256 >= words && (grid - 1) * 256 < words);
                }
            }
        }
        assert_eq!(member_dispatches(1, 4096).unwrap(), [5, 2]);
        assert_eq!(member_dispatches(4096, 4096).unwrap(), [20480, 5120]);
        for (members, count) in [(0, 1), (2, 1), (1, 0), (4097, 4096), (1, 4097)] {
            assert!(member_dispatches(members, count).is_err());
        }
    }
}
