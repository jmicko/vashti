//! MIT Strata Intel prefill.cpp's native FP16 expert arithmetic, connected to
//! original global Sources, the real router map and the parent private owners.
//! An eight-slot transfer ring and paired FP16 products are connected. Full
//! chunk/queue scheduling remains part of the complete prefill port.
use super::*;
use crate::{quant_matvec as q, strata_global_sources::Sources};
use kuno_text::model::BoundGraph;

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
        let mut s = section("COMMON").to_owned();
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
                (q::frozen_down_integer::decode(dtype).replace("down","gate"),"let e=gate_prepare(row*p.row_bytes,col/32u);let codes=gate_packed(e,(col%32u)/4u);return e.d*signed_byte((codes>>((col%4u)*8u))&255u);".to_owned())
            } else {
                let decode = format!(
                    "{}\n{}",
                    q::frozen_gu_integer::decode("gate", dtype),
                    q::frozen_gu_integer::decode("up", dtype)
                );
                let value = "if (row%2u)==0u {let e=gate_prepare((row/2u)*p.row_bytes,col/32u);let codes=gate_packed(e,(col%32u)/4u);return gate_factor(e,(col%32u)/16u)*signed_byte((codes>>((col%4u)*8u))&255u);}let e=up_prepare((row/2u)*p.row_bytes,col/32u);let codes=up_packed(e,(col%32u)/4u);return up_factor(e,(col%32u)/16u)*signed_byte((codes>>((col%4u)*8u))&255u);";
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
            s.push_str(&format!("fn value(row:u32,col:u32)->f32 {{{value}}}\n"));
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
                ("packed", "read_write", "u32"),
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
            .info((*self, "prompt-raw-slot-v2"))
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
        return Err(Error::InvalidBufferLength);
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
    let bytes = roles
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(n))
        .ok_or(Error::ArithmeticOverflow)?;
    if slot >= RAW_SLOTS
        || roles[0] != roles[1]
        || roles.iter().any(|&n| n == 0 || !n.is_multiple_of(256))
        || bytes
            .checked_mul(RAW_SLOTS)
            .is_none_or(|n| n > 64 * 1024 * 1024)
        || root.size_in_used() != allocation::ROOT_BYTES
        || root.offset_start.unwrap_or(0) != 0
        || root.offset_end.unwrap_or(0) != 0
    {
        return Err(Error::InvalidBufferLength);
    }
    let first = slot * bytes;
    let end = first + bytes;
    Ok(root
        .clone()
        .offset_start(first as u64)
        .offset_end(allocation::ROOT_BYTES - end as u64))
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
    sources.validate_graph_ready(graph)?;
    let stage = stages()?;
    if !(1..=super::groups::CAPACITY).contains(&count)
        || fault.size_in_used() != 4
        || owners
            .iter()
            .chain(stage.iter())
            .chain([fault])
            .any(|h| h.stream != owners[0].stream)
        || owners[0].size_in_used() != 32768
        || owners[1].size_in_used() < (count * 2560 * 4) as u64
        || owners[2].size_in_used() != 2 * 1024 * 1024
        || owners[3].size_in_used() != 32 * 1024 * 1024
        || owners[6].size_in_used() != 32768
        || (DEQUANT_SLOTS * DQ_WORDS * 4) as u64 > stage[1].size_in_used()
    {
        return Err(Error::InvalidBufferLength);
    }
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
            return Err(Error::InvalidBufferLength);
        }
    }
    if projections[0].dtype() != projections[1].dtype() {
        return Err(Error::InvalidBufferLength);
    }
    let role_bytes = projections.map(|p| p.expert_bytes());
    let raw_stride = role_bytes.iter().sum::<usize>();
    if (RAW_SLOTS * raw_stride) as u64 > 64 * 1024 * 1024 {
        return Err(Error::InvalidBufferLength);
    }
    // Native Intel prefill groups on the host once per layer. This is a bounded
    // routing-only readback, not decode's CPU expert/input contribution.
    let groups = super::groups::validate(map, count)?;
    let mut issued = 0;
    for group in 0..groups {
        while issued < groups && issued < group + RAW_SLOTS {
            sources.stage_prompt_group(
                graph,
                layer,
                map[4 + issued] as usize,
                role_bytes,
                &stage[2],
                (issued % RAW_SLOTS) * raw_stride,
            )?;
            issued += 1;
        }
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
            0,
            0,
            0,
            0,
        ];
        let args = KernelArguments::new().with_buffers(vec![
            owners[0].clone().binding(),
            owners[1].clone().binding(),
            stage[0].clone().binding(),
            fault.clone().binding(),
        ]);
        dispatch(
            client,
            Op::Gather,
            [(count * 2560 / 2).div_ceil(256), 1, 1],
            args,
            base,
        )?;
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
            let mut args = KernelArguments::new().with_buffer(
                sources
                    .prototype(graph, layer, if down { 2 } else { 0 })?
                    .binding(),
            );
            if !down {
                args = args.with_buffer(sources.prototype(graph, layer, 1)?.binding());
            }
            args = args.with_buffers(vec![
                owners[0].clone().binding(),
                raw_slot(&stage[2], group % RAW_SLOTS, role_bytes)?.binding(),
                stage[1].clone().binding(),
                fault.clone().binding(),
            ]);
            args = crate::strata_intel_core::bind(args, owners[0].stream)?;
            args = sources.bind_prompt_image(layer, args)?;
            dispatch(
                client,
                Op::Dequant {
                    dtype: p.dtype(),
                    down,
                },
                [
                    (words[1] as usize * words[2] as usize / 2).div_ceil(256),
                    1,
                    1,
                ],
                args,
                words,
            )?;
        }
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
        let args = KernelArguments::new().with_buffers(vec![
            owners[0].clone().binding(),
            owners[2].clone().binding(),
            stage[0].clone().binding(),
            owners[6].clone().binding(),
            fault.clone().binding(),
        ]);
        dispatch(
            client,
            Op::Activate,
            [(count * 640 / 2).div_ceil(256), 1, 1],
            args,
            base,
        )?;
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
    fn exact_views_cover_original_gu_up_down_for_every_supported_quant_pair() {
        let root = Handle::new(StreamId { value: 91 }, allocation::ROOT_BYTES);
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
                    let end = allocation::ROOT_BYTES - view.offset_end.unwrap();
                    assert_eq!(first, prior_end);
                    assert_eq!(view.size_in_used(), total as u64);
                    assert!(end <= 64 * 1024 * 1024);
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
    fn slot_descriptors_reject_partial_roots_bad_geometry_and_upper_half_use() {
        let root = Handle::new(StreamId { value: 91 }, allocation::ROOT_BYTES);
        let roles = [1024, 1024, 2048];
        assert!(raw_slot(&root, RAW_SLOTS, roles).is_err());
        assert!(raw_slot(&root.clone().offset_end(256), 0, roles).is_err());
        assert!(raw_slot(&root, 0, [1024, 512, 2048]).is_err());
        assert!(raw_slot(&root, 0, [1025, 1025, 2048]).is_err());
        assert!(raw_slot(&root, 0, [0, 0, 2048]).is_err());
        assert!(raw_slot(&root, 0, [4 * 1024 * 1024; 3]).is_err());
    }
}
