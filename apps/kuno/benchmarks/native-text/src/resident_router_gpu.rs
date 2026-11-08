//! Enqueue-only exact count1 ideal-resident router mapper.
//! No router arithmetic, replacement expert, host readback or publication.
//! Caller proves the seven complete owners and original bank lifetimes are
//! physically disjoint on this client's context before lending their handles.
//! Their immutable table contains only actual original expert/slot identities.
//! The same stream must retain owners through deferred all-family acceptance
//! and full erasure; a successful enqueue does not establish numeric acceptance.
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};

#[cfg(test)]
#[path = "count1_hybrid_moe.rs"]
#[allow(dead_code)] // The shared fixture source also defines unused GPU helpers.
#[allow(clippy::duplicate_mod)] // Reuse the private wire oracle without widening it.
mod canonical;
// Only the existing pure Map wire constructor is exercised here. Its source
// also declares unused kernel launch helpers that expect this parent import.
#[cfg(test)]
use crate::resident_util::arg;

pub(crate) const PACKET_WORDS: usize = 8192;
pub(crate) const TABLE_WORDS: usize = 48 * 512;
pub(crate) const MAP_WORDS: usize = 804;
pub(crate) const PROB_WORDS: usize = 80;
pub(crate) const OWNER_WORDS: usize = 244;
pub(crate) const FRAME_WORDS: usize = PACKET_WORDS + MAP_WORDS + PROB_WORDS + OWNER_WORDS + 2;
pub(crate) const AUDIT_WORDS: usize = 48 * FRAME_WORDS;
pub(crate) const PREFLIGHT_WORDS: usize = 524288 - AUDIT_WORDS;
pub(crate) const PREFLIGHT_WORKGROUPS: u32 = PREFLIGHT_WORDS.div_ceil(256) as u32;
pub(crate) const INFO_CALLS_PER_LAYER: usize = 2;
pub(crate) const EXTRA_INFO_CALLS_PER_TOKEN: usize = 48 * (INFO_CALLS_PER_LAYER - 1);
pub(crate) const EXTRA_INFO_BYTES_PER_TOKEN: u64 = EXTRA_INFO_CALLS_PER_TOKEN as u64 * 2048;
// Frozen full-scan launch geometry and the extra conservative INFO envelope.
// The consumer must charge this increment before enabling the paired phase.
const _: () = assert!(PREFLIGHT_WORDS == 76832 && PREFLIGHT_WORKGROUPS == 301);
const _: () = assert!(INFO_CALLS_PER_LAYER == 2 && EXTRA_INFO_BYTES_PER_TOKEN == 98304);
const OWNED_BYTES: [u64; 7] = [
    (PACKET_WORDS * 4) as u64,
    (TABLE_WORDS * 4).next_power_of_two() as u64,
    32768,
    32768,
    32768,
    (AUDIT_WORDS * 4).next_power_of_two() as u64,
    32768,
];

struct Kernel;
struct PreflightKernel;
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "ResidentCount1RouterMapAudit"
    }
    fn id(&self) -> KernelId {
        // Token position and layer are private transient INFO, never keys.
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(64))
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
            source: include_str!("resident_router_gpu.wgsl")
                .replace("4096u", &format!("{}u", crate::qsa_gpu::MAX_CONTEXT)),
            repr: None,
            cube_dim: CubeDim::new_1d(64),
            debug_info: None,
        })
    }
}
impl KernelMetadata for PreflightKernel {
    fn name(&self) -> &'static str {
        "ResidentCount1RouterOwningPreflight"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for PreflightKernel {
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
            source: include_str!("resident_router_gpu_preflight.wgsl")
                .replace("4096u", &format!("{}u", crate::qsa_gpu::MAX_CONTEXT)),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
            debug_info: None,
        })
    }
}

fn validate(owners: &[Handle; 7], layer: usize, start: u64) -> Result<()> {
    if layer >= 48 || start >= crate::qsa_gpu::MAX_CONTEXT as u64 {
        return Err(Error::InvalidFormat("resident router map coordinates"));
    }
    for (owner, bytes) in owners.iter().zip(OWNED_BYTES) {
        if owner.stream != owners[0].stream
            || owner.offset_start.unwrap_or(0) != 0
            || owner.offset_end.unwrap_or(0) != 0
            || owner.size() != bytes
            || owner.size_in_used() != bytes
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    Ok(())
}

/// Order: original packet, immutable expert→slot table, canonical MAP, PROB,
/// OWNER, complete per-token audit, original whole-token fault. All private
/// headers/full padding/current audit frame are zero before this phase.
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 7],
    layer: usize,
    start: u64,
) -> Result<()> {
    validate(&owners, layer, start)?;
    let arguments = || {
        let mut args = KernelArguments::new();
        for owner in &owners {
            args = args.with_buffer(owner.clone().binding());
        }
        args.with_info(MetadataBindingInfo::custom(vec![
            layer as u64 | (start << 32),
            0,
        ]))
    };
    // SAFETY: the same fixed complete owners/coordinates are retained across
    // both ordered dispatches. Parent proves physical alias/client/bank and
    // epoch identity. Complete preflight writes only sticky fault; publication
    // first rejects it and independently validates actual numeric routes.
    unsafe {
        client.launch_unchecked(
            Box::new(PreflightKernel),
            CubeCount::Static(PREFLIGHT_WORKGROUPS, 1, 1),
            arguments(),
        );
        client.launch_unchecked(Box::new(Kernel), CubeCount::Static(1, 1, 1), arguments());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::private_arena::BufferLayout;
    use cubecl::stream_id::StreamId;

    fn owners() -> [Handle; 7] {
        OWNED_BYTES.map(|bytes| Handle::new(StreamId { value: 43 }, bytes))
    }

    #[test]
    fn exact_wire_geometry_and_complete_owner_charges() {
        assert_eq!(MAP_WORDS, crate::quant_matvec::grouped_moe::MAP_WORDS);
        assert_eq!(
            OWNER_WORDS,
            4 + crate::quant_matvec::grouped_moe::MAX_ROUTES * 3
        );
        assert_eq!(FRAME_WORDS, 9322);
        assert_eq!(AUDIT_WORDS, 447456);
        assert_eq!(EXTRA_INFO_CALLS_PER_TOKEN, 48);
        assert_eq!(EXTRA_INFO_BYTES_PER_TOKEN, 98304);
        assert_eq!(BufferLayout::new(TABLE_WORDS).unwrap().owned_bytes, 131072);
        assert_eq!(BufferLayout::new(AUDIT_WORDS).unwrap().owned_bytes, 2097152);
        assert!(validate(&owners(), 0, 0).is_ok());
        assert!(validate(&owners(), 47, 4095).is_ok());
        for (layer, start) in [(48, 0), (usize::MAX, 0), (0, 4096), (0, u64::MAX)] {
            assert!(validate(&owners(), layer, start).is_err());
        }
    }

    #[test]
    fn prelaunch_rejects_partial_owners_foreign_stream_and_extent_changes() {
        for index in 0..7 {
            for bytes in [0, OWNED_BYTES[index] - 4, OWNED_BYTES[index] + 4] {
                let mut h = owners();
                h[index] = Handle::new(StreamId { value: 43 }, bytes);
                assert!(validate(&h, 1, 1).is_err());
            }
            for end in [false, true] {
                let mut h = owners();
                h[index] = if end {
                    h[index].clone().offset_end(4)
                } else {
                    h[index].clone().offset_start(4)
                };
                assert!(validate(&h, 1, 1).is_err());
            }
            let mut h = owners();
            h[index].stream = StreamId { value: 44 };
            assert!(validate(&h, 1, 1).is_err());
        }
    }

    #[test]
    fn canonical_all_gpu_headers_preserve_actual_rank_and_probability_bits() {
        // Independent existing CPU wire constructor establishes the exact ABI
        // including all inactive MAX8 members/owners/probabilities, not only
        // the ten selected slot cells. Deliberately permute ID and slot order.
        let ids = std::array::from_fn(|rank| (rank * 53 + 17) % 512);
        let probabilities = [0.1f32; 10];
        let pairs: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(rank, &id)| (id as u32, ((rank * 17 + 9) % 144) as u32))
            .collect();
        let split = std::array::from_fn(|index| {
            if index % 2 == 0 {
                0
            } else {
                (index / 2) as u32
            }
        });
        let canonical =
            super::canonical::Map::new(&ids, &probabilities, &split, &pairs, 144).unwrap();
        let mut expected = [0u32; MAP_WORDS];
        expected[..4].copy_from_slice(&[1, 0, 1, 10]);
        let mut owner = [0u32; OWNER_WORDS];
        owner[..4].copy_from_slice(&[1, 0, 10, 0]);
        for rank in 0..10 {
            expected[4 + rank] = pairs[rank].1;
            expected[84 + rank] = 1;
            expected[164 + rank * 8] = rank as u32;
            owner[5 + rank * 3] = rank as u32;
        }
        assert_eq!(canonical.words(), &expected);
        assert_eq!(canonical.owner_words(), &owner);
        assert_eq!(&canonical.probabilities()[..10], &probabilities);
        assert!(
            canonical.probabilities()[10..]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
    }

    #[test]
    fn shader_compiles_with_only_dense_active_resources_and_uniform_validation() {
        let source = include_str!("resident_router_gpu.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        let mut bindings = Vec::new();
        for (handle, var) in module.global_variables.iter() {
            if let Some(binding) = &var.binding {
                assert!(!info.get_entry_point(0)[handle].is_empty());
                assert_eq!(binding.group, 0);
                bindings.push(binding.binding);
            }
        }
        bindings.sort_unstable();
        assert_eq!(bindings, (0..8).collect::<Vec<_>>());
        let spirv = wgpu::naga::back::spv::write_vec(
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
        assert!(!spirv.is_empty());
        let validation = source.rfind("if atomicLoad(&bad)!=0u").unwrap();
        for store in [
            "map[index]=",
            "probability[index]=",
            "owner[index]=",
            "audit[frame+index]=",
        ] {
            assert!(source.find(store).unwrap() > validation);
        }
        assert!(source.contains("storageBarrier();"));
        assert!(!source.contains("exp("));
        assert!(!source.contains("select("));
    }

    #[test]
    fn parallel_preflight_preserves_every_full_scan_and_only_writes_sticky_fault() {
        let source = include_str!("resident_router_gpu_preflight.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
        let mut bindings = Vec::new();
        for (handle, var) in module.global_variables.iter() {
            if let Some(binding) = &var.binding {
                assert!(!info.get_entry_point(0)[handle].is_empty());
                assert_eq!(binding.group, 0);
                bindings.push(binding.binding);
                // The preflight has no capability to mutate any consumed wire
                // or source. Only the atomic fault owner has write access.
                if let wgpu::naga::AddressSpace::Storage { access } = var.space {
                    assert_eq!(
                        access.contains(wgpu::naga::StorageAccess::STORE),
                        binding.binding == 6
                    );
                }
            }
        }
        bindings.sort_unstable();
        assert_eq!(bindings, (0..8).collect::<Vec<_>>());
        wgpu::naga::back::spv::write_vec(
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
        let invocations = PREFLIGHT_WORKGROUPS as usize * 256;
        assert!((PREFLIGHT_WORDS..PREFLIGHT_WORDS + 256).contains(&invocations));
        // Fixed domains are independently derived from complete owner layouts.
        // The one global-index partition visits each required word once, with
        // no grid-stride serial scan or dynamically allocated owner.
        let domains = [
            8192,
            OWNED_BYTES[1] as usize / 4,
            FRAME_WORDS,
            OWNED_BYTES[5] as usize / 4 - AUDIT_WORDS,
        ];
        for words in domains {
            assert!((0..invocations).filter(|&i| i < words).eq(0..words));
        }
        for fragment in [
            "index<8192u",
            "index<32768u",
            "index<9322u",
            "index<76832u",
            "audit[447456u+index]",
            "atomicOr(&fault[0],4u)",
        ] {
            assert!(source.contains(fragment));
        }
        let publication = include_str!("resident_router_gpu.wgsl");
        assert!(publication.contains("atomicLoad(&fault[0])!=0u {reject();}"));
        assert!(!publication.contains("for(var index=local;index<32768u"));
        assert!(!publication.contains("for(var index=447456u+local"));
    }
}
