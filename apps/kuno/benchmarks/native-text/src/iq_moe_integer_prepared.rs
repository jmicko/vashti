//! Once-prepared two-residual-Q8 planes for the explicit integer IQ candidate.
//! The caller lends a complete private owner for one GPU-only phase. All
//! scratch consumers are queued before returning; the original caller then
//! overwrites that complete owner with CPU partials on the same stream. No
//! private allocation, partial-owner padding, new arithmetic or publication.
//! Original packed/activation provenance remains iq_moe_integer-NOTICE.txt.
use super::*;
use std::marker::PhantomData;

pub(crate) const HEADER_WORDS: usize = 32;
pub(crate) const GATE_WORDS: usize = 1440;
pub(crate) const DOWN_WORDS: usize = 360;
pub(crate) const DOWN_FIRST: usize = HEADER_WORDS + GATE_WORDS;
pub(crate) const SCRATCH_WORDS: usize = DOWN_FIRST + 10 * DOWN_WORDS;
pub(crate) const MAX_PROJECT_SHARED_BYTES: usize = 13_952;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PrepareKernel {
    down: bool,
}
impl PrepareKernel {
    fn source(self) -> String {
        let columns = if self.down { 640 } else { 2560 };
        let body = include_str!("iq_moe_integer.wgsl")
            .split_once("// Q8_PREPARE_BEGIN\n")
            .unwrap()
            .1
            .split_once("// Q8_PREPARE_END")
            .unwrap()
            .0;
        include_str!("iq_moe_integer_prepare.wgsl")
            .replace("Q8_PREPARE_VALUE", body)
            .replace("COLUMNS_VALUE", &columns.to_string())
            .replace("PACKED_VALUE", &(columns / 4).to_string())
            .replace("BLOCKS_VALUE", &(columns / 32).to_string())
            .replace("DOWN_VALUE", if self.down { "true" } else { "false" })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProjectKernel(Kernel);

macro_rules! metadata {
    ($ty:ty,$name:literal,$source:expr) => {
        impl KernelMetadata for $ty {
            fn name(&self) -> &'static str {
                $name
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
        impl CubeTask<AutoCompiler> for $ty {
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
                    source: ($source)(self),
                    repr: None,
                    cube_dim: CubeDim::new_1d(128),
                    debug_info: None,
                })
            }
        }
    };
}
metadata!(PrepareKernel, "IQMoePreparePlanes", |k: &PrepareKernel| k
    .source());
metadata!(
    ProjectKernel,
    "IQMoePreparedProjection",
    |k: &ProjectKernel| k.0.source_variant(true)
);

/// All validation happens before the first enqueue. Geometry covers every
/// route/stride; scratch uses an explicit logical interval, never owning tail.
pub(crate) fn validate_layout(
    geometry: g::Geometry,
    spans: &g::Spans,
    owner_words: [usize; 7],
    scratch_words: usize,
    groups: usize,
    active: [bool; 2],
) -> Result<()> {
    validate_geometry(geometry, 0, 72)?;
    validate_geometry(geometry, 72, 72)?;
    spans
        .validate(geometry, owner_words)
        .map_err(Error::InvalidFormat)?;
    if scratch_words < SCRATCH_WORDS
        || !(1..=10).contains(&groups)
        || !active.into_iter().any(|v| v)
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

struct Unprepared;
struct GatePrepared;
struct AllGatesQueued;
struct DownPrepared;
/// Owning handles are retained across the entire ordered scratch phase.
/// Only DownPrepared::consume can release the capability to the CPU overwrite.
struct Phase<'a, State> {
    client: &'a ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &'a g::Spans,
    banks: [[Handle; 3]; 2],
    owners: [Handle; 7],
    scratch: Handle,
    active: [bool; 2],
    groups: usize,
    state: PhantomData<State>,
}
impl<'a, State> Phase<'a, State> {
    fn next<Next>(self) -> Phase<'a, Next> {
        Phase {
            client: self.client,
            geometry: self.geometry,
            spans: self.spans,
            banks: self.banks,
            owners: self.owners,
            scratch: self.scratch,
            active: self.active,
            groups: self.groups,
            state: PhantomData,
        }
    }
    fn prepare(&self, down: bool) {
        let s = self.spans;
        let info = [
            s.map_first(),
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
            0,
        ];
        let mut args = KernelArguments::new();
        for handle in [
            &self.owners[if down { 2 } else { 1 }],
            &self.owners[0],
            &self.owners[6],
            &self.scratch,
        ] {
            args = args.with_buffer(handle.clone().binding());
        }
        args = args.with_info(info_binding(info));
        // SAFETY: launch_pair validates the complete map/inputs/status/scratch
        // before entering this phase. Runtime guards cover every compact group.
        unsafe {
            self.client.launch_unchecked(
                Box::new(PrepareKernel { down }),
                CubeCount::Static(1, if down { self.groups as u32 } else { 1 }, 1),
                args,
            );
        }
    }
    fn project(&self, down: bool) {
        let s = self.spans;
        let info = [
            s.map_first(),
            if down { s.act_first() } else { s.input_first() },
            if down {
                s.act_stride()
            } else {
                s.input_stride()
            },
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
            0,
        ];
        let p = self.geometry.projections()[if down { 2 } else { 0 }];
        for (segment, &enabled) in self.active.iter().enumerate() {
            if !enabled {
                continue;
            }
            let banks = &self.banks[segment];
            let mut args = KernelArguments::new();
            let handles: &[&Handle] = if down {
                &[
                    &banks[2],
                    &self.owners[2],
                    &self.owners[0],
                    &self.owners[3],
                    &self.owners[6],
                    &self.scratch,
                ]
            } else {
                &[
                    &banks[0],
                    &banks[1],
                    &self.owners[1],
                    &self.owners[0],
                    &self.owners[2],
                    &self.owners[6],
                    &self.scratch,
                ]
            };
            for h in handles {
                args = args.with_buffer((*h).clone().binding());
            }
            args = args.with_info(info_binding(info));
            // SAFETY: exact original count1 shards/full owners prevalidated;
            // scratch readiness and complete route guards precede all writes.
            unsafe {
                self.client.launch_unchecked(
                    Box::new(ProjectKernel(Kernel {
                        down,
                        dtype: p.dtype(),
                        row_bytes: p.row_bytes(),
                        first: segment * 72,
                        slots: 72,
                    })),
                    CubeCount::Static((if down { 2560 } else { 640 }) / 4, self.groups as u32, 1),
                    args,
                );
            }
        }
    }
}
impl<'a> Phase<'a, Unprepared> {
    fn gate(self) -> Phase<'a, GatePrepared> {
        self.prepare(false);
        self.next()
    }
}
impl<'a> Phase<'a, GatePrepared> {
    fn project_all(self) -> Phase<'a, AllGatesQueued> {
        self.project(false);
        self.next()
    }
}
impl<'a> Phase<'a, AllGatesQueued> {
    fn down(self) -> Phase<'a, DownPrepared> {
        self.prepare(true);
        self.next()
    }
}
impl Phase<'_, DownPrepared> {
    fn consume(self) {
        self.project(true);
    }
}
fn info_binding(info: [usize; 8]) -> MetadataBindingInfo {
    MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    )
}

/// The caller proves physical disjointness, original pending wave/source and
/// full owning extents. Scratch is a complete private owner whose CPU phase
/// must start AFTER this returns, on the same stream. No CPU/GPU overlap in
/// this owner; CPU computations may still use their separate host-only owners.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_pair(
    client: &ComputeClient<WgpuRuntime>,
    geometry: g::Geometry,
    spans: &g::Spans,
    banks: [[Handle; 3]; 2],
    owners: [Handle; 7],
    scratch: Handle,
    active: [bool; 2],
    groups: usize,
) -> Result<()> {
    validate_layout(
        geometry,
        spans,
        owners.each_ref().map(|h| h.size_in_used() as usize / 4),
        scratch.size_in_used() as usize / 4,
        groups,
        active,
    )?;
    for (segment, bank) in banks.iter().enumerate() {
        let mut words = [0; 3];
        for (i, (p, handle)) in geometry.projections().iter().zip(bank).enumerate() {
            words[i] = p.segment_words(72).map_err(Error::InvalidFormat)?;
            if handle.size_in_used() != (words[i] * 4) as u64 {
                return Err(Error::InvalidBufferLength);
            }
        }
        geometry
            .validate_shard(segment * 72, 72, words)
            .map_err(Error::InvalidFormat)?;
    }
    if banks
        .iter()
        .flatten()
        .chain(&owners)
        .chain([&scratch])
        .any(|h| h.stream != owners[0].stream || !h.size_in_used().is_multiple_of(4))
    {
        return Err(Error::InvalidBufferLength);
    }
    Phase::<Unprepared> {
        client,
        geometry,
        spans,
        banks,
        owners,
        scratch,
        active,
        groups,
        state: PhantomData,
    }
    .gate()
    .project_all()
    .down()
    .consume();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn compile(source: &str) -> wgpu::naga::Module {
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::SUBGROUP
                | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)
        .unwrap();
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
        module
    }
    #[test]
    fn preparation_preserves_accepted_quantizer_and_compiles_both_phases() {
        let body = include_str!("iq_moe_integer.wgsl")
            .split_once("// Q8_PREPARE_BEGIN\n")
            .unwrap()
            .1
            .split_once("// Q8_PREPARE_END")
            .unwrap()
            .0;
        for down in [false, true] {
            let source = PrepareKernel { down }.source();
            assert!(source.contains(&body.replace("BLOCKS_VALUE", if down { "20" } else { "80" })));
            assert_eq!(source.matches("prepare_q8(local)").count(), 1);
            assert!(
                source.contains("planes[flag]=select(0u,1u,valid && atomicLoad(&quant_fault)==0u)")
            );
            compile(&source);
        }
    }
    #[test]
    fn all_original_packed_projection_sources_compile_without_quantizer() {
        for (down, dtype, block, bytes) in [
            (false, crate::quant_matvec::IQ3_XXS, 256, 98),
            (false, crate::quant_matvec::IQ3_S, 256, 110),
            (false, crate::quant_matvec::IQ2_S, 256, 82),
            (false, crate::quant_matvec::IQ4_XS, 256, 136),
            (true, crate::quant_matvec::IQ4_NL, 32, 18),
            (true, crate::quant_matvec::Q8_0, 32, 34),
            (true, crate::quant_matvec::Q2_0, 64, 18),
        ] {
            for first in [0, 72] {
                let source = Kernel {
                    down,
                    dtype,
                    row_bytes: (if down { 640 } else { 2560 }) / block * bytes,
                    first,
                    slots: 72,
                }
                .source_variant(true);
                assert!(!source.contains("fn prepare_q8") && !source.contains("cached_input"));
                assert!(source.contains("dot4I8Packed") && source.contains("planes[flag]==1u"));
                let module = compile(&source);
                let shared = module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| v.space == wgpu::naga::AddressSpace::WorkGroup)
                    .map(|(_, v)| match module.types[v.ty].inner {
                        wgpu::naga::TypeInner::Array {
                            size: wgpu::naga::ArraySize::Constant(n),
                            stride,
                            ..
                        } => n.get() as usize * stride as usize,
                        _ => panic!("only prepared workgroup arrays"),
                    })
                    .sum::<usize>();
                assert!(shared <= MAX_PROJECT_SHARED_BYTES);
                assert_eq!(
                    shared,
                    if down {
                        DOWN_WORDS * 4
                    } else {
                        GATE_WORDS * 4 + original_grid_words(dtype).len() * 4
                    }
                );
            }
        }
    }
    #[test]
    fn logical_planes_are_disjoint_bounded_and_never_owning_tail() {
        assert_eq!(SCRATCH_WORDS, 5072);
        assert_eq!(SCRATCH_WORDS * 4, 20_288);
        const { assert!(SCRATCH_WORDS < 25_600) };
        let mut covered = [false; SCRATCH_WORDS];
        for (start, length) in std::iter::once((HEADER_WORDS, GATE_WORDS))
            .chain((0..10).map(|group| (DOWN_FIRST + group * DOWN_WORDS, DOWN_WORDS)))
        {
            for item in &mut covered[start..start + length] {
                assert!(!*item);
                *item = true;
            }
        }
        assert!(covered[HEADER_WORDS..].iter().all(|v| *v));
        assert!(covered[..HEADER_WORDS].iter().all(|v| !v));
    }
    #[test]
    fn layout_rejects_short_scratch_bad_groups_and_empty_segments_before_enqueue() {
        let geometry = g::Geometry::new(
            [
                g::Projection::new(crate::quant_matvec::IQ3_XXS, 2560, 640, 512).unwrap(),
                g::Projection::new(crate::quant_matvec::IQ3_XXS, 2560, 640, 512).unwrap(),
                g::Projection::new(crate::quant_matvec::IQ4_NL, 640, 2560, 512).unwrap(),
            ],
            1,
            144,
        )
        .unwrap();
        let s = g::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
        let words = [
            804,
            8 * 2560,
            80 * 640,
            80 * 2560,
            80,
            8 * 2560,
            geometry.status_words(),
        ];
        assert!(validate_layout(geometry, &s, words, SCRATCH_WORDS, 10, [true, true]).is_ok());
        for (scratch, groups, active) in [
            (SCRATCH_WORDS - 1, 10, [true, true]),
            (SCRATCH_WORDS, 0, [true, true]),
            (SCRATCH_WORDS, 11, [true, true]),
            (SCRATCH_WORDS, 10, [false, false]),
        ] {
            assert!(validate_layout(geometry, &s, words, scratch, groups, active).is_err());
        }
    }
}
