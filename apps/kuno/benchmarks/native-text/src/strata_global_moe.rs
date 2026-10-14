//! Connected native global expert family. All routing/remaps/quantized inputs,
//! selected HOST copies, intermediate rows and acceptance stamps are private.
use crate::{
    private_arena::{BufferId, PrivateArena},
    quant_matvec::grouped_moe as g,
    resident_util as u,
    strata_global_sources::Sources,
};
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::{CubeCount, CubeDim},
    wgpu::AutoCompiler,
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
use kuno_text::{
    Error, Result,
    model::{BoundGraph, GraphBindingId},
};
use std::sync::atomic::AtomicBool;
use zeroize::Zeroizing;

const ROOT_BYTES: [u64; 9] = [
    32768, 262144, 1048576, 32768, 32768, 65536, 131072, 131072, 32768,
];
pub(crate) const MAIN_BYTES: u64 = 32768
    + 262144
    + 1048576
    + 32768
    + 32768
    + 65536
    + 131072
    + 131072
    + 32768
    + 3 * crate::strata_global_experts::STAGE_BYTES;
fn reject() -> Error {
    Error::InvalidFormat("Strata global routed family")
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Route,
    Reduce,
}
impl Kernel {
    fn source(self) -> String {
        match self {
            Self::Route => include_str!("strata_global_route.wgsl").into(),
            Self::Reduce => include_str!("strata_global_reduce.wgsl").into(),
        }
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Route => "StrataGlobalRouteAndPlan",
            Self::Reduce => "StrataGlobalRouteReduceAndStamp",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(256))
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
            debug_info: None,
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
        })
    }
}
fn info(words: &[u32]) -> MetadataBindingInfo {
    MetadataBindingInfo::custom(
        words
            .chunks_exact(2)
            .map(|w| w[0] as u64 | ((w[1] as u64) << 32))
            .collect(),
    )
}
pub(crate) struct Workspace {
    binding: GraphBindingId,
    geometries: [g::Geometry; 48],
    ids: [BufferId; 9],
    stages: [BufferId; 3],
    epoch: u32,
    active: Option<Zeroizing<[usize; 3]>>, // active count, next layer, accepted flag
}
impl Workspace {
    pub(crate) fn requests() -> Vec<u64> {
        ROOT_BYTES
            .into_iter()
            .chain([crate::strata_global_experts::STAGE_BYTES; 3])
            .collect()
    }
    pub(crate) fn allocate(graph: BoundGraph<'_, '_>, arena: &mut PrivateArena) -> Result<Self> {
        if !crate::strata_intel_core::enabled() {
            return Err(reject());
        }
        let mut geometries = Vec::with_capacity(48);
        for layer in 0..48 {
            let mut projections = Vec::with_capacity(3);
            for name in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                let t = graph.tensor(&format!("blk.{layer}.{name}"))?;
                let d = t.info().dimensions();
                if d.len() != 3 {
                    return Err(reject());
                }
                projections.push(
                    g::Projection::new(
                        t.info().dtype().gguf_id(),
                        d[0] as usize,
                        d[1] as usize,
                        d[2] as usize,
                    )
                    .map_err(Error::InvalidFormat)?,
                );
            }
            geometries.push(
                g::Geometry::global_sources(projections.try_into().map_err(|_| reject())?)
                    .map_err(Error::InvalidFormat)?,
            );
        }
        let mut ids = Vec::with_capacity(9);
        for (index, bytes) in ROOT_BYTES.iter().enumerate() {
            let id = if index == 7 {
                let mut remap = Zeroizing::new(vec![0u32; 32768]);
                for layer in 0..48 {
                    for expert in 0..512 {
                        remap[layer * 512 + expert] = expert as u32;
                    }
                }
                arena.upload_u32(&remap).map_err(u::fault)?
            } else {
                arena
                    .allocate_zeroed(*bytes as usize / 4)
                    .map_err(u::fault)?
            };
            ids.push(id);
        }
        let stage_words = (crate::strata_global_experts::STAGE_BYTES / 4) as usize;
        let stages = [
            arena.allocate_zeroed(stage_words).map_err(u::fault)?,
            arena.allocate_zeroed(stage_words).map_err(u::fault)?,
            arena.allocate_zeroed(stage_words).map_err(u::fault)?,
        ];
        graph.verify_unchanged()?;
        Ok(Self {
            binding: graph.binding_id(),
            geometries: geometries.try_into().map_err(|_| reject())?,
            ids: ids.try_into().map_err(|_| reject())?,
            stages,
            epoch: 0,
            active: None,
        })
    }
    pub(crate) fn stage_ids(&self) -> [BufferId; 3] {
        self.stages
    }
    pub(crate) fn stage_handles(&self, arena: &PrivateArena) -> Result<[Handle; 3]> {
        Ok([
            arena.handle(self.stages[0]).map_err(u::fault)?.clone(),
            arena.handle(self.stages[1]).map_err(u::fault)?.clone(),
            arena.handle(self.stages[2]).map_err(u::fault)?.clone(),
        ])
    }
    pub(crate) fn begin(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        count: usize,
    ) -> Result<()> {
        self.validate_idle()?;
        if graph.binding_id() != self.binding || !(1..=8).contains(&count) {
            return Err(reject());
        }
        self.epoch = self.epoch.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        u::zero_many(arena, &[self.ids[8]])?;
        self.active = Some(Zeroizing::new([count, 0, 0]));
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        sources: &Sources,
        layer: usize,
        input: BufferId,
        output: BufferId,
        logits: BufferId,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        let active = self.active.as_ref().ok_or_else(reject)?;
        let count = active[0];
        if active[1] != layer || active[2] != 0 || graph.binding_id() != self.binding {
            return Err(reject());
        }
        sources.plan().check_binding(graph)?;
        let client = arena.client().map_err(u::fault)?;
        u::zero_many(arena, &self.ids[..7])?;
        let full = |id| arena.handle(id).cloned().map_err(u::fault);
        let owners = [
            full(self.ids[0])?,
            full(input)?,
            full(self.ids[1])?,
            full(self.ids[2])?,
            full(self.ids[3])?,
            full(output)?,
            full(self.ids[4])?,
        ];
        let spans = g::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
        let geometry = *self.geometries.get(layer).ok_or_else(reject)?;
        spans
            .validate(
                geometry,
                owners.each_ref().map(|h| h.size_in_used() as usize / 4),
            )
            .map_err(Error::InvalidFormat)?;
        let args = KernelArguments::new()
            .with_buffer(full(logits)?.binding())
            .with_buffer(full(self.ids[7])?.binding())
            .with_buffer(owners[0].clone().binding())
            .with_buffer(owners[4].clone().binding())
            .with_buffer(full(fault)?.binding())
            .with_info(info(&[count as u32, layer as u32, self.epoch, 0]));
        // SAFETY: exact private routing/remap roots; no host readback or cache
        // transfer can bypass the owning graph's dependencies.
        unsafe {
            client.launch_unchecked(Box::new(Kernel::Route), CubeCount::Static(1, 1, 1), args);
        }
        let stages = self.stage_handles(arena)?;
        let gu = full(self.ids[5])?;
        let down = full(self.ids[6])?;
        crate::strata_global_experts::prepare(client, &spans, &owners, &gu, count, false)?;
        crate::strata_global_experts::fetch(
            client, graph, sources, layer, geometry, &owners[0], &owners[6], &stages, count,
        )?;
        for role in [1, 2] {
            crate::strata_global_experts::role(
                client, graph, sources, layer, geometry, &spans, &owners, &gu, &stages, count, role,
            )?;
        }
        crate::strata_global_experts::activate(client, &spans, &owners, count)?;
        crate::strata_global_experts::prepare(client, &spans, &owners, &down, count, true)?;
        crate::strata_global_experts::role(
            client, graph, sources, layer, geometry, &spans, &owners, &down, &stages, count, 3,
        )?;
        let args = KernelArguments::new()
            .with_buffers(vec![
                owners[3].clone().binding(),
                owners[4].clone().binding(),
                owners[6].clone().binding(),
                owners[0].clone().binding(),
                owners[5].clone().binding(),
                full(self.ids[8])?.binding(),
                full(fault)?.binding(),
            ])
            .with_info(info(&[
                count as u32,
                layer as u32,
                self.epoch,
                0,
                0,
                2560,
                0,
                0,
            ]));
        // SAFETY: all ten actual rows and role stamps precede this reduction;
        // sticky fault joins every row and the whole-model acceptance stamp.
        unsafe {
            client.launch_unchecked(
                Box::new(Kernel::Reduce),
                CubeCount::Static(10, count as u32, 1),
                args,
            );
        }
        self.active.as_mut().ok_or_else(reject)?[1] += 1;
        Ok(())
    }
    pub(crate) fn accept(
        &mut self,
        arena: &PrivateArena,
        admission: &crate::executor::ResidentAdmission,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let state = self.active.as_ref().ok_or_else(reject)?;
        if state[1] != 48 || state[2] != 0 {
            return Err(reject());
        }
        let stamp =
            admission.readback_audited(cancel, || u::read_prefix_u32(arena, self.ids[8], 192))?;
        for layer in 0..48 {
            if stamp[layer * 4..layer * 4 + 4]
                != [state[0] as u32, self.epoch, 127, layer as u32 + 1]
            {
                return Err(reject());
            }
        }
        self.active.as_mut().ok_or_else(reject)?[2] = 1;
        Ok(())
    }
    pub(crate) fn finish(&mut self) -> Result<()> {
        if self.active.as_ref().is_none_or(|a| a[1] != 48 || a[2] != 1) {
            return Err(reject());
        }
        self.active = None;
        Ok(())
    }
    pub(crate) fn validate_idle(&self) -> Result<()> {
        if self.active.is_some() {
            Err(reject())
        } else {
            Ok(())
        }
    }
}
pub(crate) fn compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [Kernel::Route, Kernel::Reduce]
        .into_iter()
        .map(|k| {
            k.compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap()
        })
        .collect()
}
