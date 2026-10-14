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

const ROOT_BYTES: [u64; 11] = [
    32768, 262144, 1048576, 32768, 32768, 65536, 131072, 131072, 32768, 131072, 1048576,
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
    + 131072
    + 1048576
    + 3 * crate::strata_global_experts::STAGE_BYTES;
fn reject() -> Error {
    Error::InvalidFormat("Strata global routed family")
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Route,
    Reduce,
    PromptRoute,
    PromptReduce,
    CpuMerge,
}
impl Kernel {
    fn source(self) -> String {
        match self {
            Self::Route => include_str!("strata_global_route.wgsl").into(),
            Self::Reduce => include_str!("strata_global_reduce.wgsl").into(),
            Self::PromptRoute => include_str!("strata_prompt_route.wgsl").into(),
            Self::PromptReduce => include_str!("strata_global_reduce.wgsl")
                .replace("p.count>8u", "p.count>256u")
                .replace(
                    "arrayLength(&partial)!=262144u",
                    "arrayLength(&partial)!=8388608u",
                )
                .replace("status[80u+route]", "status[2560u+route]"),
            Self::CpuMerge => include_str!("strata_global_cpu_merge.wgsl").into(),
        }
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Route => "StrataGlobalRouteAndPlan",
            Self::Reduce => "StrataGlobalRouteReduceAndStamp",
            Self::PromptRoute => "StrataPromptNativeRoutes",
            Self::PromptReduce => "StrataPromptRankReduceAndStamp",
            Self::CpuMerge => "StrataGlobalNativeCpuJoin",
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
    ids: [BufferId; 11],
    pub(crate) cpu: std::sync::Arc<kuno_text::cpu_expert::native::Pool>,
    layer_pending: bool,
    adaptive: Option<crate::strata_global_adapt::Cache>,
    stages: [BufferId; 3],
    epoch: u32,
    verify_usage: bool,
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
        let mut ids = Vec::with_capacity(11);
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
            cpu: std::sync::Arc::new(kuno_text::cpu_expert::native::Pool::new()?),
            layer_pending: false,
            geometries: geometries.try_into().map_err(|_| reject())?,
            ids: ids.try_into().map_err(|_| reject())?,
            stages,
            epoch: 0,
            verify_usage: false,
            active: None,
            adaptive: None,
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
        let capacity = if crate::strata_prompt::active() {
            crate::strata_prompt::groups::CAPACITY
        } else {
            8
        };
        if graph.binding_id() != self.binding || !(1..=capacity).contains(&count) {
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
        admission: &crate::executor::ResidentAdmission,
        layer: usize,
        input: BufferId,
        output: BufferId,
        logits: BufferId,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<crate::strata_global_cpu::Plan> {
        u::cancel(cancel)?;
        let active = self.active.as_ref().ok_or_else(reject)?;
        let count = active[0];
        if self.layer_pending
            || active[1] != layer
            || active[2] != 0
            || graph.binding_id() != self.binding
        {
            return Err(reject());
        }
        sources.plan().check_binding(graph)?;
        let client = arena.client().map_err(u::fault)?;
        u::zero_many(arena, &self.ids[..7])?;
        let full = |id| arena.handle(id).cloned().map_err(u::fault);
        let mut owners = [
            full(self.ids[0])?,
            full(input)?,
            full(self.ids[1])?,
            full(self.ids[2])?,
            full(self.ids[3])?,
            full(output)?,
            full(self.ids[4])?,
        ];
        let prompt = crate::strata_prompt::active();
        if prompt {
            let [gu, partial] = crate::strata_prompt::groups::scratch()?;
            owners[2] = gu;
            owners[3] = partial;
            if owners[1].size_in_used() < (count * 2560 * 4) as u64
                || owners[5].size_in_used() < (count * 2560 * 4) as u64
            {
                return Err(reject());
            }
        }
        let spans = g::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
        let geometry = *self.geometries.get(layer).ok_or_else(reject)?;
        if !prompt {
            spans
                .validate(
                    geometry,
                    owners.each_ref().map(|h| h.size_in_used() as usize / 4),
                )
                .map_err(Error::InvalidFormat)?;
        }
        let args = KernelArguments::new()
            .with_buffer(full(logits)?.binding())
            .with_buffer(full(self.ids[7])?.binding())
            .with_buffer(owners[0].clone().binding())
            .with_buffer(owners[4].clone().binding())
            .with_buffer(full(fault)?.binding())
            .with_buffer(full(self.ids[9])?.binding())
            .with_info(info(&[
                count as u32,
                layer as u32,
                self.epoch,
                u32::from(self.verify_usage),
            ]));
        // SAFETY: exact private routing/remap roots; no host readback or cache
        // transfer can bypass the owning graph's dependencies.
        unsafe {
            client.launch_unchecked(
                Box::new(if prompt {
                    Kernel::PromptRoute
                } else {
                    Kernel::Route
                }),
                CubeCount::Static(1, 1, 1),
                args,
            );
        }
        if prompt {
            // Intel prompt grouping is a single bounded router readback. No
            // activation readback or CPU expert contribution is scheduled.
            let raw = admission.readback_audited(cancel, || {
                u::read_prefix_u32(
                    arena,
                    self.ids[0],
                    crate::strata_prompt::groups::LOGICAL + count * 10,
                )
            })?;
            let map = crate::strata_prompt::groups::group(&raw, count)?;
            let mut bytes = Zeroizing::new(Vec::with_capacity(32768));
            for &word in map.iter() {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            client
                .write_private(owners[0].clone(), std::mem::take(&mut *bytes))
                .map_err(u::fault)?;
            crate::strata_prompt::experts::enqueue(
                client,
                graph,
                sources,
                layer,
                geometry,
                &owners,
                &full(fault)?,
                count,
                &map,
            )?;
            self.layer_pending = true;
            // All prompt expert work is already queued. There are no CPU jobs
            // or remaining decode GPU groups for the caller to schedule; this
            // does not claim that the router selected zero experts.
            return Ok(crate::strata_global_cpu::Plan {
                jobs: Vec::new(),
                input: Zeroizing::new(Vec::new()),
                gpu_groups: 0,
            });
        }
        let cpu_plan = crate::strata_global_cpu::prepare(
            graph,
            arena,
            admission,
            sources,
            layer,
            count,
            self.ids[0],
            input,
            cancel,
        )?;
        if cpu_plan.gpu_groups > 0 {
            let stages = self.stage_handles(arena)?;
            let gu = full(self.ids[5])?;
            let down = full(self.ids[6])?;
            crate::strata_global_experts::prepare(client, &spans, &owners, &gu, count, false)?;
            crate::strata_global_experts::fetch(
                client, graph, sources, layer, geometry, &owners[0], &owners[6], &stages, count,
            )?;
            for role in [1, 2] {
                crate::strata_global_experts::role(
                    client, graph, sources, layer, geometry, &spans, &owners, &gu, &stages, count,
                    role,
                )?;
            }
            crate::strata_global_experts::activate(client, &spans, &owners, count)?;
            crate::strata_global_experts::prepare(client, &spans, &owners, &down, count, true)?;
            crate::strata_global_experts::role(
                client, graph, sources, layer, geometry, &spans, &owners, &down, &stages, count, 3,
            )?;
        }
        self.layer_pending = true;
        Ok(cpu_plan)
    }
    pub(crate) fn finish_layer(
        &mut self,
        arena: &PrivateArena,
        layer: usize,
        output: BufferId,
        fault: BufferId,
        cpu: Option<&kuno_text::cpu_expert::native::Partials>,
    ) -> Result<()> {
        let active = self.active.as_ref().ok_or_else(reject)?;
        let count = active[0];
        if !self.layer_pending || active[1] != layer || active[2] != 0 {
            return Err(reject());
        }
        let client = arena.client().map_err(u::fault)?;
        let full = |id| arena.handle(id).cloned().map_err(u::fault);
        let prompt = crate::strata_prompt::active();
        if prompt && cpu.is_some() {
            return Err(reject());
        }
        if let Some(cpu) = cpu {
            if cpu.values().len() != 80 * 2560 {
                return Err(reject());
            }
            let mut bytes = Zeroizing::new(vec![0u8; 1048576]);
            for (i, &x) in cpu.values().iter().enumerate() {
                if !x.is_finite() {
                    return Err(reject());
                }
                bytes[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
            client
                .write_private(full(self.ids[10])?, std::mem::take(&mut *bytes))
                .map_err(u::fault)?;
            let args = KernelArguments::new()
                .with_buffers(vec![
                    full(self.ids[10])?.binding(),
                    full(self.ids[0])?.binding(),
                    full(self.ids[2])?.binding(),
                    full(self.ids[4])?.binding(),
                    full(fault)?.binding(),
                ])
                .with_info(info(&[count as u32, 0, 0, 0]));
            // SAFETY: real joined CPU result and authenticated private masks;
            // GPU and CPU routes are disjoint, all parent owners retained.
            unsafe {
                client.launch_unchecked(
                    Box::new(Kernel::CpuMerge),
                    CubeCount::Static(10, (count * 10) as u32, 1),
                    args,
                );
            }
        }
        let args = KernelArguments::new()
            .with_buffers(vec![
                if prompt {
                    crate::strata_prompt::groups::scratch()?[1].clone()
                } else {
                    full(self.ids[2])?
                }
                .binding(),
                full(self.ids[3])?.binding(),
                full(self.ids[4])?.binding(),
                full(self.ids[0])?.binding(),
                full(output)?.binding(),
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
                Box::new(if prompt {
                    Kernel::PromptReduce
                } else {
                    Kernel::Reduce
                }),
                CubeCount::Static(10, count as u32, 1),
                args,
            );
        }
        self.active.as_mut().ok_or_else(reject)?[1] += 1;
        self.layer_pending = false;
        Ok(())
    }
    pub(crate) fn accept(
        &mut self,
        arena: &PrivateArena,
        admission: &crate::executor::ResidentAdmission,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let state = self.active.as_ref().ok_or_else(reject)?;
        if self.layer_pending || state[1] != 48 || state[2] != 0 {
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
    /// Whole model/family acceptance and commit precede all cache mutations.
    /// The private routing map is published only after paired original weights
    /// and both metadata uploads have joined. Failure retires the source owner.
    pub(crate) fn adapt_after_finished(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        admission: &crate::executor::ResidentAdmission,
        sources: &mut Sources,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_idle()?;
        u::cancel(cancel)?;
        if self.adaptive.is_none() {
            self.adaptive = Some(crate::strata_global_adapt::Cache::new(sources.plan())?);
        }
        let cache = self.adaptive.as_mut().ok_or_else(reject)?;
        if !cache.tick()? {
            return Ok(());
        }
        let usage =
            admission.readback_audited(cancel, || u::read_prefix_f32(arena, self.ids[9], 24576))?;
        let proposal = cache.propose(&usage)?;
        u::cancel(cancel)?;
        sources.exchange(graph, &proposal.swaps)?;
        // Full private owners are retained through their write/fence. Padding
        // is zero and moved upload allocations receive the runtime wipe guard.
        let mut mapping = Zeroizing::new(vec![0u8; 131072]);
        let mut counts = Zeroizing::new(vec![0u8; 131072]);
        for (i, (&cell, &score)) in proposal
            .mapping
            .iter()
            .zip(proposal.usage.iter())
            .enumerate()
        {
            mapping[i * 4..i * 4 + 4].copy_from_slice(&u32::from(cell).to_le_bytes());
            counts[i * 4..i * 4 + 4].copy_from_slice(&score.to_le_bytes());
        }
        let client = arena.client().map_err(u::fault)?;
        client
            .write_private(
                arena.handle(self.ids[7]).map_err(u::fault)?.clone(),
                std::mem::take(&mut *mapping),
            )
            .map_err(u::fault)?;
        client
            .write_private(
                arena.handle(self.ids[9]).map_err(u::fault)?.clone(),
                std::mem::take(&mut *counts),
            )
            .map_err(u::fault)?;
        sources.finish_publication(|| cache.commit(proposal))?;
        Ok(())
    }
    pub(crate) fn validate_idle(&self) -> Result<()> {
        if self.active.is_some() || self.layer_pending {
            Err(reject())
        } else {
            Ok(())
        }
    }
    pub(crate) fn set_verify_usage(&mut self, enabled: bool) -> Result<()> {
        self.validate_idle()?;
        if self.verify_usage == enabled {
            return Err(reject());
        }
        self.verify_usage = enabled;
        Ok(())
    }
}
pub(crate) fn compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Route,
        Kernel::Reduce,
        Kernel::PromptRoute,
        Kernel::PromptReduce,
        Kernel::CpuMerge,
    ]
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
