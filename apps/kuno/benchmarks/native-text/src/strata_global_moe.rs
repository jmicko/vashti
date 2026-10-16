//! Connected native global expert family. All routing/remaps/quantized inputs,
//! intermediate rows and acceptance stamps are private. The selected Intel
//! device plan reads both original VRAM cells and registered HOST aliases on
//! the GPU; no CPU expert service or decode staging is part of this path.
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
    32768,
    262144,
    1048576,
    32768,
    32768,
    65536,
    131072,
    crate::strata_global_adapt::ROUTE_MAP_BYTES as u64,
    32768,
    131072,
    1048576,
];
pub(crate) const MAIN_BYTES: u64 = crate::strata_sampler::request::MAIN_BYTES
    + 32768
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
    + crate::strata_dense::INPUT_BYTES
    + crate::strata_prompt::allocation::MAIN_BYTES;
fn reject() -> Error {
    Error::InvalidFormat("Strata global routed family")
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Route,
    Reduce,
    PromptRoute,
    PromptReduce,
}
impl Kernel {
    fn source(self) -> String {
        match self {
            Self::Route => include_str!("strata_global_route.wgsl").into(),
            Self::Reduce => include_str!("strata_global_reduce.wgsl").into(),
            Self::PromptRoute => {
                crate::strata_prompt::groups::shader_prefix()
                    + include_str!("strata_prompt_route.wgsl")
            }
            Self::PromptReduce => {
                crate::strata_prompt::groups::shader_prefix()
                    + include_str!("strata_prompt_reduce.wgsl")
            }
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
    .with_native_parameter(2, cubecl_runtime::server::NativeStrataValue::RouteEpoch)
}
pub(crate) struct Adaptation {
    proposal: crate::strata_global_adapt::Proposal,
    flight: Option<crate::public_host_expert_mirror::strata_global::exchange::Flight>,
}
pub(crate) struct Workspace {
    binding: GraphBindingId,
    geometries: [g::Geometry; 48],
    ids: [BufferId; 11],
    sampler: [BufferId; 7],
    layer_pending: bool,
    adaptive: Option<crate::strata_global_adapt::Cache>,
    dense_input: BufferId,
    prompt_stages: Option<[BufferId; 3]>,
    epoch: u32,
    verify_usage: bool,
    active: Option<Zeroizing<[usize; 3]>>, // active count, next layer, accepted flag
}
impl Workspace {
    pub(crate) fn invocation_controls(
        &self,
        position: usize,
    ) -> Result<cubecl_runtime::server::NativeStrataInvocation> {
        cubecl_runtime::server::NativeStrataInvocation::target_with_usage(
            u32::try_from(position).map_err(u::fault)?,
            self.invocation_epoch()?,
            self.verify_usage,
        )
        .map_err(u::fault)
    }
    pub(crate) fn invocation_epoch(&self) -> Result<u32> {
        if self.epoch == 0
            || self.layer_pending
            || self.active.as_ref().is_none_or(|a| a[1] != 0 || a[2] != 0)
        {
            return Err(reject());
        }
        Ok(self.epoch)
    }
    /// All48 native routed reductions ran in the retained graph. Preserve the
    /// normal epoch-stamp readback in accept(); this changes only queue-state
    /// bookkeeping and cannot mark the window accepted.
    pub(crate) fn complete_native_replay(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        fault: BufferId,
        position: u64,
        completed: &crate::resident_model::target_replay::Completed,
    ) -> Result<()> {
        let active = self.active.as_mut().ok_or_else(reject)?;
        completed.check(fault, position, active[0])?;
        completed.check_fault_owner(arena, fault)?;
        if self.binding != graph.binding_id()
            || self.layer_pending
            || active[1] != 0
            || active[2] != 0
            || self.epoch != completed.route_epoch()
            || self.verify_usage != completed.route_usage()
            || crate::strata_prompt::active()
        {
            return Err(reject());
        }
        active[1] = 48;
        Ok(())
    }
    pub(crate) fn sampler_ids(&self) -> [BufferId; 7] {
        self.sampler
    }
    pub(crate) fn requests() -> Vec<u64> {
        Self::requests_for_context(4096).expect("fixed prompt context")
    }
    pub(crate) fn requests_for_context(context: usize) -> Result<Vec<u64>> {
        Ok(ROOT_BYTES
            .into_iter()
            .chain(crate::strata_sampler::request::ROOT_BYTES)
            .chain([crate::strata_dense::INPUT_BYTES])
            .chain(crate::strata_prompt::allocation::Plan::new(context)?.requests())
            .collect())
    }
    pub(crate) fn main_bytes_for_context(context: usize) -> Result<u64> {
        Self::requests_for_context(context)?
            .into_iter()
            .try_fold(0u64, |a, b| {
                a.checked_add(b).ok_or(Error::ArithmeticOverflow)
            })
    }
    pub(crate) fn allocate(
        graph: BoundGraph<'_, '_>,
        arena: &mut PrivateArena,
        context: usize,
    ) -> Result<Self> {
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
        let sampler = crate::strata_sampler::request::ROOT_BYTES
            .into_iter()
            .map(|b| arena.allocate_zeroed(b as usize / 4).map_err(u::fault))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| reject())?;
        let dense_input = arena
            .allocate_zeroed(crate::strata_dense::INPUT_WORDS)
            .map_err(u::fault)?;
        let prompt_stages =
            crate::strata_prompt::allocation::Plan::new(context)?.allocate(arena)?;
        graph.verify_unchanged()?;
        Ok(Self {
            binding: graph.binding_id(),
            layer_pending: false,
            geometries: geometries.try_into().map_err(|_| reject())?,
            ids: ids.try_into().map_err(|_| reject())?,
            dense_input,
            sampler,
            prompt_stages,
            epoch: 0,
            verify_usage: false,
            active: None,
            adaptive: None,
        })
    }
    pub(crate) fn prompt_stage_ids(&self) -> Option<[BufferId; 3]> {
        self.prompt_stages
    }
    pub(crate) fn prompt_stage_handles(&self, arena: &PrivateArena) -> Result<Option<[Handle; 3]>> {
        let Some(ids) = self.prompt_stages else {
            return Ok(None);
        };
        Ok(Some([
            u::handle(arena, ids[0])?,
            u::handle(arena, ids[1])?,
            u::handle(arena, ids[2])?,
        ]))
    }
    pub(crate) fn dense_input_id(&self) -> BufferId {
        self.dense_input
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
    ) -> Result<()> {
        let h = |id| arena.handle(id).cloned().map_err(u::fault);
        self.enqueue_data(
            graph,
            arena,
            sources,
            admission,
            layer,
            [h(input)?, h(output)?, h(logits)?],
            fault,
            cancel,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_prompt(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        sources: &Sources,
        admission: &crate::executor::ResidentAdmission,
        layer: usize,
        data: [Handle; 3],
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !crate::strata_prompt::active() {
            return Err(reject());
        }
        for h in &data {
            crate::strata_prompt::validate_output(h)?;
        }
        self.enqueue_data(graph, arena, sources, admission, layer, data, fault, cancel)?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_data(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        sources: &Sources,
        admission: &crate::executor::ResidentAdmission,
        layer: usize,
        data: [Handle; 3],
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
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
        // Decode requires the complete attached source partition before any
        // pointer consumer is enqueued. Prompt has its separate loan lifecycle.
        if !crate::strata_prompt::active() {
            sources.validate_graph_ready(graph)?;
        }
        let client = arena.client().map_err(u::fault)?;
        u::zero_many(arena, &self.ids[..7])?;
        let full = |id| arena.handle(id).cloned().map_err(u::fault);
        let mut owners = [
            full(self.ids[0])?,
            data[0].clone(),
            full(self.ids[1])?,
            full(self.ids[2])?,
            full(self.ids[3])?,
            data[1].clone(),
            full(self.ids[4])?,
        ];
        let prompt = crate::strata_prompt::active();
        if prompt {
            let views = crate::strata_prompt::groups::views()?;
            views.reset(client)?;
            owners[0] = views.map;
            owners[2] = views.gu;
            owners[3] = views.partial;
            owners[4] = views.probability;
            owners[6] = views.status;
            if owners[1].size_in_used() < (count * 2560 * 4) as u64
                || owners[5].size_in_used() < (count * 2560 * 4) as u64
                || data[2].size_in_used() < (count * 512 * 4) as u64
                || data.iter().enumerate().any(|(i, a)| {
                    data[..i]
                        .iter()
                        .any(|b| crate::strata_prompt::storage::overlaps(a, b).unwrap_or(true))
                })
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
            .with_buffer(data[2].clone().binding())
            .with_buffer(full(self.ids[7])?.binding())
            .with_buffer(
                if prompt {
                    crate::strata_prompt::groups::views()?.raw
                } else {
                    owners[0].clone()
                }
                .binding(),
            )
            .with_buffer(owners[4].clone().binding())
            .with_buffer(full(fault)?.binding())
            .with_buffer(full(self.ids[9])?.binding())
            .with_info(
                info(&[
                    count as u32,
                    layer as u32,
                    self.epoch,
                    u32::from(self.verify_usage),
                ])
                .with_native_parameter(3, cubecl_runtime::server::NativeStrataValue::RouteUsage),
            );
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
                u::read_handle_prefix_u32(
                    arena,
                    crate::strata_prompt::groups::views()?.raw,
                    crate::strata_prompt::groups::LOGICAL + count * 10,
                )
            })?;
            let map = crate::strata_prompt::groups::group(&raw, count)?;
            let mut bytes = Zeroizing::new(Vec::with_capacity(
                crate::strata_prompt::groups::MAP_WORDS * 4,
            ));
            for &word in map.iter() {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            crate::strata_prompt::stage_private_view(
                arena,
                self.ids[0],
                owners[0].clone(),
                &bytes,
            )?;
            crate::strata_prompt::experts::enqueue(
                client,
                graph,
                sources,
                layer,
                geometry,
                &owners,
                // Consumers bind the exact logical fault word; the arena
                // retains the complete32KiB owner for original final erasure.
                &u::handle(arena, fault)?,
                count,
                &map,
            )?;
            self.layer_pending = true;
            return Ok(());
        }
        // Pinned Intel launcher selects VERIFY_DEVICE_PLAN + VERIFY_NO_HOST.
        // Its resident_plan puts both cached cells and HOST mirror aliases in
        // the primary GPU group list. Every real group is consumed directly;
        // no CPU routing readback, HOST staging or second fetch list is needed.
        let gu = full(self.ids[5])?;
        let down = full(self.ids[6])?;
        crate::strata_global_experts::prepare(client, &spans, &owners, &gu, count, false)?;
        for role in [1, 2] {
            crate::strata_global_experts::role(
                client, graph, sources, layer, geometry, &spans, &owners, &gu, count, role,
            )?;
        }
        crate::strata_global_experts::activate(client, &spans, &owners, count)?;
        crate::strata_global_experts::prepare(client, &spans, &owners, &down, count, true)?;
        crate::strata_global_experts::role(
            client, graph, sources, layer, geometry, &spans, &owners, &down, count, 3,
        )?;
        self.layer_pending = true;
        Ok(())
    }
    pub(crate) fn finish_layer(
        &mut self,
        arena: &PrivateArena,
        layer: usize,
        output: BufferId,
        fault: BufferId,
    ) -> Result<()> {
        self.finish_data(
            arena,
            layer,
            arena.handle(output).cloned().map_err(u::fault)?,
            fault,
        )
    }
    pub(crate) fn finish_prompt(
        &mut self,
        arena: &PrivateArena,
        layer: usize,
        output: Handle,
        fault: BufferId,
    ) -> Result<()> {
        if !crate::strata_prompt::active() {
            return Err(reject());
        }
        crate::strata_prompt::validate_output(&output)?;
        self.finish_data(arena, layer, output, fault)
    }
    fn finish_data(
        &mut self,
        arena: &PrivateArena,
        layer: usize,
        output: Handle,
        fault: BufferId,
    ) -> Result<()> {
        let active = self.active.as_ref().ok_or_else(reject)?;
        let count = active[0];
        if !self.layer_pending || active[1] != layer || active[2] != 0 {
            return Err(reject());
        }
        let client = arena.client().map_err(u::fault)?;
        let full = |id| arena.handle(id).cloned().map_err(u::fault);
        let prompt = crate::strata_prompt::active();
        let [partial, probability, status, map] = if prompt {
            let views = crate::strata_prompt::groups::views()?;
            [views.partial, views.probability, views.status, views.map]
        } else {
            [
                full(self.ids[2])?,
                full(self.ids[3])?,
                full(self.ids[4])?,
                full(self.ids[0])?,
            ]
        };
        let args = KernelArguments::new()
            .with_buffers(vec![
                partial.binding(),
                probability.binding(),
                status.binding(),
                map.binding(),
                output.binding(),
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
    /// Real target rows/INFO are observed; no next target reader exists. Start
    /// original copies BEFORE commit submission/next MTP drafting. Acceptance
    /// remains independent and all maps stay unpublished until finish below.
    pub(crate) fn begin_adaptation_after_verified(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        admission: &crate::executor::ResidentAdmission,
        sources: &mut Sources,
        cancel: &AtomicBool,
    ) -> Result<Option<Adaptation>> {
        if !self.verify_usage {
            return Ok(None);
        }
        if self.layer_pending || self.active.as_ref().is_none_or(|s| s[1] != 48 || s[2] != 0) {
            return Err(reject());
        }
        u::cancel(cancel)?;
        if self.adaptive.is_none() {
            self.adaptive = Some(crate::strata_global_adapt::Cache::new(sources.plan())?);
        }
        let cache = self.adaptive.as_mut().ok_or_else(reject)?;
        if !cache.tick()? {
            return Ok(None);
        }
        let usage =
            admission.readback_audited(cancel, || u::read_prefix_f32(arena, self.ids[9], 24576))?;
        let proposal = cache.propose(&usage)?;
        u::cancel(cancel)?;
        let raw = self.prompt_stages.ok_or_else(reject)?[2];
        let flight = sources.submit_exchange(graph, &proposal.swaps, arena, raw)?;
        Ok(Some(Adaptation { proposal, flight }))
    }
    /// Original target/MTP commit/draft consumers have completed. Join the exact
    /// copy flight and its actual copy->MAIN semaphore dependency, then publish
    /// all three physical roles and the private map/usage before target reuse.
    pub(crate) fn finish_adaptation(
        &mut self,
        arena: &PrivateArena,
        sources: &mut Sources,
        adaptation: Option<Adaptation>,
    ) -> Result<()> {
        self.validate_idle()?;
        let Some(Adaptation { proposal, flight }) = adaptation else {
            return Ok(());
        };
        if let Some(flight) = flight {
            flight.finish()?;
        }
        let mut mapping = crate::strata_global_adapt::encode_route_map(&proposal.mapping)?;
        let mut counts = Zeroizing::new(vec![0u8; 131072]);
        for (i, &score) in proposal.usage.iter().enumerate() {
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
        let cache = self.adaptive.as_mut().ok_or_else(reject)?;
        sources.finish_publication(|| cache.commit(proposal))
    }
    pub(crate) fn validate_idle(&self) -> Result<()> {
        if self.active.is_some() || self.layer_pending {
            Err(reject())
        } else {
            Ok(())
        }
    }
    /// Restore only control data erased with the private arena. The original
    /// physical source placement is authoritative; no profile reset or expert
    /// transfer occurs. Original weights, epochs and adaptive rounds survive.
    pub(crate) fn restore_retained_route_map(
        &self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        admission: &crate::executor::ResidentAdmission,
        sources: &Sources,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_idle()?;
        u::cancel(cancel)?;
        if self.binding != graph.binding_id()
            || self.verify_usage
            || admission.request_io_reuse_active()
        {
            return Err(reject());
        }
        let client = arena.client().map_err(u::fault)?;
        admission.check_client(client)?;
        let mapping = sources.retained_route_map(graph, client)?;
        if let Some(cache) = &self.adaptive {
            cache.check_retained_route_map(&mapping)?;
        } else if mapping
            .iter()
            .enumerate()
            .any(|(i, &cell)| usize::from(cell) != i % crate::strata_global_plan::EXPERTS)
        {
            return Err(reject());
        }
        let mut payload = crate::strata_global_adapt::encode_route_map(&mapping)?;
        let root = arena.handle(self.ids[7]).map_err(u::fault)?.clone();
        if arena.layout(self.ids[7]).map_err(u::fault)?
            != crate::private_arena::BufferLayout::new(
                crate::strata_global_adapt::ROUTE_MAP_BYTES / 4,
            )
            .map_err(u::fault)?
            || root.size_in_used() != crate::strata_global_adapt::ROUTE_MAP_BYTES as u64
            || root.offset_start.unwrap_or(0) != 0
            || root.offset_end.unwrap_or(0) != 0
        {
            return Err(reject());
        }
        admission.readback_audited(cancel, || {
            client
                .write_private(root, std::mem::take(&mut *payload))
                .map_err(u::fault)?;
            let actual = u::read_prefix_u32(
                arena,
                self.ids[7],
                crate::strata_global_adapt::ROUTE_MAP_BYTES / 4,
            )?;
            if actual.len() != crate::strata_global_adapt::ROUTE_MAP_BYTES / 4
                || actual
                    .iter()
                    .enumerate()
                    .any(|(i, &cell)| cell != mapping.get(i).map_or(0, |&v| u32::from(v)))
            {
                return Err(reject());
            }
            sources.validate_graph_ready(graph)?;
            u::cancel(cancel)
        })
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

#[cfg(all(test, feature = "strata-intel-core"))]
mod direct_quote_tests {
    use super::*;
    #[test]
    fn direct_source_plan_prices_dense_planes_and_prompt_roots_without_decode_fetch_stages() {
        let requests = Workspace::requests_for_context(32768).unwrap();
        assert_eq!(requests.len(), 22);
        assert_eq!(requests[18], 512 * 1024);
        assert_eq!(
            &requests[19..],
            &[756 * 1024 * 1024, 642 * 1024 * 1024, 32 * 1024 * 1024]
        );
        assert!(!requests.contains(&(128 * 1024 * 1024)));
        assert_eq!(requests.iter().sum::<u64>(), 1_503_494_144);
        assert_eq!(MAIN_BYTES, 1_503_494_144);
        assert_eq!(
            3 * 128 * 1024 * 1024 - crate::strata_dense::INPUT_BYTES,
            402_128_896
        );
    }
}
