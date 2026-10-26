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
#[path = "strata_prompt_route_numeric_probe.rs"]
mod prompt_numeric;
#[cfg(any(test, feature = "strata-route-registers"))]
#[path = "strata_prompt_route_registers.rs"]
mod route_registers;
pub(crate) use prompt_numeric::cases as public_prompt_numeric;
#[cfg(feature = "strata-expert-ordered-output")]
#[path = "strata_prompt_ordered_probe.rs"]
mod prompt_ordered;
#[cfg(feature = "strata-expert-ordered-output")]
pub(crate) use prompt_ordered::cases as public_prompt_ordered;
#[cfg(feature="strata-cohort-row-order")]
pub(crate) use prompt_ordered::row_order_cases as public_prompt_row_order;

/// Compiled-public paired producer probes bind real arena owners directly.
/// This launches the exact consumer used by finish_prompt_ordered_shared.
#[cfg(feature = "strata-expert-ordered-output")]
pub(crate) fn launch_public_ordered_shared(
    client: &cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
    handles: [Handle; 11],
    count: usize,
    layer: usize,
    epoch: u32,
) -> Result<()> {
    if !(1..=crate::strata_prompt::groups::CAPACITY).contains(&count)
        || layer >= 48 || epoch == 0 || handles.iter().any(|h| h.stream != handles[0].stream)
    {
        return Err(reject());
    }
    for (i, handle) in handles.iter().enumerate() {
        for previous in &handles[..i] {
            if crate::strata_prompt::storage::overlaps(handle, previous)? {
                return Err(reject());
            }
        }
    }
    let args = KernelArguments::new()
        .with_buffers(handles.into_iter().map(|h| h.binding()).collect())
        .with_info(MetadataBindingInfo::custom(vec![
            count as u64 | ((layer as u64) << 32), u64::from(epoch), 2560u64 << 32, 0,
        ]));
    // SAFETY: caller supplies only fixed-public, real retained arena owners;
    // production WGSL validates every original bound extent before access.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::PromptOrderedSharedReduce),
            CubeCount::Static(count as u32, 1, 1), args);
    }
    Ok(())
}

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
    + crate::strata_prompt::allocation::MAIN_BYTES
    + crate::strata_prompt::allocation::SECONDARY_RAW_BYTES;
fn reject() -> Error {
    Error::InvalidFormat("Strata global routed family")
}
/// The selected Intel prompt kernel owns eight independent token subgroups.
/// This changes execution coverage only; each token keeps its32-lane math.
fn prompt_route_groups(count: usize) -> Result<u32> {
    if !(1..=crate::strata_prompt::groups::CAPACITY).contains(&count) {
        return Err(reject());
    }
    u32::try_from(count.div_ceil(8)).map_err(|_| reject())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Route,
    Plan,
    Reduce,
    SharedReduce,
    PromptRoute,
    PromptRouteFinalize,
    PromptReduce,
    PromptOrderedSharedReduce,
}
impl Kernel {
    fn dim(self) -> CubeDim {
        CubeDim::new_1d(if matches!(self, Self::Plan | Self::SharedReduce) {
            128
        } else {
            256
        })
    }
    fn source(self) -> String {
        match self {
            Self::Route => include_str!("strata_global_route.wgsl").into(),
            Self::Plan => include_str!("strata_global_plan.wgsl").into(),
            Self::Reduce => include_str!("strata_global_reduce.wgsl").into(),
            Self::SharedReduce => include_str!("strata_global_shared_reduce.wgsl").into(),
            Self::PromptRoute => {
                let source = crate::strata_prompt::groups::shader_prefix()
                    + include_str!("strata_prompt_route.wgsl");
                #[cfg(feature = "strata-route-registers")]
                {
                    return route_registers::source(source);
                }
                #[cfg(not(feature = "strata-route-registers"))]
                {
                    source
                }
            }
            Self::PromptRouteFinalize => {
                crate::strata_prompt::groups::shader_prefix()
                    + include_str!("strata_prompt_route_finalize.wgsl")
            }
            Self::PromptReduce => {
                crate::strata_prompt::groups::shader_prefix()
                    + include_str!("strata_prompt_reduce.wgsl")
            }
            Self::PromptOrderedSharedReduce => {
                crate::strata_prompt::groups::shader_prefix()
                    + include_str!("strata_prompt_ordered_shared_reduce.wgsl")
            }
        }
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Route => "StrataGlobalNativeRoutes",
            Self::Plan => "StrataGlobalResidentPlan",
            Self::Reduce => "StrataGlobalRouteReduceAndStamp",
            Self::SharedReduce => "StrataGlobalSharedReduceAndStamp",
            Self::PromptRoute => "StrataPromptNativeRoutes",
            Self::PromptRouteFinalize => "StrataPromptNativeRouteHeader",
            Self::PromptReduce => "StrataPromptRankReduceAndStamp",
            Self::PromptOrderedSharedReduce => "StrataPromptOrderedSharedReduceAndStamp",
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
            debug_info: None,
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: self.dim(),
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
pub(crate) enum Adaptation {
    NoHost(crate::strata_global_adapt::SelectionFlight),
    GpuCounters(Box<crate::public_host_expert_mirror::strata_global::exchange::Flight>),
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
    prompt_secondary_raw: Option<BufferId>,
    epoch: u32,
    verified_window: bool, // window/job accounting, independent of GPU counter policy
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
            crate::strata_global_adapt::collect_gpu_route_usage(self.verified_window),
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
            || crate::strata_global_adapt::collect_gpu_route_usage(self.verified_window)
                != completed.route_usage()
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
        let prompt_plan = crate::strata_prompt::allocation::Plan::new(context)?;
        let prompt_stages = prompt_plan.allocate(arena)?;
        let prompt_secondary_raw = prompt_plan.allocate_secondary_raw_bank(arena)?;
        if prompt_stages.is_none() && prompt_secondary_raw.is_some() {
            return Err(reject());
        }
        graph.verify_unchanged()?;
        Ok(Self {
            binding: graph.binding_id(),
            layer_pending: false,
            geometries: geometries.try_into().map_err(|_| reject())?,
            ids: ids.try_into().map_err(|_| reject())?,
            dense_input,
            sampler,
            prompt_stages,
            prompt_secondary_raw,
            epoch: 0,
            verified_window: false,
            active: None,
            adaptive: None,
        })
    }
    pub(crate) fn prompt_stage_ids(&self) -> Option<[BufferId; 3]> {
        self.prompt_stages
    }
    pub(crate) fn prompt_secondary_raw_id(&self) -> Option<BufferId> {
        self.prompt_secondary_raw
    }
    /// Resolve the admitted owner through the same live arena ledger as the
    /// original roots. No synthetic or sliced Handle can supply this bank.
    pub(crate) fn prompt_secondary_raw_handle(
        &self,
        arena: &PrivateArena,
    ) -> Result<Option<Handle>> {
        self.prompt_secondary_raw
            .map(|id| u::handle(arena, id))
            .transpose()
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
    pub(crate) fn enqueue_with_shared(
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
        fork: &mut dyn FnMut() -> Result<()>,
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
            Some(fork),
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
        self.enqueue_data(
            graph, arena, sources, admission, layer, data, fault, cancel, None,
        )?;
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
        fork: Option<&mut dyn FnMut() -> Result<()>>,
    ) -> Result<()> {
        let mut boundary = crate::strata_prefill_profile::BoundaryTimer::new();
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
        boundary.mark("expert_boundary_owner_reset");
        let route_args = || -> Result<KernelArguments> {
            Ok(KernelArguments::new()
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
                        u32::from(crate::strata_global_adapt::collect_gpu_route_usage(
                            self.verified_window,
                        )),
                    ])
                    .with_native_parameter(
                        3,
                        cubecl_runtime::server::NativeStrataValue::RouteUsage,
                    ),
                ))
        };
        // SAFETY: exact private routing/remap roots; no host readback or cache
        // transfer can bypass the owning graph's dependencies.
        unsafe {
            client.launch_unchecked(
                Box::new(if prompt {
                    Kernel::PromptRoute
                } else {
                    Kernel::Route
                }),
                CubeCount::Static(
                    if prompt {
                        prompt_route_groups(count)?
                    } else {
                        count as u32
                    },
                    1,
                    1,
                ),
                route_args()?,
            );
            if prompt {
                // The independent token groups never publish a prefix header.
                // Their real map/probability/fault dependencies precede this
                // checked serial footer and the original audited host readback.
                client.launch_unchecked(
                    Box::new(Kernel::PromptRouteFinalize),
                    CubeCount::Static(1, 1, 1),
                    route_args()?,
                );
            } else {
                // Selected Intel schedule: route<512> per token, then the
                // original128-thread resident-plan scan. Both nodes bind the
                // SAME private map root, so capture/replay owns the real RAW
                // dependency before any original source-table expert read.
                client.launch_unchecked(
                    Box::new(Kernel::Plan),
                    CubeCount::Static(1, 1, 1),
                    route_args()?,
                );
            }
        }
        boundary.mark("expert_boundary_route_enqueue");
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
            boundary.mark("expert_boundary_route_readback_checkpoint");
            let map = crate::strata_prompt::groups::group_for_prefill(&raw, count).map_err(|error| {
                // Fixed compiled-public diagnostics only. No input, route,
                // expert ID, tensor value or private-worker observation.
                if std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1") {
                    eprintln!("public_prefill_phase scratch_group_failure_layer_{layer}");
                }
                error
            })?;
            drop(raw);
            boundary.mark("expert_boundary_host_group");
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
            boundary.mark("expert_boundary_map_upload");
            #[cfg(feature = "strata-expert-ordered-output")]
            {
                let inverse = crate::strata_prompt::groups::ordered_inverse(&map, count)?;
                // Reuse the priced MAP upload capacity. Clear the previous
                // full initialized span before shortening this private Vec.
                bytes.fill(0);
                bytes.clear();
                for &word in inverse.iter() {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                crate::strata_prompt::stage_private_view(
                    arena,
                    self.ids[0],
                    crate::strata_prompt::groups::views()?.raw,
                    &bytes,
                )?;
                boundary.mark("expert_boundary_inverse_upload");
            }
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
            boundary.mark("expert_boundary_expert_collect_backend");
            self.layer_pending = true;
            return Ok(());
        }
        if let Some(fork) = fork {
            fork()?;
        }
        // Pinned Intel launcher selects VERIFY_DEVICE_PLAN + VERIFY_NO_HOST.
        // Its resident_plan puts both cached cells and HOST mirror aliases in
        // the primary GPU group list. Every real group is consumed directly;
        // no CPU routing readback, HOST staging or second fetch list is needed.
        let gu = full(self.ids[5])?;
        let down = full(self.ids[6])?;
        crate::strata_global_experts::prepare_gu(client, &spans, &owners, &gu, count)?;
        crate::strata_global_experts::role(
            client, graph, sources, layer, geometry, &spans, &owners, &gu, count, 4,
        )?;
        crate::strata_global_experts::prepare_down(client, &spans, &owners, &down, count)?;
        crate::strata_global_experts::role(
            client, graph, sources, layer, geometry, &spans, &owners, &down, count, 3,
        )?;
        self.layer_pending = true;
        Ok(())
    }
    pub(crate) fn finish_layer_shared(
        &mut self,
        arena: &PrivateArena,
        layer: usize,
        output: BufferId,
        fault: BufferId,
        shared: Handle,
    ) -> Result<()> {
        if !crate::strata_intel_core::enabled()
            || crate::strata_prompt::active()
            || shared.size_in_used() != 8 * 2560 * 4
        {
            return Err(reject());
        }
        self.finish_data(
            arena,
            layer,
            arena.handle(output).cloned().map_err(u::fault)?,
            fault,
            Some(shared),
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
        self.finish_data(arena, layer, output, fault, None)
    }
    /// Ordered PART consumer plus the original prompt SharedMerge. The old
    /// routed owner remains bound RW so failed reductions retain its exact
    /// prior value before the shared merge, as the separate original did.
    #[cfg(feature = "strata-expert-ordered-output")]
    pub(crate) fn finish_prompt_ordered_shared(
        &mut self,
        arena: &PrivateArena,
        layer: usize,
        routed: Handle,
        shared: Handle,
        scalar: Handle,
        combined: Handle,
        fault: BufferId,
    ) -> Result<()> {
        if !crate::strata_prompt::active() {
            return Err(reject());
        }
        let active = self.active.as_ref().ok_or_else(reject)?;
        let count = active[0];
        if !self.layer_pending || active[1] != layer || active[2] != 0 {
            return Err(reject());
        }
        let data = [&routed, &shared, &scalar, &combined];
        for (index, handle) in data.iter().enumerate() {
            crate::strata_prompt::validate_output(handle)?;
            let needed = count * (if index == 2 { 1 } else { 2560 }) * 4;
            if handle.size_in_used() < needed as u64 {
                return Err(reject());
            }
            for other in &data[..index] {
                if crate::strata_prompt::storage::overlaps(handle, other)? {
                    return Err(reject());
                }
            }
        }
        let views = crate::strata_prompt::groups::views()?;
        let audit = u::handle(arena, self.ids[8])?;
        let fault = u::handle(arena, fault)?;
        for handle in data {
            for owner in views.handles().into_iter().chain([&audit, &fault]) {
                if crate::strata_prompt::storage::overlaps(handle, owner)? {
                    return Err(reject());
                }
            }
        }
        let client = arena.client().map_err(u::fault)?;
        let args = KernelArguments::new()
            .with_buffers(vec![
                views.partial.binding(), views.probability.binding(),
                views.status.binding(), views.map.binding(), routed.binding(),
                audit.binding(), fault.binding(), views.raw.binding(),
                shared.binding(), scalar.binding(), combined.binding(),
            ])
            .with_info(info(&[count as u32, layer as u32, self.epoch, 0, 0, 2560, 0, 0]));
        // SAFETY: all original private owners, checked inverse upload, expert
        // role publications and shared projections precede this dispatch.
        unsafe {
            client.launch_unchecked(Box::new(Kernel::PromptOrderedSharedReduce),
                CubeCount::Static(count as u32, 1, 1), args);
        }
        self.active.as_mut().ok_or_else(reject)?[1] += 1;
        self.layer_pending = false;
        Ok(())
    }
    fn finish_data(
        &mut self,
        arena: &PrivateArena,
        layer: usize,
        output: Handle,
        fault: BufferId,
        shared: Option<Handle>,
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
        let native_shared = shared.is_some();
        let mut buffers = vec![
            partial.binding(),
            probability.binding(),
            status.binding(),
            map.binding(),
            output.binding(),
            full(self.ids[8])?.binding(),
            full(fault)?.binding(),
        ];
        if let Some(shared) = shared {
            buffers.push(shared.binding());
        }
        let args = KernelArguments::new()
            .with_buffers(buffers)
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
                } else if native_shared {
                    Kernel::SharedReduce
                } else {
                    Kernel::Reduce
                }),
                CubeCount::Static(if native_shared { 5 } else { 10 }, count as u32, 1),
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
    /// the original host rank/copy job BEFORE commit/MTP drafting. Its owned
    /// copy authority is independent; maps stay unpublished until actual join.
    pub(crate) fn begin_adaptation_after_verified(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        admission: &crate::executor::ResidentAdmission,
        sources: &mut Sources,
        cancel: &AtomicBool,
    ) -> Result<Option<Adaptation>> {
        if !self.verified_window {
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
        let usage = if crate::strata_global_adapt::collect_gpu_route_usage(true) {
            admission.readback_audited(cancel, || u::read_prefix_f32(arena, self.ids[9], 24576))?
        } else {
            // The pinned callback-free path owns CPU zero counters. No unused
            // GPU counter readback is added to its every4-window host job.
            Zeroizing::new(cache.retained_usage().to_vec())
        };
        u::cancel(cancel)?;
        let cache = self.adaptive.take().ok_or_else(reject)?;
        if !crate::strata_global_adapt::collect_gpu_route_usage(true) {
            cache.check_retained_route_map(
                &sources.retained_route_map(graph, arena.client().map_err(u::fault)?)?,
            )?;
            return Ok(Some(Adaptation::NoHost(
                crate::strata_global_adapt::SelectionFlight::new(cache, usage)?,
            )));
        }
        let raw = self.prompt_stages.ok_or_else(reject)?[2];
        let flight = sources.submit_exchange(graph, arena, raw, cache, usage)?;
        Ok(Some(Adaptation::GpuCounters(Box::new(flight))))
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
        let Some(adaptation) = adaptation else {
            return Ok(());
        };
        if self.adaptive.is_some() {
            return Err(reject());
        }
        let flight = match adaptation {
            Adaptation::NoHost(job) => {
                // Pinned callback-free mode never increments usage. Preserve
                // its empty host job/decay, without inventing GPU copies,
                // registry revocation or unchanged map/usage uploads.
                self.adaptive = Some(job.finish()?);
                return Ok(());
            }
            Adaptation::GpuCounters(flight) => *flight,
        };
        let (cache, proposal) = sources.complete_exchange(flight)?;
        self.adaptive = Some(cache);
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
        let swaps = proposal.swaps.len();
        sources.finish_publication(|| cache.commit(proposal))?;
        crate::strata_global_adapt::record_published_swaps(swaps);
        Ok(())
    }
    pub(crate) fn validate_idle(&self) -> Result<()> {
        if self.active.is_some() || self.layer_pending {
            Err(reject())
        } else {
            Ok(())
        }
    }
    pub(crate) fn validate_completed_request_boundary(
        &self,
        graph: BoundGraph<'_, '_>,
    ) -> Result<()> {
        self.validate_idle()?;
        if self.epoch == 0 || self.verified_window || self.binding != graph.binding_id() {
            return Err(reject());
        }
        graph.verify_unchanged()
    }
    /// Only completed, closed request scopes reach this hook. Keep aggregate
    /// expert counters with the original loaded model before FULL arena erase.
    /// No token, prompt, seed, suffix or KV data crosses this boundary.
    pub(crate) fn retain_completed_usage(
        &mut self,
        arena: &PrivateArena,
        admission: &crate::executor::ResidentAdmission,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_idle()?;
        if self.verified_window || admission.request_io_reuse_active() {
            return Err(reject());
        }
        let usage =
            admission.readback_audited(cancel, || u::read_prefix_f32(arena, self.ids[9], 32768))?;
        if usage[24576..].iter().any(|v| v.to_bits() != 0) {
            return Err(reject());
        }
        self.adaptive
            .as_mut()
            .ok_or_else(reject)?
            .retain_usage(&usage[..24576])
    }
    /// The request arena is still erased and inspected in FULL. Rebind only
    /// original physical routing and bounded model-local aggregate usage, with
    /// exact completed readback. No request input/output state is restored.
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
            || self.verified_window
            || admission.request_io_reuse_active()
        {
            return Err(reject());
        }
        let client = arena.client().map_err(u::fault)?;
        admission.check_client(client)?;
        let mapping = sources.retained_route_map(graph, client)?;
        let empty = Zeroizing::new(vec![0.; 24576]);
        let usage = if let Some(cache) = &self.adaptive {
            cache.check_retained_route_map(&mapping)?;
            cache.retained_usage()
        } else {
            if mapping
                .iter()
                .enumerate()
                .any(|(i, &cell)| usize::from(cell) != i % crate::strata_global_plan::EXPERTS)
            {
                return Err(reject());
            }
            &empty
        };
        let mut map_bytes = crate::strata_global_adapt::encode_route_map(&mapping)?;
        let mut usage_bytes = crate::strata_global_adapt::encode_usage(usage)?;
        let mut roots = Vec::new();
        for (id, bytes) in [
            (self.ids[7], crate::strata_global_adapt::ROUTE_MAP_BYTES),
            (self.ids[9], crate::strata_global_adapt::USAGE_BYTES),
        ] {
            let root = arena.handle(id).map_err(u::fault)?.clone();
            if arena.layout(id).map_err(u::fault)?
                != crate::private_arena::BufferLayout::new(bytes / 4).map_err(u::fault)?
                || root.size_in_used() != bytes as u64
                || root.offset_start.unwrap_or(0) != 0
                || root.offset_end.unwrap_or(0) != 0
            {
                return Err(reject());
            }
            roots.push(root);
        }
        if roots[0].clone().binding().memory.allocation_id()
            == roots[1].clone().binding().memory.allocation_id()
        {
            return Err(reject());
        }
        admission.readback_audited(cancel, || {
            client
                .write_private(roots[0].clone(), std::mem::take(&mut *map_bytes))
                .map_err(u::fault)?;
            client
                .write_private(roots[1].clone(), std::mem::take(&mut *usage_bytes))
                .map_err(u::fault)?;
            for (index, id) in [self.ids[7], self.ids[9]].into_iter().enumerate() {
                let actual = u::read_prefix_u32(arena, id, 32768)?;
                if actual.len() != 32768
                    || actual.iter().enumerate().any(|(i, &word)| {
                        let expected = if index == 0 {
                            mapping.get(i).map_or(0, |&v| u32::from(v))
                        } else {
                            usage.get(i).map_or(0, |v| v.to_bits())
                        };
                        word != expected
                    })
                {
                    return Err(reject());
                }
            }
            sources.validate_graph_ready(graph)?;
            u::cancel(cancel)
        })
    }
    pub(crate) fn set_verified_window(&mut self, enabled: bool) -> Result<()> {
        self.validate_idle()?;
        if self.verified_window == enabled {
            return Err(reject());
        }
        self.verified_window = enabled;
        Ok(())
    }
}
pub(crate) fn compiled_sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Route,
        Kernel::Plan,
        Kernel::Reduce,
        Kernel::SharedReduce,
        Kernel::PromptRoute,
        Kernel::PromptRouteFinalize,
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

/// Same Naga options, unchecked addressing and checked workgroup-layout
/// canonicalization as the genuine native graph factory. The original shared
/// multiply/add policy has no product-rounding annotation postpass.
#[cfg(any(test, feature = "strata-expert-ordered-output"))]
fn prompt_consumer_shader_words(compiled: &CompiledKernel<AutoCompiler>)
    -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>>
{
    use wgpu::naga as n;
    if compiled.source.contains("STRATA_ROUND_PRODUCTS") {
        return Err("prompt consumer changed original shared product policy".into());
    }
    let module = n::front::wgsl::parse_str(&compiled.source)?;
    let validation = n::valid::Validator::new(n::valid::ValidationFlags::all(),
        n::valid::Capabilities::all()).validate(&module)?;
    if compiled.cube_dim != CubeDim::new_1d(module.entry_points[0].workgroup_size[0]) {
        return Err("prompt consumer actual metadata dimension".into());
    }
    let words = n::back::spv::write_vec(&module,&validation,&n::back::spv::Options {
        lang_version:(1,3),
        zero_initialize_workgroup_memory:n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        bounds_check_policies:n::proc::BoundsCheckPolicies {
            index:n::proc::BoundsCheckPolicy::Unchecked,
            buffer:n::proc::BoundsCheckPolicy::Unchecked,
            image_load:n::proc::BoundsCheckPolicy::Unchecked,
            binding_array:n::proc::BoundsCheckPolicy::Unchecked,
        },
        ..Default::default()
    },Some(&n::back::spv::PipelineOptions {
        shader_stage:n::ShaderStage::Compute,entry_point:compiled.entrypoint_name.clone(),
    }))?;
    Ok(wgpu::hal::vulkan::kuno_native_workgroup_layout(&words)
        .map_err(|e|format!("prompt consumer native workgroup layout: {e}"))?)
}

/// Additional to the unchanged original429-module population.
#[cfg(feature = "strata-expert-ordered-output")]
pub(crate) fn validate_prompt_ordered_sources()
    -> std::result::Result<(), Box<dyn std::error::Error>>
{
    use std::io::Write;
    let compiled = Kernel::PromptOrderedSharedReduce.compile(
        &mut AutoCompiler::Wgsl(Default::default()),&Default::default(),
        ExecutionMode::Unchecked,StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
    )?;
    let words = prompt_consumer_shader_words(&compiled)?;
    let mut constants = std::collections::BTreeMap::new();
    let mut barriers = Vec::new();let mut multiply = 0;let mut at = 5;
    while at < words.len() {
        let n = (words[at] >> 16) as usize;let op = words[at] & 65535;
        if n == 0 || at+n > words.len() {return Err("ordered consumer opcode extent".into());}
        if op == 43 && n == 4 {constants.insert(words[at+2],words[at+3]);}
        if op == 224 {barriers.push(words[at+3]);}
        if op == 133 {multiply += 1;}
        at += n;
    }
    if barriers.len()!=2 || constants.get(&barriers[0]).is_none_or(|s|s&0x100==0)
        || constants.get(&barriers[1]).is_none_or(|s|s&0x40==0) || multiply==0
    {return Err("ordered consumer original product and snapshot barriers".into());}
    let mut validator = std::process::Command::new("spirv-val")
        .args(["--target-env","vulkan1.3","-"]).stdin(std::process::Stdio::piped()).spawn()?;
    validator.stdin.take().ok_or("ordered consumer validator stdin")?.write_all(
        &words.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>())?;
    if !validator.wait()?.success() {return Err("ordered consumer SPIRV validation".into());}
    println!("strata_prompt_ordered_shader_accepted storage_bindings=11 workgroup_barriers=1 storage_barriers=1 original_shared_policy=1");
    Ok(())
}

#[cfg(all(test, feature = "strata-intel-core"))]
mod direct_quote_tests {
    use super::*;
    #[test]
    fn direct_source_plan_prices_dense_planes_and_prompt_roots_without_decode_fetch_stages() {
        let requests = Workspace::requests_for_context(32768).unwrap();
        let secondary = crate::strata_prompt::allocation::SECONDARY_RAW_BYTES;
        assert_eq!(requests.len(), 22 + usize::from(secondary != 0));
        assert_eq!(requests[18], 512 * 1024);
        assert_eq!(
            &requests[19..22],
            &[756 * 1024 * 1024, 642 * 1024 * 1024 + crate::strata_prompt::layout::layer_extra_bytes(), 32 * 1024 * 1024]
        );
        assert!(!requests.contains(&(128 * 1024 * 1024)));
        assert_eq!(requests[..22].iter().sum::<u64>(), 1_503_494_144 + crate::strata_prompt::layout::layer_extra_bytes());
        assert_eq!(requests[22..].iter().sum::<u64>(), secondary);
        assert_eq!(requests.iter().sum::<u64>(), 1_503_494_144 + crate::strata_prompt::layout::layer_extra_bytes() + secondary);
        assert_eq!(MAIN_BYTES - secondary, 1_503_494_144 + crate::strata_prompt::layout::layer_extra_bytes());
        assert_eq!(
            3 * 128 * 1024 * 1024 - crate::strata_dense::INPUT_BYTES,
            402_128_896
        );
    }
}

#[cfg(test)]
mod route_plan_tests {
    #[test]
    fn ordered_prompt_consumer_compiles_with_snapshot_barriers_and_original_rounding() {
        use super::*;
        use wgpu::naga as n;
        let kernel = Kernel::PromptOrderedSharedReduce;
        let source = kernel.source();
        let module = n::front::wgsl::parse_str(&source).unwrap();
        let validation = n::valid::Validator::new(n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all()).validate(&module).unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
        assert_eq!(kernel.dim(), CubeDim::new_1d(256));
        let bindings = module.global_variables.iter().filter_map(|(_, g)|
            g.binding.as_ref().map(|b| b.binding)).collect::<Vec<_>>();
        assert_eq!(bindings, (0..12).collect::<Vec<_>>());
        let uses = validation.get_entry_point(0);
        for (id, global) in module.global_variables.iter() {
            if let Some(binding) = &global.binding {
                let writes = uses[id].contains(n::valid::GlobalUse::WRITE);
                assert_eq!(writes, matches!(binding.binding, 0 | 2 | 4 | 5 | 6 | 10));
            }
        }
        let compiled = kernel.compile(&mut AutoCompiler::Wgsl(Default::default()),
            &Default::default(),ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32))).unwrap();
        let words = prompt_consumer_shader_words(&compiled).unwrap();
        let policy = |words: &[u32]| {
            let mut multiply = std::collections::BTreeSet::new();
            let mut decorations = std::collections::BTreeSet::new();
            let mut barriers = 0;let mut at = 5;
            while at < words.len() {
                let count = (words[at] >> 16) as usize;let opcode = words[at] & 65535;
                if opcode == 133 { multiply.insert(words[at + 2]); }
                if opcode == 71 && count >= 3 && words[at + 2] == 42 {
                    decorations.insert(words[at + 1]);
                }
                if opcode == 224 { barriers += 1; }
                at += count;
            }
            (multiply.len(),multiply.intersection(&decorations).count(),barriers)
        };
        let old = crate::strata_prompt::Kernel::SharedMerge.compile(
            &mut AutoCompiler::Wgsl(Default::default()), &Default::default(),
            ExecutionMode::Unchecked, StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        ).unwrap();
        let old_words = prompt_consumer_shader_words(&old).unwrap();
        let new_policy = policy(&words);let old_policy = policy(&old_words);
        assert!(new_policy.0 > 0 && old_policy.0 > 0);
        assert_eq!(new_policy.1,old_policy.1);
        assert_eq!(new_policy.2,2);
        assert!(!old.source.contains("STRATA_ROUND_PRODUCTS"));
        let snapshot = source.find("valid=finite(partial[ordinal*640u].x)").unwrap();
        let barrier = source.find("storageBarrier();").unwrap();
        let normalize = source.find("status[PROMPT_ROUTES+row*10u+lane]=route_valid[lane]").unwrap();
        let sanitize = source.find("partial[ordinal*640u+column]=x").unwrap();
        assert!(snapshot < barrier && barrier < normalize && barrier < sanitize);
        assert!(source.contains("sum=fma(x,vec4<f32>(weight),sum);"));
        assert!(source.contains("weight_sum+=weight;"));
        assert!(!source.contains("if rank==0u"));
        assert!(!source.contains("STRATA_ROUND_PRODUCTS"));
        assert!(source.contains("markers[rank]==1u || markers[rank]==2u"));
    }
    #[test]
    fn prompt_route_workgroups_cover_each_token_once_without_live_tail_rows() {
        use super::*;
        for count in [1, 8, 9, 35, crate::strata_prompt::groups::CAPACITY] {
            let groups = prompt_route_groups(count).unwrap();
            let mut written = vec![0u8; count];
            for block in 0..groups as usize {
                for subgroup in 0..8 {
                    let token = block * 8 + subgroup;
                    if token < count {
                        written[token] += 1;
                    }
                }
            }
            assert!(written.iter().all(|&writes| writes == 1));
            assert!((groups as usize) * 8 >= count);
            assert!(((groups - 1) as usize) * 8 < count);
        }
        assert_eq!(prompt_route_groups(1).unwrap(), 1);
        assert_eq!(prompt_route_groups(8).unwrap(), 1);
        assert_eq!(prompt_route_groups(9).unwrap(), 2);
        assert_eq!(prompt_route_groups(4096).unwrap(), 512);
        assert!(prompt_route_groups(0).is_err());
        assert!(prompt_route_groups(crate::strata_prompt::groups::CAPACITY + 1).is_err());
    }

    #[test]
    fn prompt_route_and_checked_header_compile_with_actual_original_owner_bindings() {
        use super::*;
        use wgpu::naga as n;
        for kernel in [Kernel::PromptRoute, Kernel::PromptRouteFinalize] {
            let source = kernel.source();
            let module = n::front::wgsl::parse_str(&source).unwrap();
            let validation = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
            assert_eq!(kernel.dim(), CubeDim::new_1d(256));
            let bindings = module
                .global_variables
                .iter()
                .filter_map(|(_, g)| g.binding.as_ref().map(|b| (b.group, b.binding)))
                .collect::<Vec<_>>();
            assert_eq!(bindings, (0..7).map(|b| (0, b)).collect::<Vec<_>>());
            let uses = validation.get_entry_point(0);
            for (handle, _) in module.global_variables.iter().filter(|(_, g)| {
                g.binding
                    .as_ref()
                    .is_some_and(|b| b.binding == 2 || b.binding == 4)
            }) {
                assert!(uses[handle].contains(n::valid::GlobalUse::WRITE));
            }
            n::back::spv::write_vec(
                &module,
                &validation,
                &n::back::spv::Options::default(),
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
        }
        let routes = Kernel::PromptRoute.source();
        assert!(routes.contains("let token=block.x*8u+sg;let live=token<p.count;"));
        assert!(!routes.contains("first+=8u"));
        assert!(!routes.contains("map[0]="));
        let header = Kernel::PromptRouteFinalize.source();
        let fault_gate = header
            .find("if atomicLoad(&fault[0])!=0u {return;}")
            .unwrap();
        let publication = header.find("map[0]=p.count;").unwrap();
        assert!(fault_gate < publication);
    }

    #[test]
    fn actual_joined_vector_combine_has_native_geometry_and_binds_the_original_shared_result() {
        use super::*;
        use wgpu::naga as n;
        let kernel = Kernel::SharedReduce;
        let source = kernel.source();
        let module = n::front::wgsl::parse_str(&source).unwrap();
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
        assert_eq!(kernel.dim(), CubeDim::new_1d(128));
        let uses = info.get_entry_point(0);
        let mut owners = std::collections::BTreeMap::new();
        for (id, g) in module.global_variables.iter() {
            if let Some(b) = &g.binding {
                owners.insert(
                    b.binding,
                    (
                        uses[id].contains(n::valid::GlobalUse::READ),
                        uses[id].contains(n::valid::GlobalUse::WRITE),
                    ),
                );
            }
        }
        assert_eq!(owners.len(), 9);
        for binding in [0, 1, 2, 3, 7] {
            assert_eq!(owners[&binding], (true, false));
        }
        for binding in [4, 5, 6] {
            assert!(owners[&binding].1);
        }
        assert!(
            source.contains("if rank==0u {sum=x*weight;}else{sum=fma(x,vec4<f32>(weight),sum);}")
        );
        assert!(source.contains("let sh=shared_result[row*640u+id.x];sum+=sh;"));
        n::back::spv::write_vec(
            &module,
            &info,
            &n::back::spv::Options::default(),
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
    }

    use super::*;

    #[test]
    fn selected_route_and_plan_compile_with_original_intel_launch_geometry() {
        use wgpu::naga as n;
        for (kernel, dimension) in [(Kernel::Route, 256), (Kernel::Plan, 128)] {
            let source = kernel.source();
            let module = n::front::wgsl::parse_str(&source).unwrap();
            let validation = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [dimension, 1, 1]);
            assert_eq!(kernel.dim(), CubeDim::new_1d(dimension));
            let bindings = module
                .global_variables
                .iter()
                .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
                .collect::<Vec<_>>();
            assert_eq!(bindings, (0..7).map(|b| (0, b)).collect::<Vec<_>>());
            n::back::spv::write_vec(
                &module,
                &validation,
                &n::back::spv::Options::default(),
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
        }
        // The last80 logical route words and the existing fixed8 group planes
        // share the original full root without a new allocation or overlap.
        assert_eq!(ROOT_BYTES[0] / 4, 8192);
        assert!(164 + 80 * 8 <= 8112);
        assert_eq!(8112 + 80, 8192);
    }

    #[test]
    fn packed_parallel_plan_matches_independent_stable_groups_across_all_warp_boundaries() {
        for count in 1..=8 {
            for pattern in 0..4 {
                let logical = (0..count * 10)
                    .map(|i| match pattern {
                        0 => i % 10,
                        1 => i,
                        2 => ((i / 10) * 7 + (i % 10) * 37) % 512,
                        _ => 511 - (((i / 10) * 83 + (i % 10) * 13) % 512),
                    })
                    .collect::<Vec<_>>();
                for remapping in 0..3 {
                    let physical = |id: usize| match remapping {
                        0 => id,
                        1 => (id * 13 + 17) % 512,
                        _ => 511 - id,
                    };
                    // Independent scalar oracle: groups by first occurrence,
                    // members in original route order, no scan or packed sums.
                    let mut oracle = Vec::<(usize, Vec<usize>)>::new();
                    for (route, &id) in logical.iter().enumerate() {
                        if let Some((_, members)) = oracle.iter_mut().find(|(e, _)| *e == id) {
                            members.push(route);
                        } else {
                            oracle.push((id, vec![route]));
                        }
                    }
                    let mut first = [0usize; 128];
                    let mut ranks = [0usize; 128];
                    let mut lengths = [0usize; 128];
                    let mut packed = [0u32; 128];
                    for i in 0..logical.len() {
                        first[i] = logical.iter().position(|&e| e == logical[i]).unwrap();
                        ranks[i] = logical[..i].iter().filter(|&&e| e == logical[i]).count();
                        lengths[i] = logical.iter().filter(|&&e| e == logical[i]).count();
                        if first[i] == i {
                            packed[i] = ((lengths[i] as u32) << 16) | 1;
                        }
                    }
                    let mut exclusive = [0u32; 128];
                    let mut sums = [0u32; 4];
                    for warp in 0..4 {
                        let mut prefix: [u32; 32] =
                            packed[warp * 32..warp * 32 + 32].try_into().unwrap();
                        for offset in [1, 2, 4, 8, 16] {
                            let previous = prefix;
                            for lane in offset..32 {
                                prefix[lane] += previous[lane - offset];
                            }
                        }
                        sums[warp] = prefix[31];
                        for lane in 0..32 {
                            exclusive[warp * 32 + lane] = prefix[lane] - packed[warp * 32 + lane];
                        }
                    }
                    let mut actual = [0u32; 8192];
                    let mut writes = [false; 8192];
                    for (i, &id) in logical.iter().enumerate() {
                        actual[8112 + i] = id as u32;
                    }
                    let mut write = |at: usize, value: usize| {
                        assert!(!writes[at], "overlapping parallel output write {at}");
                        writes[at] = true;
                        actual[at] = value as u32;
                    };
                    for i in 0..logical.len() {
                        let total = exclusive[first[i]] + sums[..first[i] / 32].iter().sum::<u32>();
                        let group = (total & 65535) as usize;
                        assert!(ranks[i] < count && group < logical.len());
                        write(164 + group * 8 + ranks[i], i);
                        if first[i] == i {
                            write(4 + group, physical(logical[i]));
                            write(84 + group, lengths[i]);
                        }
                    }
                    let total = sums.iter().sum::<u32>();
                    assert_eq!(total >> 16, logical.len() as u32);
                    assert_eq!((total & 65535) as usize, oracle.len());
                    for (group, (id, members)) in oracle.iter().enumerate() {
                        assert_eq!(actual[4 + group], physical(*id) as u32);
                        assert_eq!(actual[84 + group], members.len() as u32);
                        for (rank, &route) in members.iter().enumerate() {
                            assert_eq!(actual[164 + group * 8 + rank], route as u32);
                        }
                    }
                    assert!(writes[164 + 80 * 8..].iter().all(|&v| !v));
                    assert_eq!(
                        &actual[8112..8112 + logical.len()],
                        logical.iter().map(|&v| v as u32).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
