//! Fixed-fixture laboratory whole-token owner. No HTTP/private request support.
//! GR, GDN, QSA and all-ten routed MoE stay on the device; projected PLE
//! uses the exact Rust host oracle. Any partially queued failure destroys the
//! entire sequence. Owned runtime I/O is opt-in pending both-GPU acceptance.
use crate::{
    executor::ResidentAdmission,
    private_arena::{BufferId, BufferLayout, PrivateArena},
    resident_gdn::{PendingGdn, PublicGdnParameters, ResidentGdn, SequenceKey},
    resident_gr::{GrWorkspace, PublicVectors, ResidentProjection, project},
    resident_util as u,
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::model::{AttentionKind, BoundGraph};
use kuno_text::ops::{self, CausalConvState, PleTokenHistory};
use kuno_text::{Error, Result};
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;
use zeroize::{Zeroize, Zeroizing};

// Candidate MAX8 shared expert; scalar execution below remains unchanged.
#[path = "resident_shared_batch.rs"]
pub(crate) mod shared_batch;

#[path = "resident_ple_batch.rs"]
pub(crate) mod ple_batch;

#[path = "resident_model_batch.rs"]
pub(crate) mod batch;

#[path = "resident_batch_io.rs"]
pub(crate) mod batch_io;

#[path = "resident_lifecycle_hooks.rs"]
mod lifecycle_hooks;
#[path = "resident_model_lifecycle_probe.rs"]
pub(crate) mod lifecycle_probe;
#[path = "resident_numeric_boundaries.rs"]
pub(crate) mod numeric_boundaries;

enum Layer<'f> {
    Gdn(Box<ResidentGdn<'f>>, PublicGdnParameters),
    Qsa(
        Box<crate::resident_qsa::ResidentQsa<'f>>,
        crate::resident_qsa::PublicQsaParameters,
    ),
}
struct Buffers {
    residual: [BufferId; 2],
    block: BufferId,
    router: BufferId,
    router_packet: BufferId,
    logits: BufferId,
    logit_packet: BufferId,
    fault: BufferId,
    ple_emb: BufferId,
    ple_key: BufferId,
    ple_value: BufferId,
    shared_gate: BufferId,
    shared_up: BufferId,
    shared_act: BufferId,
    shared_down: BufferId,
    shared_scale: BufferId,
}
pub struct ResidentModel<'m, 'f> {
    graph: BoundGraph<'m, 'f>,
    arena: PrivateArena,
    io: PrivateArena,
    public: PublicVectors,
    gr: GrWorkspace,
    native_batch: Option<batch::Workspace>,
    layers: Vec<Layer<'f>>,
    moe_layers: Vec<crate::resident_moe::ResidentMoe>,
    pending_qsa: Vec<(usize, crate::resident_qsa::PendingQsa)>,
    pending_moe: Vec<(usize, crate::resident_moe::PendingMoe)>,
    b: Buffers,
    ple_history: PleTokenHistory,
    ple_conv: CausalConvState,
    ple_batch_pending: Option<ple_batch::Anchor>,
    ple_batch_poison: Option<Arc<AtomicBool>>,
    position: usize,
    completed_cpu_misses: Zeroizing<u64>,
    context: usize,
    poisoned: bool,
    retired: Arc<AtomicBool>,
    admission: ResidentAdmission,
    cancel_after_attention: Option<usize>,
    lifecycle_hooks: lifecycle_hooks::Hooks,
    numeric_boundary: numeric_boundaries::Collector,
    pub phase_ns: BTreeMap<&'static str, u128>,
}
#[cube]
#[allow(unused_assignments)]
fn sigmoid(x: f32) -> f32 {
    let mut result = 0.;
    if x >= 0. {
        result = 1. / (1. + (-x).exp());
    } else {
        let e = x.exp();
        result = e / (1. + e);
    }
    result
}
#[cube(launch)]
fn shared_activation(gate: &Array<f32>, up: &Array<f32>, output: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = gate[i] * sigmoid(gate[i]) * up[i];
    }
}
#[cube(launch)]
fn add_shared(
    routed: &Array<f32>,
    shared: &Array<f32>,
    scale: &Array<f32>,
    output: &mut Array<f32>,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = routed[i] + shared[i] * scale[0];
    }
}
#[cube(launch)]
fn shared_gate_dot(
    input: &Array<f32>,
    weight: &Array<f32>,
    scale: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let lane = UNIT_POS_X as usize;
    let mut sum = 0.0_f32;
    for tile in 0..input.len().div_ceil(64_usize) {
        let i = tile * 64 + lane;
        if i < input.len() {
            sum += input[i] * weight[i];
        }
    }
    let mut sums = SharedMemory::<f32>::new(64_usize);
    sums[lane] = sum;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            sums[lane] += sums[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        let value = sigmoid(sums[0]);
        if sums[0] <= 3.4028235e38_f32 && sums[0] >= -3.4028235e38_f32 {
            scale[0] = value;
        } else {
            scale[0] = 0.;
            fault[0].fetch_or(4);
        }
    }
}
/// Reuses every constructor's public layout, including full owned padding.
/// GR/final scratch follows the fixed Qwen graph's existing allocation order.
fn private_requests(graph: BoundGraph<'_, '_>) -> Result<Vec<u64>> {
    let g = graph.geometry();
    let mut words = Vec::new();
    for layer in 0..g.layers {
        match graph.layer_kind(layer)? {
            AttentionKind::Gdn => {
                let layout = crate::resident_gdn::StageLayout::from_graph(graph, layer)?;
                words.extend(layout.allocation_words());
                words.extend(crate::resident_gdn::batch::strata_replay::allocation_words(
                    layout,
                ));
            }
            AttentionKind::Qsa => words.extend(
                crate::resident_qsa::StageLayout::from_graph(graph, layer)?.allocation_words(),
            ),
        }
    }
    for layer in 0..g.layers {
        let l = crate::resident_moe::MoeLayout::from_graph(graph, layer)?;
        words.extend([10, 10, l.activation_words, l.partial_words, l.status_words]);
    }
    let r = crate::residual_gpu::ResidualLayout::new(g.residual, g.low_rank).map_err(u::fault)?;
    words.extend(workspace_words(r, g.vocabulary)?);
    words
        .into_iter()
        .map(|n| Ok(BufferLayout::new(n).map_err(u::fault)?.owned_bytes as u64))
        .collect()
}
fn workspace_words(
    r: crate::residual_gpu::ResidualLayout,
    vocabulary: usize,
) -> Result<Vec<usize>> {
    let mut words = vec![
        r.shape.width,
        r.shape.streams,
        r.residual_elements,
        r.low_rank,
        r.low_rank,
        r.residual_elements,
        r.shape.streams,
        r.low_rank,
        r.shape.width,
        r.residual_elements,
    ];
    words.extend([
        10240,
        10240,
        2560,
        512,
        513,
        vocabulary,
        vocabulary.checked_add(1).ok_or(Error::ArithmeticOverflow)?,
        1,
        2560,
        10240,
        2560,
        640,
        640,
        640,
        2560,
        1,
    ]);
    Ok(words)
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum StepPurpose {
    Logits,
    Advance,
}
fn completed_output(
    purpose: StepPurpose,
    logits: impl FnOnce() -> Result<Zeroizing<Vec<f32>>>,
    advance: impl FnOnce() -> Result<()>,
) -> Result<Zeroizing<Vec<f32>>> {
    match purpose {
        StepPurpose::Logits => logits(),
        StepPurpose::Advance => {
            advance()?;
            Ok(Zeroizing::new(Vec::new()))
        }
    }
}
/// State commits can run only after an accepted completed readback. A late
/// cancellation still rejects the whole token through the existing poison path.
fn commit_after_fence<T>(
    fence: Result<T>,
    cancel: &AtomicBool,
    commit: impl FnOnce() -> Result<()>,
) -> Result<T> {
    let value = fence?;
    u::cancel(cancel)?;
    commit()?;
    u::cancel(cancel)?;
    Ok(value)
}

fn admitted_upload(
    admission: &ResidentAdmission,
    io: &mut PrivateArena,
    arena: &PrivateArena,
    target: BufferId,
    values: &[f32],
) -> Result<()> {
    let page = BufferLayout::new(values.len())
        .map_err(u::fault)?
        .owned_bytes as u64;
    admission.allocate(&[page], || u::upload_f32(io, arena, target, values))
}
struct AdmittedSelector<'a> {
    inner: u::TransientSelector<'a>,
    admission: ResidentAdmission,
}
impl crate::resident_qsa::ResidentQsaSelector for AdmittedSelector<'_> {
    fn select_transient(
        &mut self,
        arena: &PrivateArena,
        layout: crate::qsa_gpu::QsaLayout,
        scores: crate::resident_gdn::StatusRegion,
        status: crate::resident_gdn::StatusRegion,
        cancel: &AtomicBool,
    ) -> Result<crate::resident_qsa::PrivateSelection> {
        // The original selector completes both score/status reads before this
        // guard samples/reclaims; no added numerical-success assumption.
        self.admission.readback_intermediate(|| {
            self.inner
                .select_transient(arena, layout, scores, status, cancel)
        })
    }
    fn upload_selected(
        &mut self,
        arena: &PrivateArena,
        target: BufferId,
        selection: &crate::resident_qsa::PrivateSelection,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let page = BufferLayout::new(u::words(arena, target)?)
            .map_err(u::fault)?
            .owned_bytes as u64;
        self.admission.allocate(&[page], || {
            self.inner.upload_selected(arena, target, selection, cancel)
        })
    }
}

impl<'m, 'f> ResidentModel<'m, 'f> {
    pub fn new(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut crate::executor::GpuExecutor<'f>,
    ) -> Result<Self> {
        Self::construct(graph, client, context, executor, false)
    }
    /// Counts-only diagnostic for compiled fixed-public fixtures. Ordinary
    /// ownership, polling, numerical work and scheduling remain the default.
    pub fn new_with_public_owned_zero_audit(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut crate::executor::GpuExecutor<'f>,
        public_zero_audit: bool,
    ) -> Result<Self> {
        if !public_zero_audit {
            return Self::new(graph, client, context, executor);
        }
        Self::construct(graph, client, context, executor, true)
    }
    fn construct(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut crate::executor::GpuExecutor<'f>,
        public_zero_audit: bool,
    ) -> Result<Self> {
        Self::construct_native(graph, client, context, executor, public_zero_audit, None)
    }
    fn construct_native(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut crate::executor::GpuExecutor<'f>,
        public_zero_audit: bool,
        native: Option<batch::Config>,
    ) -> Result<Self> {
        let g = graph.geometry();
        if context == 0 || context > g.max_context {
            return Err(Error::LimitExceeded("resident context"));
        }
        executor.check_client(&client)?;
        let mut requests = private_requests(graph)?;
        let native = native
            .map(|config| batch::Plan::new(graph, &client, executor, config, &requests))
            .transpose()?;
        if let Some(plan) = &native {
            requests.extend_from_slice(&plan.requests);
        }
        let private_owned = requests.iter().try_fold(0_u64, |sum, &n| {
            sum.checked_add(n).ok_or(Error::ArithmeticOverflow)
        })?;
        let public_pages = PublicVectors::prewarm_page_bytes(graph)?;
        if let Some(plan) = &native {
            plan.check_public_with_audit(graph, public_pages, public_zero_audit)?;
        }
        let admission = executor.begin_resident(graph, private_owned, public_pages)?;
        if (public_zero_audit || native.is_some()) && !admission.transient_io() {
            admission.poison();
            return Err(Error::InvalidFormat(
                "owned zero audit requires transient resident I/O",
            ));
        }
        let mut public = PublicVectors::new(client.clone(), graph.binding_id());
        admission.prewarm(graph, &mut public)?;
        if public.full_page_bytes() != public_pages {
            admission.poison();
            return Err(Error::InvalidFormat("resident public page plan"));
        }
        admission.allocate_model(&requests, |allocation| {
            let retired = admission.retired_context();
            let make_arena = |budget| {
                if admission.transient_io() {
                    // The outer fixed-stream runner configured ONCE before
                    // all uniforms/readbacks. Never reconfigure on rebinding.
                    PrivateArena::new_transient_on_configured(
                        client.clone(),
                        budget,
                        retired.clone(),
                    )
                } else {
                    PrivateArena::new(client.clone(), budget, retired.clone())
                }
            };
            let mut arena = make_arena(2 * 1024 * 1024 * 1024).map_err(u::fault)?;
            if public_zero_audit {
                arena.enable_public_owned_zero_audit().map_err(u::fault)?;
            }
            let mut io = if admission.transient_io() {
                PrivateArena::new_transient_source_on_configured(
                    client.clone(),
                    crate::executor::RESIDENT_IO_MAX as usize,
                    retired.clone(),
                )
            } else {
                PrivateArena::new(
                    client.clone(),
                    crate::executor::RESIDENT_IO_MAX as usize,
                    retired.clone(),
                )
            }
            .map_err(u::fault)?;
            if public_zero_audit {
                io.enable_public_owned_zero_audit().map_err(u::fault)?;
            }
            let key = SequenceKey::new()?;
            let mut layers = Vec::new();
            for l in 0..g.layers {
                layers.push(match graph.layer_kind(l)? {
                    AttentionKind::Gdn => {
                        let params = PublicGdnParameters::prepare(graph, l, &mut public)?;
                        let stage = ResidentGdn::allocate(graph, l, key, &mut arena, &params)?;
                        if stage.layout().max_context < context {
                            return Err(Error::LimitExceeded("resident stage context"));
                        }
                        Layer::Gdn(Box::new(stage), params)
                    }
                    AttentionKind::Qsa => {
                        let p = crate::resident_qsa::PublicQsaParameters::prepare(
                            graph,
                            l,
                            &mut public,
                        )?;
                        let stage = crate::resident_qsa::ResidentQsa::allocate(
                            graph, l, key, &mut arena, &p,
                        )?;
                        if stage.layout().max_context < context {
                            return Err(Error::LimitExceeded("resident attention context"));
                        }
                        Layer::Qsa(Box::new(stage), p)
                    }
                });
            }
            let mut moe_layers = Vec::new();
            for l in 0..g.layers {
                moe_layers.push(crate::resident_moe::ResidentMoe::allocate(
                    graph, l, &mut arena,
                )?);
            }
            let gr = GrWorkspace::allocate(graph, &mut arena)?;
            let mut alloc = |n| arena.allocate_zeroed(n).map_err(u::fault);
            let b = Buffers {
                residual: [alloc(10240)?, alloc(10240)?],
                block: alloc(2560)?,
                router: alloc(512)?,
                router_packet: alloc(513)?,
                logits: alloc(g.vocabulary)?,
                logit_packet: alloc(g.vocabulary + 1)?,
                fault: alloc(1)?,
                ple_emb: alloc(2560)?,
                ple_key: alloc(10240)?,
                ple_value: alloc(2560)?,
                shared_gate: alloc(640)?,
                shared_up: alloc(640)?,
                shared_act: alloc(640)?,
                shared_down: alloc(2560)?,
                shared_scale: alloc(1)?,
            };
            let ple_history = PleTokenHistory::new(graph.ple_hash());
            let ple_conv = CausalConvState::new(10240, 4, 3).map_err(u::fault)?;
            let mut model = Self {
                graph,
                arena,
                io,
                public,
                gr,
                native_batch: None,
                layers,
                moe_layers,
                pending_qsa: Vec::new(),
                pending_moe: Vec::new(),
                b,
                ple_history,
                ple_conv,
                ple_batch_pending: None,
                ple_batch_poison: None,
                position: 0,
                completed_cpu_misses: Zeroizing::new(0),
                context,
                poisoned: false,
                retired,
                admission: admission.clone(),
                cancel_after_attention: None,
                lifecycle_hooks: lifecycle_hooks::Hooks::default(),
                numeric_boundary: numeric_boundaries::Collector::default(),
                phase_ns: BTreeMap::new(),
            };
            if let Some(plan) = native {
                model.native_batch = Some(batch::Workspace::allocate(
                    &mut model,
                    executor,
                    key,
                    plan.config,
                    allocation,
                )?);
            }
            if model.arena.charged_bytes() as u64 != private_owned {
                return Err(Error::InvalidFormat("resident private page plan"));
            }
            Ok(model)
        })
    }
    pub fn arm_public_mid_token_cancel(&mut self, layer: usize) -> Result<()> {
        if layer >= self.layers.len() {
            return Err(Error::OutOfBounds);
        }
        self.cancel_after_attention = Some(layer);
        Ok(())
    }
    fn record(&mut self, key: &'static str, start: Instant) {
        *self.phase_ns.entry(key).or_default() += start.elapsed().as_nanos();
    }
    pub fn erased_after_failure(&self) -> bool {
        self.poisoned
            && self.layers.is_empty()
            && self.moe_layers.is_empty()
            && self.arena.charged_bytes() == 0
            && self.io.charged_bytes() == 0
            && !self.retired.load(Ordering::Acquire)
            && self.ple_history.tokens().iter().all(|x| *x == -1)
            && self.ple_conv.history().iter().all(|x| *x == 0.)
    }
    /// Available only after BOTH arenas accepted explicit close. Never use
    /// Drop telemetry or a partial audit as a whole-owner erasure receipt.
    pub fn public_owned_zero_audit_receipt(
        &self,
    ) -> Option<(
        crate::private_arena::OwnedZeroAudit,
        crate::private_arena::OwnedZeroAudit,
    )> {
        if !self.erased_after_failure() {
            return None;
        }
        Some((
            self.arena.owned_zero_audit_receipt()?,
            self.io.owned_zero_audit_receipt()?,
        ))
    }
    pub fn owned_bytes(&self) -> usize {
        self.arena.charged_bytes()
    }
    /// Explicit checked full-span cleanup for the fixed public acceptance
    /// runner. Failure retains arena ownership and retires the shared context.
    /// It never resets/reuses the admission ledger or emits a clean receipt.
    pub fn close(&mut self) -> Result<()> {
        self.poisoned = true;
        self.completed_cpu_misses.zeroize();
        if let Some(poison) = &self.ple_batch_poison {
            poison.store(true, Ordering::Release);
        }
        self.ple_batch_pending = None;
        self.native_batch = None; // Sticky-only workspace Drop; no admission callback.
        self.admission.poison();
        // Explicit close only: no callback from Drop/constructor guards into
        // admission. Cancelled short QSA can leave up to40MiB of ordered source
        // staging live; checked drain/erase must precede full MAIN inspection.
        let request_io = self.admission.end_request_io_reuse();
        let sources = self.admission.cleanup_abandoned_ordered_sources();
        let io = self.io.close();
        let all = self.arena.close();
        self.layers.clear();
        for m in &mut self.moe_layers {
            m.poison();
        }
        self.moe_layers.clear();
        self.pending_moe.clear();
        self.pending_qsa.clear();
        self.cancel_after_attention = None;
        self.ple_history.reset();
        self.ple_conv.reset();
        if request_io.is_err()
            || sources.is_err()
            || io.is_err()
            || all.is_err()
            || self.retired.load(Ordering::Acquire)
        {
            self.retired.store(true, Ordering::Release);
            return Err(Error::InvalidFormat("resident full-span cleanup failed"));
        }
        Ok(())
    }
    pub fn step(
        &mut self,
        executor: &mut crate::executor::GpuExecutor<'f>,
        token: u32,
        cancel: &AtomicBool,
    ) -> Result<Zeroizing<Vec<f32>>> {
        self.step_for(executor, token, cancel, StepPurpose::Logits)
    }
    /// Explicitly advance a prompt token whose logits will not be sampled.
    /// Every layer and final HC combine still runs and accepts its full fault
    /// fence before state/cursor commits. No logits are exposed by this API.
    pub fn advance_without_logits(
        &mut self,
        executor: &mut crate::executor::GpuExecutor<'f>,
        token: u32,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.step_for(executor, token, cancel, StepPurpose::Advance)
            .map(drop)
    }
    fn step_for(
        &mut self,
        executor: &mut crate::executor::GpuExecutor<'f>,
        token: u32,
        cancel: &AtomicBool,
        purpose: StepPurpose,
    ) -> Result<Zeroizing<Vec<f32>>> {
        if self.poisoned
            || self.ple_batch_pending.is_some()
            || self
                .ple_batch_poison
                .as_ref()
                .is_some_and(|p| p.load(Ordering::Acquire))
            || self.position >= self.context
            || token as usize >= self.graph.geometry().vocabulary
        {
            return Err(Error::InvalidFormat("invalid resident sequence"));
        }
        self.poisoned = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.step_checked(executor, token, cancel, purpose)
        }));
        match result {
            Ok(Ok(logits)) => {
                self.position += 1;
                self.poisoned = false;
                Ok(logits)
            }
            other => {
                executor.poison_resident();
                self.admission.poison();
                for layer in &mut self.layers {
                    match layer {
                        Layer::Gdn(g, _) => g.poison(),
                        Layer::Qsa(q, _) => q.poison(),
                    }
                }
                let _ = self.close();
                match other {
                    Ok(Err(e)) => Err(e),
                    _ => Err(Error::InvalidFormat("resident graph execution failed")),
                }
            }
        }
    }
    fn step_checked(
        &mut self,
        executor: &mut crate::executor::GpuExecutor<'f>,
        token: u32,
        cancel: &AtomicBool,
        purpose: StepPurpose,
    ) -> Result<Zeroizing<Vec<f32>>> {
        executor.check_client(self.arena.client().map_err(u::fault)?)?;
        executor.check_resident_admission(&self.admission)?;
        self.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        u::zero(&self.arena, self.b.fault)?;
        let mut embedding = Zeroizing::new(vec![0.; 2560]);
        self.graph
            .tensor("token_embd.weight")?
            .decode_row(token as u64, &mut embedding)?;
        let residual = Zeroizing::new(embedding.repeat(4));
        admitted_upload(
            &self.admission,
            &mut self.io,
            &self.arena,
            self.b.residual[0],
            &residual,
        )?;
        let mut current = 0;
        let mut pending: Vec<(usize, PendingGdn)> = Vec::new();
        for l in 0..self.layers.len() {
            u::cancel(cancel)?;
            let start = Instant::now();
            if l == 1 {
                self.numeric_boundary.observe(
                    numeric_boundaries::Boundary::Input,
                    &self.arena,
                    &self.admission,
                    self.b.residual[current],
                    self.b.fault,
                    10240,
                    self.position,
                    1,
                    cancel,
                )?;
                self.ple(executor, token, self.b.residual[current], cancel)?;
                self.numeric_boundary.observe(
                    numeric_boundaries::Boundary::Injected,
                    &self.arena,
                    &self.admission,
                    self.b.residual[current],
                    self.b.fault,
                    10240,
                    self.position,
                    1,
                    cancel,
                )?;
            }
            self.record("ple_and_previous_owner", start);
            let start = Instant::now();
            let mut projector = ResidentProjection {
                executor,
                fault: self.b.fault,
            };
            self.gr.read(
                self.graph,
                &self.arena,
                &mut self.public,
                &mut projector,
                &format!("blk.{l}.hc_attn"),
                self.b.residual[current],
                true,
                self.b.fault,
                cancel,
            )?;
            self.record("attention_gr_enqueue", start);
            let start = Instant::now();
            match &mut self.layers[l] {
                Layer::Gdn(g, p) => {
                    if g.requires_erasure_if_abandoned() {
                        return Err(Error::InvalidFormat("uncommitted resident layer"));
                    }
                    let receipt = g.enqueue(
                        self.graph,
                        &self.arena,
                        p,
                        &mut projector,
                        self.gr.mixed,
                        self.b.block,
                        self.position as u64,
                        cancel,
                    )?;
                    if receipt.output() != self.b.block
                        || receipt.position() != self.position as u64
                    {
                        return Err(Error::InvalidFormat("resident receipt mismatch"));
                    }
                    for region in receipt.statuses() {
                        u::status(&self.arena, region.buffer, self.b.fault)?;
                    }
                    pending.push((l, receipt));
                }
                Layer::Qsa(q, p) => {
                    if q.requires_erasure_if_abandoned() {
                        return Err(Error::InvalidFormat("uncommitted resident attention"));
                    }
                    let mut selector = AdmittedSelector {
                        inner: u::TransientSelector { io: &mut self.io },
                        admission: self.admission.clone(),
                    };
                    let receipt = q.enqueue(
                        self.graph,
                        &self.arena,
                        p,
                        &mut projector,
                        &mut selector,
                        self.gr.mixed,
                        self.b.block,
                        self.position as u64,
                        cancel,
                    )?;
                    if receipt.output() != self.b.block
                        || receipt.position() != self.position as u64
                    {
                        return Err(Error::InvalidFormat("resident attention receipt"));
                    }
                    for r in receipt.statuses() {
                        u::status_prefix(&self.arena, r.buffer, r.words, self.b.fault)?;
                    }
                    self.pending_qsa.push((l, receipt));
                }
            }
            if self.cancel_after_attention == Some(l) {
                self.cancel_after_attention = None;
                cancel.store(true, Ordering::Release);
            }
            self.record("attention_enqueue_and_host_fences", start);
            let start = Instant::now();
            self.gr.combine(
                &self.arena,
                self.b.residual[current],
                self.b.block,
                self.b.residual[1 - current],
                self.b.fault,
            )?;
            current = 1 - current;
            let mut projector = ResidentProjection {
                executor,
                fault: self.b.fault,
            };
            self.gr.read(
                self.graph,
                &self.arena,
                &mut self.public,
                &mut projector,
                &format!("blk.{l}.hc_ffn"),
                self.b.residual[current],
                true,
                self.b.fault,
                cancel,
            )?;
            self.record("attention_combine_and_ffn_gr_enqueue", start);
            let start = Instant::now();
            self.moe(executor, l, cancel)?;
            self.record("moe_route_and_upload_enqueue", start);
            self.gr.combine(
                &self.arena,
                self.b.residual[current],
                self.b.block,
                self.b.residual[1 - current],
                self.b.fault,
            )?;
            current = 1 - current;
        }
        let fence = completed_output(
            purpose,
            || {
                let mut projector = ResidentProjection {
                    executor,
                    fault: self.b.fault,
                };
                self.gr.read(
                    self.graph,
                    &self.arena,
                    &mut self.public,
                    &mut projector,
                    "output_hc",
                    self.b.residual[current],
                    false,
                    self.b.fault,
                    cancel,
                )?;
                project(
                    self.graph,
                    &mut projector,
                    &self.arena,
                    "output.weight",
                    self.gr.mixed,
                    self.b.logits,
                    cancel,
                )?;
                self.admission.readback(|| {
                    u::read_fence(
                        &self.arena,
                        self.b.logits,
                        self.b.fault,
                        self.b.logit_packet,
                    )
                })
            },
            || {
                // GrWorkspace::combine already checks every final residual
                // element and ORs all status failures into b.fault. Reading that
                // word completes those commands without the output head.
                self.admission
                    .readback(|| u::read_fault_fence(&self.arena, self.b.fault))
            },
        );
        commit_after_fence(fence, cancel, || {
            for (l, p) in pending {
                if let Layer::Gdn(g, _) = &mut self.layers[l] {
                    g.commit_after_whole_token_acceptance(p, &self.arena, cancel)?;
                } else {
                    return Err(Error::InvalidFormat("resident layer identity"));
                }
            }
            for (l, p) in self.pending_qsa.drain(..) {
                if let Layer::Qsa(q, _) = &mut self.layers[l] {
                    q.commit_after_whole_token_acceptance(p, &self.arena, cancel)?;
                } else {
                    return Err(Error::InvalidFormat("resident attention identity"));
                }
            }
            for (l, p) in self.pending_moe.drain(..) {
                self.moe_layers[l].accept_after_whole_token_acceptance(p)?;
            }
            self.io.reset().map_err(u::fault)?;
            Ok(())
        })
    }
    fn ple(
        &mut self,
        executor: &mut crate::executor::GpuExecutor<'f>,
        token: u32,
        residual: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let rows = Zeroizing::new(
            self.graph
                .ple_hash()
                .rows_and_advance(token as i32, &mut self.ple_history)
                .map_err(u::fault)?,
        );
        let mut emb = Zeroizing::new(vec![0.; 2560]);
        for (row, out) in rows.iter().zip(emb.chunks_exact_mut(160)) {
            self.graph
                .tensor("per_layer_token_embd.weight")?
                .decode_row(*row as u64, out)?;
        }
        admitted_upload(
            &self.admission,
            &mut self.io,
            &self.arena,
            self.b.ple_emb,
            &emb,
        )?;
        let mut p = ResidentProjection {
            executor,
            fault: self.b.fault,
        };
        project(
            self.graph,
            &mut p,
            &self.arena,
            "blk.1.ple_key.weight",
            self.b.ple_emb,
            self.b.ple_key,
            cancel,
        )?;
        project(
            self.graph,
            &mut p,
            &self.arena,
            "blk.1.ple_value.weight",
            self.b.ple_emb,
            self.b.ple_value,
            cancel,
        )?;
        for (boundary, id, words) in [
            (
                numeric_boundaries::Boundary::ProjectedKey,
                self.b.ple_key,
                10240,
            ),
            (
                numeric_boundaries::Boundary::ProjectedValue,
                self.b.ple_value,
                2560,
            ),
        ] {
            self.numeric_boundary.observe(
                boundary,
                &self.arena,
                &self.admission,
                id,
                self.b.fault,
                words,
                self.position,
                1,
                cancel,
            )?;
        }
        let (key, value, original) = self.admission.readback_intermediate(|| {
            Ok((
                u::read_f32(&self.arena, self.b.ple_key)?,
                u::read_f32(&self.arena, self.b.ple_value)?,
                u::read_f32(&self.arena, residual)?,
            ))
        })?;
        let mut result = ops::ple_inject_projected(
            &key,
            &value,
            &original,
            self.graph.static_vector("blk.1.ple_norm_key.weight")?,
            self.graph.static_vector("blk.1.ple_norm_query.weight")?,
            self.graph.static_vector("blk.1.ple_norm_conv.weight")?,
            self.graph.static_vector("blk.1.ple_conv1d.weight")?,
            &mut self.ple_conv,
            self.graph.geometry().residual,
            self.graph.geometry().epsilon,
        )
        .map_err(u::fault)?;
        let uploaded = admitted_upload(
            &self.admission,
            &mut self.io,
            &self.arena,
            residual,
            &result.residual,
        );
        result.residual.zeroize();
        result.gates.zeroize();
        result.normalized.zeroize();
        uploaded
    }
    fn moe(
        &mut self,
        executor: &mut crate::executor::GpuExecutor<'f>,
        l: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let base = format!("blk.{l}.");
        let mut p = ResidentProjection {
            executor,
            fault: self.b.fault,
        };
        project(
            self.graph,
            &mut p,
            &self.arena,
            &format!("{base}ffn_gate_inp.weight"),
            self.gr.mixed,
            self.b.router,
            cancel,
        )?;
        let start = Instant::now();
        let router = self.admission.readback_router(|| {
            u::read_fence(
                &self.arena,
                self.b.router,
                self.b.fault,
                self.b.router_packet,
            )
        })?;
        self.record("router_fence", start);
        let selected = Zeroizing::new(ops::router_topk(&router, 10, true).map_err(u::fault)?);
        let start = Instant::now();
        let receipt = executor.enqueue_routed_moe(
            self.graph,
            &self.arena,
            &mut self.moe_layers[l],
            self.gr.mixed,
            self.b.block,
            &selected,
            cancel,
        )?;
        self.record("expert_upload_and_enqueue", start);
        let status = receipt.status();
        if u::words(&self.arena, status.buffer)? != status.words {
            return Err(Error::InvalidBufferLength);
        }
        u::status(&self.arena, status.buffer, self.b.fault)?;
        self.pending_moe.push((l, receipt));
        let mut p = ResidentProjection {
            executor,
            fault: self.b.fault,
        };
        for (name, out) in [
            ("ffn_gate_shexp.weight", self.b.shared_gate),
            ("ffn_up_shexp.weight", self.b.shared_up),
        ] {
            project(
                self.graph,
                &mut p,
                &self.arena,
                &format!("{base}{name}"),
                self.gr.mixed,
                out,
                cancel,
            )?;
        }
        shared_activation::launch::<WgpuRuntime>(
            self.arena.client().map_err(u::fault)?,
            u::grid(640)?,
            CubeDim::new_1d(64),
            u::arg(&self.arena, self.b.shared_gate)?,
            u::arg(&self.arena, self.b.shared_up)?,
            u::arg(&self.arena, self.b.shared_act)?,
        );
        u::validate(&self.arena, self.b.shared_act, self.b.fault)?;
        project(
            self.graph,
            &mut p,
            &self.arena,
            &format!("{base}ffn_down_shexp.weight"),
            self.b.shared_act,
            self.b.shared_down,
            cancel,
        )?;
        let gamma = self
            .public
            .vector(self.graph, &format!("{base}ffn_gate_inp_shexp.weight"))?;
        shared_gate_dot::launch::<WgpuRuntime>(
            self.arena.client().map_err(u::fault)?,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            u::arg(&self.arena, self.gr.mixed)?,
            u::raw_arg(gamma, 2560)?,
            u::arg(&self.arena, self.b.shared_scale)?,
            u::arg(&self.arena, self.b.fault)?,
        );
        // Distinct scratch receives sum; block then copied, never alias-write.
        add_shared::launch::<WgpuRuntime>(
            self.arena.client().map_err(u::fault)?,
            u::grid(2560)?,
            CubeDim::new_1d(64),
            u::arg(&self.arena, self.b.block)?,
            u::arg(&self.arena, self.b.shared_down)?,
            u::arg(&self.arena, self.b.shared_scale)?,
            u::arg(&self.arena, self.b.ple_value)?,
        );
        u::validate(&self.arena, self.b.ple_value, self.b.fault)?;
        u::copy_words::launch::<WgpuRuntime>(
            self.arena.client().map_err(u::fault)?,
            u::grid(2560)?,
            CubeDim::new_1d(64),
            u::arg(&self.arena, self.b.ple_value)?,
            u::arg(&self.arena, self.b.block)?,
        );
        Ok(())
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use kuno_text::ops::{GatedResidualShape, GdnShape};

    #[test]
    fn explicit_advance_skips_head_but_sample_keeps_full_output() {
        use std::cell::Cell;
        for purpose in [StepPurpose::Logits, StepPurpose::Advance] {
            let heads = Cell::new(0);
            let advances = Cell::new(0);
            let result = completed_output(
                purpose,
                || {
                    heads.set(heads.get() + 1);
                    Ok(Zeroizing::new(vec![3., -2.]))
                },
                || {
                    advances.set(advances.get() + 1);
                    u::accept_fault_packet(&[0; 4])
                },
            )
            .unwrap();
            if purpose == StepPurpose::Logits {
                assert_eq!(heads.get(), 1);
                assert_eq!(advances.get(), 0);
                assert_eq!(&*result, &[3., -2.]);
            } else {
                assert_eq!(heads.get(), 0);
                assert_eq!(advances.get(), 1);
                assert!(result.is_empty());
            }
        }
    }
    #[test]
    fn advance_fault_read_failure_or_cancel_never_commits() {
        use std::cell::Cell;
        for packet in [
            vec![0; 3],
            1_u32.to_le_bytes().to_vec(),
            u32::MAX.to_le_bytes().to_vec(),
        ] {
            let committed = Cell::new(false);
            let fence = completed_output(
                StepPurpose::Advance,
                || panic!("advance executed vocabulary head"),
                || u::accept_fault_packet(&packet),
            );
            assert!(
                commit_after_fence(fence, &AtomicBool::new(false), || {
                    committed.set(true);
                    Ok(())
                })
                .is_err()
            );
            assert!(!committed.get());
        }
        let committed = Cell::new(false);
        let read_error: Result<()> = Err(Error::InvalidFormat("fixed public read failure"));
        assert!(
            commit_after_fence(read_error, &AtomicBool::new(false), || {
                committed.set(true);
                Ok(())
            })
            .is_err()
        );
        assert!(!committed.get());
        assert!(
            commit_after_fence(Ok(()), &AtomicBool::new(true), || {
                committed.set(true);
                Ok(())
            })
            .is_err()
        );
        assert!(!committed.get());
        let cancel = AtomicBool::new(false);
        assert!(
            commit_after_fence(Ok(()), &cancel, || {
                cancel.store(true, Ordering::Release);
                Ok(())
            })
            .is_err()
        );
    }
    #[test]
    fn successful_fault_fence_precedes_state_commit() {
        use std::cell::RefCell;
        let events = RefCell::new(Vec::new());
        let fence = completed_output(
            StepPurpose::Advance,
            || panic!("advance executed vocabulary head"),
            || {
                u::accept_fault_packet(&[0; 4])?;
                events.borrow_mut().push("accepted fence");
                Ok(())
            },
        );
        assert!(
            commit_after_fence(fence, &AtomicBool::new(false), || {
                events.borrow_mut().push("state commit");
                Ok(())
            })
            .unwrap()
            .is_empty()
        );
        assert_eq!(*events.borrow(), ["accepted fence", "state commit"]);
    }
    #[test]
    fn exact_public_qwen_owner_plan_includes_full_padding_for_every_handle() {
        // Fixed public geometry, no GGUF file, client, private content or device.
        let shape = GdnShape::qwen38();
        let gdn = crate::resident_gdn::StageLayout {
            pointwise: crate::gdn_pointwise::PointwiseLayout::new(shape).unwrap(),
            recurrence: crate::gdn_gpu::GdnLayout::new(shape).unwrap(),
            input_words: 2560,
            epsilon: 1e-6,
            max_context: 4096,
        };
        let qsa = crate::resident_qsa::StageLayout {
            input_words: 2560,
            max_context: 4096,
            epsilon: 1e-6,
        };
        let moe = crate::resident_moe::MoeLayout::new(2560, 640).unwrap();
        let gr = crate::residual_gpu::ResidualLayout::new(
            GatedResidualShape::new(4, 2560).unwrap(),
            320,
        )
        .unwrap();
        let mut words = Vec::new();
        for layer in 0..48 {
            if layer % 4 == 3 {
                words.extend(qsa.allocation_words());
            } else {
                words.extend(gdn.allocation_words());
            }
        }
        for _ in 0..48 {
            words.extend([
                10,
                10,
                moe.activation_words,
                moe.partial_words,
                moe.status_words,
            ]);
        }
        words.extend(workspace_words(gr, 248320).unwrap());
        let requests = words
            .iter()
            .map(|&n| BufferLayout::new(n).unwrap().owned_bytes as u64)
            .collect::<Vec<_>>();
        let total = requests.iter().sum::<u64>();
        assert_eq!(total, 785_186_816);
        // The largest private span is the 4096-token QSA cache (32 MiB).
        assert_eq!(requests.iter().copied().max(), Some(32 * 1024 * 1024));
        assert_eq!(requests.len(), 36 * 18 + 12 * 31 + 48 * 5 + 10 + 16);
        assert!(requests.iter().all(|&n| n >= 32768 && n.is_power_of_two()));
        let mut ledger = crate::main_admission::MainAdmission::new(total, 0).unwrap();
        assert_eq!(ledger.precharge(&requests).unwrap(), total);
        assert_eq!(ledger.upper_bytes(), total);
        assert!(workspace_words(gr, usize::MAX).is_err());
    }
}
