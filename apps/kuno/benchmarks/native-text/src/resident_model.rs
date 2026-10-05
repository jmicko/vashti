//! Fixed-fixture laboratory whole-token owner. No HTTP/private request support.
//! GR, GDN, QSA and all-ten routed MoE stay on the device; projected PLE
//! uses the exact Rust host oracle. Any partially queued failure destroys the
//! entire sequence. Runtime staging erasure is explicitly still unaccepted.
use crate::{
    private_arena::{BufferId, PrivateArena},
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
    layers: Vec<Layer<'f>>,
    moe_layers: Vec<crate::resident_moe::ResidentMoe>,
    pending_qsa: Vec<(usize, crate::resident_qsa::PendingQsa)>,
    pending_moe: Vec<(usize, crate::resident_moe::PendingMoe)>,
    b: Buffers,
    ple_history: PleTokenHistory,
    ple_conv: CausalConvState,
    position: usize,
    context: usize,
    poisoned: bool,
    retired: Arc<AtomicBool>,
    cancel_after_attention: Option<usize>,
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
impl<'m, 'f> ResidentModel<'m, 'f> {
    pub fn new(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
    ) -> Result<Self> {
        let g = graph.geometry();
        if context == 0 || context > g.max_context {
            return Err(Error::LimitExceeded("resident context"));
        }
        let retired = Arc::new(AtomicBool::new(false));
        let mut arena = PrivateArena::new(client.clone(), 2 * 1024 * 1024 * 1024, retired.clone())
            .map_err(u::fault)?;
        let io = PrivateArena::new(client.clone(), 64 * 1024 * 1024, retired.clone())
            .map_err(u::fault)?;
        let mut public = PublicVectors::new(client, graph.binding_id());
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
                    let p =
                        crate::resident_qsa::PublicQsaParameters::prepare(graph, l, &mut public)?;
                    let stage =
                        crate::resident_qsa::ResidentQsa::allocate(graph, l, key, &mut arena, &p)?;
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
        Ok(Self {
            graph,
            arena,
            io,
            public,
            gr,
            layers,
            moe_layers,
            pending_qsa: Vec::new(),
            pending_moe: Vec::new(),
            b,
            ple_history,
            ple_conv,
            position: 0,
            context,
            poisoned: false,
            retired,
            cancel_after_attention: None,
            phase_ns: BTreeMap::new(),
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
    pub fn owned_bytes(&self) -> usize {
        self.arena.charged_bytes()
    }
    pub fn step(
        &mut self,
        executor: &mut crate::executor::GpuExecutor<'f>,
        token: u32,
        cancel: &AtomicBool,
    ) -> Result<Zeroizing<Vec<f32>>> {
        if self.poisoned
            || self.position >= self.context
            || token as usize >= self.graph.geometry().vocabulary
        {
            return Err(Error::InvalidFormat("invalid resident sequence"));
        }
        self.poisoned = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.step_checked(executor, token, cancel)
        }));
        match result {
            Ok(Ok(logits)) => {
                self.position += 1;
                self.poisoned = false;
                Ok(logits)
            }
            other => {
                for layer in &mut self.layers {
                    match layer {
                        Layer::Gdn(g, _) => g.poison(),
                        Layer::Qsa(q, _) => q.poison(),
                    }
                }
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
                if io.is_err() || all.is_err() {
                    self.retired.store(true, Ordering::Release);
                }
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
    ) -> Result<Zeroizing<Vec<f32>>> {
        executor.check_client(self.arena.client().map_err(u::fault)?)?;
        self.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        u::zero(&self.arena, self.b.fault)?;
        let mut embedding = Zeroizing::new(vec![0.; 2560]);
        self.graph
            .tensor("token_embd.weight")?
            .decode_row(token as u64, &mut embedding)?;
        let residual = Zeroizing::new(embedding.repeat(4));
        u::upload_f32(&mut self.io, &self.arena, self.b.residual[0], &residual)?;
        let mut current = 0;
        let mut pending: Vec<(usize, PendingGdn)> = Vec::new();
        for l in 0..self.layers.len() {
            u::cancel(cancel)?;
            let start = Instant::now();
            if l == 1 {
                self.ple(executor, token, self.b.residual[current], cancel)?;
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
                    let mut selector = u::TransientSelector { io: &mut self.io };
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
        let logits = u::read_fence(
            &self.arena,
            self.b.logits,
            self.b.fault,
            self.b.logit_packet,
        )?;
        // The device completion/finiteness decision covers ALL layer outputs,
        // including the output projection. A late cancellation discards all.
        u::cancel(cancel)?;
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
        u::cancel(cancel)?;
        Ok(logits)
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
        u::upload_f32(&mut self.io, &self.arena, self.b.ple_emb, &emb)?;
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
        let key = u::read_f32(&self.arena, self.b.ple_key)?;
        let value = u::read_f32(&self.arena, self.b.ple_value)?;
        let original = u::read_f32(&self.arena, residual)?;
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
        let uploaded = u::upload_f32(&mut self.io, &self.arena, residual, &result.residual);
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
        let router = u::read_fence(
            &self.arena,
            self.b.router,
            self.b.fault,
            self.b.router_packet,
        )?;
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
