//! Exact all-ten routed MoE enqueue for the isolated resident Rust GPU lab.
//!
//! Original packed banks are PUBLIC weights. Every activation, chosen slot,
//! probability and status lives in the caller's full-span private arena. Host
//! routing selects all ten experts and supplies the committed slot plan; the
//! validated plan is published using RUNTIME scalars, never shader constants.
//! Existing packed kernels then execute gate/up, down and rank-ordered reduce.
//! The shared expert remains a separate graph path owned by the whole-token
//! executor. This module does not prune, reroute or renormalize probabilities.
//!
//! Enqueue is NOT acceptance. The single whole-token owner must drain, accept
//! every status, check cancellation/backend health and only then call accept.
//! Any error/cancellation/panic or abandoned pending receipt poisons the whole
//! sequence and requires complete arena cleanup. No partial retry is supported.
//! Pending handles remain arena-owned; public bank slots must not be replaced
//! until these commands complete on the same serialized client/stream. Runtime
//! upload/uniform/readback staging still needs the separate privacy hook; this
//! module and its offline tests do not establish device parity/privacy.

use crate::{
    expert_banks::{BankInstance, ExpertBanks},
    private_arena::{self, BufferId, BufferLayout, PrivateArena},
    quant_matvec,
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::cpu::Matrix;
use kuno_text::model::{BoundGraph, GraphBindingId, RoutedWeights};
use kuno_text::{Error, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const SELECTED: usize = 10;
const LANES: usize = 64;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn fault(s: &'static str) -> Error {
    Error::InvalidFormat(s)
}
fn af(_: private_arena::ArenaError) -> Error {
    fault("resident MoE arena unavailable")
}
fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(fault("resident MoE cancelled"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeLayout {
    pub hidden: usize,
    pub feed_forward: usize,
    pub activation_words: usize,
    pub partial_words: usize,
    pub status_words: usize,
    status_offsets: [usize; 4],
    status_counts: [usize; 4],
}
impl MoeLayout {
    pub fn new(hidden: usize, feed_forward: usize) -> Result<Self> {
        if hidden == 0 || hidden > 4096 || feed_forward == 0 || feed_forward > 4096 {
            return Err(fault("resident MoE geometry"));
        }
        let shape = quant_matvec::moe::MoeShape::new(SELECTED, hidden, feed_forward, 65535)
            .map_err(fault)?;
        let counts = [
            hidden,
            shape.activation_elements,
            shape.partial_elements,
            hidden,
        ]
        .map(|n| n.div_ceil(LANES));
        let mut offsets = [0; 4];
        for i in 1..4 {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }
        let status_words = offsets[3] + counts[3];
        Ok(Self {
            hidden,
            feed_forward,
            activation_words: shape.activation_elements,
            partial_words: shape.partial_elements,
            status_words,
            status_offsets: offsets,
            status_counts: counts,
        })
    }
    pub fn from_graph(graph: BoundGraph<'_, '_>, layer: usize) -> Result<Self> {
        let g = graph.geometry();
        if layer >= g.layers || g.experts != 512 || g.experts_used != SELECTED {
            return Err(fault("resident MoE admitted geometry"));
        }
        Self::new(g.residual.width, g.expert_hidden)
    }
    pub fn owned_bytes(self) -> Result<usize> {
        if self != Self::new(self.hidden, self.feed_forward)? {
            return Err(fault("resident MoE modified layout"));
        }
        [
            SELECTED,
            SELECTED,
            self.activation_words,
            self.partial_words,
            self.status_words,
        ]
        .into_iter()
        .try_fold(0usize, |n, w| {
            n.checked_add(BufferLayout::new(w).map_err(af)?.owned_bytes)
                .ok_or(Error::ArithmeticOverflow)
        })
    }
}

fn selection(
    slots: &[usize],
    probabilities: &[f32],
    capacity: usize,
) -> Result<([u32; SELECTED], [f32; SELECTED])> {
    if slots.len() != SELECTED
        || probabilities.len() != SELECTED
        || !crate::bank_slots::routed_capacity(capacity)
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut s = [0; SELECTED];
    let mut p = [0.; SELECTED];
    let mut sum = 0.0_f64;
    for i in 0..SELECTED {
        if slots[i] >= capacity
            || slots[..i].contains(&slots[i])
            || !probabilities[i].is_finite()
            || probabilities[i] < 0.
        {
            return Err(fault("resident MoE invalid selection"));
        }
        s[i] = slots[i] as u32;
        p[i] = probabilities[i];
        sum += f64::from(p[i]);
    }
    if (sum - 1.).abs() > 1e-5 {
        return Err(fault("resident MoE unnormalized probabilities"));
    }
    Ok((s, p))
}

// One status per 64-element chunk; inactive tail lanes succeed. Explicit flag
// stores avoid CubeCL 0.10's conditional-literal assignment lowering bug.
#[cube(launch)]
pub fn finite_chunks(input: &Array<f32>, status: &mut Array<u32>, status_offset: usize) {
    let i = ABSOLUTE_POS;
    let lane = UNIT_POS_X as usize;
    let mut success = 1_u32;
    if i < input.len() {
        let x = input[i];
        let valid = x.abs() <= f32::MAX;
        if !valid {
            success = 0_u32;
        }
    }
    let mut flags = SharedMemory::<u32>::new(LANES);
    flags[lane] = success;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            flags[lane] = flags[lane] & flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        status[status_offset + CUBE_POS_X as usize] = flags[0];
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn publish_selection(
    slots: &mut Array<u32>,
    probabilities: &mut Array<f32>,
    s0: u32,
    p0: f32,
    s1: u32,
    p1: f32,
    s2: u32,
    p2: f32,
    s3: u32,
    p3: f32,
    s4: u32,
    p4: f32,
    s5: u32,
    p5: f32,
    s6: u32,
    p6: f32,
    s7: u32,
    p7: f32,
    s8: u32,
    p8: f32,
    s9: u32,
    p9: f32,
) {
    let i = ABSOLUTE_POS;
    if i == 0 {
        slots[i] = s0;
        probabilities[i] = p0;
    }
    if i == 1 {
        slots[i] = s1;
        probabilities[i] = p1;
    }
    if i == 2 {
        slots[i] = s2;
        probabilities[i] = p2;
    }
    if i == 3 {
        slots[i] = s3;
        probabilities[i] = p3;
    }
    if i == 4 {
        slots[i] = s4;
        probabilities[i] = p4;
    }
    if i == 5 {
        slots[i] = s5;
        probabilities[i] = p5;
    }
    if i == 6 {
        slots[i] = s6;
        probabilities[i] = p6;
    }
    if i == 7 {
        slots[i] = s7;
        probabilities[i] = p7;
    }
    if i == 8 {
        slots[i] = s8;
        probabilities[i] = p8;
    }
    if i == 9 {
        slots[i] = s9;
        probabilities[i] = p9;
    }
}

#[derive(Clone, Copy)]
pub struct StatusRegion {
    pub buffer: BufferId,
    pub words: usize,
}
#[derive(Clone, Copy)]
pub struct PendingMoe {
    identity: u64,
    attempt: u64,
    status: StatusRegion,
}
impl PendingMoe {
    pub fn status(&self) -> StatusRegion {
        self.status
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Ready,
    Pending(u64),
    Poisoned,
}

fn accept_phase(
    phase: &mut Phase,
    identity: u64,
    received_identity: u64,
    attempt: u64,
) -> Result<()> {
    if received_identity != identity || *phase != Phase::Pending(attempt) {
        *phase = Phase::Poisoned;
        return Err(fault("resident MoE stale acceptance"));
    }
    *phase = Phase::Ready;
    Ok(())
}
#[cfg(test)]
fn validate_status_values(expected: usize, values: &[u32]) -> Result<()> {
    if expected == 0 || values.len() != expected || values.iter().any(|&v| v != 1) {
        Err(fault("resident MoE numerical/completion failure"))
    } else {
        Ok(())
    }
}

/// Only allocation identities belong in this transient cache, never selected
/// expert IDs/probabilities. BufferId includes arena identity + reset epoch.
#[derive(Clone, Copy, PartialEq, Eq)]
struct AliasProofKey<A> {
    bank: BankInstance,
    private: [A; 7],
}
struct AliasProofCache<A> {
    validated: Option<AliasProofKey<A>>,
}
impl<A: Copy + Eq> AliasProofCache<A> {
    fn requires_validation(&self, key: AliasProofKey<A>) -> bool {
        self.validated != Some(key)
    }
    /// Call only AFTER the physical allocation lookup/overlap proof succeeded.
    fn record_validated(&mut self, key: AliasProofKey<A>) {
        self.validated = Some(key);
    }
    fn invalidate(&mut self) {
        self.validated = None;
    }
}

pub struct ResidentMoe {
    binding: GraphBindingId,
    alias_proof: AliasProofCache<BufferId>,
    layer: usize,
    identity: u64,
    attempt: u64,
    phase: Phase,
    layout: MoeLayout,
    slots: BufferId,
    probabilities: BufferId,
    activations: BufferId,
    partials: BufferId,
    status: BufferId,
}
impl ResidentMoe {
    pub fn allocate(
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &mut PrivateArena,
    ) -> Result<Self> {
        let layout = MoeLayout::from_graph(graph, layer)?;
        if arena
            .charged_bytes()
            .checked_add(layout.owned_bytes()?)
            .is_none_or(|n| n > arena.budget_bytes())
        {
            return Err(Error::LimitExceeded("resident MoE arena budget"));
        }
        let identity = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        // Partial allocation failure remains arena-owned. Whole-sequence owner
        // must close that arena; it must not retry with these partial buffers.
        Ok(Self {
            binding: graph.binding_id(),
            alias_proof: AliasProofCache { validated: None },
            layer,
            identity,
            attempt: 0,
            phase: Phase::Ready,
            layout,
            slots: arena.allocate_zeroed(SELECTED).map_err(af)?,
            probabilities: arena.allocate_zeroed(SELECTED).map_err(af)?,
            activations: arena.allocate_zeroed(layout.activation_words).map_err(af)?,
            partials: arena.allocate_zeroed(layout.partial_words).map_err(af)?,
            status: arena.allocate_zeroed(layout.status_words).map_err(af)?,
        })
    }
    pub fn weights<'file>(&self, graph: BoundGraph<'_, 'file>) -> Result<RoutedWeights<'file>> {
        if graph.binding_id() != self.binding
            || MoeLayout::from_graph(graph, self.layer)? != self.layout
        {
            return Err(fault("resident MoE graph binding"));
        }
        let prefix = format!("blk.{}.", self.layer);
        Ok(RoutedWeights {
            gate: graph.tensor(&format!("{prefix}ffn_gate_exps.weight"))?,
            up: graph.tensor(&format!("{prefix}ffn_up_exps.weight"))?,
            down: graph.tensor(&format!("{prefix}ffn_down_exps.weight"))?,
        })
    }
    /// Fixed-public operator diagnostic only. These IDs remain governed by
    /// the arena epoch and whole-operation fence; no host inference is added.
    /// Read-only fixed-public lab inventory; the caller still proves actual
    /// full resource/client/stream identity and nonaliasing before private fills.
    pub(crate) fn fixed_public_probe_owners(&self) -> [(BufferId, usize); 5] {
        [
            (self.slots, SELECTED),
            (self.probabilities, SELECTED),
            (self.activations, self.layout.activation_words),
            (self.partials, self.layout.partial_words),
            (self.status, self.layout.status_words),
        ]
    }
    pub(crate) fn fixed_public_probe_outputs(&self) -> [(BufferId, usize); 2] {
        [
            (self.activations, self.layout.activation_words),
            (self.partials, self.layout.partial_words),
        ]
    }
    pub fn poison(&mut self) {
        self.phase = Phase::Poisoned;
        self.alias_proof.invalidate();
    }
    /// Caller certifies successful WHOLE-token acceptance, including these
    /// statuses, all separate shared-expert/projector statuses and cancellation.
    pub fn accept_after_whole_token_acceptance(&mut self, pending: PendingMoe) -> Result<()> {
        let result = accept_phase(
            &mut self.phase,
            self.identity,
            pending.identity,
            pending.attempt,
        );
        if result.is_err() {
            self.poison();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        banks: &ExpertBanks,
        input: BufferId,
        output: BufferId,
        slots: &[usize],
        probabilities: &[f32],
        cancel: &AtomicBool,
        block_moe: bool,
        word_down: bool,
    ) -> Result<PendingMoe> {
        if self.phase != Phase::Ready {
            self.poison();
            return Err(fault("resident MoE pending or poisoned"));
        }
        // Poison BEFORE any mutation or potentially partial enqueue. Successful
        // queue submission below publishes a pending receipt, never acceptance.
        self.phase = Phase::Poisoned;
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.enqueue_checked(
                graph,
                arena,
                banks,
                input,
                output,
                slots,
                probabilities,
                cancel,
                block_moe,
                word_down,
            )
        })) {
            Ok(result) => result,
            Err(_) => Err(fault("resident MoE enqueue failed")),
        };
        if result.is_err() {
            self.poison();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_checked(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        banks: &ExpertBanks,
        input: BufferId,
        output: BufferId,
        slots: &[usize],
        probabilities: &[f32],
        cancel: &AtomicBool,
        block_moe: bool,
        word_down: bool,
    ) -> Result<PendingMoe> {
        cancelled(cancel)?;
        graph.verify_unchanged()?;
        let weights = self.weights(graph)?;
        let shape = quant_matvec::moe::MoeShape::new(SELECTED, banks.hidden, banks.ff, 65535)
            .map_err(fault)?;
        if word_down {
            quant_matvec::moe_word_down::validate(weights.down.info().dtype().gguf_id(), banks.ff)
                .map_err(|_| Error::InvalidFormat("word down packed layout"))?;
        }
        let segmented = banks.slots.capacity() == crate::bank_slots::SEGMENTED_CAPACITY;
        if segmented && (!block_moe || word_down) {
            return Err(fault(
                "segmented resident MoE requires block gate/up and generic down",
            ));
        }
        let (s, p) = selection(slots, probabilities, banks.slots.capacity())?;
        if banks.hidden != self.layout.hidden
            || banks.ff != self.layout.feed_forward
            || banks.experts != 512
        {
            return Err(fault("resident MoE bank geometry"));
        }
        for (tensor, columns, rows, bank) in [
            (weights.gate, banks.hidden, banks.ff, banks.gate()),
            (weights.up, banks.hidden, banks.ff, banks.up()),
            (weights.down, banks.ff, banks.hidden, banks.down()),
        ] {
            let matrix = Matrix::new(tensor)?;
            if matrix.inputs() != columns
                || matrix.outputs() != rows
                || matrix.experts() != banks.experts
                || bank.expert_bytes().checked_mul(banks.slots.capacity()) != Some(bank.words() * 4)
            {
                return Err(fault("resident MoE packed bank bounds"));
            }
            quant_matvec::validate_layout(
                tensor.info().dtype().gguf_id(),
                rows,
                columns,
                bank.expert_bytes(),
                columns,
                rows,
            )
            .map_err(fault)?;
        }
        // check() binds original names; request input remains GPU-only. A
        // zero host fixture is unnecessary: exact descriptor/name proof here.
        banks.check_names(weights)?;
        let ids = [
            input,
            self.slots,
            self.probabilities,
            self.activations,
            self.partials,
            output,
            self.status,
        ];
        let lengths = [
            self.layout.hidden,
            SELECTED,
            SELECTED,
            self.layout.activation_words,
            self.layout.partial_words,
            self.layout.hidden,
            self.layout.status_words,
        ];
        for (i, &id) in ids.iter().enumerate() {
            if ids[..i].contains(&id) || arena.layout(id).map_err(af)?.logical_words != lengths[i] {
                return Err(fault("resident MoE aliased or sized buffer"));
            }
        }
        let client = arena.client().map_err(af)?;
        banks.check_client(client)?;
        let handles: Vec<_> = ids
            .iter()
            .map(|&id| arena.handle(id).cloned().map_err(af))
            .collect::<Result<_>>()?;
        let public: Vec<_> = [banks.gate(), banks.up(), banks.down()]
            .into_iter()
            .flat_map(|bank| bank.segments().iter().map(|segment| segment.handle()))
            .collect();
        if public.len() != if segmented { 6 } else { 3 } {
            return Err(fault("resident MoE immutable segment count"));
        }
        // Every call still validates active IDs/epochs/exact sizes above and
        // checks the same immutable bank client and all streams. Keep handles
        // live while using the cached physical proof; no allocation may be
        // replaced under its identity. Rebuilt bank or arena/new input/output
        // IDs misses this proof and performs all resource/overlap checks again.
        check_streams(&handles, &public)?;
        let key = AliasProofKey {
            bank: banks.instance(),
            private: ids,
        };
        if self.alias_proof.requires_validation(key) {
            check_regions(client, &handles, &public)?;
            self.alias_proof.record_validated(key);
        }
        let status_span = arena.layout(self.status).map_err(af)?;
        private_arena::erase_full_span::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(status_span.workgroups_x, status_span.workgroups_y, 1),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(handles[6].clone(), status_span.owned_words) },
        );
        publish_selection::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(handles[1].clone(), SELECTED) },
            unsafe { ArrayArg::from_raw_parts(handles[2].clone(), SELECTED) },
            s[0],
            p[0],
            s[1],
            p[1],
            s[2],
            p[2],
            s[3],
            p[3],
            s[4],
            p[4],
            s[5],
            p[5],
            s[6],
            p[6],
            s[7],
            p[7],
            s[8],
            p[8],
            s[9],
            p[9],
        );
        let x = handles[0].clone();
        let sl = handles[1].clone();
        let prob = handles[2].clone();
        let act = handles[3].clone();
        let parts = handles[4].clone();
        let out = handles[5].clone();
        cancelled(cancel)?;
        // SAFETY: graph/packing/capacities checked; validated runtime slot
        // publisher precedes every read; same-stream full arena spans and
        // immutable banks are retained through whole-token drain/acceptance.
        unsafe {
            if segmented {
                for (gate, up) in banks.gate().segments().iter().zip(banks.up().segments()) {
                    cancelled(cancel)?;
                    if gate.first_slot() != up.first_slot() || gate.capacity() != up.capacity() {
                        return Err(fault("resident gate/up segment identity"));
                    }
                    quant_matvec::segmented_moe::moe_gate_up::launch_unchecked::<WgpuRuntime>(
                        client,
                        CubeCount::Static(shape.gate_up_cubes, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(gate.handle().clone(), gate.words()),
                        ArrayArg::from_raw_parts(up.handle().clone(), up.words()),
                        ArrayArg::from_raw_parts(x.clone(), banks.hidden),
                        ArrayArg::from_raw_parts(sl.clone(), SELECTED),
                        ArrayArg::from_raw_parts(act.clone(), SELECTED * banks.ff),
                        SELECTED,
                        banks.hidden,
                        banks.ff,
                        gate.first_slot(),
                        gate.capacity(),
                        weights.gate.info().dtype().gguf_id(),
                        weights.up.info().dtype().gguf_id(),
                    );
                }
                for down in banks.down().segments() {
                    cancelled(cancel)?;
                    quant_matvec::segmented_moe::moe_down::launch_unchecked::<WgpuRuntime>(
                        client,
                        CubeCount::Static(shape.down_cubes, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(down.handle().clone(), down.words()),
                        ArrayArg::from_raw_parts(act.clone(), SELECTED * banks.ff),
                        ArrayArg::from_raw_parts(sl.clone(), SELECTED),
                        ArrayArg::from_raw_parts(parts.clone(), SELECTED * banks.hidden),
                        SELECTED,
                        banks.hidden,
                        banks.ff,
                        down.first_slot(),
                        down.capacity(),
                        weights.down.info().dtype().gguf_id(),
                    );
                }
            } else {
                if block_moe {
                    quant_matvec::moe_block::moe_gate_up::launch_unchecked::<WgpuRuntime>(
                        client,
                        CubeCount::Static(shape.gate_up_cubes, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(
                            banks.gate().single()?.handle().clone(),
                            banks.gate().words(),
                        ),
                        ArrayArg::from_raw_parts(
                            banks.up().single()?.handle().clone(),
                            banks.up().words(),
                        ),
                        ArrayArg::from_raw_parts(x, banks.hidden),
                        ArrayArg::from_raw_parts(sl.clone(), SELECTED),
                        ArrayArg::from_raw_parts(act.clone(), SELECTED * banks.ff),
                        SELECTED,
                        banks.hidden,
                        banks.ff,
                        weights.gate.info().dtype().gguf_id(),
                        weights.up.info().dtype().gguf_id(),
                    );
                } else {
                    quant_matvec::moe::moe_gate_up::launch_unchecked::<WgpuRuntime>(
                        client,
                        CubeCount::Static(shape.gate_up_cubes, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(
                            banks.gate().single()?.handle().clone(),
                            banks.gate().words(),
                        ),
                        ArrayArg::from_raw_parts(
                            banks.up().single()?.handle().clone(),
                            banks.up().words(),
                        ),
                        ArrayArg::from_raw_parts(x, banks.hidden),
                        ArrayArg::from_raw_parts(sl.clone(), SELECTED),
                        ArrayArg::from_raw_parts(act.clone(), SELECTED * banks.ff),
                        SELECTED,
                        banks.hidden,
                        banks.ff,
                        weights.gate.info().dtype().gguf_id(),
                        weights.up.info().dtype().gguf_id(),
                    );
                }
                if word_down {
                    quant_matvec::moe_word_down::moe_down::launch_unchecked::<WgpuRuntime>(
                        client,
                        CubeCount::Static(shape.down_cubes, 1, 1),
                        CubeDim::new_1d(quant_matvec::moe_word_down::PHYSICAL_LANES as u32),
                        ArrayArg::from_raw_parts(
                            banks.down().single()?.handle().clone(),
                            banks.down().words(),
                        ),
                        ArrayArg::from_raw_parts(act, SELECTED * banks.ff),
                        ArrayArg::from_raw_parts(sl, SELECTED),
                        ArrayArg::from_raw_parts(parts.clone(), SELECTED * banks.hidden),
                        SELECTED,
                        banks.hidden,
                        banks.ff,
                        weights.down.info().dtype().gguf_id(),
                    );
                } else {
                    quant_matvec::moe::moe_down::launch_unchecked::<WgpuRuntime>(
                        client,
                        CubeCount::Static(shape.down_cubes, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(
                            banks.down().single()?.handle().clone(),
                            banks.down().words(),
                        ),
                        ArrayArg::from_raw_parts(act, SELECTED * banks.ff),
                        ArrayArg::from_raw_parts(sl, SELECTED),
                        ArrayArg::from_raw_parts(parts.clone(), SELECTED * banks.hidden),
                        SELECTED,
                        banks.hidden,
                        banks.ff,
                        weights.down.info().dtype().gguf_id(),
                    );
                }
            }
            cancelled(cancel)?;
            quant_matvec::moe::moe_reduce::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(shape.reduce_cubes, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(parts, SELECTED * banks.hidden),
                ArrayArg::from_raw_parts(prob, SELECTED),
                ArrayArg::from_raw_parts(out.clone(), banks.hidden),
                SELECTED,
                banks.hidden,
            );
        }
        for (index, handle) in [
            handles[0].clone(),
            handles[3].clone(),
            handles[4].clone(),
            handles[5].clone(),
        ]
        .into_iter()
        .enumerate()
        {
            cancelled(cancel)?;
            let words = [
                banks.hidden,
                shape.activation_elements,
                shape.partial_elements,
                banks.hidden,
            ][index];
            finite_chunks::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(self.layout.status_counts[index] as u32, 1, 1),
                CubeDim::new_1d(64),
                unsafe { ArrayArg::from_raw_parts(handle, words) },
                unsafe { ArrayArg::from_raw_parts(handles[6].clone(), self.layout.status_words) },
                self.layout.status_offsets[index],
            );
        }
        cancelled(cancel)?;
        self.attempt = self
            .attempt
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        self.phase = Phase::Pending(self.attempt);
        Ok(PendingMoe {
            identity: self.identity,
            attempt: self.attempt,
            status: StatusRegion {
                buffer: self.status,
                words: self.layout.status_words,
            },
        })
    }
}

fn overlaps(a: u64, b: u64, c: u64, d: u64) -> bool {
    a < d && c < b
}
fn check_streams(private: &[Handle], public: &[&Handle]) -> Result<()> {
    let stream = private.first().ok_or(Error::InvalidBufferLength)?.stream;
    if private.iter().any(|h| h.stream != stream) || public.iter().any(|h| h.stream != stream) {
        return Err(fault("resident MoE cross-stream buffers"));
    }
    Ok(())
}
fn check_regions(
    client: &ComputeClient<WgpuRuntime>,
    private: &[Handle],
    public: &[&Handle],
) -> Result<()> {
    let mut handles = private.to_vec();
    handles.extend(public.iter().map(|h| (*h).clone()));
    let resources = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::resident_util::resources(client, handles)
    }))
    .map_err(|_| fault("retained physical metadata channel"))??;
    for (i, l) in resources.iter().enumerate() {
        let l = l.resource();
        let end = l
            .offset
            .checked_add(l.size)
            .ok_or(Error::ArithmeticOverflow)?;
        for r in resources.iter().skip(i + 1) {
            let r = r.resource();
            let rend = r
                .offset
                .checked_add(r.size)
                .ok_or(Error::ArithmeticOverflow)?;
            if l.buffer == r.buffer && overlaps(l.offset, end, r.offset, rend) {
                return Err(fault("resident MoE physical alias"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct FixtureAllocation {
        arena: u64,
        epoch: u64,
        slot: usize,
    }
    fn fixture_key(bank: BankInstance, arena: u64, epoch: u64) -> AliasProofKey<FixtureAllocation> {
        AliasProofKey {
            bank,
            private: std::array::from_fn(|slot| FixtureAllocation { arena, epoch, slot }),
        }
    }
    #[test]
    fn alias_proof_reuses_only_the_same_live_immutable_allocations() {
        let bank = crate::expert_banks::fixture_bank_instance();
        let key = fixture_key(bank, 7, 3);
        let mut cache = AliasProofCache { validated: None };
        assert!(cache.requires_validation(key));
        cache.record_validated(key);
        assert!(!cache.requires_validation(key));
        // Changed buffer positions/identities cannot reuse a physical proof,
        // even when dimensions and model/bank routing contents are unchanged.
        for slot in 0..7 {
            let mut changed = key;
            changed.private[slot].slot += 8;
            assert!(cache.requires_validation(changed));
        }
        assert!(cache.requires_validation(fixture_key(bank, 8, 3)));
        assert!(cache.requires_validation(fixture_key(bank, 7, 4)));
        let rebuilt = fixture_key(crate::expert_banks::fixture_bank_instance(), 7, 3);
        assert!(cache.requires_validation(rebuilt));
        assert!(!cache.requires_validation(key));
    }
    #[test]
    fn alias_proof_is_published_only_after_validation_and_cleared_on_poison() {
        let key = fixture_key(crate::expert_banks::fixture_bank_instance(), 1, 1);
        let mut other = key;
        other.private[0].epoch = 2;
        let mut cache = AliasProofCache { validated: None };
        cache.record_validated(key);
        // A required but failed/unrecorded physical check cannot authorize its
        // new key. The sequence owner poisons on failure; invalidate that proof
        // even if original allocations still happen to occupy the same spans.
        assert!(cache.requires_validation(other));
        assert!(cache.requires_validation(other));
        cache.invalidate();
        assert!(cache.requires_validation(key));
        assert!(cache.requires_validation(other));
        cache.record_validated(other);
        assert!(!cache.requires_validation(other));
        assert!(cache.requires_validation(key));
    }
    #[test]
    fn actual_geometry_status_chunks_and_padded_budget() {
        let l = MoeLayout::new(2560, 640).unwrap();
        assert_eq!(l.activation_words, 6400);
        assert_eq!(l.partial_words, 25600);
        assert_eq!(l.status_offsets, [0, 40, 140, 540]);
        assert_eq!(l.status_counts, [40, 100, 400, 40]);
        assert_eq!(l.status_words, 580);
        assert_eq!(l.owned_bytes().unwrap(), 262144);
        let tail = MoeLayout::new(5, 3).unwrap();
        assert_eq!(tail.status_words, 4);
        for (d, f) in [
            (0, 640),
            (2560, 0),
            (4097, 640),
            (2560, 4097),
            (usize::MAX, 2),
        ] {
            assert!(MoeLayout::new(d, f).is_err());
        }
        let mut modified = l;
        modified.status_words += 1;
        assert!(modified.owned_bytes().is_err());
    }
    #[test]
    fn selection_never_prunes_or_changes_rank_weights() {
        let slots = [63, 1, 7, 2, 10, 8, 13, 0, 33, 4];
        let p = [
            0.25, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
        ];
        let (s, out) = selection(&slots, &p, 64).unwrap();
        assert_eq!(s, slots.map(|s| s as u32));
        assert_eq!(out.map(f32::to_bits), p.map(f32::to_bits));
        let mut duplicate = slots;
        duplicate[9] = duplicate[0];
        assert!(selection(&duplicate, &p, 64).is_err());
        assert!(selection(&slots, &p, 63).is_err());
        assert!(selection(&slots[..9], &p, 64).is_err());
        assert!(selection(&slots, &p[..9], 64).is_err());
        for invalid in [f32::NAN, f32::INFINITY, -0.125] {
            let mut bad = p;
            bad[0] = invalid;
            assert!(selection(&slots, &bad, 64).is_err());
        }
        assert!(selection(&slots, &[0.; 10], 64).is_err());
        let extended = [71, 70, 69, 68, 67, 66, 65, 64, 1, 0];
        assert_eq!(
            selection(&extended, &p, 72).unwrap().0,
            extended.map(|s| s as u32)
        );
        assert!(selection(&extended, &p, 71).is_err());
        assert!(selection(&extended, &p, 73).is_err());
        let split = [95, 0, 63, 64, 1, 94, 31, 65, 62, 32];
        let (s, probabilities) = selection(&split, &p, 96).unwrap();
        assert_eq!(s, split.map(|s| s as u32));
        assert_eq!(probabilities.map(f32::to_bits), p.map(f32::to_bits));
        for capacity in [73, 95, 97] {
            assert!(selection(&split, &p, capacity).is_err());
        }
    }
    #[test]
    fn regions_reject_overlap_but_allow_adjacent_owned_spans() {
        assert!(overlaps(0, 32768, 0, 32768));
        assert!(overlaps(0, 32768, 32767, 65536));
        assert!(!overlaps(0, 32768, 32768, 65536));
        assert!(!overlaps(65536, 98304, 0, 32768));
    }
    #[test]
    fn finite_chunk_host_oracle_rejects_every_failure_position_and_tail() {
        fn oracle(v: &[f32]) -> Vec<u32> {
            v.chunks(64)
                .map(|chunk| u32::from(chunk.iter().all(|x| x.is_finite())))
                .collect()
        }
        let mut values = vec![1.; 65];
        assert_eq!(oracle(&values), [1, 1]);
        for index in [0, 32, 63, 64] {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                values[index] = bad;
                let status = oracle(&values);
                assert_eq!(status[index / 64], 0);
                values[index] = 1.;
            }
        }
        values[0] = f32::MAX;
        values[1] = -f32::MAX;
        assert_eq!(oracle(&values), [1, 1]);
    }
    #[test]
    fn whole_token_receipts_cannot_be_reused_crossed_or_accept_incomplete_work() {
        let mut phase = Phase::Pending(7);
        accept_phase(&mut phase, 42, 42, 7).unwrap();
        assert!(phase == Phase::Ready);
        assert!(accept_phase(&mut phase, 42, 42, 7).is_err());
        assert!(phase == Phase::Poisoned);
        for (other_id, attempt) in [(43, 7), (42, 6), (42, 8)] {
            let mut phase = Phase::Pending(7);
            assert!(accept_phase(&mut phase, 42, other_id, attempt).is_err());
            assert!(phase == Phase::Poisoned);
        }
        for phase_value in [Phase::Ready, Phase::Poisoned] {
            let mut phase = phase_value;
            assert!(accept_phase(&mut phase, 42, 42, 7).is_err());
        }
        validate_status_values(3, &[1, 1, 1]).unwrap();
        for values in [&[1, 0, 1][..], &[1, 2, 1][..], &[1, 1][..], &[][..]] {
            assert!(validate_status_values(3, values).is_err());
        }
        assert!(validate_status_values(0, &[]).is_err());
    }
    #[test]
    fn publish_and_status_compile_with_runtime_scalars_without_device() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        for finite in [false, true] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let mut b = KernelBuilder::default();
                b.scope.register_type::<u32>(u);
                b.scope.register_type::<usize>(u);
                b.scope.register_type::<f32>(f);
                b.scope
                    .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                if finite {
                    let x = b.input_array(Type::new(f)).into();
                    let status = b.output_array(Type::new(u)).into();
                    let offset = b.scalar(u).into();
                    finite_chunks::expand(&mut b.scope, x, status, offset);
                } else {
                    let slots = b.output_array(Type::new(u)).into();
                    let p = b.output_array(Type::new(f)).into();
                    let s0 = b.scalar(u).into();
                    let p0 = b.scalar(f).into();
                    let s1 = b.scalar(u).into();
                    let p1 = b.scalar(f).into();
                    let s2 = b.scalar(u).into();
                    let p2 = b.scalar(f).into();
                    let s3 = b.scalar(u).into();
                    let p3 = b.scalar(f).into();
                    let s4 = b.scalar(u).into();
                    let p4 = b.scalar(f).into();
                    let s5 = b.scalar(u).into();
                    let p5 = b.scalar(f).into();
                    let s6 = b.scalar(u).into();
                    let p6 = b.scalar(f).into();
                    let s7 = b.scalar(u).into();
                    let p7 = b.scalar(f).into();
                    let s8 = b.scalar(u).into();
                    let p8 = b.scalar(f).into();
                    let s9 = b.scalar(u).into();
                    let p9 = b.scalar(f).into();
                    publish_selection::expand(
                        &mut b.scope,
                        slots,
                        p,
                        s0,
                        p0,
                        s1,
                        p1,
                        s2,
                        p2,
                        s3,
                        p3,
                        s4,
                        p4,
                        s5,
                        p5,
                        s6,
                        p6,
                        s7,
                        p7,
                        s8,
                        p8,
                        s9,
                        p9,
                    );
                }
                let src = WgslCompiler::default()
                    .compile(
                        b.build(
                            KernelSettings::default()
                                .cube_dim(CubeDim::new_1d(64))
                                .kernel_name("resident_moe"),
                        ),
                        &WgpuCompilationOptions::default(),
                        mode,
                        u,
                    )
                    .unwrap()
                    .to_string();
                assert!(src.contains("@compute"));
                if finite {
                    assert!(src.contains("workgroupBarrier"));
                    assert!(src.contains("u32(1)"));
                    assert!(src.contains("u32(0)"));
                } else {
                    assert!(src.contains("scalars_u32"));
                    assert!(src.contains("scalars_f32"));
                }
            }
        }
    }
}
