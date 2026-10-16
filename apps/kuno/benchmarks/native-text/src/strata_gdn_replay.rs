//! Retained inputs for state-only accepted-prefix commit. No attention/FFN
//! projection, router, expert execution or LM head runs during this replay.
use super::*;
use cubecl::prelude::*;
use zeroize::Zeroize;
pub(crate) const PROMPT_INFO_BYTES: u64 = 36 * 2048;

pub(crate) const MAIN_BYTES: u64 = if cfg!(feature = "strata-rust-port") {
    36 * (524288 + 65536 + 65536 + 262144 + 32768)
} else {
    0
};
pub(crate) const EXTRA_BUFFERS: u64 = if cfg!(feature = "strata-rust-port") {
    36 * 5
} else {
    0
};
pub(crate) struct InputCache {
    ids: [BufferId; 5],
    start: u64,
    count: usize,
    chunk: u64,
}
impl Drop for InputCache {
    fn drop(&mut self) {
        self.start.zeroize();
        self.count.zeroize();
        self.chunk.zeroize();
    }
}
pub(crate) fn allocation_words(layout: StageLayout) -> Vec<usize> {
    if !crate::strata_port::enabled() {
        return Vec::new();
    }
    let p = layout.pointwise;
    [
        p.channels,
        p.qk_elements,
        p.qk_elements,
        p.value_elements,
        p.control_elements,
    ]
    .map(|n| n * MAX_BATCH)
    .to_vec()
}
impl InputCache {
    /// The original retained graph contains the five full-owner copies queued
    /// by retain(). Update only their CPU stamp after its real completion;
    /// accepted-prefix replay still requires the original family checkpoint.
    pub(super) fn record_native_replay(
        &mut self,
        arena: &PrivateArena,
        source: [BufferId; 5],
        start: u64,
        count: usize,
        chunk: u64,
    ) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&count) {
            return Err(fault("Strata GDN replay cache count"));
        }
        for (&from, &to) in source.iter().zip(&self.ids) {
            if from == to
                || arena.layout(from).map_err(arena_fault)?
                    != arena.layout(to).map_err(arena_fault)?
            {
                return Err(fault("Strata GDN replay cache extent"));
            }
        }
        self.start = start;
        self.count = count;
        self.chunk = chunk;
        Ok(())
    }
    pub(crate) fn allocate(arena: &mut PrivateArena, layout: StageLayout) -> Result<Option<Self>> {
        if !crate::strata_port::enabled() {
            return Ok(None);
        }
        let mut ids = Vec::with_capacity(5);
        for words in allocation_words(layout) {
            ids.push(arena.allocate_zeroed(words).map_err(arena_fault)?);
        }
        Ok(Some(Self {
            ids: ids
                .try_into()
                .map_err(|_| fault("Strata GDN input cache"))?,
            start: 0,
            count: 0,
            chunk: 0,
        }))
    }
    pub(super) fn retain(
        &mut self,
        arena: &PrivateArena,
        source: [BufferId; 5],
        prompt_raw: Option<cubecl_runtime::server::Handle>,
        start: u64,
        count: usize,
        chunk: u64,
    ) -> Result<()> {
        for (role, (&from, &to)) in source.iter().zip(&self.ids).enumerate() {
            if from == to
                || arena.layout(from).map_err(arena_fault)?
                    != arena.layout(to).map_err(arena_fault)?
            {
                return Err(fault("Strata GDN cache role extent"));
            }
            if role == 0
                && let Some(raw) = &prompt_raw
            {
                let words = u::words(arena, to)?;
                if !(1..=MAX_BATCH).contains(&count) || !words.is_multiple_of(MAX_BATCH) {
                    return Err(fault("Strata GDN prompt cache extent"));
                }
                crate::strata_gdn::prompt::retain(
                    arena,
                    count,
                    words / MAX_BATCH,
                    raw.clone(),
                    u::handle(arena, to)?,
                )?;
                continue;
            }
            u::copy_words::launch::<WgpuRuntime>(
                arena.client().map_err(arena_fault)?,
                u::grid(u::words(arena, to)?)?,
                CubeDim::new_1d(64),
                u::arg(arena, from)?,
                u::arg(arena, to)?,
            );
        }
        self.start = start;
        self.count = count;
        self.chunk = chunk;
        Ok(())
    }
}
pub(crate) struct PrefixReplay<'a, 'f> {
    checkpoint: LatestGdnCheckpoint,
    stage: &'a mut ResidentGdn<'f>,
    accepted: usize,
}
impl LatestGdnCheckpoint {
    pub(crate) fn prepare_prefix<'a, 'f>(
        self,
        stage: &'a mut ResidentGdn<'f>,
        workspace: &GdnBatchWorkspace,
        arena: &PrivateArena,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<PrefixReplay<'a, 'f>> {
        self.validate_completed(stage, workspace, arena, cancel)?;
        let cache = stage
            .strata_inputs
            .as_ref()
            .ok_or_else(|| fault("Strata GDN prefix inputs absent"))?;
        if !crate::strata_port::enabled()
            || accepted == 0
            || accepted >= self.metadata.count
            || cache.start != self.metadata.start
            || cache.count != self.metadata.count
            || cache.chunk != self.metadata.chunk + 1
        {
            return Err(fault("Strata GDN prefix input stamp"));
        }
        Ok(PrefixReplay {
            checkpoint: self,
            stage,
            accepted,
        })
    }
}
impl PrefixReplay<'_, '_> {
    pub(crate) fn enqueue(
        &self,
        workspace: &GdnBatchWorkspace,
        public: &PublicGdnParameters,
        arena: &PrivateArena,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        cancelled(cancel)?;
        let s = &self.stage;
        let p = s.layout.pointwise;
        let ids = &workspace.ids;
        if public.identity != s.public_identity
            || public.layer != s.layer
            || public.binding != s.binding
        {
            return Err(fault("Strata GDN prefix public origin"));
        }
        let cache = s
            .strata_inputs
            .as_ref()
            .ok_or_else(|| fault("Strata GDN prefix cache"))?;
        let client = arena.client().map_err(arena_fault)?;
        if crate::strata_intel_core::enabled() {
            use crate::strata_gdn::{self as native, Kernel};
            let h = |id| u::handle(arena, id);
            let info = [
                self.accepted as u32,
                self.checkpoint.metadata.start as u32,
                p.channels as u32,
                p.shape.key_heads as u32,
                p.shape.value_heads as u32,
                (public.gamma_words == 128) as u32,
                s.layout.epsilon.to_bits(),
                0,
            ];
            // Restore actual RAW history and the native recurrence from the
            // separately retained pre-window owners. No output graph runs.
            native::launch(
                arena,
                Kernel::Conv,
                [p.channels / 128, self.accepted, 1],
                vec![
                    h(cache.ids[0])?,
                    public.convolution.clone(),
                    h(s.buffers.raw_conv[1])?,
                    h(s.buffers.raw_conv[0])?,
                    h(ids[Q])?,
                    h(ids[K])?,
                    h(ids[V])?,
                    h(ids[STATUS])?,
                    h(ids[STATUS + 1])?,
                ],
                info,
            )?;
            native::launch(
                arena,
                Kernel::Replay,
                [p.shape.value_heads, 1, 1],
                vec![
                    h(cache.ids[1])?,
                    h(cache.ids[2])?,
                    h(cache.ids[3])?,
                    h(cache.ids[4])?,
                    h(s.buffers.recurrent[1])?,
                    h(s.buffers.recurrent[0])?,
                    h(ids[Z])?,
                    public.gamma.clone(),
                    h(ids[GATED])?,
                    h(ids[STATUS + 3])?,
                    h(ids[STATUS + 4])?,
                ],
                info,
            )?;
        } else {
            let conv = bounded_view(&public.convolution, 0, p.conv_weight_elements)?;
            point::conv_silu::launch::<WgpuRuntime>(
                client,
                u::grid(p.channels)?,
                CubeDim::new_1d(64),
                u::arg(arena, cache.ids[0])?,
                raw_f32_arg(conv, p.conv_weight_elements),
                u::arg(arena, s.buffers.raw_conv[1])?,
                u::arg(arena, s.buffers.raw_conv[0])?,
                u::arg(arena, ids[ACT])?,
                u::arg(arena, ids[STATUS])?,
                self.accepted,
                0,
                p.channels,
                0,
                3,
                0,
                3,
                0,
                p.channels,
                0,
                p.channels,
                p.channels,
            );
            gdn_chunk::enqueue_with_policy(
                arena,
                p.shape,
                workspace.layout.chunk(self.accepted)?,
                &ChunkBuffers([
                    cache.ids[1],
                    cache.ids[2],
                    cache.ids[3],
                    cache.ids[4],
                    s.buffers.recurrent[1],
                    s.buffers.recurrent[0],
                    ids[RECURRENT],
                    ids[STATUS + 3],
                ]),
                cancel,
                if self.accepted <= 3 {
                    RecurrencePolicy::ThreeStreaming
                } else {
                    RecurrencePolicy::Chunk
                },
            )?;
        }
        prefix_validate::launch::<WgpuRuntime>(
            client,
            u::grid(
                s.layout
                    .recurrence
                    .state_elements
                    .max(p.ring_elements)
                    .max(self.accepted * p.value_elements),
            )?,
            CubeDim::new_1d(64),
            u::arg(arena, s.buffers.recurrent[0])?,
            u::arg(arena, s.buffers.raw_conv[0])?,
            u::arg(arena, ids[STATUS + 3])?,
            u::arg(arena, ids[STATUS])?,
            u::arg(arena, fault_id)?,
            s.layout.recurrence.state_elements,
            p.ring_elements,
            self.accepted * p.value_elements,
            self.accepted * p.channels,
        );
        Ok(())
    }
    pub(crate) fn commit(mut self) {
        self.stage.lifecycle.phase =
            Phase::Ready(self.checkpoint.metadata.start + self.accepted as u64);
        self.checkpoint.armed = false;
    }
}
#[allow(clippy::manual_range_contains)] // Cube expansion keeps explicit finite comparisons.
#[cube(launch)]
fn prefix_validate(
    state: &Array<f32>,
    ring: &Array<f32>,
    status: &Array<u32>,
    raw_status: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    state_words: usize,
    ring_words: usize,
    status_words: usize,
    raw_status_words: usize,
) {
    let i = ABSOLUTE_POS;
    if i < state_words {
        let x = state[i];
        if !(x >= -3.4028235e38_f32 && x <= 3.4028235e38_f32) {
            fault[0].fetch_or(4);
        }
    }
    if i < ring_words {
        let x = ring[i];
        if !(x >= -3.4028235e38_f32 && x <= 3.4028235e38_f32) {
            fault[0].fetch_or(4);
        }
    }
    if i < status_words && status[i] != 1 {
        fault[0].fetch_or(2);
    }
    if i < raw_status_words && raw_status[i] != 1 {
        fault[0].fetch_or(2);
    }
}
