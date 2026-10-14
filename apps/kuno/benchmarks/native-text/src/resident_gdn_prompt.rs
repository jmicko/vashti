//! Whole-prompt GDN stage ownership. Verification receipts stay MAX8; this
//! ticket can commit only after the large model's completed fault/I/O fence.
use super::*;
use crate::{executor::GpuExecutor, quant_matvec::matmul::BatchSpans, resident_util as u};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

pub(crate) struct Pending {
    binding: Option<GraphBindingId>,
    stamp: Zeroizing<[u64; 5]>, // stage, sequence, layer, start, count
    roots: [BufferId; 5],
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for Pending {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.roots.zeroize();
        self.binding = None;
    }
}
impl ResidentGdn<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_prompt_full(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        executor: &mut GpuExecutor<'_>,
        public: &PublicGdnParameters,
        input: Handle,
        output: Handle,
        start: usize,
        count: usize,
        fault_id: BufferId,
        poison: Arc<AtomicBool>,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        cancelled(cancel)?;
        graph.verify_unchanged()?;
        if !crate::strata_prompt::active()
            || poison.load(Ordering::Acquire)
            || !(1..=crate::strata_prompt::groups::CAPACITY).contains(&count)
            || start
                .checked_add(count)
                .is_none_or(|end| end > self.layout.max_context)
            || self.binding != graph.binding_id()
            || public.binding != self.binding
            || public.layer != self.layer
            || public.identity != self.public_identity
            || self.lifecycle.phase != Phase::Ready(start as u64)
            || self.batch_pending.is_some()
            || self
                .batch_poison
                .as_ref()
                .is_some_and(|p| !Arc::ptr_eq(p, &poison))
        {
            return Err(fault("native prompt GDN binding/state"));
        }
        executor.check_client(arena.client().map_err(u::fault)?)?;
        crate::strata_prompt::validate_output(&input)?;
        crate::strata_prompt::validate_output(&output)?;
        if input.size_in_used() < (count * 2560 * 4) as u64
            || output.size_in_used() < (count * 2560 * 4) as u64
            || crate::strata_prompt::storage::overlaps(&input, &output)?
        {
            return Err(Error::InvalidBufferLength);
        }
        let shape = self.layout.pointwise.shape;
        let channels = self.layout.pointwise.channels;
        if shape.head_dim != 128 || channels != 10240 || self.layout.input_words != 2560 {
            return Err(Error::InvalidBufferLength);
        }
        let roots = [
            self.buffers.raw_conv[0],
            self.buffers.raw_conv[1],
            self.buffers.recurrent[0],
            self.buffers.recurrent[1],
            fault_id,
        ];
        let lengths = [
            channels * 3,
            channels * 3,
            shape.value_heads * 16384,
            shape.value_heads * 16384,
            1,
        ];
        let physical = roots
            .into_iter()
            .zip(lengths)
            .map(|(id, n)| crate::resident_gr::batch::full_private(arena, id, n))
            .collect::<Result<Vec<_>>>()?;
        crate::resident_gr::batch::physical(arena, &physical)?;
        self.lifecycle.phase = Phase::UncommittedBatch(start as u64, count);
        self.batch_poison = Some(poison.clone());
        let pending = Pending {
            binding: Some(self.binding),
            stamp: Zeroizing::new([
                self.identity,
                self.sequence.0,
                self.layer as u64,
                start as u64,
                count as u64,
            ]),
            roots,
            poison,
            armed: true,
        };
        let h = |id| u::handle(arena, id);
        let [raw, z, prepared, recurrent] = crate::strata_prompt::gdn_workspace()?;
        for (projection, target) in [
            (self.projections.qkv, raw.clone()),
            (self.projections.z, z.clone()),
        ] {
            executor.enqueue_strata_prompt_projection(
                projection.tensor,
                &input,
                &target,
                &h(fault_id)?,
                &BatchSpans::new(count, 0, 0, 2560, 0, projection.matrix.outputs()),
                cancel,
            )?;
        }
        let [_, controls, status] = crate::strata_prompt::gdn_controls()?;
        let ab = executor
            .strata_gdn_ab_weights([self.projections.alpha.tensor, self.projections.beta.tensor])?;
        crate::strata_prompt::ab_handles(
            crate::strata_prompt::AbHandles {
                client: arena.client().map_err(u::fault)?,
                input,
                params: public.controls.clone(),
                controls: controls.clone(),
                status,
                fault: h(fault_id)?,
                count,
                heads: shape.value_heads,
                first: 0,
                stride: 2560,
            },
            ab,
        )?;
        let p = [
            count as u32,
            start as u32,
            channels as u32,
            shape.key_heads as u32,
            shape.value_heads as u32,
            (public.gamma_words == 128) as u32,
            self.layout.epsilon.to_bits(),
            0,
        ];
        crate::strata_gdn::prompt::convolution_full(
            arena,
            p,
            [
                raw,
                public.convolution.clone(),
                h(roots[0])?,
                h(roots[1])?,
                prepared.clone(),
                h(fault_id)?,
            ],
        )?;
        crate::strata_gdn::prompt::recurrence_full(
            arena,
            p,
            [
                prepared,
                controls,
                h(roots[2])?,
                h(roots[3])?,
                recurrent,
                z,
                public.gamma.clone(),
                h(fault_id)?,
            ],
        )?;
        executor.enqueue_strata_prompt_prepacked_projection(
            self.projections.output.tensor,
            &crate::strata_prompt::activation_stage()?,
            &output,
            &h(fault_id)?,
            &BatchSpans::new(count, 0, 0, shape.value_heads * 128, 0, 2560),
            cancel,
        )?;
        cancelled(cancel)?;
        Ok(pending)
    }
    /// Read-only preflight across all stages precedes any commit. Caller must
    /// already have the actual whole-model completed fault and runtime proof.
    pub(crate) fn validate_prompt_full(
        &self,
        p: &Pending,
        start: usize,
        count: usize,
        fault_id: BufferId,
        poison: &Arc<AtomicBool>,
    ) -> Result<()> {
        if !p.armed
            || p.binding != Some(self.binding)
            || *p.stamp
                != [
                    self.identity,
                    self.sequence.0,
                    self.layer as u64,
                    start as u64,
                    count as u64,
                ]
            || p.roots
                != [
                    self.buffers.raw_conv[0],
                    self.buffers.raw_conv[1],
                    self.buffers.recurrent[0],
                    self.buffers.recurrent[1],
                    fault_id,
                ]
            || self.lifecycle.phase != Phase::UncommittedBatch(start as u64, count)
            || self.batch_pending.is_some()
            || !Arc::ptr_eq(&p.poison, poison)
            || poison.load(Ordering::Acquire)
        {
            return Err(fault("native prompt GDN completion identity"));
        }
        Ok(())
    }
    /// Infallible publication after all stage preflights in the exclusive model.
    pub(crate) fn commit_prompt_full(&mut self, mut p: Pending) {
        self.buffers.raw_conv.swap(0, 1);
        self.buffers.recurrent.swap(0, 1);
        self.lifecycle.phase = Phase::Ready(p.stamp[3] + p.stamp[4]);
        p.armed = false;
    }
}
