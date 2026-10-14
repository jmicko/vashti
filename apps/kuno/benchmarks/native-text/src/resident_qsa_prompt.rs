//! Whole-prompt QSA state ownership, separate from the MAX8 verifier receipts.
use super::*;
use crate::{executor::GpuExecutor, quant_matvec::matmul::BatchSpans, resident_util as u};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};
pub(crate) struct Pending {
    binding: Option<GraphBindingId>,
    sequence: Zeroizing<SequenceKey>,
    stamp: Zeroizing<[u64; 4]>, // stage, layer, start, count
    roots: [BufferId; 4],
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
impl ResidentQsa<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_prompt_full(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        executor: &mut GpuExecutor<'_>,
        public: &PublicQsaParameters,
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
            return Err(fault("native prompt QSA binding/state"));
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
        batch::latest_qsa_roots(self, arena, fault_id)?;
        let roots = [
            self.ids[CACHE],
            self.ids[POSITIONS],
            self.ids[POOLED],
            fault_id,
        ];
        self.lifecycle.phase = Phase::PendingBatch(start as u64, count);
        self.batch_poison = Some(poison.clone());
        let pending = Pending {
            binding: Some(self.binding),
            sequence: Zeroizing::new(self.sequence),
            stamp: Zeroizing::new([self.identity, self.layer as u64, start as u64, count as u64]),
            roots,
            poison,
            armed: true,
        };
        let h = |id| u::handle(arena, id);
        let views = crate::strata_prompt::qsa_workspace()?;
        for (i, projection) in [
            self.projections.query,
            self.projections.key,
            self.projections.value,
            self.projections.iq,
            self.projections.ik,
        ]
        .into_iter()
        .enumerate()
        {
            executor.enqueue_strata_prompt_projection(
                projection.tensor,
                &input,
                &views[i],
                &h(fault_id)?,
                &BatchSpans::new(count, 0, 0, 2560, 0, projection.matrix.outputs()),
                cancel,
            )?;
        }
        let p = [
            count as u32,
            start as u32,
            self.layout.max_context as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            qsa::EPSILON.to_bits(),
            qsa::THETA.powf(-2.0 / 64.0).to_bits(),
            0,
        ];
        crate::strata_qsa::prompt::run(
            arena,
            p,
            std::array::from_fn(|i| public.vectors[i].clone()),
            [h(roots[0])?, h(roots[1])?, h(roots[2])?, h(fault_id)?],
        )?;
        executor.enqueue_strata_prompt_prepacked_projection(
            self.projections.output.tensor,
            &crate::strata_prompt::activation_stage()?,
            &output,
            &h(fault_id)?,
            &BatchSpans::new(count, 0, 0, 6144, 0, 2560),
            cancel,
        )?;
        cancelled(cancel)?;
        Ok(pending)
    }
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
            || *p.sequence != self.sequence
            || *p.stamp != [self.identity, self.layer as u64, start as u64, count as u64]
            || p.roots
                != [
                    self.ids[CACHE],
                    self.ids[POSITIONS],
                    self.ids[POOLED],
                    fault_id,
                ]
            || self.lifecycle.phase != Phase::PendingBatch(start as u64, count)
            || self.batch_pending.is_some()
            || !Arc::ptr_eq(&p.poison, poison)
            || poison.load(Ordering::Acquire)
        {
            return Err(fault("native prompt QSA completion identity"));
        }
        Ok(())
    }
    pub(crate) fn commit_prompt_full(&mut self, mut p: Pending) {
        self.lifecycle.phase = Phase::Ready(p.stamp[2] + p.stamp[3]);
        p.armed = false;
    }
}
