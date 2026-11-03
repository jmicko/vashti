//! Original GPU convolution-history ownership; only two ngram token IDs need
//! CPU publication. Legacy paths retain their independent CPU float oracle.
use super::*;

pub(crate) enum Commit<'a> {
    Legacy(core::PleChunkCommit<'a>),
    Device(core::DeviceHistoryTokens<'a>),
}
impl Commit<'_> {
    pub(super) fn device(&self) -> bool {
        matches!(self, Self::Device(_))
    }
    pub(crate) fn preflight(&self, cancel: &AtomicBool) -> Result<()> {
        match self {
            Self::Legacy(p) => p.preflight(cancel),
            Self::Device(p) => p.preflight(cancel),
        }
        .map_err(u::fault)
    }
    pub(crate) fn commit(self) {
        match self {
            Self::Legacy(p) => p.commit(),
            Self::Device(p) => p.commit(),
        }
    }
}
pub(super) enum Restore<'a> {
    Legacy(ops::ple_checkpoint::Restore<'a>),
    Device(core::DeviceHistoryTokens<'a>),
}
impl Restore<'_> {
    fn commit(self) {
        match self {
            Self::Legacy(p) => p.commit(),
            Self::Device(p) => p.commit(),
        }
    }
}
pub(crate) struct QueuedRestore<'a>(LatestPleRestore<'a>);
impl QueuedRestore<'_> {
    /// Caller accepts the real joint GDN/QSA/PLE fault fence first.
    pub(crate) fn commit(mut self) {
        self.0.restore.commit();
        self.0.workspace.next = self.0.checkpoint.metadata.start + self.0.accepted;
        self.0.checkpoint.armed = false;
    }
}
pub(super) fn enqueue_restore<'a>(
    p: LatestPleRestore<'a>,
    arena: &PrivateArena,
    fault: BufferId,
    cancel: &AtomicBool,
) -> Result<QueuedRestore<'a>> {
    u::cancel(cancel)?;
    p.workspace.validate_ids(arena)?;
    if let Restore::Device(tokens) = &p.restore {
        tokens.preflight(cancel).map_err(u::fault)?;
        let original = &p.checkpoint.metadata.ids;
        // Full-window commit swapped roles: NEXT is the original prewindow
        // GPU history and OLD is the candidate being repaired. No CPU mirror.
        if p.workspace.ids[NEXT] != original[OLD] || p.workspace.ids[OLD] != original[NEXT] {
            return Err(error());
        }
        let h = |id| u::handle(arena, id);
        if p.accepted == 0 {
            u::copy_words::launch::<WgpuRuntime>(
                arena.client().map_err(u::fault)?,
                u::grid(words(OLD))?,
                CubeDim::new_1d(64),
                u::arg(arena, original[OLD])?,
                u::arg(arena, original[NEXT])?,
            );
        } else {
            let mut info = [0; 12];
            info[0] = u32::try_from(p.accepted).map_err(u::fault)?;
            crate::strata_ple::launch(
                arena,
                crate::strata_ple::Kernel::PromptHistory,
                [CHANNELS / 256, 1, 1],
                vec![
                    h(original[OLD])?,
                    h(original[NORMALIZED])?,
                    h(original[NEXT])?,
                    h(fault)?,
                ],
                info,
            )?;
        }
    }
    u::cancel(cancel)?;
    Ok(QueuedRestore(p))
}
impl PleBatchWorkspace {
    pub(super) fn prepare_device_history_after_whole_model_fence<'a>(
        &'a mut self,
        c: Preparation<'a, '_, '_>,
        receipt: PendingPleBatch,
        cancel: &AtomicBool,
    ) -> Result<PleBatchAcceptance<'a>> {
        let accepted = receipt.metadata.count;
        self.prepare_device_prefix_after_whole_model_fence(c, receipt, accepted, cancel)
    }
    pub(crate) fn native_recording_owner(&self) -> u64 {
        self.identity
    }
    pub(crate) fn check_pending_prefix(
        &self,
        model: &ResidentModel<'_, '_>,
        receipt: &PendingPleBatch,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<(usize, usize)> {
        self.receipt(model, receipt, cancel)?;
        if !crate::strata_intel_core::enabled() || !(1..=receipt.metadata.count).contains(&accepted)
        {
            return Err(error());
        }
        Ok((receipt.metadata.start, receipt.metadata.count))
    }
    pub(crate) fn enqueue_pending_prefix(
        &self,
        model: &ResidentModel<'_, '_>,
        receipt: &PendingPleBatch,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.check_pending_prefix(model, receipt, accepted, cancel)?;
        if receipt.metadata.count > 1 {
            let mut info = [0; 12];
            info[0] = accepted as u32;
            crate::strata_ple::launch_commit_history(
                &model.arena,
                vec![
                    u::handle(&model.arena, self.ids[OLD])?,
                    u::handle(&model.arena, self.ids[NORMALIZED])?,
                    u::handle(&model.arena, self.ids[NEXT])?,
                    u::handle(&model.arena, model.b.fault)?,
                ],
                info,
            )?;
        }
        u::cancel(cancel)
    }
    pub(crate) fn prepare_device_prefix_after_whole_model_fence<'a>(
        &'a mut self,
        c: Preparation<'a, '_, '_>,
        receipt: PendingPleBatch,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<PleBatchAcceptance<'a>> {
        let poison = self.poison.clone();
        let diagnostic = guarded(&poison, || {
            if !(1..=receipt.metadata.count).contains(&accepted) {
                return Err(error());
            }

            self.receipt_parts(&c, &receipt, cancel)?;
            // The grouped policy used to read the three GPU history owners back
            // here and drop them: a pure audit costing ~7 ms per decode round.
            // Completion is still proven by the fault readback that follows.
            if !c.admission.checkpoint_request_io_reuse(cancel)? {
                audit(c.arena, c.retired)?;
            }
            self.receipt_parts(&c, &receipt, cancel)?;
            Ok(self.public_grouped_reads)
        });
        let public_grouped_read = match diagnostic {
            Ok(v) => v,
            Err(e) => {
                *c.owner_poisoned = true;
                c.retired.store(true, Ordering::Release);
                return Err(e);
            }
        };
        let m = &receipt.metadata;
        let transition = core::prepare_device_history_tokens(
            c.history,
            c.graph.ple_hash(),
            m.old_tokens,
            &m.tokens[..accepted],
            cancel,
        )
        .map_err(u::fault);
        match transition {
            Ok(p) => Ok(PleBatchAcceptance {
                workspace: self,
                anchor: c.anchor,
                arena: c.arena,
                retired: c.retired,
                transition: Some(Commit::Device(p)),
                accepted_count: accepted,
                receipt,
                armed: true,
                public_grouped_read,
            }),
            Err(e) => {
                *c.owner_poisoned = true;
                poison.store(true, Ordering::Release);
                Err(e)
            }
        }
    }
}
