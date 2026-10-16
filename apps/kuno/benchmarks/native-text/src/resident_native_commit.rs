//! Retained original accepted-prefix graph. Controls are runtime INFO; original
//! transactions stay armed until the caller's real joint fault/I/O proof.
use super::*;
use cubecl_runtime::server::{NativeStrataBranch, NativeStrataInvocation};

impl Workspace {
    pub(super) fn run_native_prefix_commit(
        &self,
        model: &ResidentModel<'_, '_>,
        pending: &Pending,
        proposed: usize,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !crate::strata_intel_core::enabled()
            || !(1..=8).contains(&proposed)
            || !(1..=proposed).contains(&accepted)
        {
            return Err(error());
        }
        // Every warm replay consumes the original full-window state/input
        // receipts, not merely a cache hit or matching shape/position numbers.
        let mut role = None;
        for (layer, stage) in model.layers.iter().enumerate() {
            match stage {
                Layer::Gdn(stage, public) => {
                    let extent = self.gdn.check_pending_prefix(
                        stage,
                        public,
                        pending.gdn[layer].as_ref().ok_or(error())?,
                        &model.arena,
                        accepted,
                        cancel,
                    )?;
                    if extent != (model.position, proposed) {
                        return Err(error());
                    }
                    let next = stage.native_capture_state_role()?;
                    if role.is_some_and(|old| old != next) {
                        return Err(error());
                    }
                    role = Some(next);
                }
                Layer::Qsa(stage, _) => {
                    let extent = self.qsa.check_pending_prefix(
                        stage,
                        pending.qsa[layer].as_ref().ok_or(error())?,
                        &model.arena,
                        accepted,
                        cancel,
                    )?;
                    if extent != (model.position, proposed) {
                        return Err(error());
                    }
                }
            }
        }
        let extent = self.ple.check_pending_prefix(
            model,
            pending.ple.as_ref().ok_or(error())?,
            accepted,
            cancel,
        )?;
        if extent != (model.position, proposed) {
            return Err(error());
        }
        model.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        if proposed == 1 {
            return Ok(());
        } // Native verified one-row candidates self-commit.
        let role = role.ok_or(error())? | (self.ple.native_capture_history_role()? << 1);
        let key = NativeStrataBranch::target_commit(proposed as u8, model.context as u16, role)
            .and_then(|key| key.with_model_owner(self.ple.native_recording_owner()))
            .map_err(u::fault)?;
        let invocation = NativeStrataInvocation::commit(model.position as u32, accepted as u32)
            .map_err(u::fault)?;
        let client = model.arena.client().map_err(u::fault)?;
        // The current API guarantees an actual joined submission on a hit.
        // This is not fault/state acceptance and cannot publish a model cursor.
        if client
            .replay_native_strata_branch(key, invocation)
            .map_err(u::fault)?
        {
            return Ok(());
        }
        client
            .begin_native_strata_invocation(key, invocation)
            .map_err(u::fault)?;
        for (layer, stage) in model.layers.iter().enumerate() {
            match stage {
                Layer::Gdn(stage, public) => self.gdn.enqueue_pending_prefix(
                    stage,
                    public,
                    pending.gdn[layer].as_ref().ok_or(error())?,
                    &model.arena,
                    accepted,
                    cancel,
                )?,
                Layer::Qsa(stage, _) => self.qsa.enqueue_pending_prefix(
                    stage,
                    pending.qsa[layer].as_ref().ok_or(error())?,
                    &model.arena,
                    accepted,
                    cancel,
                )?,
            }
        }
        self.ple.enqueue_pending_prefix(
            model,
            pending.ple.as_ref().ok_or(error())?,
            accepted,
            cancel,
        )?;
        client.end_native_strata_branch(key).map_err(u::fault)?;
        u::cancel(cancel)
    }
}
