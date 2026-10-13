//! Exclusive one-latest-chunk public target repair coordinator.
//! `verify` runs the ORIGINAL all-row graph. Its output lease must be dropped
//! before `finish`. Rejection preflights ALL actual families, clears the exact
//! appended QSA suffix, accepts the original complete fault/I/O audit, restores
//! actual GDN/PLE state and reruns ONLY the accepted input prefix. No generations
//! or poison are reset. All-accepted consume performs no extra graph work.
//! No default selection, private inference API or speculative speed claim.
#![allow(dead_code)] // Parent owns protected registration and physical tests.

use super::*;
use kuno_text::model::GraphBindingId;
use latest_checkpoint::{HOST_INCREMENT, LatestTargetChunk};

fn rejected() -> Error {
    Error::InvalidFormat("native latest target round rejected")
}
pub(super) fn checkpoint_configuration(
    outputs: bool,
    already_reserved: bool,
    hybrid: bool,
    router: bool,
    tile: usize,
) -> Result<()> {
    if !outputs || already_reserved || hybrid || router || tile != 8 {
        return Err(rejected());
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Captured,
    Verified,
    Consumed,
}
/// The SAME immutable packet supplies capture, original execution and repair.
/// The original entry admits three tokens; the explicit reserved policy admits
/// four real rows. Unused token cells stay zero and wipe with the full owner.
/// Arbitrary prompt inputs are not an API for this public lab. Binding identity
/// is process-local, never persisted.
struct Packet<B = GraphBindingId> {
    tokens: Zeroizing<[u32; crate::strata_port::verify_capacity()]>,
    geometry: Zeroizing<[usize; 2]>,
    binding: Option<B>,
    phase: Phase,
}
impl<B: Copy + PartialEq> Packet<B> {
    fn new(start: usize, context: usize, tokens: &[u32], binding: B) -> Result<Self> {
        Self::new_selected(start, context, tokens, binding, false)
    }
    fn new_selected(
        start: usize,
        context: usize,
        tokens: &[u32],
        binding: B,
        four: bool,
    ) -> Result<Self> {
        target_rows::coordinates_selected(tokens.len(), start, four)?;
        if context == 0
            || context > 4096
            || start
                .checked_add(tokens.len())
                .is_none_or(|end| end > context)
            || tokens.iter().any(|&t| t >= 248320)
        {
            return Err(rejected());
        }
        let mut packet = Zeroizing::new([0; crate::strata_port::verify_capacity()]);
        packet[..tokens.len()].copy_from_slice(tokens);
        Ok(Self {
            tokens: packet,
            geometry: Zeroizing::new([start, tokens.len()]),
            binding: Some(binding),
            phase: Phase::Captured,
        })
    }
    fn inputs(&self) -> &[u32] {
        &self.tokens[..self.geometry[1]]
    }
    fn verified(&mut self, binding: B, start: usize, count: usize, cursor: usize) -> Result<()> {
        if self.phase != Phase::Captured
            || self.binding != Some(binding)
            || [start, count] != *self.geometry
            || cursor != start.checked_add(count).ok_or(Error::ArithmeticOverflow)?
        {
            return Err(rejected());
        }
        self.phase = Phase::Verified;
        Ok(())
    }
    fn accepted(&self, accepted: usize) -> Result<bool> {
        if self.phase != Phase::Verified || accepted == 0 || accepted > self.geometry[1] {
            return Err(rejected());
        }
        Ok(accepted < self.geometry[1])
    }
}
impl<B> Packet<B> {
    fn wipe(&mut self) {
        self.tokens.zeroize();
        self.geometry.zeroize();
        self.binding = None;
        self.phase = Phase::Consumed;
    }
}
impl<B> Drop for Packet<B> {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Holds BOTH exclusive owners from capture through the single verification
/// and consume/repair. No model/executor accessor permits an intervening chunk.
pub(crate) struct LatestTargetRound<'a, 'm, 'f> {
    model: &'a mut ResidentModel<'m, 'f>,
    executor: &'a mut GpuExecutor<'f>,
    packet: Packet,
    checkpoint: Option<LatestTargetChunk>,
    captured: bool,
    complete: bool,
}
impl<'a, 'm, 'f> LatestTargetRound<'a, 'm, 'f> {
    pub(crate) fn begin(
        model: &'a mut ResidentModel<'m, 'f>,
        executor: &'a mut GpuExecutor<'f>,
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> Result<Self> {
        // Nonmutating construction preflight is outside the armed round. A
        // configuration/cancel/input error here does NOT close a normal model.
        u::cancel(cancel)?;
        let native = model.native_batch.as_ref().ok_or(rejected())?;
        native.validate_target_decode_policy(executor, &model.admission)?;
        if !native.latest_target_checkpoint_reserved
            || native.target_rows.is_none()
            || native.moe.cpu_miss_cap().is_some()
            || native.moe.cpu_gpu_overlap().is_some()
            || native.moe.resident_router_active()
        {
            return Err(rejected());
        }
        native
            .target_rows
            .as_ref()
            .ok_or(rejected())?
            .check_no_live_output()?;
        executor.check_target_rows_policy()?;
        if !executor.target_rows_reserved {
            return Err(rejected());
        }
        entry_boundary(model, executor)?;
        let packet = Packet::new_selected(
            model.position,
            model.context,
            tokens,
            model.graph.binding_id(),
            executor.target_four_rows_active,
        )?;
        let mut result = Self {
            model,
            executor,
            packet,
            checkpoint: None,
            captured: false,
            complete: false,
        };
        let captured = result.capture(cancel)?;
        result.checkpoint = Some(captured);
        Ok(result)
    }
    fn capture(&mut self, cancel: &AtomicBool) -> Result<LatestTargetChunk> {
        let model = &*self.model;
        let native = model.native_batch.as_ref().ok_or(rejected())?;
        let mut gdn = Vec::new();
        let mut qsa = Vec::new();
        reserve_exact(&mut gdn, 36)?;
        reserve_exact(&mut qsa, 12)?;
        for owner in &model.layers {
            match owner {
                Layer::Gdn(stage, _) => {
                    let ticket = native.gdn.checkpoint_latest_with_four_rows(
                        stage,
                        &model.arena,
                        model.position as u64,
                        self.packet.inputs().len(),
                        self.executor.target_four_rows_active,
                        cancel,
                    )?;
                    self.captured = true;
                    gdn.push(ticket);
                }
                Layer::Qsa(stage, _) => {
                    let ticket = native.qsa.checkpoint_latest_with_four_rows(
                        stage,
                        &model.arena,
                        model.position as u64,
                        self.packet.inputs().len(),
                        model.b.fault,
                        self.executor.target_four_rows_active,
                        cancel,
                    )?;
                    self.captured = true;
                    qsa.push(ticket);
                }
            }
        }
        let ple = native.ple.checkpoint_latest_with_four_rows(
            model,
            self.packet.inputs(),
            self.executor.target_four_rows_active,
            cancel,
        )?;
        self.captured = true;
        let checkpoint = LatestTargetChunk::new_selected(
            model.graph.binding_id(),
            model.position,
            self.packet.inputs(),
            gdn,
            qsa,
            ple,
            self.executor.target_four_rows_active,
        )?;
        check_peak(checkpoint.owned_bytes()?)?;
        Ok(checkpoint)
    }
    /// Exactly one original all-row verification. Returned rows are provisional
    /// target inputs until this round's finish succeeds, not an accepted prefix.
    pub(crate) fn verify(&mut self, cancel: &AtomicBool) -> Result<NativeTargetRows> {
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.packet.phase != Phase::Captured || self.checkpoint.is_none() {
                return Err(rejected());
            }
            let output =
                self.model
                    .prefill_chunk_with_carry(self.executor, self.packet.inputs(), cancel)?;
            let cursor = self.model.completed_target_rows_cursor(self.executor)?;
            self.packet
                .verified(output.binding_id(), output.start(), output.count(), cursor)?;
            self.checkpoint
                .as_mut()
                .ok_or(rejected())?
                .mark_completed(self.model.graph.binding_id(), cursor)?;
            Ok(output)
        }));
        match attempt {
            Ok(Ok(output)) => Ok(output),
            failure => {
                self.terminate();
                Err(match failure {
                    Ok(Err(cause)) => cause,
                    _ => rejected(),
                })
            }
        }
    }
    /// Caller has already inspected/extracted the real greedy/carry and dropped
    /// the verification output. None means all accepted with no extra execution;
    /// Some contains ONLY the real recomputed accepted input prefix.
    pub(crate) fn finish(
        mut self,
        accepted_inputs: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<NativeTargetRows>> {
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.finish_inner(accepted_inputs, cancel)
        }));
        match attempt {
            Ok(Ok(output)) => {
                self.complete = true;
                self.packet.phase = Phase::Consumed;
                Ok(output)
            }
            failure => {
                self.terminate();
                Err(match failure {
                    Ok(Err(cause)) => cause,
                    _ => rejected(),
                })
            }
        }
    }
    fn finish_inner(
        &mut self,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<NativeTargetRows>> {
        let repair = self.packet.accepted(accepted)?;
        let native = self.model.native_batch.as_ref().ok_or(rejected())?;
        native
            .target_rows
            .as_ref()
            .ok_or(rejected())?
            .check_no_live_output()?;
        let cursor = self.model.completed_target_rows_cursor(self.executor)?;
        if cursor != self.packet.geometry[0] + self.packet.geometry[1] {
            return Err(rejected());
        }
        let checkpoint = self.checkpoint.as_ref().ok_or(rejected())?;
        check_peak(checkpoint.owned_bytes()?)?;
        let mut gd = checkpoint.gdn().iter();
        let mut qs = checkpoint.qsa().iter();
        for owner in &self.model.layers {
            match owner {
                Layer::Gdn(stage, _) => gd.next().ok_or(rejected())?.validate_completed(
                    stage,
                    &native.gdn,
                    &self.model.arena,
                    cancel,
                )?,
                Layer::Qsa(stage, _) => qs.next().ok_or(rejected())?.validate_completed(
                    stage,
                    &native.qsa,
                    &self.model.arena,
                    cancel,
                )?,
            }
        }
        if gd.next().is_some() || qs.next().is_some() {
            return Err(rejected());
        }
        checkpoint
            .ple()?
            .validate_completed(&native.ple, self.model, cancel)?;
        // Every real family/full-owner validation completed before consuming
        // any capability. All-accept does not change a cursor or state buffer.
        let parts = self
            .checkpoint
            .take()
            .ok_or(rejected())?
            .into_parts(accepted, repair)?;
        if parts.start() != self.packet.geometry[0]
            || parts.proposed_count() != self.packet.inputs().len()
            || parts.accepted_tokens() != &self.packet.inputs()[..accepted]
        {
            return Err(rejected());
        }
        if !repair {
            let native = self.model.native_batch.as_ref().ok_or(rejected())?;
            let mut gd = parts.gdn.into_iter();
            let mut qs = parts.qsa.into_iter();
            for owner in &self.model.layers {
                match owner {
                    Layer::Gdn(stage, _) => gd.next().ok_or(rejected())?.accept_completed(
                        stage,
                        &native.gdn,
                        &self.model.arena,
                        cancel,
                    )?,
                    Layer::Qsa(stage, _) => qs.next().ok_or(rejected())?.accept_completed(
                        stage,
                        &native.qsa,
                        &self.model.arena,
                        cancel,
                    )?,
                }
            }
            if gd.next().is_some() || qs.next().is_some() {
                return Err(rejected());
            }
            parts
                .ple
                .accept_completed(&native.ple, self.model, cancel)?;
            u::cancel(cancel)?;
            return Ok(None);
        }
        if crate::strata_port::enabled() {
            self.replay_prefix(parts, accepted, cancel)?;
            return Ok(None);
        }
        self.restore(parts, cancel)?;
        let output = self.model.prefill_chunk_with_carry(
            self.executor,
            &self.packet.inputs()[..accepted],
            cancel,
        )?;
        let cursor = self.model.completed_target_rows_cursor(self.executor)?;
        if output.binding_id() != self.model.graph.binding_id()
            || output.start() != self.packet.geometry[0]
            || output.count() != accepted
            || cursor != self.packet.geometry[0] + accepted
        {
            return Err(rejected());
        }
        Ok(Some(output))
    }
    fn replay_prefix(
        &mut self,
        parts: latest_checkpoint::Parts,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let model = &mut *self.model;
        let native = model.native_batch.as_mut().ok_or(rejected())?;
        let start = parts.start();
        let mut gp = Vec::with_capacity(36);
        let mut qp = Vec::with_capacity(12);
        let mut cleared = Vec::with_capacity(12);
        let mut gd = parts.gdn.into_iter();
        let mut qs = parts.qsa.into_iter();
        for owner in &mut model.layers {
            match owner {
                Layer::Gdn(stage, public) => gp.push((
                    gd.next().ok_or(rejected())?.prepare_prefix(
                        stage,
                        &native.gdn,
                        &model.arena,
                        accepted,
                        cancel,
                    )?,
                    public,
                )),
                Layer::Qsa(stage, _) => qp.push(qs.next().ok_or(rejected())?.prepare_prefix(
                    stage,
                    &native.qsa,
                    &model.arena,
                    accepted,
                    cancel,
                )?),
            }
        }
        if gd.next().is_some() || qs.next().is_some() {
            return Err(rejected());
        }
        let pp = parts.ple.prepare_prefix(
            &mut native.ple,
            ple_batch::Preparation {
                graph: model.graph,
                arena: &model.arena,
                admission: &model.admission,
                retired: &model.retired,
                context: model.context,
                position: model.position,
                owner_poisoned: &mut model.poisoned,
                sequence_poison: &model.ple_batch_poison,
                conv: &mut model.ple_conv,
                history: &mut model.ple_history,
                anchor: &mut model.ple_batch_pending,
                fault: model.b.fault,
                scalar_ids: [model.b.ple_emb, model.b.ple_key, model.b.ple_value],
            },
            accepted,
            cancel,
        )?;
        native.gr.validate_all_consumers_queued()?;
        native.shared.validate_all_consumers_queued()?;
        model.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        for (permit, public) in &gp {
            permit.enqueue(&native.gdn, public, &model.arena, model.b.fault, cancel)?;
        }
        for permit in qp {
            cleared.push(permit.enqueue_clear(&model.arena, model.b.fault, cancel)?);
        }
        commit_repair_after_fence(
            || batch_io::read_fault(&model.arena, &model.admission, model.b.fault, cancel),
            cancel,
            || {
                for (permit, _) in gp {
                    permit.commit();
                }
                for permit in cleared {
                    permit.commit();
                }
                pp.commit();
                model.position = start + accepted;
            },
        )
    }
    fn restore(&mut self, parts: latest_checkpoint::Parts, cancel: &AtomicBool) -> Result<()> {
        let model = &mut *self.model;
        let native = model.native_batch.as_mut().ok_or(rejected())?;
        let mut gp = Vec::new();
        let mut qp = Vec::new();
        let mut cleared = Vec::new();
        reserve_exact(&mut gp, 36)?;
        reserve_exact(&mut qp, 12)?;
        reserve_exact(&mut cleared, 12)?;
        let start = parts.start();
        let mut gd = parts.gdn.into_iter();
        let mut qs = parts.qsa.into_iter();
        for owner in &mut model.layers {
            match owner {
                Layer::Gdn(stage, _) => gp.push(gd.next().ok_or(rejected())?.prepare_restore(
                    stage,
                    &native.gdn,
                    &model.arena,
                    cancel,
                )?),
                Layer::Qsa(stage, _) => qp.push(qs.next().ok_or(rejected())?.prepare_restore(
                    stage,
                    &native.qsa,
                    &model.arena,
                    cancel,
                )?),
            }
        }
        if gd.next().is_some() || qs.next().is_some() {
            return Err(rejected());
        }
        let pp = parts.ple.prepare_restore(
            &mut native.ple,
            ple_batch::Preparation {
                graph: model.graph,
                arena: &model.arena,
                admission: &model.admission,
                retired: &model.retired,
                context: model.context,
                position: model.position,
                owner_poisoned: &mut model.poisoned,
                sequence_poison: &model.ple_batch_poison,
                conv: &mut model.ple_conv,
                history: &mut model.ple_history,
                anchor: &mut model.ple_batch_pending,
                fault: model.b.fault,
                scalar_ids: [model.b.ple_emb, model.b.ple_key, model.b.ple_value],
            },
            cancel,
        )?;
        // Every prepared permit now exclusively pins its exact actual owner.
        // Original GR/shared are already Available from completed verification;
        // no phase/policy reset is performed, before OR after restoration.
        native.gr.validate_all_consumers_queued()?;
        native.shared.validate_all_consumers_queued()?;
        model.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        for permit in qp {
            cleared.push(permit.enqueue_clear(&model.arena, model.b.fault, cancel)?);
        }
        commit_repair_after_fence(
            || batch_io::read_fault(&model.arena, &model.admission, model.b.fault, cancel),
            cancel,
            || {
                for permit in gp {
                    permit.commit();
                }
                for permit in cleared {
                    permit.commit();
                }
                pp.commit();
                model.position = start;
            },
        )
    }
    fn terminate(&mut self) {
        self.checkpoint = None;
        self.model.poisoned = true;
        self.executor.poison_resident();
        self.complete = true; // Never retry cleanup from Drop or reset poison.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.model.close();
            let _ = self.executor.close_transient_resident();
        }));
    }
}
impl Drop for LatestTargetRound<'_, '_, '_> {
    fn drop(&mut self) {
        if self.captured && !self.complete {
            self.terminate();
        }
    }
}

fn reserve_exact<T>(values: &mut Vec<T>, count: usize) -> Result<()> {
    values.try_reserve_exact(count).map_err(|_| rejected())?;
    if values.capacity() != count {
        return Err(rejected());
    }
    Ok(())
}
/// No completed-projection capability is minted for a constructor at cursor0.
/// The original completed boundary stays strict. Family captures subsequently
/// authenticate ALL actual Ready0 owners; any later cursor uses the original
/// whole-model completed boundary rather than output metadata.
fn entry_boundary(model: &ResidentModel<'_, '_>, executor: &GpuExecutor<'_>) -> Result<()> {
    if model.position != 0 {
        model.completed_native_projection_boundary(executor)?;
        return Ok(());
    }
    if model.poisoned
        || model.retired.load(Ordering::Acquire)
        || !model.pending_qsa.is_empty()
        || !model.pending_moe.is_empty()
        || model.ple_batch_pending.is_some()
        || !model.admission.transient_io()
        || !model.arena.public_owned_zero_audit_enabled()
        || !model.io.public_owned_zero_audit_enabled()
    {
        return Err(rejected());
    }
    model.graph.verify_unchanged()?;
    executor.check_client(model.arena.client().map_err(u::fault)?)?;
    model
        .admission
        .check_client(model.io.client().map_err(u::fault)?)?;
    executor.check_resident_admission(&model.admission)?;
    let native = model.native_batch.as_ref().ok_or(rejected())?;
    native.prove_outer(model)?;
    native.gr.validate_all_consumers_queued()?;
    native.shared.validate_all_consumers_queued()
}
fn check_peak(captured: usize) -> Result<()> {
    // Existing owner memory remains separately charged. Reserve all new host
    // permit vectors at once, including QSA queued-clear overlap, the round and
    // parts headers and bounded full-resource metadata before private copying.
    let extra = 36 * std::mem::size_of::<gdn::LatestGdnRestore<'_, '_>>()
        + 12 * std::mem::size_of::<qsa::LatestQsaRestore<'_, '_>>()
        + 12 * std::mem::size_of::<qsa::LatestQsaCleared<'_, '_>>()
        + std::mem::size_of::<LatestTargetRound<'_, '_, '_>>()
        + std::mem::size_of::<latest_checkpoint::Parts>()
        + 8192;
    if captured
        .checked_add(extra)
        .is_none_or(|n| n > HOST_INCREMENT)
    {
        return Err(rejected());
    }
    Ok(())
}
fn commit_repair_after_fence(
    fence: impl FnOnce() -> Result<()>,
    cancel: &AtomicBool,
    commit: impl FnOnce(),
) -> Result<()> {
    fence()?;
    u::cancel(cancel)?;
    commit(); // ALL remaining family/cursor changes are infallible.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn construction_requires_reserved_outputs_and_preserves_ordinary_policy() {
        for outputs in [false, true] {
            for duplicate in [false, true] {
                for hybrid in [false, true] {
                    for router in [false, true] {
                        for tile in [1, 4, 8] {
                            assert_eq!(
                                checkpoint_configuration(outputs, duplicate, hybrid, router, tile)
                                    .is_ok(),
                                outputs && !duplicate && !hybrid && !router && tile == 8
                            );
                        }
                    }
                }
            }
        }
        let old = host_reservation(false, DEFAULT_REUSE_METADATA).unwrap();
        let rows =
            host_reservation_with_outputs(false, DEFAULT_REUSE_METADATA, false, true).unwrap();
        let checkpoint = host_reservation_with_latest_checkpoint(DEFAULT_REUSE_METADATA).unwrap();
        assert_eq!(checkpoint, (rows.0 + HOST_INCREMENT, rows.1));
        assert_eq!(checkpoint, (18_808_832, 18_874_368));
        assert_eq!(checkpoint.1 - checkpoint.0, 65_536);
        assert_eq!(old, (11_468_800, 12_582_912));
        assert!(
            host_reservation_with_checkpoint(false, DEFAULT_REUSE_METADATA, false, false, true)
                .is_err()
        );
        assert_eq!(latest_checkpoint::RUNTIME_INCREMENT, 49_152);
        for (tiles, expected) in [(16, 70_344_704), (38, 70_434_816)] {
            let repair_runtime = runtime_bytes(tiles).unwrap()
                + target_rows::RUNTIME_INCREMENT
                + latest_checkpoint::RUNTIME_INCREMENT;
            assert_eq!(repair_runtime, expected);
            assert!(repair_runtime < TARGET_ROWS_RUNTIME_LIMIT);
        }
        assert_eq!(TARGET_ROWS_PRIVATE_MAIN, 826_638_336);
        check_peak(kuno_text::ops::ple_checkpoint::MAX_OWNED_BYTES + 48 * 1024).unwrap();
        assert!(check_peak(HOST_INCREMENT).is_err());
    }
    #[test]
    fn immutable_packet_and_actual_completion_allow_only_one_verification() {
        // Production uses ONLY opaque real GraphBindingId. This private packet
        // test uses harmless IDs without inventing a graph-binding capability.
        let binding = 7u64;
        let mut original = [11, 23, 37];
        let mut packet = Packet::new(1, 4, &original, binding).unwrap();
        original[1] = 99;
        assert_eq!(original[1], 99);
        assert_eq!(packet.inputs(), [11, 23, 37]);
        assert!(packet.accepted(1).is_err());
        assert!(packet.verified(binding, 1, 3, 3).is_err());
        assert!(packet.verified(8u64, 1, 3, 4).is_err());
        packet.verified(binding, 1, 3, 4).unwrap();
        assert!(packet.verified(binding, 1, 3, 4).is_err());
        assert!(packet.accepted(0).is_err());
        assert!(packet.accepted(4).is_err());
        assert!(packet.accepted(1).unwrap());
        assert!(packet.accepted(2).unwrap());
        assert!(!packet.accepted(3).unwrap());
        packet.wipe();
        assert_eq!(*packet.tokens, [0; 4]);
        assert_eq!(*packet.geometry, [0; 2]);
        assert!(packet.binding.is_none());
        assert!(packet.accepted(1).is_err());
    }
    #[test]
    fn bad_inputs_and_context_reject_before_capture_or_execution() {
        for (start, context, tokens) in [
            (0, 4, &[][..]),
            (0, 4, &[1, 2, 3, 4][..]),
            (4, 4, &[1][..]),
            (0, 0, &[1][..]),
            (0, 4097, &[1][..]),
            (usize::MAX, 4096, &[1][..]),
            (0, 4, &[248320][..]),
        ] {
            assert!(Packet::new(start, context, tokens, 7u64).is_err());
        }
        assert!(Packet::new(0, 4, &[1, 2, 3], 7u64).is_ok());
    }
    #[test]
    fn complete_original_fault_audit_and_cancel_precede_any_state_publication() {
        use std::cell::RefCell;
        for fail in [false, true] {
            for cancelled in [false, true] {
                let order = RefCell::new(Vec::new());
                let result = commit_repair_after_fence(
                    || {
                        order.borrow_mut().push("original_fault_and_io_audit");
                        if fail { Err(rejected()) } else { Ok(()) }
                    },
                    &AtomicBool::new(cancelled),
                    || {
                        order.borrow_mut().extend([
                            "gdn_both_pairs",
                            "qsa_cursor",
                            "ple_actual_histories",
                            "outer_cursor",
                        ])
                    },
                );
                assert_eq!(result.is_ok(), !fail && !cancelled);
                assert_eq!(
                    order.borrow().len(),
                    if !fail && !cancelled { 5 } else { 1 }
                );
            }
        }
    }
}
