//! One construction-fixed, successful thirteen-row PUBLIC warm replay only.
//! Model weights/banks and every actual admission/epoch/counter remain owned.
//! This is not a recovery API or a generic snapshot/cursor rewind. Failed
//! erasure is terminal; no poisoned flag or physical retirement is cleared.
#![allow(dead_code)] // Parent owns protected registration and physical tests.
use super::*;
use crate::{expert_banks::OriginalPackedProfile, private_arena::OwnedZeroAudit};
use kuno_text::model::GraphBindingId;

const END: usize = 13;
const _: () = assert!(8192 < latest_checkpoint::HOST_INCREMENT);
fn rejected() -> Error {
    Error::InvalidFormat("public warm replay boundary rejected")
}

pub(super) fn configuration(
    target_rows: bool,
    latest: bool,
    hybrid: bool,
    router: bool,
    tile: usize,
    already: bool,
) -> Result<()> {
    if !target_rows || !latest || hybrid || router || tile != 8 || already {
        return Err(rejected());
    }
    Ok(())
}

#[derive(Default)]
struct Authority {
    consumed: bool,
}
impl Authority {
    fn validate(&self, context: usize, position: usize) -> Result<()> {
        if self.consumed || context != END || position != END {
            return Err(rejected());
        }
        Ok(())
    }
    fn consume(&mut self, context: usize, position: usize) -> Result<()> {
        self.validate(context, position)?;
        self.consumed = true;
        Ok(())
    }
}

pub(super) struct Permit {
    binding: Option<GraphBindingId>,
    authority: Authority,
}
impl Permit {
    pub(super) fn reset_consumed(&self) -> bool {
        self.authority.consumed
    }
    pub(super) fn new(
        binding: GraphBindingId,
        profile: OriginalPackedProfile,
        context: usize,
        position: usize,
        transient: bool,
        zero_audit: bool,
    ) -> Result<Self> {
        construction(profile, context, position, transient, zero_audit)?;
        Ok(Self {
            binding: Some(binding),
            authority: Authority::default(),
        })
    }
    fn validate(&self, binding: GraphBindingId, context: usize, position: usize) -> Result<()> {
        if self.binding != Some(binding) {
            return Err(rejected());
        }
        self.authority.validate(context, position)
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.binding = None;
    }
}
fn construction(
    profile: OriginalPackedProfile,
    context: usize,
    position: usize,
    transient: bool,
    zero_audit: bool,
) -> Result<()> {
    if profile != OriginalPackedProfile::DaslabIq3
        || context != END
        || position != 0
        || !transient
        || !zero_audit
    {
        return Err(rejected());
    }
    Ok(())
}

/// Distinct from arena-close erasure: these fully inspected pages remain
/// owned/charged and are independently inspected again during final close.
pub(crate) struct PublicWarmResetReceipt {
    pub(crate) main: OwnedZeroAudit,
    pub(crate) source: OwnedZeroAudit,
}
/// Only this coordinator can mint the all-arena/source/fault/final-cancel
/// capability. Family permits cannot publish merely from a Ready cursor.
pub(crate) struct CompletedReset(());
fn commit_after_reset_fence<T>(
    fence: impl FnOnce() -> Result<()>,
    cancel: &AtomicBool,
    commit: impl FnOnce(&CompletedReset) -> T,
) -> Result<T> {
    fence()?;
    u::cancel(cancel)?;
    Ok(commit(&CompletedReset(())))
}

impl ResidentModel<'_, '_> {
    pub(crate) fn reset_public_warm_replay(
        &mut self,
        executor: &mut GpuExecutor<'_>,
        cancel: &AtomicBool,
    ) -> Result<PublicWarmResetReceipt> {
        // Wrong/default construction, count or duplicate calls are read-only.
        u::cancel(cancel)?;
        self.native_batch
            .as_ref()
            .ok_or(rejected())?
            .public_warm_replay
            .as_ref()
            .ok_or(rejected())?
            .validate(self.graph.binding_id(), self.context, self.position)?;
        let mut erase_started = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.reset_public_warm_checked(executor, cancel, &mut erase_started)
        }));
        match result {
            Ok(Ok(receipt)) => Ok(receipt),
            other => {
                // Preserve non-mutating policy preflight rejection. Original
                // guard failures may already poison admission; never reuse it.
                if erase_started || executor.check_resident_admission(&self.admission).is_err() {
                    self.poisoned = true;
                    executor.poison_resident();
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = self.close();
                        let _ = executor.close_transient_resident();
                    }));
                }
                match other {
                    Ok(Err(error)) => Err(error),
                    _ => Err(rejected()),
                }
            }
        }
    }

    fn reset_public_warm_checked(
        &mut self,
        executor: &mut GpuExecutor<'_>,
        cancel: &AtomicBool,
        erase_started: &mut bool,
    ) -> Result<PublicWarmResetReceipt> {
        self.completed_native_projection_boundary(executor)?;
        if self.native_decode_kernels()? != NativeDecodeKernels::Chunk
            || executor.packed_profile != OriginalPackedProfile::DaslabIq3
            || self.cancel_after_attention.is_some()
            || !self.lifecycle_hooks.is_unarmed()
            || !self.numeric_boundary.is_unarmed()
            || self.layers.len() != LAYERS
            || self.moe_layers.len() != LAYERS
            || executor.ordered_source_pending_bytes() != 0
        {
            return Err(rejected());
        }
        let native = self.native_batch.as_ref().ok_or(rejected())?;
        if !native.latest_target_checkpoint_reserved {
            return Err(rejected());
        }
        native
            .target_rows
            .as_ref()
            .ok_or(rejected())?
            .check_no_live_output()?;
        native
            .moe
            .validate_public_warm_boundary(executor, self.graph, &self.arena)?;
        for (layer, moe) in self.moe_layers.iter().enumerate() {
            moe.validate_public_warm_boundary(self.graph.binding_id(), layer)?;
        }
        // Join the ORIGINAL actual complete fault/I/O before any erasure of
        // Fault could conceal a failed forward. A metadata cursor is not proof.
        batch_io::read_fault(&self.arena, &self.admission, self.b.fault, cancel)?;
        self.admission.end_request_io_reuse()?;
        self.admission.cleanup_abandoned_ordered_sources()?;
        let native = self.native_batch.as_mut().ok_or(rejected())?;
        let mut gdn_permits: [Option<gdn::PublicWarmGdnReset<'_, '_>>; LAYERS] =
            std::array::from_fn(|_| None);
        let mut qsa_permits: [Option<qsa::PublicWarmQsaReset<'_, '_>>; LAYERS] =
            std::array::from_fn(|_| None);
        for (layer, owner) in self.layers.iter_mut().enumerate() {
            match owner {
                Layer::Gdn(stage, _) if layer % 4 != 3 => {
                    gdn_permits[layer] = Some(native.gdn.prepare_public_warm_reset(
                        stage,
                        &self.arena,
                        END as u64,
                        cancel,
                    )?);
                }
                Layer::Qsa(stage, _) if layer % 4 == 3 => {
                    qsa_permits[layer] = Some(native.qsa.prepare_public_warm_reset(
                        stage,
                        &self.arena,
                        self.b.fault,
                        END as u64,
                        cancel,
                    )?);
                }
                _ => return Err(rejected()),
            }
        }
        let mut ple_permit = native.ple.prepare_public_warm_reset(
            ple_batch::Preparation {
                graph: self.graph,
                arena: &self.arena,
                admission: &self.admission,
                retired: &self.retired,
                context: self.context,
                position: self.position,
                owner_poisoned: &mut self.poisoned,
                sequence_poison: &self.ple_batch_poison,
                conv: &mut self.ple_conv,
                history: &mut self.ple_history,
                anchor: &mut self.ple_batch_pending,
                fault: self.b.fault,
                scalar_ids: [self.b.ple_emb, self.b.ple_key, self.b.ple_value],
            },
            cancel,
        )?;
        u::cancel(cancel)?;
        native
            .public_warm_replay
            .as_mut()
            .ok_or(rejected())?
            .authority
            .consume(self.context, self.position)?;
        for permit in gdn_permits.iter_mut().flatten() {
            permit.arm();
        }
        for permit in qsa_permits.iter_mut().flatten() {
            permit.arm();
        }
        ple_permit.arm();
        *erase_started = true;
        // No epoch/handle/charge release. The complete tracked inventory covers
        // both GDN pairs/rings, full QSA KV/rawIK/positions, ALL scalar/native
        // scratch/router/status/fault/row-output pages and all source copies.
        let main = self
            .arena
            .erase_retained_public_owners()
            .map_err(u::fault)?;
        let source = self.io.erase_retained_public_owners().map_err(u::fault)?;
        // ALL remaining commits are infallible. Only numerical cursors and the
        // two original host PLE histories change; monotonic stamps/counters and
        // physical ownership survive. No poisoning/retirement bit is cleared.
        commit_after_reset_fence(
            || {
                self.graph.verify_unchanged()?;
                executor.check_resident_admission(&self.admission)?;
                batch_io::read_fault(&self.arena, &self.admission, self.b.fault, cancel)?;
                if executor.ordered_source_pending_bytes() != 0 {
                    return Err(rejected());
                }
                Ok(())
            },
            cancel,
            |proof| {
                for permit in gdn_permits.into_iter().flatten() {
                    permit.commit(proof);
                }
                for permit in qsa_permits.into_iter().flatten() {
                    permit.commit(proof);
                }
                ple_permit.commit(proof);
                self.position = 0;
                PublicWarmResetReceipt { main, source }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_one_fixed_completed_thirteen_row_replay_is_authorized() {
        let mut a = Authority::default();
        for context in [0, 4, 12, 13, 14, 4096] {
            for cursor in [0, 1, 12, 13, 14, usize::MAX] {
                assert_eq!(
                    a.validate(context, cursor).is_ok(),
                    context == 13 && cursor == 13
                );
            }
        }
        assert!(a.consume(13, 12).is_err());
        assert!(!a.consumed);
        a.consume(13, 13).unwrap();
        assert!(a.validate(13, 13).is_err());
        assert!(a.consume(13, 13).is_err());
        assert!(a.consumed);
    }
    #[test]
    fn default_private_wrong_profile_and_hybrid_construction_cannot_enable_reset() {
        for outputs in [false, true] {
            for latest in [false, true] {
                for hybrid in [false, true] {
                    for router in [false, true] {
                        for tile in [1, 4, 8] {
                            for already in [false, true] {
                                assert_eq!(
                                    configuration(outputs, latest, hybrid, router, tile, already)
                                        .is_ok(),
                                    outputs
                                        && latest
                                        && !hybrid
                                        && !router
                                        && tile == 8
                                        && !already
                                );
                            }
                        }
                    }
                }
            }
        }
        for profile in [
            OriginalPackedProfile::Q4,
            OriginalPackedProfile::UnslothQ3,
            OriginalPackedProfile::DaslabIq3,
        ] {
            for context in [4, 13, 32] {
                for position in [0, 1, 13] {
                    for transient in [false, true] {
                        for audit in [false, true] {
                            assert_eq!(
                                construction(profile, context, position, transient, audit).is_ok(),
                                profile == OriginalPackedProfile::DaslabIq3
                                    && context == 13
                                    && position == 0
                                    && transient
                                    && audit
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn fixed_permit_metadata_fits_existing_reserved_checkpoint_envelope() {
        let bytes = std::mem::size_of::<Permit>()
            + LAYERS * std::mem::size_of::<Option<gdn::PublicWarmGdnReset<'_, '_>>>()
            + LAYERS * std::mem::size_of::<Option<qsa::PublicWarmQsaReset<'_, '_>>>()
            + std::mem::size_of::<ple_batch::PublicWarmPleReset<'_>>()
            + std::mem::size_of::<PublicWarmResetReceipt>();
        assert!(bytes < 8192);
    }
    #[test]
    fn final_fault_failure_or_late_cancellation_cannot_publish_any_reset_cursor() {
        use std::cell::RefCell;
        for fault in [false, true] {
            for late_cancel in [false, true] {
                let cancel = AtomicBool::new(false);
                let cursors = RefCell::new([13u64; 50]); //36 GDN,12 QSA,PLE,model.
                let result = commit_after_reset_fence(
                    || {
                        cancel.store(late_cancel, Ordering::Release);
                        if fault { Err(rejected()) } else { Ok(()) }
                    },
                    &cancel,
                    |_| {
                        cursors.borrow_mut().fill(0);
                    },
                );
                assert_eq!(result.is_ok(), !fault && !late_cancel);
                assert!(
                    cursors
                        .borrow()
                        .iter()
                        .all(|&p| p == if result.is_ok() { 0 } else { 13 })
                );
            }
        }
    }
}
