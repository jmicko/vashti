//! Construction-fixed request-owned I/O for compiled PUBLIC DAS target tests.
//! No environment mutation, model-state reset, new cache implementation or
//! private runtime capability. The original full scope-closing audit is kept.
#![allow(dead_code)] // Parent owns protected registration and GPU acceptance.
use super::*;
use crate::expert_banks::OriginalPackedProfile;
use kuno_text::model::GraphBindingId;
use std::time::Instant;

pub(crate) const METADATA_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const BINDING_ENTRIES: usize = 4096;
const RUNTIME_BYTES: u64 = if cfg!(feature = "strata-intel-core") {
    256
} else {
    128
} * 1024
    * 1024;

fn rejected() -> Error {
    Error::InvalidFormat("public target request I/O boundary rejected")
}
pub(super) fn configuration(
    target_rows: bool,
    already: bool,
    hybrid: bool,
    router: bool,
    tile: usize,
) -> Result<()> {
    if !target_rows || already || hybrid || router || tile != 8 {
        return Err(rejected());
    }
    Ok(())
}
fn runtime_policy(pure_inline: bool, actual_inline: bool, configured: bool) -> Result<()> {
    if !pure_inline || !actual_inline || !configured {
        return Err(rejected());
    }
    Ok(())
}
pub(super) fn check_runtime(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    // Check build and dispatch before invoking the dedicated stream getter.
    // The getter independently checks its exact frozen configuration/budget.
    if crate::target_lab_build::Build::compiled() != crate::target_lab_build::Build::Inline {
        return Err(rejected());
    }
    crate::target_lab_build::validate_runtime(true).map_err(|_| rejected())?;
    runtime_policy(
        true,
        crate::target_lab_build::actual_inline(),
        client
            .public_target_request_io_configured()
            .map_err(|_| rejected())?,
    )
}
pub(super) fn check_plan(config: &Config, executor: &GpuExecutor<'_>) -> Result<()> {
    if !config.public_target_request_io {
        return Ok(());
    }
    configuration(
        config.target_rows_reserved,
        false,
        config.cpu_operator.is_some(),
        config.resident_router_reserved,
        config.tile,
    )?;
    check_runtime(&config.client)?;
    executor.check_target_rows_policy()?;
    if config.request_reuse_metadata != METADATA_BYTES
        || executor.packed_profile != OriginalPackedProfile::DaslabIq3
        || !executor.target_rows_reserved
        || executor.resident_bank_capacity() != crate::daslab_profile::BANK_SLOTS
        || !executor.token_io_checkpoint
        || executor.resident_runtime_io_limit() != RUNTIME_BYTES
    {
        return Err(rejected());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Fresh,
    Active,
    Closed,
}
struct Authority {
    phase: Phase,
    context: usize,
    max_scopes: u64,
    scopes: u64,
    four_rows: bool,
}
impl Authority {
    fn new(context: usize, warm: bool, actual_active: bool) -> Result<Self> {
        if !matches!(context, 4 | 13) || warm && context != 13 || actual_active {
            return Err(rejected());
        }
        Ok(Self {
            four_rows: false,
            phase: Phase::Fresh,
            context,
            max_scopes: if warm { 2 } else { 1 },
            scopes: 0,
        })
    }
    fn active(&self, actual: bool) -> Result<bool> {
        let expected = self.phase == Phase::Active;
        if expected != actual
            || self.scopes > self.max_scopes
            || (self.phase == Phase::Fresh) != (self.scopes == 0)
        {
            return Err(rejected());
        }
        Ok(expected)
    }
    fn validate_run(&self, start: usize, count: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if !(1..=if self.four_rows && self.phase == Phase::Active {
            crate::strata_port::verify_capacity()
        } else {
            3
        })
            .contains(&count)
            || start
                .checked_add(count)
                .is_none_or(|end| end > self.context)
        {
            return Err(rejected());
        }
        match self.phase {
            Phase::Fresh if start == 0 && count == 1 && self.scopes == 0 => Ok(()),
            Phase::Active if start >= 1 => Ok(()),
            Phase::Closed if self.scopes < self.max_scopes && start == 0 && count == 1 => {
                // Only the separately audited one-shot warm reset can restore
                // the actual family/model cursor0. This creates no reset API.
                Ok(())
            }
            _ => Err(rejected()),
        }
    }
    fn validate_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if self.phase == Phase::Active
            || position != 1
            || self.scopes >= self.max_scopes
            || self.phase == Phase::Fresh && self.scopes != 0
        {
            return Err(rejected());
        }
        Ok(())
    }
    fn begin(&mut self, position: usize, actual: bool) -> Result<()> {
        // Validate the already checked idle phase; actual must now attest the
        // successful backend/admission begin, not just a proposed local flag.
        self.validate_begin(position, false)?;
        if !actual {
            return Err(rejected());
        }
        self.scopes = self
            .scopes
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        self.phase = Phase::Active;
        Ok(())
    }
    fn validate_end(&self, position: usize, actual: bool) -> Result<()> {
        if !self.active(actual)? || position != self.context || self.scopes == 0 {
            return Err(rejected());
        }
        Ok(())
    }
    fn end(&mut self, position: usize, actual: bool) -> Result<()> {
        self.validate_end(position, true)?;
        if actual {
            return Err(rejected());
        }
        self.phase = Phase::Closed;
        Ok(())
    }
}
impl Drop for Authority {
    fn drop(&mut self) {
        self.context.zeroize();
        self.max_scopes.zeroize();
        self.scopes.zeroize();
        self.phase = Phase::Closed;
    }
}

#[derive(Default)]
struct Counters {
    checkpoints: u64,
    reuses: u64,
    bindings: [u64; 5],
    info: [u64; 3],
}
impl Counters {
    fn wipe(&mut self) {
        self.checkpoints.zeroize();
        self.reuses.zeroize();
        self.bindings.zeroize();
        self.info.zeroize();
    }
    fn capture(admission: &ResidentAdmission) -> Self {
        let (hits, misses, pins) = cubecl::wgpu::transient_info_cache_stats();
        Self {
            checkpoints: admission.request_io_checkpoints(),
            reuses: cubecl::wgpu::transient_request_io_reuses(),
            bindings: cubecl::wgpu::transient_binding_cache_stats(),
            info: [hits, misses, pins],
        }
    }
    fn difference(&self, before: &Self) -> Result<Self> {
        if [
            self.checkpoints,
            self.reuses,
            before.checkpoints,
            before.reuses,
        ]
        .into_iter()
        .chain(self.bindings)
        .chain(before.bindings)
        .chain(self.info)
        .chain(before.info)
        .any(|value| value == u64::MAX)
        {
            // Existing process counters saturate; a saturated sample cannot
            // truthfully distinguish zero work from unrecorded additional work.
            return Err(rejected());
        }
        let mut result = Self {
            checkpoints: self
                .checkpoints
                .checked_sub(before.checkpoints)
                .ok_or(rejected())?,
            reuses: self.reuses.checked_sub(before.reuses).ok_or(rejected())?,
            ..Self::default()
        };
        for (out, (after, before)) in result
            .bindings
            .iter_mut()
            .zip(self.bindings.iter().zip(&before.bindings))
        {
            *out = after.checked_sub(*before).ok_or(rejected())?;
        }
        for (out, (after, before)) in result
            .info
            .iter_mut()
            .zip(self.info.iter().zip(&before.info))
        {
            *out = after.checked_sub(*before).ok_or(rejected())?;
        }
        Ok(result)
    }
}
impl Drop for Counters {
    fn drop(&mut self) {
        self.wipe();
    }
}
pub(super) struct Permit {
    binding: Option<GraphBindingId>,
    authority: Authority,
    before: Option<Counters>,
}
impl Permit {
    pub(super) fn enable_four_rows(&mut self) -> Result<()> {
        if self.authority.phase == Phase::Active || self.authority.four_rows {
            return Err(rejected());
        }
        self.authority.four_rows = true;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Construction captures separate ownership attestations.
    pub(super) fn new(
        binding: GraphBindingId,
        profile: OriginalPackedProfile,
        context: usize,
        position: usize,
        transient: bool,
        zero_audit: bool,
        warm: bool,
        actual_active: bool,
    ) -> Result<Self> {
        construction(profile, position, transient, zero_audit)?;
        Ok(Self {
            binding: Some(binding),
            authority: Authority::new(context, warm, actual_active)?,
            before: None,
        })
    }
    fn identity(&self, binding: GraphBindingId, context: usize) -> Result<()> {
        if !same_identity(self.binding, binding, self.authority.context, context)
            || (self.authority.phase == Phase::Active) != self.before.is_some()
        {
            return Err(rejected());
        }
        Ok(())
    }
    pub(super) fn validate_run(
        &self,
        binding: GraphBindingId,
        context: usize,
        start: usize,
        count: usize,
        actual: bool,
    ) -> Result<()> {
        self.identity(binding, context)?;
        self.authority.validate_run(start, count, actual)
    }
}
fn same_identity<T: PartialEq>(
    captured: Option<T>,
    current: T,
    captured_context: usize,
    context: usize,
) -> bool {
    captured == Some(current) && captured_context == context
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.binding = None;
        self.before = None;
    }
}
fn construction(
    profile: OriginalPackedProfile,
    position: usize,
    transient: bool,
    zero_audit: bool,
) -> Result<()> {
    if profile != OriginalPackedProfile::DaslabIq3 || position != 0 || !transient || !zero_audit {
        return Err(rejected());
    }
    Ok(())
}

pub(super) fn check_execution(
    native: &Workspace,
    model: &ResidentModel<'_, '_>,
    executor: &GpuExecutor<'_>,
) -> Result<()> {
    check_runtime(model.arena.client().map_err(u::fault)?)?;
    executor.check_target_rows_policy()?;
    native.validate_target_decode_policy(executor, &model.admission)?;
    if native.target_rows.is_none()
        || native.moe.cpu_miss_cap().is_some()
        || native.moe.cpu_gpu_overlap().is_some()
        || native.moe.resident_router_active()
        || executor.packed_profile != OriginalPackedProfile::DaslabIq3
        || !executor.token_io_checkpoint
        || !model.admission.deferred_token_io()
        || executor.resident_runtime_io_limit() != RUNTIME_BYTES
        || !model.admission.transient_io()
        || !model.arena.public_owned_zero_audit_enabled()
        || !model.io.public_owned_zero_audit_enabled()
    {
        return Err(rejected());
    }
    Ok(())
}

/// Numeric aggregate deltas only. Backend cache/INFO counters are PROCESS
/// global; concurrent work on another client may contribute to those deltas.
pub(crate) struct Receipt {
    pub(crate) scope: u64,
    pub(crate) scope_closed: bool,
    pub(crate) full_io_audit: bool,
    pub(crate) target_stream_configured: bool,
    pub(crate) metadata_bytes: usize,
    pub(crate) binding_capacity: usize,
    pub(crate) runtime_limit: u64,
    pub(crate) request_checkpoints: u64,
    pub(crate) request_io_reuses: u64,
    pub(crate) binding_stats: [u64; 5],
    pub(crate) info_stats: [u64; 3],
    pub(crate) closing_audit_ns: u64,
}
impl<'m, 'f> ResidentModel<'m, 'f> {
    fn public_target_request_boundary(
        &self,
        executor: &GpuExecutor<'f>,
        begin: bool,
    ) -> Result<()> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(rejected())?;
        check_execution(native, self, executor)?;
        native
            .target_rows
            .as_ref()
            .ok_or(rejected())?
            .check_no_live_output()?;
        let permit = native.public_target_request_io.as_ref().ok_or(rejected())?;
        permit.identity(self.graph.binding_id(), self.context)?;
        if begin {
            permit
                .authority
                .validate_begin(self.position, self.admission.request_io_reuse_active())?;
            if permit.before.is_some() {
                return Err(rejected());
            }
        } else {
            permit
                .authority
                .validate_end(self.position, self.admission.request_io_reuse_active())?;
            if permit.before.is_none() {
                return Err(rejected());
            }
        }
        Ok(())
    }

    pub(crate) fn begin_public_target_request_io(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        // A wrong/default/cursor/lease preflight does not close a healthy model.
        u::cancel(cancel)?;
        self.public_target_request_boundary(executor, true)?;
        let before = Counters::capture(&self.admission);
        self.lab_oracle_attempt(executor, |model, _| {
            model.admission.begin_public_target_request_io_reuse()?;
            u::cancel(cancel)?;
            let actual = model.admission.request_io_reuse_active();
            let permit = model
                .native_batch
                .as_mut()
                .ok_or(rejected())?
                .public_target_request_io
                .as_mut()
                .ok_or(rejected())?;
            permit.authority.begin(model.position, actual)?;
            permit.before = Some(before);
            Ok(())
        })
    }

    pub(crate) fn public_target_request_io_active(&self) -> Result<bool> {
        if self.poisoned || self.retired.load(Ordering::Acquire) {
            return Err(rejected());
        }
        let actual = self.admission.request_io_reuse_active();
        let native = self.native_batch.as_ref().ok_or(rejected())?;
        match &native.public_target_request_io {
            Some(permit) => {
                permit.identity(self.graph.binding_id(), self.context)?;
                permit.authority.active(actual)
            }
            None if !actual => Ok(false),
            None => Err(rejected()),
        }
    }

    pub(crate) fn end_public_target_request_io(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        u::cancel(cancel)?;
        self.public_target_request_boundary(executor, false)?;
        self.lab_oracle_attempt(executor, |model, _| {
            let started = Instant::now();
            // Join the original complete owning fault/token checkpoint before
            // the mandatory upload+uniform+readback full scope-closing audit.
            batch_io::read_fault(&model.arena, &model.admission, model.b.fault, cancel)?;
            model.admission.end_request_io_reuse()?;
            u::cancel(cancel)?;
            check_runtime(model.arena.client().map_err(u::fault)?)?;
            let closing_audit_ns = u64::try_from(started.elapsed().as_nanos())
                .map_err(|_| Error::ArithmeticOverflow)?;
            let after = Counters::capture(&model.admission);
            let actual = model.admission.request_io_reuse_active();
            let permit = model
                .native_batch
                .as_mut()
                .ok_or(rejected())?
                .public_target_request_io
                .as_mut()
                .ok_or(rejected())?;
            let before = permit.before.as_ref().ok_or(rejected())?;
            let difference = after.difference(before)?;
            permit.authority.end(model.position, actual)?;
            permit.before = None; // Wipe the consumed numeric snapshot.
            Ok(Receipt {
                scope: permit.authority.scopes,
                scope_closed: !actual,
                full_io_audit: true,
                target_stream_configured: true,
                metadata_bytes: METADATA_BYTES,
                binding_capacity: BINDING_ENTRIES,
                runtime_limit: RUNTIME_BYTES,
                request_checkpoints: difference.checkpoints,
                request_io_reuses: difference.reuses,
                binding_stats: difference.bindings,
                info_stats: difference.info,
                closing_audit_ns,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_request_permission_never_changes_cold_or_closed_scope() {
        let mut authority = Authority::new(13, true, false).unwrap();
        authority.four_rows = true;
        assert!(authority.validate_run(0, 4, false).is_err());
        authority.validate_run(0, 1, false).unwrap();
        authority.begin(1, true).unwrap();
        authority.validate_run(1, 4, true).unwrap();
        assert!(authority.validate_run(1, 5, true).is_err());
        authority.end(13, false).unwrap();
        assert!(authority.validate_run(1, 4, false).is_err());
        let mut original = Authority::new(13, true, false).unwrap();
        original.begin(1, true).unwrap();
        assert!(original.validate_run(1, 4, true).is_err());
    }
    #[test]
    fn construction_requires_exact_public_target_and_fixed_direct_stream() {
        for rows in [false, true] {
            for duplicate in [false, true] {
                for hybrid in [false, true] {
                    for router in [false, true] {
                        for tile in [0, 1, 3, 8, 9] {
                            assert_eq!(
                                configuration(rows, duplicate, hybrid, router, tile).is_ok(),
                                rows && !duplicate && !hybrid && !router && tile == 8
                            );
                        }
                    }
                }
            }
        }
        for compiled in [false, true] {
            for actual in [false, true] {
                for configured in [false, true] {
                    assert_eq!(
                        runtime_policy(compiled, actual, configured).is_ok(),
                        compiled && actual && configured
                    );
                }
            }
        }
        for profile in [
            OriginalPackedProfile::Q4,
            OriginalPackedProfile::UnslothQ3,
            OriginalPackedProfile::DaslabIq3,
        ] {
            for position in [0, 1, 13] {
                for transient in [false, true] {
                    for audited in [false, true] {
                        assert_eq!(
                            construction(profile, position, transient, audited).is_ok(),
                            profile == OriginalPackedProfile::DaslabIq3
                                && position == 0
                                && transient
                                && audited
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn public_prefill_then_completed_full_scope_is_exclusive_and_monotonic() {
        for context in [4, 13] {
            let mut a = Authority::new(context, false, false).unwrap();
            assert!(a.validate_run(0, 1, false).is_ok());
            for (position, count) in [(0, 2), (0, 3), (1, 1), (0, 0), (0, 4)] {
                assert!(a.validate_run(position, count, false).is_err());
            }
            assert!(a.validate_begin(0, false).is_err());
            assert!(a.validate_begin(1, true).is_err());
            assert!(a.begin(1, false).is_err());
            assert_eq!(a.scopes, 0);
            a.begin(1, true).unwrap();
            assert_eq!(a.scopes, 1);
            assert!(a.validate_begin(1, true).is_err());
            assert!(a.validate_run(0, 1, true).is_err());
            for count in 1..=3 {
                assert!(a.validate_run(context - count, count, true).is_ok());
            }
            assert!(a.validate_run(context, 1, true).is_err());
            assert!(a.validate_run(1, 1, false).is_err());
            assert!(a.validate_end(context - 1, true).is_err());
            assert!(a.end(context, true).is_err());
            assert_eq!(a.phase, Phase::Active);
            a.end(context, false).unwrap();
            assert!(!a.active(false).unwrap());
            assert!(a.end(context, false).is_err());
            assert!(a.validate_run(0, 1, false).is_err());
            assert!(a.validate_begin(1, false).is_err());
        }
    }
    #[test]
    fn warm_scope_requires_real_reset_prefill_and_is_limited_to_two() {
        for (context, warm, active) in [(0, false, false), (4, true, false), (13, true, true)] {
            assert!(Authority::new(context, warm, active).is_err());
        }
        let mut a = Authority::new(13, true, false).unwrap();
        a.begin(1, true).unwrap();
        a.end(13, false).unwrap();
        assert!(a.validate_run(13, 1, false).is_err());
        assert!(a.validate_begin(0, false).is_err());
        assert!(a.validate_run(0, 1, false).is_ok());
        a.begin(1, true).unwrap();
        assert_eq!(a.scopes, 2);
        a.end(13, false).unwrap();
        assert!(a.validate_begin(1, false).is_err());
        assert!(a.validate_run(0, 1, false).is_err());
        a.scopes = 3;
        assert!(a.active(false).is_err());
    }
    #[test]
    fn counter_differences_are_complete_checked_and_wiped() {
        let before = Counters {
            checkpoints: 11,
            reuses: 7,
            bindings: [1, 2, 3, 4, 5],
            info: [6, 7, 8],
        };
        let mut after = Counters {
            checkpoints: 15,
            reuses: 12,
            bindings: [7, 9, 11, 13, 15],
            info: [17, 19, 21],
        };
        let d = after.difference(&before).unwrap();
        assert_eq!((d.checkpoints, d.reuses), (4, 5));
        assert_eq!(d.bindings, [6, 7, 8, 9, 10]);
        assert_eq!(d.info, [11, 12, 13]);
        after.checkpoints = 10;
        assert!(after.difference(&before).is_err());
        after.checkpoints = 15;
        after.reuses = u64::MAX;
        assert!(after.difference(&before).is_err());
        after.reuses = 12;
        for i in 0..5 {
            let old = after.bindings[i];
            after.bindings[i] = before.bindings[i] - 1;
            assert!(after.difference(&before).is_err());
            after.bindings[i] = old;
        }
        for i in 0..3 {
            let old = after.info[i];
            after.info[i] = before.info[i] - 1;
            assert!(after.difference(&before).is_err());
            after.info[i] = old;
        }
        after.wipe();
        assert_eq!(after.bindings, [0; 5]);
        assert_eq!(after.info, [0; 3]);
        assert_eq!((after.checkpoints, after.reuses), (0, 0));
    }
    #[test]
    fn stale_context_binding_and_corrupt_scope_identity_are_rejected() {
        assert!(same_identity(Some(3u64), 3, 13, 13));
        assert!(!same_identity(Some(3u64), 4, 13, 13));
        assert!(!same_identity(Some(3u64), 3, 13, 4));
        assert!(!same_identity(None, 3u64, 13, 13));
        let mut a = Authority::new(13, true, false).unwrap();
        a.phase = Phase::Closed;
        assert!(a.active(false).is_err());
        assert!(a.validate_begin(1, false).is_err());
        a.phase = Phase::Fresh;
        a.scopes = 1;
        assert!(a.active(false).is_err());
        assert!(a.validate_run(0, 1, false).is_err());
    }
    #[test]
    fn full_four_mib_metadata_charge_fits_existing_target_limits_without_main_growth() {
        assert_eq!(METADATA_BYTES, 4194304);
        assert_eq!(BINDING_ENTRIES, 4096);
        assert_eq!(RUNTIME_BYTES, TARGET_ROWS_RUNTIME_LIMIT);
        assert_eq!(
            host_reservation_with_outputs(false, METADATA_BYTES, false, true).unwrap(),
            (21692416, 22806528)
        );
        assert_eq!(
            host_reservation_with_latest_checkpoint(METADATA_BYTES).unwrap(),
            (22740992, 22806528)
        );
        assert_eq!(
            host_reservation_with_prepared_verify3(METADATA_BYTES, true).unwrap(),
            (22806528, 22806528)
        );
        assert_eq!(
            host_reservation_with_latest_checkpoint(DEFAULT_REUSE_METADATA).unwrap(),
            (18808832, 18874368)
        );
        assert!(metadata_bytes() <= META_CAP);
        assert!(std::mem::size_of::<Permit>() < 1024);
    }
}
