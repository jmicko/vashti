//! A separate, single PUBLIC language request capability. The historical
//! target4/target13 permit is unchanged. It authenticates owner/lifecycle and
//! compiled fixture geometry; the language harness binds actual prompt bytes.
#![allow(dead_code)] // Parent owns constructor integration and physical acceptance.
use super::*;
use crate::expert_banks::OriginalPackedProfile;
use kuno_text::model::GraphBindingId;

pub(crate) const METADATA_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const BINDING_ENTRIES: usize = 4096;
pub(crate) const RUNTIME_BYTES: u64 = 128 * 1024 * 1024;
pub(crate) const MAX_OUTPUTS: usize = 200;

fn rejected() -> Error {
    Error::InvalidFormat("public language request I/O boundary rejected")
}

/// Identity of one of the two compiled thinking-off ChatML fixtures. No
/// arbitrary context, prompt length, generation limit or warm-reset permit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    prompt: usize,
}
impl Geometry {
    pub(crate) fn new(prompt: usize, outputs: usize) -> Result<Self> {
        if !matches!(prompt, 30 | 35) || outputs != MAX_OUTPUTS {
            return Err(rejected());
        }
        Ok(Self { prompt })
    }
    pub(crate) fn prompt(self) -> usize {
        self.prompt
    }
    /// Retain capacity for the whole public output budget. The first output
    /// was predicted by prompt prefill; at most199 continuation inputs run.
    pub(crate) fn context(self) -> usize {
        self.prompt + MAX_OUTPUTS
    }
    pub(crate) fn final_cursor(self) -> usize {
        self.context() - 1
    }
}

pub(super) fn configuration(
    target_rows: bool,
    checkpoint: bool,
    already: bool,
    hybrid: bool,
    router: bool,
    warm: bool,
    tile: usize,
) -> Result<()> {
    if !target_rows || !checkpoint || already || hybrid || router || warm || tile != 8 {
        return Err(rejected());
    }
    Ok(())
}
pub(super) fn check_runtime(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    super::public_target_request_io::check_runtime(client)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Prefill,
    Active,
    Closed,
}
struct Authority {
    geometry: Geometry,
    phase: Phase,
    scopes: u64,
    four_rows: bool,
}
impl Authority {
    fn new(geometry: Geometry, context: usize, actual: bool) -> Result<Self> {
        Geometry::new(geometry.prompt, MAX_OUTPUTS)?;
        if context != geometry.context() || actual {
            return Err(rejected());
        }
        Ok(Self {
            four_rows: false,
            geometry,
            phase: Phase::Prefill,
            scopes: 0,
        })
    }
    fn active(&self, actual: bool) -> Result<bool> {
        if actual != (self.phase == Phase::Active)
            || self.scopes != u64::from(self.phase != Phase::Prefill)
        {
            return Err(rejected());
        }
        Ok(actual)
    }
    fn run(&self, start: usize, count: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if !(1..=if self.four_rows && self.phase == Phase::Active {
            4
        } else {
            3
        })
            .contains(&count)
        {
            return Err(rejected());
        }
        let end = start.checked_add(count).ok_or(rejected())?;
        match self.phase {
            Phase::Prefill if end <= self.geometry.prompt => Ok(()),
            Phase::Active
                if start >= self.geometry.prompt && end <= self.geometry.final_cursor() =>
            {
                Ok(())
            }
            _ => Err(rejected()),
        }
    }
    fn validate_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if self.phase != Phase::Prefill || position != self.geometry.prompt || actual {
            return Err(rejected());
        }
        Ok(())
    }
    fn begin(&mut self, position: usize, actual: bool) -> Result<()> {
        self.validate_begin(position, false)?;
        if !actual {
            return Err(rejected());
        }
        self.phase = Phase::Active;
        self.scopes = 1;
        Ok(())
    }
    fn validate_end(&self, position: usize, actual: bool) -> Result<()> {
        if !self.active(actual)?
            || !(self.geometry.prompt..=self.geometry.final_cursor()).contains(&position)
        {
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
        self.geometry.prompt.zeroize();
        self.scopes.zeroize();
        self.phase = Phase::Closed;
    }
}

pub(super) struct Permit {
    binding: Option<GraphBindingId>,
    authority: Authority,
}
impl Permit {
    pub(super) fn enable_four_rows(&mut self) -> Result<()> {
        if self.authority.phase == Phase::Active || self.authority.four_rows {
            return Err(rejected());
        }
        self.authority.four_rows = true;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Separate actual construction attestations.
    pub(super) fn new(
        binding: GraphBindingId,
        profile: OriginalPackedProfile,
        geometry: Geometry,
        context: usize,
        position: usize,
        transient: bool,
        zero_audit: bool,
        actual: bool,
    ) -> Result<Self> {
        if profile != OriginalPackedProfile::DaslabIq3 || position != 0 || !transient || !zero_audit
        {
            return Err(rejected());
        }
        Ok(Self {
            binding: Some(binding),
            authority: Authority::new(geometry, context, actual)?,
        })
    }
    pub(super) fn identity(&self, binding: GraphBindingId, context: usize) -> Result<()> {
        if self.binding != Some(binding) || context != self.authority.geometry.context() {
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
        self.authority.run(start, count, actual)
    }
    pub(super) fn active(&self, actual: bool) -> Result<bool> {
        self.authority.active(actual)
    }
    pub(super) fn validate_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.authority.validate_begin(position, actual)
    }
    pub(super) fn begin(&mut self, position: usize, actual: bool) -> Result<()> {
        self.authority.begin(position, actual)
    }
    pub(super) fn validate_end(&self, position: usize, actual: bool) -> Result<()> {
        self.authority.validate_end(position, actual)
    }
    pub(super) fn end(&mut self, position: usize, actual: bool) -> Result<()> {
        self.authority.end(position, actual)
    }
    pub(super) fn scope(&self) -> u64 {
        self.authority.scopes
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.binding = None;
    }
}

/// Content-free proof of the original complete scope-close audit. Root records
/// checked process-global binding/INFO deltas separately; no runtime nonce.
pub(crate) struct Receipt {
    pub(crate) scope: u64,
    pub(crate) full_io_audit: bool,
    pub(crate) scope_closed: bool,
    pub(crate) closing_audit_ns: u64,
}
impl<'m, 'f> ResidentModel<'m, 'f> {
    fn public_language_request_boundary(
        &self,
        executor: &GpuExecutor<'f>,
        begin: bool,
    ) -> Result<()> {
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or(rejected())?;
        super::public_target_request_io::check_execution(native, self, executor)?;
        native
            .target_rows
            .as_ref()
            .ok_or(rejected())?
            .check_no_live_output()?;
        let permit = native
            .public_language_request_io
            .as_ref()
            .ok_or(rejected())?;
        permit.identity(self.graph.binding_id(), self.context)?;
        let actual = self.admission.request_io_reuse_active();
        if begin {
            permit.validate_begin(self.position, actual)
        } else {
            permit.validate_end(self.position, actual)
        }
    }
    pub(crate) fn begin_public_language_request_io(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        self.public_language_request_boundary(executor, true)?;
        self.lab_oracle_attempt(executor, |model, _| {
            model.admission.begin_public_target_request_io_reuse()?;
            u::cancel(cancel)?;
            let actual = model.admission.request_io_reuse_active();
            let permit = model
                .native_batch
                .as_mut()
                .ok_or(rejected())?
                .public_language_request_io
                .as_mut()
                .ok_or(rejected())?;
            permit.begin(model.position, actual)
        })
    }
    pub(crate) fn public_language_request_io_active(&self) -> Result<bool> {
        if self.poisoned || self.retired.load(Ordering::Acquire) {
            return Err(rejected());
        }
        let actual = self.admission.request_io_reuse_active();
        let native = self.native_batch.as_ref().ok_or(rejected())?;
        match &native.public_language_request_io {
            Some(permit) => {
                permit.identity(self.graph.binding_id(), self.context)?;
                permit.active(actual)
            }
            None if !actual => Ok(false),
            None => Err(rejected()),
        }
    }
    pub(crate) fn end_public_language_request_io(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        u::cancel(cancel)?;
        self.public_language_request_boundary(executor, false)?;
        self.lab_oracle_attempt(executor, |model, _| {
            let started = std::time::Instant::now();
            // Original full fault/token join precedes complete upload/uniform/
            // readback owning scope erasure; EOS changes only allowed cursor.
            batch_io::read_fault(&model.arena, &model.admission, model.b.fault, cancel)?;
            model.admission.end_request_io_reuse()?;
            u::cancel(cancel)?;
            check_runtime(model.arena.client().map_err(u::fault)?)?;
            let closing_audit_ns = u64::try_from(started.elapsed().as_nanos())
                .map_err(|_| Error::ArithmeticOverflow)?;
            let actual = model.admission.request_io_reuse_active();
            let permit = model
                .native_batch
                .as_mut()
                .ok_or(rejected())?
                .public_language_request_io
                .as_mut()
                .ok_or(rejected())?;
            permit.end(model.position, actual)?;
            Ok(Receipt {
                scope: permit.scope(),
                full_io_audit: true,
                scope_closed: !actual,
                closing_audit_ns,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_language_permission_keeps_prefill_original_and_requires_active_real_cursor() {
        let geometry = Geometry::new(30, MAX_OUTPUTS).unwrap();
        let mut authority = Authority::new(geometry, geometry.context(), false).unwrap();
        authority.four_rows = true;
        assert!(authority.run(0, 4, false).is_err());
        authority.run(0, 3, false).unwrap();
        authority.begin(30, true).unwrap();
        authority.run(30, 4, true).unwrap();
        assert!(authority.run(30, 5, true).is_err());
        assert!(authority.run(29, 4, true).is_err());
        let mut original = Authority::new(geometry, geometry.context(), false).unwrap();
        original.begin(30, true).unwrap();
        assert!(original.run(30, 4, true).is_err());
    }
    #[test]
    fn exact_compiled_language_geometry_rejects_old_or_arbitrary_contexts() {
        for prompt in [30, 35] {
            let g = Geometry::new(prompt, 200).unwrap();
            assert_eq!(g.context(), prompt + 200);
            assert_eq!(g.final_cursor(), prompt + 199);
            for wrong in [0, 4, 13, 229, 234, 256] {
                assert!(Authority::new(g, wrong, false).is_err());
            }
            assert!(Authority::new(g, g.context(), true).is_err());
        }
        for prompt in [0, 1, 4, 13, 29, 31, 34, 36] {
            assert!(Geometry::new(prompt, 200).is_err());
        }
        for outputs in [0, 1, 199, 201, usize::MAX] {
            assert!(Geometry::new(35, outputs).is_err());
        }
    }
    #[test]
    fn original_prefill_then_single_scope_and_early_eos_are_actual() {
        for prompt in [30, 35] {
            for end in [prompt, prompt + 1, prompt + 198, prompt + 199] {
                let g = Geometry::new(prompt, 200).unwrap();
                let mut a = Authority::new(g, g.context(), false).unwrap();
                assert!(a.run(0, 3, false).is_ok());
                assert!(a.run(prompt - 1, 2, false).is_err());
                assert!(a.validate_begin(prompt - 1, false).is_err());
                assert!(a.begin(prompt, false).is_err());
                a.begin(prompt, true).unwrap();
                assert!(a.run(prompt, 3, true).is_ok());
                assert!(a.run(prompt - 1, 1, true).is_err());
                assert!(a.run(g.final_cursor(), 1, true).is_err());
                assert!(a.run(usize::MAX, 1, true).is_err());
                assert!(a.run(prompt, 0, true).is_err());
                assert!(a.run(prompt, 4, true).is_err());
                assert!(a.active(false).is_err());
                assert!(a.validate_end(prompt - 1, true).is_err());
                assert!(a.validate_end(prompt + 200, true).is_err());
                assert!(a.end(end, true).is_err());
                a.end(end, false).unwrap();
                assert!(!a.active(false).unwrap());
                assert!(a.run(prompt, 1, false).is_err());
                assert!(a.begin(prompt, true).is_err());
                assert!(a.end(end, false).is_err());
            }
        }
    }
    #[test]
    fn configuration_requires_reserved_rows_checkpoint_without_other_permits() {
        assert!(configuration(true, true, false, false, false, false, 8).is_ok());
        for values in [
            (false, true, false, false, false, false, 8),
            (true, false, false, false, false, false, 8),
            (true, true, true, false, false, false, 8),
            (true, true, false, true, false, false, 8),
            (true, true, false, false, true, false, 8),
            (true, true, false, false, false, true, 8),
            (true, true, false, false, false, false, 4),
        ] {
            assert!(
                configuration(
                    values.0, values.1, values.2, values.3, values.4, values.5, values.6
                )
                .is_err()
            );
        }
    }
}
