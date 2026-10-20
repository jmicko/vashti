//! Native request capacity and monotonic scope capability. The historical
//! target4/target13 permit is unchanged. It authenticates owner/lifecycle and
//! compiled fixture geometry; the language harness binds actual prompt bytes.
#![allow(dead_code)] // Parent owns constructor integration and physical acceptance.
use super::*;
use crate::expert_banks::OriginalPackedProfile;
use kuno_text::model::GraphBindingId;

pub(crate) const METADATA_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const BINDING_ENTRIES: usize = 4096;
pub(crate) const RUNTIME_BYTES: u64 = if cfg!(feature = "strata-intel-core") {
    512
} else {
    128
} * 1024
    * 1024;
pub(crate) const MAX_OUTPUTS: usize = if crate::strata_intel_core::enabled() {
    crate::strata_engine::MAX_OUTPUTS
} else if crate::strata_port::enabled() {
    512
} else {
    200
};

fn rejected() -> Error {
    Error::InvalidFormat("public language request I/O boundary rejected")
}

/// Charged request geometry. Historical public fixtures keep their exact
/// contract; the connected core accepts bounded request-local lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    prompt: usize,
    outputs: usize,
}
impl Geometry {
    pub(crate) fn new(prompt: usize, outputs: usize) -> Result<Self> {
        if crate::strata_intel_core::enabled() {
            if prompt == 0
                || !(1..=MAX_OUTPUTS).contains(&outputs)
                || crate::strata_engine::request_end(prompt, outputs).is_none()
            {
                return Err(rejected());
            }
        } else if !matches!(prompt, 30 | 35) || outputs != MAX_OUTPUTS {
            return Err(rejected());
        }
        Ok(Self { prompt, outputs })
    }
    pub(crate) fn prompt(self) -> usize {
        self.prompt
    }
    /// Retain capacity for the whole public output budget. The first output
    /// was predicted by prompt prefill; at most199 continuation inputs run.
    pub(crate) fn context(self) -> usize {
        self.prompt + self.outputs
    }
    pub(crate) fn final_cursor(self) -> usize {
        self.context() - 1
    }
    fn reserved_context(self) -> usize {
        self.context()
            + if crate::strata_intel_core::enabled() {
                crate::strata_engine::CONTEXT_HEADROOM
            } else {
                0
            }
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
    ShortPrefix,   // actual bounded known-input nonce, no generation scope credit
    InitialWindow, // real short runtime nonce; not long generation authority
    Active,
    Closed,
}
struct Authority {
    geometry: Geometry,
    capacity: usize,
    phase: Phase,
    scopes: u64,
    base_scopes: u64,
    four_rows: bool,
}
impl Authority {
    fn new(geometry: Geometry, context: usize, actual: bool) -> Result<Self> {
        Geometry::new(geometry.prompt, geometry.outputs)?;
        let valid = if crate::strata_intel_core::enabled() {
            geometry.reserved_context() <= context && context <= crate::qsa_gpu::MAX_CONTEXT
        } else {
            context == geometry.context()
        };
        if !valid || actual {
            return Err(rejected());
        }
        Ok(Self {
            four_rows: false,
            geometry,
            capacity: context,
            phase: Phase::Prefill,
            scopes: 0,
            base_scopes: 0,
        })
    }
    /// Native target commit consumes the complete matching INPUT prefix even
    /// when output delivery ends earlier. The last continuing origin precedes
    /// final_cursor, and its full window can commit at most MAXT-1 cells beyond
    /// it. This bound is not a completion receipt; original model/family owners
    /// still supply and validate the actual committed cursor on close/reset.
    fn maximum_cursor(&self) -> usize {
        self.geometry.final_cursor()
            + if crate::strata_intel_core::enabled() && self.four_rows {
                crate::strata_port::verify_capacity() - 1
            } else {
                0
            }
    }
    fn active(&self, actual: bool) -> Result<bool> {
        if actual
            != matches!(
                self.phase,
                Phase::ShortPrefix | Phase::InitialWindow | Phase::Active
            )
            || self.base_scopes.checked_add(u64::from(matches!(
                self.phase,
                Phase::Active | Phase::Closed
            ))) != Some(self.scopes)
        {
            return Err(rejected());
        }
        Ok(actual)
    }
    fn run(&self, start: usize, count: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if self.phase == Phase::InitialWindow {
            return if start.checked_add(1) == Some(self.geometry.prompt) && count == 1 {
                Ok(())
            } else {
                Err(rejected())
            };
        }
        if !(1..=if self.phase == Phase::ShortPrefix
            || self.four_rows && self.phase == Phase::Active
        {
            crate::strata_port::verify_capacity()
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
            Phase::ShortPrefix if end < self.geometry.prompt => Ok(()),
            Phase::Active
                if start >= self.geometry.prompt
                    && start < self.geometry.final_cursor()
                    && end <= self.maximum_cursor()
                    && end <= self.capacity =>
            {
                Ok(())
            }
            _ => Err(rejected()),
        }
    }
    fn prompt_chunk(&self, start: usize, count: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if !crate::strata_intel_core::enabled()
            || self.phase != Phase::Prefill
            || actual
            || !(1..=crate::strata_prompt::groups::CAPACITY).contains(&count)
            || start
                .checked_add(count)
                .is_none_or(|end| end > self.geometry.prompt)
        {
            return Err(rejected());
        }
        Ok(())
    }
    fn validate_completed_native(&self, position: usize, actual: bool) -> Result<()> {
        self.retained_native_scope()?;
        if actual || !(self.geometry.prompt..=self.maximum_cursor()).contains(&position) {
            return Err(rejected());
        }
        self.active(false)?;
        Ok(())
    }
    fn retained_native_scope(&self) -> Result<u64> {
        if !crate::strata_intel_core::enabled()
            || self.phase != Phase::Closed
            || self.base_scopes.checked_add(1) != Some(self.scopes)
        {
            return Err(rejected());
        }
        Ok(self.scopes)
    }
    fn rearm_native(
        &mut self,
        geometry: Geometry,
        capacity: usize,
        erased_scope: u64,
        actual: bool,
    ) -> Result<()> {
        Geometry::new(geometry.prompt, geometry.outputs)?;
        if self.retained_native_scope()? != erased_scope
            || actual
            || capacity != self.capacity
            || geometry.reserved_context() > capacity
            || self.scopes.checked_add(1).is_none()
        {
            return Err(rejected());
        }
        self.active(false)?;
        self.geometry.prompt.zeroize();
        self.geometry.outputs.zeroize();
        self.geometry = geometry;
        self.base_scopes = self.scopes;
        self.phase = Phase::Prefill;
        Ok(())
    }
    fn starting_after_erasure(&self, scope: u64) -> Result<()> {
        self.active(false)?;
        if !crate::strata_intel_core::enabled()
            || scope == 0
            || self.phase != Phase::Prefill
            || self.base_scopes != scope
            || self.scopes != scope
        {
            return Err(rejected());
        }
        Ok(())
    }
    fn current_scope(&self, actual: bool) -> Result<u64> {
        if self.phase != Phase::Active || !self.active(actual)? {
            return Err(rejected());
        }
        Ok(self.scopes)
    }
    fn validate_short_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        let prefix = self.geometry.prompt.checked_sub(1).ok_or(rejected())?;
        if !crate::strata_intel_core::enabled()
            || self.phase != Phase::Prefill
            || position != 0
            || actual
            || !(1..=crate::strata_engine::controller::SHORT_PROMPT_PREFIX_LIMIT).contains(&prefix)
        {
            return Err(rejected());
        }
        Ok(())
    }
    fn begin_short(&mut self, position: usize, actual: bool) -> Result<()> {
        self.validate_short_begin(position, false)?;
        if !actual {
            return Err(rejected());
        }
        self.phase = Phase::ShortPrefix;
        self.active(actual).map(|_| ())
    }
    fn validate_short_run(&self, start: usize, count: usize, actual: bool) -> Result<()> {
        if self.phase != Phase::ShortPrefix {
            return Err(rejected());
        }
        self.run(start, count, actual)
    }
    fn validate_short_cursor(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if self.phase != Phase::ShortPrefix || position >= self.geometry.prompt {
            return Err(rejected());
        }
        Ok(())
    }
    fn starting_short_after_erasure(&self, scope: u64, actual: bool) -> Result<()> {
        self.validate_short_cursor(0, actual)?;
        if scope == 0 || self.base_scopes != scope || self.scopes != scope {
            return Err(rejected());
        }
        Ok(())
    }
    fn validate_short_end(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if self.phase != Phase::ShortPrefix || position.checked_add(1) != Some(self.geometry.prompt)
        {
            return Err(rejected());
        }
        Ok(())
    }
    fn end_short(&mut self, position: usize, actual: bool) -> Result<()> {
        self.validate_short_end(position, true)?;
        if actual {
            return Err(rejected());
        }
        self.phase = Phase::Prefill;
        self.active(false).map(|_| ())
    }
    fn validate_initial_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if !crate::strata_intel_core::enabled()
            || self.phase != Phase::Prefill
            || actual
            || position.checked_add(1) != Some(self.geometry.prompt)
        {
            return Err(rejected());
        }
        Ok(())
    }
    fn begin_initial(&mut self, position: usize, actual: bool) -> Result<()> {
        self.validate_initial_begin(position, false)?;
        if !actual {
            return Err(rejected());
        }
        self.phase = Phase::InitialWindow;
        self.active(actual).map(|_| ())
    }
    fn validate_initial_end(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if self.phase != Phase::InitialWindow || position != self.geometry.prompt {
            return Err(rejected());
        }
        Ok(())
    }
    fn end_initial(&mut self, position: usize, actual: bool) -> Result<()> {
        self.validate_initial_end(position, true)?;
        if actual {
            return Err(rejected());
        }
        self.phase = Phase::Prefill;
        self.active(false).map(|_| ())
    }
    fn validate_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.active(actual)?;
        if self.phase != Phase::Prefill
            || position != self.geometry.prompt
            || actual
            || self.scopes.checked_add(1).is_none()
        {
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
        self.scopes = self.scopes.checked_add(1).ok_or(rejected())?;
        Ok(())
    }
    fn validate_end(&self, position: usize, actual: bool) -> Result<()> {
        if !self.active(actual)?
            || self.phase != Phase::Active
            || !(self.geometry.prompt..=self.maximum_cursor()).contains(&position)
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
        self.geometry.outputs.zeroize();
        self.scopes.zeroize();
        self.base_scopes.zeroize();
        self.capacity.zeroize();
        self.phase = Phase::Closed;
    }
}

pub(super) struct Permit {
    binding: Option<GraphBindingId>,
    authority: Authority,
}

/// A single genuine known-input window, not generation/four-row authority.
/// Only the original live ShortPrefix permit can mint it. The executor couples
/// it to the actual admission's still-live request nonce for its lexical scope.
pub(crate) struct ProjectionWindow {
    binding: GraphBindingId,
    capacity: usize,
    prompt: usize,
    start: usize,
    count: usize,
}
fn validate_projection_geometry(
    capacity: usize,
    prompt: usize,
    start: usize,
    count: usize,
) -> Result<()> {
    if !crate::strata_intel_core::enabled()
        || !(1..=crate::strata_port::verify_capacity()).contains(&count)
        || prompt > capacity
        || prompt.checked_sub(1).is_none_or(|prefix| {
            !(1..=crate::strata_engine::controller::SHORT_PROMPT_PREFIX_LIMIT).contains(&prefix)
                || start.checked_add(count).is_none_or(|end| end > prefix)
        })
    {
        return Err(rejected());
    }
    Ok(())
}
impl ProjectionWindow {
    pub(crate) fn count(&self) -> usize {
        self.count
    }
    pub(crate) fn validate(&self, binding: GraphBindingId, count: usize) -> Result<()> {
        if self.binding != binding || self.count != count {
            return Err(rejected());
        }
        validate_projection_geometry(self.capacity, self.prompt, self.start, count)
    }
}
impl Drop for ProjectionWindow {
    fn drop(&mut self) {
        self.capacity.zeroize();
        self.prompt.zeroize();
        self.start.zeroize();
        self.count.zeroize();
    }
}
impl Permit {
    pub(super) fn projection_window(
        &self,
        binding: GraphBindingId,
        capacity: usize,
        start: usize,
        count: usize,
        actual: bool,
    ) -> Result<ProjectionWindow> {
        self.identity(binding, capacity)?;
        self.validate_short_run(start, count, actual)?;
        let window = ProjectionWindow {
            binding,
            capacity,
            prompt: self.authority.geometry.prompt,
            start,
            count,
        };
        window.validate(binding, count)?;
        Ok(window)
    }
    pub(super) fn validate_completed_native_request(
        &self,
        binding: GraphBindingId,
        context: usize,
        position: usize,
        actual: bool,
    ) -> Result<()> {
        self.identity(binding, context)?;
        self.authority.validate_completed_native(position, actual)
    }
    pub(super) fn retained_native_scope(&self) -> Result<u64> {
        self.authority.retained_native_scope()
    }
    pub(super) fn rearm_native(
        &mut self,
        binding: GraphBindingId,
        capacity: usize,
        geometry: Geometry,
        scope: u64,
        actual: bool,
    ) -> Result<()> {
        self.identity(binding, capacity)?;
        self.authority
            .rearm_native(geometry, capacity, scope, actual)
    }
    pub(super) fn starting_after_erasure(&self, scope: u64) -> Result<()> {
        self.authority.starting_after_erasure(scope)
    }
    pub(super) fn current_scope(&self, actual: bool) -> Result<u64> {
        self.authority.current_scope(actual)
    }
    pub(super) fn validate_geometry(&self, prompt: usize, outputs: usize) -> Result<()> {
        if self.authority.phase != Phase::Prefill
            || self.authority.geometry != Geometry::new(prompt, outputs)?
        {
            return Err(rejected());
        }
        Ok(())
    }
    pub(super) fn enable_four_rows(&mut self) -> Result<()> {
        if matches!(
            self.authority.phase,
            Phase::ShortPrefix | Phase::InitialWindow | Phase::Active
        ) || self.authority.four_rows
        {
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
        if self.binding != Some(binding) || context != self.authority.capacity {
            return Err(rejected());
        }
        Ok(())
    }
    pub(super) fn validate_prompt_chunk(
        &self,
        binding: GraphBindingId,
        context: usize,
        start: usize,
        count: usize,
        actual: bool,
    ) -> Result<()> {
        self.identity(binding, context)?;
        self.authority.prompt_chunk(start, count, actual)
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
    pub(super) fn validate_short_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.authority.validate_short_begin(position, actual)
    }
    pub(super) fn begin_short(&mut self, position: usize, actual: bool) -> Result<()> {
        self.authority.begin_short(position, actual)
    }
    pub(super) fn validate_short_run(
        &self,
        position: usize,
        count: usize,
        actual: bool,
    ) -> Result<()> {
        self.authority.validate_short_run(position, count, actual)
    }
    pub(super) fn validate_short_cursor(&self, position: usize, actual: bool) -> Result<()> {
        self.authority.validate_short_cursor(position, actual)
    }
    pub(super) fn starting_short_after_erasure(&self, scope: u64, actual: bool) -> Result<()> {
        self.authority.starting_short_after_erasure(scope, actual)
    }
    pub(super) fn validate_short_end(&self, position: usize, actual: bool) -> Result<()> {
        self.authority.validate_short_end(position, actual)
    }
    pub(super) fn end_short(&mut self, position: usize, actual: bool) -> Result<()> {
        self.authority.end_short(position, actual)
    }
    pub(super) fn validate_initial_begin(&self, position: usize, actual: bool) -> Result<()> {
        self.authority.validate_initial_begin(position, actual)
    }
    pub(super) fn begin_initial(&mut self, position: usize, actual: bool) -> Result<()> {
        self.authority.begin_initial(position, actual)
    }
    pub(super) fn validate_initial_end(&self, position: usize, actual: bool) -> Result<()> {
        self.authority.validate_initial_end(position, actual)
    }
    pub(super) fn end_initial(&mut self, position: usize, actual: bool) -> Result<()> {
        self.authority.end_initial(position, actual)
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
    pub(crate) fn current_native_language_scope(&self) -> Result<u64> {
        if self.poisoned || self.retired.load(Ordering::Acquire) {
            return Err(rejected());
        }
        let permit = self
            .native_batch
            .as_ref()
            .ok_or(rejected())?
            .public_language_request_io
            .as_ref()
            .ok_or(rejected())?;
        permit.identity(self.graph.binding_id(), self.context)?;
        permit.current_scope(self.admission.request_io_reuse_active())
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
    fn actual_native_scope_allows_full_target_commit_beyond_delivery_cap_with_reserved_headroom() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for outputs in 2..=9 {
            let g = Geometry::new(17, outputs).unwrap();
            for remaining in 1..outputs {
                let start = g.final_cursor() - remaining;
                for width in 1..=crate::strata_port::verify_capacity() {
                    for keep in 1..=width {
                        let mut a = Authority::new(g, g.reserved_context(), false).unwrap();
                        a.four_rows = true;
                        a.begin(g.prompt(), true).unwrap();
                        // This models the same full-window controller/commit
                        // boundary, not a clipped delivery-sized verification.
                        a.run(start, width, true).unwrap();
                        let committed = start + keep;
                        a.validate_end(committed, true).unwrap();
                        assert!(a.run(g.final_cursor(), 1, true).is_err());
                        assert!(a.run(g.final_cursor() + 1, 1, true).is_err());
                        assert!(a.validate_end(a.maximum_cursor() + 1, true).is_err());
                        a.end(committed, false).unwrap();
                        a.validate_completed_native(committed, false).unwrap();
                    }
                }
            }
            for missing in 1..=crate::strata_engine::CONTEXT_HEADROOM {
                assert!(Authority::new(g, g.reserved_context() - missing, false).is_err());
            }
        }
        let context = crate::qsa_gpu::MAX_CONTEXT;
        for missing in 0..crate::strata_engine::CONTEXT_HEADROOM {
            assert!(Geometry::new(17, context - 17 - missing).is_err());
        }
    }
    #[test]
    fn retained_native_requests_share_capacity_with_monotonic_single_use_scopes() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let capacity = crate::qsa_gpu::MAX_CONTEXT;
        let mut a = Authority::new(Geometry::new(1, 1).unwrap(), capacity, false).unwrap();
        for (index, (prompt, outputs)) in [
            (1, 1),
            (13, 16),
            (4096, 8192),
            (MAX_OUTPUTS, 1),
            (1, MAX_OUTPUTS),
        ]
        .into_iter()
        .enumerate()
        {
            let g = Geometry::new(prompt, outputs).unwrap();
            if index != 0 {
                assert!(
                    a.rearm_native(g, capacity, index as u64 - 1, false)
                        .is_err()
                );
                assert!(a.rearm_native(g, capacity, index as u64, true).is_err());
                a.rearm_native(g, capacity, index as u64, false).unwrap();
                a.starting_after_erasure(index as u64).unwrap();
                assert!(a.rearm_native(g, capacity, index as u64, false).is_err());
            }
            assert_eq!(a.scopes, index as u64);
            a.begin(prompt, true).unwrap();
            assert_eq!(a.scopes, index as u64 + 1);
            assert!(a.rearm_native(g, capacity, a.scopes, false).is_err());
            a.end(prompt, false).unwrap();
            a.validate_completed_native(prompt, false).unwrap();
        }
        a.scopes = u64::MAX;
        a.base_scopes = u64::MAX - 1;
        assert!(
            a.rearm_native(Geometry::new(1, 1).unwrap(), capacity, u64::MAX, false)
                .is_err()
        );
    }
    #[test]
    fn retained_native_reset_requires_the_closed_original_scope_and_actual_cursor() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for prompt in [1, 13, 4096] {
            let g = Geometry::new(prompt, 16).unwrap();
            let mut a = Authority::new(g, g.reserved_context(), false).unwrap();
            assert!(a.validate_completed_native(prompt, false).is_err());
            a.begin(prompt, true).unwrap();
            assert!(a.validate_completed_native(prompt, true).is_err());
            a.end(prompt, false).unwrap();
            assert_eq!(a.retained_native_scope().unwrap(), 1);
            assert!(a.validate_completed_native(prompt, false).is_ok());
            assert!(a.validate_completed_native(g.final_cursor(), false).is_ok());
            assert!(a.validate_completed_native(prompt - 1, false).is_err());
            assert!(a.validate_completed_native(g.context(), false).is_err());
            assert!(a.validate_completed_native(prompt, true).is_err());
            a.scopes = 2;
            assert!(a.validate_completed_native(prompt, false).is_err());
        }
    }
    #[test]
    fn core_geometry_charges_arbitrary_bounded_request_and_checks_lifecycle() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for (prompt, outputs) in [
            (1, 1),
            (23, 7),
            (
                crate::qsa_gpu::MAX_CONTEXT - crate::strata_engine::CONTEXT_HEADROOM - 1,
                1,
            ),
            (1024, 512),
        ] {
            let geometry = Geometry::new(prompt, outputs).unwrap();
            assert_eq!(geometry.context(), prompt + outputs);
            let mut authority =
                Authority::new(geometry, geometry.reserved_context(), false).unwrap();
            let prefix = prompt - 1;
            for start in (0..prefix).step_by(crate::strata_prompt::groups::CAPACITY) {
                authority
                    .prompt_chunk(
                        start,
                        (prefix - start).min(crate::strata_prompt::groups::CAPACITY),
                        false,
                    )
                    .unwrap();
            }
            // The normal decoder consumes the final prompt input while the
            // request is still in Prefill. Only then may output reuse begin.
            authority.run(prefix, 1, false).unwrap();
            assert!(authority.prompt_chunk(prompt, 1, false).is_err());
            assert!(
                authority
                    .prompt_chunk(0, crate::strata_prompt::groups::CAPACITY + 1, false)
                    .is_err()
            );
            assert!(authority.prompt_chunk(0, 1, true).is_err());
            assert!(authority.validate_begin(prompt - 1, false).is_err());
            authority.begin(prompt, true).unwrap();
            assert!(authority.prompt_chunk(0, 1, true).is_err());
            if outputs > 1 {
                authority.run(prompt, 1, true).unwrap();
            }
            assert!(authority.run(prompt + outputs - 1, 1, true).is_err());
            authority.end(prompt + outputs - 1, false).unwrap();
            assert!(authority.run(prompt, 1, false).is_err());
        }
        for (p, o) in [
            (0, 1),
            (1, 0),
            (1, MAX_OUTPUTS + 1),
            (crate::qsa_gpu::MAX_CONTEXT, 1),
            (usize::MAX, 1),
        ] {
            assert!(Geometry::new(p, o).is_err());
        }
    }
    #[test]
    fn four_language_permission_keeps_prefill_original_and_requires_active_real_cursor() {
        let geometry = Geometry::new(30, MAX_OUTPUTS.min(512)).unwrap();
        let mut authority = Authority::new(geometry, geometry.reserved_context(), false).unwrap();
        authority.four_rows = true;
        assert!(authority.run(0, 4, false).is_err());
        authority.run(0, 3, false).unwrap();
        authority.begin(30, true).unwrap();
        authority.run(30, 4, true).unwrap();
        for count in 4..=crate::strata_port::verify_capacity() {
            authority.run(30, count, true).unwrap();
        }
        assert!(
            authority
                .run(30, crate::strata_port::verify_capacity() + 1, true)
                .is_err()
        );
        assert!(authority.run(29, 4, true).is_err());
        let mut original = Authority::new(geometry, geometry.reserved_context(), false).unwrap();
        original.begin(30, true).unwrap();
        assert!(original.run(30, 4, true).is_err());
    }
    #[test]
    fn compiled_and_native_language_geometry_keep_distinct_bounded_contracts() {
        for prompt in [30, 35] {
            let g = Geometry::new(prompt, MAX_OUTPUTS.min(512)).unwrap();
            assert_eq!(g.context(), prompt + MAX_OUTPUTS.min(512));
            assert_eq!(g.final_cursor(), prompt + MAX_OUTPUTS.min(512) - 1);
            for wrong in [0, 4, 13, 229, 234, 256] {
                assert!(Authority::new(g, wrong, false).is_err());
            }
            assert!(Authority::new(g, g.reserved_context(), true).is_err());
        }
        for prompt in [0, 1, 4, 13, 29, 31, 34, 36] {
            assert_eq!(
                Geometry::new(prompt, MAX_OUTPUTS.min(512)).is_ok(),
                crate::strata_intel_core::enabled() && prompt > 0
            );
        }
        for outputs in [0, 1, 199, 201, MAX_OUTPUTS, MAX_OUTPUTS + 1, usize::MAX] {
            let allowed = if crate::strata_intel_core::enabled() {
                (1..=MAX_OUTPUTS).contains(&outputs)
                    && outputs
                        .checked_add(35)
                        .and_then(|n| n.checked_add(crate::strata_engine::CONTEXT_HEADROOM))
                        .is_some_and(|n| n <= crate::qsa_gpu::MAX_CONTEXT)
            } else {
                outputs == MAX_OUTPUTS
            };
            assert_eq!(Geometry::new(35, outputs).is_ok(), allowed);
        }
    }

    #[test]
    fn original_prefill_then_single_scope_and_early_eos_are_actual() {
        for prompt in [30, 35] {
            for end in [prompt, prompt + 1, prompt + 198, prompt + 199] {
                let g = Geometry::new(prompt, 200).unwrap();
                let mut a = Authority::new(g, g.reserved_context(), false).unwrap();
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

#[cfg(test)]
mod initial_window_tests {
    use super::*;
    #[test]
    fn initial_window_has_real_runtime_scope_but_no_generated_cursor_authority() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        for prompt in [1, 35, 4096, MAX_OUTPUTS] {
            let geometry = Geometry::new(prompt, 1).unwrap();
            let mut a = Authority::new(geometry, geometry.reserved_context(), false).unwrap();
            let prefix = prompt - 1;
            assert!(a.begin_initial(prefix, false).is_err());
            assert!(a.validate_initial_begin(prompt, false).is_err());
            assert!(a.validate_initial_begin(prefix, true).is_err());
            a.validate_initial_begin(prefix, false).unwrap();
            a.begin_initial(prefix, true).unwrap();
            a.run(prefix, 1, true).unwrap();
            assert!(a.current_scope(true).is_err());
            assert!(a.validate_end(prompt, true).is_err());
            assert!(a.run(prefix, 1, false).is_err());
            assert!(a.run(prefix, 2, true).is_err());
            assert!(a.run(prompt, 1, true).is_err());
            assert!(a.prompt_chunk(0, 1, true).is_err());
            assert!(a.validate_begin(prompt, false).is_err());
            assert!(a.begin_initial(prefix, true).is_err());
            assert!(a.validate_initial_end(prefix, true).is_err());
            assert!(a.end_initial(prompt, true).is_err());
            assert!(a.end_initial(prefix, false).is_err());
            a.validate_initial_end(prompt, true).unwrap();
            a.end_initial(prompt, false).unwrap();
            assert_eq!(a.phase, Phase::Prefill);
            assert_eq!(a.scopes, 0);
            a.validate_begin(prompt, false).unwrap();
            a.begin(prompt, true).unwrap();
            assert_eq!(a.scopes, 1);
            assert_eq!(a.current_scope(true).unwrap(), 1);
            a.end(prompt, false).unwrap();
            a.validate_completed_native(prompt, false).unwrap();
        }
    }
    #[test]
    fn initial_nonce_cannot_rearm_close_or_credit_a_different_request() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let capacity = crate::qsa_gpu::MAX_CONTEXT;
        let mut a = Authority::new(Geometry::new(1, 1).unwrap(), capacity, false).unwrap();
        for (index, prompt) in [1, 23, 2].into_iter().enumerate() {
            if index > 0 {
                a.rearm_native(
                    Geometry::new(prompt, 1).unwrap(),
                    capacity,
                    index as u64,
                    false,
                )
                .unwrap();
            }
            assert!(a.end_initial(prompt, false).is_err());
            a.begin_initial(prompt - 1, true).unwrap();
            assert!(a.retained_native_scope().is_err());
            assert!(a.validate_completed_native(prompt, false).is_err());
            assert!(a.end(prompt, false).is_err());
            assert!(
                a.rearm_native(Geometry::new(1, 1).unwrap(), capacity, index as u64, false)
                    .is_err()
            );
            a.end_initial(prompt, false).unwrap();
            assert_eq!(a.scopes, index as u64);
            a.begin(prompt, true).unwrap();
            a.end(prompt, false).unwrap();
            assert_eq!(a.retained_native_scope().unwrap(), index as u64 + 1);
        }
    }
}

#[cfg(test)]
mod short_prefix_scope_tests {
    use super::*;
    #[test]
    fn projection_scope_requires_genuine_short_phase_and_exact_reserved_window() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let capacity = crate::qsa_gpu::MAX_CONTEXT;
        let mut a = Authority::new(Geometry::new(65, 2).unwrap(), capacity, false).unwrap();
        for count in 1..=crate::strata_port::verify_capacity() {
            assert!(a.validate_short_run(0, count, false).is_err());
        }
        a.begin_short(0, true).unwrap();
        for count in 1..=crate::strata_port::verify_capacity() {
            a.validate_short_run(0, count, true).unwrap();
            validate_projection_geometry(capacity, 65, 0, count).unwrap();
            validate_projection_geometry(capacity, 65, 64 - count, count).unwrap();
            assert!(a.validate_short_run(0, count, false).is_err());
            assert!(validate_projection_geometry(capacity, 65, 65 - count, count).is_err());
        }
        for (prompt, start, count) in [
            (0, 0, 1),
            (1, 0, 1),
            (66, 0, 1),
            (65, 0, 0),
            (65, 0, 9),
            (65, usize::MAX, 8),
        ] {
            assert!(validate_projection_geometry(capacity, prompt, start, count).is_err());
        }
        assert!(validate_projection_geometry(64, 65, 0, 8).is_err());
        a.end_short(64, false).unwrap();
        a.begin_initial(64, true).unwrap();
        assert!(a.validate_short_run(64, 1, true).is_err());
        a.end_initial(65, false).unwrap();
        a.begin(65, true).unwrap();
        assert!(a.validate_short_run(65, 8, true).is_err());
        assert_eq!(a.scopes, 1);
    }
    #[test]
    fn short_prefix_uses_real_nonce_max_eight_and_no_generation_credit() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let capacity = crate::qsa_gpu::MAX_CONTEXT;
        for prefix in [1usize, 3, 4, 7, 8, 9, 63, 64] {
            let prompt = prefix + 1;
            let geometry = Geometry::new(prompt, 2).unwrap();
            let mut authority = Authority::new(geometry, capacity, false).unwrap();
            // Cold short reads use their own checked MAX8 reservation even
            // before generation enables its separate four/eight-row policy.
            assert!(!authority.four_rows);
            assert!(authority.validate_short_begin(1, false).is_err());
            assert!(authority.validate_short_begin(0, true).is_err());
            assert!(authority.begin_short(0, false).is_err());
            authority.validate_short_begin(0, false).unwrap();
            authority.begin_short(0, true).unwrap();
            for position in [0, prefix] {
                authority.validate_short_cursor(position, true).unwrap();
            }
            assert!(authority.validate_short_cursor(0, false).is_err());
            assert!(authority.validate_short_cursor(prompt, true).is_err());
            assert!(authority.starting_short_after_erasure(0, true).is_err());
            assert_eq!(authority.scopes, 0);
            assert!(authority.current_scope(true).is_err());
            assert!(authority.validate_short_run(0, 1, false).is_err());
            for start in (0..prefix).step_by(crate::strata_port::verify_capacity()) {
                let count = (prefix - start).min(crate::strata_port::verify_capacity());
                authority.validate_short_run(start, count, true).unwrap();
            }
            for (start, count) in [(0, 0), (0, 9), (prefix, 1), (usize::MAX, 3)] {
                assert!(authority.validate_short_run(start, count, true).is_err());
            }
            assert!(authority.prompt_chunk(0, 1, true).is_err());
            assert!(authority.begin_short(0, true).is_err());
            assert!(authority.begin_initial(prefix, true).is_err());
            assert!(authority.begin(prompt, true).is_err());
            assert!(authority.end(prompt, false).is_err());
            assert!(authority.retained_native_scope().is_err());
            assert!(authority.validate_short_end(prefix - 1, true).is_err());
            assert!(authority.validate_short_end(prompt, true).is_err());
            assert!(authority.end_short(prefix, true).is_err());
            authority.validate_short_end(prefix, true).unwrap();
            authority.end_short(prefix, false).unwrap();
            assert_eq!(authority.phase, Phase::Prefill);
            assert_eq!(authority.scopes, 0);
            assert!(authority.validate_short_run(0, 1, false).is_err());
            // The original final count1 scope still has to execute and close.
            authority.begin_initial(prefix, true).unwrap();
            authority.run(prefix, 1, true).unwrap();
            authority.end_initial(prompt, false).unwrap();
            authority.begin(prompt, true).unwrap();
            authority.end(prompt, false).unwrap();
            assert_eq!(authority.retained_native_scope().unwrap(), 1);
        }
    }
    #[test]
    fn short_prefix_rejects_empty_large_and_rearm_before_actual_generation_close() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let capacity = crate::qsa_gpu::MAX_CONTEXT;
        for prompt in [1, 66, 4096] {
            let a = Authority::new(Geometry::new(prompt, 1).unwrap(), capacity, false).unwrap();
            assert!(a.validate_short_begin(0, false).is_err());
        }
        let mut a = Authority::new(Geometry::new(2, 1).unwrap(), capacity, false).unwrap();
        for (index, prefix) in [1usize, 64, 3].into_iter().enumerate() {
            let prompt = prefix + 1;
            if index > 0 {
                a.rearm_native(
                    Geometry::new(prompt, 1).unwrap(),
                    capacity,
                    index as u64,
                    false,
                )
                .unwrap();
            }
            a.begin_short(0, true).unwrap();
            if index > 0 {
                a.starting_short_after_erasure(index as u64, true).unwrap();
            }
            assert!(a.starting_short_after_erasure(index as u64, false).is_err());
            assert!(
                a.starting_short_after_erasure(index as u64 + 1, true)
                    .is_err()
            );
            assert!(
                a.rearm_native(Geometry::new(2, 1).unwrap(), capacity, index as u64, false)
                    .is_err()
            );
            assert!(a.validate_completed_native(prompt, false).is_err());
            assert!(a.end_short(prefix - 1, false).is_err());
            a.end_short(prefix, false).unwrap();
            assert_eq!(a.scopes, index as u64);
            a.begin_initial(prefix, true).unwrap();
            a.end_initial(prompt, false).unwrap();
            a.begin(prompt, true).unwrap();
            a.end(prompt, false).unwrap();
            assert_eq!(a.retained_native_scope().unwrap(), index as u64 + 1);
        }
    }
}
