//! Connected global-source controller, derived from MIT Strata's Intel
//! generate.cpp speculative loop at fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
//! No legacy bank receipts, fabricated dispatches, operator profiles or speed
//! reports. Prompt/carries/drafts/history are bounded, transient and wiped.
use super::{MAX_OUTPUTS, REQUEST_HOST_BYTES, VOCAB};
use crate::{executor::GpuExecutor, resident_model::ResidentModel, resident_mtp};
use std::{sync::atomic::AtomicBool, time::Instant};
use zeroize::Zeroizing;
#[path = "strata_short_prompt_plan.rs"]
mod short_prompt;
// One actual MAX3 host R copy coexists with the existing compact output packet.
// Both fit in the existing charged request HOST envelope; no owner is added.
const SHORT_CARRY_HOST_BYTES: usize =
    short_prompt::WINDOW_ROWS * crate::resident_model::batch::target_rows::HC_WIDTH * 4;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn elapsed(start: Instant) -> Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).map_err(Into::into)
}
use kuno_text::Error;

pub(super) struct Request<'a> {
    pub prompt: &'a [u32],
    pub max_outputs: usize,
    pub eos: u32,
    pub speculation: crate::strata_speculation::Config,
    pub draft_min_probability: f32,
    pub sampling: crate::strata_sampler::Params,
}
impl Request<'_> {
    pub(super) fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        self.speculation.validate()?;
        if crate::strata_speculation::HOST_BYTES
            + MAX_OUTPUTS * 4
            + 64 * 1024
            + SHORT_CARRY_HOST_BYTES
            > REQUEST_HOST_BYTES
        {
            return Err("global speculative host quote rejected".into());
        }
        if !self.draft_min_probability.is_finite()
            || !(0. ..=1.).contains(&self.draft_min_probability)
        {
            return Err("global draft confidence bound".into());
        }
        if self.prompt.is_empty()
            || self.prompt.iter().any(|&t| t >= VOCAB as u32)
            || !(1..=MAX_OUTPUTS).contains(&self.max_outputs)
            || self.eos >= VOCAB as u32
            || self.speculation.window > crate::strata_port::verify_capacity()
            || super::request_end(self.prompt.len(), self.max_outputs).is_none()
        {
            return Err("global request geometry rejected".into());
        }
        Ok(())
    }
}

/// Native generate.cpp batches exactly n_prompt-1 inputs. The final input is
/// decoded with normal arithmetic; only it predicts the first generated token.
/// Request chunk rounds the prefix up to256 and caps at the actual owned stage.
struct PromptPlan(Zeroizing<[usize; 2]>);
impl PromptPlan {
    fn new(prompt: usize) -> Result<Self> {
        if !(1..=crate::qsa_gpu::MAX_CONTEXT).contains(&prompt) {
            return Err("global prompt plan geometry".into());
        }
        let prefix = prompt - 1;
        let chunk = if prefix == 0 {
            0
        } else {
            prefix
                .checked_add(255)
                .ok_or("global prompt chunk overflow")?
                / 256
                * 256
        }
        .min(crate::strata_prompt::groups::CAPACITY);
        Ok(Self(Zeroizing::new([prompt, chunk])))
    }
    fn prefix(&self) -> usize {
        self.0[0] - 1
    }
    fn chunk(&self) -> usize {
        self.0[1]
    }
    fn chunks(&self) -> impl Iterator<Item = std::ops::Range<usize>> {
        let prefix = self.prefix();
        let chunk = self.chunk();
        (0..prefix)
            .step_by(chunk.max(1))
            .map(move |first| first..(first + chunk).min(prefix))
    }
    fn inputs<'a>(
        &self,
        prompt: &'a [u32],
        chunk: std::ops::Range<usize>,
    ) -> kuno_text::Result<(&'a [u32], &'a [u32])> {
        if prompt.len() != self.0[0]
            || chunk.is_empty()
            || self.chunk() == 0
            || !chunk.start.is_multiple_of(self.chunk())
            || chunk.end > self.prefix()
            || chunk.end != (chunk.start + self.chunk()).min(self.prefix())
        {
            return Err(kuno_text::Error::InvalidFormat(
                "native prompt teacher chunk",
            ));
        }
        // Every batched R_i teaches its real next input token, including the
        // final unbatched prompt token. No result/last-output token is invented.
        Ok((
            &prompt[chunk.clone()],
            &prompt[chunk.start + 1..chunk.end + 1],
        ))
    }
}

/// Accepted *input* prefix includes the confirmed predecessor. Emits exactly
/// that many target state inputs, even when output delivery stops earlier at
/// EOS or the cap. Drafts never overrule target predictions. Native commit is
/// independent of how many predictions the request chooses to emit.
fn accepted(inputs: &[u32], predicted: &[u32]) -> Result<usize> {
    if inputs.is_empty()
        || inputs.len() != predicted.len()
        || inputs.len() > crate::strata_port::verify_capacity()
        || inputs.iter().chain(predicted).any(|&t| t >= VOCAB as u32)
    {
        return Err("global target prefix rejected".into());
    }
    let mut n = 1;
    while n < inputs.len() && inputs[n] == predicted[n - 1] {
        n += 1;
    }
    Ok(n)
}

fn emitted(predicted: &[u32], remaining: usize, eos: u32) -> usize {
    let limit = remaining.min(predicted.len());
    predicted[..limit]
        .iter()
        .position(|&t| t == eos)
        .map_or(limit, |i| i + 1)
}

/// Native costs cover this verification/commit plus the NEXT draft. This
/// observation never escapes the request; no incomplete engine is ranked.
struct PolicyObservation {
    started: Instant,
    window: usize,
    accepted: usize,
    lookup: Option<usize>,
    chain: Option<(usize, usize, usize)>, // MTP window, chain count, match
}
impl PolicyObservation {
    fn finish(self, policy: &mut crate::strata_speculation::Policy) -> Result<()> {
        let ns = elapsed(self.started)?;
        if let Some((mtp, count, matched)) = self.chain {
            policy.observe_chain(mtp, count, self.accepted, matched, ns)?;
        } else {
            policy.observe(
                self.lookup.is_some(),
                self.window,
                self.accepted,
                self.lookup.unwrap_or(0),
                ns,
            )?;
        }
        Ok(())
    }
}

/// Validate the first final-prompt handoff against the original loaded MTP
/// cursor/binding. Later windows use the same scoped original source contract.
fn validate_round_entry(
    coordinates: &[usize; 2],
    original_binding: u64,
    predicted: &[u32],
    position: usize,
    confirmed: u32,
    draft_position: usize,
    binding: u64,
) -> Result<()> {
    if !(1..=crate::strata_port::verify_capacity()).contains(&predicted.len())
        || coordinates[0].checked_add(predicted.len()) != Some(position)
        || predicted.last() != Some(&confirmed)
        || coordinates[1] != draft_position
        || original_binding != binding
    {
        return Err("global accepted-round entry rejected".into());
    }
    Ok(())
}

/// A completed original MTP producer retained for the NEXT target window.
/// This owns small wiping token/control metadata, never a target-ready receipt.
struct PreparedDraft {
    coordinates: Zeroizing<[usize; 2]>, // next target origin, actual completed MTP cursor
    binding: Zeroizing<u64>,
    inputs: Zeroizing<Vec<u32>>,
    first: Option<Zeroizing<u32>>,
    joint: Zeroizing<f64>,
}
impl PreparedDraft {
    fn validate_entry(
        &self,
        position: usize,
        confirmed: u32,
        draft_position: usize,
        draft_binding: u64,
    ) -> Result<()> {
        if self.coordinates[0] != position
            || self.coordinates[1] != draft_position
            || *self.binding != draft_binding
            || self.inputs.first() != Some(&confirmed)
            || !(1..=crate::strata_port::verify_capacity()).contains(&self.inputs.len())
            || self.inputs.iter().any(|&t| t >= VOCAB as u32)
            || self.first.as_ref().is_some_and(|t| **t >= VOCAB as u32)
            || (self.inputs.len() > 1
                && self.first.as_ref().map(|t| **t) != self.inputs.get(1).copied())
            || !self.joint.is_finite()
            || !(0. ..=1.).contains(&*self.joint)
        {
            return Err("global prepared draft identity rejected".into());
        }
        Ok(())
    }
}
struct DraftPlan {
    position: usize,
    maximum: usize,
    minimum: f32,
}
fn prepare_next_draft(
    draft: &mut resident_mtp::ResidentMtp<'_, '_>,
    start: usize,
    tokens: &[u32],
    source: &crate::resident_model::batch::target_rows::DeviceCarry<'_>,
    plan: DraftPlan,
    cancel: &AtomicBool,
) -> kuno_text::Result<PreparedDraft> {
    if plan.position == 0
        || start.checked_add(tokens.len()) != Some(plan.position)
        || tokens.is_empty()
        || !(1..=crate::strata_port::verify_capacity()).contains(&plan.maximum)
    {
        return Err(Error::InvalidFormat("global next draft source extent"));
    }
    let mut inputs = Zeroizing::new(Vec::with_capacity(crate::strata_port::verify_capacity()));
    inputs.push(*tokens.last().ok_or(Error::InvalidBufferLength)?);
    let mut first = None;
    let mut joint = 1.;
    if plan.maximum > 1 {
        let output = draft.draft_accepted_device_window(
            start,
            tokens,
            source,
            plan.maximum - 1,
            plan.minimum,
            cancel,
        )?;
        if output.start != plan.position - 1
            || output.binding != draft.binding()
            || output.tokens().is_empty()
            || output.tokens().len() > plan.maximum - 1
            || draft.position() != plan.position - 1 + output.tokens().len()
        {
            return Err(Error::InvalidFormat(
                "global confidence draft identity rejected",
            ));
        }
        let count = output.usable_window(plan.minimum, plan.maximum)?;
        first = output.tokens().first().copied().map(Zeroizing::new);
        joint = output.joint_probability(count - 1)?;
        inputs.extend_from_slice(&output.tokens()[..count - 1]);
    } else {
        draft.append_device_front(start, tokens, source, cancel)?;
        if draft.position() != plan.position {
            return Err(Error::InvalidFormat(
                "global count1 MTP front cursor rejected",
            ));
        }
    }
    Ok(PreparedDraft {
        coordinates: Zeroizing::new([plan.position, draft.position()]),
        binding: Zeroizing::new(draft.binding()),
        inputs,
        first,
        joint: Zeroizing::new(joint),
    })
}

/// The model's charged request geometry must match this request. The caller
/// controls accepted-token delivery; this module never persists text or tokens.
/// Transient counts only; only caller-selected token output leaves inference.
pub(crate) struct Completion {
    pub outputs: usize,
    pub windows: u64,
    pub state_repairs: u64,
    pub eos: bool,
    pub greedy: bool,
}

#[allow(clippy::too_many_arguments)] // Original model/runtime owners plus bounded session telemetry.
pub(super) fn run<'m, 'f>(
    target: &mut ResidentModel<'m, 'f>,
    executor: &mut GpuExecutor<'f>,
    mut draft: Option<&mut resident_mtp::ResidentMtp<'_, '_>>,
    retained_policy: &mut Option<crate::strata_speculation::Policy>,
    request: Request<'_>,
    cancel: &AtomicBool,
    emit: &mut dyn FnMut(u32) -> Result<()>,
    phase: &mut dyn FnMut(&'static str),
) -> Result<Completion> {
    phase("request_preflight");
    request.validate()?;
    target.validate_global_request_geometry(executor, request.prompt.len(), request.max_outputs)?;
    if !crate::strata_intel_core::enabled()
        || !executor.global_target_scope()?
        || target.completed_target_rows_cursor(executor)? != 0
        || crate::strata_port::profile_target()
        || !crate::strata_port::native_profile_authorized()
    {
        return Err("global request controller scope rejected".into());
    }
    let profile_request = crate::strata_port::native_profile_request();
    let plan = PromptPlan::new(request.prompt.len())?;
    if plan.prefix() != 0 {
        phase("prompt_prefix");
        if let Some(short) = short_prompt::Plan::new(request.prompt.len()) {
            // Upstream serving chooses quantized verification for<=64 fresh
            // prefix inputs. This bounded first step deliberately reuses the
            // existing MAX3 full-commit/host-R path: no nonce/prediction grant
            // is repurposed, and no prompt scratch/cache slots are borrowed.
            for chunk in short.chunks() {
                crate::resident_util::cancel(cancel)?;
                let start = chunk.start;
                let end = chunk.end;
                let (inputs, teacher) = short.inputs(request.prompt, chunk)?;
                if target.completed_target_rows_cursor(executor)? != start
                    || draft.as_ref().is_some_and(|d| d.position() != start)
                {
                    return Err("global short prompt entry cursor rejected".into());
                }
                phase("short_prompt_chunk");
                let rows = target.prefill_chunk_with_carry(executor, inputs, cancel)?;
                if rows.start() != start
                    || rows.count() != inputs.len()
                    || target.completed_target_rows_cursor(executor)? != end
                {
                    return Err("global short prompt completed rows rejected".into());
                }
                if let Some(draft) = draft.as_deref_mut() {
                    let mut carries = Zeroizing::new(Vec::with_capacity(
                        inputs.len() * crate::resident_model::batch::target_rows::HC_WIDTH,
                    ));
                    for row in 0..inputs.len() {
                        if rows.position(row)? != start + row {
                            return Err("global short prompt row identity rejected".into());
                        }
                        carries.extend_from_slice(rows.hidden(row)?);
                    }
                    // Release the original output-host lease BEFORE further
                    // model work. This is a fenced, wiping copy of actual R;
                    // teacher tokens are exact NEXT INPUTS, never greedy picks.
                    drop(rows);
                    draft.append_kv_group(teacher, &carries, cancel)?;
                } else {
                    drop(rows);
                }
                if target.completed_target_rows_cursor(executor)? != end
                    || draft.as_ref().is_some_and(|d| d.position() != end)
                {
                    return Err("global short prompt teacher cursor rejected".into());
                }
            }
        } else {
            target.with_prompt_ple_source(
                executor,
                &request.prompt[..plan.prefix()],
                plan.chunk(),
                cancel,
                |target, executor| {
                    let loan = target.begin_prompt_cache_loan(executor)?;
                    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        // The single lease spans all batched inputs and MTP
                        // teaching, and ends BEFORE normal decode reads the cache.
                        let _scope = target.prompt_scope(loan.as_ref())?;
                        for chunk in plan.chunks() {
                            let end = chunk.end;
                            let (inputs, teacher) = plan.inputs(request.prompt, chunk)?;
                            phase("prompt_chunk");
                            target.prefill_prompt_chunk(
                                executor,
                                inputs,
                                teacher,
                                draft.as_deref_mut(),
                                cancel,
                            )?;
                            if target.completed_target_rows_cursor(executor)? != end {
                                return Err(kuno_text::Error::InvalidFormat(
                                    "global committed prompt prefix cursor",
                                ));
                            }
                        }
                        Ok(())
                    }));
                    target.finish_prompt_cache_loan(executor, loan)?;
                    match attempted {
                        Ok(result) => result,
                        Err(_) => Err(kuno_text::Error::InvalidFormat(
                            "global prompt prefix execution unwound",
                        )),
                    }
                },
            )?;
        }
    }
    if crate::strata_prompt::active()
        || target.completed_target_rows_cursor(executor)? != plan.prefix()
    {
        return Err("global final prompt decode boundary rejected".into());
    }
    // Same original count1 decode path as generated continuations, not the
    // prompt BF16/FP16 matrix path. Even a one-token prompt must do this once.
    let greedy_request = request.sampling.greedy();
    // SAFETY: run owns exclusive model/executor/draft borrows throughout this
    // lexical scope. It never resets/closes them; sampling closes before return
    // and the parent's model cleanup. Errors retire before arena ownership ends.
    phase("sampling_scope");
    let sampling = unsafe {
        target.begin_sampling_request(
            executor,
            request.sampling,
            &request.prompt[..plan.prefix()],
            draft.as_ref().map(|d| d.binding()),
        )?
    };
    sampling.use_device_residuals()?;
    sampling.stage_target(plan.prefix(), &request.prompt[plan.prefix()..])?;
    // Capture actual original MTP identity BEFORE the target window. Its
    // completed front is produced inside the same Pending observer used by
    // every later window, never by an outer post-adaptation fallback.
    let initial_draft = draft
        .as_ref()
        .map(|d| (Zeroizing::new(d.position()), Zeroizing::new(d.binding())));
    if initial_draft
        .as_ref()
        .is_some_and(|(position, _)| **position != plan.prefix())
    {
        return Err("global teacher MTP alignment rejected".into());
    }
    let mut initial_result = None;
    let mut initial_delivered = false;
    let mut delivery_error = None;
    let mut prepared: Option<PreparedDraft> = None;
    phase("last_prompt_input");
    let initial = target.prefill_first_native_window(
        executor,
        &request.prompt[plan.prefix()..],
        cancel,
        &mut |phase| {
            use crate::resident_model::batch::PrefixPhase;
            match phase {
                PrefixPhase::Verified(rows) => {
                    if initial_result.is_some()
                        || rows.start() != plan.prefix()
                        || rows.count() != 1
                    {
                        return Err(Error::InvalidFormat("global initial target row rejected"));
                    }
                    initial_result =
                        Some((Zeroizing::new(rows.greedy(0)?), rows.take_device_carry()?));
                    Ok(1)
                }
                PrefixPhase::Pending {
                    source,
                    accepted: 1,
                } => {
                    let (prediction, device) = initial_result.as_ref().ok_or(
                        Error::InvalidFormat("global initial target prediction missing"),
                    )?;
                    if initial_delivered {
                        return Err(Error::InvalidFormat("global duplicate initial delivery"));
                    }
                    sampling.commit_target(1)?;
                    // Original count1 target state is not yet published. As in
                    // later windows, delivery precedes the scoped next draft;
                    // a late fault still retires the complete original session.
                    if let Err(error) = emit(**prediction) {
                        delivery_error = Some(error);
                        return Err(Error::InvalidFormat("global token delivery failed"));
                    }
                    initial_delivered = true;
                    if let Some(d) = draft
                        .as_deref_mut()
                        .filter(|_| request.max_outputs > 1 && **prediction != request.eos)
                    {
                        let (old_position, binding) = initial_draft.as_ref().ok_or(
                            Error::InvalidFormat("global initial draft identity missing"),
                        )?;
                        validate_round_entry(
                            &[plan.prefix(), **old_position],
                            **binding,
                            std::slice::from_ref(&**prediction),
                            request.prompt.len(),
                            **prediction,
                            d.position(),
                            d.binding(),
                        )
                        .map_err(|_| {
                            Error::InvalidFormat("global initial draft identity rejected")
                        })?;
                        sampling.begin_round(request.prompt.len(), **prediction)?;
                        let grant = source.borrow(device, 1)?;
                        prepared = Some(prepare_next_draft(
                            d,
                            plan.prefix(),
                            std::slice::from_ref(&**prediction),
                            &grant,
                            DraftPlan {
                                position: request.prompt.len(),
                                maximum: request.speculation.mtp_window,
                                minimum: request.draft_min_probability,
                            },
                            cancel,
                        )?);
                    }
                    Ok(1)
                }
                _ => Err(Error::InvalidFormat(
                    "global initial target prefix rejected",
                )),
            }
        },
    );
    if let Some(error) = delivery_error {
        return Err(error);
    }
    initial?;
    let (prediction, initial_device) = initial_result
        .take()
        .ok_or("global initial target prediction missing")?;
    let mut confirmed = *prediction;
    drop(initial_device);
    drop(prediction);
    drop(initial_draft);
    if !initial_delivered
        || prepared.is_some()
            != (draft.is_some() && request.max_outputs > 1 && confirmed != request.eos)
        || target.completed_target_rows_cursor(executor)? != request.prompt.len()
    {
        return Err("global final prompt decode completion rejected".into());
    }
    // Select original target family policies only at the completed boundary.
    // The count1 draft has already finished its scoped R borrow and queue work.
    target.ensure_native_generation_policies(executor, cancel)?;
    if !executor.public_q6_integer_verify_head()?.0 {
        return Err("global native head changed after prefill".into());
    }
    target.begin_public_language_request_io(executor, cancel)?;
    let request_scope = target.current_native_language_scope()?;
    let mut history = Zeroizing::new(Vec::with_capacity(request.max_outputs));
    history.push(confirmed);
    let mut policy = crate::strata_speculation::Policy::take_retained(
        retained_policy,
        request.speculation.window,
    )?;
    let mut suffix_index =
        crate::strata_speculation::Suffix::new(request.speculation.suffix_minimum.max(3))?;
    suffix_index.append(request.prompt)?;
    suffix_index.append(&history)?;
    let mut observation: Option<PolicyObservation> = None;
    let mut committed_cursor = request.prompt.len();
    let mut rounds = 0u64;
    let mut repairs = 0u64;
    while history.len() < request.max_outputs && confirmed != request.eos {
        crate::resident_util::cancel(cancel)?;
        let position = target.completed_target_rows_cursor(executor)?;
        if position != request.prompt.len() + history.len() - 1 {
            return Err("global target entry alignment rejected".into());
        }
        let remaining = request.max_outputs - history.len();
        let ready = if let Some(d) = draft.as_deref_mut() {
            let ready = prepared
                .take()
                .ok_or("global scoped draft producer missing")?;
            ready.validate_entry(position, confirmed, d.position(), d.binding())?;
            ready
        } else {
            if prepared.is_some() {
                return Err("global accepted-round owner missing".into());
            }
            sampling.begin_round(position, confirmed)?;
            PreparedDraft {
                coordinates: Zeroizing::new([position, 0]),
                binding: Zeroizing::new(0),
                inputs: Zeroizing::new(vec![confirmed]),
                first: None,
                joint: Zeroizing::new(1.),
            }
        };
        let mut count = ready.inputs.len();
        let mut inputs = ready.inputs;
        let first_draft = ready.first.as_ref().map(|x| **x);
        let joint_probability = *ready.joint;
        // The preceding round's native cost includes the draft just joined.
        // Finish it BEFORE choosing the current lookup/chain policy.
        if let Some(previous) = observation.take() {
            previous.finish(&mut policy)?;
        }
        let started = Instant::now();
        let mut suffix = None;
        if request.speculation.suffix_minimum > 0 {
            let candidate = suffix_index.propose(request.speculation.window - 1);
            if let Some(candidate) =
                candidate.filter(|p| draft.is_none() || p.tokens.first().copied() == first_draft)
                && let Some(selected) = policy.choose(count, &candidate)
            {
                count = selected;
                inputs.truncate(1);
                inputs.extend_from_slice(&candidate.tokens[..count - 1]);
                suffix = Some(candidate);
            }
        }
        let mut chain = None;
        if request.speculation.lookup_chain > 0
            && draft.is_some()
            && suffix.is_none()
            && count < request.speculation.window
        {
            let candidate = suffix_index.propose_after(
                &inputs[1..],
                request
                    .speculation
                    .lookup_chain
                    .min(request.speculation.window - count),
            )?;
            if let Some(candidate) =
                candidate.filter(|p| p.matched >= request.speculation.chain_minimum)
            {
                let extra = policy.chain(count, joint_probability, &candidate);
                if extra > 0 {
                    chain = Some((count, extra, candidate.matched));
                    inputs.extend_from_slice(&candidate.tokens[..extra]);
                    count += extra;
                }
            }
        }
        if inputs.len() != count
            || count > request.speculation.window
            || position
                .checked_add(count)
                .is_none_or(|end| end > crate::qsa_gpu::MAX_CONTEXT)
        {
            return Err("global draft window bound rejected".into());
        }
        sampling.stage_target(position, &inputs)?;
        let mut selected = None;
        let previous_total = history.len();
        let mut delivered = None;
        let mut delivery_error = None;
        let mut native_profile = Some(crate::strata_port::NativeProfileRound::new(
            profile_request,
            rounds,
        ));
        let verification = target.verify_native_prefix(executor, &inputs, cancel, &mut |phase| {
            use crate::resident_model::batch::PrefixPhase;
            match phase {
                PrefixPhase::Verified(rows) => {
                    // Target math has completed. Commit export and next-draft
                    // work retain their original unprofiled pending protocol.
                    drop(native_profile.take());
                    if rows.start() != position || rows.count() != count {
                        return Err(Error::InvalidFormat(
                            "global verification identity rejected",
                        ));
                    }
                    let mut predicted = Zeroizing::new(Vec::with_capacity(count));
                    for row in 0..count {
                        if rows.position(row)? != position + row {
                            return Err(Error::InvalidFormat("global target row stamp rejected"));
                        }
                        predicted.push(rows.greedy(row)?);
                    }
                    let n = accepted(&inputs, &predicted)
                        .map_err(|_| Error::InvalidFormat("global accepted prefix rejected"))?;
                    let device = rows.take_device_carry()?;
                    selected = Some((predicted, device));
                    Ok(n)
                }
                PrefixPhase::Pending {
                    source,
                    accepted: n,
                } => {
                    let (predicted, device) = selected
                        .as_ref()
                        .ok_or(Error::InvalidFormat("global verified prefix missing"))?;
                    if delivered.is_some() {
                        return Err(Error::InvalidFormat("global duplicate prefix delivery"));
                    }
                    sampling.commit_target(n)?;
                    let emit_count = emitted(&predicted[..n], remaining, request.eos);
                    let continuing = predicted[emit_count - 1] != request.eos
                        && previous_total + emit_count < request.max_outputs;
                    if continuing && emit_count != n {
                        return Err(Error::InvalidFormat(
                            "global continuing committed prefix rejected",
                        ));
                    }
                    // Pinned serving emits real verified predictions after
                    // commit submission and BEFORE next drafting. Output does
                    // not publish target state or claim that commit succeeded.
                    confirmed = predicted[emit_count - 1];
                    suffix_index
                        .append(&predicted[..emit_count])
                        .map_err(|_| Error::InvalidFormat("global suffix delivery rejected"))?;
                    for &token in &predicted[..emit_count] {
                        history.push(token);
                        if let Err(error) = emit(token) {
                            delivery_error = Some(error);
                            return Err(Error::InvalidFormat("global token delivery failed"));
                        }
                    }
                    delivered = Some((n, emit_count, continuing));
                    if let Some(d) = draft.as_deref_mut().filter(|_| continuing) {
                        let next_position = position + n;
                        sampling.begin_round(next_position, predicted[n - 1])?;
                        let grant = source.borrow(device, n)?;
                        prepared = Some(prepare_next_draft(
                            d,
                            position,
                            &predicted[..n],
                            &grant,
                            DraftPlan {
                                position: next_position,
                                maximum: request.speculation.mtp_window,
                                minimum: request.draft_min_probability,
                            },
                            cancel,
                        )?);
                    }
                    Ok(n)
                }
            }
        });
        drop(native_profile);
        // The original model already ran its terminal failure cleanup. Preserve
        // the original delivery error rather than replacing its typed cause.
        if let Some(error) = delivery_error {
            return Err(error);
        }
        let n = verification?;
        let (_predicted, _device) = selected.ok_or("global verified prefix observation missing")?;
        let (delivered_prefix, emit_count, continuing) =
            delivered.ok_or("global verified prefix delivery missing")?;
        if delivered_prefix != n || history.len() != previous_total + emit_count {
            return Err("global delivered prefix accounting rejected".into());
        }
        repairs += u64::from(n < count);
        if target.completed_target_rows_cursor(executor)? != position + n {
            return Err("global accepted target cursor rejected".into());
        }
        // The native next round fronts all accepted (prediction,R) rows plus
        // its confirmed next input in ONE batch. Do not eagerly repair n-1
        // rows here and then run a separate count1 front at the next entry.
        committed_cursor = position + n;
        if prepared.is_some() != (draft.is_some() && continuing) {
            return Err("global prepared next draft lifecycle rejected".into());
        }
        // Native DraftPolicy consumes ACCEPTED DRAFTS (n-1), not emitted
        // rows or output clipping. EOS exits without another policy/draft pass.
        if confirmed != request.eos {
            let measured = PolicyObservation {
                started,
                window: count,
                accepted: n - 1,
                lookup: suffix.as_ref().map(|s| s.matched),
                chain,
            };
            if continuing {
                observation = Some(measured);
            } else {
                measured.finish(&mut policy)?;
            }
        }
        rounds += 1;
    }
    if prepared.is_some()
        || observation.is_some()
        || target.completed_target_rows_cursor(executor)? != committed_cursor
    {
        return Err("global final output accounting rejected".into());
    }
    if let Some(draft) = draft {
        draft.finish_native_request()?;
    }
    sampling.close()?;
    let closed = target.end_public_language_request_io(executor, cancel)?;
    if !closed.full_io_audit
        || !closed.scope_closed
        || closed.scope != request_scope
        || target.public_language_request_io_active()?
    {
        return Err("global request erasure rejected".into());
    }
    *retained_policy = Some(policy); // only after original request/fault/I/O scope closes
    Ok(Completion {
        outputs: history.len(),
        windows: rounds,
        state_repairs: repairs,
        eos: confirmed == request.eos,
        greedy: greedy_request,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_next_draft_requires_its_original_completed_producer_and_tokens() {
        let ready = |inputs: Vec<u32>, first: Option<u32>, joint| PreparedDraft {
            coordinates: Zeroizing::new([23, 25]),
            binding: Zeroizing::new(99),
            inputs: Zeroizing::new(inputs),
            first: first.map(Zeroizing::new),
            joint: Zeroizing::new(joint),
        };
        // Confidence may reduce the usable window without rewinding the actual
        // producer cursor or losing its first candidate for suffix comparison.
        for inputs in [vec![10], vec![10, 11], vec![10, 11, 12]] {
            let p = ready(inputs, Some(11), 0.8);
            p.validate_entry(23, 10, 25, 99).unwrap();
            for (position, confirmed, cursor, binding) in [
                (24, 10, 25, 99),
                (23, 11, 25, 99),
                (23, 10, 24, 99),
                (23, 10, 25, 98),
            ] {
                assert!(
                    p.validate_entry(position, confirmed, cursor, binding)
                        .is_err()
                );
            }
        }
        for p in [
            ready(vec![], None, 1.),
            ready(vec![10, 11], Some(12), 0.8),
            ready(vec![10, 11], None, 0.8),
            ready(vec![10, VOCAB as u32], Some(VOCAB as u32), 0.8),
            ready(vec![10], Some(VOCAB as u32), 0.8),
            ready(vec![10], None, f64::NAN),
            ready(vec![10], None, -0.1),
            ready(vec![10], None, 1.1),
            ready(
                vec![10; crate::strata_port::verify_capacity() + 1],
                None,
                1.,
            ),
        ] {
            assert!(p.validate_entry(23, 10, 25, 99).is_err());
        }
        ready(vec![10], None, 1.)
            .validate_entry(23, 10, 25, 99)
            .unwrap();
    }
    #[test]
    fn native_grouped_round_entry_is_bound_to_accepted_prefix_and_original_model() {
        for n in 1..=crate::strata_port::verify_capacity() {
            let predicted: Vec<_> = (10..10 + n as u32).collect();
            let next = 9 + n as u32;
            let check = |p, t, d, b| validate_round_entry(&[20, 22], 99, &predicted, p, t, d, b);
            check(20 + n, next, 22, 99).unwrap();
            assert!(check(21 + n, next, 22, 99).is_err());
            assert!(check(20 + n, next + 1, 22, 99).is_err());
            assert!(check(20 + n, next, 23, 99).is_err());
            assert!(check(20 + n, next, 22, 100).is_err());
        }
    }
    #[test]
    fn native_prompt_prefix_and_teacher_exclude_exactly_one_normal_decode_input() {
        for n in [
            1,
            2,
            30,
            35,
            255,
            256,
            257,
            258,
            2048,
            2049,
            2050,
            3073,
            4095,
            4096,
            crate::qsa_gpu::MAX_CONTEXT,
        ] {
            let prompt: Vec<u32> = (0..n as u32).collect();
            let plan = PromptPlan::new(n).unwrap();
            assert_eq!(plan.prefix(), n - 1);
            if n == 1 {
                assert_eq!(plan.chunk(), 0);
                assert_eq!(plan.chunks().count(), 0);
            } else {
                assert!(plan.chunk().is_multiple_of(256));
                assert!(plan.chunk() <= crate::strata_prompt::groups::CAPACITY);
            }
            let mut consumed = 0;
            let mut taught = Vec::new();
            for chunk in plan.chunks() {
                assert_eq!(chunk.start, consumed);
                let end = chunk.end;
                let (inputs, teacher) = plan.inputs(&prompt, chunk).unwrap();
                assert_eq!(inputs.len(), teacher.len());
                assert!(inputs.iter().zip(teacher).all(|(a, b)| a + 1 == *b));
                consumed = end;
                taught.extend_from_slice(teacher);
            }
            assert_eq!(consumed, n - 1);
            assert_eq!(taught, prompt[1..]);
            // This SAME final input reaches count1 after scope/cache restore;
            // no batched row/head consumes it, and no token is skipped/repeated.
            assert_eq!(&prompt[plan.prefix()..], &[(n - 1) as u32]);
        }
        assert!(PromptPlan::new(0).is_err());
        assert!(PromptPlan::new(crate::qsa_gpu::MAX_CONTEXT + 1).is_err());
        let plan = PromptPlan::new(35).unwrap();
        let prompt = vec![0; 35];
        for bad in [0..0, 1..34, 0..35, 0..33] {
            assert!(plan.inputs(&prompt, bad).is_err());
        }
        assert!(plan.inputs(&prompt[..34], 0..34).is_err());
    }
    #[test]
    fn native_speculation_commit_is_independent_of_eos_and_output_cap() {
        // Native commits all exact target matches, even when delivery stops
        // at EOS or before the end of the configured verification window.
        let inputs = [5, 99, 7, 8];
        let predictions = [99, 7, 8, 9];
        assert_eq!(accepted(&inputs, &predictions).unwrap(), 4);
        assert_eq!(emitted(&predictions, 4, 99), 1);
        assert_eq!(emitted(&predictions, 2, 100), 2);
        assert_eq!(emitted(&predictions, 1, 100), 1);
        for count in 1..=crate::strata_port::verify_capacity() {
            let config = crate::strata_speculation::Config::native(count);
            config.validate().unwrap();
            assert_eq!(config.mtp_window, count); // Never clipped to remaining output.
        }
    }
    #[test]
    fn global_prefix_acceptance_covers_every_width_rejection_and_eos() {
        for count in 1..=crate::strata_port::verify_capacity() {
            let inputs: Vec<_> = (0..count as u32).collect();
            let next: Vec<_> = (1..=count as u32).collect();
            assert_eq!(accepted(&inputs, &next).unwrap(), count);
            for n in 1..count {
                let mut rejected = next.clone();
                rejected[n - 1] = 50;
                assert_eq!(accepted(&inputs, &rejected).unwrap(), n);
            }
            for row in 0..count {
                let mut eos = next.clone();
                eos[row] = 99;
                assert_eq!(accepted(&inputs, &eos).unwrap(), row + 1);
            }
        }
        assert!(accepted(&[], &[]).is_err());
        assert!(accepted(&[0], &[VOCAB as u32]).is_err());
        assert!(accepted(&[0], &[1, 2]).is_err());
    }
}
