//! Connected global-source controller, derived from MIT Strata's Intel
//! generate.cpp speculative loop at fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
//! No legacy bank receipts, fabricated dispatches, operator profiles or speed
//! reports. Prompt/carries/drafts/history are bounded, transient and wiped.
use super::*;

fn public_phase(phase: &'static str) {
    if std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1") {
        eprintln!("native_public_execution phase={phase}");
    }
}

pub(super) struct Request<'a> {
    pub prompt: &'a [u32],
    pub max_outputs: usize,
    pub eos: u32,
    pub speculation: crate::strata_speculation::Config,
    pub draft_min_probability: f32,
    pub sampling: crate::strata_sampler::Params,
}
impl Request<'_> {
    fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        self.speculation.validate()?;
        if crate::strata_speculation::HOST_BYTES + MAX_OUTPUTS * 4 + 64 * 1024 > FIXTURE_HOST_BYTES
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
            || self
                .prompt
                .len()
                .checked_add(self.max_outputs)
                .is_none_or(|n| n > 4096)
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
        if !(1..=4096).contains(&prompt) {
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

/// One accepted verifier prefix retained only until the next round's grouped
/// front. These are actual target predictions plus their original device-source
/// receipt, never rejected proposals or a host residual mirror.
struct AcceptedRound {
    coordinates: Zeroizing<[usize; 2]>, // original target start, complete old MTP cursor
    binding: Zeroizing<u64>,
    predicted: Zeroizing<Vec<u32>>,
    device: crate::resident_model::batch::target_rows::DeviceCarryReceipt,
}
impl AcceptedRound {
    fn validate_entry(
        &self,
        position: usize,
        confirmed: u32,
        draft_position: usize,
        binding: u64,
    ) -> Result<()> {
        validate_round_entry(
            &self.coordinates,
            *self.binding,
            &self.predicted,
            position,
            confirmed,
            draft_position,
            binding,
        )
    }
}
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

/// The model's charged request geometry must match this request. The caller
/// controls accepted-token delivery; this module never persists text or tokens.
pub(super) fn run<'m, 'f>(
    target: &mut ResidentModel<'m, 'f>,
    executor: &mut GpuExecutor<'f>,
    mut draft: Option<&mut resident_mtp::ResidentMtp<'_, '_>>,
    request: Request<'_>,
    cancel: &AtomicBool,
    emit: &mut dyn FnMut(u32) -> Result<()>,
) -> Result<serde_json::Value> {
    public_phase("request_preflight");
    request.validate()?;
    target.validate_global_request_geometry(executor, request.prompt.len(), request.max_outputs)?;
    if !crate::strata_intel_core::enabled()
        || !executor.global_target_scope()?
        || target.completed_target_rows_cursor(executor)? != 0
        || crate::strata_port::profile_target()
        || crate::strata_port::profile_native_target()
    {
        return Err("global request controller scope rejected".into());
    }
    let plan = PromptPlan::new(request.prompt.len())?;
    if plan.prefix() != 0 {
        public_phase("prompt_prefix");
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
                        public_phase("prompt_chunk");
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
    public_phase("sampling_scope");
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
    let mut output =
        target.prefill_chunk_with_carry(executor, &request.prompt[plan.prefix()..], cancel)?;
    if output.start() != plan.prefix() || output.count() != 1 {
        return Err("global final prompt decode identity rejected".into());
    }
    let mut confirmed = output.greedy(0)?;
    let initial_device = output.take_device_carry()?;
    drop(output);
    if target.completed_target_rows_cursor(executor)? != request.prompt.len() {
        return Err("global final prompt decode cursor rejected".into());
    }
    if draft
        .as_ref()
        .is_some_and(|d| d.position() != request.prompt.len() - 1)
    {
        return Err("global teacher MTP alignment rejected".into());
    }
    sampling.commit_target(1)?;
    // Shared state owners are activated with actual global source authority.
    target.set_public_four_rows(executor, cancel)?;
    target.set_public_grouped_ple_reads(executor, true, cancel)?;
    target.set_public_grouped_status_folds(executor, true, cancel)?;
    if !executor.public_q6_integer_verify_head()?.0 {
        return Err("global native head changed after prefill".into());
    }
    target.begin_public_language_request_io(executor, cancel)?;
    let mut history = Zeroizing::new(Vec::with_capacity(request.max_outputs));
    history.push(confirmed);
    emit(confirmed)?;
    let mut policy = crate::strata_speculation::Policy::new(request.speculation.window)?;
    let mut suffix_index =
        crate::strata_speculation::Suffix::new(request.speculation.suffix_minimum.max(3))?;
    suffix_index.append(request.prompt)?;
    suffix_index.append(&history)?;
    let mut observation: Option<PolicyObservation> = None;
    let mut committed_cursor = request.prompt.len();
    let mut pending: Option<AcceptedRound> = if request.max_outputs > 1 && confirmed != request.eos
    {
        draft.as_ref().map(|d| AcceptedRound {
            coordinates: Zeroizing::new([plan.prefix(), d.position()]),
            binding: Zeroizing::new(d.binding()),
            predicted: Zeroizing::new(vec![confirmed]),
            device: initial_device,
        })
    } else {
        None
    };
    let mut rounds = 0u64;
    let mut repairs = 0u64;
    while history.len() < request.max_outputs && confirmed != request.eos {
        crate::resident_util::cancel(cancel)?;
        let position = target.completed_target_rows_cursor(executor)?;
        if position != request.prompt.len() + history.len() - 1 {
            return Err("global target entry alignment rejected".into());
        }
        if let Some(d) = draft.as_ref() {
            if let Some(accepted) = &pending {
                accepted.validate_entry(position, confirmed, d.position(), d.binding())?;
            } else if d.position() != position - 1 {
                return Err("global draft entry alignment rejected".into());
            }
        } else if pending.is_some() {
            return Err("global accepted-round owner missing".into());
        }
        sampling.begin_round(position, confirmed)?;
        let remaining = request.max_outputs - history.len();
        let mut count = if draft.is_some() {
            request.speculation.mtp_window
        } else {
            1
        };
        let mut inputs = Zeroizing::new(Vec::with_capacity(crate::strata_port::verify_capacity()));
        inputs.push(confirmed);
        let mut joint_probability = 1.;
        let mut first_draft = None;
        if let Some(draft) = draft.as_deref_mut() {
            let accepted = pending.take().ok_or("global device front source missing")?;
            let source =
                target.borrow_target_carry(executor, &accepted.device, accepted.predicted.len())?;
            if count > 1 {
                let maximum = count;
                let output = draft.draft_accepted_device_window(
                    accepted.coordinates[0],
                    &accepted.predicted,
                    &source,
                    maximum - 1,
                    request.draft_min_probability,
                    cancel,
                )?;
                if output.start != position - 1
                    || output.binding != draft.binding()
                    || output.tokens().is_empty()
                    || output.tokens().len() > maximum - 1
                    || draft.position() != position - 1 + output.tokens().len()
                {
                    return Err("global confidence draft identity rejected".into());
                }
                count = output.usable_window(request.draft_min_probability, maximum)?;
                first_draft = output.tokens().first().copied();
                joint_probability = output.joint_probability(count - 1)?;
                inputs.extend_from_slice(&output.tokens()[..count - 1]);
            } else {
                draft.append_device_front(
                    accepted.coordinates[0],
                    &accepted.predicted,
                    &source,
                    cancel,
                )?;
                if draft.position() != position {
                    return Err("global count1 MTP front cursor rejected".into());
                }
            }
        }
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
            || pending.is_some()
            || position.checked_add(count).is_none_or(|end| end > 4096)
        {
            return Err("global draft window bound rejected".into());
        }
        sampling.stage_target(position, &inputs)?;
        let mut round = LatestTargetRound::begin(target, executor, &inputs, cancel)?;
        let mut rows = round.verify(cancel)?;
        if rows.start() != position || rows.count() != count {
            return Err("global verification identity rejected".into());
        }
        let mut predicted = Zeroizing::new(Vec::with_capacity(count));
        for row in 0..count {
            if rows.position(row)? != position + row {
                return Err("global target row stamp rejected".into());
            }
            predicted.push(rows.greedy(row)?);
        }
        let n = accepted(&inputs, &predicted)?;
        let device = rows.take_device_carry()?;
        drop(rows);
        // The core repairs only real recurrent/KV/history state. Cache routes
        // count the original verified window once; repair does not count twice.
        if round.finish(n, cancel)?.is_some() {
            return Err("global state-only repair contract rejected".into());
        }
        sampling.commit_target(n)?;
        repairs += u64::from(n < count);
        if target.completed_target_rows_cursor(executor)? != position + n {
            return Err("global accepted target cursor rejected".into());
        }
        // The native next round fronts all accepted (prediction,R) rows plus
        // its confirmed next input in ONE batch. Do not eagerly repair n-1
        // rows here and then run a separate count1 front at the next entry.
        committed_cursor = position + n;
        let emit_count = emitted(&predicted[..n], remaining, request.eos);
        let next_total = history.len() + emit_count;
        let continuing =
            predicted[emit_count - 1] != request.eos && next_total < request.max_outputs;
        if continuing && emit_count != n {
            return Err("global continuing committed prefix rejected".into());
        }
        let defer_front = draft.is_some() && continuing;
        confirmed = predicted[emit_count - 1];
        suffix_index.append(&predicted[..emit_count])?;
        for &token in &predicted[..emit_count] {
            history.push(token);
            emit(token)?;
        }
        if defer_front {
            let d = draft
                .as_ref()
                .ok_or("global accepted-round draft owner missing")?;
            predicted.truncate(n);
            pending = Some(AcceptedRound {
                coordinates: Zeroizing::new([position, d.position()]),
                binding: Zeroizing::new(d.binding()),
                predicted,
                device,
            });
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
    if pending.is_some()
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
        || closed.scope != 1
        || target.public_language_request_io_active()?
    {
        return Err("global request erasure rejected".into());
    }
    Ok(
        serde_json::json!({"controller":"global-original-sources","greedy_target_verification":greedy_request,
        "outputs":history.len(),"windows":rounds,"state_repairs":repairs,
        "eos":confirmed==request.eos,"request_scope_erased":true,"content_persisted":false,
        "speed_acceptance":false,"full_rewrite_accepted":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
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
            1, 2, 30, 35, 255, 256, 257, 258, 2048, 2049, 2050, 3073, 4095, 4096,
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
        assert!(PromptPlan::new(4097).is_err());
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
