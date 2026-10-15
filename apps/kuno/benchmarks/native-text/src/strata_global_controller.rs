//! Connected global-source controller, derived from MIT Strata's Intel
//! generate.cpp speculative loop at fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
//! No legacy bank receipts, fabricated dispatches, operator profiles or speed
//! reports. Prompt/carries/drafts/history are bounded, transient and wiped.
use super::*;

pub(super) struct Request<'a> {
    pub prompt: &'a [u32],
    pub max_outputs: usize,
    pub eos: u32,
    pub window: usize,
    pub sampling: crate::strata_sampler::Params,
}
impl Request<'_> {
    fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if self.prompt.is_empty()
            || self.prompt.iter().any(|&t| t >= VOCAB as u32)
            || !(1..=MAX_OUTPUTS).contains(&self.max_outputs)
            || self.eos >= VOCAB as u32
            || !(1..=crate::strata_port::verify_capacity()).contains(&self.window)
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
/// that many target predictions, stopping at the first target EOS. Drafts can
/// never overrule a target prediction. This is the pinned greedy contract.
fn accepted(inputs: &[u32], predicted: &[u32], eos: u32) -> Result<usize> {
    if inputs.is_empty()
        || inputs.len() != predicted.len()
        || inputs.len() > crate::strata_port::verify_capacity()
        || inputs.iter().chain(predicted).any(|&t| t >= VOCAB as u32)
    {
        return Err("global target prefix rejected".into());
    }
    let mut n = 1;
    while n < inputs.len() && predicted[n - 1] != eos && inputs[n] == predicted[n - 1] {
        n += 1;
    }
    if let Some(first) = predicted[..n].iter().position(|&t| t == eos) {
        n = first + 1;
    }
    Ok(n)
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
    let sampling = unsafe {
        target.begin_sampling_request(
            executor,
            request.sampling,
            &request.prompt[..plan.prefix()],
            draft.as_ref().map(|d| d.binding()),
        )?
    };
    sampling.stage_target(plan.prefix(), &request.prompt[plan.prefix()..])?;
    let output =
        target.prefill_chunk_with_carry(executor, &request.prompt[plan.prefix()..], cancel)?;
    if output.start() != plan.prefix() || output.count() != 1 {
        return Err("global final prompt decode identity rejected".into());
    }
    let mut confirmed = output.greedy(0)?;
    let mut carry = Zeroizing::new(output.hidden(0)?.to_vec());
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
    let mut policy = crate::strata_port::DraftPolicy::default();
    let mut rounds = 0u64;
    let mut repairs = 0u64;
    while history.len() < request.max_outputs && confirmed != request.eos {
        crate::resident_util::cancel(cancel)?;
        let position = target.completed_target_rows_cursor(executor)?;
        if position != request.prompt.len() + history.len() - 1
            || draft.as_ref().is_some_and(|d| d.position() != position - 1)
        {
            return Err("global target/draft entry alignment rejected".into());
        }
        sampling.begin_round(position, confirmed)?;
        let remaining = request.max_outputs - history.len();
        let mut count = if draft.is_some() {
            policy.choose_mtp(remaining.min(request.window))
        } else {
            1
        };
        let mut inputs = Zeroizing::new(Vec::with_capacity(crate::strata_port::verify_capacity()));
        inputs.push(confirmed);
        let started = Instant::now();
        let mut suffix = None;
        if let Some(draft) = draft.as_deref_mut() {
            let candidate = crate::strata_port::lookup(
                request.prompt,
                &history,
                remaining.min(request.window) - 1,
            );
            if count > 1 {
                if candidate.is_none() {
                    let output = draft.draft_window(confirmed, &carry, count - 1, cancel)?;
                    if output.start != position - 1
                        || output.binding != draft.binding()
                        || output.tokens().len() != count - 1
                        || draft.position() != position + count - 2
                    {
                        return Err("global queued draft identity rejected".into());
                    }
                    inputs.extend_from_slice(output.tokens());
                } else {
                    let output = draft.step_draft(confirmed, &carry, cancel)?;
                    if output.start() != position - 1 {
                        return Err("global first draft stamp rejected".into());
                    }
                    inputs.push(output.greedy());
                    drop(output);
                    if let Some(candidate) = candidate.filter(|p| p.tokens.first() == inputs.get(1))
                        && let Some(selected) = policy.choose(count, &candidate)
                    {
                        count = selected;
                        for &token in candidate.tokens.iter().skip(1).take(count - 2) {
                            draft.append_kv(
                                *inputs.last().ok_or("global suffix predecessor absent")?,
                                &carry,
                                cancel,
                            )?;
                            inputs.push(token);
                        }
                        suffix = Some(candidate);
                    }
                    while inputs.len() < count {
                        let output = draft.follow_draft(cancel)?;
                        if output.start() != position + inputs.len() - 2
                            || output.binding() != draft.binding()
                        {
                            return Err("global follow draft identity rejected".into());
                        }
                        inputs.push(output.greedy());
                    }
                }
            }
        }
        if inputs.len() != count || count > remaining {
            return Err("global draft window bound rejected".into());
        }
        sampling.stage_target(position, &inputs)?;
        let mut round = LatestTargetRound::begin(target, executor, &inputs, cancel)?;
        let rows = round.verify(cancel)?;
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
        let n = accepted(&inputs, &predicted, request.eos)?;
        let mut carries = Zeroizing::new(Vec::with_capacity(n * HIDDEN));
        for row in 0..n {
            carries.extend_from_slice(rows.hidden(row)?);
        }
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
        if let Some(draft) = draft.as_deref_mut() {
            draft.repair_accepted_cache(position, &inputs[..n], &carry, &carries, cancel)?;
            if draft.position() != position + n - 1 {
                return Err("global accepted MTP cursor rejected".into());
            }
        }
        carry.copy_from_slice(&carries[(n - 1) * HIDDEN..n * HIDDEN]);
        confirmed = predicted[n - 1];
        for &token in &predicted[..n] {
            history.push(token);
            emit(token)?;
        }
        // Native policy needs request-local cost observations. They are not
        // printed, persisted or used to choose between incomplete port paths.
        policy.observe(
            suffix.is_some(),
            count,
            n,
            suffix.as_ref().map_or(0, |s| s.matched),
            elapsed(started)?,
        );
        rounds += 1;
    }
    if target.completed_target_rows_cursor(executor)? != request.prompt.len() + history.len() - 1 {
        return Err("global final output accounting rejected".into());
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
    fn global_prefix_acceptance_covers_every_width_rejection_and_eos() {
        for count in 1..=crate::strata_port::verify_capacity() {
            let inputs: Vec<_> = (0..count as u32).collect();
            let next: Vec<_> = (1..=count as u32).collect();
            assert_eq!(accepted(&inputs, &next, 99).unwrap(), count);
            for n in 1..count {
                let mut rejected = next.clone();
                rejected[n - 1] = 50;
                assert_eq!(accepted(&inputs, &rejected, 99).unwrap(), n);
            }
            for row in 0..count {
                let mut eos = next.clone();
                eos[row] = 99;
                assert_eq!(accepted(&inputs, &eos, 99).unwrap(), row + 1);
            }
        }
        assert!(accepted(&[], &[], 99).is_err());
        assert!(accepted(&[0], &[VOCAB as u32], 99).is_err());
        assert!(accepted(&[0], &[1, 2], 99).is_err());
    }
}
