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
}
impl Request<'_> {
    fn validate(&self) -> Result<()> {
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
    let mut carry = Zeroizing::new(vec![0.0f32; HIDDEN]);
    let mut confirmed = 0;
    // Exact teacher forcing: the last prompt carry predicts the first output;
    // MTP has consumed only prompt_len-1 cells at this boundary.
    for start in (0..request.prompt.len()).step_by(3) {
        let count = (request.prompt.len() - start).min(3);
        let rows = target.prompt_projection_chunk_with_carry(
            executor,
            &request.prompt[start..start + count],
            cancel,
        )?;
        if rows.start() != start || rows.count() != count {
            return Err("global prefill rows rejected".into());
        }
        for row in 0..count {
            let position = start + row;
            if rows.position(row)? != position {
                return Err("global prefill cursor rejected".into());
            }
            if position + 1 == request.prompt.len() {
                confirmed = rows.greedy(row)?;
                carry.copy_from_slice(rows.hidden(row)?);
            }
        }
        if let Some(draft) = draft.as_deref_mut() {
            if draft.position() != start {
                return Err("global teacher group cursor rejected".into());
            }
            let teacher_count = count.min(request.prompt.len() - 1 - start);
            if teacher_count > 0 {
                let mut teacher_tokens = Zeroizing::new(Vec::with_capacity(teacher_count));
                let mut teacher_carries =
                    Zeroizing::new(Vec::with_capacity(teacher_count * HIDDEN));
                for row in 0..teacher_count {
                    teacher_tokens.push(
                        teacher_input(request.prompt, start + row, start + row)?
                            .ok_or("global teacher group rejected")?,
                    );
                    teacher_carries.extend_from_slice(rows.hidden(row)?);
                }
                draft.append_kv_group(&teacher_tokens, &teacher_carries, cancel)?;
            }
        }
        drop(rows);
        if target.completed_target_rows_cursor(executor)? != start + count {
            return Err("global committed prefill cursor rejected".into());
        }
    }
    if draft
        .as_ref()
        .is_some_and(|d| d.position() != request.prompt.len() - 1)
    {
        return Err("global teacher MTP alignment rejected".into());
    }
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
    let closed = target.end_public_language_request_io(executor, cancel)?;
    if !closed.full_io_audit
        || !closed.scope_closed
        || closed.scope != 1
        || target.public_language_request_io_active()?
    {
        return Err("global request erasure rejected".into());
    }
    Ok(
        serde_json::json!({"controller":"global-original-sources","greedy_target_verification":true,
        "outputs":history.len(),"windows":rounds,"state_repairs":repairs,
        "eos":confirmed==request.eos,"request_scope_erased":true,"content_persisted":false,
        "speed_acceptance":false,"full_rewrite_accepted":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
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
