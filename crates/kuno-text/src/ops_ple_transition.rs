//! Narrow checked native PLE history transition. This is not a state setter.
//! The caller must accept the completed WHOLE-model chunk and independently
//! audit/drop all device map leases before committing ANY model state owner.
use super::{
    CausalConvState, OpsError, PleHashConfig, PleTokenHistory, clear_f32_capacity,
    clear_i32_capacity,
};
use std::sync::atomic::{AtomicBool, Ordering};
/// Whole-prompt counterpart. The native model has already closed its full
/// computation; retain only the last nine normalized rows for an independent
/// exact history check, rather than mapping all prompt activations to the CPU.
#[allow(clippy::too_many_arguments)]
pub fn prepare_prompt<'a>(
    conv: &'a mut CausalConvState,
    history: &'a mut PleTokenHistory,
    config: &PleHashConfig,
    old_tokens: &[i32; 2],
    tokens: &[u32],
    old: &[f32],
    tail: &[f32],
    next: &[f32],
    byte_budget: usize,
    cancel: &AtomicBool,
) -> Result<PleChunkCommit<'a>> {
    cancelled(cancel)?;
    let owned = owner_capacity_bytes(conv, history)?;
    let rows = tokens.len().min(HISTORY);
    if !(1..=2048).contains(&tokens.len())
        || config.history_len() != 2
        || config.heads_per_ngram != 8
        || config.boundary_token != 248044
        || config.token_vocab_size != 248320
        || history.tokens != old_tokens
        || old.len() != HISTORY_WORDS
        || next.len() != HISTORY_WORDS
        || tail.len() != rows * CHANNELS
        || tokens.iter().any(|&t| t >= config.token_vocab_size)
        || !bits_equal(old, &conv.history)
        || old.iter().chain(next).chain(tail).any(|x| !x.is_finite())
    {
        return Err(OpsError::DimensionMismatch(CONTEXT));
    }
    let available = byte_budget
        .checked_sub(owned)
        .ok_or(OpsError::CapacityExceeded(CONTEXT))?;
    let mut candidate = candidate(old, available)?;
    for c in 0..CHANNELS {
        cancelled(cancel)?;
        let base = c * HISTORY;
        if rows < HISTORY {
            candidate.0[base..base + HISTORY - rows]
                .copy_from_slice(&old[base + rows..base + HISTORY]);
        }
        for r in 0..rows {
            candidate.0[base + HISTORY - rows + r] = tail[r * CHANNELS + c];
        }
    }
    let mut next_tokens = Zeroizing::new(*old_tokens);
    for &t in tokens {
        next_tokens[0] = next_tokens[1];
        next_tokens[1] = t as i32;
    }
    if !bits_equal(&candidate.0, next) {
        return Err(OpsError::InvalidParameter(CONTEXT));
    }
    cancelled(cancel)?;
    Ok(PleChunkCommit {
        conv,
        tokens: history,
        history: candidate,
        next_tokens,
    })
}

#[cfg(test)]
use zeroize::Zeroize;
use zeroize::Zeroizing;

pub const MAX_BATCH: usize = 8;
pub const CHANNELS: usize = 10240;
pub const HISTORY: usize = 9;
pub const HISTORY_WORDS: usize = CHANNELS * HISTORY;
const CONTEXT: &str = "checked PLE chunk transition";
type Result<T> = std::result::Result<T, OpsError>;

/// Validates the exact existing authoritative PLE owner and charges actual
/// capacities, including spare host allocation. Returns no mutable state.
pub fn owner_capacity_bytes(conv: &CausalConvState, tokens: &PleTokenHistory) -> Result<usize> {
    if conv.channels != CHANNELS
        || conv.kernel != 4
        || conv.dilation != 3
        || conv.history.len() != HISTORY_WORDS
        || tokens.tokens.len() != 2
        || tokens.tokens.iter().any(|&t| !(-1..248320).contains(&t))
        || conv.history.iter().any(|x| !x.is_finite())
    {
        return Err(OpsError::InvalidShape(CONTEXT));
    }
    conv.history
        .capacity()
        .checked_mul(4)
        .and_then(|n| {
            tokens
                .tokens
                .capacity()
                .checked_mul(4)
                .and_then(|t| n.checked_add(t))
        })
        .ok_or(OpsError::CapacityExceeded(CONTEXT))
}

struct Candidate(Vec<f32>);
impl Drop for Candidate {
    fn drop(&mut self) {
        clear_f32_capacity(&mut self.0);
    }
}
fn candidate(old: &[f32], byte_budget: usize) -> Result<Candidate> {
    // Guard is established before allocation/copy; no growth with private data.
    let mut result = Candidate(Vec::new());
    if HISTORY_WORDS.checked_mul(4).is_none_or(|n| n > byte_budget) {
        return Err(OpsError::CapacityExceeded(CONTEXT));
    }
    reserve_candidate(&mut result.0, |v| {
        v.try_reserve_exact(HISTORY_WORDS)
            .map_err(|_| OpsError::AllocationFailed(CONTEXT))
    })?;
    if result
        .0
        .capacity()
        .checked_mul(4)
        .is_none_or(|n| n > byte_budget)
    {
        return Err(OpsError::CapacityExceeded(CONTEXT));
    }
    result.0.extend_from_slice(old);
    Ok(result)
}
fn reserve_candidate(
    v: &mut Vec<f32>,
    reserve: impl FnOnce(&mut Vec<f32>) -> Result<()>,
) -> Result<()> {
    // Called only while empty and guarded. An allocation failure must not see
    // a copied prefix or leave a growing private allocation outside its guard.
    if !v.is_empty() {
        return Err(OpsError::InvalidParameter(CONTEXT));
    }
    reserve(v)
}
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(OpsError::InvalidParameter(CONTEXT))
    } else {
        Ok(())
    }
}

/// Opaque borrowing ticket. Mutable ownership pins BOTH actual committed CPU
/// histories until commit or abandonment. Preparation leaves them unchanged;
/// Drop wipes candidate and discarded full capacities on every path.
pub struct PleChunkCommit<'a> {
    conv: &'a mut CausalConvState,
    tokens: &'a mut PleTokenHistory,
    history: Candidate,
    next_tokens: Zeroizing<[i32; 2]>,
}
impl PleChunkCommit<'_> {
    /// Final cancellation preflight for the all-family coordinator, before
    /// ANY swap. The caller must preflight all other state owners too.
    pub fn preflight(&self, cancel: &AtomicBool) -> Result<()> {
        cancelled(cancel)
    }
    /// Allocation-free and infallible after all-family preflight. The external
    /// completed-fence/audit/health precondition is NOT proved by this method.
    pub fn commit(mut self) {
        std::mem::swap(&mut self.conv.history, &mut self.history.0);
        clear_i32_capacity(&mut self.tokens.tokens);
        self.tokens.tokens.copy_from_slice(&*self.next_tokens);
    }
}

/// Factory accepts only an exact ordered extension of the actual existing
/// normalized/token histories. OLD and NEXT must match bit for bit (signed
/// zeros included); finite normalized rows alone cannot import arbitrary state.
/// Diagnostics here are logical exact spans; the device caller also validates
/// full owned extents, inactive rows/padding and all kernel/global statuses.
#[allow(clippy::too_many_arguments)]
pub fn prepare<'a>(
    conv: &'a mut CausalConvState,
    history: &'a mut PleTokenHistory,
    config: &PleHashConfig,
    old_tokens: &[i32; 2],
    tokens: &[u32],
    old: &[f32],
    normalized: &[f32],
    next: &[f32],
    byte_budget: usize,
    cancel: &AtomicBool,
) -> Result<PleChunkCommit<'a>> {
    cancelled(cancel)?;
    let owned = owner_capacity_bytes(conv, history)?;
    if !(1..=MAX_BATCH).contains(&tokens.len())
        || config.history_len() != 2
        || config.heads_per_ngram != 8
        || config.boundary_token != 248044
        || config.token_vocab_size != 248320
        || history.tokens != old_tokens
        || old.len() != HISTORY_WORDS
        || next.len() != HISTORY_WORDS
        || normalized.len() != MAX_BATCH * CHANNELS
        || tokens.iter().any(|&t| t >= config.token_vocab_size)
        || !bits_equal(old, &conv.history)
        || old
            .iter()
            .chain(next)
            .chain(normalized)
            .any(|x| !x.is_finite())
        || normalized[tokens.len() * CHANNELS..]
            .iter()
            .any(|x| x.to_bits() != 0)
    {
        return Err(OpsError::DimensionMismatch(CONTEXT));
    }
    let available = byte_budget
        .checked_sub(owned)
        .ok_or(OpsError::CapacityExceeded(CONTEXT))?;
    let mut candidate = candidate(&conv.history, available)?;
    let mut next_tokens = Zeroizing::new(*old_tokens);
    for (t, &token) in tokens.iter().enumerate() {
        cancelled(cancel)?;
        for c in 0..CHANNELS {
            let row = &mut candidate.0[c * HISTORY..(c + 1) * HISTORY];
            row.copy_within(1.., 0);
            row[HISTORY - 1] = normalized[t * CHANNELS + c];
        }
        next_tokens[0] = next_tokens[1];
        next_tokens[1] = token as i32;
    }
    if !bits_equal(&candidate.0, next) {
        return Err(OpsError::InvalidParameter(CONTEXT));
    }
    cancelled(cancel)?;
    Ok(PleChunkCommit {
        conv,
        tokens: history,
        history: candidate,
        next_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> (CausalConvState, PleTokenHistory, PleHashConfig) {
        let hash = PleHashConfig::qwen38();
        let mut conv = CausalConvState::new(CHANNELS, 4, 3).unwrap();
        for (i, x) in conv.history.iter_mut().enumerate() {
            *x = (i % 103) as f32 / 109.;
        }
        let mut history = PleTokenHistory::new(&hash);
        hash.rows_and_advance(17, &mut history).unwrap();
        hash.rows_and_advance(248046, &mut history).unwrap();
        (conv, history, hash)
    }
    fn inputs(count: usize) -> Zeroizing<Vec<f32>> {
        let mut v = Zeroizing::new(vec![0.; MAX_BATCH * CHANNELS]);
        for (i, x) in v[..count * CHANNELS].iter_mut().enumerate() {
            *x = if i % 19 == 0 {
                -0.
            } else {
                (i % 79) as f32 / -83.
            };
        }
        v
    }
    fn next(old: &[f32], v: &[f32], count: usize) -> Zeroizing<Vec<f32>> {
        let mut next = Zeroizing::new(old.to_vec());
        for t in 0..count {
            for c in 0..CHANNELS {
                let row = &mut next[c * HISTORY..(c + 1) * HISTORY];
                row.copy_within(1.., 0);
                row[HISTORY - 1] = v[t * CHANNELS + c];
            }
        }
        next
    }
    #[test]
    fn large_prompt_tail_matches_scalar_history_across_the_max8_boundary() {
        for count in [1usize, 8, 9, 32, 256, 257, 1024, 2048] {
            let (mut conv, mut history, hash) = state();
            let mut reference = CausalConvState::new(CHANNELS, 4, 3).unwrap();
            reference.history.copy_from_slice(conv.history());
            let old = Zeroizing::new(conv.history().to_vec());
            let before = Zeroizing::new([history.tokens[0], history.tokens[1]]);
            let ids = Zeroizing::new((0..count as u32).collect::<Vec<_>>());
            let values = Zeroizing::new(
                (0..count * CHANNELS)
                    .map(|i| {
                        if i % 19 == 0 {
                            -0.0
                        } else {
                            (i % 79) as f32 / -83.0
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            let weights = Zeroizing::new(vec![0.25; CHANNELS * 4]);
            for row in values.chunks_exact(CHANNELS) {
                reference.step(row, &weights).unwrap().zeroize();
            }
            let tail = &values[(count - count.min(HISTORY)) * CHANNELS..];
            // Dropping a complete proposal cannot publish either history.
            drop(
                prepare_prompt(
                    &mut conv,
                    &mut history,
                    &hash,
                    &before,
                    &ids,
                    &old,
                    tail,
                    reference.history(),
                    1024 * 1024,
                    &AtomicBool::new(false),
                )
                .unwrap(),
            );
            assert!(bits_equal(conv.history(), &old));
            assert_eq!(history.tokens(), &*before);
            let ticket = prepare_prompt(
                &mut conv,
                &mut history,
                &hash,
                &before,
                &ids,
                &old,
                tail,
                reference.history(),
                1024 * 1024,
                &AtomicBool::new(false),
            )
            .unwrap();
            ticket.commit();
            assert!(bits_equal(conv.history(), reference.history()));
            let expected = if count == 1 {
                [before[1], 0]
            } else {
                [count as i32 - 2, count as i32 - 1]
            };
            assert_eq!(history.tokens(), &expected);
            let row = Zeroizing::new(vec![0.17; CHANNELS]);
            assert_eq!(
                conv.step(&row, &weights).unwrap(),
                reference.step(&row, &weights).unwrap()
            );
        }
    }

    #[test]
    fn every_count_preserves_bits_and_original_scalar_consumes_accepted_history() {
        for count in [1, 3, 5, 8] {
            let (mut conv, mut history, hash) = state();
            let mut reference = CausalConvState::new(CHANNELS, 4, 3).unwrap();
            reference.history.copy_from_slice(conv.history());
            let old = Zeroizing::new(conv.history().to_vec());
            let before = Zeroizing::new([history.tokens[0], history.tokens[1]]);
            let v = inputs(count);
            let expected = next(&old, &v, count);
            let ids = Zeroizing::new([248044, 248046, 0, 31, 248043, 17, 22, 99]);
            let mut expected_tokens = Zeroizing::new(*before);
            for &id in &ids[..count] {
                expected_tokens[0] = expected_tokens[1];
                expected_tokens[1] = id as i32;
            }
            let ticket = prepare(
                &mut conv,
                &mut history,
                &hash,
                &before,
                &ids[..count],
                &old,
                &v,
                &expected,
                1024 * 1024,
                &AtomicBool::new(false),
            )
            .unwrap();
            ticket.preflight(&AtomicBool::new(false)).unwrap();
            ticket.commit();
            assert!(bits_equal(conv.history(), &expected));
            assert_eq!(history.tokens(), &*expected_tokens);
            // Independent original step advances its own reference from OLD.
            let weights = vec![0.25; CHANNELS * 4];
            for t in 0..count {
                reference
                    .step(&v[t * CHANNELS..(t + 1) * CHANNELS], &weights)
                    .unwrap()
                    .zeroize();
            }
            let scalar = vec![0.17; CHANNELS];
            assert_eq!(
                conv.step(&scalar, &weights).unwrap(),
                reference.step(&scalar, &weights).unwrap()
            );
            assert!(bits_equal(conv.history(), reference.history()));
        }
    }
    #[test]
    fn failure_and_abandonment_never_mutate_authoritative_owners() {
        let (mut conv, mut history, hash) = state();
        let old = Zeroizing::new(conv.history().to_vec());
        let before = Zeroizing::new([history.tokens[0], history.tokens[1]]);
        let v = inputs(3);
        let expected = next(&old, &v, 3);
        let cancel = AtomicBool::new(false);
        let ticket = prepare(
            &mut conv,
            &mut history,
            &hash,
            &before,
            &[1, 2, 3],
            &old,
            &v,
            &expected,
            1024 * 1024,
            &cancel,
        )
        .unwrap();
        drop(ticket);
        assert!(bits_equal(&old, conv.history()));
        assert_eq!(history.tokens(), &*before);
        for kind in 0..8 {
            let mut n = Zeroizing::new(expected.to_vec());
            let mut o = Zeroizing::new(old.to_vec());
            let mut x = Zeroizing::new(v.to_vec());
            match kind {
                0 => n[0] += 1.,
                1 => o[0] += 1.,
                2 => x[0] = f32::NAN,
                3 => x[3 * CHANNELS] = -0.,
                4 => n[HISTORY - 1] = 0.,
                _ => {}
            }
            let ids: &[u32] = if kind == 5 {
                &[]
            } else if kind == 6 {
                &[248320]
            } else {
                &[1, 2, 3]
            };
            let budget = if kind == 7 { 1 } else { 1024 * 1024 };
            assert!(
                prepare(
                    &mut conv,
                    &mut history,
                    &hash,
                    &before,
                    ids,
                    &o,
                    &x,
                    &n,
                    budget,
                    &cancel
                )
                .is_err()
            );
            assert!(bits_equal(&old, conv.history()));
            assert_eq!(history.tokens(), &*before);
        }
        cancel.store(true, Ordering::Release);
        assert!(
            prepare(
                &mut conv,
                &mut history,
                &hash,
                &before,
                &[1, 2, 3],
                &old,
                &v,
                &expected,
                1024 * 1024,
                &cancel
            )
            .is_err()
        );
    }
    #[test]
    fn failed_reserve_and_late_cancel_leave_actual_histories_unchanged() {
        let mut storage = Candidate(Vec::new());
        assert!(
            reserve_candidate(&mut storage.0, |v| {
                assert!(v.is_empty());
                Err(OpsError::AllocationFailed(CONTEXT))
            })
            .is_err()
        );
        assert!(storage.0.is_empty());
        let (mut conv, mut history, hash) = state();
        let old = Zeroizing::new(conv.history().to_vec());
        let before = Zeroizing::new([history.tokens[0], history.tokens[1]]);
        let v = inputs(1);
        let expected = next(&old, &v, 1);
        let cancel = AtomicBool::new(false);
        let ticket = prepare(
            &mut conv,
            &mut history,
            &hash,
            &before,
            &[17],
            &old,
            &v,
            &expected,
            1024 * 1024,
            &cancel,
        )
        .unwrap();
        cancel.store(true, Ordering::Release);
        assert!(ticket.preflight(&cancel).is_err());
        drop(ticket);
        assert!(bits_equal(conv.history(), &old));
        assert_eq!(history.tokens(), &*before);
    }
    #[test]
    fn two_native_chunks_feed_original_cpu_ple_tail_and_hash_without_import_setter() {
        let (mut conv, mut history, hash) = state();
        let mut reference = CausalConvState::new(CHANNELS, 4, 3).unwrap();
        reference.history.copy_from_slice(conv.history());
        let mut expected_tokens = PleTokenHistory::new(&hash);
        hash.rows_and_advance(17, &mut expected_tokens).unwrap();
        hash.rows_and_advance(248046, &mut expected_tokens).unwrap();
        let weights = Zeroizing::new(vec![0.125; CHANNELS * 4]);
        for count in [3, 5] {
            let old = Zeroizing::new(conv.history().to_vec());
            let before = Zeroizing::new([history.tokens[0], history.tokens[1]]);
            let v = inputs(count);
            let expected = next(&old, &v, count);
            let ids = Zeroizing::new([248044, 248046, 17, 19, 23, 0, 0, 0]);
            prepare(
                &mut conv,
                &mut history,
                &hash,
                &before,
                &ids[..count],
                &old,
                &v,
                &expected,
                1024 * 1024,
                &AtomicBool::new(false),
            )
            .unwrap()
            .commit();
            for t in 0..count {
                reference
                    .step(&v[t * CHANNELS..(t + 1) * CHANNELS], &weights)
                    .unwrap()
                    .zeroize();
                hash.rows_and_advance(ids[t] as i32, &mut expected_tokens)
                    .unwrap();
            }
        }
        let key = Zeroizing::new(vec![0.2; CHANNELS]);
        let residual = Zeroizing::new(vec![-0.3; CHANNELS]);
        let value = Zeroizing::new(vec![0.17; 2560]);
        let gamma = Zeroizing::new(vec![1.; CHANNELS]);
        let mut actual = super::super::ple_inject_projected(
            &key,
            &value,
            &residual,
            &gamma,
            &gamma,
            &gamma,
            &weights,
            &mut conv,
            super::super::GatedResidualShape::new(4, 2560).unwrap(),
            1e-6,
        )
        .unwrap();
        let mut expected = super::super::ple_inject_projected(
            &key,
            &value,
            &residual,
            &gamma,
            &gamma,
            &gamma,
            &weights,
            &mut reference,
            super::super::GatedResidualShape::new(4, 2560).unwrap(),
            1e-6,
        )
        .unwrap();
        assert!(bits_equal(&actual.residual, &expected.residual));
        assert!(bits_equal(conv.history(), reference.history()));
        assert_eq!(
            hash.rows_and_advance(31, &mut history).unwrap(),
            hash.rows_and_advance(31, &mut expected_tokens).unwrap()
        );
        for values in [
            &mut actual.residual,
            &mut actual.gates,
            &mut actual.normalized,
            &mut expected.residual,
            &mut expected.gates,
            &mut expected.normalized,
        ] {
            values.zeroize();
        }
    }
    #[test]
    #[allow(unsafe_code)]
    fn candidate_guard_clears_spare_capacity_and_exact_zero_bits() {
        let mut v = Candidate(vec![123.25; 17]);
        v.0.truncate(3);
        v.0[1] = -0.;
        clear_f32_capacity(&mut v.0);
        // Original eraser covers capacity, while readable logical portion proves
        // both normal and sign-zero private words are zeroed before release.
        assert!(v.0.iter().all(|x| x.to_bits() == 0));
        for x in v.0.spare_capacity_mut() {
            // SAFETY: every capacity slot was initialized by vec![..;17],
            // then the original eraser overwrote even truncated slots.
            assert_eq!(unsafe { x.assume_init_ref() }.to_bits(), 0);
        }
        assert!(bits_equal(&[-0.], &[-0.]));
        assert!(!bits_equal(&[-0.], &[0.]));
    }
}
