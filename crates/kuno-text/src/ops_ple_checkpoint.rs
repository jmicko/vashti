//! One actual PLE owner snapshot for the latest public target verification.
//! This is not an arbitrary state setter. The native coordinator must bind the
//! snapshot to its one completed chunk, exclude intervening execution and
//! preflight every state family before consuming the restore permit.
use super::{CausalConvState, OpsError, PleTokenHistory, clear_f32_capacity, clear_i32_capacity};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};

const CONTEXT: &str = "latest PLE owner checkpoint";
pub const HISTORY_WORDS: usize = super::ple_transition::HISTORY_WORDS;
pub const MAX_OWNED_BYTES: usize = 512 * 1024;
type Result<T> = std::result::Result<T, OpsError>;

/// Request-local snapshot of the actual original histories. No Clone/Debug,
/// serialization or mutable payload API. Object identities are compared only;
/// no saved address is dereferenced. The exclusive parent round keeps both
/// original owner objects stationary until this snapshot is consumed.
pub struct Checkpoint {
    history: Vec<f32>,
    tokens: Zeroizing<[i32; 2]>,
    owners: Zeroizing<[usize; 2]>,
}
impl Checkpoint {
    pub fn capture(
        conv: &CausalConvState,
        tokens: &PleTokenHistory,
        budget: usize,
    ) -> Result<Self> {
        super::ple_transition::owner_capacity_bytes(conv, tokens)?;
        if budget > MAX_OWNED_BYTES || HISTORY_WORDS * 4 + std::mem::size_of::<Self>() > budget {
            return Err(OpsError::CapacityExceeded(CONTEXT));
        }
        let mut result = Self {
            history: Vec::new(),
            tokens: Zeroizing::new([tokens.tokens[0], tokens.tokens[1]]),
            owners: Zeroizing::new([
                std::ptr::from_ref(conv) as usize,
                std::ptr::from_ref(tokens) as usize,
            ]),
        };
        result
            .history
            .try_reserve_exact(HISTORY_WORDS)
            .map_err(|_| OpsError::AllocationFailed(CONTEXT))?;
        // Charge actual complete capacity BEFORE copying any history.
        if result.owned_bytes()? > budget {
            return Err(OpsError::CapacityExceeded(CONTEXT));
        }
        result.history.extend_from_slice(conv.history());
        Ok(result)
    }
    pub fn owned_bytes(&self) -> Result<usize> {
        self.history
            .capacity()
            .checked_mul(4)
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(OpsError::CapacityExceeded(CONTEXT))
    }
    pub fn preflight(
        &self,
        conv: &CausalConvState,
        tokens: &PleTokenHistory,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if cancel.load(Ordering::Acquire) {
            return Err(OpsError::InvalidParameter(CONTEXT));
        }
        super::ple_transition::owner_capacity_bytes(conv, tokens)?;
        if *self.owners
            != [
                std::ptr::from_ref(conv) as usize,
                std::ptr::from_ref(tokens) as usize,
            ]
            || self.history.len() != HISTORY_WORDS
            || self.history.iter().any(|x| !x.is_finite())
            || self.tokens.iter().any(|&t| !(-1..248320).contains(&t))
            || self.owned_bytes()? > MAX_OWNED_BYTES
        {
            return Err(OpsError::InvalidParameter(CONTEXT));
        }
        Ok(())
    }
    pub fn prepare_restore<'a>(
        self,
        conv: &'a mut CausalConvState,
        tokens: &'a mut PleTokenHistory,
        cancel: &AtomicBool,
    ) -> Result<Restore<'a>> {
        self.preflight(conv, tokens, cancel)?;
        Ok(Restore {
            checkpoint: self,
            conv,
            tokens,
        })
    }
    /// Prepare the accepted prefix of the one completed chunk. Its first
    /// normalized rows are retained in the live nine-row convolution history;
    /// merge them with the captured old history instead of rerunning projections.
    pub fn prepare_accepted_prefix<'a>(
        mut self,
        conv: &'a mut CausalConvState,
        tokens: &'a mut PleTokenHistory,
        proposed: &[u32],
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<Restore<'a>> {
        self.preflight(conv, tokens, cancel)?;
        if !(2..=8).contains(&proposed.len())
            || accepted == 0
            || accepted >= proposed.len()
            || proposed.iter().any(|&t| t >= 248320)
            || conv.history.iter().any(|x| !x.is_finite())
        {
            return Err(OpsError::InvalidParameter(CONTEXT));
        }
        let mut expected = Zeroizing::new(*self.tokens);
        for &token in proposed {
            expected[0] = expected[1];
            expected[1] = token as i32;
        }
        if tokens.tokens.as_slice() != expected.as_slice() {
            return Err(OpsError::InvalidParameter(CONTEXT));
        }
        for channel in 0..10240 {
            let first = channel * 9;
            self.history.copy_within(first + accepted..first + 9, first);
            let incoming = first + 9 - proposed.len();
            self.history[first + 9 - accepted..first + 9]
                .copy_from_slice(&conv.history[incoming..incoming + accepted]);
        }
        for &token in &proposed[..accepted] {
            self.tokens[0] = self.tokens[1];
            self.tokens[1] = token as i32;
        }
        Ok(Restore {
            checkpoint: self,
            conv,
            tokens,
        })
    }
    fn wipe(&mut self) {
        clear_f32_capacity(&mut self.history);
        self.tokens.zeroize();
        self.owners.zeroize();
    }
}
impl Drop for Checkpoint {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Pins both actual owners through all-family preflight. Commit has no
/// allocation, cancellation point or fallible work; the discarded complete
/// history capacity is erased by the consumed checkpoint's destructor.
pub struct Restore<'a> {
    checkpoint: Checkpoint,
    conv: &'a mut CausalConvState,
    tokens: &'a mut PleTokenHistory,
}
impl Restore<'_> {
    pub fn preflight(&self, cancel: &AtomicBool) -> Result<()> {
        self.checkpoint.preflight(self.conv, self.tokens, cancel)
    }
    pub fn commit(mut self) {
        std::mem::swap(&mut self.conv.history, &mut self.checkpoint.history);
        clear_i32_capacity(&mut self.tokens.tokens);
        self.tokens.tokens.copy_from_slice(&*self.checkpoint.tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{PleHashConfig, ple_transition};
    pub(super) fn state() -> (CausalConvState, PleTokenHistory, PleHashConfig) {
        let config = PleHashConfig::qwen38();
        let mut conv = CausalConvState::new(10240, 4, 3).unwrap();
        for (i, x) in conv.history.iter_mut().enumerate() {
            *x = if i % 17 == 0 {
                -0.
            } else {
                (i % 73) as f32 / 83.
            };
        }
        let mut tokens = PleTokenHistory::new(&config);
        config.rows_and_advance(17, &mut tokens).unwrap();
        config.rows_and_advance(31, &mut tokens).unwrap();
        (conv, tokens, config)
    }
    #[test]
    fn restores_actual_committed_chunk_bits_and_next_hash_rows() {
        for count in [1, 2, 3] {
            let (mut conv, mut tokens, config) = state();
            let old = Zeroizing::new(conv.history().to_vec());
            let original_tokens = Zeroizing::new([tokens.tokens[0], tokens.tokens[1]]);
            let expected_hash = config.rows(42, &tokens).unwrap();
            let checkpoint = Checkpoint::capture(&conv, &tokens, MAX_OWNED_BYTES).unwrap();
            let mut normalized = Zeroizing::new(vec![0.; 8 * 10240]);
            for (i, x) in normalized[..count * 10240].iter_mut().enumerate() {
                *x = (i % 79) as f32 / -89.;
            }
            let mut next = Zeroizing::new(old.to_vec());
            for row in 0..count {
                for c in 0..10240 {
                    let h = &mut next[c * 9..c * 9 + 9];
                    h.copy_within(1.., 0);
                    h[8] = normalized[row * 10240 + c];
                }
            }
            ple_transition::prepare(
                &mut conv,
                &mut tokens,
                &config,
                &original_tokens,
                &[0, 1, 2][..count],
                &old,
                &normalized,
                &next,
                2 * 1024 * 1024,
                &AtomicBool::new(false),
            )
            .unwrap()
            .commit();
            assert_ne!(tokens.tokens(), &*original_tokens);
            let restore = checkpoint
                .prepare_restore(&mut conv, &mut tokens, &AtomicBool::new(false))
                .unwrap();
            restore.preflight(&AtomicBool::new(false)).unwrap();
            restore.commit();
            assert!(
                conv.history()
                    .iter()
                    .zip(&*old)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(tokens.tokens(), &*original_tokens);
            assert_eq!(config.rows(42, &tokens).unwrap(), expected_hash);
        }
    }
    #[test]
    fn equal_payload_foreign_owners_and_cancel_cannot_restore() {
        let (conv, tokens, _) = state();
        let (mut foreign, mut foreign_tokens, _) = state();
        let checkpoint = Checkpoint::capture(&conv, &tokens, MAX_OWNED_BYTES).unwrap();
        assert!(
            checkpoint
                .prepare_restore(&mut foreign, &mut foreign_tokens, &AtomicBool::new(false))
                .is_err()
        );
        let checkpoint = Checkpoint::capture(&conv, &tokens, MAX_OWNED_BYTES).unwrap();
        assert!(
            checkpoint
                .preflight(&conv, &tokens, &AtomicBool::new(true))
                .is_err()
        );
    }
    #[test]
    fn exact_capacity_budget_and_malformed_snapshot_reject_before_mutation() {
        let (conv, tokens, _) = state();
        assert!(Checkpoint::capture(&conv, &tokens, HISTORY_WORDS * 4).is_err());
        assert!(Checkpoint::capture(&conv, &tokens, MAX_OWNED_BYTES + 1).is_err());
        let mut checkpoint = Checkpoint::capture(&conv, &tokens, MAX_OWNED_BYTES).unwrap();
        assert!(checkpoint.owned_bytes().unwrap() <= MAX_OWNED_BYTES);
        checkpoint.history[7] = f32::NAN;
        assert!(
            checkpoint
                .preflight(&conv, &tokens, &AtomicBool::new(false))
                .is_err()
        );
    }
    #[test]
    fn wipe_covers_complete_spare_capacity_and_identity() {
        let (conv, tokens, _) = state();
        let mut checkpoint = Checkpoint::capture(&conv, &tokens, MAX_OWNED_BYTES).unwrap();
        checkpoint.history.try_reserve_exact(64).unwrap();
        let cap = checkpoint.history.capacity();
        checkpoint.history.resize(cap, 17.);
        checkpoint.wipe();
        assert!(checkpoint.history.iter().all(|x| x.to_bits() == 0));
        assert_eq!(*checkpoint.tokens, [0; 2]);
        assert_eq!(*checkpoint.owners, [0; 2]);
    }
}

#[cfg(test)]
mod accepted_prefix_tests {
    use super::*;
    use crate::ops::PleHashConfig;
    #[test]
    fn prefix_commit_matches_actual_sequential_owner_history_for_all_partial_windows() {
        let cancel = AtomicBool::new(false);
        for count in 2..=8 {
            for accepted in 1..count {
                let (mut live, mut live_tokens, _) = super::tests::state();
                let (mut expected, mut expected_tokens, _) = super::tests::state();
                let snapshot = Checkpoint::capture(&live, &live_tokens, MAX_OWNED_BYTES).unwrap();
                let proposed: Vec<_> = (0..count).map(|i| 100 + i as u32).collect();
                for (i, &token) in proposed.iter().enumerate() {
                    let row: Vec<_> = (0..10240).map(|j| (j + i * 37) as f32 / 101.).collect();
                    live.advance(&row);
                    live_tokens.tokens[0] = live_tokens.tokens[1];
                    live_tokens.tokens[1] = token as i32;
                    if i < accepted {
                        expected.advance(&row);
                        expected_tokens.tokens[0] = expected_tokens.tokens[1];
                        expected_tokens.tokens[1] = token as i32;
                    }
                }
                snapshot
                    .prepare_accepted_prefix(
                        &mut live,
                        &mut live_tokens,
                        &proposed,
                        accepted,
                        &cancel,
                    )
                    .unwrap()
                    .commit();
                assert_eq!(live.history(), expected.history());
                assert_eq!(live_tokens.tokens(), expected_tokens.tokens());
            }
        }
    }
    #[test]
    fn a_different_completed_token_history_cannot_authorize_prefix_mutation() {
        let mut live = CausalConvState::new(10240, 4, 3).unwrap();
        let mut tokens = PleTokenHistory::new(&PleHashConfig::qwen38());
        let snapshot = Checkpoint::capture(&live, &tokens, MAX_OWNED_BYTES).unwrap();
        assert!(
            snapshot
                .prepare_accepted_prefix(
                    &mut live,
                    &mut tokens,
                    &[100, 101],
                    1,
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(live.history().iter().all(|&v| v == 0.));
        assert_eq!(tokens.tokens(), [-1, -1]);
    }
}
