//! Latest completed public target chunk only: restore actual old state, then
//! recompute its confirmed/accepted input prefix with the ORIGINAL graph.
//! The parent must own exclusive mutable model/executor borrows for this entire
//! round, admit HOST/RUNTIME below before capture, and drop target output leases
//! before restoration. No ordinary retry, poisoned ledger repair or API path.
//! Mechanism reference: Strata fb58e0d verify.cpp1796–1823 retains original GDN
//! inputs/state and repairs accepted recurrence/indexer/PLE. Our first version
//! instead reuses original old/next GDN roots and recomputes a rejected prefix.
use crate::{
    resident_gdn::batch::LatestGdnCheckpoint, resident_model::ple_batch::LatestPleCheckpoint,
    resident_qsa::batch::LatestQsaCheckpoint,
};
use kuno_text::{Error, Result, model::GraphBindingId};
use zeroize::{Zeroize, Zeroizing};

pub(crate) const HOST_INCREMENT: usize = 1024 * 1024;
pub(crate) const RUNTIME_INCREMENT: u64 = crate::resident_qsa::batch::LATEST_QSA_RUNTIME_INFO;
const LAYERS: usize = 48;
const MAX_ROWS: usize = crate::strata_port::verify_capacity();
fn error() -> Error {
    Error::InvalidFormat("native latest target chunk checkpoint")
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Captured,
    Completed,
    Consumed,
}
struct RoundState {
    geometry: Zeroizing<[usize; 2]>,
    phase: Phase,
}
impl RoundState {
    #[cfg(test)]
    fn new(start: usize, count: usize) -> Result<Self> {
        Self::new_selected(start, count, false)
    }
    fn new_selected(start: usize, count: usize, four: bool) -> Result<Self> {
        if !(1..=if four { MAX_ROWS } else { 3 }).contains(&count)
            || start.checked_add(count).is_none_or(|end| end > 4096)
        {
            return Err(error());
        }
        Ok(Self {
            geometry: Zeroizing::new([start, count]),
            phase: Phase::Captured,
        })
    }
    fn completed(&mut self, cursor: usize) -> Result<()> {
        if self.phase != Phase::Captured || cursor != self.geometry[0] + self.geometry[1] {
            return Err(error());
        }
        self.phase = Phase::Completed;
        Ok(())
    }
    fn consume(&mut self, accepted: usize, repair: bool) -> Result<()> {
        if self.phase != Phase::Completed
            || accepted == 0
            || accepted > self.geometry[1]
            || repair != (accepted < self.geometry[1])
        {
            return Err(error());
        }
        self.phase = Phase::Consumed;
        Ok(())
    }
}
impl Drop for RoundState {
    fn drop(&mut self) {
        self.geometry.zeroize();
        self.phase = Phase::Consumed;
    }
}

/// Owns only real per-family captures. Failed/abandoned tickets poison the
/// shared sequence; the parent must terminally close model/executor on failure.
pub(crate) struct LatestTargetChunk {
    state: RoundState,
    binding: Option<GraphBindingId>,
    tokens: Zeroizing<[u32; MAX_ROWS]>,
    gdn: Vec<LatestGdnCheckpoint>,
    qsa: Vec<LatestQsaCheckpoint>,
    ple: Option<LatestPleCheckpoint>,
}
impl LatestTargetChunk {
    #[allow(dead_code)] // Original count3 compatibility entry remains available.
    pub(crate) fn new(
        binding: GraphBindingId,
        start: usize,
        tokens: &[u32],
        gdn: Vec<LatestGdnCheckpoint>,
        qsa: Vec<LatestQsaCheckpoint>,
        ple: LatestPleCheckpoint,
    ) -> Result<Self> {
        Self::new_selected(binding, start, tokens, gdn, qsa, ple, false)
    }
    pub(crate) fn new_selected(
        binding: GraphBindingId,
        start: usize,
        tokens: &[u32],
        gdn: Vec<LatestGdnCheckpoint>,
        qsa: Vec<LatestQsaCheckpoint>,
        ple: LatestPleCheckpoint,
        four: bool,
    ) -> Result<Self> {
        let state = RoundState::new_selected(start, tokens.len(), four)?;
        if tokens.iter().any(|&t| t >= 248320)
            || gdn.len() != 36
            || qsa.len() != 12
            || gdn.capacity() != 36
            || qsa.capacity() != 12
            || ple.start() != start
            || ple.count() != tokens.len()
            || ple.binding() != Some(binding)
        {
            return Err(error());
        }
        let mut seen = 0u64;
        for (layer, begin, count, id, is_qsa) in gdn
            .iter()
            .map(|s| (s.layer(), s.start(), s.count(), s.binding(), false))
            .chain(
                qsa.iter()
                    .map(|s| (s.layer(), s.start(), s.count(), s.binding(), true)),
            )
        {
            if layer >= LAYERS
                || (layer % 4 == 3) != is_qsa
                || seen & (1u64 << layer) != 0
                || begin != start
                || count != tokens.len()
                || id != Some(binding)
            {
                return Err(error());
            }
            seen |= 1u64 << layer;
        }
        if seen != (1u64 << LAYERS) - 1 {
            return Err(error());
        }
        let mut packet = Zeroizing::new([0; MAX_ROWS]);
        packet[..tokens.len()].copy_from_slice(tokens);
        let result = Self {
            state,
            binding: Some(binding),
            tokens: packet,
            gdn,
            qsa,
            ple: Some(ple),
        };
        if result.owned_bytes()? > HOST_INCREMENT {
            return Err(error());
        }
        Ok(result)
    }
    pub(crate) fn owned_bytes(&self) -> Result<usize> {
        self.gdn
            .capacity()
            .checked_mul(std::mem::size_of::<LatestGdnCheckpoint>())
            .and_then(|n| {
                self.qsa
                    .capacity()
                    .checked_mul(std::mem::size_of::<LatestQsaCheckpoint>())
                    .and_then(|q| n.checked_add(q))
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| {
                self.ple
                    .as_ref()
                    .and_then(|p| p.owned_bytes().ok())
                    .and_then(|p| n.checked_add(p))
            })
            .ok_or(Error::ArithmeticOverflow)
    }
    /// External precondition: actual completed_target_rows_cursor/native
    /// boundary, all original proofs, health and unchanged binding passed.
    /// Output start/count getters never substitute for that completed proof.
    pub(crate) fn mark_completed(&mut self, binding: GraphBindingId, cursor: usize) -> Result<()> {
        if self.binding != Some(binding) {
            return Err(error());
        }
        self.state.completed(cursor)
    }
    pub(crate) fn gdn(&self) -> &[LatestGdnCheckpoint] {
        &self.gdn
    }
    pub(crate) fn qsa(&self) -> &[LatestQsaCheckpoint] {
        &self.qsa
    }
    pub(crate) fn ple(&self) -> Result<&LatestPleCheckpoint> {
        self.ple.as_ref().ok_or(error())
    }
    /// Preflight every live family ticket against its real owner before taking
    /// parts, for BOTH all-accepted consume and rejected-prefix restoration.
    pub(crate) fn into_parts(mut self, accepted: usize, repair: bool) -> Result<Parts> {
        self.state.consume(accepted, repair)?;
        Ok(Parts {
            gdn: std::mem::take(&mut self.gdn),
            qsa: std::mem::take(&mut self.qsa),
            ple: self.ple.take().ok_or(error())?,
            tokens: Zeroizing::new(*self.tokens),
            geometry: Zeroizing::new([self.state.geometry[0], self.state.geometry[1], accepted]),
        })
    }
}
impl Drop for LatestTargetChunk {
    fn drop(&mut self) {
        self.binding = None;
        self.tokens.zeroize();
    }
}
pub(crate) struct Parts {
    pub(crate) gdn: Vec<LatestGdnCheckpoint>,
    pub(crate) qsa: Vec<LatestQsaCheckpoint>,
    pub(crate) ple: LatestPleCheckpoint,
    tokens: Zeroizing<[u32; MAX_ROWS]>,
    geometry: Zeroizing<[usize; 3]>,
}
impl Parts {
    pub(crate) fn start(&self) -> usize {
        self.geometry[0]
    }
    pub(crate) fn proposed_count(&self) -> usize {
        self.geometry[1]
    }
    pub(crate) fn accepted_tokens(&self) -> &[u32] {
        &self.tokens[..self.geometry[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_latest_actual_cursor_and_consumption_are_bounded() {
        for start in [0, 1, 23, 4093] {
            for count in 1..=3 {
                let mut s = RoundState::new(start, count).unwrap();
                assert!(s.consume(1, true).is_err());
                assert!(s.completed(start).is_err());
                s.completed(start + count).unwrap();
                assert!(s.completed(start + count).is_err());
                assert!(s.consume(0, true).is_err());
                assert!(s.consume(count + 1, false).is_err());
                if count > 1 {
                    s.consume(count - 1, true).unwrap();
                } else {
                    s.consume(count, false).unwrap();
                }
                assert!(s.consume(count, false).is_err());
            }
        }
    }
    #[test]
    fn all_accept_has_no_restore_and_partial_accept_requires_restore() {
        for count in 1..=3 {
            for accepted in 1..=count {
                let mut s = RoundState::new(17, count).unwrap();
                s.completed(17 + count).unwrap();
                assert!(s.consume(accepted, accepted == count).is_err());
                s.consume(accepted, accepted < count).unwrap();
            }
        }
    }
    #[test]
    fn four_row_checkpoint_requires_real_completed_cursor_and_exact_rejected_prefix() {
        assert!(RoundState::new(17, 4).is_err());
        for accepted in 1..=4 {
            let mut state = RoundState::new_selected(17, 4, true).unwrap();
            assert!(state.consume(accepted, accepted < 4).is_err());
            assert!(state.completed(20).is_err());
            state.completed(21).unwrap();
            assert!(state.consume(accepted, accepted == 4).is_err());
            state.consume(accepted, accepted < 4).unwrap();
            assert!(state.consume(accepted, accepted < 4).is_err());
        }
        assert!(RoundState::new_selected(4093, 4, true).is_err());
        assert!(RoundState::new_selected(0, 5, true).is_err());
        assert!(ops_snapshot_peak() < HOST_INCREMENT);
    }
    #[test]
    fn malformed_extents_and_overflow_cannot_make_a_checkpoint() {
        for (start, count) in [(0, 0), (0, 4), (4096, 1), (4095, 2), (usize::MAX, 1)] {
            assert!(RoundState::new(start, count).is_err());
        }
        assert_eq!(RUNTIME_INCREMENT, 49_152);
        assert_eq!(HOST_INCREMENT, 1_048_576);
        assert!(ops_snapshot_peak() < HOST_INCREMENT);
    }
    fn ops_snapshot_peak() -> usize {
        kuno_text::ops::ple_checkpoint::MAX_OWNED_BYTES
            + 36 * std::mem::size_of::<LatestGdnCheckpoint>()
            + 12 * std::mem::size_of::<LatestQsaCheckpoint>()
            + std::mem::size_of::<LatestTargetChunk>()
    }
}
