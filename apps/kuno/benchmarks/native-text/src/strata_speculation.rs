//! MIT Strata fb58e0 src/spec/{suffix_drafter,draft_policy}.cpp.
//! Native request-local lookup index and cost policy. Never persisted/logged.
//! Bounds are deliberate Kuno substitutions: fixed4096 context,8192 history
//! capacity, MAX8 verification; rejected overcapacity does not grow silently.
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};

const HISTORY: usize = 8192;
const SLOTS: usize = HISTORY * 2;
const MAX_T: usize = 8;
// The actual fixed owners, plus bounded proposal/scratch copies, fit inside
// the existing4MiB request-fixture HOST envelope (not a second hidden quote).
pub(crate) const HOST_BYTES: usize = HISTORY * 4 + SLOTS * 48 + 16 * 1024;
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub window: usize,
    pub mtp_window: usize,
    pub suffix_minimum: usize,
    pub lookup_chain: usize,
    pub chain_minimum: usize,
}
impl Config {
    pub(crate) fn native(window: usize) -> Self {
        Self {
            window,
            mtp_window: window,
            suffix_minimum: 3,
            lookup_chain: 0,
            chain_minimum: 3,
        }
    }
    pub(crate) fn validate(self) -> Result<()> {
        if !(1..=MAX_T).contains(&self.window)
            || !(1..=self.window).contains(&self.mtp_window)
            || self.suffix_minimum > 64
            || self.chain_minimum > 64
            || self.lookup_chain > 7
        {
            return Err(rejected());
        }
        Ok(())
    }
}
pub(crate) struct Proposal {
    pub tokens: Zeroizing<Vec<u32>>,
    pub matched: usize,
}
pub(crate) struct Suffix {
    history: Zeroizing<Vec<u32>>,
    // key, four most recent end positions, count. All fields wiped on drop.
    table: Zeroizing<Vec<[u64; 6]>>,
    minimum: usize,
}
fn rejected() -> Error {
    Error::InvalidFormat("native speculative policy")
}
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}
impl Suffix {
    pub(crate) fn new(minimum: usize) -> Result<Self> {
        if !(3..=64).contains(&minimum) {
            return Err(rejected());
        }
        Ok(Self {
            history: Zeroizing::new(Vec::with_capacity(HISTORY)),
            table: Zeroizing::new(vec![[0; 6]; SLOTS]),
            minimum,
        })
    }
    fn key(&self, end: usize) -> u64 {
        let a = self.history[end - 2] as u64;
        let b = self.history[end - 1] as u64;
        let c = self.history[end] as u64;
        mix(a.wrapping_mul(0x9e3779b97f4a7c15) ^ mix(b + 0x632be59bd9b4e019) ^ (c << 1)) | 1
    }
    fn slot(&self, key: u64, insert: bool) -> Option<usize> {
        for p in 0..SLOTS {
            let i = (key as usize).wrapping_add(p) & (SLOTS - 1);
            if self.table[i][0] == key || insert && self.table[i][0] == 0 {
                return Some(i);
            }
            if self.table[i][0] == 0 {
                return None;
            }
        }
        None
    }
    pub(crate) fn append(&mut self, tokens: &[u32]) -> Result<()> {
        if self
            .history
            .len()
            .checked_add(tokens.len())
            .is_none_or(|n| n > HISTORY)
            || tokens.iter().any(|&t| t >= 248320)
        {
            return Err(rejected());
        }
        for &token in tokens {
            self.history.push(token);
            let end = self.history.len() - 1;
            if end < 2 {
                continue;
            }
            let key = self.key(end);
            if let Some(i) = self.slot(key, true) {
                let slot = &mut self.table[i];
                slot[0] = key;
                slot.copy_within(1..4, 2);
                slot[1] = end as u64;
                slot[5] = (slot[5] + 1).min(4);
            }
        }
        Ok(())
    }
    pub(crate) fn propose(&self, maximum: usize) -> Option<Proposal> {
        if self.history.len() < 4 || maximum == 0 {
            return None;
        }
        let cur = self.history.len() - 1;
        let slot = &self.table[self.slot(self.key(cur), false)?];
        let (mut best_end, mut best_len) = (0, 0);
        for &position in &slot[1..1 + slot[5] as usize] {
            let position = position as usize;
            if position >= cur {
                continue;
            }
            let mut len = 0;
            while len < 64
                && len <= position
                && self.history[position - len] == self.history[cur - len]
            {
                len += 1;
            }
            if len > best_len {
                (best_end, best_len) = (position, len);
            }
        }
        if best_len < self.minimum {
            return None;
        }
        let end = (best_end + 1 + maximum.min(MAX_T - 1)).min(self.history.len());
        Some(Proposal {
            tokens: Zeroizing::new(self.history[best_end + 1..end].to_vec()),
            matched: best_len,
        })
    }
    pub(crate) fn propose_after(
        &mut self,
        extra: &[u32],
        maximum: usize,
    ) -> Result<Option<Proposal>> {
        if extra.len() > MAX_T - 1
            || extra.iter().any(|&t| t >= 248320)
            || self.history.len() + extra.len() > HISTORY
        {
            return Err(rejected());
        }
        let old = self.history.len();
        self.history.extend_from_slice(extra);
        // Deliberately do NOT index provisional drafts. The native lookup
        // searches only accepted history, including its periodic continuation.
        let result = self.propose(maximum);
        self.history[old..].zeroize();
        self.history.truncate(old);
        Ok(result)
    }
}

pub(crate) struct Policy {
    maximum: usize,
    // measured cost/count, MTP emitted/count, last observed round
    by_t: Zeroizing<[[f64; 5]; 9]>,
    // lookup accepted/rejected, chained accepted/rejected
    by_match: Zeroizing<[[f64; 4]; 4]>,
    rounds: f64,
}
impl Policy {
    pub(crate) fn new(maximum: usize) -> Result<Self> {
        if !(1..=MAX_T).contains(&maximum) {
            return Err(rejected());
        }
        Ok(Self {
            maximum,
            by_t: Zeroizing::new([[0.; 5]; 9]),
            by_match: Zeroizing::new([[0.; 4]; 4]),
            rounds: 0.,
        })
    }
    fn bucket(m: usize) -> usize {
        if m < 6 {
            0
        } else if m < 12 {
            1
        } else if m < 24 {
            2
        } else {
            3
        }
    }
    fn rate(&self, matched: usize, chain: bool) -> f64 {
        let b = Self::bucket(matched);
        let (offset, priors) = if chain {
            (2, [0.45, 0.65, 0.8, 0.9])
        } else {
            (0, [0.75, 0.88, 0.93, 0.96])
        };
        let counts = self.by_match[b];
        (counts[offset] + 4. * priors[b]) / (counts[offset] + counts[offset + 1] + 4.)
    }
    fn cost(&self, t: usize) -> f64 {
        const SHAPE: [f64; 9] = [0., 1., 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6];
        if self.by_t[t][1] > 0. {
            return self.by_t[t][0];
        }
        let (mut num, mut den) = (0., 0.);
        for (u, shape) in SHAPE.iter().enumerate().skip(1) {
            let row = self.by_t[u];
            if row[1] > 0. {
                let w = row[1].min(20.);
                num += w * row[0] * SHAPE[t] / shape;
                den += w;
            }
        }
        if den > 0. { num / den } else { SHAPE[t] }
    }
    fn stale(&self, t: usize) -> bool {
        self.by_t[t][1] > 0. && self.rounds - self.by_t[t][4] > 300.
    }
    fn mtp(&self, t: usize) -> f64 {
        if self.by_t[t][3] > 0. {
            self.by_t[t][2]
        } else {
            1. + 0.7 * (t - 1) as f64
        }
    }
    pub(crate) fn choose(&self, mtp: usize, proposal: &Proposal) -> Option<usize> {
        let t = mtp.clamp(1, self.maximum);
        let full = proposal.tokens.len().min(self.maximum - 1) + 1;
        if full == 1 {
            return None;
        }
        let base = self.mtp(t) / self.cost(t);
        let q = self.rate(proposal.matched, false);
        let (mut expected, mut qi, mut best, mut selected) = (1., 1., 0., 0);
        for size in 2..=full {
            qi *= q;
            expected += qi;
            let rate = expected / self.cost(size);
            if rate > best {
                (best, selected) = (rate, size);
            }
        }
        if selected > 0 && best > base * 1.03 {
            if selected < full && q >= 0.85 && self.stale(full) {
                selected = full;
            }
            return Some(selected);
        }
        if full > t && (self.by_t[full][1] < 3. || self.stale(full)) && q >= 0.85 {
            Some(full)
        } else {
            None
        }
    }
    pub(crate) fn chain(&self, mtp: usize, probability: f64, proposal: &Proposal) -> usize {
        let t = mtp.clamp(1, self.maximum);
        let maximum = proposal.tokens.len().min(self.maximum - t);
        if maximum == 0 {
            return 0;
        }
        let p = probability.clamp(0., 1.);
        let q = self.rate(proposal.matched, true);
        let base = self.mtp(t) / self.cost(t);
        let (mut gain, mut qi, mut best, mut selected) = (0., 1., 0., 0);
        for k in 1..=maximum {
            qi *= q;
            gain += qi;
            let rate = (self.mtp(t) + p * gain) / self.cost(t + k);
            if rate > best {
                (best, selected) = (rate, k);
            }
        }
        if selected > 0 && best > base * 1.03 {
            return selected;
        }
        if (self.by_t[t + maximum][1] < 3. || self.stale(t + maximum)) && p * q >= 0.6 {
            maximum
        } else {
            0
        }
    }
    fn observe_cost(&mut self, t: usize, ns: u64) {
        let ms = ns as f64 / 1e6;
        let row = &mut self.by_t[t];
        if ms > 0. {
            row[0] = if row[1] > 0. {
                0.9 * row[0] + 0.1 * ms
            } else {
                ms
            };
            row[1] += 1.;
            row[4] = self.rounds;
        }
        self.rounds += 1.;
    }
    fn observe_mtp(&mut self, t: usize, accepted: usize) {
        let row = &mut self.by_t[t];
        let got = (accepted + 1) as f64;
        row[2] = if row[3] > 0. {
            0.95 * row[2] + 0.05 * got
        } else {
            got
        };
        row[3] += 1.;
    }
    // Accepted means accepted drafts, excluding the always-confirmed input.
    pub(crate) fn observe(
        &mut self,
        lookup: bool,
        t: usize,
        accepted: usize,
        matched: usize,
        ns: u64,
    ) -> Result<()> {
        if t == 0 || t > self.maximum || accepted >= t {
            return Err(rejected());
        }
        self.observe_cost(t, ns);
        if lookup {
            let b = Self::bucket(matched);
            self.by_match[b][0] = 0.97 * self.by_match[b][0] + accepted as f64;
            self.by_match[b][1] =
                0.97 * self.by_match[b][1] + if accepted < t - 1 { 1. } else { 0. };
        } else {
            self.observe_mtp(t, accepted);
        }
        Ok(())
    }
    pub(crate) fn observe_chain(
        &mut self,
        t: usize,
        count: usize,
        accepted: usize,
        matched: usize,
        ns: u64,
    ) -> Result<()> {
        if t == 0 || count == 0 || t + count > self.maximum || accepted >= t + count {
            return Err(rejected());
        }
        self.observe_cost(t + count, ns);
        self.observe_mtp(t, accepted.min(t - 1));
        if accepted >= t - 1 {
            let b = Self::bucket(matched);
            let ok = accepted - (t - 1);
            self.by_match[b][2] = 0.97 * self.by_match[b][2] + ok as f64;
            self.by_match[b][3] = 0.97 * self.by_match[b][3] + if ok < count { 1. } else { 0. };
        }
        Ok(())
    }
}
impl Drop for Policy {
    fn drop(&mut self) {
        self.rounds.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Translated upstream policy/suffix fixtures, not timings or model oracles.
    fn cost(t: usize) -> u64 {
        (19_000_000. + 10_500_000. * (t - 1) as f64) as u64
    }
    fn proposal(count: usize, matched: usize) -> Proposal {
        Proposal {
            tokens: Zeroizing::new(vec![1; count]),
            matched,
        }
    }
    #[test]
    fn native_speculation_suffix_matches_upstream_repeat_longest_periodic_and_after() {
        for (input, expected, matched) in [
            (
                vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10, 11, 12],
                vec![13, 14, 15, 16, 17],
                3,
            ),
            (
                vec![
                    7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3,
                ],
                vec![100, 101],
                5,
            ),
            (vec![1, 2, 3, 1, 2, 3, 1, 2, 3], vec![1, 2, 3], 6),
        ] {
            let mut d = Suffix::new(3).unwrap();
            d.append(&input).unwrap();
            let p = d.propose(expected.len()).unwrap();
            assert_eq!(&*p.tokens, &expected);
            assert_eq!(p.matched, matched);
        }
        let mut d = Suffix::new(3).unwrap();
        let original = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 1, 2];
        d.append(&original).unwrap();
        let p = d.propose_after(&[3, 4], 3).unwrap().unwrap();
        assert_eq!(&*p.tokens, &[5, 6, 7]);
        assert!(p.matched >= 4);
        assert_eq!(&*d.history, &original);
        assert!(d.propose(3).is_none());
        assert!(d.propose_after(&[99, 98, 97], 3).unwrap().is_none());
        // Those3 cells were initialized above and explicitly scrubbed before
        // truncate; Zeroizing<Vec> alone would not wipe a truncated tail.
        let tail = unsafe { std::slice::from_raw_parts(d.history.as_ptr().add(original.len()), 3) };
        assert_eq!(tail, &[0; 3]);
    }
    #[test]
    fn native_speculation_suffix_keeps_exactly_four_indexed_occurrences_and_bounds() {
        let mut d = Suffix::new(3).unwrap();
        d.append(&[1, 2, 3, 7, 1, 2, 3, 8, 1, 2, 3, 9, 1, 2, 3, 10, 1, 2, 3])
            .unwrap();
        let slot = d.table[d.slot(d.key(18), false).unwrap()];
        assert_eq!(&slot[1..], &[18, 14, 10, 6, 4]);
        assert_eq!(d.propose(1).unwrap().tokens[0], 10);
        let old = d.history.len();
        assert!(d.append(&[248320]).is_err());
        assert_eq!(d.history.len(), old);
        assert!(d.append(&vec![1; HISTORY]).is_err());
        assert_eq!(d.history.len(), old);
        assert!(HOST_BYTES + 256 * 4 < crate::daslab_language_probe::FIXTURE_HOST_BYTES);
    }
    #[test]
    fn native_speculation_policy_separates_rejected_and_accepted_match_buckets() {
        let mut p = Policy::new(6).unwrap();
        for _ in 0..50 {
            p.observe(false, 4, 2, 0, cost(4)).unwrap();
        }
        assert!(p.choose(4, &proposal(0, 0)).is_none());
        for t in 2..=6 {
            p.observe(false, t, 0, 0, cost(t)).unwrap();
        }
        for _ in 0..40 {
            p.observe(true, 6, 0, 4, cost(6)).unwrap();
        }
        assert!(p.rate(4, false) < 0.15);
        assert!(p.choose(4, &proposal(5, 4)).is_none());
        for _ in 0..40 {
            p.observe(true, 6, 5, 30, cost(6)).unwrap();
        }
        assert!(p.rate(30, false) > 0.9);
        assert_eq!(p.choose(4, &proposal(5, 30)), Some(6));
        assert!(p.choose(4, &proposal(5, 4)).is_none());
        assert!(p.choose(4, &proposal(20, 30)).unwrap_or(4) <= 6);
    }
    #[test]
    fn native_speculation_policy_probes_unknown_cost_and_retries_stale_cost() {
        let mut p = Policy::new(6).unwrap();
        for _ in 0..50 {
            p.observe(false, 4, 3, 0, cost(4)).unwrap();
        }
        assert_eq!(p.choose(4, &proposal(5, 40)), Some(6));
        for _ in 0..3 {
            p.observe(true, 6, 5, 40, 3 * cost(6)).unwrap();
        }
        assert!(p.choose(4, &proposal(5, 40)).is_none());
        for t in 2..=6 {
            for _ in 0..4 {
                p.observe(false, t, t - 1, 0, cost(t)).unwrap();
            }
        }
        for _ in 0..6 {
            p.observe(true, 6, 5, 40, 4 * cost(6)).unwrap();
        }
        for _ in 0..30 {
            p.observe(true, 4, 3, 40, cost(4)).unwrap();
        }
        let mut saw_wide = false;
        for _ in 0..400 {
            let pick = p.choose(4, &proposal(5, 40));
            if pick == Some(6) {
                saw_wide = true;
                break;
            }
            let t = pick.unwrap_or(4);
            p.observe(pick.is_some(), t, t - 1, 40, cost(t)).unwrap();
        }
        assert!(saw_wide);
    }
    #[test]
    fn native_speculation_chain_observes_only_reached_lookup_suffix() {
        let cheap = |t: usize| 40_000_000 + 2_000_000 * (t - 1) as u64;
        let mut p = Policy::new(8).unwrap();
        for t in 2..=8 {
            for _ in 0..5 {
                p.observe(false, t, 1, 0, cheap(t)).unwrap();
            }
        }
        for _ in 0..30 {
            p.observe(false, 4, 2, 0, cheap(4)).unwrap();
        }
        for _ in 0..30 {
            p.observe_chain(4, 3, 6, 12, cheap(7)).unwrap();
        }
        assert_eq!(p.chain(4, 0.9, &proposal(3, 12)), 3);
        let mut q = Policy::new(8).unwrap();
        for t in 2..=8 {
            for _ in 0..5 {
                q.observe(false, t, 1, 0, cost(t)).unwrap();
            }
        }
        for _ in 0..30 {
            q.observe_chain(4, 3, 3, 4, cost(7)).unwrap();
        }
        assert_eq!(q.chain(4, 0.9, &proposal(3, 4)), 0);
        let before = *q.by_match;
        q.observe_chain(4, 3, 1, 4, cost(7)).unwrap();
        assert_eq!(*q.by_match, before);
        assert_eq!(q.chain(8, 1., &proposal(3, 40)), 0);
        assert!(q.observe(false, 4, 4, 0, 1).is_err());
        assert!(q.observe_chain(4, 5, 0, 0, 1).is_err());
    }
}
