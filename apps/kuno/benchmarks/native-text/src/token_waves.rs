//! Guarded, host-only contiguous token-wave planning for the fixed native lab.
//!
//! Each wave exposes its COMPLETE protected union before any future bank adapter
//! chooses victims. It never mutates a bank, dispatches work, commits state,
//! normalizes probabilities, or drops a route to make a union fit. Token/rank
//! order stays separate from union order. All private storage is fixed-size and
//! guarded before filling; there are no heap allocations or reallocations here.

use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};

pub const MAX_BATCH: usize = 8;
pub const TOP: usize = 10;
pub const MAX_ROUTES: usize = MAX_BATCH * TOP;
pub const MAX_EXPERTS: usize = 512;

/// Only static categories may be formatted; this contains no private content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Configuration,
    Index,
    Expert,
    Probability,
    Duplicate,
    Incomplete,
    Unnormalized,
    Cancelled,
    Overflow,
}
pub type Result<T> = std::result::Result<T, Error>;

/// A rank's original expert and exact supplied probability bits. No Debug/Clone.
pub struct Route {
    expert: u32,
    probability: f32,
}
impl Route {
    fn empty() -> Self {
        Self {
            expert: 0,
            probability: 0.0,
        }
    }
    pub fn expert(&self) -> u32 {
        self.expert
    }
    pub fn probability(&self) -> f32 {
        self.probability
    }
}
impl Zeroize for Route {
    fn zeroize(&mut self) {
        self.expert.zeroize();
        self.probability.zeroize();
    }
}

struct InputData {
    count: usize,
    capacity: usize,
    experts: usize,
    routes: [Route; MAX_ROUTES],
    initialized: [u8; MAX_ROUTES],
}
impl Zeroize for InputData {
    fn zeroize(&mut self) {
        self.count.zeroize();
        self.capacity.zeroize();
        self.experts.zeroize();
        self.routes.zeroize();
        self.initialized.zeroize();
        #[cfg(test)]
        witness(
            0,
            self.routes.iter().all(route_zero)
                && self.initialized.iter().all(|&x| x == 0)
                && self.count == 0
                && self.capacity == 0
                && self.experts == 0,
        );
    }
}

/// Establish this owner before filling any top-ten private source packet.
pub struct Input {
    data: Zeroizing<InputData>,
}
impl Input {
    pub fn new(count: usize, capacity: usize, expert_count: usize) -> Result<Self> {
        // Check original expert bounds and all public geometry before any private
        // owner is formed. 73..95 are not admitted routed-bank capacities.
        if !(1..=MAX_BATCH).contains(&count)
            || !((TOP..=72).contains(&capacity) || capacity == 96)
            || !(capacity..=MAX_EXPERTS).contains(&expert_count)
        {
            return Err(Error::Configuration);
        }
        Ok(Self {
            data: Zeroizing::new(InputData {
                count,
                capacity,
                experts: expert_count,
                routes: std::array::from_fn(|_| Route::empty()),
                initialized: [0; MAX_ROUTES],
            }),
        })
    }
    pub fn set(&mut self, token: usize, rank: usize, expert: u32, probability: f32) -> Result<()> {
        if token >= self.data.count || rank >= TOP {
            return Err(Error::Index);
        }
        if expert as usize >= self.data.experts {
            return Err(Error::Expert);
        }
        if !probability.is_finite() || probability < 0.0 {
            return Err(Error::Probability);
        }
        let i = token
            .checked_mul(TOP)
            .and_then(|x| x.checked_add(rank))
            .ok_or(Error::Overflow)?;
        self.data.routes[i].expert = expert;
        self.data.routes[i].probability = probability;
        self.data.initialized[i] = 1;
        Ok(())
    }
    pub fn token(&self, token: usize) -> Option<&[Route]> {
        if token >= self.data.count {
            return None;
        }
        Some(&self.data.routes[token * TOP..(token + 1) * TOP])
    }
}

struct Wave {
    begin: usize,
    end: usize,
    union_begin: usize,
    union_len: usize,
}
impl Wave {
    fn empty() -> Self {
        Self {
            begin: 0,
            end: 0,
            union_begin: 0,
            union_len: 0,
        }
    }
}
impl Zeroize for Wave {
    fn zeroize(&mut self) {
        self.begin.zeroize();
        self.end.zeroize();
        self.union_begin.zeroize();
        self.union_len.zeroize();
    }
}
struct PlanData {
    count: usize,
    capacity: usize,
    wave_count: usize,
    union_count: usize,
    routes: [Route; MAX_ROUTES],
    unions: [u32; MAX_ROUTES],
    waves: [Wave; MAX_BATCH],
}
impl Zeroize for PlanData {
    fn zeroize(&mut self) {
        self.count.zeroize();
        self.capacity.zeroize();
        self.wave_count.zeroize();
        self.union_count.zeroize();
        self.routes.zeroize();
        self.unions.zeroize();
        self.waves.zeroize();
        #[cfg(test)]
        witness(
            1,
            self.routes.iter().all(route_zero)
                && self.unions.iter().all(|&x| x == 0)
                && self
                    .waves
                    .iter()
                    .all(|w| w.begin == 0 && w.end == 0 && w.union_begin == 0 && w.union_len == 0)
                && self.count == 0
                && self.capacity == 0
                && self.wave_count == 0
                && self.union_count == 0,
        );
    }
}
struct Membership([u8; MAX_EXPERTS]);
impl Zeroize for Membership {
    fn zeroize(&mut self) {
        self.0.zeroize();
        #[cfg(test)]
        witness(2, self.0.iter().all(|&x| x == 0));
    }
}

struct Cursor {
    begin: usize,
    union_begin: usize,
    union_len: usize,
}
impl Zeroize for Cursor {
    fn zeroize(&mut self) {
        self.begin.zeroize();
        self.union_begin.zeroize();
        self.union_len.zeroize();
        #[cfg(test)]
        witness(
            3,
            self.begin == 0 && self.union_begin == 0 && self.union_len == 0,
        );
    }
}

/// Immutable whole plan. All eighty rank slots and eight wave slots remain
/// owned until Drop, including inactive slots; nothing escapes by Clone/Debug.
pub struct Plan {
    data: Zeroizing<PlanData>,
}
impl Plan {
    pub fn build(input: &Input, cancel: &AtomicBool) -> Result<Self> {
        Self::build_checked(input, || cancel.load(Ordering::Acquire))
    }
    fn build_checked(input: &Input, mut cancelled: impl FnMut() -> bool) -> Result<Self> {
        let x = &input.data;
        // Whole packet acceptance precedes private plan filling. Match the
        // upstream top-ten finite/nonnegative/f64 sum tolerance contract exactly,
        // including signed zero; never clamp, sort, or renormalize supplied bits.
        for token in 0..x.count {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            if x.initialized[token * TOP..(token + 1) * TOP]
                .iter()
                .any(|&set| set != 1)
            {
                return Err(Error::Incomplete);
            }
            let row = &x.routes[token * TOP..(token + 1) * TOP];
            let mut sum = Zeroizing::new(0.0_f64);
            for (rank, route) in row.iter().enumerate() {
                if route.expert as usize >= x.experts {
                    return Err(Error::Expert);
                }
                if row[..rank].iter().any(|r| r.expert == route.expert) {
                    return Err(Error::Duplicate);
                }
                if !route.probability.is_finite() || route.probability < 0.0 {
                    return Err(Error::Probability);
                }
                *sum += f64::from(route.probability);
            }
            if (*sum - 1.0).abs() > 1e-5 {
                return Err(Error::Unnormalized);
            }
        }
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let mut data = Zeroizing::new(PlanData {
            count: x.count,
            capacity: x.capacity,
            wave_count: 0,
            union_count: 0,
            routes: std::array::from_fn(|_| Route::empty()),
            unions: [0; MAX_ROUTES],
            waves: std::array::from_fn(|_| Wave::empty()),
        });
        let mut members = Zeroizing::new(Membership([0; MAX_EXPERTS]));
        let mut cursor = Zeroizing::new(Cursor {
            begin: 0,
            union_begin: 0,
            union_len: 0,
        });
        for token in 0..x.count {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let row = &x.routes[token * TOP..(token + 1) * TOP];
            let new = row
                .iter()
                .filter(|r| members.0[r.expert as usize] == 0)
                .count();
            if cursor.union_len.checked_add(new).ok_or(Error::Overflow)? > x.capacity {
                finish_wave(
                    &mut data,
                    cursor.begin,
                    token,
                    cursor.union_begin,
                    cursor.union_len,
                )?;
                members.zeroize();
                cursor.begin = token;
                cursor.union_begin = data.union_count;
                cursor.union_len = 0;
            }
            for (rank, route) in row.iter().enumerate() {
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let i = token * TOP + rank;
                data.routes[i].expert = route.expert;
                data.routes[i].probability = route.probability;
                if members.0[route.expert as usize] == 0 {
                    if data.union_count >= MAX_ROUTES || cursor.union_len >= x.capacity {
                        return Err(Error::Overflow);
                    }
                    let at = data.union_count;
                    data.unions[at] = route.expert;
                    data.union_count += 1;
                    cursor.union_len += 1;
                    members.0[route.expert as usize] = 1;
                }
            }
        }
        finish_wave(
            &mut data,
            cursor.begin,
            x.count,
            cursor.union_begin,
            cursor.union_len,
        )?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        Ok(Self { data })
    }
    pub fn count(&self) -> usize {
        self.data.count
    }
    pub fn wave_count(&self) -> usize {
        self.data.wave_count
    }
    pub fn token(&self, token: usize) -> Option<&[Route]> {
        if token >= self.data.count {
            return None;
        }
        Some(&self.data.routes[token * TOP..(token + 1) * TOP])
    }
    pub fn wave(&self, index: usize) -> Option<WaveView<'_>> {
        if index >= self.data.wave_count {
            return None;
        }
        Some(WaveView {
            plan: self,
            wave: &self.data.waves[index],
        })
    }
}
fn finish_wave(
    data: &mut PlanData,
    begin: usize,
    end: usize,
    union_begin: usize,
    union_len: usize,
) -> Result<()> {
    if begin >= end
        || end > data.count
        || union_len < TOP
        || union_len > data.capacity
        || union_begin.checked_add(union_len).ok_or(Error::Overflow)? != data.union_count
        || data.wave_count >= MAX_BATCH
    {
        return Err(Error::Overflow);
    }
    let at = data.wave_count;
    data.waves[at] = Wave {
        begin,
        end,
        union_begin,
        union_len,
    };
    data.wave_count += 1;
    Ok(())
}
pub struct WaveView<'a> {
    plan: &'a Plan,
    wave: &'a Wave,
}
impl<'a> WaveView<'a> {
    pub fn tokens(&self) -> std::ops::Range<usize> {
        self.wave.begin..self.wave.end
    }
    /// A future bank adapter must protect this entire slice before any victims.
    pub fn protected_experts(&self) -> &'a [u32] {
        &self.plan.data.unions[self.wave.union_begin..self.wave.union_begin + self.wave.union_len]
    }
    pub fn token(&self, token: usize) -> Option<&'a [Route]> {
        if !self.tokens().contains(&token) {
            return None;
        }
        self.plan.token(token)
    }
}

#[cfg(test)]
fn route_zero(r: &Route) -> bool {
    r.expert == 0 && r.probability.to_bits() == 0
}
#[cfg(test)]
thread_local! { static WIPES: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) }; }
#[cfg(test)]
fn witness(kind: usize, zero: bool) {
    assert!(zero, "guard did not wipe its complete fixed storage");
    WIPES.with(|w| {
        let mut n = w.get();
        n[kind] += 1;
        w.set(n);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn fixture(count: usize, capacity: usize, ids: impl Fn(usize, usize) -> u32) -> Input {
        let mut x = Input::new(count, capacity, MAX_EXPERTS).unwrap();
        for token in 0..count {
            for rank in 0..TOP {
                x.set(token, rank, ids(token, rank), 0.1).unwrap();
            }
        }
        x
    }
    fn reference(x: &Input) -> Vec<(std::ops::Range<usize>, BTreeSet<u32>)> {
        let mut waves = Vec::new();
        let mut begin = 0;
        let mut union = BTreeSet::new();
        for token in 0..x.data.count {
            let next = x
                .token(token)
                .unwrap()
                .iter()
                .map(Route::expert)
                .collect::<BTreeSet<_>>();
            if union.union(&next).count() > x.data.capacity {
                waves.push((begin..token, union));
                begin = token;
                union = BTreeSet::new();
            }
            union.extend(next);
        }
        waves.push((begin..x.data.count, union));
        waves
    }
    fn verify(x: &Input) -> Plan {
        let plan = Plan::build(x, &AtomicBool::new(false)).unwrap();
        let expected = reference(x);
        assert_eq!(plan.count(), x.data.count);
        assert_eq!(plan.wave_count(), expected.len());
        let mut seen = [[0u8; TOP]; MAX_BATCH];
        let mut previous = 0;
        for (i, (tokens, expected_union)) in expected.iter().enumerate() {
            let wave = plan.wave(i).unwrap();
            assert_eq!(wave.tokens(), *tokens);
            assert_eq!(tokens.start, previous);
            previous = tokens.end;
            assert_eq!(
                wave.protected_experts()
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                *expected_union
            );
            assert_eq!(wave.protected_experts().len(), expected_union.len());
            assert!(wave.protected_experts().len() <= x.data.capacity);
            let mut appearance = Vec::new();
            for token in wave.tokens() {
                let routes = wave.token(token).unwrap();
                for (rank, route) in routes.iter().enumerate() {
                    seen[token][rank] += 1;
                    assert_eq!(route.expert(), x.token(token).unwrap()[rank].expert());
                    assert_eq!(
                        route.probability().to_bits(),
                        x.token(token).unwrap()[rank].probability().to_bits()
                    );
                    if !appearance.contains(&route.expert()) {
                        appearance.push(route.expert());
                    }
                }
            }
            assert_eq!(wave.protected_experts(), appearance);
            assert!(wave.token(tokens.end).is_none());
            if tokens.end < x.data.count {
                let next = x
                    .token(tokens.end)
                    .unwrap()
                    .iter()
                    .map(Route::expert)
                    .collect::<BTreeSet<_>>();
                assert!(expected_union.union(&next).count() > x.data.capacity);
            }
        }
        assert_eq!(previous, x.data.count);
        for (token, ranks) in seen.iter().enumerate() {
            assert!(ranks.iter().all(|&n| n == u8::from(token < x.data.count)));
        }
        assert!(plan.wave(expected.len()).is_none());
        assert!(plan.token(x.data.count).is_none());
        plan
    }

    #[test]
    fn every_route_exactly_once_all_unique_and_minimum_bank() {
        for count in [1, 3, 5, 8] {
            for capacity in [10, 11, 64, 72, 96] {
                let x = fixture(count, capacity, |t, r| (t * TOP + r) as u32);
                let p = verify(&x);
                assert_eq!(p.wave_count(), count.div_ceil(capacity / TOP));
            }
        }
        let x = fixture(8, 96, |t, r| (t * TOP + r) as u32);
        assert_eq!(verify(&x).wave(0).unwrap().protected_experts().len(), 80);
    }
    #[test]
    fn repeats_do_not_merge_token_ranks_and_complete_union_protects_later_hits() {
        for capacity in [10, 64, 72, 96] {
            let x = fixture(8, capacity, |_, r| r as u32);
            let p = verify(&x);
            assert_eq!(p.wave_count(), 1);
            assert_eq!(
                p.wave(0).unwrap().protected_experts(),
                (0..10).collect::<Vec<_>>()
            );
        }
        // A bank adapter receives ALL later-token hits before picking any victim;
        // this module has no mutation/callback through which it can choose early.
        let x = fixture(
            3,
            20,
            |t, r| if t == 0 { r as u32 } else { (10 + r) as u32 },
        );
        let p = verify(&x);
        assert_eq!(p.wave_count(), 1);
        assert_eq!(
            p.wave(0).unwrap().protected_experts(),
            (0..20).collect::<Vec<_>>()
        );
    }
    #[test]
    fn independent_set_reference_for_irregular_repeated_rank_orders() {
        for seed in 0..17 {
            for capacity in [10, 16, 23, 37, 64, 72, 96] {
                let x = fixture(8, capacity, |t, r| {
                    ((t * 7 + r * 19 + seed * 31) % 512) as u32
                });
                verify(&x);
            }
        }
    }
    #[test]
    fn preserves_signed_zero_and_original_probability_bits_without_normalizing() {
        let mut x = fixture(1, 10, |_, r| r as u32);
        for rank in 0..TOP {
            x.set(0, rank, rank as u32, if rank == 0 { -0.0 } else { 0.0 })
                .unwrap();
        }
        x.set(0, 9, 9, 1.000001).unwrap(); // valid original f64 sum tolerance.
        let p = verify(&x);
        assert_eq!(
            p.token(0).unwrap()[0].probability().to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(
            p.token(0).unwrap()[9].probability().to_bits(),
            1.000001f32.to_bits()
        );
    }
    #[test]
    fn omitted_zero_default_rank_is_not_a_supplied_route() {
        let mut x = Input::new(1, 10, 512).unwrap();
        for rank in 0..9 {
            x.set(
                0,
                rank,
                (rank + 1) as u32,
                if rank < 8 { 0.125 } else { 0.0 },
            )
            .unwrap();
        }
        // The nine explicitly supplied routes sum1 and contain no expert0.
        // A default expert0/prob0 must not fabricate the missing tenth route.
        assert!(matches!(
            Plan::build(&x, &AtomicBool::new(false)),
            Err(Error::Incomplete)
        ));
        x.set(0, 9, 0, -0.0).unwrap();
        let plan = verify(&x);
        assert_eq!(
            plan.token(0).unwrap()[9].probability().to_bits(),
            (-0.0f32).to_bits()
        );
        for token in 0..8 {
            let mut incomplete = fixture(8, 10, |_, r| r as u32);
            incomplete.data.initialized[token * TOP + 9] = 0;
            assert!(matches!(
                Plan::build(&incomplete, &AtomicBool::new(false)),
                Err(Error::Incomplete)
            ));
        }
    }

    #[test]
    fn malformed_configuration_indices_experts_and_packet_fail_before_planning() {
        WIPES.with(|w| w.set([0; 4]));
        for args in [
            (0, 10, 512),
            (9, 10, 512),
            (usize::MAX, 10, 512),
            (1, 9, 512),
            (1, 73, 512),
            (1, 95, 512),
            (1, 97, 512),
            (1, usize::MAX, 512),
            (1, 96, 95),
            (1, 10, 513),
            (1, 10, usize::MAX),
        ] {
            assert!(matches!(
                Input::new(args.0, args.1, args.2),
                Err(Error::Configuration)
            ));
        }
        WIPES.with(|w| assert_eq!(w.get(), [0; 4]));
        let mut x = fixture(1, 10, |_, r| r as u32);
        assert_eq!(x.set(usize::MAX, 0, 0, 0.1), Err(Error::Index));
        assert_eq!(x.set(0, usize::MAX, 0, 0.1), Err(Error::Index));
        assert_eq!(x.set(0, 0, 512, 0.1), Err(Error::Expert));
        assert_eq!(x.set(0, 0, u32::MAX, 0.1), Err(Error::Expert));
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01] {
            assert_eq!(x.set(0, 0, 0, invalid), Err(Error::Probability));
        }
        verify(&x); // every rejected set left the owning packet unchanged.
        x.set(0, 1, 0, 0.1).unwrap();
        assert!(matches!(
            Plan::build(&x, &AtomicBool::new(false)),
            Err(Error::Duplicate)
        ));
        x.set(0, 1, 1, 0.3).unwrap();
        assert!(matches!(
            Plan::build(&x, &AtomicBool::new(false)),
            Err(Error::Unnormalized)
        ));
        x.data.routes[1].probability = f32::NAN; // downstream defense in depth.
        assert!(matches!(
            Plan::build(&x, &AtomicBool::new(false)),
            Err(Error::Probability)
        ));
        x.data.routes[1].expert = 512;
        assert!(matches!(
            Plan::build(&x, &AtomicBool::new(false)),
            Err(Error::Expert)
        ));
    }
    #[test]
    fn cancellation_at_every_poll_never_returns_partial_plan_or_modifies_source() {
        let x = fixture(8, 10, |t, r| (t * TOP + r) as u32);
        let mut polls = 0;
        let baseline = Plan::build_checked(&x, || {
            polls += 1;
            false
        })
        .unwrap();
        assert_eq!(polls, 98);
        drop(baseline);
        for at in 1..=polls {
            let mut calls = 0;
            assert!(matches!(
                Plan::build_checked(&x, || {
                    calls += 1;
                    calls == at
                }),
                Err(Error::Cancelled)
            ));
            for token in 0..8 {
                for (rank, r) in x.token(token).unwrap().iter().enumerate() {
                    assert_eq!(r.expert(), (token * TOP + rank) as u32);
                    assert_eq!(r.probability().to_bits(), 0.1f32.to_bits());
                }
            }
        }
        assert!(matches!(
            Plan::build(&x, &AtomicBool::new(true)),
            Err(Error::Cancelled)
        ));
    }
    #[test]
    fn all_fixed_slots_wipe_on_normal_drop_partial_cancel_and_unwind() {
        WIPES.with(|w| w.set([0; 4]));
        let mut x = fixture(1, 10, |_, r| r as u32);
        // Poison unused fixed slots too: no logical-length-only wipe can pass.
        x.data.initialized[MAX_ROUTES - 1] = 1;
        x.data.routes[MAX_ROUTES - 1].expert = 511;
        x.data.routes[MAX_ROUTES - 1].probability = 0.75;
        let mut p = verify(&x);
        p.data.routes[MAX_ROUTES - 1].expert = 511;
        p.data.routes[MAX_ROUTES - 1].probability = 0.875;
        p.data.unions[MAX_ROUTES - 1] = 511;
        p.data.waves[MAX_BATCH - 1] = Wave {
            begin: 7,
            end: 8,
            union_begin: 79,
            union_len: 1,
        };
        drop(p);
        WIPES.with(|w| assert_eq!(w.get(), [0, 1, 1, 1]));
        let mut calls = 0;
        assert!(matches!(
            Plan::build_checked(&x, || {
                calls += 1;
                calls == 8
            }),
            Err(Error::Cancelled)
        ));
        WIPES.with(|w| assert_eq!(w.get(), [0, 2, 2, 2]));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut calls = 0;
            let _ = Plan::build_checked(&x, || {
                calls += 1;
                assert_ne!(calls, 8);
                false
            });
        }));
        assert!(result.is_err());
        WIPES.with(|w| assert_eq!(w.get(), [0, 3, 3, 3]));
        drop(x);
        WIPES.with(|w| assert_eq!(w.get(), [1, 3, 3, 3]));
    }
    #[test]
    fn fixed_storage_inventory_has_no_private_heap_or_growth() {
        assert_eq!(std::mem::size_of::<Route>(), 8);
        assert_eq!(
            std::mem::size_of::<Wave>(),
            4 * std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<InputData>(),
            80 * 8 + 80 + 3 * std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<PlanData>(),
            80 * 8 + 80 * 4 + (8 * 4 + 4) * std::mem::size_of::<usize>()
        );
        assert_eq!(std::mem::size_of::<Membership>(), 512);
        assert_eq!(
            std::mem::size_of::<Cursor>(),
            3 * std::mem::size_of::<usize>()
        );
        let source = include_str!("token_waves.rs");
        let production = source.split("#[cfg(test)]\nmod tests").next().unwrap();
        for forbidden in [
            "Vec<",
            "Box<",
            ".collect(",
            ".push(",
            "reserve(",
            "derive(Debug",
            "derive(Clone",
        ] {
            // Error categories may implement Clone/Debug; private owners may not.
            if forbidden.starts_with("derive(") {
                continue;
            }
            assert!(!production.contains(forbidden));
        }
    }
}
