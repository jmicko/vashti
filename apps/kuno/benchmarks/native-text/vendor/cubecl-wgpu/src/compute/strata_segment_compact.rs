//! Compact only unobserved same-queue cuts in a COMPLETE original Plan.
//! No buffer declaration, original node, physical queue or completion changes.
use super::Plan;
const MAX_SEGMENTS: usize = 32769;
// Header1 and final MAX_SEGMENTS cannot be merge-left signals. This bounded
// bitmap lives on the worker's already-reserved stack, never a new heap owner.
const WORDS: usize = 512;
const _: () = assert!(size_of::<[u64; WORDS]>() == 4096);
struct ReferenceBits([u64; WORDS]);
impl ReferenceBits {
    fn erase(&mut self) {
        for word in &mut self.0 {
            unsafe { core::ptr::write_volatile(word, 0) };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}
impl Drop for ReferenceBits {
    fn drop(&mut self) {
        self.erase();
    }
}
fn mapped(segments: &[super::Segment], original: u64) -> usize {
    segments.partition_point(|s| s.signal < original)
}
pub(super) fn compact(plan: &mut Plan) -> Result<usize, &'static str> {
    let n = plan.segments.len();
    if !(1..=MAX_SEGMENTS).contains(&n) || plan.node_segments.len() > 32768 {
        return Err("native compact original plan capacity");
    }
    // Preflight ALL original indices, queue tags and dependency identities
    // before any mutation. A header-only plan is valid but never merged.
    let mut observed = ReferenceBits([0u64; WORDS]);
    for (i, s) in plan.segments.iter().enumerate() {
        let own = s.queue();
        if s.copy && s.shared
            || s.signal != i as u64 + 1
            || s.wait != *s.waits.iter().max().unwrap()
            || s.waits[own] != 0
            || i == 0 && (own != 0 || s.wait != 0)
        {
            return Err("native compact original segment identity");
        }
        for (peer, value) in s.waits.iter().copied().enumerate() {
            if value != 0 {
                if value >= s.signal || plan.segments[value as usize - 1].queue() != peer {
                    return Err("native compact original dependency");
                }
                if value >= 2 && value < (n as u64) {
                    let bit = value as usize - 2;
                    observed.0[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
    }
    if plan.node_segments.iter().any(|&i| i == 0 || i >= n)
        || plan.last_copy
            != plan
                .segments
                .iter()
                .rev()
                .find(|s| s.queue() == 1)
                .map_or(0, |s| s.signal)
        || plan.last_shared
            != plan
                .segments
                .iter()
                .rev()
                .find(|s| s.queue() == 2)
                .map_or(0, |s| s.signal)
    {
        return Err("native compact original node or final identity");
    }
    let mut keep = 0;
    for read in 0..n {
        let merge = keep >= 2 && {
            let left = &plan.segments[keep - 1];
            let right = &plan.segments[read];
            let bit = left.signal as usize - 2;
            left.queue() == right.queue()
                && observed.0[bit / 64] & (1u64 << (bit % 64)) == 0
                && right.waits.iter().zip(left.waits).all(|(&r, l)| r <= l)
        };
        if merge {
            // Retain the last ORIGINAL signal as the endpoint key until all
            // original node/wait indices are remapped without a second array.
            plan.segments[keep - 1].signal = plan.segments[read].signal;
        } else {
            plan.segments.swap(keep, read);
            keep += 1;
        }
    }
    plan.segments.truncate(keep);
    for old in &mut plan.node_segments {
        *old = mapped(&plan.segments, *old as u64 + 1);
    }
    for i in 0..keep {
        let old = plan.segments[i].waits;
        let waits = old.map(|v| {
            if v == 0 {
                0
            } else {
                mapped(&plan.segments, v) as u64 + 1
            }
        });
        plan.segments[i].waits = waits;
        plan.segments[i].wait = *waits.iter().max().unwrap();
    }
    plan.last_copy = 0;
    plan.last_shared = 0;
    for (i, s) in plan.segments.iter_mut().enumerate() {
        s.signal = i as u64 + 1;
        if s.copy {
            plan.last_copy = s.signal;
        }
        if s.shared {
            plan.last_shared = s.signal;
        }
    }
    Ok(n - keep)
}

#[cfg(test)]
mod tests {
    use super::super::{Access, NodeAccess, Planner, Segment};
    use super::*;
    fn footprint(p: &Plan) -> Vec<(u64, u64, u64, bool)> {
        p.accesses()
            .iter()
            .map(|a| (a.buffer, a.first, a.bytes, a.write))
            .collect()
    }
    fn clocks(p: &Plan) -> Vec<[u64; 3]> {
        let mut all: Vec<[u64; 3]> = Vec::new();
        let mut previous: [Option<usize>; 3] = [None; 3];
        for (i, s) in p.segments.iter().enumerate() {
            let own = s.queue();
            let mut clock = previous[own].map_or([0; 3], |j| all[j]);
            for (peer, w) in s.waits.iter().copied().enumerate() {
                if w != 0 {
                    assert!(w < s.signal);
                    let at = w as usize - 1;
                    assert_eq!(p.segments[at].queue(), peer);
                    for q in 0..3 {
                        clock[q] = clock[q].max(all[at][q]);
                    }
                }
            }
            clock[own] = i as u64 + 1;
            all.push(clock);
            previous[own] = Some(i);
        }
        all
    }
    fn node_order(p: &Plan) -> Vec<bool> {
        let c = clocks(p);
        let mut result = Vec::new();
        for (i, &a) in p.node_segments.iter().enumerate() {
            for &b in &p.node_segments[i + 1..] {
                result.push(a == b || c[b][p.segments[a].queue()] >= a as u64 + 1);
            }
        }
        result
    }
    fn image(p: &Plan) -> (Vec<(usize, [u64; 3], u64)>, Vec<usize>, u64, u64) {
        (
            p.segments
                .iter()
                .map(|s| (s.queue(), s.waits, s.signal))
                .collect(),
            p.node_segments.clone(),
            p.last_copy,
            p.last_shared,
        )
    }
    #[test]
    fn original_real_hazards_and_complete_retained_footprints_survive_compaction() {
        let mut total = 0;
        for seed in 0..160u64 {
            let mut planner = Planner::new(96, false).unwrap();
            let mut rng = seed + 1;
            for i in 0..96 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let copy = rng & 7 == 0;
                let shared = rng & 7 == 1;
                let mut accesses = Vec::new();
                for j in 0..3 {
                    let x = rng.rotate_left(j * 17);
                    accesses.push(Access {
                        buffer: 1 + (x >> 16) % 4,
                        first: ((x >> 24) % 8) * 4,
                        bytes: 4 + 4 * ((x >> 32) % 2),
                        write: x & 2 == 0,
                    });
                }
                planner
                    .push(&NodeAccess {
                        copy,
                        shared,
                        release_after: i % 3 != 0,
                        accesses,
                    })
                    .unwrap();
            }
            let mut p = planner.finish().unwrap();
            let before = node_order(&p);
            let retained = footprint(&p);
            let queues = p
                .node_segments
                .iter()
                .map(|&i| p.segments[i].queue())
                .collect::<Vec<_>>();
            total += compact(&mut p).unwrap();
            assert_eq!(node_order(&p), before);
            assert_eq!(footprint(&p), retained);
            assert_eq!(
                p.node_segments
                    .iter()
                    .map(|&i| p.segments[i].queue())
                    .collect::<Vec<_>>(),
                queues
            );
            assert_eq!(compact(&mut p).unwrap(), 0);
        }
        assert!(total > 0);
    }
    fn segment(queue: usize, signal: u64, waits: [u64; 3]) -> Segment {
        Segment {
            copy: queue == 1,
            shared: queue == 2,
            signal,
            wait: *waits.iter().max().unwrap(),
            waits,
        }
    }
    #[test]
    fn actual_peer_observed_release_and_stronger_wait_keep_their_original_boundaries() {
        for observed in [true, false] {
            let mut p = Plan {
                segments: vec![
                    segment(0, 1, [0; 3]),
                    segment(0, 2, [0; 3]),
                    segment(0, 3, [0; 3]),
                    segment(1, 4, [if observed { 2 } else { 3 }, 0, 0]),
                ],
                node_segments: vec![1, 2, 3],
                last_copy: 4,
                last_shared: 0,
                accesses: Vec::new(),
            };
            let before = node_order(&p);
            assert_eq!(compact(&mut p).unwrap(), usize::from(!observed));
            assert_eq!(node_order(&p), before);
            if observed {
                assert_eq!(p.segments[3].waits, [2, 0, 0]);
            }
        }
        let mut p = Plan {
            segments: vec![
                segment(0, 1, [0; 3]),
                segment(1, 2, [1, 0, 0]),
                segment(0, 3, [0; 3]),
                segment(0, 4, [0, 2, 0]),
            ],
            node_segments: vec![1, 2, 3],
            last_copy: 2,
            last_shared: 0,
            accesses: Vec::new(),
        };
        let old = image(&p);
        assert_eq!(compact(&mut p).unwrap(), 0);
        assert_eq!(image(&p), old);
    }
    #[test]
    fn full_fixed_stack_bitmap_is_erased_without_allocating_a_private_copy() {
        let mut bits = ReferenceBits([u64::MAX; WORDS]);
        bits.erase();
        assert!(bits.0.iter().all(|&x| x == 0));
        assert_eq!(size_of::<ReferenceBits>(), 4096);
    }
    #[test]
    fn malformed_plan_rejects_before_mutation_and_full_node_capacity_is_bounded() {
        for bad in 0..4 {
            let mut p = Plan {
                segments: vec![
                    segment(0, 1, [0; 3]),
                    segment(0, 2, [0; 3]),
                    segment(1, 3, [2, 0, 0]),
                ],
                node_segments: vec![1, 2],
                last_copy: 3,
                last_shared: 0,
                accesses: Vec::new(),
            };
            match bad {
                0 => p.segments[2].waits = [3, 0, 0],
                1 => p.node_segments[1] = 3,
                2 => p.last_copy = 2,
                _ => p.segments[1].signal = 9,
            }
            let before = image(&p);
            assert!(compact(&mut p).is_err());
            assert_eq!(image(&p), before);
        }
        let mut p = Plan {
            segments: (1..=MAX_SEGMENTS)
                .map(|i| segment(0, i as u64, [0; 3]))
                .collect(),
            node_segments: (1..MAX_SEGMENTS).collect(),
            last_copy: 0,
            last_shared: 0,
            accesses: Vec::new(),
        };
        assert_eq!(compact(&mut p).unwrap(), MAX_SEGMENTS - 2);
        assert_eq!(p.segments.len(), 2);
        assert!(p.node_segments.iter().all(|&i| i == 1));
        assert_eq!(compact(&mut p).unwrap(), 0);
        assert_eq!(size_of::<[u64; WORDS]>(), 4096);
    }
}
