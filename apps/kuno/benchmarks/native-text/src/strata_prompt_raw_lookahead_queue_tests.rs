//! Execute the real upload-unit state machine through the original queue
//! planner. This is metadata-only; it claims no GPU completion or authority.
use super::*;
use crate::strata_queue_tests::{Access, NodeAccess, Planner};

const SLOT: u64 = 4 * 1024 * 1024;
const BANK: u64 = 8 * SLOT;

fn access(buffer: u64, first: u64, bytes: u64, write: bool) -> Access {
    Access {
        buffer,
        first,
        bytes,
        write,
    }
}
fn node(copy: bool, release_after: bool, accesses: Vec<Access>) -> NodeAccess {
    NodeAccess {
        copy,
        shared: false,
        release_after,
        accesses,
    }
}
fn prefetch(
    raw: &mut Plan,
    nodes: &mut Vec<NodeAccess>,
    uploads: &mut [usize; 12],
    copy_boundaries: bool,
) {
    while let Some(unit) = raw.issue_next().unwrap() {
        for group in unit.first..unit.end {
            let slot = raw.slot(unit, group).unwrap().unwrap();
            uploads[group] = nodes.len();
            // Three genuine role extents partition the declared slot. This
            // represents legal metadata, not a fabricated physical source.
            for (first, bytes) in [(0, SLOT / 4), (SLOT / 4, SLOT / 4), (SLOT / 2, SLOT / 2)] {
                nodes.push(node(
                    true,
                    false,
                    vec![access(
                        1 + unit.bank as u64,
                        slot as u64 * SLOT + first,
                        bytes,
                        true,
                    )],
                ));
            }
            // The production cut is after the captured expert command's final
            // real role copy, never between the roles of that expert.
            if copy_boundaries {
                nodes.last_mut().unwrap().release_after = true;
            }
        }
    }
}

#[test]
fn eight_legacy_groups_replay_real_raw_read_release_before_math_and_after_compaction() {
    // Preserve the original accepted four-expert COPY-segment oracle even
    // when the new production feature is enabled.
    check_slot_waits(false);
}

#[test]
fn per_expert_copies_keep_each_original_slot_wait_before_and_after_compaction() {
    check_slot_waits(true);
}

fn check_slot_waits(copy_boundaries: bool) {
    for dequant_release in [false, true] {
        for compact in [false, true] {
            let mut raw = Plan::new(&[64; 12], &[true; 12]).unwrap();
            let mut nodes = Vec::new();
            let mut uploads = [usize::MAX; 12];
            let mut dequant = [usize::MAX; 12];
            let mut down_math = [usize::MAX; 12];
            prefetch(&mut raw, &mut nodes, &mut uploads, copy_boundaries);
            assert!(uploads[..8].iter().all(|&i| i != usize::MAX));
            assert!(uploads[8..].iter().all(|&i| i == usize::MAX));

            for group in 0..12 {
                let unit = raw.current(group).unwrap();
                let slot = raw.slot(unit, group).unwrap().unwrap();
                let raw_buffer = 1 + unit.bank as u64;
                let raw_first = slot as u64 * SLOT;
                // Original six phases: gather, both image decodes, GU,
                // activation, DOWN. Only actual streamed DOWN-image decode
                // releases its raw reader, and every final DOWN keeps its
                // original unit-end marker even inside a larger upload unit.
                nodes.push(node(false, false, vec![access(3, 0, 4096, true)]));
                nodes.push(node(
                    false,
                    false,
                    vec![
                        access(raw_buffer, raw_first, SLOT, false),
                        access(4, 0, 4096, true),
                    ],
                ));
                dequant[group] = nodes.len();
                nodes.push(node(
                    false,
                    dequant_release,
                    vec![
                        access(raw_buffer, raw_first, SLOT, false),
                        access(5, 0, 4096, true),
                    ],
                ));
                nodes.push(node(
                    false,
                    false,
                    vec![
                        access(3, 0, 4096, false),
                        access(4, 0, 4096, false),
                        access(6, 0, 4096, true),
                    ],
                ));
                nodes.push(node(false, false, vec![access(6, 0, 4096, true)]));
                down_math[group] = nodes.len();
                nodes.push(node(
                    false,
                    true,
                    vec![
                        access(5, 0, 4096, false),
                        access(6, 0, 4096, false),
                        access(7, 0, 4096, true),
                    ],
                ));
                if raw.complete_group(unit, group).unwrap() {
                    prefetch(&mut raw, &mut nodes, &mut uploads, copy_boundaries);
                }
            }
            let (misses, resident, peak, copy_units) = raw.finish().unwrap();
            assert_eq!((misses, resident, peak, copy_units), (12, 0, 8, 3));
            assert_eq!(nodes.iter().filter(|n| n.copy).count(), 3 * misses);
            let copy_bytes = nodes
                .iter()
                .filter(|n| n.copy)
                .flat_map(|n| &n.accesses)
                .map(|a| a.bytes)
                .sum::<u64>();
            assert_eq!(copy_bytes, 12 * SLOT);
            let mut planner = Planner::new(nodes.len(), false).unwrap();
            for n in &nodes {
                planner.push(n).unwrap();
            }
            let mut actual = planner.finish().unwrap();
            if compact {
                actual.compact_segments().unwrap();
            }
            for replacement in 8..12 {
                let old = if copy_boundaries { replacement - 8 } else { 3 };
                let refill = &actual.segments[actual.node_segments[uploads[replacement]]];
                let old_read = &actual.segments[actual.node_segments[dequant[old]]];
                let old_math = &actual.segments[actual.node_segments[down_math[old]]];
                assert!(refill.copy);
                assert_eq!(refill.waits[0], old_read.signal);
                if dequant_release {
                    assert!(refill.waits[0] < old_math.signal);
                } else {
                    // Negative control: omitting the genuine early release
                    // drags each same slot behind its own original math.
                    assert_eq!(refill.waits[0], old_math.signal);
                }
            }
            for group in 0..12 {
                let first = actual.node_segments[uploads[group]];
                assert!((1..3).all(|role| actual.node_segments[uploads[group] + role] == first));
                if copy_boundaries && group % 4 != 3 {
                    assert_ne!(first, actual.node_segments[uploads[group + 1]]);
                }
            }
            if copy_boundaries {
                // The first DQ imports only its own initial upload endpoint,
                // allowing later initial-slot DMA to overlap its compute.
                let dq0 = &actual.segments[actual.node_segments[dequant[0]]];
                let copy0 = &actual.segments[actual.node_segments[uploads[0]]];
                let copy7 = &actual.segments[actual.node_segments[uploads[7]]];
                assert_eq!(dq0.waits[1], copy0.signal);
                assert!(dq0.waits[1] < copy7.signal);
            }
            for bank in 1..=2 {
                assert!(actual.conflicts(&[access(bank, 0, BANK, true)]).unwrap());
                // Legacy kernels actually bind just their slot. The separate
                // original whole-bank cohort controls keep their full reads.
                assert!(
                    !actual
                        .conflicts(&[access(bank, 4 * SLOT, 4 * SLOT, true)])
                        .unwrap()
                );
            }
        }
    }
}

#[test]
fn per_expert_copy_cuts_preserve_each_main_and_shared_reader_wait() {
    for compact in [false, true] {
        let mut nodes = Vec::new();
        let mut main_read = [0; 4];
        let mut shared_read = [0; 4];
        let mut uploads = [0; 4];
        for slot in 0..4 {
            main_read[slot] = nodes.len();
            nodes.push(node(
                false,
                true,
                vec![access(1, slot as u64 * SLOT, SLOT, false)],
            ));
            shared_read[slot] = nodes.len();
            let mut shared = node(
                false,
                true,
                vec![access(1, slot as u64 * SLOT, SLOT, false)],
            );
            shared.shared = true;
            nodes.push(shared);
        }
        for slot in 0..4 {
            uploads[slot] = nodes.len();
            nodes.push(node(
                true,
                true,
                vec![access(1, slot as u64 * SLOT, SLOT, true)],
            ));
            // A real later main reader observes this original COPY endpoint,
            // so neither that endpoint nor its required peer waits disappear.
        }
        for slot in 0..4 {
            nodes.push(node(
                false,
                true,
                vec![access(1, slot as u64 * SLOT, SLOT, false)],
            ));
        }
        let mut planner = Planner::new(nodes.len(), false).unwrap();
        for n in &nodes {
            planner.push(n).unwrap();
        }
        let mut actual = planner.finish().unwrap();
        if compact {
            actual.compact_segments().unwrap();
        }
        for slot in 0..4 {
            let upload = &actual.segments[actual.node_segments[uploads[slot]]];
            let main = &actual.segments[actual.node_segments[main_read[slot]]];
            let shared = &actual.segments[actual.node_segments[shared_read[slot]]];
            assert!(upload.copy && !upload.shared);
            assert_eq!(upload.waits, [main.signal, 0, shared.signal]);
            assert_eq!(upload.wait, main.signal.max(shared.signal));
            if slot != 3 {
                assert!(upload.waits[2]
                    < actual.segments[actual.node_segments[shared_read[slot + 1]]].signal);
            }
        }
    }
}

#[test]
fn per_expert_copy_cuts_do_not_narrow_a_whole_bank_cohort_reader() {
    for compact in [false, true] {
        let mut nodes = vec![
            node(false, true, vec![access(1, 0, BANK, false)]),
            node(false, true, vec![access(2, 0, 4096, true)]),
        ];
        let mut uploads = [0; 4];
        for slot in 0..4 {
            uploads[slot] = nodes.len();
            nodes.push(node(
                true,
                true,
                vec![access(1, slot as u64 * SLOT, SLOT, true)],
            ));
        }
        let mut planner = Planner::new(nodes.len(), false).unwrap();
        for n in &nodes {
            planner.push(n).unwrap();
        }
        let mut actual = planner.finish().unwrap();
        if compact {
            actual.compact_segments().unwrap();
        }
        let cohort = &actual.segments[actual.node_segments[0]];
        for upload in uploads {
            let copy = &actual.segments[actual.node_segments[upload]];
            assert_eq!(copy.waits[0], cohort.signal);
        }
        assert!(actual.conflicts(&[access(1, 4 * SLOT, 4 * SLOT, true)]).unwrap());
    }
}
