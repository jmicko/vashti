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
fn prefetch(raw: &mut Plan, nodes: &mut Vec<NodeAccess>, uploads: &mut [usize; 12]) {
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
        }
    }
}

#[test]
fn eight_legacy_groups_replay_real_raw_read_release_before_math_and_after_compaction() {
    for dequant_release in [false, true] {
        for compact in [false, true] {
            let mut raw = Plan::new(&[64; 12], &[true; 12]).unwrap();
            let mut nodes = Vec::new();
            let mut uploads = [usize::MAX; 12];
            let mut dequant = [usize::MAX; 12];
            let mut down_math = [usize::MAX; 12];
            prefetch(&mut raw, &mut nodes, &mut uploads);
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
                    prefetch(&mut raw, &mut nodes, &mut uploads);
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
            let refill = &actual.segments[actual.node_segments[uploads[8]]];
            let old_read = &actual.segments[actual.node_segments[dequant[3]]];
            let old_math = &actual.segments[actual.node_segments[down_math[3]]];
            assert!(refill.copy);
            assert_eq!(refill.waits[0], old_read.signal);
            if dequant_release {
                assert!(refill.waits[0] < old_math.signal);
            } else {
                // Negative control: omitting the genuine early-release marker
                // drags the same refill behind matrix work. No wait is waived.
                assert_eq!(refill.waits[0], old_math.signal);
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
