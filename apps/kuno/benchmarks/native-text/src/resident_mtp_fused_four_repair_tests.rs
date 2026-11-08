//! Offline proofs for the unused fused-four wrapper. No model or device is
//! constructed; the original repair/cache tests remain separate controls.
use super::*;

fn carries(count: usize) -> Vec<f32> {
    (0..count * k::WIDE)
        .map(|i| {
            if i % 23 == 0 {
                -0.0
            } else {
                (i % 79) as f32 / 83.0
            }
        })
        .collect()
}

fn function(source: &str, signature: &str) -> String {
    let rest = source.split_once(signature).unwrap().1;
    let opening = rest.find('{').unwrap();
    let mut depth = 0usize;
    for (index, c) in rest[opening..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return rest[opening..=opening + index].to_owned();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated source function");
}

#[test]
fn real_third_draft_prefixes_keep_original_cell_positions_and_carry_bits() {
    let start = 7;
    let old = start + 2;
    let binding = 19;
    let stamp = LastForward {
        binding,
        start: old - 1,
    };
    let preceding = carries(1);
    let tokens = [3, 5, 11, 13];
    for count in 1..=4 {
        let target_carries = carries(count);
        let plan = AcceptedCacheRepair::new_fused_four(
            old,
            start,
            stamp.valid(binding, old),
            true,
            &tokens[..count],
            &preceding,
            &target_carries,
        )
        .unwrap();
        assert!(plan.rewind);
        assert_eq!(plan.first_row, 1);
        assert_eq!(plan.end, start + count - 1);
        // Independent scalar repair: retain the first real cell, rewind the
        // whole suffix, then append every remaining accepted input in order.
        let expected: Vec<_> = (1..count)
            .map(|row| {
                (
                    start + row - 1,
                    tokens[row],
                    &target_carries[(row - 1) * k::WIDE..row * k::WIDE],
                )
            })
            .collect();
        let queued: Vec<_> = (plan.first_row..count)
            .map(|row| {
                let carry = if row == 0 {
                    &preceding[..]
                } else {
                    &target_carries[(row - 1) * k::WIDE..row * k::WIDE]
                };
                (start + row - 1, tokens[row], carry)
            })
            .collect();
        assert_eq!(queued.len(), count - 1);
        for ((p, t, c), (ep, et, ec)) in queued.iter().zip(&expected) {
            assert_eq!((p, t), (ep, et));
            assert!(c.iter().zip(*ec).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let original: Vec<u32> = (0..k::CONTEXT)
            .map(|i| (-(i as f32 / 71.)).to_bits())
            .collect();
        let mut scalar = original.clone();
        let mut joined = original;
        scalar[start..].fill(0);
        if plan.rewind {
            joined[start..].fill(0);
        }
        for &(p, t, c) in &expected {
            scalar[p] = c[t as usize].to_bits();
        }
        for &(p, t, c) in &queued {
            joined[p] = c[t as usize].to_bits();
        }
        assert_eq!(scalar, joined);
    }
}

#[test]
fn fused_only_preflight_requires_exact_cursor_and_real_binding_authority() {
    let start = 17;
    let old = start + 2;
    let preceding = carries(1);
    let tokens = [11, 13, 17, 19];
    let full = carries(4);
    for (cursor, begin, stamp, fused) in [
        (old, start, true, false),
        (old, start, false, true),
        (start - 1, start, false, true),
        (start, start, true, true),
        (start + 1, start, true, true),
        (start + 3, start, true, true),
        (2, 0, true, true),
        (k::CONTEXT + 1, k::CONTEXT - 1, true, true),
        (usize::MAX, usize::MAX - 1, true, true),
        (1, usize::MAX, true, true),
    ] {
        assert!(
            AcceptedCacheRepair::new_fused_four(
                cursor, begin, stamp, fused, &tokens, &preceding, &full,
            )
            .is_err()
        );
    }
    let stamp = LastForward {
        binding: 31,
        start: old - 1,
    };
    for (binding, cursor) in [(32, old), (31, old - 1), (31, old + 1)] {
        assert!(!stamp.valid(binding, cursor));
        assert!(
            AcceptedCacheRepair::new_fused_four(
                old,
                start,
                stamp.valid(binding, cursor),
                true,
                &tokens,
                &preceding,
                &full,
            )
            .is_err()
        );
    }
    assert!(stamp.valid(31, old));
}

#[test]
fn every_prefix_token_carry_and_context_span_is_validated_before_mutation() {
    let start = 17;
    let preceding = carries(1);
    let tokens = [11, 13, 17, 19];
    for count in 1..=4 {
        let full = carries(count);
        for row in 0..count {
            let mut bad_tokens = tokens;
            bad_tokens[row] = k::VOCAB as u32;
            assert!(
                AcceptedCacheRepair::new_fused_four(
                    start + 2,
                    start,
                    true,
                    true,
                    &bad_tokens[..count],
                    &preceding,
                    &full,
                )
                .is_err()
            );
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut bad_carries = full.clone();
                // Includes the final carry, which is not appended internally.
                bad_carries[(row + 1) * k::WIDE - 1] = bad;
                assert!(
                    AcceptedCacheRepair::new_fused_four(
                        start + 2,
                        start,
                        true,
                        true,
                        &tokens[..count],
                        &preceding,
                        &bad_carries,
                    )
                    .is_err()
                );
            }
        }
        for (before, after) in [
            (&preceding[..k::WIDE - 1], &full[..]),
            (&preceding[..], &full[..full.len() - 1]),
        ] {
            assert!(
                AcceptedCacheRepair::new_fused_four(
                    start + 2,
                    start,
                    true,
                    true,
                    &tokens[..count],
                    before,
                    after,
                )
                .is_err()
            );
        }
        let mut bad = preceding.clone();
        bad[0] = f32::NAN;
        assert!(
            AcceptedCacheRepair::new_fused_four(
                start + 2,
                start,
                true,
                true,
                &tokens[..count],
                &bad,
                &full,
            )
            .is_err()
        );
    }
    for count in [0, 5] {
        assert!(
            AcceptedCacheRepair::new_fused_four(
                start + 2,
                start,
                true,
                true,
                &vec![1; count],
                &preceding,
                &carries(count),
            )
            .is_err()
        );
    }
    let start = k::CONTEXT - 2;
    for count in 1..=3 {
        assert!(
            AcceptedCacheRepair::new_fused_four(
                k::CONTEXT,
                start,
                true,
                true,
                &tokens[..count],
                &preceding,
                &carries(count),
            )
            .is_ok()
        );
    }
    assert!(
        AcceptedCacheRepair::new_fused_four(
            k::CONTEXT,
            start,
            true,
            true,
            &tokens,
            &preceding,
            &carries(4),
        )
        .is_err()
    );
}

#[test]
fn old_api_still_rejects_third_cursor_and_four_inputs() {
    let start = 17;
    let preceding = carries(1);
    let tokens = [11, 13, 17, 19];
    for count in 1..=4 {
        assert!(
            AcceptedCacheRepair::new(
                start + 2,
                start,
                true,
                &tokens[..count],
                &preceding,
                &carries(count),
            )
            .is_err()
        );
    }
    for drafts in 0..=2 {
        for count in 1..=drafts + 1 {
            let plan = AcceptedCacheRepair::new(
                start - 1 + drafts,
                start,
                drafts != 0,
                &tokens[..count],
                &preceding,
                &carries(count),
            )
            .unwrap();
            assert_eq!(plan.first_row, usize::from(drafts != 0));
            assert_eq!(plan.end, start + count - 1);
        }
    }
}

#[test]
fn actual_dispatch_inventory_and_original_all_owner_reservations_remain_bounded() {
    let (initial, initial_count) = clear_roles(false);
    let (preserved, preserved_count) = clear_roles(true);
    assert_eq!(initial_count, 40);
    assert_eq!(preserved_count, 37);
    assert!(initial[..initial_count].contains(&(B::Fault as usize)));
    assert!(!preserved[..preserved_count].contains(&(B::Fault as usize)));
    let initial_clears = initial_count.div_ceil(7);
    let repeated_clears = preserved_count.div_ceil(7) + 1;
    let guard_dispatches = (N - 1).div_ceil(6) + 1;
    let condition = 16 + 1 + 2 * 3 + 1 + 1 + 1;
    let hc = 3 + 2 + 1 + 1 + 2 + 1 + 1 + 1;
    let kv = 2 + 2 + 3 + 1;
    assert_eq!(condition + hc + kv, 46);
    assert_eq!(
        initial_clears
            + 1
            + guard_dispatches
            + 3 * (repeated_clears + 2 * guard_dispatches + condition + hc + kv),
        MAX_FUSED_FOUR_REPAIR_DISPATCHES
    );
    assert_eq!(MAX_FUSED_FOUR_REPAIR_DISPATCHES, 222);
    assert_eq!(40 + 1 + N + 3 * (39 + 2 * N + 46), 590);
    assert_eq!(MAX_BATCHED_REPAIR_DISPATCHES, 421);
    assert_eq!(MAX_TOKEN_DISPATCHES, 512);
    assert_eq!(N, 42);
    assert_eq!(private_plan().unwrap(), 11_698_176);
    assert_eq!(INFO_BYTES, 1_048_576);
    assert_eq!(HOST_BYTES, 16 * 1024 * 1024);
    assert_eq!(PRIVATE_BYTES, 16 * 1024 * 1024);
    assert_eq!(RUNTIME_BYTES, 64 * 1024 * 1024);
    let source = include_str!("resident_mtp.rs");
    let guards = function(source, "fn guard_private_fused(");
    assert!(guards.contains("roles.chunks(6)"));
    assert_eq!(guards.matches("self.tick(1)").count(), 2);
    let tick = function(source, "fn tick(");
    assert!(tick.contains("self.dispatches > MAX_TOKEN_DISPATCHES"));
    assert!(tick.contains("traffic.explicit_dispatches"));
    let old = function(source, "pub fn repair_accepted_cache(");
    assert!(old.contains("MAX_TOKEN_DISPATCHES"));
    assert!(!old.contains("MAX_FUSED_FOUR_REPAIR_DISPATCHES"));
}

#[test]
fn shared_queue_retains_late_fault_ledger_publication_and_original_failure_wipe() {
    let source = include_str!("resident_mtp.rs");
    let selected = function(source, "pub(crate) fn repair_accepted_cache_fused_four(");
    for actual_guard in [
        "!self.fused_sweeps",
        "self.poisoned",
        "self.closed",
        "self.retired.load(Ordering::Acquire)",
        "self.live_outputs.load(Ordering::Acquire) != 0",
        "self.graph.profile() != MtpProfile::SelfContainedQ8",
        "self.public_english_vocab.is_some()",
        "stamp.valid(self.binding, self.position)",
        "AcceptedCacheRepair::new_fused_four(",
        "MAX_FUSED_FOUR_REPAIR_DISPATCHES",
    ] {
        assert!(selected.contains(actual_guard), "missing {actual_guard}");
    }
    assert!(
        selected
            .find("AcceptedCacheRepair::new_fused_four(")
            .unwrap()
            < selected.find("self.repair_accepted_cache_plan(").unwrap()
    );
    let queued = function(source, "fn repair_accepted_cache_plan(");
    let ordered = [
        "self.verify_source()?;",
        "self.physical()?;",
        ".observe_before(",
        "self.last_forward.take();",
        "self.poisoned = true;",
        "self.dispatches = 0;",
        "if plan.rewind",
        "for row in plan.first_row..accepted_tokens.len()",
        "self.key_value_at(position)?;",
        "self.guard_private(position + 1)?;",
        "self.dispatches > dispatch_limit",
        "self.check_fault()?;",
        "completed_batched_repair(",
        ".completed_fence(",
        "self.traffic.set(completed);",
        "self.position = plan.end;",
        "self.poisoned = false;",
    ];
    let mut previous = 0;
    for needle in ordered {
        let next = queued[previous..].find(needle).unwrap() + previous;
        previous = next + needle.len();
    }
    assert_eq!(queued.matches("self.check_fault()?").count(), 1);
    assert_eq!(queued.matches(".completed_fence(").count(), 1);
    assert_eq!(queued.matches("self.dispatches = 0").count(), 1);
    assert_eq!(queued.matches("self.position =").count(), 1);
    assert_eq!(queued.matches("self.fail()").count(), 2);
    assert!(!queued.contains("self.complete_fault()") && !queued.contains("self.start("));
    let clear = function(source, "fn clear_repair_scratch(");
    assert!(clear.contains("self.clear_scratch_preserving_inputs(true)?"));
    assert!(clear.contains("[B::Token as usize, B::Carry as usize]"));
    assert!(!clear.contains("B::Fault"));
    let gate = function(source, "fn check_fault(");
    assert!(gate.contains("B::Fault"));
    let fail = function(source, "fn fail(");
    assert!(fail.contains("self.last_forward.take()"));
    assert!(fail.contains("for &id in &self.ids") && fail.contains("u::zero(&self.arena, id)?"));
    assert!(fail.contains("self.position.zeroize()") && fail.contains("self.ledger.poison()"));
}

#[test]
fn provisional_joined_counters_accept_three_real_cells_without_losing_original_traffic() {
    let original = Traffic {
        host_read_calls: 7,
        host_read_bytes: 99,
        explicit_dispatches: MAX_FUSED_FOUR_REPAIR_DISPATCHES as u64,
        ..Traffic::default()
    };
    for cells in 0..=3 {
        let completed = completed_batched_repair(original, cells).unwrap();
        assert_eq!(completed.batched_repair_calls, 1);
        assert_eq!(completed.batched_repair_cells, cells as u64);
        assert_eq!(completed.batched_repair_fault_gates, 1);
        assert_eq!(completed.host_read_calls, original.host_read_calls);
        assert_eq!(completed.host_read_bytes, original.host_read_bytes);
        assert_eq!(completed.explicit_dispatches, original.explicit_dispatches);
    }
    for overflowing in [
        Traffic {
            batched_repair_calls: u64::MAX,
            ..original
        },
        Traffic {
            batched_repair_cells: u64::MAX,
            ..original
        },
        Traffic {
            batched_repair_fault_gates: u64::MAX,
            ..original
        },
    ] {
        assert!(completed_batched_repair(overflowing, 3).is_err());
        assert_eq!(
            overflowing.explicit_dispatches,
            original.explicit_dispatches
        );
    }
}
