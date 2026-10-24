//! Host-only controls against the real upload plan.
//! These create no runtime scope, GPU buffer, queue or completion authority.
use super::*;

#[test]
fn raw_lookahead_actual_legacy_runs_keep_every_role_copy_group_and_four_slot_bound() {
    for len in [1, 4, 5, 8, 9, 31, 64, 512] {
        for residency in 0..9 {
            let members: Vec<_> = (0..len).map(|g| [32, 65, 255, 4096][g % 4]).collect();
            let masks: Vec<u8> = (0..len)
                .map(|g| match residency {
                    0 => 0,
                    1 => 7,
                    2 => (g % 8) as u8,
                    _ => [1, 2, 4, 3, 5, 6][residency - 3],
                })
                .collect();
            let host: Vec<_> = masks.iter().map(|&mask| mask != 0).collect();
            let mut plan = Plan::new(&members, &host).unwrap();
            let mut copies = vec![[0u8; 3]; len];
            let mut seen = vec![0u8; len];
            let mut uploaded = vec![None; len];
            let mut group = 0;
            while group < len {
                while let Some(unit) = plan.issue_next().unwrap() {
                    assert!(unit.legacy && unit.end - unit.first <= 4);
                    let mut next_slot = 0;
                    for g in unit.first..unit.end {
                        if host[g] {
                            let slot = plan.slot(unit, g).unwrap().unwrap();
                            assert_eq!(slot, next_slot);
                            assert!(slot < 4);
                            next_slot += 1;
                            assert!(uploaded[g].replace((unit.bank, slot)).is_none());
                            // Classifying a group as streamed means ANY role
                            // is HOST, not necessarily all three. Retain each
                            // exact role-copy population under mixed tiers.
                            for (role, count) in copies[g].iter_mut().enumerate() {
                                *count += u8::from(masks[g] & (1 << role) != 0);
                            }
                        } else {
                            assert_eq!(plan.slot(unit, g).unwrap(), None);
                        }
                    }
                }
                let unit = plan.current(group).unwrap();
                assert!(plan.consumed(unit).is_err()); // no whole-unit shortcut
                assert_eq!(
                    plan.slot(unit, group).unwrap(),
                    uploaded[group].map(|(_, s)| s)
                );
                seen[group] += 1;
                assert_eq!(
                    plan.complete_group(unit, group).unwrap(),
                    group + 1 == unit.end
                );
                group += 1;
            }
            let (misses, resident, peak, copy_units) = plan.finish().unwrap();
            assert_eq!(misses, host.iter().filter(|&&h| h).count());
            assert_eq!(misses + resident, len);
            assert!(peak <= 8);
            assert!(copy_units <= len.div_ceil(4));
            assert_eq!(
                plan.legacy_geometry().unwrap(),
                [
                    len.div_ceil(4),
                    len,
                    len / 4 + usize::from(len % 4 > 1),
                    len / 4 * 4 + if len % 4 > 1 { len % 4 } else { 0 }
                ]
            );
            for g in 0..len {
                assert_eq!(seen[g], 1);
                for role in 0..3 {
                    assert_eq!(copies[g][role], u8::from(masks[g] & (1 << role) != 0));
                }
            }
        }
    }
}

#[test]
fn raw_lookahead_mixed_units_preserve_original_cohorts_and_never_cross_category_boundaries() {
    let members = [
        1, 31, 32, 65, 255, 4096, 32, 1, 2, 127, 128, 129, 256, 257, 1,
    ];
    let host: Vec<_> = (0..members.len()).map(|g| g % 3 != 1).collect();
    let expected = [
        (0, 2, false),
        (2, 6, true),
        (6, 7, true),
        (7, 9, false),
        (9, 13, true),
        (13, 14, true),
        (14, 15, false),
    ];
    let mut plan = Plan::new(&members, &host).unwrap();
    for (index, &(first, end, legacy)) in expected.iter().enumerate() {
        while plan.issue_next().unwrap().is_some() {}
        let unit = plan.current(first).unwrap();
        assert_eq!(
            (unit.index, unit.first, unit.end, unit.bank, unit.legacy),
            (index, first, end, index % 2, legacy)
        );
        if legacy {
            for group in first..end {
                assert_eq!(plan.current(group).unwrap(), unit);
                assert_eq!(plan.complete_group(unit, group).unwrap(), group + 1 == end);
            }
        } else {
            assert!(plan.complete_group(unit, first).is_err());
            assert!(members[first..end].iter().all(|&n| n < 32));
            plan.consumed(unit).unwrap();
        }
    }
    assert_eq!(
        plan.finish().unwrap().0,
        host.iter().filter(|&&h| h).count()
    );
    assert_eq!(plan.legacy_geometry().unwrap(), [4, 10, 2, 8]);
}

#[test]
fn raw_lookahead_cursor_rejects_skips_repeats_forged_units_and_early_bank_refills() {
    let mut plan = Plan::new(&[64; 13], &[true; 13]).unwrap();
    let a = plan.issue_next().unwrap().unwrap();
    let b = plan.issue_next().unwrap().unwrap();
    assert_eq!((a.first, a.end, b.first, b.end, plan.live), (0, 4, 4, 8, 8));
    assert!(plan.issue_next().unwrap().is_none());
    assert!(plan.complete_group(b, 4).is_err());
    assert!(plan.complete_group(a, 1).is_err());
    assert!(plan.current(1).is_err());
    assert!(plan.consumed(a).is_err());
    assert!(plan.legacy_geometry().is_err());
    let mut forged = a;
    forged.end -= 1;
    assert!(plan.complete_group(forged, 0).is_err());
    forged = a;
    forged.legacy = false;
    assert!(plan.consumed(forged).is_err());
    for group in 0..3 {
        assert!(!plan.complete_group(a, group).unwrap());
        assert!(plan.complete_group(a, group).is_err());
        assert!(plan.issue_next().unwrap().is_none());
        assert_eq!(plan.occupied, [Some(0), Some(1)]);
    }
    assert!(plan.complete_group(a, 3).unwrap());
    assert!(plan.slot(a, 0).is_err());
    assert!(plan.complete_group(a, 3).is_err());
    let c = plan.issue_next().unwrap().unwrap();
    assert_eq!(
        (c.index, c.bank, c.first, c.end, plan.live),
        (2, 0, 8, 12, 8)
    );
    assert!(plan.current(8).is_err());
    for group in 4..8 {
        plan.complete_group(b, group).unwrap();
    }
    let d = plan.issue_next().unwrap().unwrap();
    for group in 8..12 {
        plan.complete_group(c, group).unwrap();
    }
    assert!(plan.complete_group(d, 12).unwrap());
    assert_eq!(plan.finish().unwrap(), (13, 0, 8, 4));
}

#[test]
fn raw_lookahead_opt_in_keeps_the_original_plan_available_and_original_group_release_population() {
    let members = [64; 17];
    let host = [true; 17];
    let mut old = Plan::with_lookahead(&members, &host, false).unwrap();
    let mut new = Plan::new(&members, &host).unwrap();
    assert_eq!(old.units.len(), 17);
    assert_eq!(new.units.len(), 5);
    let mut old_markers = 0;
    let mut new_markers = 0;
    for group in 0..17 {
        while old.issue_next().unwrap().is_some() {}
        while new.issue_next().unwrap().is_some() {}
        let unit = old.current(group).unwrap();
        assert!(!unit.legacy && unit.end == group + 1);
        old_markers += 1;
        old.consumed(unit).unwrap();
        let unit = new.current(group).unwrap();
        new_markers += 1; // the actual caller retains EVERY original group end
        new.complete_group(unit, group).unwrap();
    }
    assert_eq!((old_markers, new_markers), (17, 17));
    assert_eq!(old.finish().unwrap(), (17, 0, 2, 17));
    assert_eq!(new.finish().unwrap(), (17, 0, 8, 5));
    assert_eq!(old.legacy_geometry().unwrap(), [0; 4]);
}

#[test]
fn raw_lookahead_cursor_flag_and_full_reserved_metadata_erase_inside_original_eight_kib() {
    let mut plan = Plan::new(&[64; 512], &[true; 512]).unwrap();
    let first = plan.issue_next().unwrap().unwrap();
    plan.issue_next().unwrap();
    plan.complete_group(first, 0).unwrap();
    assert_eq!(plan.next_group, 1);
    assert!(plan.units[0][1] & LEGACY_UNIT != 0);
    let uc = plan.units.capacity();
    let sc = plan.slots.capacity();
    assert!(uc * 4 + sc + 512 + size_of::<Plan>() + size_of::<Vec<bool>>() + 1024 <= HOST_BYTES);
    plan.erase();
    assert!(plan.units.is_empty() && plan.slots.is_empty());
    assert_eq!(
        (
            plan.issued,
            plan.consumed,
            plan.next_group,
            plan.live,
            plan.peak,
            plan.misses,
            plan.copy_units
        ),
        (0, 0, 0, 0, 0, 0, 0)
    );
    assert_eq!(plan.occupied, [None; 2]);
    // Allocations stay live after erase; inspect their complete reserved bytes.
    let units = unsafe { std::slice::from_raw_parts(plan.units.as_ptr().cast::<u8>(), uc * 4) };
    let slots = unsafe { std::slice::from_raw_parts(plan.slots.as_ptr(), sc) };
    assert!(units.iter().chain(slots).all(|&byte| byte == 0));
}

#[cfg(feature="strata-cohort-row-order")]
#[test]
fn reordered_real_maps_fill_cohorts_without_changing_any_source_slot_or_release_law() {
    use crate::strata_prompt::groups;
    let raw = groups::row_order_tests::packet(4096);
    let mut totals = Vec::new();
    for map in [groups::group(&raw, 4096).unwrap(), groups::group_for_prefill(&raw, 4096).unwrap()] {
        let n = groups::validate(&map, 4096).unwrap();
        let members = &map[516..516+n];
        let host = (0..n).map(|g| map[4+g] % 3 != 1).collect::<Vec<_>>();
        let mut plan = Plan::new(members, &host).unwrap();
        let mut group=0;let mut cohorts=0;let mut seen=vec![false;n];
        while group<n {
            while plan.issue_next().unwrap().is_some() {}
            let unit=plan.current(group).unwrap();
            if unit.legacy {
                for g in unit.first..unit.end {
                    assert!(!seen[g]);seen[g]=true;
                    assert_eq!(plan.slot(unit,g).unwrap().is_some(),host[g]);
                    assert_eq!(plan.complete_group(unit,g).unwrap(),g+1==unit.end);
                }
            }else{
                cohorts+=1;assert!(unit.end-unit.first<=8);
                for g in unit.first..unit.end {assert!(!seen[g]);seen[g]=true;assert!(members[g]<32);assert_eq!(plan.slot(unit,g).unwrap().is_some(),host[g]);}
                plan.consumed(unit).unwrap();
            }
            group=unit.end;
        }
        let (misses,resident,peak,_)=plan.finish().unwrap();
        assert!(peak<=16);assert_eq!(misses,host.iter().filter(|&&h|h).count());assert_eq!(misses+resident,n);assert!(seen.into_iter().all(|v|v));
        totals.push((cohorts,members.iter().filter(|&&v|v<32).count(),misses));
    }
    assert_eq!(totals[0].1,totals[1].1);assert_eq!(totals[0].2,totals[1].2);
    assert!(totals[1].0<totals[0].0);assert_eq!(totals[1].0,totals[1].1.div_ceil(8));
}
