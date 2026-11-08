//! Additive source controls for a distinct shared128 prefix in format1 units.
//! Metadata-only CPU tests; no shader, GPU, whole-model or throughput claim.
use super::*;
const SHARED: &str = "probe::strata_matrix::cohorts::large_shared::expert_large_shared_cohort::ExpertLargeSharedCohort<Runtime>";

#[cfg(feature = "kuno-large-shared-panels")]
fn shared_layer_program(units: &[Vec<u32>]) -> Vec<Entry> {
    let mut program = packed_layer_program(units);
    for entry in &mut program {
        if entry.name == LARGE_PACKED && word(&entry.args.info, 11).unwrap() == 128 {
            entry.name = SHARED;
        }
    }
    program
}

#[cfg(feature = "kuno-large-shared-panels")]
#[test]
fn shared_large_prefix_and_original_packed_tails_complete_authentic_format1_layers() {
    for tails in [false, true] {
        let units = packed_units(tails);
        let program = shared_layer_program(&units);
        let counts = run_layer(&program, true).unwrap();
        assert_eq!(counts.layer_gather_launches, 1);
        assert_eq!(
            counts.large_cohort_groups,
            units.iter().map(Vec::len).sum::<usize>()
        );
        assert_eq!(counts.large_cohort_batches, units.len());
        assert_eq!(counts.large_cohort_launches, program.len() - 1);
        assert_eq!(counts.regular_cohort_groups, 0);
        let mut prefixes = 0;
        let mut old_tails = 0;
        for entry in &program[1..] {
            assert_eq!(word(&entry.args.info, 13).unwrap(), 1);
            assert_eq!(word(&entry.args.info, 14).unwrap(), 1);
            if entry.name == SHARED {
                assert_eq!(word(&entry.args.info, 11).unwrap(), 128);
                prefixes += 1;
            } else if entry.name == LARGE_PACKED {
                assert!(matches!(word(&entry.args.info, 11).unwrap(), 8 | 16 | 32));
                old_tails += 1;
            }
        }
        assert_eq!(prefixes, units.len() * 2);
        assert_eq!(old_tails > 0, tails);
        for phase in 0..program.len() {
            let mut missing = shared_layer_program(&units);
            missing.remove(phase);
            assert!(run_layer(&missing, true).is_err());
            let mut release = shared_layer_program(&units);
            release[phase].release = !release[phase].release;
            assert!(run_layer(&release, true).is_err());
        }
    }
    // Retain both complete original-format and packed-format references even
    // when the distinct new acquisition feature is compiled.
    assert!(run_layer(&layer_program(), true).is_ok());
    assert!(run_layer(&packed_layer_program(&packed_units(true)), true).is_ok());
}

#[cfg(feature = "kuno-large-shared-panels")]
#[test]
fn shared_large_prefix_requires_exact_original_grid_info_and_physical_owners() {
    let units = packed_units(true);
    let program = shared_layer_program(&units);
    for (phase, entry) in program.iter().enumerate().filter(|(_, e)| e.name == SHARED) {
        let CubeCount::Static(x, y, z) = entry.count else {
            panic!("literal genuine prefix grid")
        };
        assert_eq!(z, 1);
        for count in [
            CubeCount::Static(x + 1, y, z),
            CubeCount::Static(x, y + 1, z),
            CubeCount::Static(x, y, 2),
        ] {
            let mut bad = shared_layer_program(&units);
            bad[phase].count = count;
            assert!(run_layer(&bad, true).is_err());
        }
        for index in [0usize, 4, 12, 13, 14, 15, 16, 17, 18, 19, 48, 54] {
            let mut bad = shared_layer_program(&units);
            let value = word(&bad[phase].args.info, index).unwrap();
            set_word(&mut bad[phase].args.info, index, value + 1);
            assert!(run_layer(&bad, true).is_err(), "INFO {index} phase {phase}");
        }
        for tile in [8, 16, 32, 64] {
            let mut bad = shared_layer_program(&units);
            set_word(&mut bad[phase].args.info, 11, tile);
            assert!(run_layer(&bad, true).is_err());
        }
        for owner in 0..6 {
            let mut bad = shared_layer_program(&units);
            let old = &bad[phase].args.buffers[owner];
            // The first DOWN legitimately establishes its full PART sink;
            // require its exact extent rather than invent an expected ID.
            let bytes = old.size_in_used() - u64::from(owner == 2) * 4;
            bad[phase].args.buffers[owner] = Handle::new(old.stream, bytes).binding();
            assert!(
                run_layer(&bad, true).is_err(),
                "owner {owner} phase {phase}"
            );
        }
    }
}

#[cfg(feature = "kuno-large-shared-panels")]
#[test]
fn shared_large_prefix_never_certifies_tail_legacy_format_or_forged_identity() {
    assert_eq!(matrix_format_variant(SHARED, false, true), None);
    assert_eq!(matrix_format_variant(SHARED, true, true), None);
    for down in [false, true] {
        assert_eq!(matrix_phase_variant(SHARED, down, true, 128), Some(down));
        assert_eq!(matrix_phase_variant(SHARED, down, false, 128), None);
        for tile in [8, 16, 32] {
            assert_eq!(matrix_phase_variant(SHARED, down, true, tile), None);
        }
    }
    let units = packed_units(true);
    let program = shared_layer_program(&units);
    for (phase, entry) in program
        .iter()
        .enumerate()
        .filter(|(_, e)| e.name == LARGE_PACKED)
    {
        let mut bad = shared_layer_program(&units);
        bad[phase].name = SHARED;
        assert!(matches!(word(&entry.args.info, 11).unwrap(), 8 | 16 | 32));
        assert!(run_layer(&bad, true).is_err());
    }
    for bad_name in [
        "probe::strata_matrix::cohorts::large_shared::expert_large_shared_cohort::ExpertLargeSharedCohortFake<R>",
        "probe::strata_matrix::cohorts::large_shared::expert_large_shared_cohort::ExpertLargeSharedCohort<R>::tail",
        "probe::strata_matrix::nested::strata_matrix::cohorts::large_shared::expert_large_shared_cohort::ExpertLargeSharedCohort<R>",
        "probe::strata_matrix::cohorts::large_shared::expert_large_shared_cohort::ExpertLargeSharedCohort<R",
    ] {
        for down in [false, true] {
            assert_eq!(matrix_phase_variant(bad_name, down, true, 128), None);
        }
        let mut bad = shared_layer_program(&units);
        bad[3].name = bad_name;
        assert!(run_layer(&bad, true).is_err());
    }
    // Exact new names do not gain historical six-phase or packed-tail credit.
    for phase in 0..6 {
        assert!(
            !super::super::super::super::native_expert_batch::native_strata_expert_operator(
                SHARED, false, phase
            )
        );
    }
    for phase in [1usize, 2, 4] {
        let mut bad = shared_layer_program(&units);
        bad[phase].name = SHARED;
        assert!(run_layer(&bad, true).is_err());
    }
    let mut original = layer_program();
    for entry in &mut original {
        if entry.name == LARGE {
            entry.name = SHARED;
        }
    }
    assert!(run_layer(&original, true).is_err());
    let mut no_layer = shared_layer_program(&units);
    no_layer.remove(0);
    assert!(run_layer(&no_layer, true).is_err());
    let mut outside = NativeStrataExpertGrammar::default();
    assert!(
        outside
            .launch(SHARED, false, &program[3].count, &program[3].args)
            .is_err()
    );
}

#[cfg(all(
    feature = "kuno-layer-gather",
    not(feature = "kuno-large-shared-panels")
))]
#[test]
fn shared_large_disabled_identity_cannot_enter_old_large_or_packed_scope() {
    assert!(
        !super::super::super::super::native_expert_batch::native_strata_large_shared_cohort(SHARED)
    );
    for down in [false, true] {
        assert_eq!(matrix_phase_variant(SHARED, down, false, 128), None);
        assert_eq!(matrix_phase_variant(SHARED, down, true, 128), None);
    }
    let mut old = layer_program();
    for entry in &mut old {
        if entry.name == LARGE {
            entry.name = SHARED;
        }
    }
    assert!(run_layer(&old, true).is_err());
    #[cfg(feature = "kuno-large-packed-tiles")]
    {
        let mut packed = packed_layer_program(&packed_units(true));
        for entry in &mut packed {
            if entry.name == LARGE_PACKED && word(&entry.args.info, 11).unwrap() == 128 {
                entry.name = SHARED;
            }
        }
        assert!(run_layer(&packed, true).is_err());
    }
}
