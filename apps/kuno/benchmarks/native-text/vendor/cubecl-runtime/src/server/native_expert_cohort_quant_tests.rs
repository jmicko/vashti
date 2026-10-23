//! Genuine packed producers and seven-buffer consumers form one closed unit.
//! These controls validate metadata and ownership, never GPU arithmetic.
use super::*;
const QUANT: &str =
    "probe::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohort<Runtime>";

fn args(handles: Vec<Handle>, words: &[u32; 48], matrix: bool) -> KernelArguments {
    let mut all = words.to_vec();
    if matrix {
        let lengths: Vec<_> = handles
            .iter()
            .enumerate()
            .map(|(i, h)| (h.size_in_used() / if i == 0 { 8 } else { 4 }) as u32)
            .collect();
        all.extend(&lengths);
        all.extend(&lengths);
    }
    KernelArguments::new()
        .with_buffers(handles.into_iter().map(Handle::binding).collect())
        .with_info(MetadataBindingInfo::new(
            packed(&all),
            if matrix { 31 } else { 0 },
        ))
}

fn quant_program(members: &[u32]) -> Vec<Entry> {
    let owner = |n| Handle::new(StreamId::current(), n);
    let parent = owner(WORKING);
    let x = parent.clone().offset_end(WORKING - X);
    let hidden = parent.offset_start(X).offset_end(WORKING - X - H);
    let input = owner(WORKING);
    let image = owner(WORKING);
    let raw = owner(RAW);
    let map = owner(MAP_BYTES);
    let fault = owner(4);
    let grid = owner(32768);
    let addresses = owner(32768);
    let gu = owner(GU_BYTES);
    let part = owner(PART_BYTES);
    let status = owner(STATUS_BYTES);
    // Authentic IQ3_S GU rows: 2560/256*110=1100 bytes. Authentic
    // IQ4_NL DOWN rows: 640/32*18=360 bytes. All three roles share one slot.
    let gate = owner(640 * 1100);
    let up = owner(640 * 1100);
    let down = owner(2560 * 360);
    let mut w = [0; 48];
    w[..3].copy_from_slice(&[32, 2560, 1280]);
    w[6..11].copy_from_slice(&[
        176000,
        176000,
        582400,
        members.len() as u32,
        *members.iter().max().unwrap(),
    ]);
    let mut row = 0;
    for (i, &count) in members.iter().enumerate() {
        // Include resident sentinel8 and authentic distinct streamed slots.
        let slot = if i % 2 == 0 { i as u32 } else { 8 };
        w[16 + i * 4..20 + i * 4].copy_from_slice(&[i as u32, count, row, slot]);
        let tile = if count <= 8 { 8 } else { 16 };
        row += count.div_ceil(tile) * tile;
    }
    let z = members.len() as u32;
    let mut out = vec![Entry {
        name: "StrataExpertCohortGather",
        release: false,
        count: CubeCount::Static(w[10] * 5, 1, z),
        args: args(
            vec![map.clone(), input, x.clone(), fault.clone()],
            &w,
            false,
        ),
    }];
    for d in [false, true] {
        w[1] = if d { 640 } else { 2560 };
        w[2] = if d { 2560 } else { 1280 };
        w[3] = if d { 360 } else { 1100 };
        w[5] = if d { 2 } else { 0 };
        let mut handles = if d {
            vec![down.clone()]
        } else {
            vec![gate.clone(), up.clone()]
        };
        handles.extend([
            map.clone(),
            raw.clone(),
            image.clone(),
            fault.clone(),
            grid.clone(),
            addresses.clone(),
        ]);
        let mut a = args(handles, &w, false);
        a.read_dependencies
            .push(if d { down.clone() } else { gate.clone() }.binding());
        out.push(Entry {
            name: if d {
                "StrataExpertCohortDownCompact"
            } else {
                "StrataExpertCohortGuCompact"
            },
            release: d,
            count: CubeCount::Static(if d { 225 } else { 172 }, 1, z),
            args: a,
        });
    }
    for d in [false, true] {
        w[1] = if d { 640 } else { 2560 };
        w[2] = if d { 2560 } else { 1280 };
        w[3] = if d { 360 } else { 1100 };
        w[5] = if d { 2 } else { 0 };
        for tile in [8, 16, 32] {
            if !members.iter().any(|&m| {
                tile == if m <= 8 {
                    8
                } else if m <= 16 {
                    16
                } else {
                    32
                }
            }) {
                continue;
            }
            w[11] = tile;
            out.push(Entry {
                name: QUANT,
                release: false,
                count: CubeCount::Static(w[2] / 64, 1, z),
                args: args(
                    vec![
                        if d { hidden.clone() } else { x.clone() },
                        image.clone(),
                        if d { part.clone() } else { gu.clone() },
                        map.clone(),
                        status.clone(),
                        fault.clone(),
                        grid.clone(),
                    ],
                    &w,
                    true,
                ),
            });
        }
        if !d {
            w[3] = 0;
            w[11] = 0;
            out.push(Entry {
                name: "StrataExpertCohortActivate",
                release: false,
                count: CubeCount::Static((w[10] * 320).div_ceil(256), 1, z),
                args: args(
                    vec![
                        map.clone(),
                        gu.clone(),
                        hidden.clone(),
                        status.clone(),
                        fault.clone(),
                    ],
                    &w,
                    false,
                ),
            });
        }
    }
    out
}

fn complete(entries: &[Entry]) -> Result<NativeStrataExpertGrammar, ServerError> {
    let mut grammar = NativeStrataExpertGrammar::default();
    for e in entries {
        grammar.launch(e.name, e.release, &e.count, &e.args)?;
    }
    grammar.finish()?;
    Ok(grammar)
}
fn replace_owner(entry: &mut Entry, i: usize) {
    let old = &entry.args.buffers[i];
    entry.args.buffers[i] = Handle::new(old.stream, old.size)
        .offset_start(old.offset_start.unwrap_or(0))
        .offset_end(old.offset_end.unwrap_or(0))
        .binding();
}

#[test]
fn packed_cohort_accepts_all_tile_edges_and_actual_ten_launch_three_family_population() {
    for members in [
        &[1][..],
        &[8],
        &[9],
        &[16],
        &[17],
        &[31],
        &[3, 12, 31],
        &[1, 8, 9, 16, 17, 31, 3, 12],
    ] {
        let p = quant_program(members);
        let c = complete(&p).unwrap().finish().unwrap();
        assert_eq!(
            (c.cohort_groups, c.quant_cohort_groups),
            (members.len(), members.len())
        );
        assert_eq!((c.cohort_batches, c.quant_cohort_batches), (1, 1));
        assert_eq!(
            (c.launches, c.cohort_launches, c.quant_cohort_launches),
            (p.len(), p.len(), p.len())
        );
        assert_eq!(
            (c.logical_phases, c.matrix_pieces, c.gu_checks),
            (members.len() * 6, members.len() * 2, members.len())
        );
        assert_eq!(p.len(), if members.len() >= 3 { 10 } else { 6 });
    }
}

#[test]
fn packed_cohort_rejects_mixed_copy_and_f16_matrix_families_and_closed_type_forgeries() {
    for (i, name) in [
        (1, "StrataExpertCohortGuImage"),
        (2, "StrataExpertCohortDownImage"),
        (3, ORIGINAL),
        (7, ORDERED),
    ] {
        let mut p = quant_program(&[3, 12, 31]);
        p[i].name = name;
        assert!(complete(&p).is_err());
    }
    for name in [
        "probe::quant_cohort::expert_quant_cohort::ExpertQuantCohort<Runtime>",
        "probe::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohortWrong<Runtime>",
        "probe::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohort<Runtime>::other",
        "probe::strata_matrix::quant_cohort::expert_quant_cohort::ExpertQuantCohort<Runtime><Runtime>",
        "probe::strata_matrix::quant_cohort::cohorts::expert_cohort::ExpertCohort<Runtime>",
    ] {
        assert!(!quant_matrix(name));
        assert_eq!(matrix_variant(name, false), None);
        let mut p = quant_program(&[3, 12, 31]);
        p[3].name = name;
        assert!(complete(&p).is_err());
    }
    let mut old = program(false, true);
    old[3].name = QUANT;
    assert!(complete(&old).is_err());
    for phase in 0..6 {
        assert!(
            !crate::server::native_expert_batch::native_strata_expert_operator(QUANT, false, phase)
        );
        assert!(
            !crate::server::native_expert_batch::native_strata_expert_operator(
                "probe::strata_matrix::quant_cohort::expert::Expert<Runtime>",
                false,
                phase
            )
        );
    }
}

#[test]
fn packed_cohort_requires_checked_activation_and_exact_release_role_and_physical_grid() {
    let mut p = quant_program(&[3, 12, 31]);
    p.remove(6);
    assert!(complete(&p).is_err());
    for i in 0..10 {
        let mut p = quant_program(&[3, 12, 31]);
        p[i].release = !p[i].release;
        assert!(complete(&p).is_err());
        let mut p = quant_program(&[3, 12, 31]);
        p[i].count = CubeCount::Static(1, 1, 1);
        assert!(complete(&p).is_err());
    }
    let mut p = quant_program(&[3, 12, 31]);
    p[6].name = "StrataPromptExpertF16Swiglu";
    assert!(complete(&p).is_err());
    for i in [1, 2, 3, 4, 5, 7, 8, 9] {
        let mut p = quant_program(&[3, 12, 31]);
        let role = word(&p[i].args.info, 5).unwrap();
        set_word(&mut p[i].args.info, 5, if role == 0 { 2 } else { 0 });
        assert!(complete(&p).is_err());
    }
    // Tile16 and tile32 have Y=1, never the original two-row tile16 grid.
    for i in [4, 5, 8, 9] {
        let mut p = quant_program(&[3, 12, 31]);
        p[i].count = CubeCount::Static(if i < 6 { 20 } else { 40 }, 2, 3);
        assert!(complete(&p).is_err());
    }
}

#[test]
fn packed_cohort_binds_all_seven_matrix_owners_and_the_original_disjoint_hidden_view() {
    for i in [4, 5, 8, 9] {
        for owner in 0..7 {
            let mut p = quant_program(&[3, 12, 31]);
            replace_owner(&mut p[i], owner);
            assert!(complete(&p).is_err(), "phase {i} owner {owner}");
        }
    }
    for owner in 0..5 {
        let mut p = quant_program(&[3, 12, 31]);
        replace_owner(&mut p[6], owner);
        assert!(complete(&p).is_err());
    }
}

#[test]
fn packed_cohort_authenticates_quant_spans_array_units_and_metadata_offsets() {
    for i in [1, 2, 3, 4, 5, 7, 8, 9] {
        for field in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16] {
            let mut p = quant_program(&[3, 12, 31]);
            let old = word(&p[i].args.info, field).unwrap();
            set_word(&mut p[i].args.info, field, old + 1);
            assert!(complete(&p).is_err(), "phase {i} word {field}");
        }
    }
    for i in [3, 4, 5, 7, 8, 9] {
        for field in 48..62 {
            let mut p = quant_program(&[3, 12, 31]);
            let old = word(&p[i].args.info, field).unwrap();
            set_word(&mut p[i].args.info, field, old + 1);
            assert!(complete(&p).is_err());
        }
        let mut p = quant_program(&[3, 12, 31]);
        p[i].args.info.dynamic_metadata_offset = 30;
        assert!(complete(&p).is_err());
        let mut p = quant_program(&[3, 12, 31]);
        p[i].args.buffers.pop();
        assert!(complete(&p).is_err());
        let mut p = quant_program(&[3, 12, 31]);
        p[i].args
            .read_dependencies
            .push(Handle::new(StreamId::current(), 4).binding());
        assert!(complete(&p).is_err());
    }
}

#[test]
fn packed_cohort_retains_whole_raw_image_table_and_source_dependency_inventory() {
    for owner in 1..7 {
        let mut p = quant_program(&[3, 12, 31]);
        replace_owner(&mut p[2], owner);
        assert!(complete(&p).is_err());
    }
    for i in [1, 2] {
        let mut p = quant_program(&[3, 12, 31]);
        p[i].args.read_dependencies.clear();
        assert!(complete(&p).is_err());
        let mut p = quant_program(&[3, 12, 31]);
        p[i].args.buffers.pop();
        assert!(complete(&p).is_err());
        let raw = if i == 1 { 3 } else { 2 };
        let mut p = quant_program(&[3, 12, 31]);
        p[i].args.buffers[raw] = Handle::new(StreamId::current(), RAW + 16)
            .offset_start(16)
            .binding();
        assert!(complete(&p).is_err());
    }
    // Original canonical MAP group, unique RAW slots and padded row partition.
    for field in [16, 18, 20, 22, 23, 28] {
        let mut p = quant_program(&[3, 12, 31]);
        let old = word(&p[0].args.info, field).unwrap();
        set_word(&mut p[0].args.info, field, old + 1);
        assert!(complete(&p).is_err());
    }
}
