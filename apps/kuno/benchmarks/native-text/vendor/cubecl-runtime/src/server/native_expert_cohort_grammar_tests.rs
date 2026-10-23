//! CPU metadata controls of the actual cohort grammar, not GPU math evidence.
use super::*;
use crate::server::Handle;
use alloc::{vec, vec::Vec};

const ORIGINAL: &str = "probe::strata_matrix::cohorts::expert_cohort::ExpertCohort<Runtime>";
const ORDERED: &str =
    "probe::strata_matrix::cohorts::ordered::expert_cohort::ExpertCohort<Runtime>";

struct Entry {
    name: &'static str,
    release: bool,
    count: CubeCount,
    args: KernelArguments,
}
fn packed(words: &[u32]) -> Vec<u64> {
    words
        .chunks_exact(2)
        .map(|v| {
            let mut b = [0; 8];
            b[..4].copy_from_slice(&v[0].to_ne_bytes());
            b[4..].copy_from_slice(&v[1].to_ne_bytes());
            u64::from_ne_bytes(b)
        })
        .collect()
}
fn arguments(handles: Vec<Handle>, words: &[u32; 48], matrix: bool) -> KernelArguments {
    let mut all = words.to_vec();
    if matrix {
        let lengths = handles
            .iter()
            .enumerate()
            .map(|(i, h)| (h.size_in_used() / if i < 2 { 8 } else { 4 }) as u32)
            .collect::<Vec<_>>();
        all.extend_from_slice(&lengths);
        all.extend_from_slice(&lengths);
    }
    KernelArguments::new()
        .with_buffers(handles.into_iter().map(Handle::binding).collect())
        .with_info(if matrix {
            MetadataBindingInfo::new(packed(&all), 30)
        } else {
            MetadataBindingInfo::custom(packed(&all))
        })
}
fn program(quad: bool, ordered: bool) -> Vec<Entry> {
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
    let gate = owner(400);
    let up = owner(400);
    let down = owner(400);
    let mut w = [0; 48];
    w[..3].copy_from_slice(&[32, 2560, 1280]);
    w[6..11].copy_from_slice(&[100, 100, 300, 2, 17]);
    w[16..20].copy_from_slice(&[0, 3, 0, 8]);
    w[20..24].copy_from_slice(&[1, 17, 8, 8]);
    let mut out = vec![Entry {
        name: "StrataExpertCohortGather",
        release: false,
        count: CubeCount::Static(85, 1, 2),
        args: arguments(
            vec![map.clone(), input, x.clone(), fault.clone()],
            &w,
            false,
        ),
    }];
    for d in [false, true] {
        w[1] = if d { 640 } else { 2560 };
        w[2] = if d { 2560 } else { 1280 };
        w[3] = 1;
        w[5] = if d { 2 } else { 0 };
        let mut owners = if d {
            vec![down.clone()]
        } else {
            vec![gate.clone(), up.clone()]
        };
        owners.extend([
            map.clone(),
            raw.clone(),
            image.clone(),
            fault.clone(),
            grid.clone(),
            addresses.clone(),
        ]);
        let mut args = arguments(owners, &w, false);
        args.read_dependencies.push(gate.clone().binding());
        out.push(Entry {
            name: match (d, quad) {
                (false, false) => "StrataExpertCohortGuImage",
                (true, false) => "StrataExpertCohortDownImage",
                (false, true) => "StrataExpertCohortGuQuad32Image",
                (true, true) => "StrataExpertCohortDownQuad32Image",
            },
            release: d,
            count: CubeCount::Static(
                (w[1] * w[2] / if quad { 8 } else { 32 }).div_ceil(256),
                1,
                2,
            ),
            args,
        });
    }
    for d in [false, true] {
        w[1] = if d { 640 } else { 2560 };
        w[2] = if d { 2560 } else { 1280 };
        w[3] = 0;
        w[5] = if d { 2 } else { 0 };
        for tile in [8, 16] {
            w[11] = tile;
            out.push(Entry {
                name: if d && ordered { ORDERED } else { ORIGINAL },
                release: false,
                count: CubeCount::Static(w[2] / 64, if tile == 8 { 1 } else { 2 }, 2),
                args: arguments(
                    vec![
                        if d { hidden.clone() } else { x.clone() },
                        image.clone(),
                        if d { part.clone() } else { gu.clone() },
                        map.clone(),
                        status.clone(),
                        fault.clone(),
                    ],
                    &w,
                    true,
                ),
            });
        }
        if !d {
            w[11] = 0;
            out.push(Entry {
                name: "StrataExpertCohortActivate",
                release: false,
                count: CubeCount::Static((17u32 * 320).div_ceil(256), 1, 2),
                args: arguments(
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
fn run(entries: &[Entry]) -> Result<(), ServerError> {
    let mut cohort = Cohort::start(0, &entries[0].args)?;
    for (i, e) in entries.iter().enumerate() {
        let complete = cohort.launch(e.name, e.release, &e.count, &e.args)?;
        assert_eq!(complete, i + 1 == entries.len());
    }
    assert_eq!(cohort.groups(), 2);
    Ok(())
}
#[cfg(feature = "kuno-expert-ordered-output")]
fn set_word(info: &mut MetadataBindingInfo, index: usize, value: u32) {
    let mut bytes = info.data[index / 2].to_ne_bytes();
    let at = index % 2 * 4;
    bytes[at..at + 4].copy_from_slice(&value.to_ne_bytes());
    info.data[index / 2] = u64::from_ne_bytes(bytes);
}

#[test]
fn original_cohort_identity_grid_owners_and_both_small_families_remain_accepted() {
    let entries = program(false, false);
    assert!(matches!(entries[1].count, CubeCount::Static(400, 1, 2)));
    assert!(matches!(entries[2].count, CubeCount::Static(200, 1, 2)));
    run(&entries).unwrap();
}

#[cfg(feature = "kuno-dequant-quad32")]
#[test]
fn quad_cohort_authenticates_genuine_new_names_and_four_times_original_grid() {
    let entries = program(true, false);
    assert!(matches!(entries[1].count, CubeCount::Static(1600, 1, 2)));
    assert!(matches!(entries[2].count, CubeCount::Static(800, 1, 2)));
    run(&entries).unwrap();
    for index in [1, 2] {
        let mut wrong = program(true, false);
        wrong[index].count = if index == 1 {
            CubeCount::Static(400, 1, 2)
        } else {
            CubeCount::Static(200, 1, 2)
        };
        assert!(run(&wrong).is_err());
        let mut old = program(false, false);
        old[index].count = if index == 1 {
            CubeCount::Static(1600, 1, 2)
        } else {
            CubeCount::Static(800, 1, 2)
        };
        assert!(run(&old).is_err());
    }
}

#[cfg(feature = "kuno-dequant-quad32")]
#[test]
fn quad_cohort_rejects_reversed_roles_and_mixed_image_families() {
    for index in [1, 2] {
        let mut wrong = program(true, false);
        wrong[index].name = if index == 1 {
            "StrataExpertCohortDownQuad32Image"
        } else {
            "StrataExpertCohortGuQuad32Image"
        };
        assert!(run(&wrong).is_err());
    }
    for quad in [false, true] {
        let mut mixed = program(quad, false);
        let mut other = program(!quad, false);
        core::mem::swap(&mut mixed[2].name, &mut other[2].name);
        core::mem::swap(&mut mixed[2].count, &mut other[2].count);
        assert!(run(&mixed).is_err());
    }
}

#[cfg(not(feature = "kuno-dequant-quad32"))]
#[test]
fn disabled_quad_cohort_identity_is_rejected_even_with_its_matching_grid() {
    assert!(run(&program(true, false)).is_err());
}

#[cfg(feature = "kuno-expert-ordered-output")]
#[test]
fn ordered_cohort_down_preserves_both_family_owners_info_and_complete_phase_order() {
    run(&program(false, true)).unwrap();
    #[cfg(feature = "kuno-dequant-quad32")]
    run(&program(true, true)).unwrap();
    for index in [6, 7] {
        let mut wrong = program(false, true);
        wrong[index].name = ORIGINAL;
        assert!(run(&wrong).is_err());
    }
}

#[cfg(feature = "kuno-expert-ordered-output")]
#[test]
fn ordered_cohort_is_down_only_closed_identity_with_original_owner_and_schema_checks() {
    for index in [3, 4] {
        let mut wrong = program(false, true);
        wrong[index].name = ORDERED;
        assert!(run(&wrong).is_err());
    }
    for name in [
        "probe::cohorts::ordered::expert_cohort::ExpertCohort<Runtime>",
        "probe::strata_matrix::cohorts::ordered::other::expert_cohort::ExpertCohort<Runtime>",
        "probe::strata_matrix::cohorts::ordered::expert_cohort::ExpertCohortWrong<Runtime>",
        "probe::strata_matrix::cohorts::ordered::expert_cohort::ExpertCohort<Runtime>::other",
    ] {
        let mut wrong = program(false, true);
        wrong[6].name = name;
        assert!(run(&wrong).is_err());
    }
    for owner in 0..6 {
        let mut wrong = program(false, true);
        let old = &wrong[6].args.buffers[owner];
        wrong[6].args.buffers[owner] = Handle::new(old.stream, old.size)
            .offset_start(old.offset_start.unwrap_or(0))
            .offset_end(old.offset_end.unwrap_or(0))
            .binding();
        assert!(run(&wrong).is_err());
    }
    for word_index in [0, 1, 2, 3, 4, 5, 9, 11, 16, 48, 54] {
        let mut wrong = program(false, true);
        let old = word(&wrong[6].args.info, word_index).unwrap();
        set_word(&mut wrong[6].args.info, word_index, old + 1);
        assert!(run(&wrong).is_err());
    }
}

#[cfg(not(feature = "kuno-expert-ordered-output"))]
#[test]
fn disabled_ordered_cohort_is_rejected_without_changing_original_identity() {
    assert!(run(&program(false, true)).is_err());
}

#[cfg(feature="kuno-dequant-direct8")]
#[path="native_expert_cohort_direct8_tests.rs"]
mod direct8;

#[cfg(all(feature="kuno-small-direct-publish",feature="kuno-expert-ordered-output"))]
#[path="native_expert_cohort_small_direct_tests.rs"]
mod small_direct;

#[cfg(all(feature="kuno-small-quantized-matrix",feature="kuno-expert-ordered-output"))]
#[path="native_expert_cohort_quant_tests.rs"]
mod quant;
