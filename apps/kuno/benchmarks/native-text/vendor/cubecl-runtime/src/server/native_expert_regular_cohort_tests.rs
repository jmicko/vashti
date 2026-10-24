//! Real closed regular cohort grammar, CPU metadata only; no GPU claim.
use super::*;
use crate::server::Handle;
use alloc::{vec, vec::Vec};
const REGULAR: &str =
    "probe::strata_matrix::cohorts::regular::expert_regular_cohort::ExpertRegularCohort<Runtime>";
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

fn program(counts: &[u32]) -> Vec<Entry> {
    let owner = |n| Handle::new(StreamId::current(), n);
    let parent = owner(WORKING);
    let x = parent.clone().offset_end(WORKING - X);
    let h = parent.offset_start(X).offset_end(WORKING - X - H);
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
    let n = counts.len() as u32;
    let maximum = *counts.iter().max().unwrap();
    let mut w = [0; 48];
    w[..3].copy_from_slice(&[4096, 2560, 1280]);
    w[6..11].copy_from_slice(&[100, 100, 300, n, maximum]);
    let mut prefix = 0;
    for (i, &members) in counts.iter().enumerate() {
        w[16 + i * 4..20 + i * 4].copy_from_slice(&[i as u32, members, prefix, i as u32]);
        prefix += members.div_ceil(16) * 16;
    }
    let mut out = vec![Entry {
        name: "StrataExpertRegularCohortGather",
        release: false,
        count: CubeCount::Static(maximum * 5, 1, n),
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
        let mut args = arguments(handles, &w, false);
        args.read_dependencies.push(gate.clone().binding());
        out.push(Entry {
            name: if d {
                "StrataExpertRegularCohortDownDirect8Image"
            } else {
                "StrataExpertRegularCohortGuDirect8Image"
            },
            release: d,
            count: CubeCount::Static((w[1] * w[2] / 8).div_ceil(32), 1, n),
            args,
        });
    }
    for d in [false, true] {
        w[1] = if d { 640 } else { 2560 };
        w[2] = if d { 2560 } else { 1280 };
        w[3] = 0;
        w[5] = if d { 2 } else { 0 };
        w[11] = 16;
        out.push(Entry {
            name: REGULAR,
            release: false,
            count: CubeCount::Static(w[2] / 64, maximum.div_ceil(16), n),
            args: arguments(
                vec![
                    if d { h.clone() } else { x.clone() },
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
        if !d {
            w[11] = 0;
            out.push(Entry {
                name: "StrataExpertRegularCohortActivate",
                release: false,
                count: CubeCount::Static((maximum * 320).div_ceil(256), 1, n),
                args: arguments(
                    vec![
                        map.clone(),
                        gu.clone(),
                        h.clone(),
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
fn run(p: &[Entry]) -> Result<NativeStrataExpertGrammarCounts, ServerError> {
    let mut g = NativeStrataExpertGrammar::default();
    for e in p {
        g.launch(e.name, e.release, &e.count, &e.args)?;
    }
    g.finish()
}
fn set_word(info: &mut MetadataBindingInfo, i: usize, v: u32) {
    let mut b = info.data[i / 2].to_ne_bytes();
    let at = i % 2 * 4;
    b[at..at + 4].copy_from_slice(&v.to_ne_bytes());
    info.data[i / 2] = u64::from_ne_bytes(b);
}
#[test]
fn regular_cohort_real_six_phases_admit_mixed_boundaries_and_report_only_genuine_new_counts() {
    for counts in [
        vec![32],
        vec![255; 8],
        vec![32, 33, 63, 64, 65, 127, 128, 255],
        vec![33, 255, 64],
    ] {
        let p = program(&counts);
        assert_eq!(p.len(), 6);
        let c = run(&p).unwrap();
        assert_eq!(
            (c.cohort_groups, c.regular_cohort_groups),
            (counts.len(), counts.len())
        );
        assert_eq!((c.cohort_batches, c.regular_cohort_batches), (1, 1));
        assert_eq!(
            (c.launches, c.cohort_launches, c.regular_cohort_launches),
            (6, 6, 6)
        );
        assert_eq!(c.logical_phases, 6 * counts.len());
        assert_eq!(c.quant_cohort_groups, 0);
    }
    for counts in [vec![31], vec![256], vec![1, 17], vec![0]] {
        assert!(run(&program(&counts)).is_err());
    }
}
#[test]
fn regular_cohort_rejects_all_omitted_reordered_release_and_legacy_role_phases() {
    for i in 0..6 {
        let mut p = program(&[33, 255]);
        p.remove(i);
        assert!(run(&p).is_err());
        let mut p = program(&[33, 255]);
        p[i].release = !p[i].release;
        assert!(run(&p).is_err());
    }
    for i in 0..5 {
        let mut p = program(&[33, 255]);
        p.swap(i, i + 1);
        assert!(run(&p).is_err());
    }
    for (i, name) in [
        (0, "StrataExpertCohortGather"),
        (1, "StrataExpertCohortGuDirect8Image"),
        (2, "StrataExpertRegularCohortDownImage"),
        (
            3,
            "probe::strata_matrix::cohorts::granular_small::expert_granular_cohort::ExpertGranularCohort<Runtime>",
        ),
        (4, "StrataExpertCohortActivate"),
        (
            5,
            "probe::strata_matrix::cohorts::expert_cohort::ExpertCohort<Runtime>",
        ),
    ] {
        let mut p = program(&[33, 255]);
        p[i].name = name;
        assert!(run(&p).is_err());
    }
}
#[test]
fn regular_cohort_requires_real_maximum_grid_and_every_same_original_matrix_owner() {
    for index in [3usize, 5] {
        for owner in 0..6 {
            let mut p = program(&[33, 255]);
            let size = View::new(&p[index].args.buffers[owner]).unwrap().used();
            if index == 5 && owner == 2 {
                // The final PART sink is first established here: this local
                // grammar cannot know a caller's unstated expected allocation.
                // Actual graph resource admission owns/charges this handle;
                // whole GPU/model checks prove the original consumer wiring.
                p[index].args.buffers[owner] = Handle::new(StreamId::current(), size).binding();
                assert!(run(&p).is_ok());
                // Reject an incorrect full output span and stale array ABI.
                p[index].args.buffers[owner] = Handle::new(StreamId::current(), size - 4).binding();
            } else {
                p[index].args.buffers[owner] = Handle::new(StreamId::current(), size).binding();
            }
            assert!(run(&p).is_err(), "matrix {index} owner {owner}");
        }
        for field in [1usize, 2, 4, 5, 9, 10, 11, 16, 48, 54] {
            let mut p = program(&[33, 255]);
            let v = word(&p[index].args.info, field).unwrap();
            set_word(&mut p[index].args.info, field, v + 1);
            assert!(run(&p).is_err());
        }
        for y in [1, 2, 15, 17] {
            let mut p = program(&[33, 255]);
            let CubeCount::Static(x, _, z) = p[index].count else {
                panic!("static")
            };
            p[index].count = CubeCount::Static(x, y, z);
            assert!(run(&p).is_err());
        }
    }
}
#[test]
fn regular_cohort_common_info_cannot_alias_raw_slots_rows_or_spare_records() {
    for (i, v) in [
        (0, 31),
        (4, 1),
        (6, 101),
        (7, 101),
        (8, 200),
        (9, 0),
        (9, 9),
        (10, 254),
        (12, 2),
        (13, 1),
        (19, 9),
        (22, 0),
        (23, 0),
        (24, 1),
    ] {
        let mut p = program(&[33, 255]);
        for e in &mut p {
            set_word(&mut e.args.info, i, v);
        }
        assert!(run(&p).is_err(), "field {i}");
    }
    for index in [0usize, 1, 2, 4] {
        for owner in 0..program(&[33, 255])[index].args.buffers.len() {
            let mut p = program(&[33, 255]);
            p[index].args.buffers[owner].stream.value =
                p[index].args.buffers[owner].stream.value.wrapping_add(1);
            assert!(run(&p).is_err(), "phase {index} source {owner}");
        }
    }
}
#[test]
fn regular_cohort_exact_new_entry_cannot_accept_suffix_or_nested_legacy_forgery() {
    for n in [
        "probe::strata_matrix::cohorts::regular::expert_regular_cohort::ExpertRegularCohortFake<Runtime>",
        "probe::strata_matrix::cohorts::regular::expert_regular_cohort::ExpertRegularCohort<Runtime>::extra",
        "probe::strata_matrix::native::expert_macro::strata_matrix::cohorts::regular::expert_regular_cohort::ExpertRegularCohort<Runtime>",
        "probe::strata_matrix::cohorts::regular::expert_cohort::ExpertCohort<Runtime>",
    ] {
        assert_eq!(matrix_variant(n, false), None);
        assert_eq!(matrix_variant(n, true), None);
        let mut p = program(&[33, 255]);
        p[3].name = n;
        assert!(run(&p).is_err());
    }
}

#[cfg(feature="kuno-small-dual-cells")]
const DUAL:&str="probe::strata_matrix::cohorts::dual_regular::expert_dual_regular_cohort::ExpertDualRegularCohort<Runtime>";
#[cfg(feature="kuno-small-dual-cells")]
fn dual_program()->Vec<Entry>{let mut p=program(&[32,33,63,64,65,127,128,255]);for e in &mut p{if e.name==REGULAR {e.name=DUAL;let CubeCount::Static(x,y,z)=e.count else{panic!("literal grid")};e.count=CubeCount::Static(x*2,y,z);}}p}
#[cfg(feature="kuno-small-dual-cells")]
#[test]
fn dual_regular_exact_original_phase_and_owners_accept_actual_n32_grid(){run(&dual_program()).unwrap();assert_eq!(matrix_variant(DUAL,true),Some(true));}
#[cfg(feature="kuno-small-dual-cells")]
#[test]
fn dual_regular_refuses_old_grid_nested_aliases_and_changed_phase_owners(){
 for index in [3usize,5] {let mut p=dual_program();let CubeCount::Static(x,y,z)=p[index].count else{panic!("literal grid")};p[index].count=CubeCount::Static(x/2,y,z);assert!(run(&p).is_err());}
 for name in ["probe::strata_matrix::cohorts::dual_regular::expert_dual_regular_cohort::ExpertDualRegularCohortFake<R>","probe::strata_matrix::cohorts::dual_regular::expert_dual_regular_cohort::ExpertDualRegularCohort<R>::tail","probe::strata_matrix::nested::strata_matrix::cohorts::dual_regular::expert_dual_regular_cohort::ExpertDualRegularCohort<R>"] {assert_eq!(matrix_variant(name,false),None);}
 let mut p=dual_program();p[5].args.buffers[0]=Handle::new(StreamId::current(),H).binding();assert!(run(&p).is_err());
}
