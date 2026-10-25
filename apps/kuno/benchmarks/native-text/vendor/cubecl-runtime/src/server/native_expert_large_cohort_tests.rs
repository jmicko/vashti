//! Real closed large cohort grammar, CPU metadata only; no GPU claim.
use super::*;
use crate::server::Handle;
use alloc::{vec, vec::Vec};
const LARGE: &str =
    "probe::strata_matrix::cohorts::large::expert_large_cohort::ExpertLargeCohort<Runtime>";
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
        prefix += members.div_ceil(32) * 32;
    }
    let mut out = vec![Entry {
        name: "StrataExpertLargeCohortGather",
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
                "StrataExpertLargeCohortDownDirect8Image"
            } else {
                "StrataExpertLargeCohortGuDirect8Image"
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
        for tile in [128u32,8,16,32] {
            let part_rows=counts.iter().map(|&m| match tile {
                128=>m/128,8 if (1..=8).contains(&(m%128))=>1,
                16 if (9..32).contains(&(m%128))=>(m%128).div_ceil(16),
                32 if m%128>=32=>(m%128).div_ceil(32),_=>0,
            }).max().unwrap();
            if part_rows==0 {continue;}
            w[11]=tile;
        out.push(Entry {
            name: LARGE,
            release: false,
            count: CubeCount::Static(w[2] / if tile==128 {128}else{64}, part_rows, n),
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
        }
        if !d {
            w[11] = 0;
            out.push(Entry {
                name: "StrataExpertLargeCohortActivate",
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
fn large_cohort_exact_prefix_tail_chains_and_receipts() {
    for (counts,launches) in [(vec![256],6),(vec![257],8),(vec![265],8),(vec![288],8),
        (vec![256,257,265,288],12),(vec![512;8],6),(vec![4095],8),(vec![4096],6)] {
        let p=program(&counts);let c=run(&p).unwrap();
        assert_eq!(p.len(),launches);assert_eq!(c.large_cohort_launches,launches);
        assert_eq!(c.large_cohort_groups,counts.len());assert_eq!(c.large_cohort_batches,1);
        assert_eq!(c.regular_cohort_groups,0);assert_eq!(c.logical_phases,counts.len()*6);
        // Every omitted role or early release must reject even if later roles
        // retain valid owners and scalar headers.
        for i in 0..p.len() {
            let mut omitted=program(&counts);omitted.remove(i);assert!(run(&omitted).is_err());
            let mut release=program(&counts);release[i].release=!release[i].release;assert!(run(&release).is_err());
        }
    }
}
#[test]
fn large_cohort_capacity_map_and_closed_role_negatives() {
    for counts in [vec![255],vec![0],vec![4097],vec![1025;4]] {
        let p=program(&counts);assert!(run(&p).is_err());
    }
    for index in [0usize,4,9,10,12,13,16,17,18,19,47] {
        let mut p=program(&[256,257,265,288]);set_word(&mut p[0].args.info,index,u32::MAX);
        assert!(run(&p).is_err());
    }
    for i in 4..12 {if i==7 {continue;}
        for binding in 0..6 {
            if i==8 && binding==2 {continue;} // first DOWN establishes its output owner
            let mut p=program(&[256,257,265,288]);
            let old=&p[i].args.buffers[binding];
            p[i].args.buffers[binding]=Handle::new(old.stream,old.size_in_used()).binding();
            assert!(run(&p).is_err());
        }
        let mut p=program(&[256,257,265,288]);p[i].count=CubeCount::Static(1,1,1);assert!(run(&p).is_err());
    }
    for name in ["probe::strata_matrix::cohorts::large::expert_large_cohort::ExpertLargeCohortFake<R>",
        "probe::strata_matrix::cohorts::large::expert_large_cohort::ExpertLargeCohort<R>::extra",
        "probe::strata_matrix::nested::strata_matrix::cohorts::large::expert_large_cohort::ExpertLargeCohort<R>"] {
        assert_eq!(matrix_variant(name,false),None);
    }
}
