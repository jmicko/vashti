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

#[cfg(feature="kuno-layer-gather")]
fn layer_program()->Vec<Entry> {
    let phase=Handle::new(StreamId::current(),780*1024*1024);
    let persistent=Handle::new(StreamId::current(),756*1024*1024);
    let view=|owner:&Handle,first:u64,bytes:u64|owner.clone().offset_start(first).offset_end(owner.size()-first-bytes).binding();
    let map=view(&phase,80*1024*1024+GU_BYTES+PART_BYTES+512*1024+3*256,MAP_BYTES);
    let xfirst=80*1024*1024+GU_BYTES+PART_BYTES+4*(131072+65536+65536+131072)+6*256;
    let x=view(&phase,xfirst,56832*2560*2);
    let h=view(&persistent,20*1024*1024,H);
    let input=Handle::new(StreamId::current(),WORKING).binding();
    let fault=Handle::new(StreamId::current(),4).binding();
    let mut out=vec![Entry{name:"StrataExpertLayerGather",release:false,count:CubeCount::Static(5,40960,1),args:KernelArguments::new().with_buffers(vec![map.clone(),input.clone(),x.clone(),h.clone(),fault.clone()]).with_info(MetadataBindingInfo::custom(packed(&[4096,10,0,40960,0,0,0,0])))}];
    for group in 0..10 {
        let mut p=program(&[4096]);
        let old_map=View::new(&p[0].args.buffers[0]).unwrap();
        let old_x=View::new(&p[0].args.buffers[2]).unwrap();
        let old_fault=View::new(&p[0].args.buffers[3]).unwrap();
        let old_h=View::new(&p.iter().find(|e|e.name=="StrataExpertLargeCohortActivate").unwrap().args.buffers[2]).unwrap();
        p.remove(0);
        for e in &mut p {
            set_word(&mut e.args.info,4,group);set_word(&mut e.args.info,16,group);set_word(&mut e.args.info,13,1);
            for binding in &mut e.args.buffers {
                let v=View::new(binding).unwrap();
                *binding=if v==old_map {map.clone()}else if v==old_x {x.clone()}else if v==old_h {h.clone()}else if v==old_fault {fault.clone()}else{binding.clone()};
            }
            if e.name==LARGE {
                for (i,b) in e.args.buffers.iter().enumerate() {
                    let len=(b.size_in_used()/if i<2 {8}else{4}) as u32;
                    set_word(&mut e.args.info,48+i,len);set_word(&mut e.args.info,54+i,len);
                }
            }
        }
        out.extend(p);
    }
    out
}
#[cfg(feature="kuno-layer-gather")]
fn run_layer(p:&[Entry],complete:bool)->Result<NativeStrataExpertGrammarCounts,ServerError> {
    let mut g=NativeStrataExpertGrammar::default();
    for e in p {g.launch(e.name,e.release,&e.count,&e.args)?;}
    if complete {g.finish_batch()}else{g.finish()}
}
#[cfg(feature="kuno-layer-gather")]
#[test]
fn layer_gather_outer_grammar_retains_real_owners_and_completes_entire_layer() {
    let p=layer_program();let c=run_layer(&p,true).unwrap();
    assert_eq!(c.layer_gather_launches,1);assert_eq!(c.large_cohort_groups,10);
    assert_eq!(c.large_cohort_batches,10);assert_eq!(c.large_cohort_launches,50);
    assert_eq!(c.launches,51);assert_eq!(c.logical_phases,60);
    assert!(run_layer(&p[..6],false).is_ok());assert!(run_layer(&p[..6],true).is_err());
    for i in [0usize,1,2,3,4,5,50] {
        let mut bad=layer_program();bad.remove(i);assert!(run_layer(&bad,true).is_err());
    }
}
#[cfg(feature="kuno-layer-gather")]
#[test]
fn layer_gather_outer_grammar_rejects_local_fallback_alias_and_unbounded_grid() {
    for (index,value) in [(0,0),(1,0),(2,2),(3,56864),(4,1)] {
        let mut bad=layer_program();set_word(&mut bad[0].args.info,index,value);assert!(run_layer(&bad,true).is_err());
    }
    for grid in [CubeCount::Static(40960*5,1,1),CubeCount::Static(5,40959,1),CubeCount::Static(5,40960,2)] {
        let mut bad=layer_program();bad[0].count=grid;assert!(run_layer(&bad,true).is_err());
    }
    for phase in [1usize,2,3,4,5] {
        for index in [12usize,13] {let mut bad=layer_program();set_word(&mut bad[phase].args.info,index,if index==12 {1}else{0});assert!(run_layer(&bad,true).is_err());}
    }
    for binding in [0usize,2,3,4] {
        let mut bad=layer_program();let old=&bad[0].args.buffers[binding];bad[0].args.buffers[binding]=Handle::new(old.stream,old.size_in_used()).binding();assert!(run_layer(&bad,true).is_err());
    }
    let mut bad=layer_program();bad[1].name="StrataExpertLargeCohortGather";assert!(run_layer(&bad,true).is_err());
    let mut bad=layer_program();bad[1].name="StrataExpertCohortGather";assert!(run_layer(&bad,true).is_err());
    let mut bad=layer_program();bad[0].args.buffers[1]=Handle::new(StreamId::current(),4).binding();assert!(run_layer(&bad,true).is_err());
    let mut bad=layer_program();bad[0].args.buffers[1]=bad[0].args.buffers[3].clone();assert!(run_layer(&bad,true).is_err());
    let mut bad=layer_program();bad[3].args.buffers[0]=bad[0].args.buffers[3].clone();assert!(run_layer(&bad,true).is_err());
    let mut bad=layer_program();bad[0].release=true;assert!(run_layer(&bad,true).is_err());
}

#[cfg(feature="kuno-large-packed-tiles")]
const LARGE_PACKED: &str = "probe::strata_matrix::cohorts::large_packed::expert_large_packed_cohort::ExpertLargePackedCohort<Runtime>";

#[cfg(feature="kuno-large-packed-tiles")]
fn packed_layer_program(units: &[Vec<u32>]) -> Vec<Entry> {
    let mut owner_program = layer_program();
    let mut layer = owner_program.remove(0);
    layer.name = "StrataExpertLayerPackedGather";
    set_word(&mut layer.args.info, 4, 1);
    let groups: usize = units.iter().map(Vec::len).sum();
    let members: u32 = units.iter().flatten().sum();
    assert_eq!(members, 40960); // Genuine 4096-token/top10 population.
    let padded = units.iter().flatten().map(|m| m.div_ceil(32)*32).sum();
    set_word(&mut layer.args.info, 1, groups as u32);
    set_word(&mut layer.args.info, 3, padded);
    layer.count = CubeCount::Static(5, padded, 1);
    let map = layer.args.buffers[0].clone();
    let x = layer.args.buffers[2].clone();
    let h = layer.args.buffers[3].clone();
    let fault = layer.args.buffers[4].clone();
    let mut out = vec![layer];
    let mut first = 0;
    for counts in units {
        let mut p = program(counts);
        let old_map = View::new(&p[0].args.buffers[0]).unwrap();
        let old_x = View::new(&p[0].args.buffers[2]).unwrap();
        let old_fault = View::new(&p[0].args.buffers[3]).unwrap();
        let old_h = View::new(&p.iter().find(|e| e.name=="StrataExpertLargeCohortActivate").unwrap().args.buffers[2]).unwrap();
        p.remove(0);
        for e in &mut p {
            set_word(&mut e.args.info, 4, first);
            set_word(&mut e.args.info, 13, 1);
            set_word(&mut e.args.info, 14, 1);
            for i in 0..counts.len() { set_word(&mut e.args.info, 16+i*4, first+i as u32); }
            for binding in &mut e.args.buffers {
                let v = View::new(binding).unwrap();
                *binding = if v==old_map {map.clone()} else if v==old_x {x.clone()}
                    else if v==old_h {h.clone()} else if v==old_fault {fault.clone()}
                    else {binding.clone()};
            }
            e.name = match e.name {
                "StrataExpertLargeCohortGuDirect8Image" => "StrataExpertLargePackedCohortGuDirect8Image",
                "StrataExpertLargeCohortDownDirect8Image" => "StrataExpertLargePackedCohortDownDirect8Image",
                "StrataExpertLargeCohortActivate" => "StrataExpertLargePackedCohortActivate",
                LARGE => LARGE_PACKED,
                _ => panic!("unexpected public fixture role"),
            };
            if e.name==LARGE_PACKED {
                let tile = word(&e.args.info, 11).unwrap();
                let part_rows = counts.iter().map(|&m| match tile {
                    128 => m/128,
                    8 if (1..=8).contains(&(m%128)) => 1,
                    16 if (9..32).contains(&(m%128)) => (m%128).div_ceil(16),
                    32 if m%128>=32 => (m%128).div_ceil(32),
                    _ => 0,
                }).sum();
                e.count = CubeCount::Static(word(&e.args.info,2).unwrap()/if tile==128 {128}else{64}, part_rows, 1);
                for (i,b) in e.args.buffers.iter().enumerate() {
                    let len=(b.size_in_used()/if i<2 {8}else{4}) as u32;
                    set_word(&mut e.args.info,48+i,len);
                    set_word(&mut e.args.info,54+i,len);
                }
            } else if e.name=="StrataExpertLargePackedCohortActivate" {
                let rows: u32 = counts.iter().map(|m| m.div_ceil(32)*32).sum();
                e.count=CubeCount::Static((rows*320).div_ceil(256),1,1);
            }
        }
        first+=counts.len() as u32;
        out.extend(p);
    }
    out
}

#[cfg(feature="kuno-large-packed-tiles")]
fn packed_units(tails:bool)->Vec<Vec<u32>> {
    if tails {
        let mut units=vec![vec![4096];8];
        units.extend([vec![257,265,288],vec![3691],vec![3691]]);
        units
    } else {vec![vec![4096];10]}
}

#[cfg(feature="kuno-large-packed-tiles")]
#[test]
fn packed_large_exact_layer_prefix_all_tail_families_and_full_closure() {
    for tails in [false,true] {
        let units=packed_units(tails);
        let p=packed_layer_program(&units);
        let c=run_layer(&p,true).unwrap();
        assert_eq!(c.layer_gather_launches,1);
        assert_eq!(c.large_cohort_groups,units.iter().map(Vec::len).sum::<usize>());
        assert_eq!(c.large_cohort_batches,units.len());
        assert_eq!(c.large_cohort_launches,p.len()-1);
        assert_eq!(c.regular_cohort_groups,0);
        assert_eq!(c.logical_phases,c.large_cohort_groups*6);
        for i in 0..p.len() {
            let mut omitted=packed_layer_program(&units);
            omitted.remove(i);
            assert!(run_layer(&omitted,true).is_err(),"missing phase {i}");
            let mut bad=packed_layer_program(&units);
            bad[i].release=!bad[i].release;
            assert!(run_layer(&bad,true).is_err(),"wrong release {i}");
        }
        assert!(run_layer(&p[..6],false).is_ok());
        assert!(run_layer(&p[..6],true).is_err());
    }
}

#[cfg(feature="kuno-large-packed-tiles")]
#[test]
fn packed_large_exact_identity_and_format_are_required_at_every_phase() {
    let units=packed_units(true);
    let p=packed_layer_program(&units);
    for i in 1..p.len() {
        let mut wrong=packed_layer_program(&units);
        wrong[i].name=match wrong[i].name {
            LARGE_PACKED=>LARGE,
            "StrataExpertLargePackedCohortGuDirect8Image"=>"StrataExpertLargeCohortGuDirect8Image",
            "StrataExpertLargePackedCohortDownDirect8Image"=>"StrataExpertLargeCohortDownDirect8Image",
            "StrataExpertLargePackedCohortActivate"=>"StrataExpertLargeCohortActivate",
            _=>unreachable!(),
        };
        assert!(run_layer(&wrong,true).is_err(),"legacy identity phase {i}");
        for (word,value) in [(13,0),(14,0),(14,2),(15,1)] {
            let mut wrong=packed_layer_program(&units);
            set_word(&mut wrong[i].args.info,word,value);
            assert!(run_layer(&wrong,true).is_err(),"INFO word {word} phase {i}");
        }
    }
    let mut wrong=packed_layer_program(&units);
    wrong[0].name="StrataExpertLayerGather";
    set_word(&mut wrong[0].args.info,4,0);
    assert!(run_layer(&wrong,true).is_err());
    let mut wrong=layer_program();
    wrong[0].name="StrataExpertLayerPackedGather";
    set_word(&mut wrong[0].args.info,4,1);
    assert!(run_layer(&wrong,true).is_err());
    for packed in [false,true] {
        let mut wrong=packed_layer_program(&units);
        wrong[0].name=if packed {"StrataExpertLayerPackedGather"}else{"StrataExpertLayerGather"};
        set_word(&mut wrong[0].args.info,4,u32::from(!packed));
        assert!(run_layer(&wrong,true).is_err());
    }
    for name in ["StrataExpertLargePackedCohortGuImage","StrataExpertLargePackedCohortGuQuad32Image",
        "StrataExpertLargePackedCohortGuDirect8ImageFake","StrataExpertLargePackedCohortGather",
        "StrataExpertLayerPackedGather::Fake"] {
        let mut wrong=packed_layer_program(&units);
        wrong[1].name=name;
        assert!(run_layer(&wrong,true).is_err());
    }
}

#[cfg(feature="kuno-large-packed-tiles")]
#[test]
fn packed_large_never_bypasses_layer_or_original_physical_owners_and_grids() {
    for format in [0,1] {
        let mut p=program(&[4096]);
        for e in &mut p { set_word(&mut e.args.info,14,format); }
        p[0].name="StrataExpertLargePackedCohortGather";
        assert!(run(&p).is_err());
    }
    for name in [LARGE_PACKED,
        "probe::strata_matrix::cohorts::large_packed::expert_large_packed_cohort::ExpertLargePackedCohortFake<R>",
        "probe::strata_matrix::cohorts::large_packed::expert_large_packed_cohort::ExpertLargePackedCohort<R>::extra",
        "probe::strata_matrix::nested::strata_matrix::cohorts::large_packed::expert_large_packed_cohort::ExpertLargePackedCohort<R>",
        "probe::strata_matrix::cohorts::large_packed::expert_large_packed_cohort::ExpertLargePackedCohort<R"] {
        assert_eq!(matrix_variant(name,false),None);
        let genuine=name==LARGE_PACKED;
        assert_eq!(matrix_format_variant(name,false,true),genuine.then_some(false));
        for phase in 0..6 {assert!(!super::super::super::native_expert_batch::native_strata_expert_operator(name,false,phase));}
    }
    let units=packed_units(false);
    for phase in 0..6 {
        let mut p=packed_layer_program(&units);
        p[phase].count=CubeCount::Static(1,1,1);
        assert!(run_layer(&p,true).is_err());
    }
    for phase in [1usize,2,3,4,5] {
        for binding in 0..6.min(packed_layer_program(&units)[phase].args.buffers.len()) {
            // The first GU legitimately establishes its output/status owners;
            // mutate only immutable/shared views or already-established owners.
            if phase==3 && [2,4].contains(&binding) {continue;}
            if phase==5 && binding==2 {continue;}
            if phase==1 && ![2,5].contains(&binding) {continue;}
            if phase==2 && binding==0 {continue;}
            let mut p=packed_layer_program(&units);
            let old=&p[phase].args.buffers[binding];
            p[phase].args.buffers[binding]=Handle::new(old.stream,old.size_in_used()).binding();
            assert!(run_layer(&p,true).is_err(),"owner phase {phase} binding {binding}");
        }
    }
}

#[cfg(feature="kuno-large-packed-tiles")]
#[test]
fn packed_layer_seal_leaves_small_and_regular_segments_in_original_format() {
    let p=packed_layer_program(&packed_units(false));
    let e=&p[0];
    let layer=super::super::layer_gather::Layer::start_packed(e.release,&e.count,&e.args).unwrap();
    for members in [1,31,32,255,256,4096] {
        let mut args=program(&[4096]).remove(0).args;
        set_word(&mut args.info,13,1);
        set_word(&mut args.info,17,members);
        set_word(&mut args.info,14,u32::from(members>=256));
        assert!(layer.unit_args(&args).is_ok());
        set_word(&mut args.info,14,u32::from(members<256));
        assert!(layer.unit_args(&args).is_err());
    }
}

#[cfg(all(feature="kuno-layer-gather",not(feature="kuno-large-packed-tiles")))]
#[test]
fn packed_roles_are_rejected_when_feature_is_disabled() {
    for (phase,name) in [(0,"StrataExpertLayerPackedGather"),
        (1,"StrataExpertLargePackedCohortGuDirect8Image"),
        (2,"StrataExpertLargePackedCohortDownDirect8Image"),
        (3,"probe::strata_matrix::cohorts::large_packed::expert_large_packed_cohort::ExpertLargePackedCohort<Runtime>"),
        (4,"StrataExpertLargePackedCohortActivate")] {
        let mut p=layer_program();p[phase].name=name;
        assert!(run_layer(&p,true).is_err());
    }
}
