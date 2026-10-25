//! CPU mapping/capacity tests; these do not claim GPU numerical acceptance.
use super::*;
fn fixture(counts:&[u32])->[u32;48] {
    let mut w=[0u32;48];w[0]=4096;w[1]=2560;w[2]=1280;w[4]=7;
    w[6]=100;w[7]=100;w[8]=300;w[9]=counts.len() as u32;
    w[10]=*counts.iter().max().unwrap();let mut prefix=0;
    for (i,&n) in counts.iter().enumerate() {
        w[16+i*4..20+i*4].copy_from_slice(&[7+i as u32,n,prefix,8]);
        prefix+=n.min(4096).div_ceil(32)*32;
    }w
}
#[test]
fn large_cohort_rows_and_each_original_prefix_tail_partition() {
    for members in 256..=4096u32 {
        let w=fixture(&[members]);assert_eq!(validate_words(&w).unwrap(),members.div_ceil(32) as usize*32);
        let mut visits=vec![0u8;members as usize];
        for tile in [128,8,16,32] {
            let blocks=crate::strata_matrix::cohorts::large::part_rows(&w,tile);
            for block in 0..blocks as usize {for lane in 0..tile {
                let member=if tile==128 {block*tile+lane}else{members as usize/128*128+block*tile+lane};
                if member<members as usize {visits[member]+=1;}
            }}
        }
        assert!(visits.into_iter().all(|n|n==1));
    }
    assert_eq!(validate_words(&fixture(&[512;8])).unwrap(),4096);
    for n in [0,1,31,32,255,4097,u32::MAX] {assert!(validate_words(&fixture(&[n])).is_err());}
    assert!(validate_words(&fixture(&[1025;4])).is_err());
    let mut w=fixture(&[256,257,265,288]);
    for i in [0usize,4,9,10,12,13,16,17,18,19,47] {let old=w[i];w[i]=u32::MAX;assert!(validate_words(&w).is_err());w[i]=old;}
    assert_eq!(IMAGE_BYTES*8,75*1024*1024);
    for op in [LargeOp::Gather,LargeOp::Activate] {
        let text=op.source();assert!(text.contains("let tile=32u;let padded=((real+tile-1u)/tile)*tile;"));
        assert!(text.contains("v.y<256u || v.y>4096u"));
    }
}
#[test]
fn large_cohort_map_route_inverse_and_reduction_order_unchanged() {
    let total=512;let mut raw=vec![0u32;groups::LOGICAL+total*10];
    raw[..3].copy_from_slice(&[total as u32,0,total as u32]);
    for token in 0..total {for rank in 0..10 {
        let route=token*10+rank;raw[groups::LOGICAL+route]=((rank*41+token%2)%512) as u32;
        raw[1540+route]=raw[groups::LOGICAL+route];
    }}
    let map=groups::group(&raw,total).unwrap();let n=groups::validate(&map,total).unwrap();
    let original=map.to_vec();let inverse=groups::ordered_inverse(&map,total).unwrap();
    let mut plan=super::raw_banks::Plan::new_large(&map[516..516+n],&vec![true;n]).unwrap();
    let mut order=Vec::new();while let Some(unit)=plan.issue_next().unwrap() {
        assert!(unit.large);order.extend(unit.first..unit.end);
        let rows=map[516+unit.first..516+unit.end].iter().map(|m|m.div_ceil(32)*32).sum::<u32>();
        assert!(rows<=4096);assert!(unit.end-unit.first<=8);plan.consumed(unit).unwrap();
    }
    assert_eq!(order,(0..n).collect::<Vec<_>>());assert_eq!(*map,original);assert_eq!(*inverse,*groups::ordered_inverse(&map,total).unwrap());
    // No route, rank, expert choice or inverse entry was rewritten by planning.
    for group in order {for i in 0..map[516+group] as usize {
        let route=map[1540+map[1028+group] as usize+i] as usize;
        assert!(route<total*10);assert_eq!(raw[groups::LOGICAL+route],map[4+group]);
    }}
}

#[test]
fn large_cohort_all_actual_wgsl_producers_validate() {
    let compiled=sources();assert_eq!(compiled.len(),8);
    for k in compiled {
        let module=wgpu::naga::front::wgsl::parse_str(&k.source).unwrap();
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(),wgpu::naga::valid::Capabilities::all()).validate(&module).unwrap();
    }
}
