//! Standalone independent logical/panel oracle. `rustc --test` needs no crates.
//! Decode producer tiles back to logical coordinates rather than repeating the
//! consumer's half-index calculation as the expected result.
fn decode(vector:usize,block:usize,k:usize)->(usize,usize) {
    let parent=vector/(block*k/4);
    let inside=vector%(block*k/4);
    let ktile=inside/(block*4);
    let lane=inside%(block*4);
    (parent*block+lane/4,ktile*16+(lane%4)*4)
}
fn staging(t:usize,k:usize,columns:usize,member:usize,row:usize,weight:usize)->[(bool,usize,usize);4] {
    let r=t/4;let q=t%4*4;
    let w=|r:usize|(weight+(r/8)*(8*columns)+(k/16)*128+(r%8)*16+q)/4;
    let x=|r:usize|((r/16)*(16*columns)+(k/16)*256+(r%16)*16+q)/4;
    [(true,w(row+r),2*t),(true,w(row+64+r),512+2*t),
     (false,x(member+r),1024+2*t),(false,x(member+64+r),1536+2*t)]
}
#[test]
fn shared_panel_unique_source_vectors_and_f32_pairs_for_every_role_k16() {
    use std::collections::BTreeSet;
    for columns in [640usize,2560] {
        for (member,row,weight) in [(0,0,0),(32+128,128,4_915_200),(288+3968,1152,7*4_915_200+3_276_800)] {
            for k in (0..columns).step_by(16) {
                let mut seen=[false;2048];let mut sources=BTreeSet::new();
                for t in 0..256 {for (is_weight,vector,dest) in staging(t,k,columns,member,row,weight) {
                    assert!(sources.insert((is_weight,vector)));
                    assert!(dest+1<2048 && dest%2==0);
                    assert!(!seen[dest]&&!seen[dest+1]);seen[dest]=true;seen[dest+1]=true;
                    let (logical_r,logical_k)=decode(vector-if is_weight {weight/4}else{0},if is_weight {8}else{16},columns);
                    let local_half=(dest%1024)*2;
                    assert_eq!(logical_r,if is_weight {row}else{member}+local_half/16);
                    assert_eq!(logical_k,k+local_half%16);
                }}
                assert!(seen.into_iter().all(|v|v));assert_eq!(sources.len(),1024);
            }
        }
    }
}
#[test]
fn shared_cmmas_reconstruct_logical_a_b_and_all_eight_c_tiles() {
    // Independently generate the dense shared panel from logical coordinates.
    // Its two f16 halves per f32 slot are distinct symbolic integers.
    let mut halves=[0u16;4096];
    for r in 0..128 {for k in 0..16 {halves[r*16+k]=(1+r*16+k) as u16;halves[2048+r*16+k]=(8193+r*16+k) as u16;}}
    let mut coverage=vec![0u8;128*128];
    for sg in 0..16 {
        for ar in 0..4 {for br in 0..2 {
            let af=((sg%4)*32+ar*8)*8;
            let bf=1024+((sg/4)*32+br*16)*8;
            for row in 0..8 {for member in 0..16 {
                let output_row=(sg%4)*32+ar*8+row;
                let output_member=(sg/4)*32+br*16+member;
                coverage[output_member*128+output_row]+=1;
                for k in 0..16 {
                    // Actual CMMA stride8 is f32 units; logical half stride16.
                    let a=halves[(af+row*8)*2+k];
                    let b=halves[(bf+member*8)*2+k];
                    assert_eq!(a,halves[output_row*16+k]);
                    assert_eq!(b,halves[2048+output_member*16+k]);
                }
            }}
        }}
    }
    assert!(coverage.into_iter().all(|v|v==1));
}
#[test]
fn shared_global_gu_prefix_cannot_be_replaced_with_cohort_prefix() {
    for columns in [640usize,2560] {
        for global in [32usize,288,1024,56832-4096] {
            let cohort=0;let t=37;let tile=128;
            for k in (0..columns).step_by(16) {
                let actual=staging(t,k,columns,global+tile,128,0);
                let wrong=staging(t,k,columns,cohort+tile,128,0);
                for i in [2,3] {
                    assert_ne!(actual[i].1,wrong[i].1);
                    assert_eq!(decode(actual[i].1,16,columns).0-decode(wrong[i].1,16,columns).0,global);
                }
            }
        }
    }
}
#[test]
fn identical_128_route_scans_are_wg_uniform_for_low_high_and_remote_halves() {
    for invalid in [None,Some(0),Some(8),Some(31),Some(32),Some(96),Some(104),Some(127)] {
        let mut map=[0usize;128];for (r,value) in map.iter_mut().enumerate(){*value=r;}
        if let Some(r)=invalid {map[r]=40960;}
        let decisions=(0..16).map(|_sg| {
            (0..16).all(|lane|(0..8).all(|round|map[round*16+lane]<40960))
        }).collect::<Vec<_>>();
        assert!(decisions.iter().all(|v|*v==invalid.is_none()));
        // Prove why the original32-route SG-local decision is not enough.
        if let Some(r)=invalid {
            let old=(0..16).map(|sg|(0..32).all(|m|map[(sg/4)*32+m]<40960)).collect::<Vec<_>>();
            assert!(old.iter().any(|v|*v));assert!(old.iter().any(|v|!*v));
            let published=(0..128).filter(|m|map[*m]<40960).collect::<Vec<_>>();
            assert_eq!(published.len(),127);assert!(!published.contains(&r));
        }
    }
}
#[test]
fn one_slab_bit_pairs_preserve_every_fp16_payload_and_fallback_lifetime() {
    // Every possible f16 payload appears in both positions of a packed f32.
    // These values include floating NaN/Inf/denormal patterns; no FP math.
    let mut slab=[0f32;2048];
    for h in 0..=u16::MAX {
        let bits=u32::from(h)|(u32::from(!h)<<16);
        let pair=[f32::from_bits(bits),f32::from_bits(bits.rotate_left(16))];
        let at=usize::from(h)%1024*2;slab[at]=pair[0];slab[at+1]=pair[1];
        assert_eq!(slab[at].to_bits(),bits);assert_eq!(slab[at+1].to_bits(),bits.rotate_left(16));
    }
    // The fallback follows a disjoint control-flow lifetime and writes genuine
    // f32 accumulator values into the SAME16×128-element publication slab.
    for sg in 0..16 {for i in 0..128 {slab[sg*128+i]=(sg*128+i) as f32;}}
    for (i,value) in slab.into_iter().enumerate(){assert_eq!(value,i as f32);}
    assert_eq!(std::mem::size_of_val(&slab),8192);
}
#[test]
fn shared_source_keeps_complete_coupled_staging_and_old_fallback() {
    let wide=include_str!("strata_matrix_large_shared_wide_products.rs");
    let packed=include_str!("strata_matrix_large_packed_wide_products.rs");
    let body=include_str!("strata_matrix_large_shared_operands.rs");
    assert_eq!(wide.matches("SharedMemory::<f32>::new(2048usize)").count(),1);
    assert!(!body.contains("SharedMemory::<f32>::new"));
    assert_eq!(body.matches("sync_cube();").count(),2);
    assert_eq!(body.matches("Vector::<f32, Const<2>>::reinterpret(").count(),4);
    assert_eq!(body.matches("            8u32,").count(),6);
    assert_eq!(body.matches("row::<f16>(").count(),4);
    assert!(body.contains("xi0 += 64;")&&body.contains("xi1 += 64;")&&body.contains("wi0 += 32;")&&body.contains("wi1 += 32;"));
    assert!(body.contains("first_input = token - (sg / 4) * 32"));
    assert!(body.contains("first_weight = out_row - (sg % 4) * 32"));
    for source in ["first_weight + r, q, columns","first_weight + 64 + r, q, columns","first_input + r, q, columns","first_input + 64 + r, q, columns"] {assert!(body.contains(source));}
    // Exact fallback byte comparison, including every map/raw-half/finite/status
    // rule. Only the old local scratch declaration moved to the shared owner.
    let marker="    } else {\n        // Original8-member publication granularity";
    let old=packed[packed.find(marker).unwrap()..].replace("                let mut scratch = SharedMemory::<f32>::new(2048usize);\n","");
    assert_eq!(&wide[wide.find(marker).unwrap()..],old);
    for round in 0..8 {assert!(wide.contains(&format!("token0 + {} + lane",round*16)));}
    let eligibility=&wide[wide.find("let lane = usize::cast_from(UNIT_POS_PLANE)").unwrap()..wide.find("    if plane_all(raw_good)").unwrap()];
    assert!(!eligibility.contains("PLANE_POS)"));assert!(!eligibility.contains("fault[0].load"));
}
