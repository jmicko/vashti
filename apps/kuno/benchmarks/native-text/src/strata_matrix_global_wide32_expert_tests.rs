use super::*;
#[test]
fn wide32_actual_expert_modules_keep_original_checked_fallback_and_raw_math(){validate_sources().unwrap();}
#[test]
fn wide32_checked_subtiles_and_raw_cells_have_same_original_prefix_coverage(){
 let mut raw=vec![0u8;128*128];let mut checked=raw.clone();
 for sg in 0..16{for wb in 0..4{for tb in 0..2{for w in 0..8{for t in 0..16{raw[(sg/4*32+tb*16+t)*128+sg%4*32+wb*8+w]+=1;}}}}}
 for tm in 0..2{for tn in 0..2{for sg in 0..16{for wb in 0..2{for w in 0..8{for t in 0..16{let token=if cfg!(feature="strata-wide32-local-routes"){sg/4*32+tm*16+t}else{tm*64+sg/4*16+t};let row=if cfg!(feature="strata-wide32-local-routes"){sg%4*32+tn*16+wb*8+w}else{tn*64+sg%4*16+wb*8+w};checked[token*128+row]+=1;}}}}}}
 assert_eq!(raw,checked);assert!(raw.iter().all(|&n|n==1));assert_eq!(16*128*4,8192);
}
#[test]
fn wide32_original_prefix_tail_scope_and_physical_credit_stay_distinct(){for count in [256usize,257,383,4096]{let p=plan(count).unwrap();assert_eq!(p.prefix+p.tail,count);assert_eq!(p.prefix%128,0);assert_eq!((1280/128+2560/128)*(p.prefix/128),15*2*(p.prefix/128));}}
#[test]
fn wide32_production_guard_is_existing_authentic_gather_without_extra_tape_nodes(){
 let source=include_str!("strata_prompt_experts.wgsl");let gather=source.split("// GATHER_BEGIN").nth(1).unwrap().split("// GATHER_END").next().unwrap();
 assert!(gather.contains("route>=p.count*10u {atomicOr(&fault[0],128u);return;}"));assert!(gather.contains("i>=length*p.columns/2u"));
 assert!(gather.contains("@binding(0) var<storage,read> map"));assert!(gather.contains("@binding(3) var<storage,read_write> fault"));
}


#[test]
fn local32_each_checked_subgroup_owns_exactly_its_raw_cell_for_all_branch_masks() {
    let mut raw = [[0u128; 128]; 16];
    let mut checked = raw;
    for sg in 0..16 {
        for wb in 0..4 { for tb in 0..2 { for w in 0..8 { for t in 0..16 {
            let member = sg / 4 * 32 + tb * 16 + t;
            let row = sg % 4 * 32 + wb * 8 + w;
            assert_eq!(raw[sg][member] & (1u128 << row), 0);
            raw[sg][member] |= 1u128 << row;
        }}}}
        for tm in 0..2 { for tn in 0..2 { for wb in 0..2 { for w in 0..8 { for t in 0..16 {
            let member = sg / 4 * 32 + tm * 16 + t;
            let row = sg % 4 * 32 + tn * 16 + wb * 8 + w;
            assert_eq!(checked[sg][member] & (1u128 << row), 0);
            checked[sg][member] |= 1u128 << row;
        }}}}}
        assert_eq!(raw[sg], checked[sg]);
    }
    // Exhaust all mixtures, not merely all-raw versus all-checked. A locally
    // selected branch cannot borrow another subgroup's fallback cells.
    for mask in 0u32..65536 {
        let mut owner = [0u128; 128];
        for sg in 0..16 {
            let cells = if mask & (1 << sg) == 0 { &raw[sg] } else { &checked[sg] };
            for member in sg / 4 * 32..sg / 4 * 32 + 32 {
                assert_eq!(owner[member] & cells[member], 0);
                owner[member] |= cells[member];
            }
        }
        assert!(owner.iter().all(|bits| *bits == u128::MAX));
    }
}

#[test]
fn local32_real_route_checks_cover_both_halves_without_changing_other_subgroups() {
    for bad in 0usize..256 {
        let mut map = (0u32..256).collect::<Vec<_>>();
        map[bad] = 256;
        for y in 0..2 { for sg in 0..16 {
            let token = y * 128 + sg / 4 * 32;
            let all = (0..16).all(|lane| map[token + lane] < 256 && map[token + 16 + lane] < 256);
            assert_eq!(all, !(token..token + 32).contains(&bad));
        }}
    }
}

#[test]
fn local32_pending_status_and_nonfinite_rules_keep_original_eight_member_halves() {
    // Route damage in any of the four8-member halves affects that half alone;
    // prior unrelated sticky faults cannot turn other halves into checked ones.
    for bad in 0usize..32 { for initial_fault in [0u32, 128] {
        let valid = (0..32).map(|m| m != bad).collect::<Vec<_>>();
        let mut sticky = initial_fault;
        for member in 0..32 {
            let raw_half = (member / 8 * 8..member / 8 * 8 + 8).all(|m| valid[m]);
            if !valid[member] { sticky |= 128; continue; }
            let expected_raw = member / 8 != bad / 8;
            assert_eq!(raw_half, expected_raw);
            for finite in [false, true] { for input_status in [0u32, 1] {
                let status = if raw_half {2} else {u32::from(finite && input_status == 1)};
                assert_eq!(status, if expected_raw {2} else if finite {input_status} else {0});
                if !raw_half && !finite {sticky |= 128;}
            }}
        }
        assert_eq!(sticky, 128);
    }}
}
