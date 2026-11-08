//! Metadata controls of genuine small raw GU / ordered DOWN entries.
//! Original fixtures and counters remain independent; no GPU math claim.
use super::*;
const MACRO: &str = "probe::strata_matrix::native::small_direct::expert_macro::ExpertMacro<Runtime>";
const MACRO64: &str = "probe::strata_matrix::native::small_direct::expert_macro64::ExpertMacro64<Runtime>";
const NARROW: &str = "probe::strata_matrix::native::small_direct::expert_narrow::ExpertNarrow<Runtime>";
const TAIL_GU: &str = "probe::strata_matrix::native::small_direct::expert_tail_gu::ExpertTailGu<Runtime>";
const TAIL_DOWN: &str = "probe::strata_matrix::native::small_direct::expert_tail_down::ExpertTailDown<Runtime>";
const ORDERED_WIDE: &str = "probe::strata_matrix::native_expert_wide::ordered::expert_wide_down::ExpertWideDown<Runtime>";
fn checked(g: &mut NativeStrataExpertGrammar, h: &[Handle; 7], n: u32) {
    g.launch("StrataPromptExpertF16SwigluCheckGu", false,
        &CubeCount::Static((n * 5).div_ceil(4), 1, 1), &checker(h, n)).unwrap();
}
fn full(g: &mut NativeStrataExpertGrammar, h: &[Handle; 7], n: u32, down: bool, name: &str, tile: u32) {
    g.launch(name, false, &CubeCount::Static(if down {40} else {20}, n.div_ceil(tile), 1),
        &matrix_args(h, n, down, false, false)).unwrap();
}
fn prefixed(h: &[Handle; 7], n: u32, down: bool, ordered: bool) -> NativeStrataExpertGrammar {
    let mut g = begin(h, n);
    if down { product(&mut g, h, n, false, true); activate(&mut g); }
    g.launch(if down {if ordered {ORDERED_WIDE} else {WIDE_DOWN}} else {WIDE_GU}, false,
        &CubeCount::Static(if down {20} else {10}, n / 128, 1),
        &matrix_args(h, n, down, true, false)).unwrap();
    g
}
fn tail_grid(n: u32, down: bool) -> CubeCount {
    let rest = n % 128;
    let tile = if rest <= 8 {8} else if rest < 32 {16} else {32};
    CubeCount::Static(if down {40} else {20}, rest.div_ceil(tile), 1)
}
#[test]
fn genuine_small_full_entries_keep_six_phases_checker_owners_and_old_prefix_counts() {
    for (name, tile, counts) in [(NARROW, 8, &[1u32, 8][..]),
        (NARROW, 16, &[9, 17, 31][..]), (MACRO, 32, &[32, 63, 256, 4096][..]),
        (MACRO64, 64, &[64, 65, 127, 128, 255][..])] {
        for &n in counts { for resident in [false, true] {
            let h = owners(); let mut g = begin_images(&h, n, resident);
            full(&mut g, &h, n, false, name, tile);
            assert!(g.finish().is_err());
            checked(&mut g, &h, n);
            full(&mut g, &h, n, true, name, tile);
            let c = g.finish().unwrap();
            assert_eq!((c.launches,c.logical_phases,c.matrix_pieces,c.gu_checks),(6,6,2,1));
            assert_eq!((c.wide_prefixes,c.tail_pieces,c.direct_gu_prefixes),(0,0,0));
        }}
    }
}
#[test]
fn raw_small_gu_cannot_skip_or_forge_the_authentic_checked_activation() {
    for (name, n, tile) in [(NARROW,17,16),(MACRO,32,32),(MACRO64,65,64)] {
        let h=owners();
        for bad in 0..5 {
            let mut g=begin(&h,n); full(&mut g,&h,n,false,name,tile);
            let mut args=checker(&h,n);
            if bad==1 {args.buffers[1]=owners()[2].clone().binding();}
            if bad==2 {args.buffers[3]=owners()[4].clone().binding();}
            let grid=if bad==3 {CubeCount::Static(1,1,1)} else {CubeCount::Static((n*5).div_ceil(4),1,1)};
            assert!(g.launch(if bad==0 {"StrataPromptExpertF16Swiglu"} else {"StrataPromptExpertF16SwigluCheckGu"},
                bad==4,&grid,&args).is_err());
        }
    }
}
#[test]
fn small_direct_names_roles_geometry_info_and_physical_owners_are_closed() {
    for name in ["probe::strata_matrix::small_direct::expert_macro::ExpertMacro<Runtime>",
        "probe::strata_matrix::native::small_direct::small_direct::expert_macro::ExpertMacro<Runtime>",
        "probe::strata_matrix::native::small_direct::expert_macro::ExpertMacroFake<Runtime>",
        "probe::strata_matrix::native::small_direct::expert_macro::ExpertMacro<Runtime>::other",
        "probe::strata_matrix::native::small_direct::expert_macro::ExpertMacro<Runtime>::other<Runtime>",
        "probe::strata_matrix::native::small_direct::expert_tail_gu::ExpertTailGu<Runtime>",
        "probe::strata_matrix::native::small_direct::expert_macro64_deep::ExpertMacro64Deep<Runtime>"] {
        for phase in 0..6 {assert!(!native_strata_expert_operator(name,false,phase));}
    }
    for name in [
        "probe::strata_matrix::native::small_direct::native_expert_wide::expert_tail_gu::ExpertTailGu<Runtime>",
        "probe::strata_matrix::native::small_direct::native_expert_wide::ordered::expert_tail_down::ExpertTailDown<Runtime>",
        "probe::strata_matrix::native::small_direct::expert_tail_gu::ExpertTailGu<Runtime>::other<Runtime>",
    ] {assert!(!is_tail(name,false)&&!is_tail(name,true)&&!is_wide(name,false)&&!is_wide(name,true));}
    for name in [NARROW,MACRO,MACRO64] { for phase in 0..6 {
        assert_eq!(native_strata_expert_operator(name,false,phase),matches!(phase,3|5));
        assert!(!native_strata_expert_operator(name,true,phase));
    }}
    for (name,n,tile) in [(NARROW,17,16),(MACRO,32,32),(MACRO64,65,64)] {
        let h=owners();
        for bad in 0..9 {
            let mut g=begin(&h,n); let mut args=matrix_args(&h,n,false,false,false);
            // All six exact metadata/view extents remain checked. First GU
            // output/status identities are established here and then bound by
            // the authentic checker; physical image provenance is separately
            // checked by the real backend owner/source factory.
            if bad<6 {args.buffers[bad]=h[bad].clone().offset_start(8).binding();}
            if bad==6 {set_word(&mut args.info,2,n+1);}
            if bad==7 {set_word(&mut args.info,6,1);}
            let grid=CubeCount::Static(if bad==8 {19} else {20},n.div_ceil(tile),1);
            assert!(g.launch(name,false,&grid,&args).is_err());
        }
    }
    for (name,n,tile) in [(NARROW,32,16),(MACRO,31,32),(MACRO64,63,64),(MACRO64,256,64)] {
        let h=owners(); assert!(begin(&h,n).launch(name,false,&CubeCount::Static(20,n.div_ceil(tile),1),
            &matrix_args(&h,n,false,false,false)).is_err());
    }
}
#[test]
fn genuine_small_raw_tails_follow_original_prefixes_and_ordered_down_layout() {
    for n in [257u32,264,287,288,383,4095] {
        let h=owners(); let mut g=prefixed(&h,n,false,false);
        g.launch(TAIL_GU,false,&tail_grid(n,false),&matrix_args(&h,n,false,true,true)).unwrap();
        assert!(g.launch("StrataPromptExpertF16Swiglu",false,&CubeCount::Static(1,1,1),&KernelArguments::new()).is_err());
        checked(&mut g,&h,n);
        g.launch(ORDERED_WIDE,false,&CubeCount::Static(20,n/128,1),&matrix_args(&h,n,true,true,false)).unwrap();
        g.launch(TAIL_DOWN,false,&tail_grid(n,true),&matrix_args(&h,n,true,true,true)).unwrap();
        let c=g.finish().unwrap(); assert_eq!((c.logical_phases,c.wide_prefixes,c.tail_pieces,c.gu_checks),(6,2,2,1));
        assert_eq!(c.direct_gu_prefixes,0);
    }
}
#[test]
fn new_tails_reject_rank_down_prefixes_wrong_roles_offsets_grids_and_owner_changes() {
    for down in [false,true] { for bad in 0..7 {
        let h=owners(); let n=257; let mut g=prefixed(&h,n,down,down&&bad!=0);
        let mut args=matrix_args(&h,n,down,true,true);
        if bad==1 {set_word(&mut args.info,5,255);}
        if bad==2 {set_word(&mut args.info,6,2);}
        if bad==3 {args.buffers[0]=h[0].clone().binding();}
        if bad==4 {args.buffers[4]=owners()[4].clone().binding();}
        let grid=if bad==5 {CubeCount::Static(if down{40}else{20},2,1)}else{tail_grid(n,down)};
        let name=if down {if bad==6 {TAIL_GU}else{TAIL_DOWN}} else {if bad==6 {TAIL_DOWN}else{TAIL_GU}};
        let result=g.launch(name,false,&grid,&args);
        // Checked GU prefixes are allowed to gain a raw GU tail; DOWN must
        // already have an ordered prefix before its ordered tail.
        if !down&&bad==0 {assert!(result.is_ok());}else{assert!(result.is_err());}
    }}
}
