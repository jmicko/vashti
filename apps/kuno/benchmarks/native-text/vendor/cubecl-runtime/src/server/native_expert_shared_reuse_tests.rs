//! CPU metadata controls of real separately generated K32 lifetime entries.
//! Reuse original fixtures; this grants no device, math or completion proof.
use super::*;

const GU: &str =
    "probe::strata_matrix::native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGu<Runtime>";
const DIRECT: &str = "probe::strata_matrix::native_expert_wide::shared_reuse::expert_wide_gu_direct::ExpertWideGuDirect<Runtime>";
const DOWN: &str = "probe::strata_matrix::native_expert_wide::shared_reuse::expert_wide_down::ExpertWideDown<Runtime>";
const ORDERED: &str = "probe::strata_matrix::native_expert_wide::ordered::shared_reuse::expert_wide_down::ExpertWideDown<Runtime>";
const ORDERED_TAIL: &str =
    "probe::strata_matrix::native_expert_wide::ordered::expert_tail_down::ExpertTailDown<Runtime>";

fn before(h: &[Handle; 7], members: u32, down: bool) -> NativeStrataExpertGrammar {
    let mut grammar = begin(h, members);
    if down {
        product(&mut grammar, h, members, false, true);
        activate(&mut grammar);
    }
    grammar
}
fn prefix(
    grammar: &mut NativeStrataExpertGrammar,
    h: &[Handle; 7],
    members: u32,
    down: bool,
    name: &str,
) {
    grammar
        .launch(
            name,
            false,
            &CubeCount::Static(if down { 20 } else { 10 }, members / 128, 1),
            &matrix_args(h, members, down, true, false),
        )
        .unwrap();
}
fn tail(
    grammar: &mut NativeStrataExpertGrammar,
    h: &[Handle; 7],
    members: u32,
    down: bool,
    ordered: bool,
) {
    let remaining = members % 128;
    if remaining == 0 {
        return;
    }
    let tile = if remaining <= 8 {
        8
    } else if remaining < 32 {
        16
    } else {
        32
    };
    grammar
        .launch(
            if down {
                if ordered { ORDERED_TAIL } else { TAIL_DOWN }
            } else {
                TAIL_GU
            },
            false,
            &CubeCount::Static(if down { 40 } else { 20 }, remaining.div_ceil(tile), 1),
            &matrix_args(h, members, down, true, true),
        )
        .unwrap();
}

#[test]
fn genuine_reuse_checked_entries_retain_complete_phases_real_prefixes_and_original_tails() {
    for members in [256u32, 257, 287, 288, 383, 384, 4096] {
        for resident in [false, true] {
            let h = owners();
            let mut grammar = begin_images(&h, members, resident);
            prefix(&mut grammar, &h, members, false, GU);
            tail(&mut grammar, &h, members, false, false);
            activate(&mut grammar);
            prefix(&mut grammar, &h, members, true, DOWN);
            tail(&mut grammar, &h, members, true, false);
            let counts = grammar.finish().unwrap();
            let tails = 2 * usize::from(members % 128 != 0);
            assert_eq!(counts.logical_phases, 6);
            assert_eq!(counts.launches, 6 + tails);
            assert_eq!(counts.matrix_pieces, 2 + tails);
            assert_eq!(counts.wide_prefixes, 2);
            assert_eq!(counts.wide_blocks, 2 * (members / 128) as usize);
            assert_eq!(counts.tail_pieces, tails);
            assert_eq!(counts.direct_gu_prefixes, 0);
        }
    }
}

#[cfg(feature = "kuno-expert-ordered-output")]
#[test]
fn reuse_direct_gu_still_requires_real_checker_before_ordered_down_and_unchanged_tail() {
    for members in [256u32, 257, 264, 383, 384, 513, 4096] {
        let h = owners();
        let ready = || {
            let mut grammar = begin_images(&h, members, true);
            prefix(&mut grammar, &h, members, false, DIRECT);
            tail(&mut grammar, &h, members, false, false);
            grammar
        };
        let mut wrong = ready();
        assert!(wrong.finish().is_err());
        assert!(
            wrong
                .launch(
                    "StrataPromptExpertF16Swiglu",
                    false,
                    &CubeCount::Static((members * 5).div_ceil(4), 1, 1),
                    &checker(&h, members)
                )
                .is_err()
        );
        let mut grammar = ready();
        grammar
            .launch(
                "StrataPromptExpertF16SwigluCheckGu",
                false,
                &CubeCount::Static((members * 5).div_ceil(4), 1, 1),
                &checker(&h, members),
            )
            .unwrap();
        prefix(&mut grammar, &h, members, true, ORDERED);
        tail(&mut grammar, &h, members, true, true);
        let counts = grammar.finish().unwrap();
        assert_eq!(counts.logical_phases, 6);
        assert_eq!(counts.direct_gu_prefixes, 1);
        assert_eq!(counts.gu_checks, 1);
    }
}

#[test]
fn reuse_entries_are_exact_and_role_bound_without_admitting_new_tail_or_deep_aliases() {
    assert!(is_wide(GU, false));
    assert!(!is_wide(GU, true));
    assert!(is_direct_gu(DIRECT));
    assert!(!is_wide(DIRECT, false));
    assert!(is_wide(DOWN, true));
    assert!(!is_wide(DOWN, false));
    for name in [
        "probe::native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGuFake<Runtime>",
        "probe::strata_matrix::native_expert_wide::shared_reuse::expert_wide_gu_direct::ExpertWideGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::shared_reuse::expert_wide_down::ExpertWideGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::shared_reuse::deep::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::deep::shared_reuse::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::shared_reuse::shared_reuse::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGu<Runtime>::other",
        "probe::strata_matrix::native_expert_wide::shared_reuse::expert_tail_gu::ExpertTailGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::ordered::shared_reuse::expert_tail_down::ExpertTailDown<Runtime>",
        "probe::strata_matrix::native_expert_wide::ordered::shared_reuse::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::strata_matrix::native_expert_wide::ordered::shared_reuse::expert_wide_gu_direct::ExpertWideGuDirect<Runtime>",
    ] {
        assert!(!is_wide(name, false) && !is_wide(name, true) && !is_direct_gu(name));
        assert!(!is_tail(name, false) && !is_tail(name, true));
    }
    let h = owners();
    assert!(
        before(&h, 256, false)
            .launch(
                DOWN,
                false,
                &CubeCount::Static(10, 2, 1),
                &matrix_args(&h, 256, false, true, false)
            )
            .is_err()
    );
    assert!(
        before(&h, 256, true)
            .launch(
                GU,
                false,
                &CubeCount::Static(20, 2, 1),
                &matrix_args(&h, 256, true, true, false)
            )
            .is_err()
    );
}

#[test]
fn reuse_prefix_and_tail_preserve_all_original_info_grid_and_actual_owner_checks() {
    for (down, name) in [(false, GU), (false, DIRECT), (true, DOWN)] {
        let h = owners();
        let members = 287;
        for index in 0..8 {
            let mut args = matrix_args(&h, members, down, true, false);
            let old = word(&args.info, index).unwrap();
            set_word(&mut args.info, index, old + 1);
            assert!(
                before(&h, members, down)
                    .launch(
                        name,
                        false,
                        &CubeCount::Static(if down { 20 } else { 10 }, 2, 1),
                        &args
                    )
                    .is_err()
            );
        }
        for wrong in [
            CubeCount::Static(10, 1, 1),
            CubeCount::Static(20, 3, 1),
            CubeCount::Static(20, 2, 2),
        ] {
            assert!(
                before(&h, members, down)
                    .launch(
                        name,
                        false,
                        &wrong,
                        &matrix_args(&h, members, down, true, false)
                    )
                    .is_err()
            );
        }
        assert!(
            before(&h, members, down)
                .launch(
                    name,
                    true,
                    &CubeCount::Static(if down { 20 } else { 10 }, 2, 1),
                    &matrix_args(&h, members, down, true, false)
                )
                .is_err()
        );
        for owner in 0..6 {
            let mut grammar = before(&h, members, down);
            prefix(&mut grammar, &h, members, down, name);
            assert!(grammar.host_copies().is_err());
            assert!(grammar.finish().is_err());
            let mut args = matrix_args(&h, members, down, true, true);
            let old = &args.buffers[owner];
            args.buffers[owner] = Handle::new(old.stream, old.size)
                .offset_start(old.offset_start.unwrap_or(0))
                .offset_end(old.offset_end.unwrap_or(0))
                .binding();
            assert!(
                grammar
                    .launch(
                        if down { TAIL_DOWN } else { TAIL_GU },
                        false,
                        &CubeCount::Static(if down { 40 } else { 20 }, 1, 1),
                        &args
                    )
                    .is_err()
            );
        }
        for index in 0..20 {
            let mut grammar = before(&h, members, down);
            prefix(&mut grammar, &h, members, down, name);
            let mut args = matrix_args(&h, members, down, true, true);
            let old = word(&args.info, index).unwrap();
            set_word(&mut args.info, index, old + 1);
            assert!(
                grammar
                    .launch(
                        if down { TAIL_DOWN } else { TAIL_GU },
                        false,
                        &CubeCount::Static(if down { 40 } else { 20 }, 1, 1),
                        &args
                    )
                    .is_err()
            );
        }
    }
}

#[cfg(feature = "kuno-expert-ordered-output")]
#[test]
fn reuse_ordered_down_keeps_one_layout_and_identical_real_tail_views() {
    let h = owners();
    for (first, second) in [(ORDERED, TAIL_DOWN), (DOWN, ORDERED_TAIL)] {
        let mut grammar = before(&h, 257, true);
        prefix(&mut grammar, &h, 257, true, first);
        assert!(
            grammar
                .launch(
                    second,
                    false,
                    &CubeCount::Static(40, 1, 1),
                    &matrix_args(&h, 257, true, true, true)
                )
                .is_err()
        );
    }
    for owner in 0..6 {
        let mut grammar = before(&h, 257, true);
        prefix(&mut grammar, &h, 257, true, ORDERED);
        let mut args = matrix_args(&h, 257, true, true, true);
        let old = &args.buffers[owner];
        args.buffers[owner] = Handle::new(old.stream, old.size)
            .offset_start(old.offset_start.unwrap_or(0))
            .offset_end(old.offset_end.unwrap_or(0))
            .binding();
        assert!(
            grammar
                .launch(ORDERED_TAIL, false, &CubeCount::Static(40, 1, 1), &args)
                .is_err()
        );
    }
}

#[cfg(feature="kuno-global-image-experts")]
mod global_register {
 use super::*;
 const GU:&str="probe::strata_matrix::native_expert_wide::global_register::expert_global_register_gu::ExpertGlobalRegisterGu<Runtime>";
 const DOWN:&str="probe::strata_matrix::native_expert_wide::global_register::expert_global_register_down::ExpertGlobalRegisterDown<Runtime>";
 #[test]
 fn global_register_entries_keep_real_checker_ordered_pending_layout_and_original_tail(){
  for members in [256,257,287,288,383,384,4096] {
   let h=owners();let mut grammar=begin_images(&h,members,true);
   prefix(&mut grammar,&h,members,false,GU);tail(&mut grammar,&h,members,false,false);
   assert!(grammar.finish().is_err());
   grammar.launch("StrataPromptExpertF16SwigluCheckGu",false,&CubeCount::Static((members*5).div_ceil(4),1,1),&checker(&h,members)).unwrap();
   prefix(&mut grammar,&h,members,true,DOWN);tail(&mut grammar,&h,members,true,true);
   let c=grammar.finish().unwrap();assert_eq!(c.direct_gu_prefixes,1);assert_eq!(c.gu_checks,1);assert_eq!(c.logical_phases,6);
  }
 }
 #[test]
 fn global_register_exact_roles_reject_forged_identity_grid_owner_and_info(){
  assert!(is_direct_gu(GU));assert!(is_wide(DOWN,true));assert!(ordered_down_layout(DOWN));
  assert!(!is_wide(GU,true));assert!(!is_direct_gu(DOWN));
  for suffix in ["::tail::ExpertTailGu<Runtime>","::expert_wide_gu_direct::ExpertWideGuDirect<Runtime>","::deep::expert_global_register_gu::ExpertGlobalRegisterGu<Runtime>","::expert_global_register_gu::ExpertGlobalRegisterGuFake<Runtime>","::expert_global_register_down::ExpertGlobalRegisterDown<Runtime>::extra"] {
   let n=format!("probe::strata_matrix::native_expert_wide::global_register{suffix}");
   assert!(!is_direct_gu(&n)&&!is_wide(&n,true)&&!is_wide(&n,false)&&!is_tail(&n,true)&&!is_tail(&n,false));
  }
  let h=owners();
  for index in 0..8 {let mut a=matrix_args(&h,287,false,true,false);let old=word(&a.info,index).unwrap();set_word(&mut a.info,index,old+1);
   assert!(begin_images(&h,287,true).launch(GU,false,&CubeCount::Static(10,2,1),&a).is_err());}
  assert!(begin_images(&h,287,true).launch(GU,false,&CubeCount::Static(10,3,1),&matrix_args(&h,287,false,true,false)).is_err());
  let mut a=matrix_args(&h,287,false,true,false);a.buffers[2]=Handle::new(h[0].stream,1).binding();
  assert!(begin_images(&h,287,true).launch(GU,false,&CubeCount::Static(10,2,1),&a).is_err());
 }
}

#[cfg(feature="kuno-global-register-n64")]
mod global_n64{
 use super::*;
 const GU:&str="probe::strata_matrix::native_expert_wide::global_n64::expert_global_n64_gu::ExpertGlobalN64Gu<Runtime>";
 const DOWN:&str="probe::strata_matrix::native_expert_wide::global_n64::expert_global_n64_down::ExpertGlobalN64Down<Runtime>";
 fn n64prefix(grammar:&mut NativeStrataExpertGrammar,h:&[Handle;7],m:u32,down:bool){grammar.launch(if down{DOWN}else{GU},false,&CubeCount::Static(if down{40}else{20},m/128,1),&matrix_args(h,m,down,true,false)).unwrap();}
 #[test]
 fn n64_genuine_prefix_grid_retains_original_checker_ordered_layout_tail_and_logical_blocks(){
  for m in [256,257,287,383,384,4096]{let h=owners();let mut g=begin_images(&h,m,true);
   n64prefix(&mut g,&h,m,false);tail(&mut g,&h,m,false,false);assert!(g.finish().is_err());
   g.launch("StrataPromptExpertF16SwigluCheckGu",false,&CubeCount::Static((m*5).div_ceil(4),1,1),&checker(&h,m)).unwrap();
   n64prefix(&mut g,&h,m,true);tail(&mut g,&h,m,true,true);let c=g.finish().unwrap();assert_eq!(c.logical_phases,6);assert_eq!(c.direct_gu_prefixes,1);assert_eq!(c.gu_checks,1);assert_eq!(c.wide_blocks,2*(m/128)as usize);
  }
 }
 #[test]
 fn n64_exact_new_roles_reject_old_grid_forged_name_info_and_changed_owner(){
  assert!(is_direct_gu(GU)&&is_wide(DOWN,true)&&ordered_down_layout(DOWN));
  for suffix in ["expert_wide_gu_direct::ExpertWideGuDirect<Runtime>","expert_global_n64_gu::ExpertGlobalN64GuFake<Runtime>","deep::expert_global_n64_gu::ExpertGlobalN64Gu<Runtime>","expert_global_n64_down::ExpertGlobalN64Down<Runtime>::extra"]{
   let n=format!("probe::strata_matrix::native_expert_wide::global_n64::{suffix}");assert!(!is_direct_gu(&n)&&!is_wide(&n,true)&&!is_wide(&n,false)&&!is_tail(&n,true)&&!is_tail(&n,false));
  }
  let h=owners();let m=287;
  for index in 0..8{let mut a=matrix_args(&h,m,false,true,false);let v=word(&a.info,index).unwrap();set_word(&mut a.info,index,v+1);assert!(begin_images(&h,m,true).launch(GU,false,&CubeCount::Static(20,2,1),&a).is_err());}
  assert!(begin_images(&h,m,true).launch(GU,false,&CubeCount::Static(10,2,1),&matrix_args(&h,m,false,true,false)).is_err());
  let mut a=matrix_args(&h,m,false,true,false);a.buffers[2]=Handle::new(h[0].stream,1).binding();assert!(begin_images(&h,m,true).launch(GU,false,&CubeCount::Static(20,2,1),&a).is_err());
 }
}
