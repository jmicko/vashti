use super::*;
fn begin_gather(h: &[Handle; 7], members: u32) -> NativeStrataExpertGrammar {
    let mut grammar = NativeStrataExpertGrammar::default();
    let mut words = [0; 16];
    words[..3].copy_from_slice(&[members, 2560, 1280]);
    let gather = KernelArguments::new()
        .with_buffers(
            [h[3].clone(), h[0].clone(), h[0].clone(), h[5].clone()]
                .into_iter()
                .map(Handle::binding)
                .collect(),
        )
        .with_info(MetadataBindingInfo::custom(packed(&words)));
    grammar
        .launch(
            "StrataPromptExpertGatherPad32",
            false,
            &CubeCount::Static(members * 5, 1, 1),
            &gather,
        )
        .unwrap();
    grammar
}
fn begin_images(h: &[Handle; 7], members: u32, resident: bool) -> NativeStrataExpertGrammar {
    let mut grammar = begin_gather(h, members);
    for down in [false, true] {
        grammar
            .launch(
                if resident {
                    if down {
                        "StrataPromptExpertResidentDownImage"
                    } else {
                        "StrataPromptExpertResidentGuImage"
                    }
                } else {
                    "StrataPromptExpertF16Image"
                },
                down && !resident,
                &CubeCount::Static(1, 1, 1),
                &KernelArguments::new(),
            )
            .unwrap();
    }
    grammar
}
fn product(
    grammar: &mut NativeStrataExpertGrammar,
    h: &[Handle; 7],
    members: u32,
    down: bool,
    wide: bool,
) {
    let rows = if down { 2560 } else { 1280 };
    let prefix = members / 128 * 128;
    let args = matrix_args(h, members, down, wide, false);
    grammar
        .launch(
            if wide {
                if down { WIDE_DOWN } else { WIDE_GU }
            } else {
                ORIGINAL
            },
            false,
            &if wide {
                CubeCount::Static(rows / 128, prefix / 128, 1)
            } else {
                CubeCount::Static(rows / 64, members.div_ceil(32), 1)
            },
            &args,
        )
        .unwrap();
    if wide && prefix < members {
        let remain = members - prefix;
        let tile = if remain <= 8 {
            8
        } else if remain < 32 {
            16
        } else {
            32
        };
        grammar
            .launch(
                if down { TAIL_DOWN } else { TAIL_GU },
                false,
                &CubeCount::Static(rows / 64, remain.div_ceil(tile), 1),
                &matrix_args(h, members, down, true, true),
            )
            .unwrap();
    }
}
fn activate(grammar: &mut NativeStrataExpertGrammar) {
    grammar
        .launch(
            "StrataPromptExpertF16Swiglu",
            false,
            &CubeCount::Static(1, 1, 1),
            &KernelArguments::new(),
        )
        .unwrap();
}

fn begin(h:&[Handle;7],members:u32)->NativeStrataExpertGrammar{begin_images(h,members,false)}
const GU32:&str="probe::strata_matrix::native::balanced_global::expert_global_macro32_gu::ExpertGlobalMacro32Gu<Runtime>";
const DN32:&str="probe::strata_matrix::native::balanced_global::expert_global_macro32_down::ExpertGlobalMacro32Down<Runtime>";
const GU64:&str="probe::strata_matrix::native::balanced_global::expert_global_macro64_gu::ExpertGlobalMacro64Gu<Runtime>";
const DN64:&str="probe::strata_matrix::native::balanced_global::expert_global_macro64_down::ExpertGlobalMacro64Down<Runtime>";
#[test]
fn balanced_global_full_roles_keep_original_six_phases_checker_and_owners(){
 #[allow(dead_code)] struct LegacyGroup {index:u32,members:u32,prompt:u32,input:View,map:View,fault:View}
 assert_eq!(core::mem::size_of::<Group>(),core::mem::size_of::<LegacyGroup>());

 for (gu,dn,tile,counts)in [(GU32,DN32,32u32,&[32u32,33,63][..]),(GU64,DN64,64,&[64,65,127,128,129,255][..])]{for &n in counts{for resident in [false,true]{
  let h=owners();let mut g=begin_images(&h,n,resident);
  g.launch(gu,false,&CubeCount::Static(20,n.div_ceil(tile),1),&matrix_args(&h,n,false,false,false)).unwrap();assert!(g.finish().is_err());
  g.launch("StrataPromptExpertF16SwigluCheckGuPad32",false,&CubeCount::Static((n*5).div_ceil(4),1,1),&checker(&h,n)).unwrap();
  g.launch(dn,false,&CubeCount::Static(40,n.div_ceil(tile),1),&matrix_args(&h,n,true,false,false)).unwrap();
  let c=g.finish().unwrap();assert_eq!((c.launches,c.logical_phases,c.matrix_pieces,c.gu_checks),(6,6,2,1));assert_eq!((c.wide_prefixes,c.tail_pieces),(0,0));
 }}}
}
#[test]
fn balanced_global_rejects_forged_roles_grids_scope_info_and_changed_private_owners(){
 // Spare bytes or old producers do not certify initialized padding.
 let h=owners();let n=33u32;let mut g=super::begin(&h,n);
 assert!(g.launch(GU32,false,&CubeCount::Static(20,n.div_ceil(32),1),&matrix_args(&h,n,false,false,false)).is_err());
 let mut g=begin(&h,n);g.launch(GU32,false,&CubeCount::Static(20,n.div_ceil(32),1),&matrix_args(&h,n,false,false,false)).unwrap();
 g.launch("StrataPromptExpertF16SwigluCheckGu",false,&CubeCount::Static((n*5).div_ceil(4),1,1),&checker(&h,n)).unwrap();
 assert!(g.launch(DN32,false,&CubeCount::Static(40,n.div_ceil(32),1),&matrix_args(&h,n,true,false,false)).is_err());
 let mut h=owners();let end=h[0].size_in_used()-u64::from(n)*2560*2;h[0]=h[0].clone().offset_end(end);let mut g=begin(&h,n);
 assert!(g.launch(GU32,false,&CubeCount::Static(20,n.div_ceil(32),1),&matrix_args(&h,n,false,false,false)).is_err());
 for name in ["StrataPromptExpertGatherPad32Fake","StrataPromptExpertF16SwigluCheckGuPad32::other"] {for phase in 0..6 {assert!(!native_strata_expert_operator(name,false,phase));}}

 for(gu,dn,n,tile)in [(GU32,DN32,33u32,32u32),(GU64,DN64,65,64)]{
  assert!(!native_strata_expert_operator(gu,false,5));assert!(!native_strata_expert_operator(dn,false,3));
  for bad in 0..10{let h=owners();let mut g=begin(&h,n);let mut args=matrix_args(&h,n,false,false,false);
   if bad<6{args.buffers[bad]=h[bad].clone().offset_start(8).binding();}
   if bad==6{set_word(&mut args.info,2,n+1);}if bad==7{set_word(&mut args.info,6,1);}
   let name=if bad==8{dn}else{gu};let grid=CubeCount::Static(if bad==9{19}else{20},n.div_ceil(tile),1);
   assert!(g.launch(name,false,&grid,&args).is_err());
  }
  let h=owners();let mut g=begin(&h,n);g.launch(gu,false,&CubeCount::Static(20,n.div_ceil(tile),1),&matrix_args(&h,n,false,false,false)).unwrap();assert!(g.launch("StrataPromptExpertF16Swiglu",false,&CubeCount::Static((n*5).div_ceil(4),1,1),&checker(&h,n)).is_err());
 }
 for(name,n,tile)in [(GU32,31u32,32u32),(GU32,64,32),(GU64,63,64),(GU64,256,64)]{let h=owners();let mut g=begin(&h,n);assert!(g.launch(name,false,&CubeCount::Static(20,n.div_ceil(tile),1),&matrix_args(&h,n,false,false,false)).is_err());}
 for name in ["probe::strata_matrix::native::balanced_global::expert_macro::ExpertMacro<Runtime>","probe::strata_matrix::native::balanced_global::expert_global_macro32_gu::ExpertGlobalMacro32GuFake<Runtime>","probe::strata_matrix::native::balanced_global::native_expert_wide::expert_tail_gu::ExpertTailGu<Runtime>"]{for phase in 0..6{assert!(!native_strata_expert_operator(name,false,phase));}}
}
