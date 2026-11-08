use super::*;
const GU32:&str="probe::strata_matrix::native::global_small::expert_global_macro32_gu::ExpertGlobalMacro32Gu<Runtime>";
const DN32:&str="probe::strata_matrix::native::global_small::expert_global_macro32_down::ExpertGlobalMacro32Down<Runtime>";
const GU64:&str="probe::strata_matrix::native::global_small::expert_global_macro64_gu::ExpertGlobalMacro64Gu<Runtime>";
const DN64:&str="probe::strata_matrix::native::global_small::expert_global_macro64_down::ExpertGlobalMacro64Down<Runtime>";
#[test]
fn global_small_full_roles_keep_original_six_phases_checker_and_owners(){
 for (gu,dn,tile,counts)in [(GU32,DN32,32u32,&[32u32,33,63][..]),(GU64,DN64,64,&[64,65,127,128,129,255][..])]{for &n in counts{for resident in [false,true]{
  let h=owners();let mut g=begin_images(&h,n,resident);
  g.launch(gu,false,&CubeCount::Static(20,n.div_ceil(tile),1),&matrix_args(&h,n,false,false,false)).unwrap();assert!(g.finish().is_err());
  g.launch("StrataPromptExpertF16SwigluCheckGu",false,&CubeCount::Static((n*5).div_ceil(4),1,1),&checker(&h,n)).unwrap();
  g.launch(dn,false,&CubeCount::Static(40,n.div_ceil(tile),1),&matrix_args(&h,n,true,false,false)).unwrap();
  let c=g.finish().unwrap();assert_eq!((c.launches,c.logical_phases,c.matrix_pieces,c.gu_checks),(6,6,2,1));assert_eq!((c.wide_prefixes,c.tail_pieces),(0,0));
 }}}
}
#[test]
fn global_small_rejects_forged_roles_grids_scope_info_and_changed_private_owners(){
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
 for name in ["probe::strata_matrix::native::global_small::expert_macro::ExpertMacro<Runtime>","probe::strata_matrix::native::global_small::expert_global_macro32_gu::ExpertGlobalMacro32GuFake<Runtime>","probe::strata_matrix::native::global_small::native_expert_wide::expert_tail_gu::ExpertTailGu<Runtime>"]{for phase in 0..6{assert!(!native_strata_expert_operator(name,false,phase));}}
}
