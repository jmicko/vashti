use super::*;
const GLOBAL:&str="probe::strata_matrix::cohorts::granular_small::expert_granular_cohort::ExpertGranularCohort<Runtime>";
fn global_program()->Vec<Entry>{let mut p=program(false,true);for e in &mut p{if(e.name==ORIGINAL||e.name==ORDERED)&&word(&e.args.info,11).unwrap()==16{e.name=GLOBAL;let CubeCount::Static(x,y,z)=&e.count else{panic!("literal fixture grid")};e.count=CubeCount::Static(*x,*y,*z);}}p}
#[test]
fn granular_small_cohort16_keeps_tile8_packed_slots_ordered_layout_and_checked_activation(){
 assert_eq!(matrix_variant(GLOBAL,false),Some(false));assert_eq!(matrix_variant(GLOBAL,true),Some(true));run(&global_program()).unwrap();
 let mut missing=global_program();missing.remove(5);assert!(run(&missing).is_err());
}
#[test]
fn granular_small_cohort_refuses_tile8_aliases_missing_phases_and_changed_matrix_owners(){
 for name in ["probe::strata_matrix::cohorts::granular_small::expert_cohort::ExpertCohort<Runtime>","probe::strata_matrix::cohorts::granular_small::expert_granular_cohort::ExpertGranularCohortFake<Runtime>","probe::strata_matrix::cohorts::granular_small::expert_granular_cohort::ExpertGranularCohort<Runtime>::other"]{assert_eq!(matrix_variant(name,false),None);assert_eq!(matrix_variant(name,true),None);}
 let mut p=global_program();p[3].name=GLOBAL;assert!(run(&p).is_err());
 for index in [4usize,7]{for owner in 0..6{let mut p=global_program();let bytes=View::new(&p[index].args.buffers[owner]).unwrap().used();p[index].args.buffers[owner]=Handle::new(StreamId::current(),bytes).binding();assert!(run(&p).is_err());}
  for field in [1usize,2,4,5,9,11,16,48,54]{let mut p=global_program();let v=word(&p[index].args.info,field).unwrap();set_word(&mut p[index].args.info,field,v+1);assert!(run(&p).is_err());}
  let mut p=global_program();p[index].count=CubeCount::Static(1,1,1);assert!(run(&p).is_err());
 }
}
