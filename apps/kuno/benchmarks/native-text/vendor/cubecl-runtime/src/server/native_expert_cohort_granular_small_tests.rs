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

#[cfg(feature="kuno-small-dual-cells")]
const DUAL:&str="probe::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohort<Runtime>";
#[cfg(feature="kuno-small-dual-cells")]
fn dual_program()->Vec<Entry>{let mut p=global_program();for e in &mut p{if e.name==GLOBAL {e.name=DUAL;let CubeCount::Static(x,y,z)=e.count else{panic!("literal grid")};e.count=CubeCount::Static(x*2,y,z);}}p}
#[cfg(feature="kuno-small-dual-cells")]
#[test]
fn dual_small_exact_n32_grid_keeps_tile8_original_packing_and_phases(){run(&dual_program()).unwrap();assert_eq!(matrix_variant(DUAL,true),Some(true));}
#[cfg(feature="kuno-small-dual-cells")]
#[test]
fn dual_small_refuses_old_grid_tile8_aliases_and_changed_owners(){
 for index in [4usize,7] {let mut p=dual_program();let CubeCount::Static(x,y,z)=p[index].count else{panic!("literal grid")};p[index].count=CubeCount::Static(x/2,y,z);assert!(run(&p).is_err());}
 let mut p=dual_program();p[3].name=DUAL;assert!(run(&p).is_err());
 for name in ["probe::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohortFake<R>","probe::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohort<R>::tail","probe::strata_matrix::nested::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohort<R>"]{assert_eq!(matrix_variant(name,false),None);}
}

#[cfg(feature="kuno-small-dual-cells")]
#[test]
fn dual_small_outer_grammar_accepts_genuine_scope_and_rejects_forged_or_unscoped_consumers() {
    let outer = |entries: &[Entry]| -> Result<NativeStrataExpertGrammarCounts,ServerError> {
        let mut grammar=NativeStrataExpertGrammar::default();
        for e in entries {grammar.launch(e.name,e.release,&e.count,&e.args)?;}
        grammar.finish()
    };
    let accepted=outer(&dual_program()).unwrap();
    assert_eq!((accepted.cohort_groups,accepted.cohort_batches,accepted.cohort_launches),(2,1,8));
    assert_eq!(accepted.regular_cohort_groups,0);
    let p=dual_program();
    for index in [4usize,7] {
        let e=&p[index];let mut outside=NativeStrataExpertGrammar::default();
        assert!(outside.launch(e.name,e.release,&e.count,&e.args).is_err());
        let mut old_grid=dual_program();
        let CubeCount::Static(x,y,z)=old_grid[index].count else {panic!("literal grid")};
        old_grid[index].count=CubeCount::Static(x/2,y,z);assert!(outer(&old_grid).is_err());
        for name in [
            "probe::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohortFake<R>",
            "probe::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohort<R>::tail",
            "probe::strata_matrix::nested::strata_matrix::cohorts::dual_small::expert_dual_small_cohort::ExpertDualSmallCohort<R>",
        ] {
            let mut forged=dual_program();forged[index].name=name;assert!(outer(&forged).is_err());
        }
    }
}
