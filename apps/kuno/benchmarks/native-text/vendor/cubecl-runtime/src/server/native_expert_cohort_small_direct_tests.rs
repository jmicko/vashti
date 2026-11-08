//! Both cohort matrix sizes retain their actual INFO/owner/activation grammar.
use super::*;
const DIRECT: &str="probe::strata_matrix::cohorts::small_direct::expert_cohort::ExpertCohort<Runtime>";
fn direct_program()->Vec<Entry>{
    let mut p=program(false,true);
    for e in &mut p {if e.name==ORIGINAL||e.name==ORDERED {e.name=DIRECT;}}
    p
}
#[test]
fn genuine_small_cohort_gu_and_ordered_down_keep_both_tiles_and_checked_activation(){
    assert_eq!(matrix_variant(DIRECT,false),Some(false));
    assert_eq!(matrix_variant(DIRECT,true),Some(true));
    run(&direct_program()).unwrap();
    let mut missing=direct_program();missing.remove(5);assert!(run(&missing).is_err());
    let mut wrong=direct_program();wrong[5].name="StrataPromptExpertF16Swiglu";assert!(run(&wrong).is_err());
    // One checked original GU tile and one raw tile still share the mandatory
    // same complete-owner checker. Ordered DOWN families share one layout.
    let mut mixed=direct_program();mixed[3].name=ORIGINAL;mixed[6].name=ORDERED;run(&mixed).unwrap();
}
#[test]
fn small_cohort_rejects_names_roles_rank_layout_aliases_and_all_changed_private_owners(){
    for name in ["probe::cohorts::small_direct::expert_cohort::ExpertCohort<Runtime>",
        "probe::strata_matrix::cohorts::small_direct::expert_cohort::ExpertCohortFake<Runtime>",
        "probe::strata_matrix::cohorts::small_direct::small_direct::expert_cohort::ExpertCohort<Runtime>",
        "probe::strata_matrix::cohorts::small_direct::expert_cohort::ExpertCohort<Runtime>::other",
        "probe::strata_matrix::cohorts::small_direct::expert_cohort::ExpertCohort<Runtime>::other<Runtime>"]{
        assert_eq!(matrix_variant(name,false),None);assert_eq!(matrix_variant(name,true),None);
    }
    let mut mixed=direct_program();mixed[7].name=ORIGINAL;assert!(run(&mixed).is_err());
    for index in [3,4,6,7] {
        for owner in 0..6 {
            let mut p=direct_program();let bytes=View::new(&p[index].args.buffers[owner]).unwrap().used();
            p[index].args.buffers[owner]=Handle::new(StreamId::current(),bytes).binding();
            assert!(run(&p).is_err());
        }
        for word_index in [1usize,2,4,5,9,11,16,48,54] {
            let mut p=direct_program();let old=word(&p[index].args.info,word_index).unwrap();
            set_word(&mut p[index].args.info,word_index,old+1);assert!(run(&p).is_err());
        }
        let mut p=direct_program();p[index].release=true;assert!(run(&p).is_err());
        let mut p=direct_program();p[index].count=CubeCount::Static(1,1,1);assert!(run(&p).is_err());
    }
}
