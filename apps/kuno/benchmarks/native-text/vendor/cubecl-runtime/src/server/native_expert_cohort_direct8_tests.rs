use super::*;
fn direct_program() -> Vec<Entry> {
    let mut entries = program(true, cfg!(feature = "kuno-expert-ordered-output"));
    entries[1].name = "StrataExpertCohortGuDirect8Image";
    entries[2].name = "StrataExpertCohortDownDirect8Image";
    entries[1].count = CubeCount::Static(12800, 1, 2);
    entries[2].count = CubeCount::Static(6400, 1, 2);
    entries
}
#[test]
fn direct8_cohort_real_names_wg32_grid_both_families_and_owners_are_accepted() {
    run(&direct_program()).unwrap();
    for index in [1, 2] {
        for old in [false, true] {
            let mut v = direct_program();
            let original = program(old, false);
            v[index].count = match &original[index].count {
                CubeCount::Static(x, y, z) => CubeCount::Static(*x, *y, *z),
                _ => unreachable!(),
            };
            assert!(run(&v).is_err());
        }
        let mut v = direct_program();
        v[index].name = if index == 1 {
            "StrataExpertCohortDownDirect8Image"
        } else {
            "StrataExpertCohortGuDirect8Image"
        };
        assert!(run(&v).is_err());
    }
}
#[test]
fn direct8_cohort_rejects_aliases_mixed_image_family_info_and_owning_view_forgeries() {
    for name in [
        "StrataExpertCohortGuDirect8ImageFake",
        "prefix::StrataExpertCohortGuDirect8Image",
    ] {
        let mut v = direct_program();
        v[1].name = name;
        assert!(run(&v).is_err());
    }
    for quad in [false, true] {
        let mut v = direct_program();
        let old = program(quad, false);
        v[2].name = old[2].name;
        v[2].count = match &old[2].count {
            CubeCount::Static(x, y, z) => CubeCount::Static(*x, *y, *z),
            _ => unreachable!(),
        };
        assert!(run(&v).is_err());
    }
    for index in [1, 2] {
        for owner in 0..direct_program()[index].args.buffers.len() {
            let mut v = direct_program();
            let old = &v[index].args.buffers[owner];
            let used = View::new(old).unwrap().used();
            v[index].args.buffers[owner] = Handle::new(StreamId::current(), used).binding();
            // Actual public quantized roots are independently authenticated by
            // physical-source lowering. Grammar binds private/shared owners.
            if owner >= if index == 1 { 2 } else { 1 } {
                assert!(run(&v).is_err());
            }
        }
        for word_index in [1usize, 2, 5, 9, 16, 17, 18, 19] {
            let mut v = direct_program();
            let mut bytes = v[index].args.info.data[word_index / 2].to_ne_bytes();
            let at = word_index % 2 * 4;
            let x = u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
            bytes[at..at + 4].copy_from_slice(&x.wrapping_add(1).to_ne_bytes());
            v[index].args.info.data[word_index / 2] = u64::from_ne_bytes(bytes);
            assert!(run(&v).is_err());
        }
    }
}
