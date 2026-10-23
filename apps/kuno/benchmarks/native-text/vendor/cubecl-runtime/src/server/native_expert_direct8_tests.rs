use super::*;

fn image_args(down: bool) -> KernelArguments {
    let mut words = [0; 16];
    words[0] = 257;
    words[1] = if down { 640 } else { 2560 };
    words[2] = if down { 2560 } else { 1280 };
    words[5] = if down { 2 } else { 0 };
    KernelArguments::new().with_info(MetadataBindingInfo::custom(packed(&words)))
}
fn pair(g: &mut NativeStrataExpertGrammar, resident: bool) -> Result<(), ServerError> {
    for down in [false, true] {
        let name = if resident {
            if down {
                "StrataPromptExpertResidentDownDirect8Image"
            } else {
                "StrataPromptExpertResidentGuDirect8Image"
            }
        } else {
            "StrataPromptExpertF16Direct8Image"
        };
        g.launch(
            name,
            down && !resident,
            &CubeCount::Static(if down { 6400 } else { 12800 }, 1, 1),
            &image_args(down),
        )?;
    }
    Ok(())
}

#[test]
fn direct8_regular_pairs_preserve_all_six_phases_full_matrix_owners_and_tail() {
    for resident in [false, true] {
        let h = owners();
        let mut g = begin_gather(&h, 257);
        pair(&mut g, resident).unwrap();
        product(&mut g, &h, 257, false, true);
        activate(&mut g);
        product(&mut g, &h, 257, true, true);
        let r = g.finish().unwrap();
        assert_eq!(r.logical_phases, 6);
        assert_eq!(r.matrix_pieces, 4);
        assert_eq!(r.tail_pieces, 2);
    }
}

#[test]
fn direct8_closed_identities_roles_release_and_wg32_grid_cannot_be_forged() {
    for (name, phase, release) in [
        ("StrataPromptExpertResidentGuDirect8Image", 2, false),
        ("StrataPromptExpertResidentDownDirect8Image", 1, false),
        ("StrataPromptExpertF16Direct8Image", 2, false),
        ("StrataPromptExpertF16Direct8ImageFake", 1, false),
        ("prefix::StrataPromptExpertF16Direct8Image", 1, false),
        ("StrataPromptExpertResidentGuDirect8Image", 1, true),
    ] {
        assert!(!native_strata_expert_operator(name, release, phase));
    }
    let h = owners();
    for grid in [
        CubeCount::Static(1600, 1, 1),
        CubeCount::Static(400, 1, 1),
        CubeCount::Static(12800, 2, 1),
    ] {
        let mut g = begin_gather(&h, 257);
        assert!(
            g.launch(
                "StrataPromptExpertF16Direct8Image",
                false,
                &grid,
                &image_args(false)
            )
            .is_err()
        );
    }
    for index in [1, 2, 5] {
        let mut args = image_args(false);
        set_word(&mut args.info, index, 999);
        let mut g = begin_gather(&h, 257);
        assert!(
            g.launch(
                "StrataPromptExpertF16Direct8Image",
                false,
                &CubeCount::Static(12800, 1, 1),
                &args
            )
            .is_err()
        );
    }
}

#[test]
fn direct8_keeps_resident_streamed_pairing_and_existing_matrix_owner_rejection() {
    let h = owners();
    let mut g = begin_gather(&h, 257);
    g.launch(
        "StrataPromptExpertResidentGuDirect8Image",
        false,
        &CubeCount::Static(12800, 1, 1),
        &image_args(false),
    )
    .unwrap();
    assert!(
        g.launch(
            "StrataPromptExpertF16Direct8Image",
            true,
            &CubeCount::Static(6400, 1, 1),
            &image_args(true)
        )
        .is_err()
    );
    let mut g = begin_gather(&h, 257);
    pair(&mut g, false).unwrap();
    let mut args = matrix_args(&h, 257, false, true, false);
    args.buffers[0] = Handle::new(StreamId::current(), 80 * 1024 * 1024).binding();
    assert!(
        g.launch(WIDE_GU, false, &CubeCount::Static(20, 2, 1), &args)
            .is_err()
    );
}
