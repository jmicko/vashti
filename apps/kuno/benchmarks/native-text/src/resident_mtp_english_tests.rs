// Public constructor/owner checks only. No model, client or GPU is created.
use super::*;

#[test]
fn english_public_budget_keeps_original_profile_private_owners_and_head_pages() {
    assert_eq!(ENGLISH_VOCAB_OWNED_BYTES, 262_144);
    assert_eq!(ENGLISH_PUBLIC_BYTES, 5_011_931_136);
    for profile in [MtpProfile::SelfContainedQ8, MtpProfile::StrataQ2] {
        assert_eq!(
            public_bytes_with_english(profile, false).unwrap(),
            public_bytes_for(profile)
        );
        assert_eq!(public_plan_for(profile).unwrap(), public_bytes_for(profile));
    }
    assert_eq!(
        public_bytes_with_english(MtpProfile::SelfContainedQ8, true).unwrap(),
        ENGLISH_PUBLIC_BYTES
    );
    assert_eq!(
        public_plan_with_english(MtpProfile::SelfContainedQ8, true).unwrap(),
        ENGLISH_PUBLIC_BYTES
    );
    assert!(public_bytes_with_english(MtpProfile::StrataQ2, true).is_err());
    assert_eq!(N, 42);
    assert_eq!(private_plan().unwrap(), 11_698_176);
    assert_eq!(WORDS[B::Logits as usize], k::VOCAB);
    assert_eq!(WORDS[B::Candidates as usize], k::GREEDY_WORDS);
    let head = MATRICES
        .iter()
        .copied()
        .find(|s| s.name == "output.weight")
        .unwrap();
    assert_eq!(
        (head.rows, head.columns, head.dtype),
        (248_320, 2_560, quant_matvec::Q8_0)
    );
    let pages = tile_plan(head).unwrap();
    assert_eq!(pages.len(), 16);
    let proof = english::EnglishVocabulary::embedded().unwrap();
    let mut dispatches = 0;
    let mut selected = 0;
    for (index, &(first, rows, logical, _)) in pages.iter().enumerate() {
        let segment = proof.segment(index).unwrap();
        assert_eq!(
            (first, rows, logical),
            (
                segment.source_first,
                segment.source_rows,
                rows * k::ROW_BYTES
            )
        );
        selected += segment.subset_rows;
        dispatches += usize::from(segment.subset_rows != 0);
    }
    assert_eq!((selected, dispatches), (40_525, 13));
    main_admission::check_capacity(
        MAIN_BYTES,
        &[
            ENGLISH_PUBLIC_BYTES,
            PRIVATE_BYTES as u64,
            RUNTIME_BYTES,
            PUBLIC_SOURCE_BYTES as u64,
        ],
    )
    .unwrap();
    const {
        assert!(
            2 * english::ENGLISH_OWNED_BYTES
                + MAX_LIVE_OUTPUTS * OUTPUT_HOST_BYTES
                + 2 * 1024 * 1024
                <= HOST_BYTES
        );
    }
    const {
        assert!(INFO_BYTES + 3 * 4 * 1024 * 1024 <= RUNTIME_BYTES);
    }
}

#[test]
fn english_public_readback_requires_exact_identity_and_every_padding_byte() {
    let mut bytes = vec![0_u8; english::ENGLISH_OWNED_BYTES];
    bytes[..english::ENGLISH_BYTES].copy_from_slice(english::ENGLISH_MAP);
    validate_english_readback(&bytes).unwrap();
    assert!(validate_english_readback(&bytes[..bytes.len() - 1]).is_err());
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(validate_english_readback(&extra).is_err());
    for offset in [
        0,
        4,
        english::ENGLISH_BYTES - 1,
        english::ENGLISH_BYTES,
        bytes.len() - 1,
    ] {
        let mut bad = bytes.clone();
        bad[offset] ^= 1;
        assert!(validate_english_readback(&bad).is_err());
    }
    // Device/public ownership checks cannot omit, forge or shrink the map.
    let owner = (
        english::ENGLISH_BYTES,
        english::ENGLISH_OWNED_BYTES,
        0,
        english::ENGLISH_ROWS,
        ENGLISH_VOCAB_OWNED_BYTES,
    );
    validate_english_owner(
        MtpProfile::SelfContainedQ8,
        true,
        ENGLISH_PUBLIC_BYTES,
        Some(owner),
    )
    .unwrap();
    validate_english_owner(MtpProfile::SelfContainedQ8, false, PUBLIC_BYTES, None).unwrap();
    assert!(
        validate_english_owner(MtpProfile::SelfContainedQ8, true, PUBLIC_BYTES, Some(owner))
            .is_err()
    );
    assert!(
        validate_english_owner(
            MtpProfile::SelfContainedQ8,
            true,
            ENGLISH_PUBLIC_BYTES,
            None
        )
        .is_err()
    );
    assert!(
        validate_english_owner(
            MtpProfile::SelfContainedQ8,
            false,
            PUBLIC_BYTES,
            Some(owner)
        )
        .is_err()
    );
    assert!(
        validate_english_owner(
            MtpProfile::StrataQ2,
            true,
            STRATA_PUBLIC_BYTES + ENGLISH_VOCAB_OWNED_BYTES,
            Some(owner)
        )
        .is_err()
    );
    for field in 0..5 {
        let mut bad = owner;
        match field {
            0 => bad.0 -= 4,
            1 => bad.1 -= 4,
            2 => bad.2 = 1,
            3 => bad.3 -= 1,
            _ => bad.4 -= 4,
        }
        assert!(
            validate_english_owner(
                MtpProfile::SelfContainedQ8,
                true,
                ENGLISH_PUBLIC_BYTES,
                Some(bad)
            )
            .is_err()
        );
    }
}

#[test]
fn english_draft_only_admission_and_completed_receipt_require_actual_healthy_forward() {
    validate_full_output(false).unwrap();
    assert!(validate_full_output(true).is_err());
    for english in [false, true] {
        for broadcast in [false, true] {
            assert_eq!(
                validate_english_broadcast(english, broadcast).is_ok(),
                !(english && broadcast)
            );
        }
    }
    // The separate shared-scales selector still admits English with no broadcast.
    validate_q8_head_modes(false, true).unwrap();
    for selected in [false, true] {
        for completed in [false, true] {
            assert_eq!(
                completed_english_policy(selected, completed, false, false, false).unwrap(),
                selected && completed
            );
            for bad in 0..3 {
                assert!(
                    completed_english_policy(selected, completed, bad == 0, bad == 1, bad == 2)
                        .is_err()
                );
            }
        }
    }
    // Full Output's original full-vocabulary assertion is retained.
    let mut scores = vec![-1_f32; k::VOCAB];
    scores[100_000] = 1.;
    assert_eq!(greedy_host(&scores).unwrap(), 100_000);
}

#[test]
fn english_actual_public_map_guard_lowers_integer_order_range_and_padding_checks() {
    use cubecl::{Compiler, WgpuCompilationOptions, server::ExecutionMode, wgpu::WgslCompiler};
    use wgpu::naga::{self, AddressSpace, Expression, Literal};
    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
        let mut builder = KernelBuilder::default();
        builder.runtime_properties(WgpuRuntime::target_properties());
        AddressType::U32.register(&mut builder.scope);
        let words = u32::as_type(&builder.scope);
        let atomic = Atomic::<u32>::as_type(&builder.scope);
        let input = builder.input_array(words).into();
        let fault = builder.output_array(atomic).into();
        let logical = builder.scalar(AddressType::U32.unsigned_type()).into();
        public_guard::expand(
            &mut builder.scope,
            input,
            fault,
            logical,
            PUBLIC_ENGLISH_IDS,
        );
        let source = WgslCompiler::default()
            .compile(
                builder.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("public_english_ids_guard"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                AddressType::U32.unsigned_type(),
            )
            .unwrap()
            .to_string();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SUBGROUP,
        )
        .validate(&module)
        .unwrap();
        let spirv = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options {
                zero_initialize_workgroup_memory:
                    naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                ..Default::default()
            },
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: module.entry_points[0].name.clone(),
            }),
        )
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert!(
            module
                .global_variables
                .iter()
                .all(|(_, variable)| variable.space != AddressSpace::WorkGroup)
        );
        assert!(!source.contains("f32") && source.contains("atomicOr"));
        let expressions = &module.entry_points[0].function.expressions;
        for constant in [english::ENGLISH_BYTES as u32, k::VOCAB as u32] {
            assert!(expressions.iter().any(|(_, expression)| matches!(expression, Expression::Literal(Literal::U32(value)) if *value == constant)));
        }
        let mut cursor = 5;
        while cursor < spirv.len() {
            let words = (spirv[cursor] >> 16) as usize;
            assert!(words > 0);
            assert_ne!(spirv[cursor] & 65535, 224); // No initialization/barrier dependency.
            cursor += words;
        }
        assert_eq!(cursor, spirv.len());
    }
}
