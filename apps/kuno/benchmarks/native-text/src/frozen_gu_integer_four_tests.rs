//! Reuse the original immutable decoder/oracle helpers; prove the fourth member.
use super::*;
use crate::quant_matvec::frozen_gu_integer::four;

fn lower(kernel: four::Kernel) -> n::Module {
    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
        let compiled = kernel
            .compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                mode,
                kernel.address_type(),
            )
            .unwrap();
        assert_eq!(compiled.source, kernel.source());
    }
    let module = n::front::wgsl::parse_str(&kernel.source()).unwrap();
    let info = n::valid::Validator::new(
        n::valid::ValidationFlags::all(),
        n::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        assert!(
            function
                .local_variables
                .iter()
                .all(|(_, var)| !matches!(module.types[var.ty].inner, n::TypeInner::Array { .. }))
        );
    }
    let words = n::back::spv::write_vec(
        &module,
        &info,
        &n::back::spv::Options {
            lang_version: (1, 3),
            zero_initialize_workgroup_memory: n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            ..Default::default()
        },
        Some(&n::back::spv::PipelineOptions {
            shader_stage: n::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .unwrap();
    assert!(!words.is_empty());
    module
}

fn remove_fourth(body: &str) -> String {
    let mut out = body.to_owned();
    while let Some(start) = out.find("if length>3u {\n") {
        let end = start + out[start..].find("\n}\n").unwrap() + 3;
        out.replace_range(start..end, "");
    }
    out.replace("vec4<f32>", "vec3<f32>")
        .replace("vec4<i32>", "vec3<i32>")
        .replace("j<4u", "j<3u")
        .replace("*4u+j", "*3u+j")
        .replace("local*4u+", "local*3u+")
}

#[test]
fn actual_four_member_gu_sources_keep_original_xyz_math_and_initialize256_cells() {
    let prepared = lower(four::Kernel::Prepare);
    assert_eq!(shared_bytes(&prepared), MAX_PREPARE_SHARED_BYTES);
    assert_eq!(barriers(&prepared.entry_points[0].function.body), 2);
    assert_eq!(four::LOGICAL_WORDS, 5824);
    assert_eq!(four::MAIN_BYTES, MAIN_BYTES);
    assert_eq!(four::RUNTIME_INCREMENT, RUNTIME_INCREMENT);
    for dtype in [
        crate::quant_matvec::IQ2_S,
        crate::quant_matvec::IQ3_XXS,
        crate::quant_matvec::IQ3_S,
        crate::quant_matvec::IQ4_XS,
    ] {
        let row_bytes = TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(2560)
            .unwrap() as usize;
        let old = Kernel::Project { dtype, row_bytes }.source();
        let four_source = four::Kernel::Project { dtype, row_bytes }.source();
        let begin = "// FROZEN_GU_PROJECT_BODY_BEGIN\n";
        let end = "// FROZEN_GU_PROJECT_BODY_END";
        let old_body = old.split_once(begin).unwrap().1.split_once(end).unwrap().0;
        let four_body = four_source
            .split_once(begin)
            .unwrap()
            .1
            .split_once(end)
            .unwrap()
            .0;
        assert_eq!(remove_fourth(four_body), old_body);
        for kernel in [
            four::Kernel::Project { dtype, row_bytes },
            four::Kernel::ProjectStride4 { dtype, row_bytes },
        ] {
            let source = kernel.source();
            let module = lower(kernel);
            assert_eq!(
                shared_bytes(&module),
                integer::original_grid_words(dtype).len() as u32 * 4 + 2048
            );
            assert!(source.contains("var gates=vec4<f32>") && source.contains("j<4u;j++"));
            assert!(source.contains("gate_sums[local*4u+j]=gates[j]"));
            assert!(source.contains("if length>3u") && source.contains("frames.w"));
            assert!(source.contains("p.h!=4u"));
            assert!(!source.contains("p.h>3u"));
            if matches!(kernel, four::Kernel::ProjectStride4 { .. }) {
                assert!(source.contains(&four_body.replace("group.y", "routed_group")));
                assert_eq!(barriers(&module.entry_points[0].function.body), 4);
                assert!(
                    source.find("if !any_valid {return;}").unwrap()
                        < source.find("workgroupBarrier();").unwrap()
                );
                assert!(source.contains("p.g>40u"));
                assert_eq!(
                    source
                        .matches("routed_group=group.y;routed_group<p.g;routed_group+=4u")
                        .count(),
                    2
                );
                assert!(
                    source.rfind("status[p.f+route]=1u").unwrap()
                        < source.rfind("workgroupBarrier();").unwrap()
                );
            } else {
                assert_eq!(barriers(&module.entry_points[0].function.body), 3);
            }
            assert_ne!(kernel.id(), Kernel::Project { dtype, row_bytes }.id());
            assert_ne!(
                kernel.id(),
                Kernel::ProjectStride4 { dtype, row_bytes }.id()
            );
        }
    }
    let audit = lower(four::Kernel::Audit);
    assert_eq!(shared_bytes(&audit), 0);
    assert!(four::Kernel::Audit.source().contains("at<5824u"));
    assert!(!four::Kernel::Audit.source().contains("atomicStore"));
}

#[test]
fn four_member_gu_full_layout_keeps_max8_owners_and_rejects_count_or_capacity_fallback() {
    let old = geometry(crate::quant_matvec::IQ3_XXS, crate::quant_matvec::IQ4_NL);
    let selected = old.with_four_members().unwrap();
    let spans = g::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
    let words = [
        g::MAP_WORDS,
        8 * 2560,
        80 * 640,
        80 * 2560,
        80,
        8 * 2560,
        selected.status_words(),
    ];
    four::validate_layout(selected, &spans, words, SCRATCH_WORDS, 4, 40).unwrap();
    assert!(four::validate_layout(old, &spans, words, SCRATCH_WORDS, 4, 40).is_err());
    for count in [0, 1, 2, 3, 5, 8, usize::MAX] {
        assert!(four::validate_layout(selected, &spans, words, SCRATCH_WORDS, count, 40).is_err());
    }
    for groups in [0, 41, 80, usize::MAX] {
        assert!(four::validate_layout(selected, &spans, words, SCRATCH_WORDS, 4, groups).is_err());
    }
    for owner in 0..7 {
        let mut short = words;
        short[owner] -= 1;
        assert!(four::validate_layout(selected, &spans, short, SCRATCH_WORDS, 4, 40).is_err());
    }
    for size in [5824, 16383, 16385] {
        assert!(four::validate_layout(selected, &spans, words, size, 4, 40).is_err());
    }
    // Selected immutable capacity4 still uses byte-identical original kernels
    // for the real count1..3 tails; that is not a four-member dispatch credit.
    for count in 1..=3 {
        validate_layout(selected, &spans, words, SCRATCH_WORDS, count, 30).unwrap();
    }
}

#[test]
fn four_member_gu_real_route_permutations_preserve_each_original_projection_bits() {
    for dtype in [
        TensorType::Iq2S,
        TensorType::Iq3Xxs,
        TensorType::Iq3S,
        TensorType::Iq4Xs,
    ] {
        let gate = iq_fixtures::packed(dtype.gguf_id(), 8, 2560, 1);
        let up = iq_fixtures::packed(dtype.gguf_id(), 8, 2560, 3);
        let bytes = dtype.row_bytes(2560).unwrap() as usize;
        let inputs: Vec<_> = (0..4)
            .map(|token| {
                (0..2560)
                    .map(|i| ((i * (17 + token * 7)) % 311) as f32 / 193. - 0.7)
                    .collect::<Vec<_>>()
            })
            .collect();
        let planes: Vec<_> = inputs
            .iter()
            .map(|x| oracle::Planes::new(x).unwrap())
            .collect();
        let packed: Vec<_> = planes.iter().map(|p| p.packed_words().unwrap()).collect();
        for routes in [[37usize, 3, 15, 27], [7, 33, 25, 14], [30, 10, 0, 20]] {
            for route in routes {
                let token = route / 10;
                for row in 0..8 {
                    for payload in [&gate, &up] {
                        let wire = &payload[row * bytes..(row + 1) * bytes];
                        let original = packed_dot(dtype, wire, &packed[token]);
                        let actual = prepared_dot(dtype, wire, &packed[token]);
                        assert_eq!(actual.to_bits(), original.to_bits());
                        let expected = oracle::project(dtype, wire, &planes[token]).unwrap();
                        assert!((actual - expected).abs() <= 1e-4 + 3e-4 * expected.abs());
                    }
                }
            }
        }
    }
}

#[test]
fn four_gu_prepare_resets_every_full_owner_word_once_and_checks_new_header_tail() {
    let mut writes = vec![0u8; SCRATCH_WORDS];
    for token in 0..4 {
        writes[token] += 1;
        for i in 0..1440 {
            writes[64 + token * 1440 + i] += 1;
        }
    }
    for item in &mut writes[4..64] {
        *item += 1;
    }
    for item in &mut writes[5824..] {
        *item += 1;
    }
    assert!(writes.into_iter().all(|n| n == 1));
    let source = four::Kernel::Prepare.source();
    assert!(source.contains("group.y>=4u") && source.contains("local>=4u"));
    assert!(source.find("p.h!=4u").unwrap() < source.find("planes[token]=0u").unwrap());
    let audit = four::Kernel::Audit.source();
    assert!(audit.contains("p.count!=4u") && audit.contains("at<5824u"));
}
