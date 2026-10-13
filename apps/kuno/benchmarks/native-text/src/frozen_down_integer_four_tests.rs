//! Original DOWN decoder/oracle helpers are reused without changing old tests.
use super::*;
use crate::quant_matvec::frozen_down_integer::four;

#[test]
fn compact_shuffle_down_lowers_without_shared_reduction_or_changed_dots() {
    for dtype in [crate::quant_matvec::IQ4_NL, crate::quant_matvec::Q2_0] {
        let row_bytes = TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(640)
            .unwrap() as usize;
        let old = four::Kernel::ProjectCompact { dtype, row_bytes };
        let candidate = four::Kernel::ProjectCompactShuffle { dtype, row_bytes };
        let original = old.source();
        let source = candidate.source();
        let module = lower(candidate);
        let compiled = candidate
            .compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                candidate.address_type(),
            )
            .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert_eq!(compiled.cube_dim, CubeDim::new_1d(64));
        assert_ne!(candidate.id(), old.id());
        assert_eq!(shared_bytes(&module), 0);
        assert_eq!(barriers(&module.entry_points[0].function.body), 0);
        assert!(!source.contains("workgroupBarrier") && !source.contains("sums["));
        assert!(!source.contains("subgroupAdd"));
        assert!(source.contains("(width!=16u && width!=32u)"));
        assert!(source.contains("sg*(width/8u)+sub_lane/8u;let lane=sub_lane%8u"));
        assert!(!source.contains("local/8u"));
        assert!(source.contains("subgroupShuffleDown(values,stride)"));
        assert!(source.contains("if lane<stride {values+=other_values;}"));
        assert!(
            source.find("subgroupShuffleDown").unwrap() < source.find("if lane<stride").unwrap()
        );
        let begin = "    var values=vec4<f32>(0.0);";
        let original_dot = original
            .split_once(begin)
            .unwrap()
            .1
            .split_once("    // Explicit initialization")
            .unwrap()
            .0;
        let candidate_dot = source
            .split_once(begin)
            .unwrap()
            .1
            .split_once("    // Exact original eight-lane")
            .unwrap()
            .0;
        assert_eq!(original_dot, candidate_dot);
        for check in [
            "p.h!=4u",
            "gu_planes[base]!=1u",
            "if group.y>=shard_count {return;}",
            "if !valid_group(routed_group) {return;}",
            "status[p.f+80u+route]=1u",
            "output[p.d+route*p.e+row]=values.w",
        ] {
            assert!(source.contains(check), "missing {check}");
        }
    }
}

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
    assert!(
        !n::back::spv::write_vec(
            &module,
            &info,
            &n::back::spv::Options {
                lang_version: (1, 3),
                zero_initialize_workgroup_memory:
                    n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                ..Default::default()
            },
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into()
            })
        )
        .unwrap()
        .is_empty()
    );
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
        .replace("vec4<u32>", "vec3<u32>")
        .replace("j<4u", "j<3u")
        .replace("*4u+j", "*3u+j")
        .replace("local*4u+", "local*3u+")
        .replace("every unused RHS column", "both unused RHS columns")
}

#[test]
fn four_down_actual_shader_none_keeps_original_xyz_producer_and256cells() {
    let prepare = lower(four::Kernel::Prepare);
    assert_eq!(shared_bytes(&prepare), MAX_PREPARE_SHARED_BYTES);
    assert_eq!(barriers(&prepare.entry_points[0].function.body), 2);
    assert_eq!(four::LOGICAL_WORDS, 14464);
    assert_eq!(four::MAIN_BYTES, MAIN_BYTES);
    assert_eq!(four::RUNTIME_INCREMENT, RUNTIME_INCREMENT);
    for dtype in [crate::quant_matvec::IQ4_NL, crate::quant_matvec::Q2_0] {
        let row_bytes = TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(640)
            .unwrap() as usize;
        let original = Kernel::Project { dtype, row_bytes }.source();
        let kernel = four::Kernel::Project { dtype, row_bytes };
        let source = kernel.source();
        let module = lower(kernel);
        assert_eq!(shared_bytes(&module), 1024);
        assert_eq!(barriers(&module.entry_points[0].function.body), 2);
        assert!(
            source.contains("values=vec4<f32>") && source.contains("sums[local*4u+j]=values[j]")
        );
        assert!(source.contains("p.h!=4u") && source.contains("p.g>40u"));
        let start = "    let row=group.x*8u+local/8u;";
        assert_eq!(
            remove_fourth(source.split_once(start).unwrap().1),
            original.split_once(start).unwrap().1
        );
        assert!(
            source.find("!valid_group(group.y)").unwrap()
                < source.find("down_prepare(offset,block)").unwrap()
        );
        assert_ne!(kernel.id(), Kernel::Project { dtype, row_bytes }.id());
    }
    let audit = lower(four::Kernel::Audit);
    assert_eq!(shared_bytes(&audit), 0);
    assert!(four::Kernel::Audit.source().contains("at<14464u"));
    assert!(!four::Kernel::Audit.source().contains("atomicStore"));
}

#[test]
fn four_down_exact_roots_count_and40route_layout_fit_unchanged_caps() {
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
    for size in [14464, 16383, 16385] {
        assert!(four::validate_layout(selected, &spans, words, size, 4, 40).is_err());
    }
    for count in 1..=3 {
        validate_layout(selected, &spans, words, SCRATCH_WORDS, count, 30).unwrap();
    }
}

#[test]
fn four_down_all40_real_routes_match_original_candidate_and_independent_oracle() {
    for dtype in [TensorType::Iq4Nl, TensorType::Q2_0] {
        let row_bytes = dtype.row_bytes(640).unwrap() as usize;
        let bytes = iq_fixtures::packed(dtype.gguf_id(), 8, 640, 1);
        for route in 0..40 {
            let input: Vec<f32> = (0..640)
                .map(|i| ((i * 31 + route * 17) % 139) as f32 * 0.012 - 0.83)
                .collect();
            let planes = oracle::Planes::new(&input).unwrap();
            let words = planes.packed_words().unwrap();
            for row in 0..8 {
                let wire = &bytes[row * row_bytes..(row + 1) * row_bytes];
                let expected = oracle::project(dtype, wire, &planes).unwrap();
                let actual = projected(dtype, wire, &words);
                let mut magnitude = 0f64;
                for block in 0..20 {
                    let (codes, d) = oracle::unpack32(dtype, wire, 640, block).unwrap();
                    for (i, code) in codes.into_iter().enumerate() {
                        let at = block * 8 + i / 4;
                        let shift = i % 4 * 8;
                        let a = (words[at] >> shift) as u8 as i8;
                        let b = (words[160 + at] >> shift) as u8 as i8;
                        let x = f64::from(f32::from_bits(words[320 + block])) * f64::from(a)
                            + f64::from(f32::from_bits(words[340 + block])) * f64::from(b);
                        magnitude += (f64::from(d) * f64::from(code) * x).abs();
                    }
                }
                // Exactly the existing candidate oracle bound; original CPU13
                // tolerance and greedy gates are untouched and still required.
                assert!(
                    f64::from((actual - expected).abs())
                        <= 32. * f64::from(f32::EPSILON) * magnitude + 1e-6
                );
            }
        }
    }
}

#[test]
fn four_down_full64k_reset_and_audit_cover_route39_and_every_physical_tail_word() {
    let mut seen = vec![0u8; SCRATCH_WORDS];
    for route in 0..40 {
        seen[route] += 1;
        for word in 0..360 {
            seen[64 + route * 360 + word] += 1;
        }
    }
    for item in &mut seen[40..64] {
        *item += 1;
    }
    for item in &mut seen[14464..] {
        *item += 1;
    }
    assert!(seen.into_iter().all(|n| n == 1));
    let source = four::Kernel::Prepare.source();
    assert!(source.contains("group.y>=40u") && source.contains("local>=40u"));
    assert!(source.find("p.h!=4u").unwrap() < source.find("planes[route]=0u").unwrap());
    let valid = |words: &[u32]| {
        words.iter().enumerate().all(|(at, &word)| {
            if at < 64 {
                return word == u32::from(at < 40);
            }
            if at >= 14464 {
                return word == 0;
            }
            let offset = (at - 64) % 360;
            offset < 320 || {
                let scale = f32::from_bits(word);
                scale.is_finite() && scale >= 0.
            }
        })
    };
    let mut words = vec![0u32; SCRATCH_WORDS];
    words[..40].fill(1);
    assert!(valid(&words));
    for index in [39usize, 40, 63, 64 + 39 * 360 + 320, 14463, 14464, 16383] {
        let old = words[index];
        words[index] = if (64..14464).contains(&index) {
            f32::NAN.to_bits()
        } else {
            old ^ 1
        };
        assert!(!valid(&words));
        words[index] = old;
    }
    assert!(four::Kernel::Audit.source().contains("p.count!=4u"));
}

#[test]
fn four_down_compact_projection_keeps_original_arithmetic_and_record_bounded_dispatch() {
    for dtype in [crate::quant_matvec::IQ4_NL, crate::quant_matvec::Q2_0] {
        let row_bytes = TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(640)
            .unwrap() as usize;
        let kernel = four::Kernel::ProjectCompact { dtype, row_bytes };
        let source = kernel.source();
        let module = lower(kernel);
        assert_eq!(shared_bytes(&module), 1024);
        assert_eq!(barriers(&module.entry_points[0].function.body), 2);
        // The compact schedule reuses the original decoder body verbatim; only
        // the group list comes from this shard's authenticated record.
        let start = "    let row=group.x*8u+local/8u;";
        let four_source = four::Kernel::Project { dtype, row_bytes }.source();
        assert!(
            source.contains(
                &four_source
                    .split_once(start)
                    .unwrap()
                    .1
                    .replace("group.y", "routed_group")
                    .replace("!valid_group(group.y)", "")
            )
        );
        // p.g is the shard, and every access stays inside the record bounds.
        assert!(source.contains("p.g>4u"));
        assert!(source.contains("let base=6144u+48u*p.g;"));
        assert!(source.contains("planes[base]!=1u"));
        assert!(
            source.contains("ordinal=group.y;ordinal<shard_count;ordinal+=4u")
                || source.contains("let routed_group=gu_planes[base+8u+group.y];")
        );
        assert!(source.contains("let routed_group=gu_planes[base+8u+group.y];"));
        assert!(source.contains("if !valid_group(routed_group) {return;}"));
        assert!(source.contains("@group(0) @binding(5) var<storage,read> gu_planes"));
        assert_ne!(kernel.id(), Kernel::Project { dtype, row_bytes }.id());
        assert_ne!(kernel.id(), four::Kernel::Project { dtype, row_bytes }.id());
    }
}
