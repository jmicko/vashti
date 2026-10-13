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

#[test]
fn compact_records_lay_out_the_five_original_partitions_inside_the_owner_tail() {
    assert_eq!(four::RECORD_BASE_WORD, 6144);
    assert_eq!(four::RECORD_BASE_WORD + 5 * four::RECORD_WORDS, 6384);
    const { assert!(6384 <= four::SCRATCH_WORDS) };
    assert_eq!(four::RECORD_BYTES, 192);
    assert_eq!(four::FIRST, [0, 72, 144, 288, 432]);
    assert_eq!(four::PARTS, [72, 72, 144, 144, 80]);
    for (shard, (&first, &parts)) in four::FIRST.iter().zip(&four::PARTS).enumerate() {
        assert_eq!(four::shard_index(first, parts).unwrap(), shard);
    }
    for (first, parts) in [
        (0usize, 0usize),
        (1, 72),
        (72, 144),
        (144, 72),
        (288, 145),
        (432, 81),
        (512, 80),
    ] {
        assert!(four::shard_index(first, parts).is_err());
    }
    // Each checked 12-byte dispatch view addresses its record's exact words
    // and never escapes the authenticated full owner.
    let stream = cubecl::stream_id::StreamId::current();
    let scratch = cubecl::server::Handle::new(stream, four::MAIN_BYTES);
    for shard in 0..5u64 {
        let gu = four::record_dispatch_view(&scratch, shard as usize, false).unwrap();
        let down = four::record_dispatch_view(&scratch, shard as usize, true).unwrap();
        assert_eq!(gu.offset_start, Some(24576 + 192 * shard + 8));
        assert_eq!(down.offset_start, Some(24576 + 192 * shard + 20));
        assert_eq!(gu.size_in_used(), four::RECORD_DISPATCH_BYTES);
        assert_eq!(down.size_in_used(), four::RECORD_DISPATCH_BYTES);
        assert_eq!(gu.size, four::MAIN_BYTES);
        assert_eq!(
            down.offset_end,
            Some(four::MAIN_BYTES - 24576 - 192 * shard - 20 - 12)
        );
    }
    for bytes in [24576u64 + 8, 24576 + 192 * 4 + 20, 16384, 65535] {
        let short = cubecl::server::Handle::new(stream, bytes);
        for shard in 0..5usize {
            assert!(four::record_dispatch_view(&short, shard, false).is_err());
            assert!(four::record_dispatch_view(&short, shard, true).is_err());
        }
    }
}

#[test]
fn compact_prepare_lowers_with_builder_rules_and_zero_group_shards() {
    let module = lower(four::Kernel::PrepareCompact);
    assert_eq!(shared_bytes(&module), MAX_PREPARE_SHARED_BYTES);
    assert_eq!(barriers(&module.entry_points[0].function.body), 2);
    let source = four::Kernel::PrepareCompact.source();
    // Only token0 lanes0..4 build records; every predicate precedes access.
    assert!(source.contains("token==0u && local<5u"));
    assert!(source.find("p.h!=4u").unwrap() < source.find("planes[token]=0u").unwrap());
    // The builder derives each disjoint shard from the immutable MAP using the
    // original partition ranges and the caller's exact group bound.
    assert!(source.contains("shard_first(shard)") && source.contains("shard_parts(shard)"));
    // p.e is the capacity bound; the MAP header holds the exact count.
    assert!(source.contains("groups<1u || groups>p.e"));
    for shard in 0..5 {
        assert!(
            source.contains(&format!("6144u+48u*{}u", shard)) || source.contains("6144u+48u*shard")
        );
    }
    assert!(source.contains("6144u+48u*shard+8u+count]=g"));
    assert!(source.contains("gu_y=count;if gu_y>4u {gu_y=4u;}"));
    // A zero-group shard still publishes ready1/count0 with bounded dispatches.
    assert!(source.contains("planes[6144u+48u*shard+0u]=1u"));
    assert!(source.contains("planes[6144u+48u*shard+6u]=count"));
    assert!(source.contains("map[p.d+4u+g]"));
    // Records build only behind the original immutable MAP header gate.
    assert!(source.contains("map[p.d]!=p.h || map[p.d+1u]!=0u || map[p.d+2u]!=p.h"));
    assert!(source.contains("804u<=arrayLength(&map)-p.d"));
}

#[test]
fn compact_gu_projection_keeps_original_arithmetic_and_record_bounded_dispatch() {
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
        let begin = "// FROZEN_GU_PROJECT_BODY_BEGIN\n";
        let end = "// FROZEN_GU_PROJECT_BODY_END";
        let old_body = old.split_once(begin).unwrap().1.split_once(end).unwrap().0;
        // The compact schedule reuses the same single arithmetic body, so the
        // only difference from the count4 original is the ordinal loop.
        let four_source = four::Kernel::Project { dtype, row_bytes }.source();
        let four_body = four_source
            .split_once(begin)
            .unwrap()
            .1
            .split_once(end)
            .unwrap()
            .0;
        assert_eq!(remove_fourth(four_body), old_body);
        let kernel = four::Kernel::ProjectCompact { dtype, row_bytes };
        let source = kernel.source();
        assert!(source.contains(&four_body.replace("group.y", "routed_group")));
        let module = lower(kernel);
        assert_eq!(
            shared_bytes(&module),
            integer::original_grid_words(dtype).len() as u32 * 4 + 2048
        );
        assert_eq!(barriers(&module.entry_points[0].function.body), 4);
        // p.g is the shard; every list access is record-bounded and the ordinal
        // keeps the original codebook reuse with stride-four.
        assert!(source.contains("p.g>4u"));
        assert!(source.contains("let base=6144u+48u*p.g;"));
        assert!(source.contains("planes[base]!=1u"));
        assert!(source.contains("ordinal=group.y;ordinal<shard_count;ordinal+=4u"));
        assert!(source.contains("let routed_group=planes[base+8u+ordinal];"));
        assert!(source.contains("if valid_group(routed_group)"));
        assert!(
            source.rfind("status[p.f+route]=1u").unwrap()
                < source.rfind("workgroupBarrier();").unwrap()
        );
        // Distinct private kernel identity from both original schedules.
        assert_ne!(kernel.id(), Kernel::Project { dtype, row_bytes }.id());
        assert_ne!(
            kernel.id(),
            Kernel::ProjectStride4 { dtype, row_bytes }.id()
        );
    }
}

#[test]
fn compact_audit_reconstructs_validates_and_erases_each_disjoint_record() {
    let module = lower(four::Kernel::AuditCompact);
    // Only the records_valid barrier latch needs workgroup storage.
    assert_eq!(shared_bytes(&module), 4);
    let source = four::Kernel::AuditCompact.source();
    // The word audit keeps every original header/frame/tail check except the
    // record range, which the dedicated workgroup validates and erases.
    assert!(source.contains("at<64u") && source.contains("at<5824u"));
    assert!(source.contains("at<6144u") && source.contains("at<6384u"));
    // Reconstruction follows the builder's exact rule from the unchanged MAP.
    assert!(source.contains("map[p.r0]!=p.count || map[p.r0+1u]!=0u || map[p.r0+2u]!=p.count"));
    assert!(source.contains("groups<1u || groups>p.r1"));
    assert!(
        source.contains("audit_shard_first(shard)") && source.contains("audit_shard_parts(shard)")
    );
    // Headers, counts, both dispatch triplets, list identity and inactive zeros.
    assert!(source.contains("planes[base+8u+count]!=g"));
    assert!(source.contains("planes[base+3u]!=gu_y"));
    assert!(source.contains("planes[base+6u]!=count"));
    assert!(source.contains("i=count;i<40u"));
    // A failed validation stickies the fault and keeps the records.
    assert!(source.contains("atomicOr(&fault[0],8u);"));
    assert!(source.contains("atomicStore(&records_valid,0u);"));
    // Erasure happens only after every shard record validated.
    assert!(
        source.find("atomicLoad(&records_valid)==1u").unwrap()
            > source.find("workgroupBarrier();").unwrap()
    );
    assert!(source.contains("planes[6144u+48u*shard+i]=0u;"));
    // A builderless (invalid MAP) round must leave all-zero records.
    assert!(source.contains("for(var i=0u;i<48u;i+=1u)"));
}
