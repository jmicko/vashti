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
        let module = n::front::wgsl::parse_str(&compiled.source).unwrap();
        assert_eq!(
            compiled.cube_dim,
            CubeDim::new_1d(module.entry_points[0].workgroup_size[0])
        );
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
fn compact_indirect_views_accept_full_allocator_handles_and_reject_trimmed_roots() {
    use cubecl::server::Handle;
    let stream = cubecl::stream_id::StreamId::current();
    use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
    use cubecl_runtime::server::{
        MemoryLayoutDescriptor, MemoryLayoutPolicy, MemoryLayoutStrategy,
    };
    let (_, layouts) = ContiguousMemoryLayoutPolicy::new(32768).apply(
        stream,
        &[MemoryLayoutDescriptor::new(
            MemoryLayoutStrategy::Contiguous,
            [four::MAIN_BYTES as usize].into(),
            1,
        )],
    );
    let allocated = layouts[0].memory.clone();
    assert_eq!(allocated.offset_start, Some(0));
    assert_eq!(allocated.offset_end, Some(0));
    let root = Handle::new(stream, four::MAIN_BYTES);
    // Matches ContiguousMemoryLayoutPolicy::offset_handles for one full owner,
    // including mixed representations equivalent to the same physical span.
    for owner in [
        allocated,
        root.clone(),
        root.clone().offset_start(0),
        root.clone().offset_end(0),
        root.clone().offset_start(0).offset_end(0),
    ] {
        for shard in 0..four::FIRST.len() {
            for down in [false, true] {
                let binding = four::record_dispatch_view(&owner, shard, down).unwrap();
                assert_eq!(binding.size_in_used(), 12);
                assert_eq!(binding.size(), four::MAIN_BYTES);
                assert_eq!(binding.stream, stream);
                assert_eq!(
                    binding.offset_start,
                    Some(24576 + 192 * shard as u64 + if down { 20 } else { 8 })
                );
            }
        }
        for shard in [5, 6, usize::MAX] {
            assert!(four::record_dispatch_view(&owner, shard, false).is_err());
            assert!(four::record_dispatch_view(&owner, shard, true).is_err());
        }
    }
    // Oversized backing memory with a logical 64KiB slice is not a full owner.
    for owner in [
        root.clone().offset_start(4),
        root.clone().offset_end(4),
        root.clone().offset_start(u64::MAX),
        root.offset_end(u64::MAX),
        Handle::new(stream, four::MAIN_BYTES + 4).offset_start(4),
        Handle::new(stream, four::MAIN_BYTES + 4).offset_end(4),
    ] {
        assert!(four::record_dispatch_view(&owner, 0, false).is_err());
        assert!(four::record_dispatch_view(&owner, 4, true).is_err());
    }
}

#[test]
fn compact_prepare_lowers_with_builder_rules_and_zero_group_shards() {
    let module = lower(four::Kernel::PrepareCompact);
    assert_eq!(shared_bytes(&module), MAX_PREPARE_SHARED_BYTES);
    assert_eq!(barriers(&module.entry_points[0].function.body), 3);
    let source = four::Kernel::PrepareCompact.source();
    assert!(
        source.find("storageBarrier();").unwrap()
            < source.find("if token==0u && local<5u").unwrap()
    );
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

#[test]
fn first_plane_compact_gu_lowers_all_formats_with_only_residual_math_removed() {
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
        let original = four::Kernel::ProjectCompact { dtype, row_bytes };
        let selected = four::Kernel::ProjectCompactFirstPlane { dtype, row_bytes };
        let old = original.source();
        let source = selected.source();
        let module = lower(selected);
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert_eq!(barriers(&module.entry_points[0].function.body), 4);
        assert_eq!(
            shared_bytes(&module),
            integer::original_grid_words(dtype).len() as u32 * 4 + 2048
        );
        assert_ne!(selected.id(), original.id());
        assert_ne!(selected.name(), original.name());
        assert_eq!(old.matches("dot4I8Packed(").count(), 16);
        assert_eq!(source.matches("dot4I8Packed(").count(), 8);
        // The complete emitted shader can differ only at the two arithmetic
        // substitutions. All record, source, route, output and barrier guards
        // remain exactly the accepted compact implementation.
        let mut only_primary = old;
        for axis in ["x", "y", "z", "w"] {
            only_primary = only_primary
                .replace(&format!("let second=planes[frames.{axis}+640u+at];"), "")
                .replace(&format!("gs.{axis}+=dot4I8Packed(gq,second);"), "")
                .replace(&format!("us.{axis}+=dot4I8Packed(uq,second);"), "")
                .replace(
                    &format!("let b=bitcast<f32>(planes[frames.{axis}+1360u+block]);"),
                    "",
                )
                .replace(&format!("+b*f32(gs.{axis})"), "")
                .replace(&format!("+b*f32(us.{axis})"), "");
        }
        assert_eq!(source, only_primary);
    }
}

// Scalar model of the selected packed instruction order, using the separately
// C-fixtured register decoder. The independent expectation below instead uses
// full dequantized weights, direct per32 quantization and an F64 scalar sum.
fn primary_prepared_dot(dtype: TensorType, bytes: &[u8], packed: &[u32]) -> f32 {
    let mut wire = Wire { bytes, loads: 0 };
    let mut lanes = [0f32; 8];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        for block in (lane..80).step_by(8) {
            let encoded = wire.prepare(dtype, 0, block);
            for half in 0..if dtype == TensorType::Iq2S { 2 } else { 1 } {
                let quartets = if dtype == TensorType::Iq2S { 4 } else { 8 };
                let mut dot = 0i32;
                for quartet in half * quartets..(half + 1) * quartets {
                    let weights = encoded.packed(dtype, quartet).to_le_bytes();
                    let values = packed[block * 8 + quartet].to_le_bytes();
                    for (&weight, &value) in weights.iter().zip(&values) {
                        dot += i32::from(weight as i8) * i32::from(value as i8);
                    }
                }
                let primary = f32::from_bits(packed[1280 + block]) * dot as f32;
                *sum += encoded.factor(dtype, half) * primary;
            }
        }
    }
    for stride in [4, 2, 1] {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
    }
    lanes[0]
}

#[test]
fn first_plane_compact_gu_matches_independent_primary_f64_and_ignores_residual_values() {
    let inputs: Vec<_> = (0..4)
        .map(|token| {
            (0..2560)
                .map(|i| ((i * (17 + token * 7)) % 311) as f32 / 193. - 0.7)
                .collect::<Vec<_>>()
        })
        .collect();
    let packed: Vec<_> = inputs
        .iter()
        .map(|input| oracle::Planes::new(input).unwrap().packed_words().unwrap())
        .collect();
    let mut differs_from_two_planes = false;
    for dtype in [
        TensorType::Iq2S,
        TensorType::Iq3Xxs,
        TensorType::Iq3S,
        TensorType::Iq4Xs,
    ] {
        let row_bytes = dtype.row_bytes(2560).unwrap() as usize;
        for phase in [1, 3] {
            let bank = iq_fixtures::packed(dtype.gguf_id(), 8, 2560, phase);
            for route in [37usize, 3, 15, 27] {
                let token = route / 10;
                for row in bank.chunks_exact(row_bytes) {
                    let actual = primary_prepared_dot(dtype, row, &packed[token]);
                    let mut weights = vec![0f32; 2560];
                    dtype.decode(row, &mut weights).unwrap();
                    let mut expected = 0f64;
                    for (input, weights) in
                        inputs[token].chunks_exact(32).zip(weights.chunks_exact(32))
                    {
                        let scale = input.iter().map(|x| x.abs()).fold(0f32, f32::max) / 127.;
                        for (&value, &weight) in input.iter().zip(weights) {
                            let q = if scale == 0. {
                                0.
                            } else {
                                (value / scale).round_ties_even().clamp(-127., 127.)
                            };
                            expected += f64::from(weight) * f64::from(scale) * f64::from(q);
                        }
                    }
                    // Existing leaf rounding allowance against the declared
                    // quantized-input oracle; this is NOT the model CPU13 gate.
                    assert!((f64::from(actual) - expected).abs() <= 1e-4 + 3e-4 * expected.abs());
                    let mut altered = packed[token].to_vec();
                    altered[640..1280].fill(0x8080_8080);
                    altered[1360..1440].fill(123f32.to_bits());
                    assert_eq!(
                        actual.to_bits(),
                        primary_prepared_dot(dtype, row, &altered).to_bits()
                    );
                    differs_from_two_planes |=
                        actual.to_bits() != prepared_dot(dtype, row, &packed[token]).to_bits();
                }
            }
        }
    }
    assert!(
        differs_from_two_planes,
        "the candidate must actually omit residual arithmetic"
    );
}

#[test]
fn compact_shuffle_gu_lowers_with_original_accumulation_and_exact_tree() {
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
        let old = four::Kernel::ProjectCompact { dtype, row_bytes };
        let candidate = four::Kernel::ProjectCompactShuffle { dtype, row_bytes };
        let source = candidate.source();
        let original = old.source();
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
        assert_eq!(barriers(&module.entry_points[0].function.body), 1);
        assert_eq!(
            shared_bytes(&module),
            integer::original_grid_words(dtype).len() as u32 * 4
        );
        assert!(!source.contains("gate_sums") && !source.contains("up_sums"));
        assert!(!source.contains("subgroupAdd"));
        assert!(source.contains("(width!=16u && width!=32u)"));
        assert!(source.contains("@builtin(num_subgroups) subgroups:u32"));
        assert!(source.contains("subgroups!=64u/width"));
        assert!(source.contains("sg*(width/8u)+sub_lane/8u;let lane=sub_lane%8u"));
        assert!(!source.contains("local/8u"));
        assert!(source.contains("subgroupShuffleDown(gates,stride)"));
        assert!(source.contains("subgroupShuffleDown(ups,stride)"));
        assert!(source.contains("if lane<stride {gates+=other_gates;ups+=other_ups;}"));
        assert!(
            source.find("subgroupShuffleDown").unwrap() < source.find("if lane<stride").unwrap()
        );
        // Preserve original per-lane decoder, packed dots, both activation
        // planes and scaling before changing only the reduction transport.
        let begin = "    var gates=vec4<f32>(0.0);var ups=vec4<f32>(0.0);";
        let original_dot = original
            .split_once(begin)
            .unwrap()
            .1
            .split_once("    for(var j=0u;j<4u;j++)")
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
            "planes[base]!=1u",
            "if group.y>=shard_count {return;}",
            "if valid_group(routed_group)",
            "ordinal+=4u",
            "status[p.f+route]=1u",
            "silu(gates.w)*ups.w",
        ] {
            assert!(source.contains(check), "missing {check}");
        }
    }
}
