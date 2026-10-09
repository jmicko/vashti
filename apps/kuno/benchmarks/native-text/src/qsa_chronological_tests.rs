use super::*;
use crate::private_arena::BufferLayout;
use cubecl::stream_id::StreamId;

#[test]
fn exact_token_budget_boundary_and_all_bounded_chunk_counts() {
    assert_eq!(MAX_VISIBLE, crate::qsa_gpu::BUDGET);
    assert_eq!(SELECT_STRIDE, crate::qsa_gpu::BUDGET + 3);
    assert_eq!(EXTRA_INFO_BYTES_PER_CHUNK, 49_152);
    assert_eq!(EXTRA_HOST_BYTES_PER_CHUNK, 4_096);
    assert_eq!(OWNED_WORDS.map(|w| w * 4), [32768, 32768, 131072, 32768]);
    for count in 1..=8 {
        for visible in [count, count.max(3), count.max(4), 2048] {
            let g = Geometry {
                start: visible - count,
                count,
                blocks: 1024,
            };
            assert!(g.supported().unwrap());
            for row in 0..count {
                let at = g.start + row + 1;
                let scores: Vec<_> = (0..at / 4)
                    .map(|i| match i % 4 {
                        0 => -0.0,
                        1 => 0.0,
                        2 => f32::MAX,
                        _ => (i % 17) as f32,
                    })
                    .collect();
                let ids = kuno_text::ops::qsa_select_blocks(&scores, at, 4, 2048).unwrap();
                assert_eq!(ids, (0..at).collect::<Vec<_>>());
            }
        }
        assert!(
            !Geometry {
                start: 2049 - count,
                count,
                blocks: 1024
            }
            .supported()
            .unwrap()
        );
    }
    for g in [
        Geometry {
            start: 0,
            count: 0,
            blocks: 1,
        },
        Geometry {
            start: 0,
            count: 9,
            blocks: 3,
        },
        Geometry {
            start: 4095,
            count: 2,
            blocks: 1024,
        },
        Geometry {
            start: usize::MAX,
            count: 1,
            blocks: 1024,
        },
        Geometry {
            start: 3,
            count: 1,
            blocks: 0,
        },
        Geometry {
            start: 2047,
            count: 1,
            blocks: 511,
        },
    ] {
        assert!(g.supported().is_err());
    }
    // Above the conservative boundary, original score-dependent pruning is
    // preserved. This fixture actually drops a complete block at2052.
    let mut scores = vec![1.0; 513];
    scores[0] = 0.0;
    let ids = kuno_text::ops::qsa_select_blocks(&scores, 2052, 4, 2048).unwrap();
    assert_eq!(ids.len(), 2048);
    assert_eq!(ids[0], 4);
    assert_eq!(ids.last(), Some(&2051));
}

#[test]
fn exact_full_handles_reject_geometry_partial_pages_and_foreign_streams() {
    let g = Geometry {
        start: 2040,
        count: 8,
        blocks: 1024,
    };
    let owners = || OWNED_WORDS.map(|w| Handle::new(StreamId { value: 73 }, w as u64 * 4));
    validate_handles(&owners(), g).unwrap();
    for index in 0..4 {
        for n in [
            0,
            OWNED_WORDS[index] as u64 * 4 - 4,
            OWNED_WORDS[index] as u64 * 4 + 4,
        ] {
            let mut h = owners();
            h[index] = Handle::new(StreamId { value: 73 }, n);
            assert!(validate_handles(&h, g).is_err());
        }
        let mut h = owners();
        h[index].stream = StreamId { value: 74 };
        assert!(validate_handles(&h, g).is_err());
        let mut h = owners();
        h[index] = h[index].clone().offset_start(4);
        assert!(validate_handles(&h, g).is_err());
        let mut h = owners();
        h[index] = h[index].clone().offset_end(4);
        assert!(validate_handles(&h, g).is_err());
    }
    assert!(
        validate_handles(
            &owners(),
            Geometry {
                start: 2048,
                count: 1,
                blocks: 1024
            }
        )
        .is_err()
    );
    for (words, bytes) in OWNED_WORDS.into_iter().zip([32768, 32768, 131072, 32768]) {
        assert_eq!(BufferLayout::new(words).unwrap().owned_bytes, bytes);
    }
}

#[test]
fn both_kernel_sources_validate_dense_bindings_fault_domination_and_spirv() {
    for (source, bindings, output_binding) in [
        (include_str!("qsa_chronological_preflight.wgsl"), 5, None),
        (include_str!("qsa_chronological_publish.wgsl"), 3, Some(0)),
    ] {
        // Both CubeTask modes intentionally share the same fully explicit
        // guards; raw WGSL does not inject unchecked index/length assumptions.
        for _mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let m = wgpu::naga::front::wgsl::parse_str(source).unwrap();
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&m)
            .unwrap();
            assert_eq!(m.entry_points[0].workgroup_size, [256, 1, 1]);
            let mut seen = Vec::new();
            for (handle, var) in m.global_variables.iter() {
                if let Some(binding) = &var.binding {
                    assert!(!info.get_entry_point(0)[handle].is_empty());
                    assert_eq!(binding.group, 0);
                    seen.push(binding.binding);
                    if let wgpu::naga::AddressSpace::Storage { access } = var.space {
                        let fault_binding = if bindings == 5 { 3 } else { 1 };
                        assert_eq!(
                            access.contains(wgpu::naga::StorageAccess::STORE),
                            Some(binding.binding) == output_binding
                                || binding.binding == fault_binding
                        );
                    }
                }
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..bindings).collect::<Vec<_>>());
            let spv = wgpu::naga::back::spv::write_vec(
                &m,
                &info,
                &wgpu::naga::back::spv::Options {
                    lang_version: (1, 3),
                    ..Default::default()
                },
                Some(&wgpu::naga::back::spv::PipelineOptions {
                    shader_stage: wgpu::naga::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
            assert!(!spv.is_empty());
        }
    }
    let pre = include_str!("qsa_chronological_preflight.wgsl");
    assert!(pre.contains("token<p.count && block<(p.start+token+1u)/4u"));
    assert!(pre.contains("status[i]!=1u"));
    assert!(pre.contains("selected[i]!=0u"));
    assert!(pre.contains("i>0u && atomicLoad(&fault[i])!=0u"));
    assert!(!pre.contains("selected[i]="));
    let publish = include_str!("qsa_chronological_publish.wgsl");
    assert!(
        publish.find("if atomicLoad(&fault[0])!=0u").unwrap()
            < publish.find("selected[i]=column").unwrap()
    );
    assert!(publish.contains("token<p.count && column<p.start+token+1u"));
}

#[test]
fn actual_device_admission_covers_every_used_resource_and_dispatch_extent() {
    let valid = wgpu::Limits::default();
    validate_limits(&valid).unwrap();
    for kind in 0..9 {
        let mut wrong = valid.clone();
        match kind {
            0 => wrong.max_compute_invocations_per_workgroup = 255,
            1 => wrong.max_compute_workgroup_size_x = 255,
            2 => wrong.max_storage_buffers_per_shader_stage = 3,
            3 => wrong.max_bindings_per_bind_group = 4,
            4 => wrong.max_bind_groups = 0,
            5 => wrong.max_uniform_buffers_per_shader_stage = 0,
            6 => wrong.max_uniform_buffer_binding_size = 15,
            7 => wrong.max_compute_workgroups_per_dimension = 127,
            _ => wrong.max_storage_buffer_binding_size = 131071,
        }
        assert!(validate_limits(&wrong).is_err());
    }
    let mut wrong = valid;
    wrong.max_buffer_size = 131071;
    assert!(validate_limits(&wrong).is_err());
}
