// CPU-only source/oracle acceptance for the frozen-down metadata candidate.
#[test]
fn shared_down_scales_preserves_exact_geometry_and_source_charges() {
    for layer in 0..48 {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
        let projections = [
            Projection::new(gate, 2560, 640, 512).unwrap(),
            Projection::new(up, 2560, 640, 512).unwrap(),
            Projection::new(down, 640, 2560, 512).unwrap(),
        ];
        let ordinary = Geometry::new(projections, 8, 144).unwrap();
        assert!(!ordinary.shared_down_scales());
        assert!(ordinary.with_shared_down_scales().is_err());
        let frozen = ordinary.with_frozen_host_sources().unwrap();
        assert!(frozen.with_shared_down_scales().is_err());
        let old = frozen.with_inactive_partition_exit().unwrap();
        let new = old.with_shared_down_scales().unwrap();
        assert!(new.shared_down_scales());
        assert!(new.with_shared_down_scales().is_err());
        assert!(new.with_live_member_reduction().is_err());
        assert_eq!(old.projections(), new.projections());
        assert_eq!(old.status_words(), new.status_words());
        assert_eq!(old.member_capacity(), new.member_capacity());
        for io in [false, true] {
            assert_eq!(old.scratch_bytes(io), new.scratch_bytes(io));
        }
        for groups in [1, 10, 30] {
            for down in [false, true] {
                assert_eq!(
                    format!("{:?}", old.grid(groups, down).unwrap()),
                    format!("{:?}", new.grid(groups, down).unwrap())
                );
            }
        }
        assert_eq!(
            old.with_private_tables()
                .unwrap()
                .with_shared_down_scales()
                .unwrap(),
            new.with_private_tables().unwrap()
        );
    }
}

#[test]
fn shared_down_scales_checked_unchecked_shader_and_q2_identity() {
    use cubecl::server::ExecutionMode;
    use wgpu::naga::{AddressSpace, ArraySize, TypeInner};
    for down in [IQ4_NL, super::super::Q2_0] {
        let projections = [
            Projection::new(super::super::IQ2_S, 2560, 640, 512).unwrap(),
            Projection::new(super::super::IQ2_S, 2560, 640, 512).unwrap(),
            Projection::new(down, 640, 2560, 512).unwrap(),
        ];
        let old = Geometry::new(projections, 8, 144)
            .unwrap()
            .with_frozen_host_sources()
            .unwrap()
            .with_inactive_partition_exit()
            .unwrap();
        let new = old.with_shared_down_scales().unwrap();
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let source = shader(1, new, mode);
            validate_shader(&source, 8, true);
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let scale_owners = module
                .global_variables
                .iter()
                .filter(|(_, v)| {
                    v.space == AddressSpace::WorkGroup
                        && matches!(module.types[v.ty].inner,
                    TypeInner::Array { size: ArraySize::Constant(n), .. } if n.get() == 20)
                })
                .count();
            assert_eq!(scale_owners, usize::from(down == IQ4_NL));
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            wgpu::naga::back::spv::write_vec(
                &module,
                &info,
                &wgpu::naga::back::spv::Options {
                    lang_version: (1, 3),
                    ..Default::default()
                },
                Some(&wgpu::naga::back::spv::PipelineOptions {
                    shader_stage: wgpu::naga::ShaderStage::Compute,
                    entry_point: module.entry_points[0].name.clone(),
                }),
            )
            .unwrap();
            if down == super::super::Q2_0 {
                assert_eq!(
                    canonical_declaration_order(&source),
                    canonical_declaration_order(&shader(1, old, mode))
                );
            }
            for kind in [0, 2, 3] {
                assert_eq!(
                    canonical_declaration_order(&shader(kind, old, mode)),
                    canonical_declaration_order(&shader(kind, new, mode))
                );
            }
        }
    }
}

#[test]
fn shared_down_scales_original_c_fixture_weights_and_64_lane_dot_are_exact() {
    use kuno_text::quant::{IQ4_NL_VALUES, f16_to_f32};
    for phase in 0..8 {
        let packed = super::super::iq_fixtures::packed(IQ4_NL, 3, 640, phase);
        let mut original = vec![0.; 3 * 640];
        kuno_text::TensorType::Iq4Nl
            .decode(&packed, &mut original)
            .unwrap();
        let mut shared = vec![0.; original.len()];
        for row in 0..3 {
            let bytes = &packed[row * 360..][..360];
            let scales: Vec<_> = bytes
                .chunks_exact(18)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            assert_eq!(scales.len(), 20);
            for column in 0..640 {
                let lane = column % 32;
                let quant = bytes[column / 32 * 18 + 2 + lane % 16];
                let code = if lane >= 16 { quant >> 4 } else { quant & 15 };
                shared[row * 640 + column] =
                    scales[column / 32] * f32::from(IQ4_NL_VALUES[code as usize]);
                assert_eq!(
                    shared[row * 640 + column].to_bits(),
                    original[row * 640 + column].to_bits()
                );
            }
            let input: Vec<_> = (0..640)
                .map(|c| ((c * 7 + phase * 3) % 31) as f32 / 32. - 0.5)
                .collect();
            assert_eq!(
                tree_dot(&shared[row * 640..][..640], &input).to_bits(),
                tree_dot(&original[row * 640..][..640], &input).to_bits()
            );
        }
    }
}
