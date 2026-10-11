// Offline source acceptance only. No GPU or service acquired.

fn down_stride_four_shader(g: Geometry, mode: cubecl::server::ExecutionMode) -> String {
    use cubecl::compute::KernelBuilder;
    use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
    use cubecl::wgpu::WgslCompiler;
    use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
    let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
    let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
    let mut b = KernelBuilder::default();
    b.scope.register_type::<u32>(uint);
    b.scope.register_type::<usize>(uint);
    b.scope.register_type::<f32>(float);
    b.scope
        .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
    b.scope
        .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
    let down = b.input_array(Type::new(uint)).into();
    let act = b.input_array(Type::new(float)).into();
    let map = b.input_array(Type::new(uint)).into();
    let partial = b.output_array(Type::new(float)).into();
    let status = b.output_array(Type::new(uint)).into();
    let map_first = b.scalar(uint).into();
    let act_first = b.scalar(uint).into();
    let act_stride = b.scalar(uint).into();
    let partial_first = b.scalar(uint).into();
    let partial_stride = b.scalar(uint).into();
    let status_first = b.scalar(uint).into();
    let first_slot = b.scalar(uint).into();
    let slot_count = b.scalar(uint).into();
    let expand = grouped_down_frozen_four_rows_stride_four::expand;
    expand(
        &mut b.scope,
        down,
        act,
        map,
        partial,
        status,
        map_first,
        act_first,
        act_stride,
        partial_first,
        partial_stride,
        status_first,
        first_slot,
        slot_count,
        g.hidden,
        g.ff,
        g.projections[2].dtype,
        g.projections[2].row_bytes,
        g.tile,
    );
    let name = "grouped_down_frozen_four_rows_stride_four";
    WgslCompiler::default()
        .compile(
            b.build(
                KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(64))
                    .kernel_name(name),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            uint,
        )
        .unwrap()
        .to_string()
}

#[test]
fn down_stride_four_actual_checked_unchecked_shader_uses_original_owner_and_group_fences() {
    use cubecl::server::ExecutionMode;
    use wgpu::naga::{AddressSpace, ArraySize, TypeInner};
    for dtype in [IQ4_NL, Q2_0] {
        let g = down_four_row_geometry(dtype).with_four_row_down().unwrap();
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let source = down_stride_four_shader(g, mode);
            validate_shader(&source, 8, true);
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
            let mut bytes = 0;
            for (owner, var) in module.global_variables.iter() {
                if var.space != AddressSpace::WorkGroup {
                    continue;
                }
                let TypeInner::Array {
                    size: ArraySize::Constant(n),
                    stride,
                    ..
                } = module.types[var.ty].inner
                else {
                    panic!("fixed reduction owner")
                };
                assert_eq!(n.get(), 512);
                assert!(var.init.is_none());
                bytes += n.get() * stride;
                assert_shared_loads_follow_barrier(&module, owner);
            }
            assert_eq!(bytes, 2048);
            assert_eq!(factor_spirv_storage_classes(&module, 512), vec![4]);
            // Reset/fold and tree sites remain; only the final per-group fence
            // is added. No workgroup zero initialization is assumed.
            assert_eq!(runtime_spirv_barriers(&module), 3);
            let events = memory_events(&module);
            let exit = events.iter().position(|&event| event == 'r').unwrap();
            let write = events.iter().position(|&event| event == 'w').unwrap();
            let barrier = events.iter().position(|&event| event == 'b').unwrap();
            assert!(exit < write && write < barrier);
        }
    }
}

#[test]
fn down_stride_four_all_original_groups_partitions_and_routes_are_covered_once() {
    let partitions = [
        (0usize, 72usize),
        (72, 72),
        (144, 144),
        (288, 144),
        (432, 80),
    ];
    for count in 1..=3 {
        for groups in 1..=count * 10 {
            let slots: Vec<_> = (0..groups).map(|g| (g * 173 + count * 71) % 512).collect();
            let mut seen = vec![0u8; groups];
            for &(first, length) in &partitions {
                for y in 0..4 {
                    for group in (y..30).step_by(4) {
                        if group < groups && slots[group] >= first && slots[group] < first + length
                        {
                            seen[group] += 1;
                        }
                    }
                }
            }
            assert!(seen.into_iter().all(|n| n == 1));
        }
    }
    for layer in 0..48 {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
        let g = Geometry::new(
            [
                Projection::new(gate, 2560, 640, 512).unwrap(),
                Projection::new(up, 2560, 640, 512).unwrap(),
                Projection::new(down, 640, 2560, 512).unwrap(),
            ],
            8,
            144,
        )
        .unwrap()
        .with_frozen_host_sources()
        .unwrap()
        .with_inactive_partition_exit()
        .unwrap()
        .with_four_row_down()
        .unwrap();
        assert!(matches!(
            g.grid(30, true).unwrap(),
            CubeCount::Static(640, 30, 1)
        ));
        assert_eq!(g.down_shared_memory_bytes(), 2048);
    }
    assert_eq!(5 * 640 * 30, 96_000);
    assert_eq!(5 * 640 * 4, 12_800);
}
