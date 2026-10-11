// Offline acceptance for four physical DOWN rows with the original 64-lane tree.
// Included by grouped_moe::tests; this file does not acquire a device.

fn down_four_row_shader(g: Geometry, mode: cubecl::server::ExecutionMode) -> String {
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
    let expand = if g.shared_down_scales() {
        grouped_down_frozen_four_rows_shared_scales::expand
    } else {
        grouped_down_frozen_four_rows::expand
    };
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
    let name = if g.shared_down_scales() {
        "grouped_down_frozen_four_rows_shared_scales"
    } else {
        "grouped_down_frozen_four_rows"
    };
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

fn down_four_row_geometry(dtype: u32) -> Geometry {
    Geometry::new(
        [
            Projection::new(super::super::IQ2_S, 2560, 640, 512).unwrap(),
            Projection::new(super::super::IQ2_S, 2560, 640, 512).unwrap(),
            Projection::new(dtype, 640, 2560, 512).unwrap(),
        ],
        8,
        144,
    )
    .unwrap()
    .with_frozen_host_sources()
    .unwrap()
    .with_inactive_partition_exit()
    .unwrap()
}

#[test]
fn down_four_rows_all_48_geometry_prerequisites_and_composition() {
    let mut down_types = [0; 2];
    for layer in 0..48 {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
        down_types[usize::from(down == super::super::Q2_0)] += 1;
        let ordinary = Geometry::new(
            [
                Projection::new(gate, 2560, 640, 512).unwrap(),
                Projection::new(up, 2560, 640, 512).unwrap(),
                Projection::new(down, 640, 2560, 512).unwrap(),
            ],
            8,
            144,
        )
        .unwrap();
        assert!(!ordinary.four_row_down());
        assert!(ordinary.with_four_row_down().is_err());
        let frozen = ordinary.with_frozen_host_sources().unwrap();
        assert!(frozen.with_four_row_down().is_err());
        let old = frozen.with_inactive_partition_exit().unwrap();
        let new = old.with_four_row_down().unwrap();
        assert!(new.four_row_down());
        assert!(new.with_four_row_down().is_err());
        assert!(new.with_live_member_reduction().is_err());
        assert_eq!(old.projections(), new.projections());
        assert_eq!(old.status_words(), new.status_words());
        assert_eq!(old.member_capacity(), 8);
        assert_eq!(old.member_capacity(), new.member_capacity());
        assert_eq!(old.tile(), new.tile());
        assert_eq!(old.down_shared_memory_bytes(), 2048);
        assert_eq!(new.down_shared_memory_bytes(), 2048);
        assert_eq!(
            old.with_shared_down_scales()
                .unwrap()
                .down_shared_memory_bytes(),
            if down == IQ4_NL { 2128 } else { 2048 }
        );
        assert_eq!(
            new.with_shared_down_scales()
                .unwrap()
                .down_shared_memory_bytes(),
            if down == IQ4_NL { 2368 } else { 2048 }
        );
        for io in [false, true] {
            assert_eq!(old.scratch_bytes(io), new.scratch_bytes(io));
        }
        for groups in [1, 10, 30] {
            assert_eq!(
                format!("{:?}", old.grid(groups, false).unwrap()),
                format!("{:?}", new.grid(groups, false).unwrap())
            );
            assert!(
                matches!(old.grid(groups, true).unwrap(), CubeCount::Static(2560, y, 1) if y == groups as u32)
            );
            assert!(
                matches!(new.grid(groups, true).unwrap(), CubeCount::Static(640, y, 1) if y == groups as u32)
            );
        }
        assert_eq!(
            old.with_shared_down_scales()
                .unwrap()
                .with_four_row_down()
                .unwrap(),
            new.with_shared_down_scales().unwrap()
        );
        assert_eq!(
            old.with_private_tables()
                .unwrap()
                .with_four_row_down()
                .unwrap(),
            new.with_private_tables().unwrap()
        );
        assert_eq!(
            old.with_workgroup_tables()
                .unwrap()
                .with_four_row_down()
                .unwrap(),
            new.with_workgroup_tables().unwrap()
        );
        assert_eq!(
            old.with_shared_gate_up_factors()
                .unwrap()
                .with_four_row_down()
                .unwrap(),
            new.with_shared_gate_up_factors().unwrap()
        );
    }
    assert_eq!(down_types, [39, 9]);
    assert_eq!(5 * 30 * 2560, 384_000);
    assert_eq!(5 * 30 * 640, 96_000);
}

fn down_four_row_value(row: &[u8], column: usize, dtype: u32, scales: Option<&[f32]>) -> f32 {
    use kuno_text::quant::{IQ4_NL_VALUES, f16_to_f32};
    if dtype == IQ4_NL {
        let block = column / 32;
        let b = &row[block * 18..][..18];
        let d = scales.map_or_else(
            || f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
            |s| s[block],
        );
        let lane = column % 32;
        let packed = b[2 + lane % 16];
        let code = if lane < 16 { packed & 15 } else { packed >> 4 };
        d * f32::from(IQ4_NL_VALUES[code as usize])
    } else {
        assert_eq!(dtype, super::super::Q2_0);
        let b = &row[column / 64 * 18..][..18];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let lane = column % 64;
        let code = i32::from((b[2 + lane / 4] >> (lane % 4 * 2)) & 3) - 1;
        d * code as f32
    }
}

#[test]
fn down_four_rows_original_c_weights_and_64_lane_tree_are_exact() {
    use kuno_text::quant::f16_to_f32;
    for dtype in [IQ4_NL, super::super::Q2_0] {
        let kind = kuno_text::TensorType::from_gguf_id(dtype).unwrap();
        let row_bytes = 640 / kind.block_elements() as usize * kind.block_bytes() as usize;
        for phase in [0, 1, 3, 7] {
            let packed = super::super::iq_fixtures::packed(dtype, 2560, 640, phase);
            let expected = super::super::iq_fixtures::oracle(dtype, 2560, 640, phase);
            let mut original = vec![0.; expected.len()];
            kind.decode(&packed, &mut original).unwrap();
            assert_eq!(
                original.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let inputs: Vec<Vec<f32>> = (0..3)
                .map(|member| {
                    (0..640)
                        .map(|c| ((c * 7 + phase * 3 + member * 5) % 31) as f32 / 32. - 0.5)
                        .collect()
                })
                .collect();
            for feature_block in 0..640 {
                let mut scales = [0.; 80];
                let mut scale_writes = [0; 80];
                if dtype == IQ4_NL {
                    for lane in 0..64 {
                        for index in [lane, lane + 64] {
                            if index < 80 {
                                let row = feature_block * 4 + index / 20;
                                let byte = row * row_bytes + index % 20 * 18;
                                scales[index] = f16_to_f32(u16::from_le_bytes([
                                    packed[byte],
                                    packed[byte + 1],
                                ]));
                                scale_writes[index] += 1;
                            }
                        }
                    }
                    assert!(scale_writes.iter().all(|&n| n == 1));
                }
                let mut sums = [0.; 512];
                let mut writes = [0; 512];
                for lane in 0..64 {
                    let row = lane / 16;
                    let sub_lane = lane % 16;
                    let feature = feature_block * 4 + row;
                    let bytes = &packed[feature * row_bytes..][..row_bytes];
                    let mut partial = [[0.; 8]; 4];
                    for (virtual_row, members) in partial.iter_mut().enumerate() {
                        let virtual_lane = sub_lane + virtual_row * 16;
                        for column_tile in 0..10 {
                            let column = column_tile * 64 + virtual_lane;
                            let weight = down_four_row_value(
                                bytes,
                                column,
                                dtype,
                                (dtype == IQ4_NL).then_some(&scales[row * 20..][..20]),
                            );
                            assert_eq!(
                                weight.to_bits(),
                                expected[feature * 640 + column].to_bits()
                            );
                            for member in 0..3 {
                                members[member] += weight * inputs[member][column];
                            }
                        }
                    }
                    for (member, &first) in partial[0].iter().enumerate() {
                        let cell = (row * 8 + member) * 16 + sub_lane;
                        sums[cell] = (first + partial[2][member])
                            + (partial[1][member] + partial[3][member]);
                        writes[cell] += 1;
                    }
                }
                assert!(writes.iter().all(|&n| n == 1));
                for stride in [8, 4, 2, 1] {
                    for lane in 0..64 {
                        let row = lane / 16;
                        let sub_lane = lane % 16;
                        if sub_lane < stride {
                            for member in 0..8 {
                                let cell = (row * 8 + member) * 16 + sub_lane;
                                sums[cell] += sums[cell + stride];
                            }
                        }
                    }
                }
                for row in 0..4 {
                    let feature = feature_block * 4 + row;
                    for member in 0..3 {
                        assert_eq!(
                            sums[(row * 8 + member) * 16].to_bits(),
                            tree_dot(&expected[feature * 640..][..640], &inputs[member]).to_bits(),
                            "dtype{dtype} phase{phase} feature{feature} member{member}"
                        );
                    }
                    for member in 3..8 {
                        assert_eq!(sums[(row * 8 + member) * 16].to_bits(), 0);
                    }
                }
            }
        }
    }
}

// Evaluate the generated initializer for every lane, including the lane+64
// round, instead of trusting only the separately simulated CPU initializer.
fn assert_down_four_row_scale_coverage(module: &wgpu::naga::Module) {
    use wgpu::naga::{
        AddressSpace, ArraySize, BinaryOperator, Expression, Literal, Statement, TypeInner,
    };
    let (owner, _) = module
        .global_variables
        .iter()
        .find(|(_, var)| {
            var.space == AddressSpace::WorkGroup
                && matches!(module.types[var.ty].inner,
            TypeInner::Array { size: ArraySize::Constant(n), .. } if n.get() == 80)
        })
        .unwrap();
    let f = &module.entry_points[0].function;
    fn eval(f: &wgpu::naga::Function, h: wgpu::naga::Handle<Expression>, lane: u32) -> u32 {
        if is_physical_lane_x(f, h) {
            return lane;
        }
        match f.expressions[h] {
            Expression::Literal(Literal::U32(n)) => n,
            Expression::Literal(Literal::Bool(b)) => u32::from(b),
            Expression::As { expr, .. } => eval(f, expr, lane),
            Expression::Binary { op, left, right } => {
                let a = eval(f, left, lane);
                let b = eval(f, right, lane);
                match op {
                    BinaryOperator::Add => a + b,
                    BinaryOperator::Subtract => a - b,
                    BinaryOperator::Multiply => a * b,
                    BinaryOperator::Divide => a / b,
                    BinaryOperator::Modulo => a % b,
                    BinaryOperator::Less => u32::from(a < b),
                    _ => panic!("unexpected scale initializer operator {op:?}"),
                }
            }
            ref other => panic!("unexpected scale initializer expression {other:?}"),
        }
    }
    fn visit(
        f: &wgpu::naga::Function,
        body: &wgpu::naga::Block,
        owner: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
        predicates: &mut Vec<(wgpu::naga::Handle<Expression>, bool)>,
        writes: &mut [u32; 80],
    ) {
        for statement in body {
            match statement {
                Statement::Store { pointer, .. }
                    if shared_pointer_global(f, *pointer) == Some(owner) =>
                {
                    let Expression::Access { index, .. } = f.expressions[*pointer] else {
                        panic!("indexed scale store");
                    };
                    assert!(
                        !predicates.is_empty(),
                        "scale initialization requires its bound"
                    );
                    for lane in 0..64 {
                        if predicates
                            .iter()
                            .all(|&(condition, accept)| (eval(f, condition, lane) != 0) == accept)
                        {
                            let i = eval(f, index, lane) as usize;
                            assert!(i < 80, "scale initializer exceeds its owner");
                            writes[i] += 1;
                        }
                    }
                }
                Statement::Block(b) => visit(f, b, owner, predicates, writes),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    predicates.push((*condition, true));
                    visit(f, accept, owner, predicates, writes);
                    predicates.pop();
                    predicates.push((*condition, false));
                    visit(f, reject, owner, predicates, writes);
                    predicates.pop();
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    visit(f, body, owner, predicates, writes);
                    visit(f, continuing, owner, predicates, writes);
                }
                _ => {}
            }
        }
    }
    let mut writes = [0; 80];
    visit(f, &f.body, owner, &mut Vec::new(), &mut writes);
    assert!(
        writes.iter().all(|&n| n == 1),
        "generated initializer must write every scale exactly once"
    );
    assert_factor_barrier_and_immutability(module, owner);
}

#[test]
fn down_four_rows_actual_checked_unchecked_shader_storage_and_barriers() {
    use cubecl::server::ExecutionMode;
    use wgpu::naga::{AddressSpace, ArraySize, ScalarKind, TypeInner};
    for dtype in [IQ4_NL, super::super::Q2_0] {
        for share in [false, true] {
            let mut g = down_four_row_geometry(dtype).with_four_row_down().unwrap();
            if share {
                g = g.with_shared_down_scales().unwrap();
            }
            let has_scales = share && dtype == IQ4_NL;
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let source = down_four_row_shader(g, mode);
                validate_shader(&source, 8, true);
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
                let mut arrays = Vec::new();
                let mut bytes = 0;
                for (owner, var) in module.global_variables.iter() {
                    if var.space != AddressSpace::WorkGroup {
                        continue;
                    }
                    assert!(var.init.is_none());
                    let TypeInner::Array {
                        base,
                        size: ArraySize::Constant(n),
                        stride,
                    } = module.types[var.ty].inner
                    else {
                        panic!("fixed shared array");
                    };
                    assert!(
                        matches!(module.types[base].inner, TypeInner::Scalar(s) if s.kind == ScalarKind::Float && s.width == 4)
                    );
                    arrays.push(n.get());
                    bytes += n.get() * stride;
                    assert_shared_loads_follow_barrier(&module, owner);
                }
                arrays.sort_unstable();
                assert_eq!(arrays, if has_scales { vec![80, 512] } else { vec![512] });
                assert_eq!(bytes, if has_scales { 2368 } else { 2048 });
                assert_eq!(factor_spirv_storage_classes(&module, 512), vec![4]);
                assert_eq!(
                    factor_spirv_storage_classes(&module, 80),
                    if has_scales { vec![4] } else { vec![] }
                );
                // Writer uses the runtime's explicit None workgroup-zeroing mode.
                // One site is inside the stride-halving loop; four iterations
                // give five dynamic calls (six with scale initialization).
                assert_eq!(runtime_spirv_barriers(&module), 2 + usize::from(has_scales));
                let mut dynamic_barriers = 1 + usize::from(has_scales);
                let mut stride = 8;
                while stride > 0 {
                    dynamic_barriers += 1;
                    stride /= 2;
                }
                assert_eq!(dynamic_barriers, 5 + usize::from(has_scales));
                if has_scales {
                    assert_down_four_row_scale_coverage(&module);
                }
                let events = memory_events(&module);
                let exit = events
                    .iter()
                    .position(|&event| event == 'r')
                    .expect("uniform inactive exit");
                let write = events
                    .iter()
                    .position(|&event| event == 'w')
                    .expect("shared initialization");
                let barrier = events
                    .iter()
                    .position(|&event| event == 'b')
                    .expect("shared barrier");
                assert!(exit < write && write < barrier);
            }
        }
    }
}
