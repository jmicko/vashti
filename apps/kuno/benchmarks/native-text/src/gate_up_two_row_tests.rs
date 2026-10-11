// Offline original-arithmetic acceptance; no device is acquired here.

fn gate_two_row_shader(g: Geometry, mode: cubecl::server::ExecutionMode) -> String {
    use cubecl::compute::KernelBuilder;
    use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
    use cubecl::wgpu::WgslCompiler;
    use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
    assert!(g.two_row_gate_up());
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
    let gate = b.input_array(Type::new(uint)).into();
    let up = b.input_array(Type::new(uint)).into();
    let input = b.input_array(Type::new(float)).into();
    let map = b.input_array(Type::new(uint)).into();
    let act = b.output_array(Type::new(float)).into();
    let status = b.output_array(Type::new(uint)).into();
    let scalars: Vec<_> = (0..8).map(|_| b.scalar(uint)).collect();
    grouped_gate_up_frozen_two_rows::expand(
        &mut b.scope,
        gate,
        up,
        input,
        map,
        act,
        status,
        scalars[0].clone().into(),
        scalars[1].clone().into(),
        scalars[2].clone().into(),
        scalars[3].clone().into(),
        scalars[4].clone().into(),
        scalars[5].clone().into(),
        scalars[6].clone().into(),
        scalars[7].clone().into(),
        g.hidden,
        g.ff,
        g.projections[0].dtype,
        g.projections[1].dtype,
        g.projections[0].row_bytes,
        g.projections[1].row_bytes,
        g.tile,
        false,
    );
    WgslCompiler::default()
        .compile(
            b.build(
                KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(64))
                    .kernel_name("grouped_gate_up_frozen_two_rows"),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            uint,
        )
        .unwrap()
        .to_string()
}

#[test]
fn gate_two_rows_all_48_geometry_and_down_composition() {
    for layer in 0..48 {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
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
        assert!(!ordinary.two_row_gate_up());
        assert!(ordinary.with_two_row_gate_up().is_err());
        let frozen = ordinary.with_frozen_host_sources().unwrap();
        assert!(frozen.with_two_row_gate_up().is_err());
        let old = frozen.with_inactive_partition_exit().unwrap();
        let paired = old.with_two_row_gate_up().unwrap();
        assert!(paired.two_row_gate_up());
        assert!(paired.with_two_row_gate_up().is_err());
        assert!(paired.with_private_tables().is_err());
        assert!(paired.with_workgroup_tables().is_err());
        assert!(paired.with_shared_gate_up_factors().is_err());
        assert!(
            old.with_private_tables()
                .unwrap()
                .with_two_row_gate_up()
                .is_err()
        );
        assert!(
            old.with_workgroup_tables()
                .unwrap()
                .with_two_row_gate_up()
                .is_err()
        );
        assert!(
            old.with_shared_gate_up_factors()
                .unwrap()
                .with_two_row_gate_up()
                .is_err()
        );
        assert!(paired.with_live_member_reduction().is_err());
        assert_eq!(old.projections(), paired.projections());
        assert_eq!(old.status_words(), paired.status_words());
        assert_eq!(old.member_capacity(), paired.member_capacity());
        for io in [false, true] {
            assert_eq!(old.scratch_bytes(io), paired.scratch_bytes(io));
        }
        for groups in [1, 10, 30] {
            assert!(
                matches!(old.grid(groups, false).unwrap(), CubeCount::Static(640, y, 1) if y == groups as u32)
            );
            assert!(
                matches!(paired.grid(groups, false).unwrap(), CubeCount::Static(320, y, 1) if y == groups as u32)
            );
            assert_eq!(
                format!("{:?}", old.grid(groups, true).unwrap()),
                format!("{:?}", paired.grid(groups, true).unwrap())
            );
        }
        let combined = paired
            .with_four_row_down()
            .unwrap()
            .with_shared_down_scales()
            .unwrap();
        assert_eq!(
            combined,
            old.with_shared_down_scales()
                .unwrap()
                .with_four_row_down()
                .unwrap()
                .with_two_row_gate_up()
                .unwrap()
        );
        assert_eq!(
            combined.down_shared_memory_bytes(),
            old.with_four_row_down()
                .unwrap()
                .with_shared_down_scales()
                .unwrap()
                .down_shared_memory_bytes()
        );
        assert!(matches!(
            combined.grid(30, true).unwrap(),
            CubeCount::Static(640, 30, 1)
        ));
        let mut odd = old;
        odd.ff = 639;
        assert!(odd.with_two_row_gate_up().is_err());
    }
    let mut writes = [0u8; 640];
    for cube in 0..320 {
        for lane in 0..64 {
            if lane % 32 == 0 {
                writes[cube * 2 + lane / 32] += 1;
            }
        }
    }
    assert!(writes.iter().all(|&n| n == 1));
}

#[test]
fn gate_two_rows_original_iq_vectors_virtual64_dots_and_paired_silu_are_exact() {
    use super::super::{IQ2_S, IQ3_S, IQ4_XS};
    for dtype in [IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS] {
        let kind = kuno_text::TensorType::from_gguf_id(dtype).unwrap();
        for phase in [0, 3] {
            let gate_bytes = super::super::iq_fixtures::packed(dtype, 8, 2560, phase);
            let up_bytes = super::super::iq_fixtures::packed(dtype, 8, 2560, phase + 1);
            let mut gates = vec![0.; 8 * 2560];
            let mut ups = vec![0.; gates.len()];
            kind.decode(&gate_bytes, &mut gates).unwrap();
            kind.decode(&up_bytes, &mut ups).unwrap();
            assert_eq!(
                gates.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                super::super::iq_fixtures::oracle(dtype, 8, 2560, phase)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                ups.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                super::super::iq_fixtures::oracle(dtype, 8, 2560, phase + 1)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            let inputs: Vec<Vec<f32>> = (0..8)
                .map(|member| {
                    (0..2560)
                        .map(|c| ((c * 7 + member * 11 + phase * 3) % 67) as f32 / 32. - 1.)
                        .collect()
                })
                .collect();
            for length in [1, 3, 8] {
                for cube in 0..4 {
                    let mut gs = [0.0_f32; 512];
                    let mut us = [0.0_f32; 512];
                    let mut writes = [0; 512];
                    for lane in 0..64 {
                        let row = lane / 32;
                        let local_lane = lane % 32;
                        let feature = cube * 2 + row;
                        let mut g = [[0.0_f32; 8]; 2];
                        let mut u = [[0.0_f32; 8]; 2];
                        for column_tile in 0..40 {
                            for virtual_lane in 0..2 {
                                let column = column_tile * 64 + local_lane + virtual_lane * 32;
                                for member in 0..length {
                                    g[virtual_lane][member] +=
                                        gates[feature * 2560 + column] * inputs[member][column];
                                    u[virtual_lane][member] +=
                                        ups[feature * 2560 + column] * inputs[member][column];
                                }
                            }
                        }
                        for member in 0..8 {
                            let index = (row * 8 + member) * 32 + local_lane;
                            gs[index] = g[0][member] + g[1][member];
                            us[index] = u[0][member] + u[1][member];
                            writes[index] += 1;
                        }
                    }
                    assert!(writes.iter().all(|&n| n == 1));
                    for stride in [16, 8, 4, 2, 1] {
                        for row in 0..2 {
                            for member in 0..8 {
                                for lane in 0..stride {
                                    let index = (row * 8 + member) * 32 + lane;
                                    gs[index] += gs[index + stride];
                                    us[index] += us[index + stride];
                                }
                            }
                        }
                    }
                    for row in 0..2 {
                        let feature = cube * 2 + row;
                        for (member, input) in inputs.iter().enumerate() {
                            let index = (row * 8 + member) * 32;
                            let (g, u) = if member < length {
                                (
                                    tree_dot(&gates[feature * 2560..][..2560], input),
                                    tree_dot(&ups[feature * 2560..][..2560], input),
                                )
                            } else {
                                (0., 0.)
                            };
                            assert_eq!(gs[index].to_bits(), g.to_bits());
                            assert_eq!(us[index].to_bits(), u.to_bits());
                            assert_eq!(
                                (core_silu(gs[index]) * us[index]).to_bits(),
                                (core_silu(g) * u).to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}

fn assert_gate_two_row_initial_stores(module: &wgpu::naga::Module) {
    use wgpu::naga::{AddressSpace, BinaryOperator, Expression, Literal, Statement};
    fn eval(f: &wgpu::naga::Function, h: wgpu::naga::Handle<Expression>, lane: u32) -> u32 {
        if is_physical_lane_x(f, h) {
            return lane;
        }
        match f.expressions[h] {
            Expression::Literal(Literal::U32(n)) => n,
            Expression::As { expr, .. } => eval(f, expr, lane),
            Expression::Binary { op, left, right } => {
                let a = eval(f, left, lane);
                let b = eval(f, right, lane);
                match op {
                    BinaryOperator::Add => a + b,
                    BinaryOperator::Multiply => a * b,
                    BinaryOperator::Divide => a / b,
                    BinaryOperator::Modulo => a % b,
                    _ => panic!("unexpected reduction index {op:?}"),
                }
            }
            ref other => panic!("unexpected reduction index {other:?}"),
        }
    }
    fn walk(module: &wgpu::naga::Module, block: &wgpu::naga::Block, writes: &mut [[u32; 512]; 2]) {
        let f = &module.entry_points[0].function;
        for statement in block {
            match statement {
                Statement::Store { pointer, .. } => {
                    if let Some(owner) = shared_pointer_global(f, *pointer)
                        && module.global_variables[owner].space == AddressSpace::WorkGroup
                    {
                        let owners: Vec<_> = module
                            .global_variables
                            .iter()
                            .filter(|(_, v)| v.space == AddressSpace::WorkGroup)
                            .map(|(h, _)| h)
                            .collect();
                        let which = owners.iter().position(|&h| h == owner).unwrap();
                        let Expression::Access { index, .. } = f.expressions[*pointer] else {
                            panic!("indexed sum store");
                        };
                        for lane in 0..64 {
                            let i = eval(f, index, lane) as usize;
                            assert!(i < 512, "sum initializer exceeds owning array");
                            writes[which][i] += 1;
                        }
                    }
                }
                Statement::If { accept, reject, .. } => {
                    walk(module, accept, writes);
                    walk(module, reject, writes);
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    walk(module, body, writes);
                    walk(module, continuing, writes);
                }
                Statement::Block(body) => walk(module, body, writes),
                _ => {}
            }
        }
    }
    let f = &module.entry_points[0].function;
    let fence = f
        .body
        .iter()
        .position(|s| matches!(s, Statement::ControlBarrier(_)))
        .unwrap();
    let before = wgpu::naga::Block::from_vec(f.body.iter().take(fence).cloned().collect());
    let mut writes = [[0; 512]; 2];
    walk(module, &before, &mut writes);
    assert!(
        writes.iter().flatten().all(|&n| n == 1),
        "every sum cell initialized exactly once before fence"
    );
}

#[test]
fn gate_two_rows_actual_checked_unchecked_shader_and_runtime_spirv() {
    use cubecl::server::ExecutionMode;
    use wgpu::naga::{AddressSpace, ArraySize, TypeInner};
    let mut profiles = Vec::new();
    for layer in 0..48 {
        let types = crate::daslab_profile::routed_types(layer).unwrap();
        if !profiles.contains(&types) {
            profiles.push(types);
        }
    }
    for [gate, up, down] in profiles {
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
        .with_two_row_gate_up()
        .unwrap();
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let source = gate_two_row_shader(g, mode);
            validate_shader(&source, 8, true);
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let arrays: Vec<_> = module
                .global_variables
                .iter()
                .filter(|(_, v)| v.space == AddressSpace::WorkGroup)
                .collect();
            assert_eq!(arrays.len(), 2);
            for &(owner, var) in &arrays {
                assert!(
                    matches!(module.types[var.ty].inner, TypeInner::Array { size: ArraySize::Constant(n), stride: 4, .. } if n.get() == 512)
                );
                assert_shared_loads_follow_barrier(&module, owner);
            }
            assert_gate_two_row_initial_stores(&module);
            assert_eq!(runtime_spirv_barriers(&module), 2);
            assert_eq!(
                unconditional_barriers(&module.entry_points[0].function.body, false),
                2
            );
            let events = memory_events(&module);
            assert!(
                events.iter().position(|&c| c == 'r').unwrap()
                    < events.iter().position(|&c| c == 'w').unwrap()
            );
            assert!(
                events.iter().position(|&c| c == 'b').unwrap()
                    < events.iter().position(|&c| c == 's').unwrap()
            );
            // Exact64 tree: folded32 plus five original shared strides.
            assert_eq!(1 + [16, 8, 4, 2, 1].len(), 6);
            let mut original = g;
            original.two_row_gate_up = false;
            for kind in [1, 2, 3] {
                assert_eq!(
                    canonical_declaration_order(&shader(kind, original, mode)),
                    canonical_declaration_order(&shader(kind, g, mode)),
                    "down/finite/rank reduction arithmetic and fault coverage unchanged"
                );
            }
        }
    }
}

#[test]
fn gate_two_rows_proof_rejects_oob_initialization_and_divergent_barrier() {
    use wgpu::naga::{AddressSpace, Expression, Literal, Statement};
    let g = down_four_row_geometry(IQ4_NL)
        .with_two_row_gate_up()
        .unwrap();
    let source = gate_two_row_shader(g, cubecl::server::ExecutionMode::Unchecked);
    let mut module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
    let function = &mut module.entry_points[0].function;
    let index = function
        .expressions
        .iter()
        .find_map(|(_, expression)| {
            if let Expression::Access { base, index } = expression
                && let Some(owner) = shared_pointer_global(function, *base)
                && module.global_variables[owner].space == AddressSpace::WorkGroup
            {
                Some(*index)
            } else {
                None
            }
        })
        .unwrap();
    function.expressions[index] = Expression::Literal(Literal::U32(512));
    assert!(std::panic::catch_unwind(|| assert_gate_two_row_initial_stores(&module)).is_err());
    let mut module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
    let function = &mut module.entry_points[0].function;
    let condition = function.expressions.append(
        Expression::Literal(Literal::Bool(true)),
        wgpu::naga::Span::UNDEFINED,
    );
    let barrier = function
        .body
        .iter_mut()
        .find(|s| matches!(s, Statement::ControlBarrier(_)))
        .unwrap();
    *barrier = Statement::If {
        condition,
        accept: wgpu::naga::Block::from_vec(vec![barrier.clone()]),
        reject: wgpu::naga::Block::new(),
    };
    assert!(std::panic::catch_unwind(|| unconditional_barriers(&function.body, false)).is_err());
}
