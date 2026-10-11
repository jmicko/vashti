// CPU/compiler-only acceptance for shrinking arithmetic scratch, not tile8 MAP.
// Included by grouped_moe::tests. No device is acquired here.

fn three_member_geometry(layer: usize) -> Geometry {
    let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
    Geometry::new(
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
}

#[test]
fn three_members_all_48_geometry_defaults_prerequisites_and_composition() {
    for layer in 0..48 {
        let old = three_member_geometry(layer);
        let ordinary = Geometry::new(old.projections(), 8, 144).unwrap();
        assert!(!ordinary.three_member_arithmetic());
        assert!(ordinary.with_three_member_arithmetic().is_err());
        assert!(
            ordinary
                .with_frozen_host_sources()
                .unwrap()
                .with_three_member_arithmetic()
                .is_err()
        );
        let new = old.with_three_member_arithmetic().unwrap();
        assert!(new.three_member_arithmetic());
        assert_eq!(old.arithmetic_members(), 8);
        assert_eq!(new.arithmetic_members(), 3);
        assert_eq!(old.down_shared_memory_bytes(), 2048);
        assert_eq!(new.down_shared_memory_bytes(), 768);
        assert_eq!(
            new.with_four_row_down().unwrap().down_shared_memory_bytes(),
            768
        );
        assert!(new.with_three_member_arithmetic().is_err());
        assert!(new.with_live_member_reduction().is_err());
        assert_eq!(new.tile(), 8);
        assert_eq!(new.member_capacity(), 8);
        assert_eq!(old.projections(), new.projections());
        assert_eq!(old.status_words(), new.status_words());
        for io in [false, true] {
            assert_eq!(old.scratch_bytes(io), new.scratch_bytes(io));
        }
        for count in 1..=3 {
            new.validate_count(count).unwrap();
        }
        for count in [0, 4, 8, usize::MAX] {
            assert!(new.validate_count(count).is_err());
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
            old.with_four_row_down()
                .unwrap()
                .with_three_member_arithmetic()
                .unwrap(),
            new.with_four_row_down().unwrap()
        );
        assert!(frozen_host_staging::Shape::new(new).is_ok());
        assert!(frozen_host_staging::Shape::new(new.with_four_row_down().unwrap()).is_ok());
        assert!(new.with_two_row_gate_up().is_err());
        assert!(
            old.with_two_row_gate_up()
                .unwrap()
                .with_three_member_arithmetic()
                .is_err()
        );
        assert!(new.with_private_tables().is_err());
        assert!(
            old.with_private_tables()
                .unwrap()
                .with_three_member_arithmetic()
                .is_err()
        );
        assert!(new.with_workgroup_tables().is_err());
        assert!(
            old.with_workgroup_tables()
                .unwrap()
                .with_three_member_arithmetic()
                .is_err()
        );
        assert!(new.with_shared_gate_up_factors().is_err());
        assert!(
            old.with_shared_gate_up_factors()
                .unwrap()
                .with_three_member_arithmetic()
                .is_err()
        );
        assert!(new.with_shared_down_scales().is_err());
        assert!(
            old.with_shared_down_scales()
                .unwrap()
                .with_three_member_arithmetic()
                .is_err()
        );
        // Direct malformed private-state fixtures exercise every fixed admission
        // prerequisite even when earlier public constructors already exclude it.
        for field in 0..8 {
            let mut bad = old;
            match field {
                0 => bad.capacity = 144,
                1 => bad.tile = 4,
                2 => bad.hidden = 2556,
                3 => bad.ff = 636,
                4 => bad.secondary_partition = true,
                5 => bad.gate_policy = GatePolicy::Q4Block,
                6 => bad.reduction_policy = ReductionPolicy::LiveMembers,
                _ => bad.inactive_partition_exit = false,
            }
            assert!(bad.with_three_member_arithmetic().is_err());
        }
    }
}

// The old and new arrays use different member strides. Each virtual lane still
// receives its original columns; only absent member planes disappear.
fn three_member_simt(
    weights: &[f32],
    input: &[Vec<f32>],
    slots: usize,
    physical: usize,
) -> Vec<f32> {
    let mut shared = vec![0_f32; slots * physical];
    let mut writes = vec![0; shared.len()];
    for lane in 0..physical {
        let mut partial = vec![0_f32; (64 / physical) * slots];
        for column_tile in 0..weights.len().div_ceil(64) {
            for virtual_lane in 0..64 / physical {
                let column = column_tile * 64 + lane + virtual_lane * physical;
                if column >= weights.len() {
                    continue;
                }
                let weight = weights[column];
                for (member, activation) in input.iter().enumerate() {
                    partial[virtual_lane * slots + member] += weight * activation[column];
                }
            }
        }
        for member in 0..slots {
            let value = if physical == 64 {
                partial[member]
            } else {
                assert_eq!(physical, 16);
                (partial[member] + partial[2 * slots + member])
                    + (partial[slots + member] + partial[3 * slots + member])
            };
            let index = member * physical + lane;
            shared[index] = value;
            writes[index] += 1;
        }
    }
    assert!(writes.iter().all(|&n| n == 1));
    let mut stride = physical / 2;
    while stride > 0 {
        for lane in 0..stride {
            for member in 0..slots {
                let index = member * physical + lane;
                shared[index] += shared[index + stride];
            }
        }
        stride /= 2;
    }
    (0..slots).map(|member| shared[member * physical]).collect()
}

#[test]
fn three_members_exact_c_weights_and_original_simt_order_for_counts_1_to_3() {
    use crate::quant_matvec::{IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, iq_fixtures};
    for dtype in [IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL, Q2_0] {
        let width = if matches!(dtype, IQ4_NL | Q2_0) {
            640
        } else {
            2560
        };
        for phase in [0, 1, 3] {
            let packed = iq_fixtures::packed(dtype, 4, width, phase);
            let expected = iq_fixtures::oracle(dtype, 4, width, phase);
            let mut decoded = vec![0.; expected.len()];
            kuno_text::TensorType::from_gguf_id(dtype)
                .unwrap()
                .decode(&packed, &mut decoded)
                .unwrap();
            assert!(
                decoded
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            for count in 1..=3 {
                let inputs: Vec<Vec<f32>> = (0..count)
                    .map(|member| {
                        (0..width)
                            .map(|column| {
                                ((column * 7 + member * 5 + phase * 3) % 31) as f32 / 32. - 0.5
                            })
                            .collect()
                    })
                    .collect();
                for row in 0..4 {
                    let weights = &decoded[row * width..][..width];
                    for physical in if width == 640 { vec![64, 16] } else { vec![64] } {
                        let old = three_member_simt(weights, &inputs, 8, physical);
                        let new = three_member_simt(weights, &inputs, 3, physical);
                        for (member, input) in inputs.iter().enumerate() {
                            let expected_dot = tree_dot(&expected[row * width..][..width], input);
                            assert_eq!(old[member].to_bits(), expected_dot.to_bits());
                            assert_eq!(new[member].to_bits(), expected_dot.to_bits());
                        }
                        assert!(old[count..].iter().all(|x| x.to_bits() == 0));
                        assert!(new[count..].iter().all(|x| x.to_bits() == 0));
                    }
                }
            }
        }
    }
}

fn three_member_shader(
    g: Geometry,
    down: bool,
    staged: bool,
    mode: cubecl::server::ExecutionMode,
) -> String {
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
    let name = if down {
        let weights = b.input_array(Type::new(uint)).into();
        let act = b.input_array(Type::new(float)).into();
        let map = b.input_array(Type::new(uint)).into();
        let output = b.output_array(Type::new(float)).into();
        let status = b.output_array(Type::new(uint)).into();
        let scalars: Vec<_> = (0..8).map(|_| b.scalar(uint)).collect();
        let expand = if staged {
            if g.four_row_down() {
                grouped_down_frozen_staged_four_rows_three_members::expand
            } else {
                grouped_down_frozen_staged_three_members::expand
            }
        } else if g.four_row_down() {
            grouped_down_frozen_four_rows_three_members::expand
        } else {
            grouped_down_frozen_three_members::expand
        };
        expand(
            &mut b.scope,
            weights,
            act,
            map,
            output,
            status,
            scalars[0].clone().into(),
            scalars[1].clone().into(),
            scalars[2].clone().into(),
            scalars[3].clone().into(),
            scalars[4].clone().into(),
            scalars[5].clone().into(),
            scalars[6].clone().into(),
            scalars[7].clone().into(),
            g.hidden(),
            g.ff(),
            g.projections()[2].dtype(),
            g.projections()[2].row_bytes(),
            g.tile(),
        );
        match (staged, g.four_row_down()) {
            (true, true) => "grouped_down_frozen_staged_four_rows_three_members",
            (true, false) => "grouped_down_frozen_staged_three_members",
            (false, true) => "grouped_down_frozen_four_rows_three_members",
            _ => "grouped_down_frozen_three_members",
        }
    } else {
        let gate = b.input_array(Type::new(uint)).into();
        let up = b.input_array(Type::new(uint)).into();
        let input = b.input_array(Type::new(float)).into();
        let map = b.input_array(Type::new(uint)).into();
        let act = b.output_array(Type::new(float)).into();
        let status = b.output_array(Type::new(uint)).into();
        let scalars: Vec<_> = (0..8).map(|_| b.scalar(uint)).collect();
        let expand = if staged {
            grouped_gate_up_frozen_staged_three_members::expand
        } else {
            grouped_gate_up_frozen_three_members::expand
        };
        let [gate_p, up_p, _] = g.projections();
        expand(
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
            g.hidden(),
            g.ff(),
            gate_p.dtype(),
            up_p.dtype(),
            gate_p.row_bytes(),
            up_p.row_bytes(),
            g.tile(),
            false,
        );
        if staged {
            "grouped_gate_up_frozen_staged_three_members"
        } else {
            "grouped_gate_up_frozen_three_members"
        }
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

// The production initializer is unrolled over three members. Evaluate only
// its lane-index expressions, stopping before the first shared barrier.
fn assert_three_member_shared_initializer(module: &wgpu::naga::Module) {
    use wgpu::naga::{
        AddressSpace, ArraySize, BinaryOperator, Expression, Literal, Statement, TypeInner,
    };
    let f = &module.entry_points[0].function;
    let owners: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(h, v)| {
            (v.space == AddressSpace::WorkGroup
                && matches!(module.types[v.ty].inner,
            TypeInner::Array{size:ArraySize::Constant(n),..}if n.get()==192))
            .then_some(h)
        })
        .collect();
    let mut writes = std::collections::HashMap::new();
    for &h in &owners {
        writes.insert(h, vec![0_u8; 192]);
    }
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
                    _ => panic!("unexpected shared initializer operator {op:?}"),
                }
            }
            ref other => panic!("unexpected shared initializer expression {other:?}"),
        }
    }
    fn visit(
        f: &wgpu::naga::Function,
        body: &wgpu::naga::Block,
        writes: &mut std::collections::HashMap<
            wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
            Vec<u8>,
        >,
    ) -> bool {
        for statement in body {
            match statement {
                Statement::ControlBarrier(_) => return true,
                Statement::Store { pointer, .. } => {
                    if let Some(owner) = shared_pointer_global(f, *pointer)
                        && let Some(cells) = writes.get_mut(&owner)
                    {
                        let Expression::Access { index, .. } = f.expressions[*pointer] else {
                            panic!("indexed shared initializer");
                        };
                        for lane in 0..64 {
                            let cell = eval(f, index, lane) as usize;
                            assert!(cell < 192);
                            cells[cell] += 1;
                        }
                    }
                }
                Statement::Block(b) if visit(f, b, writes) => return true,
                Statement::If { accept, reject, .. } => {
                    assert!(!visit(f, accept, writes));
                    assert!(!visit(f, reject, writes));
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    assert!(!visit(f, body, writes));
                    assert!(!visit(f, continuing, writes));
                }
                _ => {}
            }
        }
        false
    }
    assert!(visit(f, &f.body, &mut writes));
    assert!(!owners.is_empty());
    for owner in owners {
        assert!(
            writes[&owner].iter().all(|&n| n == 1),
            "every shared member/lane cell must be initialized once"
        );
        assert_shared_loads_follow_barrier(module, owner);
    }
}

#[test]
fn three_members_actual_checked_unchecked_shaders_scratch_and_initialization() {
    use cubecl::server::ExecutionMode;
    use wgpu::naga::{AddressSpace, ArraySize, ScalarKind, TypeInner};
    let mut seen = std::collections::HashSet::new();
    for layer in 0..48 {
        let old = three_member_geometry(layer);
        for down in [false, true] {
            let dtype = old.projections()[if down { 2 } else { 0 }].dtype();
            if !seen.insert((down, dtype)) {
                continue;
            }
            for four in if down { vec![false, true] } else { vec![false] } {
                let old = if four {
                    old.with_four_row_down().unwrap()
                } else {
                    old
                };
                let new = old.with_three_member_arithmetic().unwrap();
                for staged in [false, true] {
                    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                        let source = three_member_shader(new, down, staged, mode);
                        validate_shader(&source, 8, true);
                        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
                        let mut shared = 0;
                        let mut owners = 0;
                        for (_, var) in module.global_variables.iter() {
                            if var.space == AddressSpace::WorkGroup {
                                assert!(var.init.is_none());
                                let TypeInner::Array {
                                    base,
                                    size: ArraySize::Constant(n),
                                    stride,
                                } = module.types[var.ty].inner
                                else {
                                    panic!("fixed shared scratch");
                                };
                                assert_eq!(n.get(), 192);
                                assert!(
                                    matches!(module.types[base].inner,TypeInner::Scalar(s)if s.kind==ScalarKind::Float&&s.width==4)
                                );
                                shared += n.get() * stride;
                                owners += 1;
                            }
                        }
                        assert_eq!(owners, if down { 1 } else { 2 });
                        assert_eq!(shared, if down { 768 } else { 1536 });
                        assert_eq!(
                            factor_spirv_storage_classes(&module, 192),
                            if down { vec![4] } else { vec![4, 4] }
                        );
                        assert!(factor_spirv_storage_classes(&module, 8).is_empty());
                        assert!(factor_spirv_storage_classes(&module, 32).is_empty());
                        assert_eq!(
                            factor_spirv_storage_classes(&module, if four { 12 } else { 3 }),
                            if down { vec![7] } else { vec![7, 7] }
                        );
                        assert_three_member_shared_initializer(&module);
                        assert_eq!(runtime_spirv_barriers(&module), 2);
                        let events = memory_events(&module);
                        assert!(
                            events.iter().position(|&x| x == 'r').unwrap()
                                < events.iter().position(|&x| x == 'w').unwrap()
                        );
                        let original = if down && four {
                            down_four_row_shader(old, mode)
                        } else {
                            shader(usize::from(down), old, mode)
                        };
                        let original = wgpu::naga::front::wgsl::parse_str(&original).unwrap();
                        assert_eq!(
                            runtime_spirv_barriers(&module),
                            runtime_spirv_barriers(&original)
                        );
                        for kind in [2, 3] {
                            // Compiler HashMap iteration may reorder hoisted,
                            // uninitialized local declarations. Preserve every
                            // executable line and compare the declaration set.
                            let normalize = |source: String| {
                                let (mut declarations, body): (Vec<_>, Vec<_>) = source
                                    .lines()
                                    .partition(|line| line.starts_with("var l_mut_"));
                                assert!(declarations.iter().all(|line| !line.contains('=')));
                                declarations.sort_unstable();
                                (declarations.join("\n"), body.join("\n"))
                            };
                            assert_eq!(
                                normalize(shader(kind, old, mode)),
                                normalize(shader(kind, new, mode))
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(seen.len(), 6);
}
