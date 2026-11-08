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
        let workgroup_first = old
            .with_workgroup_tables()
            .unwrap()
            .with_shared_down_scales()
            .unwrap();
        let scales_first = new.with_workgroup_tables().unwrap();
        assert_eq!(workgroup_first, scales_first);
        assert!(workgroup_first.workgroup_tables() && workgroup_first.shared_down_scales());
        assert!(!workgroup_first.private_tables());
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
            assert_eq!(
                runtime_spirv_barriers(&module),
                2 + usize::from(down == IQ4_NL)
            );
            if down == IQ4_NL {
                assert_shared_down_scale_initializer(&module);
                let original = wgpu::naga::front::wgsl::parse_str(&shader(1, old, mode)).unwrap();
                let mut events = memory_events(&module);
                let first = events.iter().position(|&event| event == 'w').unwrap();
                assert_eq!(&events[first..first + 2], &['w', 'b']);
                assert!(events.iter().position(|&event| event == 'r').unwrap() < first);
                events.drain(first..first + 2);
                assert_eq!(events, memory_events(&original));
            }
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

// Initialization must be complete without the SPIR-V workgroup-zeroing polyfill.
fn assert_shared_down_scale_initializer(module: &wgpu::naga::Module) {
    use wgpu::naga::{
        AddressSpace, ArraySize, BinaryOperator, Expression, Literal, Statement, TypeInner,
    };
    let (array, _) = module
        .global_variables
        .iter()
        .find(|(_, var)| {
            var.space == AddressSpace::WorkGroup
                && matches!(module.types[var.ty].inner,
            TypeInner::Array { size: ArraySize::Constant(n), .. } if n.get() == 20)
        })
        .expect("twenty shared IQ4_NL block scales");
    let function = &module.entry_points[0].function;
    fn stores(
        function: &wgpu::naga::Function,
        block: &wgpu::naga::Block,
        array: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
    ) -> usize {
        block
            .iter()
            .map(|statement| match statement {
                Statement::Store { pointer, .. } => {
                    usize::from(shared_pointer_global(function, *pointer) == Some(array))
                }
                Statement::Block(body) => stores(function, body, array),
                Statement::If { accept, reject, .. } => {
                    stores(function, accept, array) + stores(function, reject, array)
                }
                Statement::Switch { cases, .. } => cases
                    .iter()
                    .map(|case| stores(function, &case.body, array))
                    .sum(),
                Statement::Loop {
                    body, continuing, ..
                } => stores(function, body, array) + stores(function, continuing, array),
                _ => 0,
            })
            .sum()
    }
    fn lane_guard(
        function: &wgpu::naga::Function,
        condition: wgpu::naga::Handle<Expression>,
    ) -> bool {
        matches!(function.expressions[condition], Expression::Binary { op: BinaryOperator::Less, left, right }
            if is_physical_lane_x(function, left) && matches!(function.expressions[right], Expression::Literal(Literal::U32(20))))
    }
    fn init_stores(
        function: &wgpu::naga::Function,
        block: &wgpu::naga::Block,
        array: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
    ) {
        for statement in block {
            match statement {
                Statement::Store { pointer, .. }
                    if shared_pointer_global(function, *pointer) == Some(array) =>
                {
                    let Expression::Access { index, .. } = function.expressions[*pointer] else {
                        panic!("scale store must use physical lane index");
                    };
                    assert!(
                        is_physical_lane_x(function, index),
                        "scale store must use physical lane index"
                    );
                }
                Statement::Block(body) => init_stores(function, body, array),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    if stores(function, accept, array) + stores(function, reject, array) != 0 {
                        assert!(
                            lane_guard(function, *condition),
                            "scale write must be covered by lane<20"
                        );
                        assert_eq!(stores(function, reject, array), 0);
                    }
                    init_stores(function, accept, array);
                    init_stores(function, reject, array);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        init_stores(function, &case.body, array);
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    init_stores(function, body, array);
                    init_stores(function, continuing, array);
                }
                _ => {}
            }
        }
    }
    fn find(
        function: &wgpu::naga::Function,
        block: &wgpu::naga::Block,
        array: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
    ) -> usize {
        let mut guards = 0;
        for (i, statement) in block.iter().enumerate() {
            if let Statement::If {
                condition,
                accept,
                reject,
            } = statement
                && lane_guard(function, *condition)
                && stores(function, accept, array) != 0
            {
                assert_eq!(stores(function, accept, array), 1);
                assert_eq!(stores(function, reject, array), 0);
                init_stores(function, accept, array);
                assert!(
                    matches!(block.get(i + 1), Some(Statement::ControlBarrier(_))),
                    "scale initializer requires barrier after lane guard"
                );
                guards += 1;
                continue;
            }
            guards += match statement {
                Statement::Block(body) => find(function, body, array),
                Statement::If { accept, reject, .. } => {
                    find(function, accept, array) + find(function, reject, array)
                }
                Statement::Switch { cases, .. } => cases
                    .iter()
                    .map(|case| find(function, &case.body, array))
                    .sum(),
                Statement::Loop {
                    body, continuing, ..
                } => find(function, body, array) + find(function, continuing, array),
                _ => 0,
            };
        }
        guards
    }
    assert_eq!(
        stores(function, &function.body, array),
        1,
        "shared scales must be stored only once at initialization"
    );
    assert_eq!(find(function, &function.body, array), 1);
    for (_, function) in module.functions.iter() {
        assert_eq!(
            stores(function, &function.body, array),
            0,
            "shared scales cannot be mutated through a helper function"
        );
    }
    assert_shared_loads_follow_barrier(module, array);
}

#[test]
#[should_panic(expected = "shared array read before initialization barrier")]
fn shared_down_scales_read_before_barrier_cannot_bypass_initialization_proof() {
    let module = wgpu::naga::front::wgsl::parse_str(
        "var<workgroup> scales:array<f32,20>; @group(0) @binding(0) var<storage,read_write> output:array<f32>; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { if id.x<20u {scales[id.x]=1.;} output[id.x]=scales[id.x%20u]; workgroupBarrier(); }"
    ).unwrap();
    let array = module
        .global_variables
        .iter()
        .find(|(_, var)| var.name.as_deref() == Some("scales"))
        .unwrap()
        .0;
    assert_shared_loads_follow_barrier(&module, array);
}

#[test]
#[should_panic(expected = "shared scales must be stored only once at initialization")]
fn shared_down_scales_post_init_mutation_cannot_bypass_initializer_proof() {
    let module = wgpu::naga::front::wgsl::parse_str(
        "var<workgroup> scales:array<f32,20>; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { if id.x<20u {scales[id.x]=1.;} workgroupBarrier(); if id.x==0u {scales[0]=2.;} }"
    ).unwrap();
    assert_shared_down_scale_initializer(&module);
}

#[test]
#[should_panic(expected = "shared scales cannot be mutated through a helper function")]
fn shared_down_scales_helper_mutation_cannot_bypass_initializer_proof() {
    let module = wgpu::naga::front::wgsl::parse_str(
        "var<workgroup> scales:array<f32,20>; fn mutate() {scales[0]=2.;} @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { if id.x<20u {scales[id.x]=1.;} workgroupBarrier(); mutate(); }"
    ).unwrap();
    assert_shared_down_scale_initializer(&module);
}

#[test]
fn shared_down_scales_workgroup_composition_actual_profiles_checked_unchecked() {
    use cubecl::server::ExecutionMode;
    let mut profiles = Vec::new();
    for layer in 0..48 {
        let profile = crate::daslab_profile::routed_types(layer).unwrap();
        if !profiles.contains(&profile) {
            profiles.push(profile);
        }
    }
    for [gate, up, down] in profiles {
        let old = Geometry::new(
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
        .unwrap();
        let table = old.with_workgroup_tables().unwrap();
        let scales = old.with_shared_down_scales().unwrap();
        let combined = table.with_shared_down_scales().unwrap();
        assert_eq!(combined, scales.with_workgroup_tables().unwrap());
        assert!(combined.workgroup_tables() && combined.shared_down_scales());
        assert!(!combined.private_tables());
        assert!(combined.with_private_tables().is_err());
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            for (kind, separate) in [(0, table), (1, scales), (2, old), (3, old)] {
                let source = shader(kind, combined, mode);
                assert_eq!(
                    canonical_declaration_order(&source),
                    canonical_declaration_order(&shader(kind, separate, mode))
                );
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                let extra = match kind {
                    0 => usize::from(gate != super::super::IQ4_XS),
                    1 => usize::from(down == IQ4_NL),
                    _ => 0,
                };
                if kind == 0 {
                    assert_eq!(spirv_tables(&module), (0, 0, 0));
                    if extra != 0 {
                        assert_workgroup_tables_immutable_after_init(&module);
                    }
                } else if kind == 1 && down == IQ4_NL {
                    assert_shared_down_scale_initializer(&module);
                }
                let source_barriers = memory_events(&module)
                    .iter()
                    .filter(|&&event| event == 'b')
                    .count();
                assert_eq!(runtime_spirv_barriers(&module), source_barriers);
                if kind < 2 {
                    assert_eq!(source_barriers, 2 + extra);
                }
            }
        }
    }
}
