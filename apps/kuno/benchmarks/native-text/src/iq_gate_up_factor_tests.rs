// Offline oracle and compiler acceptance for the construction-fixed GU factors.
// This file is included in grouped_moe::tests; it owns no GPU or runtime action.

fn factor_shader_name(g: Geometry) -> &'static str {
    if g.shared_gate_up_factors() {
        if g.workgroup_tables() {
            "grouped_gate_up_frozen_workgroup_grid_shared_factors"
        } else if g.private_tables() {
            "grouped_gate_up_frozen_private_grid_shared_factors"
        } else {
            "grouped_gate_up_frozen_shared_factors"
        }
    } else if g.workgroup_tables() {
        "grouped_gate_up_frozen_workgroup_grid"
    } else if g.private_tables() {
        "grouped_gate_up_frozen_private_grid"
    } else {
        "grouped_gate_up_frozen"
    }
}

fn factor_shader(g: Geometry, mode: cubecl::server::ExecutionMode) -> String {
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
    let gate = b.input_array(Type::new(uint)).into();
    let up = b.input_array(Type::new(uint)).into();
    let input = b.input_array(Type::new(float)).into();
    let map = b.input_array(Type::new(uint)).into();
    let act = b.output_array(Type::new(float)).into();
    let status = b.output_array(Type::new(uint)).into();
    let map_first = b.scalar(uint).into();
    let input_first = b.scalar(uint).into();
    let input_stride = b.scalar(uint).into();
    let act_first = b.scalar(uint).into();
    let act_stride = b.scalar(uint).into();
    let status_first = b.scalar(uint).into();
    let first_slot = b.scalar(uint).into();
    let slot_count = b.scalar(uint).into();
    let expand = if g.shared_gate_up_factors() {
        if g.workgroup_tables() {
            grouped_gate_up_frozen_workgroup_grid_shared_factors::expand
        } else if g.private_tables() {
            grouped_gate_up_frozen_private_grid_shared_factors::expand
        } else {
            grouped_gate_up_frozen_shared_factors::expand
        }
    } else if g.workgroup_tables() {
        grouped_gate_up_frozen_workgroup_grid::expand
    } else if g.private_tables() {
        grouped_gate_up_frozen_private_grid::expand
    } else {
        grouped_gate_up_frozen::expand
    };
    expand(
        &mut b.scope,
        gate,
        up,
        input,
        map,
        act,
        status,
        map_first,
        input_first,
        input_stride,
        act_first,
        act_stride,
        status_first,
        first_slot,
        slot_count,
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
                    .kernel_name(factor_shader_name(g)),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            uint,
        )
        .unwrap()
        .to_string()
}

fn factor_count_and_width(dtype: u32) -> (usize, usize) {
    match dtype {
        super::super::IQ2_S => (160, 16),
        IQ3_XXS | super::super::IQ3_S => (80, 32),
        IQ4_XS => (0, 0),
        _ => panic!("non-public factor profile"),
    }
}

fn factor_oracle(row: &[u8], column: usize, dtype: u32) -> f32 {
    use kuno_text::quant::f16_to_f32;
    let block_bytes = match dtype {
        super::super::IQ2_S => 82,
        IQ3_XXS => 98,
        super::super::IQ3_S => 110,
        _ => unreachable!(),
    };
    let b = &row[column / 256 * block_bytes..][..block_bytes];
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let group = column % 256 / 32;
    match dtype {
        super::super::IQ2_S => {
            let scales = b[74 + group];
            let scale = if column % 32 < 16 {
                scales & 15
            } else {
                scales >> 4
            };
            d * (0.5 + f32::from(scale)) * 0.25
        }
        IQ3_XXS => {
            let word = u32::from_le_bytes(b[66 + group * 4..][..4].try_into().unwrap());
            d * (0.5 + (word >> 28) as f32) * 0.5
        }
        super::super::IQ3_S => {
            let scales = b[106 + group / 2];
            let scale = if group.is_multiple_of(2) {
                scales & 15
            } else {
                scales >> 4
            };
            d * f32::from(1 + 2 * scale)
        }
        _ => unreachable!(),
    }
}

fn staged_factor_oracle(row: &[u8], dtype: u32) -> Vec<f32> {
    use kuno_text::quant::f16_to_f32;
    let block_bytes = match dtype {
        super::super::IQ2_S => 82,
        IQ3_XXS => 98,
        super::super::IQ3_S => 110,
        _ => unreachable!(),
    };
    let mut factors = Vec::new();
    for b in row.chunks_exact(block_bytes) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        match dtype {
            super::super::IQ2_S => {
                for &scales in &b[74..82] {
                    for scale in [scales & 15, scales >> 4] {
                        factors.push(d * (0.5 + f32::from(scale)) * 0.25);
                    }
                }
            }
            IQ3_XXS => {
                for bytes in b[66..98].chunks_exact(4) {
                    let word = u32::from_le_bytes(bytes.try_into().unwrap());
                    factors.push(d * (0.5 + (word >> 28) as f32) * 0.5);
                }
            }
            super::super::IQ3_S => {
                for &scales in &b[106..110] {
                    for scale in [scales & 15, scales >> 4] {
                        factors.push(d * f32::from(1 + 2 * scale));
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    factors
}

fn value_from_factor_oracle(row: &[u8], column: usize, factor: f32, dtype: u32) -> f32 {
    use kuno_text::quant::{IQ2_S_GRID, IQ3_S_GRID, IQ3_XXS_GRID};
    let block_bytes = match dtype {
        super::super::IQ2_S => 82,
        IQ3_XXS => 98,
        super::super::IQ3_S => 110,
        _ => unreachable!(),
    };
    let b = &row[column / 256 * block_bytes..][..block_bytes];
    let within = column % 256;
    let group = within / 32;
    let octet = within % 32 / 8;
    let lane = within % 8;
    let (magnitude, signs) = match dtype {
        super::super::IQ2_S => {
            let low = usize::from(b[2 + group * 4 + octet]);
            let high = usize::from((b[66 + group] >> (2 * octet)) & 3);
            let grid = IQ2_S_GRID[low + high * 256];
            (
                ((grid >> (lane * 8)) & 255) as u32,
                u32::from(b[34 + group * 4 + octet]),
            )
        }
        IQ3_XXS => {
            let word = u32::from_le_bytes(b[66 + group * 4..][..4].try_into().unwrap());
            let code = (word >> (octet * 7)) & 127;
            let signs = code | ((code.count_ones() & 1) << 7);
            let grid = IQ3_XXS_GRID[usize::from(b[2 + group * 8 + octet * 2 + lane / 4])];
            ((grid >> (lane % 4 * 8)) & 255, signs)
        }
        super::super::IQ3_S => {
            let quartet = octet * 2 + lane / 4;
            let low = usize::from(b[2 + group * 8 + quartet]);
            let high = usize::from((b[66 + group] >> quartet) & 1);
            let grid = IQ3_S_GRID[low + high * 256];
            (
                (grid >> (lane % 4 * 8)) & 255,
                u32::from(b[74 + group * 4 + octet]),
            )
        }
        _ => unreachable!(),
    };
    let sign = if signs & (1 << lane) == 0 { 1.0 } else { -1.0 };
    factor * magnitude as f32 * sign
}

#[test]
fn shared_gate_up_factors_original_c_weights_and_64_lane_dot_are_exact() {
    for dtype in [super::super::IQ2_S, IQ3_XXS, super::super::IQ3_S] {
        let kind = kuno_text::TensorType::from_gguf_id(dtype).unwrap();
        let row_bytes = 10 * kind.block_bytes() as usize;
        let (count, width) = factor_count_and_width(dtype);
        let mut initialized = vec![0; count * 2];
        for lane in 0..64 {
            for factor_tile in 0..(count * 2).div_ceil(64) {
                let index = factor_tile * 64 + lane;
                if index < count * 2 {
                    let projection = usize::from(index >= count);
                    let subgroup = index - projection * count;
                    assert!(subgroup < count && subgroup * width < 2560);
                    initialized[projection * count + subgroup] += 1;
                }
            }
        }
        assert!(initialized.iter().all(|&writes| writes == 1));
        for phase in 0..8 {
            let gate = super::super::iq_fixtures::packed(dtype, 3, 2560, phase);
            let up = super::super::iq_fixtures::packed(dtype, 3, 2560, phase + 1);
            let sources = [gate, up];
            for (projection, packed) in sources.iter().enumerate() {
                let expected =
                    super::super::iq_fixtures::oracle(dtype, 3, 2560, phase + projection);
                let mut decoded = vec![0.0; expected.len()];
                kind.decode(packed, &mut decoded).unwrap();
                for row in 0..3 {
                    let bytes = &packed[row * row_bytes..][..row_bytes];
                    let factors = staged_factor_oracle(bytes, dtype);
                    assert_eq!(factors.len(), count);
                    let mut combined = vec![0.0; count * 2];
                    for lane in 0..64 {
                        for factor_tile in 0..(count * 2).div_ceil(64) {
                            let index = factor_tile * 64 + lane;
                            if index < count * 2 {
                                let source = usize::from(index >= count);
                                let column = (index - source * count) * width;
                                combined[index] = factor_oracle(
                                    &sources[source][row * row_bytes..][..row_bytes],
                                    column,
                                    dtype,
                                );
                            }
                        }
                    }
                    let mut actual = vec![0.0; 2560];
                    for (column, value) in actual.iter_mut().enumerate() {
                        let shared = combined[projection * count + column / width];
                        assert_eq!(shared.to_bits(), factors[column / width].to_bits());
                        assert_eq!(
                            shared.to_bits(),
                            factor_oracle(bytes, column, dtype).to_bits()
                        );
                        *value = value_from_factor_oracle(bytes, column, shared, dtype);
                        let index = row * 2560 + column;
                        assert_eq!(
                            value.to_bits(),
                            expected[index].to_bits(),
                            "dtype{dtype} phase{phase} projection{projection} row{row} column{column}"
                        );
                        assert_eq!(decoded[index].to_bits(), expected[index].to_bits());
                    }
                    for member in 0..3 {
                        let input: Vec<_> = (0..2560)
                            .map(|c| ((c * 7 + phase * 3 + member * 5) % 31) as f32 / 32.0 - 0.5)
                            .collect();
                        assert_eq!(
                            tree_dot(&actual, &input).to_bits(),
                            tree_dot(&expected[row * 2560..][..2560], &input).to_bits()
                        );
                    }
                }
            }
        }
    }
}

fn assert_factor_barrier_and_immutability(
    module: &wgpu::naga::Module,
    owner: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
) {
    use wgpu::naga::{Expression, Function, Statement};
    fn is_owner(
        f: &Function,
        pointer: wgpu::naga::Handle<Expression>,
        owner: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
    ) -> bool {
        match f.expressions[pointer] {
            Expression::GlobalVariable(var) => var == owner,
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                is_owner(f, base, owner)
            }
            _ => false,
        }
    }
    #[derive(Clone, Copy, Default)]
    struct State {
        written: bool,
        synchronized: bool,
        sealed: bool,
    }
    fn merge(a: State, b: State) -> State {
        State {
            written: a.written || b.written,
            synchronized: a.synchronized && b.synchronized,
            sealed: a.sealed || b.sealed,
        }
    }
    fn visit(
        f: &Function,
        body: &wgpu::naga::Block,
        owner: wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
        mut state: State,
        counts: &mut [usize; 2],
    ) -> State {
        for statement in body {
            match statement {
                Statement::Store { pointer, .. } if is_owner(f, *pointer, owner) => {
                    assert!(
                        !state.sealed,
                        "shared factor was modified after initialization barrier"
                    );
                    state.written = true;
                    state.synchronized = false;
                    counts[0] += 1;
                }
                Statement::ControlBarrier(_) => {
                    state.synchronized = true;
                    state.sealed |= state.written;
                }
                Statement::Emit(range) => {
                    for expression in range.clone() {
                        if let Expression::Load { pointer } = f.expressions[expression]
                            && is_owner(f, pointer, owner)
                        {
                            assert!(
                                state.sealed && state.synchronized,
                                "shared factor read before initialization barrier"
                            );
                            counts[1] += 1;
                        }
                    }
                }
                Statement::Block(b) => state = visit(f, b, owner, state, counts),
                Statement::If { accept, reject, .. } => {
                    // A following read requires the barrier on BOTH paths;
                    // a writer on either path starts the initialization phase.
                    state = merge(
                        visit(f, accept, owner, state, counts),
                        visit(f, reject, owner, state, counts),
                    );
                }
                Statement::Switch { cases, .. } => {
                    let before = state;
                    let mut joined = None;
                    for case in cases {
                        let branch = visit(f, &case.body, owner, before, counts);
                        joined = Some(joined.map_or(branch, |prior| merge(prior, branch)));
                    }
                    state = joined.unwrap_or(before);
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    let after = visit(f, body, owner, state, counts);
                    let after = visit(f, continuing, owner, after, counts);
                    // Do not let a barrier in a possibly empty loop establish
                    // synchronization for a read after that loop.
                    state = merge(state, after);
                }
                _ => {}
            }
        }
        state
    }
    let f = &module.entry_points[0].function;
    let mut counts = [0, 0];
    let state = visit(f, &f.body, owner, State::default(), &mut counts);
    assert!(state.written && state.sealed && counts[0] >= 1 && counts[1] >= 2);
}

fn factor_spirv_storage_classes(module: &wgpu::naga::Module, count: usize) -> Vec<u32> {
    use std::collections::{HashMap, HashSet};
    let info = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(module)
    .unwrap();
    let words = wgpu::naga::back::spv::write_vec(
        module,
        &info,
        &wgpu::naga::back::spv::Options {
            lang_version: (1, 3),
            zero_initialize_workgroup_memory:
                wgpu::naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            ..Default::default()
        },
        Some(&wgpu::naga::back::spv::PipelineOptions {
            shader_stage: wgpu::naga::ShaderStage::Compute,
            entry_point: module.entry_points[0].name.clone(),
        }),
    )
    .unwrap();
    let mut constants = HashMap::new();
    let mut floats = HashSet::new();
    let mut arrays = HashMap::new();
    let mut pointers = HashMap::new();
    let mut variables = Vec::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let n = (words[cursor] >> 16) as usize;
        assert!(n > 0 && cursor + n <= words.len());
        let operand = &words[cursor + 1..cursor + n];
        match words[cursor] & 0xffff {
            22 if operand[1] == 32 => {
                floats.insert(operand[0]);
            }
            43 if operand.len() == 3 => {
                constants.insert(operand[1], operand[2]);
            }
            28 => {
                arrays.insert(operand[0], (operand[1], operand[2]));
            }
            32 => {
                pointers.insert(operand[0], (operand[1], operand[2]));
            }
            59 => variables.push(operand[0]),
            _ => {}
        }
        cursor += n;
    }
    assert_eq!(cursor, words.len());
    variables
        .into_iter()
        .filter_map(|pointer| {
            let &(space, array) = pointers.get(&pointer)?;
            let &(element, length) = arrays.get(&array)?;
            (floats.contains(&element) && constants.get(&length) == Some(&(count as u32)))
                .then_some(space)
        })
        .collect()
}

#[test]
fn shared_gate_up_factors_actual_shaders_storage_barriers_and_spirv() {
    use cubecl::server::ExecutionMode;
    use wgpu::naga::{AddressSpace, ArraySize, ScalarKind, TypeInner};
    for dtype in [super::super::IQ2_S, IQ3_XXS, super::super::IQ3_S, IQ4_XS] {
        let base = Geometry::new(
            [
                Projection::new(dtype, 2560, 640, 512).unwrap(),
                Projection::new(dtype, 2560, 640, 512).unwrap(),
                Projection::new(IQ4_NL, 640, 2560, 512).unwrap(),
            ],
            8,
            144,
        )
        .unwrap()
        .with_frozen_host_sources()
        .unwrap()
        .with_inactive_partition_exit()
        .unwrap();
        let (per_row, _) = factor_count_and_width(dtype);
        let table_bytes = match dtype {
            super::super::IQ2_S => 8192,
            IQ3_XXS => 1024,
            super::super::IQ3_S => 2048,
            _ => 0,
        };
        for policy in 0..3 {
            let old = match policy {
                1 => base.with_private_tables().unwrap(),
                2 => base.with_workgroup_tables().unwrap(),
                _ => base,
            };
            let new = old.with_shared_gate_up_factors().unwrap();
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let source = factor_shader(new, mode);
                let original = factor_shader(old, mode);
                validate_shader(&source, 8, true);
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                let owners: Vec<_> = module.global_variables.iter().filter(|(_, var)| {
                    if let TypeInner::Array { base, size: ArraySize::Constant(n), .. } = module.types[var.ty].inner {
                        n.get() as usize == per_row * 2 && matches!(module.types[base].inner, TypeInner::Scalar(s) if s.kind == ScalarKind::Float && s.width == 4)
                    } else { false }
                }).collect();
                assert_eq!(owners.len(), usize::from(per_row != 0));
                let shared_bytes: u32 = module
                    .global_variables
                    .iter()
                    .filter(|(_, var)| var.space == AddressSpace::WorkGroup)
                    .map(|(_, var)| match module.types[var.ty].inner {
                        TypeInner::Array {
                            size: ArraySize::Constant(n),
                            stride,
                            ..
                        } => n.get() * stride,
                        ref other => panic!("unexpected shared owner {other:?}"),
                    })
                    .sum();
                assert_eq!(
                    shared_bytes,
                    4096 + (per_row * 2 * 4) as u32 + if policy == 2 { table_bytes } else { 0 }
                );
                assert_eq!(
                    source.matches("workgroupBarrier();").count(),
                    original.matches("workgroupBarrier();").count() + usize::from(per_row != 0)
                );
                if let Some(&(owner, variable)) = owners.first() {
                    assert_eq!(variable.space, AddressSpace::WorkGroup);
                    assert!(variable.init.is_none());
                    assert_factor_barrier_and_immutability(&module, owner);
                    assert_eq!(factor_spirv_storage_classes(&module, per_row * 2), [4]);
                } else {
                    assert!(factor_spirv_storage_classes(&module, 160).is_empty());
                    assert!(factor_spirv_storage_classes(&module, 320).is_empty());
                    // Normalize only the deliberately distinct entrypoint's
                    // declaration; all IQ4_XS inference source must match.
                    let declaration = format!("fn {}(", factor_shader_name(new));
                    assert_eq!(source.matches(&declaration).count(), 1);
                    let source = source.replacen(
                        &declaration,
                        &format!("fn {}(", factor_shader_name(old)),
                        1,
                    );
                    assert_eq!(
                        canonical_declaration_order(&source),
                        canonical_declaration_order(&original)
                    );
                }
                for kind in [1, 2, 3] {
                    assert_eq!(
                        canonical_declaration_order(&shader(kind, old, mode)),
                        canonical_declaration_order(&shader(kind, new, mode))
                    );
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "shared factor was modified after initialization barrier")]
fn shared_gate_up_factor_element_store_cannot_bypass_barrier_proof() {
    let module = wgpu::naga::front::wgsl::parse_str("var<workgroup> factors: array<f32,160>; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { factors[id.x]=1.0; workgroupBarrier(); factors[id.x]=2.0; }").unwrap();
    let owner = module.global_variables.iter().next().unwrap().0;
    assert_factor_barrier_and_immutability(&module, owner);
}

#[test]
#[should_panic(expected = "shared factor read before initialization barrier")]
fn shared_gate_up_factor_read_cannot_bypass_initialization_barrier() {
    let module = wgpu::naga::front::wgsl::parse_str("var<workgroup> factors: array<f32,160>; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { factors[id.x]=1.0; factors[id.x]=factors[id.x]+1.0; workgroupBarrier(); }").unwrap();
    let owner = module.global_variables.iter().next().unwrap().0;
    assert_factor_barrier_and_immutability(&module, owner);
}

#[test]
#[should_panic(expected = "shared factor read before initialization barrier")]
fn shared_gate_up_factor_conditional_barrier_cannot_establish_dominance() {
    let module = wgpu::naga::front::wgsl::parse_str("var<workgroup> factors: array<f32,160>; @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) id:vec3<u32>) { factors[id.x]=1.0; if id.x==0u {workgroupBarrier();} factors[id.x]=factors[id.x]+1.0; }").unwrap();
    let owner = module.global_variables.iter().next().unwrap().0;
    assert_factor_barrier_and_immutability(&module, owner);
}

#[test]
fn shared_gate_up_factors_all_48_geometries_prerequisites_and_composition() {
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
        assert!(!ordinary.shared_gate_up_factors());
        assert!(ordinary.with_shared_gate_up_factors().is_err());
        let frozen = ordinary.with_frozen_host_sources().unwrap();
        assert!(frozen.with_shared_gate_up_factors().is_err());
        let base = frozen.with_inactive_partition_exit().unwrap();
        assert!(base.with_live_member_reduction().is_err());
        for tables in 0..3 {
            for down_scales in [false, true] {
                let mut old = match tables {
                    1 => base.with_private_tables().unwrap(),
                    2 => base.with_workgroup_tables().unwrap(),
                    _ => base,
                };
                if down_scales {
                    old = old.with_shared_down_scales().unwrap();
                }
                let new = old.with_shared_gate_up_factors().unwrap();
                assert!(new.shared_gate_up_factors());
                assert!(new.with_shared_gate_up_factors().is_err());
                assert!(new.with_live_member_reduction().is_err());
                let mut reverse = base.with_shared_gate_up_factors().unwrap();
                if down_scales {
                    reverse = reverse.with_shared_down_scales().unwrap();
                }
                reverse = match tables {
                    1 => reverse.with_private_tables().unwrap(),
                    2 => reverse.with_workgroup_tables().unwrap(),
                    _ => reverse,
                };
                assert_eq!(new, reverse);
                if tables == 1 {
                    assert!(new.with_workgroup_tables().is_err());
                } else if tables == 2 {
                    assert!(new.with_private_tables().is_err());
                }
                assert_eq!(old.projections(), new.projections());
                assert_eq!(old.hidden(), new.hidden());
                assert_eq!(old.ff(), new.ff());
                assert_eq!(old.tile(), new.tile());
                assert_eq!(old.capacity(), new.capacity());
                assert_eq!(old.status_words(), new.status_words());
                assert_eq!(old.member_capacity(), new.member_capacity());
                for io in [false, true] {
                    assert_eq!(old.scratch_bytes(io), new.scratch_bytes(io));
                }
                for groups in [1, 10, 30] {
                    for downward in [false, true] {
                        assert_eq!(
                            format!("{:?}", old.grid(groups, downward).unwrap()),
                            format!("{:?}", new.grid(groups, downward).unwrap())
                        );
                    }
                }
            }
        }
    }
}
