// CPU/compiler-only acceptance. Included in frozen_host_staging::tests.
use super::super as grouped;
use super::*;
use cubecl::compute::KernelBuilder;
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
use cubecl::server::ExecutionMode;
use cubecl::wgpu::WgslCompiler;
use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};

fn staging_builder() -> KernelBuilder {
    let mut b = KernelBuilder::default();
    let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
    b.scope.register_type::<u32>(uint);
    b.scope.register_type::<usize>(uint);
    b.scope
        .register_type::<f32>(StorageType::Scalar(ElemType::Float(FloatKind::F32)));
    b.scope
        .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
    b.scope
        .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
    b
}

fn staging_compile(b: KernelBuilder, name: &str, lanes: u32, mode: ExecutionMode) -> String {
    WgslCompiler::default()
        .compile(
            b.build(
                KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(lanes))
                    .kernel_name(name),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap()
        .to_string()
}

fn staging_copy_shader(down: bool, words: usize, mode: ExecutionMode) -> String {
    let mut b = staging_builder();
    let uint = Type::new(StorageType::Scalar(ElemType::UInt(UIntKind::U32)));
    let atomic = Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U32)));
    let first = b.input_array(uint).into();
    let second = (!down).then(|| b.input_array(uint));
    let map = b.input_array(uint).into();
    let dst = b.output_array(uint).into();
    let dst2 = (!down).then(|| b.output_array(uint));
    let fault = b.output_array(atomic).into();
    let scalar = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
    let scalars: Vec<_> = (0..3).map(|_| b.scalar(scalar)).collect();
    let name = if down {
        stage_down::expand(
            &mut b.scope,
            first,
            map,
            dst,
            fault,
            scalars[0].clone().into(),
            scalars[1].clone().into(),
            scalars[2].clone().into(),
            words,
        );
        "stage_down"
    } else {
        stage_gate_up::expand(
            &mut b.scope,
            first,
            second.unwrap().into(),
            map,
            dst,
            dst2.unwrap().into(),
            fault,
            scalars[0].clone().into(),
            scalars[1].clone().into(),
            scalars[2].clone().into(),
            words,
        );
        "stage_gate_up"
    };
    staging_compile(b, name, 256, mode)
}

fn staging_geometry(layer: usize) -> grouped::Geometry {
    let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
    grouped::Geometry::new(
        [
            grouped::Projection::new(gate, 2560, 640, 512).unwrap(),
            grouped::Projection::new(up, 2560, 640, 512).unwrap(),
            grouped::Projection::new(down, 640, 2560, 512).unwrap(),
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
fn frozen_staging_all_48_shapes_charges_and_policy_scope() {
    for layer in 0..48 {
        let base = staging_geometry(layer);
        for two in [false, true] {
            for four in [false, true] {
                let mut g = base;
                if two {
                    g = g.with_two_row_gate_up().unwrap();
                }
                if four {
                    g = g.with_four_row_down().unwrap();
                }
                let shape = Shape::new(g).unwrap();
                let [gate, up, down] = g.projections();
                assert_eq!(gate, up);
                assert_eq!(shape.gate_up_words(), gate.expert_bytes() / 4);
                assert_eq!(shape.down_words(), down.expert_bytes() / 4);
                assert_eq!(shape.stage_words(), [8_388_608; 3]);
                assert_eq!(
                    shape.payload_words(),
                    [
                        30 * gate.expert_bytes() / 4,
                        30 * up.expert_bytes() / 4,
                        30 * down.expert_bytes() / 4
                    ]
                );
                for words in shape.stage_words() {
                    assert!(words * 4 <= 32 * 1024 * 1024);
                }
                for role in [false, true] {
                    let words = if role {
                        shape.down_words()
                    } else {
                        shape.gate_up_words()
                    };
                    assert!(matches!(shape.copy_grid(role), CubeCount::Static(x, 30, 1)
                        if x == words.div_ceil(1024) as u32));
                }
            }
        }
        assert!(Shape::new(base.with_private_tables().unwrap()).is_err());
        assert!(Shape::new(base.with_workgroup_tables().unwrap()).is_err());
        assert!(Shape::new(base.with_shared_gate_up_factors().unwrap()).is_err());
        assert!(Shape::new(base.with_shared_down_scales().unwrap()).is_err());
        let ordinary = grouped::Geometry::new(base.projections(), 8, 144).unwrap();
        assert!(Shape::new(ordinary).is_err());
        assert!(Shape::new(ordinary.with_frozen_host_sources().unwrap()).is_err());
    }
    assert_eq!(3 * 32 * 1024 * 1024, 100_663_296);
}

fn staging_map() -> Vec<u32> {
    let mut map = vec![0; grouped::MAP_WORDS];
    map[..4].copy_from_slice(&[3, 0, 3, 30]);
    for group in 0..30 {
        map[grouped::SLOTS + group] = match group {
            1 => 144,
            14 => 288,
            29 => 511,
            _ => group as u32,
        };
        map[grouped::LENGTHS + group] = 1;
        map[grouped::MEMBERS + group * 8] = group as u32;
    }
    assert!((0..30).all(|g| (0..g).all(|h| map[grouped::SLOTS + g] != map[grouped::SLOTS + h])));
    map
}

fn staging_active(map: &[u32], first: usize, group: usize, start: usize, slots: usize) -> bool {
    if !matches!((start, slots), (144, 144) | (288, 144) | (432, 80))
        || first > map.len()
        || grouped::MAP_WORDS > map.len() - first
    {
        return false;
    }
    let m = &map[first..];
    let count = m[0] as usize;
    if !(1..=3).contains(&count)
        || m[1] != 0
        || m[2] as usize != count
        || !(1..=30).contains(&m[3])
        || group >= 30
        || group >= m[3] as usize
    {
        return false;
    }
    let slot = m[grouped::SLOTS + group] as usize;
    let length = m[grouped::LENGTHS + group] as usize;
    slot >= start
        && slot - start < slots
        && (1..=count).contains(&length)
        && (0..length)
            .all(|member| (m[grouped::MEMBERS + group * 8 + member] as usize) < count * 10)
}

#[test]
fn frozen_staging_copy_coverage_shards_tails_and_malformed_maps() {
    let map = staging_map();
    for words in [1_usize, 1023, 1024, 1025, 217_600, 230_400] {
        let logical = 30 * words;
        let mut stage = vec![u32::MAX; logical + 37];
        let mut writes = vec![0_u8; logical];
        for (start, slots) in [(144, 144), (288, 144), (432, 80)] {
            for group in 0..30 {
                if !staging_active(&map, 0, group, start, slots) {
                    continue;
                }
                let slot = map[grouped::SLOTS + group] as usize;
                for cube in 0..words.div_ceil(1024) {
                    for lane in 0..256 {
                        for stripe in 0..4 {
                            let i = cube * 1024 + lane + stripe * 256;
                            if i < words {
                                let source = (slot - start) * words + i;
                                let dst = group * words + i;
                                assert!(source < slots * words && dst < logical);
                                stage[dst] = (slot as u32)
                                    .wrapping_mul(0x9e37_79b9)
                                    .wrapping_add(i as u32);
                                writes[dst] += 1;
                            }
                        }
                    }
                }
            }
        }
        for group in 0..30 {
            let slot = map[grouped::SLOTS + group];
            for i in 0..words {
                let index = group * words + i;
                assert_eq!(writes[index], u8::from(slot >= 144));
                assert_eq!(
                    stage[index],
                    if slot >= 144 {
                        slot.wrapping_mul(0x9e37_79b9).wrapping_add(i as u32)
                    } else {
                        u32::MAX
                    }
                );
            }
        }
        assert!(stage[logical..].iter().all(|&v| v == u32::MAX));
    }
    for (field, value) in [
        (0, 0),
        (0, 4),
        (1, 1),
        (2, 2),
        (3, 0),
        (3, 31),
        (grouped::LENGTHS + 1, 0),
        (grouped::LENGTHS + 1, 4),
        (grouped::MEMBERS + 8, 30),
    ] {
        let mut bad = map.clone();
        bad[field] = value;
        assert!(!staging_active(&bad, 0, 1, 144, 144));
    }
    for (start, count) in [(0, 72), (143, 144), (144, 143), (432, 81)] {
        assert!(!staging_active(&map, 0, 1, start, count));
    }
    assert!(!staging_active(&map, usize::MAX, 1, 144, 144));
    assert!(!staging_active(&map[..map.len() - 1], 0, 1, 144, 144));
    assert!(!staging_active(&map, 0, 30, 144, 144));
    let mut shifted = vec![0; 13];
    shifted.extend_from_slice(&map);
    assert!(staging_active(&shifted, 13, 1, 144, 144));
}

fn staging_tree(weights: &[f32], input: &[f32], physical: usize) -> f32 {
    let mut lanes = vec![0_f32; physical];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        let mut virtuals = [0.; 4];
        for (v, partial) in virtuals.iter_mut().enumerate().take(64 / physical) {
            for column in (lane + v * physical..weights.len()).step_by(64) {
                *partial += weights[column] * input[column];
            }
        }
        *sum = match physical {
            64 => virtuals[0],
            32 => virtuals[0] + virtuals[1],
            16 => (virtuals[0] + virtuals[2]) + (virtuals[1] + virtuals[3]),
            _ => unreachable!(),
        };
    }
    let mut stride = physical / 2;
    while stride > 0 {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
        stride /= 2;
    }
    lanes[0]
}

#[test]
fn frozen_staging_original_c_packed_words_weights_and_row_trees() {
    use crate::quant_matvec::{IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, iq_fixtures};
    for dtype in [IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL, Q2_0] {
        let width = if matches!(dtype, IQ4_NL | Q2_0) {
            640
        } else {
            2560
        };
        for phase in [0, 1, 3] {
            let source = iq_fixtures::packed(dtype, 4, width, phase);
            let words = source.len() / 4;
            assert!(source.len().is_multiple_of(4));
            let mut stage = vec![0_u32; 30 * words];
            let group = [1, 14, 29][phase % 3];
            for cube in 0..words.div_ceil(1024) {
                for lane in 0..256 {
                    for stripe in 0..4 {
                        let i = cube * 1024 + lane + stripe * 256;
                        if i < words {
                            stage[group * words + i] =
                                u32::from_le_bytes(source[i * 4..][..4].try_into().unwrap());
                        }
                    }
                }
            }
            let copied: Vec<u8> = stage[group * words..][..words]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            assert_eq!(copied, source);
            let expected = iq_fixtures::oracle(dtype, 4, width, phase);
            let mut decoded = vec![0.; expected.len()];
            kuno_text::TensorType::from_gguf_id(dtype)
                .unwrap()
                .decode(&copied, &mut decoded)
                .unwrap();
            assert!(
                decoded
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            for member in 0..3 {
                let input: Vec<_> = (0..width)
                    .map(|c| ((c * 7 + member * 5 + phase * 3) % 31) as f32 / 32. - 0.5)
                    .collect();
                for row in 0..4 {
                    let actual = &decoded[row * width..][..width];
                    let old = staging_tree(&expected[row * width..][..width], &input, 64);
                    assert_eq!(staging_tree(actual, &input, 64).to_bits(), old.to_bits());
                    assert_eq!(
                        staging_tree(actual, &input, if width == 2560 { 32 } else { 16 }).to_bits(),
                        old.to_bits()
                    );
                }
            }
        }
    }
}

fn staging_module(source: &str) -> wgpu::naga::Module {
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    module
}

fn staging_spirv(module: &wgpu::naga::Module) -> Vec<u32> {
    let info = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(module)
    .unwrap();
    wgpu::naga::back::spv::write_vec(
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
    .unwrap()
}

// In the unchecked runtime shader, every copy store must load the matching
// source role at the same intra-expert word under word<expert_words.
fn staging_copy_store_proof(module: &wgpu::naga::Module, down: bool, expert_words: usize) {
    use wgpu::naga::{BinaryOperator, Expression, Literal, Statement};
    let f = &module.entry_points[0].function;
    fn global(f: &wgpu::naga::Function, h: wgpu::naga::Handle<Expression>) -> bool {
        match f.expressions[h] {
            Expression::GlobalVariable(_) => true,
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                global(f, base)
            }
            _ => false,
        }
    }
    fn indexed(
        f: &wgpu::naga::Function,
        h: wgpu::naga::Handle<Expression>,
    ) -> (
        wgpu::naga::Handle<wgpu::naga::GlobalVariable>,
        wgpu::naga::Handle<Expression>,
    ) {
        let Expression::Access { base, index } = f.expressions[h] else {
            panic!("indexed packed access");
        };
        let Expression::GlobalVariable(global) = f.expressions[base] else {
            panic!("packed global owner");
        };
        (global, index)
    }
    fn offset(
        f: &wgpu::naga::Function,
        h: wgpu::naga::Handle<Expression>,
        words: usize,
    ) -> wgpu::naga::Handle<Expression> {
        let Expression::Binary {
            op: BinaryOperator::Add,
            left,
            right,
        } = f.expressions[h]
        else {
            panic!("packed access must add intra-expert word");
        };
        let Expression::Binary {
            op: BinaryOperator::Multiply,
            right: stride,
            ..
        } = f.expressions[left]
        else {
            panic!("packed access must retain expert stride");
        };
        assert!(
            matches!(f.expressions[stride], Expression::Literal(Literal::U32(n)) if n as usize == words)
        );
        right
    }
    fn visit(
        module: &wgpu::naga::Module,
        f: &wgpu::naga::Function,
        block: &wgpu::naga::Block,
        down: bool,
        words: usize,
        guards: &mut Vec<wgpu::naga::Handle<Expression>>,
        stores: &mut [usize; 2],
    ) {
        for statement in block {
            match statement {
                Statement::Store { pointer, value } => {
                    if !global(f, *pointer) {
                        continue;
                    }
                    let (dst, dst_index) = indexed(f, *pointer);
                    let dst_binding = module.global_variables[dst]
                        .binding
                        .as_ref()
                        .unwrap()
                        .binding;
                    let source_role = if down {
                        assert_eq!(dst_binding, 2);
                        0
                    } else {
                        assert!((3..=4).contains(&dst_binding));
                        (dst_binding - 3) as usize
                    };
                    let Expression::Load { pointer: src } = f.expressions[*value] else {
                        panic!("copy cannot transform packed word");
                    };
                    let (src, src_index) = indexed(f, src);
                    assert_eq!(
                        module.global_variables[src]
                            .binding
                            .as_ref()
                            .unwrap()
                            .binding as usize,
                        source_role
                    );
                    let word = offset(f, dst_index, words);
                    assert_eq!(word, offset(f, src_index, words));
                    assert!(guards.iter().any(|&g| matches!(f.expressions[g],
                        Expression::Binary { op: BinaryOperator::Less, left, right }
                            if left == word && matches!(f.expressions[right], Expression::Literal(Literal::U32(n)) if n as usize == words))),
                        "copy store must retain its tail guard");
                    stores[source_role] += 1;
                }
                Statement::Block(body) => visit(module, f, body, down, words, guards, stores),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    guards.push(*condition);
                    visit(module, f, accept, down, words, guards, stores);
                    guards.pop();
                    visit(module, f, reject, down, words, guards, stores);
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    visit(module, f, body, down, words, guards, stores);
                    visit(module, f, continuing, down, words, guards, stores);
                }
                _ => {}
            }
        }
    }
    let mut stores = [0; 2];
    visit(
        module,
        f,
        &f.body,
        down,
        expert_words,
        &mut Vec::new(),
        &mut stores,
    );
    assert_eq!(stores, if down { [4, 0] } else { [4, 4] });
}

#[test]
fn frozen_staging_actual_copy_shaders_checked_unchecked_integer_only() {
    use wgpu::naga::{AddressSpace, ScalarKind, TypeInner};
    for down in [false, true] {
        for words in [1025, if down { 230_400 } else { 217_600 }] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let module = staging_module(&staging_copy_shader(down, words, mode));
                if mode == ExecutionMode::Unchecked {
                    staging_copy_store_proof(&module, down, words);
                }
                assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
                assert_eq!(
                    module
                        .global_variables
                        .iter()
                        .filter(|(_, v)| matches!(v.space, AddressSpace::Storage { .. }))
                        .count(),
                    // Original packed roles plus the existing runtime INFO.
                    if down { 5 } else { 7 }
                );
                assert!(
                    !module
                        .global_variables
                        .iter()
                        .any(|(_, v)| v.space == AddressSpace::WorkGroup)
                );
                assert!(!module.types.iter().any(|(_, t)| matches!(t.inner,
                    TypeInner::Scalar(s) if s.kind == ScalarKind::Float)));
                let words = staging_spirv(&module);
                let mut cursor = 5;
                let mut barriers = 0;
                while cursor < words.len() {
                    let n = (words[cursor] >> 16) as usize;
                    assert!(n > 0 && cursor + n <= words.len());
                    barriers += usize::from(words[cursor] & 0xffff == 224);
                    cursor += n;
                }
                assert_eq!(barriers, 0);
                assert_eq!(cursor, words.len());
            }
        }
    }
}

fn staging_consumer_shader(
    g: grouped::Geometry,
    down: bool,
    staged: bool,
    mode: ExecutionMode,
) -> String {
    let mut b = staging_builder();
    let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
    let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
    let name = if down {
        let weights = b.input_array(Type::new(uint)).into();
        let act = b.input_array(Type::new(float)).into();
        let map = b.input_array(Type::new(uint)).into();
        let output = b.output_array(Type::new(float)).into();
        let status = b.output_array(Type::new(uint)).into();
        let scalars: Vec<_> = (0..8).map(|_| b.scalar(uint)).collect();
        let expand = if staged {
            if g.four_row_down() {
                grouped::grouped_down_frozen_staged_four_rows::expand
            } else {
                grouped::grouped_down_frozen_staged::expand
            }
        } else if g.four_row_down() {
            grouped::grouped_down_frozen_four_rows::expand
        } else {
            grouped::grouped_down_frozen::expand
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
            (true, true) => "grouped_down_frozen_staged_four_rows",
            (true, false) => "grouped_down_frozen_staged",
            (false, true) => "grouped_down_frozen_four_rows",
            _ => "grouped_down_frozen",
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
            if g.two_row_gate_up() {
                grouped::grouped_gate_up_frozen_staged_two_rows::expand
            } else {
                grouped::grouped_gate_up_frozen_staged::expand
            }
        } else if g.two_row_gate_up() {
            grouped::grouped_gate_up_frozen_two_rows::expand
        } else {
            grouped::grouped_gate_up_frozen::expand
        };
        let [ga, up_p, _] = g.projections();
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
            ga.dtype(),
            up_p.dtype(),
            ga.row_bytes(),
            up_p.row_bytes(),
            g.tile(),
            false,
        );
        match (staged, g.two_row_gate_up()) {
            (true, true) => "grouped_gate_up_frozen_staged_two_rows",
            (true, false) => "grouped_gate_up_frozen_staged",
            (false, true) => "grouped_gate_up_frozen_two_rows",
            _ => "grouped_gate_up_frozen",
        }
    };
    staging_compile(b, name, 64, mode)
}

fn staging_float_ops(module: &wgpu::naga::Module) -> Vec<String> {
    use wgpu::naga::{Expression, Literal, ScalarKind, TypeInner};
    let f = &module.entry_points[0].function;
    fn ty_float(m: &wgpu::naga::Module, t: wgpu::naga::Handle<wgpu::naga::Type>) -> bool {
        match m.types[t].inner {
            TypeInner::Scalar(s) | TypeInner::Vector { scalar: s, .. } => {
                s.kind == ScalarKind::Float
            }
            TypeInner::Array { base, .. } | TypeInner::Pointer { base, .. } => ty_float(m, base),
            _ => false,
        }
    }
    fn is_float(
        m: &wgpu::naga::Module,
        f: &wgpu::naga::Function,
        h: wgpu::naga::Handle<Expression>,
    ) -> bool {
        match f.expressions[h] {
            Expression::Literal(Literal::F32(_)) => true,
            Expression::GlobalVariable(v) => ty_float(m, m.global_variables[v].ty),
            Expression::LocalVariable(v) => ty_float(m, f.local_variables[v].ty),
            Expression::ZeroValue(t) => ty_float(m, t),
            Expression::As { kind, .. } => kind == ScalarKind::Float,
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                is_float(m, f, base)
            }
            Expression::Load { pointer } => is_float(m, f, pointer),
            Expression::Binary { left, right, .. } => is_float(m, f, left) || is_float(m, f, right),
            Expression::Unary { expr, .. } => is_float(m, f, expr),
            Expression::Math { arg, .. } => is_float(m, f, arg),
            Expression::Select { accept, reject, .. } => {
                is_float(m, f, accept) || is_float(m, f, reject)
            }
            _ => false,
        }
    }
    let mut result = Vec::new();
    for (_, e) in f.expressions.iter() {
        match *e {
            Expression::Binary { op, left, right }
                if is_float(module, f, left) || is_float(module, f, right) =>
            {
                result.push(format!("binary:{op:?}"))
            }
            Expression::Unary { op, expr } if is_float(module, f, expr) => {
                result.push(format!("unary:{op:?}"))
            }
            Expression::Math { fun, arg, .. } if is_float(module, f, arg) => {
                result.push(format!("math:{fun:?}"))
            }
            _ => {}
        }
    }
    result.sort_unstable();
    result
}

#[test]
fn frozen_staging_actual_consumers_preserve_float_arithmetic_and_storage() {
    use wgpu::naga::AddressSpace;
    let mut seen = std::collections::HashSet::new();
    for layer in 0..48 {
        let base = staging_geometry(layer);
        for down in [false, true] {
            let dtype = base.projections()[if down { 2 } else { 0 }].dtype();
            if !seen.insert((down, dtype)) {
                continue;
            }
            for tiled in [false, true] {
                let g = if !tiled {
                    base
                } else if down {
                    base.with_four_row_down().unwrap()
                } else {
                    base.with_two_row_gate_up().unwrap()
                };
                for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                    let old = staging_module(&staging_consumer_shader(g, down, false, mode));
                    let new = staging_module(&staging_consumer_shader(g, down, true, mode));
                    assert_eq!(new.entry_points[0].workgroup_size, [64, 1, 1]);
                    assert_eq!(staging_float_ops(&new), staging_float_ops(&old));
                    let storage = |m: &wgpu::naga::Module| {
                        m.global_variables
                            .iter()
                            .filter(|(_, v)| matches!(v.space, AddressSpace::Storage { .. }))
                            .count()
                    };
                    assert_eq!(storage(&new), storage(&old));
                    let shared = |m: &wgpu::naga::Module| {
                        m.global_variables
                            .iter()
                            .filter(|(_, v)| v.space == AddressSpace::WorkGroup)
                            .map(|(_, v)| match m.types[v.ty].inner {
                                wgpu::naga::TypeInner::Array { base, size, stride } => {
                                    format!("{size:?}:{stride}:{:?}", m.types[base].inner)
                                }
                                ref other => format!("{other:?}"),
                            })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(shared(&new), shared(&old));
                    assert!(staging_spirv(&new).len() > 5);
                }
            }
        }
    }
    assert_eq!(seen.len(), 6);
}
