use super::*;
use cubecl::compute::KernelBuilder;
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
use cubecl::server::ExecutionMode;
use cubecl::wgpu::WgslCompiler;
use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
use wgpu::naga::{self, Expression, GatherMode, Literal, Statement};

fn shader(rows: usize, columns: usize, tail: bool, mode: ExecutionMode) -> String {
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
    let packed = b.input_array(Type::new(uint)).into();
    let input = b.input_array(Type::new(float)).into();
    let output = b.output_array(Type::new(float)).into();
    let fault = b
        .output_array(Type::new(StorageType::Atomic(ElemType::UInt(
            UIntKind::U32,
        ))))
        .into();
    let count = b.scalar(uint).into();
    let first = b.scalar(uint).into();
    let x = b.scalar(uint).into();
    let y = b.scalar(uint).into();
    let mf = b.scalar(uint).into();
    packed_q8_head_broadcast_fault::expand(
        &mut b.scope,
        packed,
        input,
        output,
        fault,
        count,
        first,
        x,
        y,
        mf,
        rows,
        columns,
        columns / 32 * 34,
        if rows >= 2560 {
            crate::mtp_resident_kernels::VOCAB
        } else {
            rows + 3
        },
        tail,
    );
    WgslCompiler::default()
        .compile(
            b.build(
                KernelSettings::default()
                    .cube_dim(cube())
                    .kernel_name("packed_q8_head_broadcast_fault"),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            uint,
        )
        .unwrap()
        .to_string()
}

pub(super) fn validate_runtime_spirv(module: &naga::Module) -> Vec<u32> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SUBGROUP,
    )
    .validate(module)
    .unwrap();
    naga::back::spv::write_vec(
        module,
        &info,
        &naga::back::spv::Options {
            zero_initialize_workgroup_memory:
                naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            ..Default::default()
        },
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: module.entry_points[0].name.clone(),
        }),
    )
    .unwrap()
}

// Both broadcasts must dominate every per-lane active/tail branch. A leader
// predicate may guard its public loads, never the subgroup collective itself.
fn assert_converged_broadcasts(function: &naga::Function) {
    fn walk(function: &naga::Function, block: &naga::Block, depth: usize, count: &mut usize) {
        for statement in block {
            match statement {
                Statement::SubgroupGather {
                    mode: GatherMode::Broadcast(index),
                    ..
                } => {
                    assert_eq!(depth, 0, "broadcast inside divergent branch");
                    assert!(matches!(
                        function.expressions[*index],
                        Expression::Literal(Literal::U32(0))
                    ));
                    *count += 1;
                }
                Statement::If { accept, reject, .. } => {
                    walk(function, accept, depth + 1, count);
                    walk(function, reject, depth + 1, count);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        walk(function, &case.body, depth + 1, count);
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    walk(function, body, depth, count);
                    walk(function, continuing, depth, count);
                }
                Statement::Block(block) => walk(function, block, depth, count),
                _ => {}
            }
        }
    }
    let mut count = 0;
    walk(function, &function.body, 0, &mut count);
    assert_eq!(count, 2);
}

pub(super) fn root_global(
    function: &naga::Function,
    pointer: naga::Handle<Expression>,
) -> Option<naga::Handle<naga::GlobalVariable>> {
    match function.expressions[pointer] {
        Expression::GlobalVariable(global) => Some(global),
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
            root_global(function, base)
        }
        _ => None,
    }
}

pub(super) fn active_condition(
    function: &naga::Function,
    expression: naga::Handle<Expression>,
    rows: usize,
) -> bool {
    use naga::BinaryOperator;
    fn literal(
        function: &naga::Function,
        expression: naga::Handle<Expression>,
        expected: usize,
    ) -> bool {
        match function.expressions[expression] {
            Expression::Literal(Literal::U32(n)) => n as usize == expected,
            Expression::As { expr, .. } => literal(function, expr, expected),
            _ => false,
        }
    }
    let row_bound = |expression| matches!(function.expressions[expression], Expression::Binary { op: BinaryOperator::Less, right: bound, .. } if literal(function, bound, rows));
    let count_one = |expression| matches!(function.expressions[expression], Expression::Binary { op: BinaryOperator::Equal, right: count, .. } if literal(function, count, 1));
    if let Expression::Binary {
        op: BinaryOperator::LogicalAnd,
        left,
        right,
    } = function.expressions[expression]
    {
        return row_bound(left) && count_one(right);
    }
    // Naga WGSL lowers short-circuit AND to a top-level phi: if row<rows,
    // store count==1, otherwise store false, then load that same local.
    let Expression::Load { pointer } = function.expressions[expression] else {
        return false;
    };
    let Expression::LocalVariable(local) = function.expressions[pointer] else {
        return false;
    };
    let stored = |block: &naga::Block| {
        if let [Statement::Store { pointer, value }] = block.as_ref()
            && matches!(function.expressions[*pointer], Expression::LocalVariable(target) if target == local)
        {
            Some(*value)
        } else {
            None
        }
    };
    fn phi_matches(block: &naga::Block, predicate: &impl Fn(&Statement) -> bool) -> bool {
        block.iter().any(|statement| {
            predicate(statement)
                || match statement {
                    Statement::If { accept, reject, .. } => {
                        phi_matches(accept, predicate) || phi_matches(reject, predicate)
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => phi_matches(body, predicate) || phi_matches(continuing, predicate),
                    Statement::Block(block) => phi_matches(block, predicate),
                    _ => false,
                }
        })
    }
    phi_matches(&function.body, &|statement| {
        if let Statement::If {
            condition,
            accept,
            reject,
        } = statement
        {
            row_bound(*condition)
                && stored(accept).is_some_and(count_one)
                && stored(reject).is_some_and(|value| {
                    matches!(
                        function.expressions[value],
                        Expression::Literal(Literal::Bool(false))
                    )
                })
        } else {
            false
        }
    })
}

// Every payload load must be dominated by the actual row<rows/count==1
// predicate and another bounds/leader guard. Generic nested ifs are insufficient.
pub(super) fn assert_weight_input_reads_guarded(module: &naga::Module, rows: usize) {
    fn walk(
        module: &naga::Module,
        function: &naga::Function,
        block: &naga::Block,
        guard: (usize, bool),
        rows: usize,
        count: &mut usize,
    ) {
        for statement in block {
            match statement {
                Statement::Emit(range) => {
                    for handle in range.clone() {
                        if let Expression::Load { pointer } = function.expressions[handle]
                            && let Some(global) = root_global(function, pointer)
                            && module.global_variables[global]
                                .binding
                                .as_ref()
                                .is_some_and(|b| b.binding < 2)
                        {
                            assert!(guard.1 && guard.0 >= 2, "unguarded weight/input load");
                            *count += 1;
                        }
                    }
                }
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    walk(
                        module,
                        function,
                        accept,
                        (
                            guard.0 + 1,
                            guard.1 || active_condition(function, *condition, rows),
                        ),
                        rows,
                        count,
                    );
                    walk(
                        module,
                        function,
                        reject,
                        (guard.0 + 1, guard.1),
                        rows,
                        count,
                    );
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    walk(module, function, body, guard, rows, count);
                    walk(module, function, continuing, guard, rows, count);
                }
                Statement::Block(block) => walk(module, function, block, guard, rows, count),
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        walk(
                            module,
                            function,
                            &case.body,
                            (guard.0 + 1, guard.1),
                            rows,
                            count,
                        );
                    }
                }
                _ => {}
            }
        }
    }
    let mut count = 0;
    let function = &module.entry_points[0].function;
    walk(
        module,
        function,
        &function.body,
        (0, false),
        rows,
        &mut count,
    );
    assert!(
        count >= 8,
        "Q8 public headers/quants and F32 inputs must be read"
    );
}

#[test]
fn q8_head_broadcast_checked_unchecked_runtime_spirv_and_uniform_collectives() {
    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
        for (rows, columns, tail) in [(16384, 2560, false), (2560, 2560, true), (7, 32, true)] {
            let source = shader(rows, columns, tail, mode);
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let spirv = validate_runtime_spirv(&module);
            let mut cursor = 5;
            let mut broadcasts = 0;
            let mut barriers = 0;
            while cursor < spirv.len() {
                let instruction = spirv[cursor];
                let words = (instruction >> 16) as usize;
                assert!(words > 0);
                match instruction & 65535 {
                    // Naga29 lowers WGSL broadcast to GroupNonUniformShuffle
                    // even for literal lane zero (pre-SPIR-V1.5 rule).
                    345 => broadcasts += 1,
                    224 => barriers += 1, // OpControlBarrier
                    _ => {}
                }
                cursor += words;
            }
            assert_eq!((broadcasts, barriers), (2, 1));
            assert_converged_broadcasts(&module.entry_points[0].function);
            assert_weight_input_reads_guarded(&module, rows);
            super::tests::assert_shared_reduction_is_only_for16(&module);
            assert!(source.contains("atomicOr") && source.contains("subgroupShuffleDown"));
            assert_eq!(source.matches("subgroupBroadcast(").count(), 2);
        }
    }
}

#[test]
fn q8_head_broadcast_proof_rejects_divergent_collective_and_hoisted_reads() {
    let source = shader(7, 2560, true, ExecutionMode::Unchecked);
    let mut module = naga::front::wgsl::parse_str(&source).unwrap();
    fn guard_first_broadcast(block: &mut naga::Block, condition: naga::Handle<Expression>) -> bool {
        for statement in block.iter_mut() {
            if matches!(
                statement,
                Statement::SubgroupGather {
                    mode: GatherMode::Broadcast(_),
                    ..
                }
            ) {
                let original = statement.clone();
                *statement = Statement::If {
                    condition,
                    accept: naga::Block::from_vec(vec![original]),
                    reject: naga::Block::new(),
                };
                return true;
            }
            if let Statement::Loop { body, .. } | Statement::Block(body) = statement
                && guard_first_broadcast(body, condition)
            {
                return true;
            }
        }
        false
    }
    let function = &mut module.entry_points[0].function;
    let condition = function.expressions.append(
        Expression::Literal(Literal::Bool(true)),
        naga::Span::UNDEFINED,
    );
    assert!(guard_first_broadcast(&mut function.body, condition));
    assert!(
        std::panic::catch_unwind(|| assert_converged_broadcasts(&module.entry_points[0].function))
            .is_err()
    );

    let mut module = naga::front::wgsl::parse_str(&source).unwrap();
    let function = &mut module.entry_points[0].function;
    let active = function
        .expressions
        .iter()
        .find_map(|(handle, _)| active_condition(function, handle, 7).then_some(handle))
        .expect("actual row/count active predicate");
    function.expressions[active] = Expression::Literal(Literal::Bool(true));
    // Same nested leader/column checks remain, but inactive rows now read.
    assert!(std::panic::catch_unwind(|| assert_weight_input_reads_guarded(&module, 7)).is_err());

    let mut module = naga::front::wgsl::parse_str(&source).unwrap();
    let function = &mut module.entry_points[0].function;
    let load = function
        .expressions
        .iter()
        .find_map(|(handle, expression)| {
            if let Expression::Load { pointer } = expression
                && let Some(global) = root_global(function, *pointer)
                && module.global_variables[global]
                    .binding
                    .as_ref()
                    .is_some_and(|b| b.binding == 0)
            {
                Some(handle)
            } else {
                None
            }
        })
        .unwrap();
    function.body.push(
        Statement::Emit(naga::Range::new_from_bounds(load, load)),
        naga::Span::UNDEFINED,
    );
    assert!(std::panic::catch_unwind(|| assert_weight_input_reads_guarded(&module, 7)).is_err());
}

pub(super) fn original_tree(weights: &[f32], input: &[f32]) -> f32 {
    let mut partial = [0.0_f32; 64];
    for (lane, accumulator) in partial.iter_mut().enumerate() {
        for column in (lane..weights.len()).step_by(64) {
            *accumulator += weights[column] * input[column];
        }
    }
    for stride in [32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            partial[lane] += partial[lane + stride];
        }
    }
    partial[0]
}

fn broadcast_tree(
    words: &[u32],
    first: usize,
    columns: usize,
    input: &[f32],
    width: usize,
    active: bool,
) -> f32 {
    let mut low = [0.0_f32; 32];
    let mut high = [0.0_f32; 32];
    let byte = |offset: usize| (words[offset / 4] >> (offset % 4 * 8)) & 255;
    for tile in 0..columns.div_ceil(64) {
        for subgroup in 0..32 / width {
            let base = first + tile * 68;
            let lower = if active {
                binary16(byte(base) | byte(base + 1) << 8)
            } else {
                0.0
            };
            let upper = if active && tile * 64 + 32 < columns {
                binary16(byte(base + 34) | byte(base + 35) << 8)
            } else {
                0.0
            };
            if active {
                for lane in subgroup * width..(subgroup + 1) * width {
                    let column = tile * 64 + lane;
                    if column < columns {
                        let quant = byte(base + 2 + lane) as u8 as i8;
                        low[lane] += (lower * f32::from(quant)) * input[column];
                    }
                    if column + 32 < columns {
                        let quant = byte(base + 36 + lane) as u8 as i8;
                        high[lane] += (upper * f32::from(quant)) * input[column + 32];
                    }
                }
            }
        }
    }
    let mut partial = [0.0_f32; 32];
    for lane in 0..32 {
        partial[lane] = low[lane] + high[lane];
    }
    // Both supported subgroup layouts have this exact original tree; existing
    // fast_matvec tests separately prove the16-wide shared/shuffle mapping.
    for stride in [16, 8, 4, 2, 1] {
        for lane in 0..stride {
            partial[lane] += partial[lane + stride];
        }
    }
    partial[0]
}

#[test]
fn q8_head_broadcast_full_original_vectors_and_lane_dots_are_bit_exact() {
    let scales = [
        0u16, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff, 0x9231,
    ];
    for columns in [32, 64, 320, 640, 2560] {
        for phase in 0..8 {
            let rows = 7;
            let row_bytes = columns / 32 * 34;
            let mut packed = vec![0u8; rows * row_bytes];
            for (block, bytes) in packed.chunks_exact_mut(34).enumerate() {
                bytes[..2].copy_from_slice(&scales[(block + phase) % scales.len()].to_le_bytes());
                for (lane, value) in bytes[2..].iter_mut().enumerate() {
                    *value = (block * 47 + lane * 13 + phase * 31) as u8;
                }
            }
            let mut original = vec![0.0; rows * columns];
            kuno_text::TensorType::Q8_0
                .decode(&packed, &mut original)
                .unwrap();
            // The first row starts in the upper half of a word. Odd32-block
            // row lengths then alternate row/header alignment throughout.
            let mut shifted = vec![0xa5, 0x5a];
            shifted.extend_from_slice(&packed);
            let words = super::super::pack_le_words(&shifted);
            let input: Vec<_> = (0..columns)
                .map(|column| {
                    (((column * 17 + phase * 19) % 83) as f32 - 41.0)
                        * 2.0_f32.powi((column % 23) as i32 - 19)
                })
                .collect();
            for row in 0..rows {
                let first = 2 + row * row_bytes;
                for column in 0..columns {
                    let base = first + column / 32 * 34;
                    let byte = |offset: usize| (words[offset / 4] >> (offset % 4 * 8)) & 255;
                    let d = binary16(byte(base) | byte(base + 1) << 8);
                    let quant = byte(base + 2 + column % 32) as u8 as i8;
                    assert_eq!(
                        (d * f32::from(quant)).to_bits(),
                        original[row * columns + column].to_bits()
                    );
                }
                let reference = original_tree(&original[row * columns..][..columns], &input);
                for width in [16, 32] {
                    assert_eq!(
                        broadcast_tree(&words, first, columns, &input, width, true).to_bits(),
                        reference.to_bits()
                    );
                }
            }
        }
    }
    // Inactive rows/counts may have a deliberately invalid packed/input span.
    // They still participate in collectives, with no storage dereference.
    for width in [16, 32] {
        assert_eq!(
            broadcast_tree(&[], usize::MAX - 4096, 2560, &[], width, false).to_bits(),
            0.0_f32.to_bits()
        );
    }
}
