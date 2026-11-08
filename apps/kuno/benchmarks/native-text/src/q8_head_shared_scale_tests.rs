use super::q8_head_broadcast_tests::{
    active_condition, assert_weight_input_reads_guarded, original_tree, root_global,
    validate_runtime_spirv,
};
use super::*;
use cubecl::compute::KernelBuilder;
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
use cubecl::server::ExecutionMode;
use cubecl::wgpu::WgslCompiler;
use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
use wgpu::naga::{
    self, AddressSpace, ArraySize, BinaryOperator, Expression, Literal, Statement, TypeInner,
};

fn shader(rows: usize, tail: bool, mode: ExecutionMode, shared: bool) -> String {
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
    if shared {
        packed_q8_head_shared_scales_fault::expand(
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
            2560,
            2720,
            crate::mtp_resident_kernels::VOCAB,
            tail,
        );
    } else {
        packed_single_fault::expand(
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
            2560,
            super::super::Q8_0,
            2720,
            crate::mtp_resident_kernels::VOCAB,
            tail,
        );
    }
    WgslCompiler::default()
        .compile(
            b.build(
                KernelSettings::default()
                    .cube_dim(cube())
                    .kernel_name(if shared {
                        "packed_q8_head_shared_scales_fault"
                    } else {
                        "packed_single_fault"
                    }),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            uint,
        )
        .unwrap()
        .to_string()
}

fn scale_owner(module: &naga::Module) -> naga::Handle<naga::GlobalVariable> {
    let owners: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(handle, var)| {
            (var.space == AddressSpace::WorkGroup
                && matches!(module.types[var.ty].inner,
            TypeInner::Array { size: ArraySize::Constant(n), .. } if n.get() == 320))
            .then_some(handle)
        })
        .collect();
    assert_eq!(owners.len(), 1);
    owners[0]
}

fn assert_initialization_and_immutability(module: &naga::Module, rows: usize) {
    scale_owner(module);
    let function = &module.entry_points[0].function;
    let barrier = function
        .body
        .iter()
        .position(|s| matches!(s, Statement::ControlBarrier(_)))
        .expect("unconditional scale fence before any dot work");
    fn visit(
        module: &naga::Module,
        block: &naga::Block,
        before: bool,
        stores: &mut usize,
        reads: &mut usize,
    ) {
        let f = &module.entry_points[0].function;
        let owner = scale_owner(module);
        for statement in block {
            match statement {
                Statement::Store { pointer, .. } if root_global(f, *pointer) == Some(owner) => {
                    assert!(before, "public scale modified after initializer fence");
                    *stores += 1;
                }
                Statement::Emit(range) => {
                    for handle in range.clone() {
                        if let Expression::Load { pointer } = f.expressions[handle]
                            && root_global(f, pointer) == Some(owner)
                        {
                            assert!(!before, "scale read before uniform initialization fence");
                            *reads += 1;
                        }
                    }
                }
                Statement::If { accept, reject, .. } => {
                    visit(module, accept, before, stores, reads);
                    visit(module, reject, before, stores, reads);
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    visit(module, body, before, stores, reads);
                    visit(module, continuing, before, stores, reads);
                }
                Statement::Block(block) => visit(module, block, before, stores, reads),
                _ => {}
            }
        }
    }
    let mut stores = 0;
    let mut reads = 0;
    for (index, statement) in function.body.iter().enumerate() {
        visit(
            module,
            &naga::Block::from_vec(vec![statement.clone()]),
            index < barrier,
            &mut stores,
            &mut reads,
        );
    }
    assert_eq!((stores, reads), (3, 2));
    assert_weight_input_reads_guarded(module, rows);
}

fn float_arithmetic(module: &naga::Module) -> [usize; 5] {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SUBGROUP,
    )
    .validate(module)
    .unwrap();
    let function = &module.entry_points[0].function;
    let types = info.get_entry_point(0);
    let mut counts = [0; 5];
    for (handle, expression) in function.expressions.iter() {
        if let Expression::Binary { op, .. } = expression
            && matches!(
                types[handle].ty.inner_with(&module.types),
                TypeInner::Scalar(naga::Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 4
                })
            )
        {
            let index = match op {
                BinaryOperator::Add => Some(0),
                BinaryOperator::Subtract => Some(1),
                BinaryOperator::Multiply => Some(2),
                BinaryOperator::Divide => Some(3),
                BinaryOperator::Modulo => Some(4),
                _ => None,
            };
            if let Some(index) = index {
                counts[index] += 1;
            }
        }
    }
    counts
}

#[test]
fn q8_head_shared_scales_checked_unchecked_runtime_initialization_and_original_body() {
    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
        for (rows, tail) in [(16384, false), (2560, true), (7, true)] {
            let source = shader(rows, tail, mode, true);
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            assert_initialization_and_immutability(&module, rows);
            let spirv = validate_runtime_spirv(&module);
            let mut cursor = 5;
            let mut barriers = 0;
            while cursor < spirv.len() {
                let instruction = spirv[cursor];
                let count = (instruction >> 16) as usize;
                assert!(count > 0);
                if instruction & 65535 == 224 {
                    barriers += 1;
                }
                cursor += count;
            }
            assert_eq!(
                barriers, 2,
                "one initializer fence plus original16-wide reduction"
            );
            let slm: u32 = module
                .global_variables
                .iter()
                .filter_map(|(_, v)| {
                    if v.space == AddressSpace::WorkGroup
                        && let TypeInner::Array {
                            size: ArraySize::Constant(n),
                            stride: 4,
                            ..
                        } = module.types[v.ty].inner
                    {
                        Some(n.get() * 4)
                    } else {
                        None
                    }
                })
                .sum();
            assert_eq!(slm, 1792);
            let old_source = shader(rows, tail, mode, false);
            let original = naga::front::wgsl::parse_str(&old_source).unwrap();
            assert_eq!(float_arithmetic(&module), float_arithmetic(&original));
            assert_eq!(
                source.matches("atomicOr(").count(),
                old_source.matches("atomicOr(").count()
            );
            assert!(!source.contains("subgroupBroadcast("));
            let mut after_init = module.clone();
            let function = &mut after_init.entry_points[0].function;
            let barrier = function
                .body
                .iter()
                .position(|s| matches!(s, Statement::ControlBarrier(_)))
                .unwrap();
            function.body =
                naga::Block::from_vec(function.body.iter().skip(barrier + 1).cloned().collect());
            super::tests::assert_shared_reduction_is_only_for16(&after_init);
        }
    }
}

#[test]
fn q8_head_shared_scales_proof_rejects_unfenced_reads_and_missing_active_guards() {
    let source = shader(7, true, ExecutionMode::Unchecked, true);
    let mut module = naga::front::wgsl::parse_str(&source).unwrap();
    let owner = scale_owner(&module);
    let function = &mut module.entry_points[0].function;
    let read = function
        .expressions
        .iter()
        .find_map(|(handle, expression)| {
            if let Expression::Load { pointer } = expression
                && root_global(function, *pointer) == Some(owner)
            {
                Some(handle)
            } else {
                None
            }
        })
        .unwrap();
    let mut body = vec![Statement::Emit(naga::Range::new_from_bounds(read, read))];
    body.extend(function.body.iter().cloned());
    function.body = naga::Block::from_vec(body);
    assert!(
        std::panic::catch_unwind(|| assert_initialization_and_immutability(&module, 7)).is_err()
    );
    let mut module = naga::front::wgsl::parse_str(&source).unwrap();
    let function = &mut module.entry_points[0].function;
    let guards: Vec<_> = function
        .expressions
        .iter()
        .filter_map(|(handle, _)| active_condition(function, handle, 7).then_some(handle))
        .collect();
    assert!(
        guards.len() >= 4,
        "three source-row predicates plus active dot predicate"
    );
    for guard in guards {
        function.expressions[guard] = Expression::Literal(Literal::Bool(true));
    }
    assert!(std::panic::catch_unwind(|| assert_weight_input_reads_guarded(&module, 7)).is_err());
}

#[test]
fn q8_head_shared_scales_full_vectors_exact_words_and_dot_bits() {
    let columns = 2560;
    let rows = 7;
    let row_bytes = 2720;
    for phase in 0..8 {
        let halves = [
            0u16, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff, 0x9231,
        ];
        let mut packed = vec![0u8; rows * row_bytes];
        for (block, bytes) in packed.chunks_exact_mut(34).enumerate() {
            bytes[..2].copy_from_slice(&halves[(block + phase) % halves.len()].to_le_bytes());
            for (lane, byte) in bytes[2..].iter_mut().enumerate() {
                *byte = (block * 31 + lane * 17 + phase * 43) as u8;
            }
        }
        let mut original = vec![0.; rows * columns];
        kuno_text::TensorType::Q8_0
            .decode(&packed, &mut original)
            .unwrap();
        let mut shifted = vec![0xa5, 0x5a];
        shifted.extend_from_slice(&packed);
        let words = super::super::pack_le_words(&shifted);
        let byte = |offset: usize| (words[offset / 4] >> (offset % 4 * 8)) & 255;
        let input: Vec<_> = (0..columns)
            .map(|column| {
                (((column * 23 + phase * 13) % 89) as f32 - 44.)
                    * 2.0_f32.powi((column % 23) as i32 - 19)
            })
            .collect();
        for group in 0..3 {
            for count in [0, 1, 2] {
                let mut scales = [f32::NAN; 320];
                let mut writes = [0; 320];
                for physical_lane in 0..128 {
                    for slice in 0..3 {
                        let index = physical_lane + slice * 128;
                        if index < 320 {
                            let row = group * 4 + index / 80;
                            scales[index] = if row < rows && count == 1 {
                                let base = 2 + row * row_bytes + index % 80 * 34;
                                binary16(byte(base) | byte(base + 1) << 8)
                            } else {
                                0.
                            };
                            writes[index] += 1;
                        }
                    }
                }
                assert!(writes.into_iter().all(|n| n == 1));
                for slot in 0..4 {
                    let row = group * 4 + slot;
                    let active = row < rows && count == 1;
                    let mut low = [0.0_f32; 32];
                    let mut high = [0.0_f32; 32];
                    if active {
                        for tile in 0..40 {
                            for lane in 0..32 {
                                let base = 2 + row * row_bytes + tile * 68;
                                let lower = scales[slot * 80 + tile * 2]
                                    * f32::from(byte(base + 2 + lane) as u8 as i8);
                                let upper = scales[slot * 80 + tile * 2 + 1]
                                    * f32::from(byte(base + 36 + lane) as u8 as i8);
                                let column = tile * 64 + lane;
                                assert_eq!(
                                    lower.to_bits(),
                                    original[row * columns + column].to_bits()
                                );
                                assert_eq!(
                                    upper.to_bits(),
                                    original[row * columns + column + 32].to_bits()
                                );
                                low[lane] += lower * input[column];
                                high[lane] += upper * input[column + 32];
                            }
                        }
                    }
                    let mut partial = [0.0_f32; 32];
                    for lane in 0..32 {
                        partial[lane] = low[lane] + high[lane];
                    }
                    for stride in [16, 8, 4, 2, 1] {
                        for lane in 0..stride {
                            partial[lane] += partial[lane + stride];
                        }
                    }
                    let reference = if active {
                        original_tree(&original[row * columns..][..columns], &input)
                    } else {
                        0.
                    };
                    assert_eq!(partial[0].to_bits(), reference.to_bits());
                }
            }
        }
    }
}
