// Included in gdn_chunk_probe::tests. CPU/compiler checks only: no client/device.

fn three_streaming_inputs(f: &Fixture) -> [&[f32]; 5] {
    [&f.query, &f.key, &f.value, &f.controls, &f.old]
}

fn three_streaming_assert_float_bits(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    assert!(
        left.iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

fn three_streaming_assert_exact(
    left: &(Vec<f32>, Vec<f32>, Vec<u32>),
    right: &(Vec<f32>, Vec<f32>, Vec<u32>),
) {
    three_streaming_assert_float_bits(&left.0, &right.0);
    three_streaming_assert_float_bits(&left.1, &right.1);
    assert_eq!(left.2, right.2);
}

fn three_streaming_preserves_gaps(
    shape: GdnShape,
    s: ChunkSpans,
    result: &(Vec<f32>, Vec<f32>, Vec<u32>),
) {
    let l = GdnLayout::new(shape).unwrap();
    untouched_floats(&vec![GAP; s.next.words], &result.0, |i| {
        active_state(shape, s.next, i)
    })
    .unwrap();
    untouched_floats(&vec![GAP; s.output.words], &result.1, |i| {
        active_token(s.output, l.value_elements, s.count, i)
    })
    .unwrap();
    for (i, &word) in result.2.iter().enumerate() {
        if !active_token(s.status, l.value_elements, s.count, i) {
            assert_eq!(word, if i < s.status.words { STATUS_GAP } else { 0 });
        }
    }
}

#[test]
fn three_streaming_policy_retains_default_and_full_max8_owners() {
    assert_eq!(RecurrencePolicy::default(), RecurrencePolicy::Chunk);
    assert_eq!(RecurrencePolicy::ThreeStreaming.label(), "three-streaming");
    for shape in [GdnShape::new(7, 2, 6).unwrap(), GdnShape::qwen38()] {
        let l = GdnLayout::new(shape).unwrap();
        let full = fixture_spans(shape, 8).unwrap();
        for count in [0, 1, 2, 3, 4, 8, 9, usize::MAX] {
            assert_eq!(
                RecurrencePolicy::ThreeStreaming
                    .validate_count(count)
                    .is_ok(),
                (1..=3).contains(&count)
            );
            assert_eq!(
                RecurrencePolicy::SingleStreaming
                    .validate_count(count)
                    .is_ok(),
                count == 1
            );
            assert_eq!(
                RecurrencePolicy::Chunk.validate_count(count).is_ok(),
                (1..=8).contains(&count)
            );
            if (1..=8).contains(&count) {
                let s = fixture_spans(shape, count).unwrap();
                assert_eq!(s.words(), full.words());
                assert_eq!(owned_plan(l, s).unwrap(), owned_plan(l, full).unwrap());
            }
        }
    }
    let shape = GdnShape::new(7, 2, 6).unwrap();
    for count in [4, 8] {
        let f = fixture(shape, count).unwrap();
        assert!(
            chunk::three_streaming_lane_reference(shape, f.spans, three_streaming_inputs(&f))
                .is_err()
        );
    }
    let source = include_str!("gdn_chunk.rs");
    let signature = source
        .split("pub fn gdn_three_streaming(")
        .nth(1)
        .unwrap()
        .split(") {")
        .next()
        .unwrap();
    let runtime = signature.split("#[comptime]").next().unwrap();
    for name in [
        "count",
        "q_base",
        "q_stride",
        "k_base",
        "k_stride",
        "v_base",
        "v_stride",
        "c_base",
        "c_stride",
        "old_base",
        "old_head",
        "old_key",
        "old_column",
        "next_base",
        "next_head",
        "next_key",
        "next_column",
        "out_base",
        "out_stride",
        "status_base",
        "status_stride",
    ] {
        assert!(runtime.contains(&format!("{name}: usize")));
    }
    assert_eq!(signature.matches("#[comptime]").count(), 3);
}

#[test]
fn three_streaming_exact_chunk_and_original_steps_with_independent_core() {
    for shape in [
        GdnShape::new(7, 2, 6).unwrap(),
        GdnShape::new(31, 3, 9).unwrap(),
        GdnShape::qwen38(),
    ] {
        let l = GdnLayout::new(shape).unwrap();
        for count in 1..=3 {
            let f = fixture(shape, count).unwrap();
            let s = f.spans;
            let old_bits: Vec<_> = f.old.iter().map(|x| x.to_bits()).collect();
            let streaming =
                chunk::three_streaming_lane_reference(shape, s, three_streaming_inputs(&f))
                    .unwrap();
            let ordinary = chunk::lane_reference(shape, s, three_streaming_inputs(&f)).unwrap();
            three_streaming_assert_exact(&streaming, &ordinary);
            let mut original = f.initial.to_vec();
            let mut core = GdnState::new(shape).unwrap();
            core.set_values(&original).unwrap();
            for token in 0..count {
                let q = &f.query[s.query.at(token)..][..l.qk_elements];
                let k = &f.key[s.key.at(token)..][..l.qk_elements];
                let v = &f.value[s.value.at(token)..][..l.value_elements];
                let c = &f.controls[s.controls.at(token)..][..l.control_elements];
                let scalar = gdn_gpu::lane_reference(
                    shape,
                    q,
                    k,
                    v,
                    &original,
                    &c[..shape.value_heads],
                    &c[shape.value_heads..],
                )
                .unwrap();
                three_streaming_assert_float_bits(
                    &streaming.1[s.output.at(token)..][..l.value_elements],
                    &scalar.output,
                );
                original = scalar.state;
                let output = core
                    .step(q, k, v, &c[..shape.value_heads], &c[shape.value_heads..])
                    .unwrap();
                compare(
                    &streaming.1[s.output.at(token)..][..l.value_elements],
                    &output,
                )
                .unwrap();
            }
            let final_state = packed_state(shape, s.next, &streaming.0);
            three_streaming_assert_float_bits(&final_state, &original);
            compare(&final_state, core.values()).unwrap();
            chunk::validate_download(
                shape,
                s,
                &streaming.0,
                &streaming.1,
                &streaming.2,
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(
                old_bits,
                f.old.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            three_streaming_preserves_gaps(shape, s, &streaming);
            core.reset();
        }
    }
}

#[test]
fn three_streaming_failures_at_each_token_clear_every_candidate_and_preserve_old() {
    let shape = GdnShape::new(7, 2, 6).unwrap();
    for count in 1..=3 {
        for failed_token in 0..count {
            // Invalid control, delta, prediction, update and final readout.
            for failure in 0..7 {
                let mut f = fixture(shape, count).unwrap();
                let s = f.spans;
                match failure {
                    0 => f.controls[s.controls.at(failed_token)] = 0.1,
                    1 => f.controls[s.controls.at(failed_token) + shape.value_heads] = 1.1,
                    2 => f.value[s.value.at(failed_token)] = f32::INFINITY,
                    3 => f.key[s.key.at(failed_token)] = f32::NAN,
                    _ => {
                        // Preserve a large finite column through all earlier
                        // tokens, then overflow at the intended operation.
                        for token in 0..count {
                            f.controls[s.controls.at(token)] = 0.;
                            f.controls[s.controls.at(token) + shape.value_heads] = 0.;
                            for i in 0..shape.head_dim {
                                f.key[s.key.at(token) + i] = 0.;
                                f.query[s.query.at(token) + i] = 0.;
                            }
                        }
                        for i in 0..shape.head_dim {
                            f.old[s.old.at(0, i, 0)] = if failure == 5 {
                                if i < 2 { f32::MAX / 2. } else { 0. }
                            } else {
                                f32::MAX / 4.
                            };
                        }
                        if failure == 4 {
                            for i in 0..shape.head_dim {
                                f.key[s.key.at(failed_token) + i] = 4.;
                            }
                        } else if failure == 5 {
                            f.key[s.key.at(failed_token)] = 1.;
                            f.key[s.key.at(failed_token) + 1] = -1.;
                            f.value[s.value.at(failed_token)] = f32::MAX;
                            f.controls[s.controls.at(failed_token) + shape.value_heads] = 1.;
                        } else {
                            f.query[s.query.at(failed_token)] = 4.;
                            f.query[s.query.at(failed_token) + 1] = 4.;
                        }
                    }
                }
                three_streaming_assert_failed_column(shape, &f);
            }
        }
        // A rejected first token must never consume uninitialized next GAP on
        // tokens1/2. Nonfinite old is immutable, including its exact NaN bits.
        for bad_old in [f32::from_bits(0x7fc0_1234), f32::INFINITY] {
            let mut f = fixture(shape, count).unwrap();
            f.old[f.spans.old.at(0, 0, 0)] = bad_old;
            three_streaming_assert_failed_column(shape, &f);
        }
    }
}

fn three_streaming_assert_failed_column(shape: GdnShape, f: &Fixture) {
    let s = f.spans;
    let before: Vec<_> = f.old.iter().map(|x| x.to_bits()).collect();
    let streaming =
        chunk::three_streaming_lane_reference(shape, s, three_streaming_inputs(f)).unwrap();
    let ordinary = chunk::lane_reference(shape, s, three_streaming_inputs(f)).unwrap();
    three_streaming_assert_exact(&streaming, &ordinary);
    for token in 0..s.count {
        assert_eq!(streaming.1[s.output.at(token)].to_bits(), 0);
        assert_eq!(streaming.2[s.status.at(token)], 0);
    }
    for i in 0..shape.head_dim {
        assert_eq!(streaming.0[s.next.at(0, i, 0)].to_bits(), 0);
    }
    assert!(
        chunk::validate_download(
            shape,
            s,
            &streaming.0,
            &streaming.1,
            &streaming.2,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert_eq!(
        before,
        f.old.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    three_streaming_preserves_gaps(shape, s, &streaming);
}

fn three_streaming_spirv(module: &wgpu::naga::Module) -> Vec<u32> {
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

// Original operations are still scalar f32. Compare arithmetic instructions,
// excluding the extra validity checks needed for direct candidate reads.
fn three_streaming_float_operations(module: &wgpu::naga::Module) -> Vec<String> {
    use wgpu::naga::{BinaryOperator, Expression, Literal, ScalarKind, TypeInner};
    let f = &module.entry_points[0].function;
    fn float_type(m: &wgpu::naga::Module, t: wgpu::naga::Handle<wgpu::naga::Type>) -> bool {
        match m.types[t].inner {
            TypeInner::Scalar(s) | TypeInner::Vector { scalar: s, .. } => {
                s.kind == ScalarKind::Float
            }
            TypeInner::Array { base, .. } | TypeInner::Pointer { base, .. } => float_type(m, base),
            _ => false,
        }
    }
    fn float_expr(
        m: &wgpu::naga::Module,
        f: &wgpu::naga::Function,
        h: wgpu::naga::Handle<Expression>,
    ) -> bool {
        match f.expressions[h] {
            Expression::Literal(Literal::F32(_)) => true,
            Expression::LocalVariable(v) => float_type(m, f.local_variables[v].ty),
            Expression::GlobalVariable(v) => float_type(m, m.global_variables[v].ty),
            Expression::Load { pointer } => float_expr(m, f, pointer),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                float_expr(m, f, base)
            }
            Expression::Binary { left, right, .. } => {
                float_expr(m, f, left) || float_expr(m, f, right)
            }
            Expression::Math { arg, .. } | Expression::Unary { expr: arg, .. } => {
                float_expr(m, f, arg)
            }
            _ => false,
        }
    }
    let mut operations = Vec::new();
    for (_, expression) in f.expressions.iter() {
        match *expression {
            Expression::Binary { op, left, right }
                if matches!(
                    op,
                    BinaryOperator::Add
                        | BinaryOperator::Subtract
                        | BinaryOperator::Multiply
                        | BinaryOperator::Divide
                ) && (float_expr(module, f, left) || float_expr(module, f, right)) =>
            {
                operations.push(format!("{op:?}"));
            }
            Expression::Math { fun, arg, .. } if float_expr(module, f, arg) => {
                assert_ne!(fun, wgpu::naga::MathFunction::Fma);
                operations.push(format!("{fun:?}"));
            }
            _ => {}
        }
    }
    operations.sort_unstable();
    operations
}

#[test]
fn three_streaming_actual_shaders_keep_runtime_spans_and_original_float_operations() {
    use cubecl::server::ExecutionMode;
    use wgpu::naga::{AddressSpace, TypeInner};
    for shape in [GdnShape::new(7, 2, 6).unwrap(), GdnShape::qwen38()] {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let source = shader(shape, mode, RecurrencePolicy::ThreeStreaming);
            assert!(source.contains("exp("));
            assert!(!source.contains("array<f32,"));
            assert!(!source.contains("array<f64"));
            assert!(!source.contains("fma("));
            assert!(!source.contains("workgroupBarrier"));
            for scalar in 0..21 {
                assert!(source.contains(&format!("scalars_u32[{scalar}]")));
            }
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let original =
                wgpu::naga::front::wgsl::parse_str(&shader(shape, mode, RecurrencePolicy::Chunk))
                    .unwrap();
            assert!(
                original.entry_points[0]
                    .function
                    .local_variables
                    .iter()
                    .any(|(_, v)| {
                        matches!(original.types[v.ty].inner, TypeInner::Array {
                    size: wgpu::naga::ArraySize::Constant(n), ..
                } if n.get() as usize == shape.head_dim)
                    })
            );
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
            assert!(
                !module.global_variables.iter().any(|(_, v)| matches!(
                    v.space,
                    AddressSpace::WorkGroup | AddressSpace::Private
                ))
            );
            assert!(
                module.entry_points[0]
                    .function
                    .local_variables
                    .iter()
                    .all(|(_, v)| !matches!(module.types[v.ty].inner, TypeInner::Array { .. }))
            );
            let bindings = |m: &wgpu::naga::Module| {
                m.global_variables
                    .iter()
                    .filter_map(|(_, v)| {
                        v.binding
                            .as_ref()
                            .map(|b| (b.group, b.binding, format!("{:?}", v.space), v.name.clone()))
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(bindings(&module), bindings(&original));
            assert!(
                module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| matches!(v.space, AddressSpace::Storage { .. }))
                    .count()
                    >= 8
            );
            assert_eq!(
                three_streaming_float_operations(&module),
                three_streaming_float_operations(&original)
            );
            let words = three_streaming_spirv(&module);
            let mut cursor = 5;
            while cursor < words.len() {
                let instruction = words[cursor];
                let length = (instruction >> 16) as usize;
                assert!(length > 0);
                let opcode = instruction & 0xffff;
                assert!(
                    !matches!(opcode, 224 | 225),
                    "streaming recurrence needs no memory barrier"
                );
                if opcode == 12 {
                    assert_ne!(words[cursor + 4], 50, "no GLSL Fma instruction");
                }
                cursor += length;
            }
            assert_eq!(cursor, words.len());
        }
    }
}
