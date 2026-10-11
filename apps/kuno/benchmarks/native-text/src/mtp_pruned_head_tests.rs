// Offline acceptance only. No client, model, source payload or GPU is created.
use super::super::q8_head_broadcast_tests::{original_tree, root_global, validate_runtime_spirv};
use super::*;
use cubecl::compute::KernelBuilder;
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
use cubecl::server::ExecutionMode;
use cubecl::wgpu::WgslCompiler;
use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
use wgpu::naga::{self, AddressSpace, Expression, Literal, Statement, TypeInner};

fn english_ids() -> Vec<u32> {
    let proof = EnglishVocabulary::embedded().unwrap();
    (0..ENGLISH_ROWS).map(|i| proof.id(i).unwrap()).collect()
}

#[test]
fn english_vocabulary_exact_identity_and_all_source_segment_bounds() {
    assert_eq!(ENGLISH_ROWS, 40_525);
    assert_eq!(ENGLISH_BYTES, 162_100);
    assert_eq!(ENGLISH_OWNED_BYTES, 262_144);
    assert_eq!((VOCAB, TILE_ROWS, SEGMENTS), (248_320, 16_384, 16));
    let proof = EnglishVocabulary::embedded().unwrap();
    let bytes = proof.bytes();
    assert_eq!(bytes.len(), ENGLISH_BYTES);
    EnglishVocabulary::validate_exact(bytes).unwrap();
    assert!(EnglishVocabulary::validate_exact(&bytes[..bytes.len() - 1]).is_err());
    let mut surplus = bytes.to_vec();
    surplus.extend_from_slice(&0_u32.to_le_bytes());
    assert!(EnglishVocabulary::validate_exact(&surplus).is_err());
    let ids = english_ids();
    assert!(ids.iter().all(|&id| (id as usize) < VOCAB));
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(proof.id(ENGLISH_ROWS).is_err());
    assert!(proof.id(usize::MAX).is_err());
    for (index, id) in ids.iter().enumerate() {
        assert_eq!(&bytes[index * 4..][..4], &id.to_le_bytes());
    }
    for mutation in 0..5 {
        let mut bad = bytes.to_vec();
        match mutation {
            0 => bad[..4].copy_from_slice(&(VOCAB as u32).to_le_bytes()),
            1 => bad[4..8].copy_from_slice(&ids[0].to_le_bytes()),
            2 => {
                bad[..4].copy_from_slice(&ids[1].to_le_bytes());
                bad[4..8].copy_from_slice(&ids[0].to_le_bytes());
            }
            3 => bad[..4].copy_from_slice(&u32::MAX.to_le_bytes()),
            _ => {
                // Sorted and in range is insufficient: the public identity
                // must reject even an otherwise valid single-ID substitution.
                let index = ids.windows(2).position(|p| p[1] - p[0] > 1).unwrap();
                bad[index * 4..][..4].copy_from_slice(&(ids[index] + 1).to_le_bytes());
            }
        }
        assert!(EnglishVocabulary::validate_exact(&bad).is_err());
    }
    let mut covered = vec![0_u8; ENGLISH_ROWS];
    let mut previous_end = 0;
    for page in 0..SEGMENTS {
        let segment = proof.segment(page).unwrap();
        segment.validate().unwrap();
        assert_eq!(segment.source_first, page * TILE_ROWS);
        assert_eq!(
            segment.source_rows,
            TILE_ROWS.min(VOCAB - segment.source_first)
        );
        assert_eq!(segment.subset_first, previous_end);
        let expected: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                ((id as usize) >= segment.source_first
                    && (id as usize) < segment.source_first + segment.source_rows)
                    .then_some(i)
            })
            .collect();
        assert_eq!(expected.len(), segment.subset_rows);
        for index in expected {
            assert!(
                index >= segment.subset_first && index < segment.subset_first + segment.subset_rows
            );
            covered[index] += 1;
        }
        previous_end += segment.subset_rows;
        for tail in [false, true] {
            if segment.subset_rows == 0 {
                assert!(segment.cube_count(tail, VOCAB).is_err());
            } else {
                let rows = segment.subset_rows.div_ceil(4);
                let groups = if tail {
                    rows.max(VOCAB.div_ceil(128))
                } else {
                    rows
                };
                assert_eq!(
                    format!("{:?}", segment.cube_count(tail, VOCAB).unwrap()),
                    format!(
                        "{:?}",
                        cubecl::prelude::CubeCount::Static(groups as u32, 1, 1)
                    )
                );
                assert!(segment.cube_count(tail, VOCAB - 1).is_err());
            }
        }
        for mutation in 0..4 {
            let mut bad = segment;
            match mutation {
                0 => bad.source_first = VOCAB,
                1 => bad.source_rows = usize::MAX,
                2 => bad.subset_first = ENGLISH_ROWS,
                _ => bad.subset_rows = usize::MAX,
            }
            assert!(bad.validate().is_err());
        }
    }
    assert_eq!(previous_end, ENGLISH_ROWS);
    assert!(covered.iter().all(|&n| n == 1));
    assert!(proof.segment(SEGMENTS).is_err());
    assert!(proof.segment(usize::MAX).is_err());
}

// Independent IEEE binary16 equation, without the production half helper.
fn english_half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1_f32 } else { -1_f32 };
    let exponent = i32::from((bits >> 10) & 31);
    let fraction = u32::from(bits & 1023);
    assert!(exponent < 31);
    if exponent == 0 {
        sign * (fraction as f32 * 2_f32.powi(-24))
    } else {
        sign * ((1024 + fraction) as f32 * 2_f32.powi(exponent - 25))
    }
}

fn english_q8_row(id: u32, phase: usize) -> Vec<u8> {
    let halves: [u16; 10] = [
        0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff, 0x9231,
    ];
    let mut packed = Vec::with_capacity(2720);
    for block in 0..80 {
        packed
            .extend_from_slice(&halves[(id as usize + block + phase) % halves.len()].to_le_bytes());
        for lane in 0..32 {
            packed.push((id as usize * 11 + block * 37 + lane * 29 + phase * 31) as u8);
        }
    }
    packed
}

fn english_q8_tree(words: &[u32], first: usize, input: &[f32]) -> f32 {
    let byte = |offset: usize| ((words[offset / 4] >> (offset % 4 * 8)) & 255) as u8;
    let mut partial = [0_f32; 32];
    for (lane, value) in partial.iter_mut().enumerate() {
        let mut low = 0_f32;
        let mut high = 0_f32;
        for tile in 0..40 {
            for (half, sum) in [(0, &mut low), (1, &mut high)] {
                let base = first + tile * 68 + half * 34;
                let scale = english_half(u16::from_le_bytes([byte(base), byte(base + 1)]));
                let quant = byte(base + 2 + lane) as i8;
                *sum += (scale * f32::from(quant)) * input[tile * 64 + half * 32 + lane];
            }
        }
        *value = low + high;
    }
    for stride in [16, 8, 4, 2, 1] {
        for lane in 0..stride {
            partial[lane] += partial[lane + stride];
        }
    }
    partial[0]
}

#[test]
fn english_selected_original_q8_rows_keep_exact_weights_and_original64_tree() {
    let proof = EnglishVocabulary::embedded().unwrap();
    for page in 0..SEGMENTS {
        let segment = proof.segment(page).unwrap();
        if segment.subset_rows == 0 {
            continue;
        }
        // Bound arithmetic checks to first/middle/last selected rows in each
        // original source page; the map test exhausts all40,525 row mappings.
        for index in [
            segment.subset_first,
            segment.subset_first + segment.subset_rows / 2,
            segment.subset_first + segment.subset_rows - 1,
        ] {
            let id = proof.id(index).unwrap();
            let source_row = id as usize - segment.source_first;
            assert!(source_row < segment.source_rows);
            assert_eq!(segment.source_first + source_row, id as usize);
            for phase in [0, 1, 7] {
                let packed = english_q8_row(id, phase);
                let mut original = vec![0_f32; 2560];
                kuno_text::TensorType::Q8_0
                    .decode(&packed, &mut original)
                    .unwrap();
                for (column, value) in original.iter().enumerate() {
                    let base = column / 32 * 34;
                    let scale =
                        english_half(u16::from_le_bytes(packed[base..][..2].try_into().unwrap()));
                    assert_eq!(
                        value.to_bits(),
                        (scale * f32::from(packed[base + 2 + column % 32] as i8)).to_bits()
                    );
                }
                // Exercise original two-byte header alignment and exact word
                // extraction, including adjacent34-byte block boundaries.
                let mut shifted = vec![0xa5, 0x5a];
                shifted.extend_from_slice(&packed);
                let words: Vec<u32> = shifted
                    .chunks(4)
                    .map(|chunk| {
                        let mut bytes = [0; 4];
                        bytes[..chunk.len()].copy_from_slice(chunk);
                        u32::from_le_bytes(bytes)
                    })
                    .collect();
                let input: Vec<_> = (0..2560)
                    .map(|column| {
                        (((column * 17 + phase * 19) % 83) as f32 - 41.)
                            * 2_f32.powi((column % 23) as i32 - 19)
                    })
                    .collect();
                assert_eq!(
                    english_q8_tree(&words, 2, &input).to_bits(),
                    original_tree(&original, &input).to_bits()
                );
            }
        }
    }
}

fn english_greedy_reference(ids: &[u32], logits: &[f32]) -> (Option<u32>, Vec<u32>) {
    const GROUPS: usize = 10;
    let mut candidates = vec![0x53aa_9267; 183];
    let mut global_good = ids.len() >= ENGLISH_ROWS && logits.len() >= ENGLISH_ROWS;
    if !global_good {
        return (None, candidates);
    }
    for group in 0..GROUPS {
        let mut values = [-f32::MAX; 64];
        let mut tokens = [u32::MAX; 64];
        let mut good = [true; 64];
        for lane in 0..64 {
            for tile in 0..64 {
                let index = group * 4096 + tile * 64 + lane;
                if index >= ENGLISH_ROWS {
                    continue;
                }
                let id = ids[index];
                let value = logits[index];
                good[lane] &= (id as usize) < VOCAB && value.is_finite();
                if value > values[lane] || value == values[lane] && id < tokens[lane] {
                    values[lane] = value;
                    tokens[lane] = id;
                }
            }
        }
        for stride in [32, 16, 8, 4, 2, 1] {
            for lane in 0..stride {
                if values[lane + stride] > values[lane]
                    || values[lane + stride] == values[lane] && tokens[lane + stride] < tokens[lane]
                {
                    values[lane] = values[lane + stride];
                    tokens[lane] = tokens[lane + stride];
                }
                good[lane] &= good[lane + stride];
            }
        }
        candidates[group * 3..][..3].copy_from_slice(&[
            tokens[0],
            values[0].to_bits(),
            u32::from(good[0]),
        ]);
        global_good &= good[0];
    }
    let mut best = -f32::MAX;
    let mut token = u32::MAX;
    for candidate in candidates[..30].chunks_exact(3) {
        let value = f32::from_bits(candidate[1]);
        global_good &= candidate[2] == 1 && (candidate[0] as usize) < VOCAB && value.is_finite();
        if value > best || value == best && candidate[0] < token {
            best = value;
            token = candidate[0];
        }
    }
    (global_good.then_some(token), candidates)
}

#[test]
fn english_greedy_negative_logits_and_ties_return_full_mapped_token_ids() {
    let ids = english_ids();
    for pattern in 0..4 {
        // Full original owner: omitted rows must not compete, even when their
        // finite values exceed every negative selected logit.
        let mut logits = vec![1000.; VOCAB];
        let selected = &mut logits[..ENGLISH_ROWS];
        match pattern {
            0 => selected.fill(-f32::MAX),
            1 => {
                selected.fill(-2.);
                logits[ENGLISH_ROWS - 1] = -1.;
            }
            2 => {
                selected.fill(-2.);
                logits[4095] = -1.;
                logits[4096] = -1.;
            }
            _ => {
                selected.fill(-2.);
                logits[63] = -1.;
                logits[64] = -1.;
                logits[4096] = -1.;
            }
        }
        let expected = ids
            .iter()
            .zip(&logits)
            .fold((u32::MAX, -f32::MAX), |best, (&id, &value)| {
                if value > best.1 || value == best.1 && id < best.0 {
                    (id, value)
                } else {
                    best
                }
            })
            .0;
        let (actual, candidates) = english_greedy_reference(&ids, &logits);
        assert_eq!(actual, Some(expected));
        assert!(candidates[30..].iter().all(|&word| word == 0x53aa_9267));
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = logits.clone();
            bad[ENGLISH_ROWS - 1] = value;
            assert_eq!(english_greedy_reference(&ids, &bad).0, None);
        }
    }
    let logits = vec![-1.; ENGLISH_ROWS];
    let mut bad = ids.clone();
    bad[4096] = VOCAB as u32;
    assert_eq!(english_greedy_reference(&bad, &logits).0, None);
    assert_eq!(
        english_greedy_reference(&ids[..ids.len() - 1], &logits).0,
        None
    );
}

fn english_shader(segment: HeadSegment, kind: usize, mode: ExecutionMode) -> String {
    let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
    let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
    let atomic = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
    let mut b = KernelBuilder::default();
    b.scope.register_type::<u32>(uint);
    b.scope.register_type::<usize>(uint);
    b.scope.register_type::<f32>(float);
    b.scope
        .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
    b.scope
        .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
    let name;
    let lanes;
    if kind < 2 {
        let packed = b.input_array(Type::new(uint)).into();
        let ids = b.input_array(Type::new(uint)).into();
        let input = b.input_array(Type::new(float)).into();
        let output = b.output_array(Type::new(float)).into();
        let fault = b.output_array(Type::new(atomic)).into();
        let scalars: Vec<_> = (0..5).map(|_| b.scalar(uint)).collect();
        let expand = if kind == 1 {
            packed_q8_english_head_shared_scales_fault::expand
        } else {
            packed_q8_english_head_fault::expand
        };
        expand(
            &mut b.scope,
            packed,
            ids,
            input,
            output,
            fault,
            scalars[0].clone().into(),
            scalars[1].clone().into(),
            scalars[2].clone().into(),
            scalars[3].clone().into(),
            scalars[4].clone().into(),
            segment.source_rows,
            segment.subset_rows,
            true,
        );
        name = if kind == 1 {
            "packed_q8_english_head_shared_scales_fault"
        } else {
            "packed_q8_english_head_fault"
        };
        lanes = 128;
    } else if kind == 2 {
        let logits = b.input_array(Type::new(float)).into();
        let ids = b.input_array(Type::new(uint)).into();
        let candidates = b.output_array(Type::new(uint)).into();
        let fault = b.output_array(Type::new(atomic)).into();
        english_greedy_chunks::expand(&mut b.scope, logits, ids, candidates, fault);
        name = "english_greedy_chunks";
        lanes = 64;
    } else {
        let candidates = b.input_array(Type::new(uint)).into();
        let token = b.output_array(Type::new(uint)).into();
        let fault = b.output_array(Type::new(atomic)).into();
        english_greedy_finish::expand(&mut b.scope, candidates, token, fault);
        name = "english_greedy_finish";
        lanes = 64;
    }
    WgslCompiler::default()
        .compile(
            b.build(
                KernelSettings::default()
                    .cube_dim(cubecl::prelude::CubeDim::new_1d(lanes))
                    .kernel_name(name),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            uint,
        )
        .unwrap()
        .to_string()
}

// Narrow Boolean dataflow proof: count==1 and the exact mapped source-row
// upper bound must dominate every packed load. False implies both conditions;
// true implies neither. This handles Naga's short-circuit Boolean phi stores.
fn english_assert_packed_reads_guarded(module: &naga::Module, rows: usize) {
    use naga::BinaryOperator;
    use std::collections::{HashMap, HashSet};
    let f = &module.entry_points[0].function;
    type Local = naga::Handle<naga::LocalVariable>;
    fn constant(f: &naga::Function, h: naga::Handle<Expression>, value: usize) -> bool {
        matches!(f.expressions[h], Expression::Literal(Literal::U32(n)) if n as usize == value)
    }
    fn local(f: &naga::Function, h: naga::Handle<Expression>) -> Option<Local> {
        match f.expressions[h] {
            Expression::Load { pointer } | Expression::As { expr: pointer, .. } => {
                local(f, pointer)
            }
            Expression::LocalVariable(v) => Some(v),
            _ => None,
        }
    }
    let sources: HashSet<_> = f
        .expressions
        .iter()
        .filter_map(|(_, e)| {
            if let Expression::Binary {
                op: BinaryOperator::Multiply,
                left,
                right,
            } = *e
            {
                if constant(f, right, ROW_BYTES) {
                    return local(f, left);
                }
                if constant(f, left, ROW_BYTES) {
                    return local(f, right);
                }
            }
            None
        })
        .collect();
    assert!(
        !sources.is_empty(),
        "actual source*original-row-bytes indexing"
    );
    fn facts(
        f: &naga::Function,
        h: naga::Handle<Expression>,
        rows: usize,
        sources: &HashSet<Local>,
        locals: &HashMap<Local, u8>,
    ) -> u8 {
        match f.expressions[h] {
            Expression::Literal(Literal::Bool(false)) => 3,
            Expression::Literal(Literal::Bool(true)) => 0,
            Expression::Load { pointer } => local(f, pointer)
                .and_then(|l| locals.get(&l).copied())
                .unwrap_or(0),
            Expression::Binary {
                op: BinaryOperator::Equal,
                right,
                ..
            } if constant(f, right, 1) => 1,
            Expression::Binary {
                op: BinaryOperator::Less,
                left,
                right,
            } if constant(f, right, rows)
                && local(f, left).is_some_and(|l| sources.contains(&l)) =>
            {
                2
            }
            Expression::Binary {
                op: BinaryOperator::LogicalAnd,
                left,
                right,
            } => facts(f, left, rows, sources, locals) | facts(f, right, rows, sources, locals),
            Expression::Binary {
                op: BinaryOperator::LogicalOr,
                left,
                right,
            } => facts(f, left, rows, sources, locals) & facts(f, right, rows, sources, locals),
            _ => 0,
        }
    }
    #[allow(clippy::too_many_arguments)] // AST traversal state and proof masks.
    fn walk(
        m: &naga::Module,
        f: &naga::Function,
        body: &naga::Block,
        rows: usize,
        sources: &HashSet<Local>,
        locals: &mut HashMap<Local, u8>,
        guard: u8,
        reads: &mut usize,
    ) {
        for s in body {
            match s {
                Statement::Store { pointer, value } => {
                    if let Expression::LocalVariable(l) = f.expressions[*pointer] {
                        locals.insert(l, guard | facts(f, *value, rows, sources, locals));
                    }
                }
                Statement::Emit(range) => {
                    for h in range.clone() {
                        if let Expression::Load { pointer } = f.expressions[h]
                            && root_global(f, pointer).is_some_and(|g| {
                                m.global_variables[g]
                                    .binding
                                    .as_ref()
                                    .is_some_and(|b| b.binding == 0)
                            })
                        {
                            assert_eq!(
                                guard & 3,
                                3,
                                "packed read without count and mapped source bound"
                            );
                            *reads += 1;
                        }
                    }
                }
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let mut a = locals.clone();
                    let mut b = locals.clone();
                    walk(
                        m,
                        f,
                        accept,
                        rows,
                        sources,
                        &mut a,
                        guard | facts(f, *condition, rows, sources, locals),
                        reads,
                    );
                    walk(m, f, reject, rows, sources, &mut b, guard, reads);
                    for l in a.keys().chain(b.keys()) {
                        locals.insert(
                            *l,
                            a.get(l).copied().unwrap_or(0) & b.get(l).copied().unwrap_or(0),
                        );
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    walk(m, f, body, rows, sources, locals, guard, reads);
                    walk(m, f, continuing, rows, sources, locals, guard, reads);
                }
                Statement::Block(body) => walk(m, f, body, rows, sources, locals, guard, reads),
                _ => {}
            }
        }
    }
    let mut reads = 0;
    walk(
        module,
        f,
        &f.body,
        rows,
        &sources,
        &mut HashMap::new(),
        0,
        &mut reads,
    );
    assert!(reads > 0);
}

fn english_assert_collectives(module: &naga::Module, kind: usize) {
    fn walk(
        block: &naga::Block,
        depth: usize,
        greedy: bool,
        gathers: &mut usize,
        barriers: &mut usize,
    ) {
        for statement in block {
            match statement {
                Statement::SubgroupGather { .. } => {
                    assert_eq!(depth, 0);
                    *gathers += 1;
                }
                Statement::ControlBarrier(_) => {
                    if greedy {
                        assert_eq!(depth, 0);
                    }
                    *barriers += 1;
                }
                Statement::If { accept, reject, .. } => {
                    walk(accept, depth + 1, greedy, gathers, barriers);
                    walk(reject, depth + 1, greedy, gathers, barriers);
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    walk(body, depth, greedy, gathers, barriers);
                    walk(continuing, depth, greedy, gathers, barriers);
                }
                Statement::Block(body) => walk(body, depth, greedy, gathers, barriers),
                _ => {}
            }
        }
    }
    let mut gathers = 0;
    let mut barriers = 0;
    walk(
        &module.entry_points[0].function.body,
        0,
        kind >= 2,
        &mut gathers,
        &mut barriers,
    );
    assert_eq!(barriers, if kind == 0 { 1 } else { 2 });
    if kind < 2 {
        assert!(gathers > 0);
        let mut reduction = module.clone();
        if kind == 1 {
            let f = &mut reduction.entry_points[0].function;
            let fence = f
                .body
                .iter()
                .position(|s| matches!(s, Statement::ControlBarrier(_)))
                .unwrap();
            // Isolate the original reducer for its existing subgroup16 proof;
            // scale memory has an independent unconditional initialization fence.
            f.body = naga::Block::from_vec(
                f.body
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| (i != fence).then_some(s.clone()))
                    .collect(),
            );
            for (_, v) in reduction.global_variables.iter_mut() {
                if v.space == AddressSpace::WorkGroup
                    && matches!(reduction.types[v.ty].inner, TypeInner::Array { size: naga::ArraySize::Constant(n), .. } if n.get() == 320)
                {
                    v.space = AddressSpace::Private;
                }
            }
        }
        super::super::tests::assert_shared_reduction_is_only_for16(&reduction);
    }
}

#[test]
fn english_actual_head_and_greedy_checked_unchecked_shaders_keep_uniform_reducers() {
    let proof = EnglishVocabulary::embedded().unwrap();
    for page in [0, 5, 9, 15] {
        let segment = proof.segment(page).unwrap();
        for kind in 0..4 {
            if kind >= 2 && page != 0 {
                continue;
            }
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let source = english_shader(segment, kind, mode);
                let module = naga::front::wgsl::parse_str(&source).unwrap();
                let spirv = validate_runtime_spirv(&module);
                assert!(!source.contains("fma("));
                assert_eq!(
                    module.entry_points[0].workgroup_size,
                    [if kind < 2 { 128 } else { 64 }, 1, 1]
                );
                english_assert_collectives(&module, kind);
                if kind < 2 {
                    for scalar in 0..5 {
                        assert!(source.contains(&format!("scalars_u32[{scalar}]")));
                    }
                    english_assert_packed_reads_guarded(&module, segment.source_rows);
                    assert!(
                        module.entry_points[0]
                            .function
                            .expressions
                            .iter()
                            .any(|(_, e)| matches!(e,
                        Expression::Literal(Literal::U32(n)) if *n == VOCAB as u32))
                    );
                    assert_eq!(
                        module
                            .global_variables
                            .iter()
                            .filter(|(_, v)| matches!(v.space, AddressSpace::Storage { .. }))
                            .count(),
                        6
                    );
                }
                let mut cursor = 5;
                let mut barriers = 0;
                while cursor < spirv.len() {
                    let length = (spirv[cursor] >> 16) as usize;
                    assert!(length > 0);
                    if spirv[cursor] & 65535 == 224 {
                        barriers += 1;
                    }
                    cursor += length;
                }
                assert_eq!(cursor, spirv.len());
                assert_eq!(barriers, if kind == 0 { 1 } else { 2 });
            }
        }
    }
    // A hoisted packed load must fail the actual AST guard proof.
    let segment = proof.segment(15).unwrap();
    let mut module =
        naga::front::wgsl::parse_str(&english_shader(segment, 0, ExecutionMode::Unchecked))
            .unwrap();
    let f = &mut module.entry_points[0].function;
    let load = f
        .expressions
        .iter()
        .find_map(|(h, e)| {
            if let Expression::Load { pointer } = *e
                && root_global(f, pointer).is_some_and(|g| {
                    module.global_variables[g]
                        .binding
                        .as_ref()
                        .is_some_and(|b| b.binding == 0)
                })
            {
                return Some(h);
            }
            None
        })
        .unwrap();
    f.body.push(
        Statement::Emit(naga::Range::new_from_bounds(load, load)),
        naga::Span::UNDEFINED,
    );
    assert!(
        std::panic::catch_unwind(|| english_assert_packed_reads_guarded(
            &module,
            segment.source_rows
        ))
        .is_err()
    );
}
