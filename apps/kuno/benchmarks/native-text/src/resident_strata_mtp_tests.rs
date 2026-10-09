use super::*;
use cubecl::{Compiler, WgpuCompilationOptions, server::ExecutionMode, wgpu::WgslCompiler};
use std::collections::BTreeSet;

#[test]
fn explicit_profile_admission_cannot_silently_replace_the_original_draft() {
    for target in [false, true] {
        assert!(
            validate_profile(
                MtpProfile::SelfContainedQ8,
                MtpProfile::SelfContainedQ8,
                target
            )
            .is_ok()
        );
        assert!(
            validate_profile(MtpProfile::StrataQ2, MtpProfile::SelfContainedQ8, target).is_err()
        );
        assert!(
            validate_profile(MtpProfile::SelfContainedQ8, MtpProfile::StrataQ2, target).is_err()
        );
    }
    assert!(validate_profile(MtpProfile::StrataQ2, MtpProfile::StrataQ2, false).is_err());
    assert!(validate_profile(MtpProfile::StrataQ2, MtpProfile::StrataQ2, true).is_ok());
}

#[test]
fn paired_target_roles_and_independent_original_page_plan_remain_bounded() {
    let mut names = BTreeSet::new();
    let mut matrix_owned = 0;
    for spec in matrices_for(MtpProfile::StrataQ2).iter().copied().chain([
        head_up_for(MtpProfile::StrataQ2),
        embedding_for(MtpProfile::StrataQ2),
    ]) {
        assert!(names.insert(spec.name));
        let expected_kind = if spec.name == "output.weight" {
            TensorType::Q6K
        } else if spec.name == "token_embd.weight" {
            TensorType::Iq4Xs
        } else {
            TensorType::Bf16
        };
        assert_eq!(TensorType::from_gguf_id(spec.dtype).unwrap(), expected_kind);
        let mut end = 0;
        for (first, rows, logical, owned) in tile_plan(spec).unwrap() {
            assert_eq!(first, end);
            end += rows;
            assert_eq!(
                logical as u64,
                expected_kind.row_bytes(spec.columns as u64).unwrap() * rows as u64
            );
            assert!(logical <= owned && owned <= PUBLIC_SOURCE_BYTES);
            assert!(logical.is_multiple_of(4));
            matrix_owned += owned as u64;
        }
        assert_eq!(end, spec.rows);
    }
    assert_eq!(matrix_owned, 1_732_542_464);
    assert!(!names.contains("blk.48.nextn.eh_proj.weight"));
    assert!(names.contains(strata::FC_EMBEDDING) && names.contains(strata::FC_HIDDEN));
    assert!(names.contains("blk.48.ffn_gate_inp_shexp.weight"));
    for name in [strata::GU_NAME, strata::DOWN_NAME] {
        assert!(names.insert(name));
    }
    let mut vector_owned = 0;
    for &(name, words) in vectors_for(MtpProfile::StrataQ2) {
        assert!(names.insert(name));
        vector_owned += main_admission::page_bytes((words * 4) as u64).unwrap();
    }
    assert_eq!(names.len(), 30);
    assert!(!names.iter().any(|name| name.contains("indexer")));
    assert_eq!(vector_owned, 4 * 65536 + 3 * 32768);
    let expected =
        1_732_542_464 + (4 * 65536 + 3 * 32768) + 2 * 524288 + 4 * (128 + 64) * 1024 * 1024;
    assert_eq!(expected, 2_539_257_856);
    assert_eq!(public_plan_for(MtpProfile::StrataQ2).unwrap(), expected);
    assert_eq!(public_bytes_for(MtpProfile::StrataQ2), expected);
    assert_eq!(public_plan().unwrap(), 5_011_668_992);
    assert_eq!(private_plan().unwrap(), 11_698_176);
    assert!(
        main_admission::check_capacity(
            MAIN_BYTES,
            &[
                expected,
                PRIVATE_BYTES as u64,
                RUNTIME_BYTES,
                PUBLIC_SOURCE_BYTES as u64
            ]
        )
        .is_ok()
    );
}

#[test]
fn original_q2_fused_gate_up_rows_and_every_shard_boundary_are_exact() {
    assert_eq!((strata::GU_ROW_BYTES, strata::DOWN_ROW_BYTES), (720, 180));
    assert_eq!(
        (strata::GU_EXPERT_BYTES, strata::DOWN_EXPERT_BYTES),
        (921600, 460800)
    );
    assert_eq!(strata::expert_owned_bytes().unwrap(), 805_306_368);
    let ids = [511, 0, 127, 128, 255, 256, 383, 384, 32, 477];
    let mut matches = [0; k::RANKS];
    for index in 0..4 {
        let shard = strata::Shard::new(index).unwrap();
        assert_eq!(shard.first, index * 128);
        assert_eq!(shard.owned(true).unwrap(), 128 << 20);
        assert_eq!(shard.owned(false).unwrap(), 64 << 20);
        for (rank, id) in ids.into_iter().enumerate() {
            if id >= shard.first && id - shard.first < 128 {
                matches[rank] += 1;
                let local = id - shard.first;
                for row in [0, 17, 639] {
                    let gate = (local * 1280 + row) * 720;
                    let up = gate + 640 * 720;
                    assert_eq!(shard.first * 921600 + gate, (id * 1280 + row) * 720);
                    assert_eq!(shard.first * 921600 + up, (id * 1280 + 640 + row) * 720);
                    assert!(up + 720 <= shard.bytes(true).unwrap());
                }
                for row in [0, 43, 2559] {
                    let down = (local * 2560 + row) * 180;
                    assert_eq!(shard.first * 460800 + down, (id * 2560 + row) * 180);
                    assert!(down + 180 <= shard.bytes(false).unwrap());
                }
            }
        }
    }
    assert_eq!(matches, [1; k::RANKS]);
    assert!(strata::Shard::new(4).is_err());
    for first in [1, 129, 512, usize::MAX] {
        assert!(strata::Shard { first }.bytes(true).is_err());
        assert!(strata::Shard { first }.owned(false).is_err());
    }
}

const Q2_C: &[u8] =
    include_bytes!("../../../../../crates/kuno-text/tests/fixtures/daslab-b11232/q2_0.bin");
fn tree(mut lanes: [f32; 64], mut stride: usize) -> f32 {
    while stride != 0 {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
        stride /= 2;
    }
    lanes[0]
}

#[test]
fn real_width_q2_subgroup_dot_preserves_independent_c_decode_and_original_tree() {
    const RECORD: usize = 18 + 64 * 4;
    assert!(Q2_C.len().is_multiple_of(RECORD));
    let records: Vec<_> = Q2_C.chunks_exact(RECORD).collect();
    for columns in [640, 2560] {
        for salt in [0, 1, 17, 31] {
            let mut packed = Vec::new();
            let mut weights = Vec::new();
            for block in 0..columns / 64 {
                let record = records[(block + salt) % records.len()];
                packed.extend_from_slice(&record[..18]);
                weights.extend(
                    record[18..]
                        .chunks_exact(4)
                        .map(|word| f32::from_le_bytes(word.try_into().unwrap())),
                );
            }
            let mut scalar = vec![0.; columns];
            TensorType::Q2_0.decode(&packed, &mut scalar).unwrap();
            for (&a, &b) in scalar.iter().zip(&weights) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            let input: Vec<_> = (0..columns)
                .map(|i| ((i % 61) as f32 - 30.) / 64.)
                .collect();
            let mut original = [0.; 64];
            for i in 0..columns {
                original[i % 64] += weights[i] * input[i];
            }
            let mut subgroup = [0.; 64];
            for lane in 0..32 {
                let mut low = 0.;
                let mut high = 0.;
                for block in 0..columns / 64 {
                    low += scalar[block * 64 + lane] * input[block * 64 + lane];
                    high += scalar[block * 64 + lane + 32] * input[block * 64 + lane + 32];
                }
                subgroup[lane] = low + high;
            }
            // The low+high stride32 is already performed by the subgroup dot.
            assert_eq!(tree(original, 32).to_bits(), tree(subgroup, 16).to_bits());
        }
    }
}

#[test]
fn bf16_hidden_fc_has_four_disjoint_rhs_outputs_and_dead_embedding_scratch() {
    let geometry = matmul::MatrixGeometry::new(quant_matvec::BF16, 2560, 2560, 4).unwrap();
    let spans = matmul::BatchSpans::new(4, 0, 0, 2560, 0, 2560);
    let packed_words = 2560 * 2560 * 2 / 4;
    assert_eq!(geometry.row_bytes(), 5120);
    assert!(
        geometry
            .validate_spans(&spans, packed_words, 10240, 10240)
            .is_ok()
    );
    for (p, i, o) in [
        (packed_words - 1, 10240, 10240),
        (packed_words, 10239, 10240),
        (packed_words, 10240, 10239),
    ] {
        assert!(geometry.validate_spans(&spans, p, i, o).is_err());
    }
    assert_ne!(B::HiddenNorm as usize, B::Residual0 as usize);
    assert_ne!(B::EmbeddingNorm as usize, B::Embedding as usize);
    assert_ne!(B::Embedding as usize, B::Residual0 as usize);
    for stream in 0..4 {
        assert_eq!(stream * 2560 + 2559, (stream + 1) * 2560 - 1);
    }
}

#[test]
fn packed_source_preflight_covers_all_scale_offsets_and_bf16_extremes() {
    for (dtype, block_bytes, scale_at) in [
        (quant_matvec::Q2_0, 18, 0),
        (quant_matvec::IQ4_XS, 136, 0),
        (quant_matvec::Q6_K, 210, 208),
    ] {
        let mut source = vec![0; block_bytes * 41];
        for finite_half in [0_u16, 1, 0x03ff, 0x0400, 0x3555, 0x7bff, 0x8000, 0xfbff] {
            for block in [0, 1, 20, 40] {
                let at = block * block_bytes + scale_at;
                source[at..at + 2].copy_from_slice(&finite_half.to_le_bytes());
                assert!(validate_source(&source, dtype).is_ok());
            }
        }
        for half in [0x7c00_u16, 0xfc00, 0x7c01, 0x7fff, 0xffff] {
            for block in [0, 1, 20, 40] {
                let mut bad = source.clone();
                let at = block * block_bytes + scale_at;
                bad[at..at + 2].copy_from_slice(&half.to_le_bytes());
                assert!(validate_source(&bad, dtype).is_err());
            }
        }
        assert!(validate_source(&source[..source.len() - 1], dtype).is_err());
    }
    let good: Vec<_> = [0_u16, 1, 0x8000, 0x7f7f, 0xff7f, 0x3f80]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert!(validate_source(&good, quant_matvec::BF16).is_ok());
    for half in [0x7f80_u16, 0xff80, 0x7f81, 0xffff] {
        for position in 0..good.len() / 2 {
            let mut bad = good.clone();
            bad[position * 2..position * 2 + 2].copy_from_slice(&half.to_le_bytes());
            assert!(validate_source(&bad, quant_matvec::BF16).is_err());
        }
    }
    assert!(validate_source(&good[..good.len() - 1], quant_matvec::BF16).is_err());
}

fn validate_shader(builder: KernelBuilder, mode: ExecutionMode, cube: CubeDim) -> String {
    let source = WgslCompiler::default()
        .compile(
            builder.build(
                KernelSettings::default()
                    .cube_dim(cube)
                    .kernel_name("original_strata_mtp_leaf"),
            ),
            &WgpuCompilationOptions::default(),
            mode,
            AddressType::U32.unsigned_type(),
        )
        .unwrap()
        .to_string();
    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
    let info = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::SUBGROUP,
    )
    .validate(&module)
    .unwrap();
    wgpu::naga::back::spv::write_vec(
        &module,
        &info,
        &wgpu::naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        },
        Some(&wgpu::naga::back::spv::PipelineOptions {
            shader_stage: wgpu::naga::ShaderStage::Compute,
            entry_point: module.entry_points[0].name.clone(),
        }),
    )
    .unwrap();
    assert!(source.contains("atomicOr"));
    assert!(!source.contains("fma(") && !source.contains("f64"));
    source
}
fn builder() -> KernelBuilder {
    let mut b = KernelBuilder::default();
    b.runtime_properties(WgpuRuntime::target_properties());
    AddressType::U32.register(&mut b.scope);
    b
}

#[test]
fn original_q2_experts_iq4_table_and_broadcast_generate_valid_both_execution_modes() {
    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
        for which in 0..6 {
            let mut b = builder();
            let f = f32::as_type(&b.scope);
            let u = u32::as_type(&b.scope);
            let a = Atomic::<u32>::as_type(&b.scope);
            let cube = if which < 4 {
                fast::cube()
            } else {
                CubeDim::new_1d(64)
            };
            if which < 4 {
                let weights = b.input_array(u).into();
                let input = b.input_array(f).into();
                let packet = b.input_array(u).into();
                let output = b.output_array(f).into();
                let ready = b.output_array(u).into();
                let fault = b.output_array(a).into();
                let first = if which & 1 == 0 { 0 } else { 384 };
                if which < 2 {
                    strata_k::gate_up::expand(
                        &mut b.scope,
                        weights,
                        input,
                        packet,
                        output,
                        ready,
                        fault,
                        first,
                    );
                } else {
                    strata_k::down::expand(
                        &mut b.scope,
                        weights,
                        input,
                        packet,
                        output,
                        ready,
                        fault,
                        first,
                    );
                }
            } else if which == 4 {
                let weights = b.input_array(u).into();
                let token = b.input_array(u).into();
                let output = b.output_array(f).into();
                let fault = b.output_array(a).into();
                strata_k::embedding_tile::expand(
                    &mut b.scope,
                    weights,
                    token,
                    output,
                    fault,
                    245760,
                    2560,
                );
            } else {
                let embedding = b.input_array(f).into();
                let residual = b.output_array(f).into();
                let fault = b.output_array(a).into();
                strata_k::broadcast_embedding::expand(&mut b.scope, embedding, residual, fault);
            }
            let source = validate_shader(b, mode, cube);
            if which < 4 {
                assert!(source.contains("subgroupShuffleDown"));
            }
        }
    }
}

#[test]
fn new_complete_public_scale_guards_generate_valid_full_owner_sweeps() {
    fn constant(
        function: &wgpu::naga::Function,
        at: wgpu::naga::Handle<wgpu::naga::Expression>,
    ) -> u32 {
        use wgpu::naga::{Expression, Literal};
        match function.expressions[at] {
            Expression::Literal(Literal::U32(n)) => n,
            Expression::Literal(Literal::I32(n)) => u32::try_from(n).unwrap(),
            Expression::As { expr, .. } => constant(function, expr),
            Expression::Load { pointer } => {
                let Expression::LocalVariable(local) = function.expressions[pointer] else {
                    panic!("nonconstant fault flag")
                };
                constant(function, function.local_variables[local].init.unwrap())
            }
            _ => panic!("nonconstant fault flag"),
        }
    }
    fn faults(function: &wgpu::naga::Function, block: &wgpu::naga::Block, flags: &mut Vec<u32>) {
        use wgpu::naga::{AtomicFunction, Statement};
        for statement in block {
            match statement {
                Statement::Atomic { fun, value, .. } => {
                    assert_eq!(*fun, AtomicFunction::InclusiveOr);
                    flags.push(constant(function, *value));
                }
                Statement::If { accept, reject, .. } => {
                    faults(function, accept, flags);
                    faults(function, reject, flags);
                }
                Statement::Block(block) => faults(function, block, flags),
                Statement::Loop {
                    body, continuing, ..
                } => {
                    faults(function, body, flags);
                    faults(function, continuing, flags);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        faults(function, &case.body, flags);
                    }
                }
                _ => (),
            }
        }
    }
    for dtype in [
        quant_matvec::BF16,
        quant_matvec::Q2_0,
        quant_matvec::IQ4_XS,
        quant_matvec::Q6_K,
    ] {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = builder();
            let u = u32::as_type(&b.scope);
            let a = Atomic::<u32>::as_type(&b.scope);
            let values = b.input_array(u).into();
            let fault = b.output_array(a).into();
            let logical = usize::__expand_new(&mut b.scope, 33554432);
            public_guard::expand(&mut b.scope, values, fault, logical, dtype);
            let source = validate_shader(b, mode, CubeDim::new_1d(64));
            assert!(!source.contains("workgroupBarrier"));
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let function = &module.entry_points[0].function;
            let mut flags = Vec::new();
            faults(function, &function.body, &mut flags);
            flags.sort_unstable();
            assert_eq!(
                flags,
                [1, 4],
                "one finite-scale rejection plus complete zero-padding rejection"
            );
            let bound = module
                .global_variables
                .iter()
                .filter(|(_, v)| v.binding.is_some())
                .count();
            assert_eq!(
                bound, 3,
                "complete public source/fault owners plus launch INFO"
            );
        }
    }
}
