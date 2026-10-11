//! Fixed-public source/host proofs; no device/model construction or tolerance change.
use super::*;
use crate::quant_matvec::matmul::{BatchSpans, MAX_BATCH};

const TYPES: [u32; 10] = [0, 1, 30, 6, 8, 12, 13, 14, 20, 23];

fn weight_bytes(spec: MatrixSpec) -> u64 {
    (spec.rows * spec.row_bytes).div_ceil(4) as u64 * 4
}

#[test]
fn fixed_four_requires_all_real_rows_and_original_decoder_in_full_max8_spans() {
    for dtype in TYPES {
        let spec = MatrixSpec::new(9, 256, dtype).unwrap();
        let (input_stride, output_stride) = (268, 20);
        let (input_words, output_words) =
            (4 + MAX_BATCH * input_stride, 7 + MAX_BATCH * output_stride);
        let full = BatchSpans::new(4, 0, 4, input_stride, 7, output_stride);
        validate_verify4_spans(spec, &full, weight_bytes(spec), input_words, output_words).unwrap();
        assert!(
            validate_verify3_spans(spec, &full, weight_bytes(spec), input_words, output_words)
                .is_err()
        );
        for count in [0, 1, 2, 3, 5, 8, usize::MAX] {
            let bad = BatchSpans::new(count, 0, 4, input_stride, 7, output_stride);
            assert!(
                validate_verify4_spans(spec, &bad, weight_bytes(spec), input_words, output_words)
                    .is_err()
            );
        }
        for bad in [
            BatchSpans::new(4, 4, 4, input_stride, 7, output_stride),
            BatchSpans::new(4, 0, 3, input_stride, 7, output_stride),
            BatchSpans::new(4, 0, 4, input_stride + 1, 7, output_stride),
            BatchSpans::new(4, 0, 4, 252, 7, output_stride),
            BatchSpans::new(4, 0, 4, input_stride, 7, spec.rows - 1),
            BatchSpans::new(4, 0, usize::MAX - 3, input_stride, 7, output_stride),
            BatchSpans::new(4, 0, 4, input_stride, usize::MAX, output_stride),
            BatchSpans::new(4, 0, 4, u32::MAX as usize + 1, 7, output_stride),
        ] {
            assert!(
                validate_verify4_spans(spec, &bad, weight_bytes(spec), input_words, output_words)
                    .is_err()
            );
        }
        for (bytes, inputs, outputs) in [
            (weight_bytes(spec) - 4, input_words, output_words),
            (
                weight_bytes(spec),
                4 + 3 * input_stride + spec.columns - 4,
                output_words,
            ),
            (weight_bytes(spec), input_words - 1, output_words),
            (
                weight_bytes(spec),
                input_words,
                7 + 3 * output_stride + spec.rows - 1,
            ),
        ] {
            assert!(validate_verify4_spans(spec, &full, bytes, inputs, outputs).is_err());
        }
        if matches!(dtype, 20 | 23) {
            let selected = spec.with_iq4_decoder(Iq4Decoder::PackedRegisters).unwrap();
            assert!(
                validate_verify4_spans(
                    selected,
                    &full,
                    weight_bytes(spec),
                    input_words,
                    output_words
                )
                .is_err()
            );
        }
    }
}

#[test]
fn only_fourth_component_and_fixed_guard_extend_the_unchanged_verify3_producer() {
    let original = include_str!("next_verify3_dense.wgsl")
        .split_once("fn verify3_dot")
        .unwrap()
        .1;
    let candidate = include_str!("next_verify4_dense.wgsl")
        .split_once("fn verify4_dot")
        .unwrap()
        .1;
    // Remove precisely the fourth RHS expressions. All decoder/address/factor/
    // dot/accumulation/reduction and first-three store expressions must match.
    let recovered = candidate
        .replace("column4:u32)->vec4<f32>", "column4:u32)->vec3<f32>")
        .replace("var result=vec4<f32>(0.0)", "var result=vec3<f32>(0.0)")
        .replace("var sum=vec4<f32>(0.0)", "var sum=vec3<f32>(0.0)")
        .replace("    result.w=dot(w,input4[column4+3u*(p.input_stride/4u)]);\n", "")
        .replace("        output[out+3u*p.output_stride]=total.w;\n        if !(abs(total.w)<=3.4028234663852886e38) {atomicOr(&fault[0],1u);}\n", "")
        .replace("p.count==4u", "p.count>=1u && p.count<=3u")
        .replace("verify4_dot", "verify3_dot")
        .replace("VERIFY4_", "VERIFY3_");
    assert_eq!(recovered, original);
    for dtype in TYPES {
        let spec = MatrixSpec::new(9, 2560, dtype).unwrap();
        let old = spec.source_verify3();
        let new = spec.source_verify4();
        assert_eq!(
            old.split_once("fn verify3_dot").unwrap().0,
            new.split_once("fn verify4_dot")
                .unwrap()
                .0
                .replace("Verify4 leaf", "Verify3 candidate")
                .replace("four-column", "three-column")
        );
        for tail in [false, true] {
            let fused = spec.source_verify4_finite(tail);
            let start = "    let row=group.x*(128u/width)+sg;";
            let end = "    // VERIFY4_LOGICAL_COMPLEMENT_SCAN";
            assert_eq!(
                fused
                    .split_once(start)
                    .unwrap()
                    .1
                    .split_once(end)
                    .unwrap()
                    .0,
                new.split_once(start).unwrap().1.split_once(end).unwrap().0
            );
            assert!(fused.contains("p.output_words==arrayLength(&output)"));
            assert!(fused.contains("p.output_stride>=p.matrix_rows"));
            assert_eq!(fused.contains("i+=128u*dispatched.x"), tail);
        }
    }
}

#[test]
fn actual_checked_unchecked_all_format_sources_lower_with_runtime_none_and_no_new_owners() {
    use wgpu::naga as n;
    for dtype in TYPES {
        for columns in [256, 2560] {
            let spec = MatrixSpec::new(9, columns, dtype).unwrap();
            for kernel in [
                Kernel::MatrixVerify4(spec),
                Kernel::MatrixVerify4Finite(spec, false),
                Kernel::MatrixVerify4Finite(spec, true),
            ] {
                assert_ne!(kernel.id(), Kernel::MatrixVerify3(spec).id());
                assert_ne!(kernel.id(), Kernel::MatrixVerify3Finite(spec, false).id());
                assert_ne!(kernel.id(), Kernel::MatrixVerify3Finite(spec, true).id());
                assert_eq!(kernel.dimension(), CubeDim::new_1d(128));
                for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                    let compiled = kernel
                        .compile(
                            &mut AutoCompiler::Wgsl(cubecl::wgpu::WgslCompiler::default()),
                            &cubecl::WgpuCompilationOptions::default(),
                            mode,
                            kernel.address_type(),
                        )
                        .unwrap();
                    let source = &compiled.source;
                    assert!(source.contains("p.count==4u"));
                    assert!(source.contains("let total=subgroupAdd(sum)"));
                    assert!(
                        source.contains("result.w=dot(w,input4[column4+3u*(p.input_stride/4u)])")
                    );
                    assert!(source.contains("output[out+3u*p.output_stride]=total.w"));
                    assert!(
                        source.find("if !valid").unwrap()
                            < source.find("let base=row*ROW_BYTES").unwrap()
                    );
                    assert!(!source.contains("workgroupBarrier"));
                    assert!(!source.contains("var<workgroup>"));
                    assert!(!source.contains("dot4I8Packed"));
                    assert!(!source.contains("fma("));
                    assert!(!source.contains("atomicStore"));
                    let module = n::front::wgsl::parse_str(source).unwrap();
                    let info = n::valid::Validator::new(
                        n::valid::ValidationFlags::all(),
                        n::valid::Capabilities::SUBGROUP
                            | n::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
                    )
                    .validate(&module)
                    .unwrap();
                    let options = n::back::spv::Options {
                        zero_initialize_workgroup_memory:
                            n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                        ..Default::default()
                    };
                    let spv = n::back::spv::write_vec(&module, &info, &options, None).unwrap();
                    assert!(!spv.is_empty());
                    assert_eq!(module.global_variables.len(), 5);
                    assert!(
                        module
                            .global_variables
                            .iter()
                            .all(|(_, var)| var.space != n::AddressSpace::WorkGroup)
                    );
                }
            }
        }
    }
}

#[test]
fn subgroup_grids_write_four_rhs_once_preserving_every_max8_gap_and_inactive_row() {
    for width in [16, 32] {
        for rows in [1usize, 7, 8, 9, 2560, 16384] {
            let (first, stride) = (7, rows + 11);
            let mut writes = vec![0u8; first + MAX_BATCH * stride + 13];
            for group in 0..rows.div_ceil(4) {
                for subgroup in 0..128 / width {
                    let row = group * (128 / width) + subgroup;
                    if row < rows {
                        for rhs in 0..4 {
                            writes[first + rhs * stride + row] += 1;
                        }
                    }
                }
            }
            for (i, &count) in writes.iter().enumerate() {
                let active = i >= first && (i - first) / stride < 4 && (i - first) % stride < rows;
                assert_eq!(count, u8::from(active));
            }
        }
    }
}

#[test]
fn complete_finite_interval_rejects_wrong_tiles_and_scans_only_full_matrix_complement() {
    let spec = MatrixSpec::new(7, 256, 14).unwrap();
    let spans = BatchSpans::new(4, 0, 4, 268, 17, 29);
    let (inputs, outputs) = (4 + MAX_BATCH * 268, 7 + MAX_BATCH * 29);
    for (rows, first, tail) in [(17, 7, true), (18, 7, false)] {
        validate_verify4_finite_spans(
            spec,
            &spans,
            weight_bytes(spec),
            inputs,
            outputs,
            rows,
            first,
            tail,
        )
        .unwrap();
    }
    for (rows, first, tail) in [
        (17, 7, false),
        (18, 7, true),
        (16, 7, true),
        (0, 7, false),
        (30, 7, false),
        (17, 18, true),
        (17, usize::MAX, true),
        (usize::MAX, 7, true),
    ] {
        assert!(
            validate_verify4_finite_spans(
                spec,
                &spans,
                weight_bytes(spec),
                inputs,
                outputs,
                rows,
                first,
                tail
            )
            .is_err()
        );
    }
    assert!(
        validate_verify4_finite_spans(
            spec,
            &spans,
            weight_bytes(spec),
            inputs,
            VERIFY3_FINITE_WORD_LIMIT + 1,
            17,
            7,
            true
        )
        .is_err()
    );
    for (matrix_rows, first, gap) in [(17usize, 7usize, 11usize), (248320, 3, 9)] {
        let stride = matrix_rows + gap;
        let words = first + MAX_BATCH * stride + 13;
        let mut producer = vec![false; words];
        for rhs in 0..4 {
            for row in 0..matrix_rows {
                producer[first + rhs * stride + row] = true;
            }
        }
        for tile_rows in [1usize, 7, 2560, 16384] {
            let tile_rows = tile_rows.min(matrix_rows);
            let tile = MatrixSpec::new(tile_rows, 256, 14).unwrap();
            let groups = verify4_finite_grid(tile, words, true);
            assert!((1..=4096).contains(&groups));
            assert_eq!(
                verify4_finite_grid(tile, words, false),
                tile_rows.div_ceil(4)
            );
            let mut seen = vec![0u8; words];
            for group in 0..groups {
                for local in 0..128 {
                    for i in (group * 128 + local..words).step_by(128 * groups) {
                        let active = i >= first
                            && (i - first) / stride < 4
                            && (i - first) % stride < matrix_rows;
                        if !active {
                            seen[i] += 1;
                        }
                    }
                }
            }
            for (i, &visits) in seen.iter().enumerate() {
                assert_eq!(visits, u8::from(!producer[i]));
            }
        }
    }
}

#[test]
fn four_active_self_checks_and_complement_preserve_sticky_fault_over_all_logical_regions() {
    let (rows, first, stride, words) = (17usize, 7usize, 29usize, 7 + MAX_BATCH * 29 + 13);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in 0..words {
            let mut values = vec![0.25f32; words];
            values[index] = bad;
            let mut full = 4u32;
            let mut fused = 4u32;
            for &value in &values {
                if !value.is_finite() {
                    full |= 1;
                }
            }
            for rhs in 0..4 {
                for row in 0..rows {
                    if !values[first + rhs * stride + row].is_finite() {
                        fused |= 1;
                    }
                }
            }
            for (i, &value) in values.iter().enumerate() {
                let active = i >= first && (i - first) / stride < 4 && (i - first) % stride < rows;
                if !active && !value.is_finite() {
                    fused |= 1;
                }
            }
            assert_eq!(full, fused);
            assert_eq!(fused, 5);
        }
    }
}

// Source-order CPU replay only. subgroupAdd's physical driver lowering/order
// remains a GPU acceptance requirement; no new GPU numerical claim is made.
fn dot4(w: [f32; 4], x: &[f32]) -> f32 {
    ((w[0] * x[0] + w[1] * x[1]) + w[2] * x[2]) + w[3] * x[3]
}
fn q8_quartet(packed: &[u8], row_base: usize, block: usize, at: usize) -> ([f32; 4], f32) {
    let base = row_base + block * 34;
    let scale = kuno_text::quant::f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
    (
        std::array::from_fn(|j| packed[base + 2 + at + j] as i8 as f32),
        scale,
    )
}
fn fold(mut partial: Vec<f32>) -> f32 {
    let mut step = partial.len() / 2;
    while step != 0 {
        for lane in 0..step {
            partial[lane] += partial[lane + step];
        }
        step /= 2;
    }
    partial[0]
}

#[test]
fn independent_f32_q8_scalar_and_interleaved_four_rhs_schedules_match_bits() {
    for (dtype, columns) in [(0, 256usize), (0, 2560), (8, 256), (8, 2560)] {
        let spec = MatrixSpec::new(9, columns, dtype).unwrap();
        let mut packed = vec![0u8; spec.rows * spec.row_bytes];
        if dtype == 0 {
            for (i, bytes) in packed.chunks_exact_mut(4).enumerate() {
                bytes.copy_from_slice(&(((i * 17 % 127) as f32 - 63.0) / 64.0).to_le_bytes());
            }
        } else {
            for (block, bytes) in packed.chunks_exact_mut(34).enumerate() {
                let scale = 0x2400u16 + (block % 512) as u16;
                bytes[..2].copy_from_slice(&scale.to_le_bytes());
                for (i, code) in bytes[2..].iter_mut().enumerate() {
                    *code = (block * 37 + i * 29 + 11) as u8;
                }
            }
        }
        let stride = columns + 12;
        let inputs: Vec<f32> = (0..4 + MAX_BATCH * stride)
            .map(|i| ((i * 23 % 257) as f32 - 128.0) / 129.0)
            .collect();
        for width in [16usize, 32] {
            for row in 0..spec.rows {
                let mut batched = vec![[0.0f32; 4]; width];
                for (lane, partial) in batched.iter_mut().enumerate() {
                    if dtype == 0 {
                        for col in (lane * 4..columns).step_by(width * 4) {
                            let at = row * spec.row_bytes + col * 4;
                            let w = std::array::from_fn(|j| {
                                f32::from_le_bytes(
                                    packed[at + j * 4..at + (j + 1) * 4].try_into().unwrap(),
                                )
                            });
                            for (rhs, value) in partial.iter_mut().enumerate() {
                                *value += dot4(w, &inputs[4 + rhs * stride + col..]);
                            }
                        }
                    } else {
                        for group in (0..columns / 32).step_by(width / 8) {
                            let block = group + lane / 8;
                            if block < columns / 32 {
                                let col = block * 32 + (lane % 8) * 4;
                                let (q, d) = q8_quartet(
                                    &packed,
                                    row * spec.row_bytes,
                                    block,
                                    (lane % 8) * 4,
                                );
                                for (rhs, value) in partial.iter_mut().enumerate() {
                                    *value += d * dot4(q, &inputs[4 + rhs * stride + col..]);
                                }
                            }
                        }
                    }
                }
                for rhs in 0..4 {
                    let mut scalar = vec![0.0f32; width];
                    for (lane, partial) in scalar.iter_mut().enumerate() {
                        if dtype == 0 {
                            for col in (lane * 4..columns).step_by(width * 4) {
                                let x = &inputs[4 + rhs * stride + col..];
                                let mut products = [0.0; 4];
                                for j in 0..4 {
                                    let at = row * spec.row_bytes + (col + j) * 4;
                                    products[j] =
                                        f32::from_le_bytes(packed[at..at + 4].try_into().unwrap())
                                            * x[j];
                                }
                                *partial +=
                                    ((products[0] + products[1]) + products[2]) + products[3];
                            }
                        } else {
                            for group in (0..columns / 32).step_by(width / 8) {
                                let block = group + lane / 8;
                                if block < columns / 32 {
                                    let base = row * spec.row_bytes + block * 34;
                                    let d = kuno_text::quant::f16_to_f32(u16::from_le_bytes(
                                        packed[base..base + 2].try_into().unwrap(),
                                    ));
                                    let at = (lane % 8) * 4;
                                    let x = &inputs[4 + rhs * stride + block * 32 + at..];
                                    let mut products = [0.0; 4];
                                    for j in 0..4 {
                                        products[j] = packed[base + 2 + at + j] as i8 as f32 * x[j];
                                    }
                                    *partial += d
                                        * (((products[0] + products[1]) + products[2])
                                            + products[3]);
                                }
                            }
                        }
                    }
                    let candidate = fold(batched.iter().map(|v| v[rhs]).collect());
                    assert_eq!(
                        candidate.to_bits(),
                        fold(scalar).to_bits(),
                        "dtype{dtype} width{width} row{row} rhs{rhs}"
                    );
                }
            }
        }
    }
}

#[test]
fn limits_and_info_records_retain_existing_ceiling_and_pipeline_separation() {
    let mut limits = wgpu::Limits::default();
    for (min, max) in [(16, 16), (16, 32), (32, 32)] {
        validate_verify4_limits(&limits, true, min, max).unwrap();
        validate_verify4_finite_limits(&limits, true, min, max).unwrap();
    }
    assert!(validate_verify4_limits(&limits, false, 16, 32).is_err());
    assert!(validate_verify4_limits(&limits, true, 8, 32).is_err());
    limits.max_uniform_buffer_binding_size = 32;
    validate_verify4_limits(&limits, true, 16, 32).unwrap();
    assert!(validate_verify4_finite_limits(&limits, true, 16, 32).is_err());
    limits.max_uniform_buffer_binding_size = 48;
    validate_verify4_finite_limits(&limits, true, 16, 32).unwrap();
    assert_eq!(48usize.div_ceil(2048), 32usize.div_ceil(2048));
    let spec = MatrixSpec::new(9, 256, 8).unwrap();
    assert_eq!(Kernel::MatrixVerify4(spec).name(), "NextVerify4Dense");
    assert_eq!(
        Kernel::MatrixVerify4Finite(spec, true).name(),
        "NextVerify4DenseFinite"
    );
    assert_ne!(
        Kernel::MatrixVerify4(spec).id(),
        Kernel::MatrixVerify4Finite(spec, false).id()
    );
    assert_ne!(
        Kernel::MatrixVerify4Finite(spec, false).id(),
        Kernel::MatrixVerify4Finite(spec, true).id()
    );
}

#[test]
fn original_kernel_hash_payloads_keep_all_preexisting_discriminants() {
    use std::hash::{Hash, Hasher};
    #[derive(Hash)]
    enum OriginalKernel {
        PublicValues(MatrixSpec),
        Matrix(MatrixSpec),
        MatrixFault(MatrixSpec, bool),
        MatrixVerify3(MatrixSpec),
        MatrixVerify3Finite(MatrixSpec, bool),
        Router,
        Gdn([usize; 3]),
        GrNorm(usize),
    }
    fn hash(value: &impl Hash) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }
    let spec = MatrixSpec::new(9, 256, 8).unwrap();
    for (old, actual) in [
        (
            OriginalKernel::PublicValues(spec),
            Kernel::PublicValues(spec),
        ),
        (OriginalKernel::Matrix(spec), Kernel::Matrix(spec)),
        (
            OriginalKernel::MatrixFault(spec, false),
            Kernel::MatrixFault(spec, false),
        ),
        (
            OriginalKernel::MatrixVerify3(spec),
            Kernel::MatrixVerify3(spec),
        ),
        (
            OriginalKernel::MatrixVerify3Finite(spec, true),
            Kernel::MatrixVerify3Finite(spec, true),
        ),
        (OriginalKernel::Router, Kernel::Router),
        (
            OriginalKernel::Gdn([128, 16, 32]),
            Kernel::Gdn([128, 16, 32]),
        ),
        (OriginalKernel::GrNorm(2560), Kernel::GrNorm(2560)),
    ] {
        assert_eq!(hash(&old), hash(&actual));
    }
}
