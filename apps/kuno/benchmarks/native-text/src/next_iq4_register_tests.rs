use super::*;
use crate::quant_matvec::{iq_fixtures, matmul::BatchSpans};
use std::collections::{HashMap, HashSet};
use wgpu::naga as n;

fn selected(dtype: u32, rows: usize, columns: usize) -> MatrixSpec {
    MatrixSpec::new(rows, columns, dtype)
        .unwrap()
        .with_iq4_decoder(Iq4Decoder::PackedRegisters)
        .unwrap()
}

/// Parse the actual shader's four scalar literal assignments, not a second
/// unchecked copy of those words. The independent C records remain the oracle.
fn lookup_words() -> [u32; 4] {
    let source = include_str!("next_iq4_register.wgsl");
    let words: Vec<_> = source
        .lines()
        .filter_map(|line| line.split_once("word=0x"))
        .map(|(_, literal)| u32::from_str_radix(literal.split_once('u').unwrap().0, 16).unwrap())
        .collect();
    words.try_into().unwrap()
}

fn lookup(index: usize, words: [u32; 4]) -> f32 {
    f32::from(((words[index / 4] >> ((index % 4) * 8)) & 255) as u8 as i8)
}

fn normalize_lookup(source: &str) -> String {
    let (prefix, table) = source.split_once("// Original GGML b11232").unwrap();
    let (_, suffix) = table
        .split_once("// Four natural-order original values.")
        .unwrap();
    format!("{prefix}// IQ4_LOOKUP\n// Four natural-order original values.{suffix}")
}

fn compile(kernel: Kernel, mode: ExecutionMode) -> n::Module {
    let compiled = kernel
        .compile(
            &mut AutoCompiler::Wgsl(cubecl::wgpu::WgslCompiler::default()),
            &cubecl::WgpuCompilationOptions::default(),
            mode,
            kernel.address_type(),
        )
        .unwrap();
    let module = n::front::wgsl::parse_str(&compiled.source).unwrap();
    let info = n::valid::Validator::new(
        n::valid::ValidationFlags::all(),
        n::valid::Capabilities::SUBGROUP | n::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    )
    .validate(&module)
    .unwrap();
    let words =
        n::back::spv::write_vec(&module, &info, &n::back::spv::Options::default(), None).unwrap();
    let mut arrays = HashSet::new();
    let mut pointers = HashMap::new();
    let mut variables = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        let opcode = words[at] & 65535;
        assert!(length > 0 && at + length <= words.len());
        let args = &words[at + 1..at + length];
        match opcode {
            28 | 29 => {
                arrays.insert(args[0]);
            }
            32 => {
                pointers.insert(args[0], args[2]);
            }
            59 => variables.push((args[0], args[2])),
            _ => {}
        }
        at += length;
    }
    assert!(
        variables
            .iter()
            .all(|(ty, storage)| { *storage != 7 || !arrays.contains(pointers.get(ty).unwrap()) })
    );
    for function in module
        .functions
        .iter()
        .map(|(_, function)| function)
        .chain(module.entry_points.iter().map(|entry| &entry.function))
    {
        for (_, variable) in function.local_variables.iter() {
            assert!(!matches!(
                module.types[variable.ty].inner,
                n::TypeInner::Array { .. }
            ));
        }
    }
    module
}

#[test]
fn iq4_register_policy_is_explicit_typed_default_off_and_pipeline_distinct() {
    for dtype in [0, 1, 6, 8, 12, 13, 14, 20, 23, 30] {
        let original = MatrixSpec::new(7, 256, dtype).unwrap();
        assert_eq!(original.iq4_decoder(), Iq4Decoder::Original);
        original.require_original_scalar().unwrap();
        assert_eq!(
            original.with_iq4_decoder(Iq4Decoder::Original).unwrap(),
            original
        );
        let alternate = original.with_iq4_decoder(Iq4Decoder::PackedRegisters);
        if !matches!(dtype, 20 | 23) {
            assert!(alternate.is_err());
            continue;
        }
        let alternate = alternate.unwrap();
        assert_eq!(alternate.iq4_decoder(), Iq4Decoder::PackedRegisters);
        assert!(alternate.require_original_scalar().is_err());
        assert_eq!(
            alternate.with_iq4_decoder(Iq4Decoder::Original).unwrap(),
            original
        );
        for (control, candidate) in [
            (
                Kernel::PublicValues(original),
                Kernel::PublicValues(alternate),
            ),
            (
                Kernel::MatrixVerify3(original),
                Kernel::MatrixVerify3(alternate),
            ),
            (
                Kernel::MatrixVerify3Finite(original, false),
                Kernel::MatrixVerify3Finite(alternate, false),
            ),
            (
                Kernel::MatrixVerify3Finite(original, true),
                Kernel::MatrixVerify3Finite(alternate, true),
            ),
        ] {
            assert_ne!(control.id(), candidate.id());
            assert_eq!(control.dimension(), candidate.dimension());
        }
    }
}

#[test]
fn iq4_register_shader_literals_cover_every_original_code_and_mixed_quartet() {
    // Independent original GGML signed codebook, not the runtime Rust decoder.
    let levels = [
        -127i8, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let words = lookup_words();
    let shader = include_str!("next_iq4_register.wgsl");
    for threshold in [4, 8, 12] {
        assert!(shader.contains(&format!("if index>={threshold}u")));
    }
    assert!(shader.contains("((index&3u)*8u)"));
    assert!(shader.contains("i32(byte)-select(0,256,byte>=128u)"));
    assert!(!shader.contains("array<"));
    for quartet in 0u32..65536 {
        for lane in 0..4 {
            let index = ((quartet >> (lane * 4)) & 15) as usize;
            assert_eq!(
                lookup(index, words).to_bits(),
                f32::from(levels[index]).to_bits()
            );
        }
    }
}

#[test]
fn iq4_register_all_pinned_c_records_keep_original_signed_zero_and_factor_bits() {
    let words = lookup_words();
    for (dtype, width, bytes) in [(20u32, 32usize, 18usize), (23, 256, 136)] {
        let (fixture, _) = iq_fixtures::records(dtype);
        let record_bytes = bytes + 4 * width;
        assert_eq!(fixture.len() % record_bytes, 0);
        let mut seen = [false; 16];
        for record in fixture.chunks_exact(record_bytes) {
            let packed = &record[..bytes];
            let d = kuno_text::quant::f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));
            for column in 0..width {
                let (factor, encoded, high) = if dtype == 20 {
                    (d, packed[2 + column % 16], column >= 16)
                } else {
                    let sub = column / 32;
                    let low = (packed[4 + sub / 2] >> ((sub % 2) * 4)) & 15;
                    let upper = (u16::from_le_bytes([packed[2], packed[3]]) >> (sub * 2)) & 3;
                    let scale = i32::from(low) | (i32::from(upper) << 4);
                    (
                        d * (scale - 32) as f32,
                        packed[8 + sub * 16 + column % 16],
                        column % 32 >= 16,
                    )
                };
                let code = usize::from(if high { encoded >> 4 } else { encoded & 15 });
                seen[code] = true;
                let actual = (factor * lookup(code, words)).to_bits();
                let at = bytes + column * 4;
                let expected = u32::from_le_bytes(record[at..at + 4].try_into().unwrap());
                assert_eq!(actual, expected, "dtype{dtype} column{column}");
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }
}

#[test]
fn iq4_register_changes_only_lookup_preserving_sources_math_faults_and_owner_guards() {
    for (dtype, columns) in [(20, 160), (20, 640), (20, 2560), (23, 256), (23, 2560)] {
        let original = MatrixSpec::new(7, columns, dtype).unwrap();
        let alternate = selected(dtype, 7, columns);
        for (control, candidate) in [
            (
                Kernel::PublicValues(original),
                Kernel::PublicValues(alternate),
            ),
            (
                Kernel::MatrixVerify3(original),
                Kernel::MatrixVerify3(alternate),
            ),
            (
                Kernel::MatrixVerify3Finite(original, false),
                Kernel::MatrixVerify3Finite(alternate, false),
            ),
            (
                Kernel::MatrixVerify3Finite(original, true),
                Kernel::MatrixVerify3Finite(alternate, true),
            ),
        ] {
            assert_eq!(
                normalize_lookup(&control.source()),
                normalize_lookup(&candidate.source())
            );
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let module = compile(candidate, mode);
                assert_eq!(
                    module.global_variables.len(),
                    if matches!(candidate, Kernel::PublicValues(_)) {
                        3
                    } else {
                        5
                    }
                );
            }
        }
        // Existing count1 scalar source, including routed namespacing input,
        // is never altered by this candidate's helper source builder.
        assert!(original.source().contains("levels=array<i32,16>"));
        assert!(!alternate.source_verify3().contains("levels=array<i32,16>"));
        assert!(!alternate.source_verify3().contains("dot4I8Packed"));
        assert!(!alternate.source_verify3().contains("fma("));
    }
}

#[test]
fn iq4_register_count1_to3_keep_original_max8_finite_tail_and_padding_admission() {
    for (dtype, columns) in [(20, 640), (23, 2560)] {
        let original = MatrixSpec::new(7, columns, dtype).unwrap();
        let alternate = selected(dtype, 7, columns);
        let bytes = (original.rows * original.row_bytes).div_ceil(4) as u64 * 4;
        let input_words = 4 + 8 * (columns + 12);
        let output_words = 7 + 8 * 29;
        for count in 1..=3 {
            let spans = BatchSpans::new(count, 0, 4, columns + 12, 17, 29);
            for spec in [original, alternate] {
                validate_verify3_spans(spec, &spans, bytes, input_words, output_words).unwrap();
                for (rows, tail) in [(17, true), (18, false)] {
                    validate_verify3_finite_spans(
                        spec,
                        &spans,
                        bytes,
                        input_words,
                        output_words,
                        rows,
                        7,
                        tail,
                    )
                    .unwrap();
                    assert!(
                        validate_verify3_finite_spans(
                            spec,
                            &spans,
                            bytes,
                            input_words,
                            output_words,
                            rows,
                            7,
                            !tail,
                        )
                        .is_err()
                    );
                }
                assert!(
                    validate_verify3_spans(spec, &spans, bytes - 4, input_words, output_words,)
                        .is_err()
                );
                assert!(
                    validate_verify3_spans(
                        spec,
                        &spans,
                        bytes,
                        4 + (count - 1) * (columns + 12) + columns - 1,
                        output_words,
                    )
                    .is_err()
                );
                assert!(
                    validate_verify3_spans(
                        spec,
                        &spans,
                        bytes,
                        input_words,
                        17 + (count - 1) * 29 + 6,
                    )
                    .is_err()
                );
            }
        }
        for count in [0, 4, 8, usize::MAX] {
            let spans = BatchSpans::new(count, 0, 4, columns + 12, 17, 29);
            assert!(
                validate_verify3_spans(alternate, &spans, bytes, input_words, output_words,)
                    .is_err()
            );
        }
        assert_eq!(
            verify3_finite_grid(original, output_words, true),
            verify3_finite_grid(alternate, output_words, true)
        );
    }
}
