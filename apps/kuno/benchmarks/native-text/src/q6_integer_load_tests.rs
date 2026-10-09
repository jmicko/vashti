//! Fixed public tests of original-byte vector loading, independently against
//! pinned C values and natural-order codes. No model payload/device is used.
use super::{Kernel, ProjectSpans, Spec, oracle, wide_weight_view};

fn aligned(wire: &[u8], first: usize) -> [u32; 4] {
    assert!(first.is_multiple_of(16));
    std::array::from_fn(|i| {
        u32::from_le_bytes(wire[first + i * 4..first + i * 4 + 4].try_into().unwrap())
    })
}
fn words_at(wire: &[u8], at: usize) -> [u32; 4] {
    let low = aligned(wire, at / 16 * 16);
    let high = if at.is_multiple_of(16) {
        low
    } else {
        aligned(wire, at / 16 * 16 + 16)
    };
    let offset = at % 16 / 4;
    let joined = [
        low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
    ];
    let shift = (at % 4) * 8;
    std::array::from_fn(|i| {
        let word = joined[offset + i];
        if shift == 0 {
            word
        } else {
            word >> shift | joined[offset + i + 1] << (32 - shift)
        }
    })
}
fn q6_words(wire: &[u8], base: usize, sub: usize) -> [u32; 4] {
    let half = sub / 8;
    let quarter = sub % 8 / 2;
    let offset = sub % 2 * 16;
    let low = words_at(wire, base + half * 64 + quarter % 2 * 32 + offset);
    let high = words_at(wire, base + 128 + half * 32 + offset);
    std::array::from_fn(|i| {
        let nibble = if quarter < 2 {
            low[i] & 0x0f0f_0f0f
        } else {
            low[i] >> 4 & 0x0f0f_0f0f
        };
        let codes = nibble | ((high[i] >> (quarter * 2) & 0x0303_0303) << 4);
        codes.wrapping_add(0x6060_6060) ^ 0x8080_8080
    })
}

#[test]
fn every_byte_alignment_and_original_block_offset_matches_pinned_c_bits() {
    let records: Vec<_> = oracle::FIXTURE
        .chunks_exact(oracle::RECORD_BYTES)
        .take(25)
        .collect();
    let mut count = 0;
    for prefix in 0..16 {
        let bytes = prefix + records.len() * 210;
        let mut wire = vec![0xa5; bytes.next_multiple_of(16)];
        for (block, record) in records.iter().enumerate() {
            let base = prefix + block * 210;
            wire[base..base + 210].copy_from_slice(&record[..210]);
        }
        for (block, record) in records.iter().enumerate() {
            let base = prefix + block * 210;
            for sub in 0..16 {
                let (natural, factor) = oracle::unpack16(&record[..210], 256, sub).unwrap();
                let loaded = q6_words(&wire, base, sub);
                for (word, packed) in loaded.into_iter().enumerate() {
                    for (lane, code) in packed.to_le_bytes().into_iter().enumerate() {
                        let i = word * 4 + lane;
                        assert_eq!(code as i8, natural[i]);
                        let at = 210 + (sub * 16 + i) * 4;
                        let c = u32::from_le_bytes(record[at..at + 4].try_into().unwrap());
                        assert_eq!((factor * f32::from(code as i8)).to_bits(), c);
                        count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(count, 102_400);
}

#[test]
fn wide_reads_never_exceed_checked_original_source_and_partial_views_stay_scalar() {
    // Exhaust offsets, first/last blocks and the exact head's original2100-byte
    // row. Wide ql/qh loads can consume immutable prefix/scales, never a byte
    // after the checked block; a second vector is absent for aligned runs.
    for prefix in 0..16 {
        for (rows, columns) in [
            (1, 256),
            (7, 256),
            (7, 512),
            (7, 2560),
            (2560, 2560),
            (16384, 2560),
        ] {
            let spec = Spec::new(rows, columns).unwrap();
            let end = prefix + spec.weight_bytes();
            let bytes = end.next_multiple_of(16);
            assert!(wide_weight_view(bytes / 4));
            let p = ProjectSpans {
                packed_first: prefix,
                planes_first: 0,
                output_first: 0,
                fault_first: 0,
                matrix_first: 0,
                matrix_rows: rows,
            };
            p.validate(spec, bytes, [spec.plane_words(), rows, 1])
                .unwrap();
            assert!(
                p.validate(spec, end - 1, [spec.plane_words(), rows, 1])
                    .is_err()
            );
            let blocks = spec.weight_bytes() / 210;
            for block in [0, blocks - 1] {
                let base = prefix + block * 210;
                for sub in 0..16 {
                    let half = sub / 8;
                    let quarter = sub % 8 / 2;
                    let offset = sub % 2 * 16;
                    for at in [
                        base + half * 64 + quarter % 2 * 32 + offset,
                        base + 128 + half * 32 + offset,
                    ] {
                        let read_end = at / 16 * 16 + if at.is_multiple_of(16) { 16 } else { 32 };
                        assert!(read_end <= base + 210);
                        assert!(read_end <= end);
                    }
                    // Scalar metadata bytes occupy complete vec4s only after
                    // the unchanged full source extent passed selection.
                    for at in [base + 192 + sub, base + 208, base + 209] {
                        assert!(at / 16 * 16 + 16 <= bytes);
                    }
                }
            }
        }
    }
    for words in [0, 1, 2, 3, 53, 367, 3681] {
        assert!(!wide_weight_view(words));
    }
    for words in [4, 368, 8_601_600, 1_344_000] {
        assert!(wide_weight_view(words));
    }
}

#[test]
fn scalar_and_vector_sources_keep_identical_math_grid_fault_and_tail_body() {
    let spec = Spec::new(16384, 2560).unwrap();
    for tail in [false, true] {
        let scalar = Kernel::Project(spec, tail, false).source();
        let vector = Kernel::Project(spec, tail, true).source();
        let body = "fn byte_at";
        assert_eq!(
            scalar.split_once(body).unwrap().1,
            vector.split_once(body).unwrap().1
        );
        assert!(vector.contains("if at%16u!=0u {high=weight[at/16u+1u];}"));
        assert!(!vector.contains("array<u32,4>"));
        assert!(!vector.contains("var<workgroup>"));
        assert!(!vector.contains("q6_word("));
    }
}

#[test]
fn offline_spirv_retains_vector_weight_loads_without_function_arrays() {
    let spec = Spec::new(16384, 2560).unwrap();
    for wide in [false, true] {
        let module =
            wgpu::naga::front::wgsl::parse_str(&Kernel::Project(spec, true, wide).source())
                .unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let spirv = wgpu::naga::back::spv::write_vec(
            &module,
            &info,
            &wgpu::naga::back::spv::Options {
                lang_version: (1, 3),
                ..Default::default()
            },
            Some(&wgpu::naga::back::spv::PipelineOptions {
                shader_stage: wgpu::naga::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        let mut uints = std::collections::HashSet::new();
        let mut vectors = std::collections::HashSet::new();
        let mut vector_loads = 0;
        let mut function_arrays = std::collections::HashSet::new();
        let mut arrays = std::collections::HashSet::new();
        let mut at = 5;
        while at < spirv.len() {
            let length = (spirv[at] >> 16) as usize;
            let opcode = spirv[at] & 65535;
            assert!(length > 0 && at + length <= spirv.len());
            let args = &spirv[at + 1..at + length];
            match opcode {
                21 if args[1] == 32 && args[2] == 0 => {
                    uints.insert(args[0]);
                }
                23 if args[2] == 4 && uints.contains(&args[1]) => {
                    vectors.insert(args[0]);
                }
                28 | 29 => {
                    arrays.insert(args[0]);
                }
                32 if args[1] == 7 && arrays.contains(&args[2]) => {
                    function_arrays.insert(args[0]);
                }
                59 => {
                    assert!(!function_arrays.contains(&args[0]));
                }
                61 if vectors.contains(&args[0]) => {
                    vector_loads += 1;
                }
                _ => {}
            }
            at += length;
        }
        assert_eq!(vector_loads > 0, wide);
    }
}
