use super::*;

#[test]
fn original_wide_lane_fields_cover_canonical_gguf_values_and_all_rhs() {
    // Independent GGUF decoding is the oracle, not the old generic GPU kernel.
    // Public synthetic packed bytes exercise signed codes, high scale bits,
    // minima, both halves and partial final block waves for each native shape.
    for dtype in [8, 12, 13, 20, 23] {
        let (values, bytes) = crate::quant_matvec::block_layout(dtype).unwrap();
        for blocks in [1, 2, 3, 4, 5, 10, 15, 16, 17, 31, 32, 33] {
            let mut packed = vec![0u8; blocks * bytes];
            for (block, b) in packed.chunks_exact_mut(bytes).enumerate() {
                for (i, v) in b.iter_mut().enumerate() {
                    *v = (i * 73 + block * 29 + 11) as u8;
                }
                b[..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
                if matches!(dtype, 12 | 13) {
                    b[2..4].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                }
            }
            let ty = match dtype {
                8 => kuno_text::TensorType::Q8_0,
                12 => kuno_text::TensorType::Q4K,
                13 => kuno_text::TensorType::Q5K,
                20 => kuno_text::TensorType::Iq4Nl,
                23 => kuno_text::TensorType::Iq4Xs,
                _ => unreachable!(),
            };
            let mut canonical = vec![0.0f32; blocks * values];
            ty.decode(&packed, &mut canonical).unwrap();
            let lanes = if dtype == 8 {
                2
            } else if dtype == 20 {
                1
            } else {
                8
            };
            let mut visits = vec![0u8; canonical.len()];
            let mut sums = [[0f32; 8]; 32];
            for lane in 0..32 {
                let g = lane % lanes;
                for block in (lane / lanes..blocks).step_by(32 / lanes) {
                    let b = &packed[block * bytes..(block + 1) * bytes];
                    for part in 0..if dtype == 8 { 16 } else { 32 } {
                        let (offset, weight) = wide_value(dtype, b, g, part);
                        let column = block * values + offset;
                        assert_eq!(
                            weight.to_bits(),
                            canonical[column].to_bits(),
                            "dtype={dtype} block={block} lane={lane} part={part}"
                        );
                        visits[column] += 1;
                        for member in 0..8 {
                            let x = input_code(column, member) as f32 / 32.0;
                            sums[lane][member] += weight * x;
                        }
                    }
                }
            }
            assert!(visits.iter().all(|&n| n == 1));
            for member in 0..8 {
                // Powers of two keep the independent sums exact here. This
                // tests field/column/RHS ownership, not GPU rounding parity.
                let expected: f32 = canonical
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| w * input_code(i, member) as f32 / 32.0)
                    .sum();
                let actual: f32 = sums.iter().map(|s| s[member]).sum();
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}

fn input_code(column: usize, member: usize) -> i8 {
    (column * 13 + member * 31 + 7) as u8 as i8
}

fn scale_min(b: &[u8], sub: usize, minimum: bool) -> f32 {
    let sc = &b[4..16];
    let value = if sub < 4 {
        sc[sub + if minimum { 4 } else { 0 }] & 63
    } else if minimum {
        (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4)
    } else {
        (sc[sub + 4] & 15) | ((sc[sub - 4] >> 6) << 4)
    };
    value as f32
}

fn wide_value(dtype: u32, b: &[u8], g: usize, part: usize) -> (usize, f32) {
    const LUT: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    if dtype == 8 {
        let column = g * 16 + part;
        return (column, 0.25 * b[2 + column] as i8 as f32);
    }
    if matches!(dtype, 20 | 23) {
        let sub = if dtype == 23 { g } else { 0 };
        let start = if dtype == 23 { 8 + g * 16 } else { 2 };
        let code = (b[start + part % 16] >> (4 * (part / 16))) & 15;
        let scale = if dtype == 23 {
            let hi = u16::from_le_bytes(b[2..4].try_into().unwrap());
            let ls = ((b[4 + g / 2] >> (4 * (g & 1))) & 15) | (((hi >> (2 * g)) as u8 & 3) << 4);
            (ls as i32 - 32) as f32
        } else {
            1.0
        };
        return (sub * 32 + part, (0.25 * scale) * LUT[code as usize] as f32);
    }
    let sub = 2 * (g >> 1) + part / 16;
    let half = g & 1;
    let offset = half * 16 + part % 16;
    let start = if dtype == 13 { 48 } else { 16 };
    let q = b[start + 16 * g + part % 16];
    let mut code = (q >> (4 * (part / 16))) & 15;
    if dtype == 13 {
        code |= ((b[16 + offset] >> sub) & 1) << 4;
    }
    (
        sub * 32 + offset,
        (0.25 * scale_min(b, sub, false)) * code as f32 - 0.125 * scale_min(b, sub, true),
    )
}

#[test]
fn aligned_a2_dense_loads_preserve_every_original_even_offset_within_padded_owner() {
    for payload in [18usize, 34, 136, 144, 176, 210, 680, 1360, 2100, 2720] {
        let owned = payload.div_ceil(256) * 256;
        let bytes: Vec<_> = (0..owned).map(|i| (i * 73 + 11) as u8).collect();
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for at in (0..=payload - 16).step_by(2) {
            // Two actual aligned int4 reads, followed by the shader's lane
            // permutation/half-word join; never a packed-weight repack.
            let first = at / 16 * 4;
            let low = &words[first..first + 4];
            let high = if at % 16 != 0 {
                &words[first + 4..first + 8]
            } else {
                low
            };
            for word in 0..4 {
                let index = (at / 4) % 4 + word;
                let get = |n: usize| if n < 4 { low[n] } else { high[n - 4] };
                let v = if at & 2 != 0 {
                    (get(index) >> 16) | (get(index + 1) << 16)
                } else {
                    get(index)
                };
                assert_eq!(
                    v,
                    u32::from_le_bytes(bytes[at + word * 4..at + word * 4 + 4].try_into().unwrap())
                );
            }
        }
    }
}

#[test]
fn connected_wide_dense_variants_consume_original_owners_and_exact_rhs() {
    use wgpu::naga as n;
    for dtype in [8, 12, 13, 14, 20, 23] {
        for members in 1..=8 {
            for indexed in [false, true]
                .into_iter()
                .filter(|&i| !i || dtype == 8 && members == 1)
            {
                let (values, bytes) = crate::quant_matvec::block_layout(dtype).unwrap();
                let source = Kernel {
                    dtype,
                    rows: 17,
                    columns: 2560,
                    row_bytes: 2560 / values * bytes,
                    members,
                    indexed,
                }
                .source();
                let module = n::front::wgsl::parse_str(&source).unwrap();
                let info = n::valid::Validator::new(
                    n::valid::ValidationFlags::all(),
                    n::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
                let uses = info.get_entry_point(0);
                let owners: std::collections::BTreeMap<_, _> = module
                    .global_variables
                    .iter()
                    .filter_map(|(id, g)| g.binding.as_ref().map(|b| (b.binding, uses[id])))
                    .collect();
                assert_eq!(owners.len(), if indexed { 6 } else { 5 });
                for b in [0, 1] {
                    assert!(owners[&b].contains(n::valid::GlobalUse::READ));
                    assert!(!owners[&b].contains(n::valid::GlobalUse::WRITE));
                }
                for b in [2, 3] {
                    assert!(owners[&b].contains(n::valid::GlobalUse::WRITE));
                }
                if indexed {
                    assert!(owners[&4].contains(n::valid::GlobalUse::READ));
                }
                assert!(source.contains("p.count!=MEMBERS"));
                assert!(source.contains("subgroupShuffleXor(total,offset)"));
                assert!(!source.contains("subgroupAdd("));
                let words = n::back::spv::write_vec(
                    &module,
                    &info,
                    &Default::default(),
                    Some(&n::back::spv::PipelineOptions {
                        shader_stage: n::ShaderStage::Compute,
                        entry_point: "main".into(),
                    }),
                )
                .unwrap();
                let rounded = cubecl::wgpu::preserve_native_product_rounding(&words).unwrap();
                assert!(!rounded.is_empty());
            }
        }
    }
}
