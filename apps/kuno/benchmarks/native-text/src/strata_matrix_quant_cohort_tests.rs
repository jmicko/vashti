use super::*;
fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}
fn has(s: &str, needle: &str) -> bool {
    compact(s).contains(&compact(needle))
}
fn float_bits(bits: u32) -> f32 {
    let sign = (bits & 32768) << 16;
    let e = (bits >> 10) & 31;
    let mut m = bits & 1023;
    let mut out = sign;
    if e == 0 {
        if m != 0 {
            let mut exp = 113u32;
            while m & 1024 == 0 {
                m <<= 1;
                exp -= 1;
            }
            out |= (exp << 23) | ((m & 1023) << 13);
        }
    } else if e == 31 {
        out |= 2139095040 | (m << 13);
    } else {
        out |= ((e + 112) << 23) | (m << 13);
    }
    f32::from_bits(out)
}
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 32768;
    let magnitude = bits & 2147483647;
    let e = (magnitude >> 23) as i32 - 112;
    let mut out = 0u32;
    if magnitude >= 2139095040 {
        out = 31744;
        if magnitude & 8388607 != 0 {
            out = 32256;
        }
    } else if e >= 31 {
        out = 31744;
    } else if e <= 0 {
        if e >= -10 {
            let shift = (14 - e) as u32;
            let significand = (magnitude & 8388607) | 8388608;
            out = (significand + (1u32 << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
        }
    } else {
        out = ((magnitude + 4095 + ((magnitude >> 13) & 1)) >> 13) - 114688;
    }
    (sign | out) as u16
}
#[test]
fn quant_cohort_exact_half_bits_cover_every_scale_encoding_subnormal_tie_and_overflow() {
    let s = include_str!("strata_matrix_quant_cohort_decoder.rs");
    for text in [
        "magnitude + 4095 + ((magnitude >> 13) & 1)",
        "significand + (1u32 << (shift - 1)) - 1",
        "Vector::<f16, Const<2>>::reinterpret(hx | (hy << 16))",
    ] {
        assert!(has(s, text));
    }
    assert!(!s.contains("f16::cast_from("));
    for h in 0..=65535u32 {
        let v = float_bits(h);
        let expected = half::f16::from_bits(h as u16).to_f32();
        assert!(v.to_bits() == expected.to_bits() || (v.is_nan() && expected.is_nan()));
        for factor in [1.0f32, 31., 0.25, 7.75, 0.5, 3.875] {
            for code in [-127.0f32, -1., 0., 1., 127.] {
                let x = (v * factor) * code;
                let got = half_bits(x);
                let want = half::f16::from_f32(x).to_bits();
                assert!(
                    got == want
                        || (got & 31744 == 31744
                            && got & 1023 != 0
                            && want & 31744 == 31744
                            && want & 1023 != 0)
                );
            }
        }
        if v.is_finite() {
            for bits in [
                v.to_bits().saturating_sub(1),
                v.to_bits(),
                v.to_bits().saturating_add(1),
            ] {
                let x = f32::from_bits(bits);
                let got = half_bits(x);
                let want = half::f16::from_f32(x).to_bits();
                assert!(got == want || (x.is_nan() && got & 31744 == 31744 && got & 1023 != 0));
            }
        }
    }
    for x in [
        65504.0f32,
        65519.996,
        65520.,
        65536.,
        f32::MIN_POSITIVE,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ] {
        assert_eq!(
            half_bits(x) & 31744,
            half::f16::from_f32(x).to_bits() & 31744
        );
    }
}
#[test]
fn quant_cohort_panel_mapping_keeps_every_k16_chronological_and_sg_private_bank_disjoint() {
    for tile in [8usize, 16, 32] {
        let mut a = vec![false; 4 * 2 * tile * 32];
        let mut b = vec![false; 4 * 2 * 16 * 32];
        for sg in 0..4 {
            for bank in 0..2 {
                for lane in 0..16 {
                    for round in 0..tile / 4 {
                        let job = lane + round * 16;
                        let m = job / 4;
                        let k = job % 4 * 8;
                        let first =
                            ((sg * 2 + bank) * tile * 8 + (k / 16) * tile * 4 + m * 4 + k % 16 / 4)
                                * 4;
                        for j in 0..8 {
                            assert!(!a[first + j]);
                            a[first + j] = true;
                            let local = first + j - (sg * 2 + bank) * tile * 32;
                            assert_eq!(
                                (local / (tile * 16), local % (tile * 16) / 16, local % 16),
                                (k / 16, m, k % 16 + j)
                            );
                        }
                    }
                }
                for lane in 0..16 {
                    for round in 0..4 {
                        let job = lane + round * 16;
                        let n = job / 4;
                        let k = job % 4 * 8;
                        let first = ((sg * 2 + bank) * 128 + k / 16 * 64 + n * 4 + k % 16 / 4) * 4;
                        for j in 0..8 {
                            assert!(!b[first + j]);
                            b[first + j] = true;
                            let local = first + j - (sg * 2 + bank) * 512;
                            assert_eq!(
                                (local / 256, local % 256 / 16, local % 16),
                                (k / 16, n, k % 16 + j)
                            );
                        }
                    }
                }
                for step in 0..2 {
                    for m8 in 0..tile / 8 {
                        for m in 0..8 {
                            for k in 0..16 {
                                let address =
                                    ((sg * 2 + bank) * tile * 8 + step * tile * 4 + m8 * 32) * 4
                                        + m * 16
                                        + k;
                                assert_eq!(
                                    (address - (sg * 2 + bank) * tile * 32) / (tile * 16),
                                    step
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(a.into_iter().all(|x| x));
        assert!(b.into_iter().all(|x| x));
        assert!(tile * 512 + 8192 + 2048 <= 49152);
        for k in [640usize, 2560] {
            assert_eq!(
                (0..k)
                    .step_by(32)
                    .flat_map(|a| [a, a + 16])
                    .collect::<Vec<_>>(),
                (0..k).step_by(16).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn quant_cohort_k128_global_input_and_weight_banks_cover_original_operands_once() {
    // Independent address oracle: original row-major global input and original
    // chronological K16 products must be identical after removing A staging.
    for columns in [640usize, 2560] {
        for members in 1usize..32 {
            let tile = if members <= 8 {
                8
            } else if members <= 16 {
                16
            } else {
                32
            };
            for prefix in [0usize, 8, 80] {
                let mut products = Vec::new();
                for k in (0..columns).step_by(128) {
                    for step in 0..8 {
                        let reduction = k + step * 16;
                        products.push(reduction);
                        for fragment in 0..tile / 8 {
                            let first =
                                (prefix * columns + reduction) / 4 + fragment * 8 * columns / 4;
                            for row in 0..8 {
                                for col in 0..16 {
                                    let word = first * 4 + row * columns + col;
                                    assert_eq!(word / columns, prefix + fragment * 8 + row);
                                    assert_eq!(word % columns, reduction + col);
                                    assert!(word < (prefix + tile) * columns);
                                }
                            }
                        }
                    }
                }
                assert_eq!(products, (0..columns).step_by(16).collect::<Vec<_>>());
            }
        }
    }
    let mut written = vec![false; 4 * 2 * 16 * 128];
    for sg in 0..4usize {
        for bank in 0..2usize {
            for row in 0..16usize {
                for lane in 0..16usize {
                    let col = lane * 8;
                    let first =
                        ((sg * 2 + bank) * 512 + col / 16 * 64 + row * 4 + col % 16 / 4) * 4;
                    for j in 0..8 {
                        assert!(!written[first + j]);
                        written[first + j] = true;
                        let local = first + j - (sg * 2 + bank) * 16 * 128;
                        assert_eq!(
                            (local / 256, local % 256 / 16, local % 16),
                            ((col + j) / 16, row, (col + j) % 16)
                        );
                    }
                }
            }
        }
    }
    assert!(written.into_iter().all(|v| v));
    assert_eq!(4 * 2 * 16 * 128 * 2 + 4 * 128 * 4, 34816);
    assert!(34816 < 49152);
}

#[test]
fn quant_cohort_k64_mapping_and_unpacked_signs_match_original_byte_wrapping() {
    let mut seen = vec![false; 4 * 2 * 16 * 64];
    for sg in 0..4usize {
        for bank in 0..2usize {
            for round in 0..8usize {
                for lane in 0..16usize {
                    let row = round * 2 + lane / 8;
                    let col = lane % 8 * 8;
                    let first =
                        ((sg * 2 + bank) * 256 + col / 16 * 64 + row * 4 + col % 16 / 4) * 4;
                    for j in 0..8 {
                        assert!(!seen[first + j]);
                        seen[first + j] = true;
                        let local = first + j - (sg * 2 + bank) * 16 * 64;
                        assert_eq!(
                            (local / 256, local % 256 / 16, local % 16),
                            ((col + j) / 16, row, (col + j) % 16)
                        );
                    }
                }
            }
        }
    }
    assert!(seen.into_iter().all(|v| v));
    for columns in [640usize, 2560] {
        assert_eq!(
            (0..columns)
                .step_by(64)
                .flat_map(|base| (0..4).map(move |step| base + step * 16))
                .collect::<Vec<_>>(),
            (0..columns).step_by(16).collect::<Vec<_>>()
        );
    }
    assert_eq!(4 * 2 * 16 * 64 * 2 + 4 * 128 * 4, 18432);
    // Exhaustive byte/sign oracle includes values outside the actual tables;
    // the128 wrap case must stay -128 even when negated before truncation.
    for byte in 0u32..256 {
        for negative in [false, true] {
            let signed = if negative {
                -(byte as i32)
            } else {
                byte as i32
            };
            let original = (signed as u8) as i8 as i32;
            let selected = signed.wrapping_shl(24) >> 24;
            assert_eq!(selected, original);
            for scale in [-7.75f32, -0.0, 0.0, 0.25, 31.0] {
                assert_eq!(
                    (scale * selected as f32).to_bits(),
                    (scale * original as f32).to_bits()
                );
            }
        }
    }
}

#[test]
fn quant_cohort_padded_b_tiles_are_aligned_disjoint_fully_initialized_and_reusable() {
    let mut seen = vec![false; 4 * 2 * 376 * 4];
    for sg in 0..4usize {
        for bank in 0..2usize {
            let base = (sg * 2 + bank) * 376 * 4;
            for step in 0..4usize {
                let tile = base + step * 94 * 4;
                assert_eq!(tile * 2 % 16, 0);
                for row in 0..16usize {
                    for col in 0..16usize {
                        let offset = tile + row * 24 + col;
                        assert!(!seen[offset]);
                        seen[offset] = true;
                        assert_eq!(offset - tile, row * 24 + col);
                    }
                    if row < 15 {
                        for padding in 16..24usize {
                            let offset = tile + row * 24 + padding;
                            assert!(!seen[offset]);
                            seen[offset] = true;
                        }
                    }
                }
                assert_eq!(tile + 15 * 24 + 16, base + (step + 1) * 376);
            }
        }
    }
    assert!(seen.into_iter().all(|v| v));
    assert_eq!(4 * 2 * 376 * 4 * 2, 24064);
    assert!(24064 + 2048 <= 49152);
    assert_eq!(24064 + 4 * 128 * 4, 26112);
    // Independent source/lifetime fence oracle: publication may alias B only
    // after a whole-WG fence, never after just its subgroup's final reader.
    let source = include_str!("strata_matrix_quant_cohort_kernel.rs");
    assert_eq!(source.matches("sync_cube()").count(), 1);
    let epilogue = source.split_once("sync_cube();").unwrap().1;
    assert!(epilogue.find("b.free()").unwrap() < epilogue.find("let mut scratch").unwrap());
}
#[test]
fn quant_cohort_compact_shape_requires_real_six_format_role_bytes_and_distinct_tiles() {
    for gu in [18u32, 21, 22, 23] {
        for down in [20u32, 42] {
            let mut w = [0u32; 48];
            w[6] = (640 * quant_row_bytes(gu, false) / 4) as u32;
            w[7] = w[6];
            w[8] = 2 * w[6] + (2560 * quant_row_bytes(down, true) / 4) as u32;
            assert!(role_shape(&w, gu, false));
            assert!(role_shape(&w, down, true));
            w[7] += 1;
            assert!(!role_shape(&w, gu, false));
            w[7] -= 1;
            w[8] += 1;
            assert!(!role_shape(&w, down, true));
        }
    }
    for d in [0u32, 1, 19, 24, 41, 43, u32::MAX] {
        assert!(!supported(d, false));
        assert!(!supported(d, true));
    }
    for n in 1usize..32 {
        let selected = [8, 16, 32]
            .into_iter()
            .filter(|&t| match t {
                8 => n <= 8,
                16 => n > 8 && n <= 16,
                32 => n > 16,
                _ => false,
            })
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), 1);
        assert_eq!(
            n.div_ceil(selected[0]) * selected[0],
            n.div_ceil(if n <= 8 { 8 } else { 16 }) * if n <= 8 { 8 } else { 16 }
        );
    }
}
#[test]
fn quant_cohort_source_preserves_eight_value_poison_gate_and_existing_consumers() {
    let d = include_str!("strata_matrix_quant_cohort_decoder.rs");
    let k = include_str!("strata_matrix_quant_cohort_kernel.rs");
    assert!(has(
        d,
        "good = good && x.abs() <= f32::MAX && y.abs() <= f32::MAX"
    ));
    assert!(has(d, "(hx & 31744) != 31744 && (hy & 31744) != 31744"));
    let zero = d
        .split_once("if !good {")
        .unwrap()
        .1
        .split_once("out[first]")
        .unwrap()
        .0;
    assert!(has(zero, "a = Vector::<f16, Const<4>>::new(f16::new(0.0))"));
    assert!(has(zero, "b = Vector::<f16, Const<4>>::new(f16::new(0.0))"));
    assert!(has(zero, "fault[0].fetch_or(128)"));
    assert!(!has(k, "native_release_after"));
    // Bank reuse becomes cross-subgroup only after all K products. Retain
    // subgroup-only panel handoffs, plus one whole-workgroup lifetime fence.
    assert_eq!(k.matches("sync_cube()").count(), 1);
    assert_eq!(compact(k).matches("if!plane_all(!(").count(), 2);
    let release = k.split_once("sync_cube();").unwrap().1;
    assert!(release.find("b.free()").unwrap() < release.find("let mut scratch").unwrap());
    assert!(k.matches("sync_plane()").count() >= 2);
    assert_eq!(
        k.matches("crate::strata_matrix::small_direct::publish(")
            .count(),
        4
    );
    for needle in [
        "map[4 + group] >= 512",
        "map[516 + group] != count",
        "map.len() != 65536",
        "raw > 8",
        "packed.len() != comptime!(crate::strata_prompt::groups::CAPACITY * 5120)",
        "strata_grid.len() != 8192",
    ] {
        assert!(has(k, needle));
    }
    assert!(has(k, "source_first = slot * usize::cast_from(h8) * 4"));
    assert!(has(k, "source_first += usize::cast_from(h6 + h7) * 4"));
}
#[test]
fn quant_cohort_decoder_preserves_original_six_format_integer_and_factor_associations() {
    let s = include_str!("strata_matrix_quant_cohort_decoder.rs");
    for part in [
        "dtype == 18",
        "dtype == 21",
        "dtype == 22",
        "dtype == 23",
        "dtype == 20",
        "dtype == 42",
        "256 + usize::cast_from(first)",
        "768 + usize::cast_from(index) * 2",
        "769 + usize::cast_from(index) * 2",
        "(sc.count_ones() & 1) << 7",
        "block % 2 * 8",
        "quarter % 2 * 8",
        "quarter / 2 * 4",
    ] {
        assert!(has(s, part));
    }
    assert!(has(s, "* (0.5 + f32::cast_from(aux >> 28)) * 0.5"));
    assert!(has(s, "* (0.5 + f32::cast_from(scale)) * 0.25"));
    assert!(!s.contains(".mul_add("));
    let lut = [3215825025u32, 4142587343, 639175937, 1901675829];
    for i in 0..16 {
        assert_eq!(
            (lut[i / 4] >> (i % 4 * 8)) as u8 as i8,
            kuno_text::quant::IQ4_NL_VALUES[i]
        );
    }
}

#[test]
fn quant_cohort_loop_free_scale_and_single_stage_keep_exact_original_bits_and_pipeline_trace() {
    fn leading_zero_scale(bits: u32) -> f32 {
        let sign = (bits & 32768) << 16;
        let e = (bits >> 10) & 31;
        let m = bits & 1023;
        let mut out = sign;
        if e == 0 {
            if m != 0 {
                let shift = m.leading_zeros() - 21;
                out |= ((113 - shift) << 23) | (((m << shift) & 1023) << 13);
            }
        } else if e == 31 {
            out |= 2139095040 | (m << 13);
        } else {
            out |= ((e + 112) << 23) | (m << 13);
        }
        f32::from_bits(out)
    }
    // Includes signed zero, every subnormal, both infinities and all NaN bits.
    for bits in 0..=65535 {
        assert_eq!(
            leading_zero_scale(bits).to_bits(),
            float_bits(bits).to_bits()
        );
    }
    type Event = (u8, usize, usize, usize);
    const BARRIER: Event = (2, 0, 0, 0);
    for columns in [640usize, 2560] {
        for tile in [8usize, 16, 32] {
            // Accepted prologue + prefetch loop, before this source change.
            let mut original: Vec<Event> = vec![(0, 0, 0, 0), BARRIER];
            let mut current = 0;
            for k in (0..columns).step_by(32) {
                let next = 1 - current;
                if k + 32 < columns {
                    original.push((0, k + 32, next, 0));
                }
                for step in 0..2 {
                    for fragment in 0..tile / 8 {
                        original.push((1, k + step * 16, current, fragment));
                    }
                }
                original.push(BARRIER);
                current = next;
            }
            // One producer loop, including its last drain-only iteration.
            let mut selected = Vec::new();
            let mut producer_k = 0;
            let mut producer_bank = 0;
            while producer_k <= columns {
                let previous_bank = 1 - producer_bank;
                if producer_k < columns {
                    selected.push((0, producer_k, producer_bank, 0));
                }
                if producer_k != 0 {
                    for step in 0..2 {
                        for fragment in 0..tile / 8 {
                            selected.push((
                                1,
                                producer_k - 32 + step * 16,
                                previous_bank,
                                fragment,
                            ));
                        }
                    }
                }
                selected.push(BARRIER);
                producer_bank = previous_bank;
                producer_k += 32;
            }
            assert_eq!(selected, original);
            assert_eq!(selected.iter().filter(|e| e.0 == 0).count(), columns / 32);
            assert_eq!(
                selected.iter().filter(|e| e.0 == 1).count(),
                columns / 16 * (tile / 8)
            );
            assert_eq!(
                selected.iter().filter(|&&e| e == BARRIER).count(),
                columns / 32 + 1
            );
        }
    }
}
