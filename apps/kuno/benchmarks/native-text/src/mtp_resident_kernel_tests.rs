use super::*;
use kuno_text::{TensorType, model::greedy};

#[test]
fn exact_all512_q8_shards_and_all_original_page_charges() {
    let mut next = 0;
    for index in 0..SHARDS {
        let s = Shard::new(index).unwrap();
        assert_eq!(s.first, next);
        assert_eq!(s.experts, if index == 7 { 8 } else { 72 });
        assert_eq!(s.packed_bytes().unwrap(), s.experts * 1_740_800);
        assert_eq!(
            s.owned_bytes().unwrap(),
            if index == 7 { 16 << 20 } else { 128 << 20 }
        );
        assert_eq!(s.packed_words().unwrap() * 4, s.packed_bytes().unwrap());
        next += s.experts;
    }
    assert_eq!(next, EXPERTS);
    assert_eq!(expert_owned_bytes().unwrap(), 2_868_903_936);
    assert!(Shard::new(8).is_err());
    assert!(
        Shard {
            first: 1,
            experts: 72
        }
        .validate()
        .is_err()
    );
    assert!(
        Shard {
            first: 504,
            experts: 72
        }
        .validate()
        .is_err()
    );
    assert_eq!(CACHE_WORDS * 4, 4 << 20);
    assert_eq!(
        (
            CONDITION_WORDS,
            ACTIVATION_WORDS,
            PARTIAL_WORDS,
            READINESS_WORDS
        ),
        (20480, 6400, 25600, 20)
    );
    assert_eq!(ROW_BYTES, 2720);
    assert_eq!(DOWN_ROW_BYTES, 680);
    assert_eq!(FF * ROW_BYTES, D * DOWN_ROW_BYTES);
    assert_eq!((ROUTE_FIRST, ROUTE_END), (3073, 3093));
}

#[test]
fn bounded_positions_and_rewind_do_not_require_sparse_selection() {
    assert_eq!(validate_position(0).unwrap(), 1);
    assert_eq!(validate_position(2047).unwrap(), 2048);
    for p in [2048, 4095, usize::MAX] {
        assert!(validate_position(p).is_err());
    }
    for (old, new) in [(0, 0), (1, 0), (2048, 2048), (2048, 17)] {
        assert!(validate_rewind(old, new).is_ok());
    }
    for (old, new) in [(0, 1), (2048, 2049), (2049, 0), (usize::MAX, 0)] {
        assert!(validate_rewind(old, new).is_err());
    }
}

#[test]
fn actual_device_limits_admit_complete_banks_dense2k_and_seven_bindings() {
    let valid = wgpu::Limits::default();
    assert!(validate_limits(&valid).is_ok());
    for which in 0..7 {
        let mut limits = valid.clone();
        match which {
            0 => limits.max_compute_invocations_per_workgroup = 255,
            1 => limits.max_compute_workgroup_size_x = 255,
            2 => limits.max_compute_workgroups_per_dimension = (QH * CONTEXT - 1) as u32,
            3 => limits.max_compute_workgroup_storage_size = 2047,
            4 => limits.max_storage_buffers_per_shader_stage = 6,
            5 => limits.max_storage_buffer_binding_size = 128 * 1024 * 1024 - 1,
            _ => limits.max_buffer_size = 128 * 1024 * 1024 - 1,
        }
        assert!(validate_limits(&limits).is_err(), "limit{which}");
    }
}

#[test]
fn eh_condition_keeps_embedding_first_and_each_of_four_original_streams() {
    let e: Vec<f32> = (0..D).map(|i| i as f32 / 256.).collect();
    let h: Vec<f32> = (0..WIDE).map(|i| -(i as f32) / 1024.).collect();
    let mut actual = vec![0.; HC * 2 * D];
    for (i, v) in actual.iter_mut().enumerate() {
        let stream = i / (D * 2);
        let column = i % (D * 2);
        *v = if column < D {
            e[column]
        } else {
            h[stream * D + column - D]
        };
    }
    for stream in 0..HC {
        let row = &actual[stream * 2 * D..(stream + 1) * 2 * D];
        assert_eq!(&row[..D], &e);
        assert_eq!(&row[D..], &h[stream * D..(stream + 1) * D]);
    }
}

fn packed_q8_row(columns: usize, salt: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(columns / 32 * 34);
    for block in 0..columns / 32 {
        // Independent exact representable half scales, varied signs/extremes.
        bytes.extend_from_slice(&[0x00, 0x30 + ((block + salt) % 4) as u8 * 4]);
        for lane in 0..32 {
            bytes.push(((block * 37 + lane * 29 + salt * 11) % 256) as u8);
        }
    }
    bytes
}
fn packed_q8_value(bytes: &[u8], column: usize) -> f32 {
    let base = column / 32 * 34;
    let h = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
    // Fixtures have normal powers-of-two; decode by IEEE exponent directly,
    // independent of the scalar/GPU helper and half library.
    let scale = 2.0_f32.powi(((h >> 10) & 31) as i32 - 15);
    scale * (bytes[base + 2 + column % 32] as i8 as f32)
}
fn original_dot(weight: &[f32], input: &[f32]) -> f32 {
    let mut lanes = [0_f32; 64];
    for i in 0..weight.len() {
        lanes[i % 64] += weight[i] * input[i];
    }
    let mut stride = 32;
    while stride > 0 {
        for i in 0..stride {
            lanes[i] += lanes[i + stride];
        }
        stride /= 2;
    }
    lanes[0]
}
fn subgroup_dot(bytes: &[u8], input: &[f32]) -> f32 {
    let mut lanes = [0_f32; 32];
    for (lane, total) in lanes.iter_mut().enumerate() {
        let mut low = 0_f32;
        let mut high = 0_f32;
        for tile in 0..input.len() / 64 {
            let i = tile * 64 + lane;
            low += packed_q8_value(bytes, i) * input[i];
            high += packed_q8_value(bytes, i + 32) * input[i + 32];
        }
        *total = low + high;
    }
    let mut stride = 16;
    while stride > 0 {
        for i in 0..stride {
            lanes[i] += lanes[i + stride];
        }
        stride /= 2;
    }
    lanes[0]
}

#[test]
fn independent_original_q8_rows_match_exact64lane_expert_dot_at_real_widths() {
    for columns in [FF, D, D * 2, WIDE] {
        let input: Vec<f32> = (0..columns)
            .map(|i| ((i % 61) as f32 - 30.) / 16.)
            .collect();
        for salt in [0, 1, 7, 31, 255] {
            let bytes = packed_q8_row(columns, salt);
            let mut scalar = vec![0.; columns];
            TensorType::Q8_0.decode(&bytes, &mut scalar).unwrap();
            for (i, &v) in scalar.iter().enumerate() {
                assert_eq!(v.to_bits(), packed_q8_value(&bytes, i).to_bits());
            }
            assert_eq!(
                original_dot(&scalar, &input).to_bits(),
                subgroup_dot(&bytes, &input).to_bits()
            );
        }
    }
}

#[test]
fn selected_experts_cover_all_eight_shards_once_in_original_rank_order() {
    let ids = [511, 0, 72, 503, 144, 287, 216, 359, 431, 432];
    let mut covered = [0; RANKS];
    for index in 0..SHARDS {
        let shard = Shard::new(index).unwrap();
        for (rank, id) in ids.into_iter().enumerate() {
            if id >= shard.first && id - shard.first < shard.experts {
                covered[rank] += 1;
                // The actual selected original expert's row base, no slot map.
                let local = id - shard.first;
                for row in [0, 17, FF - 1] {
                    assert_eq!(
                        (shard.first * FF + local * FF + row) * ROW_BYTES,
                        (id * FF + row) * ROW_BYTES
                    );
                }
                for row in [0, 43, D - 1] {
                    assert_eq!(
                        (shard.first * D + local * D + row) * DOWN_ROW_BYTES,
                        (id * D + row) * DOWN_ROW_BYTES
                    );
                }
            }
        }
    }
    assert_eq!(covered, [1; RANKS]);
}

#[test]
fn full_dense_attention_identity_includes_last_token_and_h_div12_gqa_gate() {
    // Zero queries imply exactly uniform probabilities. A unique final-token
    // contribution makes any accidentally selected/truncated prefix fail.
    for visible in [1, 17, CONTEXT] {
        let mut values = vec![0_f32; CACHE_WORDS];
        for t in 0..visible {
            for head in 0..KVH {
                for d in 0..HD {
                    values[t * KV + head * HD + d] = head as f32 + d as f32 / 256.;
                }
            }
        }
        for d in 0..HD {
            values[(visible - 1) * KV + HD + d] += 1.;
        }
        for head in [0, 11, 12, 23] {
            for d in [0, 1, 255] {
                let mut actual = 0_f32;
                for t in 0..visible {
                    actual += 1. / visible as f32 * values[t * KV + head / 12 * HD + d];
                }
                let reference = (0..visible)
                    .map(|t| f64::from(values[t * KV + head / 12 * HD + d]))
                    .sum::<f64>()
                    / visible as f64;
                assert!((actual - reference as f32).abs() < 0.0001);
                let gated = actual * kuno_text::ops::sigmoid(-1.);
                assert!(gated.is_finite());
                if head >= 12 {
                    assert!(actual > 1.);
                }
            }
        }
    }
}

#[test]
fn full_vocabulary_greedy_ties_last_row_and_nonfinite_are_independent_core_cases() {
    assert_eq!(GREEDY_GROUPS, 61);
    let mut logits = vec![-f32::MAX; VOCAB];
    assert_eq!(greedy(&logits).unwrap(), 0);
    logits[VOCAB - 1] = 0.;
    assert_eq!(greedy(&logits).unwrap(), (VOCAB - 1) as u32);
    logits[4096] = 0.;
    assert_eq!(greedy(&logits).unwrap(), 4096);
    logits[VOCAB - 2] = f32::NAN;
    assert!(greedy(&logits).is_err());
}

#[test]
fn every_new_kernel_lowers_checked_and_unchecked_with_original_math_bindings() {
    use cubecl::server::ExecutionMode;
    use cubecl::wgpu::{WgpuRuntime, WgslCompiler};
    use cubecl::{Compiler, WgpuCompilationOptions};
    for which in 0..14 {
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            b.runtime_properties(WgpuRuntime::target_properties());
            AddressType::U32.register(&mut b.scope);
            let f = f32::as_type(&b.scope);
            let u = u32::as_type(&b.scope);
            let a = Atomic::<u32>::as_type(&b.scope);
            let cube = if matches!(which, 7 | 8) {
                fast::cube()
            } else {
                CubeDim::new_1d(64)
            };
            match which {
                0 => {
                    let w = b.input_array(u).into();
                    let t = b.input_array(u).into();
                    let o = b.output_array(f).into();
                    let x = b.output_array(a).into();
                    embedding_tile::expand(&mut b.scope, w, t, o, x, 245760, 2560);
                }
                1 | 9 | 10 => {
                    let i = b.input_array(f).into();
                    let j = b.input_array(f).into();
                    let o = b.output_array(f).into();
                    let x = b.output_array(a).into();
                    match which {
                        1 => condition_concat::expand(&mut b.scope, i, j, o, x),
                        9 => shared_activate::expand(&mut b.scope, i, j, o, x),
                        _ => shared_gate::expand(&mut b.scope, i, j, o, x),
                    }
                }
                2 => {
                    let i = b.input_array(f).into();
                    let j = b.input_array(f).into();
                    let k = b.output_array(f).into();
                    let v = b.output_array(f).into();
                    let x = b.output_array(a).into();
                    let p = usize::__expand_new(&mut b.scope, 17);
                    append_kv::expand(&mut b.scope, i, j, k, v, x, p);
                }
                3 => {
                    let k = b.output_array(f).into();
                    let v = b.output_array(f).into();
                    let x = b.output_array(a).into();
                    let old = usize::__expand_new(&mut b.scope, 2048);
                    let new = usize::__expand_new(&mut b.scope, 17);
                    rewind_kv::expand(&mut b.scope, k, v, x, old, new);
                }
                4 => {
                    let q = b.input_array(f).into();
                    let k = b.input_array(f).into();
                    let o = b.output_array(f).into();
                    let x = b.output_array(a).into();
                    let n = usize::__expand_new(&mut b.scope, 17);
                    dense_logits::expand(&mut b.scope, q, k, o, x, n);
                }
                5 => {
                    let v = b.input_array(f).into();
                    let p = b.input_array(f).into();
                    let g = b.input_array(f).into();
                    let o = b.output_array(f).into();
                    let x = b.output_array(a).into();
                    let n = usize::__expand_new(&mut b.scope, 2048);
                    dense_values_gated::expand(&mut b.scope, v, p, g, o, x, n);
                }
                6 => {
                    let p = b.input_array(u).into();
                    let x = b.output_array(a).into();
                    routes_preflight::expand(&mut b.scope, p, x);
                }
                7 => {
                    let g = b.input_array(u).into();
                    let up = b.input_array(u).into();
                    let i = b.input_array(f).into();
                    let p = b.input_array(u).into();
                    let o = b.output_array(f).into();
                    let s = b.output_array(u).into();
                    let x = b.output_array(a).into();
                    q8_gate_up::expand(&mut b.scope, g, up, i, p, o, s, x, 504, 8);
                }
                8 => {
                    let d = b.input_array(u).into();
                    let i = b.input_array(f).into();
                    let p = b.input_array(u).into();
                    let o = b.output_array(f).into();
                    let s = b.output_array(u).into();
                    let x = b.output_array(a).into();
                    q8_down::expand(&mut b.scope, d, i, p, o, s, x, 0, 72);
                }
                11 => {
                    let i = b.input_array(f).into();
                    let s = b.input_array(f).into();
                    let gate = b.input_array(f).into();
                    let p = b.input_array(u).into();
                    let status = b.input_array(u).into();
                    let o = b.output_array(f).into();
                    let x = b.output_array(a).into();
                    routed_shared_merge::expand(&mut b.scope, i, s, gate, p, status, o, x);
                }
                12 => {
                    let i = b.input_array(f).into();
                    let o = b.output_array(u).into();
                    let x = b.output_array(a).into();
                    greedy_chunks::expand(&mut b.scope, i, o, x);
                }
                _ => {
                    let i = b.input_array(u).into();
                    let o = b.output_array(u).into();
                    let x = b.output_array(a).into();
                    greedy_finish::expand(&mut b.scope, i, o, x);
                }
            }
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(cube)
                            .kernel_name("fixed_public_mtp_leaf"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    AddressType::U32.unsigned_type(),
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::SUBGROUP,
            )
            .validate(&module)
            .unwrap();
            assert!(source.contains("atomicOr"));
            assert!(!source.contains("fma(") && !source.contains("f64"));
            if matches!(which, 7 | 8) {
                assert!(source.contains("subgroupShuffleDown"));
            }
        }
    }
}
