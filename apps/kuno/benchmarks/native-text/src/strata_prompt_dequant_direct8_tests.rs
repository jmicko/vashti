use super::*;
use crate::quant_matvec::iq_fixtures;
use kuno_text::TensorType;

fn original_store(source: &str) -> &str {
    source
        .split_once("fn store_original8(")
        .unwrap()
        .1
        .split_once("\n}\nfn write_record32")
        .unwrap()
        .0
}

#[test]
fn direct8_all_real_sources_keep_original_authority_store_and_no_cross_lane_payload() {
    for op in super::super::dequant_operators() {
        let (dtype, down) = match op {
            Op::Dequant { dtype, down }
            | Op::ResidentDequant { dtype, down }
            | Op::SlotDequant { dtype, down, .. } => (dtype, down),
            _ => unreachable!(),
        };
        let old = if cfg!(feature = "strata-dequant-record32") {
            op.source()
        } else {
            super::super::record32::source(op.source(), down)
        };
        let new = source(old.clone(), dtype, down);
        assert_eq!(original_store(&old), original_store(&new));
        // Everything before the original dispatch is byte-identical, including
        // real physical authority, exact per-role views and all guard returns.
        let prefix = old.split_once("    let record=id.x;").unwrap().0;
        assert!(new.starts_with(&prefix.replace("@workgroup_size(256)", "@workgroup_size(32)")));
        let main = new
            .split_once("fn main(")
            .unwrap()
            .1
            .split_once("\n}\n")
            .unwrap()
            .0;
        assert!(main.contains("store_original8(i,"));
        assert!(!main.contains("write_record32("));
        assert!(!new.contains("subgroupShuffle("));
        assert!(!new.contains("subgroupQuad"));
        assert!(!new.contains("var<workgroup>"));
        assert!(new.contains("@workgroup_size(32)"));
        let direct = new.split_once("fn gate_direct_halfword(").unwrap().1;
        assert!(!direct.contains("_prepare("));
        assert!(!direct.contains("gate["));
        assert!(!direct.contains("up["));
        assert!(direct.contains("strata_gate(at/4u)"));
        let m = wgpu::naga::front::wgsl::parse_str(&new).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        #[cfg(feature = "strata-dequant-direct8")]
        assert_eq!(op.selected_source(), new);
    }
    #[cfg(feature = "strata-expert-cohorts")]
    for (down, types) in [
        (false, &[q::IQ3_XXS, q::IQ3_S, q::IQ2_S, q::IQ4_XS][..]),
        (true, &[q::IQ4_NL, q::Q2_0][..]),
    ] {
        for &dtype in types {
            let op = cohorts::CohortOp::Image { dtype, down };
            let old = op.source();
            let new = source(old.clone(), dtype, down);
            assert_eq!(original_store(&old), original_store(&new));
            assert!(new.contains("cohort_begin(id.z)"));
            assert!(new.contains("arrayLength(&raw)!=8388608u"));
            let m = wgpu::naga::front::wgsl::parse_str(&new).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&m)
            .unwrap();
            #[cfg(feature = "strata-dequant-direct8")]
            assert_eq!(op.selected_source(), new);
        }
    }
}

#[test]
fn direct8_global_lane_mapping_covers_every_vector_and_original_halfscale_with_tails() {
    for (rows, cols) in [
        (1280usize, 2560usize),
        (2560, 640),
        (1, 32),
        (3, 64),
        (129, 32),
    ] {
        let n = rows * cols / 8;
        let mut writes = vec![0u8; n];
        for id in 0..n.div_ceil(32) * 32 {
            if id >= n {
                continue;
            }
            let row = id / (cols / 8);
            let col = id % (cols / 8) * 8;
            let block = col / 32;
            let quarter = col % 32 / 8;
            assert_eq!(row * (cols / 8) + block * 4 + quarter, id);
            assert_eq!(quarter / 2, col % 32 / 16);
            assert_eq!(block / 2 * 64 + block % 2 * 32 + quarter * 8, col);
            writes[id] += 1;
        }
        assert!(writes.iter().all(|&n| n == 1));
    }
}

fn half(bytes: &[u8], at: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())).to_f32()
}
fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn direct_cpu(dtype: u32, bytes: &[u8], row: usize, block: usize, quarter: usize) -> [f32; 8] {
    let b = |at: usize| u32::from(bytes[at]);
    let mut out = [0.; 8];
    let group = block % 8;
    let table = q::iq_moe_integer::original_grid_words(dtype);
    let signed = |magnitude: u32, signs: u32| -> u32 {
        let mut value = 0;
        for j in 0..4 {
            let m = ((magnitude >> (8 * j)) & 255) as i32;
            let v = if signs & (1 << j) == 0 { m } else { -m };
            value |= ((v as u32) & 255) << (8 * j);
        }
        value
    };
    let (d, packed) = match dtype {
        q::IQ3_XXS => {
            let base = row + (block / 8) * 98;
            let codes = base + 2 + group * 8 + quarter * 2;
            let aux = word(bytes, base + 66 + group * 4);
            let s = (aux >> (quarter * 7)) & 127;
            let signs = s | ((s.count_ones() & 1) << 7);
            let d = half(bytes, base) * (0.5 + (aux >> 28) as f32) * 0.5;
            (
                d,
                [
                    signed(table[b(codes) as usize], signs),
                    signed(table[b(codes + 1) as usize], signs >> 4),
                ],
            )
        }
        q::IQ3_S => {
            let base = row + (block / 8) * 110;
            let codes = base + 2 + group * 8 + quarter * 2;
            let high = b(base + 66 + group);
            let a = b(codes) | (((high >> (quarter * 2)) & 1) << 8);
            let z = b(codes + 1) | (((high >> (quarter * 2 + 1)) & 1) << 8);
            let signs = b(base + 74 + group * 4 + quarter);
            let scale = (b(base + 106 + group / 2) >> ((group % 2) * 4)) & 15;
            (
                half(bytes, base) * (1 + 2 * scale) as f32,
                [
                    signed(table[a as usize], signs),
                    signed(table[z as usize], signs >> 4),
                ],
            )
        }
        q::IQ2_S => {
            let base = row + (block / 8) * 82;
            let index = b(base + 2 + group * 4 + quarter)
                | (((b(base + 66 + group) >> (quarter * 2)) & 3) << 8);
            let signs = b(base + 34 + group * 4 + quarter);
            let scale = (b(base + 74 + group) >> ((quarter / 2) * 4)) & 15;
            (
                half(bytes, base) * (0.5 + scale as f32) * 0.25,
                [
                    signed(table[index as usize * 2], signs),
                    signed(table[index as usize * 2 + 1], signs >> 4),
                ],
            )
        }
        q::IQ4_XS | q::IQ4_NL => {
            let (base, payload, d) = if dtype == q::IQ4_XS {
                let base = row + (block / 8) * 136;
                let header = word(bytes, base);
                let lo = (b(base + 4 + group / 2) >> ((group % 2) * 4)) & 15;
                let hi = (header >> (16 + group * 2)) & 3;
                (
                    base,
                    base + 8 + group * 16 + (quarter % 2) * 8,
                    half(bytes, base) * ((lo | (hi << 4)) as i32 - 32) as f32,
                )
            } else {
                let base = row + block * 18;
                (base, base + 2 + (quarter % 2) * 8, half(bytes, base))
            };
            let _ = base;
            let values = [
                -127i8, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
            ];
            let mut packed = [0u32; 2];
            for i in 0..8 {
                let c = (b(payload + i) >> if quarter >= 2 { 4 } else { 0 }) & 15;
                packed[i / 4] |= u32::from(values[c as usize] as u8) << ((i % 4) * 8);
            }
            (d, packed)
        }
        q::Q2_0 => {
            let base = row + (block / 2) * 18;
            let payload = base + 2 + (block % 2) * 8 + quarter * 2;
            let mut packed = [0u32; 2];
            for i in 0..8 {
                let c = (b(payload + i / 4) >> ((i % 4) * 2)) & 3;
                packed[i / 4] |= (((c as i32 - 1) as u32) & 255) << ((i % 4) * 8);
            }
            (half(bytes, base), packed)
        }
        _ => unreachable!(),
    };
    for i in 0..8 {
        out[i] = d * ((packed[i / 4] >> ((i % 4) * 8)) as u8 as i8) as f32;
    }
    out
}

#[test]
fn direct8_six_lane_payloads_match_independent_pinned_c_values_and_fp16_round_points() {
    for dtype in [
        q::IQ3_XXS,
        q::IQ3_S,
        q::IQ2_S,
        q::IQ4_XS,
        q::IQ4_NL,
        q::Q2_0,
    ] {
        let kind = TensorType::from_gguf_id(dtype).unwrap();
        let cols = if matches!(dtype, q::IQ4_NL | q::Q2_0) {
            640
        } else {
            2560
        };
        let rb = kind.row_bytes(cols as u64).unwrap() as usize;
        for phase in 0..4 {
            let bytes = iq_fixtures::packed(dtype, 3, cols, phase);
            let expected = iq_fixtures::oracle(dtype, 3, cols, phase);
            for row in 0..3 {
                for block in 0..cols / 32 {
                    for quarter in 0..4 {
                        let values = direct_cpu(dtype, &bytes, row * rb, block, quarter);
                        for i in 0..8 {
                            let e = expected[row * cols + block * 32 + quarter * 8 + i];
                            assert_eq!(
                                values[i].to_bits(),
                                e.to_bits(),
                                "dtype{dtype} row{row} block{block} quarter{quarter} component{i}"
                            );
                            assert_eq!(
                                half::f16::from_f32(values[i]).to_bits(),
                                half::f16::from_f32(e).to_bits()
                            );
                        }
                    }
                }
            }
        }
    }
}
