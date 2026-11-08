//! Bounded offline checks; no device/model is created and no tolerance changes.
use super::super::{Kernel as ScalarKernel, oracle};
use super::*;
use wgpu::naga as n;

fn packets(
    count: usize,
    columns: usize,
    rows: usize,
) -> (PrepareSpans, ProjectSpans, [usize; 3], [usize; 3]) {
    let spec = Spec::new(rows, columns).unwrap();
    let prepare = PrepareSpans {
        count,
        input_first: 3,
        input_stride: columns + 7,
        planes_first: 5,
        planes_stride: spec.plane_words() + 11,
        fault_first: 2,
    };
    let project = ProjectSpans {
        count,
        packed_first: 3,
        planes_first: prepare.planes_first,
        planes_stride: prepare.planes_stride,
        output_first: 9,
        output_stride: rows + 13,
        fault_first: 2,
        matrix_first: 7,
        matrix_rows: rows + 2,
    };
    let plane_words =
        prepare.planes_first + (count - 1) * prepare.planes_stride + spec.plane_words();
    (
        prepare,
        project,
        [
            prepare.input_first + (count - 1) * prepare.input_stride + columns,
            plane_words,
            3,
        ],
        [
            plane_words,
            project.matrix_first + (count - 1) * project.output_stride + project.matrix_rows,
            3,
        ],
    )
}

#[test]
fn every_real_member_span_and_original_head_capacity_is_bounded() {
    let spec = Spec::new(7, 2560).unwrap();
    for count in 1..=4 {
        let (prepare, project, input, output) = packets(count, 2560, 7);
        prepare.validate(spec, input).unwrap();
        project
            .validate(spec, 3 + spec.weight_bytes(), output)
            .unwrap();
        for row in 0..count {
            assert!(prepare.input_first + row * prepare.input_stride + 2559 < input[0]);
            assert!(
                prepare.planes_first + row * prepare.planes_stride + spec.plane_words() - 1
                    < input[1]
            );
            assert!(
                project.output_first + row * project.output_stride + spec.rows() - 1 < output[1]
            );
        }
        for index in 0..3 {
            let mut bad = input;
            bad[index] -= 1;
            assert!(prepare.validate(spec, bad).is_err());
            let mut bad = output;
            bad[index] -= 1;
            assert!(
                project
                    .validate(spec, 3 + spec.weight_bytes(), bad)
                    .is_err()
            );
        }
        assert!(
            project
                .validate(spec, 2 + spec.weight_bytes(), output)
                .is_err()
        );
        for bad in [
            PrepareSpans {
                count: 0,
                ..prepare
            },
            PrepareSpans {
                count: 5,
                ..prepare
            },
            PrepareSpans {
                input_stride: 2559,
                ..prepare
            },
            PrepareSpans {
                planes_stride: 1439,
                ..prepare
            },
            PrepareSpans {
                input_first: usize::MAX,
                ..prepare
            },
            PrepareSpans {
                planes_first: usize::MAX,
                ..prepare
            },
        ] {
            assert!(bad.validate(spec, input).is_err());
        }
        for bad in [
            ProjectSpans {
                count: 0,
                ..project
            },
            ProjectSpans {
                count: 5,
                ..project
            },
            ProjectSpans {
                planes_stride: 1439,
                ..project
            },
            ProjectSpans {
                output_stride: project.matrix_rows - 1,
                ..project
            },
            ProjectSpans {
                matrix_rows: 8,
                ..project
            },
            ProjectSpans {
                matrix_first: 10,
                ..project
            },
            ProjectSpans {
                packed_first: usize::MAX,
                ..project
            },
            ProjectSpans {
                output_first: usize::MAX,
                ..project
            },
        ] {
            assert!(bad.validate(spec, 3 + spec.weight_bytes(), output).is_err());
        }
    }
    assert_eq!(spec.plane_words(), 1440);
    assert_eq!(4 * spec.plane_words() * 4, 23040);
    assert!(4 * spec.plane_words() * 4 < 8 * 10240 * 4);
    let head = Spec::new(16384, 2560).unwrap();
    assert_eq!(head.row_bytes(), 2100);
    assert_eq!(
        15 * head.weight_bytes() + Spec::new(2560, 2560).unwrap().weight_bytes(),
        521472000
    );
    let tiles = (0..248320).step_by(head.rows()).count();
    let batch_info_records = 1 + tiles + 1;
    assert_eq!(batch_info_records, 18);
    assert!(batch_info_records <= 2 * tiles + 1);
}

#[test]
fn prepare_keeps_the_exact_original_quantizer_and_finite_failures_per_member() {
    let guard = include_str!("q6_integer_batch.wgsl")
        .split_once("// PROJECT_BATCH_STAGE")
        .unwrap()
        .0;
    for columns in [256, 512, 2560] {
        let actual = Kernel::Prepare(columns).source();
        let restored = actual
            .replace(guard, "let block=id.x;if block>=COLUMNS/32u {return;}")
            .replace("input[input_first+", "input[p.a+")
            .replace("planes[planes_first+", "planes[p.b+")
            .replace(BIT_FINITE, OLD_FINITE)
            .replace(
                "(bitcast<u32>(x)&0x7fffffffu)>=0x7f7fffffu",
                "!(abs(x)<3.4028234663852886e38)",
            );
        assert_eq!(restored, ScalarKernel::Prepare(columns).source());
        for member in 0..4 {
            let input = oracle::public_input(columns, member);
            let plane = oracle::Planes::new(&input).unwrap();
            assert_eq!(plane.words().len(), columns / 2 + columns / 16);
            for bad in [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MAX,
                -f32::MAX,
            ] {
                let mut damaged = input.to_vec();
                damaged[columns - 1] = bad;
                assert!(oracle::Planes::new(&damaged).is_err());
            }
        }
        let zero = oracle::Planes::new(&vec![0.; columns]).unwrap();
        assert!(zero.words().iter().all(|&word| word == 0));
    }
}

fn dot4(a: u32, b: u32) -> i32 {
    a.to_le_bytes()
        .into_iter()
        .zip(b.to_le_bytes())
        .map(|(x, y)| i32::from(x as i8) * i32::from(y as i8))
        .sum()
}
fn pair(w: &[i8; 16], planes: &[u32], columns: usize, group: usize) -> f32 {
    let mut a = 0;
    let mut b = 0;
    for word in 0..4 {
        let codes = u32::from_le_bytes(std::array::from_fn(|byte| w[word * 4 + byte] as u8));
        let at = group * 4 + word;
        a += dot4(codes, planes[at]);
        b += dot4(codes, planes[columns / 4 + at]);
    }
    let d = f32::from_bits(planes[columns / 2 + group / 2]);
    let rd = f32::from_bits(planes[columns / 2 + columns / 32 + group / 2]);
    d * (a as f32) + rd * (b as f32)
}
fn reduce(mut lanes: Vec<f32>) -> f32 {
    let mut step = lanes.len() / 2;
    while step != 0 {
        for lane in 0..step {
            lanes[lane] += lanes[lane + step];
        }
        step /= 2;
    }
    lanes[0]
}

#[test]
fn shared_original_codes_replay_each_packed_member_order_against_independent_c() {
    assert_eq!(oracle::validate_fixture().unwrap(), 6400);
    for columns in [256, 512, 2560] {
        let wire = oracle::public_rows(3, columns).unwrap();
        for count in 1..=4 {
            let plans: Vec<_> = (0..count)
                .map(|member| oracle::Planes::new(&oracle::public_input(columns, member)).unwrap())
                .collect();
            let stride = columns / 2 + columns / 16 + 11;
            let first = 5;
            let mut packed = vec![0xfeed_1234; first + count * stride];
            for (member, plan) in plans.iter().enumerate() {
                let words = plan.words();
                packed[first + member * stride..first + member * stride + words.len()]
                    .copy_from_slice(&words);
            }
            for row in wire.chunks_exact(columns / 256 * 210) {
                for width in [16, 32] {
                    let mut shared = vec![vec![0f32; width]; count];
                    // One decoded group, every real member. The natural C
                    // decoder independently authenticates codes/factor.
                    for lane in 0..width {
                        for group in (lane..columns / 16).step_by(width) {
                            let (weights, factor) = oracle::unpack16(row, columns, group).unwrap();
                            for (member, lanes) in shared.iter_mut().enumerate() {
                                let p = &packed[first + member * stride
                                    ..first + member * stride + columns / 2 + columns / 16];
                                lanes[lane] += factor * pair(&weights, p, columns, group);
                            }
                        }
                    }
                    for (member, lanes) in shared.into_iter().enumerate() {
                        let original_words = plans[member].words();
                        let mut scalar = vec![0f32; width];
                        for (lane, sum) in scalar.iter_mut().enumerate() {
                            for group in (lane..columns / 16).step_by(width) {
                                let (weights, factor) =
                                    oracle::unpack16(row, columns, group).unwrap();
                                // Independent natural signed-code products,
                                // rather than the packed quartet decoder.
                                let mut a = 0i64;
                                let mut b = 0i64;
                                for (i, &weight) in weights.iter().enumerate() {
                                    let at = group * 16 + i;
                                    let shift = (at % 4) * 8;
                                    let q = (original_words[at / 4] >> shift) as u8 as i8;
                                    let rq =
                                        (original_words[columns / 4 + at / 4] >> shift) as u8 as i8;
                                    a += i64::from(weight) * i64::from(q);
                                    b += i64::from(weight) * i64::from(rq);
                                }
                                let d = f32::from_bits(original_words[columns / 2 + group / 2]);
                                let rd = f32::from_bits(
                                    original_words[columns / 2 + columns / 32 + group / 2],
                                );
                                *sum += factor * (d * (a as f32) + rd * (b as f32));
                            }
                        }
                        let actual = reduce(lanes);
                        assert_eq!(actual.to_bits(), reduce(scalar).to_bits());
                        let expected = oracle::project(row, &plans[member]).unwrap();
                        assert!((actual - expected).abs() <= 2e-4 * (1. + expected.abs()));
                    }
                }
            }
        }
    }
}

#[test]
fn row_grid_and_tail_complement_cover_true_members_without_active_tile_reads() {
    for width in [16, 32] {
        for rows in [7usize, 2560, 16384] {
            let groups = rows.div_ceil(4);
            let mut seen = vec![0u8; rows];
            for group in 0..groups {
                for sg in 0..128 / width {
                    let row = group * (128 / width) + sg;
                    if row < rows {
                        seen[row] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&n| n == 1));
        }
    }
    for count in 1..=4 {
        let (_, project, _, output) = packets(count, 2560, 7);
        let mut full = vec![0u32; output[1] + 17];
        for member in 0..count {
            for row in 0..project.matrix_rows {
                full[project.matrix_first + member * project.output_stride + row] =
                    f32::NAN.to_bits();
            }
        }
        for (i, &word) in full.iter().enumerate() {
            let active = (0..count).any(|member| {
                let first = project.matrix_first + member * project.output_stride;
                i >= first && i < first + project.matrix_rows
            });
            // Even uncomputed active tiles are excluded from the tail scan.
            assert_eq!(active, !f32::from_bits(word).is_finite());
        }
    }
}

#[test]
fn emitted_real_sources_lower_with_none_spirv_and_no_project_function_arrays() {
    for columns in [256, 2560] {
        let spec = Spec::new(7, columns).unwrap();
        let mut kernels = vec![Kernel::Prepare(columns)];
        for wide in [false, true] {
            for tail in [false, true] {
                kernels.push(Kernel::ProjectThree(spec, tail, wide));
                kernels.push(Kernel::ProjectFour(spec, tail, wide));
                kernels.push(Kernel::ProjectPairedThree(spec, tail, wide));
                kernels.push(Kernel::ProjectPairedFour(spec, tail, wide));
            }
        }
        for kernel in kernels {
            let source = kernel.source();
            assert!(!source.contains("VECTOR"));
            assert!(source.contains(BIT_FINITE));
            assert!(!source.contains(OLD_FINITE));
            assert!(!source.contains("var<workgroup>"));
            assert!(!source.contains("fma(") && !source.contains("atomicStore"));
            let module = n::front::wgsl::parse_str(&source).unwrap();
            let info = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            if !matches!(kernel, Kernel::Prepare(_)) {
                for f in module
                    .functions
                    .iter()
                    .map(|(_, f)| f)
                    .chain(module.entry_points.iter().map(|e| &e.function))
                {
                    for (_, local) in f.local_variables.iter() {
                        assert!(!matches!(
                            module.types[local.ty].inner,
                            n::TypeInner::Array { .. }
                        ));
                    }
                }
                if matches!(
                    kernel,
                    Kernel::ProjectPairedThree(..) | Kernel::ProjectPairedFour(..)
                ) {
                    assert_eq!(
                        source
                            .matches("let codes=paired_codes(base,position)")
                            .count(),
                        1
                    );
                    assert_eq!(source.matches("let low=pair_words_at(").count(), 1);
                    assert_eq!(source.matches("let high=pair_words_at(").count(), 1);
                } else {
                    assert_eq!(source.matches("let codes=q6_words(base,sub)").count(), 1);
                }
                assert!(source.contains("d*f32(a)+rd*f32(b)"));
                assert!(source.contains("subgroupAdd(sum)"));
            }
            let words = n::back::spv::write_vec(
                &module,
                &info,
                &n::back::spv::Options {
                    lang_version: (1, 3),
                    zero_initialize_workgroup_memory:
                        n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                    ..Default::default()
                },
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
            assert_eq!(words[0], 0x0723_0203);
        }
    }
}

#[test]
fn immutable_keys_original_weight_helpers_and_checked_info_are_separate() {
    let spec = Spec::new(7, 2560).unwrap();
    for wide in [false, true] {
        let scalar = ScalarKernel::Project(spec, true, wide).source();
        let original = scalar
            .split_once("fn finite")
            .unwrap()
            .1
            .split_once("@compute")
            .unwrap()
            .0;
        assert_eq!(
            weight_helpers(wide)
                .replace(BIT_FINITE, OLD_FINITE)
                .split_once("fn finite")
                .unwrap()
                .1,
            original
        );
        let three = Kernel::ProjectThree(spec, true, wide);
        let four = Kernel::ProjectFour(spec, true, wide);
        assert_ne!(three.id(), four.id());
        assert_ne!(three.id(), ScalarKernel::Project(spec, true, wide).id());
        assert_eq!(
            three.source().matches("member_pair(codes,group,").count(),
            3
        );
        assert_eq!(four.source().matches("member_pair(codes,group,").count(), 4);
        let main = four.source();
        let main = main.split_once("fn main(").unwrap().1;
        assert!(main.find("if !valid").unwrap() < main.find("let codes=q6_words").unwrap());
        assert!(main.contains("(!FOUR || p.g==4u)"));
        assert!(main.contains("p.g-1u>(pl-PLANE_WORDS)/p.h"));
        assert!(main.contains("p.g-1u>(ml-p.f)/p.i"));
        assert!(main.contains("i>=first && i<first+p.f"));
    }
}

// Direct packed-payload replay of the paired shader; the natural C-backed
// oracle below independently supplies its16-value codes and half/subscales.
fn paired_decode(wire: &[u8], block: usize, position: usize) -> [([i8; 16], f32); 2] {
    let base = block * 210;
    let words = |at| {
        std::array::from_fn::<_, 4, _>(|i| {
            u32::from_le_bytes(wire[at + 4 * i..at + 4 * i + 4].try_into().unwrap())
        })
    };
    let shift = 2 * ((position & 3) >> 1);
    let low = words(base + 16 * position);
    let high =
        words(base + 128 + 16 * (2 * (position >> 2) + (position & 1))).map(|word| word >> shift);
    let scale_offset = 8 * (position >> 2) + (position & 3);
    let d = kuno_text::quant::f16_to_f32(u16::from_le_bytes([wire[base + 208], wire[base + 209]]));
    std::array::from_fn(|half| {
        let packed: [u32; 4] = std::array::from_fn(|word| {
            let codes = if half == 0 {
                (low[word] & 0x0f0f0f0f) | ((high[word] << 4) & 0x30303030)
            } else {
                ((low[word] >> 4) & 0x0f0f0f0f) | (((high[word] >> 4) << 4) & 0x30303030)
            };
            (codes + 0x60606060) ^ 0x80808080
        });
        let codes = std::array::from_fn(|i| (packed[i / 4] >> (8 * (i % 4))) as u8 as i8);
        let factor = d * f32::from(wire[base + 192 + scale_offset + 4 * half] as i8);
        (codes, factor)
    })
}

#[test]
fn paired_raw_payloads_and_each_signed_scale_match_every_independent_c_code() {
    assert_eq!(oracle::validate_fixture().unwrap(), 6400);
    for record in oracle::FIXTURE.chunks_exact(oracle::RECORD_BYTES).take(25) {
        let wire = &record[..210];
        let mut seen = [0u8; 16];
        for position in 0..8 {
            let scale_offset = 8 * (position >> 2) + (position & 3);
            for (half, (codes, factor)) in paired_decode(wire, 0, position).into_iter().enumerate()
            {
                let group = scale_offset + 4 * half;
                let (expected, expected_factor) = oracle::unpack16(wire, 256, group).unwrap();
                assert_eq!(codes, expected);
                assert_eq!(factor.to_bits(), expected_factor.to_bits());
                seen[group] += 1;
            }
            // Actual aligned payload envelopes remain inside the original
            // complete block, for every original row/block byte alignment.
            for prefix in 0..16 {
                for offset in [
                    16 * position,
                    128 + 16 * (2 * (position >> 2) + (position & 1)),
                ] {
                    let at = prefix + offset;
                    let end = at / 16 * 16 + if at % 16 == 0 { 16 } else { 32 };
                    assert!(end <= prefix + 210);
                }
            }
        }
        assert_eq!(seen, [1; 16]);
    }
    for record in oracle::FIXTURE.chunks_exact(oracle::RECORD_BYTES).skip(25) {
        assert!(
            paired_decode(&record[..210], 0, 0)
                .iter()
                .any(|(_, factor)| !factor.is_finite())
        );
    }
}

#[test]
fn paired_lane_order_replays_true_members_exactly_and_retains_original_oracle_bound() {
    for columns in [256, 512, 2560] {
        let wire = oracle::public_rows(3, columns).unwrap();
        for count in 1..=4 {
            let planes: Vec<_> = (0..count)
                .map(|member| oracle::Planes::new(&oracle::public_input(columns, member)).unwrap())
                .collect();
            for row in wire.chunks_exact(columns / 256 * 210) {
                for width in [16, 32] {
                    let mut packed_lanes = vec![vec![0f32; width]; count];
                    let mut scalar_lanes = vec![vec![0f32; width]; count];
                    let mut seen = vec![0u8; columns / 16];
                    for lane in 0..width {
                        let position = lane & 7;
                        let scale_offset = 8 * (position >> 2) + (position & 3);
                        for block in (lane >> 3..columns / 256).step_by(width / 8) {
                            for (half, (codes, factor)) in
                                paired_decode(row, block, position).into_iter().enumerate()
                            {
                                let group = block * 16 + scale_offset + 4 * half;
                                let (natural, natural_factor) =
                                    oracle::unpack16(row, columns, group).unwrap();
                                seen[group] += 1;
                                for (member, plan) in planes.iter().enumerate() {
                                    let words = plan.words();
                                    packed_lanes[member][lane] +=
                                        factor * pair(&codes, &words, columns, group);
                                    let mut first = 0i64;
                                    let mut residual = 0i64;
                                    for (i, &weight) in natural.iter().enumerate() {
                                        let at = group * 16 + i;
                                        let shift = (at % 4) * 8;
                                        first += i64::from(weight)
                                            * i64::from((words[at / 4] >> shift) as u8 as i8);
                                        residual += i64::from(weight)
                                            * i64::from(
                                                (words[columns / 4 + at / 4] >> shift) as u8 as i8,
                                            );
                                    }
                                    let d = f32::from_bits(words[columns / 2 + group / 2]);
                                    let rd = f32::from_bits(
                                        words[columns / 2 + columns / 32 + group / 2],
                                    );
                                    scalar_lanes[member][lane] +=
                                        natural_factor * (d * first as f32 + rd * residual as f32);
                                }
                            }
                        }
                    }
                    assert!(seen.iter().all(|&n| n == 1));
                    for (member, (actual, expected)) in
                        packed_lanes.into_iter().zip(scalar_lanes).enumerate()
                    {
                        let actual = reduce(actual);
                        assert_eq!(actual.to_bits(), reduce(expected).to_bits());
                        let independent = oracle::project(row, &planes[member]).unwrap();
                        // The existing alternate operator bound is unchanged.
                        assert!((actual - independent).abs() <= 2e-4 * (1. + independent.abs()));
                    }
                }
            }
        }
    }
}

#[test]
fn paired_sources_keep_all_original_guards_and_distinct_append_only_keys() {
    let spec = Spec::new(7, 2560).unwrap();
    for wide in [false, true] {
        for tail in [false, true] {
            for (original, paired, members) in [
                (
                    Kernel::ProjectThree(spec, tail, wide),
                    Kernel::ProjectPairedThree(spec, tail, wide),
                    3,
                ),
                (
                    Kernel::ProjectFour(spec, tail, wide),
                    Kernel::ProjectPairedFour(spec, tail, wide),
                    4,
                ),
            ] {
                assert_ne!(original.id(), paired.id());
                let old = original.source();
                let new = paired.source();
                assert!(!old.contains("PairedCodes"));
                assert!(new.contains("block+=width/8u"));
                assert!(new.contains("position=lane&7u;let initial_block=lane>>3u"));
                assert!(new.contains("group0=block*16u+scale_offset;let group1=group0+4u"));
                assert_eq!(
                    new.matches("member_pair(codes.lower,group0,").count(),
                    members
                );
                assert_eq!(
                    new.matches("member_pair(codes.upper,group1,").count(),
                    members
                );
                let old_main = old.split_once("fn main(").unwrap().1;
                let new_main = new.split_once("fn main(").unwrap().1;
                assert_eq!(
                    old_main
                        .split_once("        for(var group=lane;")
                        .unwrap()
                        .0,
                    new_main
                        .split_once("        let position=lane&7u;")
                        .unwrap()
                        .0,
                );
                assert_eq!(
                    old_main
                        .split_once("    let total=subgroupAdd(sum);")
                        .unwrap()
                        .1,
                    new_main
                        .split_once("    let total=subgroupAdd(sum);")
                        .unwrap()
                        .1,
                );
                for component in ["x", "y", "z", "w"].into_iter().take(members) {
                    assert!(new.contains(&format!("sum.{component}+=factor0*{component}0;sum.{component}+=factor1*{component}1")));
                }
                assert!(
                    new_main.find("if !valid").unwrap()
                        < new_main.find("let codes=paired_codes").unwrap()
                );
            }
        }
    }
}
