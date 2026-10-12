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
                assert_eq!(source.matches("let codes=q6_words(base,sub)").count(), 1);
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
