use super::*;
use crate::private_arena::BufferLayout;
use crate::quant_matvec::iq_fixtures;
use integer::oracle;
use kuno_text::TensorType;
use std::collections::{HashMap, HashSet};
use wgpu::naga as n;

fn geometry(dtype: u32, down: u32) -> g::Geometry {
    g::Geometry::new(
        [
            g::Projection::new(dtype, 2560, 640, 512).unwrap(),
            g::Projection::new(dtype, 2560, 640, 512).unwrap(),
            g::Projection::new(down, 640, 2560, 512).unwrap(),
        ],
        8,
        144,
    )
    .unwrap()
    .with_frozen_host_sources()
    .unwrap()
    .with_inactive_partition_exit()
    .unwrap()
}

fn compile(kernel: Kernel) -> (n::Module, Vec<u32>) {
    let source = kernel.source();
    assert!(!source.contains("_VALUE"));
    let module = n::front::wgsl::parse_str(&source).unwrap();
    let info = n::valid::Validator::new(
        n::valid::ValidationFlags::all(),
        n::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let mut bindings = Vec::new();
    for (h, global) in module.global_variables.iter() {
        if let Some(binding) = &global.binding {
            assert_eq!(binding.group, 0);
            assert!(
                !info.get_entry_point(0)[h].is_empty(),
                "unused reflected binding"
            );
            bindings.push(binding.binding);
        }
    }
    bindings.sort_unstable();
    assert_eq!(bindings, (0..bindings.len() as u32).collect::<Vec<_>>());
    for function in module
        .functions
        .iter()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter().map(|e| &e.function))
    {
        for (_, variable) in function.local_variables.iter() {
            assert!(
                !matches!(module.types[variable.ty].inner, n::TypeInner::Array { .. }),
                "Function array spill"
            );
        }
    }
    let words = n::back::spv::write_vec(
        &module,
        &info,
        &n::back::spv::Options {
            lang_version: (1, 3),
            zero_initialize_workgroup_memory: n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            ..Default::default()
        },
        Some(&n::back::spv::PipelineOptions {
            shader_stage: n::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .unwrap();
    let mut arrays = HashSet::new();
    let mut pointers = HashMap::new();
    let mut vars = Vec::new();
    let mut dot_count = 0;
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
            59 => vars.push((args[0], args[2])),
            4450 => dot_count += 1,
            _ => {}
        }
        at += length;
    }
    assert!(
        vars.iter()
            .filter(|(_, storage)| *storage == 7)
            .all(|(ty, _)| !arrays.contains(pointers.get(ty).unwrap()))
    );
    assert_eq!(dot_count > 0, matches!(kernel, Kernel::Project { .. }));
    (module, words)
}

fn shared_bytes(module: &n::Module) -> u32 {
    module
        .global_variables
        .iter()
        .filter(|(_, v)| v.space == n::AddressSpace::WorkGroup)
        .map(|(_, v)| match module.types[v.ty].inner {
            n::TypeInner::Array {
                size: n::ArraySize::Constant(size),
                stride,
                ..
            } => size.get() * stride,
            n::TypeInner::Atomic(_) => 4,
            _ => panic!("unexpected workgroup type"),
        })
        .sum()
}
fn barriers(block: &n::Block) -> usize {
    block
        .iter()
        .map(|s| match s {
            n::Statement::ControlBarrier(_) => 1,
            n::Statement::Block(b) => barriers(b),
            n::Statement::If { accept, reject, .. } => barriers(accept) + barriers(reject),
            n::Statement::Loop {
                body, continuing, ..
            } => barriers(body) + barriers(continuing),
            n::Statement::Switch { cases, .. } => cases.iter().map(|c| barriers(&c.body)).sum(),
            _ => 0,
        })
        .sum()
}

#[test]
fn frozen_gu_integer_all48_actual_profiles_and_five_source_partitions_are_checked() {
    let mut counts = HashMap::new();
    for layer in 0..48 {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
        assert_eq!(gate, up);
        *counts.entry(gate).or_insert(0) += 1;
        let base = geometry(gate, down);
        validate_geometry(base).unwrap();
        validate_geometry(base.with_four_row_down().unwrap()).unwrap();
        for (first, slots) in [(0, 72), (72, 72), (144, 144), (288, 144), (432, 80)] {
            base.validate_shard(
                first,
                slots,
                base.projections().map(|p| p.segment_words(slots).unwrap()),
            )
            .unwrap();
        }
        for bad in [
            base.with_private_tables().unwrap(),
            base.with_workgroup_tables().unwrap(),
            base.with_shared_gate_up_factors().unwrap(),
            base.with_two_row_gate_up().unwrap(),
            base.with_three_member_arithmetic().unwrap(),
        ] {
            assert!(validate_geometry(bad).is_err());
        }
    }
    assert_eq!(
        counts,
        HashMap::from([
            (super::super::IQ2_S, 20),
            (super::super::IQ3_XXS, 17),
            (super::super::IQ3_S, 10),
            (super::super::IQ4_XS, 1)
        ])
    );
    assert!(validate_count(0).is_err());
    assert!(validate_count(4).is_err());
    assert_eq!(
        BufferLayout::new(SCRATCH_WORDS).unwrap().owned_bytes as u64,
        MAIN_BYTES
    );
    assert_eq!(BufferLayout::new(LOGICAL_WORDS).unwrap().owned_bytes, 32768);
}

#[test]
fn frozen_gu_integer_full_spans_counts_limits_and_complete_scratch_are_required() {
    let g = geometry(super::super::IQ2_S, super::super::IQ4_NL);
    let spans = g::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
    let words = [
        g::MAP_WORDS,
        8 * 2560,
        80 * 640,
        80 * 2560,
        80,
        8 * 2560,
        g.status_words(),
    ];
    for count in 1..=3 {
        validate_layout(g, &spans, words, SCRATCH_WORDS, count, 80).unwrap();
    }
    for index in 0..7 {
        let mut short = words;
        short[index] -= 1;
        assert!(validate_layout(g, &spans, short, SCRATCH_WORDS, 3, 80).is_err());
    }
    for scratch in [4384, 8192, 16383, 16385] {
        assert!(validate_layout(g, &spans, words, scratch, 3, 80).is_err());
    }
    for count in [0, 4, 8, usize::MAX] {
        assert!(validate_layout(g, &spans, words, SCRATCH_WORDS, count, 80).is_err());
    }
    for groups in [0, 81, usize::MAX] {
        assert!(validate_layout(g, &spans, words, SCRATCH_WORDS, 3, groups).is_err());
    }
    let overlapping = g::Spans::new([0, 0, 2559, 0, 640, 0, 2560, 0, 0, 2560, 0]);
    assert!(validate_layout(g, &overlapping, words, SCRATCH_WORDS, 3, 80).is_err());
    let limits = wgpu::Limits {
        max_compute_invocations_per_workgroup: 128,
        max_compute_workgroup_size_x: 128,
        max_compute_workgroup_storage_size: MAX_SHARED_BYTES,
        max_storage_buffers_per_shader_stage: 6,
        max_bindings_per_bind_group: 7,
        max_uniform_buffer_binding_size: 32,
        max_compute_workgroups_per_dimension: 128,
        ..Default::default()
    };
    validate_limits(&limits).unwrap();
    let mut small = limits;
    small.max_compute_workgroup_storage_size -= 1;
    assert!(validate_limits(&small).is_err());
}

#[test]
fn frozen_gu_integer_actual_shaders_spirv_none_have_packed_dots_and_initialized_shared_cells() {
    let (prepare, _) = compile(Kernel::Prepare);
    assert_eq!(prepare.entry_points[0].workgroup_size, [128, 1, 1]);
    assert_eq!(shared_bytes(&prepare), MAX_PREPARE_SHARED_BYTES);
    assert_eq!(barriers(&prepare.entry_points[0].function.body), 2);
    for dtype in [
        super::super::IQ2_S,
        super::super::IQ3_XXS,
        super::super::IQ3_S,
        super::super::IQ4_XS,
    ] {
        let row_bytes = TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(2560)
            .unwrap() as usize;
        let kernel = Kernel::Project { dtype, row_bytes };
        let source = kernel.source();
        let (module, _) = compile(kernel);
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert_eq!(
            shared_bytes(&module),
            1536 + integer::original_grid_words(dtype).len() as u32 * 4
        );
        assert!(shared_bytes(&module) <= MAX_PROJECT_SHARED_BYTES);
        assert_eq!(barriers(&module.entry_points[0].function.body), 3);
        assert!(source.contains("block=lane;block<80u;block+=8u"));
        assert!(source.contains("for(var stride=4u;stride>0u;stride/=2u)"));
        assert!(
            !source.contains("subgroupAdd")
                && !source.contains("THREE_PACKED")
                && !source.contains("var<private>")
        );
        // Every scalar reduction cell is initialized, including unused RHS.
        assert!(source.contains("j<3u;j++") && source.contains("gate_sums[local*3u+j]=gates[j]"));
        let words = integer::original_grid_words(dtype);
        for (index, word) in words.into_iter().enumerate() {
            assert_eq!(
                source
                    .matches(&format!("iq3_grid[{index}u]=0x{word:08x}u;"))
                    .count(),
                1
            );
        }
    }
    let (audit, _) = compile(Kernel::Audit);
    assert_eq!(barriers(&audit.entry_points[0].function.body), 0);
    assert!(Kernel::Audit.source().contains("atomicOr(&fault[0],8u)"));
    assert!(!Kernel::Audit.source().contains("atomicStore"));
}

fn contract(source: &str) -> bool {
    [
        "p.h<1u || p.h>3u",
        "arrayLength(&planes)!=16384u",
        "map[p.a]!=p.h",
        "slot<p.b || slot-p.b>=p.c",
        "route>=p.h*10u",
        "planes[route/10u]!=1u",
        "map[p.a+164u+group*8u+k]/10u",
        "group.x>=80u || group.z!=0u || !valid_group(group.y)",
        "79u>(arrayLength(&output)-p.d-640u)/p.e",
        "80u>arrayLength(&status)-p.f",
    ]
    .iter()
    .all(|p| source.contains(p))
        && source.find("!valid_group(group.y)").unwrap() < source.find("let offset=").unwrap()
        && source.find("let offset=").unwrap() < source.find("workgroupBarrier();").unwrap()
}
#[test]
fn frozen_gu_integer_guards_and_negative_mutations_retain_full_wave_and_inactive_exit() {
    let source = Kernel::Project {
        dtype: super::super::IQ2_S,
        row_bytes: 820,
    }
    .source();
    assert!(contract(&source));
    for guard in [
        "route>=p.h*10u",
        "slot<p.b || slot-p.b>=p.c",
        "planes[route/10u]!=1u",
        "group.x>=80u || group.z!=0u || !valid_group(group.y)",
        "79u>(arrayLength(&output)-p.d-640u)/p.e",
    ] {
        assert!(
            !contract(&source.replace(guard, "false")),
            "missing guard accepted: {guard}"
        );
    }
    let prepare = Kernel::Prepare.source();
    assert!(prepare.contains("if local>=3u && local<64u {planes[local]=0u;}"));
    assert!(prepare.contains("if local==0u {planes[token]=0u;}"));
    assert!(prepare.contains("token>(arrayLength(&input)-p.b-2560u)/p.c"));
    // All reset writes partition the full owning owner with no shared flag.
    let mut writers = vec![0u8; SCRATCH_WORDS];
    for token in 0..3 {
        writers[token] += 1;
        for word in &mut writers
            [HEADER_WORDS + token * FRAME_WORDS..HEADER_WORDS + (token + 1) * FRAME_WORDS]
        {
            *word += 1;
        }
        if token == 0 {
            for word in &mut writers[3..HEADER_WORDS] {
                *word += 1;
            }
            for word in &mut writers[LOGICAL_WORDS..] {
                *word += 1;
            }
        }
    }
    assert!(writers.into_iter().all(|w| w == 1));
}

// Independent C-backed code extraction, packed activation oracle, exact i32
// block sums, then the declared eight-lane F32 tree. No shader decoder calls.
fn packed_dot(dtype: TensorType, wire: &[u8], packed: &[u32]) -> f32 {
    let mut lanes = [0f32; 8];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        for block in (lane..80).step_by(8) {
            for half in 0..if dtype == TensorType::Iq2S { 2 } else { 1 } {
                let (codes, factor) = if dtype == TensorType::Iq2S {
                    let (c, f) = oracle::unpack_iq2_s16(wire, 2560, block * 2 + half).unwrap();
                    (c.to_vec(), f)
                } else {
                    let (c, f) = oracle::unpack32(dtype, wire, 2560, block).unwrap();
                    (c.to_vec(), f)
                };
                let mut a = 0i32;
                let mut b = 0i32;
                for (i, code) in codes.into_iter().enumerate() {
                    let column = block * 32 + half * 16 + i;
                    let word = column / 4;
                    let shift = column % 4 * 8;
                    a += i32::from(code) * i32::from((packed[word] >> shift) as u8 as i8);
                    b += i32::from(code) * i32::from((packed[640 + word] >> shift) as u8 as i8);
                }
                let pair = f32::from_bits(packed[1280 + block]) * a as f32
                    + f32::from_bits(packed[1360 + block]) * b as f32;
                *sum += factor * pair;
            }
        }
    }
    for stride in [4, 2, 1] {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
    }
    lanes[0]
}
#[test]
fn frozen_gu_integer_eight_rows_three_real_routes_match_independent_full_projection_vectors() {
    for dtype in [
        TensorType::Iq2S,
        TensorType::Iq3Xxs,
        TensorType::Iq3S,
        TensorType::Iq4Xs,
    ] {
        let gate = iq_fixtures::packed(dtype.gguf_id(), 8, 2560, 1);
        let up = iq_fixtures::packed(dtype.gguf_id(), 8, 2560, 3);
        let row_bytes = dtype.row_bytes(2560).unwrap() as usize;
        let inputs = (0..3)
            .map(|token| {
                (0..2560)
                    .map(|i| ((i * (17 + token * 7)) % 311) as f32 / 193. - 0.7)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let planes = inputs
            .iter()
            .map(|x| oracle::Planes::new(x).unwrap())
            .collect::<Vec<_>>();
        let packed = planes
            .iter()
            .map(|p| p.packed_words().unwrap())
            .collect::<Vec<_>>();
        for count in 1..=3 {
            let routes = match count {
                1 => vec![7],
                2 => vec![14, 3],
                _ => vec![27, 3, 15],
            };
            for route in routes {
                let token = route / 10;
                for row in 0..8 {
                    let gbytes = &gate[row * row_bytes..(row + 1) * row_bytes];
                    let ubytes = &up[row * row_bytes..(row + 1) * row_bytes];
                    for bytes in [gbytes, ubytes] {
                        let actual = packed_dot(dtype, bytes, &packed[token]);
                        let expected = oracle::project(dtype, bytes, &planes[token]).unwrap();
                        assert!(
                            (actual - expected).abs() <= 1e-4 + 3e-4 * expected.abs(),
                            "dtype={dtype:?} count={count} route={route} row={row}"
                        );
                    }
                    let a = packed_dot(dtype, gbytes, &packed[token]);
                    let b = packed_dot(dtype, ubytes, &packed[token]);
                    let sigmoid = if a >= 0. {
                        1. / (1. + (-a).exp())
                    } else {
                        let e = a.exp();
                        e / (1. + e)
                    };
                    assert!((a * sigmoid * b).is_finite());
                }
            }
        }
    }
}
