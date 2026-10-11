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
    assert_eq!(
        dot_count > 0,
        matches!(
            kernel,
            Kernel::Project { .. } | Kernel::ProjectStride4 { .. }
        )
    );
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

#[derive(Clone, Copy)]
struct Encoded {
    codes: [u32; 4],
    signs: u32,
    high: u32,
    scale: u32,
    d: f32,
}
// Independent host model of the new encoded-register fields, with every
// physical u32 source load counted and bounds checked. Expected codes/factors
// below come from the separate scalar original decoder/C fixture records.
struct Wire<'a> {
    bytes: &'a [u8],
    loads: usize,
}
impl Wire<'_> {
    fn u32(&mut self, index: usize) -> u32 {
        self.loads += 1;
        u32::from_le_bytes(self.bytes[index * 4..index * 4 + 4].try_into().unwrap())
    }
    fn byte(&mut self, at: usize) -> u32 {
        (self.u32(at / 4) >> ((at % 4) * 8)) & 255
    }
    fn word(&mut self, at: usize) -> u32 {
        let shift = at % 4 * 8;
        let low = self.u32(at / 4);
        if shift == 0 {
            low
        } else {
            low >> shift | self.u32(at / 4 + 1) << (32 - shift)
        }
    }
    fn load8(&mut self, at: usize) -> [u32; 2] {
        assert!(at.is_multiple_of(2));
        let low = self.u32(at / 4);
        let middle = self.u32(at / 4 + 1);
        if at.is_multiple_of(4) {
            [low, middle]
        } else {
            let high = self.u32(at / 4 + 2);
            [low >> 16 | middle << 16, middle >> 16 | high << 16]
        }
    }
    fn half(&mut self, at: usize) -> f32 {
        assert!(at.is_multiple_of(2));
        kuno_text::quant::f16_to_f32((self.u32(at / 4) >> ((at % 4) * 8)) as u16)
    }
    fn prepare(&mut self, dtype: TensorType, row: usize, block: usize) -> Encoded {
        let group = block % 8;
        let base = row + block / 8 * dtype.block_bytes() as usize;
        let mut e = Encoded {
            codes: [0; 4],
            signs: 0,
            high: 0,
            scale: 0,
            d: 0.,
        };
        match dtype {
            TensorType::Iq3Xxs => {
                e.codes[..2].copy_from_slice(&self.load8(base + 2 + group * 8));
                e.signs = self.word(base + 66 + group * 4);
                e.d = self.half(base);
            }
            TensorType::Iq3S => {
                e.codes[..2].copy_from_slice(&self.load8(base + 2 + group * 8));
                e.signs = self.word(base + 74 + group * 4);
                e.high = self.byte(base + 66 + group);
                e.scale = self.byte(base + 106 + group / 2) >> (group % 2 * 4) & 15;
                e.d = self.half(base);
            }
            TensorType::Iq2S => {
                e.codes[0] = self.word(base + 2 + group * 4);
                e.signs = self.word(base + 34 + group * 4);
                e.high = self.byte(base + 66 + group);
                e.scale = self.byte(base + 74 + group);
                e.d = self.half(base);
            }
            TensorType::Iq4Xs => {
                let at = base + 8 + group * 16;
                assert!(base.is_multiple_of(4) && at.is_multiple_of(4));
                let header = self.u32(base / 4);
                e.scale = self.byte(base + 4 + group / 2) >> (group % 2 * 4) & 15;
                e.scale |= (header >> (16 + group * 2) & 3) << 4;
                for (j, code) in e.codes.iter_mut().enumerate() {
                    *code = self.u32(at / 4 + j);
                }
                e.d = kuno_text::quant::f16_to_f32(header as u16);
            }
            _ => panic!("frozen GU dtype"),
        }
        e
    }
}
impl Encoded {
    fn factor(self, dtype: TensorType, half: usize) -> f32 {
        match dtype {
            TensorType::Iq3Xxs => self.d * (0.5 + (self.signs >> 28) as f32) * 0.5,
            TensorType::Iq3S => self.d * (1 + 2 * self.scale) as f32,
            TensorType::Iq2S => self.d * (0.5 + ((self.scale >> (half * 4)) & 15) as f32) * 0.25,
            TensorType::Iq4Xs => self.d * (self.scale as i32 - 32) as f32,
            _ => panic!("frozen GU dtype"),
        }
    }
    fn packed(self, dtype: TensorType, quartet: usize) -> u32 {
        let octet = quartet / 2;
        let (magnitude, signs) = match dtype {
            TensorType::Iq3Xxs => {
                let sign_code = self.signs >> (octet * 7) & 127;
                let signs = sign_code | ((sign_code.count_ones() & 1) << 7);
                let index = self.codes[quartet / 4] >> (quartet % 4 * 8) & 255;
                (kuno_text::quant::IQ3_XXS_GRID[index as usize], signs)
            }
            TensorType::Iq3S => {
                let index = (self.codes[quartet / 4] >> (quartet % 4 * 8) & 255)
                    | ((self.high >> quartet & 1) << 8);
                (
                    kuno_text::quant::IQ3_S_GRID[index as usize],
                    self.signs >> (octet * 8) & 255,
                )
            }
            TensorType::Iq2S => {
                let index =
                    (self.codes[0] >> (octet * 8) & 255) | ((self.high >> (octet * 2) & 3) << 8);
                let grid = kuno_text::quant::IQ2_S_GRID[index as usize];
                (
                    (grid >> (quartet % 2 * 32)) as u32,
                    self.signs >> (octet * 8) & 255,
                )
            }
            TensorType::Iq4Xs => {
                let raw = self.codes[quartet % 4];
                let lut = [0xbfad9881u32, 0xf6eaddcf, 0x26190d01, 0x71594535];
                let mut packed = 0;
                for byte in 0..4 {
                    let code = raw >> (byte * 8 + usize::from(quartet >= 4) * 4) & 15;
                    packed |= ((lut[(code / 4) as usize] >> ((code % 4) * 8)) & 255) << (byte * 8);
                }
                return packed;
            }
            _ => panic!("frozen GU dtype"),
        };
        let signs = signs >> (quartet % 2 * 4);
        let mask = (0..4).fold(0, |mask, byte| {
            mask | ((signs >> byte & 1) * 255) << (byte * 8)
        });
        (magnitude ^ mask).wrapping_add(mask & 0x01010101)
    }
}

#[test]
fn frozen_gu_integer_register_preparation_matches_every_original_code_and_factor_bit() {
    for (dtype, expected_loads) in [
        (TensorType::Iq2S, [5, 7]),
        (TensorType::Iq3Xxs, [4, 6]),
        (TensorType::Iq3S, [6, 8]),
        (TensorType::Iq4Xs, [6, 6]),
    ] {
        let row_bytes = dtype.row_bytes(2560).unwrap() as usize;
        let mut observed_loads = HashSet::new();
        for phase in 0..32 {
            let bytes = iq_fixtures::packed(dtype.gguf_id(), 8, 2560, phase);
            let mut wire = Wire {
                bytes: &bytes,
                loads: 0,
            };
            for row in 0..8 {
                let original = &bytes[row * row_bytes..(row + 1) * row_bytes];
                for block in 0..80 {
                    let before = wire.loads;
                    let e = wire.prepare(dtype, row * row_bytes, block);
                    observed_loads.insert(wire.loads - before);
                    for half in 0..if dtype == TensorType::Iq2S { 2 } else { 1 } {
                        let (codes, factor) = if dtype == TensorType::Iq2S {
                            let (codes, factor) =
                                oracle::unpack_iq2_s16(original, 2560, block * 2 + half).unwrap();
                            (codes.to_vec(), factor)
                        } else {
                            let (codes, factor) =
                                oracle::unpack32(dtype, original, 2560, block).unwrap();
                            (codes.to_vec(), factor)
                        };
                        assert_eq!(e.factor(dtype, half).to_bits(), factor.to_bits());
                        for (quartet, codes) in codes.chunks_exact(4).enumerate() {
                            let codes: [i8; 4] = codes.try_into().unwrap();
                            let expected = u32::from_le_bytes(codes.map(|v| v as u8));
                            assert_eq!(e.packed(dtype, half * 4 + quartet), expected);
                        }
                    }
                }
            }
        }
        assert_eq!(observed_loads, HashSet::from(expected_loads));
    }
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
        // Encoded fields are loaded before either half/quartet loop. Neither
        // helper can reload its source bank while expanding cached quartets.
        assert!(
            source.find("gate_prepare(offset,block)").unwrap()
                < source.find("for(var half=0u;").unwrap()
        );
        assert!(
            source.contains("gate_packed(gprepared,quartet)")
                && source.contains("gate_factor(gprepared,half)")
        );
        let bank = module
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("gate"))
            .unwrap()
            .0;
        for name in ["gate_packed", "gate_factor"] {
            let function = module
                .functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(name))
                .unwrap()
                .1;
            assert!(
                !function
                    .expressions
                    .iter()
                    .any(|(_, e)| matches!(e, n::Expression::GlobalVariable(v) if *v==bank))
            );
        }
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

#[test]
fn frozen_gu_integer_group_stride_four_keeps_one_exact_arithmetic_body_and_uniform_group_fences() {
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
        let original = Kernel::Project { dtype, row_bytes };
        let alternative = Kernel::ProjectStride4 { dtype, row_bytes };
        assert_ne!(original.id(), alternative.id());
        let original_source = original.source();
        let source = alternative.source();
        let (module, _) = compile(alternative);
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert_eq!(barriers(&module.entry_points[0].function.body), 4);
        assert_eq!(
            shared_bytes(&module),
            1536 + integer::original_grid_words(dtype).len() as u32 * 4
        );
        let original_body = original_source
            .split_once("// FROZEN_GU_PROJECT_BODY_BEGIN\n")
            .unwrap()
            .1
            .split_once("// FROZEN_GU_PROJECT_BODY_END")
            .unwrap()
            .0;
        // The arithmetic is extracted once; only the runtime group address is
        // renamed, with the same blocks, factors, members and reduction order.
        assert!(source.contains(&original_body.replace("group.y", "routed_group")));
        assert!(
            source.find("if !any_valid {return;}").unwrap()
                < source.find("workgroupBarrier();").unwrap()
        );
        if !integer::original_grid_words(dtype).is_empty() {
            assert!(
                source.find("if !any_valid {return;}").unwrap()
                    < source.find("iq3_grid[0u]=").unwrap()
            );
        }
        assert_eq!(
            source
                .matches("routed_group=group.y;routed_group<p.g;routed_group+=4u")
                .count(),
            2
        );
        assert!(source.contains("group>=groups || group>=p.g"));
        assert!(source.contains("p.g<1u || p.g>80u"));
        assert!(source.contains("group.y>=4u"));
        assert!(source.contains("if valid_group(routed_group) {"));
        assert!(source.contains("gate_sums[local*3u+j]=gates[j]"));
        assert!(source.contains("up_sums[local*3u+j]=ups[j]"));
        let last_output = source.rfind("status[p.f+route]=1u").unwrap();
        assert!(last_output < source.rfind("workgroupBarrier();").unwrap());
        for (index, word) in integer::original_grid_words(dtype).into_iter().enumerate() {
            assert_eq!(
                source
                    .matches(&format!("iq3_grid[{index}u]=0x{word:08x}u;"))
                    .count(),
                1
            );
        }
    }
    assert_eq!(GROUP_STRIDE_FOUR, 4);
    assert_eq!(5 * 80 * GROUP_STRIDE_FOUR, 1600);
}

#[test]
fn frozen_gu_integer_group_stride_four_covers_exact_original_groups_and_routes_once() {
    let partitions = [(0, 72), (72, 72), (144, 144), (288, 144), (432, 80)];
    // Every legal envelope and every public count, including envelopes below
    // and above actual MAP group count. Synthetic slots span all original
    // sources and valid groups retain distinct original route identities.
    for count in 1..=3 {
        for actual_groups in 1..=count * 10 {
            for envelope in 1..=80 {
                let slots: Vec<usize> = (0..actual_groups)
                    .map(|g| (g * 173 + count * 71) % 512)
                    .collect();
                let mut original = Vec::new();
                let mut scheduled = Vec::new();
                for (shard, &(first, length)) in partitions.iter().enumerate() {
                    original.extend(
                        (0..envelope.min(actual_groups))
                            .filter(|&g| slots[g] >= first && slots[g] < first + length)
                            .map(|g| (shard, g)),
                    );
                    for y in 0..GROUP_STRIDE_FOUR as usize {
                        for g in (y..envelope).step_by(GROUP_STRIDE_FOUR as usize) {
                            if g < actual_groups && slots[g] >= first && slots[g] < first + length {
                                scheduled.push((shard, g));
                            }
                        }
                    }
                }
                original.sort_unstable();
                scheduled.sort_unstable();
                assert_eq!(scheduled, original);
                assert_eq!(scheduled.len(), envelope.min(actual_groups));
                assert!(scheduled.windows(2).all(|pair| pair[0] != pair[1]));
            }
        }
    }
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

fn prepared_dot(dtype: TensorType, bytes: &[u8], packed: &[u32]) -> f32 {
    let mut wire = Wire { bytes, loads: 0 };
    let mut lanes = [0f32; 8];
    for (lane, sum) in lanes.iter_mut().enumerate() {
        for block in (lane..80).step_by(8) {
            let e = wire.prepare(dtype, 0, block);
            for half in 0..if dtype == TensorType::Iq2S { 2 } else { 1 } {
                let quartets = if dtype == TensorType::Iq2S { 4 } else { 8 };
                let mut a = 0i32;
                let mut b = 0i32;
                for quartet in half * quartets..(half + 1) * quartets {
                    let codes = e.packed(dtype, quartet).to_le_bytes();
                    let first = packed[block * 8 + quartet].to_le_bytes();
                    let second = packed[640 + block * 8 + quartet].to_le_bytes();
                    let mut dot_first = 0;
                    let mut dot_second = 0;
                    for i in 0..4 {
                        dot_first += i32::from(codes[i] as i8) * i32::from(first[i] as i8);
                        dot_second += i32::from(codes[i] as i8) * i32::from(second[i] as i8);
                    }
                    a += dot_first;
                    b += dot_second;
                }
                let pair = f32::from_bits(packed[1280 + block]) * a as f32
                    + f32::from_bits(packed[1360 + block]) * b as f32;
                *sum += e.factor(dtype, half) * pair;
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
                        assert_eq!(
                            prepared_dot(dtype, bytes, &packed[token]).to_bits(),
                            actual.to_bits(),
                            "register prep changed candidate math dtype={dtype:?} count={count} route={route} row={row}"
                        );
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
