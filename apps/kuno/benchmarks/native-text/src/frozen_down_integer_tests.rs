use super::*;
use crate::private_arena::BufferLayout;
use crate::quant_matvec::iq_fixtures;
use crate::quant_matvec::iq_moe_integer::oracle;
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

#[test]
fn frozen_down_integer_all48_original_profiles_and_five_partitions_are_bounded() {
    let mut counts = HashMap::new();
    for layer in 0..48 {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
        assert_eq!(gate, up);
        *counts.entry(down).or_insert(0) += 1;
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
            base.with_shared_down_scales().unwrap(),
            base.with_two_row_gate_up().unwrap(),
            base.with_three_member_arithmetic().unwrap(),
        ] {
            assert!(validate_geometry(bad).is_err());
        }
    }
    assert_eq!(
        counts,
        HashMap::from([(super::super::IQ4_NL, 39), (super::super::Q2_0, 9)])
    );
    for count in [0, 4, 8, usize::MAX] {
        assert!(validate_count(count).is_err());
    }
}

#[test]
fn frozen_down_integer_exact_owners_counts_limits_and_independent_charges() {
    let geometry = geometry(super::super::IQ2_S, super::super::IQ4_NL);
    let spans = g::Spans::new([0, 0, 2560, 0, 640, 0, 2560, 0, 0, 2560, 0]);
    let words = [
        g::MAP_WORDS,
        8 * 2560,
        80 * 640,
        80 * 2560,
        80,
        8 * 2560,
        geometry.status_words(),
    ];
    for count in 1..=3 {
        for groups in 1..=MAX_ROUTES {
            validate_layout(geometry, &spans, words, SCRATCH_WORDS, count, groups).unwrap();
        }
    }
    for owner in 0..7 {
        let mut short = words;
        short[owner] -= 1;
        assert!(validate_layout(geometry, &spans, short, SCRATCH_WORDS, 3, 30).is_err());
    }
    for short in [LOGICAL_WORDS, 16383, 16385] {
        assert!(validate_layout(geometry, &spans, words, short, 3, 30).is_err());
    }
    for groups in [0, 31, usize::MAX] {
        assert!(validate_layout(geometry, &spans, words, SCRATCH_WORDS, 3, groups).is_err());
    }
    let overlapping = g::Spans::new([0, 0, 2560, 0, 639, 0, 2560, 0, 0, 2560, 0]);
    assert!(validate_layout(geometry, &overlapping, words, SCRATCH_WORDS, 3, 30).is_err());
    assert_eq!(
        (HEADER_WORDS, FRAME_WORDS, LOGICAL_WORDS, OWNED_WORDS),
        (64, 360, 10864, 16384)
    );
    assert_eq!(
        BufferLayout::new(SCRATCH_WORDS).unwrap().owned_bytes as u64,
        65536
    );
    assert_eq!(MAIN_BYTES, 65536);
    assert_eq!(HOST_INCREMENT, 65536);
    assert_eq!(INFO_INCREMENT, 196608);
    assert_eq!(RUNTIME_INCREMENT, 462848);
    let limits = wgpu::Limits {
        max_compute_invocations_per_workgroup: 64,
        max_compute_workgroup_size_x: 64,
        max_compute_workgroup_storage_size: MAX_SHARED_BYTES,
        max_storage_buffers_per_shader_stage: 5,
        max_bindings_per_bind_group: 6,
        max_uniform_buffer_binding_size: 32,
        max_compute_workgroups_per_dimension: 320,
        max_storage_buffer_binding_size: MAIN_BYTES,
        max_buffer_size: MAIN_BYTES,
        ..Default::default()
    };
    validate_limits(&limits).unwrap();
    for axis in 0..9 {
        let mut small = limits.clone();
        match axis {
            0 => small.max_compute_invocations_per_workgroup -= 1,
            1 => small.max_compute_workgroup_size_x -= 1,
            2 => small.max_compute_workgroup_storage_size -= 1,
            3 => small.max_storage_buffers_per_shader_stage -= 1,
            4 => small.max_bindings_per_bind_group -= 1,
            5 => small.max_uniform_buffer_binding_size -= 1,
            6 => small.max_compute_workgroups_per_dimension -= 1,
            7 => small.max_storage_buffer_binding_size -= 1,
            8 => small.max_buffer_size -= 1,
            _ => unreachable!(),
        }
        assert!(validate_limits(&small).is_err(), "limit axis {axis}");
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
fn frozen_down_integer_actual_shader_spirv_none_and_cached_source_decoding() {
    let (prepare, _) = compile(Kernel::Prepare);
    assert_eq!(prepare.entry_points[0].workgroup_size, [64, 1, 1]);
    assert_eq!(shared_bytes(&prepare), MAX_PREPARE_SHARED_BYTES);
    assert_eq!(barriers(&prepare.entry_points[0].function.body), 2);
    for dtype in [super::super::IQ4_NL, super::super::Q2_0] {
        let row_bytes = TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(640)
            .unwrap() as usize;
        let kernel = Kernel::Project { dtype, row_bytes };
        let source = kernel.source();
        let (module, _) = compile(kernel);
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert_eq!(shared_bytes(&module), MAX_PROJECT_SHARED_BYTES);
        assert_eq!(barriers(&module.entry_points[0].function.body), 2);
        assert!(
            source.find("!valid_group(group.y)").unwrap()
                < source.find("down_prepare(offset,block)").unwrap()
        );
        assert!(
            source.find("down_prepare(offset,block)").unwrap()
                < source.find("for(var quartet=0u;").unwrap()
        );
        assert!(source.contains("block=lane;block<20u;block+=8u"));
        assert!(source.contains("j<3u;j++") && source.contains("sums[local*3u+j]=values[j]"));
        assert!(source.contains("for(var stride=4u;stride>0u;stride/=2u)"));
        assert!(!source.contains("subgroup") && !source.contains("var<private>"));
        let bank = module
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("down"))
            .unwrap()
            .0;
        for name in ["down_packed", "down_factor"] {
            let f = module
                .functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(name))
                .unwrap()
                .1;
            assert!(
                !f.expressions
                    .iter()
                    .any(|(_, e)| matches!(e,n::Expression::GlobalVariable(v) if *v==bank))
            );
        }
        // Both checked/direct execution compilation preserve this explicit raw
        // WGSL contract; trusted roots are the parent's independent obligation.
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let compiled = kernel
                .compile(
                    &mut AutoCompiler::Wgsl(Default::default()),
                    &Default::default(),
                    mode,
                    kernel.address_type(),
                )
                .unwrap();
            assert_eq!(compiled.source, source);
            assert!(compiled.repr.is_none());
        }
    }
    let (audit, _) = compile(Kernel::Audit);
    assert_eq!(shared_bytes(&audit), 0);
    assert_eq!(barriers(&audit.entry_points[0].function.body), 0);
    let audit = Kernel::Audit.source();
    assert!(audit.contains("atomicOr(&fault[0],8u)"));
    assert!(!audit.contains("atomicStore"));
}

struct Wire<'a> {
    bytes: &'a [u8],
    loads: usize,
}
#[derive(Clone, Copy)]
struct Encoded {
    codes: [u32; 4],
    d: f32,
}
impl Wire<'_> {
    fn word(&mut self, at: usize) -> u32 {
        self.loads += 1;
        u32::from_le_bytes(self.bytes[at * 4..at * 4 + 4].try_into().unwrap())
    }
    fn load8(&mut self, at: usize) -> [u32; 2] {
        assert!(at.is_multiple_of(2));
        let low = self.word(at / 4);
        let middle = self.word(at / 4 + 1);
        if at.is_multiple_of(4) {
            [low, middle]
        } else {
            let high = self.word(at / 4 + 2);
            [low >> 16 | middle << 16, middle >> 16 | high << 16]
        }
    }
    fn prepare(&mut self, dtype: TensorType, row: usize, block: usize) -> Encoded {
        let base = row
            + match dtype {
                TensorType::Iq4Nl => block * 18,
                TensorType::Q2_0 => block / 2 * 18,
                _ => panic!(),
            };
        let mut codes = [0; 4];
        match dtype {
            TensorType::Iq4Nl => {
                codes[..2].copy_from_slice(&self.load8(base + 2));
                codes[2..].copy_from_slice(&self.load8(base + 10));
            }
            TensorType::Q2_0 => codes[..2].copy_from_slice(&self.load8(base + 2 + block % 2 * 8)),
            _ => panic!(),
        }
        let d = kuno_text::quant::f16_to_f32((self.word(base / 4) >> (base % 4 * 8)) as u16);
        Encoded { codes, d }
    }
}
impl Encoded {
    fn packed(self, dtype: TensorType, quartet: usize) -> u32 {
        let mut result = 0;
        for byte in 0..4 {
            let code = match dtype {
                TensorType::Iq4Nl => {
                    // Scalar codebook, independent of shader's four packed LUT
                    // words. Fixture assertions use independently compiled C.
                    let index =
                        self.codes[quartet % 4] >> (byte * 8 + usize::from(quartet >= 4) * 4) & 15;
                    kuno_text::quant::IQ4_NL_VALUES[index as usize]
                }
                TensorType::Q2_0 => {
                    let bits = self.codes[quartet / 4] >> (quartet % 4 * 8);
                    ((bits >> (byte * 2)) & 3) as i8 - 1
                }
                _ => panic!(),
            };
            result |= u32::from(code as u8) << (byte * 8);
        }
        result
    }
}

#[test]
fn frozen_down_integer_original_cached_codes_and_factors_match_independent_c_bits() {
    for (dtype, loads) in [(TensorType::Iq4Nl, [5, 7]), (TensorType::Q2_0, [3, 4])] {
        let mut observed = HashSet::new();
        let row_bytes = dtype.row_bytes(640).unwrap() as usize;
        for phase in 0..32 {
            let bytes = iq_fixtures::packed(dtype.gguf_id(), 8, 640, phase);
            let expected = iq_fixtures::oracle(dtype.gguf_id(), 8, 640, phase);
            let mut wire = Wire {
                bytes: &bytes,
                loads: 0,
            };
            for row in 0..8 {
                for block in 0..20 {
                    let before = wire.loads;
                    let encoded = wire.prepare(dtype, row * row_bytes, block);
                    observed.insert(wire.loads - before);
                    let original = &bytes[row * row_bytes..(row + 1) * row_bytes];
                    let (codes, factor) = oracle::unpack32(dtype, original, 640, block).unwrap();
                    assert_eq!(encoded.d.to_bits(), factor.to_bits());
                    for quartet in 0..8 {
                        let decoded = encoded.packed(dtype, quartet).to_le_bytes();
                        for byte in 0..4 {
                            let column = block * 32 + quartet * 4 + byte;
                            let code = decoded[byte] as i8;
                            assert_eq!(code, codes[quartet * 4 + byte]);
                            assert_eq!(
                                (encoded.d * f32::from(code)).to_bits(),
                                expected[row * 640 + column].to_bits()
                            );
                        }
                    }
                    if dtype == TensorType::Q2_0 && block % 2 == 1 {
                        let previous = wire.prepare(dtype, row * row_bytes, block - 1);
                        assert_eq!(previous.d.to_bits(), encoded.d.to_bits());
                    }
                }
            }
        }
        assert_eq!(observed, HashSet::from(loads));
    }
}

// Test only the declared alternate packed-dot operator. The independent
// scalar original decoder and f64 oracle do not call this register decoder.
fn projected(dtype: TensorType, bytes: &[u8], words: &[u32]) -> f32 {
    let mut wire = Wire { bytes, loads: 0 };
    let mut lanes = [0f32; 8];
    for (lane, value) in lanes.iter_mut().enumerate() {
        for block in (lane..20).step_by(8) {
            let encoded = wire.prepare(dtype, 0, block);
            let mut first = 0i32;
            let mut second = 0i32;
            for quartet in 0..8 {
                let code = encoded.packed(dtype, quartet).to_le_bytes();
                let a = words[block * 8 + quartet].to_le_bytes();
                let b = words[160 + block * 8 + quartet].to_le_bytes();
                for i in 0..4 {
                    first += i32::from(code[i] as i8) * i32::from(a[i] as i8);
                    second += i32::from(code[i] as i8) * i32::from(b[i] as i8);
                }
            }
            assert!(first.abs() <= 32 * 127 * 127 && second.abs() <= 32 * 127 * 127);
            let pair = f32::from_bits(words[320 + block]) * first as f32
                + f32::from_bits(words[340 + block]) * second as f32;
            *value += encoded.d * pair;
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
fn frozen_down_integer_eight_rows_all30_real_routes_match_separate_quantized_oracle() {
    for dtype in [TensorType::Iq4Nl, TensorType::Q2_0] {
        let row_bytes = dtype.row_bytes(640).unwrap() as usize;
        let bytes = iq_fixtures::packed(dtype.gguf_id(), 8, 640, 1);
        for count in 1..=3 {
            for route in 0..count * 10 {
                let input: Vec<f32> = (0..640)
                    .map(|i| ((i * 31 + route * 17) % 139) as f32 * 0.012 - 0.83)
                    .collect();
                let planes = oracle::Planes::new(&input).unwrap();
                let words = planes.packed_words().unwrap();
                assert_eq!(words.len(), FRAME_WORDS);
                assert!(planes.relative_l2_error(&input).unwrap() < 1e-4);
                for row in 0..8 {
                    let row = &bytes[row * row_bytes..(row + 1) * row_bytes];
                    let expected = oracle::project(dtype, row, &planes).unwrap();
                    let actual = projected(dtype, row, &words);
                    // Bound only ordinary F32 aggregation around this alternate
                    // oracle; this does not replace original CPU13 tolerances.
                    let mut magnitude = 0f64;
                    for block in 0..20 {
                        let (codes, d) = oracle::unpack32(dtype, row, 640, block).unwrap();
                        for (i, code) in codes.into_iter().enumerate() {
                            let at = block * 8 + i / 4;
                            let shift = i % 4 * 8;
                            let a = (words[at] >> shift) as u8 as i8;
                            let b = (words[160 + at] >> shift) as u8 as i8;
                            let x = f64::from(f32::from_bits(words[320 + block])) * f64::from(a)
                                + f64::from(f32::from_bits(words[340 + block])) * f64::from(b);
                            magnitude += (f64::from(d) * f64::from(code) * x).abs();
                        }
                    }
                    assert!(
                        f64::from((actual - expected).abs())
                            <= 32. * f64::from(f32::EPSILON) * magnitude + 1e-6
                    );
                }
            }
        }
    }
}

fn audited(words: &[u32], count: usize) -> bool {
    if words.len() != SCRATCH_WORDS || !(1..=3).contains(&count) {
        return false;
    }
    words.iter().enumerate().all(|(at, &word)| {
        if at < HEADER_WORDS {
            return word == u32::from(at < count * 10);
        }
        if at >= LOGICAL_WORDS {
            return word == 0;
        }
        let route = (at - HEADER_WORDS) / FRAME_WORDS;
        let offset = (at - HEADER_WORDS) % FRAME_WORDS;
        if route >= count * 10 {
            return word == 0;
        }
        offset < 320 || {
            let scale = f32::from_bits(word);
            scale.is_finite() && scale >= 0.
        }
    })
}

#[test]
fn frozen_down_integer_full64k_reset_is_disjoint_and_audit_observes_every_word() {
    let mut writes = vec![0u8; SCRATCH_WORDS];
    for route in 0..MAX_ROUTES {
        for local in 0..64 {
            if local == 0 {
                writes[route] += 1;
            }
            for at in (HEADER_WORDS + route * FRAME_WORDS + local
                ..HEADER_WORDS + (route + 1) * FRAME_WORDS)
                .step_by(64)
            {
                writes[at] += 1;
            }
            if route == 0 {
                if local >= 30 {
                    writes[local] += 1;
                }
                for at in (LOGICAL_WORDS + local..SCRATCH_WORDS).step_by(64) {
                    writes[at] += 1;
                }
            }
        }
    }
    assert!(writes.iter().all(|&w| w == 1));
    for count in 1..=3 {
        let mut words = vec![0; SCRATCH_WORDS];
        for route in 0..count * 10 {
            let input: Vec<f32> = (0..640)
                .map(|i| ((i + route * 7) % 71) as f32 * 0.04)
                .collect();
            let planes = oracle::Planes::new(&input).unwrap().packed_words().unwrap();
            words[route] = 1;
            let first = HEADER_WORDS + route * FRAME_WORDS;
            words[first..first + FRAME_WORDS].copy_from_slice(&planes);
        }
        assert!(audited(&words, count));
        // Every header flag, inactive/tail word and every active scale matters.
        for at in 0..SCRATCH_WORDS {
            let offset = at.saturating_sub(HEADER_WORDS) % FRAME_WORDS;
            if at >= HEADER_WORDS && at < HEADER_WORDS + count * 10 * FRAME_WORDS && offset < 320 {
                continue;
            }
            let saved = words[at];
            words[at] = if at < HEADER_WORDS || at >= HEADER_WORDS + count * 10 * FRAME_WORDS {
                saved ^ 1
            } else {
                f32::NAN.to_bits()
            };
            assert!(!audited(&words, count), "unaudited owner word {at}");
            words[at] = saved;
        }
        for bad in [
            (-1f32).to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            f32::NAN.to_bits(),
        ] {
            let saved = words[HEADER_WORDS + 320];
            words[HEADER_WORDS + 320] = bad;
            assert!(!audited(&words, count));
            words[HEADER_WORDS + 320] = saved;
        }
    }
}

fn group_valid(
    map: &[u32],
    first: usize,
    slots: usize,
    envelope: usize,
    count: usize,
    group: usize,
    ready: &[u32],
) -> bool {
    if !(1..=3).contains(&count)
        || !(1..=30).contains(&envelope)
        || group >= envelope
        || map.len() != g::MAP_WORDS
        || map[0] != count as u32
        || map[1] != 0
        || map[2] != count as u32
    {
        return false;
    }
    let groups = map[3] as usize;
    if groups == 0 || groups > count * 10 || group >= groups {
        return false;
    }
    let slot = map[4 + group] as usize;
    let length = map[84 + group] as usize;
    if slot < first || slot - first >= slots || length == 0 || length > count {
        return false;
    }
    let routes = &map[164 + group * 8..164 + group * 8 + length];
    routes.iter().enumerate().all(|(i, &route)| {
        route < (count * 10) as u32
            && ready[route as usize] == 1
            && !routes[..i].iter().any(|&prior| prior / 10 == route / 10)
    })
}

#[test]
fn frozen_down_integer_original_map_full_wave_routes_and_partitions_remain_exact() {
    let geometry = geometry(super::super::IQ2_S, super::super::IQ4_NL);
    let partitions = [(0, 72), (72, 72), (144, 144), (288, 144), (432, 80)];
    for count in 1..=3 {
        for sharing in [true, false] {
            let experts: Vec<u32> = (0..count * 10)
                .map(|route| {
                    ((route % 10 * 53 + if sharing { 0 } else { route / 10 * 17 }) % 512) as u32
                })
                .collect();
            let probabilities = vec![0.1; count * 10];
            let mut protected = Vec::new();
            for &e in &experts {
                if !protected.iter().any(|&(prior, _)| prior == e) {
                    protected.push((e, e));
                }
            }
            let map = g::GroupMap::new_frozen_sources(
                count,
                &experts,
                &probabilities,
                &protected,
                geometry,
            )
            .unwrap();
            let mut readiness = [0; 30];
            readiness[..count * 10].fill(1);
            let mut coverage = [0; 30];
            for (first, slots) in partitions {
                for group in 0..30 {
                    if !group_valid(map.words(), first, slots, 30, count, group, &readiness) {
                        continue;
                    }
                    let length = map.words()[84 + group] as usize;
                    for &route in &map.words()[164 + group * 8..164 + group * 8 + length] {
                        coverage[route as usize] += 1;
                    }
                }
            }
            assert!(coverage[..count * 10].iter().all(|&n| n == 1));
            assert!(coverage[count * 10..].iter().all(|&n| n == 0));
            let slot = map.words()[4] as usize;
            let (first, slots) = *partitions
                .iter()
                .find(|&&(first, slots)| slot >= first && slot < first + slots)
                .unwrap();
            assert!(group_valid(
                map.words(),
                first,
                slots,
                30,
                count,
                0,
                &readiness
            ));
            for (at, bad) in [(0, 8), (1, 1), (2, 0), (3, 0), (84, 4), (164, 30)] {
                let mut badmap = *map.words();
                badmap[at] = bad;
                assert!(!group_valid(
                    &badmap, first, slots, 30, count, 0, &readiness
                ));
            }
            let route = map.words()[164] as usize;
            readiness[route] = 0;
            assert!(!group_valid(
                map.words(),
                first,
                slots,
                30,
                count,
                0,
                &readiness
            ));
        }
    }
    for dtype in [super::super::IQ4_NL, super::super::Q2_0] {
        let source = Kernel::Project {
            dtype,
            row_bytes: TensorType::from_gguf_id(dtype)
                .unwrap()
                .row_bytes(640)
                .unwrap() as usize,
        }
        .source();
        for guard in [
            "route>=p.h*10u",
            "slot<p.b || slot-p.b>=p.c",
            "planes[route]!=1u || status[p.f+route]!=1u",
            "map[p.a+1u]!=0u || map[p.a+2u]!=p.h",
            "route/10u==map[p.a+164u+group*8u+k]/10u",
            "arrayLength(&down)!=p.c*2560u*(ROW_BYTES/4u)",
        ] {
            assert!(source.contains(guard));
        }
    }
    //30 maximum route frames total, rather than30 groups times3 members.
    assert_eq!(HEADER_WORDS + MAX_ROUTES * FRAME_WORDS, LOGICAL_WORDS);
    assert_eq!(5 * 320 * 30, 48000);
}
