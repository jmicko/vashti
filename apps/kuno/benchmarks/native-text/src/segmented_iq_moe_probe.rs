//! Fixed PUBLIC mixed-IQ grouped-MoE segment acceptance, never a model route.
//! Original C-oracle packed IQ rows and compiled Q8 rows fill full 72+72 pages.
//! Every segment must preserve inactive ranks; complete owning spans, public
//! padding, original-rank merge, finite flags and transient-I/O audits are gates.
//! Main registration must call validate_configuration BEFORE creating a device.
use crate::{
    bank_slots::{BankSlots, StagedSlots},
    grouped_moe_chunk as c,
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::{self, grouped_moe as k, iq_fixtures},
    token_waves,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::TensorType;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type SeedPayload = (Zeroizing<Vec<u8>>, Zeroizing<Vec<f32>>);
const HIDDEN: usize = 2560;
const FF: usize = 640;
const CAP: usize = 144;
const SEGMENT: usize = 72;
const SEED_ROWS: usize = 4;
const PADDING: u32 = 0x5aa5_a55a;
const GUARD: u32 = 0x3f34_0000;
const EXPERTS: [usize; 10] = [143, 0, 72, 71, 1, 70, 73, 142, 35, 108];
const PROB: [f32; 10] = [0.05, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.1, 0.08];
const PROFILES: [(TensorType, TensorType); 3] = [
    (TensorType::Iq3Xxs, TensorType::Iq4Nl),
    (TensorType::Iq3Xxs, TensorType::Q8_0),
    (TensorType::Iq4Xs, TensorType::Q8_0),
];

pub fn validate_configuration() -> std::result::Result<(), &'static str> {
    if cfg!(any(
        feature = "direct-device",
        feature = "cpu-fma",
        feature = "cpu-buffered-weights",
        feature = "cpu-work-stealing",
        feature = "cpu-q8-two-pass"
    )) {
        return Err("segmented IQ probe requires default features");
    }
    if std::env::var_os("KUNO_LAB_SEGMENTED_IQ_PUBLIC_ONLY").as_deref()
        != Some(std::ffi::OsStr::new("1"))
        || std::env::var_os("KUNO_PUBLIC_RECEIPT").is_none_or(|v| v.is_empty())
    {
        return Err("segmented IQ fixed public gate and new receipt required");
    }
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if (name.starts_with("KUNO_LAB_")
            || name.starts_with("KUNO_TEST_")
            || name.starts_with("KUNO_PUBLIC_"))
            && !matches!(
                key.to_str(),
                Some("KUNO_LAB_SEGMENTED_IQ_PUBLIC_ONLY" | "KUNO_PUBLIC_RECEIPT")
            )
        {
            return Err("segmented IQ inherited setting rejected");
        }
    }
    Ok(())
}

fn geometry(gate: TensorType, down: TensorType) -> Result<k::Geometry> {
    if !PROFILES.contains(&(gate, down)) {
        return Err("segmented IQ profile rejected".into());
    }
    Ok(k::Geometry::new(
        [
            k::Projection::new(gate.gguf_id(), HIDDEN, FF, 512)?,
            k::Projection::new(gate.gguf_id(), HIDDEN, FF, 512)?,
            k::Projection::new(down.gguf_id(), FF, HIDDEN, 512)?,
        ],
        1,
        CAP,
    )?)
}
fn spans(g: k::Geometry) -> Result<(k::Spans, [usize; 8])> {
    let s = k::Spans::new([5, 7, 2563, 11, 645, 13, 2567, 9, 17, 2565, 3]);
    let words = [
        5 + k::MAP_WORDS + 7,
        17 + 7 * 2565 + HIDDEN + 23,
        11 + 79 * 645 + FF + 13,
        13 + 79 * 2567 + HIDDEN + 17,
        9 + k::MAX_ROUTES + 19,
        17 + 7 * 2565 + HIDDEN + 23,
        3 + g.status_words() + 29,
        1,
    ];
    s.validate(g, words[..7].try_into().unwrap())?;
    if words.iter().try_fold(0usize, |a, &w| {
        Ok::<_, Box<dyn std::error::Error>>(a + BufferLayout::new(w)?.owned_bytes)
    })? > c::PRIVATE_CAP
    {
        return Err("segmented IQ private page cap".into());
    }
    Ok((s, words))
}
fn limits(g: k::Geometry, l: &wgpu::Limits) -> Result<()> {
    if l.max_compute_invocations_per_workgroup < 256
        || l.max_compute_workgroup_size_x < 256
        || l.max_compute_workgroup_storage_size < 512
        || l.max_storage_buffers_per_shader_stage < 7
        || l.max_bindings_per_bind_group < 7
        || l.max_bind_groups < 1
        || l.max_uniform_buffer_binding_size < 512
        || l.max_compute_workgroups_per_dimension < 32768
        || l.max_storage_buffer_binding_size < 128 * 1024 * 1024
        || l.max_buffer_size < 128 * 1024 * 1024
    {
        return Err("segmented IQ actual device limits".into());
    }
    for first in [0, 72] {
        g.validate_shard(
            first,
            72,
            g.projections().map(|v| v.segment_words(72).unwrap()),
        )?;
    }
    Ok(())
}
fn bank_pages(g: k::Geometry) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(6);
    for _ in 0..2 {
        for p in g.projections() {
            out.push(main_admission::page_bytes(
                (p.segment_words(72)? * 4) as u64,
            )?);
        }
    }
    if out.iter().sum::<u64>() > 512 * 1024 * 1024 {
        return Err("segmented IQ bank page cap".into());
    }
    Ok(out)
}
fn grid(words: usize) -> CubeCount {
    let groups = words.div_ceil(256);
    CubeCount::Static(groups.min(32768) as u32, groups.div_ceil(32768) as u32, 1)
}

// The /8 term makes global slot72 different from local slot0, including IQ4
// rows whose independent public records repeat every two seed rows.
#[cube]
fn seed_index(slot: usize, row: usize, phase: usize) -> usize {
    (slot * 3 + slot / 8 + row * 5 + phase) % SEED_ROWS
}
#[cube]
fn expected_word(
    seed: &Array<u32>,
    index: usize,
    logical: usize,
    row_words: usize,
    rows: usize,
    first_slot: usize,
    phase: usize,
) -> u32 {
    let mut value = 0x5aa5_a55a_u32;
    if index < logical {
        let slot = index / (row_words * rows);
        let row = (index / row_words) % rows;
        value = seed[seed_index(first_slot + slot, row, phase) * row_words + index % row_words];
    }
    value
}
#[cube(launch)]
fn initialize_bank(
    seed: &Array<u32>,
    bank: &mut Array<u32>,
    logical: usize,
    row_words: usize,
    rows: usize,
    first_slot: usize,
    phase: usize,
) {
    let index = ABSOLUTE_POS;
    if index < bank.len() {
        bank[index] = expected_word(seed, index, logical, row_words, rows, first_slot, phase);
    }
}
#[cube(launch)]
#[allow(clippy::collapsible_if)] // Keep the source read dominated by its bound.
fn verify_bank(
    seed: &Array<u32>,
    bank: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    logical: usize,
    row_words: usize,
    rows: usize,
    first_slot: usize,
    phase: usize,
) {
    let index = ABSOLUTE_POS;
    if index < bank.len() {
        if bank[index] != expected_word(seed, index, logical, row_words, rows, first_slot, phase) {
            fault[0].fetch_or(1);
        }
    }
}
struct Seed {
    handle: Handle,
    wire: Zeroizing<Vec<u8>>,
    decoded: Zeroizing<Vec<f32>>,
    row_bytes: usize,
}
struct Bank {
    root: Handle,
    logical: usize,
    owned: usize,
    component: usize,
    first: usize,
    rows: usize,
}
impl Bank {
    fn view(&self) -> Handle {
        self.root
            .clone()
            .offset_end((self.owned - self.logical) as u64)
    }
}
fn seed_payload(p: k::Projection) -> Result<SeedPayload> {
    let mut wire = Zeroizing::new(Vec::new());
    let mut decoded = Zeroizing::new(Vec::new());
    for phase in 0..SEED_ROWS {
        if p.dtype() == quant_matvec::Q8_0 {
            let mut row = vec![0u8; p.row_bytes()];
            let mut values = vec![0f32; p.columns()];
            for (block, bytes) in row.chunks_exact_mut(34).enumerate() {
                bytes[..2].copy_from_slice(&0x2800u16.to_le_bytes());
                for lane in 0..32 {
                    let quant = (((block * 32 + lane) * 17 + phase * 13) % 31) as i8 - 15;
                    bytes[2 + lane] = quant as u8;
                    values[block * 32 + lane] = f32::from(quant) / 32.;
                }
            }
            wire.extend_from_slice(&row);
            decoded.extend_from_slice(&values);
        } else {
            wire.extend(iq_fixtures::packed(p.dtype(), 1, p.columns(), phase));
            decoded.extend(iq_fixtures::oracle(p.dtype(), 1, p.columns(), phase));
        }
    }
    let owned = main_admission::page_bytes(wire.len() as u64)? as usize;
    while wire.len() < owned {
        wire.extend_from_slice(&PADDING.to_le_bytes());
    }
    if wire.len() != owned
        || decoded.len() != SEED_ROWS * p.columns()
        || decoded.iter().any(|v| !v.is_finite())
    {
        return Err("segmented IQ compiled row extent".into());
    }
    Ok((wire, decoded))
}
fn physical(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    ids: &[BufferId],
    seeds: &[Seed; 3],
    banks: &[Bank],
    g: k::Geometry,
) -> Result<()> {
    let stream = arena.handle(ids[0])?.stream;
    let mut pages = Vec::new();
    for &id in ids {
        let h = arena.handle(id)?;
        if h.stream != stream {
            return Err("segmented IQ private stream".into());
        }
        let r = client
            .get_resource(h.clone())
            .map_err(|_| "segmented IQ private resource")?;
        let r = r.resource();
        let n = arena.layout(id)?.owned_bytes as u64;
        pages.push((
            r.buffer.clone(),
            r.offset,
            r.size,
            r.buffer.size(),
            n,
            n,
            true,
        ));
    }
    for (h, n) in seeds
        .iter()
        .map(|v| (&v.handle, v.wire.len()))
        .chain(banks.iter().map(|b| (&b.root, b.owned)))
    {
        if h.stream != stream {
            return Err("segmented IQ public stream".into());
        }
        let r = client
            .get_resource(h.clone())
            .map_err(|_| "segmented IQ public resource")?;
        let r = r.resource();
        pages.push((
            r.buffer.clone(),
            r.offset,
            r.size,
            r.buffer.size(),
            n as u64,
            n as u64,
            false,
        ));
    }
    c::validate_physical_pages(&pages)?;
    for segment in 0..2 {
        let words = [
            banks[segment * 3].logical / 4,
            banks[segment * 3 + 1].logical / 4,
            banks[segment * 3 + 2].logical / 4,
        ];
        g.validate_shard(segment * 72, 72, words)?;
        for b in &banks[segment * 3..segment * 3 + 3] {
            let full = client
                .get_resource(b.root.clone())
                .map_err(|_| "segmented IQ root lookup")?;
            let view = client
                .get_resource(b.view())
                .map_err(|_| "segmented IQ view lookup")?;
            if view.resource().buffer != full.resource().buffer
                || view.resource().offset != 0
                || view.resource().size != b.logical as u64
            {
                return Err("segmented IQ trimmed root mismatch".into());
            }
        }
    }
    Ok(())
}
fn private_arg(arena: &PrivateArena, id: BufferId) -> Result<ArrayArg<WgpuRuntime>> {
    // Full owner and physical disjointness are checked immediately by caller.
    Ok(unsafe {
        ArrayArg::from_raw_parts(arena.handle(id)?.clone(), arena.layout(id)?.owned_words)
    })
}
fn verify_public(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    fault: BufferId,
    seeds: &[Seed; 3],
    banks: &[Bank],
    retired: &AtomicBool,
) -> Result<()> {
    for seed in seeds {
        let mut bytes = client
            .read_one(seed.handle.clone())
            .map_err(|_| "segmented IQ public seed read")?;
        let valid = *bytes == *seed.wire;
        bytes.zeroize();
        drop(bytes);
        if !valid {
            return Err("segmented IQ public source or padding changed".into());
        }
    }
    for b in banks {
        let seed = &seeds[b.component];
        verify_bank::launch::<WgpuRuntime>(
            client,
            grid(b.owned / 4),
            CubeDim::new_1d(256),
            unsafe { ArrayArg::from_raw_parts(seed.handle.clone(), seed.wire.len() / 4) },
            unsafe { ArrayArg::from_raw_parts(b.root.clone(), b.owned / 4) },
            private_arg(arena, fault)?,
            b.logical / 4,
            seed.row_bytes / 4,
            b.rows,
            b.first,
            b.component,
        );
    }
    let result = c::read(arena, fault, retired)?;
    c::audit(client, retired)?;
    if result.iter().any(|&v| v != 0) {
        return Err("segmented IQ public bank or full padding changed".into());
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn launch_segment(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    ids: &[BufferId; 8],
    banks: &[Bank],
    g: k::Geometry,
    s: &k::Spans,
    groups: usize,
    segment: usize,
) -> Result<()> {
    let [ga, up, down] = g.projections();
    let bs = &banks[segment * 3..segment * 3 + 3];
    let b = |i: usize| unsafe { ArrayArg::from_raw_parts(bs[i].view(), bs[i].logical / 4) };
    unsafe {
        k::grouped_gate_up::launch_unchecked::<WgpuRuntime>(
            client,
            g.grid(groups, false)?,
            CubeDim::new_1d(64),
            b(0),
            b(1),
            private_arg(arena, ids[c::INPUT])?,
            private_arg(arena, ids[c::MAP])?,
            private_arg(arena, ids[c::ACT])?,
            private_arg(arena, ids[c::STATUS])?,
            s.map_first(),
            s.input_first(),
            s.input_stride(),
            s.act_first(),
            s.act_stride(),
            s.status_first(),
            segment * 72,
            72,
            HIDDEN,
            FF,
            ga.dtype(),
            up.dtype(),
            ga.row_bytes(),
            up.row_bytes(),
            1,
            false,
        );
        k::grouped_down::launch_unchecked::<WgpuRuntime>(
            client,
            g.grid(groups, true)?,
            CubeDim::new_1d(64),
            b(2),
            private_arg(arena, ids[c::ACT])?,
            private_arg(arena, ids[c::MAP])?,
            private_arg(arena, ids[c::PART])?,
            private_arg(arena, ids[c::STATUS])?,
            s.map_first(),
            s.act_first(),
            s.act_stride(),
            s.partial_first(),
            s.partial_stride(),
            s.status_first(),
            segment * 72,
            72,
            HIDDEN,
            FF,
            down.dtype(),
            down.row_bytes(),
            1,
        );
    }
    Ok(())
}
fn finish_math(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    ids: &[BufferId; 8],
    g: k::Geometry,
    s: &k::Spans,
) -> Result<()> {
    unsafe {
        k::grouped_reduce::launch_unchecked::<WgpuRuntime>(
            client,
            g.reduce_grid(),
            CubeDim::new_1d(64),
            private_arg(arena, ids[c::PART])?,
            private_arg(arena, ids[c::PROB])?,
            private_arg(arena, ids[c::MAP])?,
            private_arg(arena, ids[c::STATUS])?,
            private_arg(arena, ids[c::OUTPUT])?,
            s.map_first(),
            s.partial_first(),
            s.partial_stride(),
            s.prob_first(),
            s.status_first(),
            s.output_first(),
            s.output_stride(),
            HIDDEN,
        );
        k::grouped_finite::launch_unchecked::<WgpuRuntime>(
            client,
            g.finite_grid(),
            CubeDim::new_1d(64),
            private_arg(arena, ids[c::INPUT])?,
            private_arg(arena, ids[c::ACT])?,
            private_arg(arena, ids[c::PART])?,
            private_arg(arena, ids[c::OUTPUT])?,
            private_arg(arena, ids[c::MAP])?,
            private_arg(arena, ids[c::STATUS])?,
            s.map_first(),
            s.input_first(),
            s.input_stride(),
            s.act_first(),
            s.act_stride(),
            s.partial_first(),
            s.partial_stride(),
            s.output_first(),
            s.output_stride(),
            s.status_first(),
            HIDDEN,
            FF,
        );
    }
    Ok(())
}
fn tree(weights: &[f32], input: &[f32]) -> f32 {
    let mut lanes = Zeroizing::new([0f32; 64]);
    for (col, (&w, &x)) in weights.iter().zip(input).enumerate() {
        lanes[col % 64] += w * x;
    }
    for stride in [32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
    }
    lanes[0]
}
struct Oracle {
    act: Zeroizing<Vec<f32>>,
    part: Zeroizing<Vec<f32>>,
    output: Zeroizing<Vec<f32>>,
}
fn oracle(decoded: [&[f32]; 3], x: &[f32]) -> Result<Oracle> {
    let mut out = Oracle {
        act: Zeroizing::new(vec![0.; 10 * FF]),
        part: Zeroizing::new(vec![0.; 10 * HIDDEN]),
        output: Zeroizing::new(vec![0.; HIDDEN]),
    };
    for (rank, &slot) in EXPERTS.iter().enumerate() {
        for row in 0..FF {
            let g = seed_index(slot, row, 0) * HIDDEN;
            let u = seed_index(slot, row, 1) * HIDDEN;
            out.act[rank * FF + row] = kuno_text::ops::silu(tree(&decoded[0][g..g + HIDDEN], x))
                * tree(&decoded[1][u..u + HIDDEN], x);
        }
        for row in 0..HIDDEN {
            let d = seed_index(slot, row, 2) * FF;
            out.part[rank * HIDDEN + row] =
                tree(&decoded[2][d..d + FF], &out.act[rank * FF..(rank + 1) * FF]);
        }
    }
    for row in 0..HIDDEN {
        for (rank, &p) in PROB.iter().enumerate() {
            out.output[row] += p * out.part[rank * HIDDEN + row];
        }
    }
    if [&out.act, &out.part, &out.output]
        .iter()
        .any(|v| v.iter().any(|x| !x.is_finite()) || !v.iter().any(|x| *x != 0.))
    {
        return Err("segmented IQ public oracle finite/nonzero gate".into());
    }
    Ok(out)
}
fn compare(actual: f32, expected: f32) -> Result<f32> {
    let error = (actual - expected).abs();
    if !actual.is_finite() || !expected.is_finite() || error > 1e-4 + 3e-4 * expected.abs() {
        return Err("segmented IQ complete projection mismatch".into());
    }
    Ok(error)
}
fn snapshot(
    arena: &PrivateArena,
    ids: &[BufferId; 8],
    retired: &AtomicBool,
) -> Result<Vec<Zeroizing<Vec<u32>>>> {
    let values = ids
        .iter()
        .map(|&id| c::read(arena, id, retired))
        .collect::<Result<Vec<_>>>()?;
    c::audit(arena.client()?, retired)?;
    Ok(values)
}
fn check(
    values: &[Zeroizing<Vec<u32>>],
    before: &[Zeroizing<Vec<u32>>],
    g: k::Geometry,
    s: &k::Spans,
    expected: &Oracle,
    second: bool,
    finished: bool,
) -> Result<[f32; 3]> {
    let mut max = [0f32; 3];
    for (i, first, stride, width, count, want, which) in [
        (
            c::ACT,
            s.act_first(),
            s.act_stride(),
            FF,
            10,
            &expected.act,
            0,
        ),
        (
            c::PART,
            s.partial_first(),
            s.partial_stride(),
            HIDDEN,
            10,
            &expected.part,
            1,
        ),
        (
            c::OUTPUT,
            s.output_first(),
            s.output_stride(),
            HIDDEN,
            usize::from(finished),
            &expected.output,
            2,
        ),
    ] {
        for index in 0..values[i].len() {
            let relative = index.saturating_sub(first);
            let rank = relative / stride;
            let col = relative % stride;
            let active = index >= first
                && rank < count
                && col < width
                && (i == c::OUTPUT || second || EXPERTS[rank] < 72);
            if active {
                max[which] = max[which].max(compare(
                    f32::from_bits(values[i][index]),
                    want[rank * width + col],
                )?);
            } else if values[i][index] != before[i][index] {
                return Err("segmented IQ inactive rank/gap/full owning tail changed".into());
            }
        }
    }
    for i in [c::MAP, c::INPUT, c::PROB, 7] {
        if values[i] != before[i] {
            return Err("segmented IQ readonly/fault owner changed".into());
        }
    }
    if finished {
        k::validate_status(g, 1, s.status_first(), &values[c::STATUS])?;
    } else {
        let mut status = before[c::STATUS].clone();
        for (rank, &slot) in EXPERTS.iter().enumerate() {
            if second || slot < 72 {
                status[s.status_first() + rank] = 1;
                status[s.status_first() + k::MAX_ROUTES + rank] = 1;
            }
        }
        if values[c::STATUS] != status {
            return Err("segmented IQ inactive segment status changed".into());
        }
    }
    Ok(max)
}
fn inactive_preserved(
    actual: &[Zeroizing<Vec<u32>>],
    previous: &[Zeroizing<Vec<u32>>],
    s: &k::Spans,
) -> Result<()> {
    for (i, first, stride, width) in [
        (c::ACT, s.act_first(), s.act_stride(), FF),
        (c::PART, s.partial_first(), s.partial_stride(), HIDDEN),
    ] {
        for (rank, &slot) in EXPERTS.iter().enumerate() {
            if slot < SEGMENT {
                let start = first + rank * stride;
                if actual[i][start..start + width] != previous[i][start..start + width] {
                    return Err("segmented IQ inactive completed rank bits changed".into());
                }
            }
        }
    }
    Ok(())
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    l: &wgpu::Limits,
    arena: &mut PrivateArena,
    retired: &AtomicBool,
    profile: (TensorType, TensorType),
) -> Result<serde_json::Value> {
    let g = geometry(profile.0, profile.1)?;
    limits(g, l)?;
    let (s, words) = spans(g)?;
    let pages = bank_pages(g)?;
    let payloads = g
        .projections()
        .map(seed_payload)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut ledger = MainAdmission::new(c::MAIN_CAP, c::sample(client)?)?;
    let private_pages = words
        .map(|n| BufferLayout::new(n).map(|v| v.owned_bytes as u64))
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    // Before/previous/current owners plus one raw readback lease, row seeds,
    // complete CPU oracle and bounded route metadata. No full bank host copy.
    let host_upper = 4 * private_pages.iter().sum::<u64>()
        + payloads
            .iter()
            .map(|(b, v)| 3 * b.len() as u64 + 4 * v.len() as u64)
            .sum::<u64>()
        + 1024 * 1024;
    if host_upper > c::HOST_CAP as u64 {
        return Err("segmented IQ host fixture bound".into());
    }
    let mut requests = pages.clone();
    requests.extend(payloads.iter().map(|(b, _)| b.len() as u64));
    requests.extend(&private_pages);
    main_admission::check_capacity(c::MAIN_CAP, &[ledger.upper_bytes(), requests.iter().sum()])?;
    ledger.precharge(&requests)?;
    let mut seeds = Vec::new();
    for (i, (wire, decoded)) in payloads.into_iter().enumerate() {
        let handle = client.create_public_owned(wire.to_vec());
        seeds.push(Seed {
            handle,
            wire,
            decoded,
            row_bytes: g.projections()[i].row_bytes(),
        });
    }
    let seeds: [Seed; 3] = seeds.try_into().map_err(|_| "segmented IQ seed count")?;
    let ids: [BufferId; 8] = words
        .into_iter()
        .map(|n| arena.allocate_zeroed(n))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "segmented IQ owner count")?;
    let mut banks = Vec::new();
    for segment in 0..2 {
        for (component, p) in g.projections().into_iter().enumerate() {
            let logical = p.segment_words(SEGMENT)? * 4;
            let owned = pages[segment * 3 + component] as usize;
            banks.push(Bank {
                root: client.empty(owned),
                logical,
                owned,
                component,
                first: segment * SEGMENT,
                rows: p.rows(),
            });
        }
    }
    physical(client, arena, &ids, &seeds, &banks, g)?;
    for b in &banks {
        initialize_bank::launch::<WgpuRuntime>(
            client,
            grid(b.owned / 4),
            CubeDim::new_1d(256),
            unsafe {
                ArrayArg::from_raw_parts(
                    seeds[b.component].handle.clone(),
                    seeds[b.component].wire.len() / 4,
                )
            },
            unsafe { ArrayArg::from_raw_parts(b.root.clone(), b.owned / 4) },
            b.logical / 4,
            seeds[b.component].row_bytes / 4,
            b.rows,
            b.first,
            b.component,
        );
    }
    verify_public(client, arena, ids[7], &seeds, &banks, retired)?;
    ledger.completed_fence(c::sample(client)?)?;
    // All synthetic bank rows were physically initialized and checked before
    // committing their bounded source mapping. No model-history is published.
    let mut bank_slots = BankSlots::new(CAP, 512)?;
    for first in (0..CAP).step_by(10) {
        let selected: Vec<_> = (first..(first + 10).min(CAP)).collect();
        let plan = bank_slots.plan(&selected)?;
        bank_slots.commit(plan)?;
    }
    let mut routes =
        token_waves::Input::new(1, CAP, 512).map_err(|_| "segmented IQ fixed route geometry")?;
    for rank in 0..10 {
        routes
            .set(0, rank, EXPERTS[rank] as u32, PROB[rank])
            .map_err(|_| "segmented IQ fixed route rank")?;
    }
    let wave_plan = token_waves::Plan::build(&routes, &AtomicBool::new(false))
        .map_err(|_| "segmented IQ fixed wave plan")?;
    let packet = c::Packet::from_plan(&wave_plan)?;
    let wave = wave_plan.wave(0).ok_or("segmented IQ fixed wave")?;
    let selected: Vec<_> = wave
        .protected_experts()
        .iter()
        .map(|&v| v as usize)
        .collect();
    let mut staged = StagedSlots::begin(&bank_slots)?;
    let pending = staged.plan(&bank_slots, &selected)?;
    if !pending.uploads.is_empty() || pending.slots != selected {
        return Err("segmented IQ actual global slot mapping".into());
    }
    let pin = c::ProtectedWave::new(wave, &pending, &staged, &bank_slots)?;
    let map = pin.map(&packet)?;
    map.validate(&packet.experts[..10], &packet.probabilities[..10], CAP)?;
    map.validate_geometry(g)?;
    let mut before = Vec::new();
    for (i, n) in words.into_iter().enumerate() {
        let mut v = Zeroizing::new(vec![0u32; arena.layout(ids[i])?.owned_words]);
        if matches!(i, c::INPUT | c::ACT | c::PART | c::OUTPUT) {
            v[..n].fill(GUARD);
        }
        before.push(v);
    }
    let x = Zeroizing::new(
        (0..HIDDEN)
            .map(|i| ((i * 7 % 13) as f32 - 6.) / 1024.)
            .collect::<Vec<_>>(),
    );
    let expected = oracle(seeds.each_ref().map(|v| v.decoded.as_slice()), &x)?;
    for (i, &v) in x.iter().enumerate() {
        before[c::INPUT][s.input_first() + i] = v.to_bits();
    }
    before[c::MAP][s.map_first()..s.map_first() + k::MAP_WORDS].copy_from_slice(map.words());
    for (rank, &p) in map.probabilities().iter().enumerate() {
        before[c::PROB][s.prob_first() + rank] = p.to_bits();
    }
    for (i, &id) in ids.iter().enumerate() {
        c::write(arena, id, &before[i][..words[i]], retired)?;
    }
    let mut errors = [0f32; 3];
    let mut previous: Option<Vec<Zeroizing<Vec<u32>>>> = None;
    for segment in 0..2 {
        pin.validate()?;
        physical(client, arena, &ids, &seeds, &banks, g)?;
        launch_segment(client, arena, &ids, &banks, g, &s, map.groups(), segment)?;
        let got = snapshot(arena, &ids, retired)?;
        if let Some(prior) = &previous {
            inactive_preserved(&got, prior, &s)?;
        }
        let e = check(&got, &before, g, &s, &expected, segment == 1, false)?;
        for i in 0..3 {
            errors[i] = errors[i].max(e[i]);
        }
        ledger.completed_fence(c::sample(client)?)?;
        previous = Some(got);
    }
    finish_math(client, arena, &ids, g, &s)?;
    let got = snapshot(arena, &ids, retired)?;
    let e = check(&got, &before, g, &s, &expected, true, true)?;
    for i in 0..3 {
        errors[i] = errors[i].max(e[i]);
    }
    pin.validate()?;
    staged.validate_origin(&bank_slots)?;
    ledger.completed_fence(c::sample(client)?)?;
    verify_public(client, arena, ids[7], &seeds, &banks, retired)?;
    ledger.completed_fence(c::sample(client)?)?;
    arena.reset()?; // Audited complete PRIVATE owners; immutable sources retained.
    if ids.iter().any(|&id| arena.handle(id).is_ok()) {
        return Err("segmented IQ stale private owner accepted".into());
    }
    ledger.completed_fence(c::sample(client)?)?;
    ledger.precharge(&[BufferLayout::new(1)?.owned_bytes as u64])?;
    let verify = arena.allocate_zeroed(1)?;
    physical(client, arena, &[verify], &seeds, &banks, g)?;
    verify_public(client, arena, verify, &seeds, &banks, retired)?;
    ledger.completed_fence(c::sample(client)?)?;
    arena.reset()?;
    Ok(
        serde_json::json!({"gate_up_dtype":profile.0.gguf_id(),"down_dtype":profile.1.gguf_id(),
        "hidden":HIDDEN,"ff":FF,"capacity":CAP,"segment_capacity":SEGMENT,"count":1,"groups":10,
        "bank_physical_bytes":pages.iter().sum::<u64>(),"private_owned_bytes":private_pages.iter().sum::<u64>(),
        "max_abs_activation_error":errors[0],"max_abs_partial_error":errors[1],"max_abs_rank_merge_error":errors[2],
        "intermediate_inactive_rank_status_and_gap_checks":true,"original_global_local_slot_mapping":true,
        "full_public_bank_and_source_padding_preserved_after_private_reset":true}),
    )
}
pub fn run(client: &ComputeClient<WgpuRuntime>, l: &wgpu::Limits) -> Result<()> {
    validate_configuration()?;
    for (gate, down) in PROFILES {
        limits(geometry(gate, down)?, l)?;
    }
    let path = std::env::var_os("KUNO_PUBLIC_RECEIPT").ok_or("segmented IQ receipt required")?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), c::PRIVATE_CAP, c::IO_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut reports = Vec::new();
        for profile in PROFILES {
            reports.push(one(client, l, &mut arena, &retired, profile)?);
            client.memory_cleanup();
        }
        Ok(reports)
    }));
    let closed = arena.close();
    let reports = match (attempt, closed, retired.load(Ordering::Acquire)) {
        (Ok(Ok(v)), Ok(()), false) => v,
        _ => return Err("segmented IQ public operator failed; no success receipt".into()),
    };
    let audit = arena
        .owned_zero_audit_receipt()
        .ok_or("segmented IQ full owner erasure absent")?;
    let report = serde_json::json!({"scope":"fixed-public-synthetic-original-packed-segmented-IQ-MoE-only","cases":reports,
        "main_limit_bytes":c::MAIN_CAP,"bank_pages_max_bytes":512u64*1024*1024,"private_limit_bytes":c::PRIVATE_CAP,
        "runtime_io_limit_bytes":c::IO_CAP,"main_erased_buffers":audit.buffers,"main_erased_bytes":audit.bytes,
        "whole_model_accepted":false,"private_request_accepted":false,"performance_accepted":false,"physical_recovery_accepted":false});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("public_segmented_iq_moe profiles=3 segments=2 ranks=10 accepted=1");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_mixed_profiles_original_strides_and_full_page_budget() {
        for (gate, down) in PROFILES {
            let g = geometry(gate, down).unwrap();
            let (_, words) = spans(g).unwrap();
            assert_eq!(g.gate_policy(), k::GatePolicy::Generic);
            assert!(g.with_q4_block().is_err());
            let pages = bank_pages(g).unwrap();
            assert_eq!(pages.len(), 6);
            assert_eq!(
                pages.iter().sum::<u64>(),
                if down == TensorType::Iq4Nl { 384 } else { 512 } * 1024 * 1024
            );
            assert_eq!(
                words
                    .iter()
                    .map(|&n| BufferLayout::new(n).unwrap().owned_bytes)
                    .sum::<usize>(),
                1_703_936
            );
            for first in [0, 72] {
                g.validate_shard(
                    first,
                    72,
                    g.projections().map(|p| p.segment_words(72).unwrap()),
                )
                .unwrap();
            }
        }
        assert!(geometry(TensorType::Iq4Xs, TensorType::Iq4Nl).is_err());
    }
    #[test]
    fn independent_rows_and_global_offset_boundary_are_not_aliases() {
        for (gate, down) in PROFILES {
            for p in geometry(gate, down).unwrap().projections() {
                let (wire, expected) = seed_payload(p).unwrap();
                let mut got = vec![0.; SEED_ROWS * p.columns()];
                TensorType::from_gguf_id(p.dtype())
                    .unwrap()
                    .decode(&wire[..SEED_ROWS * p.row_bytes()], &mut got)
                    .unwrap();
                assert_eq!(
                    got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                let a = seed_index(0, 0, 0);
                let b = seed_index(72, 0, 0);
                assert_ne!(
                    &wire[a * p.row_bytes()..(a + 1) * p.row_bytes()],
                    &wire[b * p.row_bytes()..(b + 1) * p.row_bytes()]
                );
                assert!(
                    wire[SEED_ROWS * p.row_bytes()..]
                        .chunks_exact(4)
                        .all(|v| v == PADDING.to_le_bytes())
                );
            }
        }
    }
    #[test]
    fn projection_comparison_rejects_finite_jump_and_nonfinite() {
        assert!(compare(1., 1.).is_ok());
        assert!(compare(2., 1.).is_err());
        assert!(compare(f32::NAN, 1.).is_err());
        assert!(compare(1., f32::INFINITY).is_err());
    }
    #[test]
    fn actual_shape_cpu_tree_oracles_are_finite_and_nonzero_for_all_profiles() {
        let x = Zeroizing::new(
            (0..HIDDEN)
                .map(|i| ((i * 7 % 13) as f32 - 6.) / 1024.)
                .collect::<Vec<_>>(),
        );
        for (gate, down) in PROFILES {
            let rows = geometry(gate, down)
                .unwrap()
                .projections()
                .map(|p| seed_payload(p).unwrap().1);
            let expected = oracle(rows.each_ref().map(|v| v.as_slice()), &x).unwrap();
            assert_eq!(expected.act.len(), 10 * FF);
            assert_eq!(expected.part.len(), 10 * HIDDEN);
            assert_eq!(expected.output.len(), HIDDEN);
        }
    }
    #[test]
    fn completed_first_segment_bits_must_survive_second_segment_exactly() {
        let (s, words) = spans(geometry(TensorType::Iq3Xxs, TensorType::Iq4Nl).unwrap()).unwrap();
        let before: Vec<_> = words
            .into_iter()
            .map(|n| Zeroizing::new(vec![GUARD; BufferLayout::new(n).unwrap().owned_words]))
            .collect();
        let mut after = before.clone();
        inactive_preserved(&after, &before, &s).unwrap();
        let rank = EXPERTS.iter().position(|&v| v == 71).unwrap();
        after[c::ACT][s.act_first() + rank * s.act_stride() + FF - 1] ^= 1;
        assert!(inactive_preserved(&after, &before, &s).is_err());
        after = before.clone();
        after[c::PART][s.partial_first() + rank * s.partial_stride() + HIDDEN - 1] ^= 1;
        assert!(inactive_preserved(&after, &before, &s).is_err());
    }
    #[test]
    fn initialization_and_full_padding_verifier_compile_to_valid_guarded_wgsl() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let atomic = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
        for verify in [false, true] {
            let mut b = KernelBuilder::default();
            b.scope.register_type::<u32>(uint);
            b.scope.register_type::<usize>(uint);
            b.scope
                .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
            let seed = b.input_array(Type::new(uint)).into();
            let logical = b.scalar(uint).into();
            let row_words = b.scalar(uint).into();
            let rows = b.scalar(uint).into();
            let first = b.scalar(uint).into();
            let phase = b.scalar(uint).into();
            if verify {
                let bank = b.input_array(Type::new(uint)).into();
                let fault = b.output_array(Type::new(atomic)).into();
                verify_bank::expand(
                    &mut b.scope,
                    seed,
                    bank,
                    fault,
                    logical,
                    row_words,
                    rows,
                    first,
                    phase,
                );
            } else {
                let bank = b.output_array(Type::new(uint)).into();
                initialize_bank::expand(
                    &mut b.scope,
                    seed,
                    bank,
                    logical,
                    row_words,
                    rows,
                    first,
                    phase,
                );
            }
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(256))
                            .kernel_name("public_segmented_IQ_bank"),
                    ),
                    &WgpuCompilationOptions::default(),
                    cubecl::server::ExecutionMode::Checked,
                    uint,
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
            assert!(source.contains(&format!("u32({PADDING})")));
            for scalar in 0..5 {
                assert!(source.contains(&format!("scalars_u32[{scalar}]")));
            }
            assert_eq!(source.contains("atomicOr"), verify);
        }
    }
}
