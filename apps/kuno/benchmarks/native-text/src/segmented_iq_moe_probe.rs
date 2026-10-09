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
    quant_matvec::{self, grouped_moe as k, iq_fixtures, iq_moe_integer, q3_subgroup_moe},
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

// Only compiled fixed-public integer cases produce these markers. Never log
// backend error text, weight IDs, vectors or request-derived data.
fn integer_stage(
    integer: bool,
    profile: (TensorType, TensorType),
    stage: &'static str,
    case: usize,
    segment: usize,
) {
    if integer {
        println!(
            "public_iq_integer_stage gate={} down={} stage={} case={} segment={}",
            profile.0.gguf_id(),
            profile.1.gguf_id(),
            stage,
            case,
            segment
        );
    }
}
fn fixed_failure_class(error: &(dyn std::error::Error + 'static)) -> &'static str {
    if let Some(error) = error.downcast_ref::<kuno_text::Error>() {
        return match error {
            kuno_text::Error::InvalidFormat(_) => "core-format",
            kuno_text::Error::LimitExceeded(_) => "core-budget",
            kuno_text::Error::InvalidBufferLength => "core-length",
            kuno_text::Error::OutOfBounds => "core-bounds",
            _ => "core-other",
        };
    }
    // The displayed error is compared ONLY with this closed source-controlled
    // list. Unknown/backend messages never leave this function.
    match error.to_string().as_str() {
        "segmented IQ complete projection mismatch" => "numeric-projection",
        "segmented IQ subgroup public fault accepted" => "fault-status-accepted",
        "segmented IQ subgroup fault changed inactive math owner" => "fault-owning-write",
        "segmented IQ fault producer readiness" => "fault-producer-ready",
        "segmented IQ fault finite flag invalid" => "fault-finite-flag",
        "segmented IQ inactive rank/gap/full owning tail changed" => "numeric-owning-gap",
        "segmented IQ inactive segment status changed" => "numeric-status-gap",
        "segmented IQ public oracle finite/nonzero gate" => "original-host-reference",
        "segmented IQ integer complete oracle finite/nonzero gate" => "integer-host-reference",
        "segmented IQ fixed route geometry" => "route-geometry",
        "segmented IQ host fixture bound" => "host-budget",
        "segmented IQ physical stream" => "physical-stream",
        _ => "unclassified",
    }
}

pub fn validate_configuration() -> std::result::Result<(), &'static str> {
    validate_mode(false)
}
pub fn validate_integer_configuration() -> std::result::Result<(), &'static str> {
    validate_mode(true)
}
fn validate_mode(integer: bool) -> std::result::Result<(), &'static str> {
    if cfg!(any(
        feature = "direct-device",
        feature = "cpu-fma",
        feature = "cpu-buffered-weights",
        feature = "cpu-work-stealing",
        feature = "cpu-q8-two-pass"
    )) {
        return Err("segmented IQ probe requires default features");
    }
    let gate = if integer {
        "KUNO_LAB_IQ_INTEGER_PUBLIC_ONLY"
    } else {
        "KUNO_LAB_SEGMENTED_IQ_PUBLIC_ONLY"
    };
    if std::env::var_os(gate).as_deref() != Some(std::ffi::OsStr::new("1"))
        || std::env::var_os("KUNO_PUBLIC_RECEIPT").is_none_or(|v| v.is_empty())
    {
        return Err("segmented IQ fixed public gate and new receipt required");
    }
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if (name.starts_with("KUNO_LAB_")
            || name.starts_with("KUNO_TEST_")
            || name.starts_with("KUNO_PUBLIC_"))
            && key.to_str() != Some(gate)
            && key.to_str() != Some("KUNO_PUBLIC_RECEIPT")
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
    subgroup: bool,
    integer: bool,
) -> Result<()> {
    let [ga, up, down] = g.projections();
    let bs = &banks[segment * 3..segment * 3 + 3];
    if subgroup || integer {
        let owner = |index: usize| arena.handle(ids[index]).cloned();
        let launch = if integer {
            iq_moe_integer::launch
        } else {
            q3_subgroup_moe::launch
        };
        launch(
            client,
            g,
            s,
            [bs[0].view(), bs[1].view(), bs[2].view()],
            [
                owner(0)?,
                owner(1)?,
                owner(2)?,
                owner(3)?,
                owner(4)?,
                owner(5)?,
                owner(6)?,
            ],
            segment * SEGMENT,
            SEGMENT,
            groups,
        )?;
        return Ok(());
    }
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
impl Oracle {
    fn owned_bytes(&self) -> usize {
        (self.act.capacity() + self.part.capacity() + self.output.capacity()) * 4
    }
}
fn compact_original(values: &[Zeroizing<Vec<u32>>], s: &k::Spans) -> Result<Oracle> {
    fn rows(
        values: &[u32],
        first: usize,
        stride: usize,
        width: usize,
        count: usize,
    ) -> Result<Zeroizing<Vec<f32>>> {
        let length = width
            .checked_mul(count)
            .ok_or("segmented IQ compact row overflow")?;
        let mut out = Zeroizing::new(Vec::with_capacity(length));
        for row in 0..count {
            let start = row
                .checked_mul(stride)
                .and_then(|n| first.checked_add(n))
                .ok_or("segmented IQ compact offset overflow")?;
            let end = start
                .checked_add(width)
                .ok_or("segmented IQ compact end overflow")?;
            let slice = values
                .get(start..end)
                .ok_or("segmented IQ original snapshot extent")?;
            out.extend(slice.iter().copied().map(f32::from_bits));
        }
        if out.iter().any(|v| !v.is_finite()) || out.capacity() > length {
            return Err("segmented IQ original snapshot finite/host bound".into());
        }
        Ok(out)
    }
    if values.len() != 8 {
        return Err("segmented IQ original snapshot owner count".into());
    }
    Ok(Oracle {
        act: rows(&values[c::ACT], s.act_first(), s.act_stride(), FF, 10)?,
        part: rows(
            &values[c::PART],
            s.partial_first(),
            s.partial_stride(),
            HIDDEN,
            10,
        )?,
        output: rows(
            &values[c::OUTPUT],
            s.output_first(),
            s.output_stride(),
            HIDDEN,
            1,
        )?,
    })
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
/// Separate scalar original-byte/integer/Q8-plane oracle. Four public seed
/// rows are evaluated once and selected by the independently checked global
/// slot/phase formula; this is not a production weight cache.
fn integer_oracle(g: k::Geometry, wire: [&[u8]; 3], x: &[f32]) -> Result<Oracle> {
    use iq_moe_integer::oracle::{Planes, project};
    let input = Planes::new(x)?;
    if input.owned_bytes() > 5760 || input.relative_l2_error(x)? >= 1e-4 {
        return Err("segmented IQ integer input host bound".into());
    }
    let projections = g.projections();
    let mut gu = Zeroizing::new([[0f32; SEED_ROWS]; 2]);
    for component in 0..2 {
        let p = projections[component];
        let kind = TensorType::from_gguf_id(p.dtype())?;
        for row in 0..SEED_ROWS {
            let bytes = wire[component]
                .get(row * p.row_bytes()..(row + 1) * p.row_bytes())
                .ok_or("segmented IQ integer seed extent")?;
            gu[component][row] = project(kind, bytes, &input)?;
        }
    }
    let mut out = Oracle {
        act: Zeroizing::new(vec![0.; 10 * FF]),
        part: Zeroizing::new(vec![0.; 10 * HIDDEN]),
        output: Zeroizing::new(vec![0.; HIDDEN]),
    };
    let p = projections[2];
    let down = TensorType::from_gguf_id(p.dtype())?;
    for (rank, &slot) in EXPERTS.iter().enumerate() {
        for row in 0..FF {
            out.act[rank * FF + row] = kuno_text::ops::silu(gu[0][seed_index(slot, row, 0)])
                * gu[1][seed_index(slot, row, 1)];
        }
        let act = Planes::new(&out.act[rank * FF..(rank + 1) * FF])?;
        if act.owned_bytes() > 1440 {
            return Err("segmented IQ integer activation host bound".into());
        }
        let mut results = Zeroizing::new([0f32; SEED_ROWS]);
        for row in 0..SEED_ROWS {
            let bytes = wire[2]
                .get(row * p.row_bytes()..(row + 1) * p.row_bytes())
                .ok_or("segmented IQ integer down seed extent")?;
            results[row] = project(down, bytes, &act)?;
        }
        for row in 0..HIDDEN {
            out.part[rank * HIDDEN + row] = results[seed_index(slot, row, 2)];
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
        return Err("segmented IQ integer complete oracle finite/nonzero gate".into());
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
/// Public alternative-arithmetic diagnostics, never an original F32 gate.
/// The denominator floor is explicit so all-zero references remain finite.
#[derive(Clone, Copy, Default)]
struct Difference {
    max_abs: f64,
    relative_l2: f64,
    rms: f64,
    relative_peak: f64,
    max_scaled_error: f64,
    original_pointwise_failures: usize,
}
impl Difference {
    fn new(actual: impl Iterator<Item = f32>, expected: &[f32]) -> Result<Self> {
        if expected.is_empty() {
            return Err("integer IQ difference empty reference".into());
        }
        let mut out = Self::default();
        let mut error_energy = 0f64;
        let mut reference_energy = 0f64;
        let mut peak = 0f64;
        let mut count = 0;
        for value in actual {
            let reference = *expected
                .get(count)
                .ok_or("integer IQ difference output length")?;
            if !value.is_finite() || !reference.is_finite() {
                return Err("integer IQ difference nonfinite value".into());
            }
            let error = (f64::from(value) - f64::from(reference)).abs();
            out.max_abs = out.max_abs.max(error);
            out.max_scaled_error = out
                .max_scaled_error
                .max(error / (1e-4 + 3e-4 * f64::from(reference).abs()));
            out.original_pointwise_failures += usize::from(compare(value, reference).is_err());
            error_energy += error * error;
            reference_energy += f64::from(reference).powi(2);
            peak = peak.max(f64::from(reference).abs());
            count += 1;
        }
        if count != expected.len() {
            return Err("integer IQ difference reference length".into());
        }
        out.relative_l2 = error_energy.sqrt() / reference_energy.sqrt().max(1e-12);
        out.rms = (error_energy / count as f64).sqrt();
        out.relative_peak = out.max_abs / peak.max(1e-12);
        Ok(out)
    }
    fn report(self) -> serde_json::Value {
        serde_json::json!({"max_abs":self.max_abs,"relative_l2":self.relative_l2,
            "rms":self.rms,"relative_peak":self.relative_peak,
            "max_original_scaled_error":self.max_scaled_error,
            "original_pointwise_failures":self.original_pointwise_failures,
            "reference_norm_denominator_floor":1e-12,"quality_accepted":false})
    }
}
fn differences(
    values: &[Zeroizing<Vec<u32>>],
    s: &k::Spans,
    expected: &Oracle,
) -> Result<[Difference; 3]> {
    if values.len() != 8 {
        return Err("integer IQ difference owner count".into());
    }
    let mut out = [Difference::default(); 3];
    for (which, (i, first, stride, width, count, reference)) in [
        (c::ACT, s.act_first(), s.act_stride(), FF, 10, &expected.act),
        (
            c::PART,
            s.partial_first(),
            s.partial_stride(),
            HIDDEN,
            10,
            &expected.part,
        ),
        (
            c::OUTPUT,
            s.output_first(),
            s.output_stride(),
            HIDDEN,
            1,
            &expected.output,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let end = first
            .checked_add((count - 1) * stride)
            .and_then(|v| v.checked_add(width))
            .ok_or("integer IQ difference owning extent")?;
        if end > values[i].len() || reference.len() != count * width {
            return Err("integer IQ difference original rank extent".into());
        }
        out[which] = Difference::new(
            (0..count).flat_map(|rank| {
                values[i][first + rank * stride..first + rank * stride + width]
                    .iter()
                    .copied()
                    .map(f32::from_bits)
            }),
            reference,
        )?;
    }
    Ok(out)
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
/// Fault operands use the same complete owners. Only active math cells and
/// the original active status regions may be written; every prefix, rank gap,
/// inactive MAX8 region and rounded owning tail must retain its original bits.
fn check_fault(
    values: &[Zeroizing<Vec<u32>>],
    before: &[Zeroizing<Vec<u32>>],
    g: k::Geometry,
    s: &k::Spans,
    nonfinite: bool,
    producer_ready: bool,
) -> Result<()> {
    if values.len() != 8
        || before.len() != 8
        || values.iter().zip(before).any(|(a, b)| a.len() != b.len())
    {
        return Err("segmented IQ fault owning snapshot lengths".into());
    }
    if k::validate_status(g, 1, s.status_first(), &values[c::STATUS]).is_ok() {
        return Err("segmented IQ subgroup public fault accepted".into());
    }
    for i in [c::MAP, c::INPUT, c::PROB, 7] {
        if values[i] != before[i] {
            return Err("segmented IQ subgroup fault readonly owner changed".into());
        }
    }
    for (i, first, stride, width, count) in [
        (c::ACT, s.act_first(), s.act_stride(), FF, 10),
        (c::PART, s.partial_first(), s.partial_stride(), HIDDEN, 10),
        (c::OUTPUT, s.output_first(), s.output_stride(), HIDDEN, 1),
    ] {
        for index in 0..values[i].len() {
            let offset = index.saturating_sub(first);
            let active = index >= first && offset / stride < count && offset % stride < width;
            if (!producer_ready || !active) && values[i][index] != before[i][index] {
                return Err("segmented IQ subgroup fault changed inactive math owner".into());
            }
        }
    }
    let d = HIDDEN.div_ceil(k::LANES);
    let f = FF.div_ceil(k::LANES);
    let regions = [
        (k::MAX_ROUTES, 10),
        (k::MAX_ROUTES, 10),
        (k::MAX_BATCH * d, d),
        (k::MAX_ROUTES * f, 10 * f),
        (k::MAX_ROUTES * d, 10 * d),
        (k::MAX_BATCH * d, d),
    ];
    let status = &values[c::STATUS];
    let end = s
        .status_first()
        .checked_add(g.status_words())
        .filter(|&end| end <= status.len())
        .ok_or("segmented IQ fault status extent")?;
    if status[..s.status_first()] != before[c::STATUS][..s.status_first()]
        || status[end..] != before[c::STATUS][end..]
    {
        return Err("segmented IQ fault status owning guard".into());
    }
    let mut first = s.status_first();
    for (region, (length, active)) in regions.into_iter().enumerate() {
        for index in 0..length {
            let value = status[first + index];
            if index < active {
                if region < 2 {
                    if value != u32::from(producer_ready) {
                        return Err("segmented IQ fault producer readiness".into());
                    }
                } else if value > 1 {
                    return Err("segmented IQ fault finite flag invalid".into());
                }
            } else if value != before[c::STATUS][first + index] {
                return Err("segmented IQ fault inactive status changed".into());
            }
        }
        first += length;
    }
    if nonfinite && status[s.status_first() + k::MAX_ROUTES * 2] != 0 {
        return Err("segmented IQ subgroup active NaN finite flag missing".into());
    }
    Ok(())
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
    integer: bool,
) -> Result<serde_json::Value> {
    integer_stage(integer, profile, "profile-start", 0, 0);
    let g = geometry(profile.0, profile.1)?;
    limits(g, l)?;
    if integer && l.max_compute_workgroup_storage_size < iq_moe_integer::MAX_SHARED_BYTES as u32 {
        return Err("segmented IQ integer workgroup storage limit".into());
    }
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
    // complete CPU oracle and compact original output oracle (combined<512KiB),
    // and bounded route metadata in the1MiB frame. No full bank host copy.
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
    integer_stage(integer, profile, "public-banks-verified", 0, 0);
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
    let integer_expected = if integer {
        Some(integer_oracle(
            g,
            seeds.each_ref().map(|v| v.wire.as_slice()),
            &x,
        )?)
    } else {
        None
    };
    integer_stage(integer, profile, "host-oracles-ready", 0, 0);
    for (i, &v) in x.iter().enumerate() {
        before[c::INPUT][s.input_first() + i] = v.to_bits();
    }
    before[c::MAP][s.map_first()..s.map_first() + k::MAP_WORDS].copy_from_slice(map.words());
    for (rank, &p) in map.probabilities().iter().enumerate() {
        before[c::PROB][s.prob_first() + rank] = p.to_bits();
    }
    let mut errors = [[0f32; 3]; 3];
    let mut between_errors = [0f32; 3];
    let mut integer_original_errors = None;
    let mut integer_float_errors = None;
    let mut original = None;
    for (case, case_errors) in errors
        .iter_mut()
        .enumerate()
        .take(if integer { 3 } else { 2 })
    {
        integer_stage(integer, profile, "case-start", case, 0);
        let subgroup = case == 1;
        let integer_case = case == 2;
        let reference = if integer_case {
            integer_expected
                .as_ref()
                .ok_or("segmented IQ integer reference absent")?
        } else {
            &expected
        };
        for (i, &id) in ids.iter().enumerate() {
            c::write(arena, id, &before[i][..words[i]], retired)?;
        }
        let mut previous: Option<Vec<Zeroizing<Vec<u32>>>> = None;
        for segment in 0..2 {
            integer_stage(integer, profile, "segment-launch", case, segment);
            pin.validate()?;
            physical(client, arena, &ids, &seeds, &banks, g)?;
            launch_segment(
                client,
                arena,
                &ids,
                &banks,
                g,
                &s,
                map.groups(),
                segment,
                subgroup,
                integer_case,
            )?;
            let got = snapshot(arena, &ids, retired)?;
            integer_stage(integer, profile, "segment-readback", case, segment);
            if let Some(prior) = &previous {
                inactive_preserved(&got, prior, &s)?;
            }
            let e = check(&got, &before, g, &s, reference, segment == 1, false)?;
            for (maximum, error) in case_errors.iter_mut().zip(e) {
                *maximum = maximum.max(error);
            }
            ledger.completed_fence(c::sample(client)?)?;
            previous = Some(got);
        }
        drop(previous);
        finish_math(client, arena, &ids, g, &s)?;
        integer_stage(integer, profile, "case-final-check", case, 2);
        let got = snapshot(arena, &ids, retired)?;
        let e = check(&got, &before, g, &s, reference, true, true)?;
        for (maximum, error) in case_errors.iter_mut().zip(e) {
            *maximum = maximum.max(error);
        }
        if case != 0 {
            let original_reference = original
                .as_ref()
                .ok_or("segmented IQ original oracle absent")?;
            if integer_case {
                // The completed independent integer check above already proves
                // every original owner/status/finite guard. These F32 deltas
                // are explicit diagnostics, not a relaxed original gate.
                integer_original_errors = Some(differences(&got, &s, original_reference)?);
                integer_float_errors = Some(differences(&got, &s, &expected)?);
            } else {
                between_errors = check(&got, &before, g, &s, original_reference, true, true)?;
            }
        } else {
            original = Some(compact_original(&got, &s)?);
            if original.as_ref().unwrap().owned_bytes()
                + expected.owned_bytes()
                + integer_expected.as_ref().map_or(0, Oracle::owned_bytes)
                > 512 * 1024
            {
                return Err("segmented IQ combined compact oracle host bound".into());
            }
        }
        pin.validate()?;
        ledger.completed_fence(c::sample(client)?)?;
    }
    drop(original);
    for fault in 0..if integer { 3 } else { 2 } {
        integer_stage(integer, profile, "fault-start", fault, 0);
        let nonfinite = fault == 0;
        // Compiled public fault operands only; preserve the original pin and
        // physical owners while testing actual producer/finite guard behavior.
        let mut fault_before = before.clone();
        if nonfinite {
            fault_before[c::INPUT][s.input_first()] = f32::NAN.to_bits();
        } else if fault == 1 {
            fault_before[c::MAP][s.map_first() + 3] = 0; // No valid group.
        } else {
            fault_before[c::INPUT][s.input_first()] = f32::MAX.to_bits();
        }
        for (i, &id) in ids.iter().enumerate() {
            c::write(arena, id, &fault_before[i][..words[i]], retired)?;
        }
        for segment in 0..2 {
            pin.validate()?;
            physical(client, arena, &ids, &seeds, &banks, g)?;
            launch_segment(
                client,
                arena,
                &ids,
                &banks,
                g,
                &s,
                map.groups(),
                segment,
                !integer,
                integer,
            )?;
        }
        finish_math(client, arena, &ids, g, &s)?;
        let got = snapshot(arena, &ids, retired)?;
        integer_stage(integer, profile, "fault-final-check", fault, 2);
        check_fault(&got, &fault_before, g, &s, nonfinite, nonfinite && !integer)?;
        ledger.completed_fence(c::sample(client)?)?;
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
    let mut report = serde_json::json!({"gate_up_dtype":profile.0.gguf_id(),"down_dtype":profile.1.gguf_id(),
        "hidden":HIDDEN,"ff":FF,"capacity":CAP,"segment_capacity":SEGMENT,"count":1,"groups":10,
        "bank_physical_bytes":pages.iter().sum::<u64>(),"private_owned_bytes":private_pages.iter().sum::<u64>(),
        "max_abs_activation_error":errors[0][0],"max_abs_partial_error":errors[0][1],"max_abs_rank_merge_error":errors[0][2],
        "subgroup_count1_original64":true,"subgroup_rows_per_cube":4,"subgroup_physical_lanes":128,
        "subgroup_max_abs_activation_error":errors[1][0],"subgroup_max_abs_partial_error":errors[1][1],"subgroup_max_abs_rank_merge_error":errors[1][2],
        "subgroup_original_activation_max_difference":between_errors[0],"subgroup_original_partial_max_difference":between_errors[1],"subgroup_original_merge_max_difference":between_errors[2],
        "subgroup_active_nonfinite_input_rejected":!integer,"subgroup_invalid_group_rejected_without_output_writes":!integer,
        "intermediate_inactive_rank_status_and_gap_checks":true,"original_global_local_slot_mapping":true,
        "full_public_bank_and_source_padding_preserved_after_private_reset":true});
    if integer {
        report["integer_policy"] = serde_json::json!(
            "two-residual-Q8-planes-original-packed-weights-subgroup-integer-dot"
        );
        report["integer_f32_identical"] = serde_json::json!(false);
        report["integer_reference_max_abs_errors"] = serde_json::json!(errors[2]);
        report["integer_original_gpu_differences"] = serde_json::json!(
            integer_original_errors
                .ok_or("integer IQ original diagnostics absent")?
                .map(Difference::report)
        );
        report["integer_independent_C_float_differences"] = serde_json::json!(
            integer_float_errors
                .ok_or("integer IQ C float diagnostics absent")?
                .map(Difference::report)
        );
        report["integer_active_nonfinite_input_rejected"] = serde_json::json!(true);
        report["integer_invalid_group_rejected_without_output_writes"] = serde_json::json!(true);
        report["integer_finite_max_residual_rejected_without_output_writes"] =
            serde_json::json!(true);
        report["integer_max_workgroup_bytes"] = serde_json::json!(iq_moe_integer::MAX_SHARED_BYTES);
        report["integer_shared_codebook_bytes"] =
            serde_json::json!(if profile.0 == TensorType::Iq3Xxs {
                1024
            } else {
                0
            });
    }
    Ok(report)
}
pub fn run(client: &ComputeClient<WgpuRuntime>, l: &wgpu::Limits) -> Result<()> {
    run_mode(client, l, false)
}
pub fn run_integer(client: &ComputeClient<WgpuRuntime>, l: &wgpu::Limits) -> Result<()> {
    run_mode(client, l, true)
}
fn run_mode(client: &ComputeClient<WgpuRuntime>, l: &wgpu::Limits, integer: bool) -> Result<()> {
    validate_mode(integer)?;
    if integer && l.max_compute_workgroup_storage_size < iq_moe_integer::MAX_SHARED_BYTES as u32 {
        return Err("segmented IQ integer workgroup storage limit".into());
    }
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
            reports.push(one(client, l, &mut arena, &retired, profile, integer)?);
            client.memory_cleanup();
        }
        Ok(reports)
    }));
    if integer {
        match &attempt {
            Ok(Err(error)) => println!(
                "public_iq_integer_failure class={}",
                fixed_failure_class(error.as_ref())
            ),
            Err(_) => println!("public_iq_integer_failure class=panic"),
            _ => {}
        }
    }
    let closed = arena.close();
    if integer && closed.is_err() {
        println!("public_iq_integer_failure class=owner-close");
    }
    let reports = match (attempt, closed, retired.load(Ordering::Acquire)) {
        (Ok(Ok(v)), Ok(()), false) => v,
        _ => return Err("segmented IQ public operator failed; no success receipt".into()),
    };
    let audit = arena
        .owned_zero_audit_receipt()
        .ok_or("segmented IQ full owner erasure absent")?;
    let scope = if integer {
        "fixed-public-synthetic-original-packed-IQ-two-Q8-plane-integer-MoE-only"
    } else {
        "fixed-public-synthetic-original-packed-segmented-IQ-MoE-only"
    };
    let report = serde_json::json!({"scope":scope,"cases":reports,
        "main_limit_bytes":c::MAIN_CAP,"bank_pages_max_bytes":512u64*1024*1024,"private_limit_bytes":c::PRIVATE_CAP,
        "runtime_io_limit_bytes":c::IO_CAP,"main_erased_buffers":audit.buffers,"main_erased_bytes":audit.bytes,
        "whole_model_accepted":false,"private_request_accepted":false,"performance_accepted":false,"physical_recovery_accepted":false});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    if integer {
        println!(
            "public_integer_iq_moe profiles=3 segments=2 ranks=10 alternative_arithmetic=1 accepted=1"
        );
    } else {
        println!("public_segmented_iq_moe profiles=3 segments=2 ranks=10 accepted=1");
    }
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
    fn compact_original_keeps_every_rank_and_edge_without_owning_gaps() {
        let (s, words) = spans(geometry(TensorType::Iq3Xxs, TensorType::Iq4Nl).unwrap()).unwrap();
        let mut values: Vec<_> = words
            .into_iter()
            .map(|n| {
                Zeroizing::new(vec![
                    f32::NAN.to_bits();
                    BufferLayout::new(n).unwrap().owned_words
                ])
            })
            .collect();
        for (i, first, stride, width, count, tag) in [
            (c::ACT, s.act_first(), s.act_stride(), FF, 10, 1000usize),
            (
                c::PART,
                s.partial_first(),
                s.partial_stride(),
                HIDDEN,
                10,
                20000,
            ),
            (
                c::OUTPUT,
                s.output_first(),
                s.output_stride(),
                HIDDEN,
                1,
                60000,
            ),
        ] {
            for rank in 0..count {
                for col in 0..width {
                    values[i][first + rank * stride + col] =
                        ((tag + rank * width + col) as f32).to_bits();
                }
            }
        }
        let actual = compact_original(&values, &s).unwrap();
        assert_eq!(actual.act.len(), 6400);
        assert_eq!(actual.part.len(), 25600);
        assert_eq!(actual.output.len(), 2560);
        for rank in 0..10 {
            assert_eq!(actual.act[rank * FF], (1000 + rank * FF) as f32);
            assert_eq!(
                actual.act[(rank + 1) * FF - 1],
                (1000 + (rank + 1) * FF - 1) as f32
            );
            assert_eq!(actual.part[rank * HIDDEN], (20000 + rank * HIDDEN) as f32);
            assert_eq!(
                actual.part[(rank + 1) * HIDDEN - 1],
                (20000 + (rank + 1) * HIDDEN - 1) as f32
            );
        }
        assert_eq!(
            (actual.output[0], actual.output[HIDDEN - 1]),
            (60000., 62559.)
        );
        assert_eq!(actual.owned_bytes(), (6400 + 25600 + 2560) * 4);
        assert!(compact_original(&values[..7], &s).is_err());
        values[c::ACT][s.act_first() + 9 * s.act_stride() + FF - 1] = f32::INFINITY.to_bits();
        assert!(compact_original(&values, &s).is_err());
        values[c::ACT][s.act_first() + 9 * s.act_stride() + FF - 1] = 1f32.to_bits();
        values[c::PART].truncate(s.partial_first() + 9 * s.partial_stride() + HIDDEN - 1);
        assert!(compact_original(&values, &s).is_err());
    }
    #[test]
    fn fault_owning_gate_rejects_prefix_gaps_inactive_ranks_and_readonly_changes() {
        let g = geometry(TensorType::Iq3Xxs, TensorType::Iq4Nl).unwrap();
        let (s, words) = spans(g).unwrap();
        let before: Vec<_> = words
            .into_iter()
            .map(|n| Zeroizing::new(vec![0u32; BufferLayout::new(n).unwrap().owned_words]))
            .collect();
        let mut actual = before.clone();
        for rank in 0..10 {
            actual[c::STATUS][s.status_first() + rank] = 1;
            actual[c::STATUS][s.status_first() + k::MAX_ROUTES + rank] = 1;
            actual[c::ACT][s.act_first() + rank * s.act_stride()] = f32::NAN.to_bits();
            actual[c::PART][s.partial_first() + rank * s.partial_stride()] = f32::NAN.to_bits();
        }
        actual[c::OUTPUT][s.output_first()] = f32::NAN.to_bits();
        check_fault(&actual, &before, g, &s, true, true).unwrap();
        for (owner, index) in [
            (c::ACT, 0),
            (c::ACT, s.act_first() + FF),
            (c::ACT, s.act_first() + 10 * s.act_stride()),
            (c::PART, s.partial_first() + HIDDEN),
            (c::OUTPUT, s.output_first() + HIDDEN),
            (c::STATUS, 0),
            (c::STATUS, s.status_first() + 10),
            (c::STATUS, s.status_first() + g.status_words()),
            (c::MAP, s.map_first()),
            (c::INPUT, s.input_first()),
            (c::PROB, s.prob_first()),
            (7, 0),
        ] {
            let mut damaged = actual.clone();
            damaged[owner][index] ^= 1;
            assert!(
                check_fault(&damaged, &before, g, &s, true, true).is_err(),
                "owner{owner} index{index}"
            );
        }
        check_fault(&before, &before, g, &s, false, false).unwrap();
        check_fault(&before, &before, g, &s, true, false).unwrap();
        assert!(check_fault(&actual, &before, g, &s, false, false).is_err());
        assert!(check_fault(&actual, &before, g, &s, true, false).is_err());
        assert!(check_fault(&actual[..7], &before, g, &s, true, true).is_err());
    }
    #[test]
    fn actual_shape_cpu_tree_oracles_are_finite_and_nonzero_for_all_profiles() {
        let x = Zeroizing::new(
            (0..HIDDEN)
                .map(|i| ((i * 7 % 13) as f32 - 6.) / 1024.)
                .collect::<Vec<_>>(),
        );
        for (gate, down) in PROFILES {
            let g = geometry(gate, down).unwrap();
            let payloads = g.projections().map(|p| seed_payload(p).unwrap());
            let rows = payloads.each_ref().map(|v| v.1.as_slice());
            let expected = oracle(rows, &x).unwrap();
            assert_eq!(expected.act.len(), 10 * FF);
            assert_eq!(expected.part.len(), 10 * HIDDEN);
            assert_eq!(expected.output.len(), HIDDEN);
            let integer =
                integer_oracle(g, payloads.each_ref().map(|v| v.0.as_slice()), &x).unwrap();
            for (stage, (actual, reference)) in [
                (&integer.act, &expected.act),
                (&integer.part, &expected.part),
                (&integer.output, &expected.output),
            ]
            .into_iter()
            .enumerate()
            {
                let difference = Difference::new(actual.iter().copied(), reference).unwrap();
                println!(
                    "public_host_iq_integer gate={} down={} stage={} differences={}",
                    gate.gguf_id(),
                    down.gguf_id(),
                    stage,
                    difference.report()
                );
            }
        }
    }
    #[test]
    fn alternate_difference_diagnostics_are_complete_finite_and_not_an_original_gate() {
        let reference = [0., 1., -2.];
        let stats = Difference::new([0., 1.01, -2.02].into_iter(), &reference).unwrap();
        assert_eq!(stats.original_pointwise_failures, 2);
        assert!(stats.relative_l2 > 0.009 && stats.relative_l2 < 0.011);
        assert!(stats.max_scaled_error > 1.);
        assert!(!stats.report()["quality_accepted"].as_bool().unwrap());
        assert!(Difference::new([0.; 2].into_iter(), &reference).is_err());
        assert!(Difference::new([0.; 4].into_iter(), &reference).is_err());
        assert!(Difference::new([f32::NAN].into_iter(), &[0.]).is_err());
        assert!(Difference::new([0.].into_iter(), &[f32::INFINITY]).is_err());
        assert!(Difference::new([].into_iter(), &[]).is_err());
        let extreme = Difference::new([f32::MAX].into_iter(), &[-f32::MAX]).unwrap();
        assert!(
            extreme.max_abs.is_finite()
                && extreme.relative_l2.is_finite()
                && extreme.rms.is_finite()
                && extreme.max_scaled_error.is_finite()
        );
        assert_eq!(extreme.original_pointwise_failures, 1);
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
