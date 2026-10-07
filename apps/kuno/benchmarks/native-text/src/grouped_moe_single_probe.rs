//! Fixed-public seven-format single-member grouped operator acceptance.
//! Two block-aligned original packed slices form deterministic PUBLIC rows. This is
//! not a model projection or a timing fixture. All ten ranks use the actual
//! pinned wave/pending ledger; private MAX8/MAX80 pages remain fully owned.
use super::{PROB, Result, c, compare};
use crate::{
    bank_slots::{BankSlots, StagedSlots},
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::grouped_moe as k,
    token_waves,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{GgufFile, TensorType};
use std::sync::atomic::AtomicBool;
use zeroize::{Zeroize, Zeroizing};

const D: usize = 256;
const CAP: usize = 16;
const EXPERTS: [usize; 10] = [15, 2, 14, 1, 13, 0, 12, 3, 11, 4];
const FORMATS: [TensorType; 7] = [
    TensorType::F32,
    TensorType::F16,
    TensorType::Bf16,
    TensorType::Q4K,
    TensorType::Q5_0,
    TensorType::Q6K,
    TensorType::Q8_0,
];
struct Bank {
    handle: Handle,
    bytes: usize,
    phase: usize,
}
struct ReadLease(cubecl::bytes::Bytes);
impl Drop for ReadLease {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn geometry(kind: TensorType, tile: usize) -> Result<k::Geometry> {
    let p = k::Projection::new(kind.gguf_id(), D, D, 512)?;
    Ok(k::Geometry::new([p; 3], tile, CAP)?)
}
fn spans(g: k::Geometry) -> (k::Spans, [usize; 7]) {
    (
        k::Spans::new([5, 7, D + 3, 11, D + 5, 13, D + 7, 9, 17, D + 9, 3]),
        [
            5 + k::MAP_WORDS + 7,
            7 + 7 * (D + 3) + D + 13,
            11 + 79 * (D + 5) + D + 17,
            13 + 79 * (D + 7) + D + 19,
            9 + k::MAX_ROUTES + 11,
            17 + 7 * (D + 9) + D + 23,
            3 + g.status_words() + 29,
        ],
    )
}
fn seed_index(slot: usize, row: usize, phase: usize) -> usize {
    (slot * 3 + row * 5 + phase) % 2
}
fn tree(weights: &[f32], input: &[f32]) -> f32 {
    let mut lanes = Zeroizing::new([0f32; 64]);
    for column in 0..D {
        lanes[column % 64] += weights[column] * input[column];
    }
    for stride in [32, 16, 8, 4, 2, 1] {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
    }
    lanes[0]
}
fn active(index: usize, first: usize, stride: usize, count: usize) -> bool {
    index >= first && (index - first) / stride < count && (index - first) % stride < D
}
fn read_public(
    client: &ComputeClient<WgpuRuntime>,
    bank: &Bank,
    seed: &[u8],
    row_bytes: usize,
    retired: &AtomicBool,
) -> Result<()> {
    let bytes = ReadLease(
        client
            .read_one(bank.handle.clone())
            .map_err(|_| "single-member public owning read")?,
    );
    if bytes.0.len() != bank.bytes {
        return Err("single-member public bank changed".into());
    }
    for slot in 0..CAP {
        for row in 0..D {
            let src = seed_index(slot, row, bank.phase) * row_bytes;
            let dst = (slot * D + row) * row_bytes;
            if bytes.0[dst..dst + row_bytes] != seed[src..src + row_bytes] {
                return Err("single-member public bank changed".into());
            }
        }
    }
    drop(bytes);
    c::audit(client, retired)
}
fn physical(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    ids: &[BufferId; 7],
    banks: &[Bank; 3],
    g: k::Geometry,
) -> Result<()> {
    let private = ids
        .iter()
        .map(|&id| arena.handle(id))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if private.iter().any(|h| h.stream != private[0].stream)
        || banks.iter().any(|b| b.handle.stream != private[0].stream)
    {
        return Err("single-member stream mismatch".into());
    }
    let resources = private
        .iter()
        .map(|h| client.get_resource((*h).clone()))
        .chain(banks.iter().map(|b| client.get_resource(b.handle.clone())))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| "single-member full physical lookup")?;
    let pages = resources
        .iter()
        .enumerate()
        .map(|(i, r)| -> Result<_> {
            let r = r.resource();
            let logical = if i < 7 {
                arena.layout(ids[i])?.owned_bytes
            } else {
                banks[i - 7].bytes
            } as u64;
            let bound = if i < 7 {
                logical
            } else {
                main_admission::page_bytes(logical)?
            };
            Ok((
                r.buffer.clone(),
                r.offset,
                r.size,
                r.buffer.size(),
                logical,
                bound,
                i < 7,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    c::validate_physical_pages(&pages)?;
    g.validate_shard(0, CAP, banks.each_ref().map(|b| b.bytes / 4))?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn launch(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    ids: &[BufferId; 7],
    banks: &[Bank; 3],
    g: k::Geometry,
    s: &k::Spans,
    groups: usize,
) -> Result<()> {
    let arg = |i: usize| -> Result<_> {
        // SAFETY: complete owning private pages and nonaliasing were checked
        // against the same client immediately before this launch.
        Ok(unsafe {
            ArrayArg::from_raw_parts(
                arena.handle(ids[i])?.clone(),
                arena.layout(ids[i])?.owned_words,
            )
        })
    };
    // SAFETY: full immutable original packed bank extents passed the same
    // physical/shard proof; no selected slot is part of the binding view.
    let bank =
        |i: usize| unsafe { ArrayArg::from_raw_parts(banks[i].handle.clone(), banks[i].bytes / 4) };
    let [ga, up, down] = g.projections();
    unsafe {
        k::grouped_gate_up::launch_unchecked::<WgpuRuntime>(
            client,
            g.grid(groups, false)?,
            CubeDim::new_1d(64),
            bank(0),
            bank(1),
            arg(c::INPUT)?,
            arg(c::MAP)?,
            arg(c::ACT)?,
            arg(c::STATUS)?,
            s.map_first(),
            s.input_first(),
            s.input_stride(),
            s.act_first(),
            s.act_stride(),
            s.status_first(),
            0,
            CAP,
            D,
            D,
            ga.dtype(),
            up.dtype(),
            ga.row_bytes(),
            up.row_bytes(),
            g.tile(),
        );
        k::grouped_down::launch_unchecked::<WgpuRuntime>(
            client,
            g.grid(groups, true)?,
            CubeDim::new_1d(64),
            bank(2),
            arg(c::ACT)?,
            arg(c::MAP)?,
            arg(c::PART)?,
            arg(c::STATUS)?,
            s.map_first(),
            s.act_first(),
            s.act_stride(),
            s.partial_first(),
            s.partial_stride(),
            s.status_first(),
            0,
            CAP,
            D,
            D,
            down.dtype(),
            down.row_bytes(),
            g.tile(),
        );
        // Same readiness-gated original rank0..9 merge and full finite grids.
        k::grouped_reduce::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(4, 8, 1),
            CubeDim::new_1d(64),
            arg(c::PART)?,
            arg(c::PROB)?,
            arg(c::MAP)?,
            arg(c::STATUS)?,
            arg(c::OUTPUT)?,
            s.map_first(),
            s.partial_first(),
            s.partial_stride(),
            s.prob_first(),
            s.status_first(),
            s.output_first(),
            s.output_stride(),
            D,
        );
        k::grouped_finite::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(4, 80, 4),
            CubeDim::new_1d(64),
            arg(c::INPUT)?,
            arg(c::ACT)?,
            arg(c::PART)?,
            arg(c::OUTPUT)?,
            arg(c::MAP)?,
            arg(c::STATUS)?,
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
            D,
            D,
        );
    }
    Ok(())
}
pub(super) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    file: &GgufFile,
    arena: &mut PrivateArena,
    retired: &AtomicBool,
) -> Result<Vec<serde_json::Value>> {
    if limits.max_compute_workgroup_storage_size < 4096
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bindings_per_bind_group < 7
    {
        return Err("single-member actual device limits".into());
    }
    let mut reports = Vec::with_capacity(14);
    for kind in FORMATS {
        let base = geometry(kind, 8)?;
        let row_bytes = base.projections()[0].row_bytes();
        let info = file
            .tensors()
            .values()
            .find(|t| t.dtype() == kind && t.byte_len() >= (2 * row_bytes) as u64)
            .ok_or("single-member original format blocks absent")?;
        let mut seed = Zeroizing::new(vec![0u8; 2 * row_bytes]);
        file.tensor(info.name())
            .ok_or("single-member original block binding")?
            .read_range(0, &mut seed)?;
        let mut decoded = Zeroizing::new(vec![0f32; 2 * D]);
        kind.decode(&seed, &mut decoded)?;
        if decoded.iter().any(|x| !x.is_finite()) {
            return Err("single-member original block nonfinite".into());
        }
        let mut ledger = MainAdmission::new(c::MAIN_CAP, c::sample(client)?)?;
        let mut public = Vec::with_capacity(3);
        for phase in 0..3 {
            let mut wire = Zeroizing::new(vec![0u8; CAP * D * row_bytes]);
            for slot in 0..CAP {
                for row in 0..D {
                    let src = seed_index(slot, row, phase) * row_bytes;
                    let dst = (slot * D + row) * row_bytes;
                    wire[dst..dst + row_bytes].copy_from_slice(&seed[src..src + row_bytes]);
                }
            }
            ledger.observe_before(c::sample(client)?)?;
            ledger.precharge(&[wire.len() as u64])?;
            let handle = client.create_from_slice(&wire);
            // Complete/audit each <=4MiB public upload separately; retain all
            // original public banks through both cases, with no private source.
            let sentinel = ReadLease(
                client
                    .read_one(handle.clone().offset_end((wire.len() - 16) as u64))
                    .map_err(|_| "single-member public upload completion")?,
            );
            if &*sentinel.0 != &wire[..16] {
                return Err("single-member public upload sentinel".into());
            }
            drop(sentinel);
            c::audit(client, retired)?;
            ledger.completed_fence(c::sample(client)?)?;
            public.push(Bank {
                handle,
                bytes: wire.len(),
                phase,
            });
        }
        let banks: [Bank; 3] = public
            .try_into()
            .map_err(|_| "single-member public bank count")?;
        for tile in [8, 1] {
            let g = geometry(kind, tile)?;
            let (s, words) = spans(g);
            g.validate_count(1)?;
            s.validate(g, words)?;
            let pages = words
                .map(|w| BufferLayout::new(w).map(|l| l.owned_bytes as u64))
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if pages.iter().sum::<u64>() > c::PRIVATE_CAP as u64
                || 3 * 4 * 1024 * 1024 + 4 * pages.iter().sum::<u64>() > c::HOST_CAP as u64
            {
                return Err("single-member fixed owning/host bound".into());
            }
            ledger.observe_before(c::sample(client)?)?;
            ledger.precharge(&pages)?;
            let mut ids = Vec::with_capacity(7);
            for n in words {
                ids.push(arena.allocate_zeroed(n)?);
            }
            let ids: [BufferId; 7] = ids
                .try_into()
                .map_err(|_| "single-member private owner count")?;
            physical(client, arena, &ids, &banks, g)?;
            let mut input = token_waves::Input::new(1, CAP, 512)
                .map_err(|_| "single-member fixed route configuration")?;
            for rank in 0..10 {
                input
                    .set(0, rank, EXPERTS[rank] as u32, PROB[rank])
                    .map_err(|_| "single-member fixed rank")?;
            }
            let plan = token_waves::Plan::build(&input, &AtomicBool::new(false))
                .map_err(|_| "single-member fixed wave planning")?;
            let packet = c::Packet::from_plan(&plan)?;
            let bank_slots = BankSlots::new(CAP)?;
            let mut staged = StagedSlots::begin(&bank_slots)?;
            let wave = plan.wave(0).ok_or("single-member missing pinned wave")?;
            let selected = Zeroizing::new(
                wave.protected_experts()
                    .iter()
                    .map(|&e| e as usize)
                    .collect::<Vec<_>>(),
            );
            let pending = staged.plan(&bank_slots, &selected)?;
            let pin = c::ProtectedWave::new(wave, &pending, &staged, &bank_slots)?;
            let map = pin.map(&packet)?;
            map.validate(&packet.experts[..10], &packet.probabilities[..10], CAP)?;
            map.validate_geometry(g)?;
            let mut before = Vec::with_capacity(7);
            for (i, n) in words.into_iter().enumerate() {
                let l = arena.layout(ids[i])?;
                let mut image = Zeroizing::new(vec![0u32; l.owned_words]);
                if matches!(i, c::INPUT | c::ACT | c::PART | c::OUTPUT) {
                    image[..n].fill(0x3f340000);
                }
                before.push(image);
            }
            let x = Zeroizing::new(
                (0..D)
                    .map(|i| ((i * 7 % 13) as f32 - 6.) / 1024.)
                    .collect::<Vec<_>>(),
            );
            for (i, &v) in x.iter().enumerate() {
                before[c::INPUT][s.input_first() + i] = v.to_bits();
            }
            before[c::MAP][s.map_first()..s.map_first() + k::MAP_WORDS]
                .copy_from_slice(map.words());
            for (rank, &p) in map.probabilities().iter().enumerate() {
                before[c::PROB][s.prob_first() + rank] = p.to_bits();
            }
            let mut expected = Zeroizing::new(vec![0f32; 21 * D]);
            for rank in 0..10 {
                let at = selected
                    .iter()
                    .position(|&expert| expert == EXPERTS[rank])
                    .ok_or("single-member original rank missing from pinned union")?;
                let slot = pending.slots[at];
                for row in 0..D {
                    let gate = tree(&decoded[seed_index(slot, row, 0) * D..][..D], &x);
                    let up = tree(&decoded[seed_index(slot, row, 1) * D..][..D], &x);
                    expected[rank * D + row] = kuno_text::ops::silu(gate) * up;
                }
                for row in 0..D {
                    let value = tree(
                        &decoded[seed_index(slot, row, 2) * D..][..D],
                        &expected[rank * D..][..D],
                    );
                    expected[(10 + rank) * D + row] = value;
                }
            }
            for row in 0..D {
                for rank in 0..10 {
                    let term = PROB[rank] * expected[(10 + rank) * D + row];
                    expected[20 * D + row] += term;
                }
            }
            for i in 0..7 {
                c::write(arena, ids[i], &before[i][..words[i]], retired)?;
            }
            pin.validate()?;
            physical(client, arena, &ids, &banks, g)?;
            launch(client, arena, &ids, &banks, g, &s, map.groups())?;
            let mut actual = Vec::with_capacity(7);
            for &id in &ids {
                actual.push(c::read(arena, id, retired)?);
            }
            c::audit(client, retired)?;
            pin.validate()?;
            ledger.completed_fence(c::sample(client)?)?;
            k::validate_status(g, 1, s.status_first(), &actual[c::STATUS])?;
            let mut max = [0f32; 3];
            for (i, first, stride, count, oracle, which) in [
                (c::ACT, s.act_first(), s.act_stride(), 10, 0, 0),
                (
                    c::PART,
                    s.partial_first(),
                    s.partial_stride(),
                    10,
                    10 * D,
                    1,
                ),
                (c::OUTPUT, s.output_first(), s.output_stride(), 1, 20 * D, 2),
            ] {
                for index in 0..actual[i].len() {
                    if active(index, first, stride, count) {
                        let at = oracle + (index - first) / stride * D + (index - first) % stride;
                        max[which] = max[which].max(compare(
                            f32::from_bits(actual[i][index]),
                            expected[at],
                            1e-4,
                            3e-4,
                        )?);
                    } else if actual[i][index] != before[i][index] {
                        return Err("single-member gap/inactive/owning tail changed".into());
                    }
                }
            }
            for i in [c::MAP, c::INPUT, c::PROB] {
                if actual[i] != before[i] {
                    return Err("single-member readonly owner changed".into());
                }
            }
            for bank in &banks {
                read_public(client, bank, &seed, row_bytes, retired)?;
            }
            staged.validate_origin(&bank_slots)?;
            arena.reset()?; // mandatory full owning zero inspection on every accepted case
            reports.push(serde_json::json!({"dtype":kind.gguf_id(),"tile":tile,"count":1,"groups":10,
                "max_abs_activation_error":max[0],"max_abs_partial_error":max[1],"max_abs_rank_merge_error":max[2],
                "private_pages_bytes":pages.iter().sum::<u64>(),"all_ten_ranks":true,"full_offset_tail_status_and_erasure_checks":true}));
        }
    }
    file.verify_file_unchanged()?;
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_format_single_member_fixture_keeps_full_max8_max80_pages_and_public_grid() {
        for kind in FORMATS {
            for tile in [8, 1] {
                let g = geometry(kind, tile).unwrap();
                let (s, words) = spans(g);
                s.validate(g, words).unwrap();
                assert_eq!(g.status_words(), 864);
                assert!(matches!(
                    g.grid(10, false).unwrap(),
                    CubeCount::Static(256, 10, 1)
                ));
                let pages: usize = words
                    .into_iter()
                    .map(|w| BufferLayout::new(w).unwrap().owned_bytes)
                    .sum();
                assert!(pages < c::PRIVATE_CAP);
                assert!(3 * 4 * 1024 * 1024 + 4 * pages < c::HOST_CAP);
                for field in 0..7 {
                    let mut short = words;
                    short[field] = 0;
                    assert!(s.validate(g, short).is_err());
                }
            }
        }
    }
}
