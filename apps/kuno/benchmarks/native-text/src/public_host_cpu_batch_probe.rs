//! Fixed synthetic operator packets over the real, full attached HOST mirror.
//! No model routing claim, hybrid merge, MTP construction or installed entry.
use crate::{
    daslab_carry_reference13 as reference, daslab_profile,
    executor::GpuExecutor,
    expert_banks::OriginalPackedProfile,
    private_arena::{BufferId, PrivateArena},
    public_host_expert_attachment::{Attachment, CpuLayerBorrow, PreparedPlan},
    quant_matvec::grouped_moe as k,
    resident_model::{ResidentModel, batch::Config},
    resident_util as u,
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::{
    GgufFile, GgufLimits,
    cpu_expert::{
        Backend, ExpertOperator, WorkerPlan,
        batch::{BatchPartials, Member, Rank},
    },
    model::Qwen4Exp,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const H: usize = 2560;
const FF: usize = 640;
const RANKS: usize = 10;
const MEMBERS: usize = 8;
const HOST_CAP: usize = 1024 * 1024;
const HEADER_CAP: usize = 16 * 1024;
const MAIN_CAP: usize = 4 * 1024 * 1024;
const REPEATS: usize = 3;
const LAYER: usize = 0;
const PARTIAL_WORDS: usize = 3 * RANKS * H;
const INPUT_BYTES: usize = 131072;
const INPUT_HANDOFF_BYTES: usize = 32768;
const CALLER_CAP: usize = INPUT_HANDOFF_BYTES + HEADER_CAP;
const ABS: f32 = 1e-4;
const REL: f32 = 3e-4;
const _: () = assert!(PARTIAL_WORDS * 4 + 524288 + HEADER_CAP < HOST_CAP);
const _: () = assert!(CALLER_CAP < 128 * 1024);
fn fixed_host_mask(host: usize, sparse: bool) -> Result<u16> {
    if ![1, 3, 5, 10].contains(&host) || sparse && host != 3 {
        return Err("attached CPU batch fixed HOST mask rejected".into());
    }
    Ok(if sparse {
        (1 << 0) | (1 << 4) | (1 << 9)
    } else {
        (1 << host) - 1
    })
}

pub(crate) struct Inputs {
    pub(crate) inline_dispatch: bool,
    directory: PathBuf,
    reference: PathBuf,
    receipt: PathBuf,
}
pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    build: crate::target_lab_build::Build,
) -> std::result::Result<Inputs, &'static str> {
    if args.len() != 3 || args[1] != "intel" || args[2] != "public-host-cpu-batch-probe" {
        return Err("attached HOST CPU batch requires fixed Intel operator mode");
    }
    if build != crate::target_lab_build::Build::Inline {
        return Err("attached HOST CPU batch requires original direct request I/O build");
    }
    let mut selected = BTreeMap::new();
    for (key, value) in env {
        if [b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|prefix| key.as_encoded_bytes().starts_with(prefix))
            && selected.insert(key, value).is_some()
        {
            return Err("attached HOST CPU batch duplicate option");
        }
    }
    if selected.remove(&OsString::from("KUNO_LAB_HOST_CPU_BATCH_PUBLIC_ONLY"))
        != Some(OsString::from("1"))
    {
        return Err("attached HOST CPU batch public gate required");
    }
    if selected.remove(&OsString::from("KUNO_LAB_STATIC_CACHE_BYTES"))
        != Some(OsString::from("8589934592"))
    {
        return Err("attached HOST CPU batch fixed 8GiB static cache required");
    }
    let inline_dispatch = build.selected(
        selected
            .remove(&OsString::from("KUNO_LAB_DEVICE_DISPATCH"))
            .as_deref(),
    )?;
    let mut path = |name: &str| {
        selected
            .remove(&OsString::from(name))
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .ok_or("attached HOST CPU batch required path missing")
    };
    let directory = path("KUNO_TEST_Q3_DIR")?;
    let reference = path("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE")?;
    let receipt = path("KUNO_PUBLIC_RECEIPT")?;
    if !selected.is_empty() {
        return Err("attached HOST CPU batch inherited/unsupported policy");
    }
    Ok(Inputs {
        inline_dispatch,
        directory,
        reference,
        receipt,
    })
}

struct ScrubBytes(cubecl::bytes::Bytes);
impl Drop for ScrubBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn read(arena: &PrivateArena, id: BufferId) -> Result<ScrubBytes> {
    let bytes = ScrubBytes(arena.client()?.read_one(arena.handle(id)?.clone())?);
    if bytes.0.len() != arena.layout(id)?.owned_bytes {
        return Err("attached CPU batch full read owner rejected".into());
    }
    Ok(bytes)
}
fn aligned_words(bytes: &[u8]) -> Result<&[u32]> {
    if !cfg!(target_endian = "little")
        || !bytes.len().is_multiple_of(4)
        || bytes.as_ptr().align_offset(std::mem::align_of::<u32>()) != 0
    {
        return Err("attached CPU batch aligned little-endian read rejected".into());
    }
    Ok(u32::from_bytes(bytes))
}
fn aligned_floats(bytes: &[u8]) -> Result<&[f32]> {
    aligned_words(bytes)?;
    Ok(f32::from_bytes(bytes))
}
fn upload(arena: &PrivateArena, id: BufferId, values: &[u32]) -> Result<usize> {
    let layout = arena.layout(id)?;
    if values.len() != layout.logical_words {
        return Err("attached CPU batch upload extent rejected".into());
    }
    let mut bytes = Zeroizing::new(Vec::new());
    bytes.try_reserve_exact(layout.owned_bytes)?;
    bytes.resize(layout.owned_bytes, 0);
    if bytes.capacity() != layout.owned_bytes {
        return Err("attached CPU batch upload actual capacity rejected".into());
    }
    for (dst, value) in bytes.chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    arena
        .client()?
        .write_private(arena.handle(id)?.clone(), std::mem::take(&mut *bytes))?;
    Ok(layout.owned_bytes)
}
fn copy_row(arena: &PrivateArena, source: BufferId, target: BufferId, row: usize) -> Result<()> {
    if row >= MEMBERS
        || u::words(arena, source)? != MEMBERS * RANKS * H
        || u::words(arena, target)? != RANKS * H
        || source == target
    {
        return Err("attached CPU batch row copy bounds rejected".into());
    }
    crate::residual_batch_gpu::copy_span::launch::<WgpuRuntime>(
        arena.client()?,
        CubeCount::Static((RANKS * H).div_ceil(64) as u32, 1, 1),
        CubeDim::new_1d(64),
        u::arg(arena, source)?,
        u::arg(arena, target)?,
        1,
        row * RANKS * H,
        RANKS * H,
        0,
        RANKS * H,
        RANKS * H,
    );
    Ok(())
}
fn audit_zero_regions(
    arena: &PrivateArena,
    b: &Buffers,
    source: BufferId,
    zero: impl Fn(usize) -> bool,
) -> Result<()> {
    let owned = arena.layout(source)?.owned_words;
    for first in (0..owned).step_by(RANKS * H) {
        let width = (owned - first).min(RANKS * H);
        u::zero(arena, b.row)?;
        crate::residual_batch_gpu::copy_span::launch::<WgpuRuntime>(
            arena.client()?,
            CubeCount::Static(width.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            u::raw_arg(arena.handle(source)?.clone(), owned)?,
            u::arg(arena, b.row)?,
            1,
            first,
            width,
            0,
            width,
            width,
        );
        let bytes = read(arena, b.row)?;
        let values = aligned_floats(&bytes.0)?;
        for (i, &value) in values.iter().take(width).enumerate() {
            if !value.is_finite() || zero(first + i) && value.to_bits() != 0 {
                return Err("attached CPU batch full owning zero/finite region rejected".into());
            }
        }
        if values[width..].iter().any(|x| x.to_bits() != 0) {
            return Err("attached CPU batch copied owner padding rejected".into());
        }
    }
    Ok(())
}
fn compare(cpu: f32, gpu: f32) -> Result<f32> {
    let delta = (cpu - gpu).abs();
    if !cpu.is_finite() || !gpu.is_finite() || delta > ABS + REL * cpu.abs().max(gpu.abs()) {
        return Err("attached CPU batch original partial numeric bound rejected".into());
    }
    Ok(delta)
}
fn encode_partials(arena: &PrivateArena, id: BufferId, output: &BatchPartials) -> Result<usize> {
    if output.full().len() != PARTIAL_WORDS
        || output.hidden() != H
        || arena.layout(id)?.owned_bytes != 524288
        || output
            .owned_bytes()
            .checked_add(524288 + HEADER_CAP)
            .is_none_or(|n| n > HOST_CAP)
    {
        return Err("attached CPU batch encoding phase capacity rejected".into());
    }
    let mut bytes = Zeroizing::new(Vec::new());
    bytes.try_reserve_exact(arena.layout(id)?.owned_bytes)?;
    bytes.resize(arena.layout(id)?.owned_bytes, 0);
    if output
        .owned_bytes()
        .checked_add(bytes.capacity() + HEADER_CAP)
        .is_none_or(|n| n > HOST_CAP)
    {
        return Err("attached CPU batch actual encoding phase rejected".into());
    }
    for (word, value) in bytes.chunks_exact_mut(4).zip(output.full()) {
        if !value.is_finite() {
            return Err("attached CPU batch finite output rejected".into());
        }
        word.copy_from_slice(&value.to_le_bytes());
    }
    let copied = bytes.len();
    arena
        .client()?
        .write_private(arena.handle(id)?.clone(), std::mem::take(&mut *bytes))?;
    Ok(copied)
}

struct Buffers {
    ids: [BufferId; 7],
    cpu: BufferId,
    row: BufferId,
    fault: BufferId,
    input_handoff: BufferId,
    spans: k::Spans,
}
impl Buffers {
    fn new(arena: &mut PrivateArena, geometry: k::Geometry) -> Result<Self> {
        if geometry.hidden() != H || geometry.ff() != FF || geometry.capacity() != 512 {
            return Err("attached CPU batch original geometry rejected".into());
        }
        let words = [
            k::MAP_WORDS,
            MEMBERS * H,
            k::MAX_ROUTES * FF,
            k::MAX_ROUTES * H,
            k::MAX_ROUTES,
            MEMBERS * H,
            geometry.status_words(),
        ];
        let first = arena.allocate_zeroed(words[0])?;
        let mut ids = [first; 7];
        for (i, &n) in words.iter().enumerate().skip(1) {
            ids[i] = arena.allocate_zeroed(n)?;
        }
        let cpu = arena.allocate_zeroed(PARTIAL_WORDS)?;
        let row = arena.allocate_zeroed(RANKS * H)?;
        let fault = arena.allocate_zeroed(1)?;
        let input_handoff = arena.allocate_zeroed(3 * H)?;
        let spans = k::Spans::new([0, 0, H, 0, FF, 0, H, 0, 0, H, 0]);
        spans.validate(geometry, words)?;
        if arena.charged_bytes() > MAIN_CAP {
            return Err("attached CPU batch main cap rejected".into());
        }
        Ok(Self {
            ids,
            cpu,
            row,
            fault,
            input_handoff,
            spans,
        })
    }
    fn clear(&self, arena: &PrivateArena) -> Result<()> {
        for id in [
            self.ids[2],
            self.ids[3],
            self.ids[5],
            self.ids[6],
            self.cpu,
            self.row,
            self.fault,
            self.input_handoff,
        ] {
            u::zero(arena, id)?;
        }
        cubecl::future::block_on(arena.client()?.sync())?;
        Ok(())
    }
    fn finite_join(&self, arena: &PrivateArena, ids: &[BufferId]) -> Result<()> {
        for &id in ids {
            u::validate(arena, id, self.fault)?;
        }
        let fault = read(arena, self.fault)?;
        if fault.0.iter().any(|&byte| byte != 0) {
            return Err("attached CPU batch owning finite fault".into());
        }
        drop(fault);
        cubecl::future::block_on(arena.client()?.sync())?;
        Ok(())
    }
}

struct Packet {
    experts: Zeroizing<Vec<u32>>,
    protected: Zeroizing<Vec<(u32, u32)>>,
    host_mask: u16,
    count: usize,
}
impl Packet {
    fn new(
        source: &CpuLayerBorrow<'_, '_, '_, '_>,
        count: usize,
        host: usize,
        sparse: bool,
    ) -> Result<Self> {
        if !(1..=3).contains(&count) || ![1, 3, 5, 10].contains(&host) || (sparse && host != 3) {
            return Err("attached CPU batch fixed packet rejected".into());
        }
        let mask = fixed_host_mask(host, sparse)?;
        let mut experts = Zeroizing::new(Vec::with_capacity(count * RANKS));
        let mut protected = Zeroizing::new(Vec::with_capacity(RANKS));
        for member in 0..count {
            let (mut hi, mut vi) = (0, 0);
            for rank in 0..RANKS {
                let (expert, slot) = if mask & (1 << rank) != 0 {
                    let ordinal = (hi + member) % host;
                    hi += 1;
                    (source.expert_for_host_ordinal(ordinal)?, 144 + ordinal)
                } else {
                    let slot = vi;
                    vi += 1;
                    (source.resident_expert(slot)?, slot)
                };
                experts.push(expert as u32);
                if !protected.iter().any(|&(id, _)| id == expert as u32) {
                    protected.push((expert as u32, slot as u32));
                }
            }
        }
        Ok(Self {
            experts,
            protected,
            host_mask: mask,
            count,
        })
    }
    fn map(&self, geometry: k::Geometry) -> Result<k::GroupMap<'_>> {
        Ok(k::GroupMap::new_frozen_sources(
            self.count,
            &self.experts,
            &vec![0.1; self.count * RANKS],
            &self.protected,
            geometry,
        )?)
    }
    fn host(&self, member: usize, rank: usize) -> bool {
        member < self.count && self.host_mask & (1 << rank) != 0
    }
}

fn cpu_run(
    source: &CpuLayerBorrow<'_, '_, '_, '_>,
    operator: &ExpertOperator,
    arena: &PrivateArena,
    b: &Buffers,
    packet: &Packet,
    cancel: &AtomicBool,
) -> Result<BatchPartials> {
    let required = operator.borrowed_batch_private_bytes(source.weights())?;
    if required
        .checked_add(CALLER_CAP)
        .is_none_or(|n| n > HOST_CAP)
    {
        return Err("attached CPU batch input/worker phase cap rejected".into());
    }
    // Full packet owners are checked and released sequentially before the
    // CPU phase. Their read/validation cost is inside this completed interval.
    {
        let map = packet.map(source.geometry())?;
        let map_bytes = read(arena, b.ids[0])?;
        let words = aligned_words(&map_bytes.0)?;
        if words[..k::MAP_WORDS] != map.words()[..] || words[k::MAP_WORDS..].iter().any(|&x| x != 0)
        {
            return Err("attached CPU batch actual handoff map rejected".into());
        }
        drop(map_bytes);
        let probabilities = read(arena, b.ids[4])?;
        let values = aligned_floats(&probabilities.0)?;
        if values[..k::MAX_ROUTES] != map.probabilities()[..]
            || values[k::MAX_ROUTES..].iter().any(|x| x.to_bits() != 0)
        {
            return Err("attached CPU batch actual handoff probabilities rejected".into());
        }
    }
    // The existing copy kernel copies only the three possible member inputs
    // into a full32KiB owner; no CPU-computed input substitutes for the GPU data.
    crate::residual_batch_gpu::copy_span::launch::<WgpuRuntime>(
        arena.client()?,
        CubeCount::Static(H.div_ceil(64) as u32, 3, 1),
        CubeDim::new_1d(64),
        u::arg(arena, b.ids[1])?,
        u::arg(arena, b.input_handoff)?,
        3,
        0,
        H,
        0,
        H,
        H,
    );
    let input = read(arena, b.input_handoff)?;
    if input.0.len() != INPUT_HANDOFF_BYTES {
        return Err("attached CPU batch input handoff span rejected".into());
    }
    let inputs = aligned_floats(&input.0)?;
    if inputs.iter().any(|x| !x.is_finite())
        || inputs[packet.count * H..].iter().any(|x| x.to_bits() != 0)
    {
        return Err("attached CPU batch actual input/tail rejected".into());
    }
    let mut views = Vec::with_capacity(packet.count * RANKS);
    for (index, &expert) in packet.experts.iter().enumerate() {
        if packet.host(index / RANKS, index % RANKS) {
            views.push(source.expert(expert as usize)?);
        }
    }
    let mut rows = Vec::with_capacity(packet.count);
    let mut offset = 0;
    for member in 0..packet.count {
        let mut ranks = Vec::with_capacity(RANKS);
        for rank in 0..RANKS {
            if packet.host(member, rank) {
                ranks.push(Rank::new(rank, &views[offset])?);
                offset += 1;
            }
        }
        rows.push(ranks);
    }
    let members = (0..packet.count)
        .map(|member| Member::new(&inputs[member * H..(member + 1) * H], &rows[member]))
        .collect::<kuno_text::Result<Vec<_>>>()?;
    let headers = views.capacity()
        * std::mem::size_of::<kuno_text::cpu_expert::batch::ExpertView<'_>>()
        + rows.capacity() * std::mem::size_of::<Vec<Rank<'_>>>()
        + rows
            .iter()
            .map(|r| r.capacity() * std::mem::size_of::<Rank<'_>>())
            .sum::<usize>()
        + members.capacity() * std::mem::size_of::<Member<'_>>()
        + packet.experts.capacity() * 4
        + packet.protected.capacity() * 8
        + std::mem::size_of::<Packet>()
        + 4096;
    if headers > HEADER_CAP
        || required
            .checked_add(input.0.len() + headers)
            .is_none_or(|n| n > HOST_CAP)
    {
        return Err("attached CPU batch actual borrowed header capacity rejected".into());
    }
    let output = operator.execute_borrowed_batch_with_private_limit(
        source.weights(),
        &members,
        cancel,
        HOST_CAP - CALLER_CAP,
    )?;
    // All call-owned inputs and leaf worker/activation scratch are gone before
    // the separate, bounded output encoding phase starts in the caller.
    Ok(output)
}
fn audit_partials(
    arena: &PrivateArena,
    b: &Buffers,
    packet: &Packet,
    cpu: &BatchPartials,
) -> Result<f32> {
    let mut error = 0f32;
    let uploaded = read(arena, b.cpu)?;
    let uploaded_words = aligned_floats(&uploaded.0)?;
    if uploaded_words.len() * 4 != 524288
        || uploaded_words[..PARTIAL_WORDS]
            .iter()
            .zip(cpu.full())
            .any(|(a, b)| a.to_bits() != b.to_bits())
        || uploaded_words[PARTIAL_WORDS..]
            .iter()
            .any(|x| x.to_bits() != 0)
    {
        return Err("attached CPU batch actual full uploaded owner rejected".into());
    }
    for member in 0..MEMBERS {
        copy_row(arena, b.ids[3], b.row, member)?;
        let bytes = read(arena, b.row)?;
        let gpu = aligned_floats(&bytes.0)?;
        if gpu.len() * 4 != 131072 || gpu[RANKS * H..].iter().any(|x| x.to_bits() != 0) {
            return Err("attached CPU batch row extent/tail rejected".into());
        }
        for rank in 0..RANKS {
            let expected = if member < cpu.members() {
                cpu.rank(member, rank)?
            } else {
                None
            };
            if packet.host(member, rank) != expected.is_some() {
                return Err("attached CPU batch actual rank ownership rejected".into());
            }
            for (i, &value) in gpu[rank * H..(rank + 1) * H].iter().enumerate() {
                if let Some(values) = expected {
                    error = error.max(compare(values[i], value)?);
                } else if value.to_bits() != 0 {
                    return Err("attached CPU batch inactive GPU rank rejected".into());
                }
            }
        }
    }
    drop(uploaded);
    audit_zero_regions(arena, b, b.ids[3], |i| i >= MEMBERS * RANKS * H)?;
    audit_zero_regions(arena, b, b.ids[2], |i| {
        i >= k::MAX_ROUTES * FF || !packet.host((i / FF) / RANKS, (i / FF) % RANKS)
    })?;
    audit_zero_regions(arena, b, b.ids[5], |_| true)?;
    let input = read(arena, b.ids[1])?;
    let values = aligned_floats(&input.0)?;
    if input.0.len() != INPUT_BYTES || values[packet.count * H..].iter().any(|x| x.to_bits() != 0) {
        return Err("attached CPU batch full original input owner/tail rejected".into());
    }
    drop(input);
    let status = read(arena, b.ids[6])?;
    let words = aligned_words(&status.0)?;
    for (i, &word) in words.iter().enumerate() {
        let expected = i < 2 * k::MAX_ROUTES && packet.host((i % k::MAX_ROUTES) / RANKS, i % RANKS);
        if word != u32::from(expected) {
            return Err("attached CPU batch original readiness/tail rejected".into());
        }
    }
    Ok(error)
}

fn cases(
    source: &CpuLayerBorrow<'_, '_, '_, '_>,
    client: &ComputeClient<WgpuRuntime>,
    retired: Arc<AtomicBool>,
    cancel: &AtomicBool,
) -> Result<serde_json::Value> {
    let mut arena = PrivateArena::new_transient_on_configured(client.clone(), MAIN_CAP, retired)?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = (|| -> Result<serde_json::Value> {
        let b = Buffers::new(&mut arena, source.geometry())?;
        let main = arena.charged_bytes();
        let mut reports = Vec::new();
        let mut completed_pairs = 0u64;
        for workers in [8, 12, 15] {
            let workers_started = Instant::now();
            let op = ExpertOperator::new_pooled(WorkerPlan::physical(workers)?, Backend::Avx2)?;
            let worker_pool_setup_ns = workers_started.elapsed().as_nanos();
            let required = op.borrowed_batch_private_bytes(source.weights())?;
            if required
                .checked_add(CALLER_CAP)
                .is_none_or(|n| n > HOST_CAP)
            {
                return Err("attached CPU batch explicit workers exceed cap".into());
            }
            for count in 1..=3 {
                for host in [1, 3, 5, 10] {
                    for sparse in [false, true].into_iter().filter(|s| !s || host == 3) {
                        let packet = Packet::new(source, count, host, sparse)?;
                        let map = packet.map(source.geometry())?;
                        upload(&arena, b.ids[0], map.words())?;
                        upload(&arena, b.ids[4], &map.probabilities().map(f32::to_bits))?;
                        let mut input = Zeroizing::new(vec![0u32; MEMBERS * H]);
                        for member in 0..count {
                            for i in 0..H {
                                input[member * H + i] =
                                    (((i * 17 + member * 29) % 127) as f32 / 127.0 - 0.5).to_bits();
                            }
                        }
                        upload(&arena, b.ids[1], &input)?;
                        drop(input);
                        drop(map);
                        let (mut gpu_ns, mut cpu_ns, mut max_error, mut copied) =
                            (0u128, 0u128, 0f32, 0usize);
                        let (mut prime_gpu_ns, mut prime_cpu_ns) = (0u128, 0u128);
                        for iteration in 0..=REPEATS {
                            b.clear(&arena)?;
                            let gpu_started = Instant::now();
                            source.launch_original_gpu_partials(
                                &arena,
                                &b.spans,
                                b.ids,
                                packet.protected.len(),
                                count,
                                cancel,
                            )?;
                            b.finite_join(&arena, &[b.ids[1], b.ids[2], b.ids[3]])?;
                            let gpu_elapsed = gpu_started.elapsed().as_nanos();
                            let cpu_started = Instant::now();
                            let cpu = cpu_run(source, &op, &arena, &b, &packet, cancel)?;
                            let encoded = encode_partials(&arena, b.cpu, &cpu)?;
                            b.finite_join(&arena, &[b.cpu])?;
                            let cpu_elapsed = cpu_started.elapsed().as_nanos();
                            max_error = max_error.max(audit_partials(&arena, &b, &packet, &cpu)?);
                            completed_pairs = completed_pairs
                                .checked_add(1)
                                .ok_or("attached CPU batch completed counter overflow")?;
                            if iteration != 0 {
                                gpu_ns += gpu_elapsed;
                                cpu_ns += cpu_elapsed;
                                copied += encoded;
                            } else {
                                prime_gpu_ns = gpu_elapsed;
                                prime_cpu_ns = cpu_elapsed;
                            }
                        }
                        reports.push(serde_json::json!({"members":count,"host_ranks_per_member":host,
                            "sparse_original_ranks":sparse,"physical_workers":workers,"repeats":REPEATS,
                            "gpu_completed_ns":gpu_ns,"cpu_completed_handoff_compute_upload_ns":cpu_ns,
                            "priming_gpu_completed_ns":prime_gpu_ns,"priming_cpu_completed_ns":prime_cpu_ns,
                            "worker_pool_setup_ns":worker_pool_setup_ns,
                            "cpu_encoding_copy_bytes":copied,"max_abs_partial_error":max_error,
                            "worker_input_phase_reserved_bytes":required + CALLER_CAP,
                            "phased_private_host_cap_bytes":HOST_CAP,"all8_gpu_partial_member_rows_checked":true}));
                    }
                }
            }
        }
        Ok(
            serde_json::json!({"cases":reports,"operator_main_bytes":main,
            "completed_cpu_gpu_pairs":completed_pairs,"actual_joined_cpu_batch_calls":completed_pairs,
            "actual_original_host_gu_dispatches":completed_pairs * 3,
            "actual_original_host_down_dispatches":completed_pairs * 3,
            "logical_cpu_partial_bytes":PARTIAL_WORDS * 4,"encoded_full_cpu_owner_bytes":524288,
            "input_handoff_owner_bytes":INPUT_HANDOFF_BYTES,"main_cap_bytes":MAIN_CAP,
            "runtime_io_shared_with_original_target":true,"runtime_io_cap_bytes":128u64 * 1024 * 1024,
            "numeric_abs_bound":ABS,
            "numeric_rel_bound":REL,"validation_readbacks_inside_timing":false,
            "pipeline_priming_iterations":1,"warm_cpu_L3_optimistic":true}),
        )
    })();
    let close = arena.close();
    let audit = arena.owned_zero_audit_receipt();
    attempt.and_then(|mut record| {
        close?;
        let audit = audit.ok_or("attached CPU batch operator owning close rejected")?;
        record["operator_close"] = serde_json::json!({"buffers":audit.buffers,"bytes":audit.bytes});
        Ok(record)
    })
}

pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    input: Inputs,
    device: &wgpu::Device,
) -> Result<()> {
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_host_cpu_batch_failure panic=true")
    }));
    let expected = reference::Reference::load(&input.reference)?;
    let paths = (1..=2)
        .map(|i| {
            input.directory.join(format!(
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
            ))
        })
        .collect::<Vec<_>>();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    if file.file_len() != 83_617_662_656 {
        return Err("attached CPU batch original DAS extent rejected".into());
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.dtype().gguf_id())),
    )
    .map_err(|_| "attached CPU batch original DAS inventory rejected")?;
    // SAFETY: only the fixed protected lab runner can select this mode and it
    // verifies the complete static artifact hashes before entry.
    unsafe {
        file.enable_mapped_weights()?;
    }
    let graph = Qwen4Exp::bind(&file)?;
    let reader = file.positional_reader()?;
    client.configure_public_target_request_io()?;
    u::select_retained_metadata(true);
    u::select_compute_scratch_clear(true);
    let retired = Arc::new(AtomicBool::new(false));
    let cancel = AtomicBool::new(false);
    let mut executor = GpuExecutor::new_with_profile(
        &file,
        client.clone(),
        8 * 1024 * 1024 * 1024,
        144,
        31 * 1024 * 1024 * 1024,
        OriginalPackedProfile::DaslabIq3,
    )?;
    executor.target_rows_reserved = true;
    executor.target_verify3_dense = true;
    executor.target_verify3_fused = true;
    executor.token_io_checkpoint = true;
    executor.enable_transient_resident(retired.clone())?;
    executor.fast_dense = true;
    executor.fast_dense_fault = true;
    executor.shared_kernels = true;
    executor.shared_gdn = true;
    let prepared = std::cell::OnceCell::<PreparedPlan<'_, '_>>::new();
    let mut attachment: Option<Attachment> = None;
    println!("public_host_cpu_batch_phase before_config");
    let mut config = Config::new(client, limits, 8)?
        .reserve_target_rows()?
        .reserve_latest_target_checkpoint()?
        .reserve_public_frozen_host_sources()?
        .reserve_public_target_request_io()?
        .reserve_public_warm_replay()?;
    config = config.reserve_public_frozen_host_batch_router()?;
    executor.target_batch_router_reserved = true;
    executor.target_frozen_host_router_reserved = true;
    println!("public_host_cpu_batch_phase before_constructor");
    let mut target = ResidentModel::new_with_native_batch(
        graph.bound_graph(),
        client.clone(),
        reference::CONTEXT,
        &mut executor,
        config,
        true,
    )?;
    let attempt = (|| -> Result<serde_json::Value> {
        let started = Instant::now();
        for position in 0..reference::CONTEXT {
            let token = expected.input_token(position)?;
            let row = target.prefill_chunk_with_carry(&mut executor, &[token], &cancel)?;
            if row.start() != position || row.count() != 1 {
                return Err("attached CPU batch original cold range rejected".into());
            }
            expected.verify(position, row.logits(0)?, row.hidden(0)?)?;
            drop(row);
            expected.verify_completed_prefix(
                position,
                1,
                target.completed_target_rows_cursor(&executor)?,
            )?;
            if position == 0 {
                target.begin_public_target_request_io(&mut executor, &cancel)?;
                if !target.public_target_request_io_active()? {
                    return Err("attached CPU batch cold request scope did not activate".into());
                }
            }
        }
        let cold_request_scope = crate::target_lab_build::reuse_record(
            target.end_public_target_request_io(&mut executor, &cancel)?,
        );
        if target.public_target_request_io_active()? {
            return Err("attached CPU batch cold request scope did not close".into());
        }
        println!("public_host_cpu_batch_phase cold13_complete");
        let cold_ns = started.elapsed().as_nanos();
        let setup = Instant::now();
        let (plan, fills) = target.prepare_public_host_sources(&mut executor, &cancel, false)?;
        prepared
            .set(plan)
            .map_err(|_| "attached CPU batch duplicate plan")?;
        let reset = target.reset_public_warm_replay(&mut executor, &cancel)?;
        let prepared = prepared.get().ok_or("attached CPU batch plan missing")?;
        // SAFETY: the same original hash-verified model and actual target device
        // remain owned through full target close/drop then source detach/close.
        attachment = Some(unsafe {
            target.attach_public_host_sources(
                &mut executor,
                prepared,
                device,
                prepared.budget().total_host_reserved_bytes,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                &cancel,
            )?
        });
        println!("public_host_cpu_batch_phase attached");
        target.activate_public_frozen_host_batch_router(&executor, &cancel)?;
        let first =
            target.prefill_chunk_with_carry(&mut executor, &[expected.input_token(0)?], &cancel)?;
        expected.verify(0, first.logits(0)?, first.hidden(0)?)?;
        drop(first);
        if target.completed_target_rows_cursor(&executor)? != 1
            || target.public_target_request_io_active()?
        {
            return Err("attached CPU batch actual warm boundary rejected".into());
        }
        println!("public_host_cpu_batch_phase warm_passed");
        let setup_ns = setup.elapsed().as_nanos();
        let borrow_started = Instant::now();
        let source = target.borrow_public_host_cpu_layer(
            attachment
                .as_mut()
                .ok_or("attached CPU batch owner absent")?,
            prepared,
            &reader,
            &executor,
            LAYER,
            &cancel,
        )?;
        let mapping_ns = borrow_started.elapsed().as_nanos();
        let placements = source.placements()?.map(|p| serde_json::json!({"memory_type":p.memory_type,
            "heap":p.heap,"property_flags":p.property_flags,"heap_flags":p.heap_flags,"heap_bytes":p.heap_bytes}));
        let mut value = cases(&source, client, retired.clone(), &cancel)?;
        println!("public_host_cpu_batch_phase operators_complete");
        value["actual_readonly_host_page_placements"] = placements.into();
        drop(source);
        value["cold_original13_full_vectors_ns"] = u64::try_from(cold_ns)?.into();
        value["cold_target_request_scope"] = cold_request_scope;
        value["warm_target_request_scope_activated"] = false.into();
        value["request_reuse_metadata_reserved_bytes"] = (4 * 1024 * 1024u64).into();
        value["source_bank_completion_experts"] = fills.experts.into();
        value["source_setup_and_original_warm_token0_ns"] = u64::try_from(setup_ns)?.into();
        value["readonly_layer_mapping_ns"] = u64::try_from(mapping_ns)?.into();
        value["original_warm_reset"] = serde_json::json!({"main_buffers":reset.main.buffers,"main_bytes":reset.main.bytes,
            "source_buffers":reset.source.buffers,"source_bytes":reset.source.bytes});
        Ok(value)
    })();
    let target_close = target.close();
    let audit = target.public_owned_zero_audit_receipt();
    drop(target);
    let source_close = (|| -> Result<_> {
        target_close?;
        let audit = audit.ok_or("attached CPU batch target zero audit missing")?;
        let owner = attachment
            .take()
            .ok_or("attached CPU batch source owner missing")?;
        // SAFETY: completed original target close/audit and lexical model drop
        // released every source consumer. The scoped CPU/GPU borrow has ended.
        let close = unsafe { owner.close()? };
        Ok((audit, close))
    })();
    let device_close = executor.close_transient_resident();
    drop(executor);
    let mut value = attempt?;
    let ((main, sources), close) = source_close?;
    device_close?;
    if retired.load(Ordering::Acquire) {
        return Err("attached CPU batch target retired".into());
    }
    file.verify_file_unchanged()?;
    value["target_close"] = serde_json::json!({"main_buffers":main.buffers,"main_bytes":main.bytes,
        "source_buffers":sources.buffers,"source_bytes":sources.bytes});
    value["attached_source_close"] = serde_json::json!({"registered_pages":close.registered_pages,
        "detached_pages":close.detached_pages,"physical_allocated_bytes":close.mirror.total_allocated_bytes,
        "physical_freed_bytes":close.mirror.total_freed_bytes,"live_backing_bytes":close.mirror.live_backing_bytes});
    value["scope"] = "fixed-public-real-attached-HOST-synthetic-partial-operator".into();
    value["layer"] = LAYER.into();
    value["mtp_executed"] = false.into();
    value["whole_model_cpu_hybrid_accepted"] = false.into();
    value["gpu_control_math"] = "original-F32-HOST-GU-DOWN".into();
    value["best_integer_gpu_control_compared"] = false.into();
    serde_json::to_writer_pretty(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(input.receipt)?,
        &value,
    )?;
    println!("public_host_cpu_batch_complete partials_accepted=true owners_erased=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_registration_rejects_other_models_cards_or_inherited_policies() {
        let args = ["lab", "intel", "public-host-cpu-batch-probe"].map(OsString::from);
        let env = [
            ("KUNO_LAB_HOST_CPU_BATCH_PUBLIC_ONLY", "1"),
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
            ("KUNO_TEST_Q3_DIR", "model"),
            ("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "reference"),
            ("KUNO_PUBLIC_RECEIPT", "new-receipt"),
        ]
        .map(|(k, v)| (OsString::from(k), OsString::from(v)));
        assert!(registration(&args, env.clone(), crate::target_lab_build::Build::Inline).is_ok());
        assert!(
            registration(
                &args,
                env.clone()
                    .into_iter()
                    .filter(|(key, _)| key != "KUNO_LAB_STATIC_CACHE_BYTES"),
                crate::target_lab_build::Build::Inline,
            )
            .is_err()
        );
        for value in ["0", "8589934591", "8589934593", "08589934592", "8GiB"] {
            let wrong = env.clone().into_iter().map(|(key, original)| {
                (
                    key.clone(),
                    if key == "KUNO_LAB_STATIC_CACHE_BYTES" {
                        OsString::from(value)
                    } else {
                        original
                    },
                )
            });
            assert!(registration(&args, wrong, crate::target_lab_build::Build::Inline).is_err());
        }
        for key in [
            "KUNO_LAB_FROZEN_GU_MATH",
            "KUNO_LAB_LANGUAGE_DECODE",
            "KUNO_LAB_MTP_DEVICE",
            "KUNO_TEST_OTHER",
        ] {
            assert!(
                registration(
                    &args,
                    env.clone().into_iter().chain([(key.into(), "1".into())]),
                    crate::target_lab_build::Build::Inline
                )
                .is_err()
            );
        }
        let other = ["lab", "nvidia", "public-host-cpu-batch-probe"].map(OsString::from);
        assert!(registration(&other, env.clone(), crate::target_lab_build::Build::Inline).is_err());
        assert!(
            registration(&args, env.clone(), crate::target_lab_build::Build::Threaded).is_err()
        );
        assert!(registration(&args, env, crate::target_lab_build::Build::Rejected).is_err());
    }
    #[test]
    fn original_partial_bounds_reject_nonfinite_and_drift() {
        assert!(compare(1., 1.0001).is_ok());
        assert!(compare(0., 0.0002).is_err());
        assert!(compare(f32::NAN, 0.).is_err());
        assert!(compare(0., f32::INFINITY).is_err());
    }
    #[test]
    fn readback_views_reject_unaligned_or_partial_words() {
        #[repr(align(4))]
        struct Aligned([u8; 12]);
        let bytes = Aligned([0; 12]);
        assert!(aligned_floats(&bytes.0).is_ok());
        assert!(aligned_words(&bytes.0[1..9]).is_err());
        assert!(aligned_floats(&bytes.0[..11]).is_err());
    }
    #[test]
    fn full_owning_sizes_and_sparse_original_ranks_keep_fixed_caps() {
        use crate::private_arena::BufferLayout;
        assert_eq!(
            BufferLayout::new(PARTIAL_WORDS).unwrap().owned_bytes,
            524288
        );
        assert_eq!(
            BufferLayout::new(MEMBERS * H).unwrap().owned_bytes,
            INPUT_BYTES
        );
        assert_eq!(
            BufferLayout::new(3 * H).unwrap().owned_bytes,
            INPUT_HANDOFF_BYTES
        );
        assert_eq!(BufferLayout::new(k::MAP_WORDS).unwrap().owned_bytes, 32768);
        assert_eq!(BufferLayout::new(k::MAX_ROUTES).unwrap().owned_bytes, 32768);
        for n in [1, 3, 5, 10] {
            assert_eq!(fixed_host_mask(n, false).unwrap().count_ones(), n as u32);
        }
        assert_eq!(fixed_host_mask(3, true).unwrap(), 529);
        assert!(fixed_host_mask(5, true).is_err());
        assert!(fixed_host_mask(0, false).is_err());
        assert!(fixed_host_mask(11, false).is_err());
    }
}
