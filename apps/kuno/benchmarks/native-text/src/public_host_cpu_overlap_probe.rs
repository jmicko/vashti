//! Default-unused, fixed count3 complete-layer CPU HOST/GPU resident comparison.
//! Same original source/rank math; no normal-model hybrid or four-row permission.
#![allow(dead_code)] // Root owns fixed-mode registration and all physical execution.
#[path = "public_host_cpu_overlap_merge.rs"]
mod merge;
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
const MERGED_ABS: f32 = 1e-3;
const MERGED_REL: f32 = 3e-3;
const COUNT: usize = 3;
const RANK_OWNER_WORDS: usize = 256;
const OPERATOR_BYTES: usize = 3_866_624;
const WORKERS: usize = 15;
fn fixed_host_mask(host: usize, sparse: bool) -> Result<u16> {
    if ![0, 3, 5, 10].contains(&host) || sparse && host != 3 {
        return Err("overlap fixed HOST mask rejected".into());
    }
    Ok(if sparse { 529 } else { (1 << host) - 1 })
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
    if args.len() != 3 || args[1] != "intel" || args[2] != "public-host-cpu-overlap-probe" {
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
    if selected.remove(&OsString::from("KUNO_LAB_HOST_CPU_OVERLAP_PUBLIC_ONLY"))
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
        return Err("attached CPU overlap full read owner rejected".into());
    }
    Ok(bytes)
}
fn aligned_words(bytes: &[u8]) -> Result<&[u32]> {
    if !cfg!(target_endian = "little")
        || !bytes.len().is_multiple_of(4)
        || bytes.as_ptr().align_offset(std::mem::align_of::<u32>()) != 0
    {
        return Err("attached CPU overlap aligned little-endian read rejected".into());
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
        return Err("attached CPU overlap upload extent rejected".into());
    }
    let mut bytes = Zeroizing::new(Vec::new());
    bytes.try_reserve_exact(layout.owned_bytes)?;
    bytes.resize(layout.owned_bytes, 0);
    if bytes.capacity() != layout.owned_bytes {
        return Err("attached CPU overlap upload actual capacity rejected".into());
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
        return Err("attached CPU overlap row copy bounds rejected".into());
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
                return Err("attached CPU overlap full owning zero/finite region rejected".into());
            }
        }
        if values[width..].iter().any(|x| x.to_bits() != 0) {
            return Err("attached CPU overlap copied owner padding rejected".into());
        }
    }
    Ok(())
}
fn compare(cpu: f32, gpu: f32) -> Result<f32> {
    let delta = (cpu - gpu).abs();
    if !cpu.is_finite() || !gpu.is_finite() || delta > ABS + REL * cpu.abs().max(gpu.abs()) {
        return Err("attached CPU overlap original partial numeric bound rejected".into());
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
        return Err("attached CPU overlap encoding phase capacity rejected".into());
    }
    let mut bytes = Zeroizing::new(Vec::new());
    bytes.try_reserve_exact(arena.layout(id)?.owned_bytes)?;
    bytes.resize(arena.layout(id)?.owned_bytes, 0);
    if output
        .owned_bytes()
        .checked_add(bytes.capacity() + HEADER_CAP)
        .is_none_or(|n| n > HOST_CAP)
    {
        return Err("attached CPU overlap actual encoding phase rejected".into());
    }
    for (word, value) in bytes.chunks_exact_mut(4).zip(output.full()) {
        if !value.is_finite() {
            return Err("attached CPU overlap finite output rejected".into());
        }
        word.copy_from_slice(&value.to_le_bytes());
    }
    let copied = bytes.len();
    arena
        .client()?
        .write_private(arena.handle(id)?.clone(), std::mem::take(&mut *bytes))?;
    Ok(copied)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    AllGpu,
    Serial,
    Concurrent,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::AllGpu => "all_gpu",
            Self::Serial => "serial_cpu_then_gpu",
            Self::Concurrent => "concurrent_cpu_and_gpu",
        }
    }
}
struct Buffers {
    ids: [BufferId; 7],
    cpu: BufferId,
    row: BufferId,
    fault: BufferId,
    input_handoff: BufferId,
    reference_act: BufferId,
    reference_part: BufferId,
    reference_output: BufferId,
    owners: BufferId,
    spans: k::Spans,
}
const WORDS: [usize; 15] = [
    804, 20480, 51200, 204800, 80, 20480, 4800, 76800, 25600, 1, 7680, 51200, 204800, 20480, 256,
];
impl Buffers {
    fn new(arena: &mut PrivateArena, g: k::Geometry) -> Result<Self> {
        if g.hidden() != H || g.ff() != FF || g.capacity() != 512 || g.four_members() {
            return Err("overlap exact original geometry rejected".into());
        }
        let quoted = WORDS
            .into_iter()
            .try_fold(0usize, |sum, n| {
                sum.checked_add(crate::private_arena::BufferLayout::new(n).ok()?.owned_bytes)
            })
            .ok_or("overlap MAIN quote overflow")?;
        if quoted != OPERATOR_BYTES || quoted > MAIN_CAP {
            return Err("overlap preallocation MAIN quote rejected".into());
        }
        let first = arena.allocate_zeroed(WORDS[0])?;
        let mut all = [first; 15];
        for i in 1..15 {
            all[i] = arena.allocate_zeroed(WORDS[i])?;
        }
        if arena.charged_bytes() != OPERATOR_BYTES {
            return Err("overlap actual MAIN charge rejected".into());
        }
        let ids = all[..7].try_into()?;
        let spans = k::Spans::new([0, 0, H, 0, FF, 0, H, 0, 0, H, 0]);
        spans.validate(g, WORDS[..7].try_into()?)?;
        // Original exclusive-page mode: authenticate all actual roots once
        // before private input fill, not merely a rounded quote.
        let handles = all
            .map(|id| arena.handle(id).cloned())
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let actual = u::resources(arena.client()?, handles)?;
        if actual.len() != 15 {
            return Err("overlap MAIN physical count rejected".into());
        }
        for (i, r) in actual.iter().enumerate() {
            let r = r.resource();
            let h = arena.handle(all[i])?;
            let n = arena.layout(all[i])?.owned_bytes as u64;
            if h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || h.size() != n
                || h.size_in_used() != n
                || r.offset != 0
                || r.size != n
                || r.buffer.size() != n
                || actual[..i]
                    .iter()
                    .any(|old| old.resource().buffer == r.buffer)
                || h.stream != arena.handle(first)?.stream
            {
                return Err("overlap MAIN physical owner rejected".into());
            }
        }
        Ok(Self {
            ids,
            cpu: all[7],
            row: all[8],
            fault: all[9],
            input_handoff: all[10],
            reference_act: all[11],
            reference_part: all[12],
            reference_output: all[13],
            owners: all[14],
            spans,
        })
    }
    fn clear(&self, a: &PrivateArena) -> Result<()> {
        for id in [
            self.ids[2],
            self.ids[3],
            self.ids[5],
            self.ids[6],
            self.cpu,
            self.row,
            self.fault,
            self.input_handoff,
            self.owners,
        ] {
            u::zero(a, id)?;
        }
        cubecl::future::block_on(a.client()?.sync())?;
        Ok(())
    }
    fn finite_join(&self, a: &PrivateArena) -> Result<()> {
        for id in [self.ids[1], self.ids[2], self.ids[3], self.ids[5], self.cpu] {
            u::validate(a, id, self.fault)?;
        }
        let f = read(a, self.fault)?;
        if f.0.iter().any(|x| *x != 0) {
            return Err("overlap full finite/readiness fault".into());
        }
        drop(f);
        cubecl::future::block_on(a.client()?.sync())?;
        Ok(())
    }
    fn copy_full(&self, a: &PrivateArena, src: BufferId, dst: BufferId) -> Result<()> {
        let sw = a.layout(src)?.owned_words;
        let dw = a.layout(dst)?.owned_words;
        if src == dst || sw != dw || a.layout(src)?.logical_words != a.layout(dst)?.logical_words {
            return Err("overlap reference copy exact span rejected".into());
        }
        u::copy_words::launch::<WgpuRuntime>(
            a.client()?,
            u::grid(sw)?,
            CubeDim::new_1d(64),
            u::raw_arg(a.handle(src)?.clone(), sw)?,
            u::raw_arg(a.handle(dst)?.clone(), dw)?,
        );
        Ok(())
    }
    fn snapshot(&self, a: &PrivateArena) -> Result<()> {
        for (src, dst) in [
            (self.ids[2], self.reference_act),
            (self.ids[3], self.reference_part),
            (self.ids[5], self.reference_output),
        ] {
            self.copy_full(a, src, dst)?;
        }
        cubecl::future::block_on(a.client()?.sync())?;
        Ok(())
    }
    fn reduce(&self, a: &PrivateArena) -> Result<()> {
        k::grouped_reduce::launch::<WgpuRuntime>(
            a.client()?,
            CubeCount::Static(40, 3, 1),
            CubeDim::new_1d(64),
            u::arg(a, self.ids[3])?,
            u::arg(a, self.ids[4])?,
            u::arg(a, self.ids[0])?,
            u::arg(a, self.ids[6])?,
            u::arg(a, self.ids[5])?,
            0,
            0,
            H,
            0,
            0,
            0,
            H,
            H,
        );
        Ok(())
    }
    fn merge(&self, a: &PrivateArena) -> Result<()> {
        merge::launch(
            a,
            [
                self.ids[3],
                self.cpu,
                self.ids[4],
                self.owners,
                self.ids[6],
                self.ids[5],
                self.fault,
            ],
        )?;
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct Case {
    host: usize,
    disjoint: bool,
    sparse: bool,
}
const CASES: [Case; 5] = [
    Case {
        host: 5,
        disjoint: false,
        sparse: false,
    },
    Case {
        host: 5,
        disjoint: true,
        sparse: false,
    },
    Case {
        host: 0,
        disjoint: false,
        sparse: false,
    },
    Case {
        host: 10,
        disjoint: false,
        sparse: false,
    },
    Case {
        host: 3,
        disjoint: false,
        sparse: true,
    },
];
struct Packet {
    experts: Zeroizing<Vec<u32>>,
    protected: Zeroizing<Vec<(u32, u32)>>,
    probabilities: Zeroizing<Vec<f32>>,
    masks: [u16; 3],
    case: Case,
}
impl Packet {
    fn new(source: &CpuLayerBorrow<'_, '_, '_, '_>, case: Case) -> Result<Self> {
        let mask = fixed_host_mask(case.host, case.sparse)?;
        let mut experts = Zeroizing::new(Vec::with_capacity(30));
        let mut protected = Zeroizing::new(Vec::with_capacity(30));
        for member in 0..COUNT {
            let (mut hi, mut vi) = (0, 0);
            for rank in 0..10 {
                let (id, slot) = if mask & (1 << rank) != 0 {
                    let ordinal = if case.disjoint {
                        member * case.host + hi
                    } else {
                        (hi + member) % case.host
                    };
                    hi += 1;
                    (source.expert_for_host_ordinal(ordinal)?, 144 + ordinal)
                } else {
                    let n = 10 - case.host;
                    let slot = if case.disjoint {
                        member * n + vi
                    } else {
                        (vi + member) % n
                    };
                    vi += 1;
                    (source.resident_expert(slot)?, slot)
                };
                experts.push(id as u32);
                if !protected.iter().any(|(e, _)| *e == id as u32) {
                    protected.push((id as u32, slot as u32));
                }
            }
        }
        let probabilities = Zeroizing::new(
            (0..COUNT)
                .flat_map(|_| {
                    if case.sparse {
                        vec![0., 0.05, 0.1, 0., 0.2, 0.15, 0.05, 0.15, 0.1, 0.2]
                    } else {
                        vec![0.1; 10]
                    }
                })
                .collect(),
        );
        Ok(Self {
            experts,
            protected,
            probabilities,
            masks: [mask; 3],
            case,
        })
    }
    fn map(&self, g: k::Geometry) -> Result<k::GroupMap<'_>> {
        Ok(k::GroupMap::new_frozen_sources(
            COUNT,
            &self.experts,
            &self.probabilities,
            &self.protected,
            g,
        )?)
    }
    fn host(&self, member: usize, rank: usize) -> bool {
        member < COUNT && rank < 10 && self.masks[member] & (1 << rank) != 0
    }
    fn has_cpu(&self) -> bool {
        self.masks.iter().any(|m| *m != 0)
    }
    fn has_gpu(&self) -> bool {
        self.masks.iter().any(|m| *m != 1023)
    }
}
/// Actual common GPU packet/input handoff; all leases are call-owned and wiped.
fn handoff(
    source: &CpuLayerBorrow<'_, '_, '_, '_>,
    a: &PrivateArena,
    b: &Buffers,
    p: &Packet,
) -> Result<ScrubBytes> {
    {
        let expected = p.map(source.geometry())?;
        let raw = read(a, b.ids[0])?;
        let words = aligned_words(&raw.0)?;
        if words[..804] != expected.words()[..] || words[804..].iter().any(|w| *w != 0) {
            return Err("overlap actual MAP full owner rejected".into());
        }
        drop(raw);
        let raw = read(a, b.ids[4])?;
        let values = aligned_floats(&raw.0)?;
        if values[..80]
            .iter()
            .zip(expected.probabilities())
            .any(|(a, b)| a.to_bits() != b.to_bits())
            || values[80..].iter().any(|x| x.to_bits() != 0)
        {
            return Err("overlap actual probability full owner rejected".into());
        }
    }
    crate::residual_batch_gpu::copy_span::launch::<WgpuRuntime>(
        a.client()?,
        CubeCount::Static(40, 3, 1),
        CubeDim::new_1d(64),
        u::arg(a, b.ids[1])?,
        u::arg(a, b.input_handoff)?,
        3,
        0,
        H,
        0,
        H,
        H,
    );
    let raw = read(a, b.input_handoff)?;
    let x = aligned_floats(&raw.0)?;
    if raw.0.len() != INPUT_HANDOFF_BYTES
        || x.iter().any(|v| !v.is_finite())
        || x[3 * H..].iter().any(|v| v.to_bits() != 0)
    {
        return Err("overlap actual input full owner rejected".into());
    }
    Ok(raw)
}
#[derive(Default)]
struct Work {
    gpu_calls: u64,
    cpu_calls: u64,
    concurrent_joined: bool,
    copy_bytes: usize,
    intersection_ns: u64,
}
fn split(
    source: &CpuLayerBorrow<'_, '_, '_, '_>,
    op: &ExpertOperator,
    a: &PrivateArena,
    b: &Buffers,
    p: &Packet,
    concurrent: bool,
    cancel: &AtomicBool,
) -> Result<Work> {
    let mut work = Work::default();
    if !p.has_cpu() {
        source.launch_resident_original_gpu_partials(
            a,
            &b.spans,
            b.ids,
            p.protected.len(),
            cancel,
        )?;
        work.gpu_calls = 4;
        upload(a, b.owners, &merge::packet([0; 3], false)?)?;
        b.merge(a)?;
        return Ok(work);
    }
    let required = op.borrowed_batch_private_bytes(source.weights())?;
    if required
        .checked_add(CALLER_CAP)
        .is_none_or(|n| n > HOST_CAP)
    {
        return Err("overlap original CPU scratch precharge rejected".into());
    }
    let input = handoff(source, a, b, p)?;
    let inputs = aligned_floats(&input.0)?;
    let mut views = Vec::with_capacity(30);
    for (index, &expert) in p.experts.iter().enumerate() {
        if p.host(index / 10, index % 10) {
            views.push(source.expert(expert as usize)?);
        }
    }
    let mut rows = Vec::with_capacity(3);
    let mut vi = 0;
    for member in 0..3 {
        let mut ranks = Vec::with_capacity(10);
        for rank in 0..10 {
            if p.host(member, rank) {
                ranks.push(Rank::new(rank, &views[vi])?);
                vi += 1;
            }
        }
        rows.push(ranks);
    }
    let members = (0..3)
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
        + p.experts.capacity() * 4
        + p.protected.capacity() * 8
        + p.probabilities.capacity() * 4
        + 4096;
    if headers > HEADER_CAP
        || required
            .checked_add(input.0.len() + headers)
            .is_none_or(|n| n > HOST_CAP)
    {
        return Err("overlap actual CPU header/scratch cap rejected".into());
    }
    let weights = source.weights();
    let (cpu, gpu_interval, cpu_interval) = if concurrent && p.has_gpu() {
        // Scope never captures the non-Send source/map guard. GPU sync submits
        // and joins while the external CPU task/persistent workers are running.
        std::thread::scope(|scope| -> Result<_> {
            let task = scope.spawn(|| {
                let start = Instant::now();
                let result = op.execute_borrowed_batch_with_private_limit(
                    weights,
                    &members,
                    cancel,
                    HOST_CAP - CALLER_CAP,
                );
                (start, Instant::now(), result)
            });
            let gs = Instant::now();
            let gpu = source.launch_resident_original_gpu_partials(
                a,
                &b.spans,
                b.ids,
                p.protected.len(),
                cancel,
            );
            let ge = Instant::now();
            let joined = task
                .join()
                .map_err(|_| "overlap joined CPU task panicked")?;
            // Both consumers are joined even if either returned an error.
            gpu?;
            let (cs, ce, cpu) = joined;
            Ok((cpu?, Some((gs, ge)), Some((cs, ce))))
        })?
    } else {
        let cs = Instant::now();
        let cpu = op.execute_borrowed_batch_with_private_limit(
            weights,
            &members,
            cancel,
            HOST_CAP - CALLER_CAP,
        )?;
        let ce = Instant::now();
        if p.has_gpu() {
            source.launch_resident_original_gpu_partials(
                a,
                &b.spans,
                b.ids,
                p.protected.len(),
                cancel,
            )?;
        }
        (cpu, None, Some((cs, ce)))
    };
    for member in 0..3 {
        for rank in 0..10 {
            if cpu.rank(member, rank)?.is_some() != p.host(member, rank) {
                return Err("overlap actual CPU joined mask rejected".into());
            }
        }
    }
    work.cpu_calls = 1;
    work.gpu_calls = if p.has_gpu() { 4 } else { 0 };
    if let (Some((gs, ge)), Some((cs, ce))) = (gpu_interval, cpu_interval) {
        work.concurrent_joined = true;
        let first = gs.max(cs);
        let last = ge.min(ce);
        if last > first {
            work.intersection_ns = u64::try_from(last.duration_since(first).as_nanos())?;
        }
    }
    // Original scratch and caller input/headers release BEFORE encoding peaks.
    drop(members);
    drop(rows);
    drop(views);
    drop(input);
    work.copy_bytes = encode_partials(a, b.cpu, &cpu)?;
    drop(cpu);
    upload(a, b.owners, &merge::packet(p.masks, true)?)?;
    b.merge(a)?;
    Ok(work)
}
fn checkpoint(a: &PrivateArena) -> Result<u64> {
    let r = cubecl::future::block_on(a.client()?.sanitize_transient_io_audited())?;
    if r.inspected_buffers == 0 || r.inspected_bytes == 0 {
        return Err("overlap actual transient IO audit missing".into());
    }
    Ok(r.inspected_bytes)
}
fn complete_arm(
    source: &CpuLayerBorrow<'_, '_, '_, '_>,
    op: &ExpertOperator,
    a: &PrivateArena,
    b: &Buffers,
    p: &Packet,
    arm: Arm,
    cancel: &AtomicBool,
) -> Result<(u64, Work, u64)> {
    b.clear(a)?;
    checkpoint(a)?;
    let started = Instant::now();
    let work = if arm == Arm::AllGpu {
        source.launch_all_original_gpu_partials(a, &b.spans, b.ids, p.protected.len(), cancel)?;
        b.reduce(a)?;
        Work {
            gpu_calls: 10,
            ..Work::default()
        }
    } else {
        split(source, op, a, b, p, arm == Arm::Concurrent, cancel)?
    };
    b.finite_join(a)?;
    let inspected = checkpoint(a)?;
    Ok((
        u64::try_from(started.elapsed().as_nanos())?,
        work,
        inspected,
    ))
}
fn row_read(
    a: &PrivateArena,
    b: &Buffers,
    source: BufferId,
    first: usize,
    n: usize,
) -> Result<ScrubBytes> {
    let owned = a.layout(source)?.owned_words;
    if n == 0 || n > RANKS * H || first.checked_add(n).is_none_or(|end| end > owned) {
        return Err("overlap full owning inspection span rejected".into());
    }
    u::zero(a, b.row)?;
    crate::residual_batch_gpu::copy_span::launch::<WgpuRuntime>(
        a.client()?,
        CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
        CubeDim::new_1d(64),
        u::raw_arg(a.handle(source)?.clone(), owned)?,
        u::arg(a, b.row)?,
        1,
        first,
        n,
        0,
        n,
        n,
    );
    let bytes = read(a, b.row)?;
    let values = aligned_floats(&bytes.0)?;
    if bytes.0.len() != 131072 || values[n..].iter().any(|v| v.to_bits() != 0) {
        return Err("overlap inspection row padding rejected".into());
    }
    Ok(bytes)
}
#[allow(clippy::too_many_arguments)] // Exact original owner/region/tolerance control, fixed operator only.
fn compare_owner(
    a: &PrivateArena,
    b: &Buffers,
    source: BufferId,
    reference: BufferId,
    logical: usize,
    zero: impl Fn(usize) -> bool,
    exact: impl Fn(usize) -> bool,
    abs: f32,
    rel: f32,
) -> Result<f32> {
    if a.layout(source)?.owned_words > a.layout(reference)?.owned_words {
        return Err("overlap comparison owner mismatch".into());
    }
    let mut error = 0f32;
    let words = a.layout(source)?.owned_words;
    for first in (0..words).step_by(RANKS * H) {
        let n = (words - first).min(RANKS * H);
        let value = row_read(a, b, source, first, n)?;
        let expected = row_read(a, b, reference, first, n)?;
        let v = aligned_floats(&value.0)?;
        let e = aligned_floats(&expected.0)?;
        for i in 0..n {
            let index = first + i;
            if !v[i].is_finite()
                || !e[i].is_finite()
                || index >= logical && v[i].to_bits() != 0
                || zero(index) && v[i].to_bits() != 0
            {
                return Err("overlap full owning finite/zero rejected".into());
            }
            if index < logical && !zero(index) {
                let delta = (v[i] - e[i]).abs();
                if exact(index) && v[i].to_bits() != e[i].to_bits()
                    || delta > abs + rel * v[i].abs().max(e[i].abs())
                {
                    return Err("overlap original numeric/rank ordering bound rejected".into());
                }
                error = error.max(delta);
            }
        }
    }
    Ok(error)
}
fn check_rank_owners(
    a: &PrivateArena,
    b: &Buffers,
    expected: &[u32; RANK_OWNER_WORDS],
) -> Result<()> {
    let raw = read(a, b.owners)?;
    let w = aligned_words(&raw.0)?;
    if w[..RANK_OWNER_WORDS] != expected[..] || w[RANK_OWNER_WORDS..].iter().any(|v| *v != 0) {
        return Err("overlap full rank owner/readiness packet rejected".into());
    }
    Ok(())
}
#[cube(launch)]
fn mutate_public_fixture_word(values: &mut Array<u32>, index: usize, value: u32) {
    if ABSOLUTE_POS == 0 && index < values.len() {
        values[index] = value;
    }
}
fn mutate(a: &PrivateArena, id: BufferId, index: usize, value: u32) -> Result<()> {
    let n = a.layout(id)?.owned_words;
    if index >= n {
        return Err("overlap fixed mutation owner span rejected".into());
    }
    mutate_public_fixture_word::launch::<WgpuRuntime>(
        a.client()?,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        u::raw_arg(a.handle(id)?.clone(), n)?,
        index,
        value,
    );
    Ok(())
}
fn expect_fault(a: &PrivateArena, b: &Buffers, stage: &'static str) -> Result<()> {
    let f = read(a, b.fault)?;
    let w = aligned_words(&f.0)?;
    if w[0] == 0 || w[1..].iter().any(|v| *v != 0) {
        return Err(stage.into());
    }
    Ok(())
}
/// Fixed PUBLIC negative operator controls, outside every timing arm. Restore
/// actual full source-independent private owners, never change any weight page.
fn negative_controls(
    source: &CpuLayerBorrow<'_, '_, '_, '_>,
    a: &PrivateArena,
    b: &Buffers,
    p: &Packet,
    cancel: &AtomicBool,
) -> Result<usize> {
    if !p.case.sparse || p.probabilities[0].to_bits() != 0 {
        return Err("overlap negative zero-prob fixture rejected".into());
    }
    let original = merge::packet([0; 3], false)?;
    // Original actual all-GPU status/partials and reference snapshots remain.
    for (bad, stage) in [
        (0usize, "overlap negative01 count sticky fault missing"),
        (4, "overlap negative02 owner sticky fault missing"),
        (34, "overlap negative03 readiness sticky fault missing"),
        (
            64,
            "overlap negative04 logical padding sticky fault missing",
        ),
    ] {
        let mut packet = original;
        packet[bad] = if bad == 0 { 4 } else { 2 };
        upload(a, b.owners, &packet)?;
        u::zero(a, b.fault)?;
        b.merge(a)?;
        expect_fault(a, b, stage)?;
    }
    upload(a, b.owners, &original)?;
    // Missing genuine GPU GU readiness is rejected, with unchanged rank payload.
    mutate(a, b.ids[6], 0, 0)?;
    u::zero(a, b.fault)?;
    b.merge(a)?;
    expect_fault(
        a,
        b,
        "overlap negative05 GPU readiness sticky fault missing",
    )?;
    mutate(a, b.ids[6], 0, 1)?;
    // A p=0 route still checks its full original partial for NaN.
    mutate(a, b.ids[3], 0, f32::NAN.to_bits())?;
    u::zero(a, b.fault)?;
    b.merge(a)?;
    expect_fault(
        a,
        b,
        "overlap negative06 zero probability NaN sticky fault missing",
    )?;
    b.copy_full(a, b.reference_part, b.ids[3])?;
    // A prior sticky fault survives a successful subsequent valid merge.
    upload(a, b.fault, &[1])?;
    b.merge(a)?;
    expect_fault(a, b, "overlap negative07 prior sticky fault missing")?;
    // Logical and physical inactive rank-packet padding are fully inspected.
    u::zero(a, b.fault)?;
    upload(a, b.owners, &original)?;
    mutate(a, b.owners, RANK_OWNER_WORDS + 1, 7)?;
    if check_rank_owners(a, b, &original).is_ok() {
        return Err("overlap negative08 physical packet padding mutation accepted".into());
    }
    upload(a, b.owners, &original)?;
    let mut aliases = b.ids;
    aliases[2] = aliases[1];
    if source
        .launch_resident_original_gpu_partials(a, &b.spans, aliases, p.protected.len(), cancel)
        .is_ok()
    {
        return Err("overlap negative09 actual owner alias mutation accepted".into());
    }
    if source.expert(source.resident_expert(0)?).is_ok() {
        return Err("overlap negative10 HOST source classification mutation accepted".into());
    }
    if source.expert(512).is_ok() {
        return Err("overlap negative11 HOST source range mutation accepted".into());
    }
    u::zero(a, b.fault)?;
    u::zero(a, b.ids[5])?;
    b.reduce(a)?;
    b.finite_join(a)?;
    check_rank_owners(a, b, &original)?;
    checkpoint(a)?;
    Ok(11)
}
fn audit(
    source: &CpuLayerBorrow<'_, '_, '_, '_>,
    a: &PrivateArena,
    b: &Buffers,
    p: &Packet,
    arm: Arm,
) -> Result<(f32, f32)> {
    let split = arm != Arm::AllGpu;
    let host = |index: usize, width: usize| p.host((index / width) / 10, (index / width) % 10);
    compare_owner(
        a,
        b,
        b.ids[2],
        b.reference_act,
        80 * FF,
        |i| i >= 30 * FF || split && host(i, FF),
        |_| true,
        0.,
        0.,
    )?;
    let partial = compare_owner(
        a,
        b,
        b.ids[3],
        b.reference_part,
        80 * H,
        |i| i >= 30 * H || split && host(i, H),
        |_| true,
        0.,
        0.,
    )?;
    let cpu = compare_owner(
        a,
        b,
        b.cpu,
        b.reference_part,
        30 * H,
        |i| !split || i >= 30 * H || !host(i, H),
        |_| false,
        ABS,
        REL,
    )?;
    let output = compare_owner(
        a,
        b,
        b.ids[5],
        b.reference_output,
        8 * H,
        |i| i >= 3 * H,
        |_| !split,
        MERGED_ABS,
        MERGED_REL,
    )?;
    let input = read(a, b.ids[1])?;
    let x = aligned_floats(&input.0)?;
    if input.0.len() != INPUT_BYTES
        || x.iter().any(|v| !v.is_finite())
        || x[3 * H..].iter().any(|v| v.to_bits() != 0)
    {
        return Err("overlap full INPUT inactive tail rejected".into());
    }
    drop(input);
    let status = read(a, b.ids[6])?;
    let words = aligned_words(&status.0)?;
    for (i, &word) in words.iter().enumerate() {
        let route = i % 80;
        let active = i < 160 && route < 30 && (!split || !p.host(route / 10, route % 10));
        if word != u32::from(active) {
            return Err("overlap full original status or inactive tail rejected".into());
        }
    }
    drop(status);
    // New all-GPU merge must match the OLD grouped_reduce bits before timing.
    if arm == Arm::AllGpu {
        upload(a, b.owners, &merge::packet([0; 3], false)?)?;
        u::zero(a, b.ids[5])?;
        b.merge(a)?;
        b.finite_join(a)?;
        compare_owner(
            a,
            b,
            b.ids[5],
            b.reference_output,
            8 * H,
            |i| i >= 3 * H,
            |_| true,
            0.,
            0.,
        )?;
    }
    let expected = if split {
        merge::packet(p.masks, p.has_cpu())?
    } else {
        merge::packet([0; 3], false)?
    };
    check_rank_owners(a, b, &expected)?;
    // Same actual packet/source authority retained across all consumers.
    let _ = handoff(source, a, b, p)?;
    b.finite_join(a)?;
    checkpoint(a)?;
    Ok((partial.max(cpu), output))
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
        let started = Instant::now();
        let op = ExpertOperator::new_pooled(WorkerPlan::physical(WORKERS)?, Backend::Avx2)?;
        let pool_ns = started.elapsed().as_nanos();
        let required = op.borrowed_batch_private_bytes(source.weights())?;
        if required
            .checked_add(CALLER_CAP)
            .is_none_or(|n| n > HOST_CAP)
        {
            return Err("overlap original15-worker scratch cap rejected".into());
        }
        let mut reports = Vec::with_capacity(5);
        let mut calls = [0u64; 3];
        let mut gpu = 0u64;
        let mut cpu = 0u64;
        let mut concurrent = 0u64;
        let mut negatives = 0usize;
        for (index, case) in CASES.into_iter().enumerate() {
            let p = Packet::new(source, case)?;
            let map = p.map(source.geometry())?;
            upload(&arena, b.ids[0], map.words())?;
            upload(&arena, b.ids[4], &map.probabilities().map(f32::to_bits))?;
            drop(map);
            let mut input = Zeroizing::new(vec![0u32; 8 * H]);
            for member in 0..3 {
                for i in 0..H {
                    input[member * H + i] = if case.sparse && i % 7 == 0 {
                        0f32.to_bits()
                    } else {
                        (((i * 17 + member * 29) % 127) as f32 / 127. - 0.5).to_bits()
                    };
                }
            }
            upload(&arena, b.ids[1], &input)?;
            drop(input);
            // One genuine all-GPU reference then its full GPU owning snapshots.
            let reference_started = Instant::now();
            complete_arm(source, &op, &arena, &b, &p, Arm::AllGpu, cancel)?;
            b.snapshot(&arena)?;
            audit(source, &arena, &b, &p, Arm::AllGpu)?;
            let reference_ns = reference_started.elapsed().as_nanos();
            if case.sparse {
                negatives += negative_controls(source, &arena, &b, &p, cancel)?;
            }
            let mut runs = Vec::with_capacity(15);
            let (mut partial_error, mut merge_error) = (0f32, 0f32);
            // First numeric triplet, one priming triplet, three rotated timed triplets.
            for iteration in 0..REPEATS + 2 {
                let order = match iteration.saturating_sub(2) % 3 {
                    0 => [Arm::AllGpu, Arm::Serial, Arm::Concurrent],
                    1 => [Arm::Concurrent, Arm::AllGpu, Arm::Serial],
                    _ => [Arm::Serial, Arm::Concurrent, Arm::AllGpu],
                };
                for arm in order {
                    println!(
                        "public_host_cpu_overlap_phase before_arm case={index} iteration={iteration} arm={}",
                        arm.name()
                    );
                    let whole = Instant::now();
                    let (ns, work, io) = complete_arm(source, &op, &arena, &b, &p, arm, cancel)?;
                    let checks = Instant::now();
                    let errors = audit(source, &arena, &b, &p, arm)?;
                    let checks_ns = checks.elapsed().as_nanos();
                    partial_error = partial_error.max(errors.0);
                    merge_error = merge_error.max(errors.1);
                    let ai = match arm {
                        Arm::AllGpu => 0,
                        Arm::Serial => 1,
                        Arm::Concurrent => 2,
                    };
                    calls[ai] += 1;
                    gpu += work.gpu_calls;
                    cpu += work.cpu_calls;
                    concurrent += u64::from(work.concurrent_joined);
                    runs.push(serde_json::json!({"arm":arm.name(),"iteration":iteration,"phase":if iteration==0{"numeric"}else if iteration==1{"priming"}else{"timed"},
                        "complete_joined_compute_upload_merge_io_cleanup_ns":ns,"full_numeric_owner_checks_ns":checks_ns,"complete_with_reference_checks_ns":whole.elapsed().as_nanos(),
                        "actual_gpu_partial_dispatches":work.gpu_calls,"actual_joined_cpu_batch_calls":work.cpu_calls,"both_consumers_scoped_joined":work.concurrent_joined,
                        "cpu_task_gpu_launcher_wall_intersection_ns":work.intersection_ns,"cpu_encoding_full_owner_bytes":work.copy_bytes,"arm_io_checkpoint_inspected_bytes":io}));
                }
            }
            reports.push(serde_json::json!({"members":3,"host_ranks_per_member":case.host,"disjoint_experts":case.disjoint,"sparse_ranks":case.sparse,
                "max_abs_partial_error":partial_error,"max_abs_merged_error":merge_error,"all_gpu_reference_and_snapshot_ns":reference_ns,"runs":runs}));
        }
        Ok(
            serde_json::json!({"cases":reports,"operator_main_bytes":arena.charged_bytes(),"main_cap_bytes":MAIN_CAP,"physical_workers":WORKERS,"worker_pool_setup_ns":pool_ns,
            "worker_input_phase_reserved_bytes":required+CALLER_CAP,"private_host_cap_bytes":HOST_CAP,"actual_arm_calls":calls,"actual_operator_gpu_partial_dispatches":gpu+50,"reference_all_gpu_calls":5,"reference_gpu_partial_dispatches":50,"numeric_priming_timed_gpu_partial_dispatches":gpu,"actual_joined_cpu_batch_calls":cpu,
            "actual_scoped_dual_consumer_joins":concurrent,"fixed_negative_controls_passed":negatives,"runtime_io_cap_bytes":128u64*1024*1024,"runtime_io_shared_with_original_target":true,"io_cleanup_inside_operator_arm_timings":true,
            "original_partial_abs_bound":ABS,"original_partial_rel_bound":REL,"original_merged_abs_bound":MERGED_ABS,"original_merged_rel_bound":MERGED_REL,
            "physical_acceptance_required":true,"whole_model_cpu_hybrid_accepted":false,"shared_experts_and_priming_condition_caches":true,"device_execution_overlap_timestamped":false}),
        )
    })();
    let close = arena.close();
    let audit = arena.owned_zero_audit_receipt();
    attempt.and_then(|mut result| {
        close?;
        let r = audit.ok_or("overlap complete MAIN zero audit missing")?;
        if r.buffers != 15 || r.bytes != OPERATOR_BYTES as u64 {
            return Err("overlap exact terminal MAIN owner receipt rejected".into());
        }
        result["operator_close"] = serde_json::json!({"buffers":r.buffers,"bytes":r.bytes});
        Ok(result)
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
        eprintln!("public_host_cpu_overlap_failure panic=true")
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
        return Err("attached CPU overlap original DAS extent rejected".into());
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.dtype().gguf_id())),
    )
    .map_err(|_| "attached CPU overlap original DAS inventory rejected")?;
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
    println!("public_host_cpu_overlap_phase before_config");
    let mut config = Config::new(client, limits, 8)?
        .reserve_target_rows()?
        .reserve_latest_target_checkpoint()?
        .reserve_public_frozen_host_sources()?
        .reserve_public_target_request_io()?
        .reserve_public_warm_replay()?;
    config = config.reserve_public_frozen_host_batch_router()?;
    executor.target_batch_router_reserved = true;
    executor.target_frozen_host_router_reserved = true;
    println!("public_host_cpu_overlap_phase before_constructor");
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
                return Err("attached CPU overlap original cold range rejected".into());
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
                    return Err("attached CPU overlap cold request scope did not activate".into());
                }
            }
        }
        let cold_request_scope = crate::target_lab_build::reuse_record(
            target.end_public_target_request_io(&mut executor, &cancel)?,
        );
        if target.public_target_request_io_active()? {
            return Err("attached CPU overlap cold request scope did not close".into());
        }
        println!("public_host_cpu_overlap_phase cold13_complete");
        let cold_ns = started.elapsed().as_nanos();
        let setup = Instant::now();
        let (plan, fills) = target.prepare_public_host_sources(&mut executor, &cancel, false)?;
        prepared
            .set(plan)
            .map_err(|_| "attached CPU overlap duplicate plan")?;
        let reset = target.reset_public_warm_replay(&mut executor, &cancel)?;
        let prepared = prepared.get().ok_or("attached CPU overlap plan missing")?;
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
        println!("public_host_cpu_overlap_phase attached");
        target.activate_public_frozen_host_batch_router(&executor, &cancel)?;
        let first =
            target.prefill_chunk_with_carry(&mut executor, &[expected.input_token(0)?], &cancel)?;
        expected.verify(0, first.logits(0)?, first.hidden(0)?)?;
        drop(first);
        if target.completed_target_rows_cursor(&executor)? != 1
            || target.public_target_request_io_active()?
        {
            return Err("attached CPU overlap actual warm boundary rejected".into());
        }
        println!("public_host_cpu_overlap_phase warm_passed");
        let setup_ns = setup.elapsed().as_nanos();
        let borrow_started = Instant::now();
        let source = target.borrow_public_host_cpu_layer(
            attachment
                .as_mut()
                .ok_or("attached CPU overlap owner absent")?,
            prepared,
            &reader,
            &executor,
            LAYER,
            &cancel,
        )?;
        println!("public_host_cpu_overlap_phase layer_borrowed");
        let mapping_ns = borrow_started.elapsed().as_nanos();
        let placements = source.placements()?.map(|p| serde_json::json!({"memory_type":p.memory_type,
            "heap":p.heap,"property_flags":p.property_flags,"heap_flags":p.heap_flags,"heap_bytes":p.heap_bytes}));
        println!("public_host_cpu_overlap_phase readonly_placements_checked");
        let mut value = cases(&source, client, retired.clone(), &cancel)?;
        println!("public_host_cpu_overlap_phase operators_complete");
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
        let audit = audit.ok_or("attached CPU overlap target zero audit missing")?;
        let owner = attachment
            .take()
            .ok_or("attached CPU overlap source owner missing")?;
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
        return Err("attached CPU overlap target retired".into());
    }
    file.verify_file_unchanged()?;
    value["target_close"] = serde_json::json!({"main_buffers":main.buffers,"main_bytes":main.bytes,
        "source_buffers":sources.buffers,"source_bytes":sources.bytes});
    value["attached_source_close"] = serde_json::json!({"registered_pages":close.registered_pages,
        "detached_pages":close.detached_pages,"physical_allocated_bytes":close.mirror.total_allocated_bytes,
        "physical_freed_bytes":close.mirror.total_freed_bytes,"live_backing_bytes":close.mirror.live_backing_bytes});
    value["scope"] = "fixed-public-original-three-row-real-source-full-rank-overlap".into();
    value["layer"] = LAYER.into();
    value["mtp_executed"] = false.into();
    value["whole_model_cpu_hybrid_accepted"] = false.into();
    value["gpu_control_math"] = "original-F32-all-five-GU-DOWN-and-rank-reduce".into();
    value["best_integer_gpu_control_compared"] = false.into();
    serde_json::to_writer_pretty(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(input.receipt)?,
        &value,
    )?;
    println!("public_host_cpu_overlap_complete partials_accepted=true owners_erased=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_registration_rejects_other_modes_cards_and_policy_inheritance() {
        let args = ["lab", "intel", "public-host-cpu-overlap-probe"].map(OsString::from);
        let env = [
            ("KUNO_LAB_HOST_CPU_OVERLAP_PUBLIC_ONLY", "1"),
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
            ("KUNO_TEST_Q3_DIR", "model"),
            ("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "reference"),
            ("KUNO_PUBLIC_RECEIPT", "receipt"),
        ]
        .map(|(k, v)| (k.into(), v.into()));
        assert!(registration(&args, env.clone(), crate::target_lab_build::Build::Inline).is_ok());
        for option in [
            "KUNO_LAB_TARGET_LOOKAHEAD",
            "KUNO_LAB_FROZEN_GU_MATH",
            "KUNO_LAB_LANGUAGE_DECODE",
            "KUNO_TEST_OTHER",
        ] {
            assert!(
                registration(
                    &args,
                    env.clone().into_iter().chain([(option.into(), "1".into())]),
                    crate::target_lab_build::Build::Inline
                )
                .is_err()
            );
        }
        assert!(
            registration(&args, env.clone(), crate::target_lab_build::Build::Threaded).is_err()
        );
        let bad = ["lab", "nvidia", "public-host-cpu-overlap-probe"].map(OsString::from);
        assert!(registration(&bad, env.clone(), crate::target_lab_build::Build::Inline).is_err());
        assert!(
            registration(
                &args,
                env.into_iter()
                    .filter(|(k, _)| k != "KUNO_LAB_STATIC_CACHE_BYTES"),
                crate::target_lab_build::Build::Inline
            )
            .is_err()
        );
    }
    #[test]
    fn all_actual_owner_charges_fit_original_caps_before_private_fill() {
        let sizes = WORDS.map(|n| {
            crate::private_arena::BufferLayout::new(n)
                .unwrap()
                .owned_bytes
        });
        assert_eq!(
            sizes,
            [
                32768, 131072, 262144, 1048576, 32768, 131072, 32768, 524288, 131072, 32768, 32768,
                262144, 1048576, 131072, 32768
            ]
        );
        assert_eq!(sizes.iter().sum::<usize>(), OPERATOR_BYTES);
        assert_eq!(MAIN_CAP - OPERATOR_BYTES, 327680);
        const { assert!(2 * 131072 + HEADER_CAP < HOST_CAP) };
        const { assert!(PARTIAL_WORDS * 4 + 524288 + HEADER_CAP < HOST_CAP) };
        assert_eq!(CASES.len(), 5);
        assert!(
            CASES
                .iter()
                .all(|c| fixed_host_mask(c.host, c.sparse).unwrap().count_ones() == c.host as u32)
        );
    }
    #[test]
    fn row_inspection_and_original_partial_bounds_reject_bad_data() {
        assert!(compare(1., 1.0001).is_ok());
        assert!(compare(0., 0.0002).is_err());
        assert!(compare(0., f32::NAN).is_err());
        #[repr(align(4))]
        struct Aligned([u8; 12]);
        let bytes = Aligned([0; 12]);
        assert!(aligned_floats(&bytes.0).is_ok());
        assert!(aligned_words(&bytes.0[1..9]).is_err());
        assert_eq!(fixed_host_mask(3, true).unwrap(), 529);
        assert!(fixed_host_mask(4, false).is_err());
    }
    #[test]
    fn fixed_schedule_uses_real_three_members_and_no_four_permission() {
        let text = include_str!("public_host_cpu_overlap_probe.rs");
        let body = text.split("#[cfg(test)]").next().unwrap();
        let body: String = body.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(body.contains("scope.spawn(||"));
        assert!(body.contains("letgpu=source.launch_resident_original_gpu_partials"));
        assert!(
            body.find("letgpu=source.launch_resident_original_gpu_partials")
                .unwrap()
                < body.find("letjoined=task.join()").unwrap()
        );
        assert!(!body.contains("reserve_public_four_rows"));
        assert!(!body.contains("unsafeimplSend"));
        assert!(body.contains("sanitize_transient_io_audited"));
        assert_eq!(COUNT, kuno_text::cpu_expert::batch::MAX_MEMBERS);
    }
}
