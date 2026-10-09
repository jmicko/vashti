//! Fixed public standalone operator only; parent creates the exclusive device
//! with required_memory_configuration and registers an exact public-only mode.
//! No models, prompts, device creation or environment mutation here. Full
//! source, MAX8 gaps, consumed maps, deferred wire and sticky faults are checked
//! before the original full-owner reset/zero/allocator cleanup acceptance.
use crate::{
    main_admission::MainAdmission,
    private_arena::{BufferId, PrivateArena},
    resident_batch_router_gpu as k,
};
use cubecl::server::Handle;
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type State = [Zeroizing<Vec<u32>>; 6];
const WORDS: [usize; 6] = [8192, 32768, 8192, 8192, 524288, 8192];
const MAIN_CAP: u64 = 4 * 1024 * 1024;
const HOST_CAP: u64 = 16 * 1024 * 1024;
const IO_CAP: u64 = 64 * 1024 * 1024;
const FAULT_CASES: usize = 16;
const FROZEN_FAULT_KINDS: usize = 17;
const FROZEN_FAULT_CASES: usize = FROZEN_FAULT_KINDS + 6;
const FROZEN_NUMERIC_CASES: usize = 28; // 18 patterns + 9 ties + fresh epoch.
const TOTAL_CASES: usize = 13 + FAULT_CASES + FROZEN_NUMERIC_CASES + FROZEN_FAULT_CASES;
const VIRTUAL_BOUNDARIES: [u32; 8] = [0, 143, 144, 287, 288, 431, 432, 511];

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceFixture {
    Resident144,
    Frozen512 { pattern: usize },
}
impl SourceFixture {
    fn capacity(self) -> usize {
        match self {
            Self::Resident144 => 144,
            Self::Frozen512 { .. } => 512,
        }
    }
    fn fault_kinds(self) -> usize {
        match self {
            Self::Resident144 => FAULT_CASES,
            Self::Frozen512 { .. } => FROZEN_FAULT_KINDS,
        }
    }
}
/// Compiled-public coordinates only: no model, scores, selected IDs or maps.
fn case_marker(source: SourceFixture, layer: usize, count: usize, kind: Option<usize>) -> [i64; 6] {
    [
        source.capacity() as i64,
        layer as i64,
        count as i64,
        match source {
            SourceFixture::Resident144 => -1,
            SourceFixture::Frozen512 { pattern } => pattern as i64,
        },
        kind.filter(|&n| n < source.fault_kinds())
            .map_or(-1, |n| n as i64),
        kind.filter(|&n| n >= source.fault_kinds())
            .map_or(-1, |n| (n - source.fault_kinds()) as i64),
    ]
}
/// The caller has already attempted full arena close before passing here.
/// Case errors are authored content-free classifications in this public-only
/// probe. Preserve their exact category even if cleanup also fails; cleanup
/// status is separately reported and no branch can publish a success receipt.
/// Arbitrary panic payloads are never returned or printed.
fn after_full_arena_close<T>(
    attempt: std::thread::Result<Result<T>>,
    close_ok: bool,
    retired: bool,
) -> Result<T> {
    match attempt {
        Ok(Err(error)) => Err(error),
        Err(_) => Err("batch router public case panic; no accepted receipt".into()),
        Ok(Ok(_)) if !close_ok => {
            Err("batch router full arena close failed; no accepted receipt".into())
        }
        Ok(Ok(_)) if retired => Err("batch router arena retired; no accepted receipt".into()),
        Ok(Ok(report)) => Ok(report),
    }
}
fn state() -> State {
    WORDS.map(|n| Zeroizing::new(vec![0; n]))
}
fn scores(row: usize) -> [f32; 512] {
    let mut out = std::array::from_fn(|id| -4. - id as f32 / 1024.);
    for rank in 0..12 {
        out[(rank * 7 + row * 14) % 144] = 8. - rank as f32 / 8.;
    }
    out
}
fn fixture(count: usize) -> State {
    let mut out = state();
    out[1][..48 * 512].fill(u32::MAX);
    for layer in 0..48 {
        for id in 0..144 {
            out[1][layer * 512 + id] = ((id * 31 + 7) % 144) as u32;
        }
    }
    for row in 0..count {
        for (id, value) in scores(row).iter().enumerate() {
            out[0][row * 512 + id] = value.to_bits();
        }
    }
    out
}
fn frozen_geometry(layer: usize) -> Result<crate::quant_matvec::grouped_moe::Geometry> {
    use crate::quant_matvec::grouped_moe::{Geometry, Projection};
    let [gate, up, down] = crate::daslab_profile::routed_types(layer)
        .ok_or("batch router fixed DAS layer geometry")?;
    Ok(Geometry::new(
        [
            Projection::new(gate, 2560, 640, 512)?,
            Projection::new(up, 2560, 640, 512)?,
            Projection::new(down, 640, 2560, 512)?,
        ],
        8,
        144,
    )?
    .with_frozen_host_sources()?)
}
fn frozen_fixture(count: usize, pattern: usize) -> Result<State> {
    if !(1..=3).contains(&count) || pattern > 1 {
        return Err("batch router fixed frozen fixture geometry".into());
    }
    let mut out = state();
    // Full original512 source addressing, not a larger VRAM BankSlots owner.
    // Each row is a bijection; the first ten selected IDs span all five pages.
    for layer in 0..48 {
        let table = &mut out[1][layer * 512..(layer + 1) * 512];
        for (id, slot) in table.iter_mut().enumerate() {
            *slot = id as u32;
        }
        for (id, slot) in VIRTUAL_BOUNDARIES.into_iter().chain([71, 72]).enumerate() {
            let old = table
                .iter()
                .position(|&x| x == slot)
                .ok_or("batch router fixed frozen bijection")?;
            table.swap(id, old);
        }
    }
    for row in 0..count {
        for id in 0..512 {
            out[0][row * 512 + id] = (-4. - id as f32 / 1024.).to_bits();
        }
        for rank in 0..12 {
            // Pattern0 repeats experts in different rank orders; pattern1
            // creates a thirty-expert union without changing the map format.
            let id = if pattern == 0 {
                (rank + row * 3) % 12
            } else {
                row * 12 + rank
            };
            out[0][row * 512 + id] = (8. - rank as f32 / 8.).to_bits();
        }
    }
    k::validate_frozen_source_table(&out[1])?;
    Ok(out)
}
fn frozen_tie(before: &mut State, count: usize, tie: usize) -> Result<()> {
    for row in 0..count {
        let scores = &mut before[0][row * 512..(row + 1) * 512];
        scores.fill((-4f32).to_bits());
        for (rank, score) in scores.iter_mut().take(12).enumerate() {
            *score = (8. - rank as f32 / 8.).to_bits();
        }
        match tie {
            0 => scores[1] = scores[0],
            1 => scores[10] = scores[9],
            2 => {
                for (rank, score) in scores.iter_mut().take(12).enumerate() {
                    *score = (-(rank as f32) / 8.).to_bits();
                }
                scores[0] = 0f32.to_bits();
                scores[1] = (-0f32).to_bits();
            }
            _ => return Err("batch router fixed frozen tie inventory".into()),
        }
    }
    Ok(())
}
fn bytes(words: &[u32]) -> Zeroizing<Vec<u8>> {
    let mut out = Zeroizing::new(Vec::with_capacity(words.len() * 4));
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}
struct Read(cubecl::bytes::Bytes);
impl Drop for Read {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn io_audit(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let r = cubecl::future::block_on(client.sanitize_transient_io_audited())
        .map_err(|_| "batch router complete I/O audit failed")?;
    let total = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("batch router I/O overflow")?;
    if total == 0 || total > IO_CAP || total != r.inspected_bytes || r.inspected_buffers == 0 {
        return Err("batch router incomplete owning I/O audit".into());
    }
    Ok(())
}
fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    Ok(client
        .memory_usage()
        .map_err(|_| "batch router main telemetry failed")?
        .bytes_reserved)
}
fn physical(arena: &PrivateArena, ids: &[BufferId; 6]) -> Result<()> {
    let client = arena.client()?;
    let handles = ids
        .iter()
        .map(|&id| arena.handle(id).cloned())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    k::validate_owners(
        &handles
            .clone()
            .try_into()
            .map_err(|_| "batch router owner count")?,
        0,
        0,
        1,
    )?;
    let resources = handles
        .iter()
        .map(|h| {
            client
                .get_resource(h.clone())
                .map_err(|_| "batch router physical lookup failed")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (i, r) in resources.iter().enumerate() {
        let r = r.resource();
        if r.offset != 0
            || r.size != k::OWNED_BYTES[i]
            || r.buffer.size() != r.size
            || resources[..i]
                .iter()
                .any(|old| old.resource().buffer == r.buffer)
        {
            return Err("batch router complete owners alias or escape pages".into());
        }
    }
    Ok(())
}
fn upload(
    arena: &mut PrivateArena,
    before: &State,
    ledger: &mut MainAdmission,
) -> Result<[BufferId; 6]> {
    ledger.observe_before(sample(arena.client()?)?)?;
    ledger.precharge(&k::OWNED_BYTES)?;
    let ids = before
        .iter()
        .map(|words| arena.upload_u32(words))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "batch router owning allocation count")?;
    if arena.charged_bytes() as u64 != k::MAIN_BYTES {
        return Err("batch router actual owning charge".into());
    }
    physical(arena, &ids)?;
    ledger.observe_after(sample(arena.client()?)?)?;
    Ok(ids)
}
fn snapshot(arena: &PrivateArena, ids: &[BufferId; 6]) -> Result<State> {
    let mut actual = state();
    for (i, &id) in ids.iter().enumerate() {
        let read = Read(
            arena
                .client()?
                .read_one(arena.handle(id)?.clone())
                .map_err(|_| "batch router owning read failed")?,
        );
        if read.0.len() != WORDS[i] * 4 {
            return Err("batch router owning read extent".into());
        }
        for (dst, bytes) in actual[i].iter_mut().zip(read.0.chunks_exact(4)) {
            *dst = u32::from_le_bytes(bytes.try_into().unwrap());
        }
        drop(read);
        io_audit(arena.client()?)?;
    }
    Ok(actual)
}
fn mutate(before: &mut State, kind: usize, layer: usize, count: usize) -> Result<u32> {
    let expected = match kind {
        0 => {
            before[1][layer * 512] = u32::MAX;
            k::FAULT_MISSING
        }
        1 => {
            before[0][0] = f32::NAN.to_bits();
            k::FAULT_SHAPE
        }
        2 => {
            before[0][count * 512] = 1;
            k::FAULT_SHAPE
        }
        3 => {
            before[0][8191] = 1;
            k::FAULT_SHAPE
        }
        4 => {
            before[1][48 * 512] = 1;
            k::FAULT_SHAPE
        }
        5 => {
            before[2][803] = 1;
            k::FAULT_SHAPE
        }
        6 => {
            before[3][8191] = 1;
            k::FAULT_SHAPE
        }
        7 => {
            before[4][layer * k::FRAME_WORDS + 17] = 1;
            k::FAULT_SHAPE
        }
        8 => {
            before[4][k::AUDIT_WORDS] = 1;
            k::FAULT_SHAPE
        }
        9 => {
            before[5][8191] = 2;
            k::FAULT_SHAPE
        }
        10 => {
            before[5][0] = 32;
            32 | k::FAULT_SHAPE
        }
        11 => {
            before[0][0] = f32::MAX.to_bits();
            k::FAULT_NUMERIC
        }
        12 => {
            before[0][..512].fill((-f32::MAX).to_bits());
            before[0][0] = f32::MAX.to_bits();
            k::FAULT_NUMERIC
        }
        13 => {
            before[1][layer * 512 + 7] = before[1][layer * 512];
            k::FAULT_MISSING
        }
        14 => {
            before[1][layer * 512] = 144;
            k::FAULT_SHAPE
        }
        15 => {
            before[0][512] = f32::INFINITY.to_bits();
            k::FAULT_SHAPE
        }
        _ => return Err("batch router fixed fault inventory".into()),
    };
    Ok(expected)
}
fn mutate_frozen(before: &mut State, kind: usize, layer: usize, count: usize) -> Result<u32> {
    match kind {
        0 => {
            before[1][layer * 512] = u32::MAX;
            // Unlike partial144, a frozen original512 source has no sentinel.
            Ok(k::FAULT_SHAPE)
        }
        13 => {
            before[1][layer * 512 + 1] = before[1][layer * 512];
            Ok(k::FAULT_MISSING)
        }
        14 => {
            before[1][layer * 512] = 512;
            Ok(k::FAULT_SHAPE)
        }
        16 => {
            // Even an unselected entry in a different layer is range-checked.
            before[1][((layer + 1) % 48) * 512 + 511] = 512;
            Ok(k::FAULT_SHAPE)
        }
        _ => mutate(before, kind, layer, count),
    }
}
fn require_late_exact_id_rejection(
    wire: &[u8],
    table: &[u32],
    layer: usize,
    start: u64,
    count: usize,
) -> Result<()> {
    // Equal probabilities and equal score magnitudes cannot authorize changing
    // actual expert IDs. Copy the full selected triples so rejection is the
    // exact original-ID gate, rather than a malformed probability field.
    if wire.len() != k::AUDIT_OWNED_WORDS * 4 || layer >= 48 || !(1..=3).contains(&count) {
        return Err("batch router fixed late-rejection geometry".into());
    }
    let mut bad = Zeroizing::new(wire.to_vec());
    let select = layer * k::FRAME_WORDS + k::ROUTER_WORDS + 4;
    if wire[(select + 1) * 4..(select + 2) * 4] != wire[(select + 4) * 4..(select + 5) * 4] {
        return Err("batch router fixed equal-probability rejection fixture".into());
    }
    for byte in 0..12 {
        bad.swap(select * 4 + byte, (select + 3) * 4 + byte);
    }
    if !matches!(
        k::audit_frame_frozen_sources(&bad, table, layer, start, count, frozen_geometry(layer)?),
        Err(kuno_text::Error::InvalidFormat(
            "batch device router original IDs/selection"
        ))
    ) {
        return Err("batch router late exact original ID audit accepted".into());
    }
    Ok(())
}
fn case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    ledger: &mut MainAdmission,
    layer: usize,
    count: usize,
    kind: Option<usize>,
    source: SourceFixture,
) -> Result<serde_json::Value> {
    let [capacity, layer_marker, count_marker, pattern, fault, tie] =
        case_marker(source, layer, count, kind);
    eprintln!(
        "batch_router_public_case source_capacity={capacity} layer={layer_marker} count={count_marker} pattern={pattern} fault={fault} tie={tie}"
    );
    case_body(client, arena, ledger, layer, count, kind, source)
}
fn case_body(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    ledger: &mut MainAdmission,
    layer: usize,
    count: usize,
    kind: Option<usize>,
    source: SourceFixture,
) -> Result<serde_json::Value> {
    let start = 4096 - count as u64;
    let mut before = match source {
        SourceFixture::Resident144 => fixture(count),
        SourceFixture::Frozen512 { pattern } => frozen_fixture(count, pattern)?,
    };
    let fault_kinds = source.fault_kinds();
    if let Some(tie) = kind.filter(|&n| n >= fault_kinds) {
        if matches!(source, SourceFixture::Frozen512 { .. }) {
            frozen_tie(&mut before, count, tie - fault_kinds)?;
        } else {
            match tie - fault_kinds {
                0 => before[0][7] = before[0][0],
                1 => before[0][70] = before[0][63],
                2 => {
                    before[0][..512].fill((-4f32).to_bits());
                    for rank in 0..12 {
                        before[0][rank * 7] = (-(rank as f32) / 8.).to_bits();
                    }
                    before[0][0] = 0f32.to_bits();
                    before[0][7] = (-0f32).to_bits();
                }
                _ => return Err("batch router fixed tie inventory".into()),
            }
        }
    }
    let expected_fault = kind
        .filter(|&n| n < fault_kinds)
        .map(|n| match source {
            SourceFixture::Resident144 => mutate(&mut before, n, layer, count),
            SourceFixture::Frozen512 { .. } => mutate_frozen(&mut before, n, layer, count),
        })
        .transpose()?
        .unwrap_or(0);
    let ids = upload(arena, &before, ledger)?;
    let handles: [Handle; 6] = ids
        .iter()
        .map(|&id| arena.handle(id).cloned())
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "batch router six owner handles")?;
    match source {
        SourceFixture::Resident144 => k::launch(client, handles.clone(), layer, start, count)?,
        SourceFixture::Frozen512 { .. } => {
            k::launch_frozen_sources(client, handles.clone(), layer, start, count)?
        }
    }
    let actual = snapshot(arena, &ids)?;
    ledger.completed_fence(sample(client)?)?;
    let mut max_error = 0.;
    let mut mismatch = 0;
    let mut groups = 0;
    let mut late_exact_id_rejections = 0;
    let mut late_unselected_duplicate_rejections = 0;
    if kind.is_some_and(|n| n < fault_kinds) {
        before[5][0] = expected_fault;
        if actual != before {
            return Err("batch router fault published or altered full owner".into());
        }
    } else {
        for i in [0, 1, 5] {
            if actual[i] != before[i] {
                return Err("batch router original source/table/fault changed".into());
            }
        }
        let wire = bytes(&actual[4]);
        let routes = match source {
            SourceFixture::Resident144 => k::audit_frame(&wire, &before[1], layer, start, count)?,
            SourceFixture::Frozen512 { .. } => k::audit_frame_frozen_sources(
                &wire,
                &before[1],
                layer,
                start,
                count,
                frozen_geometry(layer)?,
            )?,
        };
        groups = routes.groups();
        mismatch = routes.probability_bit_mismatches();
        max_error = routes.max_probability_abs_error();
        if matches!(source, SourceFixture::Frozen512 { .. }) {
            if kind.is_none()
                && !VIRTUAL_BOUNDARIES
                    .iter()
                    .all(|slot| routes.pairs().iter().any(|&(_, found)| found == *slot))
            {
                return Err("batch router virtual source boundary coverage absent".into());
            }
            // The constructor/deferred gate authenticates the WHOLE bijection;
            // this is not claimed as GPU duplicate scanning/no publication.
            let mut bad_table = before[1].clone();
            let other = ((layer + 1) % 48) * 512;
            bad_table[other + 510] = bad_table[other + 511];
            if k::audit_frame_frozen_sources(
                &wire,
                &bad_table,
                layer,
                start,
                count,
                frozen_geometry(layer)?,
            )
            .is_ok()
            {
                return Err("batch router unselected duplicate deferred audit accepted".into());
            }
            late_unselected_duplicate_rejections = 1;
            if kind == Some(fault_kinds + 2) {
                require_late_exact_id_rejection(&wire, &before[1], layer, start, count)?;
                late_exact_id_rejections = 1;
            }
        }
        let frame = layer * k::FRAME_WORDS;
        let map = frame + k::ROUTER_WORDS + k::SELECTION_WORDS;
        let prob = map + k::MAP_WORDS;
        if actual[2][..k::MAP_WORDS] != actual[4][map..map + k::MAP_WORDS]
            || actual[2][k::MAP_WORDS..].iter().any(|&x| x != 0)
            || actual[3][..k::PROB_WORDS] != actual[4][prob..prob + k::PROB_WORDS]
            || actual[3][k::PROB_WORDS..].iter().any(|&x| x != 0)
            || actual[4][..frame]
                .iter()
                .chain(actual[4][frame + k::FRAME_WORDS..].iter())
                .any(|&x| x != 0)
        {
            return Err("batch router consumed headers or complete audit padding changed".into());
        }
    }
    drop(actual);
    drop(before);
    arena.reset()?;
    if ids.iter().any(|&id| arena.handle(id).is_ok()) {
        return Err("batch router stale reset owner accepted".into());
    }
    for h in &handles {
        let read = Read(
            client
                .read_one(h.clone())
                .map_err(|_| "batch router erased owner read failed")?,
        );
        if read.0.len() as u64 != h.size() || read.0.iter().any(|&x| x != 0) {
            return Err("batch router full owner zero proof failed".into());
        }
        drop(read);
        io_audit(client)?;
    }
    drop(handles);
    client.memory_cleanup();
    cubecl::future::block_on(client.sync()).map_err(|_| "batch router reset completion failed")?;
    ledger.completed_fence(sample(client)?)?;
    Ok(
        serde_json::json!({"layer":layer,"count":count,"source_capacity":source.capacity(),"frozen_route_pattern":match source { SourceFixture::Resident144=>None, SourceFixture::Frozen512{pattern}=>Some(pattern)},"fault_case":kind.filter(|&n|n<fault_kinds),"exact_tie_case":kind.filter(|&n|n>=fault_kinds).map(|n|n-fault_kinds),"sticky_fault":expected_fault,
        "groups":groups,"probability_bit_mismatches":mismatch,"max_original_probability_abs_error":max_error,
        "late_exact_original_id_rejections":late_exact_id_rejections,"late_unselected_duplicate_rejections":late_unselected_duplicate_rejections,
        "full_source_and_guards_preserved":true,"complete_erasure":true}),
    )
}
/// Parent must reject inherited lab/private/model settings and CPU feature builds
/// before constructing its exclusive context. Numeric JSON returned after all
/// normal/fault/cleanup gates; root owns receipt publication/physical recovery.
pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
) -> Result<serde_json::Value> {
    use crate::target_lab_build::Build;
    let inline = match Build::compiled() {
        Build::Threaded => false,
        Build::Inline => true,
        Build::Rejected => return Err("batch router rejects CPU arithmetic feature builds".into()),
    };
    crate::target_lab_build::validate_runtime(inline)?;
    k::validate_limits(limits)?;
    // Full source/current snapshots, readback or late copied audit, one table
    // corruption copy and bounded eighty-case numeric JSON metadata. No model
    // or host-weight payload is introduced by virtual source addressing.
    crate::main_admission::check_capacity(
        HOST_CAP,
        &[3 * k::MAIN_BYTES, 2 * 1024 * 1024, 1024 * 1024],
    )?;
    let baseline = client
        .memory_usage()
        .map_err(|_| "batch router fresh telemetry failed")?;
    if baseline.number_allocs != 0
        || baseline.bytes_in_use != 0
        || baseline.bytes_padding != 0
        || baseline.bytes_reserved != 0
    {
        return Err("batch router fresh exclusive allocator required".into());
    }
    client
        .configure_transient_io(IO_CAP)
        .map_err(|_| "batch router bounded I/O setup failed")?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena = PrivateArena::new_transient_on_configured(
        client.clone(),
        MAIN_CAP as usize,
        retired.clone(),
    )?;
    arena.enable_public_owned_zero_audit()?;
    let mut ledger = MainAdmission::new(MAIN_CAP, 0)?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut report = Vec::with_capacity(TOTAL_CASES);
        for layer in [0, 23, 47] {
            for count in 1..=3 {
                report.push(case(
                    client,
                    &mut arena,
                    &mut ledger,
                    layer,
                    count,
                    None,
                    SourceFixture::Resident144,
                )?);
            }
        }
        for fault in 0..FAULT_CASES {
            report.push(case(
                client,
                &mut arena,
                &mut ledger,
                47,
                3,
                Some(fault),
                SourceFixture::Resident144,
            )?);
        }
        for tie in 0..3 {
            report.push(case(
                client,
                &mut arena,
                &mut ledger,
                47,
                3,
                Some(FAULT_CASES + tie),
                SourceFixture::Resident144,
            )?);
        }
        for layer in [0, 23, 47] {
            for count in 1..=3 {
                for pattern in 0..2 {
                    report.push(case(
                        client,
                        &mut arena,
                        &mut ledger,
                        layer,
                        count,
                        None,
                        SourceFixture::Frozen512 { pattern },
                    )?);
                }
            }
        }
        for count in 1..=3 {
            for tie in 0..3 {
                report.push(case(
                    client,
                    &mut arena,
                    &mut ledger,
                    47,
                    count,
                    Some(FROZEN_FAULT_KINDS + tie),
                    SourceFixture::Frozen512 { pattern: 0 },
                )?);
            }
        }
        for fault in 0..FROZEN_FAULT_KINDS {
            report.push(case(
                client,
                &mut arena,
                &mut ledger,
                47,
                3,
                Some(fault),
                SourceFixture::Frozen512 { pattern: 0 },
            )?);
        }
        for count in 1..=2 {
            for fault in [10, 13, 14] {
                report.push(case(
                    client,
                    &mut arena,
                    &mut ledger,
                    0,
                    count,
                    Some(fault),
                    SourceFixture::Frozen512 { pattern: 0 },
                )?);
            }
        }
        // Return to the original addressing specialization on this same
        // client to exercise distinct kernel/pipeline identities, then replay
        // the frozen specialization on another freshly wiped arena epoch.
        report.push(case(
            client,
            &mut arena,
            &mut ledger,
            0,
            3,
            None,
            SourceFixture::Resident144,
        )?); // Original fresh epoch replay.
        report.push(case(
            client,
            &mut arena,
            &mut ledger,
            0,
            3,
            None,
            SourceFixture::Frozen512 { pattern: 1 },
        )?);
        if report.len() != TOTAL_CASES {
            return Err("batch router fixed case count".into());
        }
        Ok(report)
    }));
    let close = arena.close();
    let zero = arena.owned_zero_audit_receipt();
    let retired = retired.load(Ordering::Acquire);
    let outcome = match &attempt {
        Ok(Ok(_)) => 0,
        Ok(Err(_)) => 1,
        Err(_) => 2,
    };
    if outcome != 0 || close.is_err() || retired {
        eprintln!(
            "batch_router_public_cleanup outcome={outcome} full_arena_close={} retired={}",
            u8::from(close.is_ok()),
            u8::from(retired)
        );
    }
    let report = after_full_arena_close(attempt, close.is_ok(), retired)?;
    let zero = zero.ok_or("batch router complete zero receipt absent")?;
    drop(arena);
    let mut released = false;
    for _ in 0..2 {
        cubecl::future::block_on(client.sync())
            .map_err(|_| "batch router final completion failed")?;
        client.memory_cleanup();
        cubecl::future::block_on(client.sync())
            .map_err(|_| "batch router allocator completion failed")?;
        let m = client
            .memory_usage()
            .map_err(|_| "batch router allocator telemetry failed")?;
        if m.number_allocs == 0
            && m.bytes_in_use == 0
            && m.bytes_padding == 0
            && m.bytes_reserved == 0
        {
            released = true;
            break;
        }
    }
    if !released {
        return Err("batch router allocator cleanup unaccepted".into());
    }
    Ok(
        serde_json::json!({"scope":"fixed-public-experimental-F32-batch-router-only","cases":report,
        "numeric_cases":13+FROZEN_NUMERIC_CASES,"exact_tie_cases":12,"near_ties_require_deferred_exact_original_id_audit":true,"fault_cases":FAULT_CASES+FROZEN_FAULT_CASES,
        "resident144_numeric_cases":13,"resident144_exact_tie_cases":3,"resident144_fault_cases":FAULT_CASES,
        "frozen512_numeric_cases":FROZEN_NUMERIC_CASES,"frozen512_exact_tie_cases":9,"frozen512_fault_cases":FROZEN_FAULT_CASES,
        "frozen512_late_exact_original_id_rejections":3,"frozen512_late_unselected_duplicate_rejections":FROZEN_NUMERIC_CASES,
        "virtual_source_boundary_slots":VIRTUAL_BOUNDARIES,"full512_is_source_addressing_not_vram_residency":true,
        "host_weight_memory_or_math_accepted":false,
        "max_count":3,"main_bytes":k::MAIN_BYTES,"inline_dispatch":inline,
        "main_cap_bytes":MAIN_CAP,"host_cap_bytes":HOST_CAP,"io_cap_bytes":IO_CAP,
        "info_bytes_per_layer":k::INFO_BYTES_PER_LAYER,"info_bytes_per_chunk":k::INFO_BYTES_PER_CHUNK,
        "near_tie_margin":k::NEAR_TIE_MARGIN,"probability_absolute_bound":k::PROB_ABS,"probability_relative_bound":k::PROB_REL,
        "original_ID_order_exact":true,"original_probability_bits_claimed":false,
        "owned_zero_audit_buffers":zero.buffers,"owned_zero_audit_bytes":zero.bytes,"allocator_cleanup_complete":true,
        "whole_owning_erasure_accepted":false,"private_request_accepted":false,"whole_model_accepted":false,"performance_accepted":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_case_markers_distinguish_modes_normal_fault_and_tie_without_route_content() {
        assert_eq!(
            case_marker(SourceFixture::Resident144, 0, 1, None),
            [144, 0, 1, -1, -1, -1]
        );
        assert_eq!(
            case_marker(SourceFixture::Resident144, 47, 3, Some(13)),
            [144, 47, 3, -1, 13, -1]
        );
        assert_eq!(
            case_marker(SourceFixture::Resident144, 47, 3, Some(FAULT_CASES + 2)),
            [144, 47, 3, -1, -1, 2]
        );
        assert_eq!(
            case_marker(SourceFixture::Frozen512 { pattern: 1 }, 23, 2, None),
            [512, 23, 2, 1, -1, -1]
        );
        assert_eq!(
            case_marker(SourceFixture::Frozen512 { pattern: 0 }, 47, 3, Some(0)),
            [512, 47, 3, 0, 0, -1]
        );
        assert_eq!(
            case_marker(
                SourceFixture::Frozen512 { pattern: 0 },
                47,
                1,
                Some(FROZEN_FAULT_KINDS + 2)
            ),
            [512, 47, 1, 0, -1, 2]
        );
    }
    #[test]
    fn attempted_full_close_preserves_case_error_and_cannot_promote_cleanup_or_panic() {
        use crate::main_admission::AdmissionError;
        for close_ok in [false, true] {
            for retired in [false, true] {
                let case: Result<()> = Err(Box::new(AdmissionError::LimitExceeded));
                let error = after_full_arena_close(Ok(case), close_ok, retired).unwrap_err();
                assert_eq!(
                    error.downcast_ref::<AdmissionError>(),
                    Some(&AdmissionError::LimitExceeded)
                );
                let report = after_full_arena_close(Ok(Ok(73u32)), close_ok, retired);
                if close_ok && !retired {
                    assert_eq!(report.unwrap(), 73);
                } else {
                    assert!(report.is_err());
                }
                let payload: Box<dyn std::any::Any + Send> =
                    Box::new("private arbitrary panic payload");
                let error =
                    after_full_arena_close::<()>(Err(payload), close_ok, retired).unwrap_err();
                assert_eq!(
                    error.to_string(),
                    "batch router public case panic; no accepted receipt"
                );
            }
        }
    }
    fn host_frame(count: usize, layer: usize, start: u64) -> Result<State> {
        host_frame_source(count, layer, start, SourceFixture::Resident144, None)
    }
    fn host_frame_source(
        count: usize,
        layer: usize,
        start: u64,
        source: SourceFixture,
        tie: Option<usize>,
    ) -> Result<State> {
        let mut out = match source {
            SourceFixture::Resident144 => fixture(count),
            SourceFixture::Frozen512 { pattern } => frozen_fixture(count, pattern)?,
        };
        if let Some(tie) = tie {
            frozen_tie(&mut out, count, tie)?;
        }
        let mut ids = Zeroizing::new([0u32; 80]);
        let mut probs = Zeroizing::new([0f32; 80]);
        let mut pairs = Zeroizing::new(Vec::with_capacity(30));
        let mut gap = f32::MAX;
        let frame = layer * k::FRAME_WORDS;
        let select = frame + k::ROUTER_WORDS;
        let map = select + k::SELECTION_WORDS;
        let prob = map + k::MAP_WORDS;
        let meta = prob + k::PROB_WORDS;
        for row in 0..count {
            let logits = Zeroizing::new(std::array::from_fn::<_, 512, _>(|id| {
                f32::from_bits(out[0][row * 512 + id])
            }));
            let original = kuno_text::ops::router_topk_guarded(&*logits, 10, true)?;
            let mut order = std::array::from_fn::<_, 512, _>(|id| id);
            order.sort_unstable_by(|&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b))
            });
            out[4][meta + 8 + row * 2] = order[10] as u32;
            out[4][meta + 9 + row * 2] = logits[order[10]].to_bits();
            for (rank, &(id, p)) in original.as_slice().iter().enumerate() {
                let route = row * 10 + rank;
                ids[route] = id;
                probs[route] = p;
                if !pairs.iter().any(|&(old, _)| old == id) {
                    pairs.push((id, out[1][layer * 512 + id as usize]));
                }
                out[4][select + 4 + route * 3] = id;
                out[4][select + 5 + route * 3] = p.to_bits();
                out[4][select + 6 + route * 3] = logits[id as usize].to_bits();
                gap = gap.min(logits[order[rank]] - logits[order[rank + 1]]);
            }
        }
        let canonical = match source {
            SourceFixture::Resident144 => crate::quant_matvec::grouped_moe::GroupMap::new(
                count,
                0,
                count,
                &ids[..count * 10],
                &probs[..count * 10],
                &pairs,
                144,
            )?,
            SourceFixture::Frozen512 { .. } => {
                crate::quant_matvec::grouped_moe::GroupMap::new_frozen_sources(
                    count,
                    &ids[..count * 10],
                    &probs[..count * 10],
                    &pairs,
                    frozen_geometry(layer)?,
                )?
            }
        };
        out[2][..k::MAP_WORDS].copy_from_slice(canonical.words());
        for (dst, p) in out[3].iter_mut().zip(canonical.probabilities()) {
            *dst = p.to_bits();
        }
        let (sources, destination) = out.split_at_mut(4);
        destination[0][frame..frame + k::ROUTER_WORDS].copy_from_slice(&sources[0]);
        destination[0][select..select + 4].copy_from_slice(&[
            count as u32,
            0,
            (count * 10) as u32,
            pairs.len() as u32,
        ]);
        destination[0][map..map + k::MAP_WORDS].copy_from_slice(&sources[2][..k::MAP_WORDS]);
        destination[0][prob..prob + k::PROB_WORDS].copy_from_slice(&sources[3][..k::PROB_WORDS]);
        out[4][meta..meta + 8].copy_from_slice(&[
            0x4b425231,
            1,
            layer as u32,
            start as u32,
            count as u32,
            source.capacity() as u32,
            pairs.len() as u32,
            0,
        ]);
        out[4][meta + 14] = gap.to_bits();
        out[4][meta + 15] = k::NEAR_TIE_MARGIN.to_bits();
        Ok(out)
    }
    #[test]
    fn original_wire_deduplicates_first_seen_and_preserves_all_rank_members() {
        for count in 1..=3 {
            for layer in [0, 47] {
                let s = host_frame(count, layer, 7).unwrap();
                let b = bytes(&s[4]);
                let r = k::audit_frame(&b, &s[1], layer, 7, count).unwrap();
                assert_eq!(r.count(), count);
                assert_eq!(r.groups(), 10 + (count - 1) * 2);
                assert_eq!(r.pairs()[0].0, 0);
                assert!(r.pairs().iter().any(|&(_, slot)| slot < 72));
                assert!(r.pairs().iter().any(|&(_, slot)| slot >= 72));
                assert_eq!(r.probability_bit_mismatches(), 0);
                assert!(r.experts()[count * 10..].iter().all(|&x| x == 0));
                assert!(
                    r.probabilities()[count * 10..]
                        .iter()
                        .all(|x| x.to_bits() == 0)
                );
            }
        }
    }
    #[test]
    fn complete_audit_rejects_dirty_sources_padding_headers_probabilities_and_tables() {
        let good = host_frame(3, 47, 13).unwrap();
        let frame = 47 * k::FRAME_WORDS;
        let sel = frame + k::ROUTER_WORDS;
        let map = sel + k::SELECTION_WORDS;
        let prob = map + k::MAP_WORDS;
        for index in [
            frame,
            frame + 3 * 512,
            frame + k::ROUTER_WORDS - 1,
            sel,
            sel + 4,
            sel + 5,
            sel + 6,
            sel + 244 - 1,
            map + 803,
            prob + 79,
            prob + 80,
            k::AUDIT_WORDS,
            k::AUDIT_OWNED_WORDS - 1,
        ] {
            let mut b = bytes(&good[4]);
            let word = u32::from_le_bytes(b[index * 4..index * 4 + 4].try_into().unwrap()) ^ 1;
            b[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            assert!(
                k::audit_frame(&b, &good[1], 47, 13, 3).is_err(),
                "index {index}"
            );
        }
        for index in [47 * 512, 48 * 512, k::TABLE_WORDS - 1] {
            let mut t = good[1].clone();
            t[index] = 144;
            assert!(k::audit_frame(&bytes(&good[4]), &t, 47, 13, 3).is_err());
        }
        let mut t = good[1].clone();
        t[47 * 512] = u32::MAX;
        assert!(k::audit_frame(&bytes(&good[4]), &t, 47, 13, 3).is_err());
    }
    #[test]
    fn actual_score_changes_without_matching_original_selection_are_rejected() {
        for id in [0, 7, 63, 70] {
            let s = host_frame(3, 0, 0).unwrap();
            let mut b = bytes(&s[4]);
            let bits = if id == 0 {
                f32::NAN.to_bits()
            } else {
                scores(0)[if id == 70 { 63 } else { 0 }].to_bits()
            };
            b[id * 4..id * 4 + 4].copy_from_slice(&bits.to_le_bytes());
            assert!(k::audit_frame(&b, &s[1], 0, 0, 3).is_err());
        }
        assert!(k::validate_table(&fixture(3)[1]).is_ok());
    }
    #[test]
    fn probability_arithmetic_is_bounded_and_separate_from_exact_id_order() {
        let s = host_frame(1, 0, 0).unwrap();
        let mut b = bytes(&s[4]);
        let selection = k::ROUTER_WORDS + 5;
        let probability = k::ROUTER_WORDS + k::SELECTION_WORDS + k::MAP_WORDS;
        let old = u32::from_le_bytes(b[selection * 4..selection * 4 + 4].try_into().unwrap());
        for index in [selection, probability] {
            b[index * 4..index * 4 + 4].copy_from_slice(&(old + 1).to_le_bytes());
        }
        let r = k::audit_frame(&b, &s[1], 0, 0, 1).unwrap();
        assert_eq!(r.probability_bit_mismatches(), 1);
        assert!(r.max_probability_abs_error() > 0.);
        for index in [selection, probability] {
            b[index * 4..index * 4 + 4].copy_from_slice(&0.9f32.to_le_bytes());
        }
        assert!(k::audit_frame(&b, &s[1], 0, 0, 1).is_err());
    }
    #[test]
    fn frozen_actual_das_geometry_full512_bijection_and_original_rank_wires() {
        use crate::quant_matvec::grouped_moe::Geometry;
        for layer in 0..48 {
            let g = frozen_geometry(layer).unwrap();
            assert!(g.frozen_host_sources());
            assert_eq!(g.capacity(), 512);
            assert!(Geometry::new(g.projections(), 8, 512).is_err());
        }
        for count in 1..=3 {
            for pattern in 0..2 {
                for layer in [0, 23, 47] {
                    let source = SourceFixture::Frozen512 { pattern };
                    let s =
                        host_frame_source(count, layer, 4096 - count as u64, source, None).unwrap();
                    k::validate_frozen_source_table(&s[1]).unwrap();
                    assert!(k::validate_table(&s[1]).is_err());
                    let r = k::audit_frame_frozen_sources(
                        &bytes(&s[4]),
                        &s[1],
                        layer,
                        4096 - count as u64,
                        count,
                        frozen_geometry(layer).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(
                        r.groups(),
                        if pattern == 1 {
                            count * 10
                        } else if count == 1 {
                            10
                        } else {
                            12
                        }
                    );
                    assert!(
                        VIRTUAL_BOUNDARIES
                            .iter()
                            .all(|&slot| r.pairs().iter().any(|&(_, actual)| actual == slot))
                    );
                    for row in 0..count {
                        let logits = Zeroizing::new(std::array::from_fn::<_, 512, _>(|id| {
                            f32::from_bits(s[0][row * 512 + id])
                        }));
                        let original =
                            kuno_text::ops::router_topk_guarded(&*logits, 10, true).unwrap();
                        for (rank, &(id, p)) in original.as_slice().iter().enumerate() {
                            assert_eq!(r.experts()[row * 10 + rank], id);
                            assert_eq!(r.probabilities()[row * 10 + rank].to_bits(), p.to_bits());
                        }
                    }
                    assert!(r.experts()[count * 10..].iter().all(|&x| x == 0));
                    assert!(
                        r.probabilities()[count * 10..]
                            .iter()
                            .all(|x| x.to_bits() == 0)
                    );
                    assert!(
                        k::audit_frame(&bytes(&s[4]), &s[1], layer, 4096 - count as u64, count)
                            .is_err()
                    );
                }
            }
        }
    }
    #[test]
    fn frozen_complete_audit_rejects_all_wire_families_and_unselected_table_corruption() {
        let good =
            host_frame_source(3, 47, 4093, SourceFixture::Frozen512 { pattern: 1 }, None).unwrap();
        let frame = 47 * k::FRAME_WORDS;
        let selection = frame + k::ROUTER_WORDS;
        let map = selection + k::SELECTION_WORDS;
        let prob = map + k::MAP_WORDS;
        for index in [
            frame,
            frame + 3 * 512,
            frame + k::ROUTER_WORDS - 1,
            selection,
            selection + 4,
            selection + 5,
            selection + 6,
            selection + k::SELECTION_WORDS - 1,
            map,
            map + 4,
            map + k::MAP_WORDS - 1,
            prob,
            prob + 79,
            prob + 80,
            prob + k::PROB_WORDS + 5,
            k::AUDIT_WORDS,
            k::AUDIT_OWNED_WORDS - 1,
        ] {
            let mut wire = bytes(&good[4]);
            let old = u32::from_le_bytes(wire[index * 4..index * 4 + 4].try_into().unwrap());
            wire[index * 4..index * 4 + 4].copy_from_slice(&(old ^ 1).to_le_bytes());
            assert!(
                k::audit_frame_frozen_sources(
                    &wire,
                    &good[1],
                    47,
                    4093,
                    3,
                    frozen_geometry(47).unwrap()
                )
                .is_err(),
                "index {index}"
            );
        }
        for index in [0, 511, 47 * 512 + 511, 48 * 512, k::TABLE_WORDS - 1] {
            for bad in [512, u32::MAX] {
                let mut table = good[1].clone();
                table[index] = bad;
                assert!(
                    k::audit_frame_frozen_sources(
                        &bytes(&good[4]),
                        &table,
                        47,
                        4093,
                        3,
                        frozen_geometry(47).unwrap()
                    )
                    .is_err()
                );
            }
        }
        let mut table = good[1].clone();
        table[510] = table[511]; // Different layer, entirely unselected.
        assert!(
            k::audit_frame_frozen_sources(
                &bytes(&good[4]),
                &table,
                47,
                4093,
                3,
                frozen_geometry(47).unwrap()
            )
            .is_err()
        );
    }
    #[test]
    fn frozen_equal_probabilities_and_signed_zero_do_not_relax_late_exact_expert_ids() {
        for count in 1..=3 {
            for tie in 0..3 {
                let s = host_frame_source(
                    count,
                    47,
                    4096 - count as u64,
                    SourceFixture::Frozen512 { pattern: 0 },
                    Some(tie),
                )
                .unwrap();
                k::audit_frame_frozen_sources(
                    &bytes(&s[4]),
                    &s[1],
                    47,
                    4096 - count as u64,
                    count,
                    frozen_geometry(47).unwrap(),
                )
                .unwrap();
                if tie == 2 {
                    assert_eq!(s[0][0], 0f32.to_bits());
                    assert_eq!(s[0][1], (-0f32).to_bits());
                    require_late_exact_id_rejection(
                        &bytes(&s[4]),
                        &s[1],
                        47,
                        4096 - count as u64,
                        count,
                    )
                    .unwrap();
                }
            }
        }
        assert!(require_late_exact_id_rejection(&[], &[], 0, 0, 1).is_err());
    }
    #[test]
    fn frozen_fault_inventory_retains_original144_sentinel_and_fullowner_semantics() {
        let mut old = fixture(3);
        assert_eq!(mutate(&mut old, 0, 47, 3).unwrap(), k::FAULT_MISSING);
        k::validate_table(&old[1]).unwrap();
        for count in 1..=3 {
            for kind in [0, 10, 13, 14, 16] {
                let mut f = frozen_fixture(count, 0).unwrap();
                let prior = f.clone();
                let expected = mutate_frozen(&mut f, kind, 47, count).unwrap();
                assert_eq!(
                    expected,
                    match kind {
                        10 => 32 | k::FAULT_SHAPE,
                        13 => k::FAULT_MISSING,
                        _ => k::FAULT_SHAPE,
                    }
                );
                if kind == 10 {
                    assert_eq!(f[5][0], 32);
                    assert_eq!(f[..5], prior[..5]);
                } else {
                    assert!(k::validate_frozen_source_table(&f[1]).is_err());
                    for owner in [0, 2, 3, 4, 5] {
                        assert_eq!(f[owner], prior[owner]);
                    }
                }
            }
        }
        assert!(frozen_fixture(0, 0).is_err());
        assert!(frozen_fixture(4, 0).is_err());
        assert!(frozen_fixture(1, 2).is_err());
        assert_eq!(TOTAL_CASES, 80);
        assert_eq!(FROZEN_FAULT_CASES, 23);
    }
    #[test]
    fn both_source_capacity_specializations_preserve_readonly_input_and_dense_reflection() {
        use wgpu::naga::{AddressSpace, StorageAccess};
        let ordinary = include_str!("resident_batch_router_gpu.wgsl");
        assert_eq!(
            ordinary.matches("const SOURCE_CAPACITY:u32=144u;").count(),
            1
        );
        let frozen = ordinary.replace(
            "const SOURCE_CAPACITY:u32=144u;",
            "const SOURCE_CAPACITY:u32=512u;",
        );
        assert_eq!(
            frozen.replace(
                "const SOURCE_CAPACITY:u32=512u;",
                "const SOURCE_CAPACITY:u32=144u;"
            ),
            ordinary
        );
        for source in [ordinary, frozen.as_str()] {
            let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points.len(), 2);
            for (entry, ep) in module.entry_points.iter().enumerate() {
                let mut bindings = Vec::new();
                for (handle, var) in module.global_variables.iter() {
                    if let Some(binding) = &var.binding {
                        assert_eq!(binding.group, 0);
                        bindings.push(binding.binding);
                        assert!(!info.get_entry_point(entry)[handle].is_empty());
                        if binding.binding < 2 {
                            assert_eq!(
                                var.space,
                                AddressSpace::Storage {
                                    access: StorageAccess::LOAD
                                }
                            );
                        }
                    }
                }
                bindings.sort_unstable();
                assert_eq!(bindings, [0, 1, 2, 3, 4, 5, 6]);
                let spv = wgpu::naga::back::spv::write_vec(
                    &module,
                    &info,
                    &wgpu::naga::back::spv::Options {
                        lang_version: (1, 3),
                        ..Default::default()
                    },
                    Some(&wgpu::naga::back::spv::PipelineOptions {
                        shader_stage: wgpu::naga::ShaderStage::Compute,
                        entry_point: ep.name.clone(),
                    }),
                )
                .unwrap();
                assert!(!spv.is_empty());
            }
        }
    }
}
