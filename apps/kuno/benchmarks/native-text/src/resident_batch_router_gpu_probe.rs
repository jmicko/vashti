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
            before[0][7] = before[0][0];
            k::FAULT_NEAR_TIE
        }
        12 => {
            before[0][70] = before[0][63];
            k::FAULT_NEAR_TIE
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
fn case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    ledger: &mut MainAdmission,
    layer: usize,
    count: usize,
    kind: Option<usize>,
) -> Result<serde_json::Value> {
    let start = 4096 - count as u64;
    let mut before = fixture(count);
    let expected_fault = kind
        .map(|n| mutate(&mut before, n, layer, count))
        .transpose()?
        .unwrap_or(0);
    let ids = upload(arena, &before, ledger)?;
    let handles: [Handle; 6] = ids
        .iter()
        .map(|&id| arena.handle(id).cloned())
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "batch router six owner handles")?;
    k::launch(client, handles.clone(), layer, start, count)?;
    let actual = snapshot(arena, &ids)?;
    ledger.completed_fence(sample(client)?)?;
    let mut max_error = 0.;
    let mut mismatch = 0;
    let mut groups = 0;
    if kind.is_some() {
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
        let routes = k::audit_frame(&wire, &before[1], layer, start, count)?;
        groups = routes.groups();
        mismatch = routes.probability_bit_mismatches();
        max_error = routes.max_probability_abs_error();
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
        serde_json::json!({"layer":layer,"count":count,"fault_case":kind,"sticky_fault":expected_fault,
        "groups":groups,"probability_bit_mismatches":mismatch,"max_original_probability_abs_error":max_error,
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
    crate::main_admission::check_capacity(HOST_CAP, &[3 * k::MAIN_BYTES, 2 * 1024 * 1024])?;
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
        let mut report = Vec::with_capacity(10 + FAULT_CASES);
        for layer in [0, 23, 47] {
            for count in 1..=3 {
                report.push(case(client, &mut arena, &mut ledger, layer, count, None)?);
            }
        }
        for fault in 0..FAULT_CASES {
            report.push(case(client, &mut arena, &mut ledger, 47, 3, Some(fault))?);
        }
        report.push(case(client, &mut arena, &mut ledger, 0, 3, None)?); // Fresh epoch replay.
        Ok(report)
    }));
    let close = arena.close();
    let zero = arena.owned_zero_audit_receipt();
    let report = match (attempt, close, retired.load(Ordering::Acquire)) {
        (Ok(Ok(r)), Ok(()), false) => r,
        _ => return Err("batch router compiled-public gates failed; no accepted receipt".into()),
    };
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
        "numeric_cases":10,"fault_cases":FAULT_CASES,"max_count":3,"main_bytes":k::MAIN_BYTES,"inline_dispatch":inline,
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
    fn host_frame(count: usize, layer: usize, start: u64) -> Result<State> {
        let mut out = fixture(count);
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
            let logits = scores(row);
            let original = kuno_text::ops::router_topk_guarded(&logits, 10, true)?;
            let mut order = std::array::from_fn::<_, 512, _>(|id| id);
            order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
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
        let canonical = crate::quant_matvec::grouped_moe::GroupMap::new(
            count,
            0,
            count,
            &ids[..count * 10],
            &probs[..count * 10],
            &pairs,
            144,
        )?;
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
            144,
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
    fn original_numeric_audit_rejects_nonfinite_internal_and_boundary_ties() {
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
}
