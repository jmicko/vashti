//! Compiled-public standalone acceptance for the exact resident router mapper.
//! No model, prompt, timing or private API input. Original GPU top10 and the
//! original core router/Map constructors are the independent numeric/wire
//! references. Every complete arena owner is inspected and explicitly erased.
use crate::{
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, BufferLayout, PrivateArena},
    resident_router_gpu as k,
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::{
    ffi::{OsStr, OsString},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::{Zeroize, Zeroizing};

// Reuse the actual private original Map constructor without widening its
// production visibility. The fixture source also declares unused GPU helpers.
#[path = "count1_hybrid_moe.rs"]
#[allow(dead_code, clippy::duplicate_mod)]
mod canonical;
use crate::resident_util::arg;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type State = [Zeroizing<Vec<u32>>; 8];
const LOGICAL: [usize; 8] = [
    k::PACKET_WORDS,
    k::TABLE_WORDS,
    k::MAP_WORDS,
    k::PROB_WORDS,
    k::OWNER_WORDS,
    k::AUDIT_WORDS,
    1,
    k::PACKET_WORDS,
];
const OWNED: [usize; 8] = [8192, 32768, 8192, 8192, 8192, 524288, 8192, 8192];
const PRIVATE_BYTES: usize = 2_424_832;
const PRIVATE_CAP: usize = 3 * 1024 * 1024;
const MAIN_CAP: u64 = 16 * 1024 * 1024;
const IO_CAP: u64 = 64 * 1024 * 1024;
const HOST_CAP: usize = 16 * 1024 * 1024;
const FIRST: usize = crate::router_gpu::FIRST;
const END: usize = crate::router_gpu::END;
const FAULT_CASES: usize = 24;

fn configuration(
    features: bool,
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> std::result::Result<(), &'static str> {
    if features {
        return Err("resident mapper probe requires default features");
    }
    let (mut gate, mut receipt) = (false, false);
    for (key, value) in vars {
        match key.to_str() {
            Some("KUNO_LAB_RESIDENT_ROUTER_PUBLIC_ONLY") => gate = value == OsStr::new("1"),
            Some("KUNO_PUBLIC_RECEIPT") => receipt = !value.is_empty(),
            _ => {
                let key = key.to_string_lossy();
                if key.starts_with("KUNO_LAB_")
                    || key.starts_with("KUNO_TEST_")
                    || key.starts_with("KUNO_PUBLIC_")
                {
                    return Err("resident mapper inherited setting rejected");
                }
            }
        }
    }
    if !gate || !receipt {
        return Err("resident mapper public gate and new receipt required");
    }
    Ok(())
}
pub(crate) fn validate_configuration() -> std::result::Result<(), &'static str> {
    configuration(
        cfg!(any(
            feature = "direct-device",
            feature = "cpu-fma",
            feature = "cpu-buffered-weights",
            feature = "cpu-work-stealing",
            feature = "cpu-q8-two-pass"
        )),
        std::env::vars_os(),
    )
}
pub(crate) fn validate_limits(l: &wgpu::Limits) -> Result<()> {
    if l.max_compute_invocations_per_workgroup < 256
        || l.max_compute_workgroup_size_x < 256
        || l.max_compute_workgroup_storage_size < 300
        || l.max_storage_buffers_per_shader_stage < 7
        || l.max_bindings_per_bind_group < 8
        || l.max_bind_groups < 1
        || l.max_uniform_buffer_binding_size < 16
        || l.max_compute_workgroups_per_dimension < k::PREFLIGHT_WORKGROUPS
        || l.max_storage_buffer_binding_size < 2097152
        || l.max_buffer_size < 2097152
    {
        return Err("resident mapper actual device limits rejected".into());
    }
    Ok(())
}
fn state() -> Result<State> {
    // Two full snapshots plus the maximum one-at-a-time byte lease/upload and
    // independent fixed reference metadata are precharged before allocating.
    main_admission::check_capacity(
        HOST_CAP as u64,
        &[3 * PRIVATE_BYTES as u64, 2 * 1024 * 1024],
    )?;
    let out = OWNED.map(|n| Zeroizing::new(vec![0u32; n]));
    if out.iter().map(|v| v.capacity() * 4).sum::<usize>() != PRIVATE_BYTES {
        return Err("resident mapper actual host capacity rejected".into());
    }
    Ok(out)
}
fn logits(layer: usize) -> [f32; 512] {
    std::array::from_fn(|id| match layer % 8 {
        0 => ((id * 71 % 509) as f32 - 253.) / 31.,
        1 => 0.,
        2 => {
            if id == 511 {
                20.
            } else if id == 0 {
                19.
            } else {
                ((id * 29 % 503) as f32 - 251.) / 37.
            }
        }
        3 => (id % 17) as f32 * 10000.,
        4 => -(id as f32) / 37.,
        5 => {
            if id == 64 || id == 511 {
                f32::MAX
            } else {
                f32::MIN
            }
        }
        6 => {
            if id == 17 {
                1e20
            } else {
                -1e20
            }
        }
        _ => {
            if id == 511 {
                100.
            } else if id == 0 {
                99.
            } else {
                (id % 31) as f32
            }
        }
    })
}
fn immutable_table() -> Result<Zeroizing<Vec<u32>>> {
    let mut table = Zeroizing::new(vec![0; OWNED[1]]);
    table[..k::TABLE_WORDS].fill(u32::MAX);
    for layer in 0..48 {
        let expected = kuno_text::ops::router_topk(&logits(layer), 10, true)?;
        for (rank, &(id, _)) in expected.iter().enumerate() {
            table[layer * 512 + id] = ((rank * 17 + layer * 7) % 144) as u32;
        }
    }
    Ok(table)
}
fn initialize_numeric(expected: &mut State, table: &[u32], layer: usize) {
    for index in [0, 2, 3, 4, 6, 7] {
        expected[index].fill(0);
    }
    expected[1].copy_from_slice(table);
    for (index, word) in expected[0][..2560].iter_mut().enumerate() {
        *word = (((index * 13 + layer * 7) % 101) as f32 / 7. - 7.).to_bits();
    }
    let first = 3 + layer % 7;
    for (id, value) in logits(layer).into_iter().enumerate() {
        expected[0][2560 + id] = value.to_bits();
        expected[7][first + id] = value.to_bits();
    }
}
fn start(layer: usize) -> u64 {
    if layer == 47 {
        4095
    } else {
        (layer * 73) as u64
    }
}
struct Bytes(cubecl::bytes::Bytes);
impl Drop for Bytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn write(arena: &PrivateArena, id: BufferId, words: &[u32], retired: &AtomicBool) -> Result<()> {
    let layout = arena.layout(id)?;
    if words.len() != layout.owned_words {
        return Err("resident mapper owning write extent".into());
    }
    let mut bytes = Upload(Vec::with_capacity(layout.owned_bytes));
    if bytes.0.capacity() != layout.owned_bytes {
        return Err("resident mapper owning upload capacity".into());
    }
    for word in words {
        bytes.0.extend(word.to_le_bytes());
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arena
            .client()?
            .write_private(arena.handle(id)?.clone(), std::mem::take(&mut bytes.0))
            .map_err(|_| "resident mapper owning write failed")?;
        Ok::<_, Box<dyn std::error::Error>>(())
    }));
    if !matches!(result, Ok(Ok(()))) {
        retired.store(true, Ordering::Release);
        return Err("resident mapper owning write retired".into());
    }
    Ok(())
}
fn audit(client: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<()> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io_audited())
    }));
    match result {
        Ok(Ok(r))
            if r.upload_bytes
                .checked_add(r.uniform_bytes)
                .and_then(|n| n.checked_add(r.readback_bytes))
                .is_some_and(|n| n > 0 && n <= IO_CAP && n == r.inspected_bytes)
                && r.inspected_buffers > 0
                && !retired.load(Ordering::Acquire) =>
        {
            Ok(())
        }
        _ => {
            retired.store(true, Ordering::Release);
            Err("resident mapper complete IO audit retired".into())
        }
    }
}
fn snapshot(arena: &PrivateArena, ids: &[BufferId; 8], retired: &AtomicBool) -> Result<State> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<State> {
        let mut out = state()?;
        for (index, &id) in ids.iter().enumerate() {
            let bytes = Bytes(
                arena
                    .client()?
                    .read_one(arena.handle(id)?.clone())
                    .map_err(|_| "resident mapper owning read failed")?,
            );
            if bytes.0.len() != OWNED[index] * 4 {
                return Err("resident mapper owning read extent".into());
            }
            for (dst, word) in out[index].iter_mut().zip(bytes.0.chunks_exact(4)) {
                *dst = u32::from_le_bytes(word.try_into().unwrap());
            }
        }
        Ok(out)
    }));
    let out = match result {
        Ok(Ok(out)) => out,
        _ => {
            retired.store(true, Ordering::Release);
            return Err("resident mapper owning read retired".into());
        }
    };
    audit(arena.client()?, retired)?;
    Ok(out)
}
fn physical(arena: &PrivateArena, ids: &[BufferId; 8]) -> Result<()> {
    let mut buffers = Vec::with_capacity(8);
    let stream = arena.handle(ids[0])?.stream;
    for (index, &id) in ids.iter().enumerate() {
        let handle = arena.handle(id)?;
        if ids[..index].contains(&id)
            || arena.layout(id)? != BufferLayout::new(LOGICAL[index])?
            || handle.stream != stream
            || handle.offset_start.unwrap_or(0) != 0
            || handle.offset_end.unwrap_or(0) != 0
            || handle.size() != OWNED[index] as u64 * 4
            || handle.size_in_used() != OWNED[index] as u64 * 4
        {
            return Err("resident mapper owning physical metadata".into());
        }
        let resource = arena
            .client()?
            .get_resource(handle.clone())
            .map_err(|_| "resident mapper complete resource")?;
        let resource = resource.resource();
        if resource.offset != 0
            || resource.size != OWNED[index] as u64 * 4
            || resource.buffer.size() != resource.size
            || buffers.contains(&resource.buffer)
        {
            return Err("resident mapper physical owner extent or alias".into());
        }
        buffers.push(resource.buffer.clone());
    }
    Ok(())
}
fn mapper(arena: &PrivateArena, ids: &[BufferId; 8], layer: usize, start: u64) -> Result<()> {
    let handles: Vec<_> = ids[..7]
        .iter()
        .map(|&id| arena.handle(id).cloned())
        .collect::<std::result::Result<_, _>>()?;
    k::launch(
        arena.client()?,
        handles
            .try_into()
            .map_err(|_| "resident mapper handle count")?,
        layer,
        start,
    )?;
    Ok(())
}
fn expected_numeric(expected: &mut State, actual: &State, layer: usize) -> Result<()> {
    let reference = kuno_text::ops::router_topk(&logits(layer), 10, true)?;
    for index in 0..OWNED[0] {
        if (FIRST..END).contains(&index) {
            continue;
        }
        if actual[0][index] != expected[0][index] {
            return Err("resident mapper original packet source interval changed".into());
        }
    }
    let mut ids = [0usize; 10];
    let mut probabilities = [0f32; 10];
    let mut pairs = [(0u32, 0u32); 10];
    for (rank, &(id, weight)) in reference.iter().enumerate() {
        let got = f32::from_bits(actual[0][FIRST + rank * 2 + 1]);
        if actual[0][FIRST + rank * 2] != id as u32
            || !got.is_finite()
            || got < 0.
            || (got - weight).abs() > 2e-7 + 2e-6 * weight.abs()
        {
            return Err("resident mapper independent original top10 oracle mismatch".into());
        }
        ids[rank] = id;
        probabilities[rank] = got;
        pairs[rank] = (id as u32, expected[1][layer * 512 + id]);
    }
    let split = std::array::from_fn(|index| {
        if index % 2 == 0 {
            0
        } else {
            (index / 2) as u32
        }
    });
    let canonical = canonical::Map::new(&ids, &probabilities, &split, &pairs, 144)?;
    expected[0].copy_from_slice(&actual[0]);
    expected[2][..k::MAP_WORDS].copy_from_slice(canonical.words());
    for (dst, weight) in expected[3][..k::PROB_WORDS]
        .iter_mut()
        .zip(canonical.probabilities())
    {
        *dst = weight.to_bits();
    }
    expected[4][..k::OWNER_WORDS].copy_from_slice(canonical.owner_words());
    let frame = layer * k::FRAME_WORDS;
    let mut offset = frame;
    for (index, count) in [
        (0, k::PACKET_WORDS),
        (2, k::MAP_WORDS),
        (3, k::PROB_WORDS),
        (4, k::OWNER_WORDS),
    ] {
        let source = Zeroizing::new(expected[index][..count].to_vec());
        expected[5][offset..offset + count].copy_from_slice(&source);
        offset += count;
    }
    expected[5][offset] = layer as u32;
    expected[5][offset + 1] = start(layer) as u32;
    if expected != actual {
        return Err(
            "resident mapper complete canonical wires copied audit or owning tail mismatch".into(),
        );
    }
    Ok(())
}
fn mutate_fault(state: &mut State, case: usize) -> Result<()> {
    let id0 = state[0][FIRST] as usize;
    let id1 = state[0][FIRST + 2] as usize;
    let layer = 47;
    match case {
        0 => state[0][FIRST] = 512,
        1 => state[0][FIRST + 2] = id0 as u32,
        2 => state[1][layer * 512 + id0] = u32::MAX,
        3 => state[1][layer * 512 + id1] = state[1][layer * 512 + id0],
        4 => state[1][layer * 512 + id0] = 144,
        5 => state[0][FIRST + 1] = f32::NAN.to_bits(),
        6 => state[0][FIRST + 1] = (-0.5f32).to_bits(),
        7 => state[0][FIRST + 1] = 1.25f32.to_bits(),
        8 => {
            for rank in 0..10 {
                state[0][FIRST + rank * 2 + 1] = 0;
            }
        }
        9 => state[6][0] = 8,
        10 => state[6][OWNED[6] - 1] = 1,
        11 => state[0][FIRST - 1] = 4,
        12 => state[0][END] = 0x1234,
        13 => state[1][k::TABLE_WORDS] = 1,
        14 => state[2][k::MAP_WORDS - 1] = 1,
        15 => state[2][OWNED[2] - 1] = 1,
        16 => state[3][k::PROB_WORDS] = 1,
        17 => state[4][OWNED[4] - 1] = 1,
        18 => state[5][layer * k::FRAME_WORDS + 17] = 1,
        19 => state[5][k::AUDIT_WORDS] = 1,
        20 => state[0][0] = f32::INFINITY.to_bits(),
        21 => state[0][2560 + 511] = f32::NAN.to_bits(),
        22 => {
            state[0].swap(FIRST, FIRST + 2);
            state[0].swap(FIRST + 1, FIRST + 3);
        }
        23 => state[1][0] = 145,
        _ => return Err("resident mapper compiled fault case invalid".into()),
    }
    Ok(())
}
fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    Ok(client
        .memory_usage()
        .map_err(|_| "resident mapper main telemetry")?
        .bytes_reserved)
}
fn allocate(arena: &mut PrivateArena, ledger: &mut MainAdmission) -> Result<[BufferId; 8]> {
    ledger.observe_before(sample(arena.client()?)?)?;
    ledger.precharge(&OWNED.map(|v| v as u64 * 4))?;
    let ids: Vec<_> = LOGICAL
        .into_iter()
        .map(|n| arena.allocate_zeroed(n))
        .collect::<std::result::Result<_, _>>()?;
    let ids = ids
        .try_into()
        .map_err(|_| "resident mapper owning allocation count")?;
    if arena.charged_bytes() != PRIVATE_BYTES {
        return Err("resident mapper full private charge mismatch".into());
    }
    physical(arena, &ids)?;
    ledger.observe_after(sample(arena.client()?)?)?;
    Ok(ids)
}
fn numeric(
    arena: &PrivateArena,
    ids: &[BufferId; 8],
    expected: &mut State,
    table: &[u32],
    layer: usize,
    retired: &AtomicBool,
    ledger: &mut MainAdmission,
) -> Result<()> {
    initialize_numeric(expected, table, layer);
    for index in [0, 1, 2, 3, 4, 6, 7] {
        write(arena, ids[index], &expected[index], retired)?;
    }
    crate::next_shared::router(
        arena.client()?,
        arena.handle(ids[7])?.clone(),
        arena.handle(ids[0])?.clone(),
        3 + layer % 7,
    )?;
    mapper(arena, ids, layer, start(layer))?;
    let actual = snapshot(arena, ids, retired)?;
    expected_numeric(expected, &actual, layer)?;
    ledger.completed_fence(sample(arena.client()?)?)?;
    Ok(())
}

/// Root registers the protected exclusive mode before device creation and
/// enables SUBGROUP with physical16..32 bounds for the unchanged GPU router.
pub(crate) fn run(client: &ComputeClient<WgpuRuntime>, limits: &wgpu::Limits) -> Result<()> {
    validate_configuration()?;
    validate_limits(limits)?;
    let initial = client
        .memory_usage()
        .map_err(|_| "resident mapper initial telemetry")?;
    if initial.number_allocs != 0
        || initial.bytes_in_use != 0
        || initial.bytes_padding != 0
        || initial.bytes_reserved != 0
    {
        return Err("resident mapper fresh exclusive context required".into());
    }
    main_admission::check_capacity(MAIN_CAP, &[PRIVATE_BYTES as u64])?;
    client
        .configure_transient_io(IO_CAP)
        .map_err(|_| "resident mapper bounded IO setup")?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), PRIVATE_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let mut ledger = MainAdmission::new(MAIN_CAP, 0)?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        let ids = allocate(&mut arena, &mut ledger)?;
        let mut expected = state()?;
        let table = immutable_table()?;
        for layer in 0..48 {
            println!("public_resident_router_stage case={layer} stage=numeric-start");
            numeric(
                &arena,
                &ids,
                &mut expected,
                &table,
                layer,
                &retired,
                &mut ledger,
            )?;
        }
        let last = Zeroizing::new(expected[0].to_vec());
        for case in 0..FAULT_CASES {
            println!("public_resident_router_stage case={case} stage=fault-start");
            for words in &mut expected {
                words.fill(0);
            }
            expected[0].copy_from_slice(&last);
            expected[1].copy_from_slice(&table);
            mutate_fault(&mut expected, case)?;
            for (index, &id) in ids.iter().enumerate() {
                write(&arena, id, &expected[index], &retired)?;
            }
            mapper(&arena, &ids, 47, 4095)?;
            let actual = snapshot(&arena, &ids, &retired)?;
            expected[6][0] |= 4;
            if actual != expected {
                return Err(
                    "resident mapper fault changed complete source or published output".into(),
                );
            }
            ledger.completed_fence(sample(client)?)?;
        }
        // Fresh epoch replays original numeric operators; stale IDs cannot
        // authorize access to the newly allocated full owners.
        arena.reset()?;
        if arena.handle(ids[0]).is_ok() {
            return Err("resident mapper stale owner accepted after reset".into());
        }
        client.memory_cleanup();
        ledger.completed_fence(sample(client)?)?;
        let fresh = allocate(&mut arena, &mut ledger)?;
        for words in &mut expected {
            words.fill(0);
        }
        numeric(
            &arena,
            &fresh,
            &mut expected,
            &table,
            0,
            &retired,
            &mut ledger,
        )?;
        Ok(())
    }));
    let close = arena.close();
    if close.is_err() || retired.load(Ordering::Acquire) {
        return Err("resident mapper complete owning cleanup failed; context retired".into());
    }
    match attempt {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            // Compare only this closed source-controlled vocabulary. Never
            // print raw backend errors, vectors or selected expert identities.
            let class = match error.to_string().as_str() {
                "resident mapper independent original top10 oracle mismatch" => "top10-reference",
                "resident mapper original packet source interval changed" => "packet-source",
                "resident mapper complete canonical wires copied audit or owning tail mismatch" => {
                    "canonical-audit"
                }
                "resident mapper fault changed complete source or published output" => {
                    "fault-publication"
                }
                "resident mapper owning read retired" => "read-owner-retired",
                "resident mapper complete IO audit retired" => "io-owner-retired",
                "resident mapper physical owner extent or alias" => "physical-owner",
                _ => "unclassified",
            };
            eprintln!("public_resident_router_failure class={class}");
            return Err("resident mapper public operator failed; no acceptance receipt".into());
        }
        Err(_) => {
            return Err("resident mapper public operator failed; no acceptance receipt".into());
        }
    }
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("resident mapper complete owning zero audit missing")?;
    if zero.buffers != 16 || zero.bytes != 2 * PRIVATE_BYTES as u64 {
        return Err("resident mapper full erased owner counts mismatch".into());
    }
    let mut clean = false;
    for _ in 0..2 {
        cubecl::future::block_on(client.sync()).map_err(|_| "resident mapper final completion")?;
        client.memory_cleanup();
        cubecl::future::block_on(client.sync())
            .map_err(|_| "resident mapper final allocator completion")?;
        let m = client
            .memory_usage()
            .map_err(|_| "resident mapper final allocator telemetry")?;
        if m.number_allocs == 0
            && m.bytes_in_use == 0
            && m.bytes_padding == 0
            && m.bytes_reserved == 0
        {
            clean = true;
            break;
        }
    }
    if !clean {
        return Err("resident mapper complete allocator cleanup unaccepted".into());
    }
    let receipt = serde_json::json!({"resident_router_fixed_public_probe":true,"original_gpu_router_unchanged":true,"independent_core_top10_cases":49,"all_layer_audit_offsets":48,"fresh_owner_replay":true,"canonical_map_words":k::MAP_WORDS,"canonical_probability_words":k::PROB_WORDS,"canonical_owner_words":k::OWNER_WORDS,"literal_packet_words":k::PACKET_WORDS,"audit_frame_words":k::FRAME_WORDS,"audit_layer_and_start_identity":true,"exact_consumed_headers_and_all_owned_tails":true,"fault_cases":FAULT_CASES,"fault_no_output_publication":true,"immutable_table_and_input_unchanged":true,"private_owned_bytes":PRIVATE_BYTES,"private_cap_bytes":PRIVATE_CAP,"main_cap_bytes":MAIN_CAP,"io_cap_bytes":IO_CAP,"host_capacity_bound_bytes":HOST_CAP,"owned_zero_buffers":zero.buffers,"owned_zero_bytes":zero.bytes,"allocator_cleanup_complete":true,"context_retired":false,"private_request_accepted":false,"whole_model_accepted":false,"performance_accepted":false});
    let path =
        std::env::var_os("KUNO_PUBLIC_RECEIPT").ok_or("resident mapper new receipt required")?;
    let out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| "resident mapper new receipt destination rejected")?;
    serde_json::to_writer_pretty(out, &receipt)
        .map_err(|_| "resident mapper receipt write failed")?;
    println!("{receipt}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vars() -> Vec<(OsString, OsString)> {
        vec![
            ("KUNO_LAB_RESIDENT_ROUTER_PUBLIC_ONLY".into(), "1".into()),
            ("KUNO_PUBLIC_RECEIPT".into(), "fresh-receipt".into()),
        ]
    }
    #[test]
    fn public_gate_rejects_other_settings_features_and_missing_inputs() {
        assert!(configuration(false, vars()).is_ok());
        assert!(configuration(true, vars()).is_err());
        assert!(configuration(false, Vec::new()).is_err());
        for key in ["KUNO_LAB_MODEL", "KUNO_TEST_GGUF", "KUNO_PUBLIC_PROMPT"] {
            let mut v = vars();
            v.push((key.into(), "0".into()));
            assert!(configuration(false, v).is_err());
        }
        let mut v = vars();
        v[0].1 = "0".into();
        assert!(configuration(false, v).is_err());
    }
    #[test]
    fn actual_rounding_capacity_and_boundary_fixture_inventories() {
        let layouts = LOGICAL.map(|n| BufferLayout::new(n).unwrap());
        assert_eq!(layouts.map(|l| l.owned_words), OWNED);
        assert_eq!(OWNED.iter().sum::<usize>() * 4, PRIVATE_BYTES);
        assert!(state().is_ok());
        assert_eq!(start(0), 0);
        assert_eq!(start(47), 4095);
        let table = immutable_table().unwrap();
        assert!(table[k::TABLE_WORDS..].iter().all(|&v| v == 0));
        for boundary in [0, 71, 72, 143] {
            assert!(table[..k::TABLE_WORDS].contains(&boundary));
        }
        for layer in 0..48 {
            let expected = kuno_text::ops::router_topk(&logits(layer), 10, true).unwrap();
            let slots: Vec<_> = expected
                .iter()
                .map(|&(id, _)| table[layer * 512 + id])
                .collect();
            assert!(slots.iter().all(|&s| s < 144));
            for (index, &slot) in slots.iter().enumerate() {
                assert!(!slots[..index].contains(&slot));
            }
        }
        let edge = kuno_text::ops::router_topk(&logits(47), 10, true).unwrap();
        assert_eq!((edge[0].0, edge[1].0), (511, 0));
    }
}
