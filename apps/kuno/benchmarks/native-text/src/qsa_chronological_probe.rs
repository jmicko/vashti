//! Standalone compiled-public acceptance of original all-retained QSA IDs.
//! This operator never loads models or arbitrary prompts. Independent original
//! CPU score-selection equations verify every active row; complete source,
//! status, selected, fault pages and all owning padding are inspected/erased.
use super::chronological_select as k;
use crate::{
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, BufferLayout, PrivateArena},
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
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type State = [Zeroizing<Vec<u32>>; 4];
const LOGICAL: [usize; 4] = [8192, 8192, 8 * k::SELECT_STRIDE, 1];
const OWNED: [usize; 4] = k::OWNED_WORDS;
const PRIVATE_BYTES: usize = 229_376;
const PRIVATE_CAP: usize = 1024 * 1024;
const MAIN_CAP: u64 = 16 * 1024 * 1024;
const IO_CAP: u64 = 64 * 1024 * 1024;
const HOST_CAP: usize = 2 * 1024 * 1024;
const FAULT_CASES: usize = 24;

fn configuration(
    features: bool,
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> std::result::Result<(), &'static str> {
    if features {
        return Err("QSA chronological probe requires default features");
    }
    let (mut gate, mut receipt) = (false, false);
    for (key, value) in vars {
        match key.to_str() {
            Some("KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY") => {
                if gate || value != OsStr::new("1") {
                    return Err("QSA chronological exact public gate rejected");
                }
                gate = true;
            }
            Some("KUNO_PUBLIC_RECEIPT") => {
                if receipt || value.is_empty() {
                    return Err("QSA chronological receipt rejected");
                }
                receipt = true;
            }
            _ => {
                let key = key.to_string_lossy();
                if key.starts_with("KUNO_LAB_")
                    || key.starts_with("KUNO_TEST_")
                    || key.starts_with("KUNO_PUBLIC_")
                {
                    return Err("QSA chronological inherited setting rejected");
                }
            }
        }
    }
    if !gate || !receipt {
        return Err("QSA chronological public gate and new receipt required");
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
    k::validate_limits(l)?;
    Ok(())
}
fn state() -> Result<State> {
    // Two full snapshots, one maximum owning mapped byte lease, one maximum
    // upload, independent score/ID vectors and resource/INFO metadata bounded
    // BEFORE any allocation. Complete capacities remain zeroizing on all exits.
    main_admission::check_capacity(
        HOST_CAP as u64,
        &[
            2 * PRIVATE_BYTES as u64,
            2 * 131072,
            65536,
            k::HOST_METADATA_BYTES as u64,
        ],
    )?;
    let out = OWNED.map(|n| Zeroizing::new(vec![0u32; n]));
    if out.iter().map(|v| v.capacity() * 4).sum::<usize>() != PRIVATE_BYTES {
        return Err("QSA chronological actual host capacity rejected".into());
    }
    Ok(out)
}
fn cases() -> Vec<k::Geometry> {
    let mut cases = Vec::new();
    for count in [1, 2, 3, 5, 8] {
        for visible in [count, count.max(3), count.max(4), 2048] {
            let g = k::Geometry {
                start: visible - count,
                count,
                blocks: 1024,
            };
            if !cases.contains(&g) {
                cases.push(g);
            }
        }
    }
    // Different original row pitches and inactive logical/source-tail spans.
    for g in [
        k::Geometry {
            start: 0,
            count: 1,
            blocks: 1,
        },
        k::Geometry {
            start: 3,
            count: 1,
            blocks: 1,
        },
        k::Geometry {
            start: 3,
            count: 3,
            blocks: 2,
        },
        k::Geometry {
            start: 8,
            count: 3,
            blocks: 3,
        },
        k::Geometry {
            start: 2040,
            count: 8,
            blocks: 512,
        },
    ] {
        cases.push(g);
    }
    cases
}
fn initialize(state: &mut State, g: k::Geometry) -> Result<()> {
    if !g.supported()? {
        return Err("QSA chronological compiled fixture unsupported".into());
    }
    for words in state.iter_mut() {
        words.fill(0);
    }
    for row in 0..g.count {
        let complete = (g.start + row + 1) / 4;
        for block in 0..complete {
            state[0][row * g.blocks + block] = match (block + row) % 5 {
                0 => 0,
                1 => (-0.0f32).to_bits(),
                2 => f32::MAX.to_bits(),
                3 => ((complete - block) as f32).to_bits(),
                _ => ((block % 11) as f32).to_bits(),
            };
            state[1][row * g.blocks + block] = 1;
        }
    }
    Ok(())
}
fn expected_ids(state: &mut State, g: k::Geometry) -> Result<()> {
    for row in 0..g.count {
        let visible = g.start + row + 1;
        let scores = Zeroizing::new(
            state[0][row * g.blocks..row * g.blocks + visible / 4]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect::<Vec<_>>(),
        );
        let ids = Zeroizing::new(kuno_text::ops::qsa_select_blocks(
            &scores, visible, 4, 2048,
        )?);
        if scores.capacity() > 512 || ids.capacity() > 2048 {
            return Err("QSA chronological independent reference capacity rejected".into());
        }
        for (dst, &id) in state[2][row * k::SELECT_STRIDE..][..ids.len()]
            .iter_mut()
            .zip(ids.iter())
        {
            *dst = u32::try_from(id).map_err(|_| "QSA chronological independent ID range")?;
        }
    }
    Ok(())
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
        return Err("QSA chronological owning write extent".into());
    }
    let mut bytes = Upload(Vec::with_capacity(layout.owned_bytes));
    if bytes.0.capacity() != layout.owned_bytes {
        return Err("QSA chronological owning upload capacity".into());
    }
    for word in words {
        bytes.0.extend(word.to_le_bytes());
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arena
            .client()?
            .write_private(arena.handle(id)?.clone(), std::mem::take(&mut bytes.0))
            .map_err(|_| "QSA chronological owning write failed")?;
        Ok::<_, Box<dyn std::error::Error>>(())
    }));
    if !matches!(result, Ok(Ok(()))) {
        retired.store(true, Ordering::Release);
        return Err("QSA chronological owning write retired".into());
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
            Err("QSA chronological complete IO audit retired".into())
        }
    }
}
fn snapshot(arena: &PrivateArena, ids: &[BufferId; 4], retired: &AtomicBool) -> Result<State> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<State> {
        let mut out = state()?;
        for (index, &id) in ids.iter().enumerate() {
            let bytes = Bytes(
                arena
                    .client()?
                    .read_one(arena.handle(id)?.clone())
                    .map_err(|_| "QSA chronological owning read failed")?,
            );
            if bytes.0.len() != OWNED[index] * 4 {
                return Err("QSA chronological owning read extent".into());
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
            return Err("QSA chronological owning read retired".into());
        }
    };
    audit(arena.client()?, retired)?;
    Ok(out)
}
fn physical(arena: &PrivateArena, ids: &[BufferId; 4]) -> Result<()> {
    let mut buffers = Vec::with_capacity(4);
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
            return Err("QSA chronological owning physical metadata".into());
        }
        let resource = arena
            .client()?
            .get_resource(handle.clone())
            .map_err(|_| "QSA chronological complete resource")?;
        let resource = resource.resource();
        if resource.offset != 0
            || resource.size != OWNED[index] as u64 * 4
            || resource.buffer.size() != resource.size
            || buffers.contains(&resource.buffer)
        {
            return Err("QSA chronological physical owner extent or alias".into());
        }
        buffers.push(resource.buffer.clone());
    }
    Ok(())
}

fn launch(arena: &PrivateArena, ids: &[BufferId; 4], g: k::Geometry) -> Result<()> {
    let h = ids
        .map(|id| arena.handle(id).cloned())
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    k::launch(
        arena.client()?,
        h.try_into().map_err(|_| "QSA chronological handle count")?,
        g,
    )?;
    Ok(())
}
fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    Ok(client
        .memory_usage()
        .map_err(|_| "QSA chronological main telemetry")?
        .bytes_reserved)
}
fn allocate(arena: &mut PrivateArena, ledger: &mut MainAdmission) -> Result<[BufferId; 4]> {
    ledger.observe_before(sample(arena.client()?)?)?;
    ledger.precharge(&OWNED.map(|v| v as u64 * 4))?;
    let ids = LOGICAL
        .into_iter()
        .map(|n| arena.allocate_zeroed(n))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "QSA chronological owning allocation count")?;
    if arena.charged_bytes() != PRIVATE_BYTES {
        return Err("QSA chronological full private charge mismatch".into());
    }
    physical(arena, &ids)?;
    ledger.observe_after(sample(arena.client()?)?)?;
    Ok(ids)
}
fn upload(
    arena: &PrivateArena,
    ids: &[BufferId; 4],
    state: &State,
    retired: &AtomicBool,
) -> Result<()> {
    for (index, &id) in ids.iter().enumerate() {
        write(arena, id, &state[index], retired)?;
    }
    Ok(())
}
fn numeric(
    arena: &PrivateArena,
    ids: &[BufferId; 4],
    expected: &mut State,
    g: k::Geometry,
    retired: &AtomicBool,
    ledger: &mut MainAdmission,
) -> Result<()> {
    initialize(expected, g)?;
    upload(arena, ids, expected, retired)?;
    launch(arena, ids, g)?;
    let actual = snapshot(arena, ids, retired)?;
    expected_ids(expected, g)?;
    if actual != *expected {
        return Err("QSA chronological independent IDs source or owning tail mismatch".into());
    }
    ledger.completed_fence(sample(arena.client()?)?)?;
    Ok(())
}
fn mutate(state: &mut State, g: k::Geometry, case: usize) -> Result<()> {
    let last = (g.count - 1) * g.blocks + (g.start + g.count) / 4 - 1;
    match case {
        0 => state[0][0] = f32::NAN.to_bits(),
        1 => state[0][0] = f32::INFINITY.to_bits(),
        2 => state[0][0] = f32::NEG_INFINITY.to_bits(),
        3 => state[0][0] = (-0.5f32).to_bits(),
        4 => state[1][0] = 0,
        5 => state[1][0] = 2,
        6 => state[0][last] = f32::NAN.to_bits(),
        7 => state[1][last] = 0,
        8 => state[0][1] = 1,
        9 => state[1][1] = 1,
        10 => state[0][g.count * g.blocks] = 1,
        11 => state[1][g.count * g.blocks] = 1,
        12 => state[0][OWNED[0] - 1] = 1,
        13 => state[1][OWNED[1] - 1] = 1,
        14 => state[2][1] = 1,
        15 => state[2][k::SELECT_STRIDE - 1] = 1,
        16 => state[2][g.count * k::SELECT_STRIDE] = 1,
        17 => state[2][LOGICAL[2]] = 1,
        18 => state[2][OWNED[2] - 1] = 1,
        19 => state[3][OWNED[3] - 1] = 1,
        20 => state[3][0] = 8,
        21 => {
            state[3][0] = 8;
            state[0][0] = f32::NAN.to_bits();
        }
        22 => state[0][0] = 0xbf800000,
        23 => state[1][last] = u32::MAX,
        _ => return Err("QSA chronological compiled fault case rejected".into()),
    }
    Ok(())
}
/// Parent registers a protected exclusive mode before requesting a device.
/// No subgroup requirement: two fixed256-thread dispatches use original math
/// only in the already independent CPU oracle, never synthetic forcing.
pub(crate) fn run(client: &ComputeClient<WgpuRuntime>, limits: &wgpu::Limits) -> Result<()> {
    validate_configuration()?;
    validate_limits(limits)?;
    let initial = client
        .memory_usage()
        .map_err(|_| "QSA chronological initial telemetry")?;
    if initial.number_allocs != 0
        || initial.bytes_in_use != 0
        || initial.bytes_padding != 0
        || initial.bytes_reserved != 0
    {
        return Err("QSA chronological fresh exclusive context required".into());
    }
    main_admission::check_capacity(MAIN_CAP, &[PRIVATE_BYTES as u64])?;
    client
        .configure_transient_io(IO_CAP)
        .map_err(|_| "QSA chronological bounded IO setup")?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), PRIVATE_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let mut ledger = MainAdmission::new(MAIN_CAP, 0)?;
    let geometry_cases = cases();
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        let ids = allocate(&mut arena, &mut ledger)?;
        let mut expected = state()?;
        for (case, &g) in geometry_cases.iter().enumerate() {
            println!("public_qsa_chronological_stage case={case} stage=numeric-start");
            numeric(&arena, &ids, &mut expected, g, &retired, &mut ledger)?;
        }
        let g = k::Geometry {
            start: 3,
            count: 3,
            blocks: 2,
        };
        for case in 0..FAULT_CASES {
            println!("public_qsa_chronological_stage case={case} stage=fault-start");
            initialize(&mut expected, g)?;
            mutate(&mut expected, g, case)?;
            upload(&arena, &ids, &expected, &retired)?;
            launch(&arena, &ids, g)?;
            let actual = snapshot(&arena, &ids, &retired)?;
            if case != 20 {
                expected[3][0] |= 4;
            }
            if actual != expected {
                return Err("QSA chronological fault changed source or published IDs".into());
            }
            ledger.completed_fence(sample(client)?)?;
        }
        // Independent max-boundary fault hits the LAST active score of row7,
        // not only a conveniently early word/row in the fixed short fixture.
        let g = k::Geometry {
            start: 2040,
            count: 8,
            blocks: 1024,
        };
        initialize(&mut expected, g)?;
        mutate(&mut expected, g, 6)?;
        upload(&arena, &ids, &expected, &retired)?;
        launch(&arena, &ids, g)?;
        let actual = snapshot(&arena, &ids, &retired)?;
        expected[3][0] |= 4;
        if actual != expected {
            return Err("QSA chronological boundary fault published IDs".into());
        }
        ledger.completed_fence(sample(client)?)?;
        arena.reset()?;
        if arena.handle(ids[0]).is_ok() {
            return Err("QSA chronological stale owner accepted after reset".into());
        }
        client.memory_cleanup();
        ledger.completed_fence(sample(client)?)?;
        let fresh = allocate(&mut arena, &mut ledger)?;
        numeric(
            &arena,
            &fresh,
            &mut expected,
            k::Geometry {
                start: 2040,
                count: 8,
                blocks: 1024,
            },
            &retired,
            &mut ledger,
        )?;
        Ok(())
    }));
    let close = arena.close();
    if close.is_err() || retired.load(Ordering::Acquire) {
        return Err("QSA chronological complete owning cleanup failed; context retired".into());
    }
    match attempt {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let class = match error.to_string().as_str() {
                "QSA chronological independent IDs source or owning tail mismatch" => {
                    "independent-selection"
                }
                "QSA chronological fault changed source or published IDs" => "fault-publication",
                "QSA chronological boundary fault published IDs" => "boundary-fault",
                "QSA chronological owning read retired" => "read-owner-retired",
                "QSA chronological complete IO audit retired" => "io-owner-retired",
                _ => "unclassified",
            };
            eprintln!("public_qsa_chronological_failure class={class}");
            return Err("QSA chronological public operator failed; no acceptance receipt".into());
        }
        Err(_) => {
            return Err("QSA chronological public operator failed; no acceptance receipt".into());
        }
    }
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("QSA chronological complete owning zero audit missing")?;
    if zero.buffers != 8 || zero.bytes != 2 * PRIVATE_BYTES as u64 {
        return Err("QSA chronological full erased counts mismatch".into());
    }
    let mut clean = false;
    for _ in 0..2 {
        cubecl::future::block_on(client.sync())
            .map_err(|_| "QSA chronological final completion")?;
        client.memory_cleanup();
        cubecl::future::block_on(client.sync())
            .map_err(|_| "QSA chronological allocator completion")?;
        let m = client
            .memory_usage()
            .map_err(|_| "QSA chronological final telemetry")?;
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
        return Err("QSA chronological complete allocator cleanup unaccepted".into());
    }
    let receipt = serde_json::json!({"qsa_chronological_fixed_public_probe":true,"independent_original_selection_cases":geometry_cases.len()+1,"visible_token_limit":2048,"original_block_size":4,"selected_stride":k::SELECT_STRIDE,"chunk_counts":[1,2,3,5,8],"source_row_blocks":[1,2,3,512,1024],"above_budget_requires_original_cpu":true,"fault_cases":FAULT_CASES+1,"fault_no_id_publication":true,"complete_scores_status_select_fault_owners":true,"fresh_owner_replay":true,"private_owned_bytes":PRIVATE_BYTES,"private_cap_bytes":PRIVATE_CAP,"main_cap_bytes":MAIN_CAP,"io_cap_bytes":IO_CAP,"host_capacity_bound_bytes":HOST_CAP,"extra_model_info_bytes_per_chunk":k::EXTRA_INFO_BYTES_PER_CHUNK,"owned_zero_buffers":zero.buffers,"owned_zero_bytes":zero.bytes,"allocator_cleanup_complete":true,"context_retired":false,"private_request_accepted":false,"whole_model_accepted":false,"performance_accepted":false});
    let path =
        std::env::var_os("KUNO_PUBLIC_RECEIPT").ok_or("QSA chronological new receipt required")?;
    let out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| "QSA chronological new receipt destination rejected")?;
    serde_json::to_writer_pretty(out, &receipt)
        .map_err(|_| "QSA chronological receipt write failed")?;
    println!("{receipt}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn vars() -> Vec<(OsString, OsString)> {
        vec![
            ("KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY".into(), "1".into()),
            ("KUNO_PUBLIC_RECEIPT".into(), "new-receipt".into()),
        ]
    }
    #[test]
    fn standalone_registration_is_exact_exclusive_and_default_only() {
        assert!(configuration(false, vars()).is_ok());
        assert!(configuration(true, vars()).is_err());
        assert!(configuration(false, Vec::new()).is_err());
        for key in [
            "KUNO_LAB_PRIVATE",
            "KUNO_TEST_GGUF",
            "KUNO_PUBLIC_PROMPT",
            "KUNO_LAB_DEVICE_DISPATCH",
        ] {
            let mut v = vars();
            v.push((key.into(), "1".into()));
            assert!(configuration(false, v).is_err());
        }
        for index in 0..2 {
            let mut v = vars();
            v.push(v[index].clone());
            assert!(configuration(false, v).is_err());
        }
        for value in ["", "0", "2", "01", "true", "1 "] {
            let mut v = vars();
            v[0].1 = value.into();
            assert!(configuration(false, v).is_err());
        }
    }
    #[test]
    fn actual_full_owner_geometry_and_independent_original_fallback() {
        assert_eq!(
            LOGICAL.map(|n| BufferLayout::new(n).unwrap().owned_words),
            OWNED
        );
        assert_eq!(OWNED.iter().sum::<usize>() * 4, PRIVATE_BYTES);
        state().unwrap();
        let cases = cases();
        assert!(cases.iter().any(|g| g.start + g.count == 2048));
        for &g in &cases {
            let mut s = state().unwrap();
            initialize(&mut s, g).unwrap();
            expected_ids(&mut s, g).unwrap();
            for row in 0..g.count {
                let visible = g.start + row + 1;
                assert_eq!(
                    &s[2][row * k::SELECT_STRIDE..][..visible],
                    &(0..visible as u32).collect::<Vec<_>>()
                );
            }
        }
        assert!(
            k::Geometry {
                start: 0,
                count: 0,
                blocks: 1024
            }
            .supported()
            .is_err()
        );
        assert!(
            !k::Geometry {
                start: 2048,
                count: 1,
                blocks: 1024
            }
            .supported()
            .unwrap()
        );
        let mut scores = vec![1.0; 513];
        scores[512] = 0.;
        let ids = kuno_text::ops::qsa_select_blocks(&scores, 2052, 4, 2048).unwrap();
        assert_eq!(ids.last(), Some(&2047));
    }
    #[test]
    fn every_fault_fixture_changes_exactly_the_intended_owner_and_boundary() {
        let g = k::Geometry {
            start: 3,
            count: 3,
            blocks: 2,
        };
        for case in 0..FAULT_CASES {
            let mut s = state().unwrap();
            initialize(&mut s, g).unwrap();
            let old: Vec<_> = s.iter().map(|v| v.to_vec()).collect();
            mutate(&mut s, g, case).unwrap();
            assert!(s.iter().zip(&old).any(|(a, b)| **a != *b));
        }
        let g = k::Geometry {
            start: 2040,
            count: 8,
            blocks: 1024,
        };
        let mut s = state().unwrap();
        initialize(&mut s, g).unwrap();
        mutate(&mut s, g, 6).unwrap();
        assert!(f32::from_bits(s[0][7 * 1024 + 511]).is_nan());
    }
}
