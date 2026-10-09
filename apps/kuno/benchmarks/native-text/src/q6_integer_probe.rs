//! Fixed PUBLIC Q6 integer operator acceptance; no model or prompt inputs.
//! Requires root-owned default-build registration, SUBGROUP16..32, exclusive
//! arena pools and physical recovery. No whole-model/speed/private claim.
use crate::{
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, BufferLayout, PrivateArena},
    q6_integer_matvec::{self as k, oracle},
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use std::{
    ffi::{OsStr, OsString},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const GUARD: u32 = 0x3f34_0000;
const PUBLIC_PAD: u8 = 0xa5;
const INPUT: usize = 3;
const PLANES: usize = 5;
const OUTPUT: usize = 7;
const FAULT: usize = 2;
const PACKED: usize = 3;
const PRIVATE_CAP: usize = 1024 * 1024;
const IO_CAP: u64 = 128 * 1024 * 1024;
const MAIN_CAP: u64 = 256 * 1024 * 1024;
// Four full public payload/read leases plus eight bounded private snapshots,
// uploads and erased reference owners; all fixed geometries are checked below.
const HOST_CAP: u64 = 320 * 1024 * 1024;
const CASES: [(usize, usize); 4] = [(7, 256), (257, 512), (16384, 2560), (2560, 2560)];

fn configuration(
    features: bool,
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> std::result::Result<(), &'static str> {
    if features {
        return Err("Q6 integer probe requires default features");
    }
    let mut gate = false;
    let mut receipt = false;
    for (key, value) in vars {
        match key.to_str() {
            Some("KUNO_LAB_Q6_INTEGER_PUBLIC_ONLY") => gate = value == OsStr::new("1"),
            Some("KUNO_PUBLIC_RECEIPT") => receipt = !value.is_empty(),
            _ => {
                let name = key.to_string_lossy();
                if name.starts_with("KUNO_LAB_")
                    || name.starts_with("KUNO_TEST_")
                    || name.starts_with("KUNO_PUBLIC_")
                {
                    return Err("Q6 integer inherited setting rejected");
                }
            }
        }
    }
    if !gate || !receipt {
        return Err("Q6 integer fixed public gate and new receipt required");
    }
    Ok(())
}
/// Call before any device/model access. This mode has no model path or text.
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
    if l.max_compute_invocations_per_workgroup < 128
        || l.max_compute_workgroup_size_x < 128
        || l.max_storage_buffers_per_shader_stage < 4
        || l.max_bindings_per_bind_group < 5
        || l.max_bind_groups < 1
        || l.max_uniform_buffer_binding_size < 32
        || l.max_compute_workgroups_per_dimension < 4096
        || l.max_storage_buffer_binding_size < 64 * 1024 * 1024
        || l.max_buffer_size < 64 * 1024 * 1024
    {
        return Err("Q6 integer actual device limits".into());
    }
    Ok(())
}
struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
struct Read(cubecl::bytes::Bytes);
impl Drop for Read {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn write(arena: &PrivateArena, id: BufferId, words: &[u32], retired: &AtomicBool) -> Result<()> {
    let l = arena.layout(id)?;
    if words.len() != l.owned_words {
        return Err("Q6 owning write extent".into());
    }
    let mut bytes = Upload(Vec::with_capacity(l.owned_bytes));
    for word in words {
        bytes.0.extend(word.to_le_bytes());
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arena
            .client()?
            .write_private(arena.handle(id)?.clone(), std::mem::take(&mut bytes.0))
            .map_err(|_| "Q6 owning write failed")?;
        Ok::<_, Box<dyn std::error::Error>>(())
    }));
    if !matches!(result, Ok(Ok(()))) {
        retired.store(true, Ordering::Release);
        return Err("Q6 owning write failed; context retired".into());
    }
    Ok(())
}
fn read(
    client: &ComputeClient<WgpuRuntime>,
    handle: Handle,
    bytes: usize,
    retired: &AtomicBool,
) -> Result<Zeroizing<Vec<u32>>> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let data = Read(
            client
                .read_one(handle)
                .map_err(|_| "Q6 completed owning read")?,
        );
        if data.0.len() != bytes || !bytes.is_multiple_of(4) {
            return Err("Q6 owning read physical extent".into());
        }
        Ok::<_, Box<dyn std::error::Error>>(Zeroizing::new(
            data.0
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect(),
        ))
    }));
    match result {
        Ok(Ok(data)) => Ok(data),
        _ => {
            retired.store(true, Ordering::Release);
            Err("Q6 owning read failed; context retired".into())
        }
    }
}
/// Exact existing transient-I/O checkpoint policy; no ordinary private fallback.
fn audit(client: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<()> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io_audited())
    }));
    let r = match result {
        Ok(Ok(r)) => r,
        _ => {
            retired.store(true, Ordering::Release);
            return Err("Q6 checked I/O cleanup failed; context retired".into());
        }
    };
    let total = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes));
    if total.is_none_or(|n| n == 0 || n > IO_CAP || n != r.inspected_bytes)
        || r.inspected_buffers == 0
        || retired.load(Ordering::Acquire)
    {
        retired.store(true, Ordering::Release);
        return Err("Q6 malformed I/O audit; context retired".into());
    }
    Ok(())
}
fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    Ok(client
        .memory_usage()
        .map_err(|_| "Q6 public memory telemetry")?
        .bytes_reserved)
}
fn physical(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    ids: &[BufferId; 4],
    weights: &Handle,
    weight_bytes: usize,
) -> Result<()> {
    let stream = weights.stream;
    let mut buffers = Vec::with_capacity(5);
    for (h, n) in ids
        .iter()
        .map(|&id| Ok((arena.handle(id)?.clone(), arena.layout(id)?.owned_bytes)))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .chain(std::iter::once((weights.clone(), weight_bytes)))
    {
        if h.stream != stream || h.size_in_used() != n as u64 {
            return Err("Q6 complete owner stream or extent".into());
        }
        let r = client
            .get_resource(h)
            .map_err(|_| "Q6 complete owner resource")?;
        let r = r.resource();
        if r.offset != 0
            || r.size != n as u64
            || r.buffer.size() != n as u64
            || buffers.contains(&r.buffer)
        {
            return Err("Q6 disjoint full physical owner proof".into());
        }
        buffers.push(r.buffer.clone());
    }
    Ok(())
}
fn public_unchanged(
    client: &ComputeClient<WgpuRuntime>,
    weights: &Handle,
    wire: &[u8],
    retired: &AtomicBool,
) -> Result<()> {
    let data = read(client, weights.clone(), wire.len(), retired)?;
    if data
        .iter()
        .zip(wire.chunks_exact(4))
        .any(|(&a, b)| a != u32::from_le_bytes(b.try_into().unwrap()))
    {
        return Err("Q6 original public bytes or full padding changed".into());
    }
    drop(data);
    audit(client, retired)
}
fn snapshot(
    arena: &PrivateArena,
    ids: &[BufferId; 4],
    retired: &AtomicBool,
) -> Result<[Zeroizing<Vec<u32>>; 4]> {
    let values: Vec<_> = ids
        .iter()
        .map(|&id| {
            read(
                arena.client()?,
                arena.handle(id)?.clone(),
                arena.layout(id)?.owned_bytes,
                retired,
            )
        })
        .collect::<Result<_>>()?;
    let values = values.try_into().map_err(|_| "Q6 snapshot owner count")?;
    audit(arena.client()?, retired)?;
    Ok(values)
}
fn same_outside(before: &[u32], after: &[u32], first: usize, count: usize) -> bool {
    before.len() == after.len()
        && before[..first] == after[..first]
        && before[first + count..] == after[first + count..]
}
fn run_sequence(
    client: &ComputeClient<WgpuRuntime>,
    spec: k::Spec,
    weights: &Handle,
    wire: &[u8],
    ledger: &mut MainAdmission,
    retired: Arc<AtomicBool>,
    faults: bool,
) -> Result<(Zeroizing<Vec<u32>>, f32, u64, u64)> {
    let logical = [
        INPUT + spec.columns() + 11,
        PLANES + spec.plane_words() + 13,
        OUTPUT + spec.rows() + 17,
        FAULT + 1 + 19,
    ];
    let layouts: Vec<_> = logical
        .into_iter()
        .map(BufferLayout::new)
        .collect::<std::result::Result<_, _>>()?;
    let requests: Vec<_> = layouts.iter().map(|l| l.owned_bytes as u64).collect();
    if requests.iter().sum::<u64>() > PRIVATE_CAP as u64 {
        return Err("Q6 private owner cap".into());
    }
    ledger.observe_before(sample(client)?)?;
    ledger.precharge(&requests)?;
    let mut arena =
        PrivateArena::new_transient(client.clone(), PRIVATE_CAP, IO_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let ids: Vec<_> = logical
        .into_iter()
        .map(|n| arena.allocate_zeroed(n))
        .collect::<std::result::Result<_, _>>()?;
    let ids: [BufferId; 4] = ids.try_into().map_err(|_| "Q6 private owner count")?;
    physical(client, &arena, &ids, weights, wire.len())?;
    ledger.observe_after(sample(client)?)?;
    let input = oracle::public_input(spec.columns(), 0);
    let planes = oracle::Planes::new(&input)?;
    let expected_planes = planes.words();
    let mut before: Vec<_> = layouts
        .iter()
        .map(|l| Zeroizing::new(vec![GUARD; l.owned_words]))
        .collect();
    for (dst, x) in before[0][INPUT..INPUT + spec.columns()]
        .iter_mut()
        .zip(input.iter())
    {
        *dst = x.to_bits();
    }
    before[3][FAULT] = 0;
    for (i, &id) in ids.iter().enumerate() {
        write(&arena, id, &before[i], &retired)?;
    }
    let prepare = k::PrepareSpans {
        input_first: INPUT,
        planes_first: PLANES,
        fault_first: FAULT,
    };
    let project = k::ProjectSpans {
        packed_first: PACKED,
        planes_first: PLANES,
        output_first: OUTPUT,
        fault_first: FAULT,
        matrix_first: OUTPUT,
        matrix_rows: spec.rows(),
    };
    k::prepare(
        client,
        spec,
        arena.handle(ids[0])?.clone(),
        arena.handle(ids[1])?.clone(),
        arena.handle(ids[3])?.clone(),
        prepare,
    )?;
    k::project(
        client,
        spec,
        weights.clone(),
        arena.handle(ids[1])?.clone(),
        arena.handle(ids[2])?.clone(),
        arena.handle(ids[3])?.clone(),
        project,
        true,
    )?;
    let after = snapshot(&arena, &ids, &retired)?;
    ledger.completed_fence(sample(client)?)?;
    if before[0] != after[0]
        || after[3][FAULT] != 0
        || !same_outside(&before[1], &after[1], PLANES, spec.plane_words())
        || !same_outside(&before[2], &after[2], OUTPUT, spec.rows())
        || !same_outside(&before[3], &after[3], FAULT, 1)
        || after[1][PLANES..PLANES + spec.plane_words()] != *expected_planes
    {
        return Err("Q6 prepared planes or complete owning guards rejected".into());
    }
    let mut maximum = 0f32;
    for (row, &bits) in after[2][OUTPUT..OUTPUT + spec.rows()].iter().enumerate() {
        let start = PACKED + row * spec.row_bytes();
        let expected = oracle::project(&wire[start..start + spec.row_bytes()], &planes)?;
        let actual = f32::from_bits(bits);
        let error = (actual - expected).abs();
        if !actual.is_finite() || error > 2e-4 * (1. + expected.abs()) {
            return Err("Q6 alternate integer full-row numeric acceptance".into());
        }
        maximum = maximum.max(error);
    }
    let output = Zeroizing::new(after[2][OUTPUT..OUTPUT + spec.rows()].to_vec());
    drop(after);
    if faults {
        // Invalid finite/MAX and NaN input must set fault BEFORE project, leaving
        // every output word unchanged. Previous fault flags are never cleared.
        for value in [f32::NAN, f32::MAX] {
            before[0][INPUT + spec.columns() - 1] = value.to_bits();
            before[3][FAULT] = 0;
            for (i, &id) in ids.iter().enumerate() {
                write(&arena, id, &before[i], &retired)?;
            }
            k::prepare(
                client,
                spec,
                arena.handle(ids[0])?.clone(),
                arena.handle(ids[1])?.clone(),
                arena.handle(ids[3])?.clone(),
                prepare,
            )?;
            k::project(
                client,
                spec,
                weights.clone(),
                arena.handle(ids[1])?.clone(),
                arena.handle(ids[2])?.clone(),
                arena.handle(ids[3])?.clone(),
                project,
                true,
            )?;
            let after = snapshot(&arena, &ids, &retired)?;
            if after[0] != before[0]
                || after[2] != before[2]
                || after[3][FAULT] != 1
                || !same_outside(&before[1], &after[1], PLANES, spec.plane_words())
                || !same_outside(&before[3], &after[3], FAULT, 1)
            {
                return Err("Q6 bad-input fault or complete no-write gate".into());
            }
            let block = spec.columns() / 32 - 1;
            let packed = spec.columns() / 4;
            let blocks = spec.columns() / 32;
            if after[1][PLANES + block * 8..PLANES + block * 8 + 8]
                .iter()
                .any(|&v| v != 0)
                || after[1][PLANES + packed + block * 8..PLANES + packed + block * 8 + 8]
                    .iter()
                    .any(|&v| v != 0)
                || after[1][PLANES + packed * 2 + block] != 0
                || after[1][PLANES + packed * 2 + blocks + block] != 0
            {
                return Err("Q6 invalid activation block was not zeroed".into());
            }
            ledger.completed_fence(sample(client)?)?;
        }
        before[0][INPUT + spec.columns() - 1] = input[spec.columns() - 1].to_bits();
        before[3][FAULT] = 8;
        for (i, &id) in ids.iter().enumerate() {
            write(&arena, id, &before[i], &retired)?;
        }
        k::prepare(
            client,
            spec,
            arena.handle(ids[0])?.clone(),
            arena.handle(ids[1])?.clone(),
            arena.handle(ids[3])?.clone(),
            prepare,
        )?;
        k::project(
            client,
            spec,
            weights.clone(),
            arena.handle(ids[1])?.clone(),
            arena.handle(ids[2])?.clone(),
            arena.handle(ids[3])?.clone(),
            project,
            true,
        )?;
        let after = snapshot(&arena, &ids, &retired)?;
        if after[2] != before[2] || after[3] != before[3] {
            return Err("Q6 previous fault cleared or output published".into());
        }
        drop(after);
        ledger.completed_fence(sample(client)?)?;
        if spec.rows() == 7 && spec.columns() == 256 {
            // A non-finite ORIGINAL half is a rejected owning transaction.
            // Other active rows may have computed before the global fault;
            // no output is published, and all unrelated owning words survive.
            let mut bad_wire = wire.to_vec();
            bad_wire[PACKED + 208..PACKED + 210].copy_from_slice(&0x7e01u16.to_le_bytes());
            ledger.observe_before(sample(client)?)?;
            ledger.precharge(&[bad_wire.len() as u64])?;
            let bad_weights = client.create_public_owned(bad_wire.clone());
            ledger.observe_after(sample(client)?)?;
            physical(client, &arena, &ids, &bad_weights, bad_wire.len())?;
            before[2].fill(GUARD);
            before[3][FAULT] = 0;
            for (i, &id) in ids.iter().enumerate() {
                write(&arena, id, &before[i], &retired)?;
            }
            k::prepare(
                client,
                spec,
                arena.handle(ids[0])?.clone(),
                arena.handle(ids[1])?.clone(),
                arena.handle(ids[3])?.clone(),
                prepare,
            )?;
            k::project(
                client,
                spec,
                bad_weights.clone(),
                arena.handle(ids[1])?.clone(),
                arena.handle(ids[2])?.clone(),
                arena.handle(ids[3])?.clone(),
                project,
                true,
            )?;
            let after = snapshot(&arena, &ids, &retired)?;
            if after[3][FAULT] != 1
                || after[0] != before[0]
                || !same_outside(&before[1], &after[1], PLANES, spec.plane_words())
                || !same_outside(&before[2], &after[2], OUTPUT, spec.rows())
                || !same_outside(&before[3], &after[3], FAULT, 1)
            {
                return Err("Q6 malformed original half fault rejected".into());
            }
            drop(after);
            public_unchanged(client, &bad_weights, &bad_wire, &retired)?;
            drop(bad_weights);
            ledger.completed_fence(sample(client)?)?;
        }
        before[3][FAULT] = 0;
        before[2][0] = f32::NAN.to_bits();
        for (i, &id) in ids.iter().enumerate() {
            write(&arena, id, &before[i], &retired)?;
        }
        k::prepare(
            client,
            spec,
            arena.handle(ids[0])?.clone(),
            arena.handle(ids[1])?.clone(),
            arena.handle(ids[3])?.clone(),
            prepare,
        )?;
        k::project(
            client,
            spec,
            weights.clone(),
            arena.handle(ids[1])?.clone(),
            arena.handle(ids[2])?.clone(),
            arena.handle(ids[3])?.clone(),
            project,
            true,
        )?;
        let after = snapshot(&arena, &ids, &retired)?;
        if after[3][FAULT] != 1
            || after[0] != before[0]
            || !same_outside(&before[2], &after[2], OUTPUT, spec.rows())
            || !same_outside(&before[3], &after[3], FAULT, 1)
        {
            return Err("Q6 full output padding finite fault gate".into());
        }
        drop(after);
        ledger.completed_fence(sample(client)?)?;
    }
    public_unchanged(client, weights, wire, &retired)?;
    arena.reset()?;
    public_unchanged(client, weights, wire, &retired)?;
    arena.close()?;
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("Q6 complete owned zero acceptance")?;
    ledger.completed_fence(sample(client)?)?;
    Ok((output, maximum, zero.bytes, zero.buffers))
}
/// Caller must enable SUBGROUP with actual16..32 bounds and exclusive pools.
/// No GPU initialization or public/private production route lives in this leaf.
pub(crate) fn run(client: &ComputeClient<WgpuRuntime>, limits: &wgpu::Limits) -> Result<()> {
    validate_configuration()?;
    validate_limits(limits)?;
    let c_values = oracle::validate_fixture()?;
    let baseline = client
        .memory_usage()
        .map_err(|_| "Q6 initial main telemetry")?;
    if baseline.number_allocs != 0
        || baseline.bytes_in_use != 0
        || baseline.bytes_padding != 0
        || baseline.bytes_reserved != 0
    {
        return Err("Q6 fresh exclusive main context required".into());
    }
    let retired = Arc::new(AtomicBool::new(false));
    let mut ledger = MainAdmission::new(MAIN_CAP, 0)?;
    let mut maximum = 0f32;
    let mut zero_bytes = 0u64;
    let mut zero_buffers = 0u64;
    let mut rows = 0usize;
    for (n, c) in CASES {
        let spec = k::Spec::new(n, c)?;
        let payload = oracle::public_rows(n, c)?;
        let owned = main_admission::page_bytes((PACKED + payload.len()) as u64)? as usize;
        main_admission::check_capacity(MAIN_CAP, &[owned as u64, PRIVATE_CAP as u64])?;
        main_admission::check_capacity(HOST_CAP, &[owned as u64 * 4, PRIVATE_CAP as u64 * 8])?;
        let mut wire = vec![PUBLIC_PAD; owned];
        wire[PACKED..PACKED + payload.len()].copy_from_slice(&payload);
        drop(payload);
        ledger.observe_before(sample(client)?)?;
        ledger.precharge(&[owned as u64])?;
        let weights = client.create_public_owned(wire.clone());
        ledger.observe_after(sample(client)?)?;
        let (first, error, bytes, buffers) = run_sequence(
            client,
            spec,
            &weights,
            &wire,
            &mut ledger,
            retired.clone(),
            true,
        )?;
        maximum = maximum.max(error);
        zero_bytes += bytes;
        zero_buffers += buffers;
        let (second, error, bytes, buffers) = run_sequence(
            client,
            spec,
            &weights,
            &wire,
            &mut ledger,
            retired.clone(),
            false,
        )?;
        if first != second {
            return Err("Q6 fresh-owner full output bit replay".into());
        }
        maximum = maximum.max(error);
        zero_bytes += bytes;
        zero_buffers += buffers;
        rows += n * 2;
        drop(weights);
        client.memory_cleanup();
        cubecl::future::block_on(client.sync())
            .map_err(|_| "Q6 checked public-source completion")?;
        ledger.completed_fence(sample(client)?)?;
    }
    let mut cleanup = false;
    for _ in 0..2 {
        cubecl::future::block_on(client.sync()).map_err(|_| "Q6 checked final completion")?;
        client.memory_cleanup();
        cubecl::future::block_on(client.sync())
            .map_err(|_| "Q6 checked allocator cleanup completion")?;
        let actual = client
            .memory_usage()
            .map_err(|_| "Q6 checked allocator telemetry")?;
        if actual.number_allocs == 0
            && actual.bytes_in_use == 0
            && actual.bytes_padding == 0
            && actual.bytes_reserved == 0
        {
            cleanup = true;
            break;
        }
    }
    if !cleanup || retired.load(Ordering::Acquire) {
        return Err("Q6 allocator release not accepted".into());
    }
    let receipt = serde_json::json!({"q6_integer_fixed_public_probe":true,"gguf_id":14,"original_weight_bytes":true,"activation_two_residual_q8_planes":true,"c_reference_finite_values":c_values,"geometries":CASES.len(),"full_rows_verified":rows,"fresh_owner_replay_bit_mismatches":0,"max_abs_alternate_integer_error":maximum,"relative_tolerance":2e-4,"input_and_full_owner_guards_unchanged":true,"nonfinite_and_max_input_rejected":true,"invalid_activation_block_zeroed":true,"malformed_original_half_rejected":true,"prior_fault_preserved":true,"whole_output_padding_finite_gate":true,"original_public_bytes_and_padding_unchanged":true,"owned_zero_audit_bytes":zero_bytes,"owned_zero_audit_buffers":zero_buffers,"allocator_cleanup_complete":cleanup,"main_cap_bytes":MAIN_CAP,"private_cap_bytes":PRIVATE_CAP,"io_cap_bytes":IO_CAP,"host_capacity_bound_bytes":HOST_CAP,"whole_owning_erasure_accepted":false,"private_request_accepted":false,"original_f32_numerical_parity_accepted":false,"whole_model_accepted":false,"performance_accepted":false});
    let path =
        std::env::var_os("KUNO_PUBLIC_RECEIPT").ok_or("Q6 numeric receipt destination missing")?;
    let output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| "Q6 new numeric receipt destination rejected")?;
    serde_json::to_writer_pretty(output, &receipt)
        .map_err(|_| "Q6 numeric receipt write failed")?;
    println!("{receipt}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn vars() -> Vec<(OsString, OsString)> {
        vec![
            ("KUNO_LAB_Q6_INTEGER_PUBLIC_ONLY".into(), "1".into()),
            ("KUNO_PUBLIC_RECEIPT".into(), "new-public-receipt".into()),
        ]
    }
    #[test]
    fn fixed_gate_rejects_other_settings_features_and_missing_inputs() {
        assert!(configuration(false, vars()).is_ok());
        assert!(configuration(true, vars()).is_err());
        assert!(configuration(false, Vec::new()).is_err());
        for key in ["KUNO_LAB_FOO", "KUNO_TEST_GGUF", "KUNO_PUBLIC_PROMPT"] {
            let mut v = vars();
            v.push((key.into(), "0".into()));
            assert!(configuration(false, v).is_err());
        }
        let mut v = vars();
        v[0].1 = "0".into();
        assert!(configuration(false, v).is_err());
    }
    #[test]
    fn original_tiles_and_full_private_pages_fit_declared_caps() {
        for (n, c) in CASES {
            let s = k::Spec::new(n, c).unwrap();
            let weight = main_admission::page_bytes((PACKED + s.weight_bytes()) as u64).unwrap();
            let private = [
                INPUT + c + 11,
                PLANES + s.plane_words() + 13,
                OUTPUT + n + 17,
                FAULT + 20,
            ]
            .into_iter()
            .map(|w| BufferLayout::new(w).unwrap().owned_bytes)
            .sum::<usize>();
            assert!(private <= PRIVATE_CAP);
            main_admission::check_capacity(MAIN_CAP, &[weight, private as u64]).unwrap();
            assert!(weight <= 64 * 1024 * 1024);
            main_admission::check_capacity(HOST_CAP, &[weight * 4, PRIVATE_CAP as u64 * 8])
                .unwrap();
        }
        assert!(same_outside(&[1, 2, 3], &[1, 9, 3], 1, 1));
        assert!(!same_outside(&[1, 2, 3], &[8, 9, 3], 1, 1));
    }
}
