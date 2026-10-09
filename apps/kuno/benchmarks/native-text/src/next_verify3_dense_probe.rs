//! Compiled public Verify3 projection calibration; no model or request inputs.
//! Parent supplies the exclusive, required-memory-configuration client after
//! its actual SUBGROUP16..32/default-build gate. No device is initialized here.
//! CPU decoded original weights, f64 products and the unchanged shared count1
//! GPU projector independently check every active row. Complete MAX8 owners,
//! source pages, inactive regions and transient I/O retain their own checks.

use crate::main_admission::{self, MainAdmission};
use crate::next_shared::{self, MatrixSpec};
use crate::private_arena::{BufferLayout, PrivateArena};
use crate::quant_matvec::{iq_fixtures, matmul::BatchSpans};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const TYPES: [u32; 10] = [0, 1, 30, 6, 8, 12, 13, 14, 20, 23];
const WIDTHS: [usize; 2] = [256, 2560];
const ROWS: usize = 9; // Both subgroup widths exercise a partial final group.
const MAX8: usize = 8;
const SEED: u32 = 4;
const MAIN_CAP: u64 = 8 * 1024 * 1024;
const PRIVATE_CAP: usize = 512 * 1024;
const IO_CAP: u64 = 64 * 1024 * 1024;
const HOST_CAP: u64 = 32 * 1024 * 1024;
const ABS: f64 = 1e-4;
const REL: f64 = 3e-4;
pub(crate) fn registration(
    args: &[std::ffi::OsString],
    env: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    default: bool,
) -> std::result::Result<std::path::PathBuf, &'static str> {
    use std::{collections::BTreeMap, ffi::OsStr};
    if !default
        || args.len() != 3
        || !matches!(args[1].to_str(), Some("intel" | "nvidia"))
        || args[2] != "verify3-dense-probe"
    {
        return Err("Verify3 operator requires fixed default public card fixture");
    }
    let mut selected = BTreeMap::new();
    for (key, value) in env {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|prefix| key.as_encoded_bytes().starts_with(prefix))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some("KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY" | "KUNO_PUBLIC_RECEIPT")
        ) || value.is_empty()
            || selected.insert(key, value).is_some()
        {
            return Err("Verify3 operator rejects inherited experiments");
        }
    }
    if selected.remove(OsStr::new("KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY")) != Some("1".into()) {
        return Err("Verify3 operator public gate missing");
    }
    selected
        .remove(OsStr::new("KUNO_PUBLIC_RECEIPT"))
        .map(Into::into)
        .ok_or("Verify3 operator numeric receipt missing")
}

const NORMAL_CASES: usize = TYPES.len() * WIDTHS.len() * 3;
const FAULT_CASES: usize = TYPES.len() * 2;
const FUSED_FAULT_KINDS: [usize; 7] = [1, 2, 3, 4, 5, 7, 8];
const FUSED_FAULT_CASES: usize = TYPES.len() * 3 * FUSED_FAULT_KINDS.len();

#[derive(Clone, Copy)]
struct Layout {
    input_first: usize,
    input_stride: usize,
    input_words: usize,
    output_first: usize,
    output_stride: usize,
    output_words: usize,
}
fn layout(columns: usize, count: usize) -> Result<Layout> {
    if !WIDTHS.contains(&columns) || !(1..=3).contains(&count) {
        return Err("Verify3 fixed geometry/count".into());
    }
    let x = Layout {
        input_first: 4,
        input_stride: columns + 12,
        input_words: 4 + MAX8 * (columns + 12) + 12,
        output_first: 7,
        output_stride: ROWS + 11,
        output_words: 7 + MAX8 * (ROWS + 11) + 13,
    };
    for (first, stride, width, words) in [
        (x.input_first, x.input_stride, columns, x.input_words),
        (x.output_first, x.output_stride, ROWS, x.output_words),
    ] {
        crate::resident_batch_projection::validate_view(count, first, stride, width, words)?;
    }
    Ok(x)
}

struct Fixture {
    dtype: u32,
    columns: usize,
    prefix: usize,
    root: Handle,
    view: Handle,
    wire: Zeroizing<Vec<u8>>,
    decoded: Zeroizing<Vec<f32>>,
}

/// Small literal encodings or pinned public C blocks; never reads a GGUF.
fn packed(dtype: u32, columns: usize) -> Result<Zeroizing<Vec<u8>>> {
    if !TYPES.contains(&dtype) || !WIDTHS.contains(&columns) {
        return Err("Verify3 fixed original format".into());
    }
    if matches!(dtype, 13 | 20 | 23) {
        return Ok(Zeroizing::new(iq_fixtures::packed(dtype, ROWS, columns, 0)));
    }
    let kind = kuno_text::TensorType::from_gguf_id(dtype)?;
    let bytes = kind.block_bytes() as usize;
    let width = kind.block_elements() as usize;
    let mut out = Zeroizing::new(Vec::with_capacity(ROWS * columns / width * bytes));
    for block in 0..ROWS * columns / width {
        if dtype == 14 {
            // Original GGML b11232 independent Q6 records, first16 finite
            // blocks cover all signed subscales/codes (existing MIT NOTICE).
            let record = crate::q6_integer_matvec::oracle::FIXTURE;
            let at = (block % 16) * crate::q6_integer_matvec::oracle::RECORD_BYTES;
            out.extend_from_slice(&record[at..at + bytes]);
            continue;
        }
        let mut q = vec![0; bytes];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (block * 37 + i * 29 + 11) as u8;
        }
        match dtype {
            0 => {
                let value = ((block * 17 % 127) as f32 - 63.) / 64.;
                q.copy_from_slice(&value.to_le_bytes());
            }
            1 => {
                let value = ((block as u16 & 1) << 15) | 0x3400 | (block as u16 & 1023);
                q.copy_from_slice(&value.to_le_bytes());
            }
            30 => {
                let value = ((block as u16 & 1) << 15) | 0x3e80 | (block as u16 & 127);
                q.copy_from_slice(&value.to_le_bytes());
            }
            6 | 8 => q[..2].copy_from_slice(&0x2400_u16.to_le_bytes()),
            12 => {
                q[..2].copy_from_slice(&0x2400_u16.to_le_bytes());
                q[2..4].copy_from_slice(&0x2000_u16.to_le_bytes());
            }
            _ => return Err("Verify3 fixed encoder inventory".into()),
        }
        out.extend_from_slice(&q);
    }
    Ok(out)
}

fn decoded(dtype: u32, columns: usize, payload: &[u8]) -> Result<Zeroizing<Vec<f32>>> {
    let mut values = Zeroizing::new(vec![0.; ROWS * columns]);
    kuno_text::TensorType::from_gguf_id(dtype)?.decode(payload, &mut values)?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err("Verify3 public decoded fixture finite extent".into());
    }
    // Exact independent C bits supplement the CPU decoder for added formats.
    let independent = if matches!(dtype, 13 | 20 | 23) {
        Some(Zeroizing::new(iq_fixtures::oracle(dtype, ROWS, columns, 0)))
    } else if dtype == 14 {
        let record = crate::q6_integer_matvec::oracle::FIXTURE;
        let mut expected = Zeroizing::new(Vec::with_capacity(values.len()));
        for block in 0..ROWS * columns / 256 {
            let at = (block % 16) * crate::q6_integer_matvec::oracle::RECORD_BYTES + 210;
            expected.extend(record[at..at + 1024].chunks_exact(4).map(|b| {
                f32::from_le_bytes(b.try_into().expect("compiled public C record width"))
            }));
        }
        Some(expected)
    } else {
        None
    };
    if independent.is_some_and(|x| {
        x.len() != values.len()
            || x.iter()
                .zip(values.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
    }) {
        return Err("Verify3 CPU decoder differs from pinned public C bits".into());
    }
    Ok(values)
}

fn public_fixture(
    client: &ComputeClient<WgpuRuntime>,
    dtype: u32,
    columns: usize,
) -> Result<Fixture> {
    let payload = packed(dtype, columns)?;
    let values = decoded(dtype, columns, &payload)?;
    let prefix = if columns == 256 { 0 } else { 256 };
    let logical_bytes = payload.len().div_ceil(4) * 4;
    let owned = main_admission::page_bytes((prefix + logical_bytes) as u64)? as usize;
    let mut wire = Zeroizing::new(vec![0x53; owned]);
    wire[prefix..prefix + payload.len()].copy_from_slice(&payload);
    let root = client.create_public_owned(wire.to_vec());
    let view = root
        .clone()
        .offset_start(prefix as u64)
        .offset_end((owned - prefix - logical_bytes) as u64);
    Ok(Fixture {
        dtype,
        columns,
        prefix,
        root,
        view,
        wire,
        decoded: values,
    })
}

fn input(x: Layout, columns: usize, count: usize, bad: usize) -> Zeroizing<Vec<f32>> {
    let mut values = Zeroizing::new(vec![0.; x.input_words]);
    for token in 0..count {
        for col in 0..columns {
            values[x.input_first + token * x.input_stride + col] =
                ((col * 17 + token * 13) % 97) as f32 / 64. - 0.75;
        }
    }
    if matches!(bad, 1 | 7 | 8) {
        values[x.input_first + (count - 1) * x.input_stride + columns - 1] = match bad {
            7 => f32::INFINITY,
            8 => f32::NEG_INFINITY,
            _ => f32::NAN,
        };
    }
    values
}
fn input_bits(values: &[f32]) -> Zeroizing<Vec<u32>> {
    // Only compiled public negative fixtures may carry NaN. The ordinary
    // float uploader intentionally rejects them; original owned word uploads
    // preserve the exact fixture bits without changing its admission policy.
    Zeroizing::new(values.iter().map(|x| x.to_bits()).collect())
}
fn expected(decoded: &[f32], columns: usize, x: Layout, input: &[f32], count: usize) -> Vec<f64> {
    (0..count)
        .flat_map(|token| {
            (0..ROWS).map(move |row| {
                decoded[row * columns..(row + 1) * columns]
                    .iter()
                    .zip(&input[x.input_first + token * x.input_stride..][..columns])
                    .map(|(&w, &a)| f64::from(w) * f64::from(a))
                    .sum::<f64>()
            })
        })
        .collect()
}
fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(
        bytes[i * 4..i * 4 + 4]
            .try_into()
            .expect("checked owning width"),
    )
}
fn check_input(bytes: &[u8], values: &[f32], owned: BufferLayout) -> Result<()> {
    if bytes.len() != owned.owned_bytes || values.len() != owned.logical_words {
        return Err("Verify3 full input owner extent".into());
    }
    for i in 0..owned.owned_words {
        if word(bytes, i)
            != if i < values.len() {
                values[i].to_bits()
            } else {
                0
            }
        {
            return Err("Verify3 input prefix/active/inactive/gap/padding changed".into());
        }
    }
    Ok(())
}
fn check_output(
    bytes: &[u8],
    x: Layout,
    count: usize,
    bad: usize,
    references: &[f64],
    owned: BufferLayout,
) -> Result<(Zeroizing<Vec<f32>>, f64)> {
    if bytes.len() != owned.owned_bytes
        || owned.logical_words != x.output_words
        || references.len() != count * ROWS
    {
        return Err("Verify3 full output owner extent".into());
    }
    let mut active = Zeroizing::new(vec![0.; count * ROWS]);
    let mut maximum = 0f64;
    for i in 0..owned.owned_words {
        let relative = i.saturating_sub(x.output_first);
        let token = relative / x.output_stride;
        let row = relative % x.output_stride;
        if i >= x.output_first && token < count && row < ROWS {
            let actual = f32::from_bits(word(bytes, i));
            let reference = references[token * ROWS + row];
            if matches!(bad, 1 | 7 | 8) && token == count - 1 {
                if actual.is_finite() || bad == 1 && !actual.is_nan() {
                    return Err("Verify3 active NaN input escaped finite gate".into());
                }
            } else {
                let error = (f64::from(actual) - reference).abs();
                if !actual.is_finite()
                    || !reference.is_finite()
                    || error > ABS + REL * reference.abs()
                {
                    return Err("Verify3 complete CPU decoded projection mismatch".into());
                }
                maximum = maximum.max(error);
            }
            active[token * ROWS + row] = actual;
        } else {
            let required = bad_output_marker(x, count, bad)
                .filter(|&(at, _)| at == i)
                .map_or(0, |(_, bits)| bits);
            if word(bytes, i) != required {
                return Err("Verify3 output prefix/inactive/gap/full padding changed".into());
            }
        }
    }
    Ok((active, maximum))
}
fn bad_output_marker(x: Layout, count: usize, bad: usize) -> Option<(usize, u32)> {
    match bad {
        2 => Some((x.output_words - 1, f32::NAN.to_bits())),
        3 => Some((0, f32::NAN.to_bits())),
        4 => Some((x.output_first + ROWS, f32::INFINITY.to_bits())),
        5 => Some((
            x.output_first + count * x.output_stride,
            f32::NEG_INFINITY.to_bits(),
        )),
        _ => None,
    }
}
fn check_fault(bytes: &[u8], owned: BufferLayout, bad: usize) -> Result<()> {
    if bytes.len() != owned.owned_bytes
        || owned.logical_words != 1
        || word(bytes, 0) != (SEED | u32::from(bad != 0))
        || bytes[4..].iter().any(|&b| b != 0)
    {
        return Err("Verify3 sticky fault/full padding mismatch".into());
    }
    Ok(())
}
struct Read(cubecl::bytes::Bytes);
impl Drop for Read {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn read(client: &ComputeClient<WgpuRuntime>, handle: &Handle) -> Result<Read> {
    Ok(Read(
        client
            .read_one(handle.clone())
            .map_err(|_| "Verify3 full owning read failed")?,
    ))
}
fn audit(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let r = cubecl::future::block_on(client.sanitize_transient_io_audited())
        .map_err(|_| "Verify3 owning I/O audit failed")?;
    let all = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|x| x.checked_add(r.readback_bytes))
        .ok_or("Verify3 I/O count overflow")?;
    if all == 0 || r.inspected_bytes != all || r.inspected_buffers == 0 {
        return Err("Verify3 complete I/O audit counts".into());
    }
    Ok(())
}
fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    Ok(client
        .memory_usage()
        .map_err(|_| "Verify3 main telemetry failed")?
        .bytes_reserved)
}

fn case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    ledger: &mut MainAdmission,
    m: &Fixture,
    count: usize,
    bad: usize,
    fused: bool,
) -> Result<serde_json::Value> {
    if bad > 2 || bad != 0 && (count != 3 || m.columns != 2560) {
        return Err("Verify3 fixed fault inventory".into());
    }
    let x = layout(m.columns, count)?;
    let values = input(x, m.columns, count, bad);
    let references = Zeroizing::new(expected(&m.decoded, m.columns, x, &values, count));
    let mut initial = Zeroizing::new(vec![0.; x.output_words]);
    if let Some((index, bits)) = bad_output_marker(x, count, bad) {
        initial[index] = f32::from_bits(bits);
    }
    let plans = [x.input_words, x.output_words, x.output_words, 1, 1]
        .map(BufferLayout::new)
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let full = plans.iter().map(|p| p.owned_bytes as u64).sum::<u64>();
    main_admission::check_capacity(PRIVATE_CAP as u64, &[full])?;
    ledger.observe_before(sample(client)?)?;
    ledger.precharge(
        &plans
            .iter()
            .map(|p| p.owned_bytes as u64)
            .collect::<Vec<_>>(),
    )?;
    let ids = [
        arena.upload_u32(&input_bits(&values))?,
        arena.upload_u32(&input_bits(&initial))?,
        arena.upload_u32(&input_bits(&initial))?,
        arena.upload_u32(&[SEED])?,
        arena.upload_u32(&[SEED])?,
    ];
    let handles = ids
        .iter()
        .map(|&id| arena.handle(id).cloned())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ledger.observe_after(sample(client)?)?;
    let mut roots = vec![m.root.clone()];
    roots.extend(handles.iter().cloned());
    let resources = roots
        .iter()
        .map(|h| {
            client
                .get_resource(h.clone())
                .map_err(|_| "Verify3 physical owner lookup failed")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (i, r) in resources.iter().enumerate() {
        let r = r.resource();
        if r.offset != 0
            || r.size != roots[i].size_in_used()
            || r.buffer.size() != r.size
            || roots[i].stream != roots[0].stream
            || resources[..i]
                .iter()
                .any(|p| p.resource().buffer == r.buffer)
        {
            return Err("Verify3 full physical owners alias or escape page".into());
        }
    }
    let source = client
        .get_resource(m.view.clone())
        .map_err(|_| "Verify3 source subview lookup")?;
    if source.resource().buffer != resources[0].resource().buffer
        || source.resource().offset != m.prefix as u64
        || source.resource().size != m.view.size_in_used()
    {
        return Err("Verify3 original packed source subview escaped owner".into());
    }
    let fault = handles[3]
        .clone()
        .offset_end((plans[3].owned_bytes - 4) as u64);
    let fault_resource = client
        .get_resource(fault.clone())
        .map_err(|_| "Verify3 checked fault subview lookup")?;
    if fault_resource.resource().buffer != resources[4].resource().buffer
        || fault_resource.resource().offset != 0
        || fault_resource.resource().size != 4
    {
        return Err("Verify3 logical fault escaped full owner".into());
    }
    let spec = MatrixSpec::new(ROWS, m.columns, m.dtype)?;
    let spans = BatchSpans::new(
        count,
        0,
        x.input_first,
        x.input_stride,
        x.output_first,
        x.output_stride,
    );
    if fused {
        // Exact logical output binding, retaining the complete physical root
        // for its independent read/padding/erasure checks below. No external
        // finite scan is allowed to hide a missing candidate logical fault.
        let output = handles[1]
            .clone()
            .offset_end((plans[1].owned_bytes - x.output_words * 4) as u64);
        if m.dtype == 0 {
            // F32 row offsets are256-byte aligned for both compiled widths.
            // Two original tiles prove the final complement excludes active
            // earlier-tile words; no new public source or owner is introduced.
            let row_bytes = m.columns * 4;
            for (first, rows) in [(0usize, 4usize), (4, ROWS - 4)] {
                let weights = m
                    .view
                    .clone()
                    .offset_start((first * row_bytes) as u64)
                    .offset_end(((ROWS - first - rows) * row_bytes) as u64);
                let tile = MatrixSpec::new(rows, m.columns, m.dtype)?;
                next_shared::matrix_verify3_finite(
                    client,
                    tile,
                    weights,
                    handles[0].clone(),
                    output.clone(),
                    fault.clone(),
                    x.input_words,
                    x.output_words,
                    &BatchSpans::new(
                        count,
                        0,
                        x.input_first,
                        x.input_stride,
                        x.output_first + first,
                        x.output_stride,
                    ),
                    ROWS,
                    x.output_first,
                    first + rows == ROWS,
                )?;
            }
        } else {
            next_shared::matrix_verify3_finite(
                client,
                spec,
                m.view.clone(),
                handles[0].clone(),
                output,
                fault,
                x.input_words,
                x.output_words,
                &spans,
                ROWS,
                x.output_first,
                true,
            )?;
        }
    } else {
        next_shared::matrix_verify3(
            client,
            spec,
            m.view.clone(),
            handles[0].clone(),
            handles[1].clone(),
            fault,
            x.input_words,
            x.output_words,
            &spans,
        )?;
    }
    // Old original control uses the same frozen packed bytes, active input rows
    // and output slots. No candidate/default selection changes during a case.
    for token in 0..count {
        next_shared::matrix(
            client,
            spec,
            m.view.clone(),
            handles[0].clone(),
            handles[2].clone(),
            x.input_words,
            x.output_words,
            x.input_first + token * x.input_stride,
            x.output_first + token * x.output_stride,
        )?;
    }
    for (output, fault) in [(1, 3), (2, 4)] {
        if fused && output == 1 {
            continue;
        }
        crate::resident_util::verify_finite::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(plans[output].owned_words.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(handles[output].clone(), plans[output].owned_words) },
            unsafe { ArrayArg::from_raw_parts(handles[fault].clone(), 1) },
        );
    }
    let mut maximum = 0f64;
    let mut baseline_error = 0f64;
    let mut outputs = Vec::new();
    for (i, h) in handles.iter().enumerate() {
        let lease = read(client, h)?;
        if i == 0 {
            check_input(&lease.0, &values, plans[i])?;
        } else if i <= 2 {
            let (active, error) = check_output(&lease.0, x, count, bad, &references, plans[i])?;
            maximum = maximum.max(error);
            outputs.push(active);
        } else {
            check_fault(&lease.0, plans[i], bad)?;
        }
        drop(lease);
        audit(client)?;
    }
    for (&a, &b) in outputs[0].iter().zip(outputs[1].iter()) {
        if a.is_nan() && b.is_nan() || matches!(bad, 7 | 8) && !a.is_finite() && !b.is_finite() {
            continue;
        }
        let error = (f64::from(a) - f64::from(b)).abs();
        if !a.is_finite() || !b.is_finite() || error > ABS + REL * f64::from(b).abs() {
            return Err("Verify3 old original GPU projection mismatch".into());
        }
        baseline_error = baseline_error.max(error);
    }
    drop(outputs);
    drop(resources);
    drop(source);
    drop(fault_resource);
    drop(roots);
    arena.reset()?;
    if ids.iter().any(|&id| arena.handle(id).is_ok()) {
        return Err("Verify3 reset published stale owner IDs".into());
    }
    for (i, h) in handles.iter().enumerate() {
        let lease = read(client, h)?;
        if lease.0.len() != plans[i].owned_bytes || lease.0.iter().any(|&b| b != 0) {
            return Err("Verify3 complete private owner erasure failed".into());
        }
        drop(lease);
        audit(client)?;
    }
    Ok(
        serde_json::json!({"dtype":m.dtype,"columns":m.columns,"rows":ROWS,
        "count":count,"fault_case":bad,"source_prefix_bytes":m.prefix,
        "logical_finite_fused":fused,"external_candidate_finite_scan":!fused,
        "projection_tiles":if fused && m.dtype==0 {2}else{1},
        "max_cpu_f64_abs_error":maximum,"max_old_gpu_abs_error":baseline_error,
        "full_private_bytes":full,"active_values":count*ROWS,
        "full_owner_guards_preserved":true,"normal_unused_regions_zero":bad==0,
        "sticky_fault_expected":SEED|u32::from(bad!=0),"erased":true}),
    )
}

/// Parent retains pre-device feature/configuration gating and physical recovery.
/// Returns numeric JSON only after all owning checks and allocator release.
pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
) -> Result<serde_json::Value> {
    run_inner(client, limits, false)
}

/// Separate parent-registered fixed public candidate, no ENV/config changes.
#[allow(dead_code)] // Parent owns the canonical protected selector.
pub(crate) fn run_fused(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
) -> Result<serde_json::Value> {
    run_inner(client, limits, true)
}

fn run_inner(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    fused: bool,
) -> Result<serde_json::Value> {
    if cfg!(any(
        feature = "direct-device",
        feature = "cpu-fma",
        feature = "cpu-buffered-weights",
        feature = "cpu-work-stealing",
        feature = "cpu-q8-two-pass"
    )) {
        return Err("Verify3 compiled-public probe requires default build".into());
    }
    // Actual enabled features/subgroup interval must already have passed the
    // parent's adapter gate; this checks the same resource ceilings again.
    next_shared::validate_verify3_limits(limits, true, 16, 32)?;
    if fused {
        next_shared::validate_verify3_finite_limits(limits, true, 16, 32)?;
    }
    if limits.max_compute_workgroup_storage_size < 16 * 1024 {
        return Err("Verify3 original Q4 GPU control storage limit".into());
    }
    let baseline = client
        .memory_usage()
        .map_err(|_| "Verify3 initial main telemetry")?;
    if baseline.number_allocs != 0
        || baseline.bytes_in_use != 0
        || baseline.bytes_padding != 0
        || baseline.bytes_reserved != 0
    {
        return Err("Verify3 fresh exclusive main context required".into());
    }
    client
        .configure_transient_io(IO_CAP)
        .map_err(|_| "Verify3 bounded I/O configuration")?;
    let mut ledger = MainAdmission::new(MAIN_CAP, 0)?;
    let mut sources = Vec::with_capacity(TYPES.len() * WIDTHS.len());
    let public_bound = TYPES
        .into_iter()
        .flat_map(|dtype| WIDTHS.map(move |columns| (dtype, columns)))
        .try_fold(0u64, |sum, (dtype, columns)| -> Result<u64> {
            let size = kuno_text::TensorType::from_gguf_id(dtype)?.row_bytes(columns as u64)?
                * ROWS as u64;
            let prefix = if columns == 256 { 0 } else { 256 };
            Ok(sum + main_admission::page_bytes(prefix + size.div_ceil(4) * 4)?)
        })?;
    main_admission::check_capacity(MAIN_CAP, &[public_bound, PRIVATE_CAP as u64])?;
    main_admission::check_capacity(
        HOST_CAP,
        &[public_bound * 4, PRIVATE_CAP as u64 * 8, 16 * 1024 * 1024],
    )?;
    for dtype in TYPES {
        for columns in WIDTHS {
            let payload = kuno_text::TensorType::from_gguf_id(dtype)?.row_bytes(columns as u64)?
                * ROWS as u64;
            let prefix = if columns == 256 { 0 } else { 256 };
            ledger.observe_before(sample(client)?)?;
            ledger.precharge(&[main_admission::page_bytes(
                prefix + payload.div_ceil(4) * 4,
            )?])?;
            sources.push(public_fixture(client, dtype, columns)?);
            ledger.observe_after(sample(client)?)?;
        }
    }
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), PRIVATE_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let faults = if fused {
            FUSED_FAULT_CASES
        } else {
            FAULT_CASES
        };
        let mut reports = Vec::with_capacity(NORMAL_CASES + faults);
        for m in &sources {
            for count in 1..=3 {
                reports.push(case(client, &mut arena, &mut ledger, m, count, 0, fused)?);
                client.memory_cleanup();
                cubecl::future::block_on(client.sync()).map_err(|_| "Verify3 case completion")?;
                ledger.completed_fence(sample(client)?)?;
            }
            if m.columns == 2560 {
                for count in if fused { 1..=3 } else { 3..=3 } {
                    let kinds: &[usize] = if fused { &FUSED_FAULT_KINDS } else { &[1, 2] };
                    for &bad in kinds {
                        reports.push(case(client, &mut arena, &mut ledger, m, count, bad, fused)?);
                        client.memory_cleanup();
                        cubecl::future::block_on(client.sync())
                            .map_err(|_| "Verify3 fault case completion")?;
                        ledger.completed_fence(sample(client)?)?;
                    }
                }
            }
        }
        Ok(reports)
    }));
    let closed = arena.close();
    let reports = match (attempt, closed, retired.load(Ordering::Acquire)) {
        (Ok(Ok(r)), Ok(()), false)
            if r.len()
                == NORMAL_CASES
                    + if fused {
                        FUSED_FAULT_CASES
                    } else {
                        FAULT_CASES
                    } =>
        {
            r
        }
        _ => return Err("Verify3 fixed public operators failed; no accepted receipt".into()),
    };
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("Verify3 owned zero receipt absent")?;
    drop(arena);
    for m in &sources {
        let lease = read(client, &m.root)?;
        if &*lease.0 != m.wire.as_slice() {
            return Err("Verify3 original source/full padding changed".into());
        }
        drop(lease);
        audit(client)?;
    }
    drop(sources);
    let mut released = false;
    for _ in 0..2 {
        cubecl::future::block_on(client.sync()).map_err(|_| "Verify3 final completion")?;
        client.memory_cleanup();
        cubecl::future::block_on(client.sync())
            .map_err(|_| "Verify3 allocator cleanup completion")?;
        let m = client
            .memory_usage()
            .map_err(|_| "Verify3 final main telemetry")?;
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
        return Err("Verify3 allocator release not accepted".into());
    }
    Ok(
        serde_json::json!({"scope":if fused {"compiled-public-original-packed-Verify3-logical-finite-only"}
            else {"compiled-public-original-packed-Verify3-projection-only"},
        "logical_finite_fused":fused,"external_candidate_finite_scan":!fused,
        "format_count":TYPES.len(),"numeric_cases":NORMAL_CASES,
        "fault_cases":if fused {FUSED_FAULT_CASES} else {FAULT_CASES},
        "max_runtime_count":3,"owning_member_capacity":MAX8,"cases":reports,
        "absolute_tolerance":ABS,"relative_tolerance":REL,"cpu_decoded_f64_reference":true,
        "old_original_gpu_control":true,"prior_fault_preserved":true,
        "active_nan_and_unused_output_nan_faults_rejected":true,
        "source_full_pages_preserved":true,"owned_zero_audit_bytes":zero.bytes,
        "owned_zero_audit_buffers":zero.buffers,"allocator_cleanup_complete":released,
        "main_cap_bytes":MAIN_CAP,"private_cap_bytes":PRIVATE_CAP,"io_cap_bytes":IO_CAP,
        "host_capacity_bound_bytes":HOST_CAP,"context_retired":false,
        "whole_owning_erasure_accepted":false,"private_request_accepted":false,
        "whole_model_accepted":false,"performance_accepted":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registration_is_exact_default_public_only_and_has_no_model_inputs() {
        let env = || {
            [
                ("KUNO_LAB_VERIFY3_DENSE_PUBLIC_ONLY", "1"),
                ("KUNO_PUBLIC_RECEIPT", "/public/numeric"),
            ]
            .map(|(k, v)| (k.into(), v.into()))
            .to_vec()
        };
        for card in ["intel", "nvidia"] {
            let args: Vec<_> = ["binary", card, "verify3-dense-probe"]
                .map(Into::into)
                .to_vec();
            assert!(registration(&args, env(), true).is_ok());
            assert!(registration(&args, env(), false).is_err());
            let mut values = env();
            values.push(("KUNO_TEST_Q3_DIR".into(), "/model".into()));
            assert!(registration(&args, values, true).is_err());
            for index in 0..2 {
                let mut values = env();
                values.remove(index);
                assert!(registration(&args, values, true).is_err());
            }
        }
    }

    fn bytes(values: &[f32], full: usize) -> Vec<u8> {
        let mut result = vec![0; full];
        for (chunk, value) in result.chunks_exact_mut(4).zip(values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        result
    }
    #[test]
    fn fixed_inventory_decodes_all_original_formats_and_c_records() {
        for dtype in TYPES {
            for columns in WIDTHS {
                let payload = packed(dtype, columns).unwrap();
                let decoded = decoded(dtype, columns, &payload).unwrap();
                assert_eq!(
                    payload.len(),
                    kuno_text::TensorType::from_gguf_id(dtype)
                        .unwrap()
                        .row_bytes(columns as u64)
                        .unwrap() as usize
                        * ROWS
                );
                assert_eq!(decoded.len(), ROWS * columns);
                assert!(decoded.iter().all(|x| x.is_finite()));
                assert!(decoded.iter().any(|&x| x != 0.));
                let x = layout(columns, 3).unwrap();
                let values = input(x, columns, 3, 0);
                let references = expected(&decoded, columns, x, &values, 3);
                assert!(references.iter().all(|x| x.is_finite()));
                for count in 1..=3 {
                    next_shared::validate_verify3_spans(
                        MatrixSpec::new(ROWS, columns, dtype).unwrap(),
                        &BatchSpans::new(
                            count,
                            0,
                            x.input_first,
                            x.input_stride,
                            x.output_first,
                            x.output_stride,
                        ),
                        payload.len().div_ceil(4) as u64 * 4,
                        x.input_words,
                        x.output_words,
                    )
                    .unwrap();
                }
                for token in 0..3 {
                    for row in 0..ROWS {
                        let tree = crate::quant_matvec::cooperative::reference_tree_dot(
                            &decoded[row * columns..(row + 1) * columns],
                            &values[x.input_first + token * x.input_stride..][..columns],
                        )
                        .unwrap();
                        let reference = references[token * ROWS + row];
                        assert!((f64::from(tree) - reference).abs() <= ABS + REL * reference.abs());
                    }
                }
            }
        }
        assert_eq!(NORMAL_CASES, 60);
        assert_eq!(FAULT_CASES, 20);
    }
    #[test]
    fn full_static_source_and_private_plans_fit_declared_bounds() {
        let mut public = 0u64;
        for dtype in TYPES {
            for columns in WIDTHS {
                let payload = packed(dtype, columns).unwrap();
                public += main_admission::page_bytes(
                    ((if columns == 256 { 0 } else { 256 }) + payload.len().div_ceil(4) * 4) as u64,
                )
                .unwrap();
                let x = layout(columns, 3).unwrap();
                let full = [x.input_words, x.output_words, x.output_words, 1, 1]
                    .map(|n| BufferLayout::new(n).unwrap().owned_bytes as u64)
                    .into_iter()
                    .sum::<u64>();
                assert!(full <= PRIVATE_CAP as u64);
            }
        }
        main_admission::check_capacity(MAIN_CAP, &[public, PRIVATE_CAP as u64]).unwrap();
        main_admission::check_capacity(
            HOST_CAP,
            &[public * 4, PRIVATE_CAP as u64 * 8, 16 * 1024 * 1024],
        )
        .unwrap();
        assert_eq!(input_bits(&[f32::NAN])[0], f32::NAN.to_bits());
    }
    #[test]
    fn whole_max8_parser_rejects_active_gaps_inactive_padding_and_truncation() {
        for count in 1..=3 {
            let x = layout(256, count).unwrap();
            let owned = BufferLayout::new(x.output_words).unwrap();
            let references = vec![0.5; count * ROWS];
            let mut values = vec![0.; x.output_words];
            for token in 0..count {
                for row in 0..ROWS {
                    values[x.output_first + token * x.output_stride + row] = 0.5;
                }
            }
            let original = bytes(&values, owned.owned_bytes);
            check_output(&original, x, count, 0, &references, owned).unwrap();
            for index in [
                0,
                x.output_first,
                x.output_first + ROWS,
                x.output_first + count * x.output_stride,
                x.output_words,
                owned.owned_words - 1,
            ] {
                let mut bad = original.clone();
                bad[index * 4..index * 4 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
                assert!(check_output(&bad, x, count, 0, &references, owned).is_err());
            }
            assert!(
                check_output(
                    &original[..original.len() - 4],
                    x,
                    count,
                    0,
                    &references,
                    owned
                )
                .is_err()
            );
        }
        for n in [0, 4, 8, usize::MAX] {
            assert!(layout(256, n).is_err());
        }
    }
    #[test]
    fn complete_input_and_sticky_fault_padding_are_independent_gates() {
        let x = layout(256, 3).unwrap();
        let values = input(x, 256, 3, 0);
        let owned = BufferLayout::new(values.len()).unwrap();
        let original = bytes(&values, owned.owned_bytes);
        check_input(&original, &values, owned).unwrap();
        for index in [
            0,
            x.input_first,
            x.input_first + 256,
            x.input_first + 3 * x.input_stride,
            x.input_words,
            owned.owned_words - 1,
        ] {
            let mut bad = original.clone();
            bad[index * 4] ^= 1;
            assert!(check_input(&bad, &values, owned).is_err());
        }
        let f = BufferLayout::new(1).unwrap();
        for bad in 0..=2 {
            let mut b = vec![0; f.owned_bytes];
            b[..4].copy_from_slice(&(SEED | u32::from(bad != 0)).to_le_bytes());
            check_fault(&b, f, bad).unwrap();
            b[4] = 1;
            assert!(check_fault(&b, f, bad).is_err());
        }
    }

    #[test]
    fn fused_fault_inventory_covers_every_count_and_unwritten_logical_region() {
        assert_eq!(FUSED_FAULT_CASES, 210);
        for count in 1..=3 {
            let x = layout(2560, count).unwrap();
            let owned = BufferLayout::new(x.output_words).unwrap();
            for bad in FUSED_FAULT_KINDS {
                let mut values = vec![0.; x.output_words];
                if let Some((index, bits)) = bad_output_marker(x, count, bad) {
                    assert!(index < x.output_words);
                    let relative = index.saturating_sub(x.output_first);
                    assert!(
                        !(index >= x.output_first
                            && relative / x.output_stride < count
                            && relative % x.output_stride < ROWS)
                    );
                    values[index] = f32::from_bits(bits);
                } else {
                    for row in 0..ROWS {
                        values[x.output_first + (count - 1) * x.output_stride + row] = f32::NAN;
                    }
                }
                let wire = bytes(&values, owned.owned_bytes);
                check_output(&wire, x, count, bad, &vec![0.; count * ROWS], owned).unwrap();
                assert!(check_output(&wire, x, count, 0, &vec![0.; count * ROWS], owned).is_err());
                let f = BufferLayout::new(1).unwrap();
                check_fault(&bytes(&[f32::from_bits(SEED | 1)], f.owned_bytes), f, bad).unwrap();
            }
        }
    }
}
