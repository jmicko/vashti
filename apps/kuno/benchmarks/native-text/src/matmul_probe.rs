//! Fixed public original-GGUF multi-RHS acceptance. No arbitrary fixture input.
//! Device selection/configuration belongs to main; host tests never use a GPU.
//! Borrow <=7 rows/<=512 KiB per original format and guard all host vectors.
//! Retained main handles below are used ONLY to inspect post-erasure zeros,
//! never for inference after an arena reset. Receipts contain numeric stats.

use crate::private_arena::{BufferId, BufferLayout, PrivateArena};
use crate::quant_matvec::matmul::{BatchSpans, MAX_BATCH, MatrixGeometry, packed_matmul};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::transient::TransientIoReceipt;
use kuno_text::{GgufFile, GgufLimits, TensorType};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};

#[path = "matmul_timing.rs"]
mod timing;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAX_ROWS: usize = 7;
const MAX_COLUMNS: usize = 65536;
const MAX_PACKED_BYTES: usize = 512 * 1024;
const MAIN_BUDGET: usize = 8 * 1024 * 1024;
const IO_BUDGET: u64 = 32 * 1024 * 1024;
const OUTPUT_GAP: f32 = -12345.0;
const INPUT_GAP: f32 = 31415.0;
const NUMERIC_COUNTS: [usize; 4] = [1, 3, 5, 8];
const NUMERIC_TILES: [usize; 3] = [1, 4, 8];
const REPLAY_SNAPSHOT_BYTES: usize = 32768;
const FORMATS: [TensorType; 7] = [
    TensorType::F32,
    TensorType::F16,
    TensorType::Bf16,
    TensorType::Q4K,
    TensorType::Q5_0,
    TensorType::Q6K,
    TensorType::Q8_0,
];

struct PublicMatrix {
    kind: TensorType,
    rows: usize,
    columns: usize,
    packed_first: usize,
    words: usize,
    handle: Handle,
    wire: Zeroizing<Vec<u8>>,
    decoded: Zeroizing<Vec<f32>>,
}

#[derive(Clone, Copy)]
struct FixtureLayout {
    input_first: usize,
    input_stride: usize,
    input_words: usize,
    output_first: usize,
    output_stride: usize,
    output_words: usize,
}
fn fixture_layout(rows: usize, columns: usize) -> Result<FixtureLayout> {
    if !(1..=MAX_ROWS).contains(&rows) || !(1..=MAX_COLUMNS).contains(&columns) {
        return Err("public matmul geometry exceeds fixture bounds".into());
    }
    let input_stride = columns.checked_add(11).ok_or("public fixture overflow")?;
    let output_stride = rows.checked_add(5).ok_or("public fixture overflow")?;
    let input_words = MAX_BATCH
        .checked_mul(input_stride)
        .and_then(|n| n.checked_add(9))
        .ok_or("public fixture overflow")?;
    let output_words = MAX_BATCH
        .checked_mul(output_stride)
        .and_then(|n| n.checked_add(7))
        .ok_or("public fixture overflow")?;
    let charged = BufferLayout::new(input_words)?
        .owned_bytes
        .checked_add(BufferLayout::new(output_words)?.owned_bytes)
        .ok_or("public fixture overflow")?;
    if charged > MAIN_BUDGET {
        return Err("public fixture exceeds private arena budget".into());
    }
    Ok(FixtureLayout {
        input_first: 9,
        input_stride,
        input_words,
        output_first: 7,
        output_stride,
        output_words,
    })
}

fn load_public(client: &ComputeClient<WgpuRuntime>, file: &GgufFile) -> Result<Vec<PublicMatrix>> {
    load_public_variant(client, file, true)
}
fn load_public_variant(
    client: &ComputeClient<WgpuRuntime>,
    file: &GgufFile,
    prefix: bool,
) -> Result<Vec<PublicMatrix>> {
    let mut matrices = Vec::with_capacity(FORMATS.len());
    for kind in FORMATS {
        // Preserve main's original selection: Q5_0's PLE rows are 160 wide,
        // rather than using that format's unsupported partial quant blocks.
        let tensor = if kind == TensorType::Q5_0 {
            file.tensor("per_layer_token_embd.weight")
                .ok_or("public Q5_0 fixture missing")?
        } else {
            let info = file
                .tensors()
                .values()
                .find(|t| t.dtype() == kind)
                .ok_or("public matrix format missing")?;
            file.tensor(info.name())
                .ok_or("public matrix format missing")?
        };
        let rows = MAX_ROWS.min(
            usize::try_from(tensor.info().row_count()).map_err(|_| "public fixture overflow")?,
        );
        let columns =
            usize::try_from(tensor.info().row_len()).map_err(|_| "public fixture overflow")?;
        let row_bytes =
            usize::try_from(tensor.info().row_bytes()).map_err(|_| "public fixture overflow")?;
        let bytes = rows
            .checked_mul(row_bytes)
            .ok_or("public fixture overflow")?;
        if tensor.info().dtype() != kind || bytes > MAX_PACKED_BYTES {
            return Err("public matrix slice exceeds format/byte bounds".into());
        }
        fixture_layout(rows, columns)?;
        let geometry = MatrixGeometry::new(kind.gguf_id(), rows, columns, 4)?;
        if geometry.row_bytes() != row_bytes || geometry.total_bytes() != bytes {
            return Err("public matrix layout mismatch".into());
        }
        let mut packed = Zeroizing::new(vec![0_u8; bytes]);
        tensor
            .read_range(0, &mut packed)
            .map_err(|_| "public matrix read failed")?;
        let mut decoded = Zeroizing::new(vec![
            0_f32;
            rows.checked_mul(columns)
                .ok_or("public fixture overflow")?
        ]);
        // Independent Rust GGUF decoder, not a copy of the shader decoder.
        kind.decode(&packed, &mut decoded)
            .map_err(|_| "public matrix decode failed")?;
        if decoded.iter().any(|x| !x.is_finite()) {
            return Err("public matrix contains nonfinite values".into());
        }
        let packed_first: usize = if !prefix {
            0
        } else if kind == TensorType::F32 {
            12
        } else {
            3
        };
        let wire_len = packed_first
            .checked_add(bytes)
            .ok_or("public fixture overflow")?
            .div_ceil(4)
            * 4;
        let mut wire = Zeroizing::new(vec![0x53_u8; wire_len]);
        wire[packed_first..packed_first + bytes].copy_from_slice(&packed);
        let handle = client.create_from_slice(&wire); // immutable PUBLIC original weights
        matrices.push(PublicMatrix {
            kind,
            rows,
            columns,
            packed_first,
            words: wire_len / 4,
            handle,
            wire,
            decoded,
        });
    }
    Ok(matrices)
}

fn input_fixture(layout: FixtureLayout, columns: usize, count: usize) -> Zeroizing<Vec<f32>> {
    let mut input = Zeroizing::new(vec![INPUT_GAP; layout.input_words]);
    for token in 0..count {
        for column in 0..columns {
            input[layout.input_first + token * layout.input_stride + column] =
                ((column * 3 + token * 5) % 17) as f32 / 17.0 - 8.0 / 17.0;
        }
    }
    input
}

fn expectations(
    rows: usize,
    columns: usize,
    decoded: &[f32],
    input: &[f32],
    layout: FixtureLayout,
    count: usize,
) -> Result<Zeroizing<Vec<f32>>> {
    let mut expected = Zeroizing::new(vec![OUTPUT_GAP; layout.output_words]);
    for token in 0..count {
        let x = &input[layout.input_first + token * layout.input_stride..][..columns];
        for row in 0..rows {
            let w = &decoded[row * columns..(row + 1) * columns];
            let tree = crate::quant_matvec::cooperative::reference_tree_dot(w, x)?;
            let independent = w
                .iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>();
            if !tree.is_finite()
                || !independent.is_finite()
                || (f64::from(tree) - independent).abs() > 1e-4 + 3e-4 * independent.abs()
            {
                return Err("public CPU reduction schedule oracle failed".into());
            }
            expected[layout.output_first + token * layout.output_stride + row] = tree;
        }
    }
    Ok(expected)
}

/// Check every logical gap/inactive token bit-for-bit, active numeric results
/// with the existing tolerance, and every full owning-span padding byte zero.
fn check_output(
    actual: &[u8],
    expected: &[f32],
    rows: usize,
    layout: FixtureLayout,
    count: usize,
    full_bytes: usize,
) -> Result<f32> {
    if actual.len() != full_bytes
        || expected.len() != layout.output_words
        || actual.len() < expected.len() * 4
        || !actual.len().is_multiple_of(4)
    {
        return Err("public output length mismatch".into());
    }
    let mut max_error = 0_f32;
    for (index, (bytes, &reference)) in actual.chunks_exact(4).zip(expected).enumerate() {
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        let relative = index.saturating_sub(layout.output_first);
        let active = index >= layout.output_first
            && relative / layout.output_stride < count
            && relative % layout.output_stride < rows;
        if active {
            let error = (value - reference).abs();
            if !value.is_finite() || !error.is_finite() || error > 1e-4 + 3e-4 * reference.abs() {
                return Err("public packed multi-RHS numeric fixture failed".into());
            }
            max_error = max_error.max(error);
        } else if value.to_bits() != reference.to_bits() {
            return Err("public packed multi-RHS changed a gap or inactive token".into());
        }
    }
    if actual[expected.len() * 4..].iter().any(|&byte| byte != 0) {
        return Err("public packed multi-RHS changed owning-span padding".into());
    }
    Ok(max_error)
}

fn audited(client: &ComputeClient<WgpuRuntime>) -> Result<TransientIoReceipt> {
    let r = cubecl::future::block_on(client.sanitize_transient_io_audited())
        .map_err(|_| "public multi-RHS audited I/O cleanup failed")?;
    validate_audit(r)?;
    Ok(r)
}
fn validate_audit(r: TransientIoReceipt) -> Result<()> {
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("public audit count overflow")?;
    if full == 0 || r.inspected_bytes != full || r.inspected_buffers == 0 {
        return Err("public audit full-span counts mismatch".into());
    }
    Ok(())
}

fn receipt(r: TransientIoReceipt) -> serde_json::Value {
    serde_json::json!({"epoch":r.epoch,"upload_bytes":r.upload_bytes,"uniform_bytes":r.uniform_bytes,
        "readback_bytes":r.readback_bytes,"inspected_bytes":r.inspected_bytes,"inspected_buffers":r.inspected_buffers})
}

/// The probe constructs all three bindings locally on this same client. Still
/// verify the complete physical regions, rather than assuming distinct handles
/// imply distinct resources. No descriptor/selection is retained after a case.
fn verify_bindings(
    client: &ComputeClient<WgpuRuntime>,
    handles: [&Handle; 3],
    lengths: [usize; 3],
) -> Result<()> {
    for (index, (handle, length)) in handles.iter().zip(lengths).enumerate() {
        let span = effective_bytes(
            handle.size(),
            handle.offset_start.unwrap_or(0),
            handle.offset_end.unwrap_or(0),
        )?;
        if handle.stream != handles[0].stream
            || span != length as u64
            || (index > 0 && handle.size() != span)
        {
            return Err("public multi-RHS binding size/stream mismatch".into());
        }
    }
    let resources = handles
        .iter()
        .map(|h| client.get_resource((*h).clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| "public multi-RHS resource lookup failed")?;
    for (i, left) in resources.iter().enumerate() {
        let left = left.resource();
        if left.size != lengths[i] as u64 {
            return Err("public multi-RHS physical binding length mismatch".into());
        }
        validate_physical_extent(left.offset, left.size, left.buffer.size(), i > 0)?;
        for right in resources.iter().skip(i + 1) {
            let right = right.resource();
            if left.buffer == right.buffer
                && regions_overlap(left.offset, left.size, right.offset, right.size)?
            {
                return Err("public multi-RHS physical bindings alias".into());
            }
        }
    }
    Ok(())
}
// Public contiguous layouts may have positive alignment padding at the end;
// private bindings must cover their complete original owning span instead.
fn effective_bytes(total: u64, first: u64, tail: u64) -> Result<u64> {
    total
        .checked_sub(first)
        .and_then(|n| n.checked_sub(tail))
        .filter(|&n| n > 0)
        .ok_or_else(|| "public multi-RHS binding offsets invalid".into())
}
fn validate_physical_extent(
    offset: u64,
    length: u64,
    backing_bytes: u64,
    whole_owned: bool,
) -> Result<()> {
    let end = offset
        .checked_add(length)
        .ok_or("public physical extent overflow")?;
    if length == 0 || end > backing_bytes || (whole_owned && (offset != 0 || end != backing_bytes))
    {
        return Err("public multi-RHS physical extent is not fully owned/bounded".into());
    }
    Ok(())
}
fn regions_overlap(a: u64, a_len: u64, b: u64, b_len: u64) -> Result<bool> {
    let end_a = a
        .checked_add(a_len)
        .ok_or("public physical extent overflow")?;
    let end_b = b
        .checked_add(b_len)
        .ok_or("public physical extent overflow")?;
    if a_len == 0 || b_len == 0 {
        return Err("public physical extent is empty".into());
    }
    Ok(a < end_b && b < end_a)
}

#[allow(clippy::too_many_arguments)]
fn launch_case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    matrix: &PublicMatrix,
    geometry: MatrixGeometry,
    spans: &BatchSpans,
    x: BufferId,
    y: BufferId,
) -> Result<()> {
    let x_layout = arena.layout(x)?;
    let y_layout = arena.layout(y)?;
    geometry.validate_spans(
        spans,
        matrix.words,
        x_layout.owned_words,
        y_layout.owned_words,
    )?;
    verify_bindings(
        client,
        [&matrix.handle, arena.handle(x)?, arena.handle(y)?],
        [
            matrix.wire.len(),
            x_layout.owned_bytes,
            y_layout.owned_bytes,
        ],
    )?;
    packed_matmul::launch::<WgpuRuntime>(
        client,
        geometry.grid(),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(matrix.handle.clone(), matrix.words) },
        unsafe { ArrayArg::from_raw_parts(arena.handle(x)?.clone(), x_layout.owned_words) },
        unsafe { ArrayArg::from_raw_parts(arena.handle(y)?.clone(), y_layout.owned_words) },
        spans.count(),
        spans.packed_first(),
        spans.input_first(),
        spans.input_stride(),
        spans.output_first(),
        spans.output_stride(),
        geometry.rows(),
        geometry.columns(),
        geometry.dtype(),
        geometry.tile_tokens(),
    );
    Ok(())
}

fn write_input(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    x: BufferId,
    input: &[f32],
) -> Result<()> {
    let layout = arena.layout(x)?;
    if input.len() != layout.logical_words {
        return Err("public replay input owning length".into());
    }
    let mut upload = Zeroizing::new(vec![0_u8; layout.owned_bytes]);
    for (bytes, value) in upload.chunks_exact_mut(4).zip(input) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    client
        .write_private(arena.handle(x)?.clone(), std::mem::take(&mut *upload))
        .map_err(|_| "public owned write failed")?;
    Ok(())
}

fn output_values(bytes: &[u8], words: usize) -> Result<Zeroizing<Vec<f32>>> {
    if bytes.len() < words.checked_mul(4).ok_or("public replay overflow")? {
        return Err("public replay owning output short".into());
    }
    Ok(Zeroizing::new(
        bytes[..words * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    ))
}

/// Reuse the SAME full owners across real MAX8 prefill and count1. Preserve
/// the actual GPU prefill's inactive bits, which need not equal CPU roundoff.
/// The ordinary tile1 count1 dispatch is an independent paired GPU reference.
struct ReplayReference {
    expected: Zeroizing<Vec<f32>>,
    paired: Zeroizing<Vec<f32>>,
}
fn single_replay_reference(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    matrix: &PublicMatrix,
    x: BufferId,
    y: BufferId,
    input: &mut [f32],
) -> Result<ReplayReference> {
    let l = fixture_layout(matrix.rows, matrix.columns)?;
    let full = arena.layout(y)?.owned_bytes;
    let spans = |count| {
        BatchSpans::new(
            count,
            matrix.packed_first,
            l.input_first,
            l.input_stride,
            l.output_first,
            l.output_stride,
        )
    };
    let geometry = MatrixGeometry::new(matrix.kind.gguf_id(), matrix.rows, matrix.columns, 8)?;
    launch_case(client, arena, matrix, geometry, &spans(MAX_BATCH), x, y)?;
    let prefill = client
        .read_one(arena.handle(y)?.clone())
        .map_err(|_| "public replay prefill read failed")?;
    let prefill_expected = expectations(
        matrix.rows,
        matrix.columns,
        &matrix.decoded,
        input,
        l,
        MAX_BATCH,
    )?;
    check_output(&prefill, &prefill_expected, matrix.rows, l, MAX_BATCH, full)?;
    let mut seed = Zeroizing::new(prefill.to_vec());
    if seed.len() > REPLAY_SNAPSHOT_BYTES || seed.capacity() > REPLAY_SNAPSHOT_BYTES {
        return Err("public replay host snapshot exceeded fixed owner bound".into());
    }
    let preserved = output_values(&seed, l.output_words)?;
    drop(prefill);
    audited(client)?;
    for column in 0..matrix.columns {
        input[l.input_first + column] = -input[l.input_first + column] + 0.0625;
    }
    let mut expected = expectations(matrix.rows, matrix.columns, &matrix.decoded, input, l, 1)?;
    for i in 0..expected.len() {
        if !(l.output_first..l.output_first + matrix.rows).contains(&i) {
            expected[i] = preserved[i];
        }
    }
    write_input(client, arena, x, input)?;
    launch_case(
        client,
        arena,
        matrix,
        MatrixGeometry::new(matrix.kind.gguf_id(), matrix.rows, matrix.columns, 1)?,
        &spans(1),
        x,
        y,
    )?;
    let baseline = client
        .read_one(arena.handle(y)?.clone())
        .map_err(|_| "public replay baseline read failed")?;
    check_output(&baseline, &expected, matrix.rows, l, 1, full)?;
    let paired = output_values(&baseline, l.output_words)?;
    drop(baseline);
    audited(client)?;
    // Restore the exact accepted prefill owning span before the candidate.
    client
        .write_private(arena.handle(y)?.clone(), std::mem::take(&mut *seed))
        .map_err(|_| "public replay prefill restore failed")?;
    Ok(ReplayReference { expected, paired })
}

fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    matrices: &[PublicMatrix],
    retired: &AtomicBool,
) -> Result<Vec<serde_json::Value>> {
    let mut reports =
        Vec::with_capacity(FORMATS.len() * (NUMERIC_COUNTS.len() * NUMERIC_TILES.len() + 1));
    for matrix in matrices {
        for count in NUMERIC_COUNTS {
            for (tile, single) in NUMERIC_TILES
                .into_iter()
                .map(|tile| (tile, false))
                .chain((count == 1).then_some((1, true)))
            {
                let mut geometry =
                    MatrixGeometry::new(matrix.kind.gguf_id(), matrix.rows, matrix.columns, tile)?;
                if single {
                    geometry = geometry.with_single_member()?;
                }
                let layout = fixture_layout(geometry.rows(), geometry.columns())?;
                let spans = BatchSpans::new(
                    count,
                    matrix.packed_first,
                    layout.input_first,
                    layout.input_stride,
                    layout.output_first,
                    layout.output_stride,
                );
                let mut input = input_fixture(
                    layout,
                    geometry.columns(),
                    if single { MAX_BATCH } else { spans.count() },
                );
                let mut expected = expectations(
                    geometry.rows(),
                    geometry.columns(),
                    &matrix.decoded,
                    &input,
                    layout,
                    spans.count(),
                )?;
                let initial = Zeroizing::new(vec![OUTPUT_GAP; layout.output_words]);
                let x = arena
                    .upload_f32(&input)
                    .map_err(|_| "public owned input upload failed")?;
                let y = arena
                    .upload_f32(&initial)
                    .map_err(|_| "public owned output initialization failed")?;
                let x_layout = arena.layout(x)?;
                let y_layout = arena.layout(y)?;
                geometry.validate_spans(
                    &spans,
                    matrix.words,
                    x_layout.owned_words,
                    y_layout.owned_words,
                )?;
                let retained_x = arena.handle(x)?.clone();
                let retained_y = arena.handle(y)?.clone();
                verify_bindings(
                    client,
                    [&matrix.handle, &retained_x, &retained_y],
                    [
                        matrix.wire.len(),
                        x_layout.owned_bytes,
                        y_layout.owned_bytes,
                    ],
                )?;
                let paired = if single {
                    let replay = single_replay_reference(client, arena, matrix, x, y, &mut input)?;
                    expected = replay.expected;
                    Some(replay.paired)
                } else {
                    None
                };
                // Leave one explicitly owned upload live for the device audit.
                // Move the guarded host allocation without any ordinary copy.
                write_input(client, arena, x, &input)?;
                // SAFETY: all same-client full spans are independently allocated;
                // public cache is retained, private IDs/lengths and spans checked
                // immediately above; fixed cube/grid match validated geometry.
                launch_case(client, arena, matrix, geometry, &spans, x, y)?;
                let output = client
                    .read_one(retained_y.clone())
                    .map_err(|_| "public output readback failed")?;
                let max_error = check_output(
                    &output,
                    &expected,
                    geometry.rows(),
                    layout,
                    spans.count(),
                    y_layout.owned_bytes,
                )?;
                let paired_error = paired
                    .as_ref()
                    .map(|reference| {
                        check_output(
                            &output,
                            reference,
                            geometry.rows(),
                            layout,
                            1,
                            y_layout.owned_bytes,
                        )
                    })
                    .transpose()?;
                drop(output); // release readback lease BEFORE audited cleanup
                let io = audited(client)?;
                if io.upload_bytes == 0 || io.uniform_bytes == 0 || io.readback_bytes == 0 {
                    return Err("public multi-RHS audit missed an I/O kind".into());
                }
                arena
                    .reset()
                    .map_err(|_| "public full owning main cleanup failed")?;
                if arena.handle(x).is_ok()
                    || arena.handle(y).is_ok()
                    || retired.load(Ordering::Acquire)
                {
                    return Err("public multi-RHS reset did not invalidate ownership".into());
                }
                for (retained, full) in [
                    (retained_x, x_layout.owned_bytes),
                    (retained_y, y_layout.owned_bytes),
                ] {
                    let erased = client
                        .read_one(retained)
                        .map_err(|_| "public main erasure readback failed")?;
                    if erased.len() != full || erased.iter().any(|&b| b != 0) {
                        return Err("public main owning span was not fully erased".into());
                    }
                    drop(erased);
                }
                let main_audit = audited(client)?;
                let original = client
                    .read_one(matrix.handle.clone())
                    .map_err(|_| "public cache check readback failed")?;
                if original.len() != matrix.wire.len() || &*original != matrix.wire.as_slice() {
                    return Err("public original weights changed across private cleanup".into());
                }
                drop(original);
                let cache_audit = audited(client)?;
                let mut report = serde_json::json!({"dtype":geometry.dtype(),"rows":geometry.rows(),"columns":geometry.columns(),
                    "count":spans.count(),"tile_tokens":geometry.tile_tokens(),"max_abs_error":max_error,
                    "main_full_span_bytes":x_layout.owned_bytes+y_layout.owned_bytes,"gaps_and_inactive_preserved":true,
                    "main_erased":true,"public_cache_preserved":true,"io":receipt(io),"main_audit":receipt(main_audit),"cache_audit":receipt(cache_audit)});
                if let Some(error) = paired_error {
                    report["member_capacity"] = serde_json::json!(1);
                    report["max8_prefill_reused"] = serde_json::json!(true);
                    report["generic_max_abs_error"] = serde_json::json!(error);
                }
                reports.push(report);
                println!(
                    "public_packed_matmul dtype={} rows={} columns={} count={} tile={} max_abs_error={max_error}",
                    geometry.dtype(),
                    geometry.rows(),
                    geometry.columns(),
                    spans.count(),
                    geometry.tile_tokens()
                );
            }
        }
    }
    if reports.len() != FORMATS.len() * (NUMERIC_COUNTS.len() * NUMERIC_TILES.len() + 1)
        || reports
            .iter()
            .filter(|case| case.get("member_capacity") == Some(&serde_json::json!(1)))
            .count()
            != FORMATS.len()
    {
        return Err("public projection case inventory incomplete".into());
    }
    Ok(reports)
}

pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let path = std::env::var_os("KUNO_TEST_GGUF")
        .ok_or("existing GGUF required for public matmul probe")?;
    let file = GgufFile::open(path, GgufLimits::default())
        .map_err(|_| "public matmul GGUF open failed")?;
    let raw_timing = match std::env::var("KUNO_LAB_MATMUL_TIMING") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err("KUNO_LAB_MATMUL_TIMING must be 0 or 1".into()),
    };
    let timing_enabled = timing::enabled(raw_timing.as_deref())?;
    // PUBLIC weights are allocated before configuring any private upload path.
    let matrices = load_public(client, &file)?;
    let timing_matrices = if timing_enabled {
        Some(timing::load_public(client, &file)?)
    } else {
        None
    };
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), MAIN_BUDGET, IO_BUDGET, retired.clone())?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let reports = cases(client, &mut arena, &matrices, &retired)?;
        let timings = timing_matrices
            .as_ref()
            .map(|matrices| timing::run(client, &mut arena, matrices, &retired))
            .transpose()?;
        Ok((reports, timings))
    }));
    // Always attempt owning main + bounded I/O cleanup before returning an
    // error, including a panic in CPU checking or a partly queued GPU operation.
    let closed = arena.close();
    if closed.is_err() || retired.load(Ordering::Acquire) {
        return Err("public matmul owning cleanup failed/context retired".into());
    }
    let (reports, timings) = result.map_err(|_| "public matmul probe panicked")??;
    file.verify_file_unchanged()
        .map_err(|_| "public borrowed GGUF changed")?;
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let report = serde_json::json!({"scope":"fixed-public-synthetic-operator-test-only","fixture":"packed-matmul-probe",
            "borrowed_original_gguf":true,"format_count":FORMATS.len(),"case_count":reports.len(),"single_member_cases":FORMATS.len(),"cases":reports,
            "private_uploads_owned":true,"context_retired":false,"operator_timings":timings,
            "proof_scope":"owned API-visible main and I/O spans; not opaque physical driver padding"});
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    Ok(())
}

struct FusedReadLease(cubecl::bytes::Bytes);
impl Drop for FusedReadLease {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn decode_fused_owning_read(bytes: &[u8], layout: BufferLayout) -> Result<Zeroizing<Vec<u32>>> {
    if bytes.len() != layout.owned_bytes || !bytes.len().is_multiple_of(4) {
        return Err("fused full owning read extent".into());
    }
    let mut words = Zeroizing::new(Vec::new());
    words
        .try_reserve_exact(layout.owned_words)
        .map_err(|_| "fused owning host budget")?;
    words.resize(layout.owned_words, 0);
    for (w, b) in words.iter_mut().zip(bytes.chunks_exact(4)) {
        *w = u32::from_le_bytes(b.try_into()?);
    }
    Ok(words)
}
fn fused_input_words(input: &[f32]) -> Result<Zeroizing<Vec<u32>>> {
    let mut words = Zeroizing::new(Vec::new());
    words
        .try_reserve_exact(input.len())
        .map_err(|_| "fused guarded input words")?;
    words.extend(input.iter().map(|v| v.to_bits()));
    Ok(words)
}
/// Counts-only synthetic-fixture diagnostics never forward an arbitrary error.
fn fused_failure_class(error: &(dyn std::error::Error + 'static)) -> &'static str {
    if let Some(error) = error.downcast_ref::<crate::private_arena::ArenaError>() {
        use crate::private_arena::ArenaError;
        return match error {
            ArenaError::DeviceAllocation => "owned_device_allocation",
            ArenaError::Initialization => "owned_initialization",
            ArenaError::Cleanup => "owned_cleanup",
            _ => "owned_arena",
        };
    }
    let message = Zeroizing::new(error.to_string());
    match message.as_str() {
        "fused output/fault extent mismatch" => "fault_or_extent",
        "fused projection differs from untouched GPU" => "original_gpu_projection",
        "fused independent codec oracle" => "independent_codec",
        "fused finite OR contract" => "finite_or",
        "fused output gap changed"
        | "fused owning tail changed"
        | "fused fault owning tail changed" => "owning_padding",
        "fused readonly input changed" => "readonly_input",
        "fused full owner erasure" => "main_erasure",
        "fused original weight mutation" => "public_weight_mutation",
        "fused client/span mismatch" | "fused binding alias" => "owner_binding",
        "public multi-RHS physical extent is not fully owned/bounded"
        | "public physical extent overflow"
        | "public physical extent is empty" => "physical_extent",
        _ => "unclassified",
    }
}
/// Original packed matrices plus fixed synthetic activations. Default-off
/// finite fusion is compared with the untouched scalar-GPU projection followed
/// by the original output-only fault kernel. No arbitrary input is accepted.
pub fn run_fused_fault(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("fused fixture original GGUF required")?;
    let file =
        GgufFile::open(path, GgufLimits::default()).map_err(|_| "fused fixture GGUF metadata")?;
    // Coarse full-page public/host bounds before any model upload. Each original
    // format has <=7 rows, <=512KiB packed and <=65536 columns. No source row
    // is larger or copied outside these existing validated fixture bounds.
    for kind in FORMATS {
        let t = if kind == TensorType::Q5_0 {
            file.tensor("per_layer_token_embd.weight")
        } else {
            file.tensors()
                .values()
                .find(|t| t.dtype() == kind)
                .and_then(|t| file.tensor(t.name()))
        }
        .ok_or("fused fixture format absent")?;
        let rows = MAX_ROWS.min(usize::try_from(t.info().row_count())?);
        let cols = usize::try_from(t.info().row_len())?;
        fixture_layout(rows, cols)?;
        if rows
            .checked_mul(usize::try_from(t.info().row_bytes())?)
            .is_none_or(|b| b > MAX_PACKED_BYTES)
        {
            return Err("fused fixture public bound".into());
        }
    }
    let matrices = load_public_variant(client, &file, false)?;
    // Immutable original weights use the same PUBLIC prewarm boundary as the
    // accepted multi-RHS fixture. No activation exists before owned I/O starts.
    client
        .configure_transient_io(IO_BUDGET)
        .map_err(|_| "fused fixture owned IO setup")?;
    let public_bound = FORMATS.len() * (MAX_PACKED_BYTES + 4).next_power_of_two();
    let retired = Arc::new(AtomicBool::new(false));
    // Owned zero inspection requires the already configured transient policy.
    // A PUBLIC-fixture arena correctly rejects that audit before any kernel.
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), MAIN_BUDGET, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    // Only fixed synthetic stage labels and public dtypes reach diagnostics.
    // Backend errors and tensor values remain transient.
    let mut diagnostic = ("case_setup", 0_u32);
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Vec<serde_json::Value>> {
            let mut cases = Vec::new();
            for m in &matrices {
                for bad in [
                    None,
                    Some(f32::NAN),
                    Some(f32::INFINITY),
                    Some(f32::NEG_INFINITY),
                ] {
                    diagnostic = ("case_setup", m.kind.gguf_id());
                    let out_words = 5 + m.rows + 13;
                    let full = [m.columns, out_words, out_words, 1, 1]
                        .into_iter()
                        .try_fold(0usize, |n, w| -> Result<usize> {
                            Ok(n.checked_add(BufferLayout::new(w)?.owned_bytes)
                                .ok_or("fused main bound overflow")?)
                        })?;
                    if full > MAIN_BUDGET {
                        return Err("fused main bound".into());
                    }
                    // Worst simultaneous sources/targets plus full audit duplication
                    // fit under32MiB; no main-ledger credit is reused for runtime IO.
                    if 2 * full + 2 * 512 * 8 > IO_BUDGET as usize {
                        return Err("fused IO bound".into());
                    }
                    let mut input = Zeroizing::new(
                        (0..m.columns)
                            .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
                            .collect::<Vec<_>>(),
                    );
                    if let Some(value) = bad {
                        input[m.columns / 2] = value;
                    }
                    let initial = Zeroizing::new(vec![OUTPUT_GAP; out_words]);
                    diagnostic.0 = "owning_upload";
                    // The fixed rejection fixture intentionally injects NaN/
                    // infinity. Keep upload_f32's finite-input protection;
                    // upload the original IEEE words through the owning path.
                    let input_words = fused_input_words(&input)?;
                    let x = arena.upload_u32(&input_words)?;
                    let y = arena.upload_f32(&initial)?;
                    let original = arena.upload_f32(&initial)?;
                    let f = arena.upload_u32(&[4])?;
                    let original_f = arena.upload_u32(&[4])?;
                    let ids = [x, y, original, f, original_f];
                    if arena.charged_bytes() != full {
                        return Err("fused exact owning charge".into());
                    }
                    let handles = ids
                        .iter()
                        .map(|&id| arena.handle(id).cloned())
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let mut all = vec![m.handle.clone()];
                    all.extend(handles.iter().cloned());
                    diagnostic.0 = "physical_bindings";
                    let resources = all
                        .iter()
                        .map(|h| client.get_resource(h.clone()))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    for (i, r) in resources.iter().enumerate() {
                        let r = r.resource();
                        if all[i].stream != all[0].stream || r.size != all[i].size_in_used() {
                            return Err("fused client/span mismatch".into());
                        }
                        validate_physical_extent(r.offset, r.size, r.buffer.size(), i > 0)?;
                        if resources[..i]
                            .iter()
                            .any(|p| p.resource().buffer == r.buffer)
                        {
                            return Err("fused binding alias".into());
                        }
                    }
                    let arg = |id, words| -> Result<ArrayArg<WgpuRuntime>> {
                        Ok(unsafe { ArrayArg::from_raw_parts(arena.handle(id)?.clone(), words) })
                    };
                    let weight = || unsafe { ArrayArg::from_raw_parts(m.handle.clone(), m.words) };
                    diagnostic.0 = "projection_enqueue";
                    crate::quant_matvec::cooperative::cooperative_matvec::launch::<WgpuRuntime>(
                        client,
                        CubeCount::Static(m.rows as u32, 1, 1),
                        CubeDim::new_1d(64),
                        weight(),
                        arg(x, m.columns)?,
                        arg(original, out_words)?,
                        m.rows,
                        m.columns,
                        m.kind.gguf_id(),
                        5,
                    );
                    crate::resident_util::verify_finite::launch::<WgpuRuntime>(
                        client,
                        CubeCount::Static(out_words.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        arg(original, out_words)?,
                        arg(original_f, 1)?,
                    );
                    crate::quant_matvec::cooperative::cooperative_matvec_with_fault::launch::<
                        WgpuRuntime,
                    >(
                        client,
                        CubeCount::Static(m.rows as u32, 1, 1),
                        CubeDim::new_1d(64),
                        weight(),
                        arg(x, m.columns)?,
                        arg(y, out_words)?,
                        arg(f, 1)?,
                        m.rows,
                        m.columns,
                        m.kind.gguf_id(),
                        5,
                    );
                    let mut snapshots = Vec::new();
                    diagnostic.0 = "owning_readback";
                    for h in &handles {
                        let lease = FusedReadLease(
                            client
                                .read_one(h.clone())
                                .map_err(|_| "fused fixture read")?,
                        );
                        let i = snapshots.len();
                        let words = decode_fused_owning_read(&lease.0, arena.layout(ids[i])?)?;
                        drop(lease);
                        snapshots.push(words);
                    }
                    audited(client)?;
                    diagnostic.0 = "numeric_comparison";
                    if snapshots[1].len() != snapshots[2].len()
                        || snapshots[3][0] != snapshots[4][0]
                    {
                        return Err("fused output/fault extent mismatch".into());
                    }
                    let mut max_error = 0f32;
                    for (i, (&a, &b)) in snapshots[1].iter().zip(snapshots[2].iter()).enumerate() {
                        let av = f32::from_bits(a);
                        let bv = f32::from_bits(b);
                        if av.is_nan()
                            && bv.is_nan()
                            && bad.is_some()
                            && (5..5 + m.rows).contains(&i)
                        {
                            continue;
                        }
                        if a != b {
                            return Err("fused projection differs from untouched GPU".into());
                        }
                        if i < out_words
                            && !(5..5 + m.rows).contains(&i)
                            && a != OUTPUT_GAP.to_bits()
                        {
                            return Err("fused output gap changed".into());
                        }
                        if i >= out_words && a != 0 {
                            return Err("fused owning tail changed".into());
                        }
                        if bad.is_none() && (5..5 + m.rows).contains(&i) {
                            let expected = crate::quant_matvec::cooperative::reference_tree_dot(
                                &m.decoded[(i - 5) * m.columns..(i - 4) * m.columns],
                                &input,
                            )?;
                            let err = (av - expected).abs();
                            if !av.is_finite() || err > 1e-4 + 3e-4 * expected.abs() {
                                return Err("fused independent codec oracle".into());
                            }
                            max_error = max_error.max(err);
                        }
                    }
                    for (i, &word) in snapshots[0].iter().enumerate() {
                        let expected = if i < input.len() {
                            input[i].to_bits()
                        } else {
                            0
                        };
                        if word != expected {
                            return Err("fused readonly input changed".into());
                        }
                    }
                    if snapshots[3][1..]
                        .iter()
                        .chain(&snapshots[4][1..])
                        .any(|&w| w != 0)
                    {
                        return Err("fused fault owning tail changed".into());
                    }
                    if snapshots[3][0] != if bad.is_some() { 5 } else { 4 } {
                        return Err("fused finite OR contract".into());
                    }
                    arena.reset()?;
                    diagnostic.0 = "owning_erasure";
                    for h in handles {
                        let lease = FusedReadLease(
                            client.read_one(h.clone()).map_err(|_| "fused zero read")?,
                        );
                        if lease.0.len() != h.size() as usize || lease.0.iter().any(|&b| b != 0) {
                            return Err("fused full owner erasure".into());
                        }
                        drop(lease);
                        audited(client)?;
                    }
                    let cache = FusedReadLease(
                        client
                            .read_one(m.handle.clone())
                            .map_err(|_| "fused public cache check")?,
                    );
                    diagnostic.0 = "public_weight_check";
                    if &*cache.0 != m.wire.as_slice() {
                        return Err("fused original weight mutation".into());
                    }
                    drop(cache);
                    audited(client)?;
                    cases.push(serde_json::json!({"dtype":m.kind.gguf_id(),"rows":m.rows,"columns":m.columns,"nonfinite_input":bad.is_some(),"same_gpu_output_and_fault":true,"fault_or_preserved":true,"max_core_abs_error":max_error,"full_main_bytes":full}));
                }
            }
            Ok(cases)
        },
    ));
    if let Ok(Err(error)) = &attempt {
        eprintln!(
            "public_fused_failure class={} phase={} dtype={}",
            fused_failure_class(&**error),
            diagnostic.0,
            diagnostic.1
        );
    }
    let closed = arena.close();
    let cases = match (attempt, closed, retired.load(Ordering::Acquire)) {
        (Ok(Ok(cases)), Ok(()), false) => cases,
        _ => return Err("fused fixture failed; no success receipt".into()),
    };
    file.verify_file_unchanged()
        .map_err(|_| "fused original artifact changed")?;
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("fused zero audit absent")?;
    let report = serde_json::json!({"scope":"fixed-public-original-packed-projection-only","fixture":"fused-matrix-fault-probe","cases":cases,"main_cap_bytes":MAIN_BUDGET,"public_full_page_upper_bytes":public_bound,"runtime_io_cap_bytes":IO_BUDGET,"main_erased_bytes":zero.bytes,"main_erased_buffers":zero.buffers,"owned_cleanup":true,"context_retired":false,"whole_graph_or_performance_accepted":false});
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    println!(
        "public_fused_matrix_fault cases=28 full_main_zero_bytes={} context_retired=false",
        zero.bytes
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_failure_diagnostic_does_not_forward_arbitrary_error_payloads() {
        let known: Box<dyn std::error::Error> = "fused finite OR contract".into();
        assert_eq!(fused_failure_class(&*known), "finite_or");
        let unknown: Box<dyn std::error::Error> = "untrusted synthetic error body".into();
        assert_eq!(fused_failure_class(&*unknown), "unclassified");
        let io = std::io::Error::other("untrusted synthetic backend error");
        assert_eq!(fused_failure_class(&io), "unclassified");
        assert_eq!(
            fused_failure_class(&crate::private_arena::ArenaError::Initialization),
            "owned_initialization"
        );
    }

    #[test]
    fn fused_public_prewarm_precedes_owned_activation_initialization() {
        let source = include_str!("matmul_probe.rs");
        let source = source.split("pub fn run_fused_fault(").nth(1).unwrap();
        let prewarm = source.find("let matrices = load_public_variant").unwrap();
        let configure = source.find(".configure_transient_io(IO_BUDGET)").unwrap();
        let private = source
            .find("let x = arena.upload_u32(&input_words)")
            .unwrap();
        assert!(prewarm < configure && configure < private);
    }

    #[test]
    fn fused_rejection_inputs_preserve_ieee_bits_without_weakening_float_uploads() {
        let bits = [0x8000_0000, 0x7fc0_0042, 0x7f80_0000, 0xff80_0000];
        let input = bits.map(f32::from_bits);
        assert_eq!(fused_input_words(&input).unwrap().as_slice(), &bits);
    }

    #[test]
    fn fused_complete_owner_read_rejects_equal_truncation_before_decoding() {
        let l = BufferLayout::new(3).unwrap();
        let mut bytes = Zeroizing::new(vec![0u8; l.owned_bytes]);
        bytes[l.owned_bytes - 4..].copy_from_slice(&7u32.to_le_bytes());
        let words = decode_fused_owning_read(&bytes, l).unwrap();
        assert_eq!(words.len(), l.owned_words);
        assert_eq!(words[l.owned_words - 1], 7);
        for short in [0, 3, l.owned_bytes - 4, l.owned_bytes - 1] {
            assert!(decode_fused_owning_read(&bytes[..short], l).is_err());
        }
        bytes.push(0);
        assert!(decode_fused_owning_read(&bytes, l).is_err());
    }

    #[test]
    fn public_alignment_tail_is_allowed_but_effective_extents_are_checked() {
        validate_physical_extent(0, 252, 256, false).unwrap();
        validate_physical_extent(4, 252, 256, false).unwrap();
        validate_physical_extent(0, 32768, 32768, true).unwrap();
        for (offset, length, backing, owned) in [
            (0, 252, 256, true),
            (4, 252, 256, true),
            (1, 256, 256, false),
            (0, 0, 256, false),
            (u64::MAX, 1, u64::MAX, false),
        ] {
            assert!(validate_physical_extent(offset, length, backing, owned).is_err());
        }
        assert_eq!(effective_bytes(256, 0, 4).unwrap(), 252);
        assert_eq!(effective_bytes(256, 4, 0).unwrap(), 252);
        assert_eq!(effective_bytes(32768, 0, 0).unwrap(), 32768);
        for (total, first, tail) in [
            (256, 257, 0),
            (256, 0, 257),
            (256, 1, 256),
            (256, 0, 256),
            (u64::MAX, 1, u64::MAX),
        ] {
            assert!(effective_bytes(total, first, tail).is_err());
        }
    }

    #[test]
    fn physical_extents_reject_overlap_empty_and_overflow_but_allow_adjacency() {
        for (a, a_len, b, b_len, expected) in [
            (0, 32768, 32768, 32768, false),
            (32768, 32768, 0, 32768, false),
            (3, 11, 14, 7, false),
            (3, 11, 13, 7, true),
            (13, 7, 3, 11, true),
            (9, 32768, 9, 32768, true),
            (9, 32768, 10, 1, true),
        ] {
            assert_eq!(regions_overlap(a, a_len, b, b_len).unwrap(), expected);
        }
        for (a, a_len, b, b_len) in [
            (0, 0, 1, 1),
            (1, 1, 0, 0),
            (u64::MAX, 1, 0, 1),
            (0, 1, u64::MAX - 1, 2),
        ] {
            assert!(regions_overlap(a, a_len, b, b_len).is_err());
        }
    }

    #[test]
    fn bounded_layout_charges_full_spans_and_rejects_bad_geometry() {
        for (rows, columns) in [(0, 1), (1, 0), (8, 1), (1, 65537), (usize::MAX, usize::MAX)] {
            assert!(fixture_layout(rows, columns).is_err());
        }
        let layout = fixture_layout(7, 65536).unwrap();
        assert_eq!(layout.input_first, 9);
        assert_eq!(layout.output_first, 7);
        assert_eq!(layout.input_stride, 65547);
        assert_eq!(layout.output_stride, 12);
        assert_eq!(
            BufferLayout::new(layout.input_words).unwrap().owned_bytes,
            4 * 1024 * 1024
        );
        assert!(
            BufferLayout::new(layout.input_words).unwrap().owned_bytes
                + BufferLayout::new(layout.output_words).unwrap().owned_bytes
                < MAIN_BUDGET
        );
        assert_eq!(
            BufferLayout::new(layout.output_words).unwrap().owned_bytes,
            REPLAY_SNAPSHOT_BYTES
        );
    }

    #[test]
    fn replay_checker_retains_actual_nonzero_max8_tails_without_accepting_corruption() {
        let l = fixture_layout(3, 137).unwrap();
        let full = BufferLayout::new(l.output_words).unwrap().owned_bytes;
        let mut bytes = Zeroizing::new(vec![0_u8; full]);
        let mut expected = Zeroizing::new(vec![OUTPUT_GAP; l.output_words]);
        for token in 0..MAX_BATCH {
            for row in 0..3 {
                expected[l.output_first + token * l.output_stride + row] =
                    token as f32 + row as f32 / 8. + 0.25;
            }
        }
        for (b, value) in bytes.chunks_exact_mut(4).zip(expected.iter()) {
            b.copy_from_slice(&value.to_le_bytes());
        }
        let actual_seed = output_values(&bytes, l.output_words).unwrap();
        assert_eq!(
            check_output(&bytes, &actual_seed, 3, l, 1, full).unwrap(),
            0.
        );
        for index in [0, l.output_first + l.output_stride, l.output_words - 1] {
            let saved = bytes[index * 4..index * 4 + 4].to_vec();
            bytes[index * 4..index * 4 + 4].copy_from_slice(&(-17_f32).to_le_bytes());
            assert!(check_output(&bytes, &actual_seed, 3, l, 1, full).is_err());
            bytes[index * 4..index * 4 + 4].copy_from_slice(&saved);
        }
        bytes[full - 1] = 1;
        assert!(check_output(&bytes, &actual_seed, 3, l, 1, full).is_err());
        assert!(output_values(&bytes[..3], l.output_words).is_err());
    }

    #[test]
    fn independent_oracles_and_output_checker_cover_every_gap_tail_and_padding() {
        let rows = 3;
        let columns = 137;
        let decoded = Zeroizing::new(
            (0..rows * columns)
                .map(|i| (i % 29) as f32 / 32.0 - 14.0 / 32.0)
                .collect::<Vec<_>>(),
        );
        let layout = fixture_layout(rows, columns).unwrap();
        let full = BufferLayout::new(layout.output_words).unwrap().owned_bytes;
        for count in NUMERIC_COUNTS {
            let input = input_fixture(layout, columns, count);
            let expected = expectations(rows, columns, &decoded, &input, layout, count).unwrap();
            let mut bytes = Zeroizing::new(vec![0_u8; full]);
            for (out, value) in bytes.chunks_exact_mut(4).zip(expected.iter()) {
                out.copy_from_slice(&value.to_le_bytes());
            }
            assert_eq!(
                check_output(&bytes, &expected, rows, layout, count, full).unwrap(),
                0.0
            );
            for index in 0..expected.len() {
                let original: [u8; 4] = bytes[index * 4..index * 4 + 4].try_into().unwrap();
                bytes[index * 4..index * 4 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
                assert!(check_output(&bytes, &expected, rows, layout, count, full).is_err());
                bytes[index * 4..index * 4 + 4].copy_from_slice(&original);
            }
            bytes[full - 1] = 1;
            assert!(check_output(&bytes, &expected, rows, layout, count, full).is_err());
        }
    }

    #[test]
    fn audits_reject_partial_counts_empty_receipts_and_overflow() {
        let good = TransientIoReceipt {
            epoch: 3,
            upload_bytes: 32,
            uniform_bytes: 64,
            readback_bytes: 128,
            inspected_bytes: 224,
            inspected_buffers: 3,
        };
        validate_audit(good).unwrap();
        for bad in [
            TransientIoReceipt {
                inspected_bytes: 223,
                ..good
            },
            TransientIoReceipt {
                inspected_buffers: 0,
                ..good
            },
            TransientIoReceipt {
                upload_bytes: u64::MAX,
                ..good
            },
            TransientIoReceipt {
                upload_bytes: 0,
                uniform_bytes: 0,
                readback_bytes: 0,
                inspected_bytes: 0,
                ..good
            },
        ] {
            assert!(validate_audit(bad).is_err());
        }
    }

    #[test]
    fn probe_accepts_only_existing_model_and_receipt_paths_with_fixed_fixture_policy() {
        let source = include_str!("matmul_probe.rs");
        let run = source
            .split("pub fn run(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let run = run.split("pub fn run_fused_fault(").next().unwrap();
        assert_eq!(run.matches("std::env::var_os(").count(), 2);
        let fused = source
            .split("pub fn run_fused_fault(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert_eq!(fused.matches("std::env::var_os(").count(), 2);
        assert!(fused.contains("KUNO_TEST_GGUF") && fused.contains("KUNO_PUBLIC_RECEIPT"));
        assert!(fused.contains("create_new(true)"));
        assert!(!fused.contains("create(true)") && !fused.contains("truncate(true)"));
        assert!(run.contains("KUNO_TEST_GGUF") && run.contains("KUNO_PUBLIC_RECEIPT"));
        assert!(run.contains("create_new(true)"));
        assert!(!run.contains("create(true)") && !run.contains("truncate(true)"));
    }
}
