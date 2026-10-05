//! Optional fixed-public cached operator experiment, not an engine benchmark.
//! All original weights and activations are loaded before timing; the measured
//! interval includes submission through a completed output readback fence.
//! Numeric validation, sentinel initialization and owned-I/O cleanup are outside
//! that interval. There is no arbitrary prompt, count or iteration input.

use super::{
    FixtureLayout, IO_BUDGET, MAIN_BUDGET, OUTPUT_GAP, PublicMatrix, Result, audited, check_output,
    expectations, input_fixture, receipt, verify_bindings,
};
use crate::private_arena::{BufferLayout, PrivateArena};
use crate::quant_matvec::cooperative::cooperative_matvec;
use crate::quant_matvec::matmul::{BatchSpans, MAX_BATCH, MatrixGeometry, packed_matmul};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::{GgufFile, TensorType};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use zeroize::Zeroizing;

const MAX_OPERATOR_BYTES: usize = 2 * 1024 * 1024;
const ABBA_ROUNDS: usize = 3;
const OPERATORS: [(TensorType, usize, usize, &str); 3] = [
    (TensorType::Q4K, 640, 2560, ".ffn_gate_exps.weight"),
    (TensorType::Q5_0, 2560, 640, ".ffn_down_exps.weight"),
    (TensorType::Q8_0, 2560, 640, ".ffn_down_exps.weight"),
];

pub(super) fn enabled(raw: Option<&str>) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("KUNO_LAB_MATMUL_TIMING must be 0 or 1".into()),
    }
}

fn validate_operator(
    kind: TensorType,
    rows: usize,
    columns: usize,
    dimensions: &[u64],
    name: &str,
) -> Result<MatrixGeometry> {
    if !OPERATORS.iter().any(|&(dtype, r, c, suffix)| {
        dtype == kind
            && rows == r
            && columns == c
            && dimensions == [c as u64, r as u64, 512]
            && name.ends_with(suffix)
    }) {
        return Err("public cached operator geometry mismatch".into());
    }
    let geometry = MatrixGeometry::new(kind.gguf_id(), rows, columns, 4)?;
    if geometry.total_bytes() > MAX_OPERATOR_BYTES || !geometry.total_bytes().is_multiple_of(4) {
        return Err("public cached original operator exceeds packed byte bound".into());
    }
    timing_layout(rows, columns)?;
    Ok(geometry)
}

pub(super) fn load_public(
    client: &ComputeClient<WgpuRuntime>,
    file: &GgufFile,
) -> Result<Vec<PublicMatrix>> {
    let mut matrices = Vec::with_capacity(OPERATORS.len());
    for (kind, rows, columns, suffix) in OPERATORS {
        let info = file
            .tensors()
            .values()
            .find(|info| {
                info.dtype() == kind
                    && info.name().ends_with(suffix)
                    && info.dimensions() == [columns as u64, rows as u64, 512]
            })
            .ok_or("public cached original routed operator missing")?;
        let geometry = validate_operator(kind, rows, columns, info.dimensions(), info.name())?;
        let tensor = file
            .tensor(info.name())
            .ok_or("public cached original routed operator missing")?;
        if tensor.info().row_bytes() != geometry.row_bytes() as u64 {
            return Err("public cached routed operator row layout mismatch".into());
        }
        // Fixed PUBLIC expert index zero, not a router selection or private
        // access trace. Borrow one complete original matrix, never a modelcopy.
        let mut wire = Zeroizing::new(vec![0_u8; geometry.total_bytes()]);
        tensor
            .read_range(0, &mut wire)
            .map_err(|_| "public cached original operator read failed")?;
        let mut decoded = Zeroizing::new(vec![0_f32; rows * columns]);
        kind.decode(&wire, &mut decoded)
            .map_err(|_| "public cached original operator decode failed")?;
        if decoded.iter().any(|value| !value.is_finite()) {
            return Err("public cached original operator contains nonfinite values".into());
        }
        let handle = client.create_from_slice(&wire); // original PUBLIC cache
        matrices.push(PublicMatrix {
            kind,
            rows,
            columns,
            packed_first: 0,
            words: wire.len() / 4,
            handle,
            wire,
            decoded,
        });
    }
    Ok(matrices)
}

fn timing_layout(rows: usize, columns: usize) -> Result<FixtureLayout> {
    if !matches!((rows, columns), (640, 2560) | (2560, 640)) {
        return Err("public cached activation geometry mismatch".into());
    }
    // 64 f32 words = 256 bytes, including every token's irregular stride.
    // Existing cooperative matvec has no input-first scalar; its input Handle
    // subview must satisfy the storage-binding offset alignment instead.
    let input_stride = columns
        .checked_add(64)
        .ok_or("public timing layout overflow")?;
    let output_stride = rows
        .checked_add(32)
        .ok_or("public timing layout overflow")?;
    let input_words = MAX_BATCH
        .checked_mul(input_stride)
        .and_then(|n| n.checked_add(64))
        .ok_or("public timing layout overflow")?;
    let output_words = MAX_BATCH
        .checked_mul(output_stride)
        .and_then(|n| n.checked_add(64))
        .ok_or("public timing layout overflow")?;
    let charged = BufferLayout::new(input_words)?
        .owned_bytes
        .checked_add(BufferLayout::new(output_words)?.owned_bytes)
        .ok_or("public timing layout overflow")?;
    if charged > MAIN_BUDGET || charged as u64 > IO_BUDGET / 4 {
        return Err("public cached operator exceeds owned span budget".into());
    }
    Ok(FixtureLayout {
        input_first: 64,
        input_stride,
        input_words,
        output_first: 64,
        output_stride,
        output_words,
    })
}

/// Return checked runtime offsets into one fully owned activation buffer.
/// No slice may escape this immediate same-stream launch or the owner epoch.
fn input_view_offsets(
    layout: FixtureLayout,
    columns: usize,
    token: usize,
    full_bytes: usize,
) -> Result<(usize, usize)> {
    if token >= MAX_BATCH || columns == 0 || layout.input_stride < columns {
        return Err("public legacy input slice invalid".into());
    }
    let first = token
        .checked_mul(layout.input_stride)
        .and_then(|n| n.checked_add(layout.input_first))
        .ok_or("public legacy input slice overflow")?;
    let end = first
        .checked_add(columns)
        .ok_or("public legacy input slice overflow")?;
    let first_bytes = first
        .checked_mul(4)
        .ok_or("public legacy input slice overflow")?;
    let end_bytes = end
        .checked_mul(4)
        .ok_or("public legacy input slice overflow")?;
    if end > layout.input_words
        || end_bytes > full_bytes
        || !first_bytes.is_multiple_of(256)
        || first > u32::MAX as usize
        || end > u32::MAX as usize
    {
        return Err("public legacy input slice exceeds full span/alignment".into());
    }
    Ok((first_bytes, full_bytes - end_bytes))
}

// This local borrowed launch view never outlives its owning arena epoch and
// cannot serve as a model/pipeline cache key. Geometry alone is public.
struct Trial<'a> {
    matrix: &'a PublicMatrix,
    input: &'a Handle,
    output: &'a Handle,
    input_words: usize,
    output_words: usize,
    layout: FixtureLayout,
    spans: BatchSpans,
    geometries: [MatrixGeometry; 2],
    input_views: Vec<Handle>,
}
impl<'a> Trial<'a> {
    fn new(
        matrix: &'a PublicMatrix,
        input: &'a Handle,
        output: &'a Handle,
        input_words: usize,
        output_words: usize,
        layout: FixtureLayout,
    ) -> Result<Self> {
        let spans = BatchSpans::new(
            MAX_BATCH,
            0,
            layout.input_first,
            layout.input_stride,
            layout.output_first,
            layout.output_stride,
        );
        let geometries = [
            MatrixGeometry::new(matrix.kind.gguf_id(), matrix.rows, matrix.columns, 4)?,
            MatrixGeometry::new(matrix.kind.gguf_id(), matrix.rows, matrix.columns, 8)?,
        ];
        for geometry in geometries {
            geometry.validate_spans(&spans, matrix.words, input_words, output_words)?;
        }
        crate::quant_matvec::validate_layout(
            matrix.kind.gguf_id(),
            matrix.rows,
            matrix.columns,
            matrix.wire.len(),
            matrix.columns,
            matrix.rows,
        )?;
        let full = input_words
            .checked_mul(4)
            .ok_or("public legacy input span overflow")?;
        let mut input_views = Vec::with_capacity(MAX_BATCH);
        for token in 0..spans.count() {
            let (first, tail) = input_view_offsets(layout, matrix.columns, token, full)?;
            input_views.push(
                input
                    .clone()
                    .offset_start(first as u64)
                    .offset_end(tail as u64),
            );
        }
        Ok(Self {
            matrix,
            input,
            output,
            input_words,
            output_words,
            layout,
            spans,
            geometries,
            input_views,
        })
    }
}

// All view construction, overflow checks and span validation precede the clock.
// The measured path contains only original launches and the caller's readback.
fn dispatch(client: &ComputeClient<WgpuRuntime>, trial: &Trial<'_>, tile: usize) -> Result<()> {
    let matrix = trial.matrix;
    if tile == 0 {
        // Exactly eight existing cooperative launches; no changed decoder or
        // reduction tree. Only each input view and output-first are runtime.
        for (token, view) in trial.input_views.iter().enumerate() {
            let output_first = trial.spans.output_first() + token * trial.spans.output_stride();
            cooperative_matvec::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(matrix.rows as u32, 1, 1),
                CubeDim::new_1d(64),
                unsafe { ArrayArg::from_raw_parts(matrix.handle.clone(), matrix.words) },
                unsafe { ArrayArg::from_raw_parts(view.clone(), matrix.columns) },
                unsafe { ArrayArg::from_raw_parts(trial.output.clone(), trial.output_words) },
                matrix.rows,
                matrix.columns,
                matrix.kind.gguf_id(),
                output_first,
            );
        }
    } else {
        let geometry = match tile {
            4 => trial.geometries[0],
            8 => trial.geometries[1],
            _ => return Err("public timing candidate tile invalid".into()),
        };
        packed_matmul::launch::<WgpuRuntime>(
            client,
            geometry.grid(),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(matrix.handle.clone(), matrix.words) },
            unsafe { ArrayArg::from_raw_parts(trial.input.clone(), trial.input_words) },
            unsafe { ArrayArg::from_raw_parts(trial.output.clone(), trial.output_words) },
            trial.spans.count(),
            trial.spans.packed_first(),
            trial.spans.input_first(),
            trial.spans.input_stride(),
            trial.spans.output_first(),
            trial.spans.output_stride(),
            geometry.rows(),
            geometry.columns(),
            geometry.dtype(),
            geometry.tile_tokens(),
        );
    }
    Ok(())
}

fn sample(
    client: &ComputeClient<WgpuRuntime>,
    trial: &Trial<'_>,
    initial: &[u8],
    expected: &[f32],
    tile: usize,
) -> Result<serde_json::Value> {
    let matrix = trial.matrix;
    let y = trial.output;
    let y_words = trial.output_words;
    let layout = trial.layout;
    // Move only guarded explicitly owned host storage. This initialization and
    // its completed sanitizer fence are entirely outside the timed interval.
    let mut upload = Zeroizing::new(initial.to_vec());
    client
        .write_private(y.clone(), std::mem::take(&mut *upload))
        .map_err(|_| "public timing output initialization failed")?;
    let before = audited(client)?;
    let start = Instant::now();
    dispatch(client, trial, tile)?;
    let actual = client
        .read_one(y.clone())
        .map_err(|_| "public timing readback failed")?;
    let elapsed_ns =
        u64::try_from(start.elapsed().as_nanos()).map_err(|_| "public timing overflow")?;
    let error = check_output(
        &actual,
        expected,
        matrix.rows,
        layout,
        MAX_BATCH,
        y_words * 4,
    )?;
    drop(actual);
    let after = audited(client)?;
    if after.uniform_bytes == 0 || after.readback_bytes == 0 || before.upload_bytes == 0 {
        return Err("public timing audit missed owned I/O kind".into());
    }
    Ok(
        serde_json::json!({"tile_tokens":tile,"launches":if tile==0 {8}else {1},"elapsed_ns":elapsed_ns,
        "max_abs_error":error,"gaps_and_padding_preserved":true,"before_io":receipt(before),"after_io":receipt(after)}),
    )
}

pub(super) fn run(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    matrices: &[PublicMatrix],
    retired: &AtomicBool,
) -> Result<serde_json::Value> {
    if matrices.len() != OPERATORS.len() {
        return Err("public cached operator matrix count mismatch".into());
    }
    let mut reports = Vec::with_capacity(OPERATORS.len());
    for matrix in matrices {
        let layout = timing_layout(matrix.rows, matrix.columns)?;
        let input = input_fixture(layout, matrix.columns, MAX_BATCH);
        let expected = expectations(
            matrix.rows,
            matrix.columns,
            &matrix.decoded,
            &input,
            layout,
            MAX_BATCH,
        )?;
        let x = arena.upload_f32(&input)?;
        let y = arena.upload_f32(&Zeroizing::new(vec![OUTPUT_GAP; layout.output_words]))?;
        let x_layout = arena.layout(x)?;
        let y_layout = arena.layout(y)?;
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
        let trial = Trial::new(
            matrix,
            &retained_x,
            &retained_y,
            x_layout.owned_words,
            y_layout.owned_words,
            layout,
        )?;
        let mut initial = Zeroizing::new(vec![0_u8; y_layout.owned_bytes]);
        for word in initial[..layout.output_words * 4].chunks_exact_mut(4) {
            word.copy_from_slice(&OUTPUT_GAP.to_le_bytes());
        }
        let mut warmup = Vec::with_capacity(3);
        for tile in [0, 4, 8] {
            warmup.push(sample(client, &trial, &initial, &expected, tile)?);
        }
        let mut pairs = Vec::with_capacity(2);
        for candidate in [4, 8] {
            let mut rounds = Vec::with_capacity(ABBA_ROUNDS);
            for round in 0..ABBA_ROUNDS {
                let mut samples = Vec::with_capacity(4);
                for tile in [0, candidate, candidate, 0] {
                    let datum = sample(client, &trial, &initial, &expected, tile)?;
                    println!(
                        "public_cached_matmul dtype={} rows={} columns={} count=8 candidate={} round={} tile={} elapsed_ns={}",
                        matrix.kind.gguf_id(),
                        matrix.rows,
                        matrix.columns,
                        candidate,
                        round,
                        tile,
                        datum["elapsed_ns"]
                    );
                    samples.push(datum);
                }
                rounds.push(samples);
            }
            pairs.push(serde_json::json!({"candidate_tile":candidate,"abba_rounds":rounds}));
        }
        // Inference subviews and private runtime addressing end before owner
        // reset; retained full handles are solely post-erasure audit views.
        drop(trial);
        arena.reset()?;
        if arena.handle(x).is_ok() || arena.handle(y).is_ok() || retired.load(Ordering::Acquire) {
            return Err("public timing reset did not invalidate owning spans".into());
        }
        for (handle, bytes) in [
            (retained_x, x_layout.owned_bytes),
            (retained_y, y_layout.owned_bytes),
        ] {
            let erased = client
                .read_one(handle)
                .map_err(|_| "public timing main audit read failed")?;
            if erased.len() != bytes || erased.iter().any(|&byte| byte != 0) {
                return Err("public timing full main span not erased".into());
            }
            drop(erased);
        }
        let main_audit = audited(client)?;
        let original = client
            .read_one(matrix.handle.clone())
            .map_err(|_| "public timing cache check failed")?;
        if original.len() != matrix.wire.len() || &*original != matrix.wire.as_slice() {
            return Err("public cached original matrix changed".into());
        }
        drop(original);
        let cache_audit = audited(client)?;
        reports.push(serde_json::json!({"dtype":matrix.kind.gguf_id(),"rows":matrix.rows,"columns":matrix.columns,
            "count":8,"expert_index":0,"original_packed_bytes":matrix.wire.len(),"warmup":warmup,"pairs":pairs,
            "main_erased":true,"public_cache_preserved":true,"main_audit":receipt(main_audit),"cache_audit":receipt(cache_audit)}));
    }
    Ok(
        serde_json::json!({"scope":"fixed-public-cached-operator-experiment-only","clock":"dispatch through completed output readback",
        "model_input_transfers_timed":false,"compiler_warmup_excluded":true,"abba_repetitions":ABBA_ROUNDS,
        "legacy_launches":8,"candidate_launches":1,"matrices":reports}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_timing_flag_is_explicit_and_bounded() {
        assert!(!enabled(None).unwrap());
        assert!(!enabled(Some("0")).unwrap());
        assert!(enabled(Some("1")).unwrap());
        for bad in ["", "2", "true", "01", " 1", "-1"] {
            assert!(enabled(Some(bad)).is_err());
        }
    }

    #[test]
    fn original_routed_operator_shapes_are_bounded_exact_and_never_ple_rows() {
        for (kind, rows, columns, suffix) in OPERATORS {
            let dimensions = [columns as u64, rows as u64, 512];
            let name = format!("blk.0{suffix}");
            let geometry = validate_operator(kind, rows, columns, &dimensions, &name).unwrap();
            assert!(geometry.total_bytes() <= MAX_OPERATOR_BYTES);
            assert_eq!(
                geometry.total_bytes(),
                match kind {
                    TensorType::Q4K => 921600,
                    TensorType::Q5_0 => 1126400,
                    TensorType::Q8_0 => 1740800,
                    _ => unreachable!(),
                }
            );
            assert!(
                validate_operator(kind, rows, columns, &[columns as u64, rows as u64], &name)
                    .is_err()
            );
            assert!(
                validate_operator(
                    kind,
                    rows,
                    columns,
                    &[columns as u64, rows as u64, 511],
                    &name
                )
                .is_err()
            );
            assert!(
                validate_operator(
                    kind,
                    rows,
                    columns,
                    &dimensions,
                    "per_layer_token_embd.weight"
                )
                .is_err()
            );
            assert!(validate_operator(TensorType::F32, rows, columns, &dimensions, &name).is_err());
        }
        assert!(timing_layout(7, 160).is_err());
        assert!(timing_layout(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn all_eight_legacy_runtime_input_views_are_aligned_disjoint_and_bounded() {
        for (rows, columns) in [(640, 2560), (2560, 640)] {
            let layout = timing_layout(rows, columns).unwrap();
            let full = BufferLayout::new(layout.input_words).unwrap().owned_bytes;
            let mut prior_end = 0;
            for token in 0..MAX_BATCH {
                let (first, tail) = input_view_offsets(layout, columns, token, full).unwrap();
                let end = full - tail;
                assert_eq!(end - first, columns * 4);
                assert!(first >= prior_end && first.is_multiple_of(256));
                prior_end = end;
            }
            assert!(input_view_offsets(layout, columns, MAX_BATCH, full).is_err());
            assert!(input_view_offsets(layout, columns, 0, columns * 4).is_err());
            let mut bad = layout;
            bad.input_first += 1;
            assert!(input_view_offsets(bad, columns, 0, full).is_err());
            bad.input_first = usize::MAX;
            assert!(input_view_offsets(bad, columns, 0, full).is_err());
            bad = layout;
            bad.input_stride = usize::MAX;
            assert!(input_view_offsets(bad, columns, 2, full).is_err());
        }
    }
}
