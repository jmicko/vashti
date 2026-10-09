//! Fixed public IQ operator calibration, independent of any downloaded GGUF.
//! Packed bytes and exact expected values come from pinned original GGML C
//! decoders. The temporary C generator is never linked to this Rust engine.
//! No prompts, tokenizers, arbitrary payloads, model rows or timing mode.

use super::*;
use crate::quant_matvec::{IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, Q5_K, iq_fixtures};

const TYPES: [u32; 7] = [IQ3_XXS, IQ4_NL, IQ4_XS, IQ2_S, IQ3_S, Q5_K, Q2_0];
const ORACLE_PREFIXES: [usize; 2] = [0, 3];
const PROJECTION_CASES: usize = TYPES.len() * 3 * (4 * 3 + 1);
const SHARED_TYPES: [u32; 3] = [IQ4_NL, IQ4_XS, Q5_K];
const SHARED_DENSE_CASES: usize = 120;

/// Checked before device selection/allocation. This entry only consumes its
/// fixed compiled fixtures and produces a new numeric-only receipt.
pub fn registration(argc: usize, receipt: bool, model: bool, timing: bool) -> Result<()> {
    if argc != 3 || !receipt || model || timing {
        return Err("IQ probe requires card, fixed mode and receipt; no model/timing input".into());
    }
    Ok(())
}

fn columns(dtype: u32) -> [usize; 3] {
    let block = crate::quant_matvec::block_layout(dtype).unwrap().0;
    [block, block * 2, if dtype == IQ4_NL { 160 } else { 2560 }]
}

fn public_matrix(
    client: &ComputeClient<WgpuRuntime>,
    dtype: u32,
    columns: usize,
    prefix: usize,
) -> Result<PublicMatrix> {
    let rows = MAX_ROWS;
    let geometry = MatrixGeometry::new(dtype, rows, columns, 1)?;
    fixture_layout(rows, columns)?;
    let packed = Zeroizing::new(iq_fixtures::packed(dtype, rows, columns, 0));
    let decoded = Zeroizing::new(iq_fixtures::oracle(dtype, rows, columns, 0));
    if packed.len() != geometry.total_bytes()
        || decoded.len() != rows * columns
        || decoded.iter().any(|x| !x.is_finite())
        || packed.len() > MAX_PACKED_BYTES
    {
        return Err("fixed IQ projection fixture extent".into());
    }
    let wire_len = (prefix + packed.len()).div_ceil(4) * 4;
    let mut wire = Zeroizing::new(vec![0x53_u8; wire_len]);
    wire[prefix..prefix + packed.len()].copy_from_slice(&packed);
    let handle = client.create_from_slice(&wire); // immutable PUBLIC fixture
    Ok(PublicMatrix {
        kind: TensorType::from_gguf_id(dtype).unwrap(),
        rows,
        columns,
        packed_first: prefix,
        words: wire_len / 4,
        handle,
        wire,
        decoded,
    })
}

struct Oracle {
    dtype: u32,
    rows: usize,
    columns: usize,
    prefix: usize,
    words: usize,
    handle: Handle,
    wire: Zeroizing<Vec<u8>>,
    expected: Zeroizing<Vec<f32>>,
}

fn public_oracle(client: &ComputeClient<WgpuRuntime>, dtype: u32, prefix: usize) -> Result<Oracle> {
    let (columns, block_bytes) = crate::quant_matvec::block_layout(dtype).unwrap();
    let (records, _) = iq_fixtures::records(dtype);
    let record_bytes = block_bytes + columns * 4;
    if !records.len().is_multiple_of(record_bytes) || ![0, 3].contains(&prefix) {
        return Err("fixed IQ original oracle geometry".into());
    }
    let rows = records.len() / record_bytes;
    let mut wire = Zeroizing::new(vec![0x53; (prefix + rows * block_bytes).div_ceil(4) * 4]);
    let mut expected = Zeroizing::new(Vec::with_capacity(rows * columns));
    for (row, record) in records.chunks_exact(record_bytes).enumerate() {
        let first = prefix + row * block_bytes;
        wire[first..first + block_bytes].copy_from_slice(&record[..block_bytes]);
        expected.extend(
            record[block_bytes..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    crate::quant_matvec::validate_layout(dtype, rows, columns, rows * block_bytes, columns, rows)?;
    if expected.len() != rows * columns || expected.iter().any(|x| !x.is_finite()) {
        return Err("fixed IQ original oracle values".into());
    }
    let handle = client.create_from_slice(&wire); // immutable PUBLIC fixture
    Ok(Oracle {
        dtype,
        rows,
        columns,
        prefix,
        words: wire.len() / 4,
        handle,
        wire,
        expected,
    })
}

/// Use the actual original-packed production decoder, with no arithmetic
/// reduction to hide a sign/codebook/addressing error behind dot tolerance.
#[cube(launch)]
fn original_iq_values(
    weights: &Array<u32>,
    output: &mut Array<f32>,
    packed_first: usize,
    output_first: usize,
    output_stride: usize,
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] dtype: u32,
) {
    let index = ABSOLUTE_POS;
    if index < rows * columns {
        let row = index / columns;
        let column = index % columns;
        let row_bytes = crate::quant_matvec::packed_row_bytes(columns, dtype);
        output[output_first + row * output_stride + column] = crate::quant_matvec::packed_value(
            weights,
            packed_first + row * row_bytes,
            column,
            dtype,
        );
    }
}

fn check_oracle(
    bytes: &[u8],
    expected_values: &[f32],
    rows: usize,
    columns: usize,
    owned: BufferLayout,
) -> Result<()> {
    let output_first = 5;
    let output_stride = columns + 7;
    if bytes.len() != owned.owned_bytes
        || !bytes.len().is_multiple_of(4)
        || expected_values.len() != rows * columns
        || owned.logical_words != output_first + rows * output_stride + 9
    {
        return Err("fixed IQ oracle full owner extent".into());
    }
    for (index, word) in bytes.chunks_exact(4).enumerate() {
        let actual = u32::from_le_bytes(word.try_into()?);
        let relative = index.saturating_sub(output_first);
        let row = relative / output_stride;
        let column = relative % output_stride;
        let expected = if index >= owned.logical_words {
            0
        } else if index >= output_first && row < rows && column < columns {
            expected_values[row * columns + column].to_bits()
        } else {
            OUTPUT_GAP.to_bits()
        };
        if actual != expected {
            // This entry accepts only compiled public C records, never model
            // or request content. Report at most the first numeric mismatch;
            // gap/padding failures expose only the bounded owning word index.
            if index >= owned.logical_words {
                eprintln!("public_iq_oracle_mismatch kind=padding word={index}");
            } else if index >= output_first && row < rows && column < columns {
                eprintln!(
                    "public_iq_oracle_mismatch kind=value word={index} row={row} column={column} actual_bits={actual} expected_bits={expected}"
                );
            } else {
                eprintln!("public_iq_oracle_mismatch kind=gap word={index}");
            }
            return Err(
                "fixed IQ GPU decode differs from independent C bits or changed gap".into(),
            );
        }
    }
    Ok(())
}

// Fixed phases for this compiled-public-only probe. Never format the underlying
// error, backend payload or any arbitrary tensor/input in failure diagnostics.
fn oracle_stage<T>(oracle: &Oracle, phase: &'static str, result: Result<T>) -> Result<T> {
    if result.is_err() {
        eprintln!(
            "public_iq_oracle_failure dtype={} prefix={} phase={phase}",
            oracle.dtype, oracle.prefix
        );
    }
    result
}

fn oracle_cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    oracles: &[Oracle],
    retired: &AtomicBool,
    shared: bool,
) -> Result<Vec<serde_json::Value>> {
    let mut reports = Vec::with_capacity(TYPES.len() * ORACLE_PREFIXES.len());
    for oracle in oracles
        .iter()
        .filter(|o| !shared || SHARED_TYPES.contains(&o.dtype))
    {
        println!(
            "public_iq_oracle dtype={} prefix={} blocks={} values={}",
            oracle.dtype,
            oracle.prefix,
            oracle.rows,
            oracle.expected.len()
        );
        let output_first = 5;
        let output_stride = oracle.columns + 7;
        let output_words = output_first + oracle.rows * output_stride + 9;
        let initial = Zeroizing::new(vec![OUTPUT_GAP; output_words]);
        let output = oracle_stage(
            oracle,
            "upload",
            arena.upload_f32(&initial).map_err(Into::into),
        )?;
        let layout = oracle_stage(oracle, "layout", arena.layout(output).map_err(Into::into))?;
        let retained =
            oracle_stage(oracle, "handle", arena.handle(output).map_err(Into::into))?.clone();
        let public_resource = oracle_stage(
            oracle,
            "public_resource",
            client
                .get_resource(oracle.handle.clone())
                .map_err(Into::into),
        )?;
        let private_resource = oracle_stage(
            oracle,
            "private_resource",
            client.get_resource(retained.clone()).map_err(Into::into),
        )?;
        let public = public_resource.resource();
        let private = private_resource.resource();
        oracle_stage(
            oracle,
            "public_extent",
            validate_physical_extent(public.offset, public.size, public.buffer.size(), false),
        )?;
        oracle_stage(
            oracle,
            "private_extent",
            validate_physical_extent(private.offset, private.size, private.buffer.size(), true),
        )?;
        if public.size != oracle.wire.len() as u64
            || private.size != layout.owned_bytes as u64
            || public.buffer == private.buffer
            || oracle.handle.stream != retained.stream
        {
            return oracle_stage(
                oracle,
                "binding",
                Err("fixed IQ oracle binding ownership".into()),
            );
        }
        // SAFETY: fixed compiled original rows, same-client nonaliasing public
        // weights and complete private output owner were validated above.
        if shared {
            crate::next_shared::public_fixture_values(
                client,
                crate::next_shared::MatrixSpec::new(oracle.rows, oracle.columns, oracle.dtype)?,
                oracle.handle.clone(),
                retained.clone(),
                oracle.prefix,
                output_first,
                output_stride,
                layout.owned_words,
            )?;
        } else {
            original_iq_values::launch::<WgpuRuntime>(
                client,
                CubeCount::Static((oracle.rows * oracle.columns).div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                unsafe { ArrayArg::from_raw_parts(oracle.handle.clone(), oracle.words) },
                unsafe { ArrayArg::from_raw_parts(retained.clone(), layout.owned_words) },
                oracle.prefix,
                output_first,
                output_stride,
                oracle.rows,
                oracle.columns,
                oracle.dtype,
            );
        }
        let read = FusedReadLease(oracle_stage(
            oracle,
            "read",
            client.read_one(retained.clone()).map_err(Into::into),
        )?);
        oracle_stage(
            oracle,
            "bits",
            check_oracle(
                &read.0,
                &oracle.expected,
                oracle.rows,
                oracle.columns,
                layout,
            ),
        )?;
        drop(read);
        let io = oracle_stage(oracle, "io", audited(client))?;
        oracle_stage(oracle, "reset", arena.reset().map_err(Into::into))?;
        if arena.handle(output).is_ok() || retired.load(Ordering::Acquire) {
            return oracle_stage(
                oracle,
                "invalidation",
                Err("fixed IQ oracle owner invalidation".into()),
            );
        }
        let zero = FusedReadLease(oracle_stage(
            oracle,
            "erase_read",
            client.read_one(retained).map_err(Into::into),
        )?);
        if zero.0.len() != layout.owned_bytes || zero.0.iter().any(|&b| b != 0) {
            return oracle_stage(
                oracle,
                "erase_bits",
                Err("fixed IQ oracle full owner erase".into()),
            );
        }
        drop(zero);
        oracle_stage(oracle, "erase_io", audited(client))?;
        let original = FusedReadLease(oracle_stage(
            oracle,
            "source_read",
            client.read_one(oracle.handle.clone()).map_err(Into::into),
        )?);
        if &*original.0 != oracle.wire.as_slice() {
            return oracle_stage(
                oracle,
                "source_bytes",
                Err("fixed IQ oracle packed source mutation".into()),
            );
        }
        drop(original);
        oracle_stage(oracle, "source_io", audited(client))?;
        reports.push(
            serde_json::json!({"dtype":oracle.dtype,"blocks":oracle.rows,
            "values":oracle.expected.len(),"packed_prefix_bytes":oracle.prefix,
            "packed_row_bytes":crate::quant_matvec::block_layout(oracle.dtype).unwrap().1,
            "independent_c_bits_exact":true,"shared_packed_vec4_decoder":shared,"gaps_and_full_padding_preserved":true,
            "main_erased":true,"public_cache_preserved":true,"io":receipt(io)}),
        );
    }
    if reports.len()
        != if shared {
            SHARED_TYPES.len()
        } else {
            TYPES.len()
        } * ORACLE_PREFIXES.len()
    {
        return Err("fixed IQ oracle inventory incomplete".into());
    }
    Ok(reports)
}

fn shared_widths(dtype: u32) -> &'static [usize] {
    match dtype {
        IQ4_NL => &[32, 160, 640, 2560],
        IQ4_XS | Q5_K => &[256, 512, 2560],
        _ => &[],
    }
}
struct SharedMatrix {
    dtype: u32,
    columns: usize,
    prefix: usize,
    root: Handle,
    view: Handle,
    wire: Zeroizing<Vec<u8>>,
    decoded: Zeroizing<Vec<f32>>,
}
fn public_shared_matrix(
    client: &ComputeClient<WgpuRuntime>,
    dtype: u32,
    columns: usize,
    prefix: usize,
) -> Result<SharedMatrix> {
    if !shared_widths(dtype).contains(&columns) || ![0, 256].contains(&prefix) {
        return Err("shared IQ fixed fixture geometry".into());
    }
    let packed = iq_fixtures::packed(dtype, MAX_ROWS, columns, 0);
    let size = packed.len().div_ceil(4) * 4;
    let owned = crate::main_admission::page_bytes((prefix + size) as u64)? as usize;
    let mut wire = Zeroizing::new(vec![0x53; owned]);
    wire[prefix..prefix + packed.len()].copy_from_slice(&packed);
    let root = client.create_public_owned(wire.to_vec());
    let view = root
        .clone()
        .offset_start(prefix as u64)
        .offset_end((owned - prefix - size) as u64);
    Ok(SharedMatrix {
        dtype,
        columns,
        prefix,
        root,
        view,
        wire,
        decoded: Zeroizing::new(iq_fixtures::oracle(dtype, MAX_ROWS, columns, 0)),
    })
}
fn shared_dense_case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    m: &SharedMatrix,
    rows: usize,
    bad: usize,
) -> Result<serde_json::Value> {
    const INPUT_FIRST: usize = 4;
    const OUTPUT_FIRST: usize = 7;
    let mut input = Zeroizing::new(vec![INPUT_GAP; INPUT_FIRST + m.columns + 12]);
    for (i, x) in input[INPUT_FIRST..INPUT_FIRST + m.columns]
        .iter_mut()
        .enumerate()
    {
        *x = ((i * 17 % 97) as f32 - 48.) / 32.;
    }
    if bad == 1 {
        input[INPUT_FIRST + m.columns - 1] = f32::NAN;
    }
    let out_words = OUTPUT_FIRST + rows + 13;
    let mut initial = Zeroizing::new(vec![OUTPUT_GAP; out_words]);
    if bad == 2 {
        initial[0] = f32::NAN;
    }
    let x = arena.upload_u32(&fused_input_words(&input)?)?;
    let y = arena.upload_u32(&fused_input_words(&initial)?)?;
    let original = arena.upload_u32(&fused_input_words(&initial)?)?;
    let f = arena.upload_u32(&[4])?;
    let original_f = arena.upload_u32(&[4])?;
    let ids = [x, y, original, f, original_f];
    let handles: Vec<_> = ids
        .iter()
        .map(|&id| arena.handle(id).cloned())
        .collect::<std::result::Result<_, _>>()?;
    let layouts: Vec<_> = ids
        .iter()
        .map(|&id| arena.layout(id))
        .collect::<std::result::Result<_, _>>()?;
    let full = layouts.iter().map(|l| l.owned_bytes).sum::<usize>();
    if full > MAIN_BUDGET || 2 * full + 8 * 512 > IO_BUDGET as usize {
        return Err("shared IQ original owning budget".into());
    }
    let mut all = vec![m.root.clone()];
    all.extend(handles.iter().cloned());
    let resources = all
        .iter()
        .map(|h| client.get_resource(h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (i, r) in resources.iter().enumerate() {
        let r = r.resource();
        validate_physical_extent(r.offset, r.size, r.buffer.size(), true)?;
        if r.size != all[i].size_in_used()
            || all[i].stream != all[0].stream
            || resources[..i]
                .iter()
                .any(|p| p.resource().buffer == r.buffer)
        {
            return Err("shared IQ full physical owner binding".into());
        }
    }
    let view = client.get_resource(m.view.clone())?;
    if view.resource().buffer != resources[0].resource().buffer
        || view.resource().offset != m.prefix as u64
        || view.resource().size != m.view.size_in_used()
    {
        return Err("shared IQ original checked source subview".into());
    }
    let spec = crate::next_shared::MatrixSpec::new(rows, m.columns, m.dtype)?;
    let arg = |i: usize, n| unsafe { ArrayArg::from_raw_parts(handles[i].clone(), n) };
    crate::quant_matvec::fast::packed_single::launch::<WgpuRuntime>(
        client,
        crate::quant_matvec::fast::grid(rows),
        crate::quant_matvec::fast::cube(),
        unsafe { ArrayArg::from_raw_parts(m.view.clone(), m.view.size_in_used() as usize / 4) },
        arg(0, layouts[0].owned_words),
        arg(2, layouts[2].owned_words),
        1,
        0,
        INPUT_FIRST,
        OUTPUT_FIRST,
        rows,
        m.columns,
        m.dtype,
        MatrixGeometry::new(m.dtype, rows, m.columns, 1)?.row_bytes(),
    );
    crate::resident_util::verify_finite::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(layouts[2].owned_words.div_ceil(64) as u32, 1, 1),
        CubeDim::new_1d(64),
        arg(2, layouts[2].owned_words),
        arg(4, 1),
    );
    let fault = handles[3]
        .clone()
        .offset_end((layouts[3].owned_bytes - 4) as u64);
    let fault_resource = client.get_resource(fault.clone())?;
    if fault_resource.resource().buffer != resources[4].resource().buffer
        || fault_resource.resource().offset != 0
        || fault_resource.resource().size != 4
    {
        return Err("shared IQ fault subview escaped full owner".into());
    }
    crate::next_shared::matrix_fault(
        client,
        spec,
        m.view.clone(),
        handles[0].clone(),
        handles[1].clone(),
        layouts[0].owned_words,
        layouts[1].owned_words,
        INPUT_FIRST,
        OUTPUT_FIRST,
        fault,
        rows,
        OUTPUT_FIRST,
        true,
    )?;
    let mut snapshots = Vec::new();
    for (i, h) in handles.iter().enumerate() {
        let read = FusedReadLease(
            client
                .read_one(h.clone())
                .map_err(|_| "shared IQ full owning read")?,
        );
        snapshots.push(decode_fused_owning_read(&read.0, layouts[i])?);
    }
    audited(client)?;
    if snapshots[3][0] != if bad == 0 { 4 } else { 5 } || snapshots[3] != snapshots[4] {
        return Err("shared IQ sticky complete finite flag".into());
    }
    if snapshots[3][1..]
        .iter()
        .chain(&snapshots[4][1..])
        .any(|&v| v != 0)
    {
        return Err("shared IQ fault full padding changed".into());
    }
    let mut maximum = 0f32;
    for (i, (&actual, &baseline)) in snapshots[1].iter().zip(snapshots[2].iter()).enumerate() {
        let active = (OUTPUT_FIRST..OUTPUT_FIRST + rows).contains(&i);
        if active {
            let a = f32::from_bits(actual);
            let b = f32::from_bits(baseline);
            if bad == 1 {
                if !a.is_nan() || !b.is_nan() {
                    return Err("shared IQ nonfinite input not rejected".into());
                }
            } else {
                let r = i - OUTPUT_FIRST;
                let expected = crate::quant_matvec::cooperative::reference_tree_dot(
                    &m.decoded[r * m.columns..(r + 1) * m.columns],
                    &input[INPUT_FIRST..INPUT_FIRST + m.columns],
                )?;
                let error = (a - expected).abs();
                if !a.is_finite()
                    || !b.is_finite()
                    || error > 1e-4 + 3e-4 * expected.abs()
                    || (a - b).abs() > 1e-4 + 3e-4 * b.abs()
                {
                    return Err("shared IQ independent C and original GPU projection".into());
                }
                maximum = maximum.max(error);
            }
        } else {
            let expected = if i < out_words {
                initial[i].to_bits()
            } else {
                0
            };
            if actual != expected || baseline != expected {
                return Err("shared IQ output prefix gap or complete padding".into());
            }
        }
    }
    for (i, &value) in snapshots[0].iter().enumerate() {
        if value
            != if i < input.len() {
                input[i].to_bits()
            } else {
                0
            }
        {
            return Err("shared IQ original input or owning padding changed".into());
        }
    }
    drop(snapshots);
    drop(resources);
    drop(view);
    drop(fault_resource);
    arena.reset()?;
    for (i, h) in handles.iter().enumerate() {
        let read = FusedReadLease(
            client
                .read_one(h.clone())
                .map_err(|_| "shared IQ complete zero read")?,
        );
        if read.0.len() != layouts[i].owned_bytes || read.0.iter().any(|&v| v != 0) {
            return Err("shared IQ complete owner erasure".into());
        }
        drop(read);
        audited(client)?;
    }
    let cache = FusedReadLease(
        client
            .read_one(m.root.clone())
            .map_err(|_| "shared IQ source preservation read")?,
    );
    if &*cache.0 != m.wire.as_slice() {
        return Err("shared IQ original source prefix or full page padding changed".into());
    }
    drop(cache);
    audited(client)?;
    Ok(
        serde_json::json!({"dtype":m.dtype,"rows":rows,"columns":m.columns,"source_prefix_bytes":m.prefix,"input_prefix_words":INPUT_FIRST,"output_prefix_words":OUTPUT_FIRST,"fault_case":bad,"max_independent_c_abs_error":maximum,"original_gpu_compared":true,"full_owner_padding_preserved":true,"source_full_page_bytes":m.wire.len(),"full_private_bytes":full,"erased":true}),
    )
}
fn shared_dense_cases(
    client: &ComputeClient<WgpuRuntime>,
    matrices: &[SharedMatrix],
    public_bound: usize,
) -> Result<serde_json::Value> {
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), MAIN_BUDGET, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let mut ledger = crate::main_admission::MainAdmission::new(
        (MAIN_BUDGET + public_bound) as u64,
        client
            .memory_usage()
            .map_err(|_| "shared IQ baseline telemetry")?
            .bytes_reserved,
    )?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Vec<serde_json::Value>> {
            let mut cases = Vec::new();
            for m in matrices {
                for rows in [1, MAX_ROWS] {
                    for bad in 0..3 {
                        let requests = [
                            INPUT_LAYOUT_BYTES,
                            OUTPUT_LAYOUT_BYTES,
                            OUTPUT_LAYOUT_BYTES,
                            OUTPUT_LAYOUT_BYTES,
                            OUTPUT_LAYOUT_BYTES,
                        ];
                        ledger.observe_before(
                            client
                                .memory_usage()
                                .map_err(|_| "shared IQ precharge telemetry")?
                                .bytes_reserved,
                        )?;
                        ledger.precharge(&requests)?;
                        cases.push(shared_dense_case(client, &mut arena, m, rows, bad)?);
                        ledger.completed_fence(
                            client
                                .memory_usage()
                                .map_err(|_| "shared IQ completed telemetry")?
                                .bytes_reserved,
                        )?;
                    }
                }
            }
            Ok(cases)
        },
    ));
    let closed = arena.close();
    let cases = match (attempt, closed, retired.load(Ordering::Acquire)) {
        (Ok(Ok(c)), Ok(()), false) => c,
        _ => return Err("shared IQ dense operators failed; no success receipt".into()),
    };
    if cases.len() != SHARED_DENSE_CASES {
        return Err("shared IQ dense fixture inventory incomplete".into());
    }
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("shared IQ complete zero receipt absent")?;
    Ok(
        serde_json::json!({"scope":"fixed-public-original-F32-shared-IQ4-Q5K-dense-only","cases":cases,"format_count":SHARED_TYPES.len(),"private_cap_bytes":MAIN_BUDGET,"public_full_page_upper_bytes":public_bound,"io_cap_bytes":IO_BUDGET,"main_erased_bytes":zero.bytes,"main_erased_buffers":zero.buffers,"whole_graph_or_performance_accepted":false}),
    )
}
const INPUT_LAYOUT_BYTES: u64 = 32 * 1024;
const OUTPUT_LAYOUT_BYTES: u64 = 32 * 1024;

pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let path =
        std::env::var_os("KUNO_PUBLIC_RECEIPT").ok_or("fixed IQ receipt destination required")?;
    // Prewarm every immutable public weight buffer before configuring the
    // owning private upload/readback path. No model file is opened anywhere.
    let mut matrices = Vec::with_capacity(TYPES.len() * 3);
    let mut finite_matrices = Vec::with_capacity(TYPES.len());
    let mut oracles = Vec::with_capacity(TYPES.len() * ORACLE_PREFIXES.len());
    let mut shared_matrices = Vec::new();
    for dtype in TYPES {
        for width in columns(dtype) {
            matrices.push(public_matrix(client, dtype, width, 3)?);
        }
        finite_matrices.push(public_matrix(client, dtype, columns(dtype)[2], 0)?);
        for prefix in ORACLE_PREFIXES {
            oracles.push(public_oracle(client, dtype, prefix)?);
        }
    }
    let original_public_bound = matrices
        .iter()
        .map(|m| m.wire.len())
        .chain(finite_matrices.iter().map(|m| m.wire.len()))
        .chain(oracles.iter().map(|o| o.wire.len()))
        .try_fold(0u64, |n, bytes| -> Result<u64> {
            Ok(n + crate::main_admission::page_bytes(bytes as u64)?)
        })?;
    let mut prewarm = crate::main_admission::MainAdmission::new(
        MAIN_BUDGET as u64 + original_public_bound,
        client
            .memory_usage()
            .map_err(|_| "shared IQ original baseline")?
            .bytes_reserved,
    )?;
    for dtype in SHARED_TYPES {
        for &width in shared_widths(dtype) {
            for prefix in [0, 256] {
                let row = crate::quant_matvec::block_layout(dtype).unwrap();
                let owned = crate::main_admission::page_bytes(
                    (prefix + (MAX_ROWS * width / row.0 * row.1).div_ceil(4) * 4) as u64,
                )?;
                prewarm.precharge(&[owned])?;
                shared_matrices.push(public_shared_matrix(client, dtype, width, prefix)?);
                prewarm.observe_after(
                    client
                        .memory_usage()
                        .map_err(|_| "shared IQ source postcharge")?
                        .bytes_reserved,
                )?;
            }
        }
    }
    client.configure_transient_io(IO_BUDGET)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), MAIN_BUDGET, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let oracle = oracle_cases(client, &mut arena, &oracles, &retired, false)?;
        let shared_oracle = oracle_cases(client, &mut arena, &oracles, &retired, true)?;
        let projections = cases(client, &mut arena, &matrices, &retired)?;
        if projections.len() != PROJECTION_CASES {
            return Err("fixed IQ projection inventory incomplete".into());
        }
        Ok((oracle, projections, shared_oracle))
    }));
    let closed = arena.close();
    let (oracle, projections, shared_oracle) =
        match (attempt, closed, retired.load(Ordering::Acquire)) {
            (Ok(Ok(reports)), Ok(()), false) => reports,
            _ => return Err("fixed IQ operators failed; no success receipt".into()),
        };
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("fixed IQ owned zero receipt absent")?;
    let public_bound = TYPES.len() * (MAX_PACKED_BYTES + 4).next_power_of_two();
    let finite = fused_cases(
        client,
        &finite_matrices,
        public_bound,
        FusedProjection::Cooperative,
    )?;
    let single = fused_cases(
        client,
        &finite_matrices,
        public_bound,
        FusedProjection::SingleSubgroup,
    )?;
    if finite["cases"].as_array().map(Vec::len) != Some(TYPES.len() * 4)
        || single["cases"].as_array().map(Vec::len) != Some(TYPES.len() * 4)
    {
        return Err("fixed IQ finite projection inventory incomplete".into());
    }
    let shared_bound = shared_matrices.iter().map(|m| m.wire.len()).sum::<usize>();
    let shared_dense = shared_dense_cases(
        client,
        &shared_matrices,
        original_public_bound as usize + shared_bound,
    )?;
    let report = serde_json::json!({"scope":"fixed-public-independent-c-IQ-operators-only",
        "fixture":"iq-quant-probe","upstream":"llama.cpp b11232",
        "format_count":TYPES.len(),"oracle_cases":oracle,
        "decoded_c_values":oracles.iter().map(|o| o.expected.len()).sum::<usize>(),
        "projection_case_count":projections.len(),"projections":projections,
        "finite_case_count":TYPES.len()*4,"finite":finite,
        "single_subgroup_case_count":TYPES.len()*4,"single_subgroup":single,
        "shared_vec4_exact_c_oracles":shared_oracle,"shared_vec4_dense":shared_dense,
        "main_cap_bytes":MAIN_BUDGET,"runtime_io_cap_bytes":IO_BUDGET,
        "main_erased_bytes":zero.bytes,"main_erased_buffers":zero.buffers,
        "context_retired":false,"whole_model_or_performance_accepted":false});
    use std::io::Write;
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!(
        "public_iq_quant oracle_cases={} oracle_values={} projection_cases={} finite_cases={} subgroup_cases={}",
        oracles.len(),
        oracles.iter().map(|o| o.expected.len()).sum::<usize>(),
        PROJECTION_CASES,
        TYPES.len() * 4,
        TYPES.len() * 4
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_accepts_only_fixed_compiled_fixture_entry() {
        registration(3, true, false, false).unwrap();
        for argc in [0, 1, 2, 4, usize::MAX] {
            assert!(registration(argc, true, false, false).is_err());
        }
        for (receipt, model, timing) in [
            (false, false, false),
            (true, true, false),
            (true, false, true),
        ] {
            assert!(registration(3, receipt, model, timing).is_err());
        }
    }

    #[test]
    fn projection_inventory_uses_independent_c_bytes_and_ordinary_values() {
        let mut count = 0;
        for dtype in TYPES {
            for width in columns(dtype) {
                let packed = iq_fixtures::packed(dtype, MAX_ROWS, width, 0);
                let expected = iq_fixtures::oracle(dtype, MAX_ROWS, width, 0);
                let geometry = MatrixGeometry::new(dtype, MAX_ROWS, width, 1).unwrap();
                assert_eq!(packed.len(), geometry.total_bytes());
                assert_eq!(expected.len(), MAX_ROWS * width);
                let mut decoded = vec![0.0; expected.len()];
                TensorType::from_gguf_id(dtype)
                    .unwrap()
                    .decode(&packed, &mut decoded)
                    .unwrap();
                assert_eq!(
                    decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                let layout = fixture_layout(MAX_ROWS, width).unwrap();
                for members in NUMERIC_COUNTS {
                    let input = input_fixture(layout, width, members);
                    expectations(MAX_ROWS, width, &expected, &input, layout, members).unwrap();
                }
                count += 1;
            }
        }
        assert_eq!(count * (4 * 3 + 1), PROJECTION_CASES);
    }

    #[test]
    fn shared_dense_fixed_inventory_preserves_original_pages_and_precharge() {
        let mut matrices = 0;
        let mut values = 0;
        for dtype in SHARED_TYPES {
            let (block, bytes) = crate::quant_matvec::block_layout(dtype).unwrap();
            let (records, _) = iq_fixtures::records(dtype);
            values += records.len() / (bytes + block * 4) * block;
            for &width in shared_widths(dtype) {
                let packed = iq_fixtures::packed(dtype, MAX_ROWS, width, 0);
                let expected = iq_fixtures::oracle(dtype, MAX_ROWS, width, 0);
                assert_eq!(packed.len(), MAX_ROWS * width / block * bytes);
                let mut decoded = vec![0.; expected.len()];
                TensorType::from_gguf_id(dtype)
                    .unwrap()
                    .decode(&packed, &mut decoded)
                    .unwrap();
                assert_eq!(
                    decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                for prefix in [0usize, 256] {
                    let span = packed.len().div_ceil(4) * 4;
                    let owned = crate::main_admission::page_bytes((prefix + span) as u64).unwrap();
                    assert_eq!(owned, 32 * 1024);
                    assert!(prefix + span <= owned as usize);
                    matrices += 1;
                    for rows in [1, MAX_ROWS] {
                        let input = BufferLayout::new(4 + width + 12).unwrap();
                        let output = BufferLayout::new(7 + rows + 13).unwrap();
                        let fault = BufferLayout::new(1).unwrap();
                        assert_eq!(input.owned_bytes as u64, INPUT_LAYOUT_BYTES);
                        assert_eq!(output.owned_bytes as u64, OUTPUT_LAYOUT_BYTES);
                        assert_eq!(fault.owned_bytes as u64, OUTPUT_LAYOUT_BYTES);
                        let private =
                            input.owned_bytes + 2 * output.owned_bytes + 2 * fault.owned_bytes;
                        assert!(private <= MAIN_BUDGET);
                        assert!(2 * private + 8 * 512 <= IO_BUDGET as usize);
                        crate::next_shared::MatrixSpec::new(rows, width, dtype).unwrap();
                    }
                }
            }
        }
        assert_eq!(matrices, 20);
        assert_eq!(matrices * 2 * 3, SHARED_DENSE_CASES);
        assert_eq!(values * ORACLE_PREFIXES.len(), 18_112);
        assert_eq!(matrices * 32 * 1024, 655_360);
        assert!(shared_widths(IQ3_S).is_empty());
    }

    #[test]
    fn fixture_records_cover_all_pinned_c_values_and_half_cases() {
        let mut total = 0;
        for dtype in TYPES {
            let (width, bytes) = crate::quant_matvec::block_layout(dtype).unwrap();
            let (records, ordinary) = iq_fixtures::records(dtype);
            let stride = bytes + width * 4;
            assert!(records.len().is_multiple_of(stride));
            let blocks = records.len() / stride;
            assert_eq!(blocks, ordinary + 9);
            total += blocks * width;
        }
        assert_eq!(total * ORACLE_PREFIXES.len(), 56128);
    }

    #[test]
    fn exact_oracle_checker_rejects_numeric_gap_padding_and_extent_corruption() {
        for dtype in TYPES {
            let (columns, packed_bytes) = crate::quant_matvec::block_layout(dtype).unwrap();
            let (records, _) = iq_fixtures::records(dtype);
            let record_bytes = packed_bytes + columns * 4;
            let rows = records.len() / record_bytes;
            let expected: Vec<f32> = records
                .chunks_exact(record_bytes)
                .flat_map(|record| {
                    record[packed_bytes..]
                        .chunks_exact(4)
                        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                })
                .collect();
            let stride = columns + 7;
            let layout = BufferLayout::new(5 + rows * stride + 9).unwrap();
            let mut bytes = vec![0u8; layout.owned_bytes];
            for (index, word) in bytes
                .chunks_exact_mut(4)
                .enumerate()
                .take(layout.logical_words)
            {
                let relative = index.saturating_sub(5);
                let value = if index >= 5 && relative / stride < rows && relative % stride < columns
                {
                    expected[relative / stride * columns + relative % stride]
                } else {
                    OUTPUT_GAP
                };
                word.copy_from_slice(&value.to_le_bytes());
            }
            check_oracle(&bytes, &expected, rows, columns, layout).unwrap();
            for index in [
                0,
                5,
                5 + columns,
                layout.logical_words - 1,
                layout.owned_words - 1,
            ] {
                bytes[index * 4] ^= 1;
                assert!(check_oracle(&bytes, &expected, rows, columns, layout).is_err());
                bytes[index * 4] ^= 1;
            }
            for length in [0, 3, layout.owned_bytes - 1, layout.owned_bytes - 4] {
                assert!(check_oracle(&bytes[..length], &expected, rows, columns, layout).is_err());
            }
            assert!(
                check_oracle(
                    &bytes,
                    &expected[..expected.len() - 1],
                    rows,
                    columns,
                    layout
                )
                .is_err()
            );
        }
    }

    fn shader(dtype: u32) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope.register_type::<f32>(float);
        b.scope
            .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        let weight = b.input_array(Type::new(uint)).into();
        let output = b.output_array(Type::new(float)).into();
        let packed_first = b.scalar(uint).into();
        let output_first = b.scalar(uint).into();
        let output_stride = b.scalar(uint).into();
        original_iq_values::expand(
            &mut b.scope,
            weight,
            output,
            packed_first,
            output_first,
            output_stride,
            17,
            crate::quant_matvec::block_layout(dtype).unwrap().0,
            dtype,
        );
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("fixed_public_IQ_original_values"),
                ),
                &WgpuCompilationOptions::default(),
                cubecl::server::ExecutionMode::Checked,
                uint,
            )
            .unwrap()
            .to_string()
    }

    fn integer(
        module: &wgpu::naga::Module,
        handle: wgpu::naga::Handle<wgpu::naga::Expression>,
    ) -> i64 {
        use wgpu::naga::{Expression, Literal};
        match module.global_expressions[handle] {
            Expression::Literal(Literal::U32(value)) => i64::from(value),
            Expression::Literal(Literal::I32(value)) => i64::from(value),
            Expression::Literal(Literal::AbstractInt(value)) => value,
            Expression::As { expr, .. } => integer(module, expr),
            _ => panic!("IQ codebook initializer must be an immutable integer"),
        }
    }

    #[test]
    fn actual_shader_codebooks_are_exact_static_values_without_extra_bindings() {
        for dtype in TYPES {
            let source = shader(dtype);
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            let arrays = module
                .constants
                .iter()
                .filter(|(_, constant)| {
                    constant
                        .name
                        .as_deref()
                        .is_some_and(|name| name.starts_with("arrays_"))
                })
                .map(
                    |(_, constant)| match &module.global_expressions[constant.init] {
                        wgpu::naga::Expression::Compose { components, .. } => components
                            .iter()
                            .map(|&handle| integer(&module, handle))
                            .collect::<Vec<_>>(),
                        _ => panic!("IQ codebook must remain a constant array"),
                    },
                )
                .collect::<Vec<_>>();
            if dtype == IQ3_XXS {
                let expected = kuno_text::quant::IQ3_XXS_GRID
                    .into_iter()
                    .map(i64::from)
                    .collect::<Vec<_>>();
                assert_eq!(arrays, [expected]);
            } else if dtype == IQ2_S {
                let expected = kuno_text::quant::IQ2_S_GRID
                    .into_iter()
                    .flat_map(|word| [i64::from(word as u32), i64::from((word >> 32) as u32)])
                    .collect::<Vec<_>>();
                assert_eq!(arrays, [expected]);
            } else if dtype == IQ3_S {
                let expected = kuno_text::quant::IQ3_S_GRID
                    .into_iter()
                    .map(i64::from)
                    .collect::<Vec<_>>();
                assert_eq!(arrays, [expected]);
            } else if matches!(dtype, IQ4_NL | IQ4_XS) {
                // IQ4's16 signed entries use only four packed scalar literals.
                // No value-array constant exists for a runtime nibble to index;
                // original packed source/result/INFO bindings are unchanged.
                assert!(arrays.is_empty());
                assert!(module.constants.iter().all(|(_, constant)| {
                    !matches!(
                        module.types[constant.ty].inner,
                        wgpu::naga::TypeInner::Array { .. }
                    )
                }));
                assert!(!source.contains("arrays_"));
                // A value-producing nested CubeCL if previously emitted EMPTY
                // branches and retained only the final quartet. Host Rust
                // lookup equality did not catch that expansion defect. Require
                // all four literal words and their runtime threshold guards in
                // the actual generated decoder, not just the source function.
                for word in [0xbfad_9881_u32, 0xf6ea_ddcf, 0x2619_0d01, 0x7159_4535] {
                    assert!(source.contains(&format!("u32({word})")));
                }
                for minimum in [4, 8, 12] {
                    assert!(source.contains(&format!(">= u32({minimum})")));
                }
            } else {
                assert!(arrays.is_empty());
            }
            // Public original packed weights, private result, and original INFO.
            assert_eq!(
                module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| v.binding.is_some())
                    .count(),
                3
            );
            assert!(
                source.contains("scalars_u32[0]")
                    && source.contains("scalars_u32[1]")
                    && source.contains("scalars_u32[2]")
            );
            assert!(!source.contains("array<f32, 256>") && !source.contains("array<f32, 16>"));
        }
    }
}
