//! Fixed public IQ operator calibration, independent of any downloaded GGUF.
//! Packed bytes and exact expected values come from pinned original GGML C
//! decoders. The temporary C generator is never linked to this Rust engine.
//! No prompts, tokenizers, arbitrary payloads, model rows or timing mode.

use super::*;
use crate::quant_matvec::{IQ3_XXS, IQ4_NL, IQ4_XS, iq_fixtures};

const TYPES: [u32; 3] = [IQ3_XXS, IQ4_NL, IQ4_XS];
const ORACLE_PREFIXES: [usize; 2] = [0, 3];
const PROJECTION_CASES: usize = 3 * 3 * (4 * 3 + 1);

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
) -> Result<Vec<serde_json::Value>> {
    let mut reports = Vec::with_capacity(TYPES.len() * ORACLE_PREFIXES.len());
    for oracle in oracles {
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
            "independent_c_bits_exact":true,"gaps_and_full_padding_preserved":true,
            "main_erased":true,"public_cache_preserved":true,"io":receipt(io)}),
        );
    }
    if reports.len() != TYPES.len() * ORACLE_PREFIXES.len() {
        return Err("fixed IQ oracle inventory incomplete".into());
    }
    Ok(reports)
}

pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let path =
        std::env::var_os("KUNO_PUBLIC_RECEIPT").ok_or("fixed IQ receipt destination required")?;
    // Prewarm every immutable public weight buffer before configuring the
    // owning private upload/readback path. No model file is opened anywhere.
    let mut matrices = Vec::with_capacity(9);
    let mut finite_matrices = Vec::with_capacity(3);
    let mut oracles = Vec::with_capacity(6);
    for dtype in TYPES {
        for width in columns(dtype) {
            matrices.push(public_matrix(client, dtype, width, 3)?);
        }
        finite_matrices.push(public_matrix(client, dtype, columns(dtype)[2], 0)?);
        for prefix in ORACLE_PREFIXES {
            oracles.push(public_oracle(client, dtype, prefix)?);
        }
    }
    client.configure_transient_io(IO_BUDGET)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient_on_configured(client.clone(), MAIN_BUDGET, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let oracle = oracle_cases(client, &mut arena, &oracles, &retired)?;
        let projections = cases(client, &mut arena, &matrices, &retired)?;
        if projections.len() != PROJECTION_CASES {
            return Err("fixed IQ projection inventory incomplete".into());
        }
        Ok((oracle, projections))
    }));
    let closed = arena.close();
    let (oracle, projections) = match (attempt, closed, retired.load(Ordering::Acquire)) {
        (Ok(Ok(reports)), Ok(()), false) => reports,
        _ => return Err("fixed IQ operators failed; no success receipt".into()),
    };
    let zero = arena
        .owned_zero_audit_receipt()
        .ok_or("fixed IQ owned zero receipt absent")?;
    let public_bound = 3 * (MAX_PACKED_BYTES + 4).next_power_of_two();
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
    if finite["cases"].as_array().map(Vec::len) != Some(12)
        || single["cases"].as_array().map(Vec::len) != Some(12)
    {
        return Err("fixed IQ finite projection inventory incomplete".into());
    }
    let report = serde_json::json!({"scope":"fixed-public-independent-c-IQ-operators-only",
        "fixture":"iq-quant-probe","upstream":"llama.cpp b11232",
        "format_count":3,"oracle_cases":oracle,"decoded_c_values":16064,
        "projection_case_count":projections.len(),"projections":projections,
        "finite_case_count":12,"finite":finite,"single_subgroup_case_count":12,"single_subgroup":single,
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
        "public_iq_quant oracle_cases=6 oracle_values=16064 projection_cases=117 finite_cases=12 subgroup_cases=12"
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
        assert_eq!(total * ORACLE_PREFIXES.len(), 16064);
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
            } else {
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
