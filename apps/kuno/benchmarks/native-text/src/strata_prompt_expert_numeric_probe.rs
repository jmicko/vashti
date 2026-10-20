//! Fixed-public original quantized weights through the ACTUAL vector8 image
//! decoder. Real exclusive public owners provide physical source authority.
//! Exact FP16 bits and the complete private output owner are checked in RAM.
use super::*;
use crate::{gr_batch_probe as probe, private_arena::PrivateArena, resident_util as u};
use zeroize::Zeroizing;
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn fail(label: &'static str) -> Box<dyn std::error::Error> {
    eprintln!("public_prompt_dequant_fault {label}");
    label.into()
}

fn public(client: &ComputeClient<WgpuRuntime>, bytes: Vec<u8>) -> Result<Handle> {
    let retired = std::sync::atomic::AtomicBool::new(false);
    crate::expert_banks::exact_persistent::persistent_scope(
        |mode| {
            // Same authenticated PUBLIC exact-exclusive mode as model weights.
            unsafe { client.allocation_mode(mode) };
            Ok(())
        },
        &retired,
        || Ok(client.create_public_owned(bytes)),
    )
}

fn original(file: &kuno_text::GgufFile, name: &str, bytes: usize) -> ProbeResult<Vec<u8>> {
    let mut out = vec![0u8; bytes];
    file.tensor(name)
        .ok_or_else(|| fail("original_tensor_missing"))?
        .read_range(0, &mut out)?;
    Ok(out)
}

/// Inspect ALL actual owned bytes, without requesting an80MiB transient read.
/// Each32MiB window is wiped/sanitized before the next owning span is inspected.
fn image_difference(
    arena: &PrivateArena,
    id: crate::private_arena::BufferId,
    first: usize,
    expected: &[u32],
    may_change: std::ops::Range<usize>,
) -> ProbeResult<usize> {
    let root = arena.handle(id)?.clone();
    let owned_words = arena.layout(id)?.owned_bytes / 4;
    let window = crate::private_arena::ZERO_AUDIT_WINDOW / 4;
    let mut mismatches = 0;
    for start in (0..owned_words).step_by(window) {
        let count = window.min(owned_words - start);
        let view = root
            .clone()
            .offset_start((start * 4) as u64)
            .offset_end(((owned_words - start - count) * 4) as u64);
        let actual = u::read_handle_prefix_u32(arena, view, count)?;
        for (local, &word) in actual.iter().enumerate() {
            let at = start + local;
            if !expected.is_empty() && (first..first + expected.len()).contains(&at) {
                mismatches += usize::from(word != expected[at - first]);
            } else if !may_change.contains(&at) && word != 0 {
                return Err(fail("complete_output_owner_tail_changed"));
            }
        }
        drop(actual);
        let receipt = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
        let full = receipt
            .upload_bytes
            .checked_add(receipt.uniform_bytes)
            .and_then(|n| n.checked_add(receipt.readback_bytes))
            .ok_or_else(|| fail("output_readback_audit_count_overflow"))?;
        if full == 0 || receipt.inspected_bytes != full || receipt.inspected_buffers == 0 {
            return Err(fail("output_readback_audit_incomplete"));
        }
    }
    Ok(mismatches)
}

pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> ProbeResult<Vec<serde_json::Value>> {
    let mut table = Vec::<u32>::new();
    for dtype in [q::IQ3_XXS, q::IQ3_S, q::IQ2_S] {
        table.extend(q::iq_moe_integer::original_grid_words(dtype));
    }
    if table.len() != 2816 {
        return Err(fail("original_iq_table_geometry"));
    }
    table.resize(8192, 0);
    let table = public(
        client,
        table.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    let mut reports = Vec::new();
    for dtype in [
        q::IQ3_XXS,
        q::IQ3_S,
        q::IQ2_S,
        q::IQ4_XS,
        q::IQ4_NL,
        q::Q2_0,
    ] {
        let down = matches!(dtype, q::IQ4_NL | q::Q2_0);
        let (rows, columns, suffix) = if down {
            (2560usize, 640usize, "ffn_down_exps.weight")
        } else {
            (640usize, 2560usize, "ffn_gate_exps.weight")
        };
        let (name, tensor) = file
            .tensors()
            .iter()
            .find(|(name, tensor)| name.ends_with(suffix) && tensor.dtype().gguf_id() == dtype)
            .ok_or_else(|| fail("original_quant_format_missing"))?;
        if tensor.dimensions() != [columns as u64, rows as u64, 512] {
            return Err(fail("original_quant_geometry"));
        }
        let row_bytes = tensor.dtype().row_bytes(columns as u64)? as usize;
        let bytes = rows * row_bytes;
        let gate_bytes = original(file, name, bytes)?;
        let up_bytes = if down {
            gate_bytes.clone()
        } else {
            let up_name = name.replace(suffix, "ffn_up_exps.weight");
            let up = file
                .tensor(&up_name)
                .ok_or_else(|| fail("original_up_missing"))?;
            if up.info().dimensions() != tensor.dimensions() || up.info().dtype() != tensor.dtype()
            {
                return Err(fail("original_up_geometry"));
            }
            original(file, &up_name, bytes)?
        };
        let gate = public(client, gate_bytes.clone())?;
        let up = public(client, up_bytes.clone())?;
        let physical = client
            .native_strata_source_addresses(vec![gate.clone(), up.clone()])
            .map_err(u::fault)?;
        if physical.iter().any(|a| a[1] != bytes as u64) {
            return Err(fail("actual_physical_source_extent"));
        }
        let mut gate_values = Zeroizing::new(vec![0f32; rows * columns]);
        let mut up_values = Zeroizing::new(vec![0f32; rows * columns]);
        tensor.dtype().decode(&gate_bytes, &mut gate_values)?;
        tensor.dtype().decode(&up_bytes, &mut up_values)?;
        if gate_values
            .iter()
            .chain(up_values.iter())
            .any(|v| !v.is_finite())
        {
            return Err(fail("original_quant_nonfinite"));
        }
        let output_rows = if down { rows } else { rows * 2 };
        let output_words = output_rows * columns / 2;
        let mut reference = Zeroizing::new(vec![0u32; output_words]);
        for row in 0..output_rows {
            let source_row = if down { row } else { row / 2 };
            let decoded = if down || row % 2 == 0 {
                &gate_values
            } else {
                &up_values
            };
            for col in (0..columns).step_by(2) {
                let index = source_row * columns + col;
                reference[(row * columns + col) / 2] =
                    u32::from(half::f16::from_f32(decoded[index]).to_bits())
                        | (u32::from(half::f16::from_f32(decoded[index + 1]).to_bits()) << 16);
            }
        }
        for (host, control) in [
            (false, "valid"),
            (true, "valid"),
            (false, "bad_extent"),
            (true, "bad_alignment"),
            (true, "nonfinite"),
        ] {
            // Finite-control corruption is a compiled-public invalid input,
            // separate from the unchanged original-weight acceptance oracle.
            let gu_bytes = if down { 256 } else { bytes };
            let down_bytes = if down { bytes } else { 256 };
            let mut raw = Zeroizing::new(vec![0u8; 2 * gu_bytes + down_bytes]);
            if down {
                raw[2 * gu_bytes..].copy_from_slice(&gate_bytes);
            } else {
                raw[..gu_bytes].copy_from_slice(&gate_bytes);
                raw[gu_bytes..2 * gu_bytes].copy_from_slice(&up_bytes);
            }
            if control == "nonfinite" {
                let offset = if down { 2 * gu_bytes } else { 0 };
                raw[offset..offset + 2].copy_from_slice(&half::f16::NAN.to_le_bytes());
            }
            let raw_words = Zeroizing::new(
                raw.chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            );
            let raw_id = arena.upload_u32(&raw_words)?;
            let cell = 11usize;
            let mut addresses = vec![0u32; 8192];
            for role in if down {
                &[2usize][..]
            } else {
                &[0usize, 1][..]
            } {
                let address = physical[usize::from(*role == 1)];
                let at = (*role * 512 + cell) * 4;
                addresses[at] = address[0] as u32;
                addresses[at + 1] = (address[0] >> 32) as u32;
                addresses[at + 2] = (bytes / 4) as u32 + u32::from(control == "bad_extent");
                addresses[at + 3] = u32::from(host);
            }
            let addresses = public(
                client,
                addresses.into_iter().flat_map(u32::to_le_bytes).collect(),
            )?;
            let mut map = Zeroizing::new(vec![0u32; groups::MAP_WORDS]);
            map[..4].copy_from_slice(&[1, 0, 1, 1]);
            map[4] = cell as u32;
            map[516] = 1;
            let map_id = arena.upload_u32(&map)?;
            let output_id = arena.allocate_zeroed((storage::WORKING_BYTES / 4) as usize)?;
            let fault_id = arena.allocate_zeroed(1)?;
            let first = 16usize;
            let first_param = first as u32 + u32::from(control == "bad_alignment");
            let words = [
                1,
                columns as u32,
                output_rows as u32,
                row_bytes as u32,
                0,
                if down { 2 } else { 0 },
                0,
                (gu_bytes / 4) as u32,
                (2 * gu_bytes / 4) as u32,
                first_param,
                first_param,
                raw_words.len() as u32,
                0,
                0,
                0,
                0,
            ];
            let args = || -> Result<KernelArguments> {
                let mut args = KernelArguments::new().with_buffer(gate.clone().binding());
                if !down {
                    args = args.with_buffer(up.clone().binding());
                }
                for id in [map_id, raw_id, output_id, fault_id] {
                    args = args.with_buffer(u::handle(arena, id)?.binding());
                }
                Ok(args
                    .with_buffer(table.clone().binding())
                    .with_buffer(addresses.clone().binding())
                    .with_read_dependencies(vec![gate.clone().binding(), up.clone().binding()])
                    .with_info(MetadataBindingInfo::custom(
                        words
                            .chunks_exact(2)
                            .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
                            .collect(),
                    )))
            };
            // SAFETY: authentic physical original public source roots, actual
            // table and distinct complete private owners. Invalid controls
            // fail the same production guards before unsafe source access.
            unsafe {
                client.launch_unchecked(
                    Box::new(Op::Dequant { dtype, down }),
                    CubeCount::Static((output_rows * columns / 8).div_ceil(256) as u32, 1, 1),
                    args()?,
                );
            }
            cubecl::future::block_on(client.sync()).map_err(u::fault)?;
            let mut warm_dispatch_ns = Vec::new();
            if control == "valid" {
                // First completed original dispatch above warms compilation.
                // Fixed-public fenced wall times are diagnostic, not pure GPU
                // timestamps or a private request profiling switch.
                for _ in 0..3 {
                    let started = std::time::Instant::now();
                    unsafe {
                        client.launch_unchecked(
                            Box::new(Op::Dequant { dtype, down }),
                            CubeCount::Static(
                                (output_rows * columns / 8).div_ceil(256) as u32,
                                1,
                                1,
                            ),
                            args()?,
                        );
                    }
                    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
                    warm_dispatch_ns
                        .push(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                }
            }
            let fault = probe::read_native_public(arena, fault_id)?;
            let expected_fault = if control == "valid" { 0 } else { 128 };
            if fault[0] != expected_fault || fault[1..].iter().any(|&v| v != 0) {
                return Err(fail("terminal_fault_control_mismatch"));
            }
            let expected: &[u32] = if control == "valid" { &reference } else { &[] };
            let changed = if matches!(control, "valid" | "nonfinite") {
                first..first + output_words
            } else {
                0..0
            };
            let mismatches = image_difference(arena, output_id, first, expected, changed)?;
            println!(
                "public_prompt_dequant_numeric dtype={dtype} down={down} host={host} control={control} bit_mismatches={mismatches} warm_fenced_wall_ns={warm_dispatch_ns:?}"
            );
            reports.push(
                serde_json::json!({"family":"prompt-vector8-dequant", "dtype":dtype,
                "down":down,"host_staged":host,"control":control,"bit_mismatches":mismatches,
                "warm_fenced_wall_ns":warm_dispatch_ns,"image_bytes":output_words*4,
                "actual_native_kernels":true,"original_weights":control!="nonfinite",
                "original_weight_acceptance":control=="valid",
                "full_owner_and_tail_checked":true,"authentic_physical_sources":true}),
            );
            arena.reset()?;
            if mismatches != 0 {
                return Err(fail("fp16_original_oracle_bit_mismatch"));
            }
        }
    }
    Ok(reports)
}
