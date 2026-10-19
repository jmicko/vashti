//! Fixed-public, actual physical expert kernels: old SLM input versus direct
//! planes. Original expert bytes and synthetic Q8 inputs only; no token trace,
//! response, tensor values, or timing is persisted. Both full owners are read
//! and erased by the parent numerical probe, including on a failed comparison.
use super::*;
use crate::{private_arena::PrivateArena, resident_util as u};
use zeroize::Zeroizing;
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn public(client: &ComputeClient<WgpuRuntime>, bytes: Vec<u8>) -> Result<Handle> {
    let retired = std::sync::atomic::AtomicBool::new(false);
    crate::expert_banks::exact_persistent::persistent_scope(
        |mode| {
            // Same exact immutable PUBLIC allocation mode as original weights.
            unsafe { client.allocation_mode(mode) };
            Ok(())
        },
        &retired,
        || Ok(client.create_public_owned(bytes)),
    )
}

pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> ProbeResult<Vec<serde_json::Value>> {
    let mut table = Vec::<u32>::new();
    for dtype in [18, 21, 22] {
        table.extend(crate::quant_matvec::iq_moe_integer::original_grid_words(
            dtype,
        ));
    }
    if table.len() != 2816 {
        return Err("public expert original IQ tables".into());
    }
    table.resize(8192, 0);
    let table = public(
        client,
        table.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    let mut reports = Vec::new();
    for role in [1u8, 2, 3, 4] {
        let down = role == 3;
        for dtype in if down {
            vec![20u32, 42]
        } else {
            vec![18u32, 21, 22, 23]
        } {
            let (rows, columns) = if down { (2560, 640) } else { (640, 2560) };
            let suffix = match role {
                1 | 4 => "ffn_gate_exps.weight",
                2 => "ffn_up_exps.weight",
                _ => "ffn_down_exps.weight",
            };
            let (name, info) = file
                .tensors()
                .iter()
                .find(|(name, info)| name.ends_with(suffix) && info.dtype().gguf_id() == dtype)
                .ok_or("public original expert format missing")?;
            if info.dimensions() != [columns as u64, rows as u64, 512] {
                return Err("public expert original geometry".into());
            }
            let row_bytes = info.dtype().row_bytes(columns as u64)? as usize;
            let logical = rows * row_bytes;
            let mut packed = vec![0u8; logical.next_multiple_of(16)];
            file.tensor(name)
                .ok_or("public original expert missing")?
                .read_range(0, &mut packed[..logical])?;
            let weights = public(client, packed)?;
            let up_weights = if role == 4 {
                let up_name = name.replace("ffn_gate_exps.weight", "ffn_up_exps.weight");
                let tensor = file
                    .tensor(&up_name)
                    .ok_or("public joint GU original UP missing")?;
                if tensor.info().dimensions() != info.dimensions()
                    || tensor.info().dtype() != info.dtype()
                {
                    return Err("public joint GU original UP geometry".into());
                }
                let mut packed = vec![0u8; logical.next_multiple_of(16)];
                tensor.read_range(0, &mut packed[..logical])?;
                public(client, packed)?
            } else {
                weights.clone()
            };
            let address = client
                .native_strata_source_addresses(vec![weights.clone(), up_weights.clone()])
                .map_err(u::fault)?;
            let mut addresses = vec![0u32; 8192];
            // Both GU descriptors are authentic owning references to the same
            // original selected projection; only the tested role consumes it.
            for source_role in 0..3 {
                let at = source_role * 512 * 4;
                let source = usize::from(source_role == 1);
                addresses[at] = address[source][0] as u32;
                addresses[at + 1] = (address[source][0] >> 32) as u32;
                addresses[at + 2] = (logical / 4) as u32;
            }
            let addresses = public(
                client,
                addresses.into_iter().flat_map(u32::to_le_bytes).collect(),
            )?;
            for count in [1usize, 3, 7, 8] {
                let mut map = Zeroizing::new(vec![0u32; 804]);
                map[0] = count as u32;
                map[2] = count as u32;
                map[3] = 1;
                map[84] = count as u32;
                let mut planes = Zeroizing::new(vec![0u32; if down { 32768 } else { 16384 }]);
                let mut status = Zeroizing::new(vec![0u32; 160]);
                for token in 0..count {
                    let route = token * 10 + token % 10;
                    map[164 + token] = route as u32;
                    planes[if down { route } else { token }] = 1;
                    if down {
                        status[route] = 1;
                    }
                    let (base, words, blocks) = if down {
                        (128 + route * 360, 160, 20)
                    } else {
                        (64 + token * 1440, 640, 80)
                    };
                    for q in 0..words {
                        let mut value = 0u32;
                        for byte in 0..4 {
                            let v = ((q * 13 + token * 17 + byte * 23) % 251) as i32 - 125;
                            value |= ((v as u32) & 255) << (byte * 8);
                        }
                        planes[base + q] = value;
                        // Different second plane catches accidental residual reads.
                        planes[base + words + q] = !value;
                    }
                    for block in 0..blocks {
                        planes[base + words * 2 + block] =
                            ((1 + (block + token) % 7) as f32 / 1024.).to_bits();
                    }
                }
                let map = arena.upload_u32(&map)?;
                let planes = arena.upload_u32(&planes)?;
                let mut outputs = Vec::new();
                let mut statuses = Vec::new();
                for direct_planes in [false, true] {
                    let output = arena.allocate_zeroed(80 * 2560)?;
                    let stamps = arena.upload_u32(&status)?;
                    let roles: &[u8] = if role == 4 && !direct_planes {
                        &[1, 2]
                    } else {
                        std::slice::from_ref(&role)
                    };
                    for &actual_role in roles {
                        let mut args =
                            KernelArguments::new().with_buffer(weights.clone().binding());
                        if !down {
                            args = args.with_buffer(up_weights.clone().binding());
                        }
                        for id in [map, output, stamps, planes] {
                            args = args.with_buffer(u::handle(arena, id)?.binding());
                        }
                        if !down {
                            args = args.with_buffer(table.clone().binding());
                        }
                        args = args
                            .with_buffer(addresses.clone().binding())
                            .with_read_dependencies(vec![
                                weights.clone().binding(),
                                up_weights.clone().binding(),
                            ])
                            .with_info(params([
                                0,
                                0,
                                512,
                                0,
                                2560,
                                0,
                                if down { 80 } else { 0 },
                                count,
                            ]));
                        // Actual selected production source and physical reads.
                        // Joint GU compares against BOTH separate direct roles.
                        unsafe {
                            client.launch_unchecked(
                                Box::new(Op::Role {
                                    role: actual_role,
                                    dtype,
                                    row_bytes,
                                    direct_planes: role == 4 || direct_planes,
                                }),
                                CubeCount::Static(
                                    (rows * if actual_role == 4 { 2 } else { 1 } / 32) as u32,
                                    80,
                                    1,
                                ),
                                args,
                            );
                        }
                    }
                    outputs.push(crate::gr_batch_probe::read_native_public(arena, output)?);
                    statuses.push(crate::gr_batch_probe::read_native_public(arena, stamps)?);
                }
                let mut mismatches = 0usize;
                let mut max_abs = 0f32;
                for (&a, &b) in outputs[0].iter().zip(outputs[1].iter()) {
                    mismatches += usize::from(a != b);
                    max_abs = max_abs.max((f32::from_bits(a) - f32::from_bits(b)).abs());
                    if !f32::from_bits(a).is_finite() || !f32::from_bits(b).is_finite() {
                        return Err("public expert input nonfinite result".into());
                    }
                }
                let status_mismatches = statuses[0]
                    .iter()
                    .zip(statuses[1].iter())
                    .filter(|(a, b)| a != b)
                    .count();
                let offset = if role == 2 { 640 } else { 0 };
                for token in 0..count {
                    let route = token * 10 + token % 10;
                    let stamp = if role == 1 { route } else { 80 + route };
                    if statuses[1][stamp] != 1
                        || (role == 4 && statuses[1][route] != 1)
                        || outputs[1][route * 2560 + offset..]
                            [..if role == 4 { 1280 } else { rows }]
                            .iter()
                            .all(|&v| v == 0)
                    {
                        return Err("public expert input role did not execute".into());
                    }
                }
                println!(
                    "public_expert_input_numeric role={role} dtype={dtype} count={count} bit_mismatches={mismatches} status_mismatches={status_mismatches} max_abs={max_abs}"
                );
                reports.push(
                    serde_json::json!({"family":if role==4 {"physical-expert-joint-gu-parity"} else {"physical-expert-input-staging-parity"},
                    "role":role,"dtype":dtype,"count":count,"bit_mismatches":mismatches,
                    "status_mismatches":status_mismatches,"max_abs":max_abs,
                    "actual_native_kernels":true,"full_owner_and_tail_compared":true}),
                );
                arena.reset()?;
                if mismatches != 0 || status_mismatches != 0 {
                    return Err("public direct expert input differs from original staging".into());
                }
            }
        }
    }
    Ok(reports)
}
