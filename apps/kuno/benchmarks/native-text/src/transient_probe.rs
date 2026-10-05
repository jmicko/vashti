//! Fixed public-pattern device acceptance. Counts only; no real request input.
use crate::private_arena::PrivateArena;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::transient::TransientIoReceipt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::Zeroizing;

#[cube(launch)]
fn runtime_pattern(input: &Array<u32>, output: &mut Array<u32>, salt: u32) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = input[i] ^ salt;
    }
}

fn audited(
    client: &ComputeClient<WgpuRuntime>,
) -> Result<TransientIoReceipt, Box<dyn std::error::Error>> {
    let r = cubecl::future::block_on(client.sanitize_transient_io_audited())?;
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("public audit count overflow")?;
    if full == 0 || r.inspected_bytes != full || r.inspected_buffers == 0 {
        return Err("public audit full-span counts did not match".into());
    }
    println!(
        "public_transient_io epoch={} upload_bytes={} uniform_bytes={} readback_bytes={} inspected_bytes={} inspected_buffers={}",
        r.epoch,
        r.upload_bytes,
        r.uniform_bytes,
        r.readback_bytes,
        r.inspected_bytes,
        r.inspected_buffers
    );
    Ok(r)
}

pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<(), Box<dyn std::error::Error>> {
    // Original immutable public allocation is deliberately kept through cleanup.
    let public = client.create_from_slice(&[0x53; 128]);
    let retired = Arc::new(AtomicBool::new(false));
    let mut summaries = Vec::new();
    println!("public_transient_stage=before_configure");
    let mut arena = PrivateArena::new_transient(
        client.clone(),
        1024 * 1024,
        8 * 1024 * 1024,
        retired.clone(),
    )?;
    println!("public_transient_stage=configured");
    let pattern = Zeroizing::new((0..8193).map(|i| 0xa596_317b_u32 ^ i).collect::<Vec<_>>());
    let x = arena.upload_u32(&pattern)?;
    println!("public_transient_stage=uploaded");
    let y = arena.allocate_zeroed(pattern.len())?;
    println!("public_transient_stage=zeroed");
    // Initialization already sanitized its stage. Leave another owned private
    // stage live so the subsequent full-span device audit covers uploads too.
    let mut audit_upload = vec![0_u8; arena.layout(x)?.owned_bytes];
    for (b, v) in audit_upload.chunks_exact_mut(4).zip(pattern.iter()) {
        b.copy_from_slice(&v.to_le_bytes());
    }
    client.write_private(arena.handle(x)?.clone(), audit_upload)?;
    let full = arena.layout(y)?.owned_bytes;
    let retained = arena.handle(y)?.clone();
    let pending = client.read_async(vec![arena.handle(x)?.clone()]);
    if cubecl::future::block_on(client.sanitize_transient_io()).is_ok() {
        return Err("public audit accepted a live pending readback".into());
    }
    drop(pending);
    summaries.push(audited(client)?);
    for salt in [0x52a6_b3e1_u32, 0xf247_819b] {
        // SAFETY: same serialized client; both retained full spans contain at
        // least pattern.len() words. Private salt is a runtime scalar.
        runtime_pattern::launch::<WgpuRuntime>(
            client,
            CubeCount::Static((pattern.len() as u32).div_ceil(64), 1, 1),
            CubeDim::new_1d(64),
            unsafe { ArrayArg::from_raw_parts(arena.handle(x)?.clone(), pattern.len()) },
            unsafe { ArrayArg::from_raw_parts(arena.handle(y)?.clone(), pattern.len()) },
            salt,
        );
        let bytes = client.read_one(arena.handle(y)?.clone())?;
        if bytes.len() != full
            || bytes[..pattern.len() * 4]
                .chunks_exact(4)
                .zip(pattern.iter())
                .any(|(b, &v)| u32::from_le_bytes(b.try_into().unwrap()) != v ^ salt)
            || bytes[pattern.len() * 4..].iter().any(|&b| b != 0)
        {
            return Err("public private-I/O runtime kernel parity failed".into());
        }
        if cubecl::future::block_on(client.sanitize_transient_io()).is_ok() {
            return Err("public audit accepted a retained readback".into());
        }
        drop(bytes);
        let receipt = audited(client)?;
        summaries.push(receipt);
        if receipt.upload_bytes == 0 || receipt.uniform_bytes == 0 || receipt.readback_bytes == 0 {
            return Err("public audit did not cover each I/O kind".into());
        }
        let original = client.read_one(public.clone())?;
        if original.len() != 128 || original.iter().any(|&b| b != 0x53) {
            return Err("public cache changed across private erasure".into());
        }
        drop(original);
        summaries.push(audited(client)?);
    }
    arena.close()?;
    let erased = client.read_one(retained)?;
    if erased.len() != full
        || erased.iter().any(|&b| b != 0)
        || retired.load(Ordering::Acquire)
        || arena.handle(y).is_ok()
    {
        return Err("public owned main span was not erased".into());
    }
    drop(erased);
    summaries.push(audited(client)?);
    println!(
        "public_transient_io_acceptance main_span_bytes={full} pending_drop=true retained_lease_rejected=true public_cache_preserved=true context_retired=false"
    );
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let epochs: Vec<_> = summaries.iter().map(|r| serde_json::json!({"epoch":r.epoch,"upload_bytes":r.upload_bytes,"uniform_bytes":r.uniform_bytes,"readback_bytes":r.readback_bytes,"inspected_bytes":r.inspected_bytes,"inspected_buffers":r.inspected_buffers})).collect();
        let receipt = serde_json::json!({"scope":"fixed-public-synthetic-operator-test-only","fixture":"transient-io-probe","main_full_span_bytes":full,"pending_future_drop":true,"retained_readback_rejected":true,"public_cache_preserved":true,"context_retired":false,"epochs":epochs,"proof_scope":"owned API-visible spans including audit stages; not opaque physical driver padding"});
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
    }
    Ok(())
}
