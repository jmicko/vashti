//! Actual original whole-head PHYSICAL ADDRESS consumer. Original bounded
//! public pages, fixed synthetic input, RAM-only independent canonical oracle.
use super::*;
use crate::{resident_util as u, strata_shared_weights as shared};

fn original_page(client: &ComputeClient<WgpuRuntime>, bytes: Vec<u8>) -> Result<Handle> {
    let retired = AtomicBool::new(false);
    Ok(crate::expert_banks::exact_persistent::persistent_scope(
        |mode| {
            // Same serialized PUBLIC allocation mode as the owning model.
            unsafe { client.allocation_mode(mode) };
            Ok(())
        },
        &retired,
        || Ok(client.create_public_owned(bytes)),
    )?)
}

pub(super) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &GgufFile,
) -> Result<Vec<serde_json::Value>> {
    let kind = shared::Tensor::Head;
    let tensor = file
        .tensor(kind.name())
        .ok_or("public physical head missing")?;
    if tensor.info().dimensions() != [shared::COLUMNS as u64, shared::ROWS as u64]
        || tensor.info().dtype().gguf_id() != kind.dtype()
    {
        return Err("public physical head original geometry".into());
    }
    let tiles = shared::tiles(kind);
    if tiles.iter().map(|t| t.owned).sum::<usize>() > 1024 * 1024 * 1024 {
        return Err("public physical head byte cap".into());
    }
    let mut pages = Vec::new();
    for &tile in &tiles {
        let mut bytes = vec![0u8; tile.owned]; // Original PUBLIC model bytes only.
        tensor.read_range(
            (tile.first * kind.row_bytes()) as u64,
            &mut bytes[..tile.logical],
        )?;
        shared::validate_payload(kind, &bytes, tile)?;
        pages.push(crate::strata_head::Page {
            handle: original_page(client, bytes)?,
            first: tile.first,
            rows: tile.rows,
            logical: tile.logical,
            owned: tile.owned,
        });
    }
    let table = crate::strata_head::metadata(client, shared::ROWS, &pages)?;
    let head =
        crate::strata_head::Head::new(client, shared::ROWS, pages, original_page(client, table)?)?;
    let input = fixture(shared::COLUMNS, 11);
    let input_id = arena.upload_f32(&input)?;
    let stride = shared::COLUMNS / 2 + shared::COLUMNS / 16;
    let planes = arena.allocate_zeroed(stride)?;
    let output = arena.allocate_zeroed(shared::ROWS)?;
    let fault = arena.allocate_zeroed(1)?;
    crate::strata_dense::prepare(
        client,
        shared::COLUMNS,
        u::handle(arena, input_id)?,
        u::handle(arena, planes)?,
        u::handle(arena, fault)?,
        &BatchSpans::new(1, 0, 0, shared::COLUMNS, 0, shared::ROWS),
    )?;
    head.project(
        client,
        u::handle(arena, planes)?,
        u::handle(arena, output)?,
        u::handle(arena, fault)?,
        0,
        shared::ROWS,
        1,
    )?;
    let actual = native::read_id(arena, output)?;
    if native::read_id(arena, fault)?.iter().any(|&v| v != 0) {
        return Err("public physical head actual fault".into());
    }
    let mut quantized = Zeroizing::new(vec![0.; shared::COLUMNS]);
    for (block, row) in input[..shared::COLUMNS].chunks_exact(32).enumerate() {
        let scale = row.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.;
        let d = half::f16::from_f32(scale).to_f32();
        for (i, &v) in row.iter().enumerate() {
            quantized[block * 32 + i] = if scale == 0. {
                0.
            } else {
                d * (v / scale).round().clamp(-127., 127.)
            };
        }
    }
    let mut mismatches = 0usize;
    let mut max_abs = 0f32;
    // Stream one page at a time: never retain a full dequantized model head.
    for tile in tiles {
        let mut packed = Zeroizing::new(vec![0u8; tile.logical]);
        tensor.read_range((tile.first * kind.row_bytes()) as u64, &mut packed)?;
        let mut decoded = Zeroizing::new(vec![0.; tile.rows * shared::COLUMNS]);
        tensor.info().dtype().decode(&packed, &mut decoded)?;
        for (row, w) in decoded.chunks_exact(shared::COLUMNS).enumerate() {
            let reference = w
                .iter()
                .zip(quantized.iter())
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>() as f32;
            let value = f32::from_bits(actual[tile.first + row]);
            let error = (value - reference).abs();
            mismatches += usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
            max_abs = max_abs.max(error);
        }
    }
    if actual[shared::ROWS..].iter().any(|&v| v != 0) {
        return Err("public physical head full owner padding".into());
    }
    println!(
        "public_native_physical_numeric family=head rows={} mismatches={mismatches} max_abs={max_abs}",
        shared::ROWS
    );
    arena.reset()?;
    if mismatches != 0 {
        return Err("public original physical head numerical parity failed".into());
    }
    Ok(vec![
        serde_json::json!({"family":"physical-whole-head","rows":shared::ROWS,
        "mismatches":mismatches,"max_abs":max_abs,"actual_native_kernels":true,"full_owner_and_tail_checked":true}),
    ])
}
