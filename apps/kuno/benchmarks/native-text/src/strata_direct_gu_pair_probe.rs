//! Genuine fixed-public two-stage comparison, not a fabricated production batch.
//! Production direct selection still requires the scoped closed collector.
use super::shared::{ProbeResult, scan};
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
    strata_prompt::groups as abi,
};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use zeroize::Zeroizing;

fn snapshot(arena: &PrivateArena, id: BufferId) -> ProbeResult<Zeroizing<Vec<u32>>> {
    let mut out = Zeroizing::new(vec![0; arena.layout(id)?.owned_bytes / 4]);
    scan(arena, id, |i, w| {
        out[i] = w;
        Ok(())
    })?;
    Ok(out)
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    count: usize,
    control: usize,
) -> ProbeResult<serde_json::Value> {
    let mut raw = Zeroizing::new(vec![0; abi::LOGICAL + count * 10]);
    raw[..3].copy_from_slice(&[count as u32, 0, count as u32]);
    for token in 0..count {
        for rank in 0..10 {
            let route = token * 10 + rank;
            raw[abi::LOGICAL + route] = if rank == 0 { 400 } else { rank as u32 };
            raw[1540 + route] = raw[abi::LOGICAL + route];
        }
    }
    let mut map = abi::group(&raw, count)?;
    let group = abi::validate(&map, count)? - 1;
    if map[516 + group] as usize != count || map[4 + group] != 400 {
        return Err("direct GU original group geometry".into());
    }
    let first = map[1028 + group] as usize;
    if control >= 7 {
        let member = if control == 7 { 0 } else { count - 1 };
        map[1540 + first + member] = (count * 10) as u32;
    }
    // Exact raw FP16 bits: deliberate poison never passes a finite-only helper.
    let input = Zeroizing::new(vec![0x3c003c00; count * 1280]);
    let mut weight = Zeroizing::new(vec![0x14001400; (128 + 1280 * 2560) / 2]);
    weight[..64].fill(0x35553555);
    if (1..=6).contains(&control) {
        let half = match (control - 1) % 3 {
            0 => 0x7e00,
            1 => 0x7c00,
            _ => 0xfc00,
        };
        let row = usize::from(control >= 4);
        let at = 128 + row * 2560;
        weight[at / 2] = (weight[at / 2] & 0xffff0000) | half;
    }
    let gu = Zeroizing::new(vec![0x3f123456; abi::GU_BYTES as usize / 4]);
    let status = Zeroizing::new(vec![0xa513579b; abi::STATUS_WORDS]);
    let mut reference: Option<Vec<Zeroizing<Vec<u32>>>> = None;
    let mut checked_words = 0;
    for direct in [false, true] {
        let ii = arena.upload_u32(&input)?;
        let wi = arena.upload_u32(&weight)?;
        let gi = arena.upload_u32(&gu)?;
        let mi = arena.upload_u32(&map)?;
        let si = arena.upload_u32(&status)?;
        let fi = arena.allocate_zeroed(1)?;
        let h = [ii, wi, gi, mi, si, fi]
            .map(|id| u::handle(arena, id))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let h = h.try_into().map_err(|_| "direct GU six original owners")?;
        if !crate::strata_matrix::native_expert_wide::try_expert_mode(
            client, h, count, 2560, 1280, 128, group, false, direct,
        )? {
            return Err("direct GU actual wide launch not selected".into());
        }
        crate::strata_prompt::experts::public_gu_activation(
            arena,
            [mi, gi, ii, si, fi],
            count,
            count,
            group,
            direct,
        )?;
        let ids = [gi, ii, si, fi, wi, mi];
        let mut result = Vec::new();
        for id in ids {
            result.push(snapshot(arena, id)?);
        }
        if let Some(expected) = &reference {
            for (index, (a, b)) in result.iter().zip(expected).enumerate() {
                if a != b {
                    return Err(format!("direct GU pair full-owner mismatch role={index} count={count} control={control}").into());
                }
                checked_words += a.len();
            }
        } else {
            let fault = result[3][0];
            if fault != if control == 0 { 0 } else { 128 } {
                return Err("direct GU original fault policy".into());
            }
            reference = Some(result);
        }
        arena.reset()?;
    }
    Ok(
        serde_json::json!({"family":"direct-gu-validating-activation","count":count,"control":control,"bit_mismatches":0,"full_owned_words_checked":checked_words,"complete_gu_hidden_status_fault_sources_padding":true,"original_checked_pair":true,"actual_direct_global_matrix_and_validating_activation":true,"fabricated_prompt_or_batch_scope":false,"reset_and_owned_erasure":true}),
    )
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> ProbeResult<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled()
        || arena.buffer_count() != 0
        || !std::ptr::eq(client.info(), arena.client()?.info())
    {
        return Err("direct GU requires original empty fixed-public audit arena".into());
    }
    let mut out = Vec::new();
    for count in [256, 257] {
        for control in 0..9 {
            let result = one(client, arena, count, control);
            arena.reset()?;
            out.push(result?);
        }
    }
    Ok(out)
}
