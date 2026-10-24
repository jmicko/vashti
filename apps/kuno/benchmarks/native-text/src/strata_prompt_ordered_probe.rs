//! Additive, fixed-public paired controls. Both sides launch the original
//! production kernels with real arena owners; no Scope or owner proof is made
//! up. The complete PART owners are checked in bounded readback windows.
use super::*;
use crate::{gr_batch_probe as probe, strata_prompt::{self as p, groups}};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn fail(label: &'static str) -> Box<dyn std::error::Error> { label.into() }
fn metadata(words: &[u32]) -> MetadataBindingInfo {
    MetadataBindingInfo::custom(words.chunks_exact(2)
        .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32)).collect())
}
fn grouped(count: usize) -> Result<Zeroizing<Vec<u32>>> { grouped_policy(count, false) }
fn grouped_policy(count: usize, row_order: bool) -> Result<Zeroizing<Vec<u32>>> {
    let mut raw = Zeroizing::new(vec![0; groups::LOGICAL + count * 10]);
    raw[..3].copy_from_slice(&[count as u32, 0, count as u32]);
    for route in 0..count * 10 {
        let logical = ((route / 10 * 7 + route % 10 * 37) % 512) as u32;
        raw[1540 + route] = (logical * 13 + 17) % 512;
        raw[groups::LOGICAL + route] = logical;
    }
    if row_order { groups::group_for_prefill(&raw, count) } else { groups::group(&raw, count) }
}
/// Sparse public initialization uses the genuine original copy kernel and
/// retains BOTH real owners. Its small source avoids a400MiB host payload.
fn part(
    client: &ComputeClient<WgpuRuntime>, arena: &mut PrivateArena, prefix: &[u32],
) -> ProbeResult<BufferId> {
    let target = arena.allocate_zeroed(groups::PART_BYTES as usize / 4)?;
    let source = arena.upload_u32(prefix)?;
    let mut words = [0u32; 16];words[0] = 1;words[1] = prefix.len() as u32;
    let args = KernelArguments::new().with_buffers(vec![
        u::handle(arena, source)?.binding(), u::handle(arena, target)?.binding(),
    ]).with_info(metadata(&words));
    // SAFETY: exact fixed-public prefix and retained disjoint full arena roots.
    unsafe { client.launch_unchecked(Box::new(p::Kernel::Copy),
        CubeCount::Static(prefix.len().div_ceil(256) as u32, 1, 1), args); }
    Ok(target)
}
fn check_full(arena: &PrivateArena, id: BufferId, prefix: &[u32]) -> ProbeResult<()> {
    let root = arena.handle(id)?.clone();
    let owned = arena.layout(id)?.owned_bytes / 4;
    let window = crate::private_arena::ZERO_AUDIT_WINDOW / 4;
    for first in (0..owned).step_by(window) {
        let n = window.min(owned - first);
        let view = root.clone().offset_start((first * 4) as u64)
            .offset_end(((owned - first - n) * 4) as u64);
        let actual = u::read_handle_prefix_u32(arena, view, n)?;
        if actual.iter().enumerate().any(|(i, &word)| word != prefix.get(first+i).copied().unwrap_or(0)) {
            return Err(fail("ordered_complete_PART_owner_mismatch"));
        }
        drop(actual);
        let receipt = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
        let bytes = receipt.upload_bytes.checked_add(receipt.uniform_bytes)
            .and_then(|n| n.checked_add(receipt.readback_bytes))
            .ok_or_else(|| fail("ordered_readback_audit_overflow"))?;
        if bytes == 0 || receipt.inspected_bytes != bytes || receipt.inspected_buffers == 0 {
            return Err(fail("ordered_readback_audit_incomplete"));
        }
    }
    Ok(())
}

struct Inputs<'a> {
    count:usize,map:&'a[u32],inverse:&'a[u32],probability:&'a[u32],
    shared:&'a[u32],scalar:&'a[u32],routed:&'a[u32],combined:&'a[u32],
}
/// Only ONE complete400MiB PART owner is ever live in the original512MiB
/// arena. Across the verified reset, retain only these bounded zeroizing
/// routed/combined/status/audit/fault reference observations.
fn side(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena,input:&Inputs<'_>,
    ordered:bool,prefix:&[u32],expected_part:&[u32],status:&[u32],fault:u32)
    -> ProbeResult<Vec<Zeroizing<Vec<u32>>>>
{
    let part_id=part(client,arena,prefix)?;
    let probability_id=arena.upload_u32(input.probability)?;
    let map_id=arena.upload_u32(input.map)?;
    let inverse_id=arena.upload_u32(input.inverse)?;
    let shared_id=arena.upload_u32(input.shared)?;
    let scalar_id=arena.upload_u32(input.scalar)?;
    let status_id=arena.upload_u32(status)?;
    let routed=arena.upload_u32(input.routed)?;
    let combined=arena.upload_u32(input.combined)?;
    let audit=arena.allocate_zeroed(8192)?;
    let fault_id=arena.upload_u32(&[fault])?;
    if ordered {
        let handles=[part_id,probability_id,status_id,map_id,routed,audit,
            fault_id,inverse_id,shared_id,scalar_id,combined]
            .map(|id|u::handle(arena,id)).into_iter().collect::<Result<Vec<_>>>()?;
        launch_public_ordered_shared(client,handles.try_into()
            .map_err(|_|fail("ordered_bindings"))?,input.count,7,19)?;
    } else {
        let args=KernelArguments::new().with_buffers([
            part_id,probability_id,status_id,map_id,routed,audit,fault_id,
        ].into_iter().map(|id|u::handle(arena,id).map(|h|h.binding())).collect::<Result<Vec<_>>>()?)
            .with_info(metadata(&[input.count as u32,7,19,0,0,2560,0,0]));
        // The unchanged rank reduction and SharedMerge remain the oracle.
        unsafe {client.launch_unchecked(Box::new(Kernel::PromptReduce),
            CubeCount::Static(10,input.count as u32,1),args);}
        let mut words=[0u32;16];words[0]=input.count as u32;
        let args=KernelArguments::new().with_buffers([
            routed,shared_id,scalar_id,combined,fault_id,
        ].into_iter().map(|id|u::handle(arena,id).map(|h|h.binding())).collect::<Result<Vec<_>>>()?)
            .with_info(metadata(&words));
        unsafe {client.launch_unchecked(Box::new(p::Kernel::SharedMerge),
            CubeCount::Static((input.count*2560).div_ceil(256) as u32,1,1),args);}
    }
    let observed=[routed,combined,status_id,audit,fault_id].into_iter()
        .map(|id|probe::read_native_public(arena,id)).collect::<ProbeResult<Vec<_>>>()?;
    if observed[3].iter().enumerate().any(|(i,&word)| {
        let expected=if (28..32).contains(&i) {[input.count as u32,19,127,8][i-28]} else {0};
        word!=expected
    }) {return Err(fail("ordered_actual_layer_stamp"));}
    check_full(arena,part_id,expected_part)?;
    for (id,expected) in [(map_id,input.map),(inverse_id,input.inverse),
        (probability_id,input.probability),(shared_id,input.shared),(scalar_id,input.scalar)] {
        check_full(arena,id,expected)?;
    }
    // Reset proves complete owned erasure before the other side allocates.
    arena.reset()?;
    Ok(observed)
}

pub(crate) fn cases(client: &ComputeClient<WgpuRuntime>, arena: &mut PrivateArena)
    -> ProbeResult<Vec<serde_json::Value>> { policy_cases(client, arena, false) }
#[cfg(feature="strata-cohort-row-order")]
pub(crate) fn row_order_cases(client: &ComputeClient<WgpuRuntime>, arena: &mut PrivateArena)
    -> ProbeResult<Vec<serde_json::Value>> { policy_cases(client, arena, true) }
fn policy_cases(client: &ComputeClient<WgpuRuntime>, arena: &mut PrivateArena, row_order: bool)
    -> ProbeResult<Vec<serde_json::Value>>
{
    let mut reports = Vec::new();
    for (count, control) in [
        (1usize,"finite"), (9,"finite"), (35,"finite"),
        (9,"raw_col0_nan"), (9,"raw_other_inf"), (9,"missing_down"),
        (9,"missing_gu"), (9,"weight_nan"), (9,"weight_sum"),
        (9,"shared_nan"), (9,"scalar_nan"), (9,"inverse_wrong"),
        (9,"inverse_oob"), (9,"unknown_down"), (9,"weight_negative"),
        (9,"shared_inf"),
    ] {
        let map = grouped_policy(count, row_order)?;
        let inverse = groups::ordered_inverse(&map, count)?;
        let mut raw = Zeroizing::new((0..count*10*2560).map(|i| {
            if i % 13 == 0 { (-0.0f32).to_bits() }
            else { ((((i / 2560)*17 + i%2560*3)%97) as f32/32.0-1.5).to_bits() }
        }).collect::<Vec<_>>());
        match control {
            "raw_col0_nan" | "missing_down" => raw[0] = 0x7fc01234,
            "raw_other_inf" => raw[7] = f32::INFINITY.to_bits(),
            _ => {},
        }
        let mut old_status = Zeroizing::new(vec![0; groups::STATUS_WORDS]);
        let mut new_status = Zeroizing::new(vec![0; groups::STATUS_WORDS]);
        let mut old_prefix = Zeroizing::new(vec![0; raw.len()]);
        let mut new_prefix = Zeroizing::new(vec![0; raw.len()]);
        let mut expected_prefix = Zeroizing::new(vec![0; raw.len()]);
        let mut old_fault = 0u32;
        for route in 0..count*10 {
            let gu = u32::from(!(control == "missing_gu" && route == 0));
            let missing = control == "missing_down" && route == 0;
            let marker = if missing { 0 } else if route%2 == 0 { 2 } else { 1 };
            let down = u32::from(!missing && gu == 1 && f32::from_bits(raw[route*2560]).is_finite());
            old_status[route] = gu;new_status[route] = gu;
            old_status[groups::ROUTES+route] = if route == 0 && control.starts_with("inverse_") { 0 } else { down };
            new_status[groups::ROUTES+route] = if marker == 2 { 2 } else { down };
            if control == "unknown_down" && route == 0 {
                old_status[groups::ROUTES+route] = 3;
                new_status[groups::ROUTES+route] = 3;
            }
            let ordinal = inverse[route] as usize;
            for column in 0..2560 {
                let bits = raw[route*2560+column];
                let finite = f32::from_bits(bits).is_finite();
                let sanitized = if finite || missing { bits } else { 0 };
                if !finite && !missing { old_fault |= 128; }
                old_prefix[route*2560+column] = sanitized;
                new_prefix[ordinal*2560+column] = if marker == 2 || missing { bits } else { sanitized };
                expected_prefix[ordinal*2560+column] = sanitized;
            }
        }
        let mut probability = Zeroizing::new(vec![0; groups::PROB_WORDS]);
        probability[..count*10].fill(0.1f32.to_bits());
        if control == "weight_nan" { probability[0] = f32::NAN.to_bits(); }
        if control == "weight_sum" { probability[0] = 0.2f32.to_bits(); }
        if control == "weight_negative" { probability[0] = (-0.1f32).to_bits(); }
        let mut inverse_owner = Zeroizing::new(vec![0; groups::RAW_WORDS]);
        inverse_owner[..inverse.len()].copy_from_slice(&inverse);
        // Deliberately corrupt only the compiled-public negative fixture,
        // after the exact same checked inverse construction as production.
        if control == "inverse_wrong" { inverse_owner[0] = inverse_owner[1]; }
        if control == "inverse_oob" { inverse_owner[0] = (count*10) as u32; }
        let logical = count*2560;
        let routed_seed = Zeroizing::new(vec![(-7.25f32).to_bits(); logical+64]);
        let combined_seed = Zeroizing::new(vec![(23.5f32).to_bits(); logical+64]);
        let mut shared = Zeroizing::new((0..logical+64).map(|i|
            ((i%23) as f32/64.0-0.125).to_bits()).collect::<Vec<_>>());
        let mut scalar = Zeroizing::new((0..count+64).map(|i|
            ((i%7) as f32/4.0-0.75).to_bits()).collect::<Vec<_>>());
        if control == "shared_nan" { shared[11] = f32::NAN.to_bits(); }
        if control == "shared_inf" { shared[11] = f32::INFINITY.to_bits(); }
        if control == "scalar_nan" { scalar[0] = f32::NAN.to_bits(); }
        let input=Inputs {count,map:&map,inverse:&inverse_owner,probability:&probability,
            shared:&shared,scalar:&scalar,routed:&routed_seed,combined:&combined_seed};
        let old=side(client,arena,&input,false,&old_prefix,&old_prefix,&old_status,old_fault)?;
        let new=side(client,arena,&input,true,&new_prefix,&expected_prefix,&new_status,0)?;
        for (a,b) in old.iter().zip(&new) {
            if **a!=**b {return Err(fail("ordered_original_poststate_mismatch"));}
        }
        let fault=&new[4];
        let expected_fault = match control {
            "finite" => 0, "raw_col0_nan" => 128|256, "raw_other_inf" => 128,
            "shared_nan"|"shared_inf"|"scalar_nan" => 1, _ => 256,
        };
        if fault[0] != expected_fault || fault[1..].iter().any(|&v|v!=0) {
            return Err(fail("ordered_expected_sticky_fault"));
        }
        println!("public_prompt_ordered count={count} control={control} mismatches=0");
        reports.push(serde_json::json!({"family":"prompt-ordered-shared", "count":count,
            "control":control,"row_order_selected":row_order,"original_pair_bitwise":true,"mismatches":0,
            "fault":expected_fault,"complete_owners_checked":true,"actual_native_kernels":true}));
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_vector_workgroups_cover_all_components_once_and_keep_rank_order() {
        for count in [1,9,148,257,groups::CAPACITY] {
            let map = grouped(count).unwrap();
            let inverse = groups::ordered_inverse(&map,count).unwrap();
            let mut components = vec![0u8;640];
            for lane in 0..256 {
                for column in (lane..640).step_by(256) { components[column] += 1; }
            }
            assert!(components.iter().all(|&n|n==1));
            for token in 0..count {
                let routes = (0..10).map(|rank|token*10+rank).collect::<Vec<_>>();
                for (&route,rank) in routes.iter().zip(0..10) {
                    assert_eq!(route,token*10+rank);
                    assert_eq!(map[1540+inverse[route] as usize],route as u32);
                }
            }
        }
    }
}
