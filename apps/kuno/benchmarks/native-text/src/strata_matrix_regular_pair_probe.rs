//! Actual per-expert versus distinct regular-cohort GPU programs, all owners compared.
//! Independent original F64 acceptance remains in public_oracle controls.
use super::*;
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
    strata_prompt::groups as g,
};
use zeroize::Zeroizing;
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WINDOW_WORDS: usize = 4 * 1024 * 1024;
// A public oracle only: never attached to a private inference request. The
// largest original DOWN owner is400MiB. Keep its reference in scrubbed host
// memory rather than charging a second400MiB GPU owner to the original arena.
const REFERENCE_BYTES: usize = 400 * 1024 * 1024;
fn sanitize(arena: &PrivateArena) -> Checked<()> {
    let receipt = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
    let bytes = receipt
        .upload_bytes
        .checked_add(receipt.uniform_bytes)
        .and_then(|v| v.checked_add(receipt.readback_bytes))
        .ok_or("public cohort pair IO audit overflow")?;
    if bytes == 0 || receipt.inspected_bytes != bytes || receipt.inspected_buffers == 0 {
        return Err("public cohort pair complete transient IO erasure".into());
    }
    Ok(())
}
fn compare(arena: &PrivateArena, a: BufferId, b: BufferId) -> Checked<usize> {
    let words = arena.layout(a)?.owned_bytes / 4;
    if arena.layout(b)?.owned_bytes / 4 != words {
        return Err("public cohort paired whole-owner extent".into());
    }
    for start in (0..words).step_by(WINDOW_WORDS) {
        let count = WINDOW_WORDS.min(words - start);
        let view = |id| -> Checked<Handle> {
            Ok(arena
                .handle(id)?
                .clone()
                .offset_start((start * 4) as u64)
                .offset_end(((words - start - count) * 4) as u64))
        };
        let x = u::read_handle_prefix_u32(arena, view(a)?, count)?;
        let y = u::read_handle_prefix_u32(arena, view(b)?, count)?;
        let equal = *x == *y;
        drop(x);
        drop(y);
        sanitize(arena)?;
        if !equal {
            return Err("public cohort exact original matrix/owner/status mismatch".into());
        }
    }
    Ok(words)
}
fn window(arena: &PrivateArena, id: BufferId, start: usize) -> Checked<(Handle, usize)> {
    let words = arena.layout(id)?.owned_words;
    if start >= words {
        return Err("public cohort pair full-owner window".into());
    }
    let count = WINDOW_WORDS.min(words - start);
    let view = arena
        .handle(id)?
        .clone()
        .offset_start((start * 4) as u64)
        .offset_end(((words - start - count) * 4) as u64);
    Ok((view, count))
}
fn snapshot(arena: &PrivateArena, id: BufferId) -> Checked<Zeroizing<Vec<u32>>> {
    let words = arena.layout(id)?.owned_words;
    if words
        .checked_mul(4)
        .is_none_or(|bytes| bytes > REFERENCE_BYTES)
    {
        return Err("public cohort pair bounded host reference".into());
    }
    let mut reference = Zeroizing::new(Vec::new());
    reference.try_reserve_exact(words)?;
    if reference
        .capacity()
        .checked_mul(4)
        .is_none_or(|bytes| bytes > REFERENCE_BYTES)
    {
        return Err("public cohort pair bounded host reference capacity".into());
    }
    reference.resize(words, 0);
    for start in (0..words).step_by(WINDOW_WORDS) {
        let (view, count) = window(arena, id, start)?;
        let data = u::read_handle_prefix_u32(arena, view, count)?;
        reference[start..start + count].copy_from_slice(&data);
        drop(data);
        sanitize(arena)?;
    }
    Ok(reference)
}
fn erase_output(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
) -> Checked<()> {
    let layout = arena.layout(id)?;
    let owner = arena.handle(id)?.clone();
    if owner.offset_start.unwrap_or(0) != 0
        || owner.offset_end.unwrap_or(0) != 0
        || owner.size_in_used() != layout.owned_bytes as u64
        || owner.size() != layout.owned_bytes as u64
    {
        return Err("public cohort pair exact output owner erase".into());
    }
    // This is the unchanged original arena eraser and its exact full-span
    // launch geometry. Ownership stays with the arena until reset completes.
    crate::private_arena::erase_full_span::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(layout.workgroups_x, layout.workgroups_y, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(owner, layout.owned_words) },
    );
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    for start in (0..layout.owned_words).step_by(WINDOW_WORDS) {
        let (view, count) = window(arena, id, start)?;
        let data = u::read_handle_prefix_u32(arena, view, count)?;
        let zero = data.iter().all(|word| *word == 0);
        drop(data);
        sanitize(arena)?;
        if !zero {
            return Err("public cohort pair output owner not completely zero".into());
        }
    }
    Ok(())
}
fn compare_reference(arena: &PrivateArena, id: BufferId, reference: &[u32]) -> Checked<usize> {
    let words = arena.layout(id)?.owned_words;
    if words != reference.len()
        || words
            .checked_mul(4)
            .is_none_or(|bytes| bytes > REFERENCE_BYTES)
    {
        return Err("public cohort paired reference whole-owner extent".into());
    }
    for start in (0..words).step_by(WINDOW_WORDS) {
        let (view, count) = window(arena, id, start)?;
        let actual = u::read_handle_prefix_u32(arena, view, count)?;
        let expected = &reference[start..start + count];
        let mismatch = if actual.as_slice() == expected {
            None
        } else {
            actual
                .iter()
                .zip(expected)
                .enumerate()
                .find_map(|(i, (got, want))| (got != want).then_some((start + i, *want, *got)))
        };
        drop(actual);
        sanitize(arena)?;
        if let Some((word, original, selected)) = mismatch {
            return Err(format!("public cohort exact original matrix/owner/status mismatch word={word} original_bits={original:08x} cohort_bits={selected:08x}").into());
        }
    }
    Ok(words)
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    dual: bool,
    layer: bool,
    large: bool,
    down: bool,
    poison: bool,
    counts: &[usize],
) -> Checked<serde_json::Value> {
    let slots = counts.len();
    let total = 4096usize;
    let mut raw = Zeroizing::new(vec![0u32; g::LOGICAL + total * 10]);
    raw[..3].copy_from_slice(&[total as u32, 0, total as u32]);
    for token in 0..total {
        for rank in 0..10 {
            let route = token * 10 + rank;
            let logical = if rank < slots && token >= total - counts[rank] {
                400 + rank
            } else {
                1 + rank
            };
            raw[g::LOGICAL + route] = logical as u32;
            raw[1540 + route] = ((logical * 13 + 17) % 512) as u32;
        }
    }
    let mut map = g::group(&raw, total)?;
    let first = g::validate(&map, total)? - slots;
    if poison {
        let offset = map[1028 + first + slots - 1] as usize;
        map[1540 + offset] = g::ROUTES as u32;
    }
    let cols = if down { 640 } else { 2560 };
    let rows = if down { 2560 } else { 1280 };
    let mut words = Zeroizing::new([0u32; 48]);
    words[0] = total as u32;
    words[1] = cols;
    words[2] = rows;
    words[4] = first as u32;
    words[5] = if down { 2 } else { 0 };
    words[6] = 100;
    words[7] = 100;
    words[8] = 300;
    words[9] = slots as u32;
    words[10] = *counts.iter().max().unwrap() as u32;
    words[11] = 16;
    let mut prefix = 0;
    for (i, n) in counts.iter().copied().enumerate() {
        if map[516 + first + i] as usize != n {
            return Err("public cohort paired authentic group fixture".into());
        }
        words[16 + i * 4..20 + i * 4].copy_from_slice(&[(first + i) as u32, n as u32, prefix, 8]);
        prefix += n.div_ceil(if large {32}else{16}) as u32 * if large {32}else{16};
    }
    let mut image = Zeroizing::new(vec![
        0u32;
        8 * crate::strata_prompt::experts::cohorts::IMAGE_BYTES
            as usize
            / 4
    ]);
    let mut input = Zeroizing::new(vec![0u32; prefix as usize * cols as usize / 2]);
    for i in 0..slots {
        let row_prefix = words[18 + i * 4] as usize;
        let members = counts[i];
        for member in 0..members {
            for col in 0..cols as usize {
                let value = half::f16::from_f32(
                    0.125 + ((member * 13 + col * 7 + i * 3) % 43) as f32 / 19.0,
                );
                let at = (row_prefix + member) * cols as usize + col;
                input[at / 2] |= u32::from(value.to_bits()) << (16 * (at % 2));
            }
        }
        let half_first = i * 4_915_200 + if down { 3_276_800 } else { 0 };
        for row in 0..rows as usize {
            for col in 0..cols as usize {
                let value = if poison && i == slots - 1 && row == 0 && col == 0 {
                    half::f16::NAN
                } else {
                    half::f16::from_f32(
                        (((row * 17 + col * 11 + i * 19) % 67) as f32 - 33.0) / 41.0,
                    )
                };
                let at = half_first + row * cols as usize + col;
                image[at / 2] |= u32::from(value.to_bits()) << (16 * (at % 2));
            }
        }
    }
    let src = arena.upload_u32(&input)?;
    let weight = arena.upload_u32(&image)?;
    let map_id = arena.upload_u32(&map)?;
    let output_bytes = if down { g::PART_BYTES } else { g::GU_BYTES };
    let a = arena.allocate_zeroed(output_bytes as usize / 4)?;
    let status = Zeroizing::new(vec![1u32; g::STATUS_WORDS]);
    let sa = arena.upload_u32(&status)?;
    let sb = arena.upload_u32(&status)?;
    let fa = arena.allocate_zeroed(1)?;
    let fb = arena.allocate_zeroed(1)?;
    #[cfg(feature="strata-layer-gather")]
    let layer_input=if layer {crate::strata_prompt::layer_gather::probe_util::input(arena,&map,&words,&input,src,down)?} else {None};
    #[cfg(not(feature="strata-layer-gather"))]
    assert!(!layer);
    // Genuine preceding per-expert granular programs. One reusable original
    // Pad32 input owner, updated through scrubbed transient uploads. The DOWN
    // reference already writes original ordered PART; GU's local rows are
    // packed into a zeroizing host oracle, without a second GPU output owner.
    let ri = arena.allocate_zeroed((if large {4096}else{256}) * cols as usize / 2)?;
    let mut reference = Zeroizing::new(if down {Vec::new()}else{vec![0u32; arena.layout(a)?.owned_words]});
    for (i, &members) in counts.iter().enumerate() {
        let mut packed = Zeroizing::new(vec![0u32; arena.layout(ri)?.owned_words]);
        let start = words[18 + i * 4] as usize * cols as usize / 2;
        let real = members * cols as usize / 2;
        packed[..real].copy_from_slice(&input[start..start + real]);
        client
            .write_private(
                // Private upload requires the FULL retained owner, including
                // allocator padding; matrix inputs still use the logical view.
                arena.handle(ri)?.clone(),
                packed.iter().copied().flat_map(u32::to_le_bytes).collect(),
            )
            .map_err(u::fault)?;
        let handles=[u::handle(arena,ri)?,u::handle(arena,weight)?,u::handle(arena,a)?,
            u::handle(arena,map_id)?,u::handle(arena,sa)?,u::handle(arena,fa)?];
        let wf=i*4_915_200+if down {3_276_800}else{0};
        let selected=if large {
            #[cfg(feature="strata-large-cohorts")]
            {
                if down {
                    crate::strata_matrix::native_expert_wide::ordered::try_expert_mode_small_tail(
                        client,handles,members,cols as usize,rows as usize,wf,first+i,true,false)?
                } else {
                    crate::strata_matrix::native_expert_wide::try_expert_mode_small_tail(
                        client,handles,members,cols as usize,rows as usize,wf,first+i,false,true)?
                }
            }
            #[cfg(not(feature="strata-large-cohorts"))]
            {return Err("large cohort probe feature disabled".into());}
        }else{
            crate::strata_matrix::native::granular_small::try_expert(
                client,handles,members,cols as usize,rows as usize,wf,first+i,down)?
        };
        if !selected {
            return Err("public regular cohort actual preceding reference not selected".into());
        }
        cubecl::future::block_on(client.sync()).map_err(u::fault)?;
        // Include every padding word in the input source check before reuse.
        let uploaded = snapshot(arena, ri)?;
        if *uploaded != *packed {
            return Err("public regular per-expert input source changed".into());
        }
        if !down {
            let local = snapshot(arena, a)?;
            let at = words[18 + i * 4] as usize * 1280;
            reference[at..at + members * 1280].copy_from_slice(&local[..members * 1280]);
            erase_output(client, arena, a)?;
        }
    }
    if down {
        reference = snapshot(arena, a)?;
    }
    erase_output(client, arena, a)?;
    words[13]=u32::from(layer);
    #[cfg(feature="strata-layer-gather")]
    let selected_src=layer_input.as_ref().map_or(src,|(id,_)|*id);
    #[cfg(not(feature="strata-layer-gather"))]
    let selected_src=src;
    let handles=[u::handle(arena,selected_src)?,u::handle(arena,weight)?,u::handle(arena,a)?,
        u::handle(arena,map_id)?,u::handle(arena,sb)?,u::handle(arena,fb)?];
    if large {
        #[cfg(feature="strata-large-cohorts")]
        for tile in [128,8,16,32] {
            if large::part_rows(&words,tile)==0 {continue;}
            words[11]=tile as u32;
            large::launch(client,handles.clone(),&words,down,tile)?;
        }
        #[cfg(not(feature="strata-large-cohorts"))]
        return Err("large cohort probe feature disabled".into());
    } else {
        #[cfg(feature="strata-small-dual-cells")]
        let product=if dual {dual_regular::launch}else{regular::launch};
        #[cfg(not(feature="strata-small-dual-cells"))]
        let product={assert!(!dual);regular::launch};
        product(client,handles,&words,down,16)?;
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let checked = compare_reference(arena, a, &reference)?;
    let reference_bytes = reference.capacity() * 4;
    drop(reference);
    compare(arena, sa, sb)?;
    compare(arena, fa, fb)?;
    for (id, expected) in [(src, &*input), (weight, &*image), (map_id, &*map)] {
        let root = arena.handle(id)?.clone();
        let owned = arena.layout(id)?.owned_bytes / 4;
        for start in (0..owned).step_by(4 * 1024 * 1024) {
            let n = (4 * 1024 * 1024).min(owned - start);
            let view = root
                .clone()
                .offset_start((start * 4) as u64)
                .offset_end(((owned - start - n) * 4) as u64);
            let actual = u::read_handle_prefix_u32(arena, view, n)?;
            let unchanged = !actual
                .iter()
                .enumerate()
                .any(|(i, v)| *v != expected.get(start + i).copied().unwrap_or(0));
            drop(actual);
            sanitize(arena)?;
            if !unchanged {
                return Err("public cohort paired original source owner changed".into());
            }
        }
    }
    #[cfg(feature="strata-layer-gather")]
    if let Some((id,expected))=&layer_input {crate::strata_prompt::layer_gather::probe_util::source(arena,*id,expected)?;}
    Ok(
        serde_json::json!({"family":if large {"large-packed-cohort-original-wide-prefix-tail-pair"}else{"regular-packed-cohort-original-per-expert-pair"},
        "actual_previous_per_expert_granular_reference":!large,"actual_new_regular_cohort":!dual&&!large,"actual_large_wide_prefix_tail":large,
        "down":down,"groups":slots,"counts":counts,"poison_and_invalid_route":poison,
        "actual_dual_cells":dual,"layer_gather_input":layer,"actual_previous_and_new_gpu_programs":true,"original_map_builder_and_validator":true,
        "bit_mismatches":0,"full_output_words_checked":checked,
        "full_owner_status_fault_sources_and_tail_checked":true,
        "single_original_gpu_output_owner":true,"full_output_owner_zero_verified_between_programs":true,
        "zeroizing_host_reference_bytes":reference_bytes,"host_reference_limit_bytes":REFERENCE_BYTES,
        "readback_window_bytes":WINDOW_WORDS*4,"original_arena_budget_bytes":arena.budget_bytes(),
        "actual_arena_charged_bytes":arena.charged_bytes()}),
    )
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<Vec<serde_json::Value>> {
    cases_mode(client,arena,false,false)
}
#[cfg(feature="strata-small-dual-cells")]
pub(crate) fn dual_cases(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena)->Checked<Vec<serde_json::Value>> {
    cases_mode(client,arena,true,false)
}
#[cfg(feature="strata-large-cohorts")]
pub(crate) fn large_cases(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena)->Checked<Vec<serde_json::Value>> {
    cases_mode(client,arena,false,true)
}
fn cases_mode(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena,dual:bool,large:bool)->Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("public cohort pair original empty audit arena required".into());
    }
    let mut reports = Vec::new();
    let fixtures:Vec<&[usize]>=if large {
        vec![&[256,257,264,265,287,288,383,511],&[512;8],&[4095],&[4096]]
    }else{vec![&[32,33,63,64,65,127,128,255],&[255;8],&[32,33,255]]};
    for counts in fixtures {
        for down in [false, true] {
            for poison in [false, true] {
                let result = one(client, arena, dual, false, large, down, poison, counts).map_err(
                    |e| -> Box<dyn std::error::Error> {
                        format!(
                            "public regular paired down={down} poison={poison} groups={}: {e}",
                            counts.len()
                        )
                        .into()
                    },
                );
                arena.reset()?;
                reports.push(result?);
            }
        }
    }
    Ok(reports)
}

#[cfg(feature="strata-layer-gather")]
pub(crate) fn layer_cases(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena)->Checked<Vec<serde_json::Value>> {
 if !arena.public_owned_zero_audit_enabled()||arena.buffer_count()!=0 {return Err("layer matrix empty audited arena required".into());}
 let mut reports=Vec::new();
 for (large,counts) in [(false,&[32usize,33,63,64,65,127,128,255][..]),(true,&[256,257,264,265,287,288,383,511][..]),(true,&[4095][..]),(true,&[4096][..])] {
  for down in [false,true] {for poison in [false,true] {
   let r=one(client,arena,!large,true,large,down,poison,counts);
   match r {Ok(report)=>{arena.reset()?;reports.push(report);},Err(error)=>{let _=arena.reset();return Err(format!("layer regular/large large={large} down={down} poison={poison}: {error}").into());}}
  }}
 }
 Ok(reports)
}
