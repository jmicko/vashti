//! Additive packed LARGE matrix pairs against untouched original per-expert math.
//! Reuse each input/weight owner only after its original-source readback check.
//! Producer-specific controls live separately; this tests every actual consumer.
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
// The largest snapshotted owner is PART (all prompt routes x 2560 f32).
const REFERENCE_BYTES: usize = crate::strata_prompt::groups::PART_BYTES as usize;
// Only this compiled-public numeric fixture calls these diagnostics. The
// private worker protocol and global error logging are deliberately unchanged.
fn device_fault(stage:&'static str,cause:impl std::fmt::Debug+std::fmt::Display)->kuno_text::Error {
    eprintln!("public_packed_matrix_stage stage={stage} cause={cause:?}");
    u::fault(cause)
}
fn sanitize(arena: &PrivateArena) -> Checked<()> {
    let receipt = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())
        .map_err(|cause|device_fault("sanitize_io",cause))?;
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
    cubecl::future::block_on(client.sync()).map_err(|cause|device_fault("erase_output_sync",cause))?;
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
    down: bool,
    poison: bool,
    counts: &[usize],
) -> Checked<serde_json::Value> {
    one_mode(client,arena,down,poison,counts,false,None)
}
#[allow(clippy::too_many_arguments)]
fn one_mode(
    client: &ComputeClient<WgpuRuntime>, arena: &mut PrivateArena,
    down: bool, poison: bool, counts: &[usize], shared: bool,
    invalid_route: Option<usize>,
) -> Checked<serde_json::Value> {
    let (dual,layer,large)=(false,true,true);
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
    if poison || invalid_route.is_some() {
        let offset = map[1028 + first + slots - 1] as usize;
        let invalid=invalid_route.unwrap_or(0);
        if invalid>=128 || invalid>=counts[slots-1] {return Err("shared prefix invalid-route fixture extent".into());}
        map[1540 + offset + invalid] = g::ROUTES as u32;
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
                let value = if poison && i==slots-1 && member==1 && col==0 {
                    if down {half::f16::INFINITY}else{half::f16::NAN}
                } else {half::f16::from_f32(
                    0.125 + ((member * 13 + col * 7 + i * 3) % 43) as f32 / 19.0,
                )};
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
    // Genuine preceding per-expert granular programs. One reusable original
    // Pad32 input owner, updated through scrubbed transient uploads. The DOWN
    // reference already writes original ordered PART; GU's local rows are
    // packed into a zeroizing host oracle, without a second GPU output owner.
    let ri = arena.allocate_zeroed((if large {4096}else{256}) * cols as usize / 2)?;
    let mut reference = Zeroizing::new(if down {Vec::new()}else{vec![0u32; arena.layout(a)?.owned_words]});
    for (i, &members) in counts.iter().enumerate() {
        eprintln!("public_packed_matrix_stage stage=reference_begin slot={i} members={members} down={down} poison={poison}");
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
            .map_err(|cause|device_fault("reference_input_write",cause))?;
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
        cubecl::future::block_on(client.sync()).map_err(|cause|device_fault("reference_sync",cause))?;
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
    // Check the full ORIGINAL sources before intentionally reusing these owners
    // with the size-preserving packed representation. The original reference
    // already completed and no longer retains a read dependency.
    crate::strata_prompt::layer_gather::probe_util::source(arena,src,&input)?;
    crate::strata_prompt::layer_gather::probe_util::source(arena,weight,&image)?;
    input=pack_input(&input,&words)?;
    image=pack_weight(&image,&words,down)?;
    overwrite(client,arena,src,&input,ri)?;
    overwrite(client,arena,weight,&image,ri)?;
    words[13]=1;words[14]=1;
    let layer_input=packed_global_input(arena,&map,&words,&input,src,down)?;
    let selected_src=layer_input.as_ref().map_or(src,|(id,_)|*id);
    let handles=[u::handle(arena,selected_src)?,u::handle(arena,weight)?,u::handle(arena,a)?,
        u::handle(arena,map_id)?,u::handle(arena,sb)?,u::handle(arena,fb)?];
    if large {
        #[cfg(feature="strata-large-cohorts")]
        for tile in [128,8,16,32] {
            if large_packed::part_rows(&words,tile)==0 {continue;}
            words[11]=tile as u32;
            eprintln!("public_packed_matrix_stage stage=selected_launch tile={tile} down={down} poison={poison}");
            if shared {
                #[cfg(feature="strata-large-shared-panels")]
                large_shared::launch(client,handles.clone(),&words,down,tile)?;
                #[cfg(not(feature="strata-large-shared-panels"))]
                return Err("shared large matrix probe feature disabled".into());
            } else {large_packed::launch(client,handles.clone(),&words,down,tile)?;}
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
    cubecl::future::block_on(client.sync()).map_err(|cause|device_fault("selected_sync",cause))?;
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
    let mut report=serde_json::json!({"family":"large-packed-tile-original-wide-prefix-tail-pair",
        "actual_original_per_expert_reference":true,"actual_large_tiled_operands":true,"actual_new_regular_cohort":!dual&&!large,"actual_large_wide_prefix_tail":large,
        "down":down,"groups":slots,"counts":counts,"poison_and_invalid_route":poison,
        "actual_dual_cells":dual,"layer_gather_input":layer,"actual_previous_and_new_gpu_programs":true,"original_map_builder_and_validator":true,
        "bit_mismatches":0,"full_output_words_checked":checked,
        "poison_input_nan_or_inf_and_weight_nan":poison,"original_sources_verified_before_intentional_owner_reuse":true,
        "full_owner_status_fault_sources_and_tail_checked":true,
        "single_original_gpu_output_owner":true,"full_output_owner_zero_verified_between_programs":true,
        "zeroizing_host_reference_bytes":reference_bytes,"host_reference_limit_bytes":REFERENCE_BYTES,
        "readback_window_bytes":WINDOW_WORDS*4,"original_arena_budget_bytes":arena.budget_bytes(),
        "actual_arena_charged_bytes":arena.charged_bytes()});
    if shared {
        report["family"]=serde_json::json!("large-shared-k16-original-wide-prefix-tail-pair");
        report["actual_shared_large_prefix"]=serde_json::json!(true);
        report["shared_panel_bytes"]=serde_json::json!(8192);
        report["shared_f32_stride"]=serde_json::json!(8);
        report["mixed_invalid_route_offset"]=serde_json::json!(invalid_route);
        report["all_subgroups_same_128_routes"]=serde_json::json!(true);
    }
    Ok(report)
}

// Independent CPU bit permutation; no dequantization or arithmetic change. Each
// row-block size is fixed here, independently of the consumer's CubeCL helpers.
fn packed_half(row:usize,k:usize,columns:usize,block:usize)->usize {
    (row/block)*(block*columns)+(k/16)*(block*16)+(row%block)*16+k%16
}
fn pack_input(source:&[u32],words:&[u32;48])->Checked<Zeroizing<Vec<u32>>> {
    let columns=words[1] as usize;
    let mut packed=Zeroizing::new(vec![0u32;source.len()]);
    for slot in 0..words[9] as usize {
        let first=words[18+slot*4] as usize;
        let padded=(words[17+slot*4] as usize).div_ceil(32)*32;
        for row in 0..padded {for col in (0..columns).step_by(2) {
            let src=(first+row)*columns/2+col/2;
            let dst=first*columns/2+packed_half(row,col,columns,16)/2;
            *packed.get_mut(dst).ok_or("packed pair input destination")?=
                *source.get(src).ok_or("packed pair input source")?;
        }}
    }
    Ok(packed)
}
fn pack_weight(source:&[u32],words:&[u32;48],down:bool)->Checked<Zeroizing<Vec<u32>>> {
    let columns=words[1] as usize;let rows=words[2] as usize;
    let mut packed=Zeroizing::new(vec![0u32;source.len()]);
    for slot in 0..words[9] as usize {
        let first=slot*4_915_200+if down {3_276_800}else{0};
        for row in 0..rows {for col in (0..columns).step_by(2) {
            let src=(first+row*columns+col)/2;
            let dst=(first+packed_half(row,col,columns,8))/2;
            *packed.get_mut(dst).ok_or("packed pair weight destination")?=
                *source.get(src).ok_or("packed pair weight source")?;
        }}
    }
    Ok(packed)
}
// Reuse the now-completed reference input owner as a bounded upload packet.
// Directly rewriting the75MiB image would reserve a128MiB staging allocation;
// a subsequent readback/audit can exhaust the unchanged transient-I/O budget.
fn staging_windows(words:usize,packet:usize)->Checked<impl Iterator<Item=(usize,usize)>> {
    if words==0 || words>128*1024*1024/4 || packet==0 || packet>20*1024*1024/4 {
        return Err("packed pair bounded staging geometry".into());
    }
    Ok((0..words).step_by(packet).map(move|first|(first,packet.min(words-first))))
}
fn overwrite(client:&ComputeClient<WgpuRuntime>,arena:&PrivateArena,id:BufferId,data:&[u32],staging:BufferId)->Checked<()> {
    let layout=arena.layout(id)?;let packet=arena.layout(staging)?;
    let target=arena.handle(id)?.clone();let source=arena.handle(staging)?.clone();
    if id==staging || data.len()!=layout.logical_words || layout.owned_bytes>128*1024*1024
        || source.stream!=target.stream
        || source.clone().binding().memory.allocation_id()==target.clone().binding().memory.allocation_id()
        || [&source,&target].iter().any(|h|h.offset_start.unwrap_or(0)!=0 || h.offset_end.unwrap_or(0)!=0 || h.size()!=h.size_in_used())
        || source.size_in_used()!=packet.owned_bytes as u64
        || target.size_in_used()!=layout.owned_bytes as u64 {
        return Err("packed pair retained disjoint bounded source owners".into());
    }
    let windows=staging_windows(data.len(),packet.owned_words)?;
    let mut upload=Zeroizing::new(vec![0u32;packet.owned_words]);
    for (first,n) in windows {
        // Every upload initializes the FULL old reference owner, including the
        // final packet's zero tail. Its original contents were already verified
        // after every original per-expert reference before intentional reuse.
        upload.fill(0);upload[..n].copy_from_slice(&data[first..first+n]);
        client.write_private(source.clone(),upload.iter().copied().flat_map(u32::to_le_bytes).collect())
            .map_err(|cause|device_fault("packed_packet_write",cause))?;
        let mut copy=[0u32;16];copy[0]=1;copy[1]=u32::try_from(n)?;copy[6]=u32::try_from(first)?;
        let args=cubecl_runtime::server::KernelArguments::new()
            .with_buffers(vec![source.clone().binding(),target.clone().binding()])
            .with_info(cubecl_runtime::server::MetadataBindingInfo::custom(
                copy.chunks_exact(2).map(|p|u64::from(p[0])|(u64::from(p[1])<<32)).collect()));
        // SAFETY: same-stream original retained disjoint full owners; the
        // checked window lies inside both packet and target logical extents.
        unsafe {client.launch_unchecked(Box::new(crate::strata_prompt::Kernel::Copy),
            CubeCount::Static(n.div_ceil(256) as u32,1,1),args);}
        cubecl::future::block_on(client.sync()).map_err(|cause|device_fault("packed_packet_copy_sync",cause))?;
        // The copy completed before its packet upload/uniform stages are
        // erased and fully audited; no MAIN/source owner is released here.
        sanitize(arena)?;
    }
    crate::strata_prompt::layer_gather::probe_util::source(arena,id,data)
}
fn packed_global_input(
    arena:&mut PrivateArena,map:&[u32],words:&[u32;48],input:&[u32],src:BufferId,down:bool,
)->Checked<Option<(BufferId,Zeroizing<Vec<u32>>)>> {
    if down {return Ok(None);}
    let mut global=Zeroizing::new(vec![0u32;g::LAYER_X_BYTES as usize/4]);
    let id=arena.allocate_zeroed(global.len())?;
    let client=arena.client()?;
    for slot in 0..words[9] as usize {
        let group=words[16+slot*4] as usize;
        // Include all Pad32 gutters: packed real rows are interspersed with
        // zero rows in the final16-row tile, not a row-major real-row prefix.
        let n=(words[17+slot*4] as usize).div_ceil(32)*32*1280;
        let start=words[18+slot*4] as usize*1280;
        let dest=map[g::LAYER_PREFIX+group] as usize*1280;
        global[dest..dest+n].copy_from_slice(&input[start..start+n]);
        let mut copy=[0u32;16];copy[0]=1;copy[1]=n as u32;copy[4]=start as u32;copy[6]=dest as u32;
        let args=cubecl_runtime::server::KernelArguments::new()
            .with_buffers(vec![u::handle(arena,src)?.binding(),u::handle(arena,id)?.binding()])
            .with_info(cubecl_runtime::server::MetadataBindingInfo::custom(
                copy.chunks_exact(2).map(|p|u64::from(p[0])|(u64::from(p[1])<<32)).collect()));
        // SAFETY: retained full owners and disjoint checked Pad32 group ranges.
        unsafe {client.launch_unchecked(Box::new(crate::strata_prompt::Kernel::Copy),
            CubeCount::Static(n.div_ceil(256) as u32,1,1),args);}
    }
    cubecl::future::block_on(client.sync()).map_err(|cause|device_fault("packed_global_copy_sync",cause))?;
    crate::strata_prompt::layer_gather::probe_util::source(arena,id,&global)?;
    Ok(Some((id,global)))
}

pub(crate) fn cases(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena)->Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count()!=0 {
        return Err("packed matrix original empty audit arena required".into());
    }
    let mut reports=Vec::new();
    for counts in [&[256usize,257,264,265,287,288,383,511][..],&[512;8][..],&[4095][..],&[4096][..]] {
        for down in [false,true] {for poison in [false,true] {
            let result=one(client,arena,down,poison,counts);
            match result {Ok(report)=>{arena.reset()?;reports.push(report);},
                Err(error)=>{let _=arena.reset();return Err(format!("packed large matrix down={down} poison={poison}: {error}").into());}}
        }}
    }
    Ok(reports)
}

#[cfg(feature="strata-large-shared-panels")]
pub(crate) fn shared_cases(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena)->Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count()!=0 {
        return Err("shared matrix original empty audit arena required".into());
    }
    let mut reports=Vec::new();
    for counts in [&[256usize,257,264,265,287,288,383,511][..],&[512;8][..],&[4095][..],&[4096][..]] {
        for down in [false,true] {for poison in [false,true] {
            let result=one_mode(client,arena,down,poison,counts,true,None);
            match result {Ok(report)=>{arena.reset()?;reports.push(report);},
                Err(error)=>{let _=arena.reset();return Err(format!("shared large matrix down={down} poison={poison}: {error}").into());}}
        }}
    }
    // Invalid8-member halves in near/far SG member partitions. Unaffected SGs
    // must follow the WG-uniform fallback while preserving raw valid halves.
    for down in [false,true] {for invalid in [0usize,8,96,104] {
        let result=one_mode(client,arena,down,false,&[256],true,Some(invalid));
        match result {Ok(report)=>{arena.reset()?;reports.push(report);},
            Err(error)=>{let _=arena.reset();return Err(format!("shared large mixed route down={down} invalid={invalid}: {error}").into());}}
    }}
    if reports.len()!=24 {return Err("shared large exact24 actual controls".into());}
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_pair_staging_keeps_original_owner_and_transient_bounds() {
        for packet in [5*1024*1024/4,20*1024*1024/4] {
            for words in [1,packet-1,packet,packet+1,75*1024*1024/4,128*1024*1024/4] {
                let mut end=0;
                for (first,n) in staging_windows(words,packet).unwrap() {
                    assert_eq!(first,end);assert!(n>0 && n<=packet);end+=n;
                    assert!(end<=words);
                    // The actual ledger rounds the FULL packet to a power of
                    // two and reserves another full audit copy. Even with four
                    // original4096-byte INFO pages, both packets fit128MiB.
                    let stage=(packet*4).next_power_of_two();
                    assert!(2*stage+4*4096<128*1024*1024);
                    let mut full=vec![0u32;packet];full[..n].fill(7);
                    assert!(full[n..].iter().all(|&v|v==0));
                }
                assert_eq!(end,words);
            }
        }
        for (words,packet) in [(0,1),(1,0),(128*1024*1024/4+1,1),(1,20*1024*1024/4+1)] {
            assert!(staging_windows(words,packet).is_err());
        }
    }
    #[test]
    fn packed_pair_host_input_includes_gutters_and_nonzero_group_prefix() {
        for columns in [32usize,640,2560] {
            let mut words=[0u32;48];words[1]=columns as u32;words[9]=2;
            words[17]=257;words[18]=0;words[21]=265;words[22]=288;
            let mut source=vec![0u32;576*columns/2];
            for (at,v) in source.iter_mut().enumerate() {*v=at as u32+1;}
            // Distinctive zero gutters are part of the complete owned mapping.
            source[257*columns/2..288*columns/2].fill(0);
            source[(288+265)*columns/2..].fill(0);
            let original=source.clone();let packed=pack_input(&source,&words).unwrap();
            let mut seen=vec![false;packed.len()];
            for (first,members) in [(0usize,257usize),(288,265)] {
                for row in 0..members.div_ceil(32)*32 {for col in (0..columns).step_by(2) {
                    let source_at=(first+row)*columns/2+col/2;
                    let dest=first*columns/2+packed_half(row,col,columns,16)/2;
                    assert!(!seen[dest]);seen[dest]=true;assert_eq!(packed[dest],source[source_at]);
                }}
            }
            assert!(seen.into_iter().all(|v|v));assert_eq!(source,original);
        }
    }
    #[test]
    fn packed_pair_weight_role_parent_offsets_are_outside_tiles() {
        for down in [false,true] {
            let mut words=[0u32;48];words[1]=32;words[2]=16;words[9]=2;
            let mut source=vec![0u32;2*4_915_200/2];
            for slot in 0..2 {let first=slot*4_915_200+if down {3_276_800}else{0};
                for row in 0..16 {for col in (0..32).step_by(2) {
                    source[(first+row*32+col)/2]=(1+slot*1024+row*32+col) as u32;
                }}
            }
            let packed=pack_weight(&source,&words,down).unwrap();
            let mut checked=0;
            for slot in 0..2 {let first=slot*4_915_200+if down {3_276_800}else{0};
                for row in 0..16 {for col in (0..32).step_by(2) {
                    assert_eq!(packed[(first+packed_half(row,col,32,8))/2],source[(first+row*32+col)/2]);checked+=1;
                }}
            }
            assert_eq!(checked,512);
            assert_eq!(source.iter().filter(|&&x|x!=0).count(),512);
            assert_eq!(packed.iter().filter(|&&x|x!=0).count(),512);
        }
    }
}
