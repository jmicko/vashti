//! Compare actual padded Gather/activation against their original GPU programs.
//! Fixed public fixtures only, exact real bits plus complete owner/tail scans.
use super::*;
use crate::{private_arena::PrivateArena, resident_util as u};
use zeroize::Zeroizing;
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug)]
struct Original(Op);
impl KernelMetadata for Original {
    fn name(&self) -> &'static str {
        "PublicOriginalNarrowProducer"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.0)
            .cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Original {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: self.0.source_unpadded(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
            debug_info: None,
        })
    }
}
fn read(arena: &PrivateArena, id: crate::private_arena::BufferId) -> Checked<Zeroizing<Vec<u32>>> {
    let root = arena.handle(id)?.clone();
    let words = arena.layout(id)?.owned_bytes / 4;
    Ok(u::read_handle_prefix_u32(arena, root, words)?)
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    count: usize,
    activate: bool,
) -> Checked<serde_json::Value> {
    let total=count.next_power_of_two().max(32);
    let mut raw = Zeroizing::new(vec![0u32; groups::LOGICAL + total * 10]);
    raw[..3].copy_from_slice(&[total as u32, 0, total as u32]);
    for token in 0..total {
        for rank in 0..10 {
            let route = token * 10 + rank;
            let chosen = token >= total - count && rank == token % 10;
            let logical = if chosen { 400 } else { 1 + rank };
            raw[groups::LOGICAL + route] = logical as u32;
            raw[1540 + route] = ((logical * 13 + 17) % 512) as u32;
        }
    }
    let map = groups::group(&raw, total)?;
    let n = groups::validate(&map, total)?;
    let group = n - 1;
    if map[516 + group] as usize != count {
        return Err("public narrow authentic map".into());
    }
    let tile = if count<=8 {8} else if count<32 {16} else {32};
    let padded = count.div_ceil(tile) * tile;
    let row_words = if activate { 320 } else { 1280 };
    let logical = padded * 1280 + 256;
    let mut source = Zeroizing::new(vec![
        0u32;
        if activate {
            groups::GU_BYTES as usize / 4
        } else {
            total * 2560
        }
    ]);
    for (i, x) in source.iter_mut().enumerate() {
        *x = (((i % 41) as f32 - 20.0) / 32.0).to_bits();
    }
    let poison = Zeroizing::new(vec![0x35553555u32; logical]);
    let status = Zeroizing::new(vec![0xa513579bu32; groups::STATUS_WORDS]);
    let map_id = arena.upload_u32(&map)?;
    let src = arena.upload_u32(&source)?;
    let original = arena.upload_u32(&poison)?;
    let selected = arena.upload_u32(&poison)?;
    let status0 = arena.upload_u32(&status)?;
    let status1 = arena.upload_u32(&status)?;
    let fault0 = arena.allocate_zeroed(1)?;
    let fault1 = arena.allocate_zeroed(1)?;
    let op = if activate {
        Op::ActivateCheckGu
    } else {
        Op::Gather
    };
    let mut words = [0u32; 16];
    words[0] = total as u32;
    words[1] = 2560;
    words[2] = 1280;
    words[4] = group as u32;
    let grid = member_dispatches(count, total)?[usize::from(activate)];
    for (output, status, fault, baseline) in [
        (original, status0, fault0, true),
        (selected, status1, fault1, false),
    ] {
        let mut ids = vec![map_id, src, output];
        if activate {
            ids.push(status);
        }
        ids.push(fault);
        let handles = ids
            .into_iter()
            .map(|id| u::handle(arena, id).map(Handle::binding))
            .collect::<Result<Vec<_>>>()?;
        let args =
            KernelArguments::new()
                .with_buffers(handles)
                .with_info(MetadataBindingInfo::custom(
                    words
                        .chunks_exact(2)
                        .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                        .collect(),
                ));
        // SAFETY: authentic fixed-public map and actual independent exclusive
        // owners; no fabricated production/request scope or batch is created.
        unsafe {
            if baseline {
                client.launch_unchecked(
                    Box::new(Original(op)),
                    CubeCount::Static(grid as u32, 1, 1),
                    args,
                );
            } else {
                client.launch_unchecked(Box::new(op), CubeCount::Static(grid as u32, 1, 1), args);
            }
        }
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let old = read(arena, original)?;
    let new = read(arena, selected)?;
    if old.len() != new.len() {
        return Err("public narrow full output extent".into());
    }
    for (at, (&a, &b)) in old.iter().zip(new.iter()).enumerate() {
        let expected = if (count * row_words..padded * row_words).contains(&at) {
            0
        } else {
            a
        };
        if b != expected {
            return Err("public narrow exact real and padded output".into());
        }
        if at >= count * row_words && a != if at < logical { 0x35553555 } else { 0 } {
            return Err("public narrow baseline owner tail".into());
        }
    }
    if *read(arena, status0)? != *read(arena, status1)? {
        return Err("public narrow padded rows changed status".into());
    }
    if read(arena, fault0)?
        .iter()
        .chain(read(arena, fault1)?.iter())
        .any(|&x| x != 0)
    {
        return Err("public narrow producer fault".into());
    }
    let mut checked = old.len();
    for (id, expected) in [(map_id, &*map), (src, &*source)] {
        let got = read(arena, id)?;
        checked += got.len();
        if got
            .iter()
            .enumerate()
            .any(|(at, &x)| x != expected.get(at).copied().unwrap_or(0))
        {
            return Err("public narrow unchanged source and full padding".into());
        }
    }
    let audit = cubecl::future::block_on(client.sanitize_transient_io_audited())?;
    let total = audit.upload_bytes + audit.uniform_bytes + audit.readback_bytes;
    if total == 0 || audit.inspected_bytes != total || audit.inspected_buffers == 0 {
        return Err("public narrow transient IO erase".into());
    }
    Ok(
        serde_json::json!({"count":count,"activate":activate,"actual_original_and_selected_gpu_producers":true,"original_map_builder_and_validator":true,"original_grid_and_real_params":true,"bit_mismatches":0,"real_rows_bit_exact":true,"every_padded_word_zero":true,"complete_status_fault_sources_and_owner_tail":true,"full_owned_words_checked":checked,"fabricated_prompt_or_batch_scope":false}),
    )
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("public narrow requires empty audit arena".into());
    }
    let mut out = Vec::new();
    for activate in [false, true] {
        for count in [1, 3, 7, 8, 9, 15, 16, 17, 31] {
            let result = one(client, arena, count, activate);
            arena.reset()?;
            let mut report = result?;
            report["reset_and_owned_erasure"] = true.into();
            out.push(report);
        }
    }
    Ok(out)
}

#[cfg(feature="strata-padded-global")]
pub(crate) fn medium_cases(client:&ComputeClient<WgpuRuntime>,arena:&mut PrivateArena)->Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count()!=0 {return Err("medium padding requires empty public arena".into());}
    let mut out=Vec::new();
    for activate in [false,true] {for count in [32,33,63,64,65,127,128,129,255] {
        let result=one(client,arena,count,activate);arena.reset()?;let mut r=result?;
        r["reset_and_owned_erasure"]=true.into();out.push(r);
    }}
    Ok(out)
}
