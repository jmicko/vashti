//! Fixed-public controls only: authenticated global-X layout and full source check.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
    strata_prompt::groups as g,
};
use zeroize::Zeroizing;
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) fn input(
    arena: &mut PrivateArena,
    map: &[u32],
    words: &[u32; 48],
    packed: &[u32],
    packed_owner: BufferId,
    down: bool,
) -> Checked<Option<(BufferId, Zeroizing<Vec<u32>>)>> {
    if down {
        return Ok(None);
    }
    let mut global = Zeroizing::new(vec![0u32; g::LAYER_X_BYTES as usize / 4]);
    for slot in 0..words[9] as usize {
        let group = words[16 + slot * 4] as usize;
        let n = words[17 + slot * 4] as usize;
        let src = words[18 + slot * 4] as usize * 1280;
        let dst = map[g::LAYER_PREFIX + group] as usize * 1280;
        global[dst..dst + n * 1280].copy_from_slice(&packed[src..src + n * 1280]);
    }
    // The layer owner exceeds the original128MiB transient-upload bound.
    // Reuse the already uploaded, retained local input and original bit-copy
    // kernel. No oversized staging owner, additional MAIN or public upload.
    let id = arena.allocate_zeroed(global.len())?;
    let client = arena.client()?;
    if arena.layout(packed_owner)?.logical_words != packed.len() {
        return Err("layer local input owner extent".into());
    }
    for slot in 0..words[9] as usize {
        let group = words[16 + slot * 4] as usize;
        let n = words[17 + slot * 4] as usize * 1280;
        let src = words[18 + slot * 4] as usize * 1280;
        let dst = map[g::LAYER_PREFIX + group] as usize * 1280;
        let mut copy = [0u32; 16];
        copy[0] = 1;
        copy[1] = n as u32;
        copy[4] = src as u32;
        copy[6] = dst as u32;
        let args = cubecl_runtime::server::KernelArguments::new()
            .with_buffers(vec![u::handle(arena, packed_owner)?.binding(),
                               u::handle(arena, id)?.binding()])
            .with_info(cubecl_runtime::server::MetadataBindingInfo::custom(
                copy.chunks_exact(2).map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32)).collect(),
            ));
        // SAFETY: exact initialized disjoint owners, original bit-copy kernel,
        // bounded original local interval and independently built global prefix.
        unsafe {
            client.launch_unchecked(Box::new(crate::strata_prompt::Kernel::Copy),
                cubecl::prelude::CubeCount::Static(n.div_ceil(256) as u32, 1, 1), args);
        }
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    source(arena, id, &global)?;
    Ok(Some((id, global)))
}
pub(crate) fn source(arena: &PrivateArena, id: BufferId, expected: &[u32]) -> Checked<()> {
    let owned = arena.layout(id)?.owned_words;
    for start in (0..owned).step_by(4 * 1024 * 1024) {
        let n = (4 * 1024 * 1024).min(owned - start);
        let view = arena
            .handle(id)?
            .clone()
            .offset_start((start * 4) as u64)
            .offset_end(((owned - start - n) * 4) as u64);
        let actual = u::read_handle_prefix_u32(arena, view, n)?;
        let same = actual
            .iter()
            .enumerate()
            .all(|(i, v)| *v == expected.get(start + i).copied().unwrap_or(0));
        drop(actual);
        let r = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
        let bytes = r.upload_bytes + r.uniform_bytes + r.readback_bytes;
        if bytes == 0 || r.inspected_bytes != bytes || r.inspected_buffers == 0 {
            return Err("layer public source erasure incomplete".into());
        }
        if !same {
            return Err("layer global input entire owner source changed".into());
        }
    }
    Ok(())
}
