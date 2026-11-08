//! Opt-in consecutive scratch clears. Final owning erasure is unchanged.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::{ComputeClient, CubeCount, CubeDim},
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, KernelArguments},
};
use kuno_text::{Error, Result};
use std::sync::{Arc, atomic::AtomicBool};
use zeroize::Zeroize;
const MAX_BYTES: usize = 64 * 1024 * 1024;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Kernel(u8);
impl Kernel {
    fn source(self) -> String {
        let mut s = String::new();
        for i in 0..self.0 {
            s.push_str(&format!(
                "@group(0) @binding({i}) var<storage,read_write> owner{i}:array<u32>;\n"
            ));
        }
        s.push_str("@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>) {let index=id.x+id.y*groups.x*128u;\n");
        for i in 0..self.0 {
            s.push_str(&format!(
                "if index<arrayLength(&owner{i}) {{owner{i}[index]=0u;}}\n"
            ));
        }
        s.push('}');
        s
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "GroupedFullScratchClear"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(128))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
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
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(128),
            debug_info: None,
        })
    }
}
fn grid(words: usize) -> Result<CubeCount> {
    if words == 0 || words > MAX_BYTES / 4 {
        return Err(Error::InvalidBufferLength);
    }
    let groups = words.div_ceil(128);
    let x = groups.min(65535);
    let y = groups.div_ceil(x);
    Ok(CubeCount::Static(x as u32, y as u32, 1))
}
pub(crate) fn launch(arena: &PrivateArena, ids: &[BufferId]) -> Result<()> {
    if !(2..=4).contains(&ids.len()) {
        return Err(Error::InvalidBufferLength);
    }
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(ids.len())
        .map_err(|_| Error::LimitExceeded("grouped scratch handles"))?;
    let mut total = 0usize;
    let mut max_words = 0usize;
    for (i, &id) in ids.iter().enumerate() {
        if ids[..i].contains(&id) {
            return Err(Error::InvalidBufferLength);
        }
        let layout = arena.layout(id).map_err(u::fault)?;
        let h = arena.handle(id).map_err(u::fault)?;
        total = total
            .checked_add(layout.owned_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        if total > MAX_BYTES
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != layout.owned_bytes as u64
            || h.size_in_used() != layout.owned_bytes as u64
            || handles
                .first()
                .is_some_and(|first: &cubecl_runtime::server::Handle| first.stream != h.stream)
        {
            return Err(Error::InvalidBufferLength);
        }
        max_words = max_words.max(layout.owned_words);
        handles.push(h.clone());
    }
    let handle_bytes = handles
        .capacity()
        .checked_mul(std::mem::size_of::<cubecl_runtime::server::Handle>())
        .ok_or(Error::ArithmeticOverflow)?;
    if handle_bytes > 4096 {
        return Err(Error::LimitExceeded("grouped scratch metadata"));
    }
    let client = arena.client().map_err(u::fault)?;
    let lookup = handles.clone();
    let lookup_bytes = lookup
        .capacity()
        .checked_mul(std::mem::size_of::<cubecl_runtime::server::Handle>())
        .ok_or(Error::ArithmeticOverflow)?;
    if lookup_bytes + handle_bytes > 4096 {
        return Err(Error::LimitExceeded("grouped scratch metadata"));
    }
    let metadata = u::resources(client, lookup)?;
    if metadata.len() != ids.len()
        || metadata
            .capacity()
            .checked_mul(std::mem::size_of_val(&metadata[0]))
            .and_then(|n| n.checked_add(handle_bytes + lookup_bytes))
            .is_none_or(|n| n > 4096)
    {
        return Err(Error::LimitExceeded("grouped scratch metadata"));
    }
    for (i, r) in metadata.iter().enumerate() {
        let r = r.resource();
        let n = arena.layout(ids[i]).map_err(u::fault)?.owned_bytes as u64;
        if r.offset != 0
            || r.size != n
            || r.buffer.size() != n
            || metadata[..i]
                .iter()
                .any(|old| old.resource().buffer == r.buffer)
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let mut args = KernelArguments::new();
    for handle in handles {
        args = args.with_buffer(handle.binding());
    }
    // SAFETY: every bound span is a distinct complete live same-stream owner;
    // all bytes including padding are covered. Arena retains them through the
    // original completion/erasure boundary. No scalar INFO or new owner exists.
    unsafe {
        client.launch_unchecked(Box::new(Kernel(ids.len() as u8)), grid(max_words)?, args);
    }
    Ok(())
}
struct Scrub(cubecl::bytes::Bytes);
impl Drop for Scrub {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
pub(crate) fn probe(client: &ComputeClient<WgpuRuntime>, retired: Arc<AtomicBool>) -> Result<()> {
    let mut arena = PrivateArena::new_transient_on_configured(client.clone(), 1024 * 1024, retired)
        .map_err(u::fault)?;
    let ids = [1, 8193, 32768, 81920]
        .map(|n| arena.allocate_zeroed(n).map_err(u::fault))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    if launch(&arena, &[ids[0], ids[0]]).is_ok() {
        return Err(Error::InvalidBufferLength);
    }
    for count in 2..=4 {
        for &id in &ids {
            let n = arena.layout(id).map_err(u::fault)?.owned_bytes;
            client
                .write_private(arena.handle(id).map_err(u::fault)?.clone(), vec![0xa5; n])
                .map_err(u::fault)?;
        }
        launch(&arena, &ids[..count])?;
        for (i, &id) in ids.iter().enumerate() {
            let bytes = Scrub(
                client
                    .read_one(arena.handle(id).map_err(u::fault)?.clone())
                    .map_err(u::fault)?,
            );
            let n = arena.layout(id).map_err(u::fault)?.owned_bytes;
            if bytes.0.len() != n
                || (i < count && !cubecl_runtime::transient::full_span_is_zero(&bytes.0))
                || (i >= count && bytes.0.iter().any(|b| *b != 0xa5))
            {
                return Err(Error::InvalidFormat("grouped scratch full-owner probe"));
            }
        }
        cubecl::future::block_on(client.sanitize_transient_io_audited()).map_err(u::fault)?;
    }
    arena.close().map_err(u::fault)?;
    println!(
        "public_compute_scratch_clear_full_owners=2,3,4 unselected_preserved=true padding=true"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_owning_grid_covers_first_last_and_two_dimensional_tails() {
        for words in [1, 8192, 8193, 65535 * 128, 65535 * 128 + 1, MAX_BYTES / 4] {
            let CubeCount::Static(x, y, z) = grid(words).unwrap() else {
                panic!()
            };
            assert!(x <= 65535 && y <= 65535 && z == 1);
            assert!(x as usize * y as usize * 128 >= words);
            assert!((x as usize * (y as usize - 1)) * 128 < words);
        }
        for words in [0, MAX_BYTES / 4 + 1, usize::MAX] {
            assert!(grid(words).is_err());
        }
    }
    #[test]
    fn static_public_clear_shaders_have_no_uniform_or_private_key() {
        for count in 2..=4 {
            let s = Kernel(count).source();
            let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(s.matches("@binding(").count(), count as usize);
            assert!(!s.contains("uniform"));
            assert_eq!(s.matches("=0u;").count(), count as usize);
        }
    }
}
