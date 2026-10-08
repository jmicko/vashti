//! Opt-in status folds at existing ready-before-reuse boundaries, no acceptance.
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
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
use kuno_text::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
static SELECTED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) fn select(selected: bool) {
    SELECTED.store(selected, Ordering::Release);
}
pub(crate) fn selected() -> bool {
    SELECTED.load(Ordering::Acquire)
}
pub(crate) fn calls() -> u64 {
    CALLS.load(Ordering::Relaxed)
}
pub(crate) struct View {
    region: BufferId,
    span: Zeroizing<[usize; 4]>,
}
impl View {
    pub(crate) fn new(
        region: BufferId,
        count: usize,
        first: usize,
        stride: usize,
        width: usize,
    ) -> Self {
        Self {
            region,
            span: Zeroizing::new([count, first, stride, width]),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Kernel(u8);
impl Kernel {
    fn source(self) -> String {
        let mut s = String::from("struct Params {count:u32,first:u32,stride:u32,width:u32}\n");
        for i in 0..self.0 {
            s.push_str(&format!(
                "@group(0) @binding({i}) var<storage,read> status{i}:array<u32>;\n"
            ));
        }
        s.push_str(&format!("@group(0) @binding({}) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding({}) var<uniform> params:array<Params,{}>;\n",self.0,self.0+1,self.0));
        s.push_str(
            "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {\n",
        );
        for i in 0..self.0 {
            s.push_str(&format!("if id.y<params[{i}].count && id.x<params[{i}].width {{if status{i}[params[{i}].first+id.y*params[{i}].stride+id.x]!=1u {{atomicOr(&fault[0],2u);}}}}\n"));
        }
        s.push('}');
        s
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "GroupedActiveStatusFold"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(64))
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
            cube_dim: CubeDim::new_1d(64),
            debug_info: None,
        })
    }
}
pub(crate) fn launch(arena: &PrivateArena, views: &[View], fault: BufferId) -> Result<()> {
    if !matches!(views.len(), 3 | 5 | 8) || u::words(arena, fault)? != 1 {
        return Err(Error::InvalidBufferLength);
    }
    let client = arena.client().map_err(u::fault)?;
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(views.len() + 1)
        .map_err(|_| Error::LimitExceeded("grouped status handles"))?;
    let mut args = KernelArguments::new();
    let mut info = MetadataBindingInfo::custom(Vec::new());
    info.data
        .try_reserve_exact(views.len() * 2)
        .map_err(|_| Error::LimitExceeded("grouped status INFO"))?;
    let mut width_max = 0usize;
    let mut count_max = 0usize;
    for (i, v) in views.iter().enumerate() {
        if v.region == fault || views[..i].iter().any(|old| old.region == v.region) {
            return Err(Error::InvalidBufferLength);
        }
        let [count, first, stride, width] = *v.span;
        crate::resident_batch_projection::validate_view(
            count,
            first,
            stride,
            width,
            u::words(arena, v.region)?,
        )?;
        if [count, first, stride, width]
            .iter()
            .any(|&n| n > u32::MAX as usize)
        {
            return Err(Error::InvalidBufferLength);
        }
        handles.push(arena.handle(v.region).map_err(u::fault)?.clone());
        args = args.with_buffer(u::handle(arena, v.region)?.binding());
        info.data.push(count as u64 | ((first as u64) << 32));
        info.data.push(stride as u64 | ((width as u64) << 32));
        width_max = width_max.max(width);
        count_max = count_max.max(count);
    }
    handles.push(arena.handle(fault).map_err(u::fault)?.clone());
    args = args.with_buffer(u::handle(arena, fault)?.binding());
    let handle_bytes = handles
        .capacity()
        .checked_mul(std::mem::size_of::<Handle>())
        .ok_or(Error::ArithmeticOverflow)?;
    if handle_bytes > 4096 {
        return Err(Error::LimitExceeded("grouped status metadata"));
    }
    let stream = handles.last().unwrap().stream;
    for h in &handles {
        if h.stream != stream
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != h.size_in_used()
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let metadata = u::resources(client, handles)?;
    if metadata.len() != views.len() + 1 {
        return Err(Error::InvalidBufferLength);
    }
    let bytes = metadata
        .capacity()
        .checked_mul(std::mem::size_of_val(&metadata[0]))
        .and_then(|n| n.checked_add(handle_bytes))
        .and_then(|n| {
            n.checked_add(args.buffers.capacity() * std::mem::size_of_val(&args.buffers[0]))
        })
        .and_then(|n| n.checked_add(info.data.capacity() * 8 + std::mem::size_of_val(views)))
        .ok_or(Error::ArithmeticOverflow)?;
    if bytes > 4096 {
        return Err(Error::LimitExceeded("grouped status metadata"));
    }
    for (i, r) in metadata.iter().enumerate() {
        let r = r.resource();
        let region = if i == views.len() {
            fault
        } else {
            views[i].region
        };
        let n = arena.layout(region).map_err(u::fault)?.owned_bytes as u64;
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
    let groups = width_max.div_ceil(64);
    if groups == 0 || groups > 65535 {
        return Err(Error::InvalidBufferLength);
    }
    // SAFETY: same validated logical spans and distinct actual full owners as
    // the original folds, ready before reuse/fence. Only fault bit2 is written.
    // Private runtime coordinates remain in the original wiping INFO ledger.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel(views.len() as u8)),
            CubeCount::Static(groups as u32, count_max as u32, 1),
            args.with_info(info),
        );
    }
    CALLS
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::ArithmeticOverflow)?;
    Ok(())
}
struct Scrub(cubecl::bytes::Bytes);
impl Drop for Scrub {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
// Fixed-public probe upload; ownership transfers immediately to write_private's
// wiping guard. The source words remain separately Zeroizing.
fn word_bytes(words: &[u32]) -> Result<Vec<u8>> {
    let n = words
        .len()
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)?;
    let mut bytes = Zeroizing::new(vec![0u8; n]);
    for (dst, word) in bytes.chunks_exact_mut(4).zip(words) {
        dst.copy_from_slice(&word.to_le_bytes());
    }
    Ok(std::mem::take(&mut *bytes))
}
pub(crate) fn probe(client: &ComputeClient<WgpuRuntime>, retired: Arc<AtomicBool>) -> Result<()> {
    let mut a = PrivateArena::new_transient_on_configured(client.clone(), 512 * 1024, retired)
        .map_err(u::fault)?;
    let widths = [2, 3, 5, 65, 257, 4, 17, 129];
    let ids = widths
        .map(|n| a.allocate_zeroed(8 * n).map_err(u::fault))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let fault = a.allocate_zeroed(1).map_err(u::fault)?;
    let duplicate = [ids[0], ids[0], ids[1]].map(|id| View::new(id, 1, 0, 1, 1));
    if launch(&a, &duplicate, fault).is_ok() {
        return Err(Error::InvalidBufferLength);
    }
    for members in [3, 5, 8] {
        for count in [1, 3, 8] {
            for case in 0..5 {
                let views: Vec<_> = ids[..members]
                    .iter()
                    .zip(&widths[..members])
                    .map(|(&id, &width)| View::new(id, count, 0, width, width))
                    .collect();
                for (i, &id) in ids[..members].iter().enumerate() {
                    let n = a.layout(id).map_err(u::fault)?.owned_words;
                    let mut words = Zeroizing::new(vec![9u32; n]);
                    words[..count * widths[i]].fill(1);
                    if (case == 1 || case == 4) && i == members - 1 {
                        words[count * widths[i] - 1] = 0;
                    }
                    if case == 2 && i == 0 {
                        words[0] = 0;
                    }
                    client
                        .write_private(a.handle(id).map_err(u::fault)?.clone(), word_bytes(&words)?)
                        .map_err(u::fault)?;
                }
                let mut f =
                    Zeroizing::new(vec![0u32; a.layout(fault).map_err(u::fault)?.owned_words]);
                if case == 3 || case == 4 {
                    f[0] = 8;
                }
                client
                    .write_private(a.handle(fault).map_err(u::fault)?.clone(), word_bytes(&f)?)
                    .map_err(u::fault)?;
                launch(&a, &views, fault)?;
                let bytes = Scrub(client.read_one(u::handle(&a, fault)?).map_err(u::fault)?);
                let expected = if case == 4 {
                    10
                } else if case == 1 || case == 2 {
                    2
                } else if case == 3 {
                    8
                } else {
                    0
                };
                if bytes.0.len() != 4
                    || u32::from_le_bytes(
                        bytes.0[..]
                            .try_into()
                            .map_err(|_| Error::InvalidBufferLength)?,
                    ) != expected
                {
                    return Err(Error::InvalidFormat("grouped status active/sticky probe"));
                }
                drop(bytes);
                cubecl::future::block_on(client.sanitize_transient_io_audited())
                    .map_err(u::fault)?;
            }
        }
    }
    a.close().map_err(u::fault)?;
    println!(
        "public_grouped_status_fold groups=3,5,8 counts=1,3,8 active_and_sticky=true inactive_ignored=true"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_active_status_predicates_and_uniform_layout_validate_offline() {
        for n in [3, 5, 8] {
            let s = Kernel(n).source();
            let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(s.matches("atomicOr(&fault[0],2u)").count(), n as usize);
            assert_eq!(s.matches("id.y<params[").count(), n as usize);
            assert!(!s.contains("status0[0]="));
        }
    }
}
