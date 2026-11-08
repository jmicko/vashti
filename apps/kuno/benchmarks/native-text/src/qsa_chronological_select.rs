//! Explicit original-F32 chronological QSA selection below the token budget.
//!
//! Original indexer/score math remains queued. The first complete-owner sweep
//! validates the exact original score/status predicates, inactive rows and
//! padding; the second publishes only after its sticky model fault completes.
//! No CPU dependency map, score vector or selected-ID upload is required.

use crate::resident_util as u;
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

pub(crate) const OWNED_WORDS: [usize; 4] = [8192, 8192, 32768, 8192];
pub(crate) const SELECT_STRIDE: usize = 2051;
pub(crate) const MAX_VISIBLE: usize = 2048;
pub(crate) const WORKGROUPS: u32 = 128;
pub(crate) const INFO_CALLS_PER_LAYER: usize = 2;
pub(crate) const EXTRA_INFO_BYTES_PER_CHUNK: u64 = 12 * INFO_CALLS_PER_LAYER as u64 * 2048;
// No additional retained private/public MAIN owner or private payload is made.
// Four bounded handle/resource metadata entries and two16-byte INFO arguments
// fit this conservative temporary host reservation (before any enqueue).
pub(crate) const EXTRA_HOST_BYTES_PER_CHUNK: u64 = 4096;
pub(crate) const HOST_METADATA_BYTES: usize = EXTRA_HOST_BYTES_PER_CHUNK as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub(crate) start: usize,
    pub(crate) count: usize,
    pub(crate) blocks: usize,
}
impl Geometry {
    pub(crate) fn supported(self) -> Result<bool> {
        if !(1..=8).contains(&self.count)
            || !(1..=1024).contains(&self.blocks)
            || self
                .start
                .checked_add(self.count)
                .is_none_or(|end| end > 4096 || end / 4 > self.blocks)
        {
            return Err(Error::InvalidFormat("QSA chronological geometry rejected"));
        }
        Ok(self.start + self.count <= MAX_VISIBLE)
    }
    fn validate(self) -> Result<()> {
        if !self.supported()? {
            return Err(Error::InvalidFormat(
                "QSA chronological requires original CPU fallback",
            ));
        }
        Ok(())
    }
    fn info(self) -> MetadataBindingInfo {
        MetadataBindingInfo::custom(vec![
            self.start as u64 | ((self.count as u64) << 32),
            self.blocks as u64 | ((SELECT_STRIDE as u64) << 32),
        ])
    }
}

struct Preflight;
struct Publish;
macro_rules! kernel {
    ($ty:ty, $name:literal, $source:literal) => {
        impl KernelMetadata for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn id(&self) -> KernelId {
                KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256))
            }
            fn address_type(&self) -> StorageType {
                StorageType::Scalar(ElemType::UInt(UIntKind::U32))
            }
        }
        impl CubeTask<AutoCompiler> for $ty {
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
                    source: include_str!($source).into(),
                    repr: None,
                    cube_dim: CubeDim::new_1d(256),
                    debug_info: None,
                })
            }
        }
    };
}
kernel!(
    Preflight,
    "QsaChronologicalOwningPreflight",
    "qsa_chronological_preflight.wgsl"
);
kernel!(
    Publish,
    "QsaChronologicalOriginalIds",
    "qsa_chronological_publish.wgsl"
);

pub(crate) fn validate_limits(l: &wgpu::Limits) -> Result<()> {
    if l.max_compute_invocations_per_workgroup < 256
        || l.max_compute_workgroup_size_x < 256
        || l.max_storage_buffers_per_shader_stage < 4
        || l.max_bindings_per_bind_group < 5
        || l.max_bind_groups < 1
        || l.max_uniform_buffers_per_shader_stage < 1
        || l.max_uniform_buffer_binding_size < 16
        || l.max_compute_workgroups_per_dimension < WORKGROUPS
        || l.max_storage_buffer_binding_size < 131072
        || l.max_buffer_size < 131072
    {
        return Err(Error::LimitExceeded(
            "QSA chronological physical device limits",
        ));
    }
    Ok(())
}
fn validate_handles(owners: &[Handle; 4], geometry: Geometry) -> Result<()> {
    geometry.validate()?;
    for (owner, words) in owners.iter().zip(OWNED_WORDS) {
        if owner.stream != owners[0].stream
            || owner.offset_start.unwrap_or(0) != 0
            || owner.offset_end.unwrap_or(0) != 0
            || owner.size() != words as u64 * 4
            || owner.size_in_used() != owner.size()
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    Ok(())
}

/// Whole owners in order SCORE, scoreSTATUS, SELECT, actual model globalFault.
/// They remain charged/retained by the caller through original full-family
/// fault/finite/tail acceptance. Same-stream dispatch order is the capability;
/// enqueue is neither a completed proof nor permission to reclaim anything.
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 4],
    geometry: Geometry,
) -> Result<()> {
    validate_handles(&owners, geometry)?;
    let metadata = u::resources(client, owners.to_vec())?;
    if metadata.len() != 4
        || metadata
            .capacity()
            .checked_mul(std::mem::size_of_val(&metadata[0]))
            .and_then(|bytes| bytes.checked_add(8 * std::mem::size_of::<Handle>() + 32))
            .is_none_or(|bytes| bytes > HOST_METADATA_BYTES)
    {
        return Err(Error::LimitExceeded(
            "QSA chronological bounded owner metadata",
        ));
    }
    for (index, (resource, words)) in metadata.iter().zip(OWNED_WORDS).enumerate() {
        let page = resource.resource();
        if page.offset != 0
            || page.size != words as u64 * 4
            || page.buffer.size() != page.size
            || metadata[..index]
                .iter()
                .any(|old| old.resource().buffer == page.buffer)
        {
            return Err(Error::InvalidFormat(
                "QSA chronological full owner alias/extent",
            ));
        }
    }
    let mut preflight = KernelArguments::new();
    for owner in &owners {
        preflight = preflight.with_buffer(owner.clone().binding());
    }
    preflight = preflight.with_info(geometry.info());
    let publish = KernelArguments::new()
        .with_buffer(owners[2].clone().binding())
        .with_buffer(owners[3].clone().binding())
        .with_info(geometry.info());
    // SAFETY: fixed complete physically disjoint same-client owners, all counts
    // and bounds checked before either command. First dispatch can only raise
    // sticky fault. Second checks it before all writes; no interim clear/read.
    unsafe {
        client.launch_unchecked(
            Box::new(Preflight),
            CubeCount::Static(WORKGROUPS, 1, 1),
            preflight,
        );
        client.launch_unchecked(
            Box::new(Publish),
            CubeCount::Static(WORKGROUPS, 1, 1),
            publish,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "qsa_chronological_tests.rs"]
mod tests;
