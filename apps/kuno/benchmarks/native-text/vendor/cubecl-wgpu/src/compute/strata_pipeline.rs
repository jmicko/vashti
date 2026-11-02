//! Typed separation of ordinary executables and sealed native admission.
//! No request bindings, INFO, allocation owners or completion credit live here.
use super::{strata_graph::Program, transient_io::fault};
use cubecl_runtime::server::ServerError;
use std::sync::Arc;

#[derive(Debug)]
pub enum BackendPipeline {
    Ordinary(wgpu::ComputePipeline),
    #[cfg(feature = "kuno-native-prepared")]
    Native(NativePrepared),
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PipelineKey {
    Ordinary(wgpu::ComputePipeline),
    #[cfg(feature = "kuno-native-prepared")]
    Prepared(u64),
}
#[cfg(feature = "kuno-native-prepared")]
#[derive(Debug)]
pub struct NativePrepared {
    id: u64,
    admission: Arc<wgpu::KunoPreparedComputePipeline>,
    program: Arc<Program>,
}
#[cfg(feature = "kuno-native-prepared")]
#[path = "strata_pipeline_identity.rs"]
mod identity;
#[cfg(feature = "kuno-native-prepared")]
static NEXT_PROGRAM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
#[cfg(feature = "kuno-native-prepared")]
static TOTALS: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Content-free process counts: prepared admissions, ordinary executables,
/// created native-key graph programs, created original-key graph programs. Read only by compiled-public probes.
/// [server misses, miss wall ns, native graph creations, native wall ns].
pub static CLAUDE_PIPELINE_TIME: [std::sync::atomic::AtomicU64; 18] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 18];
pub fn claude_pipeline_time() -> [u64; 18] {
    std::array::from_fn(|i| CLAUDE_PIPELINE_TIME[i].load(std::sync::atomic::Ordering::Relaxed))
}
pub(crate) struct ClaudeTimer(pub(crate) std::time::Instant, pub(crate) usize);
impl Drop for ClaudeTimer {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        CLAUDE_PIPELINE_TIME[self.1].fetch_add(1, Relaxed);
        CLAUDE_PIPELINE_TIME[self.1 + 1].fetch_add(self.0.elapsed().as_nanos() as u64, Relaxed);
    }
}
#[cfg(feature = "kuno-native-prepared")]
pub fn native_strata_prepared_pipeline_stats() -> [u64; 4] {
    std::array::from_fn(|index| TOTALS[index].load(std::sync::atomic::Ordering::Relaxed))
}
impl BackendPipeline {
    pub(crate) fn ordinary(pipeline: wgpu::ComputePipeline) -> Self {
        #[cfg(feature = "kuno-native-prepared")]
        TOTALS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self::Ordinary(pipeline)
    }
    pub(crate) fn key(&self) -> PipelineKey {
        match self {
            Self::Ordinary(pipeline) => PipelineKey::Ordinary(pipeline.clone()),
            #[cfg(feature = "kuno-native-prepared")]
            Self::Native(value) => PipelineKey::Prepared(value.id),
        }
    }
    #[cfg(feature = "kuno-native-prepared")]
    pub(crate) fn admission_owner(&self) -> Option<Arc<wgpu::KunoPreparedComputePipeline>> {
        match self {
            Self::Ordinary(_) => None,
            Self::Native(value) => Some(Arc::clone(&value.admission)),
        }
    }
    pub(crate) fn ordinary_pipeline(&self) -> Result<&wgpu::ComputePipeline, ServerError> {
        match self {
            Self::Ordinary(pipeline) => Ok(pipeline),
            #[cfg(feature = "kuno-native-prepared")]
            Self::Native(_) => Err(fault("native preparation cannot enter ordinary dispatch")),
        }
    }
    pub(crate) fn native_program(
        &self,
        device: &wgpu::Device,
    ) -> Result<Arc<Program>, ServerError> {
        let _ = device; // The original unselected key path has no prepared token.
        match self {
            Self::Ordinary(pipeline) => super::strata_graph::original_program(pipeline),
            #[cfg(feature = "kuno-native-prepared")]
            Self::Native(value) => {
                if !value.admission.is_for_device(device) {
                    return Err(fault("native pipeline original device mismatch"));
                }
                Ok(Arc::clone(&value.program))
            }
        }
    }
    #[cfg(feature = "kuno-native-prepared")]
    pub(crate) fn prepared(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        admission: wgpu::KunoPreparedComputePipeline,
    ) -> Result<Self, ServerError> {
        if !super::strata_graph::enabled()
            || !admission.is_for_device(device)
            || !admission.is_for_module(module)
        {
            return Err(fault("native pipeline exact original admission required"));
        }
        let id = identity::next(&NEXT_PROGRAM).map_err(fault)?;
        let program = super::strata_graph::take_prepared_program(module)?;
        TOTALS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self::Native(NativePrepared {
            id,
            admission: Arc::new(admission),
            program,
        }))
    }
}
#[cfg(feature = "kuno-native-prepared")]
pub(crate) fn graph_program_created(pipeline: &BackendPipeline) {
    let index = match pipeline {
        BackendPipeline::Ordinary(_) => 3,
        BackendPipeline::Native(_) => 2,
    };
    TOTALS[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
