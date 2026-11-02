//! Identity of the Vulkan driver the worker actually opened. Kuno's Mesa build
//! (bundled next to the worker) reports `kuno-xe2` in its driver info and
//! always enables Xe2 2D block loads for 16 x 16 cooperative-matrix shaders,
//! so the direct GEMM/expert kernels are selected exactly when it is present.
use std::sync::OnceLock;

static KUNO_MESA: OnceLock<bool> = OnceLock::new();

pub(crate) fn record(info: &wgpu::AdapterInfo) {
    let _ = KUNO_MESA.set(info.driver_info.contains("kuno-xe2"));
}

/// Xe2 2D block loads are available (Kuno's Mesa build).
pub(crate) fn block2d() -> bool {
    KUNO_MESA.get().copied().unwrap_or(false)
}

/// Bundled Mesa ICD next to the worker executable, if installed.
pub(crate) fn bundled_icd() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let icd = exe
        .parent()?
        .join("kuno-mesa/share/vulkan/icd.d/intel_icd.x86_64.json");
    icd.is_file().then_some(icd)
}
