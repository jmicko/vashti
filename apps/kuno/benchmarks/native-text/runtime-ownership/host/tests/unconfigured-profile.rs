//! Isolated process: disabled configuration stays disabled through pause toggles.
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl_runtime::logging::ServerLogger;

#[test]
fn toggling_cannot_enable_an_unconfigured_profiler() {
    CubeClRuntimeConfig::set(CubeClRuntimeConfig::default());
    let logger = ServerLogger::default();
    for paused in [false, true, false] {
        logger.set_profiling_paused(paused);
        assert!(logger.profile_level().is_none());
    }
}
