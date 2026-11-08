//! Process-isolated public logger control; no adapter, private data or file log.
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel};
use cubecl_runtime::logging::{ProfileLevel, ServerLogger};

#[test]
fn pausing_a_configured_profiler_preserves_the_original_level() {
    let mut config = CubeClRuntimeConfig::default();
    config.profiling.logger.level = ProfilingLogLevel::Basic;
    CubeClRuntimeConfig::set(config);
    let logger = ServerLogger::default();
    assert!(matches!(logger.profile_level(), Some(ProfileLevel::Basic)));
    logger.set_profiling_paused(true);
    assert!(logger.profile_level().is_none());
    logger.set_profiling_paused(false);
    assert!(matches!(logger.profile_level(), Some(ProfileLevel::Basic)));
}
