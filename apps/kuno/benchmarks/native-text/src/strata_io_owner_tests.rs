//! Original private-I/O owner tests without a second vendor Cargo workspace.
//! Only native device activation is disabled. No queue/fence is simulated or
//! claimed: the actual I/O/index/eraser/parameter/cache sources run CPU tests.
#![allow(dead_code, unexpected_cfgs)]
use crate::{
    strata_info_transfer, strata_io_reuse, strata_readback_tests as strata_readback,
    strata_upload_tests as strata_upload,
};
mod strata_graph {
    pub(crate) fn enabled() -> bool {
        false
    }
}
#[path = "../vendor/cubecl-wgpu/src/compute/binding_cache.rs"]
mod binding_cache;
#[path = "../vendor/cubecl-wgpu/src/compute/strata_parameters.rs"]
mod strata_parameters;
#[path = "../vendor/cubecl-wgpu/src/compute/transient_io.rs"]
mod transient_io;
