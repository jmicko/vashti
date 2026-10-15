//! Run the actual vendor branch-layout tests without building a second GPU
//! runtime workspace. Only its error-construction boundary is substituted.
#![allow(dead_code)]
mod transient_io {
    pub(crate) fn fault(reason: &'static str) -> cubecl_runtime::server::ServerError {
        cubecl_runtime::server::ServerError::Generic {
            reason: reason.into(),
            backtrace: Default::default(),
        }
    }
}
#[path = "../vendor/cubecl-wgpu/src/compute/strata_parameters.rs"]
mod parameters;
