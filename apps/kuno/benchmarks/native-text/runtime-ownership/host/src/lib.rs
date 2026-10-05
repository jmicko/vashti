//! Compile-only isolation harness. No adapter/device actions.
extern crate alloc;
pub use cubecl_runtime::{allocator, server};
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-wgpu/src/compute/cleanup_boundary.rs"]
mod cleanup_boundary;
pub use cubecl_runtime::transient::TransientIoReceipt;
pub use cubecl_wgpu::WgpuResource;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-runtime/src/transient.rs"]
mod transient_host;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-wgpu/src/compute/transient_io.rs"]
mod transient_io;
#[cfg(test)]
pub use cubecl_core::{KernelSettings, Runtime, ScalarArgType, prelude};
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-core/src/codegen/info.rs"]
mod core_info;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-core/src/codegen/metadata.rs"]
mod core_metadata;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-core/src/codegen/scalars.rs"]
mod core_scalars;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-core/src/private.rs"]
mod private;
#[cfg(test)]
use core_info::INFO_ALIGN;
#[cfg(test)]
pub use core_info::InfoBuilder;
#[cfg(test)]
pub use core_metadata::{Metadata, MetadataBuilder};
#[cfg(test)]
pub use core_scalars::ScalarBuilder;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../../vendor/cubecl-core/src/compute/private_info.rs"]
mod core_private_info;
