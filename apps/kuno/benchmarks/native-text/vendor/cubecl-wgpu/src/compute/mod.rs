// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
pub(crate) mod controller;

mod storage;

mod cleanup_boundary;
pub(super) mod mem_manager;
pub(super) mod poll;
pub(super) mod schedule;
mod server;
pub(super) mod stream;
pub(super) mod timings;
mod transient_io;

pub use server::*;
pub use storage::*;

pub use transient_io::transient_gpu_upload_clears;
pub use transient_io::transient_inline_readback_batches;
pub use transient_io::transient_readback_reuses;
