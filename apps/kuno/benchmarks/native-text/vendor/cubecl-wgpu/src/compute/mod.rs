// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
pub(crate) mod controller;

mod public_external;
mod storage;

mod strata_allocator_scope;
mod strata_draft_scope;
mod strata_flights;
pub(crate) mod strata_graph;
mod strata_handoff;
mod strata_info_transfer;
mod strata_io_reuse;
mod strata_parameters;
mod strata_physical;
mod strata_readback;
mod strata_schedule;
mod strata_upload;
pub use strata_graph::{
    arm_native_strata_public_rolling_failure, native_strata_graph_copy_barrier_stats,
    native_strata_graph_execution_stats, native_strata_graph_rolling_stats,
    native_strata_graph_stats, native_strata_operator_profile,
    native_strata_public_rolling_failure_stats, set_native_strata_profile_active,
};
pub use strata_physical::{native_expert_shader_words, preserve_native_product_rounding};
pub use strata_schedule::NATIVE_QUEUE_HOST_BYTES;
mod binding_cache;
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
pub use transient_io::transient_request_io_reuses;
pub use transient_io::transient_request_reuse_metadata_charge;

pub use stream::transient_gpu_submission_wait_stats;
pub use transient_io::transient_scoped_read_ns;

pub use transient_io::transient_info_cache_stats;
pub use transient_io::transient_unscoped_upload_transfer_stats;
pub use transient_io::transient_upload_transfer_stats;

pub use transient_io::transient_binding_cache_stats;
