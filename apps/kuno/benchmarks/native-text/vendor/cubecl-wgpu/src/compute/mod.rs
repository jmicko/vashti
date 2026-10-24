// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
pub(crate) mod controller;

mod public_external;
mod storage;

mod strata_allocator_scope;
#[cfg(feature = "kuno-expert-cohorts")]
mod strata_cohort_selection;
#[cfg(feature = "kuno-expert-cohorts")]
pub use strata_cohort_selection::{NATIVE_COHORT_SELECTION_LABELS, native_strata_cohort_selection_stats};
mod strata_draft_scope;
mod strata_flights;
pub(crate) mod strata_graph;
pub(crate) mod strata_pipeline;
#[cfg(feature = "kuno-public-host-clock")]
mod strata_pipeline_profile;
#[cfg(feature = "kuno-public-host-clock")]
pub use strata_pipeline_profile::{
    COLUMNS as NATIVE_PIPELINE_CREATION_COLUMNS, LABELS as NATIVE_PIPELINE_CREATION_LABELS,
    PublicPipelineProfile, totals as native_strata_pipeline_creation_profile,
};
#[cfg(feature = "kuno-public-host-clock")]
pub fn begin_native_strata_public_pipeline_profile() -> Option<PublicPipelineProfile> {
    strata_pipeline_profile::begin(strata_graph::enabled())
}
mod strata_handoff;
mod strata_info_transfer;
mod strata_io_reuse;
mod strata_parameters;
mod strata_physical;
#[cfg(feature = "kuno-strata-shared")]
mod strata_quads;
#[cfg(feature = "kuno-strata-shared")]
pub use strata_quads::{lower_native_subgroup_quads, native_compute_quads_supported};
mod strata_readback;
mod strata_schedule;
mod strata_upload;
#[cfg(feature = "kuno-public-host-clock")]
pub use strata_graph::{
    NATIVE_BATCH_CPU_LABELS, NATIVE_CAPTURE_CPU_LABELS, native_strata_host_batch_profile,
    native_strata_host_capture_profile,
};
pub use strata_graph::{
    arm_native_strata_public_rolling_failure, native_strata_graph_compute_barrier_stats,
    native_strata_graph_copy_barrier_stats, native_strata_graph_execution_stats,
    native_strata_graph_rolling_stats, native_strata_graph_stats, native_strata_operator_profile,
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

#[cfg(feature = "kuno-copy-cohorts")]
pub use strata_graph::native_strata_segment_compact_profile;

#[cfg(feature = "kuno-native-prepared")]
pub use strata_pipeline::native_strata_prepared_pipeline_stats;

#[cfg(feature = "kuno-global-wide32")]
pub use strata_graph::native_strata_atomic_or_descriptor_mask;
