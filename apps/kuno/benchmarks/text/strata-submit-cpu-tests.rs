//! Standalone CPU harness for the actual native planner/submission source.
//! Uses prebuilt real ash/wgpu/CubeCL rlibs. Sibling metadata stand-ins avoid
//! constructing a GPU or rebuilding the complete worker. No hardware acceptance.
#![allow(dead_code)]
mod strata_info_transfer {
    pub const HOST_BYTES: usize = 2 * 1024 * 1024;
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Wait {
        pub semaphore: ash::vk::Semaphore,
        pub value: u64,
    }
}
mod strata_io_reuse {
    pub const HOST_BYTES: usize = 4 * 1024 * 1024;
}
mod strata_allocator_scope {
    pub const HOST_BYTES: usize = 4 * 1024 * 1024;
}
#[path = "../native-text/vendor/cubecl-wgpu/src/compute/strata_schedule.rs"]
mod strata_schedule;
