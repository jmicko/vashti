//! Versioned, content-free Kuno control responses. Ollama traffic stays opaque.

use serde::{Deserialize, Serialize};

pub const API_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct Status {
    pub api_version: u32,
    pub mode: String,
    pub admission: AdmissionStatus,
    pub gpu_placement_verified: bool,
    pub image_generation: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdmissionStatus {
    pub active: bool,
    pub waiting: usize,
    pub paused: bool,
    /// An upstream may still be working after a transport failure/cancellation.
    pub recovery_required: bool,
    pub completed: u64,
    pub rejected: u64,
}
