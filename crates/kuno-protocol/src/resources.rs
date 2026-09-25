//! Read-only GPU and runtime observations; never an execution-placement guarantee.
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub api_version: u32,
    pub sampled_at_unix_ms: u128,
    pub collection_duration_ms: u128,
    pub scheduling_enabled: bool,
    pub gpus: Vec<Gpu>,
    pub gpu_errors: Vec<String>,
    pub ollama: Residency,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Gpu {
    /// PCI topology identity, not an enumeration index or permanent serial.
    pub id: String,
    pub pci_address: String,
    pub vendor_id: u16,
    pub device_id: u16,
    pub driver: Option<String>,
    pub name: Option<String>,
    pub uuid: Option<String>,
    pub render_node: Option<String>,
    pub memory: Option<Memory>,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Memory {
    pub total_bytes: u64,
    pub used_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    pub source: String,
    pub estimated: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Residency {
    pub available: bool,
    pub error: Option<String>,
    pub models: Vec<ResidentModel>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResidentModel {
    pub name: String,
    pub digest: Option<String>,
    pub total_bytes: Option<u64>,
    pub vram_bytes: Option<u64>,
    pub context_length: Option<u64>,
    pub residency: ReportedResidency,
    // /api/ps does not identify physical devices or prove compute placement.
    pub device_assignment_verified: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReportedResidency {
    GpuReported,
    PartialOffloadReported,
    CpuReported,
    Unknown,
}
