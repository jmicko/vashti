//! Public discovery contains only built-in capabilities and fixed reason codes.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub service: String,
    pub discovery_version: u32,
    pub version: String,
    pub inference_auth: String,
    pub images: ImageCapabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageCapabilities {
    pub api_version: u32,
    pub access: String,
    pub state: String,
    pub reason: Option<String>,
    pub generate_path: String,
    pub response_formats: Vec<String>,
    pub workflows: Vec<ImageWorkflowCapability>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageWorkflowCapability {
    pub id: String,
    pub state: String,
    pub reason: Option<String>,
    pub reference_images_min: u32,
    pub reference_images_max: u32,
    pub attention: Vec<String>,
}
