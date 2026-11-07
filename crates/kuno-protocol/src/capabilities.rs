//! Public discovery contains only built-in capabilities and fixed reason codes.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub service: String,
    pub discovery_version: u32,
    pub version: String,
    pub inference_auth: String,
    pub images: ImageCapabilities,
    /// Absent on Kuno versions without video. Clients treat absence as unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub videos: Option<VideoCapabilities>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoCapabilities {
    pub api_version: u32,
    pub access: String,
    pub state: String,
    pub reason: Option<String>,
    pub generate_path: String,
    pub response_formats: Vec<String>,
    pub workflows: Vec<VideoWorkflowCapability>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoWorkflowCapability {
    pub id: String,
    pub state: String,
    pub reason: Option<String>,
    pub first_frame: String,
    pub fps: u32,
    pub frames: Vec<u32>,
    pub max_pixels: u32,
    pub dimension_multiple: u32,
    pub min_dimension: u32,
    pub max_dimension: u32,
    pub audio: bool,
    pub output_mime_type: String,
}
