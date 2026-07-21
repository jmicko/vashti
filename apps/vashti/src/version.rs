use axum::{Json, extract::State};
use serde::Serialize;

use crate::app_state::AppState;

pub const API_VERSION: u32 = 1;

#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub name: &'static str,
    pub version: &'static str,
    pub instance_id: String,
    pub api_version: u32,
}

pub async fn get_version(State(state): State<AppState>) -> Json<VersionResponse> {
    Json(VersionResponse {
        name: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
        instance_id: state.server_instance_id.to_string(),
        api_version: API_VERSION,
    })
}
