use std::time::Duration;

use crate::ollama::models::{OllamaChatRequest, OllamaModel, TagsResponse};

pub async fn is_reachable(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<bool, reqwest::Error> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    client
        .get(url)
        .timeout(Duration::from_millis(800))
        .send()
        .await?
        .error_for_status()?
        .json::<TagsResponse>()
        .await?;

    Ok(true)
}

pub async fn fetch_models(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<OllamaModel>, reqwest::Error> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let response = client.get(url).send().await?.error_for_status()?;
    let response: TagsResponse = response.json().await?;

    Ok(response.into_models())
}

pub async fn chat_stream(
    client: &reqwest::Client,
    base_url: &str,
    request: &OllamaChatRequest,
) -> Result<reqwest::Response, reqwest::Error> {
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    client
        .post(url)
        .json(request)
        .send()
        .await?
        .error_for_status()
}
