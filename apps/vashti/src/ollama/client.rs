use std::time::Duration;

use futures_util::{StreamExt, stream};

use crate::ollama::models::{
    OllamaChatRequest, OllamaChatResponse, OllamaModel, ShowModelRequest, ShowModelResponse,
    TagsResponse,
};

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

    let base_url = base_url.to_string();
    let mut models = stream::iter(response.into_models().into_iter().enumerate().map(
        |(index, mut model)| {
            let base_url = base_url.clone();
            async move {
                if let Ok(capabilities) = model_capabilities(client, &base_url, &model.name).await {
                    model.apply_capabilities(capabilities);
                }

                (index, model)
            }
        },
    ))
    .buffer_unordered(8)
    .collect::<Vec<_>>()
    .await;
    models.sort_by_key(|(index, _)| *index);

    Ok(models.into_iter().map(|(_, model)| model).collect())
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

pub async fn chat_once(
    client: &reqwest::Client,
    base_url: &str,
    request: &OllamaChatRequest,
) -> Result<OllamaChatResponse, reqwest::Error> {
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    let response = client
        .post(url)
        .timeout(Duration::from_secs(45))
        .json(request)
        .send()
        .await?
        .error_for_status()?;

    response.json::<OllamaChatResponse>().await
}

pub async fn model_capabilities(
    client: &reqwest::Client,
    base_url: &str,
    model_name: &str,
) -> Result<Vec<String>, reqwest::Error> {
    let url = format!("{}/api/show", base_url.trim_end_matches('/'));
    let response = client
        .post(url)
        .timeout(Duration::from_millis(1800))
        .json(&ShowModelRequest { model: model_name })
        .send()
        .await?
        .error_for_status()?
        .json::<ShowModelResponse>()
        .await?;

    Ok(response.capabilities)
}

pub async fn model_supports_tools(
    client: &reqwest::Client,
    base_url: &str,
    model_name: &str,
) -> bool {
    model_capabilities(client, base_url, model_name)
        .await
        .map(|capabilities| {
            capabilities
                .iter()
                .any(|capability| capability.eq_ignore_ascii_case("tools"))
        })
        .unwrap_or(false)
}
