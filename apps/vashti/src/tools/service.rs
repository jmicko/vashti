use std::{net::IpAddr, time::Duration};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    ollama::models::{OllamaTool, OllamaToolCall, OllamaToolFunction},
    settings::service::ToolSettingsPrivate,
};

const MAX_SEARCH_RESULTS: u64 = 10;
const MAX_FETCH_BYTES: usize = 512 * 1024;
const MAX_TOOL_RESULT_CHARS: usize = 24_000;

#[derive(Clone, Copy, Debug)]
pub struct ToolSelection {
    pub tool_use_enabled: bool,
    pub web_search_enabled: bool,
    pub web_fetch_enabled: bool,
}

impl Default for ToolSelection {
    fn default() -> Self {
        Self {
            tool_use_enabled: true,
            web_search_enabled: true,
            web_fetch_enabled: true,
        }
    }
}

pub fn chat_tools(settings: &ToolSettingsPrivate, selection: ToolSelection) -> Vec<OllamaTool> {
    if !settings.tools_enabled {
        return Vec::new();
    }
    if !selection.tool_use_enabled {
        return Vec::new();
    }

    let mut tools = Vec::new();
    if selection.web_search_enabled
        && ((settings.brave_search_enabled && settings.has_brave_key())
            || (settings.ollama_web_search_enabled && settings.has_ollama_key()))
    {
        tools.push(OllamaTool {
            kind: "function".to_string(),
            function: OllamaToolFunction {
                name: "web_search".to_string(),
                description: render_prompt(&settings.web_search_tool_prompt),
                parameters: json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return. Defaults to 5 and cannot exceed 10."
                        }
                    }
                }),
            },
        });
    }

    if selection.web_fetch_enabled
        && ((settings.ollama_web_fetch_enabled && settings.has_ollama_key())
            || settings.direct_web_fetch_enabled)
    {
        tools.push(OllamaTool {
            kind: "function".to_string(),
            function: OllamaToolFunction {
                name: "web_fetch".to_string(),
                description: render_prompt(&settings.web_fetch_tool_prompt),
                parameters: json!({
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The HTTP or HTTPS URL to fetch."
                        }
                    }
                }),
            },
        });
    }

    tools
}

pub fn tool_system_prompt(settings: &ToolSettingsPrivate, tools: &[OllamaTool]) -> String {
    let available_names = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<Vec<_>>();
    let available = if available_names.is_empty() {
        "none".to_string()
    } else {
        available_names.join(", ")
    };
    let disabled = ["web_search", "web_fetch"]
        .into_iter()
        .filter(|tool_name| !available_names.contains(tool_name))
        .collect::<Vec<_>>();
    let disabled = if disabled.is_empty() {
        "none".to_string()
    } else {
        disabled.join(", ")
    };

    format!(
        "{}\n\nAvailable tools in this chat: {available}.\nDisabled tools in this chat: {disabled}.\nOnly call tools listed as available in this chat. Do not call disabled or unavailable tools.",
        render_prompt(&settings.tool_system_prompt)
    )
}

fn render_prompt(prompt: &str) -> String {
    prompt.replace("{current_date}", &current_date_utc())
}

fn current_date_utc() -> String {
    let now = time::OffsetDateTime::now_utc();
    format!(
        "{:04}-{:02}-{:02}",
        now.year(),
        u8::from(now.month()),
        now.day()
    )
}

pub async fn execute_tool(
    client: &reqwest::Client,
    settings: &ToolSettingsPrivate,
    selection: ToolSelection,
    call: &OllamaToolCall,
) -> String {
    let result = match call.function.name.as_str() {
        "web_search" if selection.tool_use_enabled && selection.web_search_enabled => {
            web_search(client, settings, &call.function.arguments).await
        }
        "web_fetch" if selection.tool_use_enabled && selection.web_fetch_enabled => {
            web_fetch(client, settings, &call.function.arguments).await
        }
        "web_search" | "web_fetch" => Err(format!(
            "{} is disabled for this chat.",
            call.function.name
        )),
        name => Err(format!("Unknown tool: {name}")),
    };

    match result {
        Ok(value) => truncate_chars(&value, MAX_TOOL_RESULT_CHARS),
        Err(error) => serde_json::to_string(&json!({ "error": error }))
            .unwrap_or_else(|_| "{\"error\":\"Tool failed\"}".to_string()),
    }
}

pub fn tool_summary(call: &OllamaToolCall) -> String {
    match call.function.name.as_str() {
        "web_search" => call
            .function
            .arguments
            .get("query")
            .and_then(|value| value.as_str())
            .map(|query| format!("Searched \"{}\"", truncate_chars(query, 96)))
            .unwrap_or_else(|| "Searched the web".to_string()),
        "web_fetch" => call
            .function
            .arguments
            .get("url")
            .and_then(|value| value.as_str())
            .map(|url| format!("Fetched \"{}\"", truncate_chars(url, 120)))
            .unwrap_or_else(|| "Fetched a page".to_string()),
        name => format!("Used {name}"),
    }
}

pub fn tool_usage_block(call: &OllamaToolCall, result: &str) -> String {
    let usage = ToolUsageRecord {
        name: call.function.name.clone(),
        summary: tool_summary(call),
        arguments: call.function.arguments.clone(),
        result: result.to_string(),
    };
    let json = serde_json::to_string(&usage).unwrap_or_else(|_| {
        "{\"name\":\"tool\",\"summary\":\"Used a tool\",\"arguments\":{},\"result\":\"\"}"
            .to_string()
    });

    format!("\n\n<VASHTI_TOOL_USAGE>{json}</VASHTI_TOOL_USAGE>\n\n")
}

#[derive(Serialize)]
struct ToolUsageRecord {
    name: String,
    summary: String,
    arguments: serde_json::Value,
    result: String,
}

async fn web_search(
    client: &reqwest::Client,
    settings: &ToolSettingsPrivate,
    arguments: &serde_json::Value,
) -> Result<String, String> {
    let query = required_string(arguments, "query")?;
    let max_results = arguments
        .get("max_results")
        .and_then(|value| value.as_u64())
        .unwrap_or(5)
        .clamp(1, MAX_SEARCH_RESULTS);

    if settings.brave_search_enabled {
        if let Some(api_key) = settings.brave_search_api_key.as_deref() {
            return brave_web_search(client, api_key, &query, max_results).await;
        }
    }

    if settings.ollama_web_search_enabled {
        if let Some(api_key) = settings.ollama_api_key.as_deref() {
            return ollama_web_search(client, api_key, &query, max_results).await;
        }
    }

    Err("No enabled web search provider has an API key configured.".to_string())
}

async fn web_fetch(
    client: &reqwest::Client,
    settings: &ToolSettingsPrivate,
    arguments: &serde_json::Value,
) -> Result<String, String> {
    let url = required_string(arguments, "url")?;

    if settings.ollama_web_fetch_enabled {
        if let Some(api_key) = settings.ollama_api_key.as_deref() {
            return ollama_web_fetch(client, api_key, &url).await;
        }
    }

    if settings.direct_web_fetch_enabled {
        return direct_web_fetch(&url).await;
    }

    Err("No enabled web fetch provider is configured.".to_string())
}

async fn ollama_web_search(
    client: &reqwest::Client,
    api_key: &str,
    query: &str,
    max_results: u64,
) -> Result<String, String> {
    let response = client
        .post("https://ollama.com/api/web_search")
        .bearer_auth(api_key)
        .json(&json!({ "query": query, "max_results": max_results }))
        .send()
        .await
        .map_err(|error| format!("Ollama web search request failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Ollama web search failed: {error}"))?
        .json::<OllamaSearchResponse>()
        .await
        .map_err(|error| format!("Ollama web search response was invalid: {error}"))?;

    serde_json::to_string(&json!({
        "provider": "ollama",
        "query": query,
        "results": response.results,
    }))
    .map_err(|error| format!("Could not serialize search result: {error}"))
}

async fn ollama_web_fetch(
    client: &reqwest::Client,
    api_key: &str,
    url: &str,
) -> Result<String, String> {
    let response = client
        .post("https://ollama.com/api/web_fetch")
        .bearer_auth(api_key)
        .json(&json!({ "url": url }))
        .send()
        .await
        .map_err(|error| format!("Ollama web fetch request failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Ollama web fetch failed: {error}"))?
        .json::<OllamaFetchResponse>()
        .await
        .map_err(|error| format!("Ollama web fetch response was invalid: {error}"))?;

    serde_json::to_string(&json!({
        "provider": "ollama",
        "url": url,
        "title": response.title,
        "content": truncate_chars(&response.content, MAX_TOOL_RESULT_CHARS),
        "links": response.links,
    }))
    .map_err(|error| format!("Could not serialize fetch result: {error}"))
}

async fn brave_web_search(
    client: &reqwest::Client,
    api_key: &str,
    query: &str,
    max_results: u64,
) -> Result<String, String> {
    let count = max_results.to_string();
    let response = client
        .get("https://api.search.brave.com/res/v1/web/search")
        .query(&[
            ("q", query),
            ("count", count.as_str()),
            ("safesearch", "moderate"),
        ])
        .header("Accept", "application/json")
        .header("X-Subscription-Token", api_key)
        .send()
        .await
        .map_err(|error| format!("Brave search request failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Brave search failed: {error}"))?
        .json::<BraveSearchResponse>()
        .await
        .map_err(|error| format!("Brave search response was invalid: {error}"))?;

    let results = response
        .web
        .map(|web| web.results)
        .unwrap_or_default()
        .into_iter()
        .map(|result| {
            json!({
                "title": result.title,
                "url": result.url,
                "content": result.description.unwrap_or_default(),
            })
        })
        .collect::<Vec<_>>();

    serde_json::to_string(&json!({
        "provider": "brave",
        "query": query,
        "results": results,
    }))
    .map_err(|error| format!("Could not serialize search result: {error}"))
}

async fn direct_web_fetch(url: &str) -> Result<String, String> {
    let url = reqwest::Url::parse(url)
        .map_err(|_| "Direct fetch URL must be a valid HTTP or HTTPS URL.".to_string())?;
    validate_public_http_url(&url).await?;

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(8))
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| format!("Could not create direct fetch client: {error}"))?;
    let response = client
        .get(url.clone())
        .header("User-Agent", "Vashti/0.1 web_fetch")
        .send()
        .await
        .map_err(|error| format!("Direct page fetch failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Direct page fetch returned an error: {error}"))?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();

    while let Some(next) = stream.next().await {
        let chunk = next.map_err(|error| format!("Direct page fetch failed: {error}"))?;
        if bytes.len() + chunk.len() > MAX_FETCH_BYTES {
            return Err("Fetched page is too large.".to_string());
        }
        bytes.extend_from_slice(&chunk);
    }

    let raw = String::from_utf8_lossy(&bytes).to_string();
    let content = if content_type.contains("text/html") {
        html_to_text(&raw)
    } else if content_type.contains("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("markdown")
        || content_type.is_empty()
    {
        raw
    } else {
        return Err(format!(
            "Direct page fetch does not support this content type: {content_type}"
        ));
    };

    serde_json::to_string(&json!({
        "provider": "direct",
        "url": url.as_str(),
        "content_type": content_type,
        "content": truncate_chars(&normalize_whitespace(&content), MAX_TOOL_RESULT_CHARS),
    }))
    .map_err(|error| format!("Could not serialize fetch result: {error}"))
}

async fn validate_public_http_url(url: &reqwest::Url) -> Result<(), String> {
    match url.scheme() {
        "http" | "https" => {}
        _ => return Err("Only HTTP and HTTPS URLs can be fetched.".to_string()),
    }
    if url.username() != "" || url.password().is_some() {
        return Err("URLs with embedded credentials cannot be fetched.".to_string());
    }

    let host = url
        .host_str()
        .ok_or_else(|| "URL must include a host.".to_string())?;
    if host.eq_ignore_ascii_case("localhost") {
        return Err("Localhost URLs cannot be fetched.".to_string());
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        ensure_public_ip(ip)?;
        return Ok(());
    }

    let port = url.port_or_known_default().unwrap_or(80);
    let addresses = tokio::net::lookup_host((host, port))
        .await
        .map_err(|error| format!("Could not resolve URL host: {error}"))?;
    for address in addresses {
        ensure_public_ip(address.ip())?;
    }

    Ok(())
}

fn ensure_public_ip(ip: IpAddr) -> Result<(), String> {
    match ip {
        IpAddr::V4(ip) => {
            if ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.is_broadcast()
                || ip.is_unspecified()
            {
                return Err(
                    "Private, local, multicast, and unspecified addresses cannot be fetched."
                        .to_string(),
                );
            }
        }
        IpAddr::V6(ip) => {
            if ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
                || ip.is_multicast()
            {
                return Err(
                    "Private, local, multicast, and unspecified addresses cannot be fetched."
                        .to_string(),
                );
            }
        }
    }

    Ok(())
}

fn required_string(arguments: &serde_json::Value, key: &str) -> Result<String, String> {
    arguments
        .get(key)
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("Tool argument `{key}` is required."))
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    if truncated.len() < value.len() {
        truncated.push_str("\n[truncated]");
    }
    truncated
}

fn html_to_text(input: &str) -> String {
    let without_scripts = remove_html_block(input, "script");
    let without_styles = remove_html_block(&without_scripts, "style");
    let mut text = String::new();
    let mut in_tag = false;

    for character in without_styles.chars() {
        match character {
            '<' => {
                in_tag = true;
                text.push(' ');
            }
            '>' => {
                in_tag = false;
                text.push(' ');
            }
            _ if !in_tag => text.push(character),
            _ => {}
        }
    }

    decode_basic_entities(&text)
}

fn remove_html_block(input: &str, tag: &str) -> String {
    let mut output = input.to_string();
    loop {
        let lower = output.to_ascii_lowercase();
        let Some(start) = lower.find(&format!("<{tag}")) else {
            break;
        };
        let Some(relative_end) = lower[start..].find(&format!("</{tag}>")) else {
            output.drain(start..);
            break;
        };
        let end = start + relative_end + tag.len() + 3;
        output.replace_range(start..end, " ");
    }
    output
}

fn decode_basic_entities(input: &str) -> String {
    input
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn normalize_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct SearchResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default, alias = "description")]
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaSearchResponse {
    #[serde(default)]
    results: Vec<SearchResult>,
}

#[derive(Debug, Deserialize)]
struct OllamaFetchResponse {
    #[serde(default)]
    title: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    links: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    #[serde(default)]
    results: Vec<BraveWebResult>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    description: Option<String>,
}
