use std::{
    net::{IpAddr, SocketAddr},
    time::Duration,
};

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
pub const TOOL_BRAVE_WEB_SEARCH: &str = "brave_web_search";
pub const TOOL_OLLAMA_WEB_SEARCH: &str = "ollama_web_search";
pub const TOOL_OLLAMA_WEB_FETCH: &str = "ollama_web_fetch";
pub const TOOL_DIRECT_WEB_FETCH: &str = "direct_web_fetch";

#[derive(Clone, Copy, Debug)]
pub struct ToolSelection {
    pub tool_use_enabled: bool,
    pub brave_web_search_enabled: bool,
    pub ollama_web_search_enabled: bool,
    pub ollama_web_fetch_enabled: bool,
    pub direct_web_fetch_enabled: bool,
}

impl Default for ToolSelection {
    fn default() -> Self {
        Self {
            tool_use_enabled: true,
            brave_web_search_enabled: true,
            ollama_web_search_enabled: true,
            ollama_web_fetch_enabled: true,
            direct_web_fetch_enabled: true,
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
    if selection.brave_web_search_enabled
        && settings.brave_search_enabled
        && settings.has_brave_key()
    {
        tools.push(web_search_tool(
            TOOL_BRAVE_WEB_SEARCH,
            "Search the web using Vashti's Brave Search integration.",
            &settings.web_search_tool_prompt,
        ));
    }
    if selection.ollama_web_search_enabled
        && settings.ollama_web_search_enabled
        && settings.has_ollama_key()
    {
        tools.push(web_search_tool(
            TOOL_OLLAMA_WEB_SEARCH,
            "Search the web using Ollama's hosted search API.",
            &settings.web_search_tool_prompt,
        ));
    }

    if selection.ollama_web_fetch_enabled
        && settings.ollama_web_fetch_enabled
        && settings.has_ollama_key()
    {
        tools.push(web_fetch_tool(
            TOOL_OLLAMA_WEB_FETCH,
            "Fetch a public page using Ollama's hosted fetch API.",
            &settings.web_fetch_tool_prompt,
        ));
    }
    if selection.direct_web_fetch_enabled && settings.direct_web_fetch_enabled {
        tools.push(web_fetch_tool(
            TOOL_DIRECT_WEB_FETCH,
            "Fetch a public page directly from Vashti.",
            &settings.web_fetch_tool_prompt,
        ));
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
    let disabled = [
        TOOL_BRAVE_WEB_SEARCH,
        TOOL_OLLAMA_WEB_SEARCH,
        TOOL_OLLAMA_WEB_FETCH,
        TOOL_DIRECT_WEB_FETCH,
    ]
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

fn web_search_tool(name: &str, provider_description: &str, prompt: &str) -> OllamaTool {
    OllamaTool {
        kind: "function".to_string(),
        function: OllamaToolFunction {
            name: name.to_string(),
            description: format!("{provider_description} {}", render_prompt(prompt)),
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
    }
}

fn web_fetch_tool(name: &str, provider_description: &str, prompt: &str) -> OllamaTool {
    OllamaTool {
        kind: "function".to_string(),
        function: OllamaToolFunction {
            name: name.to_string(),
            description: format!("{provider_description} {}", render_prompt(prompt)),
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
    }
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
        TOOL_BRAVE_WEB_SEARCH
            if selection.tool_use_enabled
                && selection.brave_web_search_enabled
                && settings.brave_search_enabled
                && settings.has_brave_key() =>
        {
            match (
                search_arguments(&call.function.arguments),
                settings.brave_search_api_key.as_deref(),
            ) {
                (Ok((query, max_results)), Some(api_key)) => {
                    brave_web_search(client, api_key, &query, max_results).await
                }
                (Err(error), _) => Err(error),
                (_, None) => Err("Brave Search API key is not configured.".to_string()),
            }
        }
        TOOL_OLLAMA_WEB_SEARCH
            if selection.tool_use_enabled
                && selection.ollama_web_search_enabled
                && settings.ollama_web_search_enabled
                && settings.has_ollama_key() =>
        {
            match (
                search_arguments(&call.function.arguments),
                settings.ollama_api_key.as_deref(),
            ) {
                (Ok((query, max_results)), Some(api_key)) => {
                    ollama_web_search(client, api_key, &query, max_results).await
                }
                (Err(error), _) => Err(error),
                (_, None) => Err("Ollama API key is not configured.".to_string()),
            }
        }
        TOOL_OLLAMA_WEB_FETCH
            if selection.tool_use_enabled
                && selection.ollama_web_fetch_enabled
                && settings.ollama_web_fetch_enabled
                && settings.has_ollama_key() =>
        {
            match (
                required_string(&call.function.arguments, "url"),
                settings.ollama_api_key.as_deref(),
            ) {
                (Ok(url), Some(api_key)) => ollama_web_fetch(client, api_key, &url).await,
                (Err(error), _) => Err(error),
                (_, None) => Err("Ollama API key is not configured.".to_string()),
            }
        }
        TOOL_DIRECT_WEB_FETCH
            if selection.tool_use_enabled
                && selection.direct_web_fetch_enabled
                && settings.direct_web_fetch_enabled =>
        {
            match required_string(&call.function.arguments, "url") {
                Ok(url) => direct_web_fetch(&url).await,
                Err(error) => Err(error),
            }
        }
        TOOL_BRAVE_WEB_SEARCH
        | TOOL_OLLAMA_WEB_SEARCH
        | TOOL_OLLAMA_WEB_FETCH
        | TOOL_DIRECT_WEB_FETCH => {
            Err(format!("{} is disabled for this chat.", call.function.name))
        }
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
        TOOL_BRAVE_WEB_SEARCH | TOOL_OLLAMA_WEB_SEARCH => call
            .function
            .arguments
            .get("query")
            .and_then(|value| value.as_str())
            .map(|query| format!("Searched \"{}\"", truncate_chars(query, 96)))
            .unwrap_or_else(|| "Searched the web".to_string()),
        TOOL_OLLAMA_WEB_FETCH | TOOL_DIRECT_WEB_FETCH => call
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

fn search_arguments(arguments: &serde_json::Value) -> Result<(String, u64), String> {
    let query = required_string(arguments, "query")?;
    let max_results = arguments
        .get("max_results")
        .and_then(|value| value.as_u64())
        .unwrap_or(5)
        .clamp(1, MAX_SEARCH_RESULTS);

    Ok((query, max_results))
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
    let resolved_host = validate_public_http_url(&url).await?;

    let mut client_builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(8))
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy();
    if let Some(resolved_host) = &resolved_host {
        client_builder =
            client_builder.resolve_to_addrs(&resolved_host.host, &resolved_host.addresses);
    }
    let client = client_builder
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

struct ResolvedPublicHost {
    host: String,
    addresses: Vec<SocketAddr>,
}

async fn validate_public_http_url(
    url: &reqwest::Url,
) -> Result<Option<ResolvedPublicHost>, String> {
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
    let unbracketed_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = unbracketed_host.parse::<IpAddr>() {
        ensure_public_ip(ip)?;
        return Ok(None);
    }

    let port = url.port_or_known_default().unwrap_or(80);
    let addresses = tokio::net::lookup_host((host, port))
        .await
        .map_err(|error| format!("Could not resolve URL host: {error}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return Err("URL host did not resolve to an address.".to_string());
    }
    for address in &addresses {
        ensure_public_ip(address.ip())?;
    }

    Ok(Some(ResolvedPublicHost {
        host: host.to_string(),
        addresses,
    }))
}

fn ensure_public_ip(ip: IpAddr) -> Result<(), String> {
    let is_public = match ip {
        IpAddr::V4(ip) => is_public_ipv4(ip),
        IpAddr::V6(ip) => {
            if let Some(mapped) = ip.to_ipv4_mapped() {
                is_public_ipv4(mapped)
            } else {
                let segments = ip.segments();
                segments[0] & 0xe000 == 0x2000
                    && !(segments[0] == 0x2001 && matches!(segments[1], 0x0000 | 0x0002 | 0x0db8))
                    && segments[0] != 0x2002
            }
        }
    };
    if !is_public {
        return Err("Only publicly routable addresses can be fetched.".to_string());
    }

    Ok(())
}

fn is_public_ipv4(ip: std::net::Ipv4Addr) -> bool {
    let [first, second, third, _] = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_multicast()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || first == 0
        || first >= 224
        || (first == 100 && (64..=127).contains(&second))
        || (first == 192 && second == 0 && third == 0)
        || (first == 192 && second == 0 && third == 2)
        || (first == 192 && second == 88 && third == 99)
        || (first == 198 && matches!(second, 18 | 19))
        || (first == 198 && second == 51 && third == 100)
        || (first == 203 && second == 0 && third == 113))
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

#[cfg(test)]
mod tests {
    use std::net::IpAddr;

    use super::ensure_public_ip;

    #[test]
    fn direct_fetch_accepts_public_addresses() {
        for address in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            let ip = address.parse::<IpAddr>().expect("valid test address");
            ensure_public_ip(ip).expect("public address should be accepted");
        }
    }

    #[test]
    fn direct_fetch_rejects_local_reserved_and_tunneled_addresses() {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "100.64.0.1",
            "169.254.1.1",
            "192.0.2.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "::1",
            "fd00::1",
            "fe80::1",
            "2001:db8::1",
            "2002:0a00:0001::1",
            "::ffff:10.0.0.1",
        ] {
            let ip = address.parse::<IpAddr>().expect("valid test address");
            assert!(
                ensure_public_ip(ip).is_err(),
                "{address} should not be fetchable"
            );
        }
    }

    #[tokio::test]
    async fn direct_fetch_validates_literal_ipv4_and_ipv6_urls_without_dns() {
        let public_v4 = reqwest::Url::parse("https://1.1.1.1/").expect("valid IPv4 URL");
        let public_v6 =
            reqwest::Url::parse("https://[2606:4700:4700::1111]/").expect("valid IPv6 URL");
        let local_v6 = reqwest::Url::parse("http://[::1]/").expect("valid local IPv6 URL");

        assert!(
            super::validate_public_http_url(&public_v4)
                .await
                .expect("public IPv4")
                .is_none()
        );
        assert!(
            super::validate_public_http_url(&public_v6)
                .await
                .expect("public IPv6")
                .is_none()
        );
        assert!(super::validate_public_http_url(&local_v6).await.is_err());
    }
}
