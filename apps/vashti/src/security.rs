use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderName, Method, Request, header},
    middleware::Next,
    response::Response,
};
use reqwest::Url;

use crate::{app_state::AppState, error::ApiError, settings};

const ORIGIN: HeaderName = HeaderName::from_static("origin");
const SEC_FETCH_SITE: HeaderName = HeaderName::from_static("sec-fetch-site");
const X_FORWARDED_HOST: HeaderName = HeaderName::from_static("x-forwarded-host");
const X_FORWARDED_PROTO: HeaderName = HeaderName::from_static("x-forwarded-proto");

pub async fn origin_check(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, ApiError> {
    if !is_mutating_method(request.method()) {
        return Ok(next.run(request).await);
    }

    let app_settings = settings::service::get_app_settings(&state.db).await?;
    let headers = request.headers();

    if let Some(origin) = header_text(headers, &ORIGIN) {
        if !origin_is_allowed(origin, headers, &app_settings) {
            tracing::warn!(origin, "blocked mutating API request with invalid origin");
            return Err(ApiError::forbidden(
                "invalid_origin",
                "Request origin is not allowed",
            ));
        }
    } else if header_text(headers, &SEC_FETCH_SITE)
        .is_some_and(|fetch_site| fetch_site.eq_ignore_ascii_case("cross-site"))
    {
        tracing::warn!("blocked mutating API request with cross-site fetch metadata");
        return Err(ApiError::forbidden(
            "invalid_origin",
            "Request origin is not allowed",
        ));
    }

    Ok(next.run(request).await)
}

fn is_mutating_method(method: &Method) -> bool {
    matches!(
        method,
        &Method::POST | &Method::PUT | &Method::PATCH | &Method::DELETE
    )
}

fn origin_is_allowed(
    origin: &str,
    headers: &HeaderMap,
    app_settings: &settings::service::AppSettingsResponse,
) -> bool {
    let Some(origin) = normalize_url_origin(origin) else {
        return false;
    };

    if request_origin(headers, app_settings)
        .as_deref()
        .is_some_and(|allowed| allowed == origin)
    {
        return true;
    }

    app_settings
        .public_base_url
        .as_deref()
        .and_then(normalize_url_origin)
        .as_deref()
        .is_some_and(|allowed| allowed == origin)
}

fn request_origin(
    headers: &HeaderMap,
    app_settings: &settings::service::AppSettingsResponse,
) -> Option<String> {
    let host = if app_settings.trust_proxy_headers {
        header_text(headers, &X_FORWARDED_HOST)
            .and_then(first_header_part)
            .or_else(|| header_text(headers, &header::HOST))
    } else {
        header_text(headers, &header::HOST)
    }?;

    let scheme = if app_settings.trust_proxy_headers {
        header_text(headers, &X_FORWARDED_PROTO)
            .and_then(first_header_part)
            .and_then(normalize_scheme)
            .unwrap_or_else(|| default_scheme(app_settings))
    } else {
        default_scheme(app_settings)
    };

    normalize_url_origin(&format!("{}://{}", scheme, host.trim()))
}

fn normalize_url_origin(value: &str) -> Option<String> {
    let parsed = Url::parse(value.trim()).ok()?;
    let scheme = normalize_scheme(parsed.scheme())?;
    let host = parsed
        .host_str()?
        .trim_end_matches('.')
        .to_ascii_lowercase();
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{}]", host)
    } else {
        host
    };

    let authority = match parsed.port() {
        Some(port) if port != default_port(scheme) => format!("{}:{}", host, port),
        _ => host,
    };

    Some(format!("{}://{}", scheme, authority))
}

fn first_header_part(value: &str) -> Option<&str> {
    value
        .split(',')
        .next()
        .map(str::trim)
        .filter(|part| !part.is_empty())
}

fn normalize_scheme(value: &str) -> Option<&'static str> {
    if value.eq_ignore_ascii_case("https") {
        Some("https")
    } else if value.eq_ignore_ascii_case("http") {
        Some("http")
    } else {
        None
    }
}

fn default_port(scheme: &str) -> u16 {
    if scheme == "https" { 443 } else { 80 }
}

fn default_scheme(app_settings: &settings::service::AppSettingsResponse) -> &'static str {
    if app_settings.network_mode == "public_https_proxy" {
        "https"
    } else {
        "http"
    }
}

fn header_text<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn app_settings(
        network_mode: &str,
        public_base_url: Option<&str>,
        trust_proxy_headers: bool,
    ) -> settings::service::AppSettingsResponse {
        settings::service::AppSettingsResponse {
            allow_signup: true,
            signup_limit: 0,
            signup_count: 0,
            max_upload_bytes: 1,
            request_timeout_ms: 1,
            network_mode: network_mode.to_string(),
            public_base_url: public_base_url.map(str::to_string),
            trust_proxy_headers,
            network_recovery_notice: None,
            update_channel: "stable".to_string(),
        }
    }

    #[test]
    fn same_origin_lan_request_is_allowed() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("192.168.1.5:7771"));

        assert!(origin_is_allowed(
            "http://192.168.1.5:7771",
            &headers,
            &app_settings("lan_http", None, false),
        ));
    }

    #[test]
    fn public_base_url_origin_is_allowed() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("127.0.0.1:7771"));

        assert!(origin_is_allowed(
            "https://chat.example.com",
            &headers,
            &app_settings(
                "public_https_proxy",
                Some("https://chat.example.com"),
                false,
            ),
        ));
    }

    #[test]
    fn trusted_forwarded_headers_are_used_when_enabled() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("127.0.0.1:7771"));
        headers.insert(
            X_FORWARDED_HOST,
            HeaderValue::from_static("chat.example.com"),
        );
        headers.insert(X_FORWARDED_PROTO, HeaderValue::from_static("https"));

        assert!(origin_is_allowed(
            "https://chat.example.com",
            &headers,
            &app_settings("public_https_proxy", None, true),
        ));
    }

    #[test]
    fn cross_origin_request_is_rejected() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("192.168.1.5:7771"));

        assert!(!origin_is_allowed(
            "https://evil.example",
            &headers,
            &app_settings("lan_http", None, false),
        ));
    }

    #[test]
    fn default_ports_normalize_to_the_same_origin() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::HOST,
            HeaderValue::from_static("chat.example.com:443"),
        );

        assert!(origin_is_allowed(
            "https://chat.example.com",
            &headers,
            &app_settings("public_https_proxy", None, false),
        ));
    }
}
