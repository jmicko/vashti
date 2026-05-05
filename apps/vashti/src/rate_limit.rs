use std::{collections::HashMap, collections::VecDeque};

use axum::http::{HeaderMap, HeaderName, header};
use tokio::sync::Mutex;

use crate::{auth::service::unix_timestamp, error::ApiError};

const X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");
const X_REAL_IP: HeaderName = HeaderName::from_static("x-real-ip");

#[derive(Default)]
pub struct RateLimiter {
    buckets: Mutex<HashMap<String, VecDeque<i64>>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn check(
        &self,
        key: impl Into<String>,
        limit: usize,
        window_seconds: i64,
    ) -> Result<(), ApiError> {
        let key = key.into();
        let now = unix_timestamp();
        let cutoff = now.saturating_sub(window_seconds);
        let mut buckets = self.buckets.lock().await;
        let bucket = buckets.entry(key).or_default();

        while bucket.front().is_some_and(|timestamp| *timestamp <= cutoff) {
            bucket.pop_front();
        }

        if bucket.len() >= limit {
            return Err(ApiError::too_many_requests(
                "rate_limited",
                "Too many requests. Try again shortly.",
            ));
        }

        bucket.push_back(now);

        if buckets.len() > 10_000 {
            buckets.retain(|_, bucket| bucket.back().is_some_and(|timestamp| *timestamp > cutoff));
        }

        Ok(())
    }
}

pub fn client_key(headers: &HeaderMap, trust_proxy_headers: bool) -> String {
    if trust_proxy_headers {
        if let Some(forwarded_for) = header_text(headers, &X_FORWARDED_FOR)
            && let Some(first_ip) = forwarded_for.split(',').next().map(str::trim)
            && !first_ip.is_empty()
        {
            return format!("ip:{}", compact_key_part(first_ip, 96));
        }

        if let Some(real_ip) = header_text(headers, &X_REAL_IP)
            && !real_ip.trim().is_empty()
        {
            return format!("ip:{}", compact_key_part(real_ip.trim(), 96));
        }
    }

    let host = header_text(headers, &header::HOST).unwrap_or("unknown-host");
    let user_agent = header_text(headers, &header::USER_AGENT).unwrap_or("unknown-agent");
    format!(
        "headers:{}:{}",
        compact_key_part(host, 96),
        compact_key_part(user_agent, 160)
    )
}

pub fn compact_key_part(value: &str, max_chars: usize) -> String {
    value
        .trim()
        .chars()
        .take(max_chars)
        .collect::<String>()
        .to_ascii_lowercase()
}

pub fn user_action_key(action: &str, user_id: &str) -> String {
    format!("user:{}:{}", action, user_id)
}

fn header_text<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}
