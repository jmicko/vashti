use std::{collections::HashMap, collections::VecDeque, net::IpAddr};

use axum::http::{HeaderMap, HeaderName};
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

pub fn client_key(ip: IpAddr) -> String {
    format!("ip:{ip}")
}

pub fn client_ip(headers: &HeaderMap, trust_proxy_headers: bool, peer_ip: IpAddr) -> IpAddr {
    if trust_proxy_headers {
        if let Some(forwarded_for) = header_text(headers, &X_FORWARDED_FOR)
            && let Some(first_ip) = forwarded_for.split(',').next().map(str::trim)
            && let Ok(ip) = first_ip.parse()
        {
            return ip;
        }

        if let Some(real_ip) = header_text(headers, &X_REAL_IP)
            && let Ok(ip) = real_ip.trim().parse()
        {
            return ip;
        }
    }

    peer_ip
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

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use axum::http::{HeaderMap, HeaderValue};

    use super::*;

    #[test]
    fn client_ip_uses_peer_address_without_trusted_proxy_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(&X_FORWARDED_FOR, HeaderValue::from_static("203.0.113.10"));
        let peer_ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 20));

        assert_eq!(client_ip(&headers, false, peer_ip), peer_ip);
        assert_eq!(client_key(peer_ip), "ip:192.0.2.20");
    }

    #[test]
    fn client_ip_uses_first_valid_trusted_proxy_address() {
        let mut headers = HeaderMap::new();
        headers.insert(
            &X_FORWARDED_FOR,
            HeaderValue::from_static("203.0.113.10, 10.0.0.2"),
        );
        let peer_ip = IpAddr::V4(Ipv4Addr::LOCALHOST);

        assert_eq!(
            client_ip(&headers, true, peer_ip),
            "203.0.113.10".parse::<IpAddr>().expect("valid IP")
        );
    }

    #[test]
    fn invalid_trusted_proxy_values_fall_back_to_peer_address() {
        let mut headers = HeaderMap::new();
        headers.insert(&X_FORWARDED_FOR, HeaderValue::from_static("not-an-ip"));
        headers.insert(&X_REAL_IP, HeaderValue::from_static("also-not-an-ip"));
        let peer_ip = IpAddr::V4(Ipv4Addr::LOCALHOST);

        assert_eq!(client_ip(&headers, true, peer_ip), peer_ip);
    }
}
