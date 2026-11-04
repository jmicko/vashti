use url::Url;

pub fn validate_file_export(
    filename: &str,
    mime_type: &str,
    data_len: usize,
) -> Result<(), String> {
    let valid_name = !filename.trim().is_empty()
        && !matches!(filename, "." | "..")
        && filename.len() <= 160
        && !filename
            .chars()
            .any(|c| c.is_control() || matches!(c, '/' | '\\'));
    let valid_mime = mime_type.len() <= 128
        && mime_type.split_once('/').is_some_and(|(a, b)| {
            !a.is_empty()
                && !b.is_empty()
                && a.bytes()
                    .chain(b.bytes())
                    .all(|c| c.is_ascii_alphanumeric() || b"!#$&^_.+-".contains(&c))
        });
    if !valid_name || !valid_mime || data_len > 32 * 1024 * 1024 {
        return Err("Invalid file export".into());
    }
    Ok(())
}

pub fn normalize_base_url(value: &str, allow_insecure_http: bool) -> Result<Url, String> {
    let mut url = Url::parse(value.trim()).map_err(|_| "Enter a valid server URL".to_string())?;
    if url.username() != "" || url.password().is_some() {
        return Err("Server URLs cannot contain credentials".to_string());
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err("Server URLs cannot contain a query or fragment".to_string());
    }
    if url.path() != "" && url.path() != "/" {
        return Err("Use the server's root URL without an extra path".to_string());
    }
    match url.scheme() {
        "https" => {}
        "http" if allow_insecure_http => {}
        "http" => {
            return Err(
                "HTTP requires acknowledging the local-network security warning".to_string(),
            );
        }
        _ => return Err("Server URL must use HTTPS or HTTP".to_string()),
    }
    if url.host_str().is_none() {
        return Err("Server URL must include a host".to_string());
    }
    url.set_path("/");
    Ok(url)
}

pub fn api_url(base_url: &str, path: &str) -> Result<Url, String> {
    if !path.starts_with("/api/")
        || path.contains('\\')
        || path
            .split('/')
            .any(|segment| segment == ".." || segment == ".")
    {
        return Err("Native requests must use a valid Vashti API path".to_string());
    }
    let base = Url::parse(&format!("{}/", base_url.trim_end_matches('/')))
        .map_err(|_| "Saved server URL is invalid".to_string())?;
    let url = base
        .join(path.trim_start_matches('/'))
        .map_err(|_| "API path is invalid".to_string())?;
    if !url.path().starts_with("/api/") {
        return Err("Native requests must remain within the Vashti API".to_string());
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_exports_are_bounded_and_cannot_choose_paths_or_inject_metadata() {
        assert!(validate_file_export("answer.csv", "text/csv", 1024).is_ok());
        assert!(validate_file_export("report.pdf", "application/pdf", 1024).is_ok());
        for name in ["../secret", "..", "x\\secret", "x\n.csv", ""] {
            assert!(validate_file_export(name, "text/csv", 1).is_err());
        }
        for mime in ["text/csv\r\nother", "text/csv/extra", "", "text/"] {
            assert!(validate_file_export("answer.csv", mime, 1).is_err());
        }
        assert!(validate_file_export("answer.csv", "text/csv", 33 * 1024 * 1024).is_err());
    }

    #[test]
    fn api_urls_remain_on_the_server_and_preserve_queries() {
        let url = api_url(
            "https://chat.example.com",
            "/api/chats/abc/sync?updated_at=123",
        )
        .expect("valid API path");

        assert_eq!(
            url.as_str(),
            "https://chat.example.com/api/chats/abc/sync?updated_at=123"
        );
    }

    #[test]
    fn api_urls_reject_plain_and_encoded_parent_segments() {
        for path in [
            "/api/../admin",
            "/api/%2e%2e/admin",
            "/api/%2E%2E/admin",
            "/api/.%2e/admin",
            "/api/%2e./admin",
        ] {
            assert!(api_url("https://chat.example.com", path).is_err(), "{path}");
        }
    }

    #[test]
    fn base_urls_require_root_https_or_acknowledged_http() {
        assert!(normalize_base_url("https://chat.example.com", false).is_ok());
        assert!(normalize_base_url("http://192.168.1.50:7771", false).is_err());
        assert!(normalize_base_url("http://192.168.1.50:7771", true).is_ok());
        assert!(normalize_base_url("https://user:pass@example.com", false).is_err());
        assert!(normalize_base_url("https://example.com/vashti", false).is_err());
    }
}
