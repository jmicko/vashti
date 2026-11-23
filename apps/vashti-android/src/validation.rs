use url::Url;

pub fn response_limit(path: &str, mime_type: Option<&str>) -> usize {
    let attachment = matches!(
        mime_type,
        Some("application/octet-stream" | "image/png" | "image/jpeg" | "image/gif" | "image/webp")
    ) && path
        .strip_prefix("/api/attachments/")
        .is_some_and(|id| uuid::Uuid::parse_str(id).is_ok());
    let video = mime_type == Some("video/mp4")
        && path
            .strip_prefix("/api/videos/runs/")
            .and_then(|p| p.strip_suffix("/result"))
            .is_some_and(|id| uuid::Uuid::parse_str(id).is_ok());
    if attachment {
        128 * 1024 * 1024
    } else if video {
        64 * 1024 * 1024
    } else {
        32 * 1024 * 1024
    }
}

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
    let max_encoded = if mime_type == "video/mp4" {
        (64 * 1024 * 1024_usize).div_ceil(3) * 4
    } else {
        (128 * 1024 * 1024_usize).div_ceil(3) * 4
    };
    if !valid_name || !valid_mime || data_len > max_encoded {
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
    fn private_requests_cannot_switch_to_a_different_server() {
        assert!(check_server_instance(Some("original"), "other").is_err());
        assert!(check_server_instance(Some("original"), "original").is_ok());
        assert!(check_server_instance(None, "other").is_ok());
    }

    #[test]
    fn only_mp4_result_routes_receive_the_larger_download_bound() {
        let path = "/api/videos/runs/269ef6f3-f83d-4cc2-863b-d1d89f6a1a4d/result";
        assert_eq!(response_limit(path, Some("video/mp4")), 64 * 1024 * 1024);
        for (path, mime) in [
            (path, None),
            (path, Some("text/html")),
            ("/api/notes/export", Some("video/mp4")),
            ("/api/videos/runs/invalid/result", Some("video/mp4")),
            (
                "/api/other/api/videos/runs/269ef6f3-f83d-4cc2-863b-d1d89f6a1a4d/result",
                Some("video/mp4"),
            ),
        ] {
            assert_eq!(response_limit(path, mime), 32 * 1024 * 1024);
        }
    }

    #[test]
    fn only_valid_attachment_routes_receive_the_import_download_bound() {
        let path = "/api/attachments/269ef6f3-f83d-4cc2-863b-d1d89f6a1a4d";
        assert_eq!(
            response_limit(path, Some("application/octet-stream")),
            128 * 1024 * 1024
        );
        assert_eq!(response_limit(path, Some("image/png")), 128 * 1024 * 1024);
        for (path, mime) in [
            (path, Some("text/html")),
            ("/api/attachments/invalid", Some("application/octet-stream")),
            (
                "/api/attachments/269ef6f3-f83d-4cc2-863b-d1d89f6a1a4d/content",
                Some("application/octet-stream"),
            ),
            ("/api/other", Some("application/octet-stream")),
        ] {
            assert_eq!(response_limit(path, mime), 32 * 1024 * 1024);
        }
    }

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
        assert!(
            validate_file_export(
                "answer.csv",
                "text/csv",
                (128 * 1024 * 1024_usize).div_ceil(3) * 4 + 1
            )
            .is_err()
        );
        let limit = (64 * 1024 * 1024_usize).div_ceil(3) * 4;
        assert!(validate_file_export("clip.mp4", "video/mp4", limit).is_ok());
        assert!(validate_file_export("clip.mp4", "video/mp4", limit + 1).is_err());
        assert!(validate_file_export("clip.bin", "application/octet-stream", limit).is_ok());
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

pub fn check_server_instance(expected: Option<&str>, actual: &str) -> Result<(), String> {
    if expected.is_some_and(|id| id != actual) {
        return Err("The selected server changed. Private data was not sent.".to_string());
    }
    Ok(())
}
