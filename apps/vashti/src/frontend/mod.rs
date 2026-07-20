use axum::{
    body::Body,
    http::{StatusCode, Uri, header},
    response::Response,
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web/dist"]
struct Assets;

const FALLBACK_INDEX: &str = include_str!("../../web/dist/index.html");
const CACHE_REVALIDATE: &str = "no-cache";
const CACHE_IMMUTABLE: &str = "public, max-age=31536000, immutable";

pub async fn serve_index() -> Response {
    index_response()
}

pub async fn serve_asset(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');

    if path.is_empty() || path.starts_with("app/") {
        return index_response();
    }

    match Assets::get(path) {
        Some(asset) => asset_response(path, asset.data.into_owned()),
        None => index_response(),
    }
}

fn index_response() -> Response {
    match Assets::get("index.html") {
        Some(asset) => asset_response("index.html", asset.data.into_owned()),
        None => html_response(FALLBACK_INDEX.to_string()),
    }
}

fn asset_response(path: &str, body: Vec<u8>) -> Response {
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime.as_ref())
        .header(header::CACHE_CONTROL, cache_control_for(path))
        .body(Body::from(body))
        .expect("valid static asset response")
}

fn html_response(body: String) -> Response {
    asset_response("index.html", body.into_bytes())
}

fn cache_control_for(path: &str) -> &'static str {
    if path.starts_with("assets/") || is_workbox_runtime(path) {
        CACHE_IMMUTABLE
    } else {
        CACHE_REVALIDATE
    }
}

fn is_workbox_runtime(path: &str) -> bool {
    path.strip_prefix("workbox-")
        .is_some_and(|suffix| suffix.ends_with(".js") && suffix.len() > 3)
}

#[cfg(test)]
mod tests {
    use super::{CACHE_IMMUTABLE, CACHE_REVALIDATE, cache_control_for};

    #[test]
    fn mutable_app_entry_files_always_revalidate() {
        assert_eq!(cache_control_for("index.html"), CACHE_REVALIDATE);
        assert_eq!(cache_control_for("sw.js"), CACHE_REVALIDATE);
        assert_eq!(cache_control_for("manifest.webmanifest"), CACHE_REVALIDATE);
        assert_eq!(
            cache_control_for("brand/pwa/vashti-192.png"),
            CACHE_REVALIDATE
        );
    }

    #[test]
    fn fingerprinted_build_files_are_immutable() {
        assert_eq!(
            cache_control_for("assets/index-BtEKJPwt.css"),
            CACHE_IMMUTABLE
        );
        assert_eq!(
            cache_control_for("assets/settingsProfile-yniGc0sU.js"),
            CACHE_IMMUTABLE
        );
        assert_eq!(cache_control_for("workbox-2fbc6a65.js"), CACHE_IMMUTABLE);
    }

    #[test]
    fn similarly_named_mutable_files_are_not_misclassified() {
        assert_eq!(cache_control_for("workbox-help.txt"), CACHE_REVALIDATE);
        assert_eq!(cache_control_for("brand/assets/logo.png"), CACHE_REVALIDATE);
    }
}
