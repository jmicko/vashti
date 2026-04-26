use axum::{
    body::Body,
    http::{StatusCode, Uri, header},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web/dist"]
struct Assets;

const FALLBACK_INDEX: &str = include_str!("../../web/dist/index.html");

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
        .body(Body::from(body))
        .expect("valid static asset response")
}

fn html_response(body: String) -> Response {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], body).into_response()
}
