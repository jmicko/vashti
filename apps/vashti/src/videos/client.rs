use crate::error::ApiError;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use kuno_protocol::{
    MAX_VIDEO_BYTES, VideoCapabilities, VideoPhase, VideoRecipe, VideoStreamEvent,
};
use serde_json::{Value, json};
use tokio_util::{
    codec::{FramedRead, LinesCodec},
    io::StreamReader,
};

pub fn supported(c: &VideoCapabilities) -> bool {
    c.api_version == 1
        && c.access == "inference"
        && c.state == "ready"
        && c.generate_path == "/kuno/v1/videos/generate"
        && c.response_formats
            .iter()
            .any(|f| f == "application/x-ndjson")
}
fn invalid() -> ApiError {
    ApiError::bad_request(
        "video_stream_invalid",
        "The video service returned an incomplete or invalid response. No clip was saved.",
    )
}
fn rejected(reason: &'static str) -> ApiError {
    tracing::warn!(reason, "video service stream validation failed");
    invalid()
}
pub async fn generate(
    http: &reqwest::Client,
    base: &str,
    recipe: &VideoRecipe,
    mut publish: impl FnMut(Value),
) -> Result<Vec<u8>, ApiError> {
    let mut url =
        reqwest::Url::parse(&format!("{}/", base.trim_end_matches('/'))).map_err(|_| invalid())?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid());
    }
    url = url.join("kuno/v1/videos/generate").map_err(|_| invalid())?;
    let response = http.post(url).json(recipe).send().await.map_err(|_| {
        ApiError::bad_request(
            "video_host_unreachable",
            "Video inference was interrupted. Check the host; retry is always explicit.",
        )
    })?;
    if !response.status().is_success() {
        return Err(ApiError::bad_request(
            "video_host_rejected",
            "The video host rejected the request. Refresh model availability and check the settings.",
        ));
    }
    if response
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .is_none_or(|h| h.split(';').next() != Some("application/x-ndjson"))
    {
        return Err(invalid());
    }
    let stream = response
        .bytes_stream()
        .map(|r| r.map_err(std::io::Error::other));
    let mut lines = FramedRead::new(
        StreamReader::new(stream),
        LinesCodec::new_with_max_length(MAX_VIDEO_BYTES.div_ceil(3) * 4 + 8192),
    );
    let mut started = false;
    let mut result = None;
    let mut phase = 0;
    let mut step = 0;
    while let Some(line) = lines.next().await {
        let line = line.map_err(|_| invalid())?;
        if line.is_empty() {
            continue;
        }
        if result.is_some() {
            return Err(invalid());
        }
        let event: VideoStreamEvent = serde_json::from_str(&line).map_err(|_| invalid())?;
        match event {
            VideoStreamEvent::Started if !started => started = true,
            VideoStreamEvent::Progress {
                phase: p,
                step: s,
                steps,
            } if started => {
                let next = match p {
                    VideoPhase::Conditioning => 1,
                    VideoPhase::Sampling => 2,
                    VideoPhase::Decoding => 3,
                    VideoPhase::Encoding => 4,
                };
                if next < phase
                    || (next != 2 && (s.is_some() || steps.is_some()))
                    || (next == 2
                        && (steps != Some(8) || s.is_none_or(|n| n < step || n > 8 || n == 0)))
                    || (next > 2 && step != 8)
                {
                    return Err(rejected("progress_order"));
                }
                phase = next;
                if let Some(s) = s {
                    step = s;
                }
                publish(json!({"phase":p,"step":s,"steps":steps}));
            }
            VideoStreamEvent::Preview {
                step: s,
                steps,
                width,
                height,
                png_base64,
            } if started => {
                if steps != 8
                    || s > 8
                    || width == 0
                    || height == 0
                    || width > 512
                    || height > 512
                    || png_base64.len() > 2 * 1024 * 1024
                {
                    return Err(rejected("preview_bounds"));
                }
                let bytes = STANDARD.decode(&png_base64).map_err(|_| invalid())?;
                if bytes.get(..8) != Some(b"\x89PNG\r\n\x1a\n")
                    || bytes.get(16..20) != Some(width.to_be_bytes().as_slice())
                    || bytes.get(20..24) != Some(height.to_be_bytes().as_slice())
                {
                    return Err(rejected("preview_header"));
                }
                tokio::task::spawn_blocking(move || {
                    image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
                })
                .await
                .map_err(|_| invalid())?
                .map_err(|_| invalid())?;
                if phase <= 2 {
                    publish(
                        json!({"preview":{"png_base64":png_base64,"width":width,"height":height}}),
                    );
                }
            }
            VideoStreamEvent::Result {
                mime_type,
                width,
                height,
                frames,
                fps,
                audio,
                video_base64,
            } if started => {
                if mime_type != "video/mp4"
                    || width != recipe.width
                    || height != recipe.height
                    || frames != recipe.frames
                    || fps != 24
                    || audio != recipe.audio
                    || video_base64.len() > MAX_VIDEO_BYTES.div_ceil(3) * 4
                {
                    return Err(rejected("result_metadata"));
                }
                let expected = (width, height, frames, fps, audio);
                result = Some(
                    tokio::task::spawn_blocking(move || {
                        let bytes = STANDARD.decode(video_base64).map_err(|_| invalid())?;
                        let summary = kuno_image::video::validate_mp4(&bytes, MAX_VIDEO_BYTES)
                            .map_err(|_| invalid())?;
                        if (
                            summary.width,
                            summary.height,
                            summary.frames,
                            summary.fps,
                            summary.audio,
                        ) != expected
                        {
                            return Err(invalid());
                        }
                        Ok::<_, ApiError>(bytes)
                    })
                    .await
                    .map_err(|_| invalid())??,
                );
            }
            VideoStreamEvent::Error { .. } if started => {
                return Err(ApiError::bad_request(
                    "video_generation_failed",
                    "Video generation failed on the inference host. Nothing was saved; retry is explicit.",
                ));
            }
            _ => return Err(invalid()),
        }
    }
    result.ok_or_else(invalid)
}
