//! One transient response owns acceptance, latest previews, and terminal output.
use axum::{
    body::Body,
    http::header,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::stream;
use kuno_protocol::{IMAGE_STREAM_MEDIA_TYPE, ImageStreamEvent};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::{OwnedSemaphorePermit, oneshot, watch};

pub(crate) fn response(
    result: oneshot::Receiver<crate::image_runtime::Output>,
    previews: watch::Receiver<Option<kuno_image::preview::Preview>>,
    slot: Arc<OwnedSemaphorePermit>,
) -> Response {
    struct State {
        result: oneshot::Receiver<crate::image_runtime::Output>,
        previews: watch::Receiver<Option<kuno_image::preview::Preview>>,
        started: bool,
        terminal: bool,
        previews_open: bool,
    }
    let state = State {
        result,
        previews,
        started: false,
        terminal: false,
        previews_open: true,
    };
    let events = stream::unfold(state, |mut state| async move {
        if state.terminal {
            return None;
        }
        let event = if !state.started {
            state.started = true;
            ImageStreamEvent::Started
        } else {
            loop {
                tokio::select! {
                    biased;
                    result = &mut state.result => {
                        state.terminal = true;
                        break match result {
                            Ok(Ok(png)) => ImageStreamEvent::Result {
                                mime_type: "image/png".into(), png_base64: STANDARD.encode(png),
                            },
                            Ok(Err(error)) => ImageStreamEvent::Error { error: error.into() },
                            Err(_) => ImageStreamEvent::Error { error: "image_worker_supervisor_failed".into() },
                        };
                    }
                    changed = state.previews.changed(), if state.previews_open => {
                        if changed.is_err() { state.previews_open = false; continue; }
                        if let Some(frame) = state.previews.borrow_and_update().clone() {
                            break ImageStreamEvent::Preview {
                                step: frame.step, steps: frame.steps, width: frame.width,
                                height: frame.height, png_base64: frame.png_base64,
                            };
                        }
                    }
                }
            }
        };
        let mut bytes = serde_json::to_vec(&event).expect("image event has serializable fields");
        bytes.push(b'\n');
        Some((Ok::<_, Infallible>(bytes), state))
    });
    (
        [
            (header::CONTENT_TYPE, IMAGE_STREAM_MEDIA_TYPE),
            (header::CACHE_CONTROL, "no-store"),
            (header::HeaderName::from_static("x-accel-buffering"), "no"),
        ],
        retain_slot(Body::from_stream(events), slot),
    )
        .into_response()
}

/// Keep content capacity reserved while either cleanup or response delivery
/// still owns request/image bytes, independently of the GPU admission lease.
pub(crate) fn retain_slot(body: Body, slot: Arc<OwnedSemaphorePermit>) -> Body {
    struct GuardedBody {
        inner: Body,
        _slot: Arc<OwnedSemaphorePermit>,
    }
    impl http_body::Body for GuardedBody {
        type Data = bytes::Bytes;
        type Error = axum::Error;
        fn poll_frame(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
            std::pin::Pin::new(&mut self.inner).poll_frame(cx)
        }
        fn is_end_stream(&self) -> bool {
            self.inner.is_end_stream()
        }
        fn size_hint(&self) -> http_body::SizeHint {
            self.inner.size_hint()
        }
    }
    Body::new(GuardedBody {
        inner: body,
        _slot: slot,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body::Body as _;
    use std::{pin::Pin, time::Duration};

    async fn event(body: &mut Body) -> Option<serde_json::Value> {
        let frame = tokio::time::timeout(
            Duration::from_secs(1),
            std::future::poll_fn(|cx| Pin::new(&mut *body).poll_frame(cx)),
        )
        .await
        .unwrap()?;
        let data = frame.unwrap().into_data().unwrap();
        assert_eq!(data.last(), Some(&b'\n'));
        Some(serde_json::from_slice(&data).unwrap())
    }
    fn frame(step: u32) -> kuno_image::preview::Preview {
        kuno_image::preview::Preview {
            step,
            steps: 200,
            width: 64,
            height: 64,
            png_base64: STANDARD.encode(include_bytes!("../../vashti/src/images/fixtures/64.png")),
        }
    }
    async fn fixture() -> (
        oneshot::Sender<crate::image_runtime::Output>,
        watch::Sender<Option<kuno_image::preview::Preview>>,
        Body,
        Arc<tokio::sync::Semaphore>,
    ) {
        let (send, receive) = oneshot::channel();
        let (preview, frames) = watch::channel(None);
        let slots = Arc::new(tokio::sync::Semaphore::new(1));
        let slot = Arc::new(slots.clone().acquire_owned().await.unwrap());
        let response = response(receive, frames, slot);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            IMAGE_STREAM_MEDIA_TYPE
        );
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        (send, preview, response.into_body(), slots)
    }
    #[tokio::test]
    async fn previews_arrive_before_result_with_latest_only_backpressure_and_exact_png() {
        let (send, preview, mut body, slots) = fixture().await;
        assert_eq!(event(&mut body).await.unwrap()["type"], "started");
        for step in 1..=200 {
            preview.send_replace(Some(frame(step)));
        }
        let latest = event(&mut body).await.unwrap();
        assert_eq!(latest["type"], "preview");
        assert_eq!(latest["step"], 200);
        let png = include_bytes!("../../vashti/src/images/fixtures/64.png");
        send.send(Ok(png.to_vec())).unwrap();
        let result = event(&mut body).await.unwrap();
        assert_eq!(result["type"], "result");
        assert_eq!(
            STANDARD
                .decode(result["png_base64"].as_str().unwrap())
                .unwrap(),
            png
        );
        assert!(event(&mut body).await.is_none());
        assert_eq!(slots.available_permits(), 0);
        drop(body);
        assert_eq!(slots.available_permits(), 1);
    }
    #[tokio::test]
    async fn closed_previews_do_not_finish_response_and_terminal_error_wins() {
        let (send, preview, mut body, _) = fixture().await;
        event(&mut body).await.unwrap();
        drop(preview);
        let terminal = tokio::spawn(async move {
            tokio::task::yield_now().await;
            send.send(Err("image_generation_failed")).unwrap();
        });
        assert_eq!(
            event(&mut body).await.unwrap(),
            serde_json::json!({"type":"error","error":"image_generation_failed"})
        );
        terminal.await.unwrap();
        assert!(event(&mut body).await.is_none());
        let (send, preview, mut body, _) = fixture().await;
        event(&mut body).await.unwrap();
        preview.send_replace(Some(frame(1)));
        drop(send);
        assert_eq!(
            event(&mut body).await.unwrap()["error"],
            "image_worker_supervisor_failed"
        );
        assert!(event(&mut body).await.is_none());
    }
    #[tokio::test]
    async fn disconnect_closes_supervisor_receiver_before_and_after_preview() {
        for poll in [false, true] {
            let (mut send, preview, mut body, slots) = fixture().await;
            if poll {
                event(&mut body).await.unwrap();
                preview.send_replace(Some(frame(1)));
                event(&mut body).await.unwrap();
            }
            assert!(!send.is_closed());
            drop(body);
            tokio::time::timeout(Duration::from_secs(1), send.closed())
                .await
                .unwrap();
            assert_eq!(slots.available_permits(), 1);
        }
    }
    #[tokio::test]
    async fn response_drop_keeps_content_slot_and_admission_until_cleanup_finishes() {
        let admission = crate::admission::Admission::new(2, Duration::from_secs(1));
        let mut lease = admission.acquire().await.unwrap();
        lease.dispatched().unwrap();
        let slots = Arc::new(tokio::sync::Semaphore::new(1));
        let slot = Arc::new(slots.clone().acquire_owned().await.unwrap());
        let task_slot = slot.clone();
        let (mut send, receive) = oneshot::channel();
        let (_, frames) = watch::channel(None);
        let body = response(receive, frames, slot).into_body();
        let (cancelled, notified) = oneshot::channel();
        let cleanup = Arc::new(tokio::sync::Notify::new());
        let finish = cleanup.clone();
        let task = tokio::spawn(async move {
            let _slot = task_slot;
            send.closed().await;
            cancelled.send(()).unwrap();
            finish.notified().await;
            lease.complete();
        });
        drop(body);
        notified.await.unwrap();
        assert!(admission.status().active);
        assert_eq!(slots.available_permits(), 0);
        cleanup.notify_one();
        task.await.unwrap();
        assert!(!admission.status().active);
        assert!(!admission.status().recovery_required);
        assert_eq!(slots.available_permits(), 1);
        admission.acquire().await.unwrap().complete();
    }

    #[tokio::test]
    async fn http_response_pushes_preview_before_completion_and_disconnect_closes_receiver() {
        use axum::{Router, routing::post};
        async fn next(
            response: &mut reqwest::Response,
            pending: &mut Vec<u8>,
        ) -> serde_json::Value {
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    if let Some(end) = pending.iter().position(|b| *b == b'\n') {
                        let line: Vec<_> = pending.drain(..=end).collect();
                        return serde_json::from_slice(&line).unwrap();
                    }
                    pending.extend(
                        response
                            .chunk()
                            .await
                            .unwrap()
                            .expect("terminal event missing"),
                    );
                }
            })
            .await
            .unwrap()
        }
        let (mut send, receive) = oneshot::channel();
        let (preview, frames) = watch::channel(None);
        let slots = Arc::new(tokio::sync::Semaphore::new(1));
        let slot = Arc::new(slots.clone().acquire_owned().await.unwrap());
        let stored = Arc::new(std::sync::Mutex::new(Some(response(receive, frames, slot))));
        let app = Router::new().route(
            "/synthetic-generation",
            post(move || {
                let response = stored.lock().unwrap().take().unwrap();
                async move { response }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/synthetic-generation",
            listener.local_addr().unwrap()
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let mut response = client.post(url).send().await.unwrap();
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            IMAGE_STREAM_MEDIA_TYPE
        );
        let mut pending = Vec::new();
        assert_eq!(next(&mut response, &mut pending).await["type"], "started");
        preview.send_replace(Some(frame(1)));
        assert_eq!(next(&mut response, &mut pending).await["type"], "preview");
        assert!(!send.is_closed());
        drop(response);
        tokio::time::timeout(Duration::from_secs(2), send.closed())
            .await
            .unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn uncertain_cleanup_after_disconnect_retains_recovery() {
        let admission = crate::admission::Admission::new(1, Duration::from_secs(1));
        let mut lease = admission.acquire().await.unwrap();
        lease.dispatched().unwrap();
        let (mut send, _, body, _) = fixture().await;
        drop(body);
        send.closed().await;
        // A worker with unproven cleanup must never complete the lease.
        drop(lease);
        assert!(admission.status().recovery_required);
        assert_eq!(
            admission.acquire().await.err(),
            Some("upstream_recovery_required")
        );
    }
}
