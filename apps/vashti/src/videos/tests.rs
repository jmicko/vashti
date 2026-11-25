use super::models::Request;
use super::{
    client,
    models::{Draft, Inputs},
    service,
};
use crate::{
    app_state::AppState,
    auth::service::{register_user, unix_timestamp},
    config::Config,
    startup,
};
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{Response, header},
    routing::{get, post},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use uuid::Uuid;

const CLIP: &[u8] = include_bytes!("fixtures/clip.mp4");
fn first_frame(value: &str) -> Inputs {
    Inputs {
        first_frame: Some(value.into()),
        ..Inputs::default()
    }
}
#[derive(Clone, Default)]
struct Mock {
    hold: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Value>>>,
    broken: Arc<AtomicBool>,
    events: Arc<Mutex<Option<Vec<Value>>>>,
}
fn legacy_capabilities() -> Value {
    json!({"service":"kuno","discovery_version":1,"version":"fixture","inference_auth":"none","images":{"api_version":1,"access":"inference","state":"ready","reason":null,"generate_path":"/kuno/v1/images/generate","response_formats":["application/x-ndjson"],"workflows":[]},"videos":{"api_version":1,"access":"inference","state":"ready","reason":null,"generate_path":"/kuno/v1/videos/generate","response_formats":["application/x-ndjson"],"workflows":[{"id":kuno_protocol::MINIMAX_H3_I2V_RECIPE,"state":"ready","reason":null,"first_frame":"required","fps":24,"frames":[22,39,56,73,90,107,124],"max_pixels":452608,"dimension_multiple":32,"min_dimension":256,"max_dimension":1280,"audio":true,"output_mime_type":"video/mp4"}]}})
}
fn capabilities() -> Value {
    let mut value = legacy_capabilities();
    let first = value["videos"]["workflows"][0].clone();
    let mut general = first.clone();
    general["id"] = json!(kuno_protocol::MINIMAX_H3_VIDEO_RECIPE);
    general["first_frame"] = json!("optional");
    general["last_frame"] = json!("optional");
    general["steps"] = json!(8);
    general["max_reference_images"] = json!(0);
    let mut references = general.clone();
    references["id"] = json!(kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE);
    references["first_frame"] = json!("unsupported");
    references["last_frame"] = json!("unsupported");
    references["max_reference_images"] = json!(4);
    references["steps"] = json!(4);
    value["videos"]["workflows"] = json!([first, general, references]);
    value
}

#[test]
fn extended_clips_require_the_memory_contract_and_fit_the_resolution_tier() {
    let mut wire = capabilities()["videos"]["workflows"][1].clone();
    let legacy = serde_json::from_value(wire.clone()).unwrap();
    wire["limits"] = json!("memory");
    wire["frame_base"] = json!(5);
    wire["frame_step"] = json!(17);
    let memory = serde_json::from_value(wire.clone()).unwrap();
    let standard = Draft {
        frames: 243,
        ..Draft::default()
    };
    assert!(service::validate(&standard).is_ok());
    assert!(service::supported_clip(&standard, &memory));
    assert!(!service::supported_clip(&standard, &legacy));
    let high = Draft {
        width: 1280,
        height: 704,
        frames: 124,
        ..Draft::default()
    };
    assert!(service::validate(&high).is_ok());
    assert!(service::supported_clip(&high, &memory));
    assert!(!service::supported_clip(&high, &legacy));
    assert!(!service::supported_clip(
        &Draft {
            frames: 141,
            ..high.clone()
        },
        &memory
    ));
    assert!(!service::supported_clip(
        &Draft {
            height: 720,
            ..high
        },
        &memory
    ));
    assert!(!service::supported_clip(
        &Draft {
            frames: 242,
            ..standard.clone()
        },
        &memory
    ));
    assert!(
        service::validate(&Draft {
            frames: 260,
            ..standard.clone()
        })
        .is_err()
    );
    wire["frame_step"] = json!(0);
    assert!(!service::supported_clip(
        &standard,
        &serde_json::from_value(wire).unwrap()
    ));
}

#[tokio::test]
async fn streamed_memory_errors_give_actionable_recovery_without_provider_detail() {
    let f = Fixture::new().await;
    for (code, text) in [
        ("image_gpu_memory_unavailable", "GPU memory"),
        ("image_host_memory_unavailable", "RAM"),
    ] {
        *f.mock.events.lock().unwrap() = Some(vec![
            json!({"type":"started"}),
            json!({"type":"error","error":code}),
        ]);
        let error = client::generate(
            &f.state.http_client,
            &f.base,
            &f.draft().recipe(first_frame("AAAA")),
            |_| {},
        )
        .await
        .unwrap_err();
        assert_eq!(error.code(), "video_host_memory_unavailable");
        assert!(error.message().contains(text));
        assert!(
            error
                .message()
                .contains("Shorten the clip or lower the resolution")
        );
        assert!(!error.message().contains(code));
    }
}

async fn generate(State(mock): State<Mock>, Json(input): Json<Value>) -> Response<Body> {
    let steps = if input["workflow"] == kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE {
        4
    } else {
        8
    };
    mock.requests.lock().unwrap().push(input);
    while mock.hold.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut lines = vec![
        json!({"type":"started"}),
        json!({"type":"progress","phase":"conditioning"}),
        json!({"type":"progress","phase":"sampling","step":steps,"steps":steps}),
        json!({"type":"progress","phase":"decoding"}),
        json!({"type":"progress","phase":"encoding"}),
    ];
    if !mock.broken.load(Ordering::Relaxed) {
        lines.push(json!({"type":"result","mime_type":"video/mp4","width":256,"height":256,"frames":22,"fps":24,"audio":true,"video_base64":STANDARD.encode(CLIP)}));
    }
    if let Some(events) = mock.events.lock().unwrap().as_ref() {
        lines = events.clone();
    }
    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from(
            lines.iter().map(|v| format!("{v}\n")).collect::<String>(),
        ))
        .unwrap()
}
struct Fixture {
    state: AppState,
    owner: String,
    other: String,
    backend: String,
    mock: Mock,
    server: tokio::task::JoinHandle<()>,
    dir: PathBuf,
    base: String,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
impl Fixture {
    async fn new() -> Self {
        let mock = Mock::default();
        let router = Router::new()
            .route(
                "/kuno/v1/capabilities",
                get(|| async { Json(capabilities()) }),
            )
            .route("/kuno/v1/videos/generate", post(generate))
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::new()
                    .filename(":memory:")
                    .foreign_keys(true),
            )
            .await
            .unwrap();
        startup::migrations::run(&pool).await.unwrap();
        startup::bootstrap::ensure_app_settings(&pool)
            .await
            .unwrap();
        let owner = register_user(&pool, "video-owner".into(), None, "fixture-only".into())
            .await
            .unwrap()
            .user
            .id;
        let other = register_user(&pool, "video-other".into(), None, "fixture-only".into())
            .await
            .unwrap()
            .user
            .id;
        sqlx::query("UPDATE users SET is_disabled=0,role='admin'")
            .execute(&pool)
            .await
            .unwrap();
        let backend = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO ollama_backends(id,name,base_url,created_at,updated_at) VALUES(?,'video fixture',?,?,?)").bind(&backend).bind(&base).bind(unix_timestamp()).bind(unix_timestamp()).execute(&pool).await.unwrap();
        let dir = std::env::temp_dir().join(format!("vashti-video-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = Config {
            app_root: dir.clone(),
            data_dir: dir.clone(),
            database_path: dir.join("app.db"),
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            session_cookie_name: "vashti_session".into(),
            session_ttl_seconds: 3600,
            managed_updates: false,
            update_base_url: "https://example.invalid".into(),
        };
        let state = AppState::new(
            config,
            pool,
            reqwest::Client::new(),
            Uuid::new_v4().to_string(),
        );
        Self {
            state,
            owner,
            other,
            backend,
            mock,
            server,
            dir,
            base,
        }
    }
    fn draft(&self) -> Draft {
        Draft {
            backend_id: self.backend.clone(),
            workflow: kuno_protocol::MINIMAX_H3_I2V_RECIPE.into(),
            width: 256,
            height: 256,
            frames: 22,
            prompt: "Synthetic motion".into(),
            ..Draft::default()
        }
    }
    async fn saved_draft(&self) -> Draft {
        let value = service::frame(
            &self.state,
            &self.owner,
            include_bytes!("../images/fixtures/64.png").to_vec(),
        )
        .await
        .unwrap();
        Draft {
            first_frame_id: Some(value["id"].as_str().unwrap().into()),
            ..self.draft()
        }
    }
    async fn wait(&self, id: &str) -> String {
        for _ in 0..500 {
            let value = service::status(&self.state, &self.owner, id).await.unwrap();
            let status = value["status"].as_str().unwrap();
            if !["queued", "generating", "saving"].contains(&status) {
                return status.into();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("fixture video did not finish")
    }
}

#[tokio::test]
async fn actual_cpu_encoded_mp4_is_validated_and_incomplete_streams_fail_closed() {
    let f = Fixture::new().await;
    let mut recipe = f.draft().recipe(first_frame("synthetic"));
    let mut progress = Vec::new();
    let bytes = client::generate(&reqwest::Client::new(), &f.base, &recipe, |p| {
        progress.push(p)
    })
    .await
    .unwrap();
    assert_eq!(bytes, CLIP);
    assert!(progress.iter().any(|p| p["step"] == 8));
    recipe.frames = 39;
    assert!(
        client::generate(&reqwest::Client::new(), &f.base, &recipe, |_| {})
            .await
            .is_err()
    );
    recipe.frames = 22;
    f.mock.broken.store(true, Ordering::Relaxed);
    assert!(
        client::generate(&reqwest::Client::new(), &f.base, &recipe, |_| {})
            .await
            .is_err()
    );
}

#[tokio::test]
async fn latest_only_progress_and_independent_previews_do_not_discard_valid_clips() {
    let f = Fixture::new().await;
    let recipe = f.draft().recipe(first_frame("synthetic"));
    let result = json!({"type":"result","mime_type":"video/mp4","width":256,"height":256,"frames":22,"fps":24,"audio":true,"video_base64":STANDARD.encode(CLIP)});
    let preview = json!({"type":"preview","step":0,"steps":8,"width":64,"height":64,"png_base64":STANDARD.encode(include_bytes!("../images/fixtures/64.png"))});
    let events = vec![
        json!({"type":"started"}),
        preview.clone(),
        json!({"type":"progress","phase":"conditioning"}),
        json!({"type":"progress","phase":"sampling","step":8,"steps":8}),
        json!({"type":"progress","phase":"encoding"}),
        preview,
        result.clone(),
    ];
    *f.mock.events.lock().unwrap() = Some(events);
    let mut published = Vec::new();
    assert_eq!(
        client::generate(&reqwest::Client::new(), &f.base, &recipe, |p| published
            .push(p))
        .await
        .unwrap(),
        CLIP
    );
    assert_eq!(
        published
            .iter()
            .filter(|p| p.get("preview").is_some())
            .count(),
        1
    );
    *f.mock.events.lock().unwrap() = Some(vec![json!({"type":"started"}), result.clone()]);
    assert!(
        client::generate(&reqwest::Client::new(), &f.base, &recipe, |_| {})
            .await
            .is_ok()
    );
    for events in [
        vec![json!({"type":"started"}), result.clone(), result.clone()],
        vec![result],
        vec![
            json!({"type":"started"}),
            json!({"type":"progress","phase":"sampling","step":7,"steps":8}),
            json!({"type":"progress","phase":"decoding"}),
        ],
        vec![
            json!({"type":"started"}),
            json!({"type":"progress","phase":"sampling","step":2,"steps":8}),
            json!({"type":"progress","phase":"conditioning"}),
        ],
        vec![
            json!({"type":"started"}),
            json!({"type":"error","error":"synthetic backend content must stay private"}),
        ],
    ] {
        *f.mock.events.lock().unwrap() = Some(events);
        let error = client::generate(&reqwest::Client::new(), &f.base, &recipe, |_| {})
            .await
            .unwrap_err();
        assert!(!error.message().contains("synthetic backend content"));
    }
}

#[tokio::test]
async fn saved_videos_preserve_sources_retries_permissions_and_exact_request_identity() {
    let f = Fixture::new().await;
    let draft = f.saved_draft().await;
    let session = service::create_session(&f.state.db, &f.owner, draft.clone())
        .await
        .unwrap();
    let nonce = Uuid::new_v4().to_string();
    let request = || Request {
        request_id: nonce.clone(),
        draft: draft.clone(),
        retry_of: None,
    };
    let id = service::submit(&f.state, &f.owner, Some(&session.id), request(), None)
        .await
        .unwrap();
    assert_eq!(
        service::submit(&f.state, &f.owner, Some(&session.id), request(), None)
            .await
            .unwrap(),
        id
    );
    assert_eq!(f.wait(&id).await, "complete");
    assert_eq!(
        service::result(&f.state, &f.owner, &id)
            .await
            .unwrap()
            .as_ref(),
        CLIP
    );
    assert!(
        service::session(&f.state.db, &f.other, &session.id)
            .await
            .is_err()
    );
    assert!(service::run(&f.state.db, &f.other, &id).await.is_err());
    assert!(service::result(&f.state, &f.other, &id).await.is_err());
    assert!(
        service::read_frame(
            &f.state.db,
            &f.other,
            draft.first_frame_id.as_ref().unwrap(),
            true
        )
        .await
        .is_err()
    );
    let mut changed = request();
    changed.draft.prompt = "Changed request".into();
    assert!(
        service::submit(&f.state, &f.owner, Some(&session.id), changed, None)
            .await
            .is_err()
    );
    let retry = service::submit(
        &f.state,
        &f.owner,
        Some(&session.id),
        Request {
            request_id: Uuid::new_v4().to_string(),
            draft: Draft {
                prompt: "Should not replace the original".into(),
                first_frame_id: None,
                ..draft.clone()
            },
            retry_of: Some(id.clone()),
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(f.wait(&retry).await, "complete");
    let record = service::run(&f.state.db, &f.owner, &retry).await.unwrap();
    let actual: Draft = service::decode(&record.draft_json).unwrap();
    assert_eq!(actual.prompt, draft.prompt);
    assert_eq!(actual.first_frame_id, draft.first_frame_id);
    assert_eq!(actual.seed, "1");
    assert_eq!(record.group_id, id);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(f.dir.join("videos").join(format!("{retry}.mp4")))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}

#[tokio::test]
async fn all_video_input_modes_preserve_sources_and_reference_steps() {
    let f = Fixture::new().await;
    let source = f.saved_draft().await.first_frame_id.unwrap();
    for (workflow, first, last, count) in [
        (kuno_protocol::MINIMAX_H3_VIDEO_RECIPE, false, false, 0),
        (kuno_protocol::MINIMAX_H3_VIDEO_RECIPE, true, false, 0),
        (kuno_protocol::MINIMAX_H3_VIDEO_RECIPE, false, true, 0),
        (kuno_protocol::MINIMAX_H3_VIDEO_RECIPE, true, true, 0),
        (kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE, false, false, 1),
        (kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE, false, false, 4),
    ] {
        let draft = Draft {
            workflow: workflow.into(),
            first_frame_id: first.then(|| source.clone()),
            last_frame_id: last.then(|| source.clone()),
            reference_image_ids: vec![source.clone(); count],
            ..f.draft()
        };
        let session = service::create_session(&f.state.db, &f.owner, draft.clone())
            .await
            .unwrap();
        let id = service::submit(
            &f.state,
            &f.owner,
            Some(&session.id),
            Request {
                request_id: Uuid::new_v4().to_string(),
                draft,
                retry_of: None,
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(f.wait(&id).await, "complete");
        let requests = f.mock.requests.lock().unwrap();
        let actual = requests.last().unwrap();
        assert_eq!(actual["first_frame"].is_string(), first);
        assert_eq!(actual["last_frame"].is_string(), last);
        assert_eq!(
            actual["reference_images"].as_array().map_or(0, Vec::len),
            count
        );
    }
    let mut defaults = f.saved_draft().await;
    defaults.last_frame_id = Some(source.clone());
    defaults.workflow = kuno_protocol::MINIMAX_H3_VIDEO_RECIPE.into();
    let saved = service::save_preferences(&f.state.db, &f.owner, defaults)
        .await
        .unwrap();
    assert!(saved.image_ids().next().is_none());
    let hosts = service::hosts(&f.state).await.unwrap();
    assert_eq!(
        hosts[0]["capabilities"]["workflows"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[tokio::test]
async fn last_frames_and_references_enforce_ownership_and_shared_cleanup() {
    let f = Fixture::new().await;
    let own = f.saved_draft().await.first_frame_id.unwrap();
    let foreign = service::frame(
        &f.state,
        &f.other,
        include_bytes!("../images/fixtures/64.png").to_vec(),
    )
    .await
    .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    for reference in [false, true] {
        let draft = Draft {
            workflow: if reference {
                kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE
            } else {
                kuno_protocol::MINIMAX_H3_VIDEO_RECIPE
            }
            .into(),
            last_frame_id: (!reference).then(|| foreign.clone()),
            reference_image_ids: if reference {
                vec![foreign.clone()]
            } else {
                vec![]
            },
            ..f.draft()
        };
        let session = service::create_session(&f.state.db, &f.owner, draft.clone())
            .await
            .unwrap();
        assert!(
            service::submit(
                &f.state,
                &f.owner,
                Some(&session.id),
                Request {
                    request_id: Uuid::new_v4().to_string(),
                    draft,
                    retry_of: None
                },
                None
            )
            .await
            .is_err()
        );
    }
    assert!(f.mock.requests.lock().unwrap().is_empty());
    let reference = Draft {
        workflow: kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE.into(),
        reference_image_ids: vec![own.clone()],
        ..f.draft()
    };
    let first = service::create_session(&f.state.db, &f.owner, reference)
        .await
        .unwrap();
    let last = service::create_session(
        &f.state.db,
        &f.owner,
        Draft {
            workflow: kuno_protocol::MINIMAX_H3_VIDEO_RECIPE.into(),
            last_frame_id: Some(own.clone()),
            ..f.draft()
        },
    )
    .await
    .unwrap();
    service::delete_session(&f.state, &f.owner, &first.id)
        .await
        .unwrap();
    assert!(
        service::read_frame(&f.state.db, &f.owner, &own, false)
            .await
            .is_ok()
    );
    service::delete_session(&f.state, &f.owner, &last.id)
        .await
        .unwrap();
    assert!(
        service::read_frame(&f.state.db, &f.owner, &own, false)
            .await
            .is_err()
    );
    assert!(
        service::read_frame(&f.state.db, &f.other, &foreign, false)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn private_text_and_reference_inputs_do_not_create_server_records() {
    let f = Fixture::new().await;
    for reference in [false, true] {
        let inputs = Inputs {
            reference_images: if reference {
                vec![STANDARD.encode(include_bytes!("../images/fixtures/64.png"))]
            } else {
                vec![]
            },
            ..Inputs::default()
        };
        let draft = Draft {
            workflow: if reference {
                kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE
            } else {
                kuno_protocol::MINIMAX_H3_VIDEO_RECIPE
            }
            .into(),
            ..f.draft()
        };
        let id = service::submit(
            &f.state,
            &f.owner,
            None,
            Request {
                request_id: Uuid::new_v4().to_string(),
                draft,
                retry_of: None,
            },
            Some(inputs),
        )
        .await
        .unwrap();
        assert_eq!(f.wait(&id).await, "complete");
    }
    for table in ["video_runs", "video_frames", "video_sessions"] {
        assert_eq!(
            sqlx::query_scalar::<_, i64>(&format!("SELECT count(*) FROM {table}"))
                .fetch_one(&f.state.db)
                .await
                .unwrap(),
            0
        );
    }
}

#[tokio::test]
async fn private_videos_leave_no_saved_content_or_rows_and_are_owner_scoped() {
    let f = Fixture::new().await;
    let id = Uuid::new_v4().to_string();
    let frame = STANDARD.encode(include_bytes!("../images/fixtures/64.png"));
    service::submit(
        &f.state,
        &f.owner,
        None,
        Request {
            request_id: id.clone(),
            draft: f.draft(),
            retry_of: None,
        },
        Some(first_frame(&frame)),
    )
    .await
    .unwrap();
    assert_eq!(f.wait(&id).await, "complete");
    assert_eq!(
        service::result(&f.state, &f.owner, &id)
            .await
            .unwrap()
            .as_ref(),
        CLIP
    );
    assert!(service::status(&f.state, &f.other, &id).await.is_err());
    assert!(service::result(&f.state, &f.other, &id).await.is_err());
    service::stop(&f.state, &f.other, &id).await.unwrap();
    assert_eq!(
        service::status(&f.state, &f.owner, &id).await.unwrap()["status"],
        "complete"
    );
    for table in ["video_runs", "video_frames", "video_sessions"] {
        let count: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM {table}"))
            .fetch_one(&f.state.db)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }
    assert!(!f.dir.join("videos").exists());
    service::stop(&f.state, &f.owner, &id).await.unwrap();
}

#[tokio::test]
async fn cancellation_releases_the_media_slot_and_storage_quota_does_not_save_partial_clips() {
    let f = Fixture::new().await;
    let draft = f.saved_draft().await;
    let session = service::create_session(&f.state.db, &f.owner, draft.clone())
        .await
        .unwrap();
    f.mock.hold.store(true, Ordering::Relaxed);
    let id = service::submit(
        &f.state,
        &f.owner,
        Some(&session.id),
        Request {
            request_id: Uuid::new_v4().to_string(),
            draft: draft.clone(),
            retry_of: None,
        },
        None,
    )
    .await
    .unwrap();
    for _ in 0..100 {
        if !f.mock.requests.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    service::stop(&f.state, &f.owner, &id).await.unwrap();
    assert_eq!(f.wait(&id).await, "cancelled");
    for _ in 0..100 {
        if f.state.image_jobs.slots.available_permits() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(f.state.image_jobs.slots.available_permits(), 1);
    f.mock.hold.store(false, Ordering::Relaxed);
    sqlx::query("UPDATE image_service_settings SET max_user_bytes=1")
        .execute(&f.state.db)
        .await
        .unwrap();
    let failed = service::submit(
        &f.state,
        &f.owner,
        Some(&session.id),
        Request {
            request_id: Uuid::new_v4().to_string(),
            draft: draft.clone(),
            retry_of: None,
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(f.wait(&failed).await, "failed");
    assert!(service::result(&f.state, &f.owner, &failed).await.is_err());
    assert!(!f.dir.join("videos").join(format!("{failed}.mp4")).exists());
    assert!(
        service::read_frame(
            &f.state.db,
            &f.owner,
            draft.first_frame_id.as_ref().unwrap(),
            false
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn deletion_frees_clips_but_not_another_accounts_or_shared_first_frame() {
    let f = Fixture::new().await;
    let draft = f.saved_draft().await;
    let first = service::create_session(&f.state.db, &f.owner, draft.clone())
        .await
        .unwrap();
    let second = service::create_session(&f.state.db, &f.owner, draft.clone())
        .await
        .unwrap();
    let id = service::submit(
        &f.state,
        &f.owner,
        Some(&first.id),
        Request {
            request_id: Uuid::new_v4().to_string(),
            draft: draft.clone(),
            retry_of: None,
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(f.wait(&id).await, "complete");
    assert!(
        service::delete_session(&f.state, &f.other, &first.id)
            .await
            .is_err()
    );
    service::delete_session(&f.state, &f.owner, &first.id)
        .await
        .unwrap();
    assert!(service::result(&f.state, &f.owner, &id).await.is_err());
    assert!(!f.dir.join("videos").join(format!("{id}.mp4")).exists());
    assert!(
        service::read_frame(
            &f.state.db,
            &f.owner,
            draft.first_frame_id.as_ref().unwrap(),
            false
        )
        .await
        .is_ok()
    );
    service::delete_session(&f.state, &f.owner, &second.id)
        .await
        .unwrap();
    assert!(
        service::read_frame(
            &f.state.db,
            &f.owner,
            draft.first_frame_id.as_ref().unwrap(),
            false
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn recovery_keeps_complete_clips_and_removes_only_owned_orphan_file_names() {
    let f = Fixture::new().await;
    let draft = f.saved_draft().await;
    let session = service::create_session(&f.state.db, &f.owner, draft.clone())
        .await
        .unwrap();
    let id = service::submit(
        &f.state,
        &f.owner,
        Some(&session.id),
        Request {
            request_id: Uuid::new_v4().to_string(),
            draft,
            retry_of: None,
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(f.wait(&id).await, "complete");
    let directory = f.dir.join("videos");
    let orphan = directory.join(format!("{}.mp4", Uuid::new_v4()));
    let partial = directory.join(format!("{id}.part"));
    std::fs::write(&orphan, b"partial").unwrap();
    std::fs::write(&partial, b"partial").unwrap();
    std::fs::write(directory.join("unrelated.txt"), b"leave alone").unwrap();
    service::recover(&f.state.db, &f.dir).await.unwrap();
    assert!(!orphan.exists());
    assert!(!partial.exists());
    assert!(directory.join("unrelated.txt").exists());
    assert_eq!(
        service::result(&f.state, &f.owner, &id)
            .await
            .unwrap()
            .as_ref(),
        CLIP
    );
    sqlx::query("UPDATE video_runs SET status='generating' WHERE id=?")
        .bind(&id)
        .execute(&f.state.db)
        .await
        .unwrap();
    service::recover(&f.state.db, &f.dir).await.unwrap();
    assert_eq!(
        service::run(&f.state.db, &f.owner, &id)
            .await
            .unwrap()
            .status,
        "failed"
    );
    assert!(!directory.join(format!("{id}.mp4")).exists());
}

#[tokio::test]
async fn disabled_accounts_and_private_ids_cannot_overwrite_saved_runs() {
    let f = Fixture::new().await;
    let draft = f.saved_draft().await;
    let session = service::create_session(&f.state.db, &f.owner, draft.clone())
        .await
        .unwrap();
    let id = service::submit(
        &f.state,
        &f.owner,
        Some(&session.id),
        Request {
            request_id: Uuid::new_v4().to_string(),
            draft: draft.clone(),
            retry_of: None,
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(f.wait(&id).await, "complete");
    let frame = STANDARD.encode(include_bytes!("../images/fixtures/64.png"));
    assert!(
        service::submit(
            &f.state,
            &f.other,
            None,
            Request {
                request_id: id.clone(),
                draft: draft.clone(),
                retry_of: None
            },
            Some(first_frame(&frame))
        )
        .await
        .is_err()
    );
    service::stop(&f.state, &f.other, &id).await.unwrap();
    assert_eq!(
        service::run(&f.state.db, &f.owner, &id)
            .await
            .unwrap()
            .status,
        "complete"
    );
    sqlx::query("UPDATE users SET is_disabled=1 WHERE id=?")
        .bind(&f.owner)
        .execute(&f.state.db)
        .await
        .unwrap();
    assert!(
        service::submit(
            &f.state,
            &f.owner,
            Some(&session.id),
            Request {
                request_id: Uuid::new_v4().to_string(),
                draft,
                retry_of: None
            },
            None
        )
        .await
        .is_err()
    );
}
#[test]
fn video_settings_validate_without_accepting_image_inference_fields() {
    let mut d = Draft::default();
    service::validate(&d).unwrap();
    d.frames = 57;
    assert!(service::validate(&d).is_err());
    assert!(serde_json::from_str::<Draft>(r#"{"steps":50}"#).is_err());
}
#[test]
fn blank_prompts_use_the_same_safe_motion_default() {
    assert_eq!(
        Draft::default().recipe(first_frame("frame")).prompt,
        "Animate this image with gentle natural motion."
    );
}
#[test]
fn old_or_unready_video_discovery_is_not_usable() {
    let mut c:kuno_protocol::VideoCapabilities=serde_json::from_value(serde_json::json!({"api_version":1,"access":"inference","state":"ready","reason":null,"generate_path":"/kuno/v1/videos/generate","response_formats":["application/x-ndjson"],"workflows":[]})).unwrap();
    assert!(client::supported(&c));
    c.generate_path = "https://unexpected.invalid/".into();
    assert!(!client::supported(&c));
    c.generate_path = "/kuno/v1/videos/generate".into();
    c.state = "unavailable".into();
    assert!(!client::supported(&c));
}
