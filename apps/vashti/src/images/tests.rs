use super::{models::*, service};
use crate::{app_state::AppState, config::Config, startup};
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::HeaderMap,
    routing::{get, post},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Default)]
struct Mock {
    requests: Mutex<Vec<Value>>,
    prompt_requests: Mutex<Vec<Value>>,
    hang: bool,
    disconnected: AtomicBool,
    response_gate: Mutex<Option<Arc<tokio::sync::Notify>>>,
}
struct ImageBody {
    chunks: VecDeque<Vec<u8>>,
    mock: Arc<Mock>,
}
impl futures_util::Stream for ImageBody {
    type Item = Result<Vec<u8>, std::convert::Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(chunk) = self.chunks.pop_front() {
            Poll::Ready(Some(Ok(chunk)))
        } else if self.mock.hang {
            Poll::Pending
        } else {
            Poll::Ready(None)
        }
    }
}
impl Drop for ImageBody {
    fn drop(&mut self) {
        self.mock.disconnected.store(true, Ordering::SeqCst);
    }
}
async fn image(
    State(mock): State<Arc<Mock>>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> impl axum::response::IntoResponse {
    assert!(!headers.contains_key("authorization"));
    assert!(!headers.contains_key("x-kuno-image-preview-id"));
    let steps = request["steps"].as_u64().unwrap();
    mock.requests.lock().await.push(request);
    let gate = mock.response_gate.lock().await.clone();
    if let Some(gate) = gate {
        gate.notified().await;
    }
    let mut chunks = VecDeque::from([
        b"{\"type\":\"started\"}\n".to_vec(),
        format!("{}\n",json!({"type":"preview","step":1,"steps":steps,"width":64,"height":64,"png_base64":STANDARD.encode(include_bytes!("fixtures/64.png"))})).into_bytes(),
    ]);
    if !mock.hang {
        chunks.push_back(format!("{}\n",json!({"type":"result","mime_type":"image/png","png_base64":STANDARD.encode(include_bytes!("fixtures/64.png"))})).into_bytes());
    }
    (
        [("content-type", "application/x-ndjson")],
        Body::from_stream(ImageBody { chunks, mock }),
    )
}
async fn llm(State(mock): State<Arc<Mock>>, Json(value): Json<Value>) -> String {
    assert!(
        value["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("blue")
    );
    let mut prompts = mock.prompt_requests.lock().await;
    prompts.push(value);
    format!(
        "{}\n",
        json!({"message":{"content":format!("A white weasel in neutral daylight. Variation {}.", prompts.len())},"done":true})
    )
}
async fn setup(hang: bool) -> (AppState, Arc<Mock>, tokio::task::JoinHandle<()>) {
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
    for user in ["alice", "bob"] {
        sqlx::query("INSERT INTO users (id,username,password_hash,created_at,updated_at) VALUES (?,?,'hash',0,0)").bind(user).bind(user).execute(&pool).await.unwrap();
    }
    let mock = Arc::new(Mock {
        hang,
        ..Default::default()
    });
    let app = Router::new()
        .route(
            "/kuno/v1/capabilities",
            get(|| async { Json(super::kuno::tests::capabilities()) }),
        )
        .route("/kuno/v1/images/generate", post(image))
        .route("/api/chat", post(llm))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    sqlx::query("INSERT INTO ollama_backends (id,name,base_url,created_at,updated_at) VALUES ('backend','Mock',?,0,0)").bind(format!("http://{address}")).execute(&pool).await.unwrap();
    sqlx::query("UPDATE image_service_settings SET backend_id='backend'")
        .execute(&pool)
        .await
        .unwrap();
    let state = AppState::new(
        Config::from_env().unwrap(),
        pool,
        reqwest::Client::new(),
        "test".into(),
    );
    (state, mock, server)
}
fn draft() -> Draft {
    let mut recipe = Preferences::default().generate.recipe;
    recipe.width = 64;
    recipe.height = 64;
    recipe.steps = 1;
    recipe.prompt = "a weasel".into();
    Draft {
        backend_id: None,
        recipe,
        input_ids: vec![],
        original_input_ids: vec![],
        parent_result_id: None,
        prompt_version_id: None,
        seed_mode: SeedMode::Random,
        count: Some(1),
        auto_improve: false,
        prompt_help: None,
    }
}
fn request(draft: Draft) -> RunRequest {
    RunRequest {
        chat_target: None,
        client_request_id: Uuid::new_v4().to_string(),
        draft,
        improve_only: false,
        retry_result_id: None,
        fixed_endless_confirmed: false,
    }
}

#[tokio::test]
async fn account_defaults_are_explicit_owner_scoped_and_backward_compatible() {
    let (state, _, server) = setup(false).await;
    let mut legacy = serde_json::to_value(Preferences::default()).unwrap();
    for operation in ["generate", "edit"] {
        let profile = legacy[operation].as_object_mut().unwrap();
        for key in [
            "backend_id",
            "count",
            "auto_improve",
            "prompt_help",
            "edit_mode",
        ] {
            profile.remove(key);
        }
    }
    sqlx::query("INSERT INTO image_preferences VALUES ('alice', ?)")
        .bind(legacy.to_string())
        .execute(&state.db)
        .await
        .unwrap();
    let mut preferences = service::preferences(&state.db, "alice").await.unwrap();
    assert_eq!(preferences.generate.count, Some(1));
    assert!(!preferences.generate.auto_improve);
    assert!(preferences.edit.edit_mode == EditMode::Iterative);

    let session = service::create_session(&state.db, "alice", "Test defaults", "generate", draft())
        .await
        .unwrap();
    assert_eq!(
        service::preferences(&state.db, "alice")
            .await
            .unwrap()
            .generate
            .recipe
            .steps,
        50
    );
    preferences.generate.count = Some(3);
    preferences.generate.recipe.steps = 40;
    preferences.generate.backend_id = Some("backend".into());
    preferences.generate.auto_improve = true;
    preferences.edit.edit_mode = EditMode::Original;
    preferences.generate.prompt_help = Some(PromptHelp {
        backend_id: "backend".into(),
        model_name: "helper".into(),
        guidance: "Add detail".into(),
    });
    service::save_preferences(&state.db, "alice", preferences.clone())
        .await
        .unwrap();
    let saved = service::preferences(&state.db, "alice").await.unwrap();
    assert_eq!(saved.generate.count, Some(3));
    assert_eq!(saved.generate.recipe.steps, 40);
    assert!(saved.edit.edit_mode == EditMode::Original);
    assert_eq!(saved.generate.prompt_help.unwrap().guidance, "Add detail");
    assert_eq!(
        service::preferences(&state.db, "bob")
            .await
            .unwrap()
            .generate
            .recipe
            .steps,
        50
    );
    assert_eq!(
        service::get_session(&state.db, "alice", &session.id)
            .await
            .unwrap()
            .draft
            .recipe
            .steps,
        1
    );

    for count in [Some(0), Some(101)] {
        preferences.generate.count = count;
        assert!(
            service::save_preferences(&state.db, "alice", preferences.clone())
                .await
                .is_err()
        );
    }
    preferences.generate.count = None;
    assert!(
        service::save_preferences(&state.db, "alice", preferences.clone())
            .await
            .is_ok()
    );
    preferences.generate.prompt_help = None;
    assert!(
        service::save_preferences(&state.db, "alice", preferences)
            .await
            .is_err()
    );
    server.abort();
}
async fn completed(state: &AppState, run: &str) -> Run {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let value = service::get_run(&state.db, "alice", run).await.unwrap();
            if ["complete", "stopped", "failed"].contains(&value.status.as_str()) {
                return value;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

#[test]
fn model_permissions_cannot_be_bypassed_with_dimension_aliases() {
    let mut p = Preferences::default().generate;
    p.model_may_change.clear();
    let (recipe, ignored) =
        service::resolve_tool_recipe(&p, "weasel".into(), &json!({"width":2048,"steps":1}))
            .unwrap();
    assert_eq!(recipe.width, 1024);
    assert_eq!(recipe.steps, 50);
    assert_eq!(ignored.len(), 2);
    p.model_may_change.insert("size".into());
    let (recipe, _) =
        service::resolve_tool_recipe(&p, "weasel".into(), &json!({"width":1184,"height":1776}))
            .unwrap();
    assert_eq!(recipe.width, 1184);
    assert!(
        service::resolve_tool_recipe(&p, "weasel".into(), &json!({"workflow":"arbitrary"}))
            .is_err()
    );
    assert!(service::resolve_tool_recipe(&p, "weasel".into(), &json!({"width":1185})).is_err());
}
#[tokio::test]
async fn flows_capture_hosts_allocate_new_seeds_and_replay_submissions_exactly() {
    let (state, mock, server) = setup(false).await;
    sqlx::query("UPDATE image_service_settings SET backend_id=NULL")
        .execute(&state.db)
        .await
        .unwrap();
    let mut d = draft();
    d.seed_mode = SeedMode::Increment;
    d.recipe.seed = "40".into();
    d.backend_id = Some("backend".into());
    let session = service::create_session(&state.db, "alice", "New images", "generate", d.clone())
        .await
        .unwrap();
    let first = service::submit(&state, "alice", &session.id, request(d.clone()))
        .await
        .unwrap();
    assert_eq!(first.request.draft.backend_id.as_deref(), Some("backend"));
    assert_eq!(completed(&state, &first.id).await.status, "complete");
    let original = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .remove(0);
    for expected in [41, 42] {
        let mut retry = request(d.clone());
        retry.draft.seed_mode = SeedMode::Fixed;
        retry.draft.backend_id = Some("does-not-exist".into());
        retry.draft.recipe.prompt = "must not replace the recorded prompt".into();
        retry.retry_result_id = Some(original.id.clone());
        let run = service::submit(&state, "alice", &session.id, retry.clone())
            .await
            .unwrap();
        assert_eq!(run.request.draft.recipe.seed, expected.to_string());
        assert_eq!(run.request.draft.backend_id.as_deref(), Some("backend"));
        assert_eq!(completed(&state, &run.id).await.status, "complete");
        assert_eq!(
            service::submit(&state, "alice", &session.id, retry.clone())
                .await
                .unwrap()
                .id,
            run.id
        );
        retry.draft.recipe.prompt = "different payload with same id".into();
        assert!(
            service::submit(&state, "alice", &session.id, retry)
                .await
                .is_err()
        );
    }
    let variants = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap();
    assert!(
        variants
            .iter()
            .all(|result| result.origin_run_id == first.id)
    );
    assert_eq!(mock.requests.lock().await.len(), 3);
    let mut exhausted = original.recipe.clone();
    exhausted.seed = i64::MAX.to_string();
    sqlx::query("UPDATE image_results SET recipe_json=? WHERE id=?")
        .bind(serde_json::to_string(&exhausted).unwrap())
        .bind(&original.id)
        .execute(&state.db)
        .await
        .unwrap();
    let mut retry = request(d);
    retry.retry_result_id = Some(original.id.clone());
    assert!(
        service::submit(&state, "alice", &session.id, retry.clone())
            .await
            .is_err()
    );
    retry.draft.seed_mode = SeedMode::Random;
    let run = service::submit(&state, "alice", &session.id, retry)
        .await
        .unwrap();
    assert_eq!(completed(&state, &run.id).await.status, "complete");
    server.abort();
}

#[tokio::test]
async fn new_branches_persist_selection_without_stealing_later_navigation() {
    let (state, mock, server) = setup(false).await;
    let d = draft();
    let session = service::create_session(&state.db, "alice", "Branches", "generate", d.clone())
        .await
        .unwrap();
    let first_run = service::submit(&state, "alice", &session.id, request(d.clone()))
        .await
        .unwrap();
    assert_eq!(completed(&state, &first_run.id).await.status, "complete");
    let first = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .remove(0);
    let gate = Arc::new(tokio::sync::Notify::new());
    *mock.response_gate.lock().await = Some(gate.clone());
    let submission = request(d.clone());
    let next = service::submit(&state, "alice", &session.id, submission.clone())
        .await
        .unwrap();
    assert!(
        service::get_session(&state.db, "alice", &session.id)
            .await
            .unwrap()
            .active_result_id
            .is_none()
    );
    service::select_result(&state.db, "alice", &first.id)
        .await
        .unwrap();
    assert_eq!(
        service::submit(&state, "alice", &session.id, submission)
            .await
            .unwrap()
            .id,
        next.id
    );
    gate.notify_one();
    assert_eq!(completed(&state, &next.id).await.status, "complete");
    assert_eq!(
        service::get_session(&state.db, "alice", &session.id)
            .await
            .unwrap()
            .active_result_id
            .as_deref(),
        Some(first.id.as_str())
    );
    *mock.response_gate.lock().await = None;
    let third = service::submit(&state, "alice", &session.id, request(d))
        .await
        .unwrap();
    assert_eq!(completed(&state, &third.id).await.status, "complete");
    let result = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .into_iter()
        .find(|result| result.run_id == third.id)
        .unwrap();
    assert_eq!(
        service::get_session(&state.db, "alice", &session.id)
            .await
            .unwrap()
            .active_result_id
            .as_deref(),
        Some(result.id.as_str())
    );
    let mut edit = draft();
    edit.recipe.workflow = kuno_protocol::QWEN21_EDIT_RECIPE.into();
    edit.parent_result_id = Some(first.id);
    edit.input_ids = vec![first.asset_id];
    let edit_run = service::submit(&state, "alice", &session.id, request(edit))
        .await
        .unwrap();
    assert_eq!(completed(&state, &edit_run.id).await.status, "complete");
    let result = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .into_iter()
        .find(|result| result.run_id == edit_run.id)
        .unwrap();
    assert_eq!(
        service::get_session(&state.db, "alice", &session.id)
            .await
            .unwrap()
            .active_result_id
            .as_deref(),
        Some(result.id.as_str())
    );
    server.abort();
}

#[tokio::test]
async fn image_edits_and_branch_selection_preserve_exact_ancestry_and_titles() {
    let (state, _, server) = setup(false).await;
    let mut d = draft();
    d.seed_mode = SeedMode::Increment;
    let session = service::create_session(&state.db, "alice", "New images", "generate", d.clone())
        .await
        .unwrap();
    let run = service::submit(&state, "alice", &session.id, request(d.clone()))
        .await
        .unwrap();
    assert_eq!(completed(&state, &run.id).await.status, "complete");
    // The title job is asynchronous and must not be overwritten by subsequent draft saves.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if service::get_session(&state.db, "alice", &session.id)
                .await
                .unwrap()
                .title
                == "a weasel"
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let first = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .remove(0);
    d.recipe.workflow = kuno_protocol::QWEN21_EDIT_RECIPE.into();
    d.input_ids = vec![first.asset_id.clone()];
    d.original_input_ids = d.input_ids.clone();
    d.parent_result_id = Some(first.id.clone());
    let saved = service::update_session(
        &state.db,
        "alice",
        &session.id,
        session.revision,
        None,
        "iterative",
        d.clone(),
    )
    .await
    .unwrap();
    assert_eq!(saved.title, "a weasel");
    let edit = service::submit(&state, "alice", &session.id, request(d.clone()))
        .await
        .unwrap();
    assert_eq!(completed(&state, &edit.id).await.status, "complete");
    let child = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .remove(0);
    assert_eq!(child.parent_result_id.as_deref(), Some(first.id.as_str()));
    let mut retry = request(d.clone());
    retry.retry_result_id = Some(first.id.clone());
    let regen = service::submit(&state, "alice", &session.id, retry)
        .await
        .unwrap();
    assert_eq!(completed(&state, &regen.id).await.status, "complete");
    let newer = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .remove(0);
    assert_eq!(newer.input_ids.len(), 0);
    service::select_result(&state.db, "alice", &child.id)
        .await
        .unwrap();
    let results = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap();
    assert!(
        results
            .iter()
            .find(|result| result.id == first.id)
            .unwrap()
            .selected
    );
    assert!(
        !results
            .iter()
            .find(|result| result.id == newer.id)
            .unwrap()
            .selected
    );
    assert_eq!(
        service::get_session(&state.db, "alice", &session.id)
            .await
            .unwrap()
            .active_result_id
            .as_deref(),
        Some(child.id.as_str())
    );
    assert!(
        service::select_result(&state.db, "bob", &child.id)
            .await
            .is_err()
    );
    d.input_ids = vec![child.asset_id.clone()];
    d.parent_result_id = Some(child.id.clone());
    service::update_session(
        &state.db,
        "alice",
        &session.id,
        saved.revision,
        None,
        "original",
        d.clone(),
    )
    .await
    .unwrap();
    let original_edit = service::submit(&state, "alice", &session.id, request(d))
        .await
        .unwrap();
    assert_eq!(original_edit.request.draft.input_ids, vec![first.asset_id]);
    assert_eq!(
        completed(&state, &original_edit.id).await.status,
        "complete"
    );
    server.abort();
}
#[tokio::test]
async fn batches_are_independent_retries_stack_and_inputs_are_owner_scoped() {
    let (state, mock, server) = setup(false).await;
    let mut d = draft();
    d.count = Some(2);
    let session = service::create_session(&state.db, "alice", "Weasels", "generate", d.clone())
        .await
        .unwrap();
    let req = request(d);
    let run = service::submit(&state, "alice", &session.id, req.clone())
        .await
        .unwrap();
    let duplicate = service::submit(&state, "alice", &session.id, req)
        .await
        .unwrap();
    assert_eq!(run.id, duplicate.id);
    let done = completed(&state, &run.id).await;
    assert_eq!(done.status, "complete", "{:?}", done.error);
    assert_eq!(done.completed_count, 2);
    let original = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap();
    assert_eq!(original.len(), 2);
    assert_ne!(original[0].group_id, original[1].group_id);
    let mut retry = request(draft());
    retry.retry_result_id = Some(original[0].id.clone());
    retry.draft.recipe.prompt = "must not replace recorded prompt".into();
    let run = service::submit(&state, "alice", &session.id, retry)
        .await
        .unwrap();
    assert_eq!(completed(&state, &run.id).await.status, "complete");
    let variants = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap();
    assert_eq!(variants.len(), 3);
    assert_eq!(variants[0].group_id, original[0].group_id);
    assert_eq!(variants[0].recipe.prompt, "a weasel");
    assert_ne!(variants[0].recipe.seed, original[0].recipe.seed);
    assert_eq!(
        service::gallery(&state.db, "alice", 0).await.unwrap().len(),
        2
    );
    service::copy_to_gallery(&state.db, "alice", &original[0].asset_id)
        .await
        .unwrap();
    assert_eq!(
        service::gallery(&state.db, "alice", 0).await.unwrap().len(),
        3
    );
    assert!(
        service::asset(&state.db, "bob", &original[0].asset_id)
            .await
            .is_err()
    );
    assert!(
        service::get_session(&state.db, "bob", &session.id)
            .await
            .is_err()
    );
    assert_eq!(mock.requests.lock().await.len(), 3);
    // Changing a draft from an old tab must not overwrite a newer version.
    service::update_session(
        &state.db,
        "alice",
        &session.id,
        1,
        Some("Updated"),
        "generate",
        draft(),
    )
    .await
    .unwrap();
    assert!(
        service::update_session(
            &state.db,
            "alice",
            &session.id,
            1,
            Some("Stale"),
            "generate",
            draft()
        )
        .await
        .is_err()
    );
    server.abort();
}
#[tokio::test]
async fn auto_improvement_uses_original_for_each_image_and_preserves_guidance() {
    let (state, mock, server) = setup(false).await;
    crate::backends::service::ensure_model_record(&state.db, "backend", "model")
        .await
        .unwrap();
    let mut d = draft();
    d.count = Some(2);
    d.auto_improve = true;
    d.prompt_help = Some(PromptHelp {
        backend_id: "backend".into(),
        model_name: "model".into(),
        guidance: "The image keeps coming out blue. Correct the color cast.".into(),
    });
    let session = service::create_session(&state.db, "alice", "Prompt test", "generate", d.clone())
        .await
        .unwrap();
    let run = service::submit(&state, "alice", &session.id, request(d.clone()))
        .await
        .unwrap();
    let done = completed(&state, &run.id).await;
    assert_eq!(done.status, "complete", "{:?}", done.error);
    let versions = service::history(&state.db, "alice", &session.id)
        .await
        .unwrap();
    assert_eq!(versions.len(), 3);
    assert_eq!(versions[0].source, "improved");
    assert_eq!(versions[2].prompt, "a weasel");
    assert_eq!(
        versions[0].parent_id.as_deref(),
        Some(versions[2].id.as_str())
    );
    assert_eq!(versions[1].parent_id, versions[0].parent_id);
    assert!(versions[0].guidance.contains("blue"));
    {
        let images = mock.requests.lock().await;
        assert_eq!(images.len(), 2);
        assert_ne!(images[0]["prompt"], images[1]["prompt"]);
    }
    // Another Send starts from the same source; Retry repeats the recorded recipe.
    let next = service::submit(&state, "alice", &session.id, request(d.clone()))
        .await
        .unwrap();
    assert_eq!(completed(&state, &next.id).await.status, "complete");
    let results = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap();
    let mut retry = request(d);
    retry.retry_result_id = Some(results[0].id.clone());
    let retry = service::submit(&state, "alice", &session.id, retry)
        .await
        .unwrap();
    assert_eq!(completed(&state, &retry.id).await.status, "complete");
    let prompts = mock.prompt_requests.lock().await;
    assert_eq!(prompts.len(), 4);
    for prompt in prompts.iter() {
        let input: Value =
            serde_json::from_str(prompt["messages"][1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(input["prompt"], "a weasel");
        assert!(input["guidance"].as_str().unwrap().contains("blue"));
    }
    assert_eq!(
        mock.requests.lock().await.last().unwrap()["prompt"],
        results[0].recipe.prompt
    );
    // A suggestion never changes the user's draft on the server without acceptance.
    assert_eq!(
        service::get_session(&state.db, "alice", &session.id)
            .await
            .unwrap()
            .draft
            .recipe
            .prompt,
        "a weasel"
    );
    server.abort();
}
#[tokio::test]
async fn cancellation_stops_endless_without_saving_an_inflight_image() {
    let (state, mock, server) = setup(true).await;
    let mut d = draft();
    d.count = None;
    let session = service::create_session(&state.db, "alice", "Endless", "generate", d.clone())
        .await
        .unwrap();
    let run = service::submit(&state, "alice", &session.id, request(d))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while service::preview(&state, &run.id).is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    service::stop(&state, "alice", &run.id).await.unwrap();
    assert_eq!(completed(&state, &run.id).await.status, "stopped");
    assert!(service::preview(&state, &run.id).is_none());
    tokio::time::timeout(Duration::from_secs(2), async {
        while !mock.disconnected.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        service::results(&state.db, "alice", &session.id, 0)
            .await
            .unwrap()
            .is_empty()
    );
    server.abort();
}

#[tokio::test]
async fn forward_migration_removes_legacy_secrets_without_changing_settings_or_images() {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(":memory:")
                .foreign_keys(true),
        )
        .await
        .unwrap();
    let mut old = sqlx::migrate!("./migrations");
    old.migrations = old
        .iter()
        .filter(|m| m.version < 20260930000000)
        .cloned()
        .collect::<Vec<_>>()
        .into();
    old.run(&pool).await.unwrap();
    sqlx::query("INSERT INTO ollama_backends (id,name,base_url,created_at,updated_at) VALUES ('backend','Test','http://localhost:1',0,0)").execute(&pool).await.unwrap();
    sqlx::query("UPDATE image_service_settings SET backend_id='backend',token='obsolete-test-credential',max_user_bytes=536870912").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO users (id,username,password_hash,created_at,updated_at) VALUES ('alice','alice','hash',0,0)").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO image_assets VALUES ('asset','alice','image/png',?,64,64,?,0)")
        .bind(include_bytes!("fixtures/64.png").len() as i64)
        .bind(include_bytes!("fixtures/64.png").as_slice())
        .execute(&pool)
        .await
        .unwrap();
    startup::migrations::run(&pool).await.unwrap();
    let columns = sqlx::query("PRAGMA table_info(image_service_settings)")
        .fetch_all(&pool)
        .await
        .unwrap();
    assert!(
        !columns
            .iter()
            .any(|row| row.get::<String, _>("name") == "token")
    );
    let row = sqlx::query("SELECT backend_id,max_user_bytes FROM image_service_settings")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(row.get::<String, _>("backend_id"), "backend");
    assert_eq!(row.get::<i64, _>("max_user_bytes"), 536870912);
    let body: Vec<u8> = sqlx::query_scalar("SELECT body FROM image_assets WHERE id='asset'")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(body, include_bytes!("fixtures/64.png"));
}

#[tokio::test]
async fn settings_and_discovery_remain_session_and_admin_scoped_without_a_token() {
    let (state, _, inference) = setup(false).await;
    sqlx::query("UPDATE users SET role='admin' WHERE id='alice'")
        .execute(&state.db)
        .await
        .unwrap();
    crate::permissions::service::ensure_tool_records(&state.db)
        .await
        .unwrap();
    let alice = crate::auth::service::create_session(&state.db, "alice", 3600, None, None)
        .await
        .unwrap();
    let bob = crate::auth::service::create_session(&state.db, "bob", 3600, None, None)
        .await
        .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/api/images", listener.local_addr().unwrap());
    let app = Router::new()
        .nest("/api/images", super::handlers::router())
        .with_state(state.clone());
    let api = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();
    assert_eq!(
        client
            .get(format!("{url}/settings"))
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::UNAUTHORIZED
    );
    let cookie = |id: &str| format!("{}={id}", state.config.session_cookie_name);
    let hosts: Value = client
        .get(format!("{url}/hosts"))
        .header("cookie", cookie(&bob.id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(hosts[0]["backend"]["id"], "backend");
    assert_eq!(hosts[0]["service"]["state"], "ready");
    assert!(hosts[0]["backend"].get("base_url").is_none());
    let image_session =
        service::create_session(&state.db, "alice", "API flow", "generate", draft())
            .await
            .unwrap();
    let image_run = service::submit(&state, "alice", &image_session.id, request(draft()))
        .await
        .unwrap();
    assert_eq!(completed(&state, &image_run.id).await.status, "complete");
    let image_result = service::results(&state.db, "alice", &image_session.id, 0)
        .await
        .unwrap()
        .remove(0);
    assert_eq!(
        client
            .patch(format!("{url}/sessions/{}/title", image_session.id))
            .header("cookie", cookie(&bob.id))
            .json(&json!({"title":"Forbidden"}))
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::NOT_FOUND
    );
    assert_eq!(
        client
            .patch(format!("{url}/sessions/{}/title", image_session.id))
            .header("cookie", cookie(&alice.id))
            .json(&json!({"title":"Renamed flow"}))
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::OK
    );
    assert_eq!(
        client
            .delete(format!("{url}/sessions/{}", image_session.id))
            .header("cookie", cookie(&alice.id))
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::OK
    );
    assert!(
        service::sessions(&state.db, "alice", 0)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        service::get_session(&state.db, "alice", &image_session.id)
            .await
            .unwrap()
            .title,
        "Renamed flow"
    );
    assert_eq!(
        service::gallery(&state.db, "alice", 0).await.unwrap().len(),
        1
    );
    assert!(
        service::asset(&state.db, "alice", &image_result.asset_id)
            .await
            .is_ok()
    );
    let settings: Value = client
        .get(format!("{url}/settings"))
        .header("cookie", cookie(&bob.id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(settings["service"]["state"], "ready");
    assert!(settings.get("token").is_none());
    assert!(settings.get("token_configured").is_none());
    assert_eq!(
        client
            .get(format!("{url}/capabilities?backend_id=backend"))
            .header("cookie", cookie(&bob.id))
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::OK
    );
    let input = json!({"backend_id":"backend","max_user_bytes":536870912});
    assert_eq!(
        client
            .put(format!("{url}/settings"))
            .header("cookie", cookie(&bob.id))
            .json(&input)
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::FORBIDDEN
    );
    let saved: Value = client
        .put(format!("{url}/settings"))
        .header("cookie", cookie(&alice.id))
        .json(&input)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(saved["max_user_bytes"], 536870912);
    assert_eq!(saved["service"]["state"], "ready");
    let mut obsolete = input;
    obsolete["token"] = json!("obsolete-test-credential");
    assert_eq!(
        client
            .put(format!("{url}/settings"))
            .header("cookie", cookie(&alice.id))
            .json(&obsolete)
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::UNPROCESSABLE_ENTITY
    );
    assert_eq!(super::chat::schemas(&state, "bob").await.unwrap().len(), 2);
    sqlx::query("UPDATE image_service_settings SET backend_id=NULL")
        .execute(&state.db)
        .await
        .unwrap();
    assert_eq!(super::chat::schemas(&state, "bob").await.unwrap().len(), 2);
    assert_eq!(
        service::discovery(&state, None).await.unwrap().state,
        "ready"
    );
    sqlx::query("UPDATE ollama_backends SET is_enabled=0")
        .execute(&state.db)
        .await
        .unwrap();
    assert!(
        super::chat::schemas(&state, "bob")
            .await
            .unwrap()
            .is_empty()
    );
    inference.abort();
    api.abort();
}
#[tokio::test]
async fn restart_does_not_resume_endless_and_fixed_seed_requires_confirmation() {
    let (state, _, server) = setup(false).await;
    let mut d = draft();
    d.count = None;
    d.seed_mode = SeedMode::Fixed;
    let session = service::create_session(&state.db, "alice", "Fixed", "generate", d.clone())
        .await
        .unwrap();
    let error = service::submit(&state, "alice", &session.id, request(d))
        .await
        .err()
        .unwrap();
    assert_eq!(error.code(), "fixed_seed_endless");
    sqlx::query("INSERT INTO image_runs (id,user_id,session_id,client_request_id,request_json,status,created_at,updated_at) VALUES ('old','alice',?,'id','{}','generating',0,0)").bind(session.id).execute(&state.db).await.unwrap();
    service::recover(&state.db).await.unwrap();
    let row = sqlx::query("SELECT status FROM image_runs WHERE id='old'")
        .fetch_one(&state.db)
        .await
        .unwrap();
    assert_eq!(row.get::<String, _>("status"), "interrupted");
    server.abort();
}

#[tokio::test]
async fn accepting_or_restoring_prompts_preserves_newer_user_text() {
    let (state, _, server) = setup(false).await;
    let session = service::create_session(&state.db, "alice", "Draft", "generate", draft())
        .await
        .unwrap();
    let mut changed = draft();
    changed.recipe.prompt = "My newer, carefully written prompt".into();
    let saved = service::update_session(
        &state.db,
        "alice",
        &session.id,
        session.revision,
        Some("Draft"),
        "generate",
        changed,
    )
    .await
    .unwrap();
    let mut improved = saved.draft;
    improved.recipe.prompt = "An accepted improved prompt".into();
    service::update_session(
        &state.db,
        "alice",
        &session.id,
        saved.revision,
        Some("Draft"),
        "generate",
        improved,
    )
    .await
    .unwrap();
    let history = service::history(&state.db, "alice", &session.id)
        .await
        .unwrap();
    for prompt in [
        "a weasel",
        "My newer, carefully written prompt",
        "An accepted improved prompt",
    ] {
        assert!(history.iter().any(|v| v.prompt == prompt));
    }
    server.abort();
}

#[tokio::test]
async fn chat_image_is_committed_with_attachment_and_shared_without_copying_files() {
    let (state, _, server) = setup(false).await;
    sqlx::query("INSERT INTO chats (id,user_id,default_backend_id,default_model_name,title,created_at,updated_at,last_message_at) VALUES ('chat','alice','backend','model','Images',0,0,0)").execute(&state.db).await.unwrap();
    sqlx::query("INSERT INTO chat_messages (id,chat_id,role,active_revision_id,created_at,updated_at) VALUES ('message','chat','assistant','revision',0,0)").execute(&state.db).await.unwrap();
    sqlx::query("INSERT INTO chat_message_revisions (id,message_id,content_text,source,created_at) VALUES ('revision','message','','original',0)").execute(&state.db).await.unwrap();
    let session = service::create_session(&state.db, "alice", "Chat image", "generate", draft())
        .await
        .unwrap();
    sqlx::query("UPDATE image_sessions SET chat_id='chat' WHERE id=?")
        .bind(&session.id)
        .execute(&state.db)
        .await
        .unwrap();
    let mut req = request(draft());
    req.chat_target = Some(ChatTarget {
        chat_id: "chat".into(),
        message_id: "message".into(),
        revision_id: "revision".into(),
    });
    let run = service::submit(&state, "alice", &session.id, req)
        .await
        .unwrap();
    let done = completed(&state, &run.id).await;
    assert_eq!(done.status, "complete", "{:?}", done.error);
    let attachments = crate::uploads::service::list_revision_attachments(
        &state.db, "alice", "chat", "message", "revision",
    )
    .await
    .unwrap();
    assert_eq!(attachments.len(), 1);
    let (_, bytes) = crate::uploads::service::get_attachment_file(
        &state.db,
        std::path::Path::new("/does-not-exist"),
        "alice",
        &attachments[0].id,
    )
    .await
    .unwrap();
    assert_eq!(bytes, include_bytes!("fixtures/64.png"));
    let (text, images) = crate::uploads::service::prompt_attachment_payload(
        &state.db,
        std::path::Path::new("/does-not-exist"),
        "alice",
        "chat",
        "message",
        "revision",
    )
    .await
    .unwrap();
    assert!(text.contains(&attachments[0].id));
    assert_eq!(images.len(), 1);
    let mut tx = state.db.begin().await.unwrap();
    crate::uploads::service::attach_referenced_attachments(
        &mut tx,
        std::path::Path::new("/does-not-exist"),
        "alice",
        "chat",
        "message",
        "revision",
        &[attachments[0].id.clone()],
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();
    let image = service::results(&state.db, "alice", &session.id, 0)
        .await
        .unwrap()
        .remove(0);
    service::copy_to_gallery(&state.db, "alice", &image.asset_id)
        .await
        .unwrap();
    sqlx::query("DELETE FROM chats WHERE id='chat'")
        .execute(&state.db)
        .await
        .unwrap();
    assert_eq!(
        service::gallery(&state.db, "alice", 0).await.unwrap().len(),
        1
    );
    assert!(
        service::asset(&state.db, "alice", &image.asset_id)
            .await
            .is_ok()
    );
    sqlx::query("DELETE FROM users WHERE id='alice'")
        .execute(&state.db)
        .await
        .unwrap();
    assert!(
        service::asset(&state.db, "alice", &image.asset_id)
            .await
            .is_err()
    );
    server.abort();
}

#[tokio::test]
async fn private_images_return_only_to_owner_without_saved_content() {
    use super::transient;
    let (state, mock, server) = setup(false).await;
    let mut preferences = Preferences::default();
    preferences.generate.recipe.width = 64;
    preferences.generate.recipe.height = 64;
    service::save_preferences(&state.db, "alice", preferences)
        .await
        .unwrap();
    let id = transient::submit(
        &state,
        "alice",
        transient::Request {
            operation: "generate".into(),
            prompt: "private canary prompt".into(),
            settings: json!({}),
            reference_images: vec![],
        },
    )
    .await
    .unwrap();
    assert!(
        transient::read_status(&state.image_jobs.private, "bob", &id)
            .await
            .is_err()
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let status = transient::read_status(&state.image_jobs.private, "alice", &id)
                .await
                .unwrap();
            if status["status"] == "complete" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert!(
        transient::take_result(&state.image_jobs.private, "bob", &id)
            .await
            .is_err()
    );
    let png = transient::take_result(&state.image_jobs.private, "alice", &id)
        .await
        .unwrap();
    assert_eq!(png, include_bytes!("fixtures/64.png"));
    assert!(
        transient::read_status(&state.image_jobs.private, "alice", &id)
            .await
            .is_err()
    );
    assert_eq!(mock.requests.lock().await.len(), 1);
    for table in [
        "image_sessions",
        "image_runs",
        "image_assets",
        "image_prompt_versions",
        "image_results",
        "image_result_groups",
        "image_gallery",
        "attachments",
    ] {
        let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
            .fetch_one(&state.db)
            .await
            .unwrap();
        assert_eq!(count, 0, "private content leaked into {table}");
    }
    server.abort();
}
