use super::{models::*, service};
use crate::{app_state::AppState, config::Config, startup};
use axum::{Json, Router, extract::State, routing::post};
use serde_json::{Value, json};
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use std::{sync::Arc, time::Duration};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Default)]
struct Mock {
    requests: Mutex<Vec<Value>>,
    hang: bool,
}
async fn image(
    State(mock): State<Arc<Mock>>,
    Json(request): Json<Value>,
) -> impl axum::response::IntoResponse {
    mock.requests.lock().await.push(request);
    if mock.hang {
        std::future::pending::<()>().await;
    }
    (
        [("content-type", "image/png")],
        include_bytes!("fixtures/64.png").as_slice(),
    )
}
async fn llm(Json(value): Json<Value>) -> String {
    assert!(
        value["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("blue")
    );
    format!(
        "{}\n",
        json!({"message":{"content":"A white weasel in neutral daylight."},"done":true})
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
        .route("/kuno/v1/images/generate", post(image))
        .route("/api/chat", post(llm))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    sqlx::query("INSERT INTO ollama_backends (id,name,base_url,created_at,updated_at) VALUES ('backend','Mock',?,0,0)").bind(format!("http://{address}")).execute(&pool).await.unwrap();
    sqlx::query("UPDATE image_service_settings SET backend_id='backend',token=?")
        .bind("a".repeat(32))
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
        "Updated",
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
            "Stale",
            "generate",
            draft()
        )
        .await
        .is_err()
    );
    server.abort();
}
#[tokio::test]
async fn auto_improvement_runs_once_per_batch_and_preserves_original_guidance() {
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
    let run = service::submit(&state, "alice", &session.id, request(d))
        .await
        .unwrap();
    let done = completed(&state, &run.id).await;
    assert_eq!(done.status, "complete", "{:?}", done.error);
    let versions = service::history(&state.db, "alice", &session.id)
        .await
        .unwrap();
    assert_eq!(versions.len(), 2);
    assert_eq!(versions[0].source, "improved");
    assert_eq!(versions[1].prompt, "a weasel");
    assert_eq!(
        versions[0].parent_id.as_deref(),
        Some(versions[1].id.as_str())
    );
    assert!(versions[0].guidance.contains("blue"));
    assert!(
        mock.requests
            .lock()
            .await
            .iter()
            .all(|r| r["prompt"] == "A white weasel in neutral daylight.")
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
        while mock.requests.lock().await.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    service::stop(&state, "alice", &run.id).await.unwrap();
    assert_eq!(completed(&state, &run.id).await.status, "stopped");
    assert!(
        service::results(&state.db, "alice", &session.id, 0)
            .await
            .unwrap()
            .is_empty()
    );
    server.abort();
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
        "Draft",
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
        "Draft",
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
