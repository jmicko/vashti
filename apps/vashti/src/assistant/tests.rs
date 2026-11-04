use super::{models::*, runner, service};
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
    http::{StatusCode, header},
    response::Response,
    routing::post,
};
use serde_json::{Value, json};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Clone, Default)]
struct Script {
    replies: Arc<Mutex<VecDeque<Vec<u8>>>>,
    requests: Arc<Mutex<Vec<Value>>>,
    delay: Arc<std::sync::atomic::AtomicU64>,
}

struct Fixture {
    state: AppState,
    owner: String,
    other: String,
    backend: String,
    script: Script,
    directory: PathBuf,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

impl Fixture {
    async fn new(replies: Vec<Value>) -> Self {
        Self::raw(
            replies
                .into_iter()
                .map(|v| format!("{v}\n").into_bytes())
                .collect(),
        )
        .await
    }
    async fn raw(replies: Vec<Vec<u8>>) -> Self {
        let script = Script::default();
        script.replies.lock().unwrap().extend(replies);
        let app = Router::new()
            .route(
                "/api/show",
                post(|| async { Json(json!({"capabilities":["completion","tools","thinking"]})) }),
            )
            .route(
                "/api/chat",
                post(
                    |State(script): State<Script>, Json(body): Json<Value>| async move {
                        script.requests.lock().unwrap().push(body);
                        let delay = script.delay.load(std::sync::atomic::Ordering::Relaxed);
                        if delay > 0 {
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                        }
                        let bytes =
                            script
                                .replies
                                .lock()
                                .unwrap()
                                .pop_front()
                                .unwrap_or_else(|| {
                                    format!("{}\n", answer("Fixture result")).into_bytes()
                                });
                        Response::builder()
                            .header(header::CONTENT_TYPE, "application/x-ndjson")
                            .body(Body::from(bytes))
                            .unwrap()
                    },
                ),
            )
            .with_state(script.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
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
        let owner = register_user(
            &pool,
            "assistant-owner".into(),
            None,
            "fixture-only-password".into(),
        )
        .await
        .unwrap()
        .user
        .id;
        let other = register_user(
            &pool,
            "assistant-other".into(),
            None,
            "fixture-only-password".into(),
        )
        .await
        .unwrap()
        .user
        .id;
        sqlx::query("UPDATE users SET is_disabled=0")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("UPDATE app_settings SET tools_enabled=1")
            .execute(&pool)
            .await
            .unwrap();
        let backend = Uuid::new_v4().to_string();
        sqlx::query("INSERT INTO ollama_backends(id,name,base_url,created_at,updated_at) VALUES(?,'fixture',?,?,?)")
            .bind(&backend).bind(base).bind(unix_timestamp()).bind(unix_timestamp()).execute(&pool).await.unwrap();
        let directory =
            std::env::temp_dir().join(format!("vashti-assistant-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let config = Config {
            app_root: directory.clone(),
            data_dir: directory.clone(),
            database_path: directory.join("app.db"),
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
            script,
            directory,
            server,
        }
    }
    fn config(&self) -> TaskConfig {
        TaskConfig {
            backend_id: self.backend.clone(),
            model_name: "fixture-model".into(),
            memories: false,
            web: ActionPolicy::Block,
            max_steps: 6,
            max_output_tokens: 8192,
            ..TaskConfig::default()
        }
    }
    fn payload(&self) -> SaveTask {
        SaveTask {
            request_id: None,
            title: "Fixture task".into(),
            instructions: "Carry out the synthetic fixture task.".into(),
            config: self.config(),
            interval_seconds: None,
            next_run_at: None,
            schedule_enabled: false,
            expected_revision: None,
        }
    }
    async fn task(&self) -> Task {
        service::save_task(&self.state.db, &self.owner, None, self.payload())
            .await
            .unwrap()
    }
    async fn start(&self, task: &Task) -> Run {
        service::start_run(
            &self.state.db,
            &self.owner,
            &task.id,
            StartRun::default(),
            false,
        )
        .await
        .unwrap()
    }
    async fn advance(&self, run: &Run) {
        assert_eq!(
            runner::claim(&self.state.db, &[]).await.unwrap().as_deref(),
            Some(run.id.as_str())
        );
        runner::run(&self.state, &run.id, &CancellationToken::new()).await;
    }
    async fn current(&self, run: &Run) -> Run {
        service::get_run(&self.state.db, &self.owner, &run.id)
            .await
            .unwrap()
    }
}

fn answer(text: &str) -> Value {
    json!({"message":{"role":"assistant","content":text},"done":true,"done_reason":"stop","eval_count":12})
}
fn call(name: &str, args: Value) -> Value {
    json!({"message":{"role":"assistant","tool_calls":[{"function":{"name":name,"arguments":args,"index":0}}]},"done":true,"done_reason":"stop","eval_count":30})
}

#[tokio::test]
async fn tasks_are_owner_scoped_and_edits_have_revision_checks() {
    let f = Fixture::new(vec![]).await;
    let task = f.task().await;
    assert!(
        service::list_tasks(&f.state.db, &f.other, 0)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        service::get_task(&f.state.db, &f.other, &task.id)
            .await
            .unwrap_err()
            .code(),
        "task_not_found"
    );
    let mut payload = f.payload();
    payload.expected_revision = Some(task.revision);
    payload.title = "Renamed".into();
    let changed = service::save_task(&f.state.db, &f.owner, Some(&task.id), payload)
        .await
        .unwrap();
    assert_eq!(changed.revision, 2);
    let mut stale = f.payload();
    stale.expected_revision = Some(1);
    assert_eq!(
        service::save_task(&f.state.db, &f.owner, Some(&task.id), stale)
            .await
            .unwrap_err()
            .code(),
        "task_changed"
    );
    service::archive(&f.state.db, &f.owner, &task.id, true)
        .await
        .unwrap();
    assert!(
        service::list_tasks(&f.state.db, &f.owner, 0)
            .await
            .unwrap()
            .is_empty()
    );
    service::archive(&f.state.db, &f.owner, &task.id, false)
        .await
        .unwrap();
    assert_eq!(
        service::list_tasks(&f.state.db, &f.owner, 0)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn tool_loop_waits_for_exact_approval_and_retains_note_provenance() {
    let f = Fixture::new(vec![
        call("calculate", json!({"expression":"17*23+5"})),
        call(
            "create_note",
            json!({"title":"Fixture result","content":"396"}),
        ),
        answer("Saved the result."),
    ])
    .await;
    let task = f.task().await;
    let run = f.start(&task).await;
    f.advance(&run).await;
    assert_eq!(f.current(&run).await.status, "waiting_approval");
    let actions = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap();
    assert_eq!(actions.len(), 2);
    assert_eq!(actions[0].status, "completed");
    assert_eq!(actions[1].status, "pending");
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM notes")
        .fetch_one(&f.state.db)
        .await
        .unwrap();
    assert_eq!(count, 0);
    assert!(
        service::decide(&f.state.db, &f.other, &run.id, &actions[1].id, true)
            .await
            .is_err()
    );
    service::decide(&f.state.db, &f.owner, &run.id, &actions[1].id, true)
        .await
        .unwrap();
    assert!(
        service::decide(&f.state.db, &f.owner, &run.id, &actions[1].id, true)
            .await
            .is_err()
    );
    f.advance(&run).await;
    assert_eq!(f.current(&run).await.status, "completed");
    let (source, chat, content): (Option<String>, Option<String>, String) =
        sqlx::query_as("SELECT source_assistant_run_id,source_chat_id,content FROM note_versions")
            .fetch_one(&f.state.db)
            .await
            .unwrap();
    assert_eq!(source.as_deref(), Some(run.id.as_str()));
    assert_eq!(chat, None);
    assert_eq!(content, "396");
    let stored = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap();
    assert_eq!(stored[0].id, actions[0].id);
    assert_eq!(stored[0].status, "completed");
    assert_eq!(f.script.requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn rejection_cannot_be_retried_with_changed_arguments() {
    let f = Fixture::new(vec![
        call("create_note", json!({"title":"First","content":"fixture"})),
        call(
            "create_note",
            json!({"title":"Different","content":"changed arguments"}),
        ),
        answer("I did not create the note."),
    ])
    .await;
    let task = f.task().await;
    let run = f.start(&task).await;
    f.advance(&run).await;
    let first = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap()
        .remove(0);
    service::decide(&f.state.db, &f.owner, &run.id, &first.id, false)
        .await
        .unwrap();
    f.advance(&run).await;
    assert_eq!(f.current(&run).await.status, "completed");
    let actions = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap();
    assert_eq!(actions[0].status, "rejected");
    assert_eq!(actions[1].status, "completed");
    assert!(
        actions[1]
            .result_text
            .as_deref()
            .unwrap()
            .contains("blocked")
    );
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM notes")
        .fetch_one(&f.state.db)
        .await
        .unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn revocation_after_approval_prevents_execution() {
    let f = Fixture::new(vec![
        call(
            "create_note",
            json!({"title":"Fixture","content":"must not be saved"}),
        ),
        answer("Permission was revoked."),
    ])
    .await;
    let task = f.task().await;
    let run = f.start(&task).await;
    f.advance(&run).await;
    let action = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap()
        .remove(0);
    service::decide(&f.state.db, &f.owner, &run.id, &action.id, true)
        .await
        .unwrap();
    sqlx::query("UPDATE user_note_settings SET allow_model_create=0 WHERE user_id=?")
        .bind(&f.owner)
        .execute(&f.state.db)
        .await
        .unwrap();
    f.advance(&run).await;
    let actions = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap();
    assert!(
        actions[0]
            .result_text
            .as_deref()
            .unwrap()
            .contains("revoked")
    );
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM notes")
        .fetch_one(&f.state.db)
        .await
        .unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn questions_are_owner_scoped_persistent_pauses() {
    let f = Fixture::new(vec![
        call(
            "ask_user",
            json!({"question":"Which fixture color?","choices":["Green","Gray"]}),
        ),
        answer("I used gray."),
    ])
    .await;
    let task = f.task().await;
    let run = f.start(&task).await;
    f.advance(&run).await;
    assert_eq!(f.current(&run).await.status, "waiting_input");
    service::recover(&f.state.db).await.unwrap();
    assert_eq!(f.current(&run).await.status, "waiting_input");
    assert!(
        service::reply(&f.state.db, &f.other, &run.id, "Green")
            .await
            .is_err()
    );
    service::reply(&f.state.db, &f.owner, &run.id, "Gray")
        .await
        .unwrap();
    assert!(
        service::reply(&f.state.db, &f.owner, &run.id, "Green")
            .await
            .is_err()
    );
    f.advance(&run).await;
    assert_eq!(f.current(&run).await.status, "completed");
    let requests = f.script.requests.lock().unwrap();
    let messages = requests[1]["messages"].as_array().unwrap();
    assert_eq!(messages.last().unwrap()["tool_name"], "ask_user");
    assert!(
        messages.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("Gray")
    );
}

#[tokio::test]
async fn restart_does_not_replay_uncertain_side_effects() {
    let f = Fixture::new(vec![call(
        "create_note",
        json!({"title":"Uncertain","content":"fixture"}),
    )])
    .await;
    let task = f.task().await;
    let run = f.start(&task).await;
    f.advance(&run).await;
    let action = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap()
        .remove(0);
    sqlx::query("UPDATE assistant_actions SET status='executing' WHERE id=?")
        .bind(&action.id)
        .execute(&f.state.db)
        .await
        .unwrap();
    sqlx::query("UPDATE assistant_runs SET status='running' WHERE id=?")
        .bind(&run.id)
        .execute(&f.state.db)
        .await
        .unwrap();
    service::recover(&f.state.db).await.unwrap();
    assert_eq!(f.current(&run).await.status, "interrupted");
    assert_eq!(
        service::actions(&f.state.db, &f.owner, &run.id)
            .await
            .unwrap()[0]
            .status,
        "uncertain"
    );
    assert_eq!(
        service::control(&f.state.db, &f.owner, &run.id, "resume")
            .await
            .unwrap_err()
            .code(),
        "uncertain_action"
    );
    assert_eq!(
        service::start_run(&f.state.db, &f.owner, &task.id, StartRun::default(), false)
            .await
            .unwrap_err()
            .code(),
        "uncertain_action"
    );
    assert_eq!(f.script.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn queue_does_not_overlap_users_or_reclaim_a_cancelling_worker() {
    let f = Fixture::new(vec![]).await;
    let a = f.task().await;
    let b = f.task().await;
    let first = f.start(&a).await;
    let second = f.start(&b).await;
    assert!(
        service::start_run(&f.state.db, &f.owner, &a.id, StartRun::default(), false)
            .await
            .is_err()
    );
    assert_eq!(
        runner::claim(&f.state.db, &[]).await.unwrap().as_deref(),
        Some(first.id.as_str())
    );
    assert!(runner::claim(&f.state.db, &[]).await.unwrap().is_none());
    service::control(&f.state.db, &f.owner, &first.id, "pause")
        .await
        .unwrap();
    service::control(&f.state.db, &f.owner, &first.id, "resume")
        .await
        .unwrap();
    assert!(
        runner::claim(&f.state.db, std::slice::from_ref(&first.id))
            .await
            .unwrap()
            .is_none()
    );
    service::control(&f.state.db, &f.owner, &first.id, "cancel")
        .await
        .unwrap();
    assert_eq!(
        runner::claim(&f.state.db, &[]).await.unwrap().as_deref(),
        Some(second.id.as_str())
    );
}

#[tokio::test]
async fn step_limits_and_truncation_never_execute_incomplete_tool_requests() {
    let f = Fixture::new(vec![call("calculate", json!({"expression":"2+2"}))]).await;
    let mut payload = f.payload();
    payload.config.max_steps = 1;
    let task = service::save_task(&f.state.db, &f.owner, None, payload)
        .await
        .unwrap();
    let run = f.start(&task).await;
    f.advance(&run).await;
    assert_eq!(f.current(&run).await.status, "limit_reached");
    let mut truncated = call("create_note", json!({"title":"No","content":"Never"}));
    truncated["done_reason"] = json!("length");
    f.script
        .replies
        .lock()
        .unwrap()
        .push_back(format!("{truncated}\n").into_bytes());
    let other = f.task().await;
    let run = f.start(&other).await;
    f.advance(&run).await;
    assert_eq!(f.current(&run).await.status, "limit_reached");
    assert!(
        service::actions(&f.state.db, &f.owner, &run.id)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn malformed_and_incomplete_streams_fail_closed() {
    for body in [
        b"not JSON\n".to_vec(),
        format!(
            "{}\n",
            json!({"message":{"content":"partial"},"done":false})
        )
        .into_bytes(),
    ] {
        let f = Fixture::raw(vec![body]).await;
        let task = f.task().await;
        let run = f.start(&task).await;
        f.advance(&run).await;
        assert_eq!(f.current(&run).await.status, "failed");
        assert!(
            service::actions(&f.state.db, &f.owner, &run.id)
                .await
                .unwrap()
                .is_empty()
        );
    }
}

#[tokio::test]
async fn cancellation_releases_an_inference_request_without_completing_the_run() {
    let f = Fixture::new(vec![answer("Must not arrive")]).await;
    f.script
        .delay
        .store(30_000, std::sync::atomic::Ordering::Relaxed);
    let task = f.task().await;
    let run = f.start(&task).await;
    runner::claim(&f.state.db, &[]).await.unwrap();
    let token = CancellationToken::new();
    let state = f.state.clone();
    let id = run.id.clone();
    let child_token = token.clone();
    let work = tokio::spawn(async move {
        runner::run(&state, &id, &child_token).await;
    });
    tokio::time::timeout(Duration::from_secs(3), async {
        while f.script.requests.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    service::control(&f.state.db, &f.owner, &run.id, "pause")
        .await
        .unwrap();
    token.cancel();
    tokio::time::timeout(Duration::from_secs(2), work)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(f.current(&run).await.status, "paused");
}

#[tokio::test]
async fn schedules_coalesce_downtime_and_do_not_overlap_active_work() {
    let f = Fixture::new(vec![]).await;
    let mut payload = f.payload();
    payload.schedule_enabled = true;
    payload.interval_seconds = Some(300);
    payload.next_run_at = Some(unix_timestamp() - 3600);
    let task = service::save_task(&f.state.db, &f.owner, None, payload)
        .await
        .unwrap();
    runner::tick_schedules(&f.state).await.unwrap();
    runner::tick_schedules(&f.state).await.unwrap();
    let runs = service::list_runs(&f.state.db, &f.owner, &task.id)
        .await
        .unwrap();
    assert_eq!(runs.len(), 1);
    assert!(runs[0].scheduled);
    let updated = service::get_task(&f.state.db, &f.owner, &task.id)
        .await
        .unwrap();
    assert!(updated.next_run_at.unwrap() > unix_timestamp());
    sqlx::query("UPDATE assistant_tasks SET next_run_at=? WHERE id=?")
        .bind(unix_timestamp() - 300)
        .bind(&task.id)
        .execute(&f.state.db)
        .await
        .unwrap();
    runner::tick_schedules(&f.state).await.unwrap();
    assert_eq!(
        service::list_runs(&f.state.db, &f.owner, &task.id)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn schedule_failures_are_visible_and_disable_future_dispatch() {
    let f = Fixture::new(vec![]).await;
    let mut payload = f.payload();
    payload.schedule_enabled = true;
    payload.interval_seconds = Some(300);
    payload.next_run_at = Some(unix_timestamp() - 1);
    let task = service::save_task(&f.state.db, &f.owner, None, payload)
        .await
        .unwrap();
    sqlx::query("UPDATE ollama_backends SET is_enabled=0 WHERE id=?")
        .bind(&f.backend)
        .execute(&f.state.db)
        .await
        .unwrap();
    runner::tick_schedules(&f.state).await.unwrap();
    let task = service::get_task(&f.state.db, &f.owner, &task.id)
        .await
        .unwrap();
    assert!(!task.schedule_enabled);
    assert!(task.schedule_error.is_some());
    assert!(task.needs_attention);
}

#[tokio::test]
async fn sandbox_is_a_separate_admin_default_permission() {
    let f = Fixture::new(vec![]).await;
    assert!(
        crate::settings::service::tool_is_available_for_user(
            &f.state.db,
            &f.owner,
            SANDBOX_TOOL_ID
        )
        .await
        .unwrap()
    );
    assert!(
        !crate::settings::service::tool_is_available_for_user(
            &f.state.db,
            &f.other,
            SANDBOX_TOOL_ID
        )
        .await
        .unwrap()
    );
    let args = super::sandbox::container_args("fixture", &"a".repeat(64), "installation");
    for required in [
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=1g",
        "--pids-limit=128",
        "--log-driver=none",
        "--pull=never",
    ] {
        assert!(args.iter().any(|arg| arg == required));
    }
    assert!(
        !args
            .iter()
            .any(|arg| arg == "--privileged" || arg == "--volume" || arg.starts_with("--device"))
    );
}

#[tokio::test]
async fn http_routes_require_sessions_and_scope_every_record_and_download() {
    let f = Fixture::new(vec![]).await;
    let task = f.task().await;
    let run = f.start(&task).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .nest("/api/assistant", super::handlers::router())
        .with_state(f.state.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let owner = crate::auth::service::create_session(&f.state.db, &f.owner, 3600, None, None)
        .await
        .unwrap();
    let other = crate::auth::service::create_session(&f.state.db, &f.other, 3600, None, None)
        .await
        .unwrap();
    let client = reqwest::Client::new();
    assert_eq!(
        client
            .get(format!("{base}/api/assistant/tasks"))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    for path in [format!("tasks/{}", task.id), format!("runs/{}", run.id)] {
        assert_eq!(
            client
                .get(format!("{base}/api/assistant/{path}"))
                .header(header::COOKIE, format!("vashti_session={}", other.id))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
    }
    let artifact = Uuid::new_v4().to_string();
    let bytes = b"fixture,396\n";
    let path = super::sandbox::artifact_path(&f.state, &artifact).unwrap();
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&path, bytes).await.unwrap();
    sqlx::query("INSERT INTO assistant_artifacts(id,run_id,name,mime_type,size,created_at) VALUES(?,?,'answer.csv','text/csv',?,?)").bind(&artifact).bind(&run.id).bind(bytes.len() as i64).bind(unix_timestamp()).execute(&f.state.db).await.unwrap();
    let url = format!("{base}/api/assistant/artifacts/{artifact}");
    assert_eq!(
        client
            .get(&url)
            .header(header::COOKIE, format!("vashti_session={}", other.id))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        client
            .delete(&url)
            .header(header::COOKIE, format!("vashti_session={}", other.id))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND
    );
    let response = client
        .get(&url)
        .header(header::COOKIE, format!("vashti_session={}", owner.id))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-content-type-options"], "nosniff");
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/octet-stream"
    );
    assert!(
        response.headers()[header::CONTENT_DISPOSITION]
            .to_str()
            .unwrap()
            .starts_with("attachment;")
    );
    assert_eq!(response.bytes().await.unwrap().as_ref(), bytes);
    assert_eq!(
        client
            .delete(&url)
            .header(header::COOKIE, format!("vashti_session={}", owner.id))
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert!(!path.exists());
    server.abort();
}

#[tokio::test]
#[ignore = "requires installed rootless Podman and the approved image"]
async fn podman_workspace_is_offline_persistent_bounded_and_owner_exportable() {
    let f = Fixture::new(vec![]).await;
    let task = f.task().await;
    let run = f.start(&task).await;
    let available = super::sandbox::availability().await;
    assert!(available.available, "{:?}", available.reason);
    let image = available.image_id.unwrap();
    let token = CancellationToken::new();
    let result=super::sandbox::execute(&f.state,&run.id,&image,"printf 'value\\n396\\n' > answer.csv; test ! -e /run/podman/podman.sock; test ! -e /media/jmicko/faststore/dev/vashti/.private; if touch /should-not-write 2>/dev/null; then exit 42; fi; python3 -I -c \"import socket; s=socket.socket(); s.settimeout(1); assert s.connect_ex(('1.1.1.1',443)) != 0; print('isolated')\"",20,&token).await.unwrap();
    assert_eq!(result["exit_code"], 0);
    assert!(result["stdout"].as_str().unwrap().contains("isolated"));
    let result = super::sandbox::execute(
        &f.state,
        &run.id,
        &image,
        "cat answer.csv; ln -s /etc/passwd stolen",
        20,
        &token,
    )
    .await
    .unwrap();
    assert_eq!(result["exit_code"], 0);
    assert!(result["stdout"].as_str().unwrap().contains("396"));
    assert!(
        super::sandbox::export(&f.state, &run.id, &image, "stolen", &token)
            .await
            .is_err()
    );
    assert!(
        super::sandbox::export(&f.state, &run.id, &image, "../../etc/passwd", &token)
            .await
            .is_err()
    );
    let exported = super::sandbox::export(&f.state, &run.id, &image, "answer.csv", &token)
        .await
        .unwrap();
    let id = exported["artifact_id"].as_str().unwrap();
    assert_eq!(
        tokio::fs::read(super::sandbox::artifact_path(&f.state, id).unwrap())
            .await
            .unwrap(),
        b"value\n396\n"
    );
    assert!(
        service::artifacts(&f.state.db, &f.other, &run.id)
            .await
            .is_err()
    );
    let timed = super::sandbox::execute(&f.state, &run.id, &image, "sleep 10", 1, &token)
        .await
        .unwrap();
    assert_eq!(timed["timed_out"], true);
    let count = tokio::process::Command::new("podman")
        .args([
            "ps",
            "--all",
            "--filter",
            &format!("label=chat.vashti.assistant={}", f.state.server_instance_id),
            "--format",
            "{{.Names}}",
        ])
        .output()
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&count.stdout).trim().is_empty());
}

#[tokio::test]
#[ignore = "explicit installed-model evaluation; performs real inference and Podman work"]
async fn installed_kuno_qwen_completes_a_multi_tool_file_task() {
    let f = Fixture::new(vec![]).await;
    sqlx::query("UPDATE ollama_backends SET base_url='http://127.0.0.1:11434' WHERE id=?")
        .bind(&f.backend)
        .execute(&f.state.db)
        .await
        .unwrap();
    let mut payload = f.payload();
    payload.config.model_name = PREFERRED_MODEL.into();
    payload.config.sandbox = true;
    payload.config.notes = false;
    payload.config.max_steps = 12;
    payload.config.max_output_tokens = 16384;
    payload.instructions="Use calculate to evaluate 17*23+5. Then use sandbox_execute to create answer.csv with two lines: value and the calculated integer. Export answer.csv with sandbox_export. Verify the actual file exists using a tool. Return its download link. This is a synthetic smoke test, not a real user's information. Do not create notes or memories. Keep the plan short and do not ask questions.".into();
    let task = service::save_task(&f.state.db, &f.owner, None, payload)
        .await
        .unwrap();
    let run = f.start(&task).await;
    f.advance(&run).await;
    let current = f.current(&run).await;
    assert_eq!(current.status, "completed", "{}", current.error_text);
    let actions = service::actions(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap();
    for name in ["calculate", "sandbox_execute", "sandbox_export"] {
        assert!(
            actions
                .iter()
                .any(|a| a.tool_name == name && a.status == "completed"),
            "missing actual {name}"
        );
    }
    let artifacts = service::artifacts(&f.state.db, &f.owner, &run.id)
        .await
        .unwrap();
    let file = artifacts
        .iter()
        .find(|a| a.name == "answer.csv")
        .expect("actual exported CSV");
    let bytes = tokio::fs::read(super::sandbox::artifact_path(&f.state, &file.id).unwrap())
        .await
        .unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(text.replace('\r', "").trim(), "value\n396");
    assert!(current.result_text.contains(&file.id));
}

#[test]
fn policies_do_not_infer_authority_from_model_text() {
    let config = TaskConfig::default();
    assert_eq!(runner::policy(&config, "create_note"), ActionPolicy::Ask);
    assert_eq!(
        runner::policy(&config, "direct_web_fetch"),
        ActionPolicy::Ask
    );
    assert_eq!(runner::policy(&config, "calculate"), ActionPolicy::Allow);
    assert_eq!(
        runner::policy(&config, "approve_action"),
        ActionPolicy::Block
    );
    assert_eq!(
        runner::policy(&config, "sandbox_execute"),
        ActionPolicy::Block
    );
}

#[test]
fn limits_and_model_are_explicit() {
    let mut config = TaskConfig::default();
    assert!(service::validate_config(&config).is_err());
    config.backend_id = "test".into();
    assert!(service::validate_config(&config).is_ok());
    config.max_steps = usize::MAX;
    assert!(service::validate_config(&config).is_err());
}

#[test]
fn model_context_uses_configured_limits_without_forcing_a_larger_native_window() {
    let profile:crate::ollama::models::ShowModelResponse=serde_json::from_value(json!({"parameters":"temperature 0.7\nnum_ctx 32768\nnum_predict 1024","model_info":{"qwen4exp.context_length":32768}})).unwrap();
    assert_eq!(profile.context_window(), 32768);
    let profile: crate::ollama::models::ShowModelResponse = serde_json::from_value(
        json!({"parameters":"num_ctx 65536","model_info":{"qwen4exp.context_length":8192}}),
    )
    .unwrap();
    assert_eq!(profile.context_window(), 8192);
    let profile: crate::ollama::models::ShowModelResponse =
        serde_json::from_value(json!({})).unwrap();
    assert_eq!(profile.context_window(), 8192);
}

#[tokio::test]
async fn retry_keys_do_not_duplicate_tasks_or_finished_runs() {
    let f = Fixture::new(vec![answer("Done")]).await;
    let key = Uuid::new_v4().to_string();
    let mut request = f.payload();
    request.request_id = Some(key.clone());
    let task = service::save_task(&f.state.db, &f.owner, None, request)
        .await
        .unwrap();
    let mut retry = f.payload();
    retry.request_id = Some(key.clone());
    assert_eq!(
        service::save_task(&f.state.db, &f.owner, None, retry)
            .await
            .unwrap()
            .id,
        task.id
    );
    assert_eq!(
        service::list_tasks(&f.state.db, &f.owner, 0)
            .await
            .unwrap()
            .len(),
        1
    );
    let run_id = Uuid::new_v4().to_string();
    let run = service::start_run(
        &f.state.db,
        &f.owner,
        &task.id,
        StartRun {
            request_id: Some(run_id.clone()),
            ..StartRun::default()
        },
        false,
    )
    .await
    .unwrap();
    f.advance(&run).await;
    let retry = service::start_run(
        &f.state.db,
        &f.owner,
        &task.id,
        StartRun {
            request_id: Some(run_id.clone()),
            ..StartRun::default()
        },
        false,
    )
    .await
    .unwrap();
    assert_eq!(retry.id, run.id);
    assert_eq!(retry.status, "completed");
    assert_eq!(
        service::start_run(
            &f.state.db,
            &f.owner,
            &task.id,
            StartRun {
                request_id: Some(run_id),
                message: "Changed instructions".into(),
                ..StartRun::default()
            },
            false
        )
        .await
        .unwrap_err()
        .code(),
        "request_changed"
    );
    assert_eq!(
        service::list_runs(&f.state.db, &f.owner, &task.id)
            .await
            .unwrap()
            .len(),
        1
    );
}
