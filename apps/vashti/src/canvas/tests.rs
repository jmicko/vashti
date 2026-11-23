use super::{models::*, service::*};
use sqlx::{
    SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};

async fn fixture() -> SqlitePool {
    let db = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(":memory:")
                .foreign_keys(true),
        )
        .await
        .unwrap();
    crate::startup::migrations::run(&db).await.unwrap();
    crate::startup::bootstrap::ensure_app_settings(&db)
        .await
        .unwrap();
    sqlx::raw_sql("INSERT INTO users(id,username,password_hash,created_at,updated_at) VALUES('owner','owner','hash',100,100),('other','other','hash',100,100); INSERT INTO ollama_backends(id,name,base_url,created_at,updated_at) VALUES('backend','Local','http://localhost:11434',100,100); INSERT INTO chats(id,user_id,default_backend_id,default_model_name,title,created_at,updated_at,last_message_at) VALUES('chat','owner','backend','model','Canvas fixture',100,100,100); INSERT INTO chat_messages(id,chat_id,role,status,created_at,updated_at) VALUES('user','chat','user','complete',100,100); INSERT INTO chat_message_revisions(id,message_id,content_text,thinking_text,source,created_at) VALUES('user-r','user','A synthetic request','','original',100); UPDATE chat_messages SET active_revision_id='user-r' WHERE id='user';").execute(&db).await.unwrap();
    db
}
fn owner() -> Actor<'static> {
    Actor { model_name: None }
}
fn scope() -> Scope {
    Scope {
        message_id: Some("user".into()),
        revision_id: Some("user-r".into()),
    }
}
async fn create(db: &SqlitePool, scope: &Scope) -> Workspace {
    mutate(
        db,
        "owner",
        "chat",
        scope,
        Mutation::Create {
            document_id: None,
            title: "Draft".into(),
            language: "markdown".into(),
            content: "A short draft.\nAnother line.".into(),
        },
        owner(),
    )
    .await
    .unwrap()
}
fn update(doc: &Document, text: &str, session: Option<&str>) -> Mutation {
    Mutation::Update {
        document_id: doc.id.clone(),
        expected_version_id: doc.version.id.clone(),
        title: None,
        language: None,
        content: Some(text.into()),
        edits: None,
        edit_session_id: session.map(str::to_string),
    }
}
async fn response(db: &SqlitePool, id: &str, parent: &str) -> Scope {
    sqlx::query("INSERT INTO chat_messages(id,chat_id,parent_message_id,role,status,created_at,updated_at) VALUES(?,'chat',?,'assistant','streaming',101,101)").bind(id).bind(parent).execute(db).await.unwrap();
    let revision = format!("{id}-r");
    sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,thinking_text,source,created_at) VALUES(?,?,'','','original',101)").bind(&revision).bind(id).execute(db).await.unwrap();
    sqlx::query("UPDATE chat_messages SET active_revision_id=? WHERE id=?")
        .bind(&revision)
        .bind(id)
        .execute(db)
        .await
        .unwrap();
    capture_generation(db, "owner", "chat", id, &revision)
        .await
        .unwrap();
    Scope {
        message_id: Some(id.into()),
        revision_id: Some(revision),
    }
}

#[tokio::test]
async fn documents_follow_branches_and_snapshots_never_follow_later_edits() {
    let db = fixture().await;
    let original = create(&db, &scope()).await;
    let a = response(&db, "a", "user").await;
    let b = response(&db, "b", "user").await;
    let edited = mutate(
        &db,
        "owner",
        "chat",
        &a,
        update(&original.documents[0], "Branch A draft", None),
        Actor {
            model_name: Some("Synthetic model"),
        },
    )
    .await
    .unwrap();
    assert_eq!(edited.documents[0].version.content, "Branch A draft");
    assert_eq!(
        workspace(&db, "owner", "chat", &b).await.unwrap().documents[0]
            .version
            .id,
        original.documents[0].version.id
    );
    assert_eq!(
        workspace(&db, "owner", "chat", &scope())
            .await
            .unwrap()
            .documents[0]
            .version
            .id,
        original.documents[0].version.id
    );
    let captured: String = sqlx::query_scalar(
        "SELECT heads_json FROM canvas_generation_inputs WHERE revision_id='a-r'",
    )
    .fetch_one(&db)
    .await
    .unwrap();
    assert!(captured.contains(&original.documents[0].version.id));
    assert!(!captured.contains(&edited.documents[0].version.id));
    let child = response(&db, "child", "a").await;
    assert_eq!(
        workspace(&db, "owner", "chat", &child)
            .await
            .unwrap()
            .documents[0]
            .version
            .content,
        "Branch A draft"
    );
    let empty = response(&db, "empty", "user").await;
    create(&db, &scope()).await;
    assert_eq!(
        workspace(&db, "owner", "chat", &empty)
            .await
            .unwrap()
            .documents
            .len(),
        1
    );
}
#[tokio::test]
async fn stale_edits_ownership_and_ambiguous_patches_fail_without_partial_writes() {
    let db = fixture().await;
    let original = create(&db, &scope()).await;
    assert!(workspace(&db, "other", "chat", &scope()).await.is_err());
    assert!(
        mutate(
            &db,
            "other",
            "chat",
            &scope(),
            update(&original.documents[0], "stolen", None),
            owner()
        )
        .await
        .is_err()
    );
    let first = mutate(
        &db,
        "owner",
        "chat",
        &scope(),
        update(&original.documents[0], "New draft", None),
        owner(),
    )
    .await
    .unwrap();
    assert!(
        mutate(
            &db,
            "owner",
            "chat",
            &scope(),
            update(&original.documents[0], "Stale overwrite", None),
            owner()
        )
        .await
        .is_err()
    );
    assert_eq!(
        workspace(&db, "owner", "chat", &scope())
            .await
            .unwrap()
            .documents[0]
            .version
            .id,
        first.documents[0].version.id
    );
    assert!(
        apply_edits(
            "same same",
            &[Edit {
                find: "same".into(),
                replace: "other".into()
            }]
        )
        .is_err()
    );
    assert!(
        apply_edits(
            "one two",
            &[
                Edit {
                    find: "one".into(),
                    replace: "three".into()
                },
                Edit {
                    find: "absent".into(),
                    replace: "bad".into()
                }
            ]
        )
        .is_err()
    );
    assert_eq!(
        apply_edits(
            "one two",
            &[Edit {
                find: "two".into(),
                replace: "three".into()
            }]
        )
        .unwrap(),
        "one three"
    );
}
#[tokio::test]
async fn autosaves_coalesce_only_until_a_checkpoint_is_referenced() {
    let db = fixture().await;
    let initial = create(&db, &scope()).await;
    let first = mutate(
        &db,
        "owner",
        "chat",
        &scope(),
        update(&initial.documents[0], "One edit", Some("session")),
        owner(),
    )
    .await
    .unwrap();
    let second = mutate(
        &db,
        "owner",
        "chat",
        &scope(),
        update(&first.documents[0], "Two edits", Some("session")),
        owner(),
    )
    .await
    .unwrap();
    assert_eq!(
        second.documents[0].version.version_number,
        first.documents[0].version.version_number
    );
    assert_ne!(
        second.documents[0].version.id,
        first.documents[0].version.id
    );
    response(&db, "a", "user").await;
    let third = mutate(
        &db,
        "owner",
        "chat",
        &scope(),
        update(&second.documents[0], "After snapshot", Some("session")),
        owner(),
    )
    .await
    .unwrap();
    assert!(third.documents[0].version.version_number > second.documents[0].version.version_number);
    assert!(
        historical_version(
            &db,
            "owner",
            "chat",
            &initial.documents[0].id,
            &second.documents[0].version.id
        )
        .await
        .is_ok()
    );
    let restored = mutate(
        &db,
        "owner",
        "chat",
        &scope(),
        Mutation::Restore {
            document_id: initial.documents[0].id.clone(),
            expected_version_id: third.documents[0].version.id.clone(),
            version_id: initial.documents[0].version.id.clone(),
        },
        owner(),
    )
    .await
    .unwrap();
    assert_eq!(
        restored.documents[0].version.content,
        initial.documents[0].version.content
    );
    assert!(
        restored.documents[0].version.version_number > third.documents[0].version.version_number
    );
}
#[tokio::test]
async fn comments_and_recoverable_removal_stay_scoped_and_model_cannot_remove() {
    let db = fixture().await;
    let original = create(&db, &scope()).await;
    let doc = &original.documents[0];
    let a = response(&db, "a", "user").await;
    let commented = mutate(
        &db,
        "owner",
        "chat",
        &a,
        Mutation::Comment {
            document_id: doc.id.clone(),
            expected_version_id: doc.version.id.clone(),
            quote: "short draft".into(),
            content: "Consider a stronger opening.".into(),
        },
        Actor {
            model_name: Some("Synthetic reviewer"),
        },
    )
    .await
    .unwrap();
    assert_eq!(commented.comments.len(), 1);
    let trash = || Mutation::Trash {
        document_id: doc.id.clone(),
        expected_version_id: doc.version.id.clone(),
        trashed: true,
    };
    assert!(
        mutate(
            &db,
            "owner",
            "chat",
            &scope(),
            trash(),
            Actor {
                model_name: Some("model")
            }
        )
        .await
        .is_err()
    );
    assert!(
        mutate(&db, "owner", "chat", &scope(), trash(), owner())
            .await
            .unwrap()
            .documents[0]
            .trashed
    );
    let restore = Mutation::Trash {
        document_id: doc.id.clone(),
        expected_version_id: doc.version.id.clone(),
        trashed: false,
    };
    assert!(
        !mutate(&db, "owner", "chat", &scope(), restore, owner())
            .await
            .unwrap()
            .documents[0]
            .trashed
    );
    crate::chats::service::set_chat_archived(&db, "owner", "chat", true)
        .await
        .unwrap();
    assert_eq!(
        workspace(&db, "owner", "chat", &scope())
            .await
            .unwrap()
            .documents
            .len(),
        1
    );
    crate::chats::service::delete_chat(&db, "owner", "chat")
        .await
        .unwrap();
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM canvas_versions")
            .fetch_one(&db)
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn inherited_revisions_seal_autosaves_and_stopped_generations_cannot_write() {
    let db = fixture().await;
    let initial = create(&db, &scope()).await;
    let first = mutate(
        &db,
        "owner",
        "chat",
        &scope(),
        update(&initial.documents[0], "Before revision", Some("session")),
        owner(),
    )
    .await
    .unwrap();
    sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,thinking_text,source,created_at) VALUES('edited-r','user','Edited request','','edit',102)").execute(&db).await.unwrap();
    inherit_revision(
        &mut db.acquire().await.unwrap(),
        "chat",
        "user-r",
        "edited-r",
    )
    .await
    .unwrap();
    let later = mutate(
        &db,
        "owner",
        "chat",
        &scope(),
        update(&first.documents[0], "After revision", Some("session")),
        owner(),
    )
    .await
    .unwrap();
    assert!(later.documents[0].version.version_number > first.documents[0].version.version_number);
    assert_eq!(
        historical_version(
            &db,
            "owner",
            "chat",
            &first.documents[0].id,
            &first.documents[0].version.id
        )
        .await
        .unwrap()
        .content,
        "Before revision"
    );
    let assistant = response(&db, "stopped", "user").await;
    sqlx::query("UPDATE chat_messages SET status='stopped' WHERE id='stopped'")
        .execute(&db)
        .await
        .unwrap();
    assert!(
        mutate(
            &db,
            "owner",
            "chat",
            &assistant,
            update(&later.documents[0], "After stop", None),
            Actor {
                model_name: Some("Synthetic model")
            }
        )
        .await
        .is_err()
    );
    assert_eq!(
        workspace(&db, "owner", "chat", &assistant)
            .await
            .unwrap()
            .documents[0]
            .version
            .id,
        later.documents[0].version.id
    );
}

#[tokio::test]
async fn idempotent_creation_validation_and_unicode_read_chunks_are_bounded() {
    let db = fixture().await;
    let id = uuid::Uuid::new_v4().to_string();
    let make = || Mutation::Create {
        document_id: Some(id.clone()),
        title: "Retry-safe".into(),
        language: "text".into(),
        content: "Synthetic content".into(),
    };
    let first = mutate(&db, "owner", "chat", &scope(), make(), owner())
        .await
        .unwrap();
    let again = mutate(&db, "owner", "chat", &scope(), make(), owner())
        .await
        .unwrap();
    assert_eq!(again.documents.len(), 1);
    assert_eq!(first.documents[0].version.id, again.documents[0].version.id);
    assert!(validate_document("bad\nname", "text", "").is_err());
    assert!(validate_document("Name", "execute", "").is_err());
    assert!(validate_document("Name", "text", &"🌿".repeat(70_000)).is_err());
    assert!(
        apply_edits(
            "aaa",
            &[Edit {
                find: "aa".into(),
                replace: "x".into()
            }]
        )
        .is_err()
    );
    let mut v = first.documents.into_iter().next().unwrap().version;
    v.content = format!("{}end", "🌿".repeat(4_000));
    let a = super::tools::read_result(&v, &serde_json::json!({}));
    let b = super::tools::read_result(&v, &serde_json::json!({"offset":a["next_offset"]}));
    assert_eq!(a["content"].as_str().unwrap().len(), 12_000);
    assert_eq!(
        format!(
            "{}{}",
            a["content"].as_str().unwrap(),
            b["content"].as_str().unwrap()
        ),
        v.content
    );
    assert_eq!(b["truncated"], false);
}

#[tokio::test]
async fn model_tools_enforce_catalog_chat_permissions_and_exact_versions() {
    let db = fixture().await;
    let at = response(&db, "tool", "user").await;
    sqlx::query("UPDATE app_settings SET tools_enabled=1 WHERE id=1")
        .execute(&db)
        .await
        .unwrap();
    let call = |name: &str, args: serde_json::Value| {
        serde_json::from_value::<crate::ollama::models::OllamaToolCall>(
            serde_json::json!({"function":{"name":name,"arguments":args}}),
        )
        .unwrap()
    };
    let create = call(
        "create_canvas",
        serde_json::json!({"title":"Tool draft","language":"markdown","content":"Synthetic text"}),
    );
    let run = |call: crate::ollama::models::OllamaToolCall| {
        let db = &db;
        let revision = at.revision_id.clone().unwrap();
        async move {
            serde_json::from_str::<serde_json::Value>(
                &super::tools::execute(
                    db,
                    "owner",
                    "chat",
                    "tool",
                    &revision,
                    "Synthetic model",
                    &call,
                )
                .await,
            )
            .unwrap()
        }
    };
    let created = run(create.clone()).await;
    assert_eq!(created["saved"], true, "{created}");
    let read = run(call(
        "read_canvas",
        serde_json::json!({"document_id":created["canvas"]["document_id"]}),
    ))
    .await;
    assert_eq!(read["content"], "Synthetic text");
    let stale=run(call("update_canvas",serde_json::json!({"document_id":created["canvas"]["document_id"],"expected_version_id":"wrong","content":"overwrite"}))).await;
    assert_eq!(stale["code"], "canvas_conflict");
    sqlx::query("UPDATE chats SET tool_preferences_json=? WHERE id='chat'")
        .bind(r#"{"tool_use_enabled":true,"tools":{"canvas":false}}"#)
        .execute(&db)
        .await
        .unwrap();
    assert_eq!(run(create.clone()).await["code"], "canvas_disabled");
    sqlx::query("UPDATE chats SET tool_preferences_json=NULL WHERE id='chat'")
        .execute(&db)
        .await
        .unwrap();
    sqlx::query("DELETE FROM tool_permission_tags WHERE tool_id='canvas'")
        .execute(&db)
        .await
        .unwrap();
    assert_eq!(run(create).await["code"], "canvas_disabled");
    // The human workspace is independent of model tool permissions.
    assert_eq!(
        workspace(&db, "owner", "chat", &at)
            .await
            .unwrap()
            .documents
            .len(),
        1
    );
}

#[tokio::test]
async fn chat_stream_creates_and_edits_canvas_in_its_exact_revision() {
    use axum::{
        Json, Router,
        body::Body,
        extract::{Path, State},
        routing::post,
    };
    use axum_extra::extract::{CookieJar, cookie::Cookie};
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};
    let requests = Arc::new(Mutex::new(Vec::<Value>::new()));
    let recorded = requests.clone();
    let mock=Router::new()
        .route("/api/show",post(||async {Json(json!({"capabilities":["completion","tools"]}))}))
        .route("/api/chat",post(move |Json(input):Json<Value>| {
            let recorded=recorded.clone();async move {
                let previous=input["messages"].as_array().unwrap().iter().rev().find(|m|m["role"]=="tool").and_then(|m|serde_json::from_str::<Value>(m["content"].as_str()?).ok());
                recorded.lock().unwrap().push(input);
                let response=match previous {
                    None=>json!({"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"create_canvas","arguments":{"title":"Synthetic streamed document","language":"markdown","content":"An old sentence."}}}]},"done":true}),
                    Some(ref result) if result["operation"]=="create"=>json!({"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"update_canvas","arguments":{"document_id":result["canvas"]["document_id"],"expected_version_id":result["canvas"]["version_id"],"edits":[{"find":"old","replace":"improved"}]}}}]},"done":true}),
                    _=>json!({"message":{"role":"assistant","content":"The canvas is ready."},"done":true,"done_reason":"stop"}),
                };
                axum::response::Response::builder().header("content-type","application/x-ndjson").body(Body::from(format!("{response}\n"))).unwrap()
            }
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, mock).await.unwrap();
    });
    struct Stop(tokio::task::JoinHandle<()>);
    impl Drop for Stop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _server = Stop(server);
    let db = fixture().await;
    sqlx::query("UPDATE ollama_backends SET base_url=? WHERE id='backend'")
        .bind(base)
        .execute(&db)
        .await
        .unwrap();
    sqlx::query("UPDATE app_settings SET tools_enabled=1 WHERE id=1")
        .execute(&db)
        .await
        .unwrap();
    crate::backends::service::set_model_availability(&db, "backend", "model", true)
        .await
        .unwrap();
    let directory =
        std::env::temp_dir().join(format!("vashti-canvas-stream-{}", uuid::Uuid::new_v4()));
    let config = crate::config::Config {
        app_root: directory.clone(),
        data_dir: directory.clone(),
        database_path: directory.join("app.db"),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        session_cookie_name: "canvas_fixture_session".into(),
        session_ttl_seconds: 3600,
        managed_updates: false,
        update_base_url: "https://example.invalid".into(),
    };
    let state = crate::app_state::AppState::new(
        config,
        db.clone(),
        reqwest::Client::new(),
        uuid::Uuid::new_v4().to_string(),
    );
    let session = crate::auth::service::create_session(&db, "owner", 3600, None, None)
        .await
        .unwrap();
    let jar = CookieJar::new().add(Cookie::new("canvas_fixture_session", session.id));
    let payload=serde_json::from_value(json!({"user_message":{"content_text":"Create and improve an editable synthetic document."},"think_mode":"off"})).unwrap();
    let response = crate::chats::handlers::generate_chat(
        State(state),
        jar,
        Path("chat".into()),
        Json(payload),
    )
    .await
    .unwrap();
    let body = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        axum::body::to_bytes(response.into_body(), 1024 * 1024),
    )
    .await
    .unwrap()
    .unwrap();
    let stream = String::from_utf8(body.to_vec()).unwrap();
    assert!(!stream.contains("\"type\":\"error\""), "{stream}");
    assert!(stream.contains("The canvas is ready."), "{stream}");
    let (_, messages) = crate::chats::service::list_messages(&db, "owner", "chat")
        .await
        .unwrap();
    let assistant = messages.iter().find(|m| m.role == "assistant").unwrap();
    let current = workspace(
        &db,
        "owner",
        "chat",
        &Scope {
            message_id: Some(assistant.id.clone()),
            revision_id: assistant.active_revision_id.clone(),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        current.documents[0].version.content,
        "An improved sentence."
    );
    assert_eq!(current.documents[0].version.actor, "model");
    assert_eq!(current.documents[0].version.version_number, 2);
    let input: String =
        sqlx::query_scalar("SELECT heads_json FROM canvas_generation_inputs WHERE revision_id=?")
            .bind(&assistant.active_revision_id)
            .fetch_one(&db)
            .await
            .unwrap();
    assert_eq!(input, "[]");
    assert!(
        workspace(&db, "owner", "chat", &scope())
            .await
            .unwrap()
            .documents
            .is_empty()
    );
    let requests = requests.lock().unwrap();
    assert!(
        requests[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["function"]["name"] == "create_canvas")
    );
    assert!(
        requests.iter().any(
            |r| r["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["role"] == "tool"
                    && m["content"]
                        .as_str()
                        .is_some_and(|t| t.contains("\"saved\":true")))
        )
    );
}
