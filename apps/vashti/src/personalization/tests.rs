use super::{models::*, service};
use crate::{app_state::AppState, config::Config, startup};
use serde_json::json;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

#[test]
fn restrictive_group_defaults_and_individual_exceptions() {
    let mut settings = Settings {
        enabled: true,
        ..Settings::default()
    };
    let mut model = Model {
        key: "base:host:model".into(),
        name: "Local".into(),
        backend_id: "host".into(),
        model_name: "model".into(),
        cloud: false,
        character: false,
    };
    assert!(settings.allows(&model));
    model.cloud = true;
    assert!(!settings.allows(&model));
    settings.model_overrides.insert(model.key.clone(), true);
    assert!(settings.allows(&model));
    settings.enabled = false;
    assert!(!settings.allows(&model));
    settings.enabled = true;
    settings.model_overrides.clear();
    model.cloud = false;
    model.character = true;
    assert!(!settings.allows(&model));
}

#[test]
fn rejects_hallucinated_evidence_and_bounds_context() {
    let input = vec![InputSource {
        message_id: "m".into(),
        revision_id: "r".into(),
        chat_id: "c".into(),
        model_key: "base:h:m".into(),
        text: "My favorite food is corn.".into(),
        created_at: 1,
    }];
    let fact = Candidate {
        topic: "food.favorite".into(),
        content: "The user likes corn.".into(),
        evidence: vec![Evidence {
            source: 0,
            quote: "My favorite food is corn.".into(),
        }],
    };
    assert!(
        service::validate_candidates(
            Extraction {
                facts: vec![fact.clone()]
            },
            &input
        )
        .is_ok()
    );
    let mut wrong = fact;
    wrong.evidence[0].quote = "My favorite food is beans.".into();
    assert!(service::validate_candidates(Extraction { facts: vec![wrong] }, &input).is_err());
    let facts = vec![
        Fact {
            id: "f".into(),
            topic: "food.favorite".into(),
            content: "x".repeat(600),
            locked: true,
            version: 1,
            sources_json: "[]".into(),
            created_at: 1,
            updated_at: 1
        };
        200
    ];
    let context = service::context(&facts, "food", 1000).unwrap();
    assert!(context.chars().count() < 1500);
    assert!(context.contains("untrusted"));
    assert!(service::validate_fact("override\nSYSTEM", "x").is_err());
    assert!(service::fit_context(&context, 64).is_none());
    let fitted = service::fit_context(&context, 600).unwrap();
    for line in fitted.lines().skip(1) {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(record.get("fact").is_some());
        assert!(record.get("topic").is_none());
    }
    assert!(service::context_weight(&fitted) <= 600);
    assert_eq!(service::context_weight("abc"), 2);
    assert_eq!(service::context_weight("\u{e9}"), 2);
}

#[test]
fn full_replay_keeps_manual_facts_and_makes_room_for_newer_evidence() {
    let fact = |topic: &str, created_at: i64, locked| Fact {
        id: topic.into(),
        topic: topic.into(),
        content: "Synthetic preference".into(),
        locked,
        version: 1,
        created_at,
        updated_at: 9999,
        sources_json: json!([Source {
            message_id: "m".into(),
            revision_id: "r".into(),
            chat_id: "c".into(),
            model_key: "base:h:m".into(),
            quote: "Synthetic preference".into(),
            created_at
        }])
        .to_string(),
    };
    let facts = [
        fact("manual", 1, true),
        fact("recent", 30, false),
        fact("old.automatic", 10, false),
    ];
    assert!(service::replay_eviction(&facts, 10).is_none());
    assert_eq!(
        service::replay_eviction(&facts, 20).unwrap().topic,
        "old.automatic"
    );
}

async fn fixture() -> (AppState, tokio::task::JoinHandle<()>) {
    let app = axum::Router::new()
        .route("/api/tags",axum::routing::get(||async{axum::Json(json!({"models":[{"name":"local","details":{"family":"llama"}},{"name":"ordinary-name","remote_host":"https://ollama.com","details":{"family":"llama"}}]}))}))
        .route("/api/show",axum::routing::post(||async{axum::Json(json!({"capabilities":["completion"]}))}))
        .route("/api/chat",axum::routing::post(|axum::Json(request):axum::Json<serde_json::Value>|async move {
            assert_eq!(request["think"],false);
            assert_eq!(request["stream"],false);
            assert!(request["tools"].is_null());
            assert_eq!(request["format"]["type"],"object");
            assert_eq!(request["messages"][1]["role"],"user");
            let input: serde_json::Value = serde_json::from_str(request["messages"][1]["content"].as_str().unwrap()).unwrap();
            let sources = input["sources"].as_array().unwrap();
            assert!(sources.len() <= service::MAX_SOURCES);
            let facts: Vec<_> = sources.iter().rev().find_map(|source| {
                let quote = source["text"].as_str().unwrap();
                let content = match quote {
                    "My favorite food is corn." => "The user prefers corn.",
                    "My favorite food is imported beans." => "The user prefers beans.",
                    _ => return None,
                };
                Some(json!({"topic":"food.favorite","content":content,"evidence":[{"source":source["source"],"quote":quote}]}))
            }).into_iter().collect();
            axum::Json(json!({"done":true,"message":{"content":json!({"facts":facts}).to_string()}}))
        }));
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
    for user in ["owner", "other"] {
        sqlx::query("INSERT INTO users(id,username,password_hash,role,created_at,updated_at) VALUES (?,?,'unused','admin',1,1)").bind(user).bind(user).execute(&pool).await.unwrap();
    }
    sqlx::query("INSERT INTO ollama_backends(id,name,base_url,created_at,updated_at) VALUES('host','Fixture',?,1,1)").bind(base).execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO chats(id,user_id,default_backend_id,default_model_name,title,created_at,updated_at,last_message_at) VALUES('chat','owner','host','local','Synthetic',1,1,1)").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO chat_messages(id,chat_id,role,status,active_revision_id,created_at,updated_at) VALUES('message','chat','user','complete','revision',1,1)").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) VALUES('revision','message','My favorite food is corn.','original',1)").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO chat_messages(id,chat_id,parent_message_id,role,status,backend_id,model_name,created_at,updated_at) VALUES('answer','chat','message','assistant','complete','host','local',2,2)").execute(&pool).await.unwrap();
    let state = AppState::new(
        Config::from_env().unwrap(),
        pool,
        reqwest::Client::new(),
        "fixture".into(),
    );
    state
        .model_cache
        .refresh_all(&state.db, &state.http_client)
        .await
        .unwrap();
    (state, server)
}

async fn enable(state: &AppState) -> Profile {
    service::save_settings(
        state,
        "owner",
        SaveSettings {
            expected_revision: 0,
            settings: Settings {
                enabled: true,
                processor_backend_id: Some("host".into()),
                processor_model: Some("local".into()),
                ..Settings::default()
            },
        },
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn processor_extracts_owned_evidence_and_deleted_or_edited_sources_stop_injection() {
    let (state, server) = fixture().await;
    enable(&state).await;
    assert!(
        super::worker::process_one(&state, "owner", true)
            .await
            .unwrap()
    );
    let profile = service::profile(&state.db, "owner").await.unwrap();
    assert_eq!(profile.facts.len(), 1);
    let context = service::chat_context(
        &state,
        "owner",
        "base:host:local",
        "host",
        "local",
        None,
        "food",
    )
    .await
    .unwrap()
    .unwrap();
    assert!(context.contains("corn"));
    assert!(
        service::chat_context(
            &state,
            "other",
            "base:host:local",
            "host",
            "local",
            None,
            "food"
        )
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        service::fact_history(&state.db, "other", &profile.facts[0].id)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        service::chat_context(
            &state,
            "owner",
            "base:host:ordinary-name",
            "host",
            "ordinary-name",
            None,
            "food"
        )
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        service::authorize_processor(&state, "owner", "host", "ordinary-name", false)
            .await
            .is_err()
    );
    sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) VALUES('changed','message','I no longer have a favorite food.','edit',3)").execute(&state.db).await.unwrap();
    sqlx::query("UPDATE chat_messages SET active_revision_id='changed' WHERE id='message'")
        .execute(&state.db)
        .await
        .unwrap();
    assert!(
        service::chat_context(
            &state,
            "owner",
            "base:host:local",
            "host",
            "local",
            None,
            "food"
        )
        .await
        .unwrap()
        .is_none()
    );
    sqlx::query(
        "UPDATE chat_messages SET active_revision_id='revision',is_deleted=1 WHERE id='message'",
    )
    .execute(&state.db)
    .await
    .unwrap();
    assert!(
        service::chat_context(
            &state,
            "owner",
            "base:host:local",
            "host",
            "local",
            None,
            "food"
        )
        .await
        .unwrap()
        .is_none()
    );
    server.abort();
}

#[tokio::test]
async fn human_corrections_win_forget_erases_history_and_stale_jobs_cannot_publish() {
    let (state, server) = fixture().await;
    let first = enable(&state).await;
    let sources = service::sources(&state, "owner", &first.settings, true)
        .await
        .unwrap();
    let candidate = Candidate {
        topic: "food.favorite".into(),
        content: "The user prefers corn.".into(),
        evidence: vec![Evidence {
            source: 0,
            quote: sources[0].text.clone(),
        }],
    };
    let validated = || {
        service::validate_candidates(
            Extraction {
                facts: vec![candidate.clone()],
            },
            &sources,
        )
        .unwrap()
    };
    service::publish(&state, "owner", &first, &sources, validated())
        .await
        .unwrap();
    assert!(
        service::publish(&state, "owner", &first, &sources, validated())
            .await
            .is_err()
    );
    let profile = service::profile(&state.db, "owner").await.unwrap();
    let fact = &profile.facts[0];
    let fixed = service::edit_fact(
        &state.db,
        "owner",
        Some(&fact.id),
        EditFact {
            expected_revision: profile.revision,
            topic: fact.topic.clone(),
            content: "The user prefers peas.".into(),
        },
    )
    .await
    .unwrap();
    service::publish(&state, "owner", &fixed, &sources, validated())
        .await
        .unwrap();
    let profile = service::profile(&state.db, "owner").await.unwrap();
    assert!(profile.facts[0].content.contains("peas"));
    assert_eq!(
        service::fact_history(&state.db, "owner", &fact.id)
            .await
            .unwrap()
            .len(),
        2
    );
    let forgotten = service::forget(&state.db, "owner", Some(&fact.id), profile.revision)
        .await
        .unwrap();
    assert!(forgotten.facts.is_empty());
    assert!(
        service::fact_history(&state.db, "owner", &fact.id)
            .await
            .unwrap()
            .is_empty()
    );
    service::publish(&state, "owner", &forgotten, &sources, validated())
        .await
        .unwrap();
    assert!(
        service::profile(&state.db, "owner")
            .await
            .unwrap()
            .facts
            .is_empty()
    );
    server.abort();
}

#[tokio::test]
async fn settings_and_source_changes_during_work_are_rechecked() {
    let (state, server) = fixture().await;
    let before = enable(&state).await;
    let sources = service::sources(&state, "owner", &before.settings, true)
        .await
        .unwrap();
    let values = service::validate_candidates(
        Extraction {
            facts: vec![Candidate {
                topic: "food.favorite".into(),
                content: "The user prefers corn.".into(),
                evidence: vec![Evidence {
                    source: 0,
                    quote: sources[0].text.clone(),
                }],
            }],
        },
        &sources,
    )
    .unwrap();
    sqlx::query("UPDATE chat_messages SET is_deleted=1 WHERE id='message'")
        .execute(&state.db)
        .await
        .unwrap();
    assert!(
        service::publish(&state, "owner", &before, &sources, values.clone())
            .await
            .is_err()
    );
    sqlx::query("UPDATE chat_messages SET is_deleted=0 WHERE id='message'")
        .execute(&state.db)
        .await
        .unwrap();
    let mut settings = before.settings.clone();
    settings.enabled = false;
    service::save_settings(
        &state,
        "owner",
        SaveSettings {
            expected_revision: before.revision,
            settings,
        },
    )
    .await
    .unwrap();
    assert!(
        service::publish(&state, "owner", &before, &sources, values)
            .await
            .is_err()
    );
    assert!(
        service::chat_context(
            &state,
            "owner",
            "base:host:local",
            "host",
            "local",
            None,
            ""
        )
        .await
        .unwrap()
        .is_none()
    );
    server.abort();
}

#[tokio::test]
async fn historical_character_sources_stay_excluded_after_model_settings_change() {
    let (state, server) = fixture().await;
    sqlx::query("INSERT INTO personas(id,owner_user_id,current_version_id,created_at,updated_at) VALUES('character','owner','general-version',1,1)").execute(&state.db).await.unwrap();
    for (version, number, kind) in [
        ("character-version", 1, "character"),
        ("general-version", 2, "general"),
    ] {
        sqlx::query("INSERT INTO persona_versions(id,persona_id,version_number,display_name,base_backend_id,base_model_name,system_prompt,model_type,created_at) VALUES (?,'character',?,'Fixture character','host','local','',?,1)")
            .bind(version).bind(number).bind(kind).execute(&state.db).await.unwrap();
    }
    sqlx::query("UPDATE personalization_chat_sources SET persona_id='character',persona_version_id='character-version' WHERE message_id='message'").execute(&state.db).await.unwrap();
    let profile = enable(&state).await;
    assert!(
        service::sources(&state, "owner", &profile.settings, true)
            .await
            .unwrap()
            .is_empty()
    );
    let mut settings = profile.settings.clone();
    settings
        .model_overrides
        .insert("persona:character".into(), true);
    let profile = service::save_settings(
        &state,
        "owner",
        SaveSettings {
            expected_revision: profile.revision,
            settings,
        },
    )
    .await
    .unwrap();
    assert_eq!(
        service::sources(&state, "owner", &profile.settings, true)
            .await
            .unwrap()
            .len(),
        1
    );
    let mut settings = profile.settings.clone();
    settings.model_overrides.clear();
    let profile = service::save_settings(
        &state,
        "owner",
        SaveSettings {
            expected_revision: profile.revision,
            settings,
        },
    )
    .await
    .unwrap();
    assert!(
        service::sources(&state, "owner", &profile.settings, true)
            .await
            .unwrap()
            .is_empty()
    );
    server.abort();
}

#[test]
fn metadata_not_display_name_controls_cloud_and_native_format_classification() {
    let tags: crate::ollama::models::TagsResponse = serde_json::from_value(json!({"models":[
        {"name":"innocent-name","remote_host":"https://ollama.com"},
        {"name":"native-qwen","kuno_backend":"rust-intel"},
        {"name":"local"}
    ]}))
    .unwrap();
    let models = tags.into_models();
    assert!(models[0].is_cloud);
    assert!(!models[1].is_cloud);
    assert!(!models[1].supports_structured_output);
    assert!(models[2].supports_structured_output);
}

#[tokio::test]
async fn edited_revisions_keep_their_own_model_classification() {
    let (state, server) = fixture().await;
    let profile = enable(&state).await;
    sqlx::query("UPDATE chats SET default_model_name='ordinary-name' WHERE id='chat'")
        .execute(&state.db)
        .await
        .unwrap();
    sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) VALUES('cloud-edit','message','My favorite food is corn.','edit',3)")
        .execute(&state.db).await.unwrap();
    sqlx::query("UPDATE chat_messages SET active_revision_id='cloud-edit' WHERE id='message'")
        .execute(&state.db)
        .await
        .unwrap();
    // Switching the conversation back must not grant access to the excluded edit.
    sqlx::query("UPDATE chats SET default_model_name='local' WHERE id='chat'")
        .execute(&state.db)
        .await
        .unwrap();
    assert!(
        service::sources(&state, "owner", &profile.settings, true)
            .await
            .unwrap()
            .is_empty()
    );
    sqlx::query("INSERT INTO chat_messages(id,chat_id,parent_message_id,role,status,backend_id,model_name,created_at,updated_at) VALUES('later-answer','chat','message','assistant','complete','host','local',4,4)")
        .execute(&state.db).await.unwrap();
    assert!(
        service::sources(&state, "owner", &profile.settings, true)
            .await
            .unwrap()
            .is_empty()
    );
    sqlx::query("UPDATE chat_messages SET active_revision_id='revision' WHERE id='message'")
        .execute(&state.db)
        .await
        .unwrap();
    let original = service::sources(&state, "owner", &profile.settings, true)
        .await
        .unwrap();
    assert_eq!(original.len(), 1);
    assert_eq!(original[0].model_key, "base:host:local");
    sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) VALUES('local-edit','message','My favorite food is carrots.','edit',5)")
        .execute(&state.db).await.unwrap();
    sqlx::query("UPDATE chat_messages SET active_revision_id='local-edit' WHERE id='message'")
        .execute(&state.db)
        .await
        .unwrap();
    let edited = service::sources(&state, "owner", &profile.settings, true)
        .await
        .unwrap();
    assert_eq!(edited.len(), 1);
    assert_eq!(edited[0].revision_id, "local-edit");
    server.abort();
}

#[tokio::test]
async fn excluded_history_cannot_hide_later_sources_and_foreground_work_has_priority() {
    let (state, server) = fixture().await;
    let profile = enable(&state).await;
    sqlx::query("UPDATE chat_message_revisions SET created_at=5000 WHERE id='revision'")
        .execute(&state.db)
        .await
        .unwrap();
    sqlx::query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2001) INSERT INTO chat_messages(id,chat_id,active_revision_id,role,status,created_at,updated_at) SELECT 'excluded-'||x,'chat','excluded-revision-'||x,'user','complete',x,x FROM n").execute(&state.db).await.unwrap();
    sqlx::query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2001) INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) SELECT 'excluded-revision-'||x,'excluded-'||x,'A synthetic cloud-only message.','original',x FROM n").execute(&state.db).await.unwrap();
    sqlx::query("UPDATE personalization_chat_sources SET model_name='ordinary-name' WHERE message_id LIKE 'excluded-%'").execute(&state.db).await.unwrap();
    let sources = service::sources(&state, "owner", &profile.settings, true)
        .await
        .unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].message_id, "message");
    let activity = crate::app_state::PrivateActivityGuard::enter(state.private_activity.clone());
    assert!(super::worker::foreground_busy(&state).await);
    assert!(
        !super::worker::process_one(&state, "owner", true)
            .await
            .unwrap()
    );
    drop(activity);
    assert!(!super::worker::foreground_busy(&state).await);
    let lane = state
        .personalization_lane
        .clone()
        .try_acquire_owned()
        .unwrap();
    assert!(
        !super::worker::process_one(&state, "owner", true)
            .await
            .unwrap()
    );
    drop(lane);
    server.abort();
}

#[tokio::test]
async fn replay_reopens_a_thread_in_time_order_with_only_its_own_ancestor_context() {
    let (state, server) = fixture().await;
    let profile = enable(&state).await;
    sqlx::raw_sql("INSERT INTO chats(id,user_id,default_backend_id,default_model_name,title,created_at,updated_at,last_message_at) VALUES('thread-b','owner','host','local','Thread B',10,10,10),('thread-c','owner','host','local','Thread C',40,40,40);
        INSERT INTO chat_messages(id,chat_id,parent_message_id,role,status,active_revision_id,created_at,updated_at) VALUES
           ('sibling','chat','message','user','complete','sibling-rev',3,3),
           ('b','thread-b',NULL,'user','complete','b-rev',10,10),
           ('reopen','chat','answer','user','complete','reopen-rev',30,30),
           ('c','thread-c',NULL,'user','complete','c-rev',40,40);
        INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) VALUES
           ('sibling-rev','sibling','This different branch mentions fictional moon rocks.','original',3),
           ('b-rev','b','My other permanent preference is paper checklists.','original',10),
           ('reopen-rev','reopen','Actually, make that tamales going forward.','original',30),
           ('c-rev','c','My preferred way to test changes is a tiny prototype.','original',40);")
        .execute(&state.db).await.unwrap();
    let sources = service::sources(&state, "owner", &profile.settings, true)
        .await
        .unwrap();
    assert_eq!(
        sources
            .iter()
            .map(|source| source.message_id.as_str())
            .collect::<Vec<_>>(),
        ["message", "sibling", "b", "reopen", "c"]
    );
    let hints = service::replay_hints(&state, "owner", &profile.settings, &sources, &[])
        .await
        .unwrap();
    let payload = super::worker::replay_input(&sources, &[], &hints);
    assert_eq!(payload["conversation_runs"], json!([[0, 1], [2], [3], [4]]));
    assert_eq!(
        payload["earlier_turns"],
        json!([{"source":3,"excerpts":["My favorite food is corn."]}])
    );
    assert!(!payload.to_string().contains("thread-b"));
    sqlx::query("INSERT INTO personalization_processed(user_id,revision_id,explicitly_forgotten) VALUES('owner','revision',1)")
        .execute(&state.db).await.unwrap();
    let hints = service::replay_hints(&state, "owner", &profile.settings, &sources, &[])
        .await
        .unwrap();
    assert!(hints.earlier_turns.is_empty());
    server.abort();
}

#[tokio::test]
async fn upgrade_makes_existing_imports_eligible_without_reimporting() {
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
        .filter(|m| m.version < 20261122010000)
        .cloned()
        .collect::<Vec<_>>()
        .into();
    old.run(&pool).await.unwrap();
    startup::bootstrap::ensure_app_settings(&pool)
        .await
        .unwrap();
    sqlx::raw_sql("INSERT INTO users(id,username,password_hash,role,created_at,updated_at) VALUES('owner','owner','unused','admin',1,1);
        INSERT INTO ollama_backends(id,name,base_url,created_at,updated_at) VALUES('host','Fixture','http://127.0.0.1:1',1,1);
        INSERT INTO chats(id,user_id,default_backend_id,default_model_name,title,created_at,updated_at,last_message_at) VALUES('old-chat','owner','host','local','Previously imported',1,1,1);
        INSERT INTO chat_messages(id,chat_id,role,status,created_at,updated_at,import_source) VALUES('old-message','old-chat','user','complete',1,1,'chatgpt');
        INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) VALUES('old-revision','old-message','My favorite food is corn.','original',1);
        UPDATE chat_messages SET active_revision_id='old-revision' WHERE id='old-message';
        INSERT INTO chat_import_records(id,chat_id,user_id,source_id,source_hash,original_json,created_at) VALUES('old-import','old-chat','owner','old-source','hash','{}',2);")
        .execute(&pool).await.unwrap();
    startup::migrations::run(&pool).await.unwrap();
    let state = AppState::new(
        Config::from_env().unwrap(),
        pool,
        reqwest::Client::new(),
        "fixture".into(),
    );
    let sources = service::sources(
        &state,
        "owner",
        &Settings {
            enabled: true,
            ..Settings::default()
        },
        true,
    )
    .await
    .unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].revision_id, "old-revision");
    assert_eq!(sources[0].model_key, "import:chatgpt");
}

#[tokio::test]
async fn bulk_chatgpt_import_rebuilds_in_batches_and_notifies_once() {
    let (state, server) = fixture().await;
    enable(&state).await;
    assert!(
        super::worker::process_one(&state, "owner", true)
            .await
            .unwrap()
    );
    for index in 0..1500 {
        let source_id = format!("synthetic-{index}");
        let created = 10 + index;
        let content = if index == 1499 {
            "My favorite food is corn."
        } else {
            "My favorite food is imported beans."
        };
        let input = serde_json::from_value(json!({
            "backend_id":"host", "model_name":"local", "allow_history_search":index%2==0,
            "learn_from_import":if index == 2 { Some(false) } else { None },
            "conversation":{
                "source_id":source_id, "title":"Synthetic imported chat", "created_at":created,
                "updated_at":created+1, "archived":index%2==0, "do_not_remember":index==1,
                "current_node_id":"a", "assets":[], "messages":[
                    {"source_id":"q","parent_id":null,"role":"user","content":content,"thinking":"","model":null,"created_at":created},
                    {"source_id":"a","parent_id":"q","role":"assistant","content":"Imported reply.","thinking":"","model":"original","created_at":created+1}
                ], "original_json":json!({"id":source_id}).to_string()
            }
        })).unwrap();
        let imported = crate::imports::service::import_conversation(&state.db, "owner", input)
            .await
            .unwrap();
        assert!(!imported.existing);
    }
    let before = service::profile(&state.db, "owner").await.unwrap();
    let mut disabled = before.settings.clone();
    disabled.learn_from_imports = false;
    assert_eq!(
        service::sources(&state, "owner", &disabled, true)
            .await
            .unwrap()
            .len(),
        0
    );
    let sources = service::sources(&state, "owner", &before.settings, true)
        .await
        .unwrap();
    assert_eq!(sources.len(), 16);
    assert_eq!(sources[0].model_key, "import:chatgpt");
    assert_eq!(sources[0].text, "My favorite food is imported beans.");
    let manual = service::edit_fact(
        &state.db,
        "owner",
        None,
        EditFact {
            expected_revision: before.revision,
            topic: "answer.style".into(),
            content: "Keep explanations concise.".into(),
        },
    )
    .await
    .unwrap();
    let queued = service::rebuild(&state, "owner", manual.revision)
        .await
        .unwrap();
    assert!(!queued.requested);
    let id = queued.rebuild_id.unwrap();
    assert!(
        super::worker::process_one(&state, "owner", false)
            .await
            .unwrap()
    );
    let ongoing = service::profile(&state.db, "owner").await.unwrap();
    assert_eq!(
        ongoing
            .facts
            .iter()
            .find(|fact| fact.topic == "food.favorite")
            .unwrap()
            .content,
        "The user prefers corn."
    );
    let draft = service::processing_facts(&state.db, "owner", &ongoing)
        .await
        .unwrap();
    assert_eq!(
        draft
            .iter()
            .find(|fact| fact.topic == "food.favorite")
            .unwrap()
            .content,
        "The user prefers beans."
    );
    let mut batches = 1;
    while super::worker::process_one(&state, "owner", false)
        .await
        .unwrap()
    {
        batches += 1;
        assert!(batches <= 94);
    }
    assert_eq!(batches, 94);
    let processed: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM personalization_processed WHERE user_id='owner'")
            .fetch_one(&state.db)
            .await
            .unwrap();
    assert_eq!(processed, 1499); // 1498 eligible imports and one native user message.
    let after = service::profile(&state.db, "owner").await.unwrap();
    assert!(after.rebuild_id.is_none());
    assert!(
        after
            .facts
            .iter()
            .any(|fact| fact.locked && fact.topic == "answer.style")
    );
    assert_eq!(
        after
            .facts
            .iter()
            .find(|fact| fact.topic == "food.favorite")
            .unwrap()
            .content,
        "The user prefers corn."
    );
    let notices: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM assistant_notices WHERE user_id='owner' AND read_at IS NULL",
    )
    .fetch_all(&state.db)
    .await
    .unwrap();
    assert_eq!(notices, vec![id]);
    assert!(
        service::chat_context(
            &state,
            "owner",
            "base:host:local",
            "host",
            "local",
            None,
            "favorite food"
        )
        .await
        .unwrap()
        .unwrap()
        .contains("corn")
    );
    server.abort();
}

#[tokio::test]
async fn a_manual_request_drains_multiple_bounded_batches() {
    let (state, server) = fixture().await;
    enable(&state).await;
    sqlx::query("UPDATE user_personalization SET requested=1 WHERE user_id='owner'")
        .execute(&state.db)
        .await
        .unwrap();
    sqlx::query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<16) INSERT INTO chat_messages(id,chat_id,active_revision_id,role,status,created_at,updated_at) SELECT 'extra-'||x,'chat','extra-revision-'||x,'user','complete',x+2,x+2 FROM n").execute(&state.db).await.unwrap();
    sqlx::query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<16) INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) SELECT 'extra-revision-'||x,'extra-'||x,'My favorite food is corn.','original',x+2 FROM n").execute(&state.db).await.unwrap();
    assert!(
        super::worker::process_one(&state, "owner", true)
            .await
            .unwrap()
    );
    assert!(
        service::profile(&state.db, "owner")
            .await
            .unwrap()
            .requested
    );
    let processed: i64 =
        sqlx::query_scalar("SELECT count(*) FROM personalization_processed WHERE user_id='owner'")
            .fetch_one(&state.db)
            .await
            .unwrap();
    assert_eq!(processed, 16);
    assert!(
        super::worker::process_one(&state, "owner", true)
            .await
            .unwrap()
    );
    assert!(
        !service::profile(&state.db, "owner")
            .await
            .unwrap()
            .requested
    );
    let processed: i64 =
        sqlx::query_scalar("SELECT count(*) FROM personalization_processed WHERE user_id='owner'")
            .fetch_one(&state.db)
            .await
            .unwrap();
    assert_eq!(processed, 17);
    server.abort();
}

#[tokio::test]
async fn repeated_facts_keep_fresh_sources_without_spamming_history() {
    let (state, server) = fixture().await;
    enable(&state).await;
    super::worker::process_one(&state, "owner", true)
        .await
        .unwrap();
    sqlx::query("INSERT INTO chat_messages(id,chat_id,active_revision_id,role,status,created_at,updated_at) VALUES('confirmation','chat','confirmed-revision','user','complete',3,3)").execute(&state.db).await.unwrap();
    sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,source,created_at) VALUES('confirmed-revision','confirmation','My favorite food is corn.','original',3)").execute(&state.db).await.unwrap();
    super::worker::process_one(&state, "owner", true)
        .await
        .unwrap();
    let profile = service::profile(&state.db, "owner").await.unwrap();
    assert_eq!(profile.facts[0].version, 1);
    assert_eq!(
        service::fact_history(&state.db, "owner", &profile.facts[0].id)
            .await
            .unwrap()
            .len(),
        1
    );
    sqlx::query("UPDATE chat_messages SET is_deleted=1 WHERE id='message'")
        .execute(&state.db)
        .await
        .unwrap();
    assert!(
        service::chat_context(
            &state,
            "owner",
            "base:host:local",
            "host",
            "local",
            None,
            "food"
        )
        .await
        .unwrap()
        .unwrap()
        .contains("corn")
    );
    server.abort();
}

#[tokio::test]
async fn forgetting_tokens_do_not_retain_personal_values_and_legacy_tokens_upgrade() {
    let (state, server) = fixture().await;
    sqlx::query("INSERT INTO personalization_suppressed VALUES('owner','private_topic_with_personal_value')").execute(&state.db).await.unwrap();
    service::migrate_suppression_tokens(&state.db)
        .await
        .unwrap();
    service::migrate_suppression_tokens(&state.db)
        .await
        .unwrap();
    let values = sqlx::query_scalar::<_, String>(
        "SELECT topic FROM personalization_suppressed WHERE user_id='owner'",
    )
    .fetch_all(&state.db)
    .await
    .unwrap();
    assert_eq!(
        values,
        vec![service::topic_token("private_topic_with_personal_value")]
    );
    assert!(!values[0].contains("personal_value"));
    assert_eq!(values[0].len(), 71);
    server.abort();
}
