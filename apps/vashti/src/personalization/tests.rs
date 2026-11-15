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
        assert!(serde_json::from_str::<serde_json::Value>(line).is_ok());
    }
    assert!(service::context_weight(&fitted) <= 600);
    assert_eq!(service::context_weight("abc"), 2);
    assert_eq!(service::context_weight("\u{e9}"), 2);
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
            axum::Json(json!({"done":true,"message":{"content":json!({"facts":[{"topic":"food.favorite","content":"The user prefers corn.","evidence":[{"source":0,"quote":"My favorite food is corn."}]}]}).to_string()}}))
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
