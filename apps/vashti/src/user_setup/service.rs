use std::collections::{HashMap, HashSet};

use serde::Serialize;
use sqlx::{SqliteConnection, SqlitePool};

use crate::{auth::service::unix_timestamp, error::ApiError};

use super::handlers::SetupChoiceInput;

pub const NOTES_ALLOW_MODEL_READ: &str = "notes.allow_model_read";
pub const NOTES_ALLOW_MODEL_CREATE: &str = "notes.allow_model_create";
pub const NOTES_ALLOW_MODEL_EDIT: &str = "notes.allow_model_edit";
pub const NOTES_ALLOW_MODEL_TRASH: &str = "notes.allow_model_trash";
pub const MEMORIES_ALLOW_MODEL_READ: &str = "memories.allow_model_read";
pub const MEMORIES_ALLOW_MODEL_CREATE: &str = "memories.allow_model_create";
pub const MEMORIES_ALLOW_MODEL_EDIT: &str = "memories.allow_model_edit";
pub const MEMORIES_ALLOW_MODEL_FORGET: &str = "memories.allow_model_forget";
pub const CHAT_HISTORY_ALLOW_MODEL_SEARCH: &str = "chat_history.allow_model_search";

const SOURCE_SETUP: &str = "setup";
pub const SOURCE_SETTINGS: &str = "settings";

#[derive(Clone, Copy)]
struct ChoiceDefinition {
    key: &'static str,
    group_id: &'static str,
    title: &'static str,
    description: &'static str,
    recommended_value: bool,
}

const CHOICES: &[ChoiceDefinition] = &[
    ChoiceDefinition {
        key: NOTES_ALLOW_MODEL_READ,
        group_id: "notes",
        title: "Find and read notes",
        description: "Search and open notes that you make available to models.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: NOTES_ALLOW_MODEL_CREATE,
        group_id: "notes",
        title: "Create notes",
        description: "Make new notes when you ask during a chat.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: NOTES_ALLOW_MODEL_EDIT,
        group_id: "notes",
        title: "Edit notes",
        description: "Update available notes while preserving their version history.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: NOTES_ALLOW_MODEL_TRASH,
        group_id: "notes",
        title: "Move notes to Trash",
        description: "Move notes out of the library without permanently deleting them.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: MEMORIES_ALLOW_MODEL_READ,
        group_id: "memories",
        title: "Find and read memories",
        description: "Use relevant saved memories when they can improve a response.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: MEMORIES_ALLOW_MODEL_CREATE,
        group_id: "memories",
        title: "Create memories",
        description: "Save durable facts and preferences when they are clearly useful later.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: MEMORIES_ALLOW_MODEL_EDIT,
        group_id: "memories",
        title: "Update memories",
        description: "Correct or refine saved memories while preserving their history.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: MEMORIES_ALLOW_MODEL_FORGET,
        group_id: "memories",
        title: "Forget memories",
        description: "Move memories to the recoverable Forgotten list.",
        recommended_value: true,
    },
    ChoiceDefinition {
        key: CHAT_HISTORY_ALLOW_MODEL_SEARCH,
        group_id: "past_chats",
        title: "Search past chats",
        description: "Search completed server chats and read the branch around a useful match. Private chats stay on this device.",
        recommended_value: true,
    },
];

#[derive(Debug, Serialize)]
pub struct SetupStatusResponse {
    pub pending_count: usize,
    pub groups: Vec<SetupChoiceGroupResponse>,
}

#[derive(Debug, Serialize)]
pub struct SetupChoiceGroupResponse {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub choices: Vec<SetupChoiceResponse>,
}

#[derive(Debug, Serialize)]
pub struct SetupChoiceResponse {
    pub key: &'static str,
    pub kind: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub recommended_value: bool,
}

pub async fn has_pending_choices(pool: &SqlitePool, user_id: &str) -> Result<bool, sqlx::Error> {
    let decisions = decision_keys(pool, user_id).await?;
    Ok(CHOICES.iter().any(|choice| !decisions.contains(choice.key)))
}

pub async fn get_setup_status(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<SetupStatusResponse, ApiError> {
    let decisions = decision_keys(pool, user_id).await?;
    let mut groups = Vec::new();
    for (id, title, description) in [
        (
            "notes",
            "Notes",
            "Choose what models may do with your server notes when Notes is enabled for a chat.",
        ),
        (
            "memories",
            "Memories",
            "Choose how models may use durable facts and preferences saved to your account.",
        ),
        (
            "past_chats",
            "Past Chats",
            "Choose whether models may search your completed server conversation history.",
        ),
    ] {
        let choices = CHOICES
            .iter()
            .filter(|choice| choice.group_id == id && !decisions.contains(choice.key))
            .map(|choice| SetupChoiceResponse {
                key: choice.key,
                kind: "toggle",
                title: choice.title,
                description: choice.description,
                recommended_value: choice.recommended_value,
            })
            .collect::<Vec<_>>();
        if !choices.is_empty() {
            groups.push(SetupChoiceGroupResponse {
                id,
                title,
                description,
                choices,
            });
        }
    }
    let pending_count = groups.iter().map(|group| group.choices.len()).sum();
    Ok(SetupStatusResponse {
        pending_count,
        groups,
    })
}

pub async fn apply_setup_choices(
    pool: &SqlitePool,
    user_id: &str,
    choices: Vec<SetupChoiceInput>,
) -> Result<(), ApiError> {
    if choices.is_empty() {
        return Err(ApiError::bad_request(
            "setup_choices_required",
            "Choose at least one setting",
        ));
    }

    let known = CHOICES
        .iter()
        .map(|choice| choice.key)
        .collect::<HashSet<_>>();
    let decided = decision_keys(pool, user_id).await?;
    let mut values = HashMap::new();
    for choice in choices {
        if !known.contains(choice.key.as_str()) {
            return Err(ApiError::bad_request(
                "unknown_setup_choice",
                format!("Unknown setup setting: {}", choice.key),
            ));
        }
        if decided.contains(choice.key.as_str()) {
            return Err(ApiError::conflict(
                "setup_choice_already_decided",
                "One of these settings was already saved. Reload setup and try again.",
            ));
        }
        if values.insert(choice.key.clone(), choice.value).is_some() {
            return Err(ApiError::bad_request(
                "duplicate_setup_choice",
                format!("Setup setting was included more than once: {}", choice.key),
            ));
        }
    }

    let note_settings = if values.keys().any(|key| key.starts_with("notes.")) {
        Some(crate::notes::service::get_note_settings(pool, user_id).await?)
    } else {
        None
    };
    let memory_settings = if values
        .keys()
        .any(|key| key.starts_with("memories.") || key.starts_with("chat_history."))
    {
        Some(crate::memories::service::get_memory_settings(pool, user_id).await?)
    } else {
        None
    };

    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    if let Some(settings) = note_settings {
        let allow_model_read = value_or(&values, NOTES_ALLOW_MODEL_READ, settings.allow_model_read);
        let allow_model_create = value_or(
            &values,
            NOTES_ALLOW_MODEL_CREATE,
            settings.allow_model_create,
        );
        let allow_model_edit = value_or(&values, NOTES_ALLOW_MODEL_EDIT, settings.allow_model_edit);
        let allow_model_trash =
            value_or(&values, NOTES_ALLOW_MODEL_TRASH, settings.allow_model_trash);
        if !allow_model_read && (allow_model_edit || allow_model_trash) {
            return Err(ApiError::bad_request(
                "invalid_notes_setup",
                "Models must be allowed to read notes before they can edit or trash them",
            ));
        }
        sqlx::query(
            r#"
            INSERT INTO user_note_settings (
                user_id, allow_model_read, allow_model_create, allow_model_edit,
                allow_model_trash, default_ai_access, default_all_models, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                allow_model_read = excluded.allow_model_read,
                allow_model_create = excluded.allow_model_create,
                allow_model_edit = excluded.allow_model_edit,
                allow_model_trash = excluded.allow_model_trash,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(user_id)
        .bind(allow_model_read)
        .bind(allow_model_create)
        .bind(allow_model_edit)
        .bind(allow_model_trash)
        .bind(settings.default_ai_access.as_str())
        .bind(settings.default_model_scope.all_models)
        .bind(now)
        .execute(&mut *tx)
        .await?;
    }

    if let Some(settings) = memory_settings {
        let allow_model_read = value_or(
            &values,
            MEMORIES_ALLOW_MODEL_READ,
            settings.allow_model_read,
        );
        let allow_model_create = value_or(
            &values,
            MEMORIES_ALLOW_MODEL_CREATE,
            settings.allow_model_create,
        );
        let allow_model_edit = value_or(
            &values,
            MEMORIES_ALLOW_MODEL_EDIT,
            settings.allow_model_edit,
        );
        let allow_model_forget = value_or(
            &values,
            MEMORIES_ALLOW_MODEL_FORGET,
            settings.allow_model_forget,
        );
        if !allow_model_read && (allow_model_edit || allow_model_forget) {
            return Err(ApiError::bad_request(
                "invalid_memories_setup",
                "Models must be allowed to read memories before they can update or forget them",
            ));
        }
        sqlx::query(
            r#"
            INSERT INTO user_memory_settings (
                user_id, allow_model_read, allow_model_create, allow_model_edit,
                allow_model_forget, allow_model_chat_history, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                allow_model_read = excluded.allow_model_read,
                allow_model_create = excluded.allow_model_create,
                allow_model_edit = excluded.allow_model_edit,
                allow_model_forget = excluded.allow_model_forget,
                allow_model_chat_history = excluded.allow_model_chat_history,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(user_id)
        .bind(allow_model_read)
        .bind(allow_model_create)
        .bind(allow_model_edit)
        .bind(allow_model_forget)
        .bind(value_or(
            &values,
            CHAT_HISTORY_ALLOW_MODEL_SEARCH,
            settings.allow_model_chat_history,
        ))
        .bind(now)
        .execute(&mut *tx)
        .await?;
    }

    let recorded = values
        .iter()
        .map(|(key, value)| (key.as_str(), *value))
        .collect::<Vec<_>>();
    record_boolean_decisions(&mut tx, user_id, SOURCE_SETUP, &recorded, now).await?;
    tx.commit().await?;
    Ok(())
}

pub async fn record_boolean_decisions(
    connection: &mut SqliteConnection,
    user_id: &str,
    source: &str,
    values: &[(&str, bool)],
    now: i64,
) -> Result<(), sqlx::Error> {
    for (key, value) in values {
        sqlx::query(
            r#"
            INSERT INTO user_setting_decisions (
                user_id, setting_key, value_json, source, decided_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, setting_key) DO UPDATE SET
                value_json = excluded.value_json,
                source = excluded.source,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(user_id)
        .bind(key)
        .bind(if *value { "true" } else { "false" })
        .bind(source)
        .bind(now)
        .bind(now)
        .execute(&mut *connection)
        .await?;
    }
    Ok(())
}

async fn decision_keys(pool: &SqlitePool, user_id: &str) -> Result<HashSet<String>, sqlx::Error> {
    Ok(
        sqlx::query_scalar("SELECT setting_key FROM user_setting_decisions WHERE user_id = ?")
            .bind(user_id)
            .fetch_all(pool)
            .await?
            .into_iter()
            .collect(),
    )
}

fn value_or(values: &HashMap<String, bool>, key: &str, fallback: bool) -> bool {
    values.get(key).copied().unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    use super::*;
    use crate::{
        memories::models::UpdateMemorySettingsRequest,
        notes::models::{NoteAiAccess, NoteModelScope, UpdateNoteSettingsRequest},
        startup,
    };

    async fn test_pool() -> SqlitePool {
        let options = SqliteConnectOptions::new()
            .filename(":memory:")
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .expect("connect test database");
        startup::migrations::run(&pool)
            .await
            .expect("run migrations");
        startup::bootstrap::ensure_app_settings(&pool)
            .await
            .expect("ensure app settings");
        sqlx::query(
            r#"
            INSERT INTO users (
                id, username, password_hash, role, is_disabled, created_at, updated_at
            ) VALUES ('new-user', 'new-user', 'hash', 'user', 0, 1, 1)
            "#,
        )
        .execute(&pool)
        .await
        .expect("insert user after setup migration");
        pool
    }

    #[tokio::test]
    async fn new_users_receive_every_registered_choice() {
        let pool = test_pool().await;
        let status = get_setup_status(&pool, "new-user")
            .await
            .expect("load setup");
        assert_eq!(status.pending_count, CHOICES.len());
        assert_eq!(status.groups.len(), 3);
        assert!(has_pending_choices(&pool, "new-user").await.unwrap());
    }

    #[tokio::test]
    async fn recommended_choices_apply_atomically_and_clear_setup() {
        let pool = test_pool().await;
        let choices = CHOICES
            .iter()
            .map(|choice| SetupChoiceInput {
                key: choice.key.to_string(),
                value: choice.recommended_value,
            })
            .collect();
        apply_setup_choices(&pool, "new-user", choices)
            .await
            .expect("apply setup");

        assert!(!has_pending_choices(&pool, "new-user").await.unwrap());
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM user_setting_decisions WHERE user_id = 'new-user'",
            )
            .fetch_one(&pool)
            .await
            .unwrap(),
            CHOICES.len() as i64
        );
        let memories = crate::memories::service::get_memory_settings(&pool, "new-user")
            .await
            .unwrap();
        assert!(memories.allow_model_read);
        assert!(memories.allow_model_chat_history);
    }

    #[tokio::test]
    async fn ordinary_settings_saves_resolve_only_the_choices_the_client_saw() {
        let pool = test_pool().await;
        crate::notes::service::update_note_settings(
            &pool,
            "new-user",
            UpdateNoteSettingsRequest {
                allow_model_read: false,
                allow_model_create: false,
                allow_model_edit: false,
                allow_model_trash: false,
                default_ai_access: NoteAiAccess::None,
                default_model_scope: NoteModelScope {
                    all_models: true,
                    model_keys: Vec::new(),
                },
            },
        )
        .await
        .expect("save note settings");
        crate::memories::service::update_memory_settings(
            &pool,
            "new-user",
            UpdateMemorySettingsRequest {
                allow_model_read: false,
                allow_model_create: false,
                allow_model_edit: false,
                allow_model_forget: false,
                allow_model_chat_history: None,
            },
        )
        .await
        .expect("save old-client memory settings");

        let status = get_setup_status(&pool, "new-user")
            .await
            .expect("load remaining setup");
        assert_eq!(status.pending_count, 1);
        assert_eq!(status.groups[0].id, "past_chats");

        crate::memories::service::update_memory_settings(
            &pool,
            "new-user",
            UpdateMemorySettingsRequest {
                allow_model_read: false,
                allow_model_create: false,
                allow_model_edit: false,
                allow_model_forget: false,
                allow_model_chat_history: Some(false),
            },
        )
        .await
        .expect("save current memory settings");
        assert!(!has_pending_choices(&pool, "new-user").await.unwrap());
    }

    #[tokio::test]
    async fn setup_rejects_incoherent_permissions() {
        let pool = test_pool().await;
        let error = apply_setup_choices(
            &pool,
            "new-user",
            vec![
                SetupChoiceInput {
                    key: NOTES_ALLOW_MODEL_READ.to_string(),
                    value: false,
                },
                SetupChoiceInput {
                    key: NOTES_ALLOW_MODEL_EDIT.to_string(),
                    value: true,
                },
            ],
        )
        .await
        .expect_err("edit without read must fail");
        assert_eq!(error.code(), "invalid_notes_setup");
        assert!(has_pending_choices(&pool, "new-user").await.unwrap());
    }
}
