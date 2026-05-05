use sqlx::{Row, SqlitePool, Transaction};
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp,
    error::ApiError,
    personas::{
        handlers::{CopyPersonaRequest, CreatePersonaRequest, UpdatePersonaRequest},
        models::{PersonaResponse, PersonaVersionResponse},
    },
};

const MAX_DISPLAY_NAME_LEN: usize = 80;
const MAX_SYSTEM_PROMPT_LEN: usize = 60_000;
const MAX_TOOL_POLICY_LEN: usize = 20_000;

struct PersonaVisibility {
    is_member: bool,
}

#[derive(Clone, Debug)]
pub struct ResolvedPersonaVersion {
    pub persona_id: String,
    pub persona_version_id: String,
    pub display_name: String,
    pub base_backend_id: String,
    pub base_model_name: String,
    pub system_prompt: String,
}

pub async fn list_personas(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<Vec<PersonaResponse>, ApiError> {
    let rows = sqlx::query(
        r#"
        SELECT p.id,
               p.owner_user_id,
               u.username AS owner_username,
               p.visibility,
               p.lifecycle_state,
               p.created_at,
               p.updated_at,
               v.id AS version_id,
               v.persona_id AS version_persona_id,
               v.version_number,
               v.display_name,
               v.avatar_attachment_id,
               v.base_backend_id,
               v.base_model_name,
               v.system_prompt,
               v.tool_policy_json,
               v.created_by_user_id,
               v.created_at AS version_created_at,
               CASE WHEN p.owner_user_id = ? THEN 1 ELSE 0 END AS is_owner,
               CASE WHEN pm.user_id IS NULL THEN 0 ELSE 1 END AS is_member
        FROM personas p
        JOIN persona_versions v ON v.id = p.current_version_id
        LEFT JOIN users u ON u.id = p.owner_user_id
        LEFT JOIN persona_members pm ON pm.persona_id = p.id AND pm.user_id = ?
        WHERE p.lifecycle_state != 'deleted'
          AND (
            p.owner_user_id = ?
            OR (p.visibility = 'public' AND p.lifecycle_state = 'active')
            OR pm.user_id IS NOT NULL
          )
        ORDER BY v.display_name COLLATE NOCASE ASC, p.created_at ASC
        "#,
    )
    .bind(user_id)
    .bind(user_id)
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(row_to_persona)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ApiError::from)
}

pub async fn create_persona(
    pool: &SqlitePool,
    user_id: &str,
    payload: CreatePersonaRequest,
) -> Result<PersonaResponse, ApiError> {
    let visibility = validate_visibility(&payload.visibility)?;
    let display_name = validate_display_name(&payload.display_name)?;
    let base_backend_id = validate_base_backend(pool, &payload.base_backend_id).await?;
    let base_model_name = validate_base_model_name(&payload.base_model_name)?;
    let system_prompt = validate_system_prompt(&payload.system_prompt)?;
    let tool_policy_json = validate_tool_policy(payload.tool_policy_json)?;
    let avatar_attachment_id = normalize_optional(payload.avatar_attachment_id);
    let now = unix_timestamp();
    let persona_id = Uuid::new_v4().to_string();
    let version_id = Uuid::new_v4().to_string();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        INSERT INTO personas (
            id,
            owner_user_id,
            visibility,
            lifecycle_state,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, 'active', ?, ?)
        "#,
    )
    .bind(&persona_id)
    .bind(user_id)
    .bind(&visibility)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    insert_persona_version(
        &mut tx,
        InsertPersonaVersion {
            version_id: &version_id,
            persona_id: &persona_id,
            version_number: 1,
            display_name: &display_name,
            avatar_attachment_id: avatar_attachment_id.as_deref(),
            base_backend_id: &base_backend_id,
            base_model_name: &base_model_name,
            system_prompt: &system_prompt,
            tool_policy_json: tool_policy_json.as_deref(),
            created_by_user_id: Some(user_id),
            now,
        },
    )
    .await?;
    set_current_version(&mut tx, &persona_id, &version_id, now).await?;
    if visibility == "public" {
        upsert_member(&mut tx, &persona_id, user_id, "creator", now).await?;
    }
    tx.commit().await?;

    get_visible_persona(pool, user_id, &persona_id).await
}

pub async fn update_persona(
    pool: &SqlitePool,
    user_id: &str,
    persona_id: &str,
    payload: UpdatePersonaRequest,
) -> Result<PersonaResponse, ApiError> {
    let current = get_persona_for_update(pool, user_id, persona_id).await?;
    if current.lifecycle_state == "deleted" {
        return Err(ApiError::not_found(
            "persona_not_found",
            "Persona not found",
        ));
    }
    if current.owner_user_id.as_deref() != Some(user_id) {
        return Err(ApiError::forbidden(
            "persona_not_owned",
            "Only the persona owner can edit it",
        ));
    }

    let current_version = get_version(pool, &current.current_version_id).await?;
    let next_visibility = match payload.visibility {
        Some(visibility) => validate_visibility(&visibility)?,
        None => current.visibility.clone(),
    };
    let display_name = match payload.display_name {
        Some(display_name) => validate_display_name(&display_name)?,
        None => current_version.display_name.clone(),
    };
    let avatar_attachment_id = payload
        .avatar_attachment_id
        .map(Some)
        .unwrap_or(current_version.avatar_attachment_id.clone())
        .and_then(|value| normalize_optional(Some(value)));
    let base_backend_id = match payload.base_backend_id {
        Some(backend_id) => validate_base_backend(pool, &backend_id).await?,
        None => current_version.base_backend_id.clone(),
    };
    let base_model_name = match payload.base_model_name {
        Some(model_name) => validate_base_model_name(&model_name)?,
        None => current_version.base_model_name.clone(),
    };
    let system_prompt = match payload.system_prompt {
        Some(system_prompt) => validate_system_prompt(&system_prompt)?,
        None => current_version.system_prompt.clone(),
    };
    let tool_policy_json = match payload.tool_policy_json {
        Some(tool_policy) => validate_tool_policy(Some(tool_policy))?,
        None => current_version.tool_policy_json.clone(),
    };

    if current.visibility == "public" && next_visibility == "private" {
        let other_members = count_other_members(pool, persona_id, user_id).await?;
        if other_members > 0 {
            return Err(ApiError::conflict(
                "persona_has_members",
                "Public personas used by other users cannot be made private; disown it instead",
            ));
        }
    }

    let has_version_change = display_name != current_version.display_name
        || avatar_attachment_id != current_version.avatar_attachment_id
        || base_backend_id != current_version.base_backend_id
        || base_model_name != current_version.base_model_name
        || system_prompt != current_version.system_prompt
        || tool_policy_json != current_version.tool_policy_json;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    if has_version_change {
        let version_id = Uuid::new_v4().to_string();
        let version_number = next_version_number(&mut tx, persona_id).await?;
        insert_persona_version(
            &mut tx,
            InsertPersonaVersion {
                version_id: &version_id,
                persona_id,
                version_number,
                display_name: &display_name,
                avatar_attachment_id: avatar_attachment_id.as_deref(),
                base_backend_id: &base_backend_id,
                base_model_name: &base_model_name,
                system_prompt: &system_prompt,
                tool_policy_json: tool_policy_json.as_deref(),
                created_by_user_id: Some(user_id),
                now,
            },
        )
        .await?;
        set_current_version(&mut tx, persona_id, &version_id, now).await?;
    }

    sqlx::query(
        r#"
        UPDATE personas
        SET visibility = ?,
            lifecycle_state = 'active',
            updated_at = ?
        WHERE id = ?
        "#,
    )
    .bind(&next_visibility)
    .bind(now)
    .bind(persona_id)
    .execute(&mut *tx)
    .await?;

    if next_visibility == "public" {
        upsert_member(&mut tx, persona_id, user_id, "creator", now).await?;
    } else {
        sqlx::query("DELETE FROM persona_members WHERE persona_id = ? AND user_id = ?")
            .bind(persona_id)
            .bind(user_id)
            .execute(&mut *tx)
            .await?;
    }

    tx.commit().await?;
    get_visible_persona(pool, user_id, persona_id).await
}

pub async fn copy_persona(
    pool: &SqlitePool,
    user_id: &str,
    persona_id: &str,
    payload: CopyPersonaRequest,
) -> Result<PersonaResponse, ApiError> {
    ensure_persona_visible(pool, user_id, persona_id).await?;
    let visibility = validate_visibility(payload.visibility.as_deref().unwrap_or("private"))?;
    let source = get_version(pool, &payload.persona_version_id).await?;
    if source.persona_id != persona_id {
        return Err(ApiError::bad_request(
            "invalid_persona_version",
            "Persona version does not belong to this persona",
        ));
    }

    create_persona(
        pool,
        user_id,
        CreatePersonaRequest {
            visibility,
            display_name: source.display_name,
            avatar_attachment_id: source.avatar_attachment_id,
            base_backend_id: source.base_backend_id,
            base_model_name: source.base_model_name,
            system_prompt: source.system_prompt,
            tool_policy_json: source.tool_policy_json,
        },
    )
    .await
}

pub async fn disown_persona(
    pool: &SqlitePool,
    user_id: &str,
    persona_id: &str,
) -> Result<(), ApiError> {
    let current = get_persona_for_update(pool, user_id, persona_id).await?;
    let visibility = ensure_persona_visible(pool, user_id, persona_id).await?;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query("DELETE FROM persona_members WHERE persona_id = ? AND user_id = ?")
        .bind(persona_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;

    let remaining_members: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM persona_members WHERE persona_id = ?")
            .bind(persona_id)
            .fetch_one(&mut *tx)
            .await?;
    let is_owner = current.owner_user_id.as_deref() == Some(user_id);
    let next_state = if remaining_members == 0 {
        "deleted"
    } else if is_owner {
        "disowned"
    } else {
        current.lifecycle_state.as_str()
    };

    if is_owner || remaining_members == 0 || visibility.is_member {
        sqlx::query(
            r#"
            UPDATE personas
            SET owner_user_id = CASE WHEN owner_user_id = ? THEN NULL ELSE owner_user_id END,
                lifecycle_state = ?,
                updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(user_id)
        .bind(next_state)
        .bind(now)
        .bind(persona_id)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(())
}

pub async fn list_versions(
    pool: &SqlitePool,
    user_id: &str,
    persona_id: &str,
) -> Result<Vec<PersonaVersionResponse>, ApiError> {
    ensure_persona_visible(pool, user_id, persona_id).await?;
    let rows = sqlx::query(
        r#"
        SELECT id,
               persona_id,
               version_number,
               display_name,
               avatar_attachment_id,
               base_backend_id,
               base_model_name,
               system_prompt,
               tool_policy_json,
               created_by_user_id,
               created_at
        FROM persona_versions
        WHERE persona_id = ?
        ORDER BY version_number DESC
        "#,
    )
    .bind(persona_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(row_to_version)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ApiError::from)
}

pub async fn resolve_persona_version_for_use(
    pool: &SqlitePool,
    user_id: &str,
    persona_version_id: &str,
) -> Result<ResolvedPersonaVersion, ApiError> {
    let persona_version_id = persona_version_id.trim();
    if persona_version_id.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_persona",
            "Persona version is required",
        ));
    }

    let row = sqlx::query(
        r#"
        SELECT p.id AS persona_id,
               p.owner_user_id,
               p.visibility,
               p.lifecycle_state,
               pm.user_id AS member_user_id,
               v.id AS persona_version_id,
               v.display_name,
               v.base_backend_id,
               v.base_model_name,
               v.system_prompt
        FROM persona_versions v
        JOIN personas p ON p.id = v.persona_id
        LEFT JOIN persona_members pm ON pm.persona_id = p.id AND pm.user_id = ?
        WHERE v.id = ?
          AND p.lifecycle_state != 'deleted'
        "#,
    )
    .bind(user_id)
    .bind(persona_version_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("persona_not_found", "Persona not found"))?;

    let owner_user_id: Option<String> = row.try_get("owner_user_id")?;
    let visibility: String = row.try_get("visibility")?;
    let lifecycle_state: String = row.try_get("lifecycle_state")?;
    let persona_id: String = row.try_get("persona_id")?;
    let is_owner = owner_user_id.as_deref() == Some(user_id);
    let is_member = row
        .try_get::<Option<String>, _>("member_user_id")?
        .is_some();
    let is_globally_visible = visibility == "public" && lifecycle_state == "active";

    if !is_owner && !is_member && !is_globally_visible {
        return Err(ApiError::not_found(
            "persona_not_found",
            "Persona not found",
        ));
    }

    if visibility == "public" && !is_member {
        sqlx::query(
            r#"
            INSERT INTO persona_members (persona_id, user_id, membership_role, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(persona_id, user_id) DO NOTHING
            "#,
        )
        .bind(&persona_id)
        .bind(user_id)
        .bind(if is_owner { "creator" } else { "member" })
        .bind(unix_timestamp())
        .execute(pool)
        .await?;
    }

    Ok(ResolvedPersonaVersion {
        persona_id,
        persona_version_id: row.try_get("persona_version_id")?,
        display_name: row.try_get("display_name")?,
        base_backend_id: row.try_get("base_backend_id")?,
        base_model_name: row.try_get("base_model_name")?,
        system_prompt: row.try_get("system_prompt")?,
    })
}

struct PersonaRecord {
    owner_user_id: Option<String>,
    current_version_id: String,
    visibility: String,
    lifecycle_state: String,
}

struct InsertPersonaVersion<'a> {
    version_id: &'a str,
    persona_id: &'a str,
    version_number: i64,
    display_name: &'a str,
    avatar_attachment_id: Option<&'a str>,
    base_backend_id: &'a str,
    base_model_name: &'a str,
    system_prompt: &'a str,
    tool_policy_json: Option<&'a str>,
    created_by_user_id: Option<&'a str>,
    now: i64,
}

async fn insert_persona_version(
    tx: &mut Transaction<'_, sqlx::Sqlite>,
    params: InsertPersonaVersion<'_>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO persona_versions (
            id,
            persona_id,
            version_number,
            display_name,
            avatar_attachment_id,
            base_backend_id,
            base_model_name,
            system_prompt,
            tool_policy_json,
            created_by_user_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(params.version_id)
    .bind(params.persona_id)
    .bind(params.version_number)
    .bind(params.display_name)
    .bind(params.avatar_attachment_id)
    .bind(params.base_backend_id)
    .bind(params.base_model_name)
    .bind(params.system_prompt)
    .bind(params.tool_policy_json)
    .bind(params.created_by_user_id)
    .bind(params.now)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn set_current_version(
    tx: &mut Transaction<'_, sqlx::Sqlite>,
    persona_id: &str,
    version_id: &str,
    now: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        UPDATE personas
        SET current_version_id = ?,
            updated_at = ?
        WHERE id = ?
        "#,
    )
    .bind(version_id)
    .bind(now)
    .bind(persona_id)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn upsert_member(
    tx: &mut Transaction<'_, sqlx::Sqlite>,
    persona_id: &str,
    user_id: &str,
    membership_role: &str,
    now: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO persona_members (persona_id, user_id, membership_role, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(persona_id, user_id) DO UPDATE SET membership_role = excluded.membership_role
        "#,
    )
    .bind(persona_id)
    .bind(user_id)
    .bind(membership_role)
    .bind(now)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn next_version_number(
    tx: &mut Transaction<'_, sqlx::Sqlite>,
    persona_id: &str,
) -> Result<i64, sqlx::Error> {
    let version_number: i64 = sqlx::query_scalar(
        r#"
        SELECT COALESCE(MAX(version_number), 0) + 1
        FROM persona_versions
        WHERE persona_id = ?
        "#,
    )
    .bind(persona_id)
    .fetch_one(&mut **tx)
    .await?;

    Ok(version_number)
}

async fn get_visible_persona(
    pool: &SqlitePool,
    user_id: &str,
    persona_id: &str,
) -> Result<PersonaResponse, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT p.id,
               p.owner_user_id,
               u.username AS owner_username,
               p.visibility,
               p.lifecycle_state,
               p.created_at,
               p.updated_at,
               v.id AS version_id,
               v.persona_id AS version_persona_id,
               v.version_number,
               v.display_name,
               v.avatar_attachment_id,
               v.base_backend_id,
               v.base_model_name,
               v.system_prompt,
               v.tool_policy_json,
               v.created_by_user_id,
               v.created_at AS version_created_at,
               CASE WHEN p.owner_user_id = ? THEN 1 ELSE 0 END AS is_owner,
               CASE WHEN pm.user_id IS NULL THEN 0 ELSE 1 END AS is_member
        FROM personas p
        JOIN persona_versions v ON v.id = p.current_version_id
        LEFT JOIN users u ON u.id = p.owner_user_id
        LEFT JOIN persona_members pm ON pm.persona_id = p.id AND pm.user_id = ?
        WHERE p.id = ?
          AND p.lifecycle_state != 'deleted'
          AND (
            p.owner_user_id = ?
            OR (p.visibility = 'public' AND p.lifecycle_state = 'active')
            OR pm.user_id IS NOT NULL
          )
        "#,
    )
    .bind(user_id)
    .bind(user_id)
    .bind(persona_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("persona_not_found", "Persona not found"))?;

    row_to_persona(row).map_err(ApiError::from)
}

async fn ensure_persona_visible(
    pool: &SqlitePool,
    user_id: &str,
    persona_id: &str,
) -> Result<PersonaVisibility, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT p.owner_user_id,
               p.visibility,
               p.lifecycle_state,
               pm.user_id AS member_user_id
        FROM personas p
        LEFT JOIN persona_members pm ON pm.persona_id = p.id AND pm.user_id = ?
        WHERE p.id = ?
          AND p.lifecycle_state != 'deleted'
        "#,
    )
    .bind(user_id)
    .bind(persona_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("persona_not_found", "Persona not found"))?;

    let owner_user_id: Option<String> = row.try_get("owner_user_id")?;
    let visibility: String = row.try_get("visibility")?;
    let lifecycle_state: String = row.try_get("lifecycle_state")?;
    let is_owner = owner_user_id.as_deref() == Some(user_id);
    let is_member = row
        .try_get::<Option<String>, _>("member_user_id")?
        .is_some();
    let is_globally_visible = visibility == "public" && lifecycle_state == "active";
    if lifecycle_state == "deleted" || (!is_owner && !is_globally_visible && !is_member) {
        return Err(ApiError::not_found(
            "persona_not_found",
            "Persona not found",
        ));
    }

    Ok(PersonaVisibility { is_member })
}

async fn get_persona_for_update(
    pool: &SqlitePool,
    user_id: &str,
    persona_id: &str,
) -> Result<PersonaRecord, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT owner_user_id, current_version_id, visibility, lifecycle_state
        FROM personas
        WHERE id = ?
          AND (
            owner_user_id = ?
            OR visibility = 'public'
            OR EXISTS (
                SELECT 1
                FROM persona_members
                WHERE persona_id = personas.id
                  AND user_id = ?
            )
          )
        "#,
    )
    .bind(persona_id)
    .bind(user_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("persona_not_found", "Persona not found"))?;

    Ok(PersonaRecord {
        owner_user_id: row.try_get("owner_user_id")?,
        current_version_id: row.try_get("current_version_id")?,
        visibility: row.try_get("visibility")?,
        lifecycle_state: row.try_get("lifecycle_state")?,
    })
}

async fn get_version(
    pool: &SqlitePool,
    version_id: &str,
) -> Result<PersonaVersionResponse, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT id,
               persona_id,
               version_number,
               display_name,
               avatar_attachment_id,
               base_backend_id,
               base_model_name,
               system_prompt,
               tool_policy_json,
               created_by_user_id,
               created_at
        FROM persona_versions
        WHERE id = ?
        "#,
    )
    .bind(version_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("persona_version_not_found", "Persona version not found"))?;

    row_to_version(row).map_err(ApiError::from)
}

async fn count_other_members(
    pool: &SqlitePool,
    persona_id: &str,
    user_id: &str,
) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM persona_members
        WHERE persona_id = ?
          AND user_id != ?
        "#,
    )
    .bind(persona_id)
    .bind(user_id)
    .fetch_one(pool)
    .await
}

async fn validate_base_backend(pool: &SqlitePool, backend_id: &str) -> Result<String, ApiError> {
    let backend_id = backend_id.trim();
    if backend_id.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_backend",
            "A base backend is required",
        ));
    }

    let exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM ollama_backends
            WHERE id = ?
              AND is_enabled = 1
        )
        "#,
    )
    .bind(backend_id)
    .fetch_one(pool)
    .await?;

    if exists == 0 {
        return Err(ApiError::bad_request(
            "invalid_backend",
            "Base backend not found or disabled",
        ));
    }

    Ok(backend_id.to_string())
}

fn validate_visibility(visibility: &str) -> Result<String, ApiError> {
    let visibility = visibility.trim().to_ascii_lowercase();
    if !matches!(visibility.as_str(), "private" | "public") {
        return Err(ApiError::bad_request(
            "invalid_visibility",
            "Persona visibility must be private or public",
        ));
    }

    Ok(visibility)
}

fn validate_display_name(display_name: &str) -> Result<String, ApiError> {
    let display_name = display_name.trim();
    if display_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_display_name",
            "Persona name is required",
        ));
    }
    if display_name.len() > MAX_DISPLAY_NAME_LEN {
        return Err(ApiError::bad_request(
            "invalid_display_name",
            "Persona name is too long",
        ));
    }

    Ok(display_name.to_string())
}

fn validate_base_model_name(model_name: &str) -> Result<String, ApiError> {
    let model_name = model_name.trim();
    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "A base model is required",
        ));
    }

    Ok(model_name.to_string())
}

fn validate_system_prompt(system_prompt: &str) -> Result<String, ApiError> {
    let system_prompt = system_prompt.trim();
    if system_prompt.len() > MAX_SYSTEM_PROMPT_LEN {
        return Err(ApiError::bad_request(
            "invalid_system_prompt",
            "System prompt is too long",
        ));
    }

    Ok(system_prompt.to_string())
}

fn validate_tool_policy(tool_policy_json: Option<String>) -> Result<Option<String>, ApiError> {
    let Some(tool_policy_json) = tool_policy_json else {
        return Ok(None);
    };
    let tool_policy_json = tool_policy_json.trim();
    if tool_policy_json.is_empty() {
        return Ok(None);
    }
    if tool_policy_json.len() > MAX_TOOL_POLICY_LEN {
        return Err(ApiError::bad_request(
            "invalid_tool_policy",
            "Tool policy metadata is too long",
        ));
    }

    Ok(Some(tool_policy_json.to_string()))
}

fn normalize_optional(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn row_to_persona(row: sqlx::sqlite::SqliteRow) -> Result<PersonaResponse, sqlx::Error> {
    Ok(PersonaResponse {
        id: row.try_get("id")?,
        owner_user_id: row.try_get("owner_user_id")?,
        owner_username: row.try_get("owner_username")?,
        visibility: row.try_get("visibility")?,
        lifecycle_state: row.try_get("lifecycle_state")?,
        current_version: PersonaVersionResponse {
            id: row.try_get("version_id")?,
            persona_id: row.try_get("version_persona_id")?,
            version_number: row.try_get("version_number")?,
            display_name: row.try_get("display_name")?,
            avatar_attachment_id: row.try_get("avatar_attachment_id")?,
            base_backend_id: row.try_get("base_backend_id")?,
            base_model_name: row.try_get("base_model_name")?,
            system_prompt: row.try_get("system_prompt")?,
            tool_policy_json: row.try_get("tool_policy_json")?,
            created_by_user_id: row.try_get("created_by_user_id")?,
            created_at: row.try_get("version_created_at")?,
        },
        is_owner: row.try_get::<i64, _>("is_owner")? == 1,
        is_member: row.try_get::<i64, _>("is_member")? == 1,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn row_to_version(row: sqlx::sqlite::SqliteRow) -> Result<PersonaVersionResponse, sqlx::Error> {
    Ok(PersonaVersionResponse {
        id: row.try_get("id")?,
        persona_id: row.try_get("persona_id")?,
        version_number: row.try_get("version_number")?,
        display_name: row.try_get("display_name")?,
        avatar_attachment_id: row.try_get("avatar_attachment_id")?,
        base_backend_id: row.try_get("base_backend_id")?,
        base_model_name: row.try_get("base_model_name")?,
        system_prompt: row.try_get("system_prompt")?,
        tool_policy_json: row.try_get("tool_policy_json")?,
        created_by_user_id: row.try_get("created_by_user_id")?,
        created_at: row.try_get("created_at")?,
    })
}

#[cfg(test)]
mod tests {
    use sqlx::{
        SqlitePool,
        sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    };

    use super::*;
    use crate::{auth::service::register_user, backends::service as backends_service, startup};

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

        pool
    }

    async fn create_test_backend(pool: &SqlitePool) -> String {
        backends_service::create_backend(
            pool,
            "Local Ollama".to_string(),
            "http://127.0.0.1:11434".to_string(),
        )
        .await
        .expect("create backend")
        .id
    }

    #[tokio::test]
    async fn persona_edits_create_immutable_versions() {
        let pool = test_pool().await;
        let user = register_user(&pool, "admin".to_string(), None, "secret".to_string())
            .await
            .expect("register user")
            .user;
        let backend_id = create_test_backend(&pool).await;

        let persona = create_persona(
            &pool,
            &user.id,
            CreatePersonaRequest {
                visibility: "private".to_string(),
                display_name: "Careful Researcher".to_string(),
                avatar_attachment_id: None,
                base_backend_id: backend_id.clone(),
                base_model_name: "gemma4:e2b".to_string(),
                system_prompt: "Answer carefully.".to_string(),
                tool_policy_json: None,
            },
        )
        .await
        .expect("create persona");

        assert_eq!(persona.current_version.version_number, 1);
        assert_eq!(persona.current_version.system_prompt, "Answer carefully.");

        let updated = update_persona(
            &pool,
            &user.id,
            &persona.id,
            UpdatePersonaRequest {
                visibility: Some("public".to_string()),
                display_name: Some("Careful Researcher".to_string()),
                avatar_attachment_id: None,
                base_backend_id: Some(backend_id),
                base_model_name: Some("gemma4:e2b".to_string()),
                system_prompt: Some("Answer carefully and cite uncertainty.".to_string()),
                tool_policy_json: None,
            },
        )
        .await
        .expect("update persona");

        assert_eq!(updated.visibility, "public");
        assert_eq!(updated.current_version.version_number, 2);
        assert_ne!(updated.current_version.id, persona.current_version.id);

        let versions = list_versions(&pool, &user.id, &persona.id)
            .await
            .expect("list versions");
        assert_eq!(versions.len(), 2);
        assert!(
            versions.iter().any(|version| version.version_number == 1
                && version.system_prompt == "Answer carefully.")
        );
        assert!(versions.iter().any(|version| version.version_number == 2
            && version.system_prompt == "Answer carefully and cite uncertainty."));
    }

    #[tokio::test]
    async fn visible_public_personas_can_be_copied_to_private_personas() {
        let pool = test_pool().await;
        let owner = register_user(&pool, "owner".to_string(), None, "secret".to_string())
            .await
            .expect("register owner")
            .user;
        let other = register_user(&pool, "other".to_string(), None, "secret".to_string())
            .await
            .expect("register other")
            .user;
        let backend_id = create_test_backend(&pool).await;
        let persona = create_persona(
            &pool,
            &owner.id,
            CreatePersonaRequest {
                visibility: "public".to_string(),
                display_name: "Shared Helper".to_string(),
                avatar_attachment_id: None,
                base_backend_id: backend_id,
                base_model_name: "gemma4:e2b".to_string(),
                system_prompt: "Be helpful.".to_string(),
                tool_policy_json: None,
            },
        )
        .await
        .expect("create public persona");

        let copied = copy_persona(
            &pool,
            &other.id,
            &persona.id,
            CopyPersonaRequest {
                persona_version_id: persona.current_version.id,
                visibility: Some("private".to_string()),
            },
        )
        .await
        .expect("copy persona");

        assert_ne!(copied.id, persona.id);
        assert_eq!(copied.owner_user_id.as_deref(), Some(other.id.as_str()));
        assert_eq!(copied.visibility, "private");
        assert_eq!(copied.current_version.version_number, 1);
        assert_eq!(copied.current_version.display_name, "Shared Helper");
        assert_eq!(copied.current_version.system_prompt, "Be helpful.");
    }
}
