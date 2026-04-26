use serde::Serialize;
use sqlx::{Row, SqlitePool};

use crate::{admin::handlers::UpdateUserRequest, auth::service::unix_timestamp, error::ApiError};

#[derive(Debug, Serialize)]
pub struct AdminUserResponse {
    pub id: String,
    pub username: String,
    pub email: Option<String>,
    pub role: String,
    pub is_disabled: bool,
    pub created_at: i64,
    pub updated_at: i64,
    pub last_login_at: Option<i64>,
}

pub async fn list_users(pool: &SqlitePool) -> Result<Vec<AdminUserResponse>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT id, username, email, role, is_disabled, created_at, updated_at, last_login_at
        FROM users
        ORDER BY is_disabled DESC, created_at ASC
        "#,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter().map(row_to_admin_user).collect()
}

pub async fn update_user(
    pool: &SqlitePool,
    admin_user_id: &str,
    user_id: &str,
    payload: UpdateUserRequest,
) -> Result<AdminUserResponse, ApiError> {
    if payload.role.is_none() && payload.is_disabled.is_none() {
        return Err(ApiError::bad_request(
            "empty_update",
            "No user changes were provided",
        ));
    }

    let role = match payload.role {
        Some(role) => Some(normalize_role(role)?),
        None => None,
    };

    if admin_user_id == user_id {
        if payload.is_disabled == Some(true) {
            return Err(ApiError::bad_request(
                "cannot_disable_self",
                "Admins cannot disable their own account",
            ));
        }

        if role.as_deref() == Some("user") {
            return Err(ApiError::bad_request(
                "cannot_demote_self",
                "Admins cannot demote their own account",
            ));
        }
    }

    let now = unix_timestamp();
    let is_disabled = payload.is_disabled.map(i64::from);

    let row = sqlx::query(
        r#"
        UPDATE users
        SET role = COALESCE(?, role),
            is_disabled = COALESCE(?, is_disabled),
            updated_at = ?
        WHERE id = ?
        RETURNING id, username, email, role, is_disabled, created_at, updated_at, last_login_at
        "#,
    )
    .bind(role)
    .bind(is_disabled)
    .bind(now)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("user_not_found", "User not found"))?;

    row_to_admin_user(row).map_err(ApiError::from)
}

pub async fn delete_user(
    pool: &SqlitePool,
    admin_user_id: &str,
    user_id: &str,
) -> Result<(), ApiError> {
    if admin_user_id == user_id {
        return Err(ApiError::bad_request(
            "cannot_delete_self",
            "Admins cannot delete their own account",
        ));
    }

    let result = sqlx::query("DELETE FROM users WHERE id = ?")
        .bind(user_id)
        .execute(pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found("user_not_found", "User not found"));
    }

    Ok(())
}

fn normalize_role(role: String) -> Result<String, ApiError> {
    let role = role.trim().to_ascii_lowercase();
    match role.as_str() {
        "admin" | "user" => Ok(role),
        _ => Err(ApiError::bad_request(
            "invalid_role",
            "Role must be admin or user",
        )),
    }
}

fn row_to_admin_user(row: sqlx::sqlite::SqliteRow) -> Result<AdminUserResponse, sqlx::Error> {
    Ok(AdminUserResponse {
        id: row.try_get("id")?,
        username: row.try_get("username")?,
        email: row.try_get("email")?,
        role: row.try_get("role")?,
        is_disabled: row.try_get::<i64, _>("is_disabled")? != 0,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        last_login_at: row.try_get("last_login_at")?,
    })
}
