use serde::Serialize;
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::{
    admin::handlers::{CreateUserRequest, UpdateUserRequest},
    auth::service::{hash_password, unix_timestamp},
    error::ApiError,
};

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

pub async fn create_user(
    pool: &SqlitePool,
    payload: CreateUserRequest,
) -> Result<AdminUserResponse, ApiError> {
    let username = payload.username.trim().to_string();
    if username.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_username",
            "Username is required",
        ));
    }

    if payload.password.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_password",
            "Password is required",
        ));
    }

    let email = payload
        .email
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let role = match payload.role {
        Some(role) => normalize_role(role)?,
        None => "user".to_string(),
    };
    let is_disabled = i64::from(payload.is_disabled.unwrap_or(false));
    let password_hash = hash_password(&payload.password)?;
    let user_id = Uuid::new_v4().to_string();
    let now = unix_timestamp();

    let mut tx = pool.begin().await?;

    let insert = sqlx::query(
        r#"
        INSERT INTO users (
            id,
            username,
            email,
            password_hash,
            role,
            is_disabled,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id, username, email, role, is_disabled, created_at, updated_at, last_login_at
        "#,
    )
    .bind(&user_id)
    .bind(username)
    .bind(email)
    .bind(password_hash)
    .bind(role)
    .bind(is_disabled)
    .bind(now)
    .bind(now)
    .fetch_one(&mut *tx)
    .await;

    let row = match insert {
        Ok(row) => row,
        Err(error) => {
            if let sqlx::Error::Database(database_error) = &error {
                if database_error.is_unique_violation() {
                    return Err(ApiError::conflict(
                        "user_exists",
                        "Username or email is already in use",
                    ));
                }
            }
            return Err(error.into());
        }
    };

    sqlx::query(
        r#"
        INSERT INTO user_settings (user_id, created_at, updated_at)
        VALUES (?, ?, ?)
        "#,
    )
    .bind(&user_id)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    row_to_admin_user(row).map_err(ApiError::from)
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
