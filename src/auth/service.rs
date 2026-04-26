use std::time::{SystemTime, UNIX_EPOCH};

use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
};
use axum_extra::extract::cookie::{Cookie, SameSite};
use rand_core::OsRng;
use serde::Serialize;
use sqlx::{Row, SqlitePool};
use time::Duration;
use uuid::Uuid;

use crate::{config::Config, error::ApiError};

#[derive(Debug, Clone, Serialize)]
pub struct UserPublic {
    pub id: String,
    pub username: String,
    pub email: Option<String>,
    pub role: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisteredUser {
    pub id: String,
    pub username: String,
    pub email: Option<String>,
    pub role: String,
    pub is_disabled: bool,
}

#[derive(Debug)]
pub struct RegistrationResult {
    pub requires_approval: bool,
    pub user: RegisteredUser,
}

#[derive(Debug)]
pub struct SessionRecord {
    pub id: String,
}

pub fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs() as i64
}

pub async fn register_user(
    pool: &SqlitePool,
    username: String,
    email: Option<String>,
    password: String,
) -> Result<RegistrationResult, ApiError> {
    let username = username.trim().to_string();
    if username.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_username",
            "Username is required",
        ));
    }

    if password.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_password",
            "Password is required",
        ));
    }

    let email = email
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);

    let now = unix_timestamp();
    let password_hash = hash_password(&password)?;
    let mut user = RegisteredUser {
        id: Uuid::new_v4().to_string(),
        username,
        email,
        role: "user".to_string(),
        is_disabled: true,
    };

    let mut tx = pool.begin().await?;

    let admin_exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS (
            SELECT 1 FROM users WHERE role = 'admin' AND is_disabled = 0
        )
        "#,
    )
    .fetch_one(&mut *tx)
    .await?;
    let admin_exists = admin_exists != 0;
    let role = if admin_exists { "user" } else { "admin" };
    let is_disabled = i64::from(admin_exists);

    let signup_limit = if admin_exists {
        let row = sqlx::query(
            r#"
            SELECT allow_signup, signup_limit, signup_count
            FROM app_settings
            WHERE id = 1
            "#,
        )
        .fetch_one(&mut *tx)
        .await?;

        let allow_signup = row.try_get::<i64, _>("allow_signup")? != 0;
        let signup_limit: i64 = row.try_get("signup_limit")?;
        let signup_count: i64 = row.try_get("signup_count")?;

        if !allow_signup {
            return Err(ApiError::forbidden(
                "signup_disabled",
                "Account creation is disabled",
            ));
        }

        if signup_count >= signup_limit {
            sqlx::query(
                r#"
                UPDATE app_settings
                SET allow_signup = 0,
                    updated_at = ?
                WHERE id = 1
                "#,
            )
            .bind(now)
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;

            return Err(ApiError::forbidden(
                "signup_limit_reached",
                "Account creation limit has been reached",
            ));
        }

        Some(signup_limit)
    } else {
        None
    };

    let insert_user = sqlx::query(
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
        "#,
    )
    .bind(&user.id)
    .bind(&user.username)
    .bind(&user.email)
    .bind(password_hash)
    .bind(role)
    .bind(is_disabled)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await;

    match insert_user {
        Ok(_) => {}
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
    }

    user.role = role.to_string();
    user.is_disabled = is_disabled != 0;

    sqlx::query(
        r#"
        INSERT INTO user_settings (user_id, created_at, updated_at)
        VALUES (?, ?, ?)
        "#,
    )
    .bind(&user.id)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    if let Some(signup_limit) = signup_limit {
        sqlx::query(
            r#"
            UPDATE app_settings
            SET signup_count = signup_count + 1,
                allow_signup = CASE
                    WHEN signup_count + 1 >= ? THEN 0
                    ELSE allow_signup
                END,
                updated_at = ?
            WHERE id = 1
            "#,
        )
        .bind(signup_limit)
        .bind(now)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;

    Ok(RegistrationResult {
        requires_approval: user.is_disabled,
        user,
    })
}

pub async fn can_create_account(pool: &SqlitePool) -> Result<bool, sqlx::Error> {
    let admin_exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS (
            SELECT 1 FROM users WHERE role = 'admin' AND is_disabled = 0
        )
        "#,
    )
    .fetch_one(pool)
    .await?;

    if admin_exists == 0 {
        return Ok(true);
    }

    let row = sqlx::query(
        r#"
        SELECT allow_signup, signup_limit, signup_count
        FROM app_settings
        WHERE id = 1
        "#,
    )
    .fetch_one(pool)
    .await?;

    let allow_signup = row.try_get::<i64, _>("allow_signup")? != 0;
    let signup_limit: i64 = row.try_get("signup_limit")?;
    let signup_count: i64 = row.try_get("signup_count")?;

    Ok(allow_signup && signup_count < signup_limit)
}

pub async fn authenticate_user(
    pool: &SqlitePool,
    identifier: String,
    password: String,
) -> Result<UserPublic, ApiError> {
    let identifier = identifier.trim().to_string();
    if identifier.is_empty() || password.is_empty() {
        return Err(ApiError::invalid_credentials());
    }

    let Some(row) = sqlx::query(
        r#"
        SELECT id, username, email, role, password_hash, is_disabled
        FROM users
        WHERE username = ? OR email = ?
        LIMIT 1
        "#,
    )
    .bind(&identifier)
    .bind(&identifier)
    .fetch_optional(pool)
    .await?
    else {
        return Err(ApiError::invalid_credentials());
    };

    let password_hash: String = row.try_get("password_hash")?;
    if !verify_password(&password, &password_hash)? {
        return Err(ApiError::invalid_credentials());
    }

    if row.try_get::<i64, _>("is_disabled")? != 0 {
        return Err(ApiError::forbidden(
            "account_pending_approval",
            "Account is pending admin approval",
        ));
    }

    Ok(UserPublic {
        id: row.try_get("id")?,
        username: row.try_get("username")?,
        email: row.try_get("email")?,
        role: row.try_get("role")?,
    })
}

pub async fn create_session(
    pool: &SqlitePool,
    user_id: &str,
    ttl_seconds: i64,
    ip_address: Option<String>,
    user_agent: Option<String>,
) -> Result<SessionRecord, sqlx::Error> {
    let now = unix_timestamp();
    let session = SessionRecord {
        id: Uuid::new_v4().to_string(),
    };

    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        INSERT INTO sessions (id, user_id, created_at, expires_at, last_seen_at, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&session.id)
    .bind(user_id)
    .bind(now)
    .bind(now + ttl_seconds)
    .bind(now)
    .bind(ip_address)
    .bind(user_agent)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        UPDATE users
        SET last_login_at = ?, updated_at = ?
        WHERE id = ?
        "#,
    )
    .bind(now)
    .bind(now)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    Ok(session)
}

pub async fn current_user_from_cookie(
    pool: &SqlitePool,
    jar: &axum_extra::extract::CookieJar,
    cookie_name: &str,
) -> Result<Option<UserPublic>, sqlx::Error> {
    let Some(cookie) = jar.get(cookie_name) else {
        return Ok(None);
    };

    let now = unix_timestamp();
    let user = sqlx::query(
        r#"
        SELECT u.id, u.username, u.email, u.role
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.id = ?
          AND s.expires_at > ?
          AND u.is_disabled = 0
        "#,
    )
    .bind(cookie.value())
    .bind(now)
    .fetch_optional(pool)
    .await?
    .map(|row| UserPublic {
        id: row.try_get("id").expect("id selected"),
        username: row.try_get("username").expect("username selected"),
        email: row.try_get("email").expect("email selected"),
        role: row.try_get("role").expect("role selected"),
    });

    if user.is_some() {
        sqlx::query("UPDATE sessions SET last_seen_at = ? WHERE id = ?")
            .bind(now)
            .bind(cookie.value())
            .execute(pool)
            .await?;
    }

    Ok(user)
}

pub async fn require_user(
    pool: &SqlitePool,
    jar: &axum_extra::extract::CookieJar,
    cookie_name: &str,
) -> Result<UserPublic, ApiError> {
    current_user_from_cookie(pool, jar, cookie_name)
        .await?
        .ok_or_else(ApiError::unauthorized)
}

pub async fn require_admin(
    pool: &SqlitePool,
    jar: &axum_extra::extract::CookieJar,
    cookie_name: &str,
) -> Result<UserPublic, ApiError> {
    let user = require_user(pool, jar, cookie_name).await?;
    if user.role != "admin" {
        return Err(ApiError::forbidden(
            "admin_required",
            "Admin access is required",
        ));
    }

    Ok(user)
}

pub async fn delete_session(pool: &SqlitePool, session_id: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM sessions WHERE id = ?")
        .bind(session_id)
        .execute(pool)
        .await?;

    Ok(())
}

pub fn session_cookie(config: &Config, session_id: &str) -> Cookie<'static> {
    Cookie::build((config.session_cookie_name.clone(), session_id.to_string()))
        .http_only(true)
        .same_site(SameSite::Lax)
        .secure(config.cookie_secure)
        .path("/")
        .max_age(Duration::seconds(config.session_ttl_seconds))
        .build()
}

pub fn expired_session_cookie(config: &Config) -> Cookie<'static> {
    Cookie::build((config.session_cookie_name.clone(), ""))
        .http_only(true)
        .same_site(SameSite::Lax)
        .secure(config.cookie_secure)
        .path("/")
        .max_age(Duration::seconds(0))
        .build()
}

fn hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    Ok(argon2
        .hash_password(password.as_bytes(), &salt)?
        .to_string())
}

fn verify_password(
    password: &str,
    password_hash: &str,
) -> Result<bool, argon2::password_hash::Error> {
    let parsed_hash = PasswordHash::new(password_hash)?;
    Ok(Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .is_ok())
}
