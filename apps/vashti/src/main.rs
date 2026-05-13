mod admin;
mod app_state;
mod auth;
mod backends;
mod chats;
mod config;
mod db;
mod error;
mod frontend;
mod model_cache;
mod ollama;
mod permissions;
mod personas;
mod private;
mod rate_limit;
mod security;
mod settings;
mod startup;
mod tools;
mod uploads;
mod version;

use std::{error::Error, time::Duration};

use app_state::AppState;
use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{HeaderName, HeaderValue, header},
    routing::{get, patch, post},
};
use config::Config;
use error::ApiError;
use tokio::signal;
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    if std::env::args().any(|arg| arg == "--version" || arg == "-V") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vashti=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::from_env()?;
    startup::prepare_data_dir(&config).await?;

    let db = db::connect(&config).await?;
    startup::migrations::run(&db).await?;
    startup::bootstrap::ensure_app_settings(&db).await?;
    startup::network_recovery::recover_network_if_requested(&db, &config).await?;
    auth::service::delete_expired_sessions(&db).await?;

    let http_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;

    startup::bootstrap::detect_localhost_ollama_if_empty(&db, &http_client).await?;

    let bind_addr = config.bind_addr;
    let state = AppState::new(config, db, http_client);
    spawn_session_cleanup(state.db.clone());
    spawn_model_cache_refresh(state.clone());
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    tracing::info!("listening on http://{}", listener.local_addr()?);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/version", get(version::get_version))
        .route("/auth/session", get(auth::handlers::session))
        .route("/auth/register", post(auth::handlers::register))
        .route("/auth/login", post(auth::handlers::login))
        .route("/auth/logout", post(auth::handlers::logout))
        .route(
            "/settings",
            get(settings::handlers::get_app_settings)
                .patch(settings::handlers::update_app_settings),
        )
        .route(
            "/settings/network",
            patch(settings::handlers::update_network_settings),
        )
        .route(
            "/settings/tools",
            get(settings::handlers::get_tool_settings)
                .patch(settings::handlers::update_tool_settings),
        )
        .route("/tools", get(settings::handlers::get_available_tools))
        .route(
            "/settings/network-recovery-notice/dismiss",
            post(settings::handlers::dismiss_network_recovery_notice),
        )
        .route(
            "/user-settings",
            get(settings::handlers::get_user_settings)
                .patch(settings::handlers::update_user_settings),
        )
        .route(
            "/admin/users",
            get(admin::handlers::list_users).post(admin::handlers::create_user),
        )
        .route(
            "/admin/users/{user_id}",
            patch(admin::handlers::update_user).delete(admin::handlers::delete_user),
        )
        .route(
            "/backends",
            get(backends::handlers::list_backends).post(backends::handlers::create_backend),
        )
        .route(
            "/backends/detect-localhost",
            post(backends::handlers::detect_localhost),
        )
        .route(
            "/backends/scan-local-network",
            post(backends::handlers::scan_local_network),
        )
        .route(
            "/backends/{backend_id}",
            patch(backends::handlers::update_backend).delete(backends::handlers::delete_backend),
        )
        .route("/models", get(backends::handlers::list_models))
        .route(
            "/user-models",
            get(backends::handlers::list_user_models)
                .patch(backends::handlers::update_user_model_visibility),
        )
        .route(
            "/user-models/refresh",
            post(backends::handlers::refresh_user_models),
        )
        .route(
            "/admin/models",
            get(backends::handlers::list_admin_models)
                .patch(backends::handlers::update_model_availability),
        )
        .route(
            "/admin/models/refresh",
            post(backends::handlers::refresh_admin_models),
        )
        .route(
            "/admin/models/tags",
            patch(backends::handlers::update_model_tags),
        )
        .route(
            "/admin/models/default-tags",
            patch(backends::handlers::update_default_model_tags),
        )
        .route(
            "/admin/models/backend",
            patch(backends::handlers::update_backend_model_availability),
        )
        .route(
            "/personas",
            get(personas::handlers::list_personas).post(personas::handlers::create_persona),
        )
        .route(
            "/personas/{persona_id}",
            patch(personas::handlers::update_persona),
        )
        .route(
            "/personas/{persona_id}/copy",
            post(personas::handlers::copy_persona),
        )
        .route(
            "/personas/{persona_id}/disown",
            post(personas::handlers::disown_persona),
        )
        .route(
            "/personas/{persona_id}/versions",
            get(personas::handlers::list_versions),
        )
        .route(
            "/chats",
            get(chats::handlers::list_chats).post(chats::handlers::create_chat),
        )
        .route(
            "/chats/{chat_id}",
            get(chats::handlers::get_chat)
                .patch(chats::handlers::update_chat)
                .delete(chats::handlers::delete_chat),
        )
        .route("/chats/{chat_id}/sync", get(chats::handlers::sync_chat))
        .route(
            "/chats/{chat_id}/active-root",
            patch(chats::handlers::set_active_root),
        )
        .route(
            "/chats/{chat_id}/messages",
            get(chats::handlers::list_messages).post(chats::handlers::create_message),
        )
        .route(
            "/chats/{chat_id}/messages/{message_id}/active-child",
            patch(chats::handlers::set_active_child),
        )
        .route(
            "/chats/{chat_id}/messages/{message_id}/active-revision",
            patch(chats::handlers::set_active_revision),
        )
        .route(
            "/chats/{chat_id}/messages/{message_id}",
            patch(chats::handlers::edit_message).delete(chats::handlers::delete_message),
        )
        .route(
            "/chats/{chat_id}/messages/{message_id}/branch",
            post(chats::handlers::branch_message),
        )
        .route(
            "/chats/{chat_id}/messages/{message_id}/regenerate",
            post(chats::handlers::regenerate_message),
        )
        .route(
            "/chats/{chat_id}/generate",
            post(chats::handlers::generate_chat),
        )
        .route(
            "/chats/{chat_id}/messages/{message_id}/stop",
            post(chats::handlers::stop_generation),
        )
        .route(
            "/chats/{chat_id}/attachments",
            post(uploads::handlers::upload_attachment),
        )
        .route(
            "/attachments/{attachment_id}",
            get(uploads::handlers::get_attachment).delete(uploads::handlers::delete_attachment),
        )
        .route("/private/vault-key", get(private::handlers::vault_key))
        .route("/private/generate", post(private::handlers::generate));

    #[cfg(debug_assertions)]
    let api = api.route(
        "/dev/private-stream-test",
        post(private::handlers::generate_stream_test),
    );

    let api = api
        .fallback(api_not_found)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            security::origin_check,
        ));

    Router::new()
        .nest("/api", api)
        .route("/", get(frontend::serve_index))
        .route("/app", get(frontend::serve_index))
        .route("/app/{*path}", get(frontend::serve_index))
        .fallback(frontend::serve_asset)
        .with_state(state)
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .layer(SetResponseHeaderLayer::if_not_present(
            HeaderName::from_static("content-security-policy"),
            HeaderValue::from_static(
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
            ),
        ))
        .layer(SetResponseHeaderLayer::if_not_present(
            HeaderName::from_static("permissions-policy"),
            HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
        ))
        .layer(SetResponseHeaderLayer::if_not_present(
            header::REFERRER_POLICY,
            HeaderValue::from_static("no-referrer"),
        ))
        .layer(SetResponseHeaderLayer::if_not_present(
            header::X_FRAME_OPTIONS,
            HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::if_not_present(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(TraceLayer::new_for_http())
}

async fn api_not_found() -> ApiError {
    ApiError::not_found("not_found", "API route not found")
}

fn spawn_session_cleanup(db: sqlx::SqlitePool) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60 * 60));

        loop {
            interval.tick().await;

            match auth::service::delete_expired_sessions(&db).await {
                Ok(deleted) if deleted > 0 => {
                    tracing::debug!(deleted, "deleted expired sessions");
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(?error, "failed to delete expired sessions");
                }
            }
        }
    });
}

fn spawn_model_cache_refresh(state: AppState) {
    tokio::spawn(async move {
        loop {
            match state
                .model_cache
                .refresh_all(&state.db, &state.http_client)
                .await
            {
                Ok(snapshot) => {
                    let model_count: usize = snapshot
                        .backends
                        .values()
                        .map(|backend| backend.models.len())
                        .sum();
                    tracing::debug!(
                        backend_count = snapshot.backends.len(),
                        model_count,
                        "refreshed Ollama model cache"
                    );
                }
                Err(error) => {
                    tracing::warn!(?error, "failed to refresh Ollama model cache");
                }
            }

            tokio::time::sleep(Duration::from_secs(5 * 60)).await;
        }
    });
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install terminate signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
