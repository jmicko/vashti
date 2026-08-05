mod admin;
mod app_state;
mod auth;
mod backends;
mod chats;
mod config;
mod context_blocks;
mod db;
mod error;
mod frontend;
mod model_cache;
mod ollama;
mod permissions;
mod persona_avatars;
mod personas;
mod private;
mod rate_limit;
mod security;
mod settings;
mod startup;
mod tools;
mod updates;
mod uploads;
mod version;

use std::{error::Error, net::SocketAddr, time::Duration};

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
use tower_http::{
    compression::{
        CompressionLayer,
        predicate::{DefaultPredicate, NotForContentType, Predicate},
    },
    set_header::SetResponseHeaderLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const DEFAULT_REQUEST_BODY_LIMIT: usize = 1024 * 1024;
const LARGE_REQUEST_BODY_LIMIT: usize = 64 * 1024 * 1024;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|arg| arg == "--version" || arg == "-V")
    {
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

    if arguments.iter().any(|arg| arg == "--apply-update") {
        updates::worker::apply_requested_update().await?;
        return Ok(());
    }

    let config = Config::from_env()?;
    startup::prepare_data_dir(&config).await?;

    let db = db::connect(&config).await?;
    startup::secure_data_files(&config).await?;
    startup::migrations::run(&db).await?;
    let interrupted_generations = chats::service::recover_interrupted_generations(&db).await?;
    if interrupted_generations > 0 {
        tracing::warn!(
            interrupted_generations,
            "marked generations interrupted by the previous server shutdown as stopped"
        );
    }
    startup::bootstrap::ensure_app_settings(&db).await?;
    let server_instance_id = startup::bootstrap::ensure_server_identity(&db).await?;
    startup::network_recovery::recover_network_if_requested(&db, &config).await?;
    auth::service::delete_expired_sessions(&db).await?;

    let http_client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;

    startup::bootstrap::detect_localhost_ollama_if_empty(&db, &http_client).await?;

    let bind_addr = config.bind_addr;
    let state = AppState::new(config, db, http_client, server_instance_id);
    spawn_session_cleanup(state.db.clone());
    spawn_model_cache_refresh(state.clone());
    spawn_update_checks(state.clone());
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    tracing::info!("listening on http://{}", listener.local_addr()?);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    Ok(())
}

fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/version", get(version::get_version))
        .route("/admin/update", get(updates::handlers::get_update_status))
        .route(
            "/admin/update/check",
            post(updates::handlers::check_for_update),
        )
        .route(
            "/admin/update/install",
            post(updates::handlers::install_update),
        )
        .route("/auth/session", get(auth::handlers::session))
        .route("/auth/register", post(auth::handlers::register))
        .route("/auth/login", post(auth::handlers::login))
        .route("/auth/profile", patch(auth::handlers::update_profile))
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
                .patch(backends::handlers::update_user_model_preference),
        )
        .route(
            "/user-models/refresh",
            post(backends::handlers::refresh_user_models),
        )
        .route(
            "/user-models/avatar",
            patch(backends::handlers::update_user_model_avatar),
        )
        .route(
            "/user-models/background",
            patch(backends::handlers::update_user_model_background),
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
            "/admin/models/avatar",
            patch(backends::handlers::update_admin_model_avatar),
        )
        .route(
            "/admin/models/background",
            patch(backends::handlers::update_admin_model_background),
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
            "/context-library",
            get(context_blocks::handlers::get_library),
        )
        .route(
            "/context-categories",
            post(context_blocks::handlers::create_category),
        )
        .route(
            "/context-categories/{category_id}",
            patch(context_blocks::handlers::update_category)
                .delete(context_blocks::handlers::delete_category),
        )
        .route(
            "/context-blocks",
            post(context_blocks::handlers::create_block),
        )
        .route(
            "/context-blocks/{block_id}",
            patch(context_blocks::handlers::update_block)
                .delete(context_blocks::handlers::delete_block),
        )
        .route(
            "/context-blocks/{block_id}/versions",
            get(context_blocks::handlers::list_block_versions),
        )
        .route(
            "/persona-avatars",
            post(persona_avatars::handlers::upload_avatar)
                .layer(DefaultBodyLimit::max(LARGE_REQUEST_BODY_LIMIT)),
        )
        .route(
            "/persona-avatars/{asset_id}",
            get(persona_avatars::handlers::get_avatar)
                .delete(persona_avatars::handlers::delete_avatar),
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
            "/chats/{chat_id}/messages/{message_id}/continue",
            post(chats::handlers::continue_message),
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
            post(uploads::handlers::upload_attachment)
                .layer(DefaultBodyLimit::max(LARGE_REQUEST_BODY_LIMIT)),
        )
        .route(
            "/attachments/{attachment_id}",
            get(uploads::handlers::get_attachment).delete(uploads::handlers::delete_attachment),
        )
        .route("/private/vault-key", get(private::handlers::vault_key))
        .route(
            "/private/generate",
            post(private::handlers::generate)
                .layer(DefaultBodyLimit::max(LARGE_REQUEST_BODY_LIMIT)),
        );

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
        ))
        .layer(SetResponseHeaderLayer::if_not_present(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-store"),
        ));

    Router::new()
        .nest("/api", api)
        .route("/", get(frontend::serve_index))
        .route("/app", get(frontend::serve_index))
        .route("/app/{*path}", get(frontend::serve_index))
        .fallback(frontend::serve_asset)
        .with_state(state)
        .layer(DefaultBodyLimit::max(DEFAULT_REQUEST_BODY_LIMIT))
        .layer(
            CompressionLayer::new().compress_when(
                DefaultPredicate::new()
                    .and(NotForContentType::const_new("application/json"))
                    .and(NotForContentType::const_new("application/x-ndjson")),
            ),
        )
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

fn spawn_update_checks(state: AppState) {
    tokio::spawn(async move {
        loop {
            if let Err(error) = state
                .updates
                .check_for_update(&state.db, &state.http_client, &state.config)
                .await
            {
                tracing::warn!(?error, "failed to check for a Vashti update");
            }

            tokio::time::sleep(Duration::from_secs(6 * 60 * 60)).await;
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
