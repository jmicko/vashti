mod admin;
mod app_state;
mod auth;
mod backends;
mod chats;
mod config;
mod db;
mod error;
mod frontend;
mod ollama;
mod private;
mod settings;
mod startup;
mod uploads;

use std::{error::Error, time::Duration};

use app_state::AppState;
use axum::{
    Router,
    routing::{get, patch, post},
};
use config::Config;
use error::ApiError;
use tokio::signal;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
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

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;

    startup::bootstrap::detect_localhost_ollama_if_empty(&db, &http_client).await?;

    let bind_addr = config.bind_addr;
    let state = AppState::new(config, db, http_client);
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
        .route("/auth/session", get(auth::handlers::session))
        .route("/auth/register", post(auth::handlers::register))
        .route("/auth/login", post(auth::handlers::login))
        .route("/auth/logout", post(auth::handlers::logout))
        .route(
            "/settings",
            get(settings::handlers::get_app_settings)
                .patch(settings::handlers::update_app_settings),
        )
        .route("/admin/users", get(admin::handlers::list_users))
        .route(
            "/admin/users/{user_id}",
            patch(admin::handlers::update_user).delete(admin::handlers::delete_user),
        )
        .route("/backends", get(backends::handlers::list_backends))
        .route("/models", get(backends::handlers::list_models))
        .fallback(api_not_found);

    Router::new()
        .nest("/api", api)
        .route("/", get(frontend::serve_index))
        .route("/app", get(frontend::serve_index))
        .route("/app/{*path}", get(frontend::serve_index))
        .fallback(frontend::serve_asset)
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

async fn api_not_found() -> ApiError {
    ApiError::not_found("not_found", "API route not found")
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
