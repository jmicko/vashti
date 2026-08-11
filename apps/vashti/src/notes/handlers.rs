use axum::{
    Json,
    extract::{Path, Query, State},
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    auth,
    error::ApiError,
    notes::{
        models::{
            CreateNoteRequest, ExpectedNoteVersionRequest, NoteResponse, NoteSettingsResponse,
            NoteSummaryResponse, NoteVersionResponse, UpdateNoteRequest, UpdateNoteSettingsRequest,
        },
        service::{self, ListNotesOptions, NoteListStatus, NoteMutationActor, NoteSort},
    },
};

#[derive(Debug, Deserialize)]
pub struct NoteListQuery {
    pub query: Option<String>,
    pub status: Option<String>,
    pub sort: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct NoteListResponse {
    pub notes: Vec<NoteSummaryResponse>,
    pub total: i64,
}

#[derive(Debug, Serialize)]
pub struct NoteMutationResponse {
    pub note: NoteResponse,
}

#[derive(Debug, Serialize)]
pub struct NoteVersionsResponse {
    pub versions: Vec<NoteVersionResponse>,
}

#[derive(Debug, Serialize)]
pub struct NoteDeleteResponse {
    pub ok: bool,
}

pub async fn list_notes(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(query): Query<NoteListQuery>,
) -> Result<Json<NoteListResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let options = ListNotesOptions {
        query: query.query,
        status: parse_status(query.status.as_deref())?,
        sort: parse_sort(query.sort.as_deref())?,
        limit: query.limit.unwrap_or(50),
        offset: query.offset.unwrap_or(0),
    };
    let result = service::list_notes(&state.db, &user.id, options).await?;
    Ok(Json(NoteListResponse {
        notes: result.notes,
        total: result.total,
    }))
}

pub async fn get_note(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(note_id): Path<String>,
) -> Result<Json<NoteMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let note = service::get_note(&state.db, &user.id, &note_id).await?;
    Ok(Json(NoteMutationResponse { note }))
}

pub async fn create_note(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreateNoteRequest>,
) -> Result<Json<NoteMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let note = service::create_note(
        &state.db,
        &user.id,
        payload,
        &NoteMutationActor::human(&user.id),
    )
    .await?;
    Ok(Json(NoteMutationResponse { note }))
}

pub async fn update_note(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(note_id): Path<String>,
    Json(payload): Json<UpdateNoteRequest>,
) -> Result<Json<NoteMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let note = service::update_note(
        &state.db,
        &user.id,
        &note_id,
        payload,
        &NoteMutationActor::human(&user.id),
    )
    .await?;
    Ok(Json(NoteMutationResponse { note }))
}

pub async fn trash_note(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(note_id): Path<String>,
    Json(payload): Json<ExpectedNoteVersionRequest>,
) -> Result<Json<NoteMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let note = service::trash_note(
        &state.db,
        &user.id,
        &note_id,
        payload.expected_version,
        &NoteMutationActor::human(&user.id),
    )
    .await?;
    Ok(Json(NoteMutationResponse { note }))
}

pub async fn restore_note(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(note_id): Path<String>,
    Json(payload): Json<ExpectedNoteVersionRequest>,
) -> Result<Json<NoteMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let note =
        service::restore_note(&state.db, &user.id, &note_id, payload.expected_version).await?;
    Ok(Json(NoteMutationResponse { note }))
}

pub async fn permanently_delete_note(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(note_id): Path<String>,
) -> Result<Json<NoteDeleteResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    service::permanently_delete_note(&state.db, &user.id, &note_id).await?;
    Ok(Json(NoteDeleteResponse { ok: true }))
}

pub async fn list_versions(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(note_id): Path<String>,
) -> Result<Json<NoteVersionsResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let versions = service::list_versions(&state.db, &user.id, &note_id).await?;
    Ok(Json(NoteVersionsResponse { versions }))
}

pub async fn restore_version(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((note_id, version_id)): Path<(String, String)>,
    Json(payload): Json<ExpectedNoteVersionRequest>,
) -> Result<Json<NoteMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let note = service::restore_version(
        &state.db,
        &user.id,
        &note_id,
        &version_id,
        payload.expected_version,
        &NoteMutationActor::human(&user.id),
    )
    .await?;
    Ok(Json(NoteMutationResponse { note }))
}

pub async fn get_settings(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<NoteSettingsResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    Ok(Json(service::get_note_settings(&state.db, &user.id).await?))
}

pub async fn update_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateNoteSettingsRequest>,
) -> Result<Json<NoteSettingsResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    Ok(Json(
        service::update_note_settings(&state.db, &user.id, payload).await?,
    ))
}

async fn require_user(
    state: &AppState,
    jar: &CookieJar,
) -> Result<crate::auth::service::UserPublic, ApiError> {
    auth::service::require_user(&state.db, jar, &state.config.session_cookie_name).await
}

fn parse_status(value: Option<&str>) -> Result<NoteListStatus, ApiError> {
    match value.unwrap_or("active") {
        "active" => Ok(NoteListStatus::Active),
        "trashed" => Ok(NoteListStatus::Trashed),
        "all" => Ok(NoteListStatus::All),
        _ => Err(ApiError::bad_request(
            "invalid_note_status",
            "Note status must be active, trashed, or all",
        )),
    }
}

fn parse_sort(value: Option<&str>) -> Result<NoteSort, ApiError> {
    match value.unwrap_or("updated") {
        "updated" => Ok(NoteSort::Updated),
        "created" => Ok(NoteSort::Created),
        "title" => Ok(NoteSort::Title),
        _ => Err(ApiError::bad_request(
            "invalid_note_sort",
            "Note sort must be updated, created, or title",
        )),
    }
}
