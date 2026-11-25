use super::{client, models::*};
use crate::{app_state::AppState, auth::service::unix_timestamp, error::ApiError};
use axum::body::Bytes;
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sqlx::{Row, SqlitePool};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, Semaphore};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub struct Jobs {
    submissions: Mutex<()>,
    entries: Mutex<HashMap<String, Entry>>,
    http: reqwest::Client,
    decoders: std::sync::Arc<Semaphore>,
}
struct Entry {
    owner: String,
    signature: String,
    status: String,
    progress: Value,
    output: Option<Bytes>,
    cancel: CancellationToken,
    touched: Instant,
}
impl Default for Jobs {
    fn default() -> Self {
        Self {
            submissions: Mutex::new(()),
            entries: Mutex::new(HashMap::new()),
            decoders: std::sync::Arc::new(Semaphore::new(2)),
            http: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("valid video client"),
        }
    }
}
pub fn invalid(text: &str) -> ApiError {
    ApiError::bad_request("video_request_invalid", text)
}
fn missing() -> ApiError {
    ApiError::not_found("video_not_found", "Video flow or clip not found")
}
pub fn encode(value: &impl serde::Serialize) -> Result<String, ApiError> {
    serde_json::to_string(value).map_err(|_| ApiError::internal("Could not encode video state"))
}
pub fn decode<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, ApiError> {
    serde_json::from_str(text).map_err(|_| invalid("Invalid saved video settings"))
}
pub fn validate(d: &Draft) -> Result<(), ApiError> {
    if !d.within_clip_limits() {
        return Err(invalid(
            "This clip is too long or too large. Shorten it or lower the resolution.",
        ));
    }
    let mut check = d.clone();
    if check.prompt.trim().is_empty() {
        check.prompt = "Draft".into();
    }
    let reference = d.workflow == kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE;
    let inputs = if reference {
        Inputs {
            reference_images: vec!["placeholder".into(); d.reference_image_ids.len().max(1)],
            ..Inputs::default()
        }
    } else {
        Inputs {
            first_frame: (d.first_frame_id.is_some()
                || d.workflow == kuno_protocol::MINIMAX_H3_I2V_RECIPE)
                .then(|| "placeholder".into()),
            last_frame: d.last_frame_id.as_ref().map(|_| "placeholder".into()),
            reference_images: vec!["placeholder".into(); d.reference_image_ids.len()],
        }
    };
    check
        .recipe(inputs)
        .validate()
        .map_err(|_| invalid("Unsupported video settings"))?;
    if d.backend_id.len() > 128
        || !matches!(d.seed_mode.as_str(), "fixed" | "increment" | "random")
        || d.image_ids().any(|id| Uuid::parse_str(id).is_err())
        || d.reference_image_ids.len() > kuno_protocol::VIDEO_MAX_REFERENCE_IMAGES
        || (reference && (d.first_frame_id.is_some() || d.last_frame_id.is_some()))
        || d.input_mode.as_deref().is_some_and(|mode| match mode {
            "text" => {
                reference
                    || d.first_frame_id.is_some()
                    || d.last_frame_id.is_some()
                    || d.workflow == kuno_protocol::MINIMAX_H3_I2V_RECIPE
            }
            "keyframes" => reference,
            "references" => !reference,
            _ => true,
        })
    {
        return Err(invalid("Invalid host, seed mode, or video input images"));
    }
    Ok(())
}

pub(super) fn supported_clip(
    draft: &Draft,
    workflow: &kuno_protocol::VideoWorkflowCapability,
) -> bool {
    if !draft.within_clip_limits()
        || (draft.audio && !workflow.audio)
        || workflow.dimension_multiple == 0
        || [draft.width, draft.height]
            .iter()
            .any(|side| !side.is_multiple_of(workflow.dimension_multiple))
    {
        return false;
    }
    if workflow.limits == "memory" && workflow.frame_base > 0 && workflow.frame_step > 0 {
        draft
            .frames
            .checked_sub(workflow.frame_base)
            .is_some_and(|offset| offset.is_multiple_of(workflow.frame_step))
    } else {
        workflow.frames.contains(&draft.frames)
            && draft.pixels() <= u64::from(workflow.max_pixels)
            && [draft.width, draft.height]
                .iter()
                .all(|side| (workflow.min_dimension..=workflow.max_dimension).contains(side))
    }
}
pub async fn hosts(state: &AppState) -> Result<Value, ApiError> {
    let rows = sqlx::query(
        "SELECT id,name,base_url FROM ollama_backends WHERE is_enabled=1 ORDER BY name,id",
    )
    .fetch_all(&state.db)
    .await?;
    let mut hosts = Vec::new();
    for row in rows {
        let base: String = row.try_get("base_url")?;
        let discovery = state.image_jobs.kuno.discover(&base).await;
        if let Some(mut c) = discovery
            .capabilities
            .and_then(|c| c.videos)
            .filter(client::supported)
        {
            c.workflows.retain(|w| {
                w.state == "ready"
                    && matches!(
                        w.id.as_str(),
                        kuno_protocol::MINIMAX_H3_I2V_RECIPE
                            | kuno_protocol::MINIMAX_H3_VIDEO_RECIPE
                            | kuno_protocol::MINIMAX_H3_REFERENCE_RECIPE
                    )
                    && w.fps == 24
                    && w.output_mime_type == "video/mp4"
            });
            if !c.workflows.is_empty() {
                hosts.push(json!({"backend":{"id":row.get::<String,_>("id"),"name":row.get::<String,_>("name")},"capabilities":c}));
            }
        }
    }
    Ok(json!(hosts))
}
pub async fn preferences(pool: &SqlitePool, owner: &str) -> Result<Draft, ApiError> {
    let saved: Option<String> =
        sqlx::query_scalar("SELECT value_json FROM video_preferences WHERE user_id=?")
            .bind(owner)
            .fetch_optional(pool)
            .await?;
    saved
        .map(|s| decode(&s))
        .unwrap_or_else(|| Ok(Draft::default()))
}
pub async fn save_preferences(
    pool: &SqlitePool,
    owner: &str,
    mut draft: Draft,
) -> Result<Draft, ApiError> {
    draft.prompt.clear();
    draft.first_frame_id = None;
    draft.last_frame_id = None;
    draft.reference_image_ids.clear();
    validate(&draft)?;
    sqlx::query("INSERT INTO video_preferences VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET value_json=excluded.value_json").bind(owner).bind(encode(&draft)?).execute(pool).await?;
    Ok(draft)
}
pub async fn frame(state: &AppState, owner: &str, bytes: Vec<u8>) -> Result<Value, ApiError> {
    if bytes.len() > 12 * 1024 * 1024 {
        return Err(invalid("An input image must be at most 12 MiB"));
    }
    let input = STANDARD.encode(&bytes);
    let permit = state
        .video_jobs
        .decoders
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| invalid("Image decoding unavailable"))?;
    let (w, h, thumbnail) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut decoded = kuno_image::references::decode(&[input])
            .map_err(|_| invalid("Use a still PNG, JPEG, or WebP up to 4096 pixels per side"))?;
        let p = decoded.remove(0);
        let (w, h) = (p.width, p.height);
        let image =
            image::RgbaImage::from_raw(w, h, p.rgba).ok_or_else(|| invalid("Invalid image"))?;
        let thumbnail = image::DynamicImage::ImageRgba8(image).thumbnail(512, 512);
        let mut out = std::io::Cursor::new(Vec::new());
        thumbnail
            .write_to(&mut out, image::ImageFormat::Png)
            .map_err(|_| invalid("Invalid image"))?;
        Ok::<_, ApiError>((w, h, out.into_inner()))
    })
    .await
    .map_err(|_| invalid("Could not decode the input image"))??;
    let mime = if bytes.starts_with(b"\x89PNG") {
        "image/png"
    } else if bytes.starts_with(b"\xff\xd8") {
        "image/jpeg"
    } else {
        "image/webp"
    };
    let id = Uuid::new_v4().to_string();
    let mut tx = state.db.begin().await?;
    sqlx::query("UPDATE image_service_settings SET id=id WHERE id=1")
        .execute(&mut *tx)
        .await?;
    crate::images::service::check_quota(&mut tx, owner, bytes.len() + thumbnail.len()).await?;
    sqlx::query("INSERT INTO video_frames VALUES(?,?,?,?,?,?,?,?)")
        .bind(&id)
        .bind(owner)
        .bind(mime)
        .bind(w)
        .bind(h)
        .bind(&bytes)
        .bind(&thumbnail)
        .bind((bytes.len() + thumbnail.len()) as i64)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(json!({"id":id,"width":w,"height":h,"mime_type":mime}))
}
pub async fn read_frame(
    pool: &SqlitePool,
    owner: &str,
    id: &str,
    thumbnail: bool,
) -> Result<(String, Vec<u8>), ApiError> {
    let row =
        sqlx::query(if thumbnail {"SELECT 'image/png' AS mime_type,thumbnail AS bytes FROM video_frames WHERE id=? AND user_id=?"}else{"SELECT mime_type,body AS bytes FROM video_frames WHERE id=? AND user_id=?"})
            .bind(id)
            .bind(owner)
            .fetch_optional(pool)
            .await?
            .ok_or_else(missing)?;
    Ok((row.try_get("mime_type")?, row.try_get("bytes")?))
}
pub async fn session(pool: &SqlitePool, owner: &str, id: &str) -> Result<Session, ApiError> {
    sqlx::query_as("SELECT * FROM video_sessions WHERE id=? AND user_id=?")
        .bind(id)
        .bind(owner)
        .fetch_optional(pool)
        .await?
        .ok_or_else(missing)
}
pub async fn sessions(
    pool: &SqlitePool,
    owner: &str,
    q: &str,
    offset: i64,
) -> Result<Vec<Session>, ApiError> {
    Ok(sqlx::query_as("SELECT * FROM video_sessions WHERE user_id=? AND archived_at IS NULL AND instr(lower(title),lower(?))>0 ORDER BY updated_at DESC,rowid DESC LIMIT 50 OFFSET ?").bind(owner).bind(q).bind(offset.clamp(0,100000)).fetch_all(pool).await?)
}
pub async fn create_session(
    pool: &SqlitePool,
    owner: &str,
    draft: Draft,
) -> Result<Session, ApiError> {
    validate(&draft)?;
    let id = Uuid::new_v4().to_string();
    let now = unix_timestamp();
    sqlx::query("INSERT INTO video_sessions(id,user_id,title,draft_json,created_at,updated_at) VALUES(?,?,'New video',?,?,?)").bind(&id).bind(owner).bind(encode(&draft)?).bind(now).bind(now).execute(pool).await?;
    session(pool, owner, &id).await
}
pub async fn runs(pool: &SqlitePool, owner: &str, id: &str) -> Result<Vec<Run>, ApiError> {
    session(pool, owner, id).await?;
    Ok(sqlx::query_as(
        "SELECT * FROM (SELECT *,rowid AS sequence FROM video_runs WHERE user_id=? AND session_id=? ORDER BY rowid DESC LIMIT 100) ORDER BY sequence",
    )
    .bind(owner)
    .bind(id)
    .fetch_all(pool)
    .await?)
}
pub async fn run(pool: &SqlitePool, owner: &str, id: &str) -> Result<Run, ApiError> {
    sqlx::query_as("SELECT * FROM video_runs WHERE id=? AND user_id=?")
        .bind(id)
        .bind(owner)
        .fetch_optional(pool)
        .await?
        .ok_or_else(missing)
}
async fn allowed(state: &AppState, owner: &str) -> Result<(), ApiError> {
    let enabled: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM users WHERE id=? AND is_disabled=0)")
            .bind(owner)
            .fetch_one(&state.db)
            .await?;
    crate::permissions::service::ensure_tool_records(&state.db).await?;
    let tags = crate::permissions::service::effective_user_tag_ids(&state.db, owner).await?;
    let permissions = crate::permissions::service::tool_tags_by_tool(&state.db).await?;
    if !enabled
        || !permissions
            .get("images")
            .is_some_and(|allowed| crate::permissions::service::has_matching_tag(&tags, allowed))
    {
        return Err(ApiError::forbidden(
            "videos_not_allowed",
            "Video access is unavailable for this account",
        ));
    }
    Ok(())
}
pub async fn submit(
    state: &AppState,
    owner: &str,
    session_id: Option<&str>,
    mut input: Request,
    private_frame: Option<Inputs>,
) -> Result<String, ApiError> {
    Uuid::parse_str(&input.request_id).map_err(|_| invalid("Invalid request ID"))?;
    validate(&input.draft)?;
    let signature = encode(&json!({"request":input,"frame":private_frame}))?;
    let _submission = state.video_jobs.submissions.lock().await;
    if session_id.is_none()
        && sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM video_runs WHERE id=?)")
            .bind(&input.request_id)
            .fetch_one(&state.db)
            .await?
    {
        return Err(missing());
    }
    if let Some(session_id) = session_id {
        session(&state.db, owner, session_id).await?;
        if let Ok(saved) = run(&state.db, owner, &input.request_id).await {
            let old: String = sqlx::query_scalar("SELECT request_json FROM video_runs WHERE id=?")
                .bind(&saved.id)
                .fetch_one(&state.db)
                .await?;
            if saved.session_id != session_id || old != signature {
                return Err(ApiError::conflict(
                    "request_changed",
                    "This request ID was already used for another video",
                ));
            }
            return Ok(saved.id);
        }
    }
    let jobs = state.video_jobs.entries.lock().await;
    if let Some(entry) = jobs.get(&input.request_id) {
        if entry.owner == owner && entry.signature == signature {
            return Ok(input.request_id);
        }
        return Err(missing());
    }
    if jobs.len() >= 8
        || jobs
            .values()
            .filter(|e| {
                e.owner == owner
                    && !matches!(e.status.as_str(), "complete" | "failed" | "cancelled")
            })
            .count()
            >= 2
    {
        return Err(ApiError::conflict(
            "video_busy",
            "Wait for or stop an existing video generation",
        ));
    }
    drop(jobs);
    allowed(state, owner).await?;
    let mut group = input.request_id.clone();
    if let Some(original) = &input.retry_of {
        let saved = run(&state.db, owner, original).await?;
        if Some(saved.session_id.as_str()) != session_id {
            return Err(missing());
        }
        group = saved.group_id;
        // A retry's immutable prompt, source and inference recipe come from the stored run.
        let origin = run(&state.db, owner, &group).await?;
        input.draft = decode(&origin.draft_json)?;
        let count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM video_runs WHERE user_id=? AND group_id=?")
                .bind(owner)
                .bind(&group)
                .fetch_one(&state.db)
                .await?;
        input.draft.seed = crate::images::service::seed(
            &input.draft.seed,
            if input.draft.seed_mode == "fixed" {
                crate::images::models::SeedMode::Fixed
            } else if input.draft.seed_mode == "random" {
                crate::images::models::SeedMode::Random
            } else {
                crate::images::models::SeedMode::Increment
            },
            count as u32,
        )?;
    }
    let backend = crate::backends::service::get_enabled_backend(&state.db, &input.draft.backend_id)
        .await?
        .ok_or_else(|| invalid("Choose an available video model"))?;
    let capability = state
        .image_jobs
        .kuno
        .discover(&backend.base_url)
        .await
        .capabilities
        .and_then(|c| c.videos)
        .filter(client::supported)
        .ok_or_else(|| invalid("Video is unavailable on this host. Refresh available models."))?;
    let workflow = capability
        .workflows
        .iter()
        .find(|w| w.id == input.draft.workflow && w.state == "ready")
        .ok_or_else(|| invalid("This video workflow is unavailable"))?;
    if !supported_clip(&input.draft, workflow) {
        return Err(invalid(
            "These settings are not advertised by this video model",
        ));
    }
    let inputs = if let Some(inputs) = private_frame {
        inputs
    } else {
        let mut inputs = Inputs::default();
        for (id, target) in [
            (&input.draft.first_frame_id, &mut inputs.first_frame),
            (&input.draft.last_frame_id, &mut inputs.last_frame),
        ] {
            if let Some(id) = id {
                *target = Some(STANDARD.encode(read_frame(&state.db, owner, id, false).await?.1));
            }
        }
        for id in &input.draft.reference_image_ids {
            inputs
                .reference_images
                .push(STANDARD.encode(read_frame(&state.db, owner, id, false).await?.1));
        }
        inputs
    };
    if (workflow.first_frame == "required" && inputs.first_frame.is_none())
        || (workflow.first_frame == "unsupported" && inputs.first_frame.is_some())
        || (workflow.last_frame != "optional" && inputs.last_frame.is_some())
        || inputs.reference_images.len() > workflow.max_reference_images as usize
    {
        return Err(invalid(
            "These input images are not supported by the selected video model",
        ));
    }
    let recipe = input.draft.recipe(inputs);
    recipe.validate().map_err(|code| match code {
        "video_prompt_invalid" => invalid("Describe the video, or attach an input image"),
        "video_first_frame_invalid" => invalid("Choose a valid start frame for this video model"),
        "video_last_frame_invalid" => {
            invalid("This video model does not accept the selected end frame")
        }
        "video_reference_images_invalid" => {
            invalid("Choose between one and four reference images, without keyframes")
        }
        _ => invalid("Invalid video recipe"),
    })?;
    let encoded: Vec<_> = recipe
        .first_frame
        .iter()
        .chain(recipe.last_frame.iter())
        .chain(recipe.reference_images.iter())
        .cloned()
        .collect();
    if encoded.iter().map(String::len).sum::<usize>() + recipe.prompt.len() + 8192
        > 64 * 1024 * 1024
    {
        return Err(invalid(
            "The input images are too large together. Use smaller images.",
        ));
    }
    let permit = state
        .video_jobs
        .decoders
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| invalid("Image decoding unavailable"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        kuno_image::references::decode(&encoded)
    })
    .await
    .map_err(|_| invalid("Invalid video input image"))?
    .map_err(|_| invalid("Invalid video input image"))?;
    if let Some(session_id) = session_id {
        let mut tx = state.db.begin().await?;
        let changed=sqlx::query("UPDATE video_sessions SET revision=revision+1 WHERE id=? AND user_id=? AND archived_at IS NULL").bind(session_id).bind(owner).execute(&mut *tx).await?;
        if changed.rows_affected() != 1 {
            return Err(ApiError::conflict(
                "video_flow_archived",
                "Restore this flow before generating",
            ));
        }
        let busy:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM video_runs WHERE session_id=? AND status IN ('queued','generating','saving'))").bind(session_id).fetch_one(&mut *tx).await?;
        if busy {
            return Err(ApiError::conflict(
                "video_busy",
                "This conversation already has an active generation",
            ));
        }
        let title = recipe
            .prompt
            .lines()
            .next()
            .unwrap_or("New video")
            .chars()
            .take(80)
            .collect::<String>();
        sqlx::query("UPDATE video_sessions SET draft_json=?,title=CASE WHEN title='New video' THEN ? ELSE title END,updated_at=? WHERE id=? AND user_id=? AND archived_at IS NULL").bind(encode(&input.draft)?).bind(title).bind(unix_timestamp()).bind(session_id).bind(owner).execute(&mut *tx).await?;
        sqlx::query("INSERT INTO video_runs(id,session_id,user_id,group_id,request_json,draft_json,status,created_at) VALUES(?,?,?,?,?,?,'queued',?)").bind(&input.request_id).bind(session_id).bind(owner).bind(group).bind(&signature).bind(encode(&input.draft)?).bind(unix_timestamp()).execute(&mut *tx).await?;
        tx.commit().await?;
    }
    let id = input.request_id;
    let cancel = CancellationToken::new();
    state.video_jobs.entries.lock().await.insert(
        id.clone(),
        Entry {
            owner: owner.into(),
            signature,
            status: "queued".into(),
            progress: json!({}),
            output: None,
            cancel: cancel.clone(),
            touched: Instant::now(),
        },
    );
    let state = state.clone();
    let owner = owner.to_owned();
    let job = id.clone();
    let saved = session_id.is_some();
    tokio::spawn(async move {
        let result = {
            let begin = Instant::now();
            let work = async {
                let _permit = state
                    .image_jobs
                    .slots
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| invalid("Media service unavailable"))?;
                allowed(&state, &owner).await?;
                let backend = crate::backends::service::get_enabled_backend(
                    &state.db,
                    &input.draft.backend_id,
                )
                .await?
                .ok_or_else(|| invalid("The selected video host was disabled before generation"))?;
                if saved {
                    sqlx::query("UPDATE video_runs SET status='generating' WHERE id=? AND user_id=? AND status='queued'").bind(&job).bind(&owner).execute(&state.db).await?;
                }
                let jobs = state.video_jobs.clone();
                let job_id = job.clone();
                let bytes = client::generate(
                    &state.video_jobs.http,
                    &backend.base_url,
                    &recipe,
                    move |value| {
                        if let Ok(mut entries) = jobs.entries.try_lock()
                            && let Some(entry) = entries.get_mut(&job_id)
                        {
                            entry.status = "generating".into();
                            if let (Some(target), Some(new)) =
                                (entry.progress.as_object_mut(), value.as_object())
                            {
                                for (k, v) in new {
                                    target.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    },
                )
                .await?;
                allowed(&state, &owner).await?;
                if saved {
                    let directory = state.config.data_dir.join("videos");
                    tokio::fs::create_dir_all(&directory)
                        .await
                        .map_err(|_| invalid("Could not store video"))?;
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        tokio::fs::set_permissions(
                            &directory,
                            std::fs::Permissions::from_mode(0o700),
                        )
                        .await
                        .map_err(|_| invalid("Could not protect video"))?;
                    }
                    let path = directory.join(format!("{job}.part"));
                    let mut options = tokio::fs::OpenOptions::new();
                    options.create_new(true).write(true);
                    #[cfg(unix)]
                    options.mode(0o600);
                    let mut file = options
                        .open(&path)
                        .await
                        .map_err(|_| invalid("Could not store video"))?;
                    use tokio::io::AsyncWriteExt;
                    file.write_all(&bytes)
                        .await
                        .map_err(|_| invalid("Could not store video"))?;
                    drop(file);
                    tokio::fs::rename(&path, directory.join(format!("{job}.mp4")))
                        .await
                        .map_err(|_| invalid("Could not store video"))?;
                    let mut tx = state.db.begin().await?;
                    sqlx::query("UPDATE image_service_settings SET id=id WHERE id=1")
                        .execute(&mut *tx)
                        .await?;
                    crate::images::service::check_quota(&mut tx, &owner, bytes.len()).await?;
                    let updated=sqlx::query("UPDATE video_runs SET status='complete',size_bytes=?,duration_ms=? WHERE id=? AND user_id=? AND status IN ('queued','generating')").bind(bytes.len() as i64).bind(begin.elapsed().as_millis() as i64).bind(&job).bind(&owner).execute(&mut *tx).await?;
                    if updated.rows_affected() != 1 {
                        return Err(invalid("Generation was stopped before saving"));
                    }
                    tx.commit().await?;
                }
                Ok::<_, ApiError>(Bytes::from(bytes))
            };
            tokio::pin!(work);
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tokio::select! {
                    biased;
                    _=cancel.cancelled()=>break None,
                    _=tick.tick(),if !saved=>{if state.video_jobs.entries.lock().await.get(&job).is_none_or(|e|e.touched.elapsed()>Duration::from_secs(45)){break None;}},
                    result=&mut work=>break Some(result),
                }
            }
        };
        let mut entries = state.video_jobs.entries.lock().await;
        if let Some(entry) = entries.get_mut(&job) {
            match result {
                Some(Ok(bytes)) if !cancel.is_cancelled() => {
                    entry.status = "complete".into();
                    if !saved {
                        entry.output = Some(bytes);
                    }
                }
                Some(Err(error)) => {
                    entry.status = "failed".into();
                    entry.progress = json!({"error":error.message()});
                    if saved {
                        let _=sqlx::query("UPDATE video_runs SET status='failed',error=? WHERE id=? AND user_id=? AND status IN ('queued','generating')").bind(error.message()).bind(&job).bind(&owner).execute(&state.db).await;
                    }
                }
                _ => {
                    entry.status = "cancelled".into();
                    if saved {
                        let _=sqlx::query("UPDATE video_runs SET status='cancelled' WHERE id=? AND user_id=? AND status IN ('queued','generating')").bind(&job).bind(&owner).execute(&state.db).await;
                    }
                }
            }
        }
        if saved {
            entries.remove(&job);
        }
        drop(entries);
        let complete = run(&state.db, &owner, &job)
            .await
            .is_ok_and(|r| r.status == "complete");
        if saved && !complete {
            let _ = tokio::fs::remove_file(
                state
                    .config
                    .data_dir
                    .join("videos")
                    .join(format!("{job}.mp4")),
            )
            .await;
            let _ = tokio::fs::remove_file(
                state
                    .config
                    .data_dir
                    .join("videos")
                    .join(format!("{job}.part")),
            )
            .await;
        }
        if !saved {
            tokio::select! {_=cancel.cancelled()=>{},_=tokio::time::sleep(Duration::from_secs(120))=>{}}
            state.video_jobs.entries.lock().await.remove(&job);
        }
    });
    Ok(id)
}
pub async fn status(state: &AppState, owner: &str, id: &str) -> Result<Value, ApiError> {
    let mut entries = state.video_jobs.entries.lock().await;
    if let Some(entry) = entries.get_mut(id).filter(|e| e.owner == owner) {
        entry.touched = Instant::now();
        return Ok(json!({"status":entry.status,"progress":entry.progress}));
    }
    drop(entries);
    let run = run(&state.db, owner, id).await?;
    Ok(json!({"status":run.status,"progress":{"error":run.error}}))
}
pub async fn stop(state: &AppState, owner: &str, id: &str) -> Result<(), ApiError> {
    if let Some(entry) = state
        .video_jobs
        .entries
        .lock()
        .await
        .get(id)
        .filter(|e| e.owner == owner)
    {
        entry.cancel.cancel();
    }
    sqlx::query("UPDATE video_runs SET status='cancelled' WHERE id=? AND user_id=? AND status IN ('queued','generating')").bind(id).bind(owner).execute(&state.db).await?;
    Ok(())
}
pub async fn result(state: &AppState, owner: &str, id: &str) -> Result<Bytes, ApiError> {
    if let Some(entry) = state
        .video_jobs
        .entries
        .lock()
        .await
        .get(id)
        .filter(|e| e.owner == owner)
    {
        if let Some(output) = &entry.output {
            return Ok(output.clone());
        }
        if entry.status != "complete" {
            return Err(ApiError::conflict("video_pending", "The clip is not ready"));
        }
    }
    let run = run(&state.db, owner, id).await?;
    if run.status != "complete" {
        return Err(invalid("The clip is not complete"));
    }
    let bytes = tokio::fs::read(
        state
            .config
            .data_dir
            .join("videos")
            .join(format!("{}.mp4", run.id)),
    )
    .await
    .map_err(|_| missing())?;
    Ok(Bytes::from(bytes))
}
pub async fn delete_session(state: &AppState, owner: &str, id: &str) -> Result<(), ApiError> {
    let _submission = state.video_jobs.submissions.lock().await;
    session(&state.db, owner, id).await?;
    let mut tx = state.db.begin().await?;
    sqlx::query("UPDATE video_sessions SET id=id WHERE id=? AND user_id=?")
        .bind(id)
        .bind(owner)
        .execute(&mut *tx)
        .await?;
    let active: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM video_runs WHERE session_id=? AND user_id=? AND status IN ('queued','generating','saving'))").bind(id).bind(owner).fetch_one(&mut *tx).await?;
    if active {
        return Err(ApiError::conflict(
            "video_busy",
            "Stop the active video before deleting this flow",
        ));
    }
    let clips: Vec<String> =
        sqlx::query_scalar("SELECT id FROM video_runs WHERE session_id=? AND user_id=?")
            .bind(id)
            .bind(owner)
            .fetch_all(&mut *tx)
            .await?;
    let frames: Vec<String> = sqlx::query_scalar("WITH drafts(draft) AS (SELECT draft_json FROM video_runs WHERE session_id=? AND user_id=? UNION ALL SELECT draft_json FROM video_sessions WHERE id=? AND user_id=?) SELECT value FROM drafts,json_each(json_array(json_extract(draft,'$.first_frame_id'),json_extract(draft,'$.last_frame_id'))) WHERE value IS NOT NULL UNION SELECT value FROM drafts,json_each(draft,'$.reference_image_ids') WHERE type='text'").bind(id).bind(owner).bind(id).bind(owner).fetch_all(&mut *tx).await?;
    sqlx::query("DELETE FROM video_sessions WHERE id=? AND user_id=?")
        .bind(id)
        .bind(owner)
        .execute(&mut *tx)
        .await?;
    for frame in frames {
        sqlx::query("DELETE FROM video_frames WHERE id=? AND user_id=? AND NOT EXISTS(SELECT 1 FROM (SELECT draft_json AS draft FROM video_runs WHERE user_id=? UNION ALL SELECT draft_json FROM video_sessions WHERE user_id=?) WHERE json_extract(draft,'$.first_frame_id')=? OR json_extract(draft,'$.last_frame_id')=? OR EXISTS(SELECT 1 FROM json_each(draft,'$.reference_image_ids') WHERE value=?))").bind(&frame).bind(owner).bind(owner).bind(owner).bind(&frame).bind(&frame).bind(&frame).execute(&mut *tx).await?;
    }
    tx.commit().await?;
    for clip in clips {
        for extension in ["mp4", "part"] {
            let _ = tokio::fs::remove_file(
                state
                    .config
                    .data_dir
                    .join("videos")
                    .join(format!("{clip}.{extension}")),
            )
            .await;
        }
    }
    Ok(())
}
pub async fn recover(
    pool: &SqlitePool,
    data_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    sqlx::query("UPDATE video_runs SET status='failed',error='The server restarted during generation. Your prompt and input images are retained; retry is explicit.' WHERE status IN ('queued','generating','saving')").execute(pool).await?;
    let directory = data_dir.join("videos");
    let mut files = match tokio::fs::read_dir(directory).await {
        Ok(files) => files,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    while let Some(file) = files.next_entry().await? {
        let path = file.path();
        let Some(id) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| Uuid::parse_str(s).is_ok())
        else {
            continue;
        };
        let extension = path.extension().and_then(|s| s.to_str());
        if !matches!(extension, Some("mp4" | "part")) || !file.file_type().await?.is_file() {
            continue;
        }
        let keep: bool = extension == Some("mp4")
            && sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM video_runs WHERE id=? AND status='complete')",
            )
            .bind(id)
            .fetch_one(pool)
            .await?;
        if !keep {
            tokio::fs::remove_file(path).await?;
        }
    }
    Ok(())
}
