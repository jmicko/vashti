use serde_json::{Value, json};
use sqlx::{Row, SqlitePool, sqlite::SqliteRow};
use uuid::Uuid;

use super::models::*;
use crate::{auth::service::unix_timestamp, backends, error::ApiError};

const TASK_SELECT: &str = "SELECT t.*, r.status AS last_status, r.finished_at AS last_finished_at FROM assistant_tasks t LEFT JOIN assistant_runs r ON r.id=t.last_run_id";

pub fn encode<T: serde::Serialize>(value: &T) -> Result<String, ApiError> {
    serde_json::to_string(value).map_err(|_| ApiError::internal("Could not encode assistant state"))
}

pub fn decode<T: serde::de::DeserializeOwned>(value: &str) -> Result<T, ApiError> {
    serde_json::from_str(value).map_err(|_| ApiError::internal("Stored assistant state is invalid"))
}

pub fn validate_config(config: &TaskConfig) -> Result<(), ApiError> {
    if config.backend_id.is_empty()
        || config.backend_id.len() > 128
        || config.model_name.trim().is_empty()
        || config.model_name.len() > 512
        || config.instructions.len() > 16_000
        || !(1..=64).contains(&config.max_steps)
        || !(256..=131_072).contains(&config.max_output_tokens)
    {
        return Err(ApiError::bad_request(
            "invalid_assistant_config",
            "Choose a model, 1-64 steps, and a 256-131072 output-token budget. Instructions are limited to 16000 bytes.",
        ));
    }
    Ok(())
}

pub async fn authorize_model(
    pool: &SqlitePool,
    owner: &str,
    config: &TaskConfig,
) -> Result<String, ApiError> {
    validate_config(config)?;
    let enabled: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM users WHERE id=? AND is_disabled=0)")
            .bind(owner)
            .fetch_one(pool)
            .await?;
    if !enabled {
        return Err(ApiError::forbidden(
            "account_disabled",
            "This account cannot run assistant tasks",
        ));
    }
    let backend = backends::service::get_enabled_backend(pool, &config.backend_id)
        .await?
        .ok_or_else(|| {
            ApiError::forbidden(
                "backend_unavailable",
                "The selected backend is unavailable or disabled",
            )
        })?;
    backends::service::ensure_model_enabled_for_user(
        pool,
        owner,
        &config.backend_id,
        &config.model_name,
    )
    .await?;
    Ok(backend.base_url)
}

pub async fn preferences(pool: &SqlitePool, owner: &str) -> Result<TaskConfig, ApiError> {
    let saved: Option<String> =
        sqlx::query_scalar("SELECT config_json FROM assistant_preferences WHERE user_id=?")
            .bind(owner)
            .fetch_optional(pool)
            .await?;
    saved
        .map(|value| decode(&value))
        .unwrap_or_else(|| Ok(TaskConfig::default()))
}

pub async fn save_preferences(
    pool: &SqlitePool,
    owner: &str,
    config: TaskConfig,
) -> Result<TaskConfig, ApiError> {
    authorize_model(pool, owner, &config).await?;
    sqlx::query("INSERT INTO assistant_preferences(user_id,config_json,updated_at) VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET config_json=excluded.config_json,updated_at=excluded.updated_at")
        .bind(owner).bind(encode(&config)?).bind(unix_timestamp()).execute(pool).await?;
    Ok(config)
}

fn task_row(row: SqliteRow) -> Result<Task, ApiError> {
    let last_status: Option<String> = row.try_get("last_status")?;
    let finished: Option<i64> = row.try_get("last_finished_at")?;
    let viewed: i64 = row.try_get("viewed_at")?;
    let schedule_error: Option<String> = row.try_get("schedule_error")?;
    let needs_attention = schedule_error.is_some()
        || last_status.as_deref().is_some_and(|status| {
            matches!(
                status,
                "waiting_approval" | "waiting_input" | "interrupted" | "limit_reached"
            )
        })
        || finished.is_some_and(|at| at > viewed);
    Ok(Task {
        id: row.try_get("id")?,
        title: row.try_get("title")?,
        instructions: row.try_get("instructions")?,
        config: decode(&row.try_get::<String, _>("config_json")?)?,
        interval_seconds: row.try_get("interval_seconds")?,
        next_run_at: row.try_get("next_run_at")?,
        schedule_enabled: row.try_get("schedule_enabled")?,
        schedule_error,
        archived_at: row.try_get("archived_at")?,
        last_run_id: row.try_get("last_run_id")?,
        last_status,
        needs_attention,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        revision: row.try_get("revision")?,
    })
}

pub async fn list_tasks(
    pool: &SqlitePool,
    owner: &str,
    offset: i64,
) -> Result<Vec<Task>, ApiError> {
    let rows=sqlx::query(&format!("{TASK_SELECT} WHERE t.user_id=? AND t.archived_at IS NULL ORDER BY t.updated_at DESC,t.id LIMIT 50 OFFSET ?"))
        .bind(owner).bind(offset.clamp(0,100_000)).fetch_all(pool).await?;
    rows.into_iter().map(task_row).collect()
}

pub async fn get_task(pool: &SqlitePool, owner: &str, id: &str) -> Result<Task, ApiError> {
    let row = sqlx::query(&format!("{TASK_SELECT} WHERE t.user_id=? AND t.id=?"))
        .bind(owner)
        .bind(id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ApiError::not_found("task_not_found", "Assistant task not found"))?;
    task_row(row)
}

pub async fn list_archived_tasks(
    pool: &SqlitePool,
    owner: &str,
    offset: i64,
) -> Result<Vec<Task>, ApiError> {
    sqlx::query(&format!("{TASK_SELECT} WHERE t.user_id=? AND t.archived_at IS NOT NULL ORDER BY t.updated_at DESC,t.id LIMIT 50 OFFSET ?"))
        .bind(owner).bind(offset.clamp(0,100_000)).fetch_all(pool).await?.into_iter().map(task_row).collect()
}

pub async fn save_task(
    pool: &SqlitePool,
    owner: &str,
    id: Option<&str>,
    payload: SaveTask,
) -> Result<Task, ApiError> {
    let title = payload.title.trim();
    if title.is_empty()
        || title.chars().count() > 160
        || payload.instructions.trim().is_empty()
        || payload.instructions.len() > 32_000
    {
        return Err(ApiError::bad_request(
            "invalid_task",
            "A title (up to 160 characters) and instructions (up to 32000 bytes) are required",
        ));
    }
    if payload
        .interval_seconds
        .is_some_and(|n| !(300..=2_592_000).contains(&n))
        || payload
            .next_run_at
            .is_some_and(|at| at < 0 || at > unix_timestamp() + 315_576_000)
        || (payload.schedule_enabled && payload.next_run_at.is_none())
    {
        return Err(ApiError::bad_request(
            "invalid_schedule",
            "Choose a start time and, for recurring work, an interval between 5 minutes and 30 days",
        ));
    }
    authorize_model(pool, owner, &payload.config).await?;
    let now = unix_timestamp();
    let key = id
        .map(str::to_owned)
        .or_else(|| payload.request_id.clone())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    Uuid::parse_str(&key)
        .map_err(|_| ApiError::bad_request("invalid_request_id", "Request IDs must be UUIDs"))?;
    if id.is_none() {
        let exists: bool =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM assistant_tasks WHERE id=?)")
                .bind(&key)
                .fetch_one(pool)
                .await?;
        if exists {
            let existing = get_task(pool, owner, &key).await?;
            if existing.title == title
                && existing.instructions == payload.instructions.trim()
                && encode(&existing.config)? == encode(&payload.config)?
            {
                return Ok(existing);
            }
            return Err(ApiError::conflict(
                "request_changed",
                "This request ID already created a different task. Open it before submitting another draft.",
            ));
        }
    }
    if id.is_some() {
        get_task(pool, owner, &key).await?;
        let updated=sqlx::query("UPDATE assistant_tasks SET title=?,instructions=?,config_json=?,interval_seconds=?,next_run_at=?,schedule_enabled=?,schedule_error=NULL,updated_at=?,revision=revision+1 WHERE id=? AND user_id=? AND revision=? AND archived_at IS NULL")
            .bind(title).bind(payload.instructions.trim()).bind(encode(&payload.config)?)
            .bind(payload.interval_seconds).bind(payload.next_run_at).bind(payload.schedule_enabled).bind(now)
            .bind(&key).bind(owner).bind(payload.expected_revision.unwrap_or(-1)).execute(pool).await?;
        if updated.rows_affected() == 0 {
            return Err(ApiError::conflict(
                "task_changed",
                "This task changed elsewhere. Reload before saving.",
            ));
        }
    } else {
        let count: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM assistant_tasks WHERE user_id=? AND archived_at IS NULL",
        )
        .bind(owner)
        .fetch_one(pool)
        .await?;
        if count >= 200 {
            return Err(ApiError::too_many_requests(
                "task_limit",
                "Archive an existing task before adding another",
            ));
        }
        sqlx::query("INSERT INTO assistant_tasks(id,user_id,title,instructions,config_json,interval_seconds,next_run_at,schedule_enabled,viewed_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)")
            .bind(&key).bind(owner).bind(title).bind(payload.instructions.trim()).bind(encode(&payload.config)?)
            .bind(payload.interval_seconds).bind(payload.next_run_at).bind(payload.schedule_enabled)
            .bind(now-1).bind(now).bind(now).execute(pool).await?;
    }
    get_task(pool, owner, &key).await
}

pub async fn list_runs(
    pool: &SqlitePool,
    owner: &str,
    task_id: &str,
) -> Result<Vec<Run>, ApiError> {
    get_task(pool, owner, task_id).await?;
    Ok(sqlx::query_as::<_,Run>("SELECT * FROM assistant_runs WHERE task_id=? AND user_id=? ORDER BY created_at DESC,rowid DESC LIMIT 50")
        .bind(task_id).bind(owner).fetch_all(pool).await?)
}

pub async fn get_run(pool: &SqlitePool, owner: &str, id: &str) -> Result<Run, ApiError> {
    sqlx::query_as::<_, Run>("SELECT * FROM assistant_runs WHERE id=? AND user_id=?")
        .bind(id)
        .bind(owner)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ApiError::not_found("run_not_found", "Assistant run not found"))
}

pub async fn start_run(
    pool: &SqlitePool,
    owner: &str,
    task_id: &str,
    request: StartRun,
    scheduled: bool,
) -> Result<Run, ApiError> {
    let task = get_task(pool, owner, task_id).await?;
    if let Some(id) = &request.request_id {
        Uuid::parse_str(id).map_err(|_| {
            ApiError::bad_request("invalid_request_id", "Request IDs must be UUIDs")
        })?;
        if let Some(row) =
            sqlx::query("SELECT user_id,task_id,request_message FROM assistant_runs WHERE id=?")
                .bind(id)
                .fetch_optional(pool)
                .await?
        {
            if row.try_get::<String, _>("user_id")? != owner {
                return Err(ApiError::not_found(
                    "run_not_found",
                    "Assistant run not found",
                ));
            }
            if row.try_get::<String, _>("task_id")? != task_id
                || row.try_get::<String, _>("request_message")? != request.message
            {
                return Err(ApiError::conflict(
                    "request_changed",
                    "This request ID was already used for different instructions",
                ));
            }
            return get_run(pool, owner, id).await;
        }
    }
    if task.archived_at.is_some() {
        return Err(ApiError::conflict(
            "task_archived",
            "Restore this task before running it",
        ));
    }
    if request.message.len() > 32_000 {
        return Err(ApiError::bad_request(
            "message_too_long",
            "Follow-up is limited to 32000 bytes",
        ));
    }
    authorize_model(pool, owner, &task.config).await?;
    let mut tx = pool.begin().await?;
    if scheduled {
        let now = unix_timestamp();
        let claimed=sqlx::query("UPDATE assistant_tasks SET next_run_at=?,schedule_enabled=?,schedule_error=NULL,revision=revision+1 WHERE id=? AND user_id=? AND revision=? AND schedule_enabled=1 AND archived_at IS NULL AND next_run_at=? AND next_run_at<=?")
            .bind(task.interval_seconds.map(|n|now+n)).bind(task.interval_seconds.is_some()).bind(task_id).bind(owner).bind(task.revision).bind(task.next_run_at).bind(now).execute(&mut *tx).await?;
        if claimed.rows_affected() != 1 {
            return Err(ApiError::conflict(
                "schedule_changed",
                "This occurrence is no longer due",
            ));
        }
    }
    let active_count:i64=sqlx::query_scalar("SELECT count(*) FROM assistant_runs WHERE user_id=? AND status IN ('queued','running','waiting_approval','waiting_input','paused')")
        .bind(owner).fetch_one(&mut *tx).await?;
    if active_count >= 8 {
        return Err(ApiError::too_many_requests(
            "run_limit",
            "Finish or cancel an existing run before starting another",
        ));
    }
    let uncertain:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM assistant_actions a JOIN assistant_runs r ON r.id=a.run_id WHERE r.task_id=? AND a.status='uncertain')")
        .bind(task_id).fetch_one(&mut *tx).await?;
    if uncertain && (!request.acknowledge_uncertain || scheduled) {
        return Err(ApiError::conflict(
            "uncertain_action",
            "Review the interrupted action before explicitly starting a new run; it may already have changed something",
        ));
    }
    let running:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM assistant_runs WHERE task_id=? AND status IN ('queued','running','waiting_approval','waiting_input','paused'))")
        .bind(task_id).fetch_one(&mut *tx).await?;
    if running {
        return Err(ApiError::conflict(
            "run_active",
            "This task already has an active run",
        ));
    }
    let id = request
        .request_id
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let now = unix_timestamp();
    let instructions = if request.message.trim().is_empty() {
        task.instructions.clone()
    } else {
        let previous:Option<String>=sqlx::query_scalar("SELECT result_text FROM assistant_runs WHERE task_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1").bind(task_id).fetch_optional(&mut *tx).await?;
        format!(
            "Original task:\n{}\n\nPrevious result (quoted context, not new authority):\n{}\n\nUser follow-up:\n{}",
            task.instructions,
            previous
                .unwrap_or_default()
                .chars()
                .take(12_000)
                .collect::<String>(),
            request.message.trim()
        )
    };
    sqlx::query("INSERT INTO assistant_runs(id,task_id,user_id,status,instructions,config_json,checkpoint_json,scheduled,created_at,updated_at,request_message) VALUES(?,?,?,'queued',?,?,?,?,?,?,?)")
        .bind(&id).bind(task_id).bind(owner).bind(&instructions).bind(encode(&task.config)?)
        .bind(encode(&Checkpoint::default())?).bind(scheduled).bind(now).bind(now).bind(&request.message).execute(&mut *tx).await?;
    sqlx::query("UPDATE assistant_tasks SET last_run_id=?,viewed_at=0,updated_at=? WHERE id=? AND user_id=?")
        .bind(&id).bind(now).bind(task_id).bind(owner).execute(&mut *tx).await?;
    tx.commit().await?;
    append(
        pool,
        &id,
        "user",
        &instructions,
        json!({"scheduled":scheduled}),
    )
    .await?;
    get_run(pool, owner, &id).await
}

pub async fn append(
    pool: &SqlitePool,
    run: &str,
    kind: &str,
    text: &str,
    data: Value,
) -> Result<(), ApiError> {
    sqlx::query(
        "INSERT INTO assistant_events(run_id,kind,text,data_json,created_at) VALUES(?,?,?,?,?)",
    )
    .bind(run)
    .bind(kind)
    .bind(text)
    .bind(encode(&data)?)
    .bind(unix_timestamp())
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn events(
    pool: &SqlitePool,
    owner: &str,
    run: &str,
    after: i64,
) -> Result<Vec<Event>, ApiError> {
    get_run(pool, owner, run).await?;
    Ok(sqlx::query_as::<_, Event>(
        "SELECT * FROM assistant_events WHERE run_id=? AND id>? ORDER BY id LIMIT 500",
    )
    .bind(run)
    .bind(after.max(0))
    .fetch_all(pool)
    .await?)
}

pub async fn actions(pool: &SqlitePool, owner: &str, run: &str) -> Result<Vec<Action>, ApiError> {
    get_run(pool, owner, run).await?;
    Ok(sqlx::query_as::<_, Action>(
        "SELECT * FROM assistant_actions WHERE run_id=? ORDER BY ordinal",
    )
    .bind(run)
    .fetch_all(pool)
    .await?)
}

pub async fn artifacts(
    pool: &SqlitePool,
    owner: &str,
    run: &str,
) -> Result<Vec<Artifact>, ApiError> {
    get_run(pool, owner, run).await?;
    Ok(sqlx::query_as::<_, Artifact>(
        "SELECT * FROM assistant_artifacts WHERE run_id=? ORDER BY created_at,id",
    )
    .bind(run)
    .fetch_all(pool)
    .await?)
}

pub async fn decide(
    pool: &SqlitePool,
    owner: &str,
    run_id: &str,
    action: &str,
    approve: bool,
) -> Result<(), ApiError> {
    let run = get_run(pool, owner, run_id).await?;
    if !matches!(run.status.as_str(), "waiting_approval" | "paused") {
        return Err(ApiError::conflict(
            "not_waiting",
            "This run is not waiting for an approval",
        ));
    }
    let row = sqlx::query("SELECT * FROM assistant_actions WHERE id=? AND run_id=?")
        .bind(action)
        .bind(run_id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ApiError::not_found("action_not_found", "Action not found"))?;
    let expires: i64 = row.try_get("expires_at")?;
    if approve && expires < unix_timestamp() {
        return Err(ApiError::conflict(
            "approval_expired",
            "This request expired. Reject it and let the assistant request a fresh action.",
        ));
    }
    let config: String = sqlx::query_scalar("SELECT config_json FROM assistant_runs WHERE id=?")
        .bind(run_id)
        .fetch_one(pool)
        .await?;
    if approve {
        authorize_model(pool, owner, &decode::<TaskConfig>(&config)?).await?;
    }
    let mut tx = pool.begin().await?;
    let result=sqlx::query("UPDATE assistant_actions SET status=?,decided_at=? WHERE id=? AND run_id=? AND status='pending'")
        .bind(if approve{"approved"}else{"rejected"}).bind(unix_timestamp()).bind(action).bind(run_id).execute(&mut *tx).await?;
    if result.rows_affected() != 1 {
        return Err(ApiError::conflict(
            "already_decided",
            "This request has already been answered",
        ));
    }
    sqlx::query("UPDATE assistant_runs SET status='queued',updated_at=? WHERE id=? AND status='waiting_approval'").bind(unix_timestamp()).bind(run_id).execute(&mut *tx).await?;
    tx.commit().await?;
    append(
        pool,
        run_id,
        "decision",
        if approve {
            "Approved action"
        } else {
            "Rejected action"
        },
        json!({"action_id":action,"approved":approve}),
    )
    .await
}

pub async fn control(
    pool: &SqlitePool,
    owner: &str,
    id: &str,
    command: &str,
) -> Result<(), ApiError> {
    let run = get_run(pool, owner, id).await?;
    match command {
        "pause" | "cancel" if active(&run.status) => {
            let status = if command == "pause" {
                "paused"
            } else {
                "cancelled"
            };
            let mut tx = pool.begin().await?;
            sqlx::query("UPDATE assistant_runs SET status=?,updated_at=?,finished_at=? WHERE id=? AND user_id=? AND status IN ('queued','running','waiting_approval','waiting_input','paused')")
                .bind(status).bind(unix_timestamp()).bind(if command=="cancel"{Some(unix_timestamp())}else{None}).bind(id).bind(owner).execute(&mut *tx).await?;
            if command == "cancel" {
                sqlx::query("UPDATE assistant_actions SET status='rejected',decided_at=? WHERE run_id=? AND status IN ('ready','pending','approved')")
                    .bind(unix_timestamp()).bind(id).execute(&mut *tx).await?;
            }
            tx.commit().await?;
            append(pool, id, "status", status, json!({})).await?;
        }
        "resume" if matches!(run.status.as_str(), "paused" | "interrupted") => {
            let uncertain:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM assistant_actions WHERE run_id=? AND status IN ('executing','uncertain'))").bind(id).fetch_one(pool).await?;
            if uncertain {
                return Err(ApiError::conflict(
                    "uncertain_action",
                    "An interrupted action may already have run. Review it and start a new run instead.",
                ));
            }
            sqlx::query("UPDATE assistant_runs SET status='queued',error_text='',finished_at=NULL,updated_at=? WHERE id=? AND user_id=? AND status IN ('paused','interrupted')")
                .bind(unix_timestamp()).bind(id).bind(owner).execute(pool).await?;
        }
        _ => {
            return Err(ApiError::conflict(
                "invalid_run_transition",
                "This action is not available for the current run state",
            ));
        }
    }
    Ok(())
}

pub async fn reply(pool: &SqlitePool, owner: &str, id: &str, text: &str) -> Result<(), ApiError> {
    let run = get_run(pool, owner, id).await?;
    if run.status != "waiting_input" || text.trim().is_empty() || text.len() > 16_000 {
        return Err(ApiError::bad_request(
            "invalid_reply",
            "This run needs a nonempty reply of at most 16000 bytes",
        ));
    }
    let mut tx = pool.begin().await?;
    let updated=sqlx::query("UPDATE assistant_actions SET status='completed',result_text=?,finished_at=? WHERE run_id=? AND tool_name='ask_user' AND status='pending'")
        .bind(json!({"answer":text.trim()}).to_string()).bind(unix_timestamp()).bind(id).execute(&mut *tx).await?;
    if updated.rows_affected() != 1 {
        return Err(ApiError::conflict(
            "already_answered",
            "This question has already been answered",
        ));
    }
    sqlx::query("UPDATE assistant_runs SET status='queued',updated_at=? WHERE id=? AND status='waiting_input'")
        .bind(unix_timestamp()).bind(id).execute(&mut *tx).await?;
    tx.commit().await?;
    append(pool, id, "user", text.trim(), json!({"reply":true})).await
}

pub async fn archive(
    pool: &SqlitePool,
    owner: &str,
    id: &str,
    archived: bool,
) -> Result<(), ApiError> {
    get_task(pool, owner, id).await?;
    let active:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM assistant_runs WHERE task_id=? AND status IN ('queued','running','waiting_approval','waiting_input','paused'))").bind(id).fetch_one(pool).await?;
    if active {
        return Err(ApiError::conflict(
            "run_active",
            "Finish or cancel this task's run before archiving",
        ));
    }
    sqlx::query("UPDATE assistant_tasks SET archived_at=?,schedule_enabled=CASE WHEN ? THEN 0 ELSE schedule_enabled END,revision=revision+1,updated_at=? WHERE id=? AND user_id=?")
        .bind(archived.then(unix_timestamp)).bind(archived).bind(unix_timestamp()).bind(id).bind(owner).execute(pool).await?;
    Ok(())
}

pub async fn viewed(pool: &SqlitePool, owner: &str, id: &str) -> Result<(), ApiError> {
    let task = get_task(pool, owner, id).await?;
    sqlx::query("UPDATE assistant_tasks SET viewed_at=? WHERE id=? AND user_id=?")
        .bind(if task.last_status.as_deref().is_some_and(active) {
            0
        } else {
            unix_timestamp()
        })
        .bind(id)
        .bind(owner)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn recover(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query("UPDATE assistant_actions SET status='uncertain' WHERE status='executing'")
        .execute(&mut *tx)
        .await?;
    sqlx::query("UPDATE assistant_runs SET status='interrupted',error_text='The server stopped during this run. Review any uncertain actions before continuing.',updated_at=?,finished_at=? WHERE status='running'")
        .bind(unix_timestamp()).bind(unix_timestamp()).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(())
}
