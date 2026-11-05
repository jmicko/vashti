use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use super::{models::*, service};
use crate::{
    app_state::AppState, auth::service::unix_timestamp, error::ApiError,
    ollama::models::OllamaChatMessage,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewConversation {
    pub request_id: String,
    pub title: String,
    pub config: TaskConfig,
}

pub async fn create(
    pool: &SqlitePool,
    owner: &str,
    input: NewConversation,
) -> Result<Task, ApiError> {
    Uuid::parse_str(&input.request_id)
        .map_err(|_| ApiError::bad_request("invalid_request_id", "Request IDs must be UUIDs"))?;
    let original = service::encode(&json!({"title":input.title,"config":input.config}))?;
    if let Some(saved) = sqlx::query_scalar::<_, Option<String>>(
        "SELECT create_request_json FROM assistant_tasks WHERE id=?",
    )
    .bind(&input.request_id)
    .fetch_optional(pool)
    .await?
    {
        let task = get(pool, owner, &input.request_id).await?;
        if saved.as_deref() != Some(original.as_str()) {
            return Err(ApiError::conflict(
                "request_changed",
                "This request already created a different conversation",
            ));
        }
        return Ok(task);
    }
    service::authorize_model(pool, owner, &input.config).await?;
    let title = input.title.trim().chars().take(80).collect::<String>();
    let now = unix_timestamp();
    let changed = sqlx::query("INSERT INTO assistant_tasks(id,user_id,kind,title,instructions,config_json,created_at,updated_at,viewed_at,create_request_json) SELECT ?,?,'conversation',?,'Conversation',?,?,?,?,? WHERE (SELECT count(*) FROM assistant_tasks WHERE user_id=? AND kind='conversation' AND archived_at IS NULL)<100")
        .bind(&input.request_id).bind(owner).bind(if title.is_empty() { "Assistant" } else { &title }).bind(service::encode(&input.config)?)
        .bind(now).bind(now).bind(now).bind(original).bind(owner).execute(pool).await?;
    if changed.rows_affected() != 1 {
        return Err(ApiError::too_many_requests(
            "conversation_limit",
            "Archive an older conversation first",
        ));
    }
    get(pool, owner, &input.request_id).await
}

pub async fn get(pool: &SqlitePool, owner: &str, id: &str) -> Result<Task, ApiError> {
    let task = service::get_task(pool, owner, id).await?;
    if task.kind != "conversation" {
        return Err(ApiError::not_found(
            "conversation_not_found",
            "Assistant conversation not found",
        ));
    }
    Ok(task)
}

pub async fn list(pool: &SqlitePool, owner: &str) -> Result<Value, ApiError> {
    list_archived(pool, owner, false).await
}

pub async fn list_archived(
    pool: &SqlitePool,
    owner: &str,
    archived: bool,
) -> Result<Value, ApiError> {
    let rows = sqlx::query("SELECT id,title,updated_at FROM assistant_tasks WHERE user_id=? AND kind='conversation' AND (archived_at IS NOT NULL)=? ORDER BY updated_at DESC,rowid DESC LIMIT 100")
        .bind(owner).bind(archived).fetch_all(pool).await?;
    Ok(
        json!({"conversations":rows.into_iter().map(|r|json!({"id":r.get::<String,_>("id"),"title":r.get::<String,_>("title"),"updated_at":r.get::<i64,_>("updated_at")})).collect::<Vec<_>>() }),
    )
}

pub async fn page(
    pool: &SqlitePool,
    owner: &str,
    id: &str,
    before: Option<&str>,
) -> Result<Value, ApiError> {
    let task = get(pool, owner, id).await?;
    let cursor: i64 = if let Some(before) = before {
        sqlx::query_scalar(
            "SELECT rowid FROM assistant_runs WHERE id=? AND task_id=? AND user_id=?",
        )
        .bind(before)
        .bind(id)
        .bind(owner)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| ApiError::not_found("run_not_found", "Conversation turn not found"))?
    } else {
        i64::MAX
    };
    let mut runs = sqlx::query_as::<_, TurnSummary>("SELECT id,task_id,status,created_at,updated_at,finished_at FROM assistant_runs WHERE user_id=? AND task_id=? AND rowid<? ORDER BY rowid DESC LIMIT 21")
        .bind(owner).bind(id).bind(cursor).fetch_all(pool).await?;
    let more = runs.len() > 20;
    runs.truncate(20);
    runs.reverse();
    // Task summaries stay bounded; full activity and files use the existing owner-checked run API.
    let mut rows=sqlx::query("SELECT t.id,t.title,(t.schedule_enabled AND r.id IS t.last_run_id) AS schedule_enabled,t.next_run_at,t.schedule_error,r.id AS last_run_id,r.status,substr(r.result_text,1,4000) AS result_text,r.error_text,COALESCE(r.created_at,t.created_at) AS created_at FROM assistant_tasks t LEFT JOIN assistant_runs r ON r.task_id=t.id AND r.user_id=t.user_id WHERE t.user_id=? AND t.conversation_id=? ORDER BY COALESCE(r.rowid,0) DESC,t.rowid DESC LIMIT 51")
        .bind(owner).bind(id).fetch_all(pool).await?;
    let more_jobs = rows.len() > 50;
    rows.truncate(50);
    rows.reverse();
    let jobs:Vec<Value> = rows.into_iter().map(|r|json!({"id":r.get::<String,_>("id"),"title":r.get::<String,_>("title"),"schedule_enabled":r.get::<bool,_>("schedule_enabled"),"next_run_at":r.get::<Option<i64>,_>("next_run_at"),"schedule_error":r.get::<Option<String>,_>("schedule_error"),"run_id":r.get::<Option<String>,_>("last_run_id"),"status":r.get::<Option<String>,_>("status"),"result":r.get::<Option<String>,_>("result_text").map(|v|v.chars().take(4000).collect::<String>()),"error":r.get::<Option<String>,_>("error_text"),"created_at":r.get::<i64,_>("created_at")})).collect();
    let requires_review:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM assistant_actions a JOIN assistant_runs r ON r.id=a.run_id WHERE r.user_id=? AND r.task_id=? AND (a.status='uncertain' OR (a.status='executing' AND r.status NOT IN ('running','queued'))))")
        .bind(owner).bind(id).fetch_one(pool).await?;
    Ok(
        json!({"task":task,"runs":runs,"more":more,"jobs":jobs,"more_jobs":more_jobs,"requires_review":requires_review}),
    )
}

pub async fn history(
    pool: &SqlitePool,
    owner: &str,
    task: &str,
    run: &str,
    budget: i64,
) -> Result<Vec<OllamaChatMessage>, ApiError> {
    // Reload actual turns, not a copied checkpoint. Tool bodies and private reasoning
    // are not carried into new turns; protected records must be re-read with current permissions.
    let rows=sqlx::query("SELECT e.kind,e.text FROM assistant_events e JOIN assistant_runs r ON r.id=e.run_id WHERE r.task_id=? AND r.user_id=? AND r.rowid<(SELECT rowid FROM assistant_runs WHERE id=?) AND e.kind IN ('user','assistant') AND e.text<>'' ORDER BY e.id DESC LIMIT 160")
        .bind(task).bind(owner).bind(run).fetch_all(pool).await?;
    let mut bytes = 0;
    let mut weight = 0;
    let mut messages = Vec::new();
    for row in rows {
        let text: String = row.try_get("text")?;
        weight += super::runner::context_weight(&text) + 32;
        if bytes + text.len() > 24_000 || weight > budget {
            break;
        }
        bytes += text.len();
        messages.push(OllamaChatMessage {
            role: if row.get::<String, _>("kind") == "user" {
                "user"
            } else {
                "assistant"
            }
            .into(),
            content: text,
            thinking: None,
            images: None,
            tool_name: None,
            tool_calls: None,
        });
    }
    messages.reverse();
    Ok(messages)
}

pub fn is_tool(name: &str) -> bool {
    matches!(
        name,
        "create_task" | "update_task" | "list_tasks" | "read_task" | "cancel_task"
    )
}

pub async fn coordinator(pool: &SqlitePool, run: &str) -> Result<Option<String>, ApiError> {
    Ok(sqlx::query_scalar("SELECT t.id FROM assistant_tasks t JOIN assistant_runs r ON r.task_id=t.id WHERE r.id=? AND t.kind='conversation'")
        .bind(run).fetch_optional(pool).await?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskArguments {
    title: String,
    instructions: String,
    next_run_at: Option<i64>,
    interval_seconds: Option<i64>,
}

pub async fn dispatch(
    state: &AppState,
    owner: &str,
    run: &str,
    config: &TaskConfig,
    name: &str,
    args: &Value,
    action: &str,
) -> Result<String, ApiError> {
    let conversation = coordinator(&state.db, run).await?.ok_or_else(|| {
        ApiError::forbidden(
            "delegation_not_allowed",
            "Background tasks cannot create or manage other tasks",
        )
    })?;
    get(&state.db, owner, &conversation).await?;
    if name == "list_tasks" {
        let rows=sqlx::query("SELECT t.id,t.title,t.schedule_enabled,t.next_run_at,r.status,substr(r.result_text,1,4000) AS result FROM assistant_tasks t LEFT JOIN assistant_runs r ON r.id=t.last_run_id WHERE t.user_id=? AND t.conversation_id=? ORDER BY t.updated_at DESC,t.rowid DESC LIMIT 50").bind(owner).bind(&conversation).fetch_all(&state.db).await?;
        let summaries:Vec<_>=rows.into_iter().map(|r|json!({"task_id":r.get::<String,_>("id"),"title":r.get::<String,_>("title"),"schedule_enabled":r.get::<bool,_>("schedule_enabled"),"next_run_at":r.get::<Option<i64>,_>("next_run_at"),"status":r.get::<Option<String>,_>("status"),"result":r.get::<Option<String>,_>("result")})).collect();
        return Ok(json!({"tasks":summaries}).to_string());
    }
    if name == "create_task" {
        let input: TaskArguments = serde_json::from_value(args.clone()).map_err(|_| {
            ApiError::bad_request(
                "invalid_task",
                "Provide title, instructions, and optional next_run_at/interval_seconds only",
            )
        })?;
        if input.interval_seconds.is_some() && input.next_run_at.is_none() {
            return Err(ApiError::bad_request(
                "invalid_schedule",
                "A recurring task needs its first run time",
            ));
        }
        let count: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM assistant_tasks WHERE user_id=? AND conversation_id=?",
        )
        .bind(owner)
        .bind(&conversation)
        .fetch_one(&state.db)
        .await?;
        if count >= 50 {
            return Err(ApiError::too_many_requests(
                "task_limit",
                "This conversation has reached its 50-task limit",
            ));
        }
        let task = service::save_delegated_task(
            &state.db,
            owner,
            &conversation,
            SaveTask {
                request_id: Some(action.into()),
                title: input.title,
                instructions: input.instructions,
                config: config.clone(),
                interval_seconds: input.interval_seconds,
                next_run_at: input.next_run_at,
                schedule_enabled: input.next_run_at.is_some(),
                expected_revision: None,
            },
        )
        .await?;
        let started = if task.schedule_enabled {
            Ok(None)
        } else {
            service::start_run(
                &state.db,
                owner,
                &task.id,
                StartRun {
                    request_id: Some(action.into()),
                    ..StartRun::default()
                },
                false,
            )
            .await
            .map(Some)
        };
        state.assistant_jobs.wake.notify_one();
        return Ok(json!({"task_id":task.id,"title":task.title,"run_id":started.as_ref().ok().and_then(|r|r.as_ref()).map(|r|&r.id),"start_error":started.err().map(|e|e.message().to_owned()),"scheduled":task.schedule_enabled,"next_run_at":task.next_run_at,"url":format!("/app/assistant/tasks/{}",task.id)}).to_string());
    }
    let id = args
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let task = service::get_task(&state.db, owner, id).await?;
    if task.conversation_id.as_deref() != Some(&conversation) {
        return Err(ApiError::not_found(
            "task_not_found",
            "No task with that ID belongs to this conversation",
        ));
    }
    match name {
        "read_task" => Ok(json!({"task":task,"runs":service::list_runs(&state.db,owner,id).await?.into_iter().take(5).collect::<Vec<_>>()} ).to_string()),
        "cancel_task" => {
            sqlx::query("UPDATE assistant_tasks SET schedule_enabled=0,revision=revision+1,updated_at=? WHERE id=? AND user_id=?").bind(unix_timestamp()).bind(id).bind(owner).execute(&state.db).await?;
            for run in service::list_runs(&state.db,owner,id).await?.into_iter().filter(|r|active(&r.status)) {
                service::control(&state.db,owner,&run.id,"cancel").await?;
                state.assistant_jobs.cancel(&run.id).await;
            }
            Ok(json!({"task_id":id,"stopped":true,"schedule_enabled":false}).to_string())
        }
        "update_task" => {
            let revision=args.get("expected_revision").and_then(Value::as_i64).ok_or_else(||ApiError::bad_request("revision_required", "Read the task first and provide its expected_revision"))?;
            let allowed=["task_id","expected_revision","title","instructions","next_run_at","interval_seconds","schedule_enabled"];
            if args.as_object().is_none_or(|a|a.keys().any(|k|!allowed.contains(&k.as_str()))) { return Err(ApiError::bad_request("invalid_task", "Task capabilities cannot be changed by a model")); }
            let mut inherited=task.config;
            inherited.notes &= config.notes;
            inherited.memories &= config.memories;
            inherited.chat_history &= config.chat_history;
            inherited.sandbox &= config.sandbox;
            inherited.max_steps=inherited.max_steps.min(config.max_steps);
            inherited.max_output_tokens=inherited.max_output_tokens.min(config.max_output_tokens);
            for (old,current) in [(&mut inherited.writes,config.writes),(&mut inherited.web,config.web)] {
                *old=match (*old,current) { (ActionPolicy::Block,_)|(_,ActionPolicy::Block)=>ActionPolicy::Block,(ActionPolicy::Ask,_)|(_,ActionPolicy::Ask)=>ActionPolicy::Ask,_=>ActionPolicy::Allow };
            }
            let mut next=json!({"title":task.title,"instructions":task.instructions,"config":inherited,"next_run_at":task.next_run_at,"interval_seconds":task.interval_seconds,"schedule_enabled":task.schedule_enabled,"expected_revision":revision});
            for key in ["title","instructions","next_run_at","interval_seconds","schedule_enabled"] { if let Some(value)=args.get(key) { next[key]=value.clone(); } }
            let payload:SaveTask=serde_json::from_value(next).map_err(|_|ApiError::bad_request("invalid_task", "Invalid task settings"))?;
            let task=service::save_task(&state.db,owner,Some(id),payload).await?;
            state.assistant_jobs.wake.notify_one();
            Ok(json!({"task":task,"notice":"Changes apply to future runs, not an already running occurrence."}).to_string())
        }
        _=>Err(ApiError::bad_request("unknown_tool", "Unknown task tool")),
    }
}
