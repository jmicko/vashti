use super::{models::*, service};
use crate::{
    error::ApiError,
    ollama::models::{OllamaTool, OllamaToolCall, OllamaToolFunction},
};
use serde_json::{Value, json};
use sqlx::SqlitePool;

pub const NAMES: [&str; 5] = [
    "list_canvases",
    "read_canvas",
    "create_canvas",
    "update_canvas",
    "comment_canvas",
];
pub fn schemas() -> Vec<OllamaTool> {
    NAMES.iter().filter_map(|name| schema(name)).collect()
}
pub fn schema(name: &str) -> Option<OllamaTool> {
    let (description, parameters) = match name {
        "list_canvases" => (
            "List the editable documents in this chat branch. Canvas is separate from Notes.",
            json!({"type":"object","properties":{},"additionalProperties":false}),
        ),
        "read_canvas" => (
            "Read a canvas before editing it. Returns its exact version_id and a bounded content chunk. offset counts Unicode characters; use next_offset to read more of a long document.",
            json!({"type":"object","properties":{"document_id":{"type":"string"},"offset":{"type":"integer","minimum":0}},"required":["document_id"],"additionalProperties":false}),
        ),
        "create_canvas" => (
            "Create an editable writing draft or code file in this branch. Use markdown for prose, or text/javascript/typescript/python/rust/html/css/json/sql. Content is saved, not executed.",
            json!({"type":"object","properties":{"title":{"type":"string"},"language":{"type":"string","enum":LANGUAGES},"content":{"type":"string"}},"required":["title","language","content"],"additionalProperties":false}),
        ),
        "update_canvas" => (
            "Edit a canvas using the exact version_id from read_canvas. Prefer edits with unique literal find and replace strings; include sufficient context to match exactly once. For a requested complete rewrite, provide content instead. Conflicts must be re-read, never guessed away.",
            json!({"type":"object","properties":{"document_id":{"type":"string"},"expected_version_id":{"type":"string"},"title":{"type":"string"},"language":{"type":"string","enum":LANGUAGES},"content":{"type":"string"},"edits":{"type":"array","maxItems":24,"items":{"type":"object","properties":{"find":{"type":"string"},"replace":{"type":"string"}},"required":["find","replace"],"additionalProperties":false}}},"required":["document_id","expected_version_id"],"additionalProperties":false}),
        ),
        "comment_canvas" => (
            "Leave a review comment on a canvas version. quote must be a literal passage from that version, or an empty string for a general comment. This does not edit the document.",
            json!({"type":"object","properties":{"document_id":{"type":"string"},"expected_version_id":{"type":"string"},"quote":{"type":"string"},"content":{"type":"string"}},"required":["document_id","expected_version_id","quote","content"],"additionalProperties":false}),
        ),
        _ => return None,
    };
    Some(OllamaTool {
        kind: "function".into(),
        function: OllamaToolFunction {
            name: name.into(),
            description: description.into(),
            parameters,
        },
    })
}

pub fn summary(version: &Version) -> Value {
    json!({"document_id":version.document_id,"version_id":version.id,"version_number":version.version_number,"title":version.title,"language":version.language})
}
pub fn read_result(version: &Version, args: &Value) -> Value {
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(MAX_CONTENT as u64) as usize;
    let total = version.content.chars().count();
    let mut content = String::new();
    let mut next = offset.min(total);
    let mut encoded = 0;
    for ch in version.content.chars().skip(offset) {
        let quoted = serde_json::to_string(&ch.to_string())
            .unwrap()
            .chars()
            .count()
            - 2;
        if content.len() + ch.len_utf8() > 12_000 || encoded + quoted > 12_000 {
            break;
        }
        content.push(ch);
        encoded += quoted;
        next += 1;
    }
    json!({"canvas":summary(version),"offset":offset,"next_offset":next,"total_characters":total,"content":content,"truncated":next<total})
}

pub async fn execute(
    pool: &SqlitePool,
    owner: &str,
    chat: &str,
    message: &str,
    revision: &str,
    model: &str,
    call: &OllamaToolCall,
) -> String {
    match execute_inner(pool, owner, chat, message, revision, model, call).await {
        Ok(result) => result.to_string(),
        Err(error) => json!({"error":error.message(),"code":error.code()}).to_string(),
    }
}
#[allow(clippy::too_many_arguments)]
async fn execute_inner(
    pool: &SqlitePool,
    owner: &str,
    chat: &str,
    message: &str,
    revision: &str,
    model: &str,
    call: &OllamaToolCall,
) -> Result<Value, ApiError> {
    if !crate::settings::service::tool_is_available_for_user(pool, owner, "canvas").await? {
        return Err(ApiError::forbidden(
            "canvas_disabled",
            "Canvas tools are unavailable",
        ));
    }
    let chat_record = crate::chats::service::get_chat(pool, owner, chat).await?;
    if !chat_record.tool_preferences.tool_use_enabled
        || !chat_record.tool_preferences.tool_enabled("canvas")
    {
        return Err(ApiError::forbidden(
            "canvas_disabled",
            "Canvas tools are disabled for this chat",
        ));
    }
    let active: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM chat_messages WHERE id=? AND chat_id=? AND active_revision_id=? AND status='streaming' AND is_deleted=0)").bind(message).bind(chat).bind(revision).fetch_one(pool).await?;
    if !active {
        return Err(service::conflict());
    }
    let scope = Scope {
        message_id: Some(message.into()),
        revision_id: Some(revision.into()),
    };
    let workspace = service::workspace(pool, owner, chat, &scope).await?;
    let args = &call.function.arguments;
    if call.function.name == "list_canvases" {
        return Ok(
            json!({"canvases":workspace.documents.iter().filter(|d| !d.trashed).map(|d| summary(&d.version)).collect::<Vec<_>>()}),
        );
    }
    if call.function.name == "read_canvas" {
        let id = args
            .get("document_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let doc = workspace
            .documents
            .iter()
            .find(|d| d.id == id && !d.trashed)
            .ok_or_else(|| {
                ApiError::not_found("canvas_missing", "Canvas not found in this branch")
            })?;
        let mut result = read_result(&doc.version, args);
        let comments = workspace
            .comments
            .iter()
            .filter(|c| c.document_id == doc.id && c.version_id == doc.version.id && !c.resolved)
            .collect::<Vec<_>>();
        result["comments"]=json!(comments.iter().take(3).map(|c| json!({"id":c.id,"quote":c.quote.chars().take(200).collect::<String>(),"content":c.content.chars().take(400).collect::<String>()})).collect::<Vec<_>>());
        result["comments_truncated"] = json!(comments.len() > 3);
        return Ok(result);
    }
    let operation = match call.function.name.as_str() {
        "create_canvas" => "create",
        "update_canvas" => "update",
        "comment_canvas" => "comment",
        _ => {
            return Err(ApiError::bad_request(
                "unknown_canvas_tool",
                "Unknown canvas tool",
            ));
        }
    };
    let mut input = args.as_object().cloned().ok_or_else(|| {
        ApiError::bad_request("invalid_canvas", "Tool arguments must be an object")
    })?;
    if input.contains_key("operation") || input.contains_key("edit_session_id") {
        return Err(ApiError::bad_request(
            "invalid_canvas",
            "Unexpected canvas tool argument",
        ));
    }
    input.insert("operation".into(), json!(operation));
    let mutation: Mutation = serde_json::from_value(Value::Object(input))
        .map_err(|_| ApiError::bad_request("invalid_canvas", "Invalid canvas tool arguments"))?;
    let result = service::mutate(
        pool,
        owner,
        chat,
        &scope,
        mutation,
        Actor {
            model_name: Some(model),
        },
    )
    .await?;
    let document = if operation == "create" {
        result.documents.last()
    } else {
        args.get("document_id")
            .and_then(Value::as_str)
            .and_then(|id| result.documents.iter().find(|d| d.id == id))
    }
    .ok_or_else(|| ApiError::internal("Canvas result unavailable"))?;
    Ok(json!({"canvas":summary(&document.version),"operation":operation,"saved":true}))
}
