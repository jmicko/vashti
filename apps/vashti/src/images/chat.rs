//! Image tools return typed attachments independently of assistant prose.
use super::{models::*, service};
use crate::{
    app_state::AppState,
    error::ApiError,
    ollama::models::{OllamaTool, OllamaToolCall, OllamaToolFunction},
    uploads,
};
use serde_json::{Value, json};
use sqlx::Row;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
pub const GENERATE: &str = "generate_image";
pub const EDIT: &str = "edit_image";
pub fn is_image_tool(name: &str) -> bool {
    matches!(name, GENERATE | EDIT)
}
pub async fn schemas(state: &AppState, user: &str) -> Result<Vec<OllamaTool>, ApiError> {
    let ready: bool = sqlx::query_scalar(
        "SELECT backend_id IS NOT NULL AND token!='' FROM image_service_settings WHERE id=1",
    )
    .fetch_one(&state.db)
    .await?;
    if !ready {
        return Ok(vec![]);
    }
    let p = service::preferences(&state.db, user).await?;
    Ok([(GENERATE,&p.generate),(EDIT,&p.edit)].into_iter().map(|(name,profile)|{
        let properties=json!({"prompt":{"type":"string","description":"The complete positive prompt, or instructions describing the edit."},"image_ids":{"type":"array","items":{"type":"string"},"maxItems":10,"description":"For editing only: ordered image IDs supplied in this chat or returned by previous image tools."},"settings":{"type":"object","description":"Optional overrides. Only allowed fields can change; omitted values use the user's defaults.","properties":{"width":{"type":"integer"},"height":{"type":"integer"},"steps":{"type":"integer"},"cfg":{"type":"number"},"seed":{"type":"string"},"negative_prompt":{"type":"string"},"attention":{"type":"string","enum":["standard","fast"]},"flow_shift":{"type":"number"}},"additionalProperties":false}});
        OllamaTool{kind:"function".into(),function:OllamaToolFunction{name:name.into(),description:format!("{}. Returns an image displayed directly to the user and an image ID for further editing. User defaults and override permissions: {}. Never claim to have visually inspected an image unless pixels were supplied to you.",if name==EDIT{"Edit one or more images available in this conversation"}else{"Generate an image from a written prompt"},serde_json::to_string(profile).unwrap_or_default()),parameters:json!({"type":"object","properties":properties,"required":if name==EDIT{vec!["prompt","image_ids"]}else{vec!["prompt"]},"additionalProperties":false})}}
    }).collect())
}
pub struct Outcome {
    pub text: String,
    pub attachment: uploads::models::Attachment,
}
pub struct Context<'a> {
    pub state: &'a AppState,
    pub user: &'a str,
    pub target: ChatTarget,
    pub call: &'a OllamaToolCall,
    pub cancel: &'a CancellationToken,
    pub previews: &'a tokio::sync::watch::Sender<Option<kuno_image::preview::Preview>>,
}
pub async fn execute(context: Context<'_>) -> Result<Outcome, ApiError> {
    let Context {
        state,
        user,
        target,
        call,
        cancel,
        previews,
    } = context;
    let args = &call.function.arguments;
    let prompt = args
        .get("prompt")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            ApiError::bad_request("image_prompt_required", "An image prompt is required")
        })?;
    let edit = call.function.name == EDIT;
    let preferences = service::preferences(&state.db, user).await?;
    let profile = if edit {
        &preferences.edit
    } else {
        &preferences.generate
    };
    let overrides = args.get("settings").cloned().unwrap_or_else(|| json!({}));
    let (recipe, ignored) = service::resolve_tool_recipe(profile, prompt.into(), &overrides)?;
    let image_ids: Vec<String> =
        serde_json::from_value(args.get("image_ids").cloned().unwrap_or_else(|| json!([])))
            .map_err(|_| {
                ApiError::bad_request("image_ids_invalid", "Use image IDs from this conversation")
            })?;
    if (edit && image_ids.is_empty()) || (!edit && !image_ids.is_empty()) || image_ids.len() > 10 {
        return Err(ApiError::bad_request(
            "image_ids_invalid",
            "Editing requires one to ten image IDs; generation takes no input images",
        ));
    }
    let mut input_ids = Vec::new();
    for image_id in image_ids {
        // Only attachments on this assistant's actual ancestry are tool-visible.
        let row=sqlx::query("WITH RECURSIVE branch AS (SELECT id,parent_message_id,active_revision_id FROM chat_messages WHERE id=? AND chat_id=? UNION ALL SELECT m.id,m.parent_message_id,m.active_revision_id FROM chat_messages m JOIN branch b ON b.parent_message_id=m.id WHERE m.chat_id=?) SELECT a.image_asset_id FROM attachments a JOIN branch b ON b.id=a.message_id AND b.active_revision_id=a.revision_id WHERE a.id=? AND a.user_id=? AND a.chat_id=? AND a.attachment_kind='image' LIMIT 1")
            .bind(&target.message_id).bind(&target.chat_id).bind(&target.chat_id).bind(&image_id).bind(user).bind(&target.chat_id).fetch_optional(&state.db).await?
            .ok_or_else(||ApiError::forbidden("image_not_in_chat","Choose an image ID available in this conversation branch"))?;
        if let Some(asset) = row.try_get::<Option<String>, _>("image_asset_id")? {
            input_ids.push(asset);
        } else {
            let (_, bytes) = uploads::service::get_attachment_file(
                &state.db,
                &state.config.uploads_dir(),
                user,
                &image_id,
            )
            .await?;
            let asset = service::upload(&state.db, user, bytes).await?;
            input_ids.push(
                asset["id"]
                    .as_str()
                    .ok_or_else(|| ApiError::internal("Image import failed"))?
                    .to_string(),
            );
        }
    }
    let explicit_seed =
        profile.model_may_change.contains("seed") && overrides.get("seed").is_some();
    let draft = Draft {
        recipe,
        input_ids: input_ids.clone(),
        original_input_ids: input_ids,
        parent_result_id: None,
        prompt_version_id: None,
        seed_mode: if explicit_seed {
            SeedMode::Fixed
        } else {
            profile.seed_mode
        },
        count: Some(1),
        auto_improve: false,
        prompt_help: None,
    };
    let session = service::create_session(
        &state.db,
        user,
        "Chat image",
        if edit { "original" } else { "generate" },
        draft.clone(),
    )
    .await?;
    sqlx::query("UPDATE image_sessions SET chat_id=? WHERE id=? AND user_id=?")
        .bind(&target.chat_id)
        .bind(&session.id)
        .bind(user)
        .execute(&state.db)
        .await?;
    let run = service::submit(
        state,
        user,
        &session.id,
        RunRequest {
            chat_target: Some(target.clone()),
            client_request_id: Uuid::new_v4().to_string(),
            draft,
            improve_only: false,
            retry_result_id: None,
            fixed_endless_confirmed: false,
        },
    )
    .await?;
    let mut last_step = None;
    loop {
        if let Some(preview) = service::preview(state, &run.id)
            && last_step != Some(preview.step)
        {
            last_step = Some(preview.step);
            previews.send_replace(Some(preview));
        }
        let current = service::get_run(&state.db, user, &run.id).await?;
        if ["complete", "stopped", "failed", "interrupted"].contains(&current.status.as_str()) {
            let results = service::results(&state.db, user, &session.id, 0).await?;
            if let Some(result) = results.first() {
                let attachment_id:String=sqlx::query_scalar("SELECT id FROM attachments WHERE image_asset_id=? AND user_id=? AND message_id=? AND revision_id=? LIMIT 1")
                    .bind(&result.asset_id).bind(user).bind(&target.message_id).bind(&target.revision_id).fetch_one(&state.db).await?;
                let attachment =
                    uploads::service::get_attachment(&state.db, user, &attachment_id).await?;
                return Ok(Outcome{text:json!({"image_id":attachment.id,"status":"complete","settings":result.recipe,"ignored_locked_overrides":ignored,"message":"The image is displayed to the user. This image ID can be used for another edit."}).to_string(),attachment});
            }
            return Err(ApiError::conflict(
                "image_tool_failed",
                current
                    .error
                    .unwrap_or_else(|| "Image generation stopped".into()),
            ));
        }
        tokio::select! {_ = cancel.cancelled()=>{service::stop(state,user,&run.id).await?;},_ = tokio::time::sleep(std::time::Duration::from_millis(100))=>{}}
        if cancel.is_cancelled() {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}
