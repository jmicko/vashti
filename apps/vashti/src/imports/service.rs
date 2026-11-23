use crate::{auth::service::unix_timestamp, backends, error::ApiError};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{Row, SqlitePool};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use uuid::Uuid;

pub const MAX_ASSET_BYTES: u64 = 128 * 1024 * 1024;
const MAX_IMPORT_BYTES: i64 = 2 * 1024 * 1024 * 1024;
const MAX_CHUNK_BYTES: usize = 512 * 1024;
const PAGE_PREFIX: &str = "chatgpt-page-";
const FILE_LINK_PREFIX: &str = "/.vashti/imported-file/";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conversation {
    pub source_id: String,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub archived: bool,
    pub do_not_remember: bool,
    pub current_node_id: Option<String>,
    pub messages: Vec<Message>,
    pub assets: Vec<Asset>,
    pub original_json: String,
    #[serde(default)]
    pub workspace: Option<Workspace>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workspace {
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub markdown: Option<String>,
}

impl Workspace {
    fn display(&self) -> (&str, &str) {
        self.markdown.as_deref().map_or(("text", self.content.as_str()), |markdown| ("markdown", markdown))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub source_id: String,
    pub parent_id: Option<String>,
    pub role: String,
    pub content: String,
    pub thinking: String,
    pub model: Option<String>,
    pub created_at: i64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub source_id: String,
    pub message_id: String,
    pub filename: String,
    pub size_bytes: i64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportRequest {
    pub expected_user_id: Option<String>,
    pub backend_id: String,
    pub model_name: String,
    pub allow_history_search: bool,
    pub learn_from_import: Option<bool>,
    pub conversation: Conversation,
}
#[derive(Serialize)]
pub struct ImportResult {
    pub chat_id: String,
    pub existing: bool,
    pub changed_source: bool,
    pub assets: Vec<AssetStatus>,
}
#[derive(Serialize)]
pub struct AssetStatus {
    pub id: String,
    pub source_id: String,
    pub uploaded_bytes: i64,
    pub complete: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    pub expected_user_id: Option<String>,
    pub offset: i64,
    pub data: String,
    pub sha256: Option<String>,
}

fn invalid() -> ApiError {
    ApiError::bad_request(
        "invalid_import",
        "This conversation has an invalid or unsupported structure",
    )
}
fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}
fn valid_time(value: i64) -> bool {
    (0..=32_503_680_000).contains(&value)
}
fn filename(value: &str) -> String {
    let base = value.rsplit(['/', '\\']).next().unwrap_or("attachment");
    let name: String = base
        .chars()
        .map(|c| if c.is_control() { '_' } else { c })
        .take(160)
        .collect();
    if name.trim_matches(['.', ' ']).is_empty() {
        "attachment".into()
    } else {
        name
    }
}

fn validate(conversation: &Conversation) -> Result<(), ApiError> {
    if !valid_id(&conversation.source_id)
        || conversation.title.len() > 4096
        || !valid_time(conversation.created_at)
        || !valid_time(conversation.updated_at)
        || conversation.messages.is_empty()
        || conversation.messages.len() > 10_000
        || conversation.assets.len() > 5000
        || conversation.original_json.len() > 16 * 1024 * 1024
    {
        return Err(invalid());
    }
    let original: serde_json::Value =
        serde_json::from_str(&conversation.original_json).map_err(|_| invalid())?;
    if original
        .get("conversation_id")
        .or_else(|| original.get("id"))
        .and_then(|v| v.as_str())
        != Some(conversation.source_id.as_str())
    {
        return Err(invalid());
    }
    if let Some(workspace) = &conversation.workspace {
        let valid_source = conversation
            .source_id
            .strip_prefix(PAGE_PREFIX)
            .is_some_and(|id| id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()));
        if !valid_source
            || !conversation.archived
            || !conversation.do_not_remember
            || !conversation.assets.is_empty()
            || conversation.messages.len() != 1
            || workspace.content.trim().is_empty()
            || original
                .pointer("/page/projection/materialized_search_text")
                .and_then(|v| v.as_str())
                != Some(workspace.content.as_str())
        {
            return Err(invalid());
        }
        if let Some(markdown) = &workspace.markdown {
            let head = original.pointer("/page/head").ok_or_else(invalid)?;
            let checkpoint = original.pointer("/checkpoint/checkpoint").ok_or_else(invalid)?;
            let document = original.pointer("/checkpoint/document").ok_or_else(invalid)?;
            let sequence = head["current_sequence"].as_u64().ok_or_else(invalid)?;
            if markdown.trim().is_empty()
                || document["format"] != "openai.page.yjs"
                || document["schema_version"] != 2
                || document["page_id"] != head["page_id"]
                || checkpoint["page_id"] != head["page_id"]
                || checkpoint["checkpoint_id"] != head["checkpoint_id"]
                || checkpoint["stream_kind"] != "content"
                || checkpoint["through_sequence"] != sequence
                || document["through_sequence"] != sequence
                || document["yjs_state_update"].as_str().is_none_or(str::is_empty)
            {
                return Err(invalid());
            }
        }
        let (language, body) = workspace.display();
        crate::canvas::service::validate_document(&workspace.title, language, body)?;
    } else if conversation.source_id.starts_with(PAGE_PREFIX) {
        return Err(invalid());
    }
    let mut known = HashSet::new();
    for message in &conversation.messages {
        if !valid_id(&message.source_id)
            || !known.insert(message.source_id.as_str())
            || !matches!(message.role.as_str(), "user" | "assistant")
            || message.content.len() + message.thinking.len() > 2 * 1024 * 1024
            || message
                .model
                .as_ref()
                .is_some_and(|model| model.len() > 256 || model.chars().any(char::is_control))
            || !valid_time(message.created_at)
        {
            return Err(invalid());
        }
        // Parent-first ordering also proves that the graph is acyclic.
        if message
            .parent_id
            .as_ref()
            .is_some_and(|parent| parent == &message.source_id || !known.contains(parent.as_str()))
        {
            return Err(invalid());
        }
    }
    if conversation
        .current_node_id
        .as_ref()
        .is_some_and(|id| !known.contains(id.as_str()))
    {
        return Err(invalid());
    }
    let mut asset_ids = HashSet::new();
    for asset in &conversation.assets {
        if !valid_id(&asset.source_id)
            || !asset_ids.insert(&asset.source_id)
            || !known.contains(asset.message_id.as_str())
            || asset.filename.len() > 1024
            || !(1..=MAX_ASSET_BYTES as i64).contains(&asset.size_bytes)
        {
            return Err(invalid());
        }
    }
    Ok(())
}

fn bind_file_links(content: &str, ids: Option<&HashMap<&str, String>>) -> String {
    let Some(ids) = ids else {
        return content.to_owned();
    };
    let mut parts = content.split(FILE_LINK_PREFIX);
    let mut result = String::from(parts.next().unwrap_or_default());
    for part in parts {
        result.push_str(FILE_LINK_PREFIX);
        let replacement = part.get(..70).and_then(|key| {
            let tail = &part[70..];
            if !tail.is_empty() && !tail.starts_with([')', '>', ' ', '\t', '\n', '\r']) {
                return None;
            }
            ids.get(key).map(|id| (id, tail))
        });
        if let Some((id, tail)) = replacement {
            result.push_str(id);
            result.push_str(tail);
        } else {
            result.push_str(part);
        }
    }
    result
}

#[derive(Serialize, sqlx::FromRow)]
pub struct ImportedDocument {
    pub id: String,
    pub title: String,
    pub updated_at: i64,
}

pub async fn imported_documents(
    pool: &SqlitePool,
    owner: &str,
) -> Result<Vec<ImportedDocument>, ApiError> {
    Ok(sqlx::query_as("SELECT c.id,c.title,c.updated_at FROM chat_import_records r JOIN chats c ON c.id=r.chat_id WHERE r.user_id=? AND substr(r.source_id,1,13)='chatgpt-page-' ORDER BY c.updated_at DESC,c.id")
        .bind(owner).fetch_all(pool).await?)
}

pub async fn import_conversation(
    pool: &SqlitePool,
    owner: &str,
    input: ImportRequest,
) -> Result<ImportResult, ApiError> {
    validate(&input.conversation)?;
    let c = &input.conversation;
    let hash = format!("{:x}", Sha256::digest(c.original_json.as_bytes()));
    if let Some(row) = sqlx::query(
        "SELECT id,chat_id,source_hash FROM chat_import_records WHERE user_id=? AND source_id=?",
    )
    .bind(owner)
    .bind(&c.source_id)
    .fetch_optional(pool)
    .await?
    {
        let id: String = row.get("id");
        return Ok(ImportResult {
            chat_id: row.get("chat_id"),
            existing: true,
            changed_source: row.get::<String, _>("source_hash") != hash,
            assets: asset_status(pool, owner, &id).await?,
        });
    }
    let enabled: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM ollama_backends WHERE id=? AND is_enabled=1)",
    )
    .bind(&input.backend_id)
    .fetch_one(pool)
    .await?;
    if !enabled || input.model_name.trim().is_empty() || input.model_name.len() > 256 {
        return Err(ApiError::bad_request(
            "invalid_model",
            "Choose an available model for future replies",
        ));
    }
    backends::service::ensure_model_enabled_for_user(
        pool,
        owner,
        &input.backend_id,
        &input.model_name,
    )
    .await?;
    let chat_id = Uuid::new_v4().to_string();
    let import_id = Uuid::new_v4().to_string();
    let ids: HashMap<&str, (String, String)> = c
        .messages
        .iter()
        .map(|m| {
            (
                m.source_id.as_str(),
                (Uuid::new_v4().to_string(), Uuid::new_v4().to_string()),
            )
        })
        .collect();
    let source_messages: HashMap<&str, &Message> = c
        .messages
        .iter()
        .map(|m| (m.source_id.as_str(), m))
        .collect();
    let mut active_path = HashMap::new();
    let mut current = c
        .current_node_id
        .as_deref()
        .or_else(|| c.messages.last().map(|m| m.source_id.as_str()));
    let mut root = None;
    while let Some(id) = current {
        root = Some(ids[id].0.clone());
        let parent = source_messages[id].parent_id.as_deref();
        if let Some(parent) = parent {
            active_path.insert(parent, id);
        }
        current = parent;
    }
    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
    if let Some(row) = sqlx::query(
        "SELECT id,chat_id,source_hash FROM chat_import_records WHERE user_id=? AND source_id=?",
    )
    .bind(owner)
    .bind(&c.source_id)
    .fetch_optional(&mut *tx)
    .await?
    {
        let id: String = row.get("id");
        let chat_id = row.get("chat_id");
        let changed_source = row.get::<String, _>("source_hash") != hash;
        tx.rollback().await?;
        return Ok(ImportResult {
            chat_id,
            existing: true,
            changed_source,
            assets: asset_status(pool, owner, &id).await?,
        });
    }
    let used: i64 = sqlx::query_scalar("SELECT COALESCE(SUM(a.size_bytes),0) FROM chat_import_assets a JOIN chat_import_records r ON r.id=a.import_id WHERE r.user_id=?").bind(owner).fetch_one(&mut *tx).await?;
    if c.assets.iter().map(|a| a.size_bytes).sum::<i64>() > MAX_IMPORT_BYTES - used {
        return Err(ApiError::bad_request(
            "import_quota",
            "Imported attachments exceed the 2 GiB account import limit",
        ));
    }
    sqlx::query("INSERT INTO chats(id,user_id,default_backend_id,default_model_name,title,created_at,updated_at,last_message_at,archived_at) VALUES(?,?,?,?,?,?,?,?,?)")
        .bind(&chat_id).bind(owner).bind(&input.backend_id).bind(&input.model_name).bind(if c.title.trim().is_empty() { "Imported conversation" } else { &c.title })
        .bind(c.created_at).bind(c.updated_at).bind(c.updated_at).bind(c.archived.then_some(unix_timestamp())).execute(&mut *tx).await?;
    let mut asset_ids: HashMap<&str, HashMap<&str, String>> = HashMap::new();
    for asset in &c.assets {
        asset_ids
            .entry(&asset.message_id)
            .or_default()
            .insert(&asset.source_id, Uuid::new_v4().to_string());
    }
    for m in &c.messages {
        let (id, revision) = &ids[m.source_id.as_str()];
        let content = bind_file_links(&m.content, asset_ids.get(m.source_id.as_str()));
        sqlx::query("INSERT INTO chat_messages(id,chat_id,parent_message_id,role,status,model_name,created_at,updated_at,import_source) VALUES(?,?,?,?,'complete',?,?,?,'chatgpt')")
            .bind(id).bind(&chat_id).bind(m.parent_id.as_deref().map(|p| &ids[p].0)).bind(&m.role).bind(&m.model).bind(m.created_at).bind(m.created_at).execute(&mut *tx).await?;
        sqlx::query("INSERT INTO chat_message_revisions(id,message_id,content_text,thinking_text,source,created_at) VALUES(?,?,?,?,'original',?)")
            .bind(revision).bind(id).bind(content).bind(&m.thinking).bind(m.created_at).execute(&mut *tx).await?;
        sqlx::query("UPDATE chat_messages SET active_revision_id=? WHERE id=?")
            .bind(revision)
            .bind(id)
            .execute(&mut *tx)
            .await?;
    }
    let mut first_children = HashMap::new();
    for m in &c.messages {
        if let Some(parent) = m.parent_id.as_deref() {
            first_children.entry(parent).or_insert(m.source_id.as_str());
        }
    }
    for m in &c.messages {
        let child = active_path
            .get(m.source_id.as_str())
            .or_else(|| first_children.get(m.source_id.as_str()))
            .copied();
        if let Some(child) = child {
            sqlx::query("UPDATE chat_messages SET active_child_message_id=? WHERE id=?")
                .bind(&ids[child].0)
                .bind(&ids[m.source_id.as_str()].0)
                .execute(&mut *tx)
                .await?;
        }
    }
    sqlx::query("UPDATE chats SET active_root_message_id=? WHERE id=?")
        .bind(root)
        .bind(&chat_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("INSERT INTO chat_import_records(id,chat_id,user_id,source_id,source_hash,original_json,allow_history_search,created_at,do_not_remember,allow_profile_learning) VALUES(?,?,?,?,?,?,?,?,?,?)")
        .bind(&import_id).bind(&chat_id).bind(owner).bind(&c.source_id).bind(hash).bind(&c.original_json).bind(input.allow_history_search && !c.do_not_remember).bind(unix_timestamp()).bind(c.do_not_remember).bind(input.learn_from_import.unwrap_or(true) && !c.do_not_remember).execute(&mut *tx).await?;
    for asset in &c.assets {
        let (message, revision) = &ids[asset.message_id.as_str()];
        sqlx::query("INSERT INTO chat_import_assets(id,import_id,source_id,message_id,revision_id,filename,size_bytes) VALUES(?,?,?,?,?,?,?)")
            .bind(&asset_ids[asset.message_id.as_str()][asset.source_id.as_str()]).bind(&import_id).bind(&asset.source_id).bind(message).bind(revision).bind(filename(&asset.filename)).bind(asset.size_bytes).execute(&mut *tx).await?;
    }
    if let Some(workspace) = &c.workspace {
        let (language, body) = workspace.display();
        crate::canvas::service::import_document(
            &mut tx,
            owner,
            &chat_id,
            &workspace.title,
            body,
            language,
            c.updated_at,
        )
        .await?;
    }
    tx.commit().await?;
    Ok(ImportResult {
        chat_id,
        existing: false,
        changed_source: false,
        assets: asset_status(pool, owner, &import_id).await?,
    })
}

async fn asset_status(
    pool: &SqlitePool,
    owner: &str,
    id: &str,
) -> Result<Vec<AssetStatus>, ApiError> {
    let rows = sqlx::query("SELECT a.id,a.source_id,a.uploaded_bytes,a.complete FROM chat_import_assets a JOIN chat_import_records r ON r.id=a.import_id WHERE r.id=? AND r.user_id=? ORDER BY a.rowid")
        .bind(id).bind(owner).fetch_all(pool).await?;
    Ok(rows
        .iter()
        .map(|r| AssetStatus {
            id: r.get("id"),
            source_id: r.get("source_id"),
            uploaded_bytes: r.get("uploaded_bytes"),
            complete: r.get("complete"),
        })
        .collect())
}

pub async fn upload_chunk(
    pool: &SqlitePool,
    tmp: &Path,
    uploads: &Path,
    owner: &str,
    id: &str,
    chunk: Chunk,
) -> Result<AssetStatus, ApiError> {
    if chunk.data.len() > MAX_CHUNK_BYTES.div_ceil(3) * 4
        || chunk.offset < 0
        || Uuid::parse_str(id).is_err()
    {
        return Err(invalid());
    }
    let bytes = STANDARD.decode(&chunk.data).map_err(|_| invalid())?;
    if bytes.is_empty() || bytes.len() > MAX_CHUNK_BYTES {
        return Err(invalid());
    }
    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
    let row = sqlx::query("SELECT a.*,r.chat_id FROM chat_import_assets a JOIN chat_import_records r ON r.id=a.import_id WHERE a.id=? AND r.user_id=?")
        .bind(id).bind(owner).fetch_optional(&mut *tx).await?.ok_or_else(|| ApiError::not_found("import_asset_missing","Imported attachment not found"))?;
    let mut status = AssetStatus {
        id: id.into(),
        source_id: row.get("source_id"),
        uploaded_bytes: row.get("uploaded_bytes"),
        complete: row.get("complete"),
    };
    if status.complete || chunk.offset < status.uploaded_bytes {
        return Ok(status);
    }
    let size: i64 = row.get("size_bytes");
    if chunk.offset != status.uploaded_bytes || bytes.len() as i64 > size - chunk.offset {
        return Err(ApiError::conflict(
            "import_offset",
            "Attachment upload position changed; retry the import",
        ));
    }
    let staging = tmp.join("chat-imports");
    tokio::fs::create_dir_all(&staging)
        .await
        .map_err(io_error)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o700))
            .await
            .map_err(io_error)?;
    }
    let path = staging.join(format!("{id}.part"));
    let staged_size = match tokio::fs::symlink_metadata(&path).await {
        Ok(meta) if meta.file_type().is_file() => meta.len(),
        Ok(_) => return Err(invalid()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(io_error(error)),
    };
    if staged_size < chunk.offset as u64 {
        sqlx::query("UPDATE chat_import_assets SET uploaded_bytes=0 WHERE id=?")
            .bind(id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        status.uploaded_bytes = 0;
        return Ok(status);
    }
    let mut options = tokio::fs::OpenOptions::new();
    options.create(true).truncate(false).write(true).read(true);
    #[cfg(unix)]
    {
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(&path).await.map_err(io_error)?;
    file.set_len(chunk.offset as u64).await.map_err(io_error)?;
    file.seek(std::io::SeekFrom::Start(chunk.offset as u64))
        .await
        .map_err(io_error)?;
    file.write_all(&bytes).await.map_err(io_error)?;
    file.sync_data().await.map_err(io_error)?;
    status.uploaded_bytes += bytes.len() as i64;
    status.complete = status.uploaded_bytes == size;
    if status.complete {
        let expected = chunk
            .sha256
            .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(invalid)?;
        file.seek(std::io::SeekFrom::Start(0))
            .await
            .map_err(io_error)?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];
        let mut signature = Vec::new();
        loop {
            let n = file.read(&mut buffer).await.map_err(io_error)?;
            if n == 0 {
                break;
            }
            if signature.is_empty() {
                signature.extend_from_slice(&buffer[..n.min(512)]);
            }
            hash.update(&buffer[..n]);
        }
        if format!("{:x}", hash.finalize()) != expected.to_ascii_lowercase() {
            sqlx::query("UPDATE chat_import_assets SET uploaded_bytes=0 WHERE id=?")
                .bind(id)
                .execute(&mut *tx)
                .await?;
            tx.commit().await?;
            let _ = tokio::fs::remove_file(&path).await;
            return Err(ApiError::bad_request(
                "import_checksum",
                "Attachment checksum did not match; retry the import",
            ));
        }
        let chat_id: String = row.get("chat_id");
        let destination = uploads.join("chat-imports").join(&chat_id).join(id);
        tokio::fs::create_dir_all(destination.parent().ok_or_else(invalid)?)
            .await
            .map_err(io_error)?;
        tokio::fs::copy(&path, &destination)
            .await
            .map_err(io_error)?;
        let (kind, mime) = file_type(&signature);
        sqlx::query("INSERT INTO attachments(id,user_id,chat_id,message_id,revision_id,original_filename,storage_path,mime_type,size_bytes,attachment_kind,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)")
            .bind(id).bind(owner).bind(&chat_id).bind(row.get::<String,_>("message_id")).bind(row.get::<String,_>("revision_id")).bind(row.get::<String,_>("filename"))
            .bind(format!("chat-imports/{chat_id}/{id}")).bind(mime).bind(size).bind(kind).bind(unix_timestamp()).execute(&mut *tx).await?;
    }
    sqlx::query(
        "UPDATE chat_import_assets SET uploaded_bytes=?,complete=? WHERE id=? AND uploaded_bytes=?",
    )
    .bind(status.uploaded_bytes)
    .bind(status.complete)
    .bind(id)
    .bind(chunk.offset)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    if status.complete {
        let _ = tokio::fs::remove_file(path).await;
    }
    Ok(status)
}

pub async fn chat_asset_ids(
    pool: &SqlitePool,
    owner: &str,
    chat_id: &str,
) -> Result<Vec<String>, ApiError> {
    Ok(sqlx::query_scalar("SELECT a.id FROM chat_import_assets a JOIN chat_import_records r ON r.id=a.import_id WHERE r.user_id=? AND r.chat_id=?")
        .bind(owner).bind(chat_id).fetch_all(pool).await?)
}

pub async fn cleanup_deleted_files(
    tmp: &Path,
    uploads: &Path,
    chat_id: &str,
    asset_ids: &[String],
) {
    if Uuid::parse_str(chat_id).is_err() {
        return;
    }
    for id in asset_ids.iter().filter(|id| Uuid::parse_str(id).is_ok()) {
        let _ = tokio::fs::remove_file(tmp.join("chat-imports").join(format!("{id}.part"))).await;
        let _ = tokio::fs::remove_file(uploads.join("chat-imports").join(chat_id).join(id)).await;
    }
    let _ = tokio::fs::remove_dir(uploads.join("chat-imports").join(chat_id)).await;
}

// Startup has no active uploads: remove only importer-owned, unreferenced files.
pub async fn recover(pool: &SqlitePool, tmp: &Path, uploads: &Path) -> Result<(), sqlx::Error> {
    let staging = tmp.join("chat-imports");
    if let Ok(mut entries) = tokio::fs::read_dir(&staging).await {
        while let Some(entry) = entries.next_entry().await.map_err(sqlx::Error::Io)? {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(id) = name
                .strip_suffix(".part")
                .filter(|id| Uuid::parse_str(id).is_ok())
            else {
                continue;
            };
            let keep: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM chat_import_assets WHERE id=? AND complete=0)",
            )
            .bind(id)
            .fetch_one(pool)
            .await?;
            if !keep && entry.file_type().await.map_err(sqlx::Error::Io)?.is_file() {
                tokio::fs::remove_file(entry.path())
                    .await
                    .map_err(sqlx::Error::Io)?;
            }
        }
    }
    let root = uploads.join("chat-imports");
    if let Ok(mut chats) = tokio::fs::read_dir(&root).await {
        while let Some(chat) = chats.next_entry().await.map_err(sqlx::Error::Io)? {
            let chat_id = chat.file_name().to_string_lossy().into_owned();
            if Uuid::parse_str(&chat_id).is_err()
                || !chat.file_type().await.map_err(sqlx::Error::Io)?.is_dir()
            {
                continue;
            }
            let mut files = tokio::fs::read_dir(chat.path())
                .await
                .map_err(sqlx::Error::Io)?;
            while let Some(file) = files.next_entry().await.map_err(sqlx::Error::Io)? {
                let id = file.file_name().to_string_lossy().into_owned();
                if Uuid::parse_str(&id).is_err()
                    || !file.file_type().await.map_err(sqlx::Error::Io)?.is_file()
                {
                    continue;
                }
                let keep: bool = sqlx::query_scalar(
                    "SELECT EXISTS(SELECT 1 FROM attachments WHERE id=? AND storage_path=?)",
                )
                .bind(&id)
                .bind(format!("chat-imports/{chat_id}/{id}"))
                .fetch_one(pool)
                .await?;
                if !keep {
                    tokio::fs::remove_file(file.path())
                        .await
                        .map_err(sqlx::Error::Io)?;
                }
            }
            let _ = tokio::fs::remove_dir(chat.path()).await;
        }
    }
    Ok(())
}
fn io_error(_: std::io::Error) -> ApiError {
    ApiError::internal("Could not store the imported attachment")
}

pub fn file_type(bytes: &[u8]) -> (&'static str, &'static str) {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        ("image", "image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        ("image", "image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        ("image", "image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        ("image", "image/webp")
    } else {
        ("file", "application/octet-stream")
    }
}
