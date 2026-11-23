use super::service::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use sha2::{Digest, Sha256};
use sqlx::{
    SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};

async fn pool() -> SqlitePool {
    let db = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(":memory:")
                .foreign_keys(true),
        )
        .await
        .unwrap();
    crate::startup::migrations::run(&db).await.unwrap();
    crate::startup::bootstrap::ensure_app_settings(&db)
        .await
        .unwrap();
    sqlx::query("INSERT INTO users(id,username,password_hash,created_at,updated_at) VALUES ('owner','owner','hash',100,100),('other','other','hash',100,100); INSERT INTO ollama_backends(id,name,base_url,created_at,updated_at) VALUES('backend','Local','http://127.0.0.1:11434',100,100)").execute(&db).await.unwrap();
    db
}
fn request() -> ImportRequest {
    serde_json::from_value(serde_json::json!({
        "backend_id":"backend","model_name":"local-model","allow_history_search":false,
        "conversation":{"source_id":"synthetic-chat","title":"Synthetic imported conversation","created_at":100,"updated_at":103,"archived":true,"do_not_remember":false,"current_node_id":"answer-b",
        "messages":[{"source_id":"question","parent_id":null,"role":"user","content":"A synthetic question","thinking":"","model":null,"created_at":100},
        {"source_id":"answer-a","parent_id":"question","role":"assistant","content":"First answer","thinking":"Reasoning","model":"original-model","created_at":101},
        {"source_id":"answer-b","parent_id":"question","role":"assistant","content":"Alternative answer","thinking":"","model":"original-model","created_at":102}],
        "assets":[{"source_id":"fixture-file","message_id":"question","filename":"../../fixture.pdf","size_bytes":8}],"original_json":"{\"id\":\"synthetic-chat\"}"}
    })).unwrap()
}

#[tokio::test]
async fn workspace_page_is_owner_scoped_archived_and_idempotent() {
    use crate::canvas::{models::Scope, service::workspace};
    let db = pool().await;
    let page_id = format!("chatgpt-page-{}", "a".repeat(64));
    let expected = "First section\n\nSecond section";
    let page = || {
        serde_json::from_value::<ImportRequest>(serde_json::json!({
        "backend_id":"backend", "model_name":"local-model", "allow_history_search":true,
        "conversation":{"source_id":page_id,"title":"A synthetic imported page","created_at":100,"updated_at":103,
        "archived":true,"do_not_remember":true,"current_node_id":"page","assets":[],
        "messages":[{"source_id":"page","parent_id":null,"role":"assistant","content":"Imported page","thinking":"","model":null,"created_at":100}],
        "workspace":{"title":"A synthetic imported page","content":expected},
        "original_json":serde_json::json!({"id":page_id,"page":{"projection":{"materialized_search_text":expected}}}).to_string()}
    })).unwrap()
    };
    let mut forged = page();
    forged.conversation.workspace.as_mut().unwrap().content = "Not the exported content".into();
    assert!(import_conversation(&db, "owner", forged).await.is_err());
    let imported = import_conversation(&db, "owner", page()).await.unwrap();
    assert!(imported_documents(&db, "other").await.unwrap().is_empty());
    assert_eq!(
        imported_documents(&db, "owner").await.unwrap()[0].id,
        imported.chat_id
    );
    assert!(
        crate::chats::service::get_chat(&db, "owner", &imported.chat_id)
            .await
            .unwrap()
            .archived_at
            .is_some()
    );
    assert!(
        workspace(&db, "other", &imported.chat_id, &Scope::default())
            .await
            .is_err()
    );
    let doc = workspace(&db, "owner", &imported.chat_id, &Scope::default())
        .await
        .unwrap()
        .documents
        .remove(0);
    assert_eq!(doc.version.content, expected);
    assert_eq!(doc.version.language, "text");
    let again = import_conversation(&db, "owner", page()).await.unwrap();
    assert!(again.existing);
    assert_eq!(imported_documents(&db, "owner").await.unwrap().len(), 1);
    assert_eq!(
        workspace(&db, "owner", &imported.chat_id, &Scope::default())
            .await
            .unwrap()
            .documents[0]
            .version
            .id,
        doc.version.id
    );
    let profile_or_search:bool=sqlx::query_scalar("SELECT do_not_remember=0 OR allow_history_search!=0 FROM chat_import_records WHERE chat_id=?")
        .bind(&imported.chat_id).fetch_one(&db).await.unwrap();
    assert!(!profile_or_search);
}

#[tokio::test]
async fn imported_links_bind_only_to_their_messages_attachment() {
    let db = pool().await;
    let mut input = request();
    let source = format!("asset-{}", "b".repeat(64));
    let link = format!("/.vashti/imported-file/{source}");
    input.conversation.assets[0].source_id = source;
    input.conversation.messages[0].content = format!("Download [this file]({link})");
    input.conversation.messages[1].content = format!("Other message [cannot use it]({link})");
    let result = import_conversation(&db, "owner", input).await.unwrap();
    let mut messages = crate::chats::service::list_messages(&db, "owner", &result.chat_id)
        .await
        .unwrap()
        .1;
    let question = messages
        .iter_mut()
        .find(|m| m.role == "user")
        .unwrap()
        .active_revision
        .take()
        .unwrap();
    let answer = messages
        .iter()
        .find_map(|m| {
            m.active_revision
                .as_ref()
                .filter(|r| r.content_text.starts_with("Other message"))
        })
        .unwrap();
    assert!(
        question
            .content_text
            .ends_with(&format!("/.vashti/imported-file/{})", result.assets[0].id))
    );
    assert!(answer.content_text.ends_with(&format!("{link})")));
}
#[tokio::test]
async fn import_preserves_branches_timestamps_archive_and_originals_without_inference() {
    let db = pool().await;
    let result = import_conversation(&db, "owner", request()).await.unwrap();
    assert!(!result.existing);
    let chat = crate::chats::service::get_chat(&db, "owner", &result.chat_id)
        .await
        .unwrap();
    assert_eq!(chat.created_at, 100);
    assert_eq!(chat.updated_at, 103);
    assert!(chat.archived_at.is_some());
    assert_eq!(chat.default_model_name, "local-model");
    let (root, messages) = crate::chats::service::list_messages(&db, "owner", &result.chat_id)
        .await
        .unwrap();
    assert_eq!(messages.len(), 3);
    let parent = messages.iter().find(|m| m.role == "user").unwrap();
    assert_eq!(root.as_deref(), Some(parent.id.as_str()));
    let active = messages
        .iter()
        .find(|m| Some(&m.id) == parent.active_child_message_id.as_ref())
        .unwrap();
    assert_eq!(
        active.active_revision.as_ref().unwrap().content_text,
        "Alternative answer"
    );
    assert_eq!(active.model_name.as_deref(), Some("original-model"));
    assert!(active.backend_id.is_none());
    assert!(
        messages
            .iter()
            .all(|m| m.import_source.as_deref() == Some("chatgpt"))
    );
    let again = import_conversation(&db, "owner", request()).await.unwrap();
    assert!(again.existing);
    assert_eq!(again.chat_id, result.chat_id);
    let other = import_conversation(&db, "other", request()).await.unwrap();
    assert_ne!(other.chat_id, result.chat_id);
    let retrieval = crate::chats::retrieval::ConversationRetrieval::new(db.clone());
    retrieval.rebuild_all().await.unwrap();
    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM conversation_search_documents WHERE chat_id=?")
            .bind(&result.chat_id)
            .fetch_one(&db)
            .await
            .unwrap();
    assert_eq!(count, 0);
    let source: String =
        sqlx::query_scalar("SELECT original_json FROM chat_import_records WHERE chat_id=?")
            .bind(&result.chat_id)
            .fetch_one(&db)
            .await
            .unwrap();
    assert!(source.contains("synthetic-chat"));
}
#[tokio::test]
async fn attachment_chunks_are_resumable_owner_checked_and_binary_is_download_only() {
    let db = pool().await;
    let imported = import_conversation(&db, "owner", request()).await.unwrap();
    let id = &imported.assets[0].id;
    let root = std::env::temp_dir().join(format!("vashti-import-test-{}", uuid::Uuid::new_v4()));
    let tmp = root.join("tmp");
    let uploads = root.join("uploads");
    let chunk = || Chunk {
        expected_user_id: None,
        offset: 0,
        data: STANDARD.encode(b"%PDF"),
        sha256: None,
    };
    assert!(
        upload_chunk(&db, &tmp, &uploads, "other", id, chunk())
            .await
            .is_err()
    );
    let first = upload_chunk(&db, &tmp, &uploads, "owner", id, chunk())
        .await
        .unwrap();
    assert_eq!(first.uploaded_bytes, 4);
    assert_eq!(
        upload_chunk(&db, &tmp, &uploads, "owner", id, chunk())
            .await
            .unwrap()
            .uploaded_bytes,
        4
    );
    assert!(
        upload_chunk(
            &db,
            &tmp,
            &uploads,
            "owner",
            id,
            Chunk {
                expected_user_id: None,
                offset: 4,
                data: STANDARD.encode(b"data"),
                sha256: Some("0".repeat(64)),
            }
        )
        .await
        .is_err()
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT uploaded_bytes FROM chat_import_assets WHERE id=?")
            .bind(id)
            .fetch_one(&db)
            .await
            .unwrap(),
        0
    );
    upload_chunk(&db, &tmp, &uploads, "owner", id, chunk())
        .await
        .unwrap();
    let done = upload_chunk(
        &db,
        &tmp,
        &uploads,
        "owner",
        id,
        Chunk {
            expected_user_id: None,
            offset: 4,
            data: STANDARD.encode(b"data"),
            sha256: Some(format!("{:x}", Sha256::digest(b"%PDFdata"))),
        },
    )
    .await
    .unwrap();
    assert!(done.complete);
    let attachment = crate::uploads::service::get_attachment(&db, "owner", id)
        .await
        .unwrap();
    assert_eq!(attachment.attachment_kind, "file");
    assert_eq!(attachment.original_filename, "fixture.pdf");
    let payload = crate::uploads::service::prompt_attachment_payload(
        &db,
        &uploads,
        "owner",
        &imported.chat_id,
        attachment.message_id.as_deref().unwrap(),
        attachment.revision_id.as_deref().unwrap(),
        &mut crate::uploads::models::PdfPromptContext::new(72_000),
    )
    .await
    .unwrap();
    assert!(payload.1.is_empty());
    assert!(!payload.0.contains("%PDF"));
    assert!(payload.0.contains("download only"));
    assert!(
        crate::uploads::service::get_attachment(&db, "other", id)
            .await
            .is_err()
    );
    tokio::fs::remove_dir_all(root).await.unwrap();
}
#[tokio::test]
async fn invalid_imports_are_atomic_and_never_replay_system_or_tool_authority() {
    let db = pool().await;
    for case in 0..4 {
        let mut input = request();
        match case {
            0 => input.conversation.messages[0].role = "system".into(),
            1 => input.conversation.messages[0].parent_id = Some("answer-b".into()),
            2 => input.conversation.messages[1].source_id = "question".into(),
            _ => input.conversation.assets[0].message_id = "foreign".into(),
        }
        assert!(import_conversation(&db, "owner", input).await.is_err());
    }
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM chats")
            .fetch_one(&db)
            .await
            .unwrap(),
        0
    );
}
#[tokio::test]
async fn forward_attachment_migration_preserves_legacy_avatar_references() {
    let db = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(":memory:")
                .foreign_keys(true),
        )
        .await
        .unwrap();
    let migrations = sqlx::migrate!("./migrations");
    for m in migrations.iter().filter(|m| m.version < 20261121000000) {
        sqlx::raw_sql(&m.sql).execute(&db).await.unwrap();
    }
    sqlx::raw_sql("INSERT INTO users(id,username,password_hash,created_at,updated_at) VALUES('owner','owner','hash',100,100); INSERT INTO ollama_backends(id,name,base_url,created_at,updated_at) VALUES('backend','Local','http://127.0.0.1:11434',100,100); INSERT INTO chats(id,user_id,default_backend_id,default_model_name,title,created_at,updated_at,last_message_at) VALUES('chat','owner','backend','model','Fixture',100,100,100); INSERT INTO attachments(id,user_id,chat_id,original_filename,storage_path,mime_type,size_bytes,attachment_kind,created_at) VALUES('legacy','owner','chat','avatar.png','chat/legacy','image/png',100,'image',100); INSERT INTO personas(id,owner_user_id,created_at,updated_at) VALUES('persona','owner',100,100); INSERT INTO persona_versions(id,persona_id,version_number,display_name,base_backend_id,base_model_name,system_prompt,avatar_attachment_id,created_at) VALUES('version','persona',1,'Fixture','backend','model','','legacy',100);").execute(&db).await.unwrap();
    let mut tx = db.begin().await.unwrap();
    sqlx::raw_sql(include_str!(
        "../../migrations/20261121000000_chatgpt_import.sql"
    ))
    .execute(&mut *tx)
    .await
    .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(
        sqlx::query_scalar::<_, String>(
            "SELECT avatar_attachment_id FROM persona_versions WHERE id='version'"
        )
        .fetch_one(&db)
        .await
        .unwrap(),
        "legacy"
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM attachments")
            .fetch_one(&db)
            .await
            .unwrap(),
        1
    );
    let rows = sqlx::query("PRAGMA foreign_key_check")
        .fetch_all(&db)
        .await
        .unwrap();
    assert!(rows.is_empty());
}

#[tokio::test]
async fn imports_respect_profile_and_history_opt_outs_including_followups() {
    let db = pool().await;
    for private in [false, true] {
        let mut input = request();
        input.conversation.source_id = format!("privacy-{private}");
        input.conversation.original_json = format!(r#"{{"id":"privacy-{private}"}}"#);
        input.conversation.do_not_remember = private;
        input.allow_history_search = true;
        let imported = import_conversation(&db, "owner", input).await.unwrap();
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM personalization_chat_sources s JOIN chat_messages m ON m.id=s.message_id WHERE m.chat_id=?").bind(&imported.chat_id).fetch_one(&db).await.unwrap();
        assert_eq!(count, 0);
        crate::chats::service::create_user_message(
            &db,
            "owner",
            &imported.chat_id,
            crate::chats::handlers::CreateMessageRequest {
                content_text: "Synthetic follow-up".into(),
                parent_message_id: None,
            },
        )
        .await
        .unwrap();
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM personalization_chat_sources s JOIN chat_messages m ON m.id=s.message_id WHERE m.chat_id=?").bind(&imported.chat_id).fetch_one(&db).await.unwrap();
        assert_eq!(count, if private { 0 } else { 1 });
        crate::chats::retrieval::ConversationRetrieval::new(db.clone())
            .rebuild_all()
            .await
            .unwrap();
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM conversation_search_documents WHERE chat_id=?",
        )
        .bind(&imported.chat_id)
        .fetch_one(&db)
        .await
        .unwrap();
        assert_eq!(count, if private { 0 } else { 4 });
    }
}

#[tokio::test]
async fn continued_imported_responses_use_the_selected_vashti_model() {
    let db = pool().await;
    let imported = import_conversation(&db, "owner", request()).await.unwrap();
    let (_, messages) = crate::chats::service::list_messages(&db, "owner", &imported.chat_id)
        .await
        .unwrap();
    let response = messages.iter().find(|m| m.role == "assistant").unwrap();
    let prepared = crate::chats::service::prepare_continuation(
        &db,
        std::path::Path::new("/tmp"),
        "owner",
        &imported.chat_id,
        &response.id,
        crate::chats::handlers::ContinueMessageRequest {
            inference_settings: None,
            tool_preferences: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(prepared.model_name, "local-model");
    assert_eq!(prepared.backend_base_url, "http://127.0.0.1:11434");
}

#[tokio::test]
async fn concurrent_imports_are_idempotent() {
    let db = pool().await;
    let (a, b) = tokio::join!(
        import_conversation(&db, "owner", request()),
        import_conversation(&db, "owner", request())
    );
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!(a.chat_id, b.chat_id);
    assert_ne!(a.existing, b.existing);
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM chats")
            .fetch_one(&db)
            .await
            .unwrap(),
        1
    );
}

#[tokio::test]
async fn partial_uploads_restart_safely_and_deleted_import_files_are_recovered() {
    let db = pool().await;
    let imported = import_conversation(&db, "owner", request()).await.unwrap();
    let id = &imported.assets[0].id;
    let root =
        std::env::temp_dir().join(format!("vashti-import-recovery-{}", uuid::Uuid::new_v4()));
    let tmp = root.join("tmp");
    let uploads = root.join("uploads");
    let chunk = |offset, bytes: &[u8]| Chunk {
        expected_user_id: None,
        offset,
        data: STANDARD.encode(bytes),
        sha256: Some(format!("{:x}", Sha256::digest(b"%PDFdata"))),
    };
    upload_chunk(&db, &tmp, &uploads, "owner", id, chunk(0, b"%PDF"))
        .await
        .unwrap();
    tokio::fs::write(tmp.join("chat-imports").join(format!("{id}.part")), b"%P")
        .await
        .unwrap();
    assert_eq!(
        upload_chunk(&db, &tmp, &uploads, "owner", id, chunk(4, b"data"))
            .await
            .unwrap()
            .uploaded_bytes,
        0
    );
    upload_chunk(&db, &tmp, &uploads, "owner", id, chunk(0, b"%PDFdata"))
        .await
        .unwrap();
    let destination = uploads
        .join("chat-imports")
        .join(&imported.chat_id)
        .join(id);
    recover(&db, &tmp, &uploads).await.unwrap();
    assert!(destination.exists());
    crate::chats::service::delete_chat(&db, "owner", &imported.chat_id)
        .await
        .unwrap();
    tokio::fs::write(
        tmp.join("chat-imports").join(format!("{id}.part")),
        b"orphan",
    )
    .await
    .unwrap();
    recover(&db, &tmp, &uploads).await.unwrap();
    assert!(!destination.exists());
    assert!(!tmp.join("chat-imports").join(format!("{id}.part")).exists());
    tokio::fs::remove_dir_all(root).await.unwrap();
}
