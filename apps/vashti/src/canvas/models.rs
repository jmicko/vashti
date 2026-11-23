use serde::{Deserialize, Serialize};

pub const MAX_CONTENT: usize = 256 * 1024;
pub const MAX_DOCUMENTS: usize = 16;
pub const LANGUAGES: &[&str] = &[
    "markdown",
    "text",
    "javascript",
    "typescript",
    "python",
    "rust",
    "html",
    "css",
    "json",
    "sql",
];

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub message_id: Option<String>,
    pub revision_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Head {
    pub document_id: String,
    pub version_id: String,
    #[serde(default)]
    pub trashed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Version {
    pub id: String,
    pub document_id: String,
    pub parent_id: Option<String>,
    pub version_number: i64,
    pub title: String,
    pub language: String,
    pub content: String,
    pub actor: String,
    pub model_name: Option<String>,
    pub edit_session_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Document {
    pub id: String,
    pub trashed: bool,
    pub version: Version,
}

#[derive(Clone, Debug, Serialize)]
pub struct Comment {
    pub id: String,
    pub document_id: String,
    pub version_id: String,
    pub quote: String,
    pub content: String,
    pub actor: String,
    pub model_name: Option<String>,
    pub resolved: bool,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Workspace {
    pub scope_id: String,
    pub documents: Vec<Document>,
    pub comments: Vec<Comment>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub find: String,
    pub replace: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mutation {
    Create {
        document_id: Option<String>,
        title: String,
        language: String,
        content: String,
    },
    Update {
        document_id: String,
        expected_version_id: String,
        title: Option<String>,
        language: Option<String>,
        content: Option<String>,
        edits: Option<Vec<Edit>>,
        edit_session_id: Option<String>,
    },
    Restore {
        document_id: String,
        expected_version_id: String,
        version_id: String,
    },
    Trash {
        document_id: String,
        expected_version_id: String,
        trashed: bool,
    },
    Comment {
        document_id: String,
        expected_version_id: String,
        quote: String,
        content: String,
    },
    ResolveComment {
        comment_id: String,
        resolved: bool,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MutateRequest {
    pub scope: Scope,
    pub mutation: Mutation,
}

pub struct Actor<'a> {
    pub model_name: Option<&'a str>,
}

impl Actor<'_> {
    pub fn kind(&self) -> &'static str {
        if self.model_name.is_some() {
            "model"
        } else {
            "human"
        }
    }
}
