export type NoteAiAccess = "none" | "read" | "edit" | "manage";
export type NoteListStatus = "active" | "trashed";
export type NoteSort = "updated" | "created" | "title";

export type NoteModelScope = {
  all_models: boolean;
  model_keys: string[];
};

export type NoteVersion = {
  id: string;
  note_id: string;
  version_number: number;
  title: string;
  content: string;
  actor_type: string;
  actor_user_id: string;
  actor_model_key: string | null;
  actor_model_name: string | null;
  source_chat_id: string | null;
  source_message_id: string | null;
  source_tool_call_id: string | null;
  created_at: number;
};

export type NoteSummary = {
  id: string;
  title: string;
  excerpt: string;
  current_version_id: string;
  current_version_number: number;
  tags: string[];
  is_pinned: boolean;
  ai_access: NoteAiAccess;
  model_scope: NoteModelScope;
  deleted_at: number | null;
  created_at: number;
  updated_at: number;
};

export type Note = {
  id: string;
  current_version: NoteVersion;
  tags: string[];
  is_pinned: boolean;
  ai_access: NoteAiAccess;
  model_scope: NoteModelScope;
  deleted_at: number | null;
  created_at: number;
  updated_at: number;
};

export type NoteDraft = {
  title: string;
  content: string;
  tags: string[];
  is_pinned: boolean;
  ai_access: NoteAiAccess;
  model_scope: NoteModelScope;
};

export type NoteSettings = {
  allow_model_read: boolean;
  allow_model_create: boolean;
  allow_model_edit: boolean;
  allow_model_trash: boolean;
  default_ai_access: NoteAiAccess;
  default_model_scope: NoteModelScope;
};
