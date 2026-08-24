export type MemoryModelScope = {
  all_models: boolean;
  model_keys: string[];
};

export type MemoryVersion = {
  id: string;
  memory_id: string;
  version_number: number;
  content: string;
  actor_type: "human" | "model";
  actor_user_id: string;
  actor_model_key: string | null;
  actor_model_name: string | null;
  source_chat_id: string | null;
  source_message_id: string | null;
  source_tool_call_id: string | null;
  created_at: number;
};

export type MemorySummary = {
  id: string;
  excerpt: string;
  current_version_id: string;
  current_version_number: number;
  model_scope: MemoryModelScope;
  deleted_at: number | null;
  created_at: number;
  updated_at: number;
};

export type Memory = {
  id: string;
  current_version: MemoryVersion;
  model_scope: MemoryModelScope;
  deleted_at: number | null;
  created_at: number;
  updated_at: number;
};

export type MemorySettings = {
  allow_model_read: boolean;
  allow_model_create: boolean;
  allow_model_edit: boolean;
  allow_model_forget: boolean;
};
