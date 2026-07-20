export type User = {
  id: string;
  username: string;
  display_name: string | null;
  email: string | null;
  role: string;
};

export type RegisteredUser = User & {
  is_disabled: boolean;
};

export type AdminUser = User & {
  is_disabled: boolean;
  created_at: number;
  updated_at: number;
  last_login_at: number | null;
  permission_tags: PermissionTag[];
};

export type PermissionTag = {
  id: string;
  label: string;
  kind: string;
};

export type SessionResponse = {
  is_authenticated: boolean;
  user: User | null;
  can_create_account: boolean;
  private_vault_key: {
    user_id: string;
    key_material: string;
  } | null;
};

export type RegisterResponse = {
  requires_approval: boolean;
  user: RegisteredUser;
};

export type AdminUsersResponse = {
  users: AdminUser[];
  available_tags: PermissionTag[];
};

export type AdminUserMutationResponse = {
  user: AdminUser;
};

export type Backend = {
  id: string;
  name: string;
  base_url: string;
  is_enabled: boolean;
  last_health_status: string | null;
  last_error: string | null;
};

export type BackendsResponse = {
  backends: Backend[];
};

export type DetectLocalhostResponse = {
  detected: Array<{
    name: string;
    base_url: string;
  }>;
};

export type ModelBackgroundMode = "fill" | "fit" | "stretch" | "tile";

export type ModelBackgroundSettings = {
  background_asset_id: string | null;
  background_dim: number;
  background_message_dim: number;
  background_landscape_mode: ModelBackgroundMode;
  background_landscape_x: number;
  background_landscape_y: number;
  background_landscape_scale: number;
  background_portrait_mode: ModelBackgroundMode;
  background_portrait_x: number;
  background_portrait_y: number;
  background_portrait_scale: number;
  background_is_private?: boolean;
};

export type ModelInfo = {
  name: string;
  supports_images: boolean;
  supports_thinking?: boolean;
  capabilities?: string[];
  is_favorite?: boolean;
  is_default?: boolean;
  avatar_asset_id: string | null;
  avatar_crop_x: number;
  avatar_crop_y: number;
  avatar_crop_size: number;
} & ModelBackgroundSettings;

export type AdminModelInfo = ModelInfo & {
  is_enabled: boolean;
  permission_tags: PermissionTag[];
  default_permission_tags: PermissionTag[];
};

export type UserModelInfo = ModelInfo & {
  is_visible: boolean;
  is_favorite: boolean;
  is_default: boolean;
  personal_avatar_asset_id: string | null;
  personal_avatar_crop_x: number;
  personal_avatar_crop_y: number;
  personal_avatar_crop_size: number;
  default_avatar_asset_id: string | null;
  default_avatar_crop_x: number;
  default_avatar_crop_y: number;
  default_avatar_crop_size: number;
  personal_background_asset_id: string | null;
  personal_background_dim: number;
  personal_background_message_dim: number;
  personal_background_landscape_mode: ModelBackgroundMode;
  personal_background_landscape_x: number;
  personal_background_landscape_y: number;
  personal_background_landscape_scale: number;
  personal_background_portrait_mode: ModelBackgroundMode;
  personal_background_portrait_x: number;
  personal_background_portrait_y: number;
  personal_background_portrait_scale: number;
  default_background_asset_id: string | null;
  default_background_dim: number;
  default_background_message_dim: number;
  default_background_landscape_mode: ModelBackgroundMode;
  default_background_landscape_x: number;
  default_background_landscape_y: number;
  default_background_landscape_scale: number;
  default_background_portrait_mode: ModelBackgroundMode;
  default_background_portrait_x: number;
  default_background_portrait_y: number;
  default_background_portrait_scale: number;
};

export type BackendModelGroup = {
  backend: {
    id: string;
    name: string;
  };
  models: ModelInfo[];
};

export type AdminBackendModelGroup = {
  backend: {
    id: string;
    name: string;
  };
  models: AdminModelInfo[];
};

export type AdminModelTagPatchPayload = {
  backend_id: string;
  model_name: string;
  permission_tags?: string[];
  default_permission_tags?: string[];
};

export type UserBackendModelGroup = {
  backend: {
    id: string;
    name: string;
  };
  models: UserModelInfo[];
};

export type ModelsResponse = {
  backends: BackendModelGroup[];
  is_refreshing?: boolean;
  cache_updated_at?: number | null;
};

export type UserModelsResponse = {
  backends: UserBackendModelGroup[];
  is_refreshing: boolean;
  cache_updated_at: number | null;
};

export type AdminModelsResponse = {
  backends: AdminBackendModelGroup[];
  available_tags: PermissionTag[];
  default_permission_tags: PermissionTag[];
  is_refreshing: boolean;
  cache_updated_at: number | null;
};

export type PersonaVersion = {
  id: string;
  persona_id: string;
  version_number: number;
  display_name: string;
  avatar_asset_id: string | null;
  avatar_crop_x: number;
  avatar_crop_y: number;
  avatar_crop_size: number;
  background_asset_id: string | null;
  background_dim: number;
  background_message_dim: number;
  background_landscape_mode: ModelBackgroundMode;
  background_landscape_x: number;
  background_landscape_y: number;
  background_landscape_scale: number;
  background_portrait_mode: ModelBackgroundMode;
  background_portrait_x: number;
  background_portrait_y: number;
  background_portrait_scale: number;
  base_backend_id: string;
  base_model_name: string;
  system_prompt: string;
  tool_policy_json: string | null;
  created_by_user_id: string | null;
  created_at: number;
};

export type Persona = {
  id: string;
  owner_user_id: string | null;
  owner_username: string | null;
  visibility: string;
  lifecycle_state: string;
  current_version: PersonaVersion;
  is_owner: boolean;
  is_member: boolean;
  created_at: number;
  updated_at: number;
};

export type PersonasResponse = {
  personas: Persona[];
};

export type ModelPickerCache = {
  models: ModelsResponse;
  personas: PersonasResponse;
};

export type PersonaMutationResponse = {
  persona: Persona;
};

export type ChatSummary = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name?: string | null;
  updated_at: number;
  last_message_at: number;
  message_count: number;
};

export type ChatToolPreferences = {
  tool_use_enabled: boolean;
  tools: Record<string, boolean>;
  web_search_enabled?: boolean;
  web_fetch_enabled?: boolean;
};

export type ChatInferenceSettings = {
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
  min_p?: number | null;
  repeat_penalty?: number | null;
  repeat_last_n?: number | null;
  presence_penalty?: number | null;
  frequency_penalty?: number | null;
  num_ctx?: number | null;
  num_predict?: number | null;
  num_gpu?: number | null;
  num_thread?: number | null;
  seed?: number | null;
};

export type ContextSelectionMode = "single" | "multiple";

export type ContextCategory = {
  id: string;
  name: string;
  selection_mode: ContextSelectionMode;
  sort_order: number;
  created_at: number;
  updated_at: number;
};

export type ContextBlockVersion = {
  id: string;
  block_id: string;
  version_number: number;
  name: string;
  content: string;
  created_at: number;
};

export type ContextBlock = {
  id: string;
  category_id: string | null;
  sort_order: number;
  current_version: ContextBlockVersion;
  created_at: number;
  updated_at: number;
};

export type ContextLibraryResponse = {
  categories: ContextCategory[];
  blocks: ContextBlock[];
};

export type ContextBlockSelection = {
  block_id: string;
  block_version_id: string;
  category_id: string | null;
  category_name: string | null;
  category_selection_mode: ContextSelectionMode | null;
  version_number: number;
  name: string;
  content: string;
  position: number;
};

export type ChatDetail = {
  id: string;
  title: string;
  default_backend_id: string;
  backend_name: string;
  default_model_name: string;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name?: string | null;
  system_prompt_override?: string | null;
  tool_preferences: ChatToolPreferences;
  inference_settings: ChatInferenceSettings;
  context_blocks: ContextBlockSelection[];
  active_root_message_id: string | null;
  created_at: number;
  updated_at: number;
};

export type ChatResponse = {
  chat: ChatDetail;
};

export type ListChatsResponse = {
  chats: ChatSummary[];
};

export type ChatMessageRevision = {
  id: string;
  content_text: string;
  thinking_text: string;
  source: string;
  created_at: number;
};

export type AttachmentInfo = {
  id: string;
  chat_id?: string;
  message_id: string | null;
  revision_id: string | null;
  original_filename: string;
  mime_type: string;
  size_bytes: number;
  attachment_kind: string;
  created_at?: number;
  data_url?: string;
  text_content?: string;
};

export type ImageOpenHandler = (attachment: AttachmentInfo, attachments?: AttachmentInfo[]) => void;

export type ImageViewerState = {
  attachments: AttachmentInfo[];
  index: number;
};

export type ComposerAttachment = AttachmentInfo & {
  status: "ready" | "uploaded" | "uploading" | "error";
  error?: string;
  file?: File;
  isExisting?: boolean;
};

export type ComposerSubmitPayload = {
  prompt: string;
  attachments: ComposerAttachment[];
  toolPreferences?: ChatToolPreferences;
  thinkMode?: ThinkingMode;
  systemPromptOverride?: string | null;
  inferenceSettings?: ChatInferenceSettings;
  contextBlocks?: ContextBlockSelection[];
};

export type ThinkingMode = "auto" | "false" | "low" | "medium" | "high";

export type MessageStats = {
  total_duration?: number | null;
  load_duration?: number | null;
  prompt_eval_count?: number | null;
  prompt_eval_duration?: number | null;
  eval_count?: number | null;
  eval_duration?: number | null;
};

export type ChatMessage = {
  id: string;
  parent_message_id: string | null;
  active_child_message_id: string | null;
  active_revision_id: string | null;
  role: string;
  status: string;
  is_deleted: boolean;
  backend_id: string | null;
  model_name: string | null;
  persona_id?: string | null;
  persona_version_id?: string | null;
  persona_name_snapshot?: string | null;
  think_mode: string | null;
  done_reason: string | null;
  error_text: string | null;
  stats?: MessageStats | null;
  started_at: number | null;
  completed_at: number | null;
  created_at: number;
  updated_at: number;
  active_revision: ChatMessageRevision | null;
  revisions: ChatMessageRevision[];
  revision_count: number;
  attachments: AttachmentInfo[];
  context_blocks: ContextBlockSelection[];
};

export type ListMessagesResponse = {
  active_root_message_id: string | null;
  messages: ChatMessage[];
};

export type ChatSyncResponse = {
  changed: boolean;
  chat: ChatDetail;
  active_root_message_id: string | null;
  messages: ChatMessage[] | null;
};

export type MessageResponse = {
  message: ChatMessage;
};

export type ToolUsageRecord = {
  name: string;
  summary: string;
  arguments: unknown;
  result: string;
};

export type ThinkingSegment =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "tool";
      usage: ToolUsageRecord;
    };

export type MessageStreamSegment =
  | {
      type: "thinking";
      text: string;
    }
  | {
      type: "content";
      text: string;
    }
  | {
      type: "tool";
      usage: ToolUsageRecord;
    };

export type ParsedThinkingText = {
  thinkingText: string;
  segments: ThinkingSegment[];
};

export type AttachmentResponse = {
  attachment: AttachmentInfo;
};

export type GenerateEvent =
  | {
      type: "message_start";
      user_message: ChatMessage | null;
      assistant_message: ChatMessage;
    }
  | {
      type: "thinking_delta";
      assistant_message_id: string;
      delta: string;
    }
  | {
      type: "content_delta";
      assistant_message_id: string;
      delta: string;
    }
  | {
      type: "message_done";
      assistant_message_id: string;
      done_reason: string | null;
      stats?: MessageStats | null;
    }
  | {
      type: "chat_title";
      chat_id: string;
      title: string;
    }
  | {
      type: "message_stopped";
      assistant_message_id: string;
    }
  | {
      type: "error";
      assistant_message_id: string | null;
      message: string;
    };

export type AppSettings = {
  allow_signup: boolean;
  signup_limit: number;
  signup_count: number;
  max_upload_bytes: number;
  request_timeout_ms: number;
  network_mode: "lan_http" | "public_https_proxy";
  public_base_url: string | null;
  trust_proxy_headers: boolean;
  network_recovery_notice: string | null;
};

export type UserSettings = {
  default_backend_id: string | null;
  default_model_name: string | null;
  theme: string | null;
};

export type ToolSettings = {
  tools_enabled: boolean;
  ollama_web_search_enabled: boolean;
  ollama_web_fetch_enabled: boolean;
  ollama_api_key_configured: boolean;
  brave_search_enabled: boolean;
  brave_search_api_key_configured: boolean;
  direct_web_fetch_enabled: boolean;
  tool_system_prompt: string;
  default_tool_system_prompt: string;
  web_search_tool_prompt: string;
  default_web_search_tool_prompt: string;
  web_fetch_tool_prompt: string;
  default_web_fetch_tool_prompt: string;
  available_tags: PermissionTag[];
  default_tool_permission_tags: PermissionTag[];
  tool_permissions: ToolPermission[];
};

export type ToolPermission = {
  tool_id: string;
  permission_tags: PermissionTag[];
};

export type AvailableTool = {
  id: string;
  label: string;
  description: string;
};

export type AvailableToolsResponse = {
  tools_enabled: boolean;
  tools: AvailableTool[];
};

export type VersionResponse = {
  name: string;
  version: string;
};

export type LoadState =
  | { status: "loading" }
  | { status: "loaded"; session: SessionResponse }
  | { status: "error"; message: string };

export type FormState = {
  isSubmitting: boolean;
  error: string | null;
};

export type Page = "chat" | "private-chat" | "settings";
export type SettingsSection =
  | "profile"
  | "context"
  | "users"
  | "models"
  | "tools"
  | "app"
  | "backends";
export type NewChatMode = "standard" | "private";
export type AppRoute =
  | { page: "chat"; chatId?: string }
  | { page: "private-chat"; chatId: string }
  | { page: "settings"; section: SettingsSection };
export type AppSettingsGuard = {
  isDirty: boolean;
  save: () => Promise<boolean>;
  discard: () => void;
};
export type AutoScrollMode = "top" | "bottom" | "paused";
export type BranchScrollAnchor = {
  messageId: string;
  topOffset: number;
};
export type MessageVersion = {
  message: ChatMessage;
  revision: ChatMessageRevision;
};
export type VersionInfo = {
  index: number;
  total: number;
  canPrevious: boolean;
  canNext: boolean;
  onPrevious: () => void;
  onNext: () => void;
};
