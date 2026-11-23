export type ProfileSettings = {
  enabled: boolean;
  processor_backend_id: string | null;
  processor_model: string | null;
  allow_cloud_processor: boolean;
  learn_from_imports: boolean;
  allow_local: boolean;
  allow_cloud: boolean;
  allow_characters: boolean;
  model_overrides: Record<string, boolean>;
  idle_minutes: number;
  context_chars: number;
};

export type ProfileModel = {
  key: string;
  name: string;
  backend_id: string;
  model_name: string;
  cloud: boolean;
  character: boolean;
};

export type ProfileSource = {
  message_id: string;
  revision_id: string;
  chat_id: string;
  model_key: string;
  quote: string;
  created_at: number;
};

export type InputSource = Omit<ProfileSource, "quote"> & { text: string };
export type ProfileFact = {
  id: string;
  topic: string;
  content: string;
  locked: boolean;
  version: number;
  sources_json: string;
  created_at: number;
  updated_at: number;
};
export type ProfileVersion = {
  version: number;
  content: string;
  sources_json: string;
  actor: "human" | "processor";
  created_at: number;
};
export type PersonalProfile = {
  settings: ProfileSettings;
  revision: number;
  facts: ProfileFact[];
  last_attempt_at: number | null;
  last_processed_at: number | null;
  last_error: string | null;
  requested: boolean;
  rebuild_id: string | null;
};
export type ProfileNotice = { id: string; created_at: number; read_at: number | null };
export type ReplayDraft = {
  facts: ProfileFact[];
  processed: string[];
  history: Record<string, ProfileVersion[]>;
};
export type DeviceProfile = PersonalProfile & {
  processed: string[];
  forgotten?: string[];
  notices?: ProfileNotice[];
  rebuild_draft?: ReplayDraft;
  suppressed: string[];
  history: Record<string, ProfileVersion[]>;
};
export type ProfileMode = "server" | "device";

export type ProfilePromptPayload = {
  backend_id: string;
  model_name: string;
  inference_settings?: { num_ctx?: number | null };
  messages: Array<{ role: string; content_text: string; thinking_text?: string }>;
};
