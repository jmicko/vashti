export type Policy = "block" | "ask" | "allow";
export type TaskConfig = {
  backend_id: string;
  model_name: string;
  instructions: string;
  notes: boolean;
  memories: boolean;
  chat_history: boolean;
  writes: Policy;
  web: Policy;
  sandbox: boolean;
  reasoning: boolean;
  max_steps: number;
  max_output_tokens: number;
};
export type AssistantTask = {
  kind?: "task" | "conversation"; conversation_id?: string | null;
  id: string; title: string; instructions: string; config: TaskConfig;
  interval_seconds: number | null; next_run_at: number | null; schedule_enabled: boolean;
  schedule_error: string | null; archived_at: number | null; last_run_id: string | null;
  last_status: string | null; needs_attention: boolean; created_at: number; updated_at: number; revision: number;
};
export type TaskSummary = Omit<AssistantTask, "config">;
export type RunStatus = "queued" | "running" | "waiting_approval" | "waiting_input" | "paused" | "completed" | "failed" | "cancelled" | "interrupted" | "limit_reached";
export type AssistantRun = {
  id: string; task_id: string; status: RunStatus; instructions: string;
  progress_text: string; progress_thinking: string; result_text: string | null; error_text: string | null;
  steps: number; output_tokens: number; scheduled: boolean; created_at: number; updated_at: number; finished_at: number | null;
};
export type AssistantAction = {
  id: string; run_id: string; ordinal: number; tool_name: string; arguments_json: string; summary: string;
  status: "ready" | "pending" | "approved" | "rejected" | "executing" | "completed" | "uncertain";
  result_text: string | null; requested_at: number; expires_at: number; decided_at: number | null; finished_at: number | null;
};
export type AssistantEvent = { id: number; kind: string; text: string; data_json: string | null; created_at: number };
export type AssistantArtifact = { id: string; run_id: string; name: string; mime_type: string; size: number; created_at: number };
export type RunResponse = { run: AssistantRun; events: AssistantEvent[]; more_events: boolean; actions: AssistantAction[]; artifacts: AssistantArtifact[] };
export type TaskResponse = { task: AssistantTask; runs: AssistantRun[] };
export type SandboxStatus = { available: boolean; image: string; reason: string | null };
export type Bootstrap = { config: TaskConfig; storage: "server"; preferred_model: string };
export type TaskDraft = {
  start_request_id?: string;
  request_id: string; title: string; instructions: string; config: TaskConfig;
  interval_seconds: number | null; next_run_at: number | null; schedule_enabled: boolean; expected_revision?: number;
};
export type ReplyDraft = { text: string; request_id: string; question_id?: string };
export type ConversationJob = { id: string; title: string; run_id: string | null; status: RunStatus | null; result: string | null; error: string | null; schedule_enabled: boolean; next_run_at: number | null; schedule_error: string | null; created_at: number };
export type ConversationTurnSummary = Pick<AssistantRun, "id" | "task_id" | "status" | "created_at" | "updated_at" | "finished_at">;
export type ConversationPage = { task: AssistantTask; runs: ConversationTurnSummary[]; jobs: ConversationJob[]; more: boolean; more_jobs?: boolean; requires_review?: boolean };
