import { requestJson } from "../api";
import type { AssistantTask, AssistantRun, Bootstrap, RunResponse, TaskResponse, TaskDraft } from "./types";

export const assistantApi = {
  bootstrap: () => requestJson<Bootstrap>("/api/assistant"),
  task: (id: string) => requestJson<TaskResponse>(`/api/assistant/tasks/${encodeURIComponent(id)}`),
  run: (id: string, after = 0, actionsAfter = -1) => requestJson<RunResponse>(`/api/assistant/runs/${encodeURIComponent(id)}?after=${after}&actions_after=${actionsAfter}`),
  save: (draft: TaskDraft, id?: string) => {
    const { start_request_id: _localRunId, ...payload } = draft;
    return requestJson<{ task: AssistantTask }>(`/api/assistant/tasks${id ? `/${encodeURIComponent(id)}` : ""}`, { method: id ? "PUT" : "POST", body: JSON.stringify(payload) });
  },
  start: (id: string, request_id: string, message: string, acknowledge_uncertain = false) => requestJson<{ run: AssistantRun }>(`/api/assistant/tasks/${encodeURIComponent(id)}/runs`, { method: "POST", body: JSON.stringify({ request_id, message, acknowledge_uncertain }) }),
  post: (path: string, body: unknown) => requestJson(`/api/assistant/${path}`, { method: "POST", body: JSON.stringify(body) })
};

export function newRequestId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, n => n.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export const activeStatuses = new Set(["queued", "running", "waiting_approval", "waiting_input", "paused"]);
export const statusLabel = (value: string | null) => ({ waiting_approval: "Approval needed", waiting_input: "Your reply needed", limit_reached: "Limit reached", interrupted: "Interrupted", cancelled: "Stopped" }[value ?? ""] ?? (value ? value[0].toUpperCase() + value.slice(1) : "Not started"));
export const parseData = (value: string | null): Record<string, unknown> => {
  try { const parsed: unknown = JSON.parse(value ?? "{}"); return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {}; }
  catch { return {}; }
};
