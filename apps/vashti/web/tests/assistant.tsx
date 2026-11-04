// Synthetic, fully intercepted browser fixture. No live model or account requests.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AssistantWorkspace } from "../src/assistant/AssistantWorkspace";
import { setPrivateStorageUser } from "../src/privateChatStore";
import { UnsavedSettingsDialog } from "../src/settingsControls";
import type { AppRoute, BackendModelGroup, SettingsGuard } from "../src/types";
import type { AssistantAction, AssistantArtifact, AssistantEvent, AssistantRun, AssistantTask, TaskConfig } from "../src/assistant/types";
import "../src/styles.css";

const now = Math.floor(Date.now() / 1000);
const config: TaskConfig = { backend_id: "fixture", model_name: "kuno-qwen3.8-flash-next:iq3_s", instructions: "", notes: true, memories: true, chat_history: false, writes: "ask", web: "ask", sandbox: false, reasoning: true, max_steps: 24, max_output_tokens: 24576 };
const groups: BackendModelGroup[] = [{ backend: { id: "fixture", name: "Kuno fixture" }, models: [{ name: config.model_name, supports_images: false, supports_thinking: true, avatar_asset_id: null, avatar_crop_x: 0, avatar_crop_y: 0, avatar_crop_size: 1, capabilities: ["completion", "tools", "thinking"] } as BackendModelGroup["models"][number]] }];
let defaults = config;
const task: AssistantTask = { id: "fixture-task", title: "Prepare a weekly garden summary", instructions: "Review the synthetic garden notes, calculate supplies, and prepare a short summary.", config, interval_seconds: 86400, next_run_at: now + 3600, schedule_enabled: true, schedule_error: null, archived_at: null, last_run_id: "fixture-run", last_status: "waiting_approval", needs_attention: true, created_at: now - 3600, updated_at: now, revision: 1 };
const tasks = [task];
const run: AssistantRun = { id: "fixture-run", task_id: task.id, instructions: task.instructions, status: "waiting_approval", progress_text: "", progress_thinking: "", result_text: null, error_text: null, steps: 3, output_tokens: 2048, scheduled: false, created_at: now - 90, updated_at: now, finished_at: null };
const runs = [run];
const events: Record<string, AssistantEvent[]> = { [run.id]: [
  { id: 1, kind: "user", text: task.instructions, data_json: "{}", created_at: now - 90 },
  { id: 2, kind: "plan", text: "", data_json: JSON.stringify({ steps: [{ label: "Review the garden note", status: "completed" }, { label: "Check quantities", status: "completed" }, { label: "Save the summary", status: "in_progress" }] }), created_at: now - 80 },
  { id: 3, kind: "assistant", text: "The garden needs **12 bags** of mulch. I have prepared a summary and am waiting for approval to save it.\n\n![Remote image fixture](https://example.invalid/must-not-load.png)", data_json: JSON.stringify({ thinking: Array(45).fill("Check the synthetic quantities and retain the original source.").join("\n\n") }), created_at: now - 40 }
] };
const actions: Record<string, AssistantAction[]> = { [run.id]: [
  { id: "read", run_id: run.id, ordinal: 0, tool_name: "read_note", arguments_json: JSON.stringify({ note_id: "synthetic-note" }), summary: "Read garden note", status: "completed", result_text: JSON.stringify({ note_id: "synthetic-note", title: "Garden plan", content: "Four beds, three bags per bed." }), requested_at: now - 75, expires_at: now + 86400, decided_at: null, finished_at: now - 74 },
  { id: "write", run_id: run.id, ordinal: 1, tool_name: "create_note", arguments_json: JSON.stringify({ title: "Garden summary", content: "Buy 12 bags of mulch." }), summary: "Create garden summary", status: "pending", result_text: null, requested_at: now - 30, expires_at: now + 86400, decided_at: null, finished_at: null }
] };
const files: Record<string, AssistantArtifact[]> = { [run.id]: [{ id: "synthetic-csv", run_id: run.id, name: "supplies.csv", mime_type: "text/csv", size: 26, created_at: now - 40 }] };
let failure = false, starts = 0, writes = 0;
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
window.fetch = async (input, options = {}) => {
  const url = new URL(String(input), location.origin), path = url.pathname;
  const body = typeof options.body === "string" ? JSON.parse(options.body) : {};
  if (path === "/api/assistant") return json({ config: defaults, storage: "server", preferred_model: config.model_name });
  if (path === "/api/assistant/sandbox-status") return json({ sandbox: { available: true, image: "python-fixture", reason: null } });
  if (path === "/api/assistant/preferences") { defaults = body; return json({ config: defaults }); }
  if (path === "/api/assistant/tasks") {
    if (options.method === "POST") {
      let saved = tasks.find(t => t.id === body.request_id);
      if (!saved) { saved = { ...task, ...body, id: body.request_id, archived_at: null, last_run_id: null, last_status: null, needs_attention: false, revision: 1 }; tasks.unshift(saved!); }
      return json({ task: saved });
    }
    const matches = tasks.filter(t => Boolean(t.archived_at) === (url.searchParams.get("archived") === "true") && (url.searchParams.get("attention") !== "true" || t.needs_attention) && (url.searchParams.get("scheduled") !== "true" || t.schedule_enabled) && `${t.title} ${t.instructions}`.toLowerCase().includes((url.searchParams.get("search") ?? "").toLowerCase()));
    return json({ tasks: matches.slice(Number(url.searchParams.get("offset")) || 0, 50) });
  }
  if (path === "/api/assistant/artifacts/synthetic-csv") {
    if (options.method === "DELETE") { files[run.id] = []; return json({ ok: true }); }
    return new Response("item,quantity\nmulch,12\n", { headers: { "Content-Type": "application/octet-stream" } });
  }
  const taskMatch = /^\/api\/assistant\/tasks\/([^/]+)(?:\/(viewed|archive|runs))?$/.exec(path);
  if (taskMatch) {
    const selected = tasks.find(t => t.id === taskMatch[1]);
    if (!selected) return json({ error: { message: "Task not found" } }, 404);
    if (taskMatch[2] === "viewed") return json({ ok: true });
    if (taskMatch[2] === "archive") { selected.archived_at = body.archived ? now : null; selected.schedule_enabled = false; return json({ ok: true }); }
    if (taskMatch[2] === "runs") {
      const existing = runs.find(r => r.id === body.request_id);
      if (existing) return json({ run: existing });
      if (failure) { failure = false; return json({ error: { message: "Synthetic connection failure; draft preserved" } }, 503); }
      starts++;
      const created: AssistantRun = { ...run, id: body.request_id, task_id: selected.id, status: "running", instructions: body.message || selected.instructions, created_at: now + starts, updated_at: now + starts, steps: 0, output_tokens: 0 };
      selected.last_run_id = created.id; selected.last_status = created.status;
      runs.unshift(created); events[created.id] = [{ id: 100 + starts, kind: "user", text: created.instructions, data_json: "{}", created_at: now + starts }]; actions[created.id] = []; files[created.id] = [];
      return json({ run: created });
    }
    if (options.method === "PUT") { Object.assign(selected, body, { revision: selected.revision + 1 }); writes++; return json({ task: selected }); }
    return json({ task: selected, runs: runs.filter(r => r.task_id === selected.id) });
  }
  const runMatch = /^\/api\/assistant\/runs\/([^/]+)(?:\/(control|reply|actions)(?:\/([^/]+)\/decision)?)?$/.exec(path);
  if (runMatch) {
    const selected = runs.find(r => r.id === runMatch[1]);
    if (!selected) return json({ error: { message: "Run not found" } }, 404);
    if (runMatch[2]) {
      if (runMatch[2] === "control") selected.status = body.command === "pause" ? "paused" : body.command === "resume" ? "running" : "cancelled";
      if (runMatch[2] === "reply") { selected.status = "running"; for (const a of actions[selected.id]) if (a.tool_name === "ask_user" && a.status === "pending") { a.status = "completed"; a.result_text = JSON.stringify({ answer: body.text }); a.finished_at = now + 5; } events[selected.id].push({ id: 900 + events[selected.id].length, kind: "user", text: body.text, data_json: "{}", created_at: now + 5 }); }
      if (runMatch[2] === "actions") {
        const action = actions[selected.id].find(a => a.id === runMatch[3])!;
        action.status = body.approve ? "completed" : "rejected"; action.result_text = JSON.stringify(body.approve ? { note_id: "synthetic-summary", title: "Garden summary" } : { error: "User rejected the action" });
        selected.status = "completed"; selected.finished_at = now + 1; selected.result_text = body.approve ? "Summary saved." : "I did not save the note.";
        events[selected.id].push({ id: 950, kind: "assistant", text: selected.result_text, data_json: "{}", created_at: now + 1 });
      }
      const t = tasks.find(t => t.id === selected.task_id)!; t.last_status = selected.status;
      return json({ ok: true });
    }
    return json({ run: selected, events: (events[selected.id] ?? []).filter(e => e.id > Number(url.searchParams.get("after") ?? 0)), more_events: false, actions: (actions[selected.id] ?? []).filter(a => a.ordinal > Number(url.searchParams.get("actions_after") ?? -1)), artifacts: files[selected.id] ?? [] });
  }
  throw new Error(`Unmocked fixture request: ${path}`);
};
setPrivateStorageUser("assistant-browser-fixture", { user_id: "assistant-browser-fixture", key_material: btoa("a".repeat(32)) });

function Fixture() {
  const [route, setRoute] = useState<Extract<AppRoute, { page: "assistant" }>>({ page: "assistant" });
  const [guard, setGuard] = useState<SettingsGuard | null>(null);
  const guardRef = useRef<SettingsGuard | null>(null);
  const [pending, setPending] = useState<AppRoute | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [notice, setNotice] = useState("");
  const onGuardChange = useCallback((next: SettingsGuard | null) => { guardRef.current = next; setGuard(next); }, []);
  function navigate(next: AppRoute) {
    if (guardRef.current?.isDirty) { setPending(next); return; }
    if (next.page === "assistant") setRoute(next);
  }
  useEffect(() => {
    const listener = () => setEpoch(n => n + 1);
    window.addEventListener("fixture-update", listener);
    return () => window.removeEventListener("fixture-update", listener);
  }, []);
  return <div style={{ height: "100dvh", display: "flex", flexDirection: "column" }}>
    <header style={{ padding: 8, display: "flex", gap: 6, flexWrap: "wrap", borderBottom: "1px solid var(--line-soft)" }}>
      <button onClick={() => navigate({ page: "assistant" })}>Tasks</button>
      <button onClick={() => { run.status = "waiting_input"; actions[run.id] = [{ ...actions[run.id][0], id: "question", tool_name: "ask_user", arguments_json: JSON.stringify({ question: "Which garden should I summarize?", choices: ["Front garden", "Back garden"] }), status: "pending", result_text: null }]; setRoute({ page: "assistant", taskId: task.id, runId: run.id }); setEpoch(n => n + 1); }}>Fixture question</button>
      <button onClick={() => { failure = true; setNotice("Next run request fails"); }}>Fail next send</button>
      <output style={{ fontSize: 11 }}>{notice} · starts {starts} · writes {writes}</output>
    </header>
    <div className="main-pane" style={{ flex: 1, minHeight: 0, gridTemplateRows: "minmax(0,1fr)" }}><AssistantWorkspace key={epoch} route={route} groups={groups} onNavigate={navigate} onGuardChange={onGuardChange} onManageModels={() => setNotice("Manage models opened")} onOpenNote={id => setNotice(`Opened note ${id}`)} /></div>
    {pending && <UnsavedSettingsDialog isSaving={false} onCancel={() => setPending(null)} onDiscard={() => { guard?.discard(); setGuard(null); if (pending.page === "assistant") setRoute(pending); setPending(null); }} onSave={() => void guard?.save().then(saved => { if (saved) { if (pending.page === "assistant") setRoute(pending); setPending(null); } })} />}
  </div>;
}
createRoot(document.getElementById("root")!).render(<Fixture />);
