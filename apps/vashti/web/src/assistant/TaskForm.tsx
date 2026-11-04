import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Play, Save, Server, ShieldCheck } from "lucide-react";
import { ModelPicker } from "../ModelPicker";
import { ConfirmDialog } from "../common";
import { modelParts, modelValue } from "../modelSelection";
import { ApiRequestError, requestJson } from "../api";
import type { BackendModelGroup, SettingsGuard } from "../types";
import { assistantApi, newRequestId } from "./api";
import type { AssistantTask, SandboxStatus, TaskConfig, TaskDraft } from "./types";
import { useAssistantDraft } from "./useDraft";

export function TaskForm({ config, task, defaults, groups, sandbox, onGuardChange, onBack, onSaved, onManageModels }: {
  config: TaskConfig; task?: AssistantTask; defaults?: boolean; groups: BackendModelGroup[]; sandbox: SandboxStatus | null;
  onGuardChange: (guard: SettingsGuard | null) => void; onBack: () => void;
  onSaved: (task?: AssistantTask, runId?: string, config?: TaskConfig) => void; onManageModels: () => void;
}) {
  const initial = useMemo<TaskDraft>(() => {
    const selected = { ...config };
    if (!selected.backend_id) {
      const group = groups.find(g => g.models.some(m => m.name === selected.model_name));
      if (group) selected.backend_id = group.backend.id;
    }
    return { request_id: newRequestId(), start_request_id: newRequestId(), title: task?.title ?? "", instructions: task?.instructions ?? "", config: selected,
      interval_seconds: task?.interval_seconds ?? null, next_run_at: task?.next_run_at ?? null,
      schedule_enabled: task?.schedule_enabled ?? false, expected_revision: task?.revision };
  }, []);
  const draft = useAssistantDraft<TaskDraft>(defaults ? "defaults" : `task:${task?.id ?? "new"}`, initial);
  const [baseline, setBaseline] = useState(JSON.stringify(initial));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [stale, setStale] = useState(false);
  const [reload, setReload] = useState(false);
  const runRequest = useRef(newRequestId());
  const form = useRef<HTMLFormElement>(null);
  const value = draft.value;
  const dirty = draft.ready && JSON.stringify(value) !== baseline;
  const saveRef = useRef<() => Promise<boolean>>(async () => false);
  const discardRef = useRef(() => {});
  saveRef.current = () => save(false);
  discardRef.current = () => { draft.reset(initial); onGuardChange(null); };
  useEffect(() => {
    onGuardChange({ isDirty: dirty, save: () => saveRef.current(), discard: () => discardRef.current() });
    return () => onGuardChange(null);
  }, [dirty, onGuardChange]);
  useEffect(() => {
    const prevent = (event: BeforeUnloadEvent) => { if (dirty) event.preventDefault(); };
    window.addEventListener("beforeunload", prevent);
    return () => window.removeEventListener("beforeunload", prevent);
  }, [dirty]);
  const change = (next: Partial<TaskDraft>) => { if (!busy) { draft.change({ ...value, ...next }); setSaved(false); } };
  const set = <K extends keyof TaskConfig>(key: K, next: TaskConfig[K]) => change({ config: { ...value.config, [key]: next } });

  async function save(run: boolean): Promise<boolean> {
    if (busy || !draft.ready || !form.current?.reportValidity()) return false;
    setBusy(true); setError(null); setStale(false);
    try {
      await draft.change(value);
      if (defaults) {
        const response = await requestJson<{ config: TaskConfig }>("/api/assistant/preferences", { method: "PUT", body: JSON.stringify(value.config) });
        await draft.clear(); setBaseline(JSON.stringify(value)); setSaved(true); onGuardChange(null);
        onSaved(undefined, undefined, response.config);
      } else {
        const result = await assistantApi.save({ ...value, title: value.title.trim() || value.instructions.trim().split(/\r?\n/)[0].slice(0, 80) }, task?.id);
        // Retrying after a lost response uses the same task and run IDs.
        let runId: string | undefined;
        if (run) runId = (await assistantApi.start(result.task.id, value.start_request_id ?? runRequest.current, "")).run.id;
        await draft.clear(); setBaseline(JSON.stringify(value)); onGuardChange(null);
        onSaved(result.task, runId);
      }
      return true;
    } catch (e) { setError(e instanceof Error ? e.message : "Could not save task"); setStale(e instanceof ApiRequestError && e.code === "task_changed"); return false; }
    finally { setBusy(false); }
  }
  const when = value.next_run_at ? localDate(value.next_run_at) : "";
  if (!draft.ready) return <p role="status">Loading draft...</p>;
  return <form ref={form} className="assistant-form" onSubmit={e => { e.preventDefault(); void save(false); }}>
    <div className="assistant-page-heading">
      <button type="button" className="icon-button" title="Back" aria-label="Back" disabled={busy} onClick={onBack}><ArrowLeft /></button>
      <h1>{defaults ? "Assistant defaults" : task ? "Task settings" : "New task"}</h1>
      <span className="assistant-storage"><Server size={14} />Server</span>
    </div>
    {!defaults && <>
      <label className="setting-field"><span>Name</span><input maxLength={160} value={value.title} placeholder="Optional" disabled={busy} onChange={e => change({ title: e.target.value })} /></label>
      <label className="setting-field"><span>Task</span><textarea className="assistant-instructions" required maxLength={32000} value={value.instructions} placeholder="What would you like done?" disabled={busy} onChange={e => change({ instructions: e.target.value })} /></label>
    </>}
    <div className="setting-field"><span id="assistant-model-label">Model</span><ModelPicker groups={groups} personas={[]} privatePersonas={[]} isLoading={false} error={null}
      value={value.config.backend_id ? modelValue(value.config.backend_id, value.config.model_name) : ""}
      onChange={selected => { const parts = modelParts(selected); if (parts) change({ config: { ...value.config, backend_id: parts.backendId, model_name: parts.modelName } }); }}
      disabled={busy} ariaLabel="Assistant model" modelUnavailable={(backend, name) => { const model = groups.find(g => g.backend.id === backend)?.models.find(m => m.name === name); return model?.capabilities?.length && !model.capabilities.includes("tools") ? "Tool support required" : undefined; }} onManageModels={onManageModels} /></div>
    <details className="assistant-options" open={defaults || undefined}><summary><ShieldCheck size={17} />Access & permissions</summary>
      <div className="assistant-option-fields">
        <label className="assistant-check"><input type="checkbox" checked={value.config.notes} onChange={e => set("notes", e.target.checked)} />Notes</label>
        <label className="assistant-check"><input type="checkbox" checked={value.config.memories} onChange={e => set("memories", e.target.checked)} />Memories</label>
        <label className="assistant-check"><input type="checkbox" checked={value.config.chat_history} onChange={e => set("chat_history", e.target.checked)} />Search past server chats</label>
        <label className="setting-field"><span>Note & memory changes</span><select value={value.config.writes} onChange={e => set("writes", e.target.value as TaskConfig["writes"])}><option value="block">Blocked</option><option value="ask">Ask each time</option><option value="allow">Allow without asking</option></select></label>
        <label className="setting-field"><span>Web searches & page reads</span><select value={value.config.web} onChange={e => set("web", e.target.value as TaskConfig["web"])}><option value="block">Blocked</option><option value="ask">Ask each time</option><option value="allow">Allow without asking</option></select></label>
        <label className="assistant-check"><input type="checkbox" checked={value.config.sandbox} disabled={!sandbox?.available && !value.config.sandbox} onChange={e => set("sandbox", e.target.checked)} />Isolated code workspace</label>
        {!sandbox?.available && <p className="assistant-caption">{sandbox?.reason ?? "Checking workspace availability..."}</p>}
        <p className="assistant-caption">Existing note, memory, model, and tool permissions still apply. Device-only data is not available.</p>
      </div>
    </details>
    {!defaults && <details className="assistant-options"><summary>Schedule</summary><div className="assistant-option-fields">
      <label className="assistant-check"><input type="checkbox" checked={value.schedule_enabled} onChange={e => change({ schedule_enabled: e.target.checked, next_run_at: value.next_run_at ?? Math.floor(Date.now() / 1000) + 3600 })} />Run on a schedule</label>
      {value.schedule_enabled && <>
        <label className="setting-field"><span>First run ({Intl.DateTimeFormat().resolvedOptions().timeZone})</span><input type="datetime-local" required value={when} onChange={e => change({ next_run_at: e.target.value ? Math.floor(new Date(e.target.value).getTime() / 1000) : null })} /></label>
        <label className="setting-field"><span>Repeat</span><select value={value.interval_seconds ?? ""} onChange={e => change({ interval_seconds: e.target.value ? Number(e.target.value) : null })}><option value="">Once</option>{[300, 900, 3600, 21600, 86400, 604800, 2592000].map(seconds => <option value={seconds} key={seconds}>{seconds < 3600 ? `Every ${seconds / 60} minutes` : seconds < 86400 ? `Every ${seconds / 3600} hours` : `Every ${seconds / 86400} days`}</option>)}{value.interval_seconds && ![300,900,3600,21600,86400,604800,2592000].includes(value.interval_seconds) && <option value={value.interval_seconds}>Every {value.interval_seconds} seconds</option>}</select></label>
        <p className="assistant-caption">Runs while the server is on. Missed intervals are skipped, not replayed. Approvals wait for you.</p>
      </>}
    </div></details>}
    <details className="assistant-options"><summary>Instructions & limits</summary><div className="assistant-option-fields">
      <label className="setting-field"><span>Assistant instructions</span><textarea maxLength={16000} value={value.config.instructions} onChange={e => set("instructions", e.target.value)} /></label>
      <label className="assistant-check"><input type="checkbox" checked={value.config.reasoning} onChange={e => set("reasoning", e.target.checked)} />Reasoning, when supported</label>
      <div className="assistant-field-pair"><label className="setting-field"><span>Maximum steps</span><input type="number" min={1} max={64} required value={value.config.max_steps} onChange={e => set("max_steps", Number(e.target.value))} /></label>
        <label className="setting-field"><span>Output token budget</span><input type="number" min={256} max={131072} step={1} required value={value.config.max_output_tokens} onChange={e => set("max_output_tokens", Number(e.target.value))} /></label></div>
    </div></details>
    {task && <p className="assistant-caption">Changes apply to the next run, not a run already in progress.</p>}
    {draft.storageError && <p role="status" className="assistant-caption">{draft.storageError}</p>}
    {error && <p role="alert" className="error-message">{error}</p>}
    {stale && task && <button type="button" className="secondary-button" onClick={() => setReload(true)}>Reload saved task</button>}
    {saved && <p role="status">Defaults saved.</p>}
    <div className="assistant-form-actions">
      <button type="submit" className={defaults ? "primary-action" : "secondary-button"} disabled={busy || !value.config.backend_id}><Save />{busy ? "Saving..." : defaults ? "Save defaults" : "Save task"}</button>
      {!defaults && !task && <button type="button" className="primary-action" disabled={busy || !value.instructions.trim() || !value.config.backend_id} onClick={() => void save(true)}><Play />Create & run</button>}
    </div>
    {reload && <ConfirmDialog title="Reload saved task?" message="This discards the local draft and loads the latest saved settings. Your current draft has not been applied." confirmLabel="Reload" confirmTone="primary" onCancel={() => setReload(false)} onConfirm={() => void draft.clear().then(() => { onGuardChange(null); onSaved(task); })} />}
  </form>;
}

function localDate(seconds: number) {
  const date = new Date(seconds * 1000);
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}
