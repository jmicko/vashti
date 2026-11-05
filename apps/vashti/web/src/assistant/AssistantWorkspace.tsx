import { useEffect, useRef, useState } from "react";
import { Archive, ArrowLeft, Check, Clock, Download, FileText, LoaderCircle, Pause, Play, Plus, RefreshCw, Search, Send, Settings2, Square, Trash2, Undo2 } from "lucide-react";
import { requestBlob, requestJson } from "../api";
import { ActionMenu } from "../ActionMenu";
import { ConfirmDialog } from "../common";
import { MarkdownContent as SharedMarkdownContent } from "../MarkdownContent";
import { saveFile } from "../saveFile";
import type { AppRoute, BackendModelGroup, SettingsGuard } from "../types";
import { assistantApi, activeStatuses, newRequestId, parseData, statusLabel } from "./api";
import { TaskForm } from "./TaskForm";
import { useAssistantDraft } from "./useDraft";
import { Activity, Plan, humanTool } from "./Activity";
import { useRun } from "./useRun";
import { ConversationView } from "./ConversationView";
import type { AssistantArtifact, Bootstrap, ReplyDraft, SandboxStatus, TaskResponse, TaskSummary } from "./types";

export type Props = {
  headerTarget?: HTMLDivElement | null;
  route: Extract<AppRoute, { page: "assistant" }>; groups: BackendModelGroup[];
  onNavigate: (route: AppRoute) => void; onGuardChange: (guard: SettingsGuard | null) => void;
  onManageModels: () => void; onOpenNote: (id: string) => void;
};

// Retrieved text must not trigger an outbound request through an image tag.
function MarkdownContent({ content }: { content: string }) {
  return <SharedMarkdownContent content={content} deferImages />;
}

export function AssistantWorkspace(props: Props) {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [sandbox, setSandbox] = useState<SandboxStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let current = true;
    void assistantApi.bootstrap().then(value => { if (current) { setBootstrap(value); setError(null); } }).catch(e => { if (current) setError(message(e)); });
    void requestJson<{ sandbox: SandboxStatus }>("/api/assistant/sandbox-status").then(value => { if (current) setSandbox(value.sandbox); }).catch(e => { if (current) setSandbox({ available: false, image: "", reason: message(e) }); });
    return () => { current = false; };
  }, [retry]);
  const home = () => props.onNavigate({ page: "assistant" });
  if (!bootstrap) return <section className="assistant-workspace"><div className="assistant-empty">{error ? <><p role="alert">{error}</p><button type="button" onClick={() => setRetry(v => v + 1)}><RefreshCw />Retry</button></> : <p role="status">Loading assistant...</p>}</div></section>;
  if (props.route.view === "new" || props.route.view === "defaults") return <section className="assistant-workspace assistant-scroll"><TaskForm key={props.route.view} config={bootstrap.config} defaults={props.route.view === "defaults"} groups={props.groups} sandbox={sandbox} onGuardChange={props.onGuardChange}
    onBack={home} onManageModels={props.onManageModels} onSaved={(task, runId, config) => { if (config) setBootstrap(previous => previous ? { ...previous, config } : previous); if (task) props.onNavigate({ page: "assistant", taskId: task.id, runId }); }} /></section>;
  if (props.route.taskId || props.route.runId) return <TaskDetail key={props.route.taskId ?? props.route.runId} {...props} sandbox={sandbox} />;
  if (props.route.view === "tasks") return <TaskInbox onNavigate={props.onNavigate} />;
  return <ConversationView key={props.route.conversationId ?? props.route.view ?? "home"} {...props} config={bootstrap.config} onConfig={config => setBootstrap(previous => previous ? { ...previous, config } : previous)} />;
}

function TaskInbox({ onNavigate }: Pick<Props, "onNavigate">) {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [archived, setArchived] = useState(false);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  useEffect(() => { if (search === query) return; const timer = setTimeout(() => { setQuery(search); setTasks([]); setLoading(true); }, 200); return () => clearTimeout(timer); }, [search, query]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [more, setMore] = useState(false);
  const [busyMore, setBusyMore] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const generation = useRef(0);
  useEffect(() => {
    const epoch = ++generation.current;
    let timer = 0;
    async function load() {
      try {
        const response = await requestJson<{ tasks: TaskSummary[] }>(`/api/assistant/tasks?${new URLSearchParams({ archived: String(archived), attention: String(filter === "attention"), scheduled: String(filter === "scheduled"), search: query })}`);
        if (generation.current !== epoch) return;
        setTasks(previous => [...response.tasks, ...previous.filter(t => !response.tasks.some(n => n.id === t.id) && previous.indexOf(t) >= 50)]);
        setMore(response.tasks.length === 50); setError(null);
      } catch (e) { if (generation.current === epoch) setError(message(e)); }
      finally { if (generation.current === epoch) { setLoading(false); timer = window.setTimeout(load, document.hidden ? 30000 : 8000); } }
    }
    void load();
    return () => { generation.current++; clearTimeout(timer); };
  }, [archived, filter, query, refresh]);
  async function loadMore() {
    const epoch = generation.current;
    setBusyMore(true);
    try {
      const response = await requestJson<{ tasks: TaskSummary[] }>(`/api/assistant/tasks?${new URLSearchParams({ archived: String(archived), attention: String(filter === "attention"), scheduled: String(filter === "scheduled"), search: query, offset: String(tasks.length) })}`);
      if (epoch !== generation.current) return;
      setTasks(previous => [...previous, ...response.tasks.filter(t => !previous.some(p => p.id === t.id))]); setMore(response.tasks.length === 50);
    } catch (e) { if (epoch === generation.current) setError(message(e)); }
    finally { setBusyMore(false); }
  }
  const visible = tasks.filter(t => (filter !== "attention" || t.needs_attention) && (filter !== "scheduled" || t.schedule_enabled));
  return <section className="assistant-workspace assistant-scroll"><div className="assistant-inbox">
    <div className="assistant-page-heading"><button type="button" className="icon-button" title="Back to assistant" aria-label="Back to assistant" onClick={() => onNavigate({ page: "assistant" })}><ArrowLeft /></button><h1>Tasks</h1><div className="assistant-heading-actions">
      <button type="button" className="icon-button" title="Assistant defaults" aria-label="Assistant defaults" onClick={() => onNavigate({ page: "assistant", view: "defaults" })}><Settings2 /></button>
      <button type="button" className="primary-action" onClick={() => onNavigate({ page: "assistant", view: "new" })}><Plus />New task</button>
    </div></div>
    <div className="assistant-inbox-filters"><label className="history-search"><Search size={16} /><input aria-label="Search tasks" placeholder="Search tasks" value={search} onChange={e => setSearch(e.target.value)} /></label>
      <select aria-label="Filter tasks" value={archived ? "archived" : filter} onChange={e => { const next = e.target.value; setTasks([]); setLoading(true); setArchived(next === "archived"); setFilter(next === "archived" ? "all" : next); }}><option value="all">All tasks</option><option value="attention">Needs attention</option><option value="scheduled">Scheduled</option><option value="archived">Archived</option></select>
      <button type="button" className="icon-button" aria-label="Refresh tasks" title="Refresh" onClick={() => setRefresh(v => v + 1)}><RefreshCw /></button>
    </div>
    {error && <p role="alert" className="error-message">{error}</p>}
    {loading && !tasks.length ? <p role="status">Loading tasks...</p> : !visible.length ? <div className="assistant-empty"><p>{search || filter !== "all" ? "No matching tasks" : archived ? "No archived tasks" : "No tasks yet"}</p></div> : <ul className="assistant-task-list">{visible.map(task => <li key={task.id}><button type="button" className="assistant-task-row" onClick={() => onNavigate({ page: "assistant", taskId: task.id })}>
      <span className={`assistant-task-dot${task.needs_attention ? " attention" : ""}${task.last_status === "running" ? " running" : ""}`} />
      <span className="assistant-task-copy"><strong>{task.title}</strong><span>{task.instructions}</span><small>{task.schedule_error ? `Schedule stopped: ${task.schedule_error}` : task.schedule_enabled && task.next_run_at ? `Next: ${dateTime(task.next_run_at)}` : dateTime(task.updated_at)}</small></span>
      <span className="assistant-task-status">{task.schedule_enabled && <Clock size={14} />}{statusLabel(task.last_status)}</span>
    </button></li>)}</ul>}
    {more && <button type="button" className="secondary-button" disabled={busyMore} onClick={() => void loadMore()}>{busyMore ? "Loading..." : "Load more"}</button>}
  </div></section>;
}


function TaskDetail(props: Props & { sandbox: SandboxStatus | null }) {
  const [detail, setDetail] = useState<TaskResponse | null>(null);
  const [runId, setRunId] = useState<string | null>(props.route.runId ?? null);
  const [refresh, setRefresh] = useState(0);
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<"stop" | "archive" | "uncertain" | AssistantArtifact | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fallbackRequest = useRef(newRequestId());
  const transcript = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const draft = useAssistantDraft<ReplyDraft | string>(`reply:${props.route.taskId ?? props.route.runId}`, { text: "", request_id: fallbackRequest.current });
  const follow = typeof draft.value === "string" ? { text: draft.value, request_id: fallbackRequest.current } : draft.value;
  const clearReply = () => draft.reset({ text: "", request_id: newRequestId() });
  const { data, error: runError } = useRun(runId, refresh);
  useEffect(() => {
    if (follow.question_id && data?.events.some(e => e.kind === "user" && e.text.trim() === follow.text.trim() && parseData(e.data_json).question_id === follow.question_id)) clearReply();
  }, [data?.events, follow.question_id, follow.text]);
  const currentTaskId = props.route.taskId ?? data?.run.task_id;
  useEffect(() => {
    if (!currentTaskId) return;
    let current = true;
    void assistantApi.task(currentTaskId).then(value => {
      if (!current) return;
      setDetail(value); if (!runId) setRunId(value.task.last_run_id);
      void assistantApi.post(`tasks/${currentTaskId}/viewed`, {}).catch(() => undefined);
    }).catch(e => { if (current) setError(message(e)); });
    return () => { current = false; };
  }, [currentTaskId, refresh, data?.run.status]);
  useEffect(() => { if (props.route.runId) setRunId(props.route.runId); }, [props.route.runId]);
  useEffect(() => { pinned.current = true; transcript.current?.scrollTo({ top: 0 }); }, [runId]);
  useEffect(() => {
    if (pinned.current && transcript.current) transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [data?.events.length, data?.run.progress_text, data?.run.progress_thinking, data?.actions]);
  const sourceConversation = detail?.task.kind === "conversation" ? detail.task.id : detail?.task.conversation_id;
  const goHome = () => props.onNavigate(sourceConversation ? { page: "assistant", conversationId: sourceConversation } : { page: "assistant", view: "tasks" });
  async function action(work: () => Promise<unknown>) {
    if (busy) return;
    setBusy(true); setError(null); setNotice(null);
    try { await work(); setRefresh(v => v + 1); setConfirm(null); }
    catch (e) { setError(message(e)); }
    finally { setBusy(false); }
  }
  async function start(acknowledge = false) {
    if (!detail || !draft.ready) return;
    await action(async () => {
      await draft.change(follow);
      draft.assertCurrent();
      const response = await assistantApi.start(detail.task.id, follow.request_id, follow.text, acknowledge);
      await draft.clear(); clearReply();
      setRunId(response.run.id); props.onNavigate({ page: "assistant", taskId: detail.task.id, runId: response.run.id });
    });
  }
  async function download(file: AssistantArtifact) {
    await action(async () => {
      const result = await saveFile(await requestBlob(`/api/assistant/artifacts/${encodeURIComponent(file.id)}`), file.name, file.mime_type);
      setNotice(result === "saved" ? "File saved." : result === "cancelled" ? "Save cancelled." : "Download requested.");
    });
  }
  if (!detail) return <section className="assistant-workspace"><div className="assistant-empty">{error || runError ? <><p role="alert">{error ?? runError}</p><button type="button" className="secondary-button" onClick={() => setRefresh(v => v + 1)}><RefreshCw />Retry</button><button type="button" className="secondary-button" onClick={goHome}><ArrowLeft />Tasks</button></> : <p role="status">Loading task...</p>}</div></section>;
  if (data && data.run.task_id !== detail.task.id) return <section className="assistant-workspace"><div className="assistant-empty"><p role="alert">This run belongs to a different task.</p><button type="button" className="secondary-button" onClick={() => props.onNavigate({ page: "assistant", taskId: data.run.task_id, runId: data.run.id })}>Open the matching task</button></div></section>;
  if (editing) return <section className="assistant-workspace assistant-scroll"><TaskForm key={detail.task.revision} task={detail.task} config={detail.task.config} groups={props.groups} sandbox={props.sandbox} onGuardChange={props.onGuardChange} onManageModels={props.onManageModels} onBack={() => { props.onGuardChange(null); setEditing(false); }} onSaved={() => { setEditing(false); setRefresh(v => v + 1); }} /></section>;
  const task = detail.task;
  const run = data?.run;
  const live = detail.runs.find(r => activeStatuses.has(r.status));
  const active = run && activeStatuses.has(run.status);
  const uncertain = data?.actions.some(a => a.status === "uncertain" || a.status === "executing") && !active;
  const pending = active ? data?.actions.filter(a => a.status === "pending") ?? [] : [];
  const question = pending.find(a => a.tool_name === "ask_user");
  const questionData = question ? parseData(question.arguments_json) : {};
  const changeReply = (text: string) => draft.change({ text, request_id: text === follow.text ? follow.request_id : newRequestId(), question_id: question?.id });
  const plans = data?.events.filter(e => e.kind === "plan");
  const lastPlan = plans?.[plans.length - 1];
  const choices = Array.isArray(questionData.choices) ? questionData.choices.filter((s): s is string => typeof s === "string") : [];
  return <section className="assistant-workspace assistant-detail">
    <header className="assistant-detail-heading">
      <button type="button" className="icon-button" title={sourceConversation ? "Back to conversation" : "All tasks"} aria-label={sourceConversation ? "Back to conversation" : "All tasks"} onClick={goHome}><ArrowLeft /></button>
      <div className="assistant-detail-title"><h1>{task.title}</h1><span className="assistant-caption">{task.config.model_name} · Server</span></div>
      <ActionMenu label="Task actions">
        {task.kind !== "conversation" && <button type="button" disabled={task.archived_at !== null} onClick={() => setEditing(true)}><Settings2 />Task settings</button>}
        <button type="button" disabled={Boolean(live)} onClick={() => setConfirm("archive")}>{task.archived_at ? <Undo2 /> : <Archive />}{task.archived_at ? "Restore task" : "Archive task"}</button>
      </ActionMenu>
    </header>
    <div className="assistant-run-bar">
      <span className={`assistant-run-status status-${run?.status ?? "idle"}`} role="status">{run?.status === "running" && <LoaderCircle className="assistant-spinner" size={15} />}{statusLabel(run?.status ?? null)}</span>
      {detail.runs.length > 0 && <select aria-label="Run history" value={runId ?? ""} onChange={e => { setRunId(e.target.value); props.onNavigate({ page: "assistant", taskId: task.id, runId: e.target.value }); }}>
        {detail.runs.map((r, index) => <option key={r.id} value={r.id}>{index === 0 ? "Latest · " : ""}{dateTime(r.created_at)} · {statusLabel(r.status)}</option>)}
      </select>}
      <div className="assistant-run-controls">
        {run && ["running", "queued"].includes(run.status) && <button type="button" className="icon-button" title="Pause run" aria-label="Pause run" disabled={busy} onClick={() => void action(() => assistantApi.post(`runs/${run.id}/control`, { command: "pause" }))}><Pause /></button>}
        {run && ["paused", "interrupted"].includes(run.status) && !uncertain && <button type="button" className="icon-button" title="Resume run" aria-label="Resume run" disabled={busy} onClick={() => void action(() => assistantApi.post(`runs/${run.id}/control`, { command: "resume" }))}><Play /></button>}
        {active && <button type="button" className="icon-button" title="Stop run" aria-label="Stop run" disabled={busy} onClick={() => setConfirm("stop")}><Square /></button>}
      </div>
    </div>
    {live && live.id !== runId && <button type="button" className="assistant-jump" onClick={() => { setRunId(live.id); props.onNavigate({ page: "assistant", taskId: task.id, runId: live.id }); }}>Go to current run · {statusLabel(live.status)}</button>}
    {task.schedule_error && <p role="alert" className="assistant-banner">Schedule stopped: {task.schedule_error}</p>}
    {error || runError ? <p className="assistant-banner error-message" role="alert">{error ?? runError}</p> : null}
    <div className="assistant-transcript" ref={transcript} onClick={e => {
      const anchor = e.target instanceof Element ? e.target.closest("a[href]") : null;
      if (!(anchor instanceof HTMLAnchorElement)) return;
      const url = new URL(anchor.href, location.origin);
      if (url.origin !== location.origin || !url.pathname.startsWith("/api/assistant/artifacts/")) return;
      e.preventDefault();
      const file = data?.artifacts.find(file => url.pathname === `/api/assistant/artifacts/${encodeURIComponent(file.id)}`);
      if (file) void download(file); else setError("This file is no longer available in this run.");
    }} onScroll={() => { const element = transcript.current; if (element) pinned.current = element.scrollHeight - element.clientHeight - element.scrollTop < 80; }}>
      <div className="assistant-transcript-inner">
        {lastPlan && <Plan event={lastPlan} />}
        {!runId && <article className="assistant-entry assistant-user-entry"><h2>Task</h2><MarkdownContent content={task.instructions} /></article>}
        {data && <Activity events={data.events} actions={data.actions} terminal={!activeStatuses.has(data.run.status)} onOpenNote={props.onOpenNote} />}
        {run && (run.progress_thinking || run.progress_text) && <article className="assistant-entry">
          {run.progress_thinking && <details className="assistant-thinking"><summary>Thinking</summary><MarkdownContent content={run.progress_thinking} /></details>}
          {run.progress_text && <MarkdownContent content={run.progress_text} />}
        </article>}
        {run?.error_text && <p className="error-message" role="alert">{run.error_text}</p>}
        {uncertain && <p className="assistant-banner">An action may have run before the interruption. Review its outcome before starting again; it will not be replayed automatically.</p>}
        {data && data.artifacts.length > 0 && <section className="assistant-files"><h2>Files</h2>{data.artifacts.map(file => <div className="assistant-file" key={file.id}><FileText /><span><strong>{file.name}</strong><small>{(file.size / 1024).toFixed(1)} KiB</small></span><button type="button" className="icon-button" title={`Download ${file.name}`} aria-label={`Download ${file.name}`} disabled={busy} onClick={() => void download(file)}><Download /></button><button type="button" className="message-icon-button" title={`Delete ${file.name}`} aria-label={`Delete ${file.name}`} disabled={busy} onClick={() => setConfirm(file)}><Trash2 /></button></div>)}</section>}
        {notice && <p role="status">{notice}</p>}
        {pending.filter(a => a.tool_name !== "ask_user").map(a => <section key={a.id} className="assistant-approval" aria-label="Approval request"><h2>Approval needed</h2><strong>{humanTool(a.tool_name)}</strong><pre>{JSON.stringify(parseData(a.arguments_json), null, 2)}</pre><p className="assistant-caption">This exact action only · Expires {dateTime(a.expires_at)}</p><div className="assistant-form-actions"><button type="button" className="secondary-button" disabled={busy || run?.status !== "waiting_approval"} onClick={() => void action(() => assistantApi.post(`runs/${runId}/actions/${a.id}/decision`, { approve: false }))}>Reject</button><button type="button" className="primary-action" disabled={busy || run?.status !== "waiting_approval" || a.expires_at * 1000 < Date.now()} onClick={() => void action(() => assistantApi.post(`runs/${runId}/actions/${a.id}/decision`, { approve: true }))}><Check />Approve</button></div></section>)}
        {question && <section className="assistant-question"><h2>Your reply needed</h2><MarkdownContent content={String(questionData.question ?? "")} />{choices.length > 0 && <div className="assistant-choices">{choices.map(choice => <button type="button" className="secondary-button" disabled={busy || run?.status !== "waiting_input"} key={choice} onClick={() => void changeReply(choice)}>{choice}</button>)}</div>}</section>}
        {run && <p className="assistant-run-metrics">{run.steps} steps · {run.output_tokens.toLocaleString()} output tokens{run.scheduled ? " · Scheduled" : ""}</p>}
      </div>
    </div>
    {!task.archived_at && <form className="assistant-followup" onSubmit={e => { e.preventDefault(); if (run?.status === "waiting_input" && question) void action(async () => { await draft.change({ ...follow, question_id: question.id }); await assistantApi.post(`runs/${run.id}/reply`, { text: follow.text, question_id: question.id }); await draft.clear(); clearReply(); }); else if (uncertain) setConfirm("uncertain"); else void start(); }}>
      <textarea aria-label={run?.status === "waiting_input" ? "Reply to assistant" : "Follow-up instructions"} placeholder={run?.status === "waiting_input" ? "Your reply..." : active ? "A run is in progress" : run ? "Follow-up instructions..." : "Additional instructions (optional)"} rows={2} maxLength={run?.status === "waiting_input" ? 16000 : 32000} value={follow.text} disabled={!draft.ready || busy || Boolean(live && run?.status !== "waiting_input")} onChange={e => void changeReply(e.target.value)} />
      <button type="submit" className="primary-action icon-button" title={run?.status === "waiting_input" ? "Send reply" : "Run task"} aria-label={run?.status === "waiting_input" ? "Send reply" : "Run task"} disabled={busy || !draft.ready || Boolean(live && run?.status !== "waiting_input") || (run?.status === "waiting_input" && !follow.text.trim())}><Send /></button>
      {draft.storageError && <p className="assistant-caption">{draft.storageError}</p>}
    </form>}
    {confirm && <ConfirmDialog title={typeof confirm === "object" ? "Delete file?" : confirm === "stop" ? "Stop this run?" : confirm === "uncertain" ? "Start a new run?" : task.archived_at ? "Restore task?" : "Archive task?"}
      message={typeof confirm === "object" ? `${confirm.name} will be permanently removed. Download it first if needed.` : confirm === "stop" ? "Completed actions remain recorded. An action already in progress may have an uncertain outcome." : confirm === "uncertain" ? "I have reviewed the uncertain actions. Starting again could repeat changes if the task asks for them." : "The task and its run history are retained. Archived tasks do not run on a schedule."}
      confirmLabel={typeof confirm === "object" ? "Delete file" : confirm === "stop" ? "Stop run" : confirm === "uncertain" ? "Start new run" : task.archived_at ? "Restore" : "Archive"} confirmTone={confirm === "stop" || typeof confirm === "object" ? "danger" : "primary"} isBusy={busy} onCancel={() => setConfirm(null)} onConfirm={() => {
        if (typeof confirm === "object") void action(() => requestJson(`/api/assistant/artifacts/${encodeURIComponent(confirm.id)}`, { method: "DELETE" }));
        else if (confirm === "stop") void action(() => assistantApi.post(`runs/${runId}/control`, { command: "cancel" }));
        else if (confirm === "uncertain") void start(true);
        else void action(async () => { await assistantApi.post(`tasks/${task.id}/archive`, { archived: !task.archived_at }); goHome(); });
      }} />}
  </section>;
}

const dateTime = (time: number) => new Date(time * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
const message = (e: unknown) => e instanceof Error ? e.message : "Request failed";
