import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Archive, ArrowLeft, Check, Clock, Download, History, ListTodo, LoaderCircle, MessageSquarePlus, Pause, Play, Settings2, Square } from "lucide-react";
import { requestBlob, requestJson } from "../api";
import { ActionMenu } from "../ActionMenu";
import { StartChatComposer } from "../Composer";
import { MarkdownContent } from "../MarkdownContent";
import { ModelPicker } from "../ModelPicker";
import { modelParts, modelValue } from "../modelSelection";
import { saveFile } from "../saveFile";
import type { Props } from "./AssistantWorkspace";
import { Activity } from "./Activity";
import { useRun } from "./useRun";
import { assistantApi, activeStatuses, newRequestId, parseData, statusLabel } from "./api";
import { useAssistantDraft } from "./useDraft";
import type { AssistantRun, ConversationJob, ConversationPage, TaskConfig } from "./types";

type ConversationDraft = { text: string; request_id: string; create_id: string; conversation_id?: string; create_title?: string; create_config?: TaskConfig };
type ConversationSummary = { id: string; title: string; updated_at: number };
const errorMessage = (e: unknown) => e instanceof Error ? e.message : "Request failed";

export function ConversationView(props: Props & { config: TaskConfig; onConfig: (config: TaskConfig) => void }) {
  const id = props.route.conversationId;
  const [page, setPage] = useState<ConversationPage | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[] | null>(null);
  const [archived, setArchived] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [olderBusy, setOlderBusy] = useState(false);
  const [reviewed, setReviewed] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const transcript = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const sent = useRef(false);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const initialDraft = useMemo(() => ({ text: "", request_id: newRequestId(), create_id: newRequestId() }), []);
  const draft = useAssistantDraft<ConversationDraft>(`conversation:${id ?? "new"}`, initialDraft);
  const config = useMemo(() => {
    if (props.config.backend_id) return props.config;
    const group = props.groups.find(g => g.models.some(m => m.name === props.config.model_name));
    return { ...props.config, backend_id: group?.backend.id ?? "" };
  }, [props.config, props.groups]);
  useEffect(() => {
    let current = true;
    void requestJson<{ conversations: ConversationSummary[] }>(`/api/assistant/conversations?archived=${archived}`).then(value => {
      if (!current) return;
      setConversations(value.conversations);
    }).catch(e => { if (current) setError(errorMessage(e)); });
    return () => { current = false; };
  }, [refresh, archived]);
  useEffect(() => {
    if (!id && !props.route.view && conversations?.length && draft.ready && !draft.value.text) props.onNavigate({ page: "assistant", conversationId: conversations[0].id });
  }, [id, conversations, draft.ready]);
  useEffect(() => {
    if (!id) return;
    let current = true, timer = 0;
    async function poll() {
      try {
        const response = await requestJson<ConversationPage>(`/api/assistant/conversations/${encodeURIComponent(id!)}`);
        if (!current) return;
        setPage(previous => {
          const older = previous?.runs.filter(r => !response.runs.some(n => n.id === r.id)) ?? [];
          return { ...response, runs: [...older, ...response.runs], more: older.length ? previous!.more : response.more };
        });
        setError(null);
      } catch (e) { if (current) setError(errorMessage(e)); }
      finally { if (current) timer = window.setTimeout(poll, document.hidden ? 15000 : 2000); }
    }
    void poll();
    return () => { current = false; clearTimeout(timer); };
  }, [id, refresh]);
  useEffect(() => {
    const element = transcript.current;
    if (!element) return;
    // Observe content growth, including expanded thinking, without a second scroller.
    const observer = new ResizeObserver(() => { if (pinned.current) element.scrollTop = element.scrollHeight; });
    if (element.firstElementChild) observer.observe(element.firstElementChild);
    return () => observer.disconnect();
  }, [id, Boolean(page)]);
  const current = page?.runs.find(r => activeStatuses.has(r.status));
  useEffect(() => { setReviewed(false); }, [page?.task.last_run_id, page?.requires_review]);
  async function changeConfig(next: TaskConfig) {
    setBusy(true); setError(null);
    try { const saved = await requestJson<{ config: TaskConfig }>("/api/assistant/preferences", { method: "PUT", body: JSON.stringify(next) }); props.onConfig(saved.config); }
    catch (e) { setError(errorMessage(e)); }
    finally { setBusy(false); }
  }
  async function send(text: string) {
    if (busy || sent.current || !draft.ready || current) return false;
    sent.current = true; setBusy(true); setError(null);
    try {
      draft.assertCurrent();
      let conversation = id ?? draft.value.conversation_id;
      await draft.change(draft.value);
      draft.assertCurrent();
      if (!conversation) {
        const creation = { ...draft.value, create_title: draft.value.create_title ?? text.trim().split(/\r?\n/)[0].slice(0, 80), create_config: draft.value.create_config ?? config };
        await draft.change(creation);
        draft.assertCurrent();
        const response = await requestJson<{ task: { id: string } }>("/api/assistant/conversations", { method: "POST", body: JSON.stringify({ request_id: creation.create_id, title: creation.create_title, config: creation.create_config }) });
        conversation = response.task.id;
        draft.assertCurrent();
        await draft.change({ ...creation, conversation_id: conversation });
      }
      draft.assertCurrent();
      const response = await assistantApi.start(conversation, draft.value.request_id, text, reviewed);
      draft.assertCurrent();
      setPage(previous => previous && !previous.runs.some(r => r.id === response.run.id) ? { ...previous, runs: [...previous.runs, response.run], task: { ...previous.task, last_run_id: response.run.id } } : previous);
      await draft.clear();
      if (!mounted.current) return true;
      pinned.current = true;
      if (!id) props.onNavigate({ page: "assistant", conversationId: conversation });
      else { draft.reset({ text: "", request_id: newRequestId(), create_id: newRequestId() }); setRefresh(v => v + 1); }
      return true;
    } catch (e) { setError(errorMessage(e)); return false; }
    finally { sent.current = false; setBusy(false); }
  }
  async function older() {
    if (!id || !page?.runs.length || olderBusy) return;
    setOlderBusy(true);
    try {
      const element = transcript.current, height = element?.scrollHeight ?? 0, top = element?.scrollTop ?? 0;
      const response = await requestJson<ConversationPage>(`/api/assistant/conversations/${encodeURIComponent(id)}?before=${encodeURIComponent(page.runs[0].id)}`);
      setPage(previous => previous ? { ...previous, more: response.more, runs: [...response.runs.filter(r => !previous.runs.some(p => p.id === r.id)), ...previous.runs] } : response);
      requestAnimationFrame(() => { if (element) element.scrollTop = top + element.scrollHeight - height; });
    } catch (e) { setError(errorMessage(e)); }
    finally { setOlderBusy(false); }
  }
  const home = () => props.onNavigate({ page: "assistant" });
  if (props.route.view === "conversations") return <section className="assistant-workspace assistant-scroll"><div className="assistant-inbox"><div className="assistant-page-heading"><button type="button" className="icon-button" title="Back to assistant" aria-label="Back to assistant" onClick={home}><ArrowLeft /></button><h1>Conversations</h1><button type="button" className="icon-button" title={archived ? "Active conversations" : "Archived conversations"} aria-label={archived ? "Active conversations" : "Archived conversations"} aria-pressed={archived} onClick={() => { setConversations(null); setArchived(v => !v); }}><Archive /></button><button type="button" className="icon-button" title="New conversation" aria-label="New conversation" onClick={() => props.onNavigate({ page: "assistant", view: "compose" })}><MessageSquarePlus /></button></div>{error && <p role="alert">{error}</p>}{!conversations ? <p role="status">Loading conversations...</p> : !conversations.length ? <p>No {archived ? "archived " : ""}conversations</p> : <ul className="assistant-task-list">{conversations.map(c => <li key={c.id}><button type="button" className="assistant-task-row" onClick={() => props.onNavigate({ page: "assistant", conversationId: c.id })}><span className="assistant-task-copy"><strong>{c.title}</strong><small>{new Date(c.updated_at * 1000).toLocaleString()}</small></span></button></li>)}</ul>}</div></section>;
  const header = <header className="assistant-conversation-heading">
      <ModelPicker groups={props.groups} personas={[]} privatePersonas={[]} isLoading={false} error={null} value={config.backend_id ? modelValue(config.backend_id, config.model_name) : ""} ariaLabel="Assistant model" disabled={busy} onManageModels={props.onManageModels}
        modelUnavailable={(backend, name) => { const model = props.groups.find(g => g.backend.id === backend)?.models.find(m => m.name === name); return model?.capabilities?.length && !model.capabilities.includes("tools") ? "Tool support required" : undefined; }}
        onChange={value => { const model = modelParts(value); if (model) void changeConfig({ ...config, backend_id: model.backendId, model_name: model.modelName }); }} />
      <span className="assistant-caption assistant-server-label">Server</span>
      <button type="button" className="icon-button" title="Tasks & schedules" aria-label="Tasks and schedules" onClick={() => props.onNavigate({ page: "assistant", view: "tasks" })}><ListTodo /></button>
      <ActionMenu label="Assistant options"><button type="button" onClick={() => props.onNavigate({ page: "assistant", view: "compose" })}><MessageSquarePlus />New conversation</button><button type="button" onClick={() => props.onNavigate({ page: "assistant", view: "conversations" })}><History />Conversations</button><button type="button" onClick={() => props.onNavigate({ page: "assistant", view: "defaults" })}><Settings2 />Assistant settings</button>{id && <button type="button" disabled={busy || Boolean(current)} onClick={() => { setBusy(true); void assistantApi.post(`tasks/${id}/archive`, { archived: !page?.task.archived_at }).then(home).catch(e => setError(errorMessage(e))).finally(() => setBusy(false)); }}><Archive />{page?.task.archived_at ? "Restore conversation" : "Archive conversation"}</button>}</ActionMenu>
    </header>;
  return <section className="assistant-workspace assistant-detail assistant-conversation chat-view">
    {props.headerTarget ? createPortal(header, props.headerTarget) : header}
    {error && <p className="assistant-banner error-message" role="alert">{error}</p>}
    <div className="assistant-transcript" ref={transcript} onClickCapture={e => { if (e.target instanceof Element && e.target.closest("summary")) pinned.current = false; }} onScroll={() => { const element = transcript.current; if (element) pinned.current = element.scrollHeight - element.clientHeight - element.scrollTop < 80; }}>
      <div className="assistant-transcript-inner">
        {id && !page ? <p role="status">Loading conversation...</p> : <>
          {page?.more && <button type="button" className="secondary-button" disabled={olderBusy} onClick={() => void older()}>{olderBusy ? "Loading..." : "Earlier messages"}</button>}
          {!page?.runs.length && <div className="assistant-chat-empty"><h1>What can I help with?</h1></div>}
          {page?.runs.map(run => <ConversationTurn key={run.id} run={run} onOpenNote={props.onOpenNote} />)}
          {Boolean(page?.jobs.length) && <section className="assistant-background-results" aria-label="Background work">{page!.more_jobs && <button type="button" className="secondary-button" onClick={() => props.onNavigate({ page: "assistant", view: "tasks" })}>Earlier task activity</button>}{page!.jobs.map(job => <BackgroundJob key={job.run_id ?? job.id} job={job} onOpenNote={props.onOpenNote} onOpen={() => props.onNavigate({ page: "assistant", taskId: job.id, runId: job.run_id ?? undefined })} />)}</section>}
        </>}
      </div>
    </div>
    <div className="assistant-chat-composer">
      {page?.task.archived_at ? <button type="button" className="secondary-button" disabled={busy} onClick={() => void assistantApi.post(`tasks/${id}/archive`, { archived: false }).then(() => setRefresh(v => v + 1)).catch(e => setError(errorMessage(e)))}>Restore conversation</button> : <>
      {page?.requires_review && <label className="assistant-check assistant-caption"><input type="checkbox" checked={reviewed} onChange={e => setReviewed(e.target.checked)} />I reviewed the interrupted actions. A new response could repeat a change.</label>}
      {current && <div className="assistant-current-turn" role="status"><LoaderCircle size={14} className={current.status === "running" ? "assistant-spinner" : undefined} />{statusLabel(current.status)}<button type="button" className="message-icon-button" aria-label="Stop response" title="Stop response" onClick={() => void assistantApi.post(`runs/${current.id}/control`, { command: "cancel" }).then(() => setRefresh(v => v + 1)).catch(e => setError(errorMessage(e)))}><Square size={16} /></button></div>}
      <StartChatComposer isBusy={busy || Boolean(current)} isDisabled={!draft.ready || !config.backend_id || Boolean(id && !page) || Boolean(page?.requires_review && !reviewed)} placeholder={config.backend_id ? "Message Assistant" : "Choose a model"} autoFocusOnReady={false}
        draft={{ text: draft.value.text, onChange: text => { if (!busy) void draft.change({ ...draft.value, text, request_id: text === draft.value.text ? draft.value.request_id : newRequestId() }); } }} onSubmit={send} />
      {draft.storageError && <p className="assistant-caption" role="status">{draft.storageError}</p>}
      </>}
    </div>
  </section>;
}

function BackgroundJob({ job, onOpenNote, onOpen }: { job: ConversationJob; onOpenNote: Props["onOpenNote"]; onOpen: () => void }) {
  const element = useRef<HTMLElement>(null);
  const needsInput = job.status === "waiting_input" || job.status === "waiting_approval";
  const [expanded, setExpanded] = useState(needsInput);
  useEffect(() => { if (needsInput) setExpanded(true); }, [needsInput]);
  useEffect(() => {
    if (!element.current || !job.status || activeStatuses.has(job.status)) return;
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { void assistantApi.post(`tasks/${job.id}/viewed`, {}).catch(() => undefined); observer.disconnect(); }
    }, { root: element.current.closest(".assistant-transcript"), threshold: .1 });
    observer.observe(element.current);
    return () => observer.disconnect();
  }, [job.run_id, job.status]);
  return <article ref={element} className="assistant-background-job">
    <div className="assistant-job-heading"><button type="button" className="assistant-job-toggle" aria-expanded={expanded} onClick={() => setExpanded(v => !v)}>{job.schedule_enabled ? <Clock size={17} /> : <ListTodo size={17} />}<span><strong>{job.title}</strong><small>{job.status ? statusLabel(job.status) : "Scheduled"}{job.next_run_at && job.schedule_enabled ? ` · ${new Date(job.next_run_at * 1000).toLocaleString()}` : ""}</small></span></button><button type="button" className="icon-button" aria-label={`Open task: ${job.title}`} title="Task details" onClick={onOpen}><Settings2 /></button></div>
    {job.schedule_error && <p role="alert">{job.schedule_error}</p>}
    {expanded && job.run_id ? <ConversationTurn key={job.run_id} run={{ id: job.run_id, status: job.status ?? undefined }} onOpenNote={onOpenNote} background /> : job.result ? <div className="assistant-job-preview"><MarkdownContent content={job.result} deferImages /></div> : job.error ? <p role="alert">{job.error}</p> : null}
  </article>;
}

function ConversationTurn({ run, onOpenNote, background = false }: { run: Pick<AssistantRun,"id"> & Partial<AssistantRun>; onOpenNote: Props["onOpenNote"]; background?: boolean }) {
  const [refresh, setRefresh] = useState(0);
  const { data, error } = useRun(run.id, refresh);
  useEffect(() => { if (data && run.status && data.run.status !== run.status && !activeStatuses.has(data.run.status) && activeStatuses.has(run.status)) setRefresh(v => v + 1); }, [run.status, data?.run.status]);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const reply = useAssistantDraft<string>(`question:${run.id}`, "");
  const current = data?.run ?? run;
  const active = activeStatuses.has(current.status ?? "");
  async function action(work: () => Promise<unknown>) {
    if (busy) return;
    setBusy(true); setFailure(null);
    try { await work(); setRefresh(v => v + 1); }
    catch (e) { setFailure(errorMessage(e)); }
    finally { setBusy(false); }
  }
  async function download(id: string) {
    const file = data?.artifacts.find(f => f.id === id);
    if (!file) { setFailure("This file is no longer available in this response."); return; }
    await action(async () => { const result = await saveFile(await requestBlob(`/api/assistant/artifacts/${encodeURIComponent(file.id)}`), file.name, file.mime_type); setNotice(result === "saved" ? "File saved." : result === "cancelled" ? "Save cancelled." : "Download requested."); });
  }
  return <div className="assistant-conversation-turn" onClick={e => {
    const anchor = e.target instanceof Element ? e.target.closest("a[href]") : null;
    if (!(anchor instanceof HTMLAnchorElement)) return;
    const url = new URL(anchor.href, location.origin);
    if (url.origin === location.origin && url.pathname.startsWith("/api/assistant/artifacts/")) {
      e.preventDefault();
      const file = data?.artifacts.find(file => url.pathname === `/api/assistant/artifacts/${encodeURIComponent(file.id)}`);
      if (file) void download(file.id); else setFailure("This file is no longer available in this response.");
    }
  }}>
    {!data ? <p role="status">{error ?? "Loading message..."}</p> : <>
      <Activity events={background ? data.events.filter(e => e.kind !== "user") : data.events} actions={data.actions} terminal={!active} onOpenNote={onOpenNote} />
      {(current.progress_thinking || current.progress_text) && <article className="assistant-entry"><h2>Assistant</h2>{current.progress_thinking && <details className="assistant-thinking"><summary>Thinking</summary><MarkdownContent content={current.progress_thinking} deferImages /></details>}{current.progress_text && <MarkdownContent content={current.progress_text} deferImages />}</article>}
      {current.error_text && <p className="error-message" role="alert">{current.error_text}</p>}
      {data.actions.some(a => a.status === "uncertain") && <p role="alert">An interrupted action may already have run. Review its task activity before starting again; it will not be replayed automatically.</p>}
      {data.artifacts.map(file => <button type="button" className="assistant-file-download" key={file.id} disabled={busy} onClick={() => void download(file.id)}><Download size={16} /><span>{file.name}</span></button>)}
      {active && data.actions.filter(a => a.status === "pending").map(a => a.tool_name === "ask_user" ? <form key={a.id} className="assistant-question" onSubmit={e => { e.preventDefault(); void action(async () => { await assistantApi.post(`runs/${run.id}/reply`, { text: reply.value, question_id: a.id }); await reply.clear(); reply.reset(""); }); }}><h2>Your reply needed</h2><MarkdownContent content={String(parseData(a.arguments_json).question ?? "")} deferImages /><textarea aria-label="Reply to assistant question" value={reply.value} maxLength={16000} disabled={!reply.ready || busy} onChange={e => void reply.change(e.target.value)} /><button type="submit" className="primary-action" disabled={busy || !reply.value.trim() || current.status !== "waiting_input"}>Reply</button></form> : <section key={a.id} className="assistant-approval" aria-label="Approval request"><h2>Approval needed</h2><strong>{a.tool_name.replace(/_/g, " ")}</strong><pre>{JSON.stringify(parseData(a.arguments_json), null, 2)}</pre><div className="assistant-form-actions"><button type="button" className="secondary-button" disabled={busy || current.status !== "waiting_approval"} onClick={() => void action(() => assistantApi.post(`runs/${run.id}/actions/${a.id}/decision`, { approve: false }))}>Reject</button><button type="button" className="primary-action" disabled={busy || current.status !== "waiting_approval" || a.expires_at * 1000 < Date.now()} onClick={() => void action(() => assistantApi.post(`runs/${run.id}/actions/${a.id}/decision`, { approve: true }))}><Check />Approve</button></div></section>)}
      {active && background && <div className="assistant-form-actions"><span className="assistant-caption">{statusLabel(current.status ?? null)}</span><button type="button" className="icon-button" aria-label="Stop background task" title="Stop task" disabled={busy} onClick={() => void action(() => assistantApi.post(`runs/${run.id}/control`, { command: "cancel" }))}><Square /></button></div>}
      {current.status === "running" && !background && <button type="button" className="message-icon-button" aria-label="Pause response" title="Pause response" disabled={busy} onClick={() => void action(() => assistantApi.post(`runs/${run.id}/control`, { command: "pause" }))}><Pause /></button>}
      {["paused", "interrupted"].includes(current.status ?? "") && !data.actions.some(a => ["executing", "uncertain"].includes(a.status)) && <button type="button" className="secondary-button" disabled={busy} onClick={() => void action(() => assistantApi.post(`runs/${run.id}/control`, { command: "resume" }))}><Play />Resume</button>}
    </>}
    {(failure || error) && <p className="error-message" role="alert">{failure ?? error}</p>}{notice && <p role="status" className="assistant-caption">{notice}</p>}
  </div>;
}
