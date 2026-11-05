import { memo } from "react";
import { Check, FileText, LoaderCircle } from "lucide-react";
import { MarkdownContent } from "../MarkdownContent";
import { ToolRecordLinks } from "../ToolRecordLinks";
import { noteToolReferencesFromSegments } from "../notes/toolReferences";
import type { ToolUsageRecord } from "../types";
import { activityEntries } from "./activity";
import { parseData, statusLabel } from "./api";
import type { AssistantAction, AssistantEvent } from "./types";

export const humanTool = (name: string) => name.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());
const pretty = (text: string) => { try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; } };

export const Activity = memo(function Activity({ events, actions, terminal, onOpenNote }: { events: AssistantEvent[]; actions: AssistantAction[]; terminal: boolean; onOpenNote: (id: string) => void }) {
  return <>{activityEntries(events, actions).map(entry => entry.action ? <ToolAction key={entry.key} action={entry.action} terminal={terminal} onOpenNote={onOpenNote} /> : <EventEntry key={entry.key} event={entry.event!} />)}</>;
});

const EventEntry = memo(function EventEntry({ event }: { event: AssistantEvent }) {
  if (event.kind === "status" || event.kind === "decision") return event.text ? <p className="assistant-activity-line">{event.kind === "status" ? statusLabel(event.text) : event.text}</p> : null;
  const thinking = parseData(event.data_json).thinking;
  return <article className={`assistant-entry${event.kind === "user" ? " assistant-user-entry" : ""}`}>
    {(event.text.trim() || event.kind === "user") && <h2>{event.kind === "user" ? "You" : "Assistant"}</h2>}
    {typeof thinking === "string" && thinking && <details className="assistant-thinking"><summary>Thinking</summary><MarkdownContent content={thinking} deferImages /></details>}
    {event.text && <MarkdownContent content={event.text} deferImages />}
  </article>;
});

const ToolAction = memo(function ToolAction({ action, terminal, onOpenNote }: { action: AssistantAction; terminal: boolean; onOpenNote: (id: string) => void }) {
  const usage: ToolUsageRecord = { name: action.tool_name, summary: action.summary, arguments: parseData(action.arguments_json), result: action.result_text ?? "" };
  const notes = noteToolReferencesFromSegments([{ type: "tool", usage }]);
  return <details className={`assistant-tool status-${action.status}`}><summary><span>{humanTool(action.tool_name)}</span><small>{terminal && ["pending", "ready", "approved"].includes(action.status) ? "Not executed" : action.status === "executing" ? "Working" : action.status === "pending" ? "Waiting" : statusLabel(action.status)}</small></summary>
    <ToolRecordLinks usage={usage} />
    {notes.length > 0 && <nav className="tool-record-links" aria-label="Referenced notes">{notes.map(n => <button type="button" key={n.noteId} onClick={() => onOpenNote(n.noteId)}><FileText size={16} />{n.title}</button>)}</nav>}
    <h3>Arguments</h3><pre>{JSON.stringify(usage.arguments, null, 2)}</pre>
    {action.result_text && <><h3>Result</h3><pre>{pretty(action.result_text)}</pre></>}
  </details>;
});

export function Plan({ event }: { event: AssistantEvent }) {
  const steps = parseData(event.data_json).steps;
  if (!Array.isArray(steps)) return null;
  return <details className="assistant-plan" open><summary>Plan</summary><ol>{steps.map((step: unknown, index) => {
    if (!step || typeof step !== "object" || !("label" in step)) return null;
    const s = step as { label: string; status?: string };
    return <li key={index}>{s.status === "completed" ? <Check size={16} /> : s.status === "in_progress" ? <LoaderCircle size={16} /> : <span className="assistant-step-dot" />}<span>{s.label}</span></li>;
  })}</ol></details>;
}
