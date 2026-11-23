import { useEffect, useRef, useState } from "react";
import { FileArchive, FilePenLine, LoaderCircle, Upload, X } from "lucide-react";
import { ModelPicker } from "../ModelPicker";
import { modelParts, modelValue } from "../modelSelection";
import { chatModelGroups } from "../modelEligibility";
import type { BackendModelGroup } from "../types";
import { RebuildProfileButton } from "../personalization/RebuildProfileButton";
import { ArchiveClient } from "./client";
import { warningLabels, type ArchiveSummary } from "./chatgpt";
import { listImportedDocuments, runImport, type ImportedDocument, type ImportProgress } from "./repository";

export function ImportPanel({ groups, onBusyChange,onOpenImportedDocument }: { groups: BackendModelGroup[]; onBusyChange?: (busy: boolean, cancel: () => void) => void; onOpenImportedDocument?: (chatId:string,mode:ImportedDocument["mode"])=>void }) {
  const [expanded, setExpanded] = useState(false), [summary, setSummary] = useState<ArchiveSummary | null>(null);
  const [mode, setMode] = useState<"server" | "device">("server"), [model, setModel] = useState("");
  const [phase, setPhase] = useState<"idle" | "reading" | "ready" | "importing" | "done" | "stopped">("idle");
  const [scanned, setScanned] = useState(0), [filename, setFilename] = useState("");
  const [error, setError] = useState(""), [progress, setProgress] = useState<ImportProgress | null>(null);
  const [allowSearch, setAllowSearch] = useState(false);
  const [learnFromImport, setLearnFromImport] = useState(true);
  const [documents,setDocuments]=useState<ImportedDocument[]>([]);
  const source = useRef<ArchiveClient | null>(null), controller = useRef<AbortController | null>(null), epoch = useRef(0);
  const available = chatModelGroups(groups);
  const busy = phase === "reading" || phase === "importing";
  const selected = modelParts(model);
  const replyAvailable = selected && available.some(group => group.backend.id === selected.backendId && group.models.some(item => item.name === selected.modelName));
  const total = summary ? summary.conversations+summary.workspace_pages : 0;
  useEffect(()=>{
    if(!expanded||busy)return;
    let active=true;
    void Promise.allSettled([listImportedDocuments("server"),listImportedDocuments("device")]).then(results=>{
      if(active)setDocuments(results.flatMap(r=>r.status==="fulfilled"?r.value:[]).sort((a,b)=>b.updated_at-a.updated_at||a.title.localeCompare(b.title)));
    });
    return ()=>{active=false;};
  },[expanded,busy]);
  useEffect(() => { onBusyChange?.(busy, () => { controller.current?.abort(); source.current?.close(); }); }, [busy, onBusyChange]);
  useEffect(() => () => { epoch.current++; controller.current?.abort(); source.current?.close(); window.dispatchEvent(new Event("vashti:chats-imported")); }, []);
  useEffect(() => {
    if (!busy) return;
    const unload = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", unload);
    return () => window.removeEventListener("beforeunload", unload);
  }, [busy]);
  async function inspect(file: File) {
    const request = ++epoch.current;
    controller.current?.abort(); source.current?.close();
    setError(""); setSummary(null); setProgress(null); setScanned(0); setFilename(file.name); setPhase("reading");
    setLearnFromImport(true);
    let client: ArchiveClient | null = null;
    try {
      client = new ArchiveClient(count => { if (request === epoch.current) setScanned(count); }); source.current = client;
      const result = await client.open(file);
      if (request !== epoch.current) return;
      setSummary(result); setPhase("ready");
      const first = available.find(group => group.models.length);
      if (!model && first) setModel(modelValue(first.backend.id, first.models[0].name));
    } catch (cause) { if (request === epoch.current) { setError(!client ? "The import worker could not start. Reload the app and try again." : cause instanceof Error ? cause.message : "Could not read export"); setPhase("idle"); client?.close(); source.current = null; } }
  }
  async function start() {
    const selected = modelParts(model), client = source.current;
    if (!selected || !client || !summary || busy || controller.current || !replyAvailable) return;
    const group = available.find(g => g.backend.id === selected.backendId);
    if (!group) return;
    const request = epoch.current, abort = new AbortController(); controller.current = abort;
    setPhase("importing"); setError("");
    try {
      const result = await runImport(client, { mode, backend_id: selected.backendId, backend_name: group.backend.name, model_name: selected.modelName, allow_history_search: allowSearch, learn_from_import: learnFromImport }, abort.signal, value => { if (request === epoch.current) setProgress(value); });
      if (request === epoch.current) { setProgress(result); setPhase("done"); client.close(); source.current = null; }
    } catch (cause) {
      if (request === epoch.current) { setError(abort.signal.aborted ? "Import stopped. Saved chats are kept; select the same ZIP to resume without duplicating them." : cause instanceof Error ? cause.message : "Import failed; select the ZIP again to resume"); setPhase("stopped"); client.close(); source.current = null; }
    } finally { if (controller.current === abort) controller.current = null; window.dispatchEvent(new Event("vashti:chats-imported")); }
  }
  function stop() { controller.current?.abort(); if (phase === "reading") { epoch.current++; source.current?.close(); source.current = null; setPhase("idle"); } }
  return <section className="settings-subsection chat-import-panel">
    <button type="button" className="secondary-button" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}><FileArchive /><span>Import ChatGPT data</span></button>
    {expanded && <>
      {documents.length>0&&onOpenImportedDocument&&<details><summary>Imported items ({documents.length})</summary><div className="chat-import-documents">
        {documents.map(doc=><button className="secondary-button" type="button" key={`${doc.mode}:${doc.id}`} disabled={busy} onClick={()=>onOpenImportedDocument(doc.id,doc.mode)}>
          <FilePenLine/><span>{doc.title}</span><small>{doc.mode==="device"?"This device":"Server"}</small>
        </button>)}
      </div></details>}
      <label className="field"><span>ChatGPT export ZIP</span><input type="file" accept=".zip,application/zip,application/x-zip-compressed" disabled={busy} onChange={e => { const file = e.target.files?.[0]; e.target.value = ""; if (file) void inspect(file); }} /></label>
      <fieldset disabled={busy} className="chat-import-options">
        <legend>Destination</legend><div className="segmented-control" role="group" aria-label="Import storage"><button type="button" aria-pressed={mode === "server"} onClick={() => setMode("server")}>Server</button><button type="button" aria-pressed={mode === "device"} onClick={() => setMode("device")}>This device</button></div>
        <p>{mode === "device" ? "Encrypted on this device. The import itself does not upload the ZIP or chats." : "Saved to your account on this Vashti server. The ZIP is read locally; only conversation records and their files are uploaded."}</p>
      </fieldset>
      {filename && <small className="chat-import-filename">{filename}</small>}
      {phase === "reading" && <p role="status"><LoaderCircle className="spin" />Reading export · {scanned.toLocaleString()} conversations</p>}
      {summary && <>
        <dl className="chat-import-summary"><div><dt>Conversations</dt><dd>{summary.conversations.toLocaleString()}</dd></div><div><dt>Messages</dt><dd>{summary.messages.toLocaleString()}</dd></div><div><dt>Files</dt><dd>{summary.assets.toLocaleString()}</dd></div><div><dt>File storage</dt><dd>{(summary.asset_bytes / 1024 ** 2).toFixed(1)} MiB</dd></div>{summary.workspace_pages>0&&<div><dt>Workspace Pages</dt><dd>{summary.workspace_pages.toLocaleString()}</dd></div>}</dl>
        <details><summary>Compatibility review</summary><p>{summary.branches.toLocaleString()} alternative branches. Original titles, dates, model labels, Markdown, and exported thinking are preserved.</p><p>Hidden messages and rich metadata stay in original records. PDFs can be previewed; audio, video and other binary files are download-only. Missing export files cannot be restored. Existing imports are not overwritten.</p>{summary.workspace_pages>0&&<p>Pages appear under Imported items as editable canvases. Supported checkpoints retain Markdown formatting. Earlier edit history is not reconstructed.</p>}<ul>{Object.entries(summary.warnings).filter(([, count]) => count).map(([key, count]) => <li key={key}>{warningLabels[key] ?? key}: {count.toLocaleString()}</li>)}</ul></details>
        <fieldset disabled={busy || phase === "done" || phase === "stopped"} className="chat-import-options">
          <legend>Import options</legend>
          <label><span>Model for future replies</span><ModelPicker groups={available} personas={[]} privatePersonas={[]} isLoading={false} error={null} value={model} ariaLabel="Import reply model" onChange={setModel} /></label>
          {mode === "server" && <label className="toggle-row"><input type="checkbox" checked={allowSearch} onChange={e => setAllowSearch(e.target.checked)} /><span>Allow models to search imported chats</span></label>}
          <label className="toggle-row"><input type="checkbox" checked={learnFromImport} onChange={e => setLearnFromImport(e.target.checked)} /><span>Learn from imported chats</span></label>
          <small>This applies to newly imported chats. When automatic personalization is on, eligible messages can be sent to your chosen profile processor. Chats marked “do not remember” are excluded.</small>
        </fieldset>
        {progress && <div role="status"><progress aria-label="Import progress" max={total} value={progress.completed} /><p>{progress.completed.toLocaleString()} / {total.toLocaleString()} {summary.workspace_pages?"items":"conversations"} · {progress.files.toLocaleString()} files saved</p><p>{progress.imported.toLocaleString()} new · {progress.skipped.toLocaleString()} already imported</p>{progress.changed > 0 && <p>{progress.changed.toLocaleString()} changed exports skipped to protect existing edits.</p>}</div>}
        {phase === "ready" && <button type="button" disabled={!replyAvailable} onClick={() => void start()}><Upload /><span>Import {total.toLocaleString()} {summary.workspace_pages?"items":"conversations"}</span></button>}
        {phase === "done" && <p role="status">Import complete.</p>}
      </>}
      {busy && <button type="button" className="secondary-button" onClick={stop}><X />Stop import</button>}
      {error && <p className="error" role="alert">{error}</p>}
      <RebuildProfileButton key={mode} mode={mode} disabled={busy} />
    </>}
  </section>;
}
