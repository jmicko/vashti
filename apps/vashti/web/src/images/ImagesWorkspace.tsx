import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Copy, Download, GitBranch, History, ImagePlus, Pencil, Plus, RotateCcw, Settings2, Sparkles, Square, Trash2, X } from "lucide-react";
import { requestJson, requestMultipartJson } from "../api";
import type { BackendModelGroup } from "../types";
import { ImageAsset, ImageAssetProvider, downloadImage } from "./ImageAsset";
import { ImagePanel, NumberField } from "./controls";
import { GenerationControls, imageControls as controls } from "./GenerationControls";
import { ModelPicker } from "../ModelPicker";
import { modelParts, modelValue } from "../modelSelection";
import { activeRun, imageLineage, newDraft, type Draft, type GalleryItem, type ImageResult, type ImageSession, type ImageView, type Preferences, type PromptVersion, type Recipe, type Run, type RunRequest, type ServiceSettings } from "./types";
import "./images.css";

const path = "/api/images";
const terminal = (run: Run) => !activeRun(run);
const errorText = (error: unknown) => error instanceof Error ? error.message : "Image operation failed";
const write = <T,>(url: string, value: unknown, method = "POST") => requestJson<T>(url, { method, body: JSON.stringify(value) });
type Props = { view: ImageView; sessionId?: string; isAdmin: boolean; modelGroups: BackendModelGroup[]; onNavigate: (view: ImageView, sessionId?: string) => void };

export function ImagesWorkspace({ view, sessionId, isAdmin, modelGroups, onNavigate }: Props) {
  return <ImageAssetProvider><Workspace view={view} sessionId={sessionId} isAdmin={isAdmin} modelGroups={modelGroups} onNavigate={onNavigate} /></ImageAssetProvider>;
}

function Workspace({ view, sessionId, isAdmin, modelGroups, onNavigate }: Props) {
  const [preferences, setPreferences] = useState<Preferences | null>(null);
  const [settings, setSettings] = useState<ServiceSettings | null>(null);
  const [sessions, setSessions] = useState<ImageSession[]>([]);
  const [session, setSession] = useState<ImageSession | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [mode, setMode] = useState<ImageSession["mode"]>(view === "edit" ? "iterative" : "generate");
  const [results, setResults] = useState<ImageResult[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [history, setHistory] = useState<PromptVersion[]>([]);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [suggestion, setSuggestion] = useState<Run | null>(null);
  const [viewer, setViewer] = useState<{ ids: string[]; results: ImageResult[]; index: number } | null>(null);
  const [confirmEndless, setConfirmEndless] = useState(false);
  const [token, setToken] = useState("");
  const [serviceOpen, setServiceOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [persisted, setPersisted] = useState("");
  const [saving, setSaving] = useState(false);
  const persistence = useRef<{ session: ImageSession | null }>({ session: null });
  const saves = useRef<Promise<unknown>>(Promise.resolve());
  const mounted = useRef(true);
  const flushOnLeave = useRef<(() => Promise<unknown>) | null>(null);
  const remembered = useRef<Partial<Record<"generate" | "edit", string>>>({});
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => () => { void flushOnLeave.current?.().catch(() => {}); }, [sessionId, view]);
  const handledRuns = useRef(new Set<string>());
  const selectionRevision = useRef(0);
  const selectionPending = useRef(false);
  const loadedSession = useRef<string | undefined>(undefined);
  const sessionRef = useRef(session); sessionRef.current = session;
  const draftRef = useRef(draft); draftRef.current = draft;
  const modeRef = useRef(mode); modeRef.current = mode;
  const scrollRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLElement>(null);
  const following = useRef(true);
  const [unseen, setUnseen] = useState(false);
  const submission = useRef<{ key: string; id: string } | null>(null);

  const guard = useCallback(async (action: () => Promise<void>) => {
    setError(""); setBusy(true);
    try { await action(); } catch (e) { setError(errorText(e)); } finally { setBusy(false); }
  }, []);
  useEffect(() => {
    let live = true;
    Promise.all([requestJson<Preferences>(`${path}/preferences`), requestJson<ServiceSettings>(`${path}/settings`), requestJson<ImageSession[]>(`${path}/sessions`)])
      .then(([p, s, list]) => { if (live) { setPreferences(p); setSettings(s); setSessions(list); } })
      .catch(e => { if (live) setError(errorText(e)); });
    return () => { live = false; };
  }, []);
  useEffect(() => {
    setOffset(0); setError("");
    if (sessionId && loadedSession.current !== sessionId) {
      let live = true; persistence.current = { session: null }; sessionRef.current = null; setSession(null); setDraft(null); setResults([]); setRuns([]); setHistory([]); setSuggestion(null);
      requestJson<ImageSession>(`${path}/sessions/${sessionId}`).then(s => {
        if (!live) return; loadedSession.current = s.id; persistence.current = { session: s }; setSession(s); setDraft(s.draft); setMode(s.mode); setPersisted(JSON.stringify([s.mode, s.draft]));
        remembered.current[s.mode === "generate" ? "generate" : "edit"] = s.id;
      }).catch(e => { if (live) setError(errorText(e)); });
      return () => { live = false; };
    }
    if (!sessionId && view !== "gallery" && preferences) {
      loadedSession.current = undefined; setSession(null); setResults([]); setRuns([]); setHistory([]); setSuggestion(null);
      setMode(view === "edit" ? "iterative" : "generate"); setDraft(newDraft(view === "edit" ? preferences.edit : preferences.generate));
      persistence.current = { session: null };
      setPersisted(JSON.stringify([view === "edit" ? "iterative" : "generate", newDraft(view === "edit" ? preferences.edit : preferences.generate)]));
    }
  }, [sessionId, view, preferences === null]);

  const refreshGallery = useCallback(async () => { setGallery(await requestJson<GalleryItem[]>(`${path}/gallery?offset=${offset}`)); }, [offset]);
  useEffect(() => { if (view === "gallery") void refreshGallery().catch(e => setError(errorText(e))); }, [view, refreshGallery]);
  useEffect(() => {
    if (!session?.id) return;
    let live = true; let knownResults: Set<string> | null = null; let timer: ReturnType<typeof setTimeout>;
    const sessionKey = session.id;
    async function adoptDraft(next: Draft, previous: Draft) {
      const current = sessionRef.current;
      if (!live || !current || current.id !== sessionKey) return;
      const saved = await saveDraft(next);
      if (!live || sessionRef.current?.id !== sessionKey) return;
      sessionRef.current=saved; setSession(saved);
      if (draftRef.current === previous) { draftRef.current=saved.draft; setDraft(saved.draft); }
    }
    async function poll() {
      try {
        const selectionAtStart = selectionRevision.current;
        const [nextRuns, nextResults, nextHistory] = await Promise.all([
          requestJson<Run[]>(`${path}/sessions/${sessionKey}/runs`),
          requestJson<ImageResult[]>(`${path}/sessions/${sessionKey}/results?offset=${offset}`),
          requestJson<PromptVersion[]>(`${path}/sessions/${sessionKey}/history`)
        ]);
        if (!live) return;
        if (selectionPending.current || selectionAtStart !== selectionRevision.current) {
          timer = setTimeout(() => void poll(), 1800);
          return;
        }
        const hasNewResults = knownResults && nextResults.some(r => !knownResults!.has(r.id));
        knownResults = new Set(nextResults.map(r => r.id));
        setRuns(nextRuns); setResults(nextResults); setHistory(nextHistory);
        if (hasNewResults) {
          if (following.current) requestAnimationFrame(() => resultsRef.current?.lastElementChild?.scrollIntoView({ block: "start", behavior: "smooth" }));
          else setUnseen(true);
        }
        for (const run of nextRuns.slice(0, 1)) {
          if (!terminal(run) || handledRuns.current.has(run.id)) continue;
          handledRuns.current.add(run.id);
          if (run.error) setError(run.error);
          if (run.completed_count && run.request.draft.seed_mode === "increment" && !run.request.retry_result_id) {
            const base = run.request.draft.recipe.seed;
            const nextSeed = BigInt(base) + BigInt(run.completed_count);
            if (nextSeed <= 9223372036854775807n) { const current=draftRef.current; if (current?.recipe.seed === base) await adoptDraft({ ...current, recipe:{ ...current.recipe, seed:nextSeed.toString() } }, current); }
          }
          if (run.status === "complete" && run.effective_prompt) {
            if (run.request.improve_only) setSuggestion(run);

          }
        }
        const activeOrLatest = nextRuns[0];
        if (activeOrLatest && !activeOrLatest.request.improve_only && modeRef.current === "iterative") {
          const first = [...nextResults].reverse().find(r => r.run_id === activeOrLatest.id);
          const current = draftRef.current;
          const retry = activeOrLatest.request.retry_result_id;
          const unchanged = current && (retry ? current.parent_result_id === retry : current.parent_result_id === activeOrLatest.request.draft.parent_result_id && JSON.stringify(current.input_ids) === JSON.stringify(activeOrLatest.request.draft.input_ids));
          if (first && current && unchanged) await adoptDraft({ ...current, parent_result_id:first.id, input_ids:[first.asset_id] }, current);
        }
      } catch (e) { if (live) setError(errorText(e)); }
      if (live) timer = setTimeout(() => void poll(), 1800);
    }
    void poll(); return () => { live = false; clearTimeout(timer); };
  }, [session?.id, offset]);

  const active = runs.find(activeRun);
  const shownGroups = useMemo(() => {
    if (mode === "iterative" && draft?.parent_result_id && !offset) {
      const lineage = imageLineage(results, draft.parent_result_id);
      if (lineage.length) return lineage;
    }
    return [...results].reverse().filter(r => r.selected);
  }, [results, draft?.parent_result_id, mode, offset]);
  function patchRecipe(patch: Partial<Recipe>) { if (draftRef.current) patchDraft({ recipe: { ...draftRef.current.recipe, ...patch } }); }
  function patchDraft(patch: Partial<Draft>) { if (draftRef.current) { const next = { ...draftRef.current, ...patch }; draftRef.current = next; setDraft(next); } }
  async function saveDraft(value: Draft = draftRef.current!) {
    const target = persistence.current;
    const targetMode = modeRef.current;
    const task = saves.current.catch(() => {}).then(async () => {
      const current = target.session;
      const placeholderTitle = !current || current.title === "New images" || current.title === "Image edit";
      const title = placeholderTitle ? value.recipe.prompt.trim().slice(0, 70) || current?.title || "New images" : current.title;
      const body = { title, mode: targetMode, draft: value, revision: current?.revision };
      const saved = await write<ImageSession>(current ? `${path}/sessions/${current.id}` : `${path}/sessions`, body, current ? "PUT" : "POST");
      target.session = saved;
      if (mounted.current) setSessions(list => [saved, ...list.filter(s => s.id !== saved.id)]);
      if (mounted.current && target === persistence.current) {
        sessionRef.current = saved; loadedSession.current = saved.id; setSession(saved);
        setPersisted(JSON.stringify([targetMode, value]));
        remembered.current[targetMode === "generate" ? "generate" : "edit"] = saved.id;
        if (!current) onNavigate(targetMode === "generate" ? "generate" : "edit", saved.id);
      }
      return saved;
    });
    saves.current = task;
    return task;
  }
  const draftSignature = JSON.stringify([mode, draft]);
  flushOnLeave.current = draft && persisted && draftSignature !== persisted ? () => saveDraft(draft) : null;
  useEffect(() => {
    if (!draft || !persisted || draftSignature === persisted) return;
    const timer = setTimeout(() => {
      setSaving(true);
      void saveDraft(draft).catch(e => { if (mounted.current) setError(errorText(e)); }).finally(() => { if (mounted.current) setSaving(false); });
    }, 700);
    return () => clearTimeout(timer);
  }, [draftSignature, persisted]);
  useEffect(() => {
    if (!draft || draftSignature === persisted) return;
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [draftSignature, persisted]);
  async function navigate(next: ImageView, id?: string, fresh = false) {
    await guard(async () => {
      if (draftRef.current && JSON.stringify([modeRef.current, draftRef.current]) !== persisted) await saveDraft();
      await saves.current;
      setHistoryOpen(false);
      if (fresh) {
        loadedSession.current = undefined;
        persistence.current = { session: null }; sessionRef.current = null; setSession(null);
        setResults([]); setRuns([]); setHistory([]); setSuggestion(null);
        const nextMode = next === "edit" ? "iterative" : "generate";
        const nextDraft = newDraft(preferences![next === "edit" ? "edit" : "generate"]);
        modeRef.current = nextMode; setMode(nextMode); draftRef.current = nextDraft; setDraft(nextDraft);
        setPersisted(JSON.stringify([nextMode, nextDraft]));
      }
      const recent = next === "gallery" ? undefined : sessions.find(s => (s.mode === "generate" ? "generate" : "edit") === next)?.id;
      onNavigate(next, fresh ? undefined : id ?? (next === "gallery" ? sessionRef.current?.id : remembered.current[next] ?? recent));
    });
  }
  async function start(improveOnly = false, retry?: ImageResult, confirmed = false) {
    if (!draftRef.current || active) return;
    const value = structuredClone(draftRef.current);
    if (!improveOnly && !retry && value.count === null && value.seed_mode === "fixed" && !confirmed) { setConfirmEndless(true); return; }
    await guard(async () => {
      const saved = await saveDraft(value);
      const key = JSON.stringify([saved.id, value, improveOnly, retry?.id, confirmed]);
      if (submission.current?.key !== key) submission.current = { key, id: crypto.randomUUID() };
      const request: RunRequest = { client_request_id: submission.current.id, draft: value, improve_only: improveOnly, retry_result_id: retry?.id ?? null, fixed_endless_confirmed: confirmed };
      const run = await write<Run>(`${path}/sessions/${saved.id}/runs`, request);
      submission.current = null; setSuggestion(null); setRuns(list => [run, ...list.filter(r => r.id !== run.id)]); setConfirmEndless(false);
    });
  }
  async function applySuggestion() {
    if (!suggestion?.effective_prompt || !draft) return;
    const previous = draft;
    const next = { ...draft, recipe: { ...draft.recipe, prompt: suggestion.effective_prompt }, prompt_version_id: suggestion.prompt_version_id };
    patchDraft(next); setSuggestion(null);
    await guard(async () => { await saveDraft(previous); await saveDraft(); });
  }
  async function selectVersion(result: ImageResult) {
    if (selectionPending.current) return;
    selectionPending.current = true; selectionRevision.current++;
    const target = persistence.current;
    const previous = draftRef.current;
    let leaf = result; const visited = new Set<string>();
    while (!visited.has(leaf.id)) { visited.add(leaf.id); const child = results.find(r => r.parent_result_id === leaf.id && r.selected); if (!child) break; leaf = child; }
    const editHere = modeRef.current === "iterative" && previous && sessionRef.current?.id === result.session_id;
    if (editHere) patchDraft({ parent_result_id: leaf.id, input_ids: [leaf.asset_id] });
    setResults(list => list.map(r => r.group_id === result.group_id ? { ...r, selected: r.id === result.id } : r));
    await guard(async () => {
      try {
        await write(`${path}/results/${result.id}/select`, {});
        if (editHere && draftRef.current && target === persistence.current) await saveDraft();
        if (view === "gallery") {
          const source = await requestJson<ImageSession>(`${path}/sessions/${result.session_id}`);
          await navigate(source.mode === "generate" ? "generate" : "edit", source.id);
        }
      } catch (e) {
        if (target === persistence.current) {
          setResults(results);
          if (editHere && draftRef.current?.parent_result_id === leaf.id) patchDraft({ parent_result_id: previous.parent_result_id, input_ids: previous.input_ids });
        }
        throw e;
      } finally { selectionPending.current = false; selectionRevision.current++; }
    });
  }
  async function beginEdit(assetId: string) {
    if (!preferences) return;
    if (view === "edit" && draft) {
      const result = results.find(r => r.asset_id === assetId);
      setMode("iterative"); modeRef.current = "iterative";
      patchDraft({ input_ids: [assetId], parent_result_id: result?.id ?? null });
      setViewer(null); composerRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      return;
    }
    await guard(async () => {
      if (draftRef.current && JSON.stringify([modeRef.current, draftRef.current]) !== persisted) await saveDraft();
      const next = newDraft(preferences.edit); next.input_ids = [assetId]; next.original_input_ids = [assetId];
      const created = await write<ImageSession>(`${path}/sessions`, { title: "Image edit", mode: "iterative", draft: next });
      setViewer(null); onNavigate("edit", created.id);
    });
  }
  async function uploadFiles(files: FileList | null) {
    if (!files || !draft) return;
    const target = persistence.current;
    const selected = [...files];
    if (draft.input_ids.length + selected.length > 10) { setError("Choose at most ten reference images."); return; }
    await guard(async () => {
      const ids: string[] = [];
      for (const file of selected) { const form = new FormData(); form.append("image", file); const image = await requestMultipartJson<{ id: string }>(`${path}/assets`, form, { method: "POST" }); ids.push(image.id); }
      const current = draftRef.current;
      if (!current || target !== persistence.current) return;
      const input_ids = [...current.input_ids, ...ids];
      patchDraft({ input_ids, original_input_ids: current.parent_result_id && modeRef.current === "iterative" ? current.original_input_ids : input_ids });
    });
  }
  async function openGallery(item: GalleryItem) {
    await guard(async () => {
      const all = item.session_id && item.group_id ? await requestJson<ImageResult[]>(`${path}/sessions/${item.session_id}/results`) : [];
      const variants = all.filter(r => r.group_id === item.group_id).reverse();
      const ids = variants.length ? variants.map(r => r.asset_id) : [item.asset_id];
      setViewer({ ids, results: variants, index: Math.max(0, ids.indexOf(item.asset_id)) });
    });
  }
  function moveViewer(delta: number) { setViewer(v => v ? { ...v, index: Math.max(0, Math.min(v.ids.length - 1, v.index + delta)) } : null); }
  useEffect(() => {
    if (!viewer) return;
    const key = (event: KeyboardEvent) => { if (event.key === "Escape") setViewer(null); if (event.key === "ArrowLeft") moveViewer(-1); if (event.key === "ArrowRight") moveViewer(1); };
    window.addEventListener("keydown", key); return () => window.removeEventListener("keydown", key);
  }, [Boolean(viewer)]);
  const swipeStart = useRef<{ x: number; y: number } | null>(null);
  const suppressImageClick = useRef(false);
  const hidden = (key: string) => preferences?.hidden_controls.includes(key);
  const controlsProfile = mode === "generate" ? "generate" : "edit";
  function toggleAllowed(key: string, allowed: boolean) {
    if (!preferences) return;
    const profile = preferences[controlsProfile];
    setPreferences({ ...preferences, [controlsProfile]: { ...profile, model_may_change: allowed ? [...profile.model_may_change, key] : profile.model_may_change.filter(v => v !== key) } });
  }
  const promptModelValue = draft?.prompt_help ? modelValue(draft.prompt_help.backend_id, draft.prompt_help.model_name) : "";
  const promptModels = useMemo(() => modelGroups.map(group => ({ ...group, models: group.models.filter(model => !model.capabilities?.includes("embedding") || model.capabilities.includes("completion")) })), [modelGroups]);

  return <div className="images-workspace" ref={scrollRef} onScroll={() => { const el = scrollRef.current; if (el) following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 220; }}>
    <nav className="images-tabs" aria-label="Images views"><div className="images-segments">{(["gallery", "generate", "edit"] as const).map(v => <button key={v} type="button" disabled={busy || !preferences} aria-current={view === v ? "page" : undefined} onClick={() => void navigate(v)}>{v === "gallery" ? "Gallery" : v === "generate" ? "Generate" : "Edit"}</button>)}</div><div className="images-actions">
      <button className="images-icon" title="New session" aria-label="New session" disabled={busy || !preferences} onClick={() => void navigate(view === "edit" ? "edit" : "generate", undefined, true)}><Plus size={20} /></button>
      <button className="images-icon" title="History" aria-label="Image history" onClick={() => setHistoryOpen(true)}><History size={20} /></button>
      <button className="images-icon" title="Image preferences" aria-label="Image preferences" onClick={() => setServiceOpen(true)}><Settings2 size={20} /></button>
    </div></nav>
    {error && <p className="images-error" role="alert">{error}<button type="button" onClick={() => setError("")} aria-label="Dismiss error"><X size={16} /></button></p>}
    {settings && !settings.token_configured && <div className="images-notice">Image service not connected.{isAdmin && <button onClick={() => setServiceOpen(true)}>Connect</button>}</div>}
    {serviceOpen && <ImagePanel title="Image preferences" onClose={() => setServiceOpen(false)}>
      {preferences && draft && <>
        <details open><summary>Visible generation controls</summary>{controls.map(([key, label]) => <label className="images-check" key={key}><input type="checkbox" checked={!hidden(key)} onChange={e => setPreferences({ ...preferences, hidden_controls: e.target.checked ? preferences.hidden_controls.filter(v => v !== key) : [...preferences.hidden_controls, key] })} />{label}</label>)}<button disabled={busy} onClick={() => void guard(async () => { setPreferences(await write<Preferences>(`${path}/preferences`, preferences, "PUT")); setServiceOpen(false); })}>Save preferences</button></details>
        <details><summary>Chat tool permissions and defaults</summary><p>Settings the model may change when generating images in chats:</p>{controls.filter(([key]) => key !== "count").map(([key, label]) => <label className="images-check" key={key}><input type="checkbox" checked={preferences[controlsProfile].model_may_change.includes(key)} onChange={e => toggleAllowed(key, e.target.checked)} />{label}</label>)}<button disabled={busy} onClick={() => void guard(async () => { const updated = { ...preferences, [controlsProfile]: { ...preferences[controlsProfile], recipe: { ...draft.recipe, prompt: "" }, seed_mode: draft.seed_mode } }; setPreferences(await write<Preferences>(`${path}/preferences`, updated, "PUT")); setServiceOpen(false); })}>Save as {controlsProfile} defaults</button></details>
      </>}
      {settings && <details open={!settings.token_configured}><summary>Image service</summary><div className="images-settings">{isAdmin ? <>
      <label>Inference host<select value={settings.backend_id ?? ""} onChange={e => setSettings({ ...settings, backend_id: e.target.value || null })}><option value="">Choose Kuno host</option>{modelGroups.map(g => <option key={g.backend.id} value={g.backend.id}>{g.backend.name}</option>)}</select></label>
      <label>Kuno access token<input type="password" autoComplete="new-password" value={token} placeholder={settings.token_configured ? "Configured; leave blank to keep" : "Required"} onChange={e => setToken(e.target.value)} /></label>
      <NumberField label="Storage per user (MiB)" min={20} max={1048576} value={settings.max_user_bytes / 1048576} onChange={value => setSettings({ ...settings, max_user_bytes: value * 1048576 })} />
      <button disabled={busy} onClick={() => void guard(async () => { setSettings(await write<ServiceSettings>(`${path}/settings`, { backend_id: settings.backend_id, token: token || null, max_user_bytes: settings.max_user_bytes }, "PUT")); setToken(""); })}>Save connection</button>
    </> : <p>Your administrator manages the inference host and storage limit.</p>}
    </div></details>}
    </ImagePanel>}
    {historyOpen && <ImagePanel title="Image history" onClose={() => setHistoryOpen(false)}>
      <details open><summary>Sessions</summary><div className="images-sessions">{sessions.map(s => <button disabled={busy} key={s.id} onClick={() => void navigate(s.mode === "generate" ? "generate" : "edit", s.id)} aria-current={session?.id === s.id ? "true" : undefined}>{s.title}<small>{s.mode === "generate" ? "Generate" : "Edit"}</small></button>)}{!sessions.length && <p>No saved sessions yet.</p>}</div></details>
      {results.length > 0 && <details><summary>Results and branches</summary><div className="images-branch-list">{results.map(r => <button disabled={busy} key={r.id} onClick={() => { void selectVersion(r); setHistoryOpen(false); }}><ImageAsset id={r.asset_id} /><span>{r.recipe.prompt.slice(0, 90)}<small>Seed {r.recipe.seed}</small></span></button>)}</div><div className="images-actions"><button disabled={!offset} onClick={() => setOffset(n => Math.max(0, n - 100))}>Newer</button><button disabled={results.length < 100} onClick={() => setOffset(n => n + 100)}>Older</button></div></details>}
      {history.length > 0 && <details><summary>Prompt history ({history.length})</summary>{history.map(version => <article className="images-prompt-version" key={version.id}><small>{version.source === "improved" ? `Improved · ${version.model_name}` : "Your prompt"}</small><p>{version.prompt}</p>{version.guidance && <p className="muted">Guidance: {version.guidance}</p>}<button disabled={busy || !draft} onClick={() => { patchRecipe({ prompt: version.prompt }); patchDraft({ prompt_version_id: version.id }); setHistoryOpen(false); }}>Restore this prompt</button></article>)}</details>}
    </ImagePanel>}
    {active && <aside className="images-run" role="status"><span>{active.status === "improving" ? "Improving prompt…" : `${active.status[0].toUpperCase()}${active.status.slice(1)}…`} · {active.completed_count} saved{active.request.draft.count !== null && !active.request.improve_only ? ` of ${active.request.draft.count}` : ""}</span><button disabled={active.status === "stopping"} onClick={() => void guard(async () => { const stopped = await write<Run>(`${path}/runs/${active.id}/stop`, {}); setRuns(list => list.map(r => r.id === stopped.id ? stopped : r)); })}><Square size={16} />Stop</button></aside>}
    {view === "gallery" ? <>
      <div className="images-gallery">{gallery.map(item => <article key={item.id}>
        <ImageAsset id={item.asset_id} onClick={() => void openGallery(item)} />
        <div className="images-result-toolbar"><span className="images-stack-count">{item.variants > 1 ? `${item.variants} versions` : ""}</span><div className="images-actions">
          <button className="images-icon" disabled={busy} title="Edit image" aria-label="Edit image" onClick={() => void beginEdit(item.asset_id)}><Pencil size={18} /></button>
          <button className="images-icon" title="Download" aria-label="Download" onClick={() => void guard(() => downloadImage(item.asset_id))}><Download size={18} /></button>
          {item.session_id && <button className="images-icon" disabled={busy} title="Open session" aria-label="Open session" onClick={() => void guard(async () => { const source = await requestJson<ImageSession>(`${path}/sessions/${item.session_id}`); await navigate(source.mode === "generate" ? "generate" : "edit", source.id); })}><History size={18} /></button>}
          <button className="images-icon" title="Remove from gallery" aria-label="Remove from gallery" onClick={() => void guard(async () => { await requestJson(`${path}/gallery/${item.id}`, { method: "DELETE" }); await refreshGallery(); })}><Trash2 size={18} /></button>
        </div></div>
      </article>)}</div>
      {!gallery.length && <p className="images-empty">Your generated images will appear here.</p>}
      <div className="images-actions"><button disabled={!offset} onClick={() => setOffset(n => Math.max(0, n - 50))}>Newer</button><button disabled={gallery.length < 50} onClick={() => setOffset(n => n + 50)}>Older</button></div>
    </> : draft && preferences ? <div className="images-flow">
      <section className="images-results" ref={resultsRef} aria-label="Image history">{shownGroups.map(result => {
        const variants = results.filter(r => r.group_id === result.group_id).reverse(); const index = variants.findIndex(r => r.id === result.id);
        return <article className="images-result" key={result.group_id}>
          <div className="images-swipe"
            onPointerDown={e => { suppressImageClick.current = false; swipeStart.current = { x: e.clientX, y: e.clientY }; }}
            onPointerCancel={() => { swipeStart.current = null; }}
            onPointerUp={e => {
              const start = swipeStart.current; swipeStart.current = null;
              if (!start || busy) return;
              const x = e.clientX - start.x, y = e.clientY - start.y;
              const next = index + (x < 0 ? 1 : -1);
              if (Math.abs(x) > 65 && Math.abs(x) > Math.abs(y) * 1.5) {
                suppressImageClick.current = true;
                if (variants[next]) void selectVersion(variants[next]);
              }
            }}>
            <ImageAsset id={result.asset_id} neighbors={[variants[index - 1]?.asset_id, variants[index + 1]?.asset_id].filter((id): id is string => Boolean(id))}
              onClick={() => { if (!suppressImageClick.current) setViewer({ ids: variants.map(r => r.asset_id), results: variants, index }); }} />
          </div>
          <div className="images-result-toolbar"><div className="images-actions">
            <button className="images-icon" title="Retry with a new seed" aria-label="Retry with a new seed" disabled={busy || Boolean(active)} onClick={() => void start(false, result)}><RotateCcw size={18} /></button>
            <button className="images-icon" title="Edit image" aria-label="Edit image" disabled={busy} onClick={() => void beginEdit(result.asset_id)}><Pencil size={18} /></button>
            <button className="images-icon" title="Download" aria-label="Download" onClick={() => void guard(() => downloadImage(result.asset_id))}><Download size={18} /></button>
            <button className="images-icon" title="Branch from this prompt" aria-label="Branch from this prompt" onClick={() => { patchDraft({ recipe: { ...result.recipe }, input_ids: [...result.input_ids], parent_result_id: result.parent_result_id }); composerRef.current?.scrollIntoView({ block: "start", behavior: "smooth" }); }}><GitBranch size={18} /></button></div>
          {variants.length > 1 && <div className="images-variants"><button aria-label="Previous version" disabled={index <= 0 || busy} onClick={() => void selectVersion(variants[index - 1])}><ArrowLeft size={18} /></button><span>{index + 1} / {variants.length}</span><button aria-label="Next version" disabled={index >= variants.length - 1 || busy} onClick={() => void selectVersion(variants[index + 1])}><ArrowRight size={18} /></button></div>}
          </div><details><summary>Prompt and settings · {(result.duration_ms / 1000).toFixed(1)}s</summary><p className="images-result-prompt">{result.recipe.prompt}</p><p>{result.recipe.width} × {result.recipe.height} · {result.recipe.steps} steps · CFG {result.recipe.cfg} · seed {result.recipe.seed} · {result.recipe.attention} attention</p>{result.recipe.negative_prompt && <p>Negative: {result.recipe.negative_prompt}</p>}</details>
        </article>;
      })}</section>
      {unseen && <button onClick={() => { resultsRef.current?.lastElementChild?.scrollIntoView({ block: "start", behavior: "smooth" }); setUnseen(false); }}>New images ↓</button>}
      {active?.preview && <figure className="images-live-preview" aria-label="Image generation preview"><div className="images-asset"><img src={`data:image/png;base64,${active.preview.png_base64}`} alt="Approximate image being generated" /></div><figcaption>Preview · step {active.preview.step} of {active.preview.steps} · the final image will be sharper</figcaption></figure>}
      <div className="images-composer" ref={composerRef}>
        {mode !== "generate" && <>
          <fieldset><legend>Edit from</legend><div className="images-segments">{(["iterative", "original"] as const).map(value => <button key={value} disabled={Boolean(active)} aria-pressed={mode === value} onClick={() => { setMode(value); modeRef.current = value; if (value === "original") patchDraft({ input_ids: [...draft.original_input_ids], parent_result_id: null }); }}>{value === "iterative" ? "Selected result" : "Original images"}</button>)}</div></fieldset>
          <div className="images-inputs">{draft.input_ids.map((asset, index) => <div key={`${asset}:${index}`}>
            <ImageAsset id={asset} alt={`Reference ${index + 1}`} onClick={() => setViewer({ ids: draft.input_ids, results: [], index })} />
            <div className="images-actions"><small>{index + 1}</small>
              <button className="images-icon" title="Remove reference" aria-label={`Remove reference ${index + 1}`} onClick={() => { const ids = draft.input_ids.filter((_, i) => i !== index); patchDraft({ input_ids: ids, ...(!draft.parent_result_id || mode === "original" ? { original_input_ids: ids } : {}) }); }}><X size={16} /></button>
              {index > 0 && <button className="images-icon" title="Move earlier" aria-label={`Move reference ${index + 1} earlier`} onClick={() => { const ids = [...draft.input_ids]; [ids[index - 1], ids[index]] = [ids[index], ids[index - 1]]; patchDraft({ input_ids: ids, ...(!draft.parent_result_id || mode === "original" ? { original_input_ids: ids } : {}) }); }}><ArrowLeft size={16} /></button>}
              <button className="images-icon" title="Import embedded generation settings" aria-label={`Import settings from reference ${index + 1}`} onClick={() => void guard(async () => { const value = await requestJson<{ recipe: Recipe; required_reference_images: number; warnings: string[] }>(`${path}/assets/${asset}/settings`); patchRecipe(value.recipe); if (value.required_reference_images) setError(`Imported settings require ${value.required_reference_images} original reference image(s), in their original order.`); })}><Settings2 size={16} /></button>
            </div>
          </div>)}</div>
          <label className="images-upload"><ImagePlus size={18} />Add reference images<input type="file" multiple accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={e => { void uploadFiles(e.target.files); e.target.value = ""; }} /></label>
        </>}
        <label>Prompt<textarea rows={4} value={draft.recipe.prompt} onChange={e => patchRecipe({ prompt: e.target.value })} placeholder={mode === "generate" ? "Describe the image you want…" : "Describe what to change…"} /></label>
        {suggestion && <section className="images-suggestion"><h3>Suggested prompt</h3><p>{suggestion.effective_prompt}</p>{draft.recipe.prompt !== suggestion.request.draft.recipe.prompt && <p>Your draft changed since this suggestion was requested. Accepting will replace the current text.</p>}<div className="images-actions"><button disabled={busy} onClick={() => void applySuggestion()}>Use this prompt</button><button onClick={() => setSuggestion(null)}>Keep my prompt</button></div><p className="muted">Your original wording stays in prompt history.</p></section>}
        <div className="images-send-actions"><small role="status">{saving ? "Saving…" : draftSignature === persisted && session ? "Draft saved" : ""}</small><button title="Improve prompt once" disabled={busy || Boolean(active) || !draft.recipe.prompt.trim() || !draft.prompt_help?.model_name} onClick={() => void start(true)}><Sparkles size={18} />Improve</button>{active ? <button disabled={active.status === "stopping"} onClick={() => void guard(async () => { const stopped = await write<Run>(`${path}/runs/${active.id}/stop`, {}); setRuns(list => list.map(r => r.id === stopped.id ? stopped : r)); })}><Square size={18} />Stop</button> : <button className="primary-action" disabled={busy || !draft.recipe.prompt.trim() || !settings?.token_configured || (mode !== "generate" && !draft.input_ids.length)} onClick={() => void start()}><ImagePlus size={18} />Generate</button>}</div>
        <details className="images-prompt-help"><summary>Prompt improvement</summary>
          <div className="images-model-picker"><span>Assistant model</span><ModelPicker groups={promptModels} personas={[]} privatePersonas={[]} isLoading={false} error={null} value={promptModelValue} ariaLabel="Prompt assistant model" onChange={value => { const parts = modelParts(value); if (parts) patchDraft({ prompt_help: { backend_id: parts.backendId, model_name: parts.modelName, guidance: draft.prompt_help?.guidance ?? "" } }); }} /></div>
          <label>Improvement guidance<textarea rows={3} disabled={!draft.prompt_help} value={draft.prompt_help?.guidance ?? ""} placeholder="For example: The images keep coming out blue. Help correct that." onChange={e => { if (draft.prompt_help) patchDraft({ prompt_help: { ...draft.prompt_help, guidance: e.target.value } }); }} /></label>
          <label className="images-check"><input type="checkbox" checked={draft.auto_improve} disabled={!draft.prompt_help} onChange={e => patchDraft({ auto_improve: e.target.checked })} />Improve each image's prompt</label><p className="muted">Each image starts from your original prompt above, including batches. Your draft stays unchanged. Retry uses the recorded prompt.</p>
        </details>
        <details><summary>Generation settings <small>{draft.recipe.width} × {draft.recipe.height} · {draft.count ?? "∞"} {draft.count === 1 ? "image" : "images"}</small></summary><GenerationControls draft={draft} hidden={preferences.hidden_controls} patchRecipe={patchRecipe} patchDraft={patchDraft} /></details>
      </div>
    </div> : <p role="status">Loading image workspace…</p>}
    {confirmEndless && <ImagePanel title="Repeat the same image?" onClose={() => setConfirmEndless(false)}><p>Endless generation with a fixed seed and unchanged settings will generally repeat the same image until you stop it.</p><div className="images-actions"><button onClick={() => setConfirmEndless(false)}>Go back</button><button disabled={busy} onClick={() => void start(false, undefined, true)}>Generate anyway</button></div></ImagePanel>}
    {viewer && <ImagePanel title={`Image ${viewer.index + 1} of ${viewer.ids.length}`} className="images-viewer-panel" onClose={() => setViewer(null)}>
      <div className="images-swipe" onPointerDown={e => { swipeStart.current = { x: e.clientX, y: e.clientY }; }} onPointerCancel={() => { swipeStart.current = null; }} onPointerUp={e => { const start = swipeStart.current; swipeStart.current = null; if (start && Math.abs(e.clientX - start.x) > 65 && Math.abs(e.clientX - start.x) > Math.abs(e.clientY - start.y) * 1.5) moveViewer(e.clientX < start.x ? 1 : -1); }}>
        <ImageAsset id={viewer.ids[viewer.index]} neighbors={[viewer.ids[viewer.index - 1], viewer.ids[viewer.index + 1]].filter(Boolean)} />
      </div>
      <div className="images-actions">
        <button className="images-icon" title="Previous image" aria-label="Previous image" disabled={viewer.index === 0} onClick={() => moveViewer(-1)}><ArrowLeft size={20} /></button>
        <button className="images-icon" title="Next image" aria-label="Next image" disabled={viewer.index === viewer.ids.length - 1} onClick={() => moveViewer(1)}><ArrowRight size={20} /></button>
        <button className="images-icon" title="Edit image" aria-label="Edit image" disabled={busy} onClick={() => void beginEdit(viewer.ids[viewer.index])}><Pencil size={20} /></button>
        <button className="images-icon" title="Download" aria-label="Download" onClick={() => void guard(() => downloadImage(viewer.ids[viewer.index]))}><Download size={20} /></button>
        <button className="images-icon" title="Copy to gallery" aria-label="Copy to gallery" disabled={busy} onClick={() => void guard(async () => { await write(`${path}/gallery/copy/${viewer.ids[viewer.index]}`, {}); if (view === "gallery") await refreshGallery(); })}><Copy size={20} /></button>
        {viewer.results[viewer.index] && <button disabled={busy} onClick={() => { void selectVersion(viewer.results[viewer.index]); setViewer(null); }}>Use this version</button>}
      </div>
    </ImagePanel>}
  </div>;
}
