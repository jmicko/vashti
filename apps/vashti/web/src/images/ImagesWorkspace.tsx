import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Download, ImagePlus, Sparkles, Square, X } from "lucide-react";
import { requestJson, requestMultipartJson } from "../api";
import type { BackendModelGroup } from "../types";
import { ImageAsset, downloadImage } from "./ImageAsset";
import { activeRun, imageLineage, newDraft, type Draft, type GalleryItem, type ImageResult, type ImageSession, type ImageView, type Preferences, type PromptVersion, type Recipe, type Run, type RunRequest, type ServiceSettings } from "./types";
import "./images.css";

const path = "/api/images";
const controls = [["negative_prompt", "Negative prompt"], ["size", "Image size"], ["steps", "Steps"], ["cfg", "CFG"], ["seed", "Seed"], ["count", "Image count"], ["attention", "Attention"]] as const;
const terminal = (run: Run) => !activeRun(run);
const errorText = (error: unknown) => error instanceof Error ? error.message : "Image operation failed";
const write = <T,>(url: string, value: unknown, method = "POST") => requestJson<T>(url, { method, body: JSON.stringify(value) });
type Props = { view: ImageView; sessionId?: string; isAdmin: boolean; modelGroups: BackendModelGroup[]; onNavigate: (view: ImageView, sessionId?: string) => void };

export function ImagesWorkspace({ view, sessionId, isAdmin, modelGroups, onNavigate }: Props) {
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
  const handledRuns = useRef(new Set<string>());
  const adoptedPrompts = useRef(new Set<string>());
  const loadedSession = useRef<string | undefined>(undefined);
  const sessionRef = useRef(session); sessionRef.current = session;
  const draftRef = useRef(draft); draftRef.current = draft;
  const modeRef = useRef(mode); modeRef.current = mode;
  const scrollRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLDivElement>(null);
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
      .then(([p, s, list]) => { if (live) { setPreferences(p); setSettings(s); setSessions(list); if (!sessionId) setDraft(newDraft(view === "edit" ? p.edit : p.generate)); } })
      .catch(e => { if (live) setError(errorText(e)); });
    return () => { live = false; };
  }, []);
  useEffect(() => {
    setOffset(0); setError("");
    if (sessionId && loadedSession.current !== sessionId) {
      let live = true; setDraft(null); setResults([]); setRuns([]); setHistory([]); setSuggestion(null);
      requestJson<ImageSession>(`${path}/sessions/${sessionId}`).then(s => {
        if (!live) return; loadedSession.current = s.id; setSession(s); setDraft(s.draft); setMode(s.mode);
      }).catch(e => { if (live) setError(errorText(e)); });
      return () => { live = false; };
    }
    if (!sessionId && view !== "gallery" && preferences) {
      loadedSession.current = undefined; setSession(null); setResults([]); setRuns([]); setHistory([]); setSuggestion(null);
      setMode(view === "edit" ? "iterative" : "generate"); setDraft(newDraft(view === "edit" ? preferences.edit : preferences.generate));
    }
  }, [sessionId, view, preferences === null]);

  const refreshGallery = useCallback(async () => { setGallery(await requestJson<GalleryItem[]>(`${path}/gallery?offset=${offset}`)); }, [offset]);
  useEffect(() => { if (view === "gallery") void refreshGallery().catch(e => setError(errorText(e))); }, [view, refreshGallery]);
  useEffect(() => {
    if (!session?.id) return;
    let live = true; let timer: ReturnType<typeof setTimeout>;
    const sessionKey = session.id;
    async function adoptDraft(next: Draft, previous: Draft) {
      const current = sessionRef.current;
      if (!live || !current || current.id !== sessionKey) return;
      const saved = await write<ImageSession>(`${path}/sessions/${sessionKey}`, { title:current.title, mode:modeRef.current, draft:next, revision:current.revision }, "PUT");
      if (!live || sessionRef.current?.id !== sessionKey) return;
      sessionRef.current=saved; setSession(saved);
      if (draftRef.current === previous) { draftRef.current=saved.draft; setDraft(saved.draft); }
    }
    async function poll() {
      try {
        const [nextRuns, nextResults, nextHistory] = await Promise.all([
          requestJson<Run[]>(`${path}/sessions/${sessionKey}/runs`),
          requestJson<ImageResult[]>(`${path}/sessions/${sessionKey}/results?offset=${offset}`),
          requestJson<PromptVersion[]>(`${path}/sessions/${sessionKey}/history`)
        ]);
        if (!live) return;
        setRuns(nextRuns); setResults(old => {
          if (nextResults.some(r => !old.some(previous => previous.id === r.id))) {
            if (following.current) requestAnimationFrame(() => composerRef.current?.scrollIntoView({ block: "end", behavior: "smooth" }));
            else setUnseen(true);
          }
          return nextResults;
        }); setHistory(nextHistory);
        const latest = nextRuns[0];
        if (latest?.effective_prompt && latest.request.draft.auto_improve && !latest.request.improve_only && !adoptedPrompts.current.has(latest.id)) {
          adoptedPrompts.current.add(latest.id);
          const current = draftRef.current;
          if (current && current.recipe.prompt === latest.request.draft.recipe.prompt) {
            const updated = { ...current, recipe: { ...current.recipe, prompt: latest.effective_prompt }, prompt_version_id: latest.prompt_version_id };
            await adoptDraft(updated, current);
          }
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
  function patchRecipe(patch: Partial<Recipe>) { setDraft(d => d ? { ...d, recipe: { ...d.recipe, ...patch } } : d); }
  function patchDraft(patch: Partial<Draft>) { setDraft(d => d ? { ...d, ...patch } : d); }
  async function saveDraft(value: Draft = draftRef.current!) {
    const current = sessionRef.current;
    const body = { title: current?.title ?? (value.recipe.prompt.trim().slice(0, 70) || "New images"), mode: modeRef.current, draft: value, revision: current?.revision };
    const saved = await write<ImageSession>(current ? `${path}/sessions/${current.id}` : `${path}/sessions`, body, current ? "PUT" : "POST");
    sessionRef.current = saved; loadedSession.current = saved.id; setSession(saved);
    if (!current) { setSessions(list => [saved, ...list]); onNavigate(modeRef.current === "generate" ? "generate" : "edit", saved.id); }
    return saved;
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
    const next = { ...draft, recipe: { ...draft.recipe, prompt: suggestion.effective_prompt }, prompt_version_id: suggestion.prompt_version_id };
    await guard(async () => { await saveDraft(draft); await saveDraft(next); setDraft(next); setSuggestion(null); });
  }
  async function selectVersion(result: ImageResult) {
    await guard(async () => {
      await write(`${path}/results/${result.id}/select`, {});
      setResults(list => list.map(r => r.group_id === result.group_id ? { ...r, selected: r.id === result.id } : r));
      let leaf = result; const visited = new Set<string>();
      while (!visited.has(leaf.id)) { visited.add(leaf.id); const child = results.find(r => r.parent_result_id === leaf.id && r.selected); if (!child) break; leaf = child; }
      if (mode === "iterative" && draft && session?.id === result.session_id) { const next = { ...draft, parent_result_id: leaf.id, input_ids: [leaf.asset_id] }; setDraft(next); await saveDraft(next); }
    });
  }
  async function beginEdit(assetId: string) {
    if (!preferences) return;
    await guard(async () => {
      const next = newDraft(preferences.edit); next.input_ids = [assetId]; next.original_input_ids = [assetId];
      const created = await write<ImageSession>(`${path}/sessions`, { title: "Image edit", mode: "iterative", draft: next });
      setViewer(null); onNavigate("edit", created.id);
    });
  }
  async function uploadFiles(files: FileList | null) {
    if (!files || !draft) return;
    const selected = [...files];
    if (draft.input_ids.length + selected.length > 10) { setError("Choose at most ten reference images."); return; }
    await guard(async () => {
      const ids: string[] = [];
      for (const file of selected) { const form = new FormData(); form.append("image", file); const image = await requestMultipartJson<{ id: string }>(`${path}/assets`, form); ids.push(image.id); }
      const input_ids = [...draft.input_ids, ...ids];
      patchDraft({ input_ids, original_input_ids: draft.parent_result_id && mode === "iterative" ? draft.original_input_ids : input_ids });
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
  const hidden = (key: string) => preferences?.hidden_controls.includes(key);
  const controlsProfile = mode === "generate" ? "generate" : "edit";
  function toggleAllowed(key: string, allowed: boolean) {
    if (!preferences) return;
    const profile = preferences[controlsProfile];
    setPreferences({ ...preferences, [controlsProfile]: { ...profile, model_may_change: allowed ? [...profile.model_may_change, key] : profile.model_may_change.filter(v => v !== key) } });
  }
  const promptModelValue = draft?.prompt_help ? JSON.stringify([draft.prompt_help.backend_id, draft.prompt_help.model_name]) : "";

  return <div className="images-workspace" ref={scrollRef} onScroll={() => { const el = scrollRef.current; if (el) following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 220; }}>
    <nav className="images-tabs" aria-label="Images views">{(["gallery", "generate", "edit"] as const).map(v => <button key={v} type="button" aria-current={view === v ? "page" : undefined} onClick={() => onNavigate(v)}>{v === "gallery" ? "Gallery" : v === "generate" ? "Generate" : "Edit"}</button>)}<button type="button" onClick={() => setServiceOpen(v => !v)}>Settings</button></nav>
    {error && <p className="images-error" role="alert">{error}<button type="button" onClick={() => setError("")} aria-label="Dismiss error"><X size={16} /></button></p>}
    {settings && !settings.token_configured && <p className="images-notice">Connect Kuno in Images settings to generate images. Prompt improvement can be used independently.</p>}
    {serviceOpen && settings && <section className="images-settings"><h2>Image service</h2>{isAdmin ? <>
      <label>Inference host<select value={settings.backend_id ?? ""} onChange={e => setSettings({ ...settings, backend_id: e.target.value || null })}><option value="">Choose Kuno host</option>{modelGroups.map(g => <option key={g.backend.id} value={g.backend.id}>{g.backend.name}</option>)}</select></label>
      <label>Kuno access token<input type="password" autoComplete="new-password" value={token} placeholder={settings.token_configured ? "Configured; leave blank to keep" : "Required"} onChange={e => setToken(e.target.value)} /></label>
      <label>Storage per user (MiB)<input type="number" min={20} value={settings.max_user_bytes / 1048576} onChange={e => setSettings({ ...settings, max_user_bytes: Number(e.target.value) * 1048576 })} /></label>
      <button disabled={busy} onClick={() => void guard(async () => { setSettings(await write<ServiceSettings>(`${path}/settings`, { backend_id: settings.backend_id, token: token || null, max_user_bytes: settings.max_user_bytes }, "PUT")); setToken(""); })}>Save connection</button>
    </> : <p>Your administrator manages the inference host and storage limit.</p>}
      <p>Images and prompts are stored by Vashti. Kuno only performs inference.</p>
    </section>}
    {active && <aside className="images-run" role="status"><span>{active.status === "improving" ? "Improving prompt…" : `${active.status[0].toUpperCase()}${active.status.slice(1)}…`} · {active.completed_count} saved{active.request.draft.count !== null && !active.request.improve_only ? ` of ${active.request.draft.count}` : ""}</span><button disabled={active.status === "stopping"} onClick={() => void guard(async () => { const stopped = await write<Run>(`${path}/runs/${active.id}/stop`, {}); setRuns(list => list.map(r => r.id === stopped.id ? stopped : r)); })}><Square size={16} />Stop</button></aside>}
    {view === "gallery" ? <>
      <div className="images-gallery">{gallery.map(item => <article key={item.id}><ImageAsset id={item.asset_id} onClick={() => void openGallery(item)} />{item.variants > 1 && <span className="images-stack-count">{item.variants} versions</span>}<div className="images-actions"><button onClick={() => void beginEdit(item.asset_id)}>Edit</button><button onClick={() => void guard(() => downloadImage(item.asset_id))}>Download</button>{item.session_id && <button onClick={() => void guard(async () => { const source = await requestJson<ImageSession>(`${path}/sessions/${item.session_id}`); onNavigate(source.mode === "generate" ? "generate" : "edit", source.id); })}>Open history</button>}<button onClick={() => void guard(async () => { await requestJson(`${path}/gallery/${item.id}`, { method: "DELETE" }); await refreshGallery(); })}>Remove from gallery</button></div></article>)}</div>
      {!gallery.length && <p className="images-empty">Your generated images will appear here.</p>}
      <div className="images-actions"><button disabled={!offset} onClick={() => setOffset(n => Math.max(0, n - 50))}>Newer</button><button disabled={gallery.length < 50} onClick={() => setOffset(n => n + 50)}>Older</button></div>
    </> : draft && preferences ? <div className="images-flow">
      <details className="images-sessions"><summary>Saved sessions</summary>{sessions.map(s => <button key={s.id} onClick={() => onNavigate(s.mode === "generate" ? "generate" : "edit", s.id)}>{s.title}</button>)}</details>
      <section className="images-results" aria-label="Image history">{shownGroups.map(result => {
        const variants = results.filter(r => r.group_id === result.group_id).reverse(); const index = variants.findIndex(r => r.id === result.id);
        return <article className="images-result" key={result.group_id}>
          <p className="images-result-prompt">{result.recipe.prompt}</p>
          <div onPointerDown={e => { swipeStart.current = { x: e.clientX, y: e.clientY }; }} onPointerUp={e => { const start = swipeStart.current; swipeStart.current = null; if (!start) return; const x = e.clientX - start.x, y = e.clientY - start.y; const next = index + (x < 0 ? 1 : -1); if (Math.abs(x) > 65 && Math.abs(x) > Math.abs(y) * 1.5 && variants[next]) void selectVersion(variants[next]); }}><ImageAsset id={result.asset_id} onClick={() => setViewer({ ids: variants.map(r => r.asset_id), results: variants, index })} /></div>
          <div className="images-actions"><button disabled={busy || Boolean(active)} onClick={() => void start(false, result)}>Retry · new seed</button><button onClick={() => void beginEdit(result.asset_id)}>Edit</button><button onClick={() => void guard(() => downloadImage(result.asset_id))}><Download size={16} />Download</button><button onClick={() => { patchDraft({ recipe: { ...result.recipe }, input_ids: [...result.input_ids], parent_result_id: result.parent_result_id }); composerRef.current?.scrollIntoView({ behavior: "smooth" }); }}>Edit prompt / branch</button></div>
          {variants.length > 1 && <div className="images-variants"><button aria-label="Previous version" disabled={index <= 0 || busy} onClick={() => void selectVersion(variants[index - 1])}><ArrowLeft size={18} /></button><span>{index + 1} / {variants.length}</span><button aria-label="Next version" disabled={index >= variants.length - 1 || busy} onClick={() => void selectVersion(variants[index + 1])}><ArrowRight size={18} /></button></div>}
          <details><summary>Generation settings · {(result.duration_ms / 1000).toFixed(1)}s</summary><p>{result.recipe.width} × {result.recipe.height} · {result.recipe.steps} steps · CFG {result.recipe.cfg} · seed {result.recipe.seed} · {result.recipe.attention} attention</p>{result.recipe.negative_prompt && <p>Negative: {result.recipe.negative_prompt}</p>}</details>
        </article>;
      })}</section>
      {results.length > 0 && <details><summary>All results and branches</summary><div className="images-branch-list">{results.map(r => <button key={r.id} onClick={() => void selectVersion(r)}>{r.recipe.prompt.slice(0, 70)} · seed {r.recipe.seed}</button>)}</div><div className="images-actions"><button disabled={!offset} onClick={() => setOffset(n => Math.max(0, n - 100))}>Newer</button><button disabled={results.length < 100} onClick={() => setOffset(n => n + 100)}>Older</button></div></details>}
      {unseen && <button onClick={() => { composerRef.current?.scrollIntoView({ behavior: "smooth" }); setUnseen(false); }}>New images ↓</button>}
      {active?.preview && <figure className="images-live-preview" aria-label="Image generation preview"><div className="images-asset"><img src={`data:image/png;base64,${active.preview.png_base64}`} alt="Approximate image being generated" /></div><figcaption>Preview · step {active.preview.step} of {active.preview.steps} · the final image will be sharper</figcaption></figure>}
      <div className="images-composer" ref={composerRef}>
        {mode !== "generate" && <>
          <label>Editing mode<select value={mode} disabled={Boolean(active)} onChange={e => { const next = e.target.value as typeof mode; setMode(next); if (next === "original") patchDraft({ input_ids: [...draft.original_input_ids], parent_result_id: null }); }}><option value="iterative">Iterative — use the selected result</option><option value="original">Original images — start from the originals</option></select></label>
          <p className="muted">{mode === "iterative" ? "Each next edit uses only the selected result. You can add references for this edit." : "Every edit uses the original image set. Include every change you want in the prompt."}</p>
          <div className="images-inputs">{draft.input_ids.map((asset, index) => <div key={`${asset}:${index}`}><ImageAsset id={asset} alt={`Reference ${index + 1}`} /><span>{index + 1}</span><button aria-label={`Remove reference ${index + 1}`} onClick={() => { const ids = draft.input_ids.filter((_, i) => i !== index); patchDraft({ input_ids: ids, ...(!draft.parent_result_id || mode === "original" ? { original_input_ids: ids } : {}) }); }}>Remove</button>{index > 0 && <button aria-label={`Move reference ${index + 1} earlier`} onClick={() => { const ids = [...draft.input_ids]; [ids[index - 1], ids[index]] = [ids[index], ids[index - 1]]; patchDraft({ input_ids: ids, ...(!draft.parent_result_id || mode === "original" ? { original_input_ids: ids } : {}) }); }}>←</button>}<button onClick={() => void guard(async () => { const value = await requestJson<{ recipe: Recipe; required_reference_images: number; warnings: string[] }>(`${path}/assets/${asset}/settings`); patchRecipe(value.recipe); if (value.required_reference_images) setError(`Imported settings require ${value.required_reference_images} original reference image(s), in their original order.`); })}>Use settings</button></div>)}</div>
          <label className="images-upload"><ImagePlus size={18} />Add reference images<input type="file" multiple accept="image/png,image/jpeg,image/webp" disabled={busy} onChange={e => { void uploadFiles(e.target.files); e.target.value = ""; }} /></label>
        </>}
        <label>Prompt<textarea rows={4} value={draft.recipe.prompt} onChange={e => patchRecipe({ prompt: e.target.value })} placeholder={mode === "generate" ? "Describe the image you want…" : "Describe what to change…"} /></label>
        {suggestion && <section className="images-suggestion"><h3>Suggested prompt</h3><p>{suggestion.effective_prompt}</p>{draft.recipe.prompt !== suggestion.request.draft.recipe.prompt && <p>Your draft changed since this suggestion was requested. Accepting will replace the current text.</p>}<div className="images-actions"><button disabled={busy} onClick={() => void applySuggestion()}>Use this prompt</button><button onClick={() => setSuggestion(null)}>Keep my prompt</button></div><p className="muted">Your original wording stays in prompt history.</p></section>}
        <div className="images-send-actions"><button disabled={busy || Boolean(active) || !draft.recipe.prompt.trim() || !draft.prompt_help?.model_name} onClick={() => void start(true)}><Sparkles size={18} />Improve prompt</button>{active ? <button disabled={active.status === "stopping"} onClick={() => void guard(async () => { await write(`${path}/runs/${active.id}/stop`, {}); })}><Square size={18} />Stop</button> : <button className="primary-action" disabled={busy || !draft.recipe.prompt.trim()} onClick={() => void start()}>Send</button>}</div>
        <details className="images-prompt-help"><summary>Prompt improvement</summary>
          <label>Assistant model<select value={promptModelValue} onChange={e => { if (!e.target.value) { patchDraft({ prompt_help: null, auto_improve: false }); return; } const [backend_id, model_name] = JSON.parse(e.target.value) as string[]; patchDraft({ prompt_help: { backend_id, model_name, guidance: draft.prompt_help?.guidance ?? "" } }); }}><option value="">Choose a model</option>{modelGroups.map(g => <optgroup key={g.backend.id} label={g.backend.name}>{g.models.map(m => <option key={m.name} value={JSON.stringify([g.backend.id, m.name])}>{m.name}</option>)}</optgroup>)}</select></label>
          <label>Improvement guidance<textarea rows={3} disabled={!draft.prompt_help} value={draft.prompt_help?.guidance ?? ""} placeholder="For example: The images keep coming out blue. Help correct that." onChange={e => { if (draft.prompt_help) patchDraft({ prompt_help: { ...draft.prompt_help, guidance: e.target.value } }); }} /></label>
          <label className="images-check"><input type="checkbox" checked={draft.auto_improve} disabled={!draft.prompt_help} onChange={e => patchDraft({ auto_improve: e.target.checked })} />Improve on every Send</label><p className="muted">When enabled, Send improves and uses your prompt automatically, once for the whole batch. Both versions are saved. Retry keeps the recorded prompt.</p>
        </details>
        <details><summary>Prompt history ({history.length})</summary>{history.map(version => <article className="images-prompt-version" key={version.id}><small>{version.source === "improved" ? `Improved · ${version.model_name}` : "Your prompt"}</small><p>{version.prompt}</p>{version.guidance && <p className="muted">Guidance: {version.guidance}</p>}<button disabled={busy} onClick={() => void guard(async () => { const next = { ...draft, recipe: { ...draft.recipe, prompt: version.prompt }, prompt_version_id: version.id }; await saveDraft(next); setDraft(next); })}>Restore this prompt</button></article>)}</details>
        <details><summary>Customize controls</summary>{controls.map(([key, label]) => <label className="images-check" key={key}><input type="checkbox" checked={!hidden(key)} onChange={e => setPreferences({ ...preferences, hidden_controls: e.target.checked ? preferences.hidden_controls.filter(v => v !== key) : [...preferences.hidden_controls, key] })} />{label}</label>)}<button onClick={() => void guard(async () => { setPreferences(await write<Preferences>(`${path}/preferences`, preferences, "PUT")); })}>Save visibility</button></details>
        <div className="images-controls">
          {!hidden("negative_prompt") && <label className="images-wide">Negative prompt<textarea rows={2} value={draft.recipe.negative_prompt} onChange={e => patchRecipe({ negative_prompt: e.target.value })} /></label>}
          {!hidden("size") && <><label>Aspect ratio<select value="" onChange={e => { const [w, h] = e.target.value.split(":").map(Number); const scale = Math.sqrt(1024 * 1024 / (w * h)); patchRecipe({ width: Math.round(w * scale / 16) * 16, height: Math.round(h * scale / 16) * 16 }); }}><option value="">{draft.recipe.width} × {draft.recipe.height}</option>{["1:1", "3:2", "2:3", "16:9", "9:16"].map(r => <option key={r}>{r}</option>)}</select></label><label>Width<input type="number" min={64} max={2048} step={16} value={draft.recipe.width} onChange={e => patchRecipe({ width: Number(e.target.value) })} /></label><label>Height<input type="number" min={64} max={2048} step={16} value={draft.recipe.height} onChange={e => patchRecipe({ height: Number(e.target.value) })} /></label></>}
          {!hidden("steps") && <label>Steps<input type="number" min={1} max={200} value={draft.recipe.steps} onChange={e => patchRecipe({ steps: Number(e.target.value) })} /></label>}
          {!hidden("cfg") && <label>CFG<input type="number" min={0} max={30} step={0.1} value={draft.recipe.cfg} onChange={e => patchRecipe({ cfg: Number(e.target.value) })} /></label>}
          {!hidden("seed") && <><label>Seed mode<select value={draft.seed_mode} onChange={e => patchDraft({ seed_mode: e.target.value as Draft["seed_mode"] })}><option value="random">Random</option><option value="increment">Increment</option><option value="fixed">Fixed</option></select></label><label>Seed<input inputMode="numeric" value={draft.recipe.seed} onChange={e => patchRecipe({ seed: e.target.value })} /></label></>}
          {!hidden("count") && <><label className="images-check"><input type="checkbox" checked={draft.count === null} onChange={e => patchDraft({ count: e.target.checked ? null : 1 })} />Endless</label>{draft.count !== null && <label>Images<input type="number" min={1} max={100} value={draft.count} onChange={e => patchDraft({ count: Number(e.target.value) })} /></label>}</>}
          {!hidden("attention") && <label>Attention<select value={draft.recipe.attention} onChange={e => patchRecipe({ attention: e.target.value as Recipe["attention"] })}><option value="standard">Standard</option><option value="fast">Fast (Qwen 2.1 on CUDA)</option></select></label>}
        </div>
        <details><summary>Defaults and chat tool controls</summary><p>Choose which settings the model can change in chats. Hidden controls still apply.</p>{controls.filter(([key]) => key !== "count").map(([key, label]) => <label className="images-check" key={key}><input type="checkbox" checked={preferences[controlsProfile].model_may_change.includes(key)} onChange={e => toggleAllowed(key, e.target.checked)} />Allow model to change {label.toLowerCase()}</label>)}<button disabled={busy} onClick={() => void guard(async () => { const updated = { ...preferences, [controlsProfile]: { ...preferences[controlsProfile], recipe: { ...draft.recipe, prompt: "" }, seed_mode: draft.seed_mode } }; setPreferences(await write<Preferences>(`${path}/preferences`, updated, "PUT")); })}>Save current settings as {controlsProfile} defaults</button></details>
        <button disabled={busy} onClick={() => void guard(async () => { await saveDraft(); })}>Save draft</button>
      </div>
    </div> : <p role="status">Loading image workspace…</p>}
    {confirmEndless && <div className="images-modal" role="dialog" aria-modal="true" aria-label="Repeated fixed-seed images"><section><h2>Repeat the same image?</h2><p>Endless generation with a fixed seed and unchanged settings will generally repeat the same image until you stop it.</p><button onClick={() => setConfirmEndless(false)}>Go back</button><button onClick={() => void start(false, undefined, true)}>Generate anyway</button></section></div>}
    {viewer && <div className="images-modal images-viewer" role="dialog" aria-modal="true" aria-label="Image viewer" onPointerDown={e => { swipeStart.current = { x: e.clientX, y: e.clientY }; }} onPointerUp={e => { const start = swipeStart.current; swipeStart.current = null; if (start && Math.abs(e.clientX - start.x) > 65 && Math.abs(e.clientX - start.x) > Math.abs(e.clientY - start.y) * 1.5) moveViewer(e.clientX < start.x ? 1 : -1); }}><button className="images-viewer-close" aria-label="Close viewer" onClick={() => setViewer(null)}><X /></button><ImageAsset id={viewer.ids[viewer.index]} /><div className="images-actions"><button aria-label="Previous image" disabled={viewer.index === 0} onClick={() => moveViewer(-1)}><ArrowLeft /></button><span>{viewer.index + 1} / {viewer.ids.length}</span><button aria-label="Next image" disabled={viewer.index === viewer.ids.length - 1} onClick={() => moveViewer(1)}><ArrowRight /></button><button onClick={() => void beginEdit(viewer.ids[viewer.index])}>Edit</button><button onClick={() => void guard(() => downloadImage(viewer.ids[viewer.index]))}>Download</button><button onClick={() => void guard(async () => { await write(`${path}/gallery/copy/${viewer.ids[viewer.index]}`, {}); if (view === "gallery") await refreshGallery(); })}>Copy to gallery</button>{viewer.results[viewer.index] && <button onClick={() => void selectVersion(viewer.results[viewer.index])}>Use this version</button>}</div></div>}
  </div>;
}
