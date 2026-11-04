// Manual browser regression fixture. No request can reach a real inference host.
// Open /tests/images.html with Vite; this entry is not part of the production build.
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ImagesWorkspace } from "../src/images/ImagesWorkspace";
import { ImageFlowHistory } from "../src/images/ImageFlowHistory";
import { setPrivateStorageUser, writePrivateImage, readPrivateImage } from "../src/privateChatStore";
import { imageDataUrl } from "../src/saveImage";
import { newDraft, type Preferences, type ImageSession, type Run, type ImageResult, type ImageView } from "../src/images/types";
import "../src/styles.css";

const edit = "qwen-image-2.1-edit/v1";
const generate = "qwen-image-2.1-text-to-image/v1";
const recipe = { workflow: generate, prompt: "", negative_prompt: "", seed: "1",
  width: 768, height: 512, steps: 20, cfg: 4, attention: "standard" as const };
const preferences: Preferences = {
  generate: { recipe, backend_id: "fixture-host", count: 1, seed_mode: "increment", model_may_change: [] },
  edit: { recipe: { ...recipe, workflow: edit }, backend_id: "fixture-host", count: 1,
    seed_mode: "increment", edit_mode: "original", model_may_change: [] },
  hidden_controls: [],
};
const backend = { id: "fixture-host", name: "Simulated host" };
const hosts = [{ backend, service: { state: "ready", message: "Simulated inference",
  capabilities: { service: "kuno", discovery_version: 1, version: "fixture", inference_auth: "none",
    images: { api_version: 1, state: "ready", access: "public", reason: null,
      generate_path: "/unused", response_formats: ["ndjson"], workflows: [generate, edit].map(id => ({
        id, state: "ready", reason: null, reference_images_min: id === edit ? 1 : 0,
        reference_images_max: id === edit ? 10 : 0, attention: ["standard"],
      })) } } } }];
const blobs = new Map<string, Blob>();
function picture(id: string): Promise<Blob> {
  if (blobs.has(id)) return Promise.resolve(blobs.get(id)!);
  const canvas = document.createElement("canvas");
  canvas.width = 768; canvas.height = 512;
  const context = canvas.getContext("2d")!;
  const number = Number(id.replace(/\D/g, "")) || 1;
  context.fillStyle = ["#872e43", "#267260", "#3866ac", "#785d26"][number % 4];
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#ffffff"; context.lineWidth = 6;
  context.strokeRect(32, 32, 704, 448);
  context.fillStyle = "#ffffff"; context.font = "bold 64px sans-serif";
  context.fillText(id, 70, 280);
  return new Promise(resolve => canvas.toBlob(blob => { blobs.set(id, blob!); resolve(blob!); }, "image/png"));
}
const session: ImageSession = {
  id: "fixture-flow", title: "Reference ordering test", mode: "original",
  draft: { ...newDraft(preferences.edit), recipe: { ...preferences.edit.recipe, prompt: "Fixture edit" },
    input_ids: ["reference-1", "reference-2", "reference-3"],
    original_input_ids: ["reference-1", "reference-2", "reference-3"] },
  revision: 1, updated_at: 1, active_result_id: null,
};
const runs: Run[] = [], results: ImageResult[] = [];
const now = Math.floor(Date.now() / 1000);
const gallery = Array.from({ length: 550 }, (_, index) => ({
  id: `fixture-tile-${index + 1}`, asset_id: `photo-${index % 60 + 1}`,
  group_id: index === 3 ? "stack" : null, session_id: index === 3 ? "stack-session" : null,
  width: 768, height: 512, variants: index === 3 ? 2 : 1,
  created_at: now - Math.floor(index / 15) * 86400 - index % 15 * 60,
}));
const stack = ["photo-4-older", "photo-4"].map((asset_id, index) => ({
  id: `stack-${index}`, asset_id, run_id: "stack-run", group_id: "stack", session_id: "stack-session",
  parent_result_id: null, recipe, input_ids: ["reference-1"], duration_ms: 1000, created_at: index,
  selected: index === 1,
})).reverse();
let thumbnailRequests = 0, originalRequests = 0;
let failingOriginals = 0;
let preferenceWrites = 0, policyWrites = 0;
let policy = { backend_id: null as string | null, max_user_bytes: 512 * 1048576, service: hosts[0].service };
const sessions: ImageSession[] = [session, ...Array.from({length:65}, (_, index) => ({
  ...structuredClone(session), id: `history-${index}`, title: `Saved drawing ${index}`, updated_at: now - index,
}))];
const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const diagnostics = () => {
  const output = document.querySelector("#fixture-status");
  if (output) output.textContent = JSON.stringify({
    thumbnailRequests, originalRequests, turns: runs.length,
    preferenceWrites, policyWrites, maxUserMiB: policy.max_user_bytes / 1048576,
    parents: results.map(result => result.parent_result_id),
    submitted: runs.map(run => run.request.draft.input_ids),
  });
};
window.fetch = async (input, options = {}) => {
  const url = new URL(String(input), location.origin), path = url.pathname;
  const method = options.method ?? "GET";
  const body = typeof options.body === "string" ? JSON.parse(options.body) : {};
  const asset = /^\/api\/images\/assets\/([^/]+)(\/thumbnail)?$/.exec(path);
  if (asset) {
    if (asset[2]) thumbnailRequests++; else originalRequests++;
    diagnostics();
    await delay(asset[2] ? 800 : 200);
    if (!asset[2] && failingOriginals > 0) {
      failingOriginals--;
      return new Response("Fixture failure", {status:503});
    }
    return new Response(await picture(asset[1]), { headers: { "Content-Type": "image/png" } });
  }
  if (path === "/api/images/hosts") return json(hosts);
  if (path === "/api/images/preferences") {
    if (method === "PUT") { Object.assign(preferences, structuredClone(body)); preferenceWrites++; diagnostics(); }
    return json(preferences);
  }
  if (path === "/api/images/settings") {
    if (method === "PUT") { policy = { ...policy, ...body }; policyWrites++; diagnostics(); }
    return json(policy);
  }
  if (path === "/api/images/runs/active") return json(runs.filter(run => run.status === "generating"));
  if (path === "/api/images/gallery") {
    await delay(200);
    const start = Number(url.searchParams.get("offset") ?? 0);
    const selectedSession = url.searchParams.get("session");
    return json(gallery.filter(item => !selectedSession || item.session_id === selectedSession).slice(start, start + 50));
  }
  if (path === "/api/images/sessions") {
    const query = (url.searchParams.get("q") ?? "").toLowerCase();
    const offset = Number(url.searchParams.get("offset") ?? 0);
    await delay(query ? 300 : 600);
    return json(sessions.filter(value => value.title.toLowerCase().includes(query)).slice(offset, offset + 50));
  }
  const history = /^\/api\/images\/sessions\/([^/]+)(\/title)?$/.exec(path);
  if (history && history[1] !== session.id) {
    const value = sessions.find(value => value.id === history[1]);
    if (history[2] && value && method === "PATCH") { value.title = body.title; return json(value); }
    if (method === "DELETE") { const index = sessions.findIndex(value => value.id === history[1]); if (index >= 0) sessions.splice(index,1); return json({}); }
  }
  if (path === "/api/images/sessions/stack-session/results") return json(stack);
  if (path === "/api/images/sessions/fixture-flow") {
    if (method === "PUT") Object.assign(session, body, { revision: session.revision + 1 });
    return json(session);
  }
  if (path === "/api/images/sessions/fixture-flow/results") return json(results);
  if (path === "/api/images/sessions/fixture-flow/prompts") return json([]);
  if (path === "/api/images/sessions/fixture-flow/runs") {
    if (method === "GET") return json(runs);
    const run: Run = { id: crypto.randomUUID(), session_id: session.id, status: "generating",
      completed_count: 0, effective_prompt: body.draft.recipe.prompt, prompt_version_id: null,
      error: null, request: body, created_at: Math.floor(Date.now() / 1000) };
    runs.unshift(run);
    session.active_result_id = body.draft.parent_result_id;
    diagnostics();
    setTimeout(() => {
      const original = body.retry_result_id && results.find(result => result.id === body.retry_result_id);
      const result: ImageResult = { id: crypto.randomUUID(), asset_id: `output-${runs.length}`,
        run_id: run.id, origin_run_id: original?.origin_run_id ?? run.id, session_id: session.id,
        group_id: original?.group_id ?? run.id,
        parent_result_id: original ? original.parent_result_id : body.draft.parent_result_id,
        recipe: original?.recipe ?? body.draft.recipe, input_ids: original?.input_ids ?? body.draft.input_ids,
        duration_ms: 1200, created_at: run.created_at, selected: true };
      results.unshift(result);
      run.status = "complete"; run.completed_count = 1;
      session.active_result_id = result.id;
      diagnostics();
    }, 1200);
    return json(run);
  }
  const run = runs.find(value => path === `/api/images/runs/${value.id}`);
  if (run) return json(run);
  if (/^\/api\/images\/results\/[^/]+\/select$/.test(path)) {
    session.active_result_id = path.split("/")[4]; return json({});
  }
  throw new Error(`Fixture blocked unexpected request: ${method} ${path}`);
};

// A separate encrypted vault keeps the user's account and private images untouched.
setPrivateStorageUser("image-browser-fixture-v2", {
  user_id: "image-browser-fixture-v2", key_material: btoa("f".repeat(32)),
}, "image-browser-fixture-v2");
for (const item of gallery) {
  if (!await readPrivateImage(`asset:${item.asset_id}`))
    await writePrivateImage(`asset:${item.asset_id}`, { id: item.asset_id, width: 768, height: 512,
      mime_type: "image/png", data: await imageDataUrl(await picture(item.asset_id)) });
  await writePrivateImage(`gallery:${item.id}`, { ...item, group_id: null, session_id: null, variants: 1 });
}
for (const value of sessions) await writePrivateImage(`flow:${value.id}`, { session:value, results:[], runs:[], hiddenGroups:[] });
function Fixture() {
  const [view, setView] = useState<ImageView>("gallery");
  const [mode, setMode] = useState<"server" | "device">("server");
  const [selectedSession, setSelectedSession] = useState<string | undefined>();
  const [history, setHistory] = useState(false);
  function openGallery(storageMode: "server" | "device") {
    setMode(storageMode); setView("gallery"); setSelectedSession(undefined); setHistory(false);
  }
  return <div style={{ display: "grid", gridTemplateRows: "auto auto minmax(0, 1fr)", height: "100dvh" }}>
    <nav style={{ display: "flex", gap: 8, padding: 8, flexWrap: "wrap" }}>
      <button onClick={() => openGallery("server")}>Server gallery</button>
      <button onClick={() => openGallery("device")}>Device gallery</button>
      <button onClick={() => { setMode("server"); setView("flow"); setSelectedSession(session.id); setHistory(false); }}>Edit flow</button>
      <button onClick={() => setHistory(value => !value)}>History</button>
      <button onClick={() => { failingOriginals = 3; }}>Fail next originals</button>
    </nav>
    <output id="fixture-status" style={{ fontSize: 11, overflowWrap: "anywhere", maxHeight: 48, overflow: "auto" }}>Fixture ready</output>
    {history ? <div className="chat-history" style={{padding:16}}><ImageFlowHistory key={mode} storageMode={mode}
      onOpen={() => { setSelectedSession(session.id); setView("flow"); setHistory(false); }} onRemoved={() => {}} /></div>
      : <ImagesWorkspace key={mode} storageMode={mode} view={view} sessionId={selectedSession}
      newFlowKey={0} headerTarget={null} isAdmin={true} modelGroups={[]}
      onNavigate={(next,id) => { setView(next); setSelectedSession(id); }} onOpenBackends={() => {}} />}
  </div>;
}
createRoot(document.getElementById("root")!).render(<Fixture />);
