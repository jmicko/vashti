// Explicit local-only inference smoke test. Synthetic flows are retained for development.
import assert from "node:assert/strict";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";

const base = new URL(process.env.VASHTI_DEV_URL ?? "http://127.0.0.1:7771");
assert(["localhost", "127.0.0.1", "[::1]"].includes(base.hostname), "Local development only");
assert(process.env.VASHTI_DEV_VIDEO_PASSWORD, "Set VASHTI_DEV_VIDEO_PASSWORD");
let cookie = "";
async function api(path, body, method = "POST") {
  const response = await fetch(new URL(path, base), {
    method: body === undefined ? "GET" : method,
    headers: { cookie, ...(body instanceof FormData ? {} : { "content-type": "application/json" }) },
    body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body),
    signal: AbortSignal.timeout(30000)
  });
  const header = response.headers.get("set-cookie");
  if (header) cookie = header.split(";")[0];
  assert(response.ok, `${path}: HTTP ${response.status}`);
  return response.headers.get("content-type")?.includes("video/mp4") ? new Uint8Array(await response.arrayBuffer()) : response.json();
}
const { user } = await api("/api/auth/login", {
  identifier: process.env.VASHTI_DEV_VIDEO_USER ?? "demo-profile-alex",
  password: process.env.VASHTI_DEV_VIDEO_PASSWORD
});
assert(user.username.startsWith("demo-"), "Use a synthetic development account");
const hosts = await api("/api/videos/hosts");
const host = hosts.find(h => h.capabilities.workflows.some(w => w.id === "minimax-h3-reference-video/v1"));
assert(host, "The installed video modes are not advertised by Vashti");
const inputs = [];
for (const relative of ["../../vashti-android/icons/64x64.png", "../../vashti-android/icons/128x128.png"]) {
  const bytes = await readFile(new URL(relative, import.meta.url));
  const form = new FormData();
  form.set("image", new Blob([bytes], { type: "image/png" }), "synthetic-reference.png");
  inputs.push({ ...(await api("/api/videos/frames", form)), bytes });
}
const modes = [
  { name: "text", workflow: "minimax-h3-video/v1", input_mode: "text", prompt: "A small red cube rotates slowly on a neutral gray surface, soft studio light." },
  { name: "keyframes", workflow: "minimax-h3-video/v1", input_mode: "keyframes", prompt: "Gentle natural motion, preserve the green logo.", first_frame_id: inputs[0].id, last_frame_id: inputs[1].id },
  { name: "references", workflow: "minimax-h3-reference-video/v1", input_mode: "references", prompt: "The green logo in <Picture 1> and <Picture 2> rotates gently on a black background.", reference_image_ids: inputs.map(i => i.id) }
];
const set = process.env.VASHTI_DEV_VIDEO_SET ?? "modes";
assert(["modes", "memory"].includes(set), "Unknown video test set");
const scaled = {
  workflow: "minimax-h3-image-to-video/v1", input_mode: "keyframes",
  prompt: "Gentle camera motion, preserve the full green logo.", first_frame_id: inputs[1].id,
};
const cases = set === "memory" ? [
  { ...scaled, name: "high", width: 1280, height: 704, frames: 22 },
  { ...scaled, name: "long", width: 832, height: 544, frames: 243 },
  { ...scaled, name: "cancel", width: 832, height: 544, frames: 243 },
] : modes;
if (set === "memory") {
  const workflow = host.capabilities.workflows.find((w) => w.id === scaled.workflow);
  assert.equal(workflow?.limits, "memory", "Vashti must forward the installed memory contract");
}
const directory = process.env.VASHTI_VIDEO_OUTPUT ?? "/tmp/vashti-video-modes";
await mkdir(directory, { recursive: true });
for (const test of cases) {
  const { name, ...patch } = test;
  const draft = { backend_id: host.backend.id, seed: "42", seed_mode: "increment", width: 512, height: 512, frames: 22, audio: true, first_frame_id: null, last_frame_id: null, reference_image_ids: [], ...patch };
  const session = await api("/api/videos/sessions", { draft });
  await api(`/api/videos/sessions/${session.id}/title`, { title: `Video integration check / ${name}` }, "PATCH");
  const started = Date.now();
  const { id } = await api(`/api/videos/sessions/${session.id}/runs`, { request_id: randomUUID(), draft, retry_of: null });
  let status;
  while (Date.now() - started < 10 * 60 * 1000) {
    status = await api(`/api/videos/runs/${id}`);
    if (name === "cancel" && status.status === "generating") {
      await api(`/api/videos/runs/${id}/stop`, {});
      status = await api(`/api/videos/runs/${id}`);
      break;
    }
    if (!["queued", "generating", "saving"].includes(status.status)) break;
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  if (name === "cancel") {
    assert.equal(status?.status, "cancelled");
    console.log(JSON.stringify({ mode: name, session: session.id, run: id, elapsed_ms: Date.now() - started }));
    continue;
  }
  assert.equal(status?.status, "complete", `${name}: ${JSON.stringify(status)}`);
  const bytes = await api(`/api/videos/runs/${id}/result`);
  assert(bytes.length > 1000, "Missing video bytes");
  await writeFile(`${directory}/${name}.mp4`, bytes);
  console.log(JSON.stringify({ mode: name, session: session.id, run: id, bytes: bytes.length, elapsed_ms: Date.now() - started, artifact: `${directory}/${name}.mp4` }));
}
