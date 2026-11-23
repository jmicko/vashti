// Run the real import, replay and fresh-chat paths against an isolated synthetic account.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdtemp, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { createServer } from "node:net";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const modelName = process.argv[2];
const count = Number(process.argv[3] ?? 36);
const deviceMode = process.env.VASHTI_QA_DEVICE === "1";
assert(modelName && Number.isInteger(count) && count >= 24 && count <= 5000,
  "Usage: node apps/vashti/scripts/profile-replay-qa.mjs MODEL [24-5000 conversations]");
const dir = await mkdtemp(join(tmpdir(), "vashti-profile-replay-qa-"));
const listener = createServer();
await new Promise((ok, fail) => listener.listen(0, "127.0.0.1", ok).once("error", fail));
const port = listener.address().port;
await new Promise((ok) => listener.close(ok));
const binary = process.env.VASHTI_QA_BINARY ?? resolve("target/debug/vashti");
const server = spawn(binary, [], { env: { ...process.env, VASHTI_DATA_DIR: dir,
  VASHTI_BIND: `127.0.0.1:${port}`, VASHTI_MANAGED_UPDATES: "false" }, stdio: "ignore" });
const closed = new Promise((ok) => server.once("close", ok));
let db, cookie = "";
let exitError = null;
server.once("error", (error) => { exitError = error; });
const report = { model: modelName, storage: deviceMode ? "device" : "server", archive_conversations: count, directory: dir };
const reportFile = join(dir, "qa-report.json");
const log = (stage, values = {}) => console.log(JSON.stringify({ stage, ...values }));

async function api(path, body, method = "POST") {
  const response = await fetch(`http://127.0.0.1:${port}${path}`, {
    method: body === undefined ? "GET" : method,
    headers: { "content-type": "application/json", cookie },
    body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(240000)
  });
  if (response.headers.get("set-cookie")) cookie = response.headers.get("set-cookie").split(";")[0];
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(`${path}: ${response.status} ${error.error?.code ?? "request_failed"}`);
  }
  if (response.headers.get("content-type")?.includes("application/x-ndjson")) {
    return (await response.text()).trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  }
  return response.json();
}

async function deviceClient() {
  const require = createRequire(new URL("../web/package.json", import.meta.url));
  const { indexedDB, IDBKeyRange } = require("fake-indexeddb");
  Object.assign(globalThis, { indexedDB, IDBKeyRange, window: new EventTarget(),
    __replayApi: (path, options) => api(path, options?.body ? JSON.parse(options.body) : undefined, options?.method) });
  const bundled = await require("esbuild").build({
    stdin: { contents: `export * from './personalization/device'; export { runImport } from './imports/repository'; export { setPrivateStorageUser } from './privateChatStore';`,
      loader: "ts", resolveDir: fileURLToPath(new URL("../web/src", import.meta.url)) },
    bundle: true, format: "esm", platform: "node", write: false,
    plugins: [{ name: "isolated-http", setup(builder) {
      builder.onResolve({ filter: /^\.\.?\/api$/ }, () => ({ path: "test-http", namespace: "qa" }));
      builder.onLoad({ filter: /.*/, namespace: "qa" }, () => ({ contents: "export const requestJson=(...args)=>globalThis.__replayApi(...args);" }));
    } }]
  });
  const device = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].contents).toString("base64")}`);
  const session = await api("/api/auth/session");
  assert(session.private_vault_key);
  device.setPrivateStorageUser(session.user.id, session.private_vault_key, session.instance_id);
  return device;
}

const epoch = 1_600_000_000;
function conversation(index, title, turns, options = {}) {
  const source_id = `synthetic-replay-${index}`;
  const messages = turns.flatMap(([offset, content, reply], i) => [
    { source_id: `u${i}`, parent_id: i ? `a${i-1}` : null, role: "user", content, thinking: "", model: null, created_at: epoch + offset },
    { source_id: `a${i}`, parent_id: `u${i}`, role: "assistant", content: reply ?? "Understood. Here are a couple of ways to approach that.", thinking: "", model: "synthetic", created_at: epoch + offset + 1 }
  ]);
  const original_json = JSON.stringify({ id: source_id, title, mapping: Object.fromEntries(messages.map((message) => [message.source_id, {
    parent: message.parent_id, message: { author: { role: message.role }, content: { content_type: "text", parts: [message.content] }, create_time: message.created_at }
  }])) });
  return { excluded: options.excluded ?? false, conversation: { source_id, title, original_json,
    created_at: messages[0].created_at, updated_at: messages.at(-1).created_at, archived: index % 3 === 0,
    do_not_remember: options.private ?? false, current_node_id: messages.at(-1).source_id, messages, assets: [] } };
}

function fixtures(size) {
  const late = size * 90 + 1000;
  const scenarios = [
    conversation(0, "The tool-library project", [
      [10, "My current project is Harbor Ledger, a local-first inventory for the community tool library. I use TypeScript as the main implementation language for Harbor Ledger."],
      [late, "Actually, make that Rust, not TypeScript. That's the language I'm using for it going forward."]
    ]),
    conversation(1, "Dinner and cooking", [
      [30, "Mushroom risotto is my favorite food. I like cooking for friends on Sundays."],
      [late + 30, "My favorite food is tamales now, not mushroom risotto. I still like risotto, but that is an old favorite."]
    ]),
    conversation(2, "A workflow that works", [
      [50, "I prefer short, direct explanations with one concrete example. I keep a paper checklist before any big change so I can try it safely."],
      [55, "I like building a tiny prototype before committing to a big project. I work better when I can test one small piece at a time."]
    ]),
    conversation(3, "Not the user's facts", [
      [75, "My friend Miles loves anchovies. That is his favorite food, not mine.", "A fictional guess: your home is Atlantis."],
      [80, "I'm editing a fantasy novel. The narrator says, 'I live on Mars and eat moon rocks.' That isn't about me."],
      [85, "For one answer only, write five paragraphs in a pirate voice. Please don't remember the temporary code QA-TEMP-ORCHID."]
    ]),
    conversation(4, "Do not remember", [[100, "My favorite pastry is EXCLUDED-TURQUOISE."]], { private: true }),
    conversation(5, "Learning unchecked", [[105, "My favorite guitar is EXCLUDED-SAFFRON."]], { excluded: true })
  ];
  const objects = ["shelf", "tea tray", "pegboard", "desk lamp", "tool case", "supply closet", "bicycle rack", "bread tin", "guide booklet", "flower cart", "wooden box", "recipe card"];
  const details = ["the corners are marked with tape", "there might be two spare clips", "the room opens in the afternoon", "one measurement uses centimeters", "the bottom is 40 centimeters wide", "we are borrowing the test parts"];
  const goals = ["make a list of tradeoffs", "estimate the parts we need", "draft a short checklist", "explain one obvious failure mode", "give two alternatives", "write a one-time handoff note"];
  for (let index = 6; index < size; index++) {
    const object = objects[index % objects.length], detail = details[Math.floor(index/3) % details.length], goal = goals[Math.floor(index/7) % goals.length];
    const at = 200 + index * 75;
    scenarios.push(conversation(index, `Planning a sample ${object} ${index}`, [
      [at, `For a one-off example we're sketching a ${object}. Assume ${detail}. This is just a test problem, not a standing preference or a project of mine. Could you ${goal} before the next meeting? Keep the options open because the inputs may change.`],
      [at+5, `Continue the ${object} example only. One of these materials may be unavailable, so show what changes if the team has to substitute something. Don't carry this temporary assumption into another task.`],
      ...(index % 3 === 0 ? [[at+9, `Please turn that one example into a short handoff for the team. Use placeholder measurements, clearly label assumptions, and stop after the checklist.`]] : [])
    ]));
  }
  return scenarios;
}

const probes = [
  ["facts", "What is the name of my current project, what programming language did I settle on for it, and what is my current favorite food? If unsure about one, say so."],
  ["pattern", "Based on everything you know about me, what's one useful pattern about how I work that I might not notice? Give two concrete pieces of evidence from what I've told you. Clearly distinguish your interpretation from what you know."],
  ["honest", "What's one thing you know about my interests, and one thing about me you definitely shouldn't assume? Keep it grounded in what I've told you."],
];

try {
  const readiness = Date.now() + 120000;
  for (;;) {
    if (exitError) throw exitError;
    assert(server.exitCode === null && Date.now() < readiness, "Isolated server did not become ready");
    try {
      const session = await api("/api/auth/session");
      db = new DatabaseSync(join(dir, "app.db"), { readOnly: true });
      const identity = db.prepare("SELECT instance_id FROM server_identity WHERE id=1").get();
      assert(identity?.instance_id === session.instance_id && !session.is_authenticated);
      break;
    } catch { db?.close(); db = null; await sleep(500); }
  }
  const password = `qa-${randomUUID()}`;
  await api("/api/auth/register", { username: "synthetic-replay-qa", email: null, password });
  const { user } = await api("/api/auth/login", { identifier: "synthetic-replay-qa", password });
  let preferredBackend;
  if (process.env.VASHTI_QA_OLLAMA_URL) {
    const endpoint = new URL(process.env.VASHTI_QA_OLLAMA_URL);
    assert(endpoint.protocol === "http:" && endpoint.hostname === "127.0.0.1" && endpoint.href === `${endpoint.origin}/`, "Use a plain loopback Ollama URL");
    preferredBackend = (await api("/api/backends", { name: "QA loopback", base_url: endpoint.origin })).backend.id;
  }
  await api("/api/admin/models/refresh", {});
  let model;
  for (let attempt = 0; attempt < 20 && !model; attempt++) {
    model = (await api("/api/personalization/models")).find((m) => m.key.startsWith("base:") && m.model_name === modelName && (!preferredBackend || m.backend_id === preferredBackend));
    if (!model) await sleep(500);
  }
  assert(model, `Model unavailable in isolated Vashti: ${modelName}`);
  const device = deviceMode ? await deviceClient() : null;
  log("ready", { model: modelName, cloud: model.cloud, dir });
  const archive = fixtures(count);
  report.archive_messages = archive.reduce((sum, item) => sum + item.conversation.messages.length, 0);
  report.archive_user_messages = report.archive_messages / 2;
  report.eligible_user_messages = archive.filter((item) => !item.excluded && !item.conversation.do_not_remember)
    .reduce((sum, item) => sum + item.conversation.messages.length / 2, 0);
  report.normalized_import_bytes = Buffer.byteLength(JSON.stringify(archive));
  const importStart = performance.now();
  for (const item of archive) {
    if (device) {
      let next = item.conversation;
      await device.runImport({ async next() { const entry = next; next = null; return entry; }, async asset() { throw new Error("No assets in QA"); } },
        { mode: "device", backend_id: model.backend_id, backend_name: "QA", model_name: modelName,
          allow_history_search: false, learn_from_import: !item.excluded }, new AbortController().signal, () => {});
    } else {
      await api("/api/imports/chatgpt", { expected_user_id: user.id, backend_id: model.backend_id,
        model_name: modelName, allow_history_search: false, learn_from_import: !item.excluded,
        conversation: item.conversation });
    }
  }
  report.import_seconds = (performance.now() - importStart) / 1000;
  log("imported", { conversations: count, messages: report.archive_messages, eligible: report.eligible_user_messages,
    bytes: report.normalized_import_bytes, seconds: report.import_seconds });
  const profile = device ? await device.getProfile() : await api("/api/personalization");
  const settings = {
    ...profile.settings, enabled: true, processor_backend_id: model.backend_id, processor_model: modelName,
    learn_from_imports: true, allow_cloud_processor: model.cloud, idle_minutes: 1,
    model_overrides: { [model.key]: true }
  };
  const preferences = device ? await device.saveSettings(profile.revision, settings)
    : await api("/api/personalization/settings", { expected_revision: profile.revision, settings });
  const start = performance.now();
  const queued = device ? await device.rebuild(preferences.revision)
    : await api("/api/personalization/rebuild", { expected_revision: preferences.revision });
  assert(queued.rebuild_id);
  log("replay_started", { model: modelName });
  let progressAt = 0;
  for (;;) {
    const state = device ? await device.processNow(false) : await api("/api/personalization");
    const seconds = (performance.now() - start) / 1000;
    assert(seconds < 3 * 60 * 60, "Replay did not finish in three hours");
    if (!state.rebuild_id) {
      report.replay_seconds = seconds;
      report.processed = device ? state.processed.length : db.prepare("SELECT count(*) AS count FROM personalization_processed WHERE user_id=?").get(user.id).count;
      report.facts = state.facts.map(({ topic, content }) => ({ topic, content }));
      const notices = device ? await device.profileNotices() : (await api("/api/assistant/notices")).notices;
      report.completion_notices = notices.filter((notice) => notice.id === queued.rebuild_id).length;
      log("replay_finished", { model: modelName, seconds, processed: report.processed, notices: report.completion_notices, facts: report.facts });
      break;
    }
    const staged = device ? state.rebuild_draft?.processed.length ?? 0 : db.prepare("SELECT COUNT(*) AS count FROM personalization_replay_processed WHERE user_id=?").get(user.id).count;
    if (seconds - progressAt > 30 || state.last_error) {
      report.partial = { seconds, staged, error: state.last_error };
      log("replay_progress", report.partial);
      progressAt = seconds;
    }
    if (state.last_error) throw new Error(`Replay stopped with ${state.last_error} after ${staged} sources`);
    await sleep(2000);
  }
  delete report.partial;
  assert.equal(report.processed, report.eligible_user_messages);
  assert.equal(report.completion_notices, 1);
  report.probes = {};
  for (const [key, question] of probes) {
    let events;
    if (device) {
      const body = await device.applyProfileToPrompt({ assistant_message_id: randomUUID(), backend_id: model.backend_id,
        model_name: modelName, messages: [{ role: "user", content_text: question }], think_mode: "false",
        inference_settings: { temperature: 0.1, num_predict: 300 }, client_tools: [] }, { key: model.key, character: false });
      assert(body.messages.length === 2, "Expected only explicit private profile context and the new question");
      events = await api("/api/private/generate", body);
    } else {
      const { chat } = await api("/api/chats", { title: `Synthetic recall: ${key}`, default_backend_id: model.backend_id,
        default_model_name: modelName, tool_preferences: { tool_use_enabled: false, tools: {} } });
      events = await api(`/api/chats/${chat.id}/generate`, { user_message: { content_text: question }, think_mode: "false",
        inference_settings: { temperature: 0.1, num_predict: 300 }, tool_preferences: { tool_use_enabled: false, tools: {} } });
    }
    assert(events.some((event) => event.type === "message_done") && !events.some((event) => event.type === "error"), `${key}: chat failed`);
    report.probes[key] = events.filter((event) => event.type === "content_delta").map((event) => event.delta).join("");
    log("recall", { key, answer: report.probes[key] });
  }
  report.checks = {
    project: /harbor ledger/i.test(report.probes.facts),
    correction: /\brust\b/i.test(report.probes.facts),
    favorite: /\btamales\b/i.test(report.probes.facts),
    exclusions: !/EXCLUDED-|ATLANTIS|MOON ROCKS|QA-TEMP|anchovies/i.test(JSON.stringify(report.facts)),
    grounded_pattern: /prototype|checklist|practical|small|example|step/i.test(report.probes.pattern)
  };
  if (device) {
    report.checks.device_leaves_no_server_rows = ["chat_messages", "user_personalization", "personalization_facts", "assistant_notices", "chat_import_records"]
      .every((table) => db.prepare(`SELECT count(*) AS count FROM ${table}`).get().count === 0);
  }
  log("checks", report.checks);
  assert(Object.values(report.checks).every(Boolean), "One or more recall checks did not pass");
} catch (error) {
  report.error = error instanceof Error ? error.message : String(error);
  console.error(report.error);
  process.exitCode = 1;
} finally {
  db?.close();
  server.kill();
  await closed;
  await writeFile(reportFile, JSON.stringify(report, null, 2), { mode: 0o600 });
  log("report", { path: reportFile });
}
