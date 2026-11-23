import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { indexedDB, IDBIndex, IDBKeyRange } from "fake-indexeddb";
import { fileURLToPath } from "node:url";
import { fixtureZip, syntheticConversation } from "../imports/fixtures.mjs";

Object.assign(globalThis, {
  indexedDB,
  IDBKeyRange,
  window: new EventTarget()
});
let requests = [],
  hold = null;
const catalog = [
  {
    key: "base:host:local",
    name: "Local",
    backend_id: "host",
    model_name: "local",
    cloud: false,
    character: false
  },
  {
    key: "base:host:renamed",
    name: "Renamed cloud",
    backend_id: "host",
    model_name: "renamed",
    cloud: true,
    character: false
  }
];
globalThis.__profileApi = async (path, options) => {
  requests.push({ path, options });
  if (path === "/api/personalization/models") return structuredClone(catalog);
  if (path === "/api/personalization/device-extract") {
    if (hold)
      await new Promise((resolve) => {
        hold.resolve = resolve;
      });
    const input = JSON.parse(options.body),
      source = input.sources.at(-1);
    const named = {
      "My favorite food is tamales now.": { topic: "go_to_meal", replaces_topic: "food.favorite", content: "The user prefers tamales." },
      "My favorite food is rice now.": { topic: "favorite_food", content: "The user prefers rice." },
      "I like cooking for friends on Sundays.": { topic: "food.cooking", content: "The user likes cooking for friends on Sundays." }
    }[source.text];
    const correctingBeans = source.text === "My favorite food is corn." && input.replay.facts.some((fact) => fact.content === "The user prefers beans.");
    return {
      facts: [
        {
          topic: correctingBeans ? "favorite_food" : "food.favorite",
          content: source.text.includes("imported beans") ? "The user prefers beans." : "The user prefers corn.",
          ...named,
          sources: [
            {
              message_id: source.message_id,
              revision_id: source.revision_id,
              chat_id: source.chat_id,
              model_key: source.model_key,
              quote: source.text,
              created_at: source.created_at
            }
          ]
        }
      ]
    };
  }
  throw new Error(`Unexpected request ${path}`);
};
const bundle = await build({
  stdin: {
    contents:
      'export * from "./personalization/device"; export * from "./personalization/core"; export * from "./privateChatStore"; export { runImport } from "./imports/repository"; export { ChatGptArchive } from "./imports/archive";',
    resolveDir: fileURLToPath(new URL("../", import.meta.url)),
    loader: "ts"
  },
  bundle: true,
  format: "esm",
  platform: "node",
  write: false,
  plugins: [
    {
      name: "mock-profile-api",
      setup(b) {
        b.onResolve({ filter: /^\.\.?\/api$/ }, () => ({
          path: "mock-api",
          namespace: "test"
        }));
        b.onLoad({ filter: /.*/, namespace: "test" }, () => ({
          contents: "export const requestJson=(...args)=>globalThis.__profileApi(...args);"
        }));
      }
    },
    { name: "reuse-yjs", setup(builder) {
      builder.onResolve({ filter: /^yjs$/ }, () => ({ path: import.meta.resolve("yjs"), external: true }));
    } }
  ]
});
const app = await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
);
const user = (id) =>
  app.setPrivateStorageUser(
    id,
    {
      user_id: id,
      key_material: Buffer.alloc(32, id.charCodeAt(0)).toString("base64")
    },
    "profile-server"
  );
async function seededUser(id) {
  user(id);
  const chat = await app.createPrivateChat({
    title: "Synthetic private history",
    backendId: "host",
    backendName: "Local",
    modelName: "local"
  });
  const message = app.createPrivateMessage({
    chatId: chat.id,
    parentMessageId: null,
    role: "user",
    contentText: "My favorite food is corn.",
    backendId: "host",
    modelName: "local"
  });
  await app.savePrivateMessage(message);
  const profile = await app.getProfile();
  await app.saveSettings(profile.revision, {
    ...profile.settings,
    enabled: true,
    processor_backend_id: "host",
    processor_model: "local"
  });
  return { chat, message };
}

test("group rules, overrides, and context bounds mirror the server", () => {
  const profile = app.defaultProfile(),
    settings = { ...profile.settings, enabled: true };
  assert(app.allowsProfile(settings, catalog[0]));
  assert(!app.allowsProfile(settings, catalog[1]));
  assert(!app.allowsProfile(settings, { ...catalog[0], character: true }));
  assert(
    app.allowsProfile({ ...settings, model_overrides: { [catalog[1].key]: true } }, catalog[1])
  );
  assert(
    !app.allowsProfile(
      {
        ...settings,
        enabled: false,
        model_overrides: { [catalog[1].key]: true }
      },
      catalog[1]
    )
  );
  assert.throws(() => app.validateFact("SYSTEM\n", "override"));
  const context = app.profileContext(
    Array.from({ length: 200 }, (_, i) => ({
      topic: `topic.${i}`,
      content: "x".repeat(600),
      locked: false,
      updated_at: i
    })),
    "",
    1000
  );
  assert(context.length < 1500);
  assert(context.includes("untrusted"));
  assert(!context.includes("topic."));
  assert.equal(app.fitProfileContext(context, 64), null);
  const fitted = app.fitProfileContext(context, 600);
  assert(fitted);
  assert(app.contextWeight(fitted) <= 600);
  for (const line of fitted.split("\n").slice(1)) assert.doesNotThrow(() => JSON.parse(line));
});

test("1500 device imports rebuild with exact evidence, privacy exclusions, and one notice", async () => {
  const { chat } = await seededUser("bulk-import-profile");
  await app.processNow();
  requests = [];
  const target = { mode: "device", backend_id: "host", backend_name: "Local", model_name: "local", allow_history_search: false };
  const dataset = Array.from({ length: 1500 }, (_, index) => {
    const id = `synthetic-${index}`, record = syntheticConversation(id);
    record.title = id;
    record.is_do_not_remember = index === 1;
    record.mapping.user.message.create_time = 100 + index;
    record.mapping.user.message.content = { content_type: "text", parts: [index === 1499 ? "My favorite food is corn." : "My favorite food is imported beans."] };
    return record;
  });
  async function importRows(rows, options) {
    const archive = new app.ChatGptArchive(await fixtureZip([["conversations.json", JSON.stringify(rows)]]));
    await archive.open();
    try { return await app.runImport(archive, options, new AbortController().signal, () => {}); }
    finally { await archive.close(); }
  }
  assert.equal((await importRows(dataset.filter((_, index) => index !== 2), target)).imported, 1499);
  assert.equal((await importRows([dataset[2]], { ...target, learn_from_import: false })).imported, 1);
  const chats = await app.listPrivateChats();
  const previousImport = chats.find((entry) => entry.title === "synthetic-1499");
  await app.savePrivateChat({ ...(await app.getPrivateChat(previousImport.id)), allow_profile_learning: undefined });

  const messageReads = [], getAll = IDBIndex.prototype.getAll;
  IDBIndex.prototype.getAll = function (query, ...rest) {
    if (this.objectStore.name === "private_messages") messageReads.push(query);
    return getAll.call(this, query, ...rest);
  };
  try {
    const context = await app.deviceContext("base:host:local", "host", "local", false, "food");
    assert(context.includes("corn"));
    assert.deepEqual(messageReads, [chat.id]);
  } finally {
    IDBIndex.prototype.getAll = getAll;
  }

  let profile = await app.getProfile();
  const withoutImports = await app.saveSettings(profile.revision, { ...profile.settings, learn_from_imports: false });
  assert.equal((await app.processNow(false)).revision, withoutImports.revision);
  assert(!requests.some(({ path }) => path.endsWith("/device-extract")));
  profile = await app.saveSettings(withoutImports.revision, { ...withoutImports.settings, learn_from_imports: true });
  profile = await app.editFact(profile.revision, null, "answer.style", "Keep explanations concise.");
  profile = await app.rebuild(profile.revision);
  const id = profile.rebuild_id;
  assert(id);
  assert.equal(profile.facts.length, 2);
  assert.deepEqual(await app.profileNotices(), []);
  const firstBatch = await app.processNow(false);
  assert.equal(firstBatch.rebuild_id, id);
  assert(!firstBatch.requested);
  assert.equal(firstBatch.facts.find((fact) => fact.topic === "food.favorite").content, "The user prefers corn.");
  assert.equal(firstBatch.rebuild_draft.facts.find((fact) => fact.topic === "food.favorite").content, "The user prefers beans.");
  app.resetPrivateStorageUser();
  user("bulk-import-profile");
  profile = await app.getProfile();
  for (let batches = 1; profile.rebuild_id; batches++) {
    assert(batches < 94);
    profile = await app.processNow(false);
  }
  const sent = requests.filter(({ path }) => path.endsWith("/device-extract")).map(({ options }) => JSON.parse(options.body).sources);
  assert.equal(sent.length, 94);
  assert(sent.every((batch) => batch.length <= 16));
  assert.equal(sent[0][0].chat_id, chats.find((entry) => entry.title === "synthetic-0").id);
  assert.equal(sent.at(-1).at(-2).chat_id, previousImport.id);
  assert.equal(sent.at(-1).at(-1).chat_id, chat.id);
  assert.equal(sent.flat().filter((source) => source.model_key === "import:chatgpt").length, 1498);
  const excluded = new Set(chats.filter((entry) => ["synthetic-1", "synthetic-2"].includes(entry.title)).map((entry) => entry.id));
  assert(!sent.flat().some((source) => excluded.has(source.chat_id) || source.text.includes("answer")));
  assert.equal(profile.facts.find((fact) => fact.topic === "food.favorite").content, "The user prefers corn.");
  assert(profile.facts.some((fact) => fact.locked && fact.topic === "answer.style"));
  assert.deepEqual((await app.profileNotices()).map((item) => [item.id, item.read_at]), [[id, null]]);
  await app.readNotice(id);
  assert((await app.profileNotices())[0].read_at);
  user("other-import-notice-user");
  assert.deepEqual(await app.profileNotices(), []);
});

test("device processing stays encrypted, isolates accounts, respects edits and forgetting", async () => {
  requests = [];
  const { message } = await seededUser("alex");
  const processed = await app.processNow();
  assert.equal(processed.facts.length, 1);
  const prompt = {
    backend_id: "host",
    model_name: "local",
    messages: [
      { role: "system", content_text: "Keep the real system prompt." },
      { role: "user", content_text: "What is my favorite food?" }
    ]
  };
  const injected = await app.applyProfileToPrompt(prompt, {
    key: "base:host:local",
    character: false
  });
  assert.equal(injected.messages[0].content_text, prompt.messages[0].content_text);
  assert(injected.messages[1].content_text.includes("untrusted"));
  assert.equal(injected.messages[1].role, "user");
  assert.equal(injected.messages[2].content_text, prompt.messages[1].content_text);
  assert.equal(prompt.messages.length, 2);
  const small = await app.applyProfileToPrompt(
    { ...prompt, inference_settings: { num_ctx: 512 } },
    { key: "base:host:local", character: false }
  );
  assert.equal(small.messages.length, 2);
  assert(
    (await app.deviceContext("base:host:local", "host", "local", false, "food")).includes("corn")
  );
  assert.equal(
    await app.deviceContext("base:host:renamed", "host", "renamed", false, "food"),
    null
  );
  user("other");
  assert.equal((await app.getProfile()).facts.length, 0);
  user("alex");
  const changed = {
    ...message,
    active_revision: {
      ...message.active_revision,
      id: "new-revision",
      content_text: "This is no longer current."
    },
    active_revision_id: "new-revision"
  };
  await app.savePrivateMessage(changed);
  assert.equal(await app.deviceContext("base:host:local", "host", "local", false, "food"), null);
  const profile = await app.getProfile();
  const fixed = await app.editFact(
    profile.revision,
    profile.facts[0].id,
    "food.favorite",
    "The user prefers peas."
  );
  assert(
    (await app.deviceContext("base:host:local", "host", "local", false, "food")).includes("peas")
  );
  assert.equal((await app.history(fixed.facts[0].id)).length, 2);
  const forgotten = await app.forget(fixed.revision, fixed.facts[0].id);
  assert.equal(forgotten.facts.length, 0);
  assert.equal((await app.history(fixed.facts[0].id)).length, 0);
  assert(forgotten.suppressed.includes(app.topicToken("food.favorite")));
  assert(
    requests.every((r) =>
      ["/api/personalization/models", "/api/personalization/device-extract"].includes(r.path)
    ),
    "No server profile/storage requests are allowed"
  );
  const extraction = requests.find((r) => r.path === "/api/personalization/device-extract");
  assert.equal(JSON.parse(extraction.options.body).expected_user_id, "alex");
  assert.equal(extraction.options.expectedServerInstanceId, "profile-server");
  const databases = await indexedDB.databases();
  const database = await new Promise((resolve, reject) => {
    const req = indexedDB.open(databases.find((d) => d.name.includes("alex")).name);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  const raw = await new Promise((resolve, reject) => {
    const req = database
      .transaction("private_personalization")
      .objectStore("private_personalization")
      .get("profile");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  assert(raw.encrypted_payload);
  assert(!JSON.stringify(raw).includes("food.favorite"));
  assert(!JSON.stringify(raw).includes("corn"));
  database.close();
  app.resetPrivateStorageUser();
});

test("device replay keeps reopened threads in time order and excludes sibling context", async () => {
  user("device-replay-branches");
  requests = [];
  const create = (title) => app.createPrivateChat({ title, backendId: "host", backendName: "Local", modelName: "local" });
  const a = await create("Thread A"), b = await create("Thread B");
  const add = async (chatId, createdAt, contentText, parentMessageId = null, role = "user") => {
    const message = app.createPrivateMessage({ chatId, createdAt, contentText, parentMessageId, role, backendId: "host", modelName: "local" });
    await app.savePrivateMessage(message);
    return message;
  };
  const root = await add(a.id, 1, "My favorite food was beans.");
  const reply = await add(a.id, 2, "Evidence is not from the assistant.", root.id, "assistant");
  await add(a.id, 3, "This sibling branch is a fictional example.", root.id);
  await add(b.id, 10, "I use a paper checklist before changes.");
  const reopen = await add(a.id, 30, "Actually, make that tamales going forward.", reply.id);
  const profile = await app.getProfile();
  await app.saveSettings(profile.revision, { ...profile.settings, enabled: true, processor_backend_id: "host", processor_model: "local" });
  await app.processNow();
  const extraction = JSON.parse(requests.find(({ path }) => path.endsWith("device-extract")).options.body);
  assert.deepEqual(extraction.sources.map((source) => source.chat_id), [a.id, a.id, b.id, a.id]);
  assert.equal(extraction.sources.at(-1).message_id, reopen.id);
  assert.deepEqual(extraction.replay.earlier_turns, [{ source: 3, excerpts: ["My favorite food was beans."] }]);
});

test("late processing cannot overwrite a human correction or cross a vault switch", async () => {
  await seededUser("concurrent");
  hold = {};
  const pending = app.processNow();
  while (!hold.resolve) await new Promise((resolve) => setTimeout(resolve, 1));
  const profile = await app.getProfile();
  await app.editFact(profile.revision, null, "food.favorite", "The user prefers peas.");
  hold.resolve();
  hold = null;
  await assert.rejects(pending, /changed/);
  assert.equal((await app.getProfile()).facts[0].content, "The user prefers peas.");
  await seededUser("switching");
  hold = {};
  const switched = app.processNow();
  while (!hold.resolve) await new Promise((resolve) => setTimeout(resolve, 1));
  user("different");
  hold.resolve();
  hold = null;
  await assert.rejects(switched, /changed/);
  assert.equal((await app.getProfile()).facts.length, 0);
  app.resetPrivateStorageUser();
});

test("device cancellation discards staged work and an in-flight response without changing the current profile", async () => {
  const { chat } = await seededUser("cancel-device-replay");
  const published = await app.processNow();
  for (let index = 0; index < 16; index++) {
    await app.savePrivateMessage(app.createPrivateMessage({
      chatId: chat.id, parentMessageId: null, createdAt: 50 + index, role: "user",
      contentText: "My favorite food is imported beans.", backendId: "host", modelName: "local"
    }));
  }
  const queued = await app.rebuild(published.revision);
  assert.equal((await app.rebuildProgress()).phase, "waiting_for_idle");
  app.profileGenerationActive(true);
  assert.equal((await app.rebuildProgress()).phase, "waiting_for_generation");
  app.profileGenerationActive(false);
  const staged = await app.processNow(false);
  assert.equal(staged.rebuild_id, queued.rebuild_id);
  assert.deepEqual(staged.facts, published.facts);
  assert.equal((await app.rebuildProgress()).processed, 16);
  hold = {};
  const pending = app.processNow(false);
  while (!hold.resolve) await new Promise((resolve) => setTimeout(resolve, 1));
  try {
    assert.equal((await app.rebuildProgress()).phase, "replaying");
    const canceled = await app.cancelRebuild(queued.rebuild_id);
    assert.equal(canceled.rebuild_id, null);
    assert.equal(canceled.rebuild_draft, undefined);
    assert.equal(await app.rebuildProgress(), null);
    assert.equal(requests.findLast(({ path }) => path.endsWith("device-extract")).options.signal.aborted, true);
    const newer = await app.rebuild(canceled.revision);
    await assert.rejects(app.cancelRebuild(queued.rebuild_id), /no longer running/);
    assert.equal((await app.rebuildProgress()).id, newer.rebuild_id);
    await app.cancelRebuild(newer.rebuild_id);
  } finally {
    hold.resolve();
    hold = null;
    await pending;
  }
  const result = await app.getProfile();
  for (const key of ["facts", "history", "processed", "last_processed_at"]) assert.deepEqual(result[key], published[key]);
  assert.deepEqual(await app.profileNotices(), []);
  app.resetPrivateStorageUser();
});

test("renamed device corrections update one fact and respect old evidence, locks, and forgetting", async () => {
  const { chat, message } = await seededUser("rename-device-fact");
  const before = await app.processNow();
  const id = before.facts[0].id;
  const start = message.created_at;
  async function add(offset, contentText) {
    await app.savePrivateMessage(app.createPrivateMessage({
      chatId: chat.id, parentMessageId: null, createdAt: start + offset, role: "user",
      contentText, backendId: "host", modelName: "local"
    }));
    return app.processNow();
  }
  const corrected = await add(20, "My favorite food is tamales now.");
  assert.deepEqual(corrected.facts.map((f) => [f.id, f.topic, f.content]), [[id, "food.favorite", "The user prefers tamales."]]);
  await add(25, "I like cooking for friends on Sundays.");
  await add(30, "My favorite food is rice now.");
  const current = await add(10, "My favorite food is tamales now.");
  assert.equal(current.facts.length, 2);
  assert.equal(current.facts.find((f) => f.id === id).content, "The user prefers rice.");
  assert.deepEqual(current.history[id].map((v) => v.version), [3, 2, 1]);
  await app.editFact(current.revision, id, "food.favorite", "The user prefers chickpeas.");
  const locked = await add(40, "My favorite food is tamales now.");
  assert.equal(locked.facts.find((f) => f.id === id).content, "The user prefers chickpeas.");
  await app.forget(locked.revision, id);
  const after = await add(50, "My favorite food is rice now.");
  assert.deepEqual(after.facts.map((f) => f.topic), ["food.cooking"]);
  assert.equal(app.candidateTopic("food_favorite", undefined, [{ topic: "favorite_food" }, { topic: "food.favorite" }]), null);
  assert.equal(app.candidateTopic("new_key", "invented_target", after.facts), null);
  app.resetPrivateStorageUser();
});

test("a signed-out attempt cannot leave the processing lane permanently busy", async () => {
  app.resetPrivateStorageUser();
  await assert.rejects(app.processNow(), /Sign in/);
  await seededUser("signed-back-in");
  assert.equal((await app.processNow()).facts.length, 1);
  app.resetPrivateStorageUser();
});

test("archiving does not change device profile eligibility or hide its sources", async () => {
  const { chat } = await seededUser("archived-profile");
  await app.setPrivateChatArchived(chat.id, true);
  assert.equal((await app.processNow()).facts.length, 1);
  assert((await app.deviceContext("base:host:local", "host", "local", false, "food")).includes("corn"));
  assert.equal((await app.getProfile()).settings.enabled, true);
  app.resetPrivateStorageUser();
});

test("corroboration keeps newer sources without another content version", async () => {
  const { chat, message } = await seededUser("confirming");
  const first = await app.processNow();
  const second = app.createPrivateMessage({
    chatId: chat.id,
    parentMessageId: null,
    role: "user",
    contentText: message.active_revision.content_text,
    backendId: "host",
    modelName: "local",
    createdAt: message.created_at + 20
  });
  await app.savePrivateMessage(second);
  const confirmed = await app.processNow();
  assert.equal(confirmed.facts[0].version, first.facts[0].version);
  assert.equal((await app.history(first.facts[0].id)).length, 1);
  await app.savePrivateMessage({ ...message, is_deleted: true });
  assert(
    (await app.deviceContext("base:host:local", "host", "local", false, "food")).includes("corn")
  );
  app.resetPrivateStorageUser();
});

test("legacy device suppression markers upgrade without retaining topic text", async () => {
  user("legacy-markers");
  const profile = await app.getProfile();
  await app.writePrivateProfile(
    {
      ...profile,
      revision: 1,
      suppressed: ["private_topic_with_personal_value"]
    },
    0
  );
  const [first, second] = await Promise.all([app.getProfile(), app.getProfile()]);
  assert.deepEqual(first.suppressed, [app.topicToken("private_topic_with_personal_value")]);
  assert.deepEqual(second.suppressed, first.suppressed);
  assert(!JSON.stringify(first).includes("personal_value"));
  app.resetPrivateStorageUser();
});

test("device edits cannot inherit an earlier revision's allowed model", async () => {
  const { message } = await seededUser("revision-provenance");
  const original = structuredClone(message.active_revision);
  const edit = {
    ...original,
    id: "cloud-edit",
    source: "edit",
    profile_source: {
      backend_id: "host",
      model_name: "renamed",
      persona_id: null,
      persona_version_id: null
    }
  };
  await app.savePrivateMessage({
    ...message,
    active_revision_id: edit.id,
    active_revision: edit,
    revisions: [original, edit]
  });
  assert.equal((await app.processNow()).facts.length, 0);
  // Missing historical provenance also fails closed, rather than using chat defaults.
  const legacy = { ...edit, id: "legacy-edit" };
  delete legacy.profile_source;
  await app.savePrivateMessage({
    ...message,
    active_revision_id: legacy.id,
    active_revision: legacy,
    revisions: [original, legacy]
  });
  assert.equal((await app.processNow()).facts.length, 0);
  await app.savePrivateMessage(message);
  assert.equal((await app.processNow()).facts.length, 1);
  app.resetPrivateStorageUser();
});

test("retry clears an old device error when every source is already processed", async () => {
  await seededUser("empty-retry");
  const profile = await app.processNow();
  await app.writePrivateProfile(
    {
      ...profile,
      revision: profile.revision + 1,
      last_error: "processing_failed"
    },
    profile.revision
  );
  const retried = await app.processNow();
  assert.equal(retried.last_error, null);
  assert.equal(retried.requested, false);
  assert.equal(retried.facts.length, 1);
  app.resetPrivateStorageUser();
});
