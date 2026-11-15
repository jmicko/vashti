import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";
import { fileURLToPath } from "node:url";

Object.assign(globalThis, { indexedDB, IDBKeyRange, window: new EventTarget() });
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
      source = input.sources[0];
    return {
      facts: [
        {
          topic: "food.favorite",
          content: "The user prefers corn.",
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
      'export * from "./personalization/device"; export * from "./personalization/core"; export * from "./privateChatStore";',
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
        b.onResolve({ filter: /^\.\.?\/api$/ }, () => ({ path: "mock-api", namespace: "test" }));
        b.onLoad({ filter: /.*/, namespace: "test" }, () => ({
          contents: "export const requestJson=(...args)=>globalThis.__profileApi(...args);"
        }));
      }
    }
  ]
});
const app = await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
);
const user = (id) =>
  app.setPrivateStorageUser(
    id,
    { user_id: id, key_material: Buffer.alloc(32, id.charCodeAt(0)).toString("base64") },
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
      { ...settings, enabled: false, model_overrides: { [catalog[1].key]: true } },
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
  assert.equal(app.fitProfileContext(context, 64), null);
  const fitted = app.fitProfileContext(context, 600);
  assert(fitted);
  assert(app.contextWeight(fitted) <= 600);
  for (const line of fitted.split("\n").slice(1)) assert.doesNotThrow(() => JSON.parse(line));
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
  assert(forgotten.suppressed.includes("food.favorite"));
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

test("a signed-out attempt cannot leave the processing lane permanently busy", async () => {
  app.resetPrivateStorageUser();
  await assert.rejects(app.processNow(), /Sign in/);
  await seededUser("signed-back-in");
  assert.equal((await app.processNow()).facts.length, 1);
  app.resetPrivateStorageUser();
});
