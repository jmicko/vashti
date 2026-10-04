import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { build } from "esbuild";
import { indexedDB, IDBKeyRange, IDBObjectStore } from "fake-indexeddb";

Object.assign(globalThis, {
  indexedDB,
  IDBKeyRange,
  window: new EventTarget(),
});
globalThis.createImageBitmap = async () => ({
  width: 64,
  height: 64,
  close() {},
});
globalThis.FileReader = class {
  readAsDataURL(blob) {
    blob
      .arrayBuffer()
      .then((bytes) => {
        this.result = `data:${blob.type};base64,${Buffer.from(bytes).toString("base64")}`;
        this.onload?.();
      })
      .catch((error) => {
        this.error = error;
        this.onerror?.();
      });
  }
};
const png = await readFile(
  new URL("../../../src/images/fixtures/64.png", import.meta.url),
);
const recipe = {
  workflow: "qwen-image-2.1-text-to-image/v1",
  prompt: "",
  negative_prompt: "",
  width: 64,
  height: 64,
  seed: "40",
  steps: 2,
  cfg: 1,
  attention: "standard",
  flow_shift: null,
};
const profile = {
  recipe,
  model_may_change: [],
  backend_id: "backend",
  seed_mode: "increment",
  count: 1,
};
const preferences = {
  generate: profile,
  edit: {
    ...profile,
    recipe: { ...recipe, workflow: "qwen-image-2.1-edit/v1" },
  },
  hidden_controls: [],
};
const remoteJobs = new Map();
let requests = [],
  exports = [],
  pending = false,
  fail = false,
  native = false,
  blobRequests = [],
  holdBlob;
globalThis.__imageTestNative = {
  isNativeRuntime: () => native,
  invokeNative: async (name, args) => {
    exports.push({ name, args });
    return { saved: true };
  },
};
globalThis.__imageTestApi = {
  requestJson: async (path, options = {}) => {
    if (options.signal?.aborted) throw new Error("Aborted");
    if (path === "/api/images/preferences") return structuredClone(preferences);
    if (path === "/api/images/private/flow") {
      const input = JSON.parse(options.body);
      requests.push(input);
      const id = crypto.randomUUID();
      remoteJobs.set(id, input);
      return { id };
    }
    const match = /^\/api\/images\/private\/([^/]+)(\/stop)?$/.exec(path);
    if (!match) throw new Error(`Unexpected network request: ${path}`);
    if (match[2]) {
      remoteJobs.delete(match[1]);
      return {};
    }
    const value = remoteJobs.get(match[1]);
    return {
      status: fail ? "failed" : pending ? "generating" : "complete",
      recipe: {
        ...value.draft.recipe,
        prompt:
          value.draft.auto_improve || value.improve_only
            ? "Improved once"
            : value.draft.recipe.prompt,
      },
    };
  },
  requestBlob: async (path) => {
    blobRequests.push(path);
    if (holdBlob) await holdBlob;
    return new Blob([png], { type: "image/png" });
  },
};
const root = fileURLToPath(new URL("../", import.meta.url));
const bundled = await build({
  stdin: {
    contents: `export * from './images/deviceRepository'; export * from './privateChatStore'; export * from './images/assetStorage'; export * from './saveImage'; export { newDraft } from './images/types'; export { imageTurns, clearPromptAfterSend } from './images/flow';`,
    resolveDir: root,
    loader: "ts",
  },
  bundle: true,
  write: false,
  format: "esm",
  platform: "node",
  plugins: [
    {
      name: "mock-transport",
      setup(build) {
        build.onResolve({ filter: /^\.\.\/api$|^\.\/api$/ }, (args) =>
          args.importer.endsWith("/privateChatStore.ts") ||
          (args.importer.includes("/images/") && args.path === "../api")
            ? { path: "mock-api", namespace: "test" }
            : null,
        );
        build.onResolve({ filter: /^\.\/runtime$/ }, () => ({
          path: "mock-native",
          namespace: "test",
        }));
        build.onLoad({ filter: /.*/, namespace: "test" }, (args) => ({
          contents:
            args.path === "mock-native"
              ? `export const isNativeRuntime = () => globalThis.__imageTestNative.isNativeRuntime(); export const invokeNative = (...args) => globalThis.__imageTestNative.invokeNative(...args);`
              : `export const requestJson = (...args) => globalThis.__imageTestApi.requestJson(...args); export const requestBlob = (...args) => globalThis.__imageTestApi.requestBlob(...args); export const requestMultipartJson = () => { throw new Error('Unexpected server upload'); };`,
        }));
      },
    },
  ],
});
const app = await import(
  `data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString("base64")}`
);
const local = (path, value, method = "POST") =>
  app.deviceImageRequest(
    path,
    value === undefined ? {} : { method, body: JSON.stringify(value) },
  );
function user(id) {
  app.setPrivateStorageUser(
    id,
    {
      user_id: id,
      key_material: Buffer.alloc(32, id.charCodeAt(0)).toString("base64"),
    },
    "test-server",
  );
}
async function completed(id) {
  for (let i = 0; i < 300; i++) {
    const run = await local(`/runs/${id}`);
    if (["complete", "failed", "stopped"].includes(run.status)) return run;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("Test generation did not finish");
}
const runRequest = (draft) => ({
  client_request_id: crypto.randomUUID(),
  draft,
  improve_only: false,
  retry_result_id: null,
  fixed_endless_confirmed: false,
});

test("private flows persist encrypted images, branch versions and gallery only on device", async () => {
  user("alice");
  requests = [];
  const draft = {
    ...app.newDraft(profile),
    recipe: { ...recipe, prompt: "A teapot" },
    count: 3,
    auto_improve: true,
    prompt_help: {
      backend_id: "backend",
      model_name: "helper",
      guidance: "Add detail",
    },
  };
  const session = await local("/sessions", { draft, mode: "generate" });
  const request = runRequest(draft);
  const run = await local(`/sessions/${session.id}/runs`, request);
  assert.equal((await completed(run.id)).status, "complete");
  assert.deepEqual(
    requests.map((value) => value.draft.recipe.prompt),
    ["A teapot", "A teapot", "A teapot"],
  );
  assert.deepEqual(
    requests.map((value) => value.draft.recipe.seed),
    ["40", "41", "42"],
  );
  const results = await local(`/sessions/${session.id}/results`);
  assert.equal(results.length, 3);
  assert.equal(
    (await app.deviceImageBlob(results[0].asset_id)).size,
    png.length,
  );
  assert.equal(
    (await local(`/sessions/${session.id}/runs`, request)).id,
    run.id,
  );
  const retry = await local(`/sessions/${session.id}/runs`, {
    ...runRequest(draft),
    retry_result_id: results[0].id,
  });
  assert.equal((await completed(retry.id)).status, "complete");
  const versions = await local(`/sessions/${session.id}/results`);
  assert.equal(versions[0].group_id, results[0].group_id);
  assert.equal(versions[0].recipe.seed, "43");
  const olderRetry = await local(`/sessions/${session.id}/runs`, {
    ...runRequest(draft),
    retry_result_id: results[0].id,
  });
  assert.equal((await completed(olderRetry.id)).status, "complete");
  assert.equal(
    (await local(`/sessions/${session.id}/results`))[0].recipe.seed,
    "44",
  );
  const randomRetry = await local(`/sessions/${session.id}/runs`, {
    ...runRequest({ ...draft, seed_mode: "random" }),
    retry_result_id: results[0].id,
  });
  assert.equal((await completed(randomRetry.id)).status, "complete");
  const randomVersions = await local(`/sessions/${session.id}/results`);
  assert.equal(randomVersions[0].group_id, results[0].group_id);
  assert.equal(randomVersions[0].origin_run_id, run.id);
  const turns = app.imageTurns(
    await local(`/sessions/${session.id}/runs`),
    randomVersions,
    randomVersions[0].id,
  );
  assert.ok(
    turns.some((turn) =>
      turn.responses.some((value) => value.id === randomVersions[0].id),
    ),
  );
  await local(`/results/${results[0].id}/select`, {});
  assert.equal(
    (await local(`/sessions/${session.id}`)).active_result_id,
    results[0].id,
  );
  const gallery = await local("/gallery");
  assert.equal(gallery.length, 3);
  assert.equal(
    gallery.find((value) => value.group_id === results[0].group_id).asset_id,
    results[0].asset_id,
  );
  app.resetPrivateStorageUser();
  user("bob");
  assert.deepEqual(await local("/sessions"), []);
  await assert.rejects(app.deviceImageBlob(results[0].asset_id), /not found/);
  user("alice");
  assert.equal(
    (await local(`/sessions/${session.id}`)).draft.recipe.prompt,
    "A teapot",
  );
  const guard = app.privateImageGuard();
  user("bob");
  assert.throws(guard, /user changed/);
  user("alice");
  // Check the actual vault database, not just the repository's decoded view.
  const databases = await indexedDB.databases();
  const name = databases.find(
    (value) => value.name.includes("alice") && value.version === 10,
  )?.name;
  assert.ok(name);
  const vault = await new Promise((resolve) => {
    const request = indexedDB.open(name);
    request.onsuccess = () => resolve(request.result);
  });
  const raw = await new Promise((resolve) => {
    const tx = vault.transaction("private_images");
    const request = tx.objectStore("private_images").getAll();
    request.onsuccess = () => resolve(request.result);
  });
  assert.ok(raw.every((value) => value.encrypted_payload));
  assert.ok(!JSON.stringify(raw).includes("A teapot"));
  assert.ok(!JSON.stringify(raw).includes("data:image"));
  vault.close();
});

test("image-only edits, candidate batches, Stop, failures and defaults remain usable", async () => {
  user("carol");
  requests = [];
  const source = await app.uploadDeviceImage(
    new Blob([png], { type: "image/png" }),
  );
  const draft = {
    ...app.newDraft(preferences.edit),
    input_ids: [source.id],
    original_input_ids: [source.id],
    count: 2,
  };
  const session = await local("/sessions", { draft, mode: "iterative" });
  const run = await local(`/sessions/${session.id}/runs`, runRequest(draft));
  assert.equal((await completed(run.id)).status, "complete");
  assert.equal(requests[0].draft.recipe.prompt, "");
  assert.equal(requests[1].draft.recipe.prompt, "");
  assert.equal(requests[0].draft.input_ids[0], requests[1].draft.input_ids[0]);
  const results = await local(`/sessions/${session.id}/results`);
  assert.equal(results[0].parent_result_id, null);
  const continuation = {
    ...draft,
    count: 1,
    input_ids: [results[0].asset_id],
    parent_result_id: results[0].id,
  };
  const next = await local(
    `/sessions/${session.id}/runs`,
    runRequest(continuation),
  );
  assert.equal((await completed(next.id)).status, "complete");
  assert.equal(
    (await local(`/sessions/${session.id}/results`))[0].parent_result_id,
    results[0].id,
  );
  await assert.rejects(
    local(
      `/sessions/${session.id}/runs`,
      runRequest({ ...continuation, input_ids: [source.id] }),
    ),
    /selected result/,
  );
  await assert.rejects(
    local(`/sessions/${session.id}/runs`, {
      ...runRequest(draft),
      improve_only: true,
    }),
    /Enter a prompt/,
  );
  pending = true;
  const waiting = await local(
    `/sessions/${session.id}/runs`,
    runRequest(draft),
  );
  await local(`/runs/${waiting.id}/stop`, {});
  assert.equal((await completed(waiting.id)).status, "stopped");
  pending = false;
  fail = true;
  const failed = await local(`/sessions/${session.id}/runs`, runRequest(draft));
  assert.equal((await completed(failed.id)).status, "failed");
  fail = false;
  await local("/preferences", preferences, "PUT");
  assert.deepEqual(await local("/preferences"), preferences);
  await local(`/gallery/${results[0].group_id}`, {}, "DELETE");
  assert.equal((await local(`/sessions/${session.id}/results`)).length, 3);
  const copy = await local(`/gallery/copy/${results[0].asset_id}`, {});
  await local(`/sessions/${session.id}`, {}, "DELETE");
  assert.deepEqual(await local("/sessions"), []);
  assert.equal(
    (await app.deviceImageBlob(results[0].asset_id)).size,
    png.length,
  );
  await assert.rejects(app.deviceImageBlob(source.id), /not found/);
  await local(`/gallery/${copy.id}`, {}, "DELETE");
  await assert.rejects(app.deviceImageBlob(results[0].asset_id), /not found/);
});

test("native download exports authenticated bytes instead of a browser blob link", async () => {
  native = true;
  exports = [];
  await app.saveImage(new Blob([png], { type: "image/png" }), "fixture");
  assert.equal(exports[0].name, "native_save_image");
  assert.equal(exports[0].args.filename, "fixture.png");
  assert.deepEqual(Buffer.from(exports[0].args.data, "base64"), png);
  native = false;
});

test("private seed exhaustion retains completed images without wrapping", async () => {
  user("seeds");
  requests = [];
  const draft = {
    ...app.newDraft(profile),
    recipe: { ...recipe, prompt: "Seed boundary", seed: "9223372036854775807" },
    count: 2,
  };
  const session = await local("/sessions", { draft, mode: "generate" });
  const run = await local(`/sessions/${session.id}/runs`, runRequest(draft));
  const finished = await completed(run.id);
  assert.equal(finished.status, "failed");
  assert.match(finished.error, /Seed range exhausted/);
  const results = await local(`/sessions/${session.id}/results`);
  assert.equal(results.length, 1);
  assert.deepEqual(
    requests.map((value) => value.draft.recipe.seed),
    ["9223372036854775807"],
  );
  await assert.rejects(
    local(`/sessions/${session.id}/runs`, {
      ...runRequest(draft),
      retry_result_id: results[0].id,
    }),
    /Seed range exhausted/,
  );
});

test("private completion does not steal a selected older branch", async () => {
  user("branches");
  const draft = {
    ...app.newDraft(profile),
    recipe: { ...recipe, prompt: "Branch fixture" },
    count: 2,
  };
  const session = await local("/sessions", { draft, mode: "generate" });
  const first = await local(`/sessions/${session.id}/runs`, runRequest(draft));
  await completed(first.id);
  const results = await local(`/sessions/${session.id}/results`);
  let release;
  holdBlob = new Promise((resolve) => {
    release = resolve;
  });
  const count = blobRequests.length;
  const next = await local(
    `/sessions/${session.id}/runs`,
    runRequest({ ...draft, count: 1, parent_result_id: results[0].id }),
  );
  for (let i = 0; i < 300 && blobRequests.length === count; i++)
    await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(blobRequests.length, count + 1);
  await local(`/results/${results[1].id}/select`, {});
  release();
  holdBlob = undefined;
  assert.equal((await completed(next.id)).status, "complete");
  assert.equal(
    (await local(`/sessions/${session.id}`)).active_result_id,
    results[1].id,
  );
});

async function cacheSettled(id) {
  for (let i = 0; i < 300; i++) {
    const index = await app.readPrivateImage("cache-index");
    if (index?.[0].id === id) return index;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("Image cache did not settle");
}
test("original image cache is bounded, least-recently-used, and survives reopen", async () => {
  user("cache");
  blobRequests = [];
  for (let i = 0; i < 10; i++) {
    await app.loadImageBlob("/api/images", `image-${i}`);
    await cacheSettled(`image-${i}`);
  }
  assert.equal(blobRequests.length, 10);
  await app.loadImageBlob("/api/images", "image-0");
  await cacheSettled("image-0");
  await app.loadImageBlob("/api/images", "image-10");
  const index = await cacheSettled("image-10");
  assert.equal(index.length, 10);
  assert.ok(index.some((value) => value.id === "image-0"));
  assert.ok(!index.some((value) => value.id === "image-1"));
  assert.equal(await app.readPrivateImage("cache:image-1"), null);
  app.resetPrivateStorageUser();
  user("cache");
  assert.deepEqual(
    Buffer.from(
      await (await app.loadImageBlob("/api/images", "image-0")).arrayBuffer(),
    ),
    png,
  );
  assert.equal(blobRequests.length, 11);
  await cacheSettled("image-0");
});

test("an account switch during a private result cannot save it under the next user", async () => {
  user("before");
  let release;
  holdBlob = new Promise((resolve) => {
    release = resolve;
  });
  const draft = {
    ...app.newDraft(profile),
    recipe: { ...recipe, prompt: "Private fixture" },
  };
  const session = await local("/sessions", { draft, mode: "generate" });
  const count = blobRequests.length;
  await local(`/sessions/${session.id}/runs`, runRequest(draft));
  for (let i = 0; i < 300 && blobRequests.length === count; i++)
    await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(blobRequests.length, count + 1);
  user("after");
  release();
  holdBlob = undefined;
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.deepEqual(await local("/sessions"), []);
  assert.deepEqual(await app.privateImageIds("asset:"), []);
  user("before");
  assert.equal(
    (await local(`/sessions/${session.id}/runs`))[0].status,
    "stopped",
  );
  assert.deepEqual(await local(`/sessions/${session.id}/results`), []);
});

test("prompt retention only clears accepted iterative edits, not newly typed text", () => {
  const generate = runRequest({
    ...app.newDraft(profile),
    recipe: { ...recipe, prompt: "Keep this" },
  });
  const edit = {
    ...generate,
    draft: {
      ...generate.draft,
      recipe: { ...generate.draft.recipe, workflow: "qwen-image-2.1-edit/v1" },
    },
  };
  assert.equal(
    app.clearPromptAfterSend(generate, "generate", "Keep this"),
    false,
  );
  assert.equal(app.clearPromptAfterSend(edit, "original", "Keep this"), false);
  assert.equal(app.clearPromptAfterSend(edit, "iterative", "Keep this"), true);
  assert.equal(
    app.clearPromptAfterSend(edit, "iterative", "New instructions"),
    false,
  );
  assert.equal(
    app.clearPromptAfterSend(
      { ...edit, retry_result_id: "retry" },
      "iterative",
      "Keep this",
    ),
    false,
  );
  assert.equal(
    app.clearPromptAfterSend(
      { ...edit, improve_only: true },
      "iterative",
      "Keep this",
    ),
    false,
  );
});

test("cache write failures do not prevent viewing or leave unindexed image bytes", async () => {
  user("quota");
  const put = IDBObjectStore.prototype.put;
  let failed = false;
  IDBObjectStore.prototype.put = function (value, ...args) {
    if (value.id === "cache-index") {
      failed = true;
      throw new DOMException("Test quota", "QuotaExceededError");
    }
    return put.call(this, value, ...args);
  };
  try {
    const result = await app.loadImageBlob("/api/images", "quota-image");
    assert.equal(result.size, png.length);
    for (let i = 0; i < 300 && !failed; i++)
      await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(failed, true);
    for (
      let i = 0;
      i < 300 && (await app.readPrivateImage("cache:quota-image"));
      i++
    )
      await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(await app.readPrivateImage("cache:quota-image"), null);
  } finally {
    IDBObjectStore.prototype.put = put;
  }
});

test("browser downloads retain the normal authenticated blob download", async () => {
  const documentBefore = globalThis.document,
    timeoutBefore = globalThis.setTimeout;
  const anchor = {
    click() {
      this.clicked = true;
    },
    remove() {
      this.removed = true;
    },
  };
  let appended = false;
  globalThis.document = {
    createElement: (tag) => {
      assert.equal(tag, "a");
      return anchor;
    },
    body: {
      append(value) {
        assert.equal(value, anchor);
        appended = true;
      },
    },
  };
  globalThis.setTimeout = (callback, delay) => {
    assert.equal(delay, 60000);
    callback();
  };
  try {
    await app.saveImage(
      new Blob([png], { type: "image/png" }),
      "browser-fixture",
    );
    assert.equal(anchor.download, "browser-fixture.png");
    assert.ok(anchor.href.startsWith("blob:"));
    assert.ok(appended && anchor.clicked && anchor.removed);
  } finally {
    globalThis.document = documentBefore;
    globalThis.setTimeout = timeoutBefore;
  }
});
