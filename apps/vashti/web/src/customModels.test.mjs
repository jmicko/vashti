import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { build } from "esbuild";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";

const session = new Map();
Object.assign(globalThis, {
  indexedDB,
  IDBKeyRange,
  window: Object.assign(new EventTarget(), {
    sessionStorage: {
      getItem: key => session.get(key) ?? null,
      setItem: (key, value) => session.set(key, value),
      removeItem: key => session.delete(key)
    }
  })
});
const bundled = await build({
  stdin: {
    contents: `export * from './privateChatStore'; export * from './inferenceSettings'; export * from './customModelDraft';`,
    resolveDir: fileURLToPath(new URL("./", import.meta.url)),
    loader: "ts"
  },
  bundle: true,
  write: false,
  platform: "node",
  format: "esm",
  plugins: [{
    name: "no-network",
    setup(build) {
      build.onResolve({ filter: /^\.\/api$/ }, () => ({ path: "api", namespace: "test" }));
      build.onLoad({ filter: /.*/, namespace: "test" }, () => ({
        contents: `export const requestJson = () => { throw new Error('Unexpected network request'); };`
      }));
    }
  }]
});
const app = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString("base64")}`);

test("inference inheritance keeps zero overrides without erasing unspecified defaults", () => {
  const defaults = { temperature: 1.3, top_p: 0.85, seed: 0 };
  const overrides = { temperature: 0, top_p: undefined, num_predict: 128 };
  const merged = app.mergeInferenceSettings(defaults, overrides);
  assert.equal(merged.temperature, 0);
  assert.equal(merged.top_p, 0.85);
  assert.equal(merged.seed, 0);
  assert.equal(merged.num_predict, 128);
  assert.deepEqual(defaults, { temperature: 1.3, top_p: 0.85, seed: 0 });
});

test("custom model drafts carry explicit inference settings across navigation", () => {
  app.storeCustomModelDraft({
    displayName: "Creative", baseModelValue: "backend:model", systemPrompt: "Be creative.",
    storageMode: "private", modelType: "general", inferenceSettings: { temperature: 1.3, seed: 0 }
  });
  const draft = app.takeCustomModelDraft();
  assert.equal(draft.inferenceSettings.temperature, 1.3);
  assert.equal(draft.inferenceSettings.seed, 0);
  assert.equal(app.takeCustomModelDraft(), null);
});

test("encrypted device custom models version defaults without changing old versions", async () => {
  app.setPrivateStorageUser("custom-model-test", {
    user_id: "custom-model-test", key_material: Buffer.alloc(32, 42).toString("base64")
  }, "isolated-custom-model-test");
  try {
    const params = {
      displayName: "Creative", baseBackendId: "backend", baseBackendName: "Backend",
      baseModelName: "model", systemPrompt: "Be creative."
    };
    const created = await app.createPrivatePersona({ ...params, inferenceSettings: { temperature: 1.3, seed: 0 } });
    assert.equal(created.current_version.version_number, 1);
    const changed = await app.updatePrivatePersona(created.id, { ...params, inferenceSettings: { temperature: 1.6, seed: 0 } });
    assert.equal(changed.current_version.version_number, 2);
    const unchanged = await app.updatePrivatePersona(created.id, { ...params, inferenceSettings: { temperature: 1.6, seed: 0 } });
    assert.equal(unchanged.current_version.id, changed.current_version.id);
    const renamed = await app.updatePrivatePersona(created.id, { ...params, displayName: "Renamed" });
    assert.equal(renamed.current_version.version_number, 3);
    assert.equal(renamed.current_version.inference_settings.temperature, 1.6);
    const recropped = await app.updatePrivatePersona(created.id, { ...params, displayName: "Renamed", avatarCropX: 30 });
    assert.equal(recropped.current_version.id, renamed.current_version.id);
    const cleared = await app.updatePrivatePersona(created.id, { ...params, displayName: "Renamed", inferenceSettings: {} });
    assert.equal(cleared.current_version.version_number, 4);
    assert.deepEqual(app.normalizeInferenceSettings(cleared.current_version.inference_settings), app.normalizeInferenceSettings({}));
    const versions = await app.listPrivatePersonaVersions(created.id);
    assert.equal(versions.length, 4);
    assert.equal(versions.find(version => version.id === created.current_version.id).inference_settings.temperature, 1.3);
    assert.equal((await app.getPrivatePersona(created.id)).current_version.id, cleared.current_version.id);
    const legacy = await app.createPrivatePersona(params);
    assert.equal(legacy.current_version.inference_settings.temperature, undefined);
  } finally {
    app.resetPrivateStorageUser();
  }
});
