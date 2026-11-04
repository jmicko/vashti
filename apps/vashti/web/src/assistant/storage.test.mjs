import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";

Object.assign(globalThis, { indexedDB, IDBKeyRange, window: new EventTarget() });
let native = false, cancelled = false, saved = [];
globalThis.__assistantNative = {
  isNativeRuntime: () => native,
  invokeNative: async (name, args) => { saved.push({ name, args }); return { saved: !cancelled }; }
};
globalThis.FileReader = class {
  readAsDataURL(blob) { void blob.arrayBuffer().then(bytes => { this.result = `data:${blob.type};base64,${Buffer.from(bytes).toString("base64")}`; this.onload?.(); }); }
};
const bundled = await build({ stdin: { contents: "export * from './privateChatStore'; export * from './saveFile';", resolveDir: fileURLToPath(new URL("../", import.meta.url)), loader: "ts" }, bundle: true, write: false, format: "esm", platform: "node", plugins: [{ name: "no-network", setup(build) {
  build.onResolve({ filter: /^\.\/runtime$/ }, () => ({ path: "native", namespace: "fixture" }));
  build.onResolve({ filter: /^\.\/api$/ }, () => ({ path: "api", namespace: "fixture" }));
  build.onLoad({ filter: /.*/, namespace: "fixture" }, args => ({ contents: args.path === "native" ? "export const isNativeRuntime = () => globalThis.__assistantNative.isNativeRuntime(); export const invokeNative = (...args) => globalThis.__assistantNative.invokeNative(...args);" : "export const requestJson = () => { throw new Error('No network expected'); };" }));
} }] });
const app = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString("base64")}`);
const user = (id, server = "fixture-server") => app.setPrivateStorageUser(id, { user_id: id, key_material: Buffer.alloc(32, id.charCodeAt(0)).toString("base64") }, server);

test("Assistant drafts are encrypted and isolated by account and server", async () => {
  user("alice");
  await app.writeAssistantDraft("reply:task", "private synthetic draft");
  assert.equal(await app.readAssistantDraft("reply:task"), "private synthetic draft");
  const databases = await indexedDB.databases();
  const names = databases.map(db => db.name).filter(name => name?.includes("alice"));
  assert.equal(names.length, 1);
  const raw = await new Promise((resolve, reject) => {
    const open = indexedDB.open(names[0]);
    open.onsuccess = () => { const db = open.result; const tx = db.transaction("assistant_drafts"); const request = tx.objectStore("assistant_drafts").get("reply:task"); request.onsuccess = () => { resolve(request.result); db.close(); }; request.onerror = () => reject(request.error); };
  });
  assert.ok(raw.encrypted_payload);
  assert.ok(!JSON.stringify(raw).includes("private synthetic draft"));
  user("bob"); assert.equal(await app.readAssistantDraft("reply:task"), null);
  user("alice", "other-server"); assert.equal(await app.readAssistantDraft("reply:task"), null);
  user("alice"); assert.equal(await app.readAssistantDraft("reply:task"), "private synthetic draft");
  await app.writeAssistantDraft("reply:task", null);
  assert.equal(await app.readAssistantDraft("reply:task"), null);
});
test("a late draft write cannot cross an account change", async () => {
  user("alice");
  const pending = app.writeAssistantDraft("reply:late", "old account data");
  user("bob");
  await assert.rejects(pending, /changed|ready/);
  assert.equal(await app.readAssistantDraft("reply:late"), null);
});
test("generic native exports preserve bytes, safe names, and cancellation", async () => {
  native = true;
  const file = new Blob(["value\n396\n"], { type: "text/csv" });
  assert.equal(await app.saveFile(file, "../../answer.csv"), "saved");
  assert.equal(saved[0].name, "native_save_file");
  assert.ok(!saved[0].args.filename.includes("/"));
  assert.equal(Buffer.from(saved[0].args.data, "base64").toString(), "value\n396\n");
  assert.equal(saved[0].args.mimeType, "text/csv");
  cancelled = true;
  assert.equal(await app.saveFile(file, "answer.csv"), "cancelled");
  await assert.rejects(app.saveFile(new Blob([new Uint8Array(20 * 1024 * 1024 + 1)]), "large.bin"), /20 MiB/);
});
