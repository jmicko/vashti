import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createContext, runInContext } from "node:vm";
import { syntheticExport } from "../web/src/imports/fixtures.mjs";

const target = process.argv[2] ?? "dist";
assert(["dist", "dist-native"].includes(target));
const directory = new URL(`../web/${target}/assets/`, import.meta.url);
const filename = (await readdir(directory)).find(name => /^worker-.*\.js$/.test(name));
assert(filename, "Build the frontend before checking its import worker");
const calls = new Map(); let sequence = 0;
const context = createContext({
  Blob, TextEncoder, TextDecoder, TransformStream, ReadableStream, WritableStream,
  URL, atob, btoa, setTimeout, clearTimeout,
  crypto: globalThis.crypto,
  location: { href: "https://synthetic.example/assets/worker.js" },
  fetch: async () => { throw new Error("An import worker must not fetch codecs or export data"); },
  // Exercise the JavaScript fallback and strict native CSP, without a DOM.
  CompressionStream: undefined, DecompressionStream: undefined, WebAssembly: undefined,
  postMessage: data => { if (data.id !== undefined) { const call = calls.get(data.id); calls.delete(data.id); if (data.error) call?.reject(new Error(data.error)); else call?.resolve(data.value); } }
});
context.self = context;
runInContext(await readFile(new URL(filename, directory), "utf8"), context, { filename: fileURLToPath(new URL(filename, directory)) });
async function request(action, extra = {}) {
  const id = ++sequence;
  const response = new Promise((resolve, reject) => { calls.set(id, { resolve, reject }); });
  context.onmessage({ data: { id, action, ...extra } });
  const timeout = setTimeout(() => { calls.get(id)?.reject(new Error("Worker response timed out")); calls.delete(id); }, 30_000);
  try { return await response; } finally { clearTimeout(timeout); }
}
const image = await readFile(new URL("../src/images/fixtures/64.png", import.meta.url));
const summary = await request("open", { file: await syntheticExport(image) });
assert.equal(summary.conversations, 2); assert.equal(summary.messages, 6); assert.equal(summary.workspace_pages,1);
const first = await request("next");
assert.match(first.messages.find(message => message.source_id === "first").content, /Sources/);
assert.match(first.messages.find(message => message.source_id === "first").content,/\/\.vashti\/imported-file\/asset-/);
const asset = await request("asset", { path: first.assets.find(asset => asset.entry_path === "file-fixture.dat").entry_path });
assert.equal(asset.size, image.length);
await request("next");
const page = await request("next");
assert.equal(page.workspace.title,"Synthetic workspace Page");
assert.equal(page.workspace.markdown,"## First section\n\nA **formatted** paragraph.\n\n- A list item");
assert.equal(JSON.parse(page.original_json).checkpoint.checkpoint.checkpoint_id,"fixture-checkpoint");
assert.equal(await request("next"), null); await request("close");
console.log(`Verified ${target} import worker: ZIP preview, Markdown, branches, Page and file links without DOM, network, native codecs, or WebAssembly.`);
