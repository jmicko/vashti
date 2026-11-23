import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";

const require = createRequire(import.meta.url);
const bundle = await build({
  entryPoints: [fileURLToPath(new URL("RebuildProfileButton.tsx", import.meta.url))],
  bundle: true, format: "esm", platform: "node", mainFields: ["module", "main"], jsx: "automatic", write: false,
  plugins: [{ name: "fixture-apis", setup(b) {
    b.onResolve({ filter: /^react(\/.*)?$/ }, ({ path }) => ({ path: pathToFileURL(require.resolve(path)).href, external: true }));
    b.onResolve({ filter: /^\.\/api$/ }, () => ({ path: "mock", namespace: "test" }));
    b.onLoad({ filter: /.*/, namespace: "test" }, () => ({ contents: `
      export const getRebuildProgress = (...args) => globalThis.rebuildUiApi.progress(...args);
      export const cancelRebuild = (...args) => globalThis.rebuildUiApi.cancel(...args);
      export const getProfile = () => {};
      export const rebuildProfile = () => {};
    ` }));
  } }]
});
const { RebuildProfileButton } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`);

for (const mode of ["server", "device"]) {
  test(`${mode} rebuild progress and cancellation remain usable with other settings disabled`, async () => {
    const dom = new JSDOM("<main></main>", { url: "http://localhost", pretendToBeVisual: true });
    const { window } = dom;
    Object.assign(globalThis, { window, document: window.document, HTMLElement: window.HTMLElement, IS_REACT_ACT_ENVIRONMENT: true });
    window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
    window.HTMLDialogElement.prototype.close = function () { this.open = false; };
    const calls = [];
    let progress = { id: `${mode}-rebuild`, processed: 42, phase: "waiting_for_generation" };
    globalThis.rebuildUiApi = {
      async progress(requestedMode) { assert.equal(requestedMode, mode); return progress; },
      async cancel(...args) { calls.push(args); progress = null; return { rebuild_id: null }; }
    };
    const root = createRoot(document.querySelector("main"));
    const click = async (label) => {
      const button = [...document.querySelectorAll("button")].find((el) => el.textContent.trim() === label);
      assert(button && !button.disabled, `${label} is available`);
      await act(async () => button.click());
    };
    try {
      await act(async () => root.render(createElement(RebuildProfileButton, { mode, disabled: true })));
      assert.match(document.querySelector('[role="status"]').textContent, /Waiting for generation to finish.*42 messages processed/);
      await click("Cancel rebuild");
      assert.match(document.querySelector("dialog").textContent, /current profile and its history won.t change/);
      await click("Cancel");
      assert.deepEqual(calls, []);
      await click("Cancel rebuild");
      await click("Discard rebuild");
      assert.deepEqual(calls, [[mode, `${mode}-rebuild`]]);
      assert.match(document.querySelector('[role="status"]').textContent, /Current profile kept/);
      assert.equal(document.querySelector("dialog"), null);
    } finally {
      await act(async () => root.unmount());
      window.close();
      delete globalThis.rebuildUiApi;
    }
  });
}
