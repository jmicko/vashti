import assert from "node:assert/strict";
import test, { after } from "node:test";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";
import { fileURLToPath } from "node:url";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

const dom = new JSDOM(
  "<!doctype html><html><body><main></main></body></html>",
  { url: "http://localhost", pretendToBeVisual: true },
);
const browser = dom.window;
for (const key of [
  "window",
  "Window",
  "document",
  "HTMLElement",
  "Element",
  "Node",
  "MutationObserver",
  "DOMRect",
  "Range",
  "Event",
  "CustomEvent",
  "KeyboardEvent",
  "MouseEvent",
  "getComputedStyle",
]) {
  Object.defineProperty(globalThis, key, {
    value: key === "window" ? browser : browser[key],
    configurable: true,
    writable: true,
  });
}
Object.assign(globalThis, {
  indexedDB,
  IDBKeyRange,
  requestAnimationFrame: browser.requestAnimationFrame.bind(browser),
  cancelAnimationFrame: browser.cancelAnimationFrame.bind(browser),
  IS_REACT_ACT_ENVIRONMENT: true,
});
browser.Range.prototype.getClientRects = () => [];
browser.Range.prototype.getBoundingClientRect = () => new browser.DOMRect();
browser.HTMLElement.prototype.hidePopover = function () {};
browser.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
browser.HTMLDialogElement.prototype.close = function () {
  this.open = false;
};
const temporary = await mkdtemp(join(tmpdir(), "vashti-canvas-editor-")),
  roots = new Set();
const require = createRequire(import.meta.url);
await build({
  stdin: {
    contents:
      'export {act,createElement,createRef} from "react"; export {createRoot} from "react-dom/client"; export {EditorView} from "@codemirror/view"; export {CanvasEditor} from "./Editor"; export {CanvasWorkspacePanel} from "./Workspace"; export * from "../privateChatStore"; export * from "./device";',
    resolveDir: fileURLToPath(new URL(".", import.meta.url)),
  },
  bundle: true,
  format: "esm",
  platform: "node",
  outfile: join(temporary, "fixture.mjs"),
  banner: {
    js: 'import {createRequire} from "node:module"; const require = createRequire(import.meta.url);',
  },
  loader: { ".css": "empty" },
  plugins: [
    {
      name: "native-react",
      setup(build) {
        build.onResolve({ filter: /^(react|react-dom)(\/.*)?$/ }, (args) => ({
          path: require.resolve(args.path),
          external: true,
        }));
      },
    },
  ],
});
const app = await import(join(temporary, "fixture.mjs"));
after(async () => {
  for (const root of roots) await app.act(async () => root.unmount());
  dom.window.close();
  await rm(temporary, { recursive: true, force: true });
});
const wait = () => new Promise((resolve) => setTimeout(resolve, 25));
const settle = () => app.act(wait);
async function until(check) {
  for (let i = 0; i < 80; i++) {
    await settle();
    if (check()) return;
  }
  assert.fail("Timed out waiting for the editor");
}
const click = async (element) => {
  assert(element, "Expected a control");
  await app.act(async () => element.click());
  await settle();
};
const button = (label) =>
  [...document.querySelectorAll("button")].find(
    (b) =>
      b.getAttribute("aria-label") === label || b.textContent.trim() === label,
  );

test("CodeMirror editing, selection, formatting, undo, readonly and external updates", async () => {
  const container = document.querySelector("main"),
    root = app.createRoot(container),
    ref = app.createRef();
  roots.add(root);
  let value = "Hello world",
    selected = "";
  const render = (readOnly) =>
    app.act(async () =>
      root.render(
        app.createElement(app.CanvasEditor, {
          ref,
          value,
          language: "markdown",
          readOnly,
          onChange: (next) => {
            value = next;
          },
          onSelection: (text) => {
            selected = text;
          },
        }),
      ),
    );
  await render(false);
  await settle();
  const view = app.EditorView.findFromDOM(
    container.querySelector(".cm-editor"),
  );
  await app.act(async () =>
    view.dispatch({ selection: { anchor: 0, head: 5 } }),
  );
  assert.equal(selected, "Hello");
  await app.act(async () => ref.current.wrap("**", "**"));
  assert.equal(value, "**Hello** world");
  await app.act(async () => ref.current.undo());
  assert.equal(value, "Hello world");
  await app.act(async () => ref.current.redo());
  assert.equal(value, "**Hello** world");
  value = "External revision";
  await render(true);
  assert.equal(view.state.doc.toString(), value);
  await app.act(async () => {
    ref.current.wrap("*", "*");
    ref.current.undo();
  });
  assert.equal(view.state.doc.toString(), "External revision");
  assert.equal(view.contentDOM.getAttribute("contenteditable"), "false");
  await app.act(async () => root.unmount());
  roots.delete(root);
});

test("workspace saves edits, reviews and restores history, comments, and recovers an unsent draft", async () => {
  app.setPrivateStorageUser(
    "canvas-editor-fixture",
    {
      user_id: "canvas-editor-fixture",
      key_material: Buffer.alloc(32, 4).toString("base64"),
    },
    "editor-fixture",
  );
  const chat = await app.createPrivateChat({
      title: "Editor fixture",
      backendId: "local",
      backendName: "Local",
      modelName: "synthetic-model",
    }),
    scope = { message_id: null, revision_id: null };
  const initial = await app.mutateDeviceCanvas(chat.id, scope, {
    operation: "create",
    title: "Editor document",
    language: "markdown",
    content: "Original synthetic draft",
  });
  const container = document.querySelector("main"),
    root = app.createRoot(container);
  roots.add(root);
  let closed = false,
    asked = "";
  const props = {
    mode: "device",
    chatId: chat.id,
    scope,
    request: null,
    onConsumed: () => {},
    onClose: () => {
      closed = true;
    },
    onAsk: async (prompt) => {
      asked = prompt;
      return true;
    },
    isGenerating: false,
  };
  await app.act(async () =>
    root.render(app.createElement(app.CanvasWorkspacePanel, props)),
  );
  await settle();
  await settle();
  assert.equal(container.querySelectorAll(".cm-editor").length, 1);
  const edit = app.EditorView.findFromDOM(
    container.querySelector(".cm-editor"),
  );
  await app.act(async () =>
    edit.dispatch({
      changes: {
        from: 0,
        to: edit.state.doc.length,
        insert: "Changed synthetic draft",
      },
    }),
  );
  await app.act(async () =>
    container.querySelector(".canvas-document").dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "s",
        ctrlKey: true,
        bubbles: true,
      }),
    ),
  );
  await settle();
  let current = await app.deviceCanvasWorkspace(chat.id, scope);
  assert.equal(current.documents[0].version.content, "Changed synthetic draft");
  await until(
    () => container.querySelector('[role="status"]').textContent === "Saved",
  );
  await click(button("History"));
  assert.match(container.querySelector(".canvas-history").textContent, /v2/);
  await click(
    [...container.querySelectorAll(".canvas-history-entry")].find((b) =>
      b.textContent.startsWith("v1"),
    ),
  );
  assert.match(
    container.querySelector(".canvas-version-view").textContent,
    /Original synthetic draft/,
  );
  await click(button("Restore"));
  assert(document.querySelector("dialog[open]"));
  await click(
    [...document.querySelectorAll("dialog button")].find(
      (b) => b.textContent === "Restore",
    ),
  );
  current = await app.deviceCanvasWorkspace(chat.id, scope);
  assert.equal(
    current.documents[0].version.content,
    initial.documents[0].version.content,
  );
  assert.equal(current.documents[0].version.version_number, 3);
  await click(button("Comments"));
  const text = container.querySelector('[aria-label="New canvas comment"]');
  await app.act(async () => {
    Object.getOwnPropertyDescriptor(
      browser.HTMLTextAreaElement.prototype,
      "value",
    ).set.call(text, "A test comment");
    text.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await click(button("Add comment"));
  assert.equal(
    (await app.deviceCanvasWorkspace(chat.id, scope)).comments[0].content,
    "A test comment",
  );
  await click(button("Back to editor"));
  const live = app.EditorView.findFromDOM(
    container.querySelector(".canvas-editor-container .cm-editor"),
  );
  await app.act(async () =>
    live.dispatch({
      changes: {
        from: 0,
        to: live.state.doc.length,
        insert: "Recover this local draft",
      },
    }),
  );
  await app.act(async () => root.unmount());
  roots.delete(root);
  await wait();
  const reopened = app.createRoot(container);
  roots.add(reopened);
  await app.act(async () =>
    reopened.render(app.createElement(app.CanvasWorkspacePanel, props)),
  );
  await settle();
  await settle();
  assert.match(
    container.querySelector(".canvas-editor-container").textContent,
    /Recover this local draft/,
  );
  assert.match(
    container.querySelector('[role="status"]').textContent,
    /Recovered draft/,
  );
  await click(button("Back to chat"));
  assert(closed);
  assert.equal(asked, "");
  await app.act(async () => reopened.unmount());
  roots.delete(reopened);
  await wait();
  app.resetPrivateStorageUser();
});

test("unmounting after an account switch cannot save the previous user's pending draft", async () => {
  const login = (user) =>
    app.setPrivateStorageUser(
      user,
      { user_id: user, key_material: Buffer.alloc(32, 6).toString("base64") },
      "editor-switch-fixture",
    );
  login("editor-old-user");
  const chat = await app.createPrivateChat({
      title: "Switch fixture",
      backendId: "local",
      backendName: "Local",
      modelName: "synthetic-model",
    }),
    scope = { message_id: null, revision_id: null };
  await app.mutateDeviceCanvas(chat.id, scope, {
    operation: "create",
    title: "Previous owner",
    language: "text",
    content: "Original text",
  });
  const container = document.querySelector("main"),
    root = app.createRoot(container);
  roots.add(root);
  await app.act(async () =>
    root.render(
      app.createElement(app.CanvasWorkspacePanel, {
        mode: "device",
        chatId: chat.id,
        scope,
        request: null,
        onConsumed: () => {},
        onClose: () => {},
        onAsk: async () => true,
        isGenerating: false,
      }),
    ),
  );
  await settle();
  await settle();
  const editor = app.EditorView.findFromDOM(
    container.querySelector(".cm-editor"),
  );
  await app.act(async () =>
    editor.dispatch({
      changes: {
        from: 0,
        to: editor.state.doc.length,
        insert: "Previous owner's pending content",
      },
    }),
  );
  login("editor-new-user");
  await app.act(async () => root.unmount());
  roots.delete(root);
  await wait();
  assert.deepEqual(await app.listPrivateCanvasRecords(chat.id), []);
  app.resetPrivateStorageUser();
});

test("typing during a pending server save stays in the editor and the next saved revision", async () => {
  app.setPrivateStorageUser(
    "server-editor-fixture",
    {
      user_id: "server-editor-fixture",
      key_material: Buffer.alloc(32, 5).toString("base64"),
    },
    "server-editor-fixture",
  );
  const now = Math.floor(Date.now() / 1000),
    version = {
      id: "v1",
      document_id: "doc",
      parent_id: null,
      version_number: 1,
      title: "Concurrent editor",
      language: "markdown",
      content: "Initial server content",
      actor: "human",
      model_name: null,
      edit_session_id: null,
      created_at: now,
      updated_at: now,
    };
  let workspace = {
      scope_id: "root:server-chat",
      documents: [{ id: "doc", trashed: false, version }],
      comments: [],
    },
    release,
    entered = false;
  const gate = new Promise((resolve) => {
      release = resolve;
    }),
    originalFetch = globalThis.fetch;
  globalThis.fetch = async (path, options = {}) => {
    assert(path.startsWith("/api/canvas/chats/server-chat"));
    if (options.method === "POST") {
      const { mutation } = JSON.parse(options.body);
      if (!entered) {
        entered = true;
        await gate;
      }
      assert.equal(
        mutation.expected_version_id,
        workspace.documents[0].version.id,
      );
      const old = workspace.documents[0].version;
      workspace = {
        ...workspace,
        documents: [
          {
            id: "doc",
            trashed: false,
            version: {
              ...old,
              id: `v${old.version_number + 1}`,
              parent_id: old.id,
              version_number: old.version_number + 1,
              content: mutation.content,
            },
          },
        ],
      };
    }
    return new Response(JSON.stringify(workspace), {
      headers: { "content-type": "application/json" },
    });
  };
  const container = document.querySelector("main"),
    root = app.createRoot(container);
  roots.add(root);
  try {
    await app.act(async () =>
      root.render(
        app.createElement(app.CanvasWorkspacePanel, {
          mode: "server",
          chatId: "server-chat",
          scope: { message_id: null, revision_id: null },
          request: null,
          onConsumed: () => {},
          onClose: () => {},
          onAsk: async () => true,
          isGenerating: false,
        }),
      ),
    );
    await settle();
    await settle();
    const editor = app.EditorView.findFromDOM(
      container.querySelector(".cm-editor"),
    );
    const replace = (content) =>
      app.act(async () =>
        editor.dispatch({
          changes: { from: 0, to: editor.state.doc.length, insert: content },
        }),
      );
    const save = () =>
      app.act(async () =>
        container
          .querySelector(".canvas-document")
          .dispatchEvent(
            new KeyboardEvent("keydown", {
              key: "s",
              ctrlKey: true,
              bubbles: true,
            }),
          ),
      );
    await replace("First submitted edit");
    await save();
    await until(() => entered);
    await replace("More typing while the first save is pending");
    await app.act(async () => release());
    await settle();
    await settle();
    assert.equal(
      editor.state.doc.toString(),
      "More typing while the first save is pending",
    );
    await save();
    await until(
      () => container.querySelector('[role="status"]').textContent === "Saved",
    );
    assert.equal(
      workspace.documents[0].version.content,
      "More typing while the first save is pending",
    );
    assert.equal(workspace.documents[0].version.version_number, 3);
  } finally {
    release();
    await app.act(async () => root.unmount());
    roots.delete(root);
    globalThis.fetch = originalFetch;
    app.resetPrivateStorageUser();
  }
});
