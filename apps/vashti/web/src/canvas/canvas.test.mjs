import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";
import { fileURLToPath } from "node:url";

Object.assign(globalThis, {
  indexedDB,
  IDBKeyRange,
  window: new EventTarget(),
});
const bundle = await build({
  stdin: {
    contents:
      'export * from "../privateChatStore"; export * from "./device"; export * from "./types"; export * from "./repository"; export * from "./references"; export * from "./deviceTools";',
    resolveDir: fileURLToPath(new URL(".", import.meta.url)),
  },
  bundle: true,
  format: "esm",
  platform: "node",
  write: false,
});
const api = await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
);
const login = (user, server = "canvas-fixture") =>
  api.setPrivateStorageUser(
    user,
    { user_id: user, key_material: Buffer.alloc(32, 9).toString("base64") },
    server,
  );
const scope = (message) => ({
  message_id: message?.id ?? null,
  revision_id: message?.active_revision_id ?? null,
});
const create = (
  chat,
  at,
  title = "Synthetic document",
  content = "A first line.\nA second line.",
) =>
  api.mutateDeviceCanvas(chat.id, at, {
    operation: "create",
    title,
    language: "markdown",
    content,
  });
const update = (doc, content, edit_session_id = "editing-session") => ({
  operation: "update",
  document_id: doc.id,
  expected_version_id: doc.version.id,
  content,
  edit_session_id,
});
async function seed(user) {
  login(user);
  const chat = await api.createPrivateChat({
    title: "Canvas fixture",
    backendId: "local",
    backendName: "Local",
    modelName: "synthetic-model",
  });
  const root = api.createPrivateMessage({
    chatId: chat.id,
    parentMessageId: null,
    role: "user",
    contentText: "A synthetic request",
  });
  chat.active_root_message_id = root.id;
  await api.savePrivateChat(chat);
  await api.savePrivateMessages([root]);
  return { chat, root, at: scope(root) };
}
async function response(chat, parent) {
  const message = api.createPrivateMessage({
    chatId: chat.id,
    parentMessageId: parent.id,
    role: "assistant",
    contentText: "",
    status: "streaming",
  });
  await api.savePrivateMessages([message]);
  await api.captureDeviceCanvas(chat.id, scope(message));
  return message;
}

test("device documents follow branches and preserve generation snapshots", async () => {
  const { chat, root, at } = await seed("canvas-branches"),
    original = await create(chat, at);
  const a = await response(chat, root),
    b = await response(chat, root);
  const edited = await api.mutateDeviceCanvas(
    chat.id,
    scope(a),
    update(original.documents[0], "Only branch A", undefined),
    "Synthetic model",
  );
  assert.equal(edited.documents[0].version.actor, "model");
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, scope(b))).documents[0].version
      .content,
    original.documents[0].version.content,
  );
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, at)).documents[0].version.id,
    original.documents[0].version.id,
  );
  const captured = await api.readPrivateCanvasRecord(
    `canvas-input:${api.canvasScopeKey(chat.id, scope(a))}`,
  );
  assert.equal(
    captured.value.heads[0].version_id,
    original.documents[0].version.id,
  );
  const child = await response(chat, a);
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, scope(child))).documents[0]
      .version.content,
    "Only branch A",
  );
  await create(chat, at, "Created later");
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, scope(b))).documents.length,
    1,
  );
});

test("autosaves coalesce, while inherited revisions, comments and snapshots seal checkpoints", async () => {
  const { chat, root, at } = await seed("canvas-checkpoints"),
    initial = await create(chat, at);
  const first = await api.mutateDeviceCanvas(
    chat.id,
    at,
    update(initial.documents[0], "One edit"),
  );
  const second = await api.mutateDeviceCanvas(
    chat.id,
    at,
    update(first.documents[0], "Two edits"),
  );
  assert.equal(
    first.documents[0].version.version_number,
    second.documents[0].version.version_number,
  );
  assert.equal(
    (await api.deviceCanvasHistory(chat.id, initial.documents[0].id)).length,
    2,
  );
  await api.inheritDeviceCanvas(
    chat.id,
    at.revision_id,
    "continuation-revision",
  );
  const third = await api.mutateDeviceCanvas(
    chat.id,
    at,
    update(second.documents[0], "After inheritance"),
  );
  assert(
    third.documents[0].version.version_number >
      second.documents[0].version.version_number,
  );
  assert.equal(
    (
      await api.deviceCanvasVersion(
        chat.id,
        initial.documents[0].id,
        second.documents[0].version.id,
      )
    ).content,
    "Two edits",
  );
  const nextRoot = {
    ...root,
    active_revision_id: "continuation-revision",
    active_revision: { ...root.active_revision, id: "continuation-revision" },
  };
  await api.savePrivateMessages([nextRoot]);
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, scope(nextRoot))).documents[0]
      .version.content,
    "Two edits",
  );
  await assert.rejects(api.deviceCanvasWorkspace(chat.id, at), /changed/);
});

test("optimistic writes and exact patches are atomic, and stale or stopped models cannot edit", async () => {
  const { chat, root, at } = await seed("canvas-conflicts"),
    original = await create(chat, at),
    doc = original.documents[0];
  const results = await Promise.allSettled([
    api.mutateDeviceCanvas(chat.id, at, update(doc, "First window")),
    api.mutateDeviceCanvas(chat.id, at, update(doc, "Second window")),
  ]);
  assert.equal(results.filter((r) => r.status === "fulfilled").length, 1);
  assert.equal(results.filter((r) => r.status === "rejected").length, 1);
  const current = (await api.deviceCanvasWorkspace(chat.id, at)).documents[0];
  await assert.rejects(
    api.mutateDeviceCanvas(chat.id, at, {
      ...update(current, ""),
      content: undefined,
      edits: [
        { find: current.version.content, replace: "Changed" },
        { find: "missing", replace: "bad" },
      ],
    }),
    /exactly once/,
  );
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, at)).documents[0].version.id,
    current.version.id,
  );
  await assert.rejects(
    api.mutateDeviceCanvas(chat.id, at, {
      operation: "update",
      document_id: current.id,
      expected_version_id: current.version.id,
    }),
    /Provide content/,
  );
  const assistant = await response(chat, root);
  await api.savePrivateMessages([{ ...assistant, status: "stopped" }]);
  await assert.rejects(
    api.mutateDeviceCanvas(
      chat.id,
      scope(assistant),
      update(current, "After stop"),
      "Synthetic model",
    ),
    /changed/,
  );
  await api.savePrivateMessages([assistant]);
  await assert.rejects(
    api.mutateDeviceCanvas(
      chat.id,
      scope(assistant),
      update(current, "Cancelled"),
      "Synthetic model",
      () => {
        throw new Error("Cancelled run");
      },
    ),
    /Cancelled run/,
  );
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, scope(assistant))).documents[0]
      .version.id,
    current.version.id,
  );
});

test("comments, branch-only removal, and restoration retain history", async () => {
  const { chat, root, at } = await seed("canvas-history"),
    initial = await create(chat, at),
    doc = initial.documents[0];
  const commented = await api.mutateDeviceCanvas(chat.id, at, {
    operation: "comment",
    document_id: doc.id,
    expected_version_id: doc.version.id,
    quote: "first line",
    content: "A synthetic comment",
  });
  assert.equal(commented.comments.length, 1);
  const child = await response(chat, root);
  const trash = {
    operation: "trash",
    document_id: doc.id,
    expected_version_id: doc.version.id,
    trashed: true,
  };
  await assert.rejects(
    api.mutateDeviceCanvas(chat.id, scope(child), trash, "Synthetic model"),
    /Only the owner/,
  );
  assert(
    (await api.mutateDeviceCanvas(chat.id, scope(child), trash)).documents[0]
      .trashed,
  );
  assert(!(await api.deviceCanvasWorkspace(chat.id, at)).documents[0].trashed);
  const restored = await api.mutateDeviceCanvas(chat.id, scope(child), {
    operation: "restore",
    document_id: doc.id,
    expected_version_id: doc.version.id,
    version_id: doc.version.id,
  });
  assert(!restored.documents[0].trashed);
  assert(
    restored.documents[0].version.version_number > doc.version.version_number,
  );
  assert.equal(
    (
      await api.mutateDeviceCanvas(chat.id, scope(child), {
        operation: "resolve_comment",
        comment_id: commented.comments[0].id,
        resolved: true,
      })
    ).comments[0].resolved,
    true,
  );
  await api.setPrivateChatArchived(chat.id, true);
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, at)).documents.length,
    1,
  );
});

test("create retries are idempotent, validation is bounded, and Unicode reads have stable offsets", async () => {
  const { chat, at } = await seed("canvas-validation"),
    document_id = crypto.randomUUID(),
    mutation = {
      operation: "create",
      document_id,
      title: "Retry-safe",
      language: "text",
      content: "same content",
    };
  const a = await api.mutateDeviceCanvas(chat.id, at, mutation),
    b = await api.mutateDeviceCanvas(chat.id, at, mutation);
  assert.equal(b.documents.length, 1);
  assert.equal(a.documents[0].version.id, b.documents[0].version.id);
  await assert.rejects(
    api.mutateDeviceCanvas(chat.id, at, {
      ...mutation,
      content: "Different retry",
    }),
    /changed/,
  );
  assert.throws(() => api.validateCanvas("bad\nname", "text", "text"), /title/);
  assert.throws(() => api.validateCanvas("Name", "exec", "text"), /supported/);
  assert.throws(
    () => api.validateCanvas("Name", "text", "🌿".repeat(70_000)),
    /256 KiB/,
  );
  assert.throws(
    () => api.exactCanvasEdits("aaa", [{ find: "aa", replace: "x" }]),
    /exactly once/,
  );
  assert.equal(
    api.exactCanvasEdits("literal $1", [{ find: "$1", replace: "$& $`" }]),
    "literal $& $`",
  );
  const content = "🌿".repeat(4000) + "end",
    v = { ...a.documents[0].version, content };
  const first = api.canvasReadResult(v, {}),
    second = api.canvasReadResult(v, { offset: first.next_offset });
  assert.equal(new TextEncoder().encode(first.content).length, 12000);
  assert.equal(first.content + second.content, content);
  assert.equal(second.truncated, false);
});

test("encrypted drafts are per-session and clearing one save never removes newer typing", async () => {
  const { chat, at } = await seed("canvas-drafts"),
    initial = await create(chat, at),
    doc = initial.documents[0];
  const draft = {
    title: doc.version.title,
    language: "markdown",
    content: "First pending draft",
    expected_version_id: doc.version.id,
    edit_session_id: "first-window",
    updated_at: 1,
  };
  await api.saveCanvasDraft("device", chat.id, at, doc.id, draft);
  const newer = { ...draft, content: "Later typing", updated_at: 2 };
  await api.saveCanvasDraft("device", chat.id, at, doc.id, newer);
  await api.saveCanvasDraft("device", chat.id, at, doc.id, {
    ...draft,
    edit_session_id: "second-window",
    updated_at: 3,
  });
  await api.clearCanvasDraft("device", chat.id, at, doc.id, draft);
  assert.deepEqual(
    (await api.readCanvasDrafts("device", chat.id, at, doc.id)).map(
      (d) => d.content,
    ),
    [draft.content, newer.content],
  );
  await api.clearCanvasDraft("device", chat.id, at, doc.id, newer);
  assert.equal(
    (await api.readCanvasDrafts("device", chat.id, at, doc.id)).length,
    1,
  );
});

test("private canvases stay encrypted, account/server isolated, and are deleted with their chat", async () => {
  const { chat, at } = await seed("canvas-owner"),
    initial = await create(
      chat,
      at,
      "Private canvas sentinel",
      "Content must stay encrypted",
    ),
    doc = initial.documents[0];
  const databases = await indexedDB.databases(),
    name = databases.find((d) => d.name.includes("canvas-owner")).name;
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open(name);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  const raw = () =>
    new Promise((resolve, reject) => {
      const request = db
        .transaction("private_canvases")
        .objectStore("private_canvases")
        .getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  assert(!JSON.stringify(await raw()).includes("sentinel"));
  assert(!JSON.stringify(await raw()).includes("Content must stay"));
  login("other-canvas-owner");
  await assert.rejects(api.deviceCanvasWorkspace(chat.id, at), /not found/);
  login("canvas-owner", "another-server");
  await assert.rejects(
    api.deviceCanvasVersion(chat.id, doc.id, doc.version.id),
    /not found/,
  );
  login("canvas-owner");
  assert.equal(
    (await api.deviceCanvasWorkspace(chat.id, at)).documents[0].version.content,
    doc.version.content,
  );
  const pending = api.deviceCanvasHistory(chat.id, doc.id);
  login("other-canvas-owner");
  await assert.rejects(pending, /changed|storage|session|user/i);
  login("canvas-owner");
  await api.deletePrivateChat(chat.id);
  assert.equal((await raw()).length, 0);
  db.close();
});

test("only Canvas tool results produce links, with one latest link for each document", () => {
  const ref = {
    document_id: "doc",
    version_id: "v1",
    title: "Draft",
    version_number: 1,
    language: "markdown",
  };
  const segment = (name, result) => ({
    type: "tool",
    usage: { name, result: JSON.stringify(result) },
  });
  assert.deepEqual(
    api.canvasReferences([
      segment("read_note", { canvas: ref }),
      { type: "text", text: JSON.stringify({ canvas: ref }) },
    ]),
    [],
  );
  assert.deepEqual(
    api.canvasReferences([
      segment("create_canvas", { canvas: ref }),
      segment("update_canvas", {
        canvas: { ...ref, version_id: "v2", version_number: 2 },
      }),
    ]),
    [{ ...ref, version_id: "v2", version_number: 2 }],
  );
});

test("device Canvas tools respect permissions, disabled chats, and cancellation", async () => {
  const { chat, root } = await seed("canvas-tools"),
    assistant = await response(chat, root),
    oldFetch = globalThis.fetch;
  let enabled = true;
  globalThis.fetch = async (path) => {
    assert.equal(path, "/api/tools");
    return new Response(
      JSON.stringify({
        tools_enabled: enabled,
        device_tools: [{ id: "canvas" }],
      }),
      { headers: { "content-type": "application/json" } },
    );
  };
  try {
    const args = {
        title: "Tool draft",
        language: "markdown",
        content: "Tool content",
      },
      check = () => {};
    const result = await api.executeDeviceCanvasTool(
      "create_canvas",
      args,
      chat.id,
      scope(assistant),
      "Synthetic model",
      check,
    );
    assert(result.saved);
    assert.equal(
      (
        await api.executeDeviceCanvasTool(
          "read_canvas",
          { document_id: result.canvas.document_id },
          chat.id,
          scope(assistant),
          "Synthetic model",
          check,
        )
      ).content,
      args.content,
    );
    enabled = false;
    await assert.rejects(
      api.executeDeviceCanvasTool(
        "create_canvas",
        args,
        chat.id,
        scope(assistant),
        "Synthetic model",
        check,
      ),
      /unavailable/,
    );
    enabled = true;
    await api.savePrivateChat({
      ...chat,
      tool_preferences: { ...chat.tool_preferences, tool_use_enabled: false },
    });
    await assert.rejects(
      api.executeDeviceCanvasTool(
        "list_canvases",
        {},
        chat.id,
        scope(assistant),
        "Synthetic model",
        check,
      ),
      /unavailable/,
    );
  } finally {
    globalThis.fetch = oldFetch;
  }
});
