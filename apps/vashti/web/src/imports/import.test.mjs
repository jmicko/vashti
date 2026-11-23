import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { fileURLToPath } from "node:url";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";
import { syntheticConversation, syntheticPageRecords, fixtureZip as zip } from "./fixtures.mjs";
Object.assign(globalThis, { indexedDB, IDBKeyRange, window: new EventTarget() });
const built = await build({ stdin: { contents: 'export * from "./imports/chatgpt"; export * from "./imports/archive"; export * from "./imports/repository"; export * from "./imports/fileLinks"; export * from "./canvas/device"; export * from "./privateChatStore"; export { privateMessagesWithImportedImages } from "./chatMessages"; export { configure as configureZip } from "@zip.js/zip.js/lib/zip-core-native.js";', resolveDir: fileURLToPath(new URL("../", import.meta.url)), loader: "ts" }, bundle: true, write: false, format: "esm", platform: "node", loader: { ".css": "empty" },
  plugins: [{ name: "reuse-yjs", setup(builder) {
    builder.onResolve({ filter: /^yjs$/ }, () => ({ path: import.meta.resolve("yjs"), external: true }));
  } }]
});
const app = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString("base64")}`);
const target={mode:"device",backend_id:"local",backend_name:"Local",model_name:"fixture-model",allow_history_search:false};

test("workspace Pages import their current editable text without mixing into chat counts",async()=>{
  const id="synthetic-page",path=`pages/${id}/content/current.json`;
  const page={head:{page_id:id,stream_kind:"content",current_sequence:4},includes_current_sequence:true,
    projection:{page_id:id,projection_through_sequence:4,title:"Fixture Page",materialized_search_text:"First section\n\nSecond section",preview:"First section",updated_at:"2026-11-01T17:00:00Z"}};
  const archive= new app.ChatGptArchive(await zip([[path,JSON.stringify(page)],["pages/stale/content/current.json",JSON.stringify({...page,head:{...page.head,page_id:"stale",current_sequence:5},projection:{...page.projection,page_id:"stale"}})]]));
  const summary=await archive.open();
  assert.equal(summary.workspace_pages,1);assert.equal(summary.conversations,0);
  assert.equal(summary.warnings.workspace_documents,1);
  app.setPrivateStorageUser("workspace-fixture",{user_id:"workspace-fixture",key_material:Buffer.alloc(32,4).toString("base64")},"workspace-test");
  const result=await app.runImport(archive,target,new AbortController().signal,()=>{});
  assert.equal(result.imported,1);
  const [document]=await app.listImportedDocuments("device");
  const chat=await app.getPrivateChat(document.id);
  assert.equal(chat.title,"Fixture Page");assert(chat.archived_at);assert.equal(chat.do_not_remember,true);
  const at={message_id:null,revision_id:null}, workspace=await app.deviceCanvasWorkspace(document.id,at);
  const version=workspace.documents[0].version;
  assert.equal(version.content,page.projection.materialized_search_text);
  assert.equal(version.language,"text");
  await app.mutateDeviceCanvas(document.id,at,{operation:"update",document_id:version.document_id,expected_version_id:version.id,content:"Keep this local edit"});
  const retry=new app.ChatGptArchive(await zip([[path,JSON.stringify(page)]]));await retry.open();
  assert.equal((await app.runImport(retry,target,new AbortController().signal,()=>{})).skipped,1);
  assert.equal((await app.deviceCanvasWorkspace(document.id,at)).documents[0].version.content,"Keep this local edit");
  await retry.close();await archive.close();await app.deletePrivateChat(document.id);assert.deepEqual(await app.listImportedDocuments("device"),[]);
  app.resetPrivateStorageUser();
});

test("matching Page checkpoints produce Markdown; stale and malformed checkpoints fall back", async () => {
  for (const [suffix, adapt, expected] of [
    ["rich", v => v, "markdown"],
    ["stale", v => ({ ...v, document: { ...v.document, through_sequence: 1 } }), "text"],
    ["malformed", v => ({ ...v, document: { ...v.document, yjs_state_update: "broken" } }), "text"]
  ]) {
    const id = `fixture-${suffix}`, prefix = `pages/${id}/content/`;
    const { page, checkpoint } = syntheticPageRecords(id, "Fixture formatting");
    const archive = new app.ChatGptArchive(await zip([
      [prefix + "current.json", JSON.stringify(page)],
      [prefix + "checkpoints/fixture-checkpoint.json", JSON.stringify(adapt(checkpoint))]
    ]));
    app.setPrivateStorageUser("rich-fixture", { user_id: "rich-fixture", key_material: Buffer.alloc(32, 3).toString("base64") }, id);
    try {
      await archive.open();
      await app.runImport(archive, target, new AbortController().signal, () => {});
      const [item] = await app.listImportedDocuments("device");
      const { documents } = await app.deviceCanvasWorkspace(item.id, { message_id: null, revision_id: null });
      assert.equal(documents[0].version.language, expected);
      assert.equal(documents[0].version.content, expected === "markdown"
        ? "## First section\n\nA **formatted** paragraph.\n\n- A list item"
        : page.projection.materialized_search_text);
      await app.deletePrivateChat(item.id);
    } finally {
      await archive.close();
      app.resetPrivateStorageUser();
    }
  }
});

test("sandbox links use only explicit ZIP paths or unambiguous conversation file metadata",()=>{
  const c=syntheticConversation();
  c.mapping.user.message.metadata.attachments=[
    {id:"file-a",name:"report with spaces.pdf"},{id:"file-b",name:"ambiguous.pdf"},{id:"file-c",name:"ambiguous.pdf"}
  ];
  c.mapping.first.message.metadata.content_references=[{type:"file",name:"structured.pdf",id:"file-structured"}];
  c.mapping.first.message.content.parts=[
    "[Report](sandbox:/mnt/data/report%20with%20spaces.pdf)\n\n![Structured](sandbox:/mnt/data/structured.pdf)\n\n[Other](sandbox:/mnt/data/ambiguous.pdf)\n\n[Cannot guess](sandbox:/mnt/data/other-people.pdf)\n\n[Traversal](sandbox:/mnt/data/%2e%2e/payload)\n\n[a ref][file]\n\n[file]: sandbox:/mnt/data/exact.txt\n\n`[Literal](sandbox:/mnt/data/structured.pdf)`"
  ];
  const files=new Map([
    ...["file-a","file-b","file-c","file-structured"].map(id=>[id+".dat",{filename:id+".dat",size:12}]),
    ["mnt/data/exact.txt",{filename:"mnt/data/exact.txt",size:12}],
    ["file-unrelated.dat",{filename:"file-unrelated.dat",size:12,originalName:"other-people.pdf"}]
  ]);
  const warnings={}, normalized=app.normalizeConversation(c,files,JSON.stringify(c),warnings);
  const message=normalized.messages.find(m=>m.source_id==="first");
  const related=normalized.assets.filter(a=>a.message_id==="first");
  assert.deepEqual(related.map(a=>a.entry_path).sort(),["file-a.dat","file-structured.dat","mnt/data/exact.txt"]);
  assert.equal(warnings.restored_file_links,3);assert.equal(warnings.sandbox_links,3);
  assert.match(message.content,/Report\]\(\/\.vashti\/imported-file\/asset-/);
  assert.match(message.content,/\[Literal\]\(sandbox:\/mnt\/data\/structured\.pdf\)/);
  assert.match(message.content,/Cannot guess \(file not available/);
  assert.match(message.content,/\[file\]: \/\.vashti\/imported-file\/asset-/);
});

test("encrypted device import binds a file link to its own saved bytes", async()=>{
  const c=syntheticConversation();
  c.mapping.first.message.metadata.content_references=[{type:"file",name:"fixture.pdf",id:"file-document"}];
  c.mapping.first.message.content.parts=["[Read PDF](sandbox:/mnt/data/fixture.pdf)"];
  const pdf=new TextEncoder().encode("%PDF synthetic");
  const archive=new app.ChatGptArchive(await zip([["conversations.json",JSON.stringify([c])],["file-document.dat",pdf]]));
  assert.equal((await archive.open()).warnings.restored_file_links,1);
  app.setPrivateStorageUser("file-link-fixture",{user_id:"file-link-fixture",key_material:Buffer.alloc(32,6).toString("base64")},"file-link-test");
  await app.runImport(archive,target,new AbortController().signal,()=>{});
  const [chat]=await app.listPrivateChats();
  const message=(await app.listPrivateMessages(chat.id)).find(m=>m.attachments.length);
  const localHref=message.active_revision.content_text.match(/\(([^)]+)\)/)[1];
  const assetId=app.importedAttachmentId(localHref);
  assert.equal(assetId,message.attachments[0].id);
  assert.deepEqual(new Uint8Array(await (await app.readPrivateImportAsset(assetId)).arrayBuffer()),pdf);
  await archive.close();app.resetPrivateStorageUser();
});
test("normalization reconstructs branches from parents and merges exported reasoning", () => {
  const c = syntheticConversation(), raw = JSON.stringify(c);
  const parsed = app.normalizeConversation(c, new Map([["file-fixture.dat", { filename: "file-fixture.dat", size: 10 }]]), raw);
  assert.equal(parsed.messages.length, 3); assert.equal(parsed.assets.length, 1);
  assert.equal(parsed.current_node_id, "alternative");
  assert.equal(parsed.messages.find(m => m.source_id === "first").parent_id, "user");
  assert.match(parsed.messages.find(m => m.source_id === "first").thinking, /Synthetic reasoning/);
  assert.equal(parsed.original_json, raw); assert.equal(parsed.archived, true);
  assert(parsed.messages.every(m => !["thought", "recap"].includes(m.source_id)));
});
test("ZIP supports split and legacy JSON files and only materializes requested assets", async () => {
  const blob = await zip([["conversations-000.json", JSON.stringify([syntheticConversation()])], ["conversations-001.json", JSON.stringify([syntheticConversation("second")])], ["file-fixture.dat", new Uint8Array([137,80,78,71,13,10,26,10])], ["conversation_asset_file_names.json", JSON.stringify({"file-fixture.dat":"original-fixture.png"})], ["unused.dat", new Uint8Array([1])], ["workspace/current.json", "{}"]]);
  const reader = new app.ChatGptArchive(blob), summary = await reader.open();
  assert.equal(summary.conversations, 2); assert.equal(summary.messages, 6); assert.equal(summary.branches, 2);
  assert.equal(summary.assets, 2);
  assert.equal(summary.warnings.unreferenced_files, 1); assert.equal(summary.warnings.workspace_documents, 1);
  const c = await reader.next(); assert.equal(c.source_id, "fixture-conversation");
  assert.equal(c.assets[0].filename, "original-fixture.png");
  assert.equal((await reader.asset(c.assets[0].entry_path)).size, 8);
  assert.equal((await reader.next()).source_id, "second"); assert.equal(await reader.next(), null); await reader.close();
  const legacy = new app.ChatGptArchive(await zip([["conversations.json", JSON.stringify([syntheticConversation()])]]));
  assert.equal((await legacy.open()).conversations, 1); await legacy.close();
});
test("fallback decompression works without network requests or WebAssembly permission", async () => {
  const c = syntheticConversation(); c.mapping.first.message.content.parts = ["Synthetic compressed text. ".repeat(500)];
  const blob = await zip([["conversations.json", JSON.stringify([c])]]);
  const originalFetch = globalThis.fetch, instantiate = WebAssembly.instantiate;
  globalThis.fetch = async () => { throw new Error("ZIP must not fetch codecs"); };
  WebAssembly.instantiate = async () => { throw new Error("WebAssembly is blocked by CSP"); };
  app.configureZip({ useCompressionStream: false });
  const archive = new app.ChatGptArchive(blob);
  try {
    assert.equal((await archive.open()).conversations, 1);
    assert.equal((await archive.next()).messages.find(m => m.source_id === "first").content, c.mapping.first.message.content.parts[0]);
  } finally { await archive.close(); app.configureZip({ useCompressionStream: true }); globalThis.fetch = originalFetch; WebAssembly.instantiate = instantiate; }
});
test("unsafe paths, cyclic graphs, malformed JSON and missing files are explicit", async () => {
  for (const p of ["../secret", "/absolute", "C:/absolute", "folder/../secret", "folder\\secret", "x\u0000y"]) assert(!app.safeEntryPath(p));
  const c = syntheticConversation(); c.mapping.user.parent = "alternative";
  assert.throws(() => app.normalizeConversation(c, new Map(), JSON.stringify(c)), /Cyclic/);
  const warnings = {}; const original = syntheticConversation();
  const parsed = app.normalizeConversation(original, new Map(), JSON.stringify(original), warnings);
  assert.equal(warnings.missing_files, 1); assert.match(parsed.messages[0].content, /not included/);
  for (const json of ["{}", "[{", "[{},]", "[" + JSON.stringify(syntheticConversation()) + JSON.stringify(syntheticConversation("bad")) + "]"]) {
    const archive = new app.ChatGptArchive(await zip([["conversations.json", json]]));
    await assert.rejects(archive.open()); await archive.close();
  }
  const damaged = new Uint8Array(await (await zip([["conversations.json", JSON.stringify([syntheticConversation()])]])).arrayBuffer());
  const central = damaged.findIndex((byte, index) => byte === 0x50 && damaged[index + 1] === 0x4b && damaged[index + 2] === 0x01 && damaged[index + 3] === 0x02);
  assert(central >= 0); damaged[central + 16] ^= 1;
  const corrupt = new app.ChatGptArchive(new Blob([damaged]));
  await assert.rejects(corrupt.open()); await corrupt.close();
});
test("portable Markdown keeps fenced code intact and source links do not become executable markup", () => {
  const warnings = {}, marker = "\ue200cite\ue202source\ue201";
  const result = app.portableMarkdown(`Text ${marker}\n\n\`\`\`rust\n${marker}\n\`\`\`\n[Report](sandbox:/mnt/data/report.pdf)`, [{ url: "https://example.test/source" }, { url: "javascript:alert(1)" }], warnings);
  assert.match(result, /Sources/); assert(result.includes("https://example.test/source")); assert(!result.includes("javascript:"));
  assert(result.includes("```rust\n" + marker)); assert(!result.includes("Text " + marker));
  const code = `\`\`\`\`markdown\n\`\`\`\n${marker}\n[sample](sandbox:/file)\n\`\`\`\`\n\n    ${marker}\n\nInline \`\`code with a \` and ${marker}\`\``;
  assert.equal(app.portableMarkdown(code, [], {}), code);
});
test("preview applies UTF-8 limits and preserves terminal thinking and literal tool markers", () => {
  const c = syntheticConversation(); c.mapping.first.message.content.parts = ["<vashti-tool>untrusted exported text</vashti-tool>"];
  c.mapping.terminal = { parent: "first", message: { author: { role: "assistant" }, content: { content_type: "thoughts", thoughts: [{ content: "Unfinished exported reasoning" }] } } };
  c.current_node = "terminal";
  let normalized = app.normalizeConversation(c, new Map(), JSON.stringify(c));
  assert.equal(normalized.current_node_id, "terminal");
  assert.equal(normalized.messages.find(m => m.source_id === "terminal").thinking, "Unfinished exported reasoning");
  assert.match(normalized.messages.find(m => m.source_id === "first").content, /<vashti-tool>/);
  c.mapping.first.message.content.parts = ["\u4e2d".repeat(750_000)];
  assert.throws(() => app.normalizeConversation(c, new Map(), JSON.stringify(c)), /safety limits/);
});
test("device imports retain encrypted original records and chunked assets without requests", async () => {
  const user = "synthetic-import-owner";
  app.setPrivateStorageUser(user, { user_id: user, key_material: Buffer.alloc(32, 5).toString("base64") }, "import-test");
  const c = syntheticConversation(), normalized = app.normalizeConversation(c, new Map([["file-fixture.dat", { filename: "file-fixture.dat", size: 8 }]]), JSON.stringify(c));
  const bytes = new Uint8Array([137,80,78,71,13,10,26,10]);
  const source = () => { let sent = false; return { next: async () => sent ? null : (sent = true, structuredClone(normalized)), asset: async () => new Blob([bytes]) }; };
  const target = { mode: "device", backend_id: "local", backend_name: "Local", model_name: "reply-model", allow_history_search: false };
  const originalFetch = globalThis.fetch; let requests = 0;
  globalThis.fetch = async () => { requests++; throw new Error("Device import must not make requests"); };
  const progress = await app.runImport(source(), target, new AbortController().signal, () => {});
  globalThis.fetch = originalFetch; assert.equal(requests, 0);
  assert.equal(progress.imported, 1);
  const chats = await app.listPrivateChats(); assert.equal(chats.length, 1); assert(chats[0].archived_at);
  assert.equal(chats[0].default_model_name, "reply-model");
  const messages = await app.listPrivateMessages(chats[0].id); assert.equal(messages.length, 3);
  assert(messages.every(m => m.backend_id === null));
  assert.equal(messages.find(m => m.role === "user").active_revision.profile_source, undefined);
  const attachment = messages.flatMap(m => m.attachments)[0]; assert.equal(attachment.data_url, undefined);
  assert.deepEqual(new Uint8Array(await (await app.readPrivateImportAsset(attachment.import_asset_id)).arrayBuffer()), bytes);
  const originalReader = globalThis.FileReader; let switchAccount = false;
  globalThis.FileReader = class {
    readAsDataURL(blob) { void blob.arrayBuffer().then(bytes => {
      this.result = `data:${blob.type};base64,${Buffer.from(bytes).toString("base64")}`;
      if (switchAccount) app.setPrivateStorageUser("during-image", { user_id: "during-image", key_material: Buffer.alloc(32, 7).toString("base64") }, "import-test");
      this.onload?.();
    }); }
  };
  try {
    const stopBefore = messages.find(m => m.role === "assistant").id;
    const promptMessages = await app.privateMessagesWithImportedImages(messages, stopBefore);
    assert.match(promptMessages.find(m => m.role === "user").attachments[0].data_url, /^data:image\/png;base64,/);
    assert.equal(messages.find(m => m.role === "user").attachments[0].data_url, undefined);
    switchAccount = true;
    await assert.rejects(app.privateMessagesWithImportedImages(messages, stopBefore), /user changed/);
  } finally {
    globalThis.FileReader = originalReader;
    app.setPrivateStorageUser(user, { user_id: user, key_material: Buffer.alloc(32, 5).toString("base64") }, "import-test");
  }
  const open = indexedDB.open("vashti-private-local-import-test-synthetic-import-owner");
  const db = await new Promise((resolve, reject) => { open.onsuccess = () => resolve(open.result); open.onerror = () => reject(open.error); });
  const tx = db.transaction("private_chat_imports", "readonly"), request = tx.objectStore("private_chat_imports").getAll();
  const stored = await new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
  assert(stored.length >= 3); assert(!JSON.stringify(stored).includes("A synthetic prompt")); assert(!JSON.stringify(stored).includes("Synthetic import fixture")); db.close();
  assert.equal((await app.runImport(source(), target, new AbortController().signal, () => {})).skipped, 1);
  assert.equal((await app.listPrivateChats()).length, 1);
  app.setPrivateStorageUser("other", { user_id: "other", key_material: Buffer.alloc(32, 6).toString("base64") }, "import-test");
  await assert.rejects(app.readPrivateImportAsset(attachment.import_asset_id));
  app.setPrivateStorageUser(user, { user_id: user, key_material: Buffer.alloc(32, 5).toString("base64") }, "import-test");
  await app.deletePrivateChat(chats[0].id); await assert.rejects(app.readPrivateImportAsset(attachment.import_asset_id));
  app.resetPrivateStorageUser();
});
test("cancellation keeps a resumable chat and account changes stop late reads", async () => {
  const user = "synthetic-resume-owner", key = { user_id: user, key_material: Buffer.alloc(32, 8).toString("base64") };
  app.setPrivateStorageUser(user, key, "resume-test");
  const c = syntheticConversation("resume-fixture"), bytes = new Uint8Array(600_000); c.is_do_not_remember = true; bytes.set([137,80,78,71,13,10,26,10]);
  const normalized = app.normalizeConversation(c, new Map([["file-fixture.dat", { filename: "file-fixture.dat", size: bytes.length }]]), JSON.stringify(c));
  const target = { mode: "device", backend_id: "local", backend_name: "Local", model_name: "reply-model", allow_history_search: false }, abort = new AbortController();
  const source = (cancel) => { let sent = false, reads = 0; return { next: async () => sent ? null : (sent = true, structuredClone(normalized)), asset: async () => { if (cancel && ++reads === 2) abort.abort(); return new Blob([bytes]); } }; };
  await assert.rejects(app.runImport(source(true), target, abort.signal, () => {}));
  const chats = await app.listPrivateChats(); assert.equal(chats.length, 1);
  assert.equal(chats[0].do_not_remember, true);
  assert.equal((await app.runImport(source(false), target, new AbortController().signal, () => {})).skipped, 1);
  const attachment = (await app.listPrivateMessages(chats[0].id)).flatMap(m => m.attachments)[0];
  assert.equal((await app.readPrivateImportAsset(attachment.import_asset_id)).size, bytes.length);
  let sent = false;
  await assert.rejects(app.runImport({ next: async () => {
    if (sent) return null; sent = true;
    app.setPrivateStorageUser("other-resume", { user_id: "other-resume", key_material: Buffer.alloc(32, 9).toString("base64") }, "resume-test");
    return structuredClone(normalized);
  }, asset: async () => new Blob([bytes]) }, target, new AbortController().signal, () => {}), /user changed/);
  assert.equal((await app.listPrivateChats()).length, 0);
  app.setPrivateStorageUser(user, key, "different-server"); assert.equal((await app.listPrivateChats()).length, 0);
  app.setPrivateStorageUser(user, key, "resume-test");
  assert.equal((await app.listPrivateChats()).length, 1); await app.deletePrivateChat(chats[0].id); app.resetPrivateStorageUser();
});
