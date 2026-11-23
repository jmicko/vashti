import assert from "node:assert/strict";
import test from "node:test";
import { build } from "esbuild";
import { indexedDB, IDBKeyRange } from "fake-indexeddb";
import { fileURLToPath } from "node:url";

Object.assign(globalThis, { indexedDB, IDBKeyRange, window: new EventTarget() });
const bundled = await build({
  entryPoints: [fileURLToPath(new URL("./privateChatStore.ts", import.meta.url))],
  bundle: true, format: "esm", platform: "node", write: false
});
const store = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString("base64")}`);
const login = (user, server = "archive-test") => store.setPrivateStorageUser(user, {
  user_id: user, key_material: Buffer.alloc(32, 7).toString("base64")
}, server);
async function seed(user) {
  login(user);
  const chat = await store.createPrivateChat({ title: "Archive fixture", backendId: "local", backendName: "Local", modelName: "model" });
  const root = store.createPrivateMessage({ chatId: chat.id, parentMessageId: null, role: "user", contentText: "Keep every branch." });
  const child = store.createPrivateMessage({ chatId: chat.id, parentMessageId: root.id, role: "assistant", contentText: "First branch" });
  const sibling = store.createPrivateMessage({ chatId: chat.id, parentMessageId: root.id, role: "assistant", contentText: "Second branch" });
  root.active_child_message_id = child.id;
  root.attachments = [{ id: "synthetic-attachment", original_filename: "fixture.txt" }];
  chat.active_root_message_id = root.id;
  await store.savePrivateChat(chat);
  await store.savePrivateMessages([root, child, sibling]);
  return chat;
}

test("private archive retains messages, branches and attachments across vault reopen", async () => {
  const chat = await seed("archive-owner");
  const before = await store.listPrivateMessages(chat.id);
  const archived = await store.setPrivateChatArchived(chat.id, true);
  assert(archived.archived_at);
  assert.equal(archived.updated_at, chat.updated_at);
  assert.equal((await store.listPrivateChats())[0].archived_at, archived.archived_at);
  assert.equal((await store.setPrivateChatArchived(chat.id, true)).archived_at, archived.archived_at);
  // A streaming response may still hold the pre-archive chat object.
  await store.savePrivateChat({ ...chat, title: "Finished while archived" });
  assert.equal((await store.getPrivateChat(chat.id)).archived_at, archived.archived_at);
  store.resetPrivateStorageUser();
  login("archive-owner");
  assert.equal((await store.getPrivateChat(chat.id)).archived_at, archived.archived_at);
  assert.deepEqual(await store.listPrivateMessages(chat.id), before);
  const restored = await store.setPrivateChatArchived(chat.id, false);
  assert.equal(restored.archived_at, null);
  assert.equal(restored.active_root_message_id, chat.active_root_message_id);
  await store.savePrivateChat(archived);
  assert.equal((await store.getPrivateChat(chat.id)).archived_at, null);
  assert.deepEqual(await store.listPrivateMessages(chat.id), before);
  await store.deletePrivateChat(chat.id);
  assert.equal(await store.getPrivateChat(chat.id), null);
  assert.deepEqual(await store.listPrivateMessages(chat.id), []);
  await assert.rejects(store.setPrivateChatArchived(chat.id, false), /not found/);
  store.resetPrivateStorageUser();
});

test("private archive is encrypted, isolated by account and server, and removed on delete", async () => {
  const chat = await seed("isolated-archive");
  const archived = await store.setPrivateChatArchived(chat.id, true);
  const databases = await indexedDB.databases();
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open(databases.find(d => d.name.includes("isolated-archive")).name);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  const read = () => new Promise((resolve, reject) => {
    const request = db.transaction("private_chat_archives").objectStore("private_chat_archives").get(chat.id);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  const raw = await read();
  assert(raw.encrypted_payload);
  assert(!JSON.stringify(raw).includes("archived_at"));
  assert(!JSON.stringify(raw).includes(String(archived.archived_at)));
  login("other-user");
  assert.equal(await store.getPrivateChat(chat.id), null);
  await assert.rejects(store.setPrivateChatArchived(chat.id, false), /not found/);
  login("isolated-archive", "other-server");
  assert.equal(await store.getPrivateChat(chat.id), null);
  await assert.rejects(store.setPrivateChatArchived(chat.id, false), /not found/);
  login("isolated-archive");
  assert((await store.getPrivateChat(chat.id)).archived_at);
  await store.deletePrivateChat(chat.id);
  assert.equal(await read(), undefined);
  db.close();
  store.resetPrivateStorageUser();
});

test("a version-12 device vault upgrades without losing its existing chats", async () => {
  const chat = { id: "legacy-chat", title: "Legacy fixture", default_backend_id: "local", backend_name: "Local", default_model_name: "model", active_root_message_id: null, created_at: 100, updated_at: 100, last_message_at: 100 };
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open("vashti-private-local-archive-test-upgrade-archive", 12);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("private_chats", { keyPath: "id" });
      request.result.createObjectStore("private_messages", { keyPath: "id" }).createIndex("chat_id", "chat_id");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  await new Promise((resolve, reject) => {
    const tx = db.transaction("private_chats", "readwrite");
    tx.objectStore("private_chats").put(chat);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
  login("upgrade-archive");
  assert.equal((await store.listPrivateChats())[0].archived_at, null);
  assert.equal((await store.getPrivateChat(chat.id)).title, chat.title);
  assert((await store.setPrivateChatArchived(chat.id, true)).archived_at);
  assert.equal((await store.setPrivateChatArchived(chat.id, false)).archived_at, null);
  store.resetPrivateStorageUser();
});
