import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import { requestJson } from "../api";
import { commitPrivateImportedChat, createPrivateMessage, listPrivateChats, privateId, privateImageGuard, privateProfileIdentity, readPrivateImport, writePrivateImport, type PrivateChatDetail, type PrivateChatMessage } from "../privateChatStore";
import { sourceToken, type ImportConversation, type ImportAsset } from "./chatgpt";
import { bindImportedFileLinks } from "./fileLinks";

type AssetStatus = { id: string; source_id: string; uploaded_bytes: number; complete: boolean };
type ImportResult = { chat_id: string; existing: boolean; changed_source: boolean; assets: AssetStatus[] };
type Source = { chat_id: string; source_hash: string; original_json: string };
export type ImportTarget = { mode: "server" | "device"; backend_id: string; backend_name: string; model_name: string; allow_history_search: boolean; learn_from_import?: boolean };
export type ImportProgress = { imported: number; skipped: number; completed: number; files: number; phase: "conversations" | "attachments"; changed: number };
export interface ArchiveSource { next(): Promise<ImportConversation | null>; asset(path: string): Promise<Blob> }
const CHUNK = 512 * 1024;
const base64 = (bytes: Uint8Array) => { let result = ""; for (let i = 0; i < bytes.length; i += 0x8000) result += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(result); };

export type ImportedDocument = {id:string;title:string;updated_at:number;mode:"server"|"device"};
export async function listImportedDocuments(mode:ImportedDocument["mode"]):Promise<ImportedDocument[]> {
  const check=privateImageGuard(), instance=privateProfileIdentity().instance_id;
  const docs= mode==="server"
    ? await requestJson<Omit<ImportedDocument,"mode">[]>("/api/imports/chatgpt/documents",{expectedServerInstanceId:instance})
    : (await listPrivateChats()).filter(chat=>chat.imported_workspace);
  check();
  return docs.map(({id,title,updated_at})=>({id,title,updated_at,mode}));
}

export function attachmentType(bytes: Uint8Array): { kind: "image" | "file"; mime: string; extension: string } {
  if (bytes[0] === 137 && bytes[1] === 80 && bytes[2] === 78 && bytes[3] === 71) return { kind: "image", mime: "image/png", extension: "png" };
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return { kind: "image", mime: "image/jpeg", extension: "jpg" };
  if (new TextDecoder().decode(bytes.subarray(0, 4)) === "GIF8") return { kind: "image", mime: "image/gif", extension: "gif" };
  if (new TextDecoder().decode(bytes.subarray(0, 4)) === "RIFF" && new TextDecoder().decode(bytes.subarray(8, 12)) === "WEBP") return { kind: "image", mime: "image/webp", extension: "webp" };
  return { kind: "file", mime: "application/octet-stream", extension: "bin" };
}

async function importDevice(c: ImportConversation, target: ImportTarget, archive: ArchiveSource, check: () => void, onFile: () => void): Promise<ImportResult> {
  const key = "chatgpt-source-" + sourceToken(c.source_id), sourceHash = sourceToken(c.original_json);
  let previous = await readPrivateImport<Source>(key); check();
  const existing = previous !== null;
  if (previous && previous.source_hash !== sourceHash) return { chat_id: previous.chat_id, existing, changed_source: true, assets: [] };
  const ids = new Map(c.messages.map(m => [m.source_id, { id: privateId("private-message"), revision: privateId("private-revision") }]));
  const assetIds = new Map(c.assets.map(a => [a.source_id, "chatgpt-asset-" + sourceToken(c.source_id + "\0" + a.source_id)]));
  const assetId = (a: ImportAsset) => assetIds.get(a.source_id)!;
  const sourceByAsset = new Map(c.assets.map(a => [assetId(a), a]));
  const assetsByMessage = new Map<string, ImportAsset[]>();
  for (const asset of c.assets) { const list = assetsByMessage.get(asset.message_id) ?? []; list.push(asset); assetsByMessage.set(asset.message_id, list); }
  if (!previous) {
    const chatId = privateId("private-chat"), active = new Map<string, string>();
    const sources = new Map(c.messages.map(m => [m.source_id, m]));
    let current = c.current_node_id ?? c.messages[c.messages.length - 1].source_id, root = current;
    while (current) { root = current; const parent = sources.get(current)?.parent_id; if (!parent) break; active.set(parent, current); current = parent; }
    const firstChild = new Map<string, string>();
    for (const m of c.messages) if (m.parent_id && !firstChild.has(m.parent_id)) firstChild.set(m.parent_id, m.source_id);
    const messages: PrivateChatMessage[] = c.messages.map(m => {
      const own = ids.get(m.source_id)!;
      const files=assetsByMessage.get(m.source_id)??[];
      const contentText=bindImportedFileLinks(m.content,new Map(files.map(a=>[a.source_id,assetId(a)])));
      const message = createPrivateMessage({ chatId, parentMessageId: m.parent_id ? ids.get(m.parent_id)!.id : null, role: m.role, contentText, thinkingText: m.thinking, createdAt: m.created_at });
      const revision = { ...message.active_revision!, id: own.revision };
      const child = active.get(m.source_id) ?? firstChild.get(m.source_id);
      return { ...message, id: own.id, import_source: "chatgpt", active_revision: revision, active_revision_id: revision.id, revisions: [revision], backend_id: null, model_name: m.model,
        active_child_message_id: child ? ids.get(child)!.id : null,
        attachments: files.map(a => ({ id: assetId(a), import_asset_id: assetId(a), message_id: own.id, revision_id: revision.id, original_filename: a.filename, mime_type: "application/octet-stream", size_bytes: a.size_bytes, attachment_kind: "file" })) };
    });
    // Classify only a short prefix; no media content is decoded here.
    for (const message of messages) for (const a of message.attachments) {
      const source = sourceByAsset.get(a.id)!;
      const blob = await archive.asset(source.entry_path); check();
      const type = attachmentType(new Uint8Array(await blob.slice(0, 16).arrayBuffer()));
      a.mime_type = type.mime; a.attachment_kind = type.kind;
      if (a.original_filename === "Attachment") a.original_filename = `Attachment.${type.extension}`;
    }
    const chat: PrivateChatDetail = { id: chatId, title: c.title, default_backend_id: target.backend_id, backend_name: target.backend_name, default_model_name: target.model_name, tool_preferences: { tool_use_enabled: true, tools: {} }, context_blocks: [], active_root_message_id: ids.get(root)!.id, created_at: c.created_at, updated_at: c.updated_at, last_message_at: c.updated_at, do_not_remember: c.do_not_remember, allow_profile_learning: !c.do_not_remember && target.learn_from_import !== false, ...(c.workspace?{imported_workspace:true}:{}) };
    previous = { chat_id: chatId, source_hash: sourceHash, original_json: c.original_json };
    const records=c.workspace?(await import("../canvas/device")).workspaceImportRecords(chatId,c.workspace.title,c.workspace.markdown??c.workspace.content,c.updated_at,c.workspace.markdown?"markdown":"text"):[];
    check(); await commitPrivateImportedChat(key, previous, chat, messages, c.archived,records); check();
  }
  for (const asset of c.assets) {
    const id = assetId(asset); if (await readPrivateImport(id)) continue;
    check(); const blob = await archive.asset(asset.entry_path); check();
    const type = attachmentType(new Uint8Array(await blob.slice(0, 16).arrayBuffer()));
    const count = Math.ceil(blob.size / CHUNK);
    for (let index = 0; index < count; index++) {
      check(); const data = base64(new Uint8Array(await blob.slice(index * CHUNK, (index + 1) * CHUNK).arrayBuffer()));
      await writePrivateImport(`${id}:${index}`, previous.chat_id, data); check();
    }
    await writePrivateImport(id, previous.chat_id, { chunks: count, mime: type.mime, size: blob.size }); check(); onFile();
  }
  return { chat_id: previous.chat_id, existing, changed_source: false, assets: [] };
}

export async function runImport(archive: ArchiveSource, target: ImportTarget, signal: AbortSignal, onProgress: (progress: ImportProgress) => void) {
  const identity = privateProfileIdentity(), guard = privateImageGuard();
  const check = () => { guard(); signal.throwIfAborted(); };
  const progress: ImportProgress = { imported: 0, skipped: 0, completed: 0, files: 0, phase: "conversations", changed: 0 };
  const emit = () => onProgress({ ...progress });
  let c: ImportConversation | null;
  while ((c = await archive.next()) !== null) {
    check(); progress.phase = "conversations"; emit();
    let result: ImportResult;
    if (target.mode === "device") result = await importDevice(c, target, archive, check, () => { progress.files++; progress.phase = "attachments"; emit(); });
    else {
      result = await requestJson<ImportResult>("/api/imports/chatgpt", { method: "POST", signal, expectedServerInstanceId: identity.instance_id, body: JSON.stringify({ expected_user_id: identity.user_id, backend_id: target.backend_id, model_name: target.model_name, allow_history_search: target.allow_history_search, learn_from_import: target.learn_from_import,
        conversation: { ...c, assets: c.assets.map(({ entry_path: _entry, ...asset }) => asset) } }) });
      check();
      if (!result.changed_source) for (const status of result.assets) {
        if (status.complete) continue;
        const asset = c.assets.find(a => a.source_id === status.source_id); if (!asset) throw new Error("Import attachment mapping changed");
        const blob = await archive.asset(asset.entry_path); check();
        const hasher = sha256.create();
        for (let start = 0; start < blob.size; start += CHUNK) { check(); hasher.update(new Uint8Array(await blob.slice(start, start + CHUNK).arrayBuffer())); }
        const hash = bytesToHex(hasher.digest());
        let offset = status.uploaded_bytes;
        progress.phase = "attachments"; emit();
        while (offset < blob.size) {
          check(); const data = base64(new Uint8Array(await blob.slice(offset, offset + CHUNK).arrayBuffer()));
          const next = await requestJson<AssetStatus>(`/api/imports/chatgpt/assets/${status.id}`, { method: "POST", signal, expectedServerInstanceId: identity.instance_id, body: JSON.stringify({ expected_user_id: identity.user_id, offset, data, sha256: hash }) });
          check(); if (next.uploaded_bytes === offset) throw new Error("Attachment upload stalled; retry the import"); offset = next.uploaded_bytes;
        }
        progress.files++; emit();
      }
    }
    if (result.existing) progress.skipped++; else progress.imported++;
    if (result.changed_source) progress.changed++;
    progress.completed++; emit();
  }
  return progress;
}
