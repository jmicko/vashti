import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import { fromMarkdown } from "mdast-util-from-markdown";
import { importedSourceLink } from "./fileLinks";

export const MAX_ASSET_BYTES = 128 * 1024 * 1024;
export const MAX_CONVERSATION_BYTES = 16 * 1024 * 1024;
export type ImportWarnings = Record<string, number>;
export type ImportMessage = { source_id: string; parent_id: string | null; role: "user" | "assistant"; content: string; thinking: string; model: string | null; created_at: number };
export type ImportAsset = { source_id: string; message_id: string; filename: string; size_bytes: number; entry_path: string };
export type ImportWorkspace = { title: string; content: string; markdown?: string };
export type ImportConversation = { source_id: string; title: string; created_at: number; updated_at: number; archived: boolean; do_not_remember: boolean; current_node_id: string | null; messages: ImportMessage[]; assets: ImportAsset[]; original_json: string; workspace?: ImportWorkspace };
export type ArchiveSummary = { conversations: number; messages: number; assets: number; asset_bytes: number; branches: number; workspace_pages:number; warnings: ImportWarnings };
export type FileIndex = Map<string, { filename: string; size: number; originalName?: string }>;
type RecordValue = Record<string, any>;
const record = (v: unknown): RecordValue => v !== null && typeof v === "object" && !Array.isArray(v) ? v as RecordValue : {};
const text = (v: unknown) => typeof v === "string" ? v : "";
const note = (warnings: ImportWarnings, key: string) => { warnings[key] = (warnings[key] ?? 0) + 1; };
const timestamp = (v: unknown, fallback: number) => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 32_503_680_000 ? Math.floor(v) : fallback;
const encoder = new TextEncoder();
const validId = (value: string) => !!value && encoder.encode(value).length <= 256 && !/[\x00-\x1f\x7f-\x9f]/.test(value);
export const sourceToken = (value: string) => bytesToHex(sha256(new TextEncoder().encode(value)));

export function safeEntryPath(path: string) {
  return path.length <= 2048 && !path.startsWith("/") && !/^[A-Za-z]:/.test(path)
    && !/[\\\x00-\x1f]/.test(path) && !path.split("/").some(p => p === "." || p === "..");
}

function urls(value: unknown, found = new Set<string>(), depth = 0): Set<string> {
  if (depth > 12 || found.size >= 40) return found;
  if (Array.isArray(value)) for (const item of value.slice(0, 200)) urls(item, found, depth + 1);
  else if (value && typeof value === "object") for (const [key, item] of Object.entries(value)) {
    if (key === "url" && typeof item === "string") {
      try { const u = new URL(item); if (["http:", "https:"].includes(u.protocol) && !u.username && !u.password && item.length <= 2048) found.add(u.href); } catch { /* Invalid exported references stay in the original record. */ }
    } else if (typeof item === "object") urls(item, found, depth + 1);
  }
  return found;
}

export function portableMarkdown(value: string, metadata: unknown, warnings: ImportWarnings, resolveFile?: (url: string) => string | undefined) {
  const clean = (part: string) => part
      .replace(/[\ue200\ue203][^\ue201\ue204]*[\ue201\ue204]/g, () => { note(warnings, "rich_references"); return ""; });
  let content = value;
  if (/[\ue200\ue203]|sandbox:/.test(value)) {
    type Node = { type: string; url?: string; value?: string; alt?: string | null; position?: { start: { offset?: number }; end: { offset?: number } }; children?: Node[] };
    const pending: Node[] = [fromMarkdown(value)], replacements: [number, number, string][] = [];
    while (pending.length) {
      const node = pending.pop()!;
      const start = node.position?.start.offset, end = node.position?.end.offset;
      if (start !== undefined && end !== undefined && ["code", "inlineCode"].includes(node.type)) {
        replacements.push([start,end,value.slice(start,end)]);
      } else if (start !== undefined && end !== undefined && node.url?.startsWith("sandbox:") && ["link","image","definition"].includes(node.type)) {
        const file = resolveFile?.(node.url);
        if (!file) note(warnings, "sandbox_links");
        else note(warnings, "restored_file_links");
        if (node.type === "definition") {
          const raw = value.slice(start,end);
          replacements.push([start,end,file ? raw.slice(0,raw.indexOf("]:")+2)+" "+file : ""]);
        } else {
          const label = (node.alt || node.children?.map(c=>c.value ?? "").join("") || "File").replace(/[\[\]\\]/g,"\\$&").replace(/[\r\n]/g," ");
          replacements.push([start,end,file ? `[${label}](${file})` : `${label} (file not available in the export)`]);
        }
      } else if (node.children) for (const child of node.children) pending.push(child);
    }
    const parts: string[] = []; let offset = 0;
    for (const [start, end, replacement] of replacements.sort((a, b) => a[0] - b[0])) { parts.push(clean(value.slice(offset, start)), replacement); offset = end; }
    parts.push(clean(value.slice(offset))); content = parts.join("");
  }
  const references = [...urls(metadata)];
  return content + (references.length ? `\n\n### Sources\n${references.map(url => `- <${url.replace(/>/g, "%3E").replace(/</g, "%3C")}>`).join("\n")}` : "");
}

function thinking(content: RecordValue) {
  if (content.content_type === "reasoning_recap") return text(content.content);
  return (Array.isArray(content.thoughts) ? content.thoughts : []).map((item: unknown) => {
    const t = record(item);
    return text(t.content) || text(t.summary) || (Array.isArray(t.chunks) ? t.chunks.map((c: unknown) => text(c) || text(record(c).content)).join("\n") : "");
  }).filter(Boolean).join("\n\n");
}

export function normalizeConversation(input: unknown, index: FileIndex, originalJson: string, warnings: ImportWarnings = {}): ImportConversation {
  const c = record(input), mapping = record(c.mapping), entries = Object.entries(mapping);
  const sourceId = text(c.conversation_id) || text(c.id);
  if (!validId(sourceId) || !entries.length || entries.length > 20_000 || encoder.encode(originalJson).length > MAX_CONVERSATION_BYTES) throw new Error("Invalid or oversized conversation graph");
  const raw = new Map(entries.map(([id, node]) => [id, record(node)]));
  const children = new Map<string, string[]>();
  for (const [id, n] of raw) if (typeof n.parent === "string") { const siblings = children.get(n.parent) ?? []; siblings.push(id); children.set(n.parent, siblings); }
  const isThought = (n: RecordValue) => ["thoughts", "reasoning_recap"].includes(record(record(n.message).content).content_type) || record(n.message).channel === "analysis";
  const visible = new Set<string>();
  for (const [id, n] of raw) {
    const m = record(n.message), meta = record(m.metadata), role = record(m.author).role;
    if (!n.message || meta.is_visually_hidden_from_conversation || role === "system" || role === "developer") { if (n.message) note(warnings, "hidden_messages"); continue; }
    if (isThought(n) && children.get(id)?.length) continue;
    if (!["user", "assistant", "tool", "function"].includes(role)) { note(warnings, "unsupported_messages"); continue; }
    visible.add(id);
  }
  const parentOf = new Map<string, string | null>();
  const skippedThoughts = new Map<string, string>();
  // Iterative ancestry walks avoid overflowing the stack on long exports.
  for (const id of visible) {
    const seen = new Set([id]); let parent = raw.get(id)?.parent; const thoughts: string[] = [];
    while (typeof parent === "string") {
      if (seen.has(parent)) throw new Error("Cyclic conversation graph");
      seen.add(parent);
      const node = raw.get(parent);
      if (!node) { note(warnings, "missing_parents"); parent = null; break; }
      if (visible.has(parent)) break;
      if (isThought(node)) thoughts.unshift(thinking(record(record(node.message).content)) || text(record(node.message).content?.parts?.join?.("\n")));
      parent = node.parent;
    }
    parentOf.set(id, typeof parent === "string" ? parent : null);
    skippedThoughts.set(id, thoughts.filter(Boolean).join("\n\n"));
  }
  const ordered: string[] = [], added = new Set<string>();
  for (const id of visible) {
    const chain: string[] = [], seen = new Set<string>(); let next: string | null = id;
    while (next && !added.has(next)) { if (seen.has(next)) throw new Error("Cyclic conversation graph"); seen.add(next); chain.push(next); next = parentOf.get(next) ?? null; }
    for (const part of chain.reverse()) { added.add(part); ordered.push(part); }
  }
  if (!ordered.length || ordered.length > 10_000) throw new Error("No supported conversation messages");
  const assets: ImportAsset[] = [];
  const conversationFiles = new Map<string, Set<string>>();
  const zipEntry = (pointer: string) => {
    const candidate = pointer.replace(/^[A-Za-z-]+:\/\//, "");
    return index.get(candidate) ?? index.get(candidate + ".dat") ?? index.get(candidate.replace(/^file_/, "file-") + ".dat");
  };
  function rememberFile(pointer:string, label:string) {
    const entry=zipEntry(pointer), name=(label || entry?.originalName || "").split(/[\\/]/).pop()!;
    if(!entry || !name)return;
    const matches=conversationFiles.get(name)??new Set<string>();
    matches.add(entry.filename); conversationFiles.set(name,matches);
  }
  for (const id of ordered) {
    const m=record(raw.get(id)?.message), meta=record(m.metadata), content=record(m.content);
    for (const a of Array.isArray(meta.attachments)?meta.attachments:[]) if(typeof a?.id==="string") rememberFile(a.id,text(a.name));
    for (const r of Array.isArray(meta.content_references)?meta.content_references:[]) if(r?.type==="file"&&typeof r.id==="string") rememberFile(r.id,text(r.name));
    for (const p of Array.isArray(content.parts)?content.parts:[]) if(typeof p?.asset_pointer==="string") rememberFile(p.asset_pointer,"");
  }
  const created = timestamp(c.create_time, 0), updated = timestamp(c.update_time, created);
  const messages = ordered.map(id => {
    const m = record(raw.get(id)?.message), content = record(m.content), meta = record(m.metadata);
    const originalRole = record(m.author).role, role = originalRole === "user" ? "user" : "assistant";
    const attachmentNames = new Map<string, string>();
    for (const attachment of Array.isArray(meta.attachments) ? meta.attachments : []) {
      if (typeof attachment?.id === "string") attachmentNames.set(attachment.id, text(attachment.name));
    }
    const attached = new Map<string, ImportAsset>();
    function attach(pointer: string, name = "", countMissing=true) {
      const candidate = pointer.replace(/^[A-Za-z-]+:\/\//, "");
      const entry = zipEntry(pointer);
      if (!entry) { if(countMissing)note(warnings, "missing_files"); return; }
      if (attached.has(entry.filename)) return attached.get(entry.filename);
      if (entry.size > MAX_ASSET_BYTES || !entry.size) { note(warnings, "oversized_files"); return; }
      const originalName = (name || attachmentNames.get(candidate) || entry.originalName || "Attachment").split(/[\\/]/).pop()!.replace(/[\x00-\x1f\x7f-\x9f]/g, "_").slice(0, 160);
      const asset={ source_id: "asset-" + sourceToken(id + "\0" + entry.filename), message_id: id, filename: originalName || "Attachment", size_bytes: entry.size, entry_path: entry.filename };
      assets.push(asset);attached.set(entry.filename,asset);
      return asset;
    }
    const parts: string[] = [];
    for (const part of Array.isArray(content.parts) ? content.parts : []) {
      if (typeof part === "string") { parts.push(part); continue; }
      const p = record(part);
      if (typeof p.asset_pointer === "string") { if (!attach(p.asset_pointer)) parts.push("[Attachment not included in the export]"); }
      else if (p.content_type === "audio_transcription") parts.push(text(p.text));
      else if (p.content_type === "real_time_user_audio_video_asset_pointer") {
        for (const key of ["audio_asset_pointer", "video_container_asset_pointer"]) {
          const v = p[key]; if (typeof v === "string") attach(v); else if (record(v).asset_pointer) attach(record(v).asset_pointer);
        }
        if (Array.isArray(p.frames_asset_pointers)) for (const f of p.frames_asset_pointers) if (typeof f?.asset_pointer === "string") attach(f.asset_pointer);
      } else { note(warnings, "unsupported_parts"); if (typeof p.text === "string") parts.push(p.text); }
    }
    for (const [pointer, name] of attachmentNames) attach(pointer, name);
    const fileRefs = Array.isArray(meta.content_references) ? meta.content_references.filter((r:RecordValue|null)=>r?.type==="file" && typeof r.id==="string") : [];
    for (const ref of fileRefs) attach(ref.id,text(ref.name),false);
    function resolveFile(url:string) {
      const path=url.slice("sandbox:".length);
      if(!path.startsWith("/mnt/data/") || /[?#\\]/.test(path))return;
      let decoded: string;
      try{decoded=decodeURIComponent(path.slice("/mnt/data/".length));}catch{return;}
      if(!decoded || !safeEntryPath(decoded) || /[?#]/.test(decoded))return;
      const name=decoded.split("/").pop()!;
      const direct=new Set<string>();
      for(const ref of fileRefs) {
        if(typeof ref.name==="string"&&ref.name.split(/[\\/]/).pop()===name){
          const entry=zipEntry(ref.id);if(entry)direct.add(entry.filename);
        }
      }
      for(const [pointer,label] of attachmentNames){if(label.split(/[\\/]/).pop()===name){const entry=zipEntry(pointer);if(entry)direct.add(entry.filename);}}
      const exact=index.get(path)??index.get(path.slice(1));
      const matches=direct.size?direct:exact?new Set([exact.filename]):conversationFiles.get(name);
      if(matches?.size!==1)return;
      const asset=attach([...matches][0],name,false);
      if(asset)return importedSourceLink(asset.source_id);
    }
    let result = parts.join("\n\n") || text(content.text) || (!isThought(raw.get(id)!) ? text(content.content) : "");
    if (originalRole === "tool" || originalRole === "function") { note(warnings, "tool_transcripts"); result = "Imported tool output (reference only):\n\n" + result; }
    if (!result && !assets.some(a => a.message_id === id) && !isThought(raw.get(id)!)) { result = "[Content retained in the original conversation record]"; note(warnings, "unsupported_messages"); }
    const ownThinking = isThought(raw.get(id)!) ? thinking(content) || parts.join("\n\n") : "";
    const thoughtText = [skippedThoughts.get(id), ownThinking].filter(Boolean).join("\n\n");
    return { source_id: id, parent_id: parentOf.get(id) ?? null, role, content: portableMarkdown(result, meta.content_references, warnings,resolveFile), thinking: thoughtText, model: text(meta.model_slug) || text(meta.resolved_model_slug) || (role === "assistant" ? text(c.default_model_slug) || "ChatGPT" : null), created_at: timestamp(m.create_time, created) } satisfies ImportMessage;
  });
  const seenCurrent = new Set<string>(); let current = text(c.current_node);
  while (current && !visible.has(current)) { if (seenCurrent.has(current)) throw new Error("Cyclic active path"); seenCurrent.add(current); current = text(raw.get(current)?.parent); }
  const normalized = { source_id: sourceId, title: text(c.title) || "Imported conversation", created_at: created, updated_at: updated, archived: c.is_archived === true, do_not_remember: c.is_do_not_remember === true, current_node_id: current || ordered[ordered.length - 1], messages, assets, original_json: originalJson };
  if (encoder.encode(normalized.title).length > 4096 || assets.length > 5000 || messages.some(m => !validId(m.source_id)
    || encoder.encode(m.content).length + encoder.encode(m.thinking).length > 2 * 1024 * 1024
    || (m.model !== null && (encoder.encode(m.model).length > 256 || /[\x00-\x1f\x7f-\x9f]/.test(m.model))))
    || encoder.encode(JSON.stringify(normalized)).length > 31 * 1024 * 1024) throw new Error("Conversation exceeds the import safety limits");
  return normalized;
}

export const warningLabels: Record<string, string> = {
  unreferenced_files: "Library files not linked to an imported message (not imported)", workspace_documents: "Workspace documents without a supported current text record",
  missing_files: "File references not included in the ZIP", oversized_files: "Files outside the 128 MiB attachment limit",
  unsupported_parts: "Special content preserved in original records", unsupported_messages: "Messages needing a plain-text fallback",
  hidden_messages: "Hidden/system messages retained only in original records", missing_parents: "Missing parent links repaired",
  rich_references: "ChatGPT widgets/citations converted to portable text and source links", sandbox_links: "File links with missing or ambiguous targets",
  restored_file_links: "File links that can be restored",
  tool_transcripts: "Tool transcripts imported as reference text", invalid_conversations: "Invalid or empty conversations skipped"
};
