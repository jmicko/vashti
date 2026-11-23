// Native-stream and JS codecs also work under Android's restrictive CSP.
import { BlobReader, ZipReader, type FileEntry } from "@zip.js/zip.js/lib/zip-core-native.js";
import { MAX_ASSET_BYTES, MAX_CONVERSATION_BYTES, normalizeConversation, safeEntryPath, type ArchiveSummary, type FileIndex, type ImportConversation } from "./chatgpt";
import { MAX_PAGE_RECORD, normalizeWorkspacePage, PAGE_PATH, retainWorkspaceCheckpoint, workspaceCheckpointPath } from "./workspace";

const MAX_ENTRIES = 100_000, MAX_ZIP_BYTES = 4 * 1024 ** 3, MAX_JSON_BYTES = 1024 ** 3;
export class ChatGptArchive {
  private reader: ZipReader<Blob>;
  private entries = new Map<string, FileEntry>();
  private index: FileIndex = new Map();
  private documents: FileEntry[] = [];
  private pages: FileEntry[] = [];
  private iterator: AsyncGenerator<ImportConversation> | null = null;
  constructor(file: Blob) {
    if (!file.size || file.size > MAX_ZIP_BYTES) throw new Error("Choose a ZIP smaller than 4 GiB");
    this.reader = new ZipReader(new BlobReader(file), { useWebWorkers: false, checkCrc32: true });
  }
  async open(onProgress: (count: number) => void = () => {}) {
    let expanded = 0;
    const basenames = new Map<string, FileEntry | null>(), names = new Set<string>();
    for await (const e of this.reader.getEntriesGenerator()) {
      if (!safeEntryPath(e.filename) || names.has(e.filename) || e.encrypted || ((e.unixMode ?? 0) & 0o170000) === 0o120000) throw new Error("The ZIP contains unsafe, duplicate, linked, or encrypted entries");
      names.add(e.filename);
      if (names.size > MAX_ENTRIES) throw new Error("The ZIP contains too many entries");
      if (e.directory) continue;
      if (this.entries.size >= MAX_ENTRIES || !Number.isSafeInteger(e.uncompressedSize) || e.uncompressedSize < 0) throw new Error("The ZIP contains too many or invalid entries");
      expanded += e.uncompressedSize;
      if (expanded > 8 * 1024 ** 3) throw new Error("The ZIP expands beyond the 8 GiB safety limit");
      this.entries.set(e.filename, e);
      const base = e.filename.split("/").pop()!;
      basenames.set(base, basenames.has(base) ? null : e);
      if (/^conversations(?:-\d+)?\.json$/i.test(base)) this.documents.push(e);
      if (PAGE_PATH.test(e.filename) && this.pages.length < 200) this.pages.push(e);
    }
    for (const [path, e] of this.entries) this.index.set(path, { filename: path, size: e.uncompressedSize });
    for (const [base, e] of basenames) if (e) this.index.set(base, { filename: e.filename, size: e.uncompressedSize });
    const namesFile = basenames.get("conversation_asset_file_names.json");
    if (namesFile && namesFile.uncompressedSize <= 2 * 1024 * 1024) {
      const names = JSON.parse(await (await this.asset(namesFile.filename)).text());
      if (names && typeof names === "object" && !Array.isArray(names)) for (const [path, name] of Object.entries(names)) {
        const entry = this.index.get(path);
        if (entry && typeof name === "string") { entry.originalName = name; const full = this.index.get(entry.filename); if (full) full.originalName = name; }
      }
    }
    this.documents.sort((a, b) => a.filename.localeCompare(b.filename, undefined, { numeric: true }));
    if (!this.documents.length && !this.pages.length) throw new Error("The ZIP has no supported conversations or workspace Pages");
    const summary: ArchiveSummary = { conversations: 0, messages: 0, assets: 0, asset_bytes: 0, branches: 0, workspace_pages:0, warnings: {} };
    let jsonBytes = 0; const ids = new Set<string>(), referencedFiles = new Set<string>();
    for (const doc of this.documents) {
      jsonBytes += doc.uncompressedSize;
      if (jsonBytes > MAX_JSON_BYTES) throw new Error("Conversation JSON exceeds the 1 GiB safety limit");
      for await (const original of this.conversations(doc)) {
        try {
          const c = normalizeConversation(JSON.parse(original), this.index, original, summary.warnings);
          if (ids.has(c.source_id)) continue;
          ids.add(c.source_id); summary.conversations++; summary.messages += c.messages.length; summary.assets += c.assets.length; summary.asset_bytes += c.assets.reduce((sum, a) => sum + a.size_bytes, 0);
          for (const asset of c.assets) referencedFiles.add(asset.entry_path);
          const parents = new Set<string>(); for (const m of c.messages) if (m.parent_id) { if (parents.has(m.parent_id)) summary.branches++; parents.add(m.parent_id); }
          if (summary.conversations > 50_000 || summary.messages > 500_000) throw new Error("Import is too large");
          onProgress(summary.conversations);
        } catch (error) {
          if (error instanceof Error && error.message === "Import is too large") throw error;
          summary.warnings.invalid_conversations = (summary.warnings.invalid_conversations ?? 0) + 1;
        }
      }
    }
    for await(const page of this.workspacePages(false)) {
      if (!ids.has(page.source_id)) { ids.add(page.source_id); summary.workspace_pages++; }
    }
    if (!summary.conversations && !summary.workspace_pages) throw new Error("No supported conversations or workspace Pages were found");
    const unused = [...this.entries.keys()].filter(path => path.endsWith(".dat") && !referencedFiles.has(path)).length;
    if (unused) summary.warnings.unreferenced_files = unused;
    const documents = [...this.entries.keys()].filter(path => path.endsWith("/current.json")).length - summary.workspace_pages;
    if (documents) summary.warnings.workspace_documents = documents;
    if (summary.asset_bytes > 2 * 1024 ** 3) throw new Error("Imported attachments exceed the 2 GiB safety limit");
    return summary;
  }
  private async *workspacePages(retainCheckpoint: boolean) {
    for (const entry of this.pages) {
      if (entry.uncompressedSize > MAX_PAGE_RECORD) continue;
      const page=normalizeWorkspacePage(entry.filename, await (await this.asset(entry.filename,MAX_PAGE_RECORD)).text());
      if(!page)continue;
      if(retainCheckpoint){
        const path=workspaceCheckpointPath(entry.filename,page.original_json);
        const checkpoint=path?this.entries.get(path):undefined;
        if(checkpoint&&checkpoint.uncompressedSize<=MAX_PAGE_RECORD)
          retainWorkspaceCheckpoint(page,await (await this.asset(checkpoint.filename,MAX_PAGE_RECORD)).text());
      }
      yield page;
    }
  }
  private async *conversations(entry: FileEntry) {
    if (entry.uncompressedSize > MAX_JSON_BYTES) throw new Error("Conversation JSON exceeds the safety limit");
    const stream = new TransformStream<Uint8Array, Uint8Array>();
    const reading = entry.getData(stream.writable).then(() => null, () => new Error("Could not read conversation JSON; the ZIP may be damaged"));
    const reader = stream.readable.getReader(), decoder = new TextDecoder("utf-8", { fatal: true });
    let buffer = "", depth = 0, inString = false, escaped = false, started = false, closed = false, total = 0;
    let separator = false, needsComma = false;
    try {
      while (true) {
        const { value, done } = await reader.read();
        total += value?.length ?? 0;
        if (total > MAX_JSON_BYTES || total > entry.uncompressedSize) throw new Error("Conversation JSON exceeds the safety limit");
        const chunk = decoder.decode(value, { stream: !done });
        for (const ch of chunk) {
          if (!started) { if (/\s|\uFEFF/.test(ch)) continue; if (ch !== "[") throw new Error("Conversations must be a JSON array"); started = true; continue; }
          if (!depth) {
            if (/\s/.test(ch)) continue;
            if (closed) throw new Error("Unexpected data after conversations");
            if (ch === "]") { if (separator) throw new Error("Invalid conversation separator"); closed = true; continue; }
            if (ch === ",") { if (separator || !needsComma) throw new Error("Invalid conversation separator"); separator = true; needsComma = false; continue; }
            if (ch !== "{" || needsComma) throw new Error("Invalid conversation JSON");
            depth = 1; buffer = "{"; inString = false; escaped = false; separator = false; continue;
          }
          buffer += ch;
          if (buffer.length > MAX_CONVERSATION_BYTES) throw new Error("A conversation exceeds the 16 MiB safety limit");
          if (inString) { if (escaped) escaped = false; else if (ch === "\\") escaped = true; else if (ch === '"') inString = false; }
          else if (ch === '"') inString = true;
          else if (ch === "{" || ch === "[") depth++;
          else if (ch === "}" || ch === "]") { depth--; if (!depth) { needsComma = true; yield buffer; buffer = ""; } }
        }
        if (done) break;
      }
      const error = await reading; if (error) throw error;
      if (!closed || depth || inString) throw new Error("Conversation JSON is incomplete");
    } finally { await reader.cancel().catch(() => undefined); await reading; }
  }
  private async *iterate() {
    const seen = new Set<string>();
    for (const doc of this.documents) for await (const original of this.conversations(doc)) {
      let c: ImportConversation;
      try { c = normalizeConversation(JSON.parse(original), this.index, original); } catch { continue; }
      if (seen.has(c.source_id)) continue;
      seen.add(c.source_id); yield c;
    }
    for await(const page of this.workspacePages(true)){
      if(!seen.has(page.source_id)){seen.add(page.source_id);yield page;}
    }
  }
  async next() { this.iterator ??= this.iterate(); const result = await this.iterator.next(); return result.done ? null : result.value; }
  async asset(path: string, maxBytes = MAX_ASSET_BYTES) {
    const entry = this.entries.get(path);
    if (!entry || entry.uncompressedSize > maxBytes) throw new Error("Attachment exceeds the safety limit");
    let size = 0; const parts: Uint8Array<ArrayBuffer>[] = [];
    await entry.getData(new WritableStream<Uint8Array>({ write(chunk) {
      size += chunk.length;
      if (size > maxBytes || size > entry.uncompressedSize) throw new Error("Attachment expands beyond its limit");
      parts.push(new Uint8Array(chunk));
    }}));
    if (size !== entry.uncompressedSize) throw new Error("Attachment size did not match");
    return new Blob(parts);
  }
  async close() { await this.iterator?.return(undefined); await this.reader.close(); }
}
