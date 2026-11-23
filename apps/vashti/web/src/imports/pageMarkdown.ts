import * as Y from "yjs";

const MAX_MARKDOWN_BYTES = 256 * 1024;
const MAX_BLOCKS = 4_096;

export function decodePageMarkdown(update: string): string | undefined {
  const doc = new Y.Doc();
  try {
    // This decoder handles full schema-2 snapshots only, not arbitrary edit deltas.
    if (update.length > 512 * 1024) return;
    const base64 = update.replace(/-/g, "+").replace(/_/g, "/");
    Y.applyUpdate(doc, Uint8Array.from(atob(base64), c => c.charCodeAt(0)));
    if (doc.store.pendingStructs || doc.store.pendingDs) return;
    const blocks = doc.getMap("blocksById");
    const order = doc.getArray("topLevelBlockIds");
    if (!order.length || order.length > MAX_BLOCKS || order.length !== blocks.size) return;
    const ids = order.toArray();
    if (new Set(ids).size !== ids.length) return;
    const markdown: string[] = [];
    let bytes = 0;
    const encoder = new TextEncoder();
    for (const id of ids) {
      const block = typeof id === "string" ? blocks.get(id) : undefined;
      if (!(block instanceof Y.Map) || block.get("kind") !== "markdown") return;
      const value = block.get("markdown");
      const text = value instanceof Y.Text ? value.toString() : value;
      if (typeof text !== "string") return;
      bytes += encoder.encode(text).length + 2;
      if (bytes > MAX_MARKDOWN_BYTES) return;
      markdown.push(text);
    }
    return markdown.join("\n\n").trim() || undefined;
  } catch {
    return;
  } finally {
    doc.destroy();
  }
}
