import { sourceToken, type ImportConversation } from "./chatgpt";
import { decodePageMarkdown } from "./pageMarkdown";

export const IMPORTED_PAGE_PREFIX = "chatgpt-page-";
export const PAGE_PATH = /(?:^|\/)pages\/([^/]+)\/content\/current\.json$/;
export const MAX_PAGE_RECORD = 2 * 1024 * 1024;
const encoder = new TextEncoder();

export function workspaceCheckpointPath(path: string, original: string) {
  try {
    const id = JSON.parse(original)?.page?.head?.checkpoint_id;
    if (typeof id === "string" && /^[A-Za-z0-9_-]{1,160}$/.test(id)) {
      return path.slice(0, -"current.json".length) + "checkpoints/" + id + ".json";
    }
  } catch { /* The portable Page can still be imported. */ }
  return null;
}

export function retainWorkspaceCheckpoint(conversation: ImportConversation, raw: string) {
  if (encoder.encode(raw).length > MAX_PAGE_RECORD) return;
  try {
    const source = JSON.parse(conversation.original_json), checkpoint = JSON.parse(raw);
    const head = source.page.head;
    if (checkpoint?.checkpoint?.checkpoint_id === head.checkpoint_id &&
        checkpoint?.checkpoint?.page_id === head.page_id &&
        checkpoint?.document?.page_id === head.page_id) {
      conversation.original_json = JSON.stringify({...source,checkpoint});
      const document = checkpoint.document;
      if (conversation.workspace &&
          document.format === "openai.page.yjs" && document.schema_version === 2 &&
          checkpoint.checkpoint.stream_kind === "content" &&
          checkpoint.checkpoint.through_sequence === head.current_sequence &&
          document.through_sequence === head.current_sequence &&
          typeof document.yjs_state_update === "string") {
        const markdown = decodePageMarkdown(document.yjs_state_update);
        if (markdown) conversation.workspace.markdown = markdown;
      }
    }
  } catch { /* Unsupported checkpoints do not affect the portable text. */ }
}

export function normalizeWorkspacePage(path: string, raw: string): ImportConversation | null {
  const exportedId = PAGE_PATH.exec(path)?.[1];
  if (!exportedId || encoder.encode(raw).length > MAX_PAGE_RECORD) return null;
  try {
    const page = JSON.parse(raw), head = page?.head, projection = page?.projection;
    if (head?.page_id !== exportedId || projection?.page_id !== exportedId ||
        head.stream_kind !== "content" || page.includes_current_sequence !== true ||
        !Number.isSafeInteger(head.current_sequence) ||
        projection.projection_through_sequence !== head.current_sequence ||
        typeof projection.title !== "string" || typeof projection.materialized_search_text !== "string") return null;
    const title = projection.title.trim() || "Untitled imported page", content = projection.materialized_search_text;
    if (/[\x00-\x1f\x7f-\x9f]/.test(title) || encoder.encode(title).length > 200 ||
        !content.trim() || encoder.encode(content).length > 256 * 1024) return null;
    const source_id = IMPORTED_PAGE_PREFIX + sourceToken(exportedId);
    const unix = (value: unknown) => typeof value === "string"
      ? Math.floor(Date.parse(value) / 1000) : NaN;
    const edited = unix(projection.updated_at), checkpoint = unix(head.last_checkpoint_at);
    const valid = (value: number) => Number.isSafeInteger(value) && value >= 0 && value <= 32_503_680_000;
    const updated = valid(edited) ? edited : valid(checkpoint) ? checkpoint : 0;
    return {
      source_id, title, created_at: updated, updated_at: updated, archived: true,
      do_not_remember: true, current_node_id: "imported-page", assets: [],
      messages: [{ source_id:"imported-page", parent_id:null, role:"assistant",
        content:"Imported ChatGPT Page. Open Canvas to edit its text.",
        thinking:"",model:null,created_at:updated }],
      original_json: JSON.stringify({id:source_id,page}),
      workspace: {title,content},
    };
  } catch {
    return null;
  }
}
