export type CanvasMode = "server" | "device";
export type CanvasScope = {
  message_id: string | null;
  revision_id: string | null;
};
export type CanvasLanguage =
  | "markdown"
  | "text"
  | "javascript"
  | "typescript"
  | "python"
  | "rust"
  | "html"
  | "css"
  | "json"
  | "sql";
export const canvasLanguages: CanvasLanguage[] = [
  "markdown",
  "text",
  "javascript",
  "typescript",
  "python",
  "rust",
  "html",
  "css",
  "json",
  "sql",
];
export type CanvasVersion = {
  id: string;
  document_id: string;
  parent_id: string | null;
  version_number: number;
  title: string;
  language: CanvasLanguage;
  content: string;
  actor: "human" | "model";
  model_name: string | null;
  edit_session_id: string | null;
  created_at: number;
  updated_at: number;
};
export type CanvasVersionSummary = Omit<
  CanvasVersion,
  "content" | "updated_at" | "edit_session_id"
>;
export type CanvasDocument = {
  id: string;
  trashed: boolean;
  version: CanvasVersion;
};
export type CanvasComment = {
  id: string;
  document_id: string;
  version_id: string;
  quote: string;
  content: string;
  actor: "human" | "model";
  model_name: string | null;
  resolved: boolean;
  created_at: number;
};
export type CanvasWorkspace = {
  scope_id: string;
  documents: CanvasDocument[];
  comments: CanvasComment[];
};
export type CanvasHead = {
  document_id: string;
  version_id: string;
  trashed: boolean;
};
export type CanvasReference = {
  document_id: string;
  version_id: string;
  title: string;
  version_number: number;
  language: CanvasLanguage;
};
export type CanvasMutation =
  | {
      operation: "create";
      document_id?: string;
      title: string;
      language: CanvasLanguage;
      content: string;
    }
  | {
      operation: "update";
      document_id: string;
      expected_version_id: string;
      title?: string;
      language?: CanvasLanguage;
      content?: string;
      edits?: { find: string; replace: string }[];
      edit_session_id?: string;
    }
  | {
      operation: "restore";
      document_id: string;
      expected_version_id: string;
      version_id: string;
    }
  | {
      operation: "trash";
      document_id: string;
      expected_version_id: string;
      trashed: boolean;
    }
  | {
      operation: "comment";
      document_id: string;
      expected_version_id: string;
      quote: string;
      content: string;
    }
  | { operation: "resolve_comment"; comment_id: string; resolved: boolean };
export const MAX_CANVAS_BYTES = 256 * 1024;
export const CANVAS_TOOL_NAMES = [
  "list_canvases",
  "read_canvas",
  "create_canvas",
  "update_canvas",
  "comment_canvas",
] as const;
export type CanvasToolName = (typeof CANVAS_TOOL_NAMES)[number];
export const canvasScopeKey = (chat: string, scope: CanvasScope) =>
  `${chat}:${scope.revision_id ?? "root"}`;
export const canvasSummary = (v: CanvasVersion): CanvasReference => ({
  document_id: v.document_id,
  version_id: v.id,
  title: v.title,
  language: v.language,
  version_number: v.version_number,
});

export function validateCanvas(
  title: string,
  language: string,
  content: string,
) {
  const bytes = new TextEncoder();
  if (
    !title.trim() ||
    bytes.encode(title).length > 200 ||
    /[\x00-\x1f\x7f-\x9f]/.test(title) ||
    !canvasLanguages.includes(language as CanvasLanguage) ||
    bytes.encode(content).length > MAX_CANVAS_BYTES
  )
    throw new Error(
      "Use a title up to 200 bytes, a supported language, and content up to 256 KiB",
    );
}
export function exactCanvasEdits(
  content: string,
  edits: { find: string; replace: string }[],
) {
  if (!edits.length || edits.length > 24)
    throw new Error("Provide between one and 24 exact edits");
  for (const edit of edits) {
    if (
      !edit.find ||
      new TextEncoder().encode(edit.find).length > MAX_CANVAS_BYTES ||
      new TextEncoder().encode(edit.replace).length > MAX_CANVAS_BYTES ||
      content.indexOf(edit.find) < 0 ||
      content.indexOf(edit.find) !== content.lastIndexOf(edit.find)
    )
      throw new Error(
        "Each edit must match exactly once. Read the canvas again or include more surrounding text.",
      );
    content = content.replace(edit.find, () => edit.replace);
    if (new TextEncoder().encode(content).length > MAX_CANVAS_BYTES)
      throw new Error("Canvas exceeds 256 KiB");
  }
  return content;
}
export function canvasReadResult(
  version: CanvasVersion,
  args: Record<string, unknown>,
) {
  const characters = Array.from(version.content),
    offset =
      typeof args.offset === "number" &&
      Number.isSafeInteger(args.offset) &&
      args.offset >= 0
        ? Math.min(args.offset, MAX_CANVAS_BYTES, characters.length)
        : 0;
  let content = "",
    next = Math.min(offset, characters.length),
    size = 0,
    encoded = 0;
  for (let i = offset; i < characters.length; i++) {
    const ch = characters[i],
      bytes = new TextEncoder().encode(ch).length,
      quoted = JSON.stringify(ch).length - 2;
    if (size + bytes > 12000 || encoded + quoted > 12000) break;
    content += ch;
    size += bytes;
    encoded += quoted;
    next++;
  }
  return {
    canvas: canvasSummary(version),
    offset,
    next_offset: next,
    total_characters: characters.length,
    content,
    truncated: next < characters.length,
  };
}
