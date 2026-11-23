import { requestJson } from "../api";
import { getPrivateChat, privateImageGuard } from "../privateChatStore";
import { toolPreferenceEnabled } from "../toolPreferences";
import { canvasChanged } from "./repository";
import { deviceCanvasWorkspace, mutateDeviceCanvas } from "./device";
import {
  canvasReadResult,
  canvasSummary,
  type CanvasLanguage,
  type CanvasMutation,
  type CanvasScope,
  type CanvasToolName,
} from "./types";

function string(args: Record<string, unknown>, key: string) {
  if (typeof args[key] !== "string") throw new Error(`Canvas requires ${key}`);
  return args[key] as string;
}
function keys(args: Record<string, unknown>, allowed: string[]) {
  if (
    !args ||
    typeof args !== "object" ||
    Array.isArray(args) ||
    Object.keys(args).some((key) => !allowed.includes(key))
  )
    throw new Error("Unexpected canvas tool argument");
}
export async function executeDeviceCanvasTool(
  name: CanvasToolName,
  args: Record<string, unknown>,
  chat: string,
  scope: CanvasScope,
  model: string,
  check: () => void,
) {
  const allowed: Record<CanvasToolName, string[]> = {
    list_canvases: [],
    read_canvas: ["document_id", "offset"],
    create_canvas: ["title", "language", "content"],
    update_canvas: [
      "document_id",
      "expected_version_id",
      "title",
      "language",
      "content",
      "edits",
    ],
    comment_canvas: ["document_id", "expected_version_id", "quote", "content"],
  };
  keys(args, allowed[name]);
  const guard = privateImageGuard(),
    catalog = await requestJson<{
      tools_enabled: boolean;
      device_tools: { id: string }[];
    }>("/api/tools");
  guard();
  check();
  const current = await getPrivateChat(chat);
  guard();
  if (
    !catalog.tools_enabled ||
    !catalog.device_tools.some((t) => t.id === "canvas") ||
    !current?.tool_preferences.tool_use_enabled ||
    !toolPreferenceEnabled(current.tool_preferences, "canvas")
  )
    throw new Error("Canvas tools are unavailable for this chat");
  const workspace = await deviceCanvasWorkspace(chat, scope);
  guard();
  check();
  if (name === "list_canvases")
    return {
      canvases: workspace.documents
        .filter((d) => !d.trashed)
        .map((d) => canvasSummary(d.version)),
    };
  if (name === "read_canvas") {
    const doc = workspace.documents.find(
      (d) => d.id === string(args, "document_id") && !d.trashed,
    );
    if (!doc) throw new Error("Canvas not found in this branch");
    const comments = workspace.comments.filter(
      (c) =>
        c.document_id === doc.id &&
        c.version_id === doc.version.id &&
        !c.resolved,
    );
    return {
      ...canvasReadResult(doc.version, args),
      comments_truncated: comments.length > 3,
      comments: comments.slice(0, 3).map((c) => ({
        id: c.id,
        quote: Array.from(c.quote).slice(0, 200).join(""),
        content: Array.from(c.content).slice(0, 400).join(""),
      })),
    };
  }
  let mutation: CanvasMutation;
  if (name === "create_canvas")
    mutation = {
      operation: "create",
      title: string(args, "title"),
      language: string(args, "language") as CanvasLanguage,
      content: string(args, "content"),
    };
  else if (name === "comment_canvas")
    mutation = {
      operation: "comment",
      document_id: string(args, "document_id"),
      expected_version_id: string(args, "expected_version_id"),
      quote: string(args, "quote"),
      content: string(args, "content"),
    };
  else {
    if (args.edit_session_id !== undefined || args.operation !== undefined)
      throw new Error("Unexpected canvas tool argument");
    let edits: { find: string; replace: string }[] | undefined;
    if (args.edits !== undefined) {
      if (!Array.isArray(args.edits))
        throw new Error("Canvas edits must be a list");
      edits = args.edits.map((value) => {
        keys(value, ["find", "replace"]);
        return {
          find: string(value, "find"),
          replace: string(value, "replace"),
        };
      });
    }
    mutation = {
      operation: "update",
      document_id: string(args, "document_id"),
      expected_version_id: string(args, "expected_version_id"),
      ...(args.title !== undefined ? { title: string(args, "title") } : {}),
      ...(args.language !== undefined
        ? { language: string(args, "language") as CanvasLanguage }
        : {}),
      ...(args.content !== undefined
        ? { content: string(args, "content") }
        : {}),
      ...(edits ? { edits } : {}),
    };
  }
  const result = await mutateDeviceCanvas(chat, scope, mutation, model, () => {
    guard();
    check();
  });
  guard();
  check();
  canvasChanged(chat, "device");
  const doc =
    mutation.operation === "create"
      ? result.documents[result.documents.length - 1]
      : result.documents.find((d) => d.id === mutation.document_id);
  if (!doc) throw new Error("Canvas result unavailable");
  return {
    canvas: canvasSummary(doc.version),
    operation: mutation.operation,
    saved: true,
  };
}
