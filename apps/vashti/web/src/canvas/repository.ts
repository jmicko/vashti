import { requestJson } from "../api";
import {
  commitPrivateCanvasRecords,
  listPrivateCanvasRecords,
  privateImageGuard,
  privateProfileIdentity,
  readPrivateCanvasRecord,
} from "../privateChatStore";
import type {
  CanvasMode,
  CanvasMutation,
  CanvasScope,
  CanvasVersion,
  CanvasVersionSummary,
  CanvasWorkspace,
} from "./types";
import { canvasScopeKey } from "./types";

function path(chat: string) {
  return `/api/canvas/chats/${encodeURIComponent(chat)}`;
}
async function request<T>(url: string, options: RequestInit = {}) {
  const guard = privateImageGuard(),
    identity = privateProfileIdentity();
  const result = await requestJson<T>(url, {
    ...options,
    expectedServerInstanceId: identity.instance_id,
  });
  guard();
  return result;
}
export async function loadCanvas(
  mode: CanvasMode,
  chat: string,
  scope: CanvasScope,
): Promise<CanvasWorkspace> {
  if (mode === "device")
    return (await import("./device")).deviceCanvasWorkspace(chat, scope);
  const query = new URLSearchParams();
  if (scope.message_id) query.set("message_id", scope.message_id);
  if (scope.revision_id) query.set("revision_id", scope.revision_id);
  return request<CanvasWorkspace>(`${path(chat)}?${query}`);
}
export function canvasChanged(chat: string, mode: CanvasMode) {
  window.dispatchEvent(
    new CustomEvent("vashti:canvas-changed", { detail: { chat, mode } }),
  );
}
export async function mutateCanvas(
  mode: CanvasMode,
  chat: string,
  scope: CanvasScope,
  mutation: CanvasMutation,
): Promise<CanvasWorkspace> {
  const result =
    mode === "device"
      ? await (
          await import("./device")
        ).mutateDeviceCanvas(chat, scope, mutation)
      : await request<CanvasWorkspace>(`${path(chat)}/mutations`, {
          method: "POST",
          body: JSON.stringify({ scope, mutation }),
        });
  canvasChanged(chat, mode);
  return result;
}
export async function canvasHistory(
  mode: CanvasMode,
  chat: string,
  document: string,
  before?: number,
): Promise<CanvasVersionSummary[]> {
  if (mode === "device")
    return (await import("./device")).deviceCanvasHistory(
      chat,
      document,
      before,
    );
  return request(
    `${path(chat)}/documents/${encodeURIComponent(document)}/versions${before === undefined ? "" : `?before=${before}`}`,
  );
}
export async function canvasVersion(
  mode: CanvasMode,
  chat: string,
  document: string,
  id: string,
): Promise<CanvasVersion> {
  if (mode === "device")
    return (await import("./device")).deviceCanvasVersion(chat, document, id);
  return request(
    `${path(chat)}/documents/${encodeURIComponent(document)}/versions/${encodeURIComponent(id)}`,
  );
}
export type CanvasDraft = {
  title: string;
  language: CanvasVersion["language"];
  content: string;
  expected_version_id: string;
  edit_session_id: string;
  updated_at: number;
};
const draftPrefix = (
  mode: CanvasMode,
  chat: string,
  scope: CanvasScope,
  document: string,
) => `canvas-draft:${mode}:${canvasScopeKey(chat, scope)}:${document}:`;
export async function readCanvasDrafts(
  mode: CanvasMode,
  chat: string,
  scope: CanvasScope,
  document: string,
) {
  return (
    await listPrivateCanvasRecords<CanvasDraft>(
      chat,
      draftPrefix(mode, chat, scope, document),
    )
  ).sort((a, b) => b.updated_at - a.updated_at);
}
export async function saveCanvasDraft(
  mode: CanvasMode,
  chat: string,
  scope: CanvasScope,
  document: string,
  value: CanvasDraft,
) {
  const guard = privateImageGuard(),
    id = draftPrefix(mode, chat, scope, document) + value.edit_session_id,
    existing = await readPrivateCanvasRecord<CanvasDraft>(id);
  guard();
  await commitPrivateCanvasRecords(
    chat,
    [{ id, token: existing?.token ?? null }],
    [{ id, value }],
  );
}
export async function clearCanvasDraft(
  mode: CanvasMode,
  chat: string,
  scope: CanvasScope,
  document: string,
  saved: CanvasDraft,
) {
  const guard = privateImageGuard(),
    id = draftPrefix(mode, chat, scope, document) + saved.edit_session_id,
    existing = await readPrivateCanvasRecord<CanvasDraft>(id);
  guard();
  const old = existing?.value;
  if (
    old &&
    old.content === saved.content &&
    old.title === saved.title &&
    old.language === saved.language
  )
    await commitPrivateCanvasRecords(
      chat,
      [{ id, token: existing!.token }],
      [],
      [id],
    );
}
