import { requestJson } from "../api";
import type {
  Note,
  NoteAiAccess,
  NoteListStatus,
  NoteModelScope,
  NoteSettings,
  NoteSort,
  NoteSummary,
  NoteVersion
} from "./types";

type NoteMutationResponse = { note: Note };

export async function listNotes({
  query = "",
  status = "active",
  sort = "updated",
  limit = 100,
  offset = 0
}: {
  query?: string;
  status?: NoteListStatus;
  sort?: NoteSort;
  limit?: number;
  offset?: number;
} = {}) {
  const params = new URLSearchParams({
    status,
    sort,
    limit: String(limit),
    offset: String(offset)
  });
  if (query.trim()) {
    params.set("query", query.trim());
  }
  return requestJson<{ notes: NoteSummary[]; total: number }>(`/api/notes?${params}`);
}

export async function getNote(noteId: string) {
  return (await requestJson<NoteMutationResponse>(`/api/notes/${noteId}`)).note;
}

export async function createNote(payload: {
  title: string;
  content?: string;
  tags?: string[];
  is_pinned?: boolean;
  ai_access?: NoteAiAccess;
  model_scope?: NoteModelScope;
}) {
  return (
    await requestJson<NoteMutationResponse>("/api/notes", {
      method: "POST",
      body: JSON.stringify(payload)
    })
  ).note;
}

export async function updateNote(
  noteId: string,
  payload: {
    expected_version: number;
    title?: string;
    content?: string;
    tags?: string[];
    is_pinned?: boolean;
    ai_access?: NoteAiAccess;
    model_scope?: NoteModelScope;
  }
) {
  return (
    await requestJson<NoteMutationResponse>(`/api/notes/${noteId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    })
  ).note;
}

export async function trashNote(noteId: string, expectedVersion: number) {
  return mutateVersionedNote(`/api/notes/${noteId}/trash`, expectedVersion);
}

export async function restoreNote(noteId: string, expectedVersion: number) {
  return mutateVersionedNote(`/api/notes/${noteId}/restore`, expectedVersion);
}

export async function permanentlyDeleteNote(noteId: string) {
  return requestJson<{ ok: boolean }>(`/api/notes/${noteId}`, { method: "DELETE" });
}

export async function listNoteVersions(noteId: string) {
  return (
    await requestJson<{ versions: NoteVersion[] }>(`/api/notes/${noteId}/versions`)
  ).versions;
}

export async function restoreNoteVersion(
  noteId: string,
  versionId: string,
  expectedVersion: number
) {
  return mutateVersionedNote(
    `/api/notes/${noteId}/versions/${versionId}/restore`,
    expectedVersion
  );
}

export async function getNoteSettings() {
  return requestJson<NoteSettings>("/api/notes/settings");
}

export async function updateNoteSettings(settings: NoteSettings) {
  return requestJson<NoteSettings>("/api/notes/settings", {
    method: "PATCH",
    body: JSON.stringify(settings)
  });
}

async function mutateVersionedNote(path: string, expectedVersion: number) {
  return (
    await requestJson<NoteMutationResponse>(path, {
      method: "POST",
      body: JSON.stringify({ expected_version: expectedVersion })
    })
  ).note;
}
