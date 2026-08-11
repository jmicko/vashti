import {
  createPrivateNote,
  getPrivateNote,
  listPrivateNotes,
  listPrivateNoteVersions,
  permanentlyDeletePrivateNote,
  restorePrivateNote,
  restorePrivateNoteVersion,
  trashPrivateNote,
  updatePrivateNote
} from "../privateChatStore";
import {
  createNote,
  getNote,
  listNotes,
  listNoteVersions,
  permanentlyDeleteNote,
  restoreNote,
  restoreNoteVersion,
  trashNote,
  updateNote
} from "./api";
import type {
  Note,
  NoteAiAccess,
  NoteListStatus,
  NoteModelScope,
  NoteSort,
  NoteSummary,
  NoteVersion
} from "./types";

export type NoteStorageMode = "server" | "device";

export type NoteListParams = {
  query?: string;
  status?: NoteListStatus;
  sort?: NoteSort;
  limit?: number;
  offset?: number;
};

export type CreateNotePayload = {
  title: string;
  content?: string;
  tags?: string[];
  is_pinned?: boolean;
  ai_access?: NoteAiAccess;
  model_scope?: NoteModelScope;
};

export type UpdateNotePayload = CreateNotePayload & {
  expected_version: number;
};

export type NoteSearchFunction = (
  query: string
) => Promise<{ notes: NoteSummary[]; total: number }>;

export type NoteRepository = {
  list: (params?: NoteListParams) => Promise<{ notes: NoteSummary[]; total: number }>;
  get: (noteId: string) => Promise<Note>;
  create: (payload: CreateNotePayload) => Promise<Note>;
  update: (noteId: string, payload: UpdateNotePayload) => Promise<Note>;
  trash: (noteId: string, expectedVersion: number) => Promise<Note>;
  restore: (noteId: string, expectedVersion: number) => Promise<Note>;
  purge: (noteId: string) => Promise<{ ok: boolean }>;
  listVersions: (noteId: string) => Promise<NoteVersion[]>;
  restoreVersion: (
    noteId: string,
    versionId: string,
    expectedVersion: number
  ) => Promise<Note>;
};

export const serverNoteRepository: NoteRepository = {
  list: listNotes,
  get: getNote,
  create: createNote,
  update: updateNote,
  trash: trashNote,
  restore: restoreNote,
  purge: permanentlyDeleteNote,
  listVersions: listNoteVersions,
  restoreVersion: restoreNoteVersion
};

export const deviceNoteRepository: NoteRepository = {
  list: listPrivateNotes,
  get: getPrivateNote,
  create: createPrivateNote,
  update: updatePrivateNote,
  trash: trashPrivateNote,
  restore: restorePrivateNote,
  purge: permanentlyDeletePrivateNote,
  listVersions: listPrivateNoteVersions,
  restoreVersion: restorePrivateNoteVersion
};

export function noteRepositoryFor(mode: NoteStorageMode) {
  return mode === "device" ? deviceNoteRepository : serverNoteRepository;
}

export const searchServerNotes: NoteSearchFunction = (query) =>
  serverNoteRepository.list({ query, status: "active", sort: "updated", limit: 20 });

export const searchDeviceNotes: NoteSearchFunction = (query) =>
  deviceNoteRepository.list({ query, status: "active", sort: "updated", limit: 20 });
