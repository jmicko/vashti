import {
  createPrivateNote,
  getPrivateNote,
  getPrivateNoteSettings,
  listPrivateNotes,
  trashPrivateNote,
  updatePrivateNote,
  type PrivateNoteMutationActor
} from "./privateChatStore";
import type { Note, NoteAiAccess, NoteModelScope, NoteSettings } from "./notes/types";

export const DEVICE_NOTES_TOOL_ID = "notes";
export const DEVICE_NOTE_TOOL_NAMES = [
  "search_notes",
  "read_note",
  "create_note",
  "update_note",
  "trash_note"
] as const;

export type DeviceNoteToolName = (typeof DEVICE_NOTE_TOOL_NAMES)[number];

export type DeviceNoteToolContext = {
  chatId: string;
  messageId: string;
  callId: string;
  modelKey: string;
  modelName: string;
};

export async function enabledDeviceNoteToolNames(): Promise<DeviceNoteToolName[]> {
  const settings = await getPrivateNoteSettings();
  const names: DeviceNoteToolName[] = [];
  if (settings.allow_model_read) {
    names.push("search_notes", "read_note");
  }
  if (settings.allow_model_create) {
    names.push("create_note");
  }
  if (settings.allow_model_read && settings.allow_model_edit) {
    names.push("update_note");
  }
  if (settings.allow_model_read && settings.allow_model_trash) {
    names.push("trash_note");
  }
  return names;
}

export async function executeDeviceNoteTool(
  name: DeviceNoteToolName,
  args: Record<string, unknown>,
  context: DeviceNoteToolContext
): Promise<unknown> {
  const settings = await getPrivateNoteSettings();
  const actor: PrivateNoteMutationActor = {
    model_key: context.modelKey,
    model_name: context.modelName,
    chat_id: context.chatId,
    message_id: context.messageId,
    tool_call_id: context.callId
  };

  switch (name) {
    case "search_notes":
      requireGlobalPermission(settings.allow_model_read, "read notes");
      return searchNotes(args, context.modelKey);
    case "read_note":
      requireGlobalPermission(settings.allow_model_read, "read notes");
      return readNote(args, context.modelKey);
    case "create_note":
      requireGlobalPermission(settings.allow_model_create, "create notes");
      return createNote(args, settings, actor);
    case "update_note":
      requireGlobalPermission(settings.allow_model_read && settings.allow_model_edit, "edit notes");
      return updateNote(args, context.modelKey, actor);
    case "trash_note":
      requireGlobalPermission(settings.allow_model_read && settings.allow_model_trash, "move notes to trash");
      return trashNote(args, context.modelKey);
  }
}

async function searchNotes(args: Record<string, unknown>, modelKey: string) {
  const query = requiredString(args, "query");
  const limit = boundedInteger(args.limit, 5, 1, 10);
  const response = await listPrivateNotes({
    query,
    status: "active",
    sort: "updated",
    limit: 200
  });
  const results = response.notes
    .filter((note) => note.ai_access !== "none" && modelInScope(note.model_scope, modelKey))
    .slice(0, limit)
    .map((note) => ({
      note_id: note.id,
      title: note.title,
      excerpt: note.excerpt,
      tags: note.tags,
      version: note.current_version_number
    }));
  return { query, count: results.length, results };
}

async function readNote(args: Record<string, unknown>, modelKey: string) {
  const note = await accessibleNote(requiredString(args, "note_id"), modelKey, "read");
  const content = truncateCharacters(note.current_version.content, 18_000);
  return {
    note_id: note.id,
    title: note.current_version.title,
    content,
    content_truncated: content.length < note.current_version.content.length,
    tags: note.tags,
    version: note.current_version.version_number
  };
}

async function createNote(
  args: Record<string, unknown>,
  settings: NoteSettings,
  actor: PrivateNoteMutationActor
) {
  const modelScope = settings.default_model_scope.all_models
    ? settings.default_model_scope
    : {
        all_models: false,
        model_keys: [...new Set([...settings.default_model_scope.model_keys, actor.model_key])]
      };
  const note = await createPrivateNote({
    title: requiredString(args, "title"),
    content: requiredString(args, "content"),
    tags: optionalStringArray(args, "tags"),
    ai_access: "manage",
    model_scope: modelScope,
    actor
  });
  return {
    note_id: note.id,
    title: note.current_version.title,
    version: note.current_version.version_number,
    created: true
  };
}

async function updateNote(
  args: Record<string, unknown>,
  modelKey: string,
  actor: PrivateNoteMutationActor
) {
  const noteId = requiredString(args, "note_id");
  await accessibleNote(noteId, modelKey, "edit");
  const title = optionalString(args, "title");
  const content = optionalString(args, "content");
  if (title === undefined && content === undefined) {
    throw new Error("update_note requires a title and/or content change");
  }
  const note = await updatePrivateNote(noteId, {
    expected_version: requiredInteger(args, "expected_version"),
    title,
    content,
    actor
  });
  return {
    note_id: note.id,
    title: note.current_version.title,
    version: note.current_version.version_number,
    updated: true
  };
}

async function trashNote(args: Record<string, unknown>, modelKey: string) {
  const noteId = requiredString(args, "note_id");
  await accessibleNote(noteId, modelKey, "manage");
  const note = await trashPrivateNote(noteId, requiredInteger(args, "expected_version"));
  return {
    note_id: note.id,
    version: note.current_version.version_number,
    trashed: true
  };
}

async function accessibleNote(
  noteId: string,
  modelKey: string,
  requiredAccess: NoteAiAccess
): Promise<Note> {
  let note: Note;
  try {
    note = await getPrivateNote(noteId);
  } catch {
    throw unavailableNoteError();
  }
  if (
    note.deleted_at ||
    accessRank(note.ai_access) < accessRank(requiredAccess) ||
    !modelInScope(note.model_scope, modelKey)
  ) {
    throw unavailableNoteError();
  }
  return note;
}

function modelInScope(scope: NoteModelScope, modelKey: string) {
  return scope.all_models || scope.model_keys.includes(modelKey);
}

function accessRank(access: NoteAiAccess) {
  return { none: 0, read: 1, edit: 2, manage: 3 }[access];
}

function requireGlobalPermission(allowed: boolean, action: string) {
  if (!allowed) {
    throw new Error(`Device note settings do not allow models to ${action}`);
  }
}

function unavailableNoteError() {
  return new Error("The requested note is unavailable to this model");
}

function requiredString(args: Record<string, unknown>, key: string) {
  const value = args[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${key} is required`);
  }
  return value.trim();
}

function optionalString(args: Record<string, unknown>, key: string) {
  const value = args[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`${key} must be text`);
  }
  return value;
}

function optionalStringArray(args: Record<string, unknown>, key: string) {
  const value = args[key];
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new Error(`${key} must be a list of text values`);
  }
  return value as string[];
}

function requiredInteger(args: Record<string, unknown>, key: string) {
  const value = args[key];
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new Error(`${key} must be an integer`);
  }
  return value;
}

function boundedInteger(value: unknown, fallback: number, min: number, max: number) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(max, Math.max(min, Math.trunc(value)))
    : fallback;
}

function truncateCharacters(value: string, max: number) {
  return [...value].slice(0, max).join("");
}
