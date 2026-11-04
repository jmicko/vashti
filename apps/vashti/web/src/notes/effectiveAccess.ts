import type { MemorySettings } from "../memories/types";
import type { NoteDraft, NoteModelScope, NoteSettings } from "./types";

export type AccessStatus = { action: string; allowed: boolean; reason: string };

function scopeRestriction(scope: NoteModelScope, modelKey: string) {
  if (scope.all_models) return null;
  if (!scope.model_keys.length) return "No models selected";
  if (modelKey && !scope.model_keys.includes(modelKey))
    return "Model not selected";
  return null;
}

export function noteAccessStatus(
  note: Pick<NoteDraft, "ai_access" | "model_scope">,
  settings: NoteSettings,
  modelKey = "",
  trashed = false,
): AccessStatus[] {
  const blocked = trashed
    ? "In Trash"
    : scopeRestriction(note.model_scope, modelKey);
  return [
    {
      action: "Read",
      account: settings.allow_model_read,
      note: note.ai_access !== "none",
    },
    {
      action: "Edit",
      account: settings.allow_model_read && settings.allow_model_edit,
      note: ["edit", "manage"].includes(note.ai_access),
    },
    {
      action: "Move to Trash",
      account: settings.allow_model_read && settings.allow_model_trash,
      note: note.ai_access === "manage",
    },
  ].map(({ action, account, note }) => {
    const reason =
      blocked ??
      (!note
        ? "Not allowed for this note"
        : !account
          ? "Disabled in account settings"
          : "Allowed");
    return { action, allowed: reason === "Allowed", reason };
  });
}

export function memoryAccessStatus(
  scope: NoteModelScope,
  settings: MemorySettings,
  modelKey = "",
  forgotten = false,
): AccessStatus[] {
  const blocked = forgotten ? "Forgotten" : scopeRestriction(scope, modelKey);
  return [
    { action: "Read", account: settings.allow_model_read },
    {
      action: "Edit",
      account: settings.allow_model_read && settings.allow_model_edit,
    },
    {
      action: "Forget",
      account: settings.allow_model_read && settings.allow_model_forget,
    },
  ].map(({ action, account }) => ({
    action,
    allowed: !blocked && account,
    reason: blocked ?? (account ? "Allowed" : "Disabled in account settings"),
  }));
}
