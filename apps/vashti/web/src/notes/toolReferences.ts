import type { MessageStreamSegment, ThinkingSegment, ToolUsageRecord } from "../types";

export type NoteToolReference = {
  noteId: string;
  title: string;
  version?: number;
};

const noteToolNames = new Set([
  "search_notes",
  "read_note",
  "create_note",
  "update_note",
  "trash_note"
]);

export function noteToolReferencesFromSegments(
  segments: Array<MessageStreamSegment | ThinkingSegment>
): NoteToolReference[] {
  const references = new Map<string, NoteToolReference>();

  for (const segment of segments) {
    if (segment.type !== "tool" || !noteToolNames.has(segment.usage.name)) {
      continue;
    }
    for (const reference of noteToolReferencesFromUsage(segment.usage)) {
      const current = references.get(reference.noteId);
      references.set(reference.noteId, {
        noteId: reference.noteId,
        title:
          reference.title !== "Note" || !current
            ? reference.title
            : current.title,
        version: reference.version ?? current?.version
      });
    }
  }

  return Array.from(references.values());
}

function noteToolReferencesFromUsage(usage: ToolUsageRecord): NoteToolReference[] {
  const result = parseObject(usage.result);
  if (!result) {
    return [];
  }

  if (usage.name === "search_notes") {
    return Array.isArray(result.results)
      ? result.results.flatMap((value) => {
          const reference = noteReferenceFromObject(asObject(value));
          return reference ? [reference] : [];
        })
      : [];
  }

  const reference = noteReferenceFromObject(result, titleFromArguments(usage.arguments));
  return reference ? [reference] : [];
}

function noteReferenceFromObject(
  value: Record<string, unknown> | null,
  fallbackTitle = "Note"
): NoteToolReference | null {
  if (!value || typeof value.note_id !== "string" || value.note_id.trim() === "") {
    return null;
  }

  return {
    noteId: value.note_id,
    title:
      typeof value.title === "string" && value.title.trim() !== ""
        ? value.title.trim()
        : fallbackTitle,
    version: typeof value.version === "number" ? value.version : undefined
  };
}

function titleFromArguments(value: unknown) {
  const argumentsObject = asObject(value);
  return typeof argumentsObject?.title === "string" && argumentsObject.title.trim() !== ""
    ? argumentsObject.title.trim()
    : "Note";
}

function parseObject(value: string) {
  try {
    return asObject(JSON.parse(value));
  } catch {
    return null;
  }
}

function asObject(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}
