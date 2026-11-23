import type { MessageStreamSegment, ThinkingSegment } from "../types";
import {
  CANVAS_TOOL_NAMES,
  canvasLanguages,
  type CanvasReference,
} from "./types";

export function canvasReferences(
  segments: (MessageStreamSegment | ThinkingSegment)[],
): CanvasReference[] {
  const found = new Map<string, CanvasReference>();
  for (const segment of segments) {
    if (
      segment.type !== "tool" ||
      !(CANVAS_TOOL_NAMES as readonly string[]).includes(segment.usage.name)
    )
      continue;
    try {
      const data = JSON.parse(segment.usage.result),
        ref = data?.canvas;
      if (
        !ref ||
        typeof ref.document_id !== "string" ||
        !ref.document_id ||
        ref.document_id.length > 160 ||
        typeof ref.version_id !== "string" ||
        !ref.version_id ||
        ref.version_id.length > 160 ||
        typeof ref.title !== "string" ||
        !Number.isInteger(ref.version_number) ||
        !canvasLanguages.includes(ref.language)
      )
        continue;
      found.set(ref.document_id, {
        document_id: ref.document_id,
        version_id: ref.version_id,
        title: ref.title.slice(0, 200),
        version_number: ref.version_number,
        language: ref.language,
      });
    } catch {
      /* Only structured, successful tool results become document links. */
    }
  }
  return [...found.values()];
}
