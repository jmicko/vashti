import type { ToolUsageRecord } from "./types";

export type ToolRecord = {
  kind: "memory" | "chat";
  id: string;
  label: string;
  excerpt?: string;
};
const memoryTools = new Set([
  "search_memories",
  "read_memory",
  "remember",
  "update_memory",
  "forget_memory",
]);
const chatTools = new Set(["search_chat_history", "read_chat_history"]);

export function toolRecords(usage: ToolUsageRecord): ToolRecord[] {
  const memory = memoryTools.has(usage.name);
  if (!memory && !chatTools.has(usage.name)) return [];
  let result;
  try {
    result = JSON.parse(usage.result);
  } catch {
    return [];
  }
  if (!result || typeof result !== "object" || result.error) return [];
  const values = Array.isArray(result.results)
    ? result.results
    : result.conversation
      ? [result.conversation]
      : [result];
  const records = new Map<string, ToolRecord>();
  for (const value of values.slice(0, 10)) {
    const id = value?.[memory ? "memory_id" : "chat_id"];
    if (typeof id !== "string" || !id.trim() || id.length > 200) continue;
    const text = memory ? (value.content ?? value.excerpt) : value.chat_title;
    const excerpt = memory
      ? undefined
      : (value.excerpt ??
        (Array.isArray(value.messages)
          ? value.messages.find(
              (message: Record<string, unknown> | null) =>
                message?.message_id === value.target_message_id,
            )?.content
          : undefined));
    records.set(id, {
      kind: memory ? "memory" : "chat",
      id,
      label:
        typeof text === "string" && text.trim()
          ? text.trim().slice(0, 120)
          : memory
            ? "Memory"
            : "Conversation",
      ...(typeof excerpt === "string" && excerpt.trim()
        ? { excerpt: excerpt.trim().slice(0, 360) }
        : {}),
    });
  }
  return [...records.values()];
}

export function toolRecordPath(record: ToolRecord) {
  return record.kind === "memory"
    ? `/app/settings/memories#${new URLSearchParams({ memory: record.id })}`
    : `/app/chats/${encodeURIComponent(record.id)}`;
}

export function toolEmptyResult(usage: ToolUsageRecord) {
  if (!memoryTools.has(usage.name) && !chatTools.has(usage.name)) return false;
  try {
    return JSON.parse(usage.result)?.count === 0;
  } catch {
    return false;
  }
}
