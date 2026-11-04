import type { BackendModelGroup, ModelInfo } from "./types";

export function canChatWith(model: Pick<ModelInfo, "capabilities">): boolean {
  const capabilities = (model.capabilities ?? []).map((value) => value.trim().toLowerCase());
  // Older Ollama-compatible hosts omit capabilities. Do not lock their models out.
  return !capabilities.includes("embedding") || capabilities.includes("completion") || capabilities.includes("chat");
}

export function chatModelGroups<T extends BackendModelGroup>(groups: T[]): T[] {
  return groups.map((group) => ({ ...group, models: group.models.filter(canChatWith) }));
}
