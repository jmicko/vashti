import { modelCapabilityBadges } from "./modelCapabilities";
import { normalizeInferenceSettings } from "./inferenceSettings";
import type { ChatDetail, ChatToolPreferences, ModelInfo } from "./types";

export const defaultToolPreferences: ChatToolPreferences = {
  tool_use_enabled: true,
  tools: {}
};

export function modelSupportsToolUse(model: ModelInfo | null | undefined) {
  if (!model) {
    return false;
  }

  return modelCapabilityBadges(model).includes("tools");
}

export function toolPreferenceEnabled(preferences: ChatToolPreferences, toolId: string) {
  return preferences.tools?.[toolId] ?? true;
}

export function updateToolPreference(
  preferences: ChatToolPreferences,
  toolId: string,
  isEnabled: boolean
): ChatToolPreferences {
  return {
    tool_use_enabled: preferences.tool_use_enabled,
    tools: {
      ...(preferences.tools ?? {}),
      [toolId]: isEnabled
    }
  };
}

export function normalizeToolPreferences(
  preferences: Partial<ChatToolPreferences> | null | undefined
): ChatToolPreferences {
  const tools = { ...(preferences?.tools ?? {}) };
  if (preferences?.web_search_enabled !== undefined) {
    tools.brave_web_search = preferences.web_search_enabled;
    tools.ollama_web_search = preferences.web_search_enabled;
  }
  if (preferences?.web_fetch_enabled !== undefined) {
    tools.ollama_web_fetch = preferences.web_fetch_enabled;
    tools.direct_web_fetch = preferences.web_fetch_enabled;
  }

  return {
    tool_use_enabled: preferences?.tool_use_enabled ?? defaultToolPreferences.tool_use_enabled,
    tools
  };
}

export function normalizeChatDetail(chat: ChatDetail): ChatDetail {
  return {
    ...chat,
    tool_preferences: normalizeToolPreferences(chat.tool_preferences),
    inference_settings: normalizeInferenceSettings(chat.inference_settings),
    context_blocks: chat.context_blocks ?? []
  };
}
