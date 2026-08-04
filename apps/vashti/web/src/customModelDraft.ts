import type { CustomModelType } from "./types";

export type CustomModelDraft = {
  displayName: string;
  baseModelValue: string;
  systemPrompt: string;
  storageMode: "local" | "private" | "public";
  modelType: CustomModelType;
};

const customModelDraftKey = "vashti:custom-model-draft";

export function storeCustomModelDraft(draft: CustomModelDraft) {
  try {
    window.sessionStorage.setItem(customModelDraftKey, JSON.stringify(draft));
  } catch {
    // The caller still navigates to the creator; storage only carries the draft across routes.
  }
}

export function takeCustomModelDraft(): CustomModelDraft | null {
  try {
    const rawDraft = window.sessionStorage.getItem(customModelDraftKey);
    window.sessionStorage.removeItem(customModelDraftKey);
    if (!rawDraft) {
      return null;
    }

    const draft = JSON.parse(rawDraft) as Partial<CustomModelDraft>;
    if (
      typeof draft.displayName !== "string" ||
      typeof draft.baseModelValue !== "string" ||
      typeof draft.systemPrompt !== "string" ||
      !isStorageMode(draft.storageMode)
    ) {
      return null;
    }

    return {
      displayName: draft.displayName,
      baseModelValue: draft.baseModelValue,
      systemPrompt: draft.systemPrompt,
      storageMode: draft.storageMode,
      modelType: draft.modelType === "character" ? "character" : "general"
    };
  } catch {
    return null;
  }
}

function isStorageMode(value: unknown): value is CustomModelDraft["storageMode"] {
  return value === "local" || value === "private" || value === "public";
}
