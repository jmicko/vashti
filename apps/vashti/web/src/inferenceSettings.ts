import type { ChatInferenceSettings } from "./types";

export const defaultInferenceSettings: ChatInferenceSettings = {};

export function normalizeInferenceSettings(
  settings: Partial<ChatInferenceSettings> | null | undefined
): ChatInferenceSettings {
  return {
    temperature: normalizedNumber(settings?.temperature),
    top_p: normalizedNumber(settings?.top_p),
    repeat_penalty: normalizedNumber(settings?.repeat_penalty),
    num_ctx: normalizedInteger(settings?.num_ctx),
    num_predict: normalizedInteger(settings?.num_predict),
    seed: normalizedInteger(settings?.seed)
  };
}

export function hasInferenceSettings(settings: ChatInferenceSettings | null | undefined) {
  const normalized = normalizeInferenceSettings(settings);
  return (
    normalized.temperature !== undefined ||
    normalized.top_p !== undefined ||
    normalized.repeat_penalty !== undefined ||
    normalized.num_ctx !== undefined ||
    normalized.num_predict !== undefined ||
    normalized.seed !== undefined
  );
}

export function inferenceSettingsEqual(
  left: ChatInferenceSettings | null | undefined,
  right: ChatInferenceSettings | null | undefined
) {
  const normalizedLeft = normalizeInferenceSettings(left);
  const normalizedRight = normalizeInferenceSettings(right);
  return (
    normalizedLeft.temperature === normalizedRight.temperature &&
    normalizedLeft.top_p === normalizedRight.top_p &&
    normalizedLeft.repeat_penalty === normalizedRight.repeat_penalty &&
    normalizedLeft.num_ctx === normalizedRight.num_ctx &&
    normalizedLeft.num_predict === normalizedRight.num_predict &&
    normalizedLeft.seed === normalizedRight.seed
  );
}

function normalizedNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizedInteger(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : undefined;
}
