import type { ChatInferenceSettings } from "./types";

export const defaultInferenceSettings: ChatInferenceSettings = {};

export function normalizeInferenceSettings(
  settings: Partial<ChatInferenceSettings> | null | undefined
): ChatInferenceSettings {
  return {
    temperature: normalizedNumber(settings?.temperature),
    top_k: normalizedInteger(settings?.top_k),
    top_p: normalizedNumber(settings?.top_p),
    min_p: normalizedNumber(settings?.min_p),
    repeat_penalty: normalizedNumber(settings?.repeat_penalty),
    repeat_last_n: normalizedInteger(settings?.repeat_last_n),
    presence_penalty: normalizedNumber(settings?.presence_penalty),
    frequency_penalty: normalizedNumber(settings?.frequency_penalty),
    num_ctx: normalizedInteger(settings?.num_ctx),
    num_predict: normalizedInteger(settings?.num_predict),
    num_gpu: normalizedInteger(settings?.num_gpu),
    num_thread: normalizedInteger(settings?.num_thread),
    seed: normalizedInteger(settings?.seed)
  };
}

export function hasInferenceSettings(settings: ChatInferenceSettings | null | undefined) {
  const normalized = normalizeInferenceSettings(settings);
  return (
    normalized.temperature !== undefined ||
    normalized.top_k !== undefined ||
    normalized.top_p !== undefined ||
    normalized.min_p !== undefined ||
    normalized.repeat_penalty !== undefined ||
    normalized.repeat_last_n !== undefined ||
    normalized.presence_penalty !== undefined ||
    normalized.frequency_penalty !== undefined ||
    normalized.num_ctx !== undefined ||
    normalized.num_predict !== undefined ||
    normalized.num_gpu !== undefined ||
    normalized.num_thread !== undefined ||
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
    normalizedLeft.top_k === normalizedRight.top_k &&
    normalizedLeft.top_p === normalizedRight.top_p &&
    normalizedLeft.min_p === normalizedRight.min_p &&
    normalizedLeft.repeat_penalty === normalizedRight.repeat_penalty &&
    normalizedLeft.repeat_last_n === normalizedRight.repeat_last_n &&
    normalizedLeft.presence_penalty === normalizedRight.presence_penalty &&
    normalizedLeft.frequency_penalty === normalizedRight.frequency_penalty &&
    normalizedLeft.num_ctx === normalizedRight.num_ctx &&
    normalizedLeft.num_predict === normalizedRight.num_predict &&
    normalizedLeft.num_gpu === normalizedRight.num_gpu &&
    normalizedLeft.num_thread === normalizedRight.num_thread &&
    normalizedLeft.seed === normalizedRight.seed
  );
}

function normalizedNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizedInteger(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : undefined;
}
