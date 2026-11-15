import { useEffect, useState } from "react";
import { Info, RotateCcw } from "lucide-react";
import { requestJson } from "./api";
import { modelParts } from "./modelSelection";
import { normalizeInferenceSettings } from "./inferenceSettings";
import type { ChatInferenceSettings } from "./types";

export type InferenceInputKey = keyof ChatInferenceSettings;

export type InferenceInputValues = Record<InferenceInputKey, string>;

type InferenceFieldDefinition = {
  key: InferenceInputKey;
  label: string;
  help: string;
  inputMode: "decimal" | "numeric";
  step?: string;
  min?: string;
  max?: string;
};

const inferenceFieldDefinitions: InferenceFieldDefinition[] = [
  {
    key: "temperature",
    label: "Temperature",
    help: "Controls randomness. Lower values are steadier; higher values are more creative.",
    inputMode: "decimal",
    step: "0.1",
    min: "0",
    max: "2"
  },
  {
    key: "num_ctx",
    label: "Context",
    help: "Sets how many tokens the model can use as context. Higher values use more memory.",
    inputMode: "numeric",
    step: "1",
    min: "512"
  },
  {
    key: "num_predict",
    label: "Max output",
    help: "Limits how many tokens the model can generate for the response.",
    inputMode: "numeric",
    step: "1",
    min: "1"
  },
  {
    key: "top_k",
    label: "Top K",
    help: "Limits each token choice to the top K likely tokens. Lower values are more conservative.",
    inputMode: "numeric",
    step: "1",
    min: "1"
  },
  {
    key: "top_p",
    label: "Top P",
    help: "Keeps the smallest set of likely tokens whose probability adds up to this value.",
    inputMode: "decimal",
    step: "0.05",
    min: "0.01",
    max: "1"
  },
  {
    key: "min_p",
    label: "Min P",
    help: "Filters out tokens below a probability floor relative to the most likely token.",
    inputMode: "decimal",
    step: "0.01",
    min: "0",
    max: "1"
  },
  {
    key: "repeat_penalty",
    label: "Repeat penalty",
    help: "Penalizes repeated tokens. Higher values reduce repetition more strongly.",
    inputMode: "decimal",
    step: "0.05",
    min: "0.5",
    max: "2"
  },
  {
    key: "repeat_last_n",
    label: "Repeat window",
    help: "Sets how far back the repeat penalty looks. Use -1 to use the context window.",
    inputMode: "numeric",
    step: "1",
    min: "-1"
  },
  {
    key: "presence_penalty",
    label: "Presence penalty",
    help: "Penalizes tokens that have appeared at all, encouraging new topics or wording.",
    inputMode: "decimal",
    step: "0.05",
    min: "-2",
    max: "2"
  },
  {
    key: "frequency_penalty",
    label: "Frequency penalty",
    help: "Penalizes tokens more as they repeat more often.",
    inputMode: "decimal",
    step: "0.05",
    min: "-2",
    max: "2"
  },
  {
    key: "num_gpu",
    label: "GPU layers",
    help: "Controls how many model layers Ollama tries to offload to GPU. Use 0 for CPU only, or leave blank for Ollama's default.",
    inputMode: "numeric",
    step: "1",
    min: "0"
  },
  {
    key: "num_thread",
    label: "CPU threads",
    help: "Controls how many CPU threads Ollama uses for inference.",
    inputMode: "numeric",
    step: "1",
    min: "1"
  },
  {
    key: "seed",
    label: "Seed",
    help: "Sets the random seed. Reusing the same seed can make similar prompts more reproducible.",
    inputMode: "numeric",
    step: "1"
  }
];
export function InferenceSettingsFields({
  inputs,
  defaults = {},
  disabled = false,
  onChange
}: {
  inputs: InferenceInputValues;
  defaults?: ChatInferenceSettings;
  disabled?: boolean;
  onChange: (key: InferenceInputKey, value: string) => void;
}) {
  return <>{inferenceFieldDefinitions.map(field => {
    const inheritedValue = defaults[field.key];
    return (
      <div key={field.key} className="model-settings-number-field">
        <span className="model-settings-number-label">{field.label}</span>
        <div className="model-settings-input-reset">
          <button type="button" className="model-settings-help" aria-label={`${field.label} help`}>
            <Info />
            <span className="model-settings-help-popover">{field.help}</span>
          </button>
          <input
            type="number"
            aria-label={field.label}
            inputMode={field.inputMode}
            step={field.step}
            min={field.min}
            max={field.max}
            placeholder={inheritedValue === undefined ? "Default" : String(inheritedValue)}
            title={inheritedValue === undefined ? "Backend default" : `Inherited default: ${inheritedValue}`}
            value={inputs[field.key]}
            disabled={disabled}
            onChange={event => onChange(field.key, event.target.value)}
          />
          <button
            type="button"
            className="icon-button model-settings-field-reset"
            aria-label={`Reset ${field.label} to default`}
            title={`Reset ${field.label} to default`}
            disabled={disabled || !inputs[field.key]}
            onClick={() => onChange(field.key, "")}
          ><RotateCcw /></button>
        </div>
      </div>
    );
  })}</>;
}

export function useModelInferenceDefaults(baseModelValue: string | null, enabled = true) {
  const [state, setState] = useState<{
    model: string | null;
    defaults: ChatInferenceSettings;
  }>({ model: null, defaults: {} });
  useEffect(() => {
    const parts = baseModelValue && modelParts(baseModelValue);
    if (!enabled || !parts) return;
    let current = true;
    const query = new URLSearchParams({ backend_id: parts.backendId, model_name: parts.modelName });
    void requestJson<ChatInferenceSettings>(`/api/models/inference-defaults?${query}`)
      .then(defaults => {
        if (current) setState({ model: baseModelValue, defaults: normalizeInferenceSettings(defaults) });
      })
      .catch(() => {
        if (current) setState({ model: baseModelValue, defaults: {} });
      });
    return () => { current = false; };
  }, [baseModelValue, enabled]);
  return state.model === baseModelValue ? state.defaults : {};
}

export function inferenceSettingsToInputs(settings: ChatInferenceSettings): InferenceInputValues {
  return {
    temperature: inferenceInputValue(settings.temperature),
    top_k: inferenceInputValue(settings.top_k),
    top_p: inferenceInputValue(settings.top_p),
    min_p: inferenceInputValue(settings.min_p),
    repeat_penalty: inferenceInputValue(settings.repeat_penalty),
    repeat_last_n: inferenceInputValue(settings.repeat_last_n),
    presence_penalty: inferenceInputValue(settings.presence_penalty),
    frequency_penalty: inferenceInputValue(settings.frequency_penalty),
    num_ctx: inferenceInputValue(settings.num_ctx),
    num_predict: inferenceInputValue(settings.num_predict),
    num_gpu: inferenceInputValue(settings.num_gpu),
    num_thread: inferenceInputValue(settings.num_thread),
    seed: inferenceInputValue(settings.seed)
  };
}

export function inferenceInputsToSettings(inputs: InferenceInputValues): ChatInferenceSettings {
  return normalizeInferenceSettings({
    temperature: parsedInferenceNumber(inputs.temperature),
    top_k: parsedInferenceInteger(inputs.top_k),
    top_p: parsedInferenceNumber(inputs.top_p),
    min_p: parsedInferenceNumber(inputs.min_p),
    repeat_penalty: parsedInferenceNumber(inputs.repeat_penalty),
    repeat_last_n: parsedInferenceInteger(inputs.repeat_last_n),
    presence_penalty: parsedInferenceNumber(inputs.presence_penalty),
    frequency_penalty: parsedInferenceNumber(inputs.frequency_penalty),
    num_ctx: parsedInferenceInteger(inputs.num_ctx),
    num_predict: parsedInferenceInteger(inputs.num_predict),
    num_gpu: parsedInferenceInteger(inputs.num_gpu),
    num_thread: parsedInferenceInteger(inputs.num_thread),
    seed: parsedInferenceInteger(inputs.seed)
  });
}

function inferenceInputValue(value: number | null | undefined) {
  return value === null || value === undefined ? "" : String(value);
}

function parsedInferenceNumber(value: string) {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }

  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parsedInferenceInteger(value: string) {
  const parsed = parsedInferenceNumber(value);
  return parsed === undefined ? undefined : Math.trunc(parsed);
}
