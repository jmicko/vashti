import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Info, Plus, RotateCcw, SlidersHorizontal } from "lucide-react";
import { requestJson } from "./api";
import type { CustomModelDraft } from "./customModelDraft";
import {
  hasInferenceSettings,
  inferenceSettingsEqual,
  normalizeInferenceSettings
} from "./inferenceSettings";
import { ModelCapabilityBadges } from "./modelCapabilities";
import {
  compactModelName,
  modelParts,
  modelValue,
  personaModelValue,
  personaVersionForValue,
  privatePersonaModelValue,
  privatePersonaVersionForValue
} from "./modelSelection";
import {
  listPrivatePersonaVersions,
  type PrivatePersona,
  type PrivatePersonaVersion
} from "./privateChatStore";
import type {
  BackendModelGroup,
  ChatInferenceSettings,
  ModelInfo,
  Persona,
  PersonaVersion
} from "./types";

type PersonaVersionsResponse = {
  versions: PersonaVersion[];
};

type InferenceInputKey = keyof ChatInferenceSettings;

type InferenceInputValues = Record<InferenceInputKey, string>;

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

export function ModelSettingsMenu({
  groups,
  personas,
  privatePersonas,
  personaVersions,
  privatePersonaVersions,
  selectedModel,
  selectedModelInfo,
  systemPromptOverride,
  inferenceSettings,
  canSaveConversationSettings,
  disabled,
  onModelSelected,
  onCreateCustomModelFromSettings,
  onPersonaVersionsLoaded,
  onPrivatePersonaVersionsLoaded,
  onSystemPromptOverrideChange,
  onInferenceSettingsChange
}: {
  groups: BackendModelGroup[];
  personas: Persona[];
  privatePersonas: PrivatePersona[];
  personaVersions: PersonaVersion[];
  privatePersonaVersions: PrivatePersonaVersion[];
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  systemPromptOverride?: string | null;
  inferenceSettings?: ChatInferenceSettings;
  canSaveConversationSettings?: boolean;
  disabled: boolean;
  onModelSelected: (value: string) => void;
  onCreateCustomModelFromSettings?: (draft: CustomModelDraft) => void;
  onPersonaVersionsLoaded: (versions: PersonaVersion[]) => void;
  onPrivatePersonaVersionsLoaded: (versions: PrivatePersonaVersion[]) => void;
  onSystemPromptOverrideChange?: (value: string | null) => void;
  onInferenceSettingsChange?: (value: ChatInferenceSettings) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [isVersionMenuOpen, setIsVersionMenuOpen] = useState(false);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [draftSystemPrompt, setDraftSystemPrompt] = useState("");
  const [draftInferenceInputs, setDraftInferenceInputs] = useState<InferenceInputValues>(() =>
    inferenceSettingsToInputs({})
  );
  const [error, setError] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const versionPickerRef = useRef<HTMLDivElement>(null);
  const loadedHostedPersonaIdsRef = useRef(new Set<string>());
  const loadedPrivatePersonaIdsRef = useRef(new Set<string>());

  const selectedHostedVersion = personaVersionForValue(personas, personaVersions, selectedModel);
  const selectedPrivateVersion = privatePersonaVersionForValue(
    privatePersonas,
    privatePersonaVersions,
    selectedModel
  );
  const selectedHostedPersona = selectedHostedVersion
    ? personas.find((persona) => persona.id === selectedHostedVersion.persona_id) ?? null
    : null;
  const selectedPrivatePersona = selectedPrivateVersion
    ? privatePersonas.find((persona) => persona.id === selectedPrivateVersion.persona_id) ?? null
    : null;
  const selectedBase = modelParts(selectedModel);
  const selectedBaseOption =
    selectedBase &&
    groups
      .flatMap((group) =>
        group.models.map((model) => ({
          backendId: group.backend.id,
          backendName: group.backend.name,
          model
        }))
      )
      .find((option) => modelValue(option.backendId, option.model.name) === selectedModel);

  const hostedVersions = useMemo(
    () =>
      selectedHostedVersion
        ? uniquePersonaVersions([
            ...personaVersions.filter(
              (version) => version.persona_id === selectedHostedVersion.persona_id
            ),
            selectedHostedVersion
          ])
        : [],
    [personaVersions, selectedHostedVersion]
  );
  const privateVersions = useMemo(
    () =>
      selectedPrivateVersion
        ? uniquePrivatePersonaVersions([
            ...privatePersonaVersions.filter(
              (version) => version.persona_id === selectedPrivateVersion.persona_id
            ),
            selectedPrivateVersion
          ])
        : [],
    [privatePersonaVersions, selectedPrivateVersion]
  );
  const isCustomModel = Boolean(selectedHostedVersion || selectedPrivateVersion);
  const isUsingNonDefaultVersion = Boolean(
    (selectedHostedPersona &&
      selectedHostedVersion &&
      selectedHostedPersona.current_version.id !== selectedHostedVersion.id) ||
      (selectedPrivatePersona &&
        selectedPrivateVersion &&
        selectedPrivatePersona.current_version.id !== selectedPrivateVersion.id)
  );
  const title =
    selectedHostedVersion?.display_name ??
    selectedPrivateVersion?.display_name ??
    (selectedBaseOption ? compactModelName(selectedBaseOption.model.name) : "Model settings");
  const subtitle = selectedHostedVersion
    ? `Custom model v${selectedHostedVersion.version_number}`
    : selectedPrivateVersion
      ? `Device custom model v${selectedPrivateVersion.version_number}`
      : selectedBaseOption
        ? `${selectedBaseOption.backendName} / ${compactModelName(selectedBaseOption.model.name)}`
        : "Select a model to view settings";
  const defaultSystemPrompt =
    selectedHostedVersion?.system_prompt ?? selectedPrivateVersion?.system_prompt ?? "";
  const effectiveSystemPrompt =
    systemPromptOverride === undefined || systemPromptOverride === null
      ? defaultSystemPrompt
      : systemPromptOverride;
  const isSystemPromptCustomized =
    systemPromptOverride !== undefined && systemPromptOverride !== null;
  const effectiveInferenceSettings = normalizeInferenceSettings(inferenceSettings);
  const draftInferenceSettings = useMemo(
    () => inferenceInputsToSettings(draftInferenceInputs),
    [draftInferenceInputs]
  );
  const inferenceSettingsChanged = !inferenceSettingsEqual(
    draftInferenceSettings,
    effectiveInferenceSettings
  );
  const isInferenceCustomized = hasInferenceSettings(effectiveInferenceSettings);
  const hasConversationSettings =
    isUsingNonDefaultVersion || isSystemPromptCustomized || isInferenceCustomized;
  const baseModelName =
    selectedHostedVersion?.base_model_name ??
    selectedPrivateVersion?.base_model_name ??
    selectedBaseOption?.model.name ??
    null;
  const baseModelDraftValue = selectedHostedVersion
    ? modelValue(selectedHostedVersion.base_backend_id, selectedHostedVersion.base_model_name)
    : selectedPrivateVersion
      ? modelValue(selectedPrivateVersion.base_backend_id, selectedPrivateVersion.base_model_name)
      : selectedBaseOption
        ? modelValue(selectedBaseOption.backendId, selectedBaseOption.model.name)
        : null;
  const selectedVersionId = selectedHostedVersion?.id ?? selectedPrivateVersion?.id ?? "";
  const displayedVersions = selectedHostedVersion ? hostedVersions : privateVersions;
  const selectedVersionLabel =
    (selectedHostedVersion ?? selectedPrivateVersion)
      ? versionOptionLabel(selectedHostedVersion ?? selectedPrivateVersion)
      : "Select version";

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      const target = event.target as Node;

      if (wrapRef.current && !wrapRef.current.contains(target)) {
        setIsOpen(false);
        setIsVersionMenuOpen(false);
        return;
      }

      if (versionPickerRef.current && !versionPickerRef.current.contains(target)) {
        setIsVersionMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    if (!isOpen) {
      setIsVersionMenuOpen(false);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    if (selectedHostedVersion) {
      void loadHostedVersions(selectedHostedVersion.persona_id);
      return;
    }

    if (selectedPrivateVersion) {
      void loadPrivateVersions(selectedPrivateVersion.persona_id);
    }
  }, [isOpen, selectedHostedVersion?.persona_id, selectedPrivateVersion?.persona_id]);

  useEffect(() => {
    setDraftSystemPrompt(effectiveSystemPrompt);
  }, [effectiveSystemPrompt, selectedModel]);

  useEffect(() => {
    setDraftInferenceInputs(inferenceSettingsToInputs(effectiveInferenceSettings));
  }, [
    effectiveInferenceSettings.temperature,
    effectiveInferenceSettings.top_k,
    effectiveInferenceSettings.top_p,
    effectiveInferenceSettings.min_p,
    effectiveInferenceSettings.repeat_penalty,
    effectiveInferenceSettings.repeat_last_n,
    effectiveInferenceSettings.presence_penalty,
    effectiveInferenceSettings.frequency_penalty,
    effectiveInferenceSettings.num_ctx,
    effectiveInferenceSettings.num_predict,
    effectiveInferenceSettings.num_gpu,
    effectiveInferenceSettings.num_thread,
    effectiveInferenceSettings.seed,
    selectedModel
  ]);

  async function loadHostedVersions(personaId: string) {
    if (loadedHostedPersonaIdsRef.current.has(personaId)) {
      return;
    }

    loadedHostedPersonaIdsRef.current.add(personaId);
    setLoadingVersions(true);
    setError(null);
    try {
      const response = await requestJson<PersonaVersionsResponse>(
        `/api/personas/${personaId}/versions`
      );
      onPersonaVersionsLoaded(response.versions);
    } catch (loadError) {
      loadedHostedPersonaIdsRef.current.delete(personaId);
      setError(loadError instanceof Error ? loadError.message : "Failed to load versions");
    } finally {
      setLoadingVersions(false);
    }
  }

  async function loadPrivateVersions(personaId: string) {
    if (loadedPrivatePersonaIdsRef.current.has(personaId)) {
      return;
    }

    loadedPrivatePersonaIdsRef.current.add(personaId);
    setLoadingVersions(true);
    setError(null);
    try {
      const versions = await listPrivatePersonaVersions(personaId);
      onPrivatePersonaVersionsLoaded(versions);
    } catch (loadError) {
      loadedPrivatePersonaIdsRef.current.delete(personaId);
      setError(loadError instanceof Error ? loadError.message : "Failed to load versions");
    } finally {
      setLoadingVersions(false);
    }
  }

  function resetVersion() {
    if (selectedHostedPersona) {
      onModelSelected(personaModelValue(selectedHostedPersona.current_version.id));
      return;
    }

    if (selectedPrivatePersona) {
      onModelSelected(privatePersonaModelValue(selectedPrivatePersona.current_version.id));
    }
  }

  function selectVersion(versionId: string) {
    if (selectedHostedVersion) {
      onModelSelected(personaModelValue(versionId));
    } else {
      onModelSelected(privatePersonaModelValue(versionId));
    }
    setIsVersionMenuOpen(false);
  }

  function updateDraftInferenceSetting(key: InferenceInputKey, value: string) {
    const nextInputs = { ...draftInferenceInputs, [key]: value };
    setDraftInferenceInputs(nextInputs);
    onInferenceSettingsChange?.(inferenceInputsToSettings(nextInputs));
  }

  function resetInferenceSettings() {
    const nextInputs = inferenceSettingsToInputs({});
    setDraftInferenceInputs(nextInputs);
    onInferenceSettingsChange?.({});
  }

  function updateSystemPromptOverride(value: string) {
    setDraftSystemPrompt(value);
    onSystemPromptOverrideChange?.(value === defaultSystemPrompt ? null : value);
  }

  function resetSystemPromptOverride() {
    setDraftSystemPrompt(defaultSystemPrompt);
    onSystemPromptOverrideChange?.(null);
  }

  function renderInferenceField(field: InferenceFieldDefinition) {
    const value = draftInferenceInputs[field.key];
    return (
      <div className="model-settings-number-field" key={field.key}>
        <span className="model-settings-number-label">{field.label}</span>
        <div className="model-settings-input-reset">
          <button
            type="button"
            className="model-settings-help"
            aria-label={`${field.label} help`}
          >
            <Info />
            <span className="model-settings-help-popover">{field.help}</span>
          </button>
          <input
            type="number"
            inputMode={field.inputMode}
            step={field.step}
            min={field.min}
            max={field.max}
            placeholder="Default"
            value={value}
            disabled={!canSaveConversationSettings}
            onChange={(event) => updateDraftInferenceSetting(field.key, event.target.value)}
          />
          <button
            type="button"
            className="icon-button model-settings-field-reset"
            aria-label={`Reset ${field.label} to default`}
            disabled={!canSaveConversationSettings || !value}
            onClick={() => updateDraftInferenceSetting(field.key, "")}
          >
            <RotateCcw />
          </button>
        </div>
      </div>
    );
  }

  function createCustomModelFromSettings() {
    if (!baseModelDraftValue || !onCreateCustomModelFromSettings) {
      return;
    }

    onCreateCustomModelFromSettings({
      displayName: isCustomModel ? `${title} copy` : compactModelName(baseModelName ?? title),
      baseModelValue: baseModelDraftValue,
      systemPrompt: draftSystemPrompt,
      storageMode: selectedPrivateVersion ? "local" : "private"
    });
    setIsOpen(false);
  }

  function isCurrentVersion(version: PersonaVersion | PrivatePersonaVersion) {
    return (
      version.id === selectedHostedPersona?.current_version.id ||
      version.id === selectedPrivatePersona?.current_version.id
    );
  }

  return (
    <div
      className={isOpen ? "model-settings-wrap model-settings-wrap-open" : "model-settings-wrap"}
      ref={wrapRef}
    >
      <button
        type="button"
        className={
          hasConversationSettings
            ? "icon-button model-settings-button model-settings-button-dirty"
            : "icon-button model-settings-button"
        }
        aria-label="Open model settings"
        aria-expanded={isOpen}
        disabled={disabled}
        onClick={() => setIsOpen((open) => !open)}
      >
        <SlidersHorizontal />
      </button>
      {isOpen && (
        <>
          <div
            className="model-settings-backdrop"
            aria-hidden="true"
            onPointerDown={(event) => {
              event.preventDefault();
              event.stopPropagation();
            }}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              setIsVersionMenuOpen(false);
              setIsOpen(false);
            }}
          />
          <div className="model-settings-menu">
            <div className="model-settings-menu-header">
              <div>
                <h2>{title}</h2>
                <p>{subtitle}</p>
              </div>
              {isUsingNonDefaultVersion && (
                <button type="button" className="secondary-button" onClick={resetVersion}>
                  <RotateCcw />
                  <span>Reset</span>
                </button>
              )}
            </div>

            {selectedModelInfo && <ModelCapabilityBadges model={selectedModelInfo} />}
            {baseModelName && (
              <p className="model-settings-meta">
                Base model: <span>{compactModelName(baseModelName)}</span>
              </p>
            )}

            {isCustomModel ? (
              <>
                <div className="model-settings-field">
                  <span>Version</span>
                  <div className="model-settings-version-picker" ref={versionPickerRef}>
                    <button
                      type="button"
                      className="model-settings-version-button"
                      aria-expanded={isVersionMenuOpen}
                      disabled={loadingVersions || displayedVersions.length === 0}
                      onClick={() => setIsVersionMenuOpen((open) => !open)}
                    >
                      <span>{selectedVersionLabel}</span>
                      <ChevronDown />
                    </button>
                    {isVersionMenuOpen && (
                      <div className="model-settings-version-menu">
                        {displayedVersions.map((version) => (
                          <button
                            type="button"
                            key={version.id}
                            className={
                              version.id === selectedVersionId
                                ? "model-settings-version-option model-option-active"
                                : "model-settings-version-option"
                            }
                            onClick={() => selectVersion(version.id)}
                          >
                            <span className="model-option-content">
                              <span className="model-option-title-row">
                                <span className="model-name">{versionOptionLabel(version)}</span>
                                {isCurrentVersion(version) && (
                                  <span className="model-settings-version-current">Current</span>
                                )}
                              </span>
                              <span className="model-subtitle">
                                Created {formatVersionTimestamp(version.created_at)}
                              </span>
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                <p className="model-settings-note">
                  Version changes apply to new messages in this conversation.
                </p>
              </>
            ) : null}

            <details className="model-settings-prompt">
              <summary>
                System prompt
                {isSystemPromptCustomized && <span>customized</span>}
              </summary>
              <textarea
                value={draftSystemPrompt}
                onChange={(event) => updateSystemPromptOverride(event.target.value)}
                placeholder={
                  defaultSystemPrompt
                    ? "Use the default system prompt, or edit it for this chat."
                    : "Add a system prompt for this chat."
                }
                disabled={!canSaveConversationSettings}
              />
              <div className="model-settings-prompt-actions">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={!canSaveConversationSettings || !isSystemPromptCustomized}
                  onClick={resetSystemPromptOverride}
                >
                  <RotateCcw />
                  <span>Reset</span>
                </button>
              </div>
              {!canSaveConversationSettings && (
                <p className="model-settings-note">
                  Start or open a chat to change conversation-specific prompts.
                </p>
              )}
            </details>

            <details className="model-settings-prompt model-settings-inference">
              <summary>
                Inference
                {isInferenceCustomized && <span>customized</span>}
              </summary>
              <div className="model-settings-inference-grid">
                {inferenceFieldDefinitions.map((field) => renderInferenceField(field))}
              </div>
              <p className="model-settings-note">Blank fields use the model/backend default.</p>
              <div className="model-settings-prompt-actions">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={
                    !canSaveConversationSettings ||
                    (!isInferenceCustomized && !inferenceSettingsChanged)
                  }
                  onClick={resetInferenceSettings}
                >
                  <RotateCcw />
                  <span>Reset</span>
                </button>
              </div>
              {!canSaveConversationSettings && (
                <p className="model-settings-note">
                  Start or open a chat to change conversation-specific inference settings.
                </p>
              )}
            </details>

            {loadingVersions && <p className="status-message">Loading versions...</p>}
            {error && <p className="error">{error}</p>}

            {baseModelDraftValue && onCreateCustomModelFromSettings && (
              <button
                type="button"
                className="secondary-button model-settings-wide-action"
                onClick={createCustomModelFromSettings}
              >
                <Plus />
                <span>New custom model from settings</span>
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function uniquePersonaVersions(versions: PersonaVersion[]) {
  return [...new Map(versions.map((version) => [version.id, version])).values()].sort(
    (left, right) => right.version_number - left.version_number
  );
}

function uniquePrivatePersonaVersions(versions: PrivatePersonaVersion[]) {
  return [...new Map(versions.map((version) => [version.id, version])).values()].sort(
    (left, right) => right.version_number - left.version_number
  );
}

function versionOptionLabel(version: PersonaVersion | PrivatePersonaVersion) {
  return `v${version.version_number} · ${version.display_name}`;
}

function formatVersionTimestamp(timestamp: number) {
  return new Date(timestamp * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric"
  });
}

function inferenceSettingsToInputs(settings: ChatInferenceSettings): InferenceInputValues {
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

function inferenceInputsToSettings(inputs: InferenceInputValues): ChatInferenceSettings {
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
