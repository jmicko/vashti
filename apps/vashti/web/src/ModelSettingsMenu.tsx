import { useEffect, useMemo, useRef, useState } from "react";
import { Plus, RotateCcw, SlidersHorizontal } from "lucide-react";
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
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [draftSystemPrompt, setDraftSystemPrompt] = useState("");
  const [draftInferenceInputs, setDraftInferenceInputs] = useState<InferenceInputValues>(() =>
    inferenceSettingsToInputs({})
  );
  const [error, setError] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
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

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

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
    effectiveInferenceSettings.top_p,
    effectiveInferenceSettings.repeat_penalty,
    effectiveInferenceSettings.num_ctx,
    effectiveInferenceSettings.num_predict,
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

  function renderInferenceField(
    key: InferenceInputKey,
    label: string,
    options: {
      inputMode: "decimal" | "numeric";
      step?: string;
      min?: string;
      max?: string;
    }
  ) {
    const value = draftInferenceInputs[key];
    return (
      <div className="model-settings-number-field">
        <span>{label}</span>
        <div className="model-settings-input-reset">
          <input
            type="number"
            inputMode={options.inputMode}
            step={options.step}
            min={options.min}
            max={options.max}
            placeholder="Default"
            value={value}
            disabled={!canSaveConversationSettings}
            onChange={(event) => updateDraftInferenceSetting(key, event.target.value)}
          />
          <button
            type="button"
            className="icon-button model-settings-field-reset"
            aria-label={`Reset ${label} to default`}
            disabled={!canSaveConversationSettings || !value}
            onClick={() => updateDraftInferenceSetting(key, "")}
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
                <label className="model-settings-field">
                  <span>Version</span>
                  <select
                    value={selectedHostedVersion?.id ?? selectedPrivateVersion?.id ?? ""}
                    disabled={loadingVersions}
                    onChange={(event) => {
                      if (selectedHostedVersion) {
                        onModelSelected(personaModelValue(event.target.value));
                      } else {
                        onModelSelected(privatePersonaModelValue(event.target.value));
                      }
                    }}
                  >
                    {(selectedHostedVersion ? hostedVersions : privateVersions).map((version) => (
                      <option key={version.id} value={version.id}>
                        v{version.version_number}
                        {version.id === selectedHostedPersona?.current_version.id ||
                        version.id === selectedPrivatePersona?.current_version.id
                          ? " current"
                          : ""}
                        {" · "}
                        {version.display_name}
                      </option>
                    ))}
                  </select>
                </label>
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
                {renderInferenceField("temperature", "Temperature", {
                  inputMode: "decimal",
                  step: "0.1",
                  min: "0",
                  max: "2"
                })}
                {renderInferenceField("num_ctx", "Context", {
                  inputMode: "numeric",
                  step: "1",
                  min: "512"
                })}
                {renderInferenceField("top_p", "Top P", {
                  inputMode: "decimal",
                  step: "0.05",
                  min: "0.01",
                  max: "1"
                })}
                {renderInferenceField("repeat_penalty", "Repeat penalty", {
                  inputMode: "decimal",
                  step: "0.05",
                  min: "0.5",
                  max: "2"
                })}
                {renderInferenceField("num_predict", "Max output", {
                  inputMode: "numeric",
                  step: "1",
                  min: "1"
                })}
                {renderInferenceField("seed", "Seed", {
                  inputMode: "numeric",
                  step: "1"
                })}
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

function inferenceSettingsToInputs(settings: ChatInferenceSettings): InferenceInputValues {
  return {
    temperature: inferenceInputValue(settings.temperature),
    top_p: inferenceInputValue(settings.top_p),
    repeat_penalty: inferenceInputValue(settings.repeat_penalty),
    num_ctx: inferenceInputValue(settings.num_ctx),
    num_predict: inferenceInputValue(settings.num_predict),
    seed: inferenceInputValue(settings.seed)
  };
}

function inferenceInputsToSettings(inputs: InferenceInputValues): ChatInferenceSettings {
  return normalizeInferenceSettings({
    temperature: parsedInferenceNumber(inputs.temperature),
    top_p: parsedInferenceNumber(inputs.top_p),
    repeat_penalty: parsedInferenceNumber(inputs.repeat_penalty),
    num_ctx: parsedInferenceInteger(inputs.num_ctx),
    num_predict: parsedInferenceInteger(inputs.num_predict),
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
