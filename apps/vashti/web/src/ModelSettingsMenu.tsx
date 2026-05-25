import { useEffect, useMemo, useRef, useState } from "react";
import { Plus, RotateCcw, SlidersHorizontal } from "lucide-react";
import { requestJson } from "./api";
import type { CustomModelDraft } from "./customModelDraft";
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
import type { BackendModelGroup, ModelInfo, Persona, PersonaVersion } from "./types";

type PersonaVersionsResponse = {
  versions: PersonaVersion[];
};

export function ModelSettingsMenu({
  groups,
  personas,
  privatePersonas,
  personaVersions,
  privatePersonaVersions,
  selectedModel,
  selectedModelInfo,
  systemPromptOverride,
  canSaveConversationSettings,
  disabled,
  onModelSelected,
  onCreateCustomModelFromSettings,
  onPersonaVersionsLoaded,
  onPrivatePersonaVersionsLoaded,
  onSystemPromptOverrideSave
}: {
  groups: BackendModelGroup[];
  personas: Persona[];
  privatePersonas: PrivatePersona[];
  personaVersions: PersonaVersion[];
  privatePersonaVersions: PrivatePersonaVersion[];
  selectedModel: string;
  selectedModelInfo: ModelInfo | null;
  systemPromptOverride?: string | null;
  canSaveConversationSettings?: boolean;
  disabled: boolean;
  onModelSelected: (value: string) => void;
  onCreateCustomModelFromSettings?: (draft: CustomModelDraft) => void;
  onPersonaVersionsLoaded: (versions: PersonaVersion[]) => void;
  onPrivatePersonaVersionsLoaded: (versions: PrivatePersonaVersion[]) => void;
  onSystemPromptOverrideSave?: (value: string | null) => Promise<void>;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [savingPromptOverride, setSavingPromptOverride] = useState(false);
  const [draftSystemPrompt, setDraftSystemPrompt] = useState("");
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
  const systemPromptChanged = draftSystemPrompt !== effectiveSystemPrompt;
  const hasConversationSettings = isUsingNonDefaultVersion || isSystemPromptCustomized;
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

  async function saveSystemPromptOverride(value: string | null) {
    if (!onSystemPromptOverrideSave) {
      return;
    }

    setSavingPromptOverride(true);
    setError(null);
    try {
      await onSystemPromptOverrideSave(value);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save system prompt");
    } finally {
      setSavingPromptOverride(false);
    }
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
    <div className="model-settings-wrap" ref={wrapRef}>
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
          ) : (
            <p className="model-settings-note">
              Inference settings like context length and temperature will live here.
            </p>
          )}

          <details className="model-settings-prompt">
            <summary>
              System prompt
              {isSystemPromptCustomized && <span>customized</span>}
            </summary>
            <textarea
              value={draftSystemPrompt}
              onChange={(event) => setDraftSystemPrompt(event.target.value)}
              placeholder={
                defaultSystemPrompt
                  ? "Use the default system prompt, or edit it for this chat."
                  : "Add a system prompt for this chat."
              }
              disabled={!canSaveConversationSettings || savingPromptOverride}
            />
            <div className="model-settings-prompt-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={
                  !canSaveConversationSettings ||
                  savingPromptOverride ||
                  (!isSystemPromptCustomized && !systemPromptChanged)
                }
                onClick={() => {
                  setDraftSystemPrompt(defaultSystemPrompt);
                  void saveSystemPromptOverride(null);
                }}
              >
                <RotateCcw />
                <span>Reset</span>
              </button>
              <button
                type="button"
                disabled={
                  !canSaveConversationSettings || savingPromptOverride || !systemPromptChanged
                }
                onClick={() => void saveSystemPromptOverride(draftSystemPrompt)}
              >
                Save Prompt
              </button>
            </div>
            {!canSaveConversationSettings && (
              <p className="model-settings-note">
                Start or open a chat to save conversation-specific prompts.
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
