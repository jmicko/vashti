import {
  FormEvent,
  useCallback,
  useEffect,
  useState
} from "react";
import {
  Copy,
  Lock,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  X
} from "lucide-react";
import { requestBlob, requestJson } from "./api";
import {
  deleteHostedAvatar as deleteHostedPersonaAvatar,
  hostedAvatarFile as hostedPersonaAvatarFile,
  uploadHostedAvatar as uploadHostedPersonaAvatar
} from "./avatarAssets";
import { ConfirmDialog, RetroLoader } from "./common";
import { takeCustomModelDraft, type CustomModelDraft } from "./customModelDraft";
import { ModelAvatar } from "./ModelAvatar";
import { ModelBackgroundButton } from "./ModelBackground";
import {
  ModelBackgroundEditorDialog,
  type ModelBackgroundEdit
} from "./ModelBackgroundEditorDialog";
import { modelParts, modelValue } from "./modelSelection";
import { PersonaAvatarField } from "./PersonaAvatarField";
import {
  createPrivatePersona,
  deleteUnusedPrivatePersonaAvatar,
  deletePrivatePersona,
  getPrivatePersonaAvatar,
  listPrivatePersonas,
  savePrivatePersonaAvatar,
  updatePrivatePersona,
  type PrivatePersona
} from "./privateChatStore";
import { backendNameFor, firstModelValue } from "./settingsModelHelpers";
import type {
  BackendModelGroup,
  ModelsResponse,
  Persona,
  PersonaVersion,
  PersonaMutationResponse,
  PersonasResponse,
  ModelBackgroundMode,
  CustomModelType
} from "./types";

export function CustomModelsSection({
  onPersonasChanged,
  onPrivatePersonasChanged
}: {
  onPersonasChanged: () => Promise<void>;
  onPrivatePersonasChanged: () => Promise<void>;
}) {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [privatePersonas, setPrivatePersonas] = useState<PrivatePersona[]>([]);
  const [modelGroups, setModelGroups] = useState<BackendModelGroup[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [busyPersonaId, setBusyPersonaId] = useState<string | null>(null);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [editingPersona, setEditingPersona] = useState<Persona | null>(null);
  const [editingPrivatePersona, setEditingPrivatePersona] = useState<PrivatePersona | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Persona | null>(null);
  const [deletePrivateTarget, setDeletePrivateTarget] = useState<PrivatePersona | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [modelType, setModelType] = useState<CustomModelType>("general");
  const [storageMode, setStorageMode] = useState("local");
  const [selectedBaseModel, setSelectedBaseModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [avatarAssetId, setAvatarAssetId] = useState<string | null>(null);
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [isReadingAvatar, setIsReadingAvatar] = useState(false);
  const [avatarCropX, setAvatarCropX] = useState(50);
  const [avatarCropY, setAvatarCropY] = useState(50);
  const [avatarCropSize, setAvatarCropSize] = useState(100);
  const [backgroundAssetId, setBackgroundAssetId] = useState<string | null>(null);
  const [backgroundFile, setBackgroundFile] = useState<File | null>(null);
  const [backgroundDim, setBackgroundDim] = useState(0.72);
  const [backgroundMessageDim, setBackgroundMessageDim] = useState(0.82);
  const [backgroundLandscape, setBackgroundLandscape] = useState<{
    mode: ModelBackgroundMode;
    x: number;
    y: number;
    scale: number;
  }>({
    mode: "fill",
    x: 50,
    y: 50,
    scale: 35
  });
  const [backgroundPortrait, setBackgroundPortrait] = useState<{
    mode: ModelBackgroundMode;
    x: number;
    y: number;
    scale: number;
  }>({
    mode: "fill",
    x: 50,
    y: 50,
    scale: 35
  });
  const [isBackgroundEditorOpen, setIsBackgroundEditorOpen] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadPersonas = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const [personaResponse, privatePersonaResponse, modelsResponse] = await Promise.all([
        requestJson<PersonasResponse>("/api/personas"),
        listPrivatePersonas(),
        requestJson<ModelsResponse>("/api/models")
      ]);
      setPersonas(personaResponse.personas);
      setPrivatePersonas(privatePersonaResponse);
      setModelGroups(modelsResponse.backends);
      setHasLoaded(true);
      setSelectedBaseModel((current) => current || firstModelValue(modelsResponse.backends));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load custom models");
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadPersonas();
  }, [loadPersonas]);

  function resetBackground() {
    setBackgroundAssetId(null);
    setBackgroundFile(null);
    setBackgroundDim(0.72);
    setBackgroundMessageDim(0.82);
    setBackgroundLandscape({ mode: "fill", x: 50, y: 50, scale: 35 });
    setBackgroundPortrait({ mode: "fill", x: 50, y: 50, scale: 35 });
    setIsBackgroundEditorOpen(false);
  }

  function loadBackground(version: PersonaVersion | PrivatePersona["current_version"]) {
    setBackgroundAssetId(version.background_asset_id ?? null);
    setBackgroundFile(null);
    setBackgroundDim(version.background_dim ?? 0.72);
    setBackgroundMessageDim(version.background_message_dim ?? 0.82);
    setBackgroundLandscape({
      mode: (version.background_landscape_mode ?? "fill") as ModelBackgroundMode,
      x: version.background_landscape_x ?? 50,
      y: version.background_landscape_y ?? 50,
      scale: version.background_landscape_scale ?? 35
    });
    setBackgroundPortrait({
      mode: (version.background_portrait_mode ?? "fill") as ModelBackgroundMode,
      x: version.background_portrait_x ?? 50,
      y: version.background_portrait_y ?? 50,
      scale: version.background_portrait_scale ?? 35
    });
    setIsBackgroundEditorOpen(false);
  }

  function resetDraft() {
    setIsEditorOpen(false);
    setEditingPersona(null);
    setEditingPrivatePersona(null);
    setDisplayName("");
    setModelType("general");
    setStorageMode("local");
    setSystemPrompt("");
    setAvatarAssetId(null);
    setAvatarFile(null);
    setAvatarCropX(50);
    setAvatarCropY(50);
    setAvatarCropSize(100);
    resetBackground();
    setSelectedBaseModel(firstModelValue(modelGroups));
    setError(null);
    setStatus(null);
  }

  function startCreatingPersona(draft?: CustomModelDraft) {
    setEditingPersona(null);
    setEditingPrivatePersona(null);
    setDisplayName(draft?.displayName ?? "");
    setModelType(draft?.modelType ?? "general");
    setStorageMode(draft?.storageMode ?? "local");
    setSelectedBaseModel(
      draft?.baseModelValue && modelParts(draft.baseModelValue)
        ? draft.baseModelValue
        : firstModelValue(modelGroups)
    );
    setSystemPrompt(draft?.systemPrompt ?? "");
    setAvatarAssetId(null);
    setAvatarFile(null);
    setAvatarCropX(50);
    setAvatarCropY(50);
    setAvatarCropSize(100);
    resetBackground();
    setError(null);
    setStatus(null);
    setIsEditorOpen(true);
  }

  useEffect(() => {
    if (!hasLoaded) {
      return;
    }

    const draft = takeCustomModelDraft();
    if (draft) {
      startCreatingPersona(draft);
    }
  }, [hasLoaded, modelGroups]);

  function startEditingPersona(persona: Persona) {
    const version = persona.current_version;
    setIsEditorOpen(true);
    setEditingPersona(persona);
    setEditingPrivatePersona(null);
    setDisplayName(version.display_name);
    setModelType(version.model_type ?? "general");
    setStorageMode(persona.visibility);
    setSelectedBaseModel(modelValue(version.base_backend_id, version.base_model_name));
    setSystemPrompt(version.system_prompt);
    setAvatarAssetId(version.avatar_asset_id ?? null);
    setAvatarFile(null);
    setAvatarCropX(version.avatar_crop_x ?? 50);
    setAvatarCropY(version.avatar_crop_y ?? 50);
    setAvatarCropSize(version.avatar_crop_size ?? 100);
    loadBackground(version);
    setError(null);
    setStatus(null);
  }

  function startEditingPrivatePersona(persona: PrivatePersona) {
    const version = persona.current_version;
    setIsEditorOpen(true);
    setEditingPrivatePersona(persona);
    setEditingPersona(null);
    setDisplayName(version.display_name);
    setModelType(version.model_type ?? "general");
    setStorageMode("local");
    setSelectedBaseModel(modelValue(version.base_backend_id, version.base_model_name));
    setSystemPrompt(version.system_prompt);
    setAvatarAssetId(version.avatar_asset_id ?? null);
    setAvatarFile(null);
    setAvatarCropX(version.avatar_crop_x ?? 50);
    setAvatarCropY(version.avatar_crop_y ?? 50);
    setAvatarCropSize(version.avatar_crop_size ?? 100);
    loadBackground(version);
    setError(null);
    setStatus(null);
  }

  async function savePersona(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const selected = modelParts(selectedBaseModel);
    if (!selected) {
      setError("Select a base model for this persona");
      return;
    }

    setIsSaving(true);
    setError(null);
    setStatus(null);

    let uploadedAssetId: string | null = null;
    let uploadedBackgroundAssetId: string | null = null;
    try {
      const backendName = backendNameFor(modelGroups, selected.backendId);
      let nextAvatarAssetId = avatarAssetId;
      let avatarUploadFile = avatarFile;
      if (!avatarUploadFile && avatarAssetId && storageMode === "local" && editingPersona) {
        avatarUploadFile = await hostedPersonaAvatarFile(avatarAssetId);
      }
      if (!avatarUploadFile && avatarAssetId && storageMode !== "local" && editingPrivatePersona) {
        avatarUploadFile = await privatePersonaAvatarFile(avatarAssetId);
      }
      if (avatarUploadFile) {
        if (storageMode === "local") {
          const asset = await savePrivatePersonaAvatar(avatarUploadFile);
          nextAvatarAssetId = asset.id;
          uploadedAssetId = asset.id;
        } else {
          const asset = await uploadHostedPersonaAvatar(avatarUploadFile);
          nextAvatarAssetId = asset.id;
          uploadedAssetId = asset.id;
        }
      }
      let nextBackgroundAssetId = backgroundAssetId;
      let backgroundUploadFile = backgroundFile;
      if (!backgroundUploadFile && backgroundAssetId && storageMode === "local" && editingPersona) {
        backgroundUploadFile = await hostedPersonaAvatarFile(backgroundAssetId);
      }
      if (
        !backgroundUploadFile &&
        backgroundAssetId &&
        storageMode !== "local" &&
        editingPrivatePersona
      ) {
        backgroundUploadFile = await privatePersonaAvatarFile(backgroundAssetId);
      }
      if (backgroundUploadFile) {
        if (storageMode === "local") {
          const asset = await savePrivatePersonaAvatar(backgroundUploadFile);
          nextBackgroundAssetId = asset.id;
          uploadedBackgroundAssetId = asset.id;
        } else {
          const asset = await uploadHostedPersonaAvatar(backgroundUploadFile);
          nextBackgroundAssetId = asset.id;
          uploadedBackgroundAssetId = asset.id;
        }
      }

      if (storageMode === "local") {
        const body = {
          displayName,
          modelType,
          baseBackendId: selected.backendId,
          baseBackendName: backendName,
          baseModelName: selected.modelName,
          systemPrompt,
          avatarAssetId: nextAvatarAssetId,
          avatarCropX,
          avatarCropY,
          avatarCropSize,
          backgroundAssetId: nextBackgroundAssetId,
          backgroundDim,
          backgroundMessageDim,
          backgroundLandscapeMode: backgroundLandscape.mode,
          backgroundLandscapeX: backgroundLandscape.x,
          backgroundLandscapeY: backgroundLandscape.y,
          backgroundLandscapeScale: backgroundLandscape.scale,
          backgroundPortraitMode: backgroundPortrait.mode,
          backgroundPortraitX: backgroundPortrait.x,
          backgroundPortraitY: backgroundPortrait.y,
          backgroundPortraitScale: backgroundPortrait.scale
        };
        if (editingPrivatePersona) {
          await updatePrivatePersona(editingPrivatePersona.id, body);
          setStatus("Device custom model updated.");
        } else {
          await createPrivatePersona(body);
          setStatus("Device custom model created.");
        }
        resetDraft();
        await loadPersonas();
        await onPrivatePersonasChanged();
        return;
      }

      const body = {
        visibility: storageMode,
        display_name: displayName,
        model_type: modelType,
        avatar_asset_id: nextAvatarAssetId,
        avatar_asset_changed: editingPersona
          ? nextAvatarAssetId !== (editingPersona.current_version.avatar_asset_id ?? null)
          : undefined,
        avatar_crop_x: avatarCropX,
        avatar_crop_y: avatarCropY,
        avatar_crop_size: avatarCropSize,
        background: {
          asset_id: nextBackgroundAssetId,
          asset_changed: editingPersona
            ? nextBackgroundAssetId !==
              (editingPersona.current_version.background_asset_id ?? null)
            : true,
          dim: backgroundDim,
          message_dim: backgroundMessageDim,
          landscape_mode: backgroundLandscape.mode,
          landscape_x: backgroundLandscape.x,
          landscape_y: backgroundLandscape.y,
          landscape_scale: backgroundLandscape.scale,
          portrait_mode: backgroundPortrait.mode,
          portrait_x: backgroundPortrait.x,
          portrait_y: backgroundPortrait.y,
          portrait_scale: backgroundPortrait.scale
        },
        base_backend_id: selected.backendId,
        base_model_name: selected.modelName,
        system_prompt: systemPrompt,
        tool_policy_json: null
      };
      if (editingPersona) {
        await requestJson<PersonaMutationResponse>(`/api/personas/${editingPersona.id}`, {
          method: "PATCH",
          body: JSON.stringify(body)
        });
        setStatus("Custom model updated.");
      } else {
        await requestJson<PersonaMutationResponse>("/api/personas", {
          method: "POST",
          body: JSON.stringify(body)
        });
        setStatus("Custom model created.");
      }
      resetDraft();
      await loadPersonas();
      await onPersonasChanged();
    } catch (saveError) {
      if (uploadedAssetId) {
        if (storageMode === "local") {
          await deleteUnusedPrivatePersonaAvatar(uploadedAssetId).catch(() => undefined);
        } else {
          await deleteHostedPersonaAvatar(uploadedAssetId).catch(() => undefined);
        }
      }
      if (uploadedBackgroundAssetId) {
        if (storageMode === "local") {
          await deleteUnusedPrivatePersonaAvatar(uploadedBackgroundAssetId).catch(() => undefined);
        } else {
          await deleteHostedPersonaAvatar(uploadedBackgroundAssetId).catch(() => undefined);
        }
      }
      setError(saveError instanceof Error ? saveError.message : "Failed to save custom model");
    } finally {
      setIsSaving(false);
    }
  }

  async function copyPersona(persona: Persona) {
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    try {
      await requestJson<PersonaMutationResponse>(`/api/personas/${persona.id}/copy`, {
        method: "POST",
        body: JSON.stringify({
          persona_version_id: persona.current_version.id,
          visibility: "private"
        })
      });
      setStatus("Custom model copied to your private server models.");
      await loadPersonas();
      await onPersonasChanged();
    } catch (copyError) {
      setError(copyError instanceof Error ? copyError.message : "Failed to copy persona");
    } finally {
      setBusyPersonaId(null);
    }
  }

  async function copyPersonaToDevice(persona: Persona) {
    const version = persona.current_version;
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    let uploadedAvatarAssetId: string | null = null;
    let uploadedBackgroundAssetId: string | null = null;
    try {
      let avatarAssetId: string | null = null;
      if (version.avatar_asset_id) {
        const blob = await requestBlob(
          `/api/persona-avatars/${encodeURIComponent(version.avatar_asset_id)}`
        );
        const file = new File([blob], "profile-image", { type: blob.type });
        avatarAssetId = (await savePrivatePersonaAvatar(file)).id;
        uploadedAvatarAssetId = avatarAssetId;
      }
      let backgroundAssetId: string | null = null;
      if (version.background_asset_id) {
        const blob = await requestBlob(
          `/api/persona-avatars/${encodeURIComponent(version.background_asset_id)}`
        );
        const file = new File([blob], "chat-background", { type: blob.type });
        backgroundAssetId = (await savePrivatePersonaAvatar(file)).id;
        uploadedBackgroundAssetId = backgroundAssetId;
      }
      await createPrivatePersona({
        displayName: version.display_name,
        modelType: version.model_type ?? "general",
        baseBackendId: version.base_backend_id,
        baseBackendName: backendNameFor(modelGroups, version.base_backend_id),
        baseModelName: version.base_model_name,
        systemPrompt: version.system_prompt,
        avatarAssetId,
        avatarCropX: version.avatar_crop_x,
        avatarCropY: version.avatar_crop_y,
        avatarCropSize: version.avatar_crop_size,
        backgroundAssetId,
        backgroundDim: version.background_dim,
        backgroundMessageDim: version.background_message_dim,
        backgroundLandscapeMode: version.background_landscape_mode,
        backgroundLandscapeX: version.background_landscape_x,
        backgroundLandscapeY: version.background_landscape_y,
        backgroundLandscapeScale: version.background_landscape_scale,
        backgroundPortraitMode: version.background_portrait_mode,
        backgroundPortraitX: version.background_portrait_x,
        backgroundPortraitY: version.background_portrait_y,
        backgroundPortraitScale: version.background_portrait_scale,
        sourcePersonaId: persona.id,
        sourcePersonaVersionId: version.id
      });
      setStatus("Custom model copied to this device for private chats.");
      await loadPersonas();
      await onPrivatePersonasChanged();
    } catch (copyError) {
      if (uploadedAvatarAssetId) {
        await deleteUnusedPrivatePersonaAvatar(uploadedAvatarAssetId).catch(() => undefined);
      }
      if (uploadedBackgroundAssetId) {
        await deleteUnusedPrivatePersonaAvatar(uploadedBackgroundAssetId).catch(() => undefined);
      }
      setError(copyError instanceof Error ? copyError.message : "Failed to copy persona to device");
    } finally {
      setBusyPersonaId(null);
    }
  }

  async function disownPersona(persona: Persona) {
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    try {
      await requestJson(`/api/personas/${persona.id}/disown`, { method: "POST" });
      setDeleteTarget(null);
      setStatus(persona.visibility === "public" ? "Custom model removed." : "Custom model deleted.");
      if (editingPersona?.id === persona.id) {
        resetDraft();
      }
      await loadPersonas();
      await onPersonasChanged();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Failed to remove custom model");
      setDeleteTarget(null);
    } finally {
      setBusyPersonaId(null);
    }
  }

  async function deletePrivatePersonaTarget(persona: PrivatePersona) {
    setBusyPersonaId(persona.id);
    setError(null);
    setStatus(null);

    try {
      await deletePrivatePersona(persona.id);
      setDeletePrivateTarget(null);
      setStatus("Device custom model deleted.");
      if (editingPrivatePersona?.id === persona.id) {
        resetDraft();
      }
      await loadPersonas();
      await onPrivatePersonasChanged();
    } catch (deleteError) {
      setError(
        deleteError instanceof Error ? deleteError.message : "Failed to delete device custom model"
      );
      setDeletePrivateTarget(null);
    } finally {
      setBusyPersonaId(null);
    }
  }

  const canSave =
    !isReadingAvatar &&
    displayName.trim() !== "" &&
    selectedBaseModel !== "" &&
    !(editingPersona && storageMode === "local") &&
    !(editingPrivatePersona && storageMode !== "local");

  const editorTitle =
    editingPersona || editingPrivatePersona ? "Edit Custom Model" : "Create Custom Model";

  return (
    <section className="model-access-group custom-models-section">
      <div className="model-access-header">
        <div>
          <h2>Custom Models</h2>
          <p>
            {isEditorOpen
              ? editorTitle
              : `${privatePersonas.length + personas.length} custom model${
                  privatePersonas.length + personas.length === 1 ? "" : "s"
                }`}
          </p>
        </div>
        <div className="model-access-actions">
          {isEditorOpen ? (
            <button
              type="button"
              className="secondary-button refresh-button"
              disabled={isSaving || isReadingAvatar}
              onClick={resetDraft}
            >
              <X />
              <span>Cancel</span>
            </button>
          ) : (
            <>
              <button
                type="button"
                className="secondary-button refresh-button"
                onClick={() => void loadPersonas()}
                disabled={isRefreshing}
              >
                {isRefreshing ? <RetroLoader /> : <RefreshCw />}
                <span>{isRefreshing ? "Loading" : "Refresh"}</span>
              </button>
              <button type="button" onClick={() => startCreatingPersona()}>
                <Plus />
                <span>New</span>
              </button>
            </>
          )}
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {status && <p className="status-message">{status}</p>}

      {isEditorOpen ? (
        <form className="settings-form persona-form persona-editor-form" onSubmit={savePersona}>
          <h3>{editorTitle}</h3>
          <label>
            <span>Name</span>
            <input
              required
              maxLength={80}
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              placeholder="Careful Researcher"
            />
          </label>
          <div className="custom-model-type-field">
            <span>Type</span>
            <div className="segmented-control custom-model-type-control">
              <button
                type="button"
                className={modelType === "general" ? "active" : undefined}
                aria-pressed={modelType === "general"}
                onClick={() => setModelType("general")}
              >
                General
              </button>
              <button
                type="button"
                className={modelType === "character" ? "active" : undefined}
                aria-pressed={modelType === "character"}
                onClick={() => setModelType("character")}
              >
                Character
              </button>
            </div>
          </div>
          <label>
            <span>Storage</span>
            <select value={storageMode} onChange={(event) => setStorageMode(event.target.value)}>
              <option value="local">This device only</option>
              <option value="private">Server private</option>
              <option value="public">Server public</option>
            </select>
          </label>
          <label>
            <span>Base Model</span>
            <select
              required
              value={selectedBaseModel}
              onChange={(event) => setSelectedBaseModel(event.target.value)}
            >
              <option value="">Select a model</option>
              {modelGroups.map((group) => (
                <optgroup key={group.backend.id} label={group.backend.name}>
                  {group.models.map((model) => (
                    <option
                      key={modelValue(group.backend.id, model.name)}
                      value={modelValue(group.backend.id, model.name)}
                    >
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <PersonaAvatarField
            displayName={displayName}
            assetId={storageMode === "local" ? null : avatarAssetId}
            privateAssetId={storageMode === "local" ? avatarAssetId : null}
            previewFile={avatarFile}
            cropX={avatarCropX}
            cropY={avatarCropY}
            cropSize={avatarCropSize}
            onFileChange={(file) => {
              setError(null);
              setAvatarFile(file);
              setAvatarCropX(50);
              setAvatarCropY(50);
              setAvatarCropSize(100);
            }}
            onReadingChange={setIsReadingAvatar}
            onRemove={() => {
              setAvatarAssetId(null);
              setAvatarFile(null);
              setAvatarCropX(50);
              setAvatarCropY(50);
              setAvatarCropSize(100);
            }}
            onCropChange={(cropX, cropY, cropSize) => {
              setAvatarCropX(cropX);
              setAvatarCropY(cropY);
              setAvatarCropSize(cropSize);
            }}
          />
          <div className="persona-background-field">
            <div>
              <span>Chat Background</span>
              <p>Shown behind conversations that use this custom model.</p>
            </div>
            <ModelBackgroundButton
              assetId={storageMode === "local" ? null : backgroundAssetId}
              privateAssetId={storageMode === "local" ? backgroundAssetId : null}
              previewFile={backgroundFile}
              label="Change custom model chat background"
              onClick={() => setIsBackgroundEditorOpen(true)}
            />
          </div>
          <label>
            <span>System Prompt</span>
            <textarea
              rows={10}
              value={systemPrompt}
              onChange={(event) => setSystemPrompt(event.target.value)}
              placeholder="Describe how this custom model should behave."
            />
          </label>
          <div className="persona-form-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={isSaving || isReadingAvatar}
              onClick={resetDraft}
            >
              <X />
              <span>Cancel</span>
            </button>
            <button type="submit" disabled={isSaving || !canSave}>
              {editingPersona || editingPrivatePersona ? <Save /> : <Plus />}
              <span>
                {isSaving
                  ? "Saving..."
                  : isReadingAvatar
                    ? "Reading image..."
                  : editingPersona || editingPrivatePersona
                    ? "Save Custom Model"
                    : "Create Custom Model"}
              </span>
            </button>
          </div>
          <p className="status-message">
            Device custom models can be used in private chats without storing their name or prompt
            on the server. Server custom models are available from signed-in standard chats.
          </p>
        </form>
      ) : (
        <>
          {!hasLoaded && <p className="status-message">Loading custom models...</p>}
          {hasLoaded && personas.length === 0 && privatePersonas.length === 0 && (
            <p className="status-message">No custom models created yet.</p>
          )}

          <div className="persona-list">
            {privatePersonas.map((persona) => (
              <PrivatePersonaRow
                key={persona.id}
                persona={persona}
                isBusy={busyPersonaId === persona.id}
                onDelete={() => setDeletePrivateTarget(persona)}
                onEdit={() => startEditingPrivatePersona(persona)}
              />
            ))}
            {personas.map((persona) => (
              <PersonaRow
                key={persona.id}
                persona={persona}
                backendName={backendNameFor(modelGroups, persona.current_version.base_backend_id)}
                isBusy={busyPersonaId === persona.id}
                canEdit={persona.is_owner && persona.lifecycle_state === "active"}
                onCopy={() => void copyPersona(persona)}
                onCopyToDevice={() => void copyPersonaToDevice(persona)}
                onDelete={() => setDeleteTarget(persona)}
                onEdit={() => startEditingPersona(persona)}
              />
            ))}
          </div>
        </>
      )}

      {deleteTarget && (
        <ConfirmDialog
          title={
            deleteTarget.visibility === "public" ? "Remove Custom Model" : "Delete Custom Model"
          }
          message={
            deleteTarget.visibility === "public"
              ? `Remove "${deleteTarget.current_version.display_name}" from your available custom models? If other users have used it, their existing chats keep working.`
              : `Delete "${deleteTarget.current_version.display_name}"? Existing chats keep their stored message labels, but this custom model will disappear from your picker.`
          }
          confirmLabel={deleteTarget.visibility === "public" ? "Remove" : "Delete"}
          isBusy={busyPersonaId === deleteTarget.id}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void disownPersona(deleteTarget)}
        />
      )}
      {deletePrivateTarget && (
        <ConfirmDialog
          title="Delete Device Custom Model"
          message={`Delete "${deletePrivateTarget.current_version.display_name}" from this device? Server chats and hosted custom models are not affected.`}
          confirmLabel="Delete"
          isBusy={busyPersonaId === deletePrivateTarget.id}
          onCancel={() => setDeletePrivateTarget(null)}
          onConfirm={() => void deletePrivatePersonaTarget(deletePrivateTarget)}
        />
      )}
      {isBackgroundEditorOpen && (
        <ModelBackgroundEditorDialog
          title={`Background for ${displayName.trim() || "custom model"}`}
          assetId={storageMode === "local" ? null : backgroundAssetId}
          privateAssetId={storageMode === "local" ? backgroundAssetId : null}
          dim={backgroundDim}
          messageDim={backgroundMessageDim}
          landscape={backgroundLandscape}
          portrait={backgroundPortrait}
          isBusy={false}
          error={null}
          onCancel={() => setIsBackgroundEditorOpen(false)}
          onSave={(edit: ModelBackgroundEdit) => {
            setBackgroundAssetId(edit.assetId);
            setBackgroundFile(edit.file);
            setBackgroundDim(edit.dim);
            setBackgroundMessageDim(edit.messageDim);
            setBackgroundLandscape(edit.landscape);
            setBackgroundPortrait(edit.portrait);
            setIsBackgroundEditorOpen(false);
          }}
        />
      )}
    </section>
  );
}

function PersonaRow({
  persona,
  backendName,
  isBusy,
  canEdit,
  onCopy,
  onCopyToDevice,
  onDelete,
  onEdit
}: {
  persona: Persona;
  backendName: string;
  isBusy: boolean;
  canEdit: boolean;
  onCopy: () => void;
  onCopyToDevice: () => void;
  onDelete: () => void;
  onEdit: () => void;
}) {
  const version = persona.current_version;
  return (
    <article className="persona-row">
      <div className="persona-main">
        <ModelAvatar
          displayName={version.display_name}
          assetId={version.avatar_asset_id}
          cropX={version.avatar_crop_x}
          cropY={version.avatar_crop_y}
          cropSize={version.avatar_crop_size}
          className="model-avatar-persona-row"
        />
        <div className="persona-main-copy">
          <h2>{version.display_name}</h2>
          <p>
            {backendName} / {version.base_model_name}
          </p>
          {persona.owner_username && <p>by {persona.owner_username}</p>}
        </div>
        <div className="badges">
          <span className="badge">custom</span>
          {version.model_type === "character" && <span className="badge">character</span>}
          <span className={persona.visibility === "public" ? "badge" : "badge badge-warning"}>
            {persona.visibility}
          </span>
          <span className="badge">v{version.version_number}</span>
          {persona.is_owner && <span className="badge">yours</span>}
        </div>
      </div>
      <details className="persona-prompt">
        <summary>System prompt</summary>
        <pre>{version.system_prompt || "No system prompt."}</pre>
      </details>
      <div className="persona-actions">
        {canEdit && (
          <button type="button" className="secondary-button" disabled={isBusy} onClick={onEdit}>
            <Pencil />
            <span>Edit</span>
          </button>
        )}
        <button type="button" className="secondary-button" disabled={isBusy} onClick={onCopy}>
          <Copy />
          <span>Copy</span>
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={isBusy}
          onClick={onCopyToDevice}
        >
          <Lock />
          <span>Device</span>
        </button>
        {(persona.is_owner || persona.is_member) && (
          <button type="button" className="danger-button" disabled={isBusy} onClick={onDelete}>
            <Trash2 />
            <span>{persona.visibility === "public" ? "Remove" : "Delete"}</span>
          </button>
        )}
      </div>
    </article>
  );
}

function PrivatePersonaRow({
  persona,
  isBusy,
  onDelete,
  onEdit
}: {
  persona: PrivatePersona;
  isBusy: boolean;
  onDelete: () => void;
  onEdit: () => void;
}) {
  const version = persona.current_version;
  return (
    <article className="persona-row">
      <div className="persona-main">
        <ModelAvatar
          displayName={version.display_name}
          privateAssetId={version.avatar_asset_id}
          cropX={version.avatar_crop_x}
          cropY={version.avatar_crop_y}
          cropSize={version.avatar_crop_size}
          className="model-avatar-persona-row"
        />
        <div className="persona-main-copy">
          <h2>{version.display_name}</h2>
          <p>
            {version.base_backend_name} / {version.base_model_name}
          </p>
          <p>stored on this device</p>
        </div>
        <div className="badges">
          <span className="badge">custom</span>
          {version.model_type === "character" && <span className="badge">character</span>}
          <span className="badge badge-warning">device</span>
          <span className="badge">v{version.version_number}</span>
        </div>
      </div>
      <details className="persona-prompt">
        <summary>System prompt</summary>
        <pre>{version.system_prompt || "No system prompt."}</pre>
      </details>
      <div className="persona-actions">
        <button type="button" className="secondary-button" disabled={isBusy} onClick={onEdit}>
          <Pencil />
          <span>Edit</span>
        </button>
        <button type="button" className="danger-button" disabled={isBusy} onClick={onDelete}>
          <Trash2 />
          <span>Delete</span>
        </button>
      </div>
    </article>
  );
}

async function privatePersonaAvatarFile(assetId: string) {
  const asset = await getPrivatePersonaAvatar(assetId);
  if (!asset) {
    throw new Error("Profile image is not available on this device");
  }
  return new File([dataUrlBytes(asset.data_url)], asset.original_filename, {
    type: asset.mime_type
  });
}

function dataUrlBytes(dataUrl: string) {
  const separator = dataUrl.indexOf(",");
  if (separator < 0) {
    throw new Error("Stored profile image is invalid");
  }
  const binary = atob(dataUrl.slice(separator + 1));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
