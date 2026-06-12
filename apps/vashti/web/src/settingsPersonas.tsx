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
import { requestJson, responseErrorMessage } from "./api";
import { ConfirmDialog, RetroLoader } from "./common";
import { takeCustomModelDraft, type CustomModelDraft } from "./customModelDraft";
import { ModelAvatar } from "./ModelAvatar";
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
  PersonaMutationResponse,
  PersonasResponse
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
  const [storageMode, setStorageMode] = useState("local");
  const [selectedBaseModel, setSelectedBaseModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [avatarAssetId, setAvatarAssetId] = useState<string | null>(null);
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [avatarCropX, setAvatarCropX] = useState(50);
  const [avatarCropY, setAvatarCropY] = useState(50);
  const [avatarCropSize, setAvatarCropSize] = useState(100);
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

  function resetDraft() {
    setIsEditorOpen(false);
    setEditingPersona(null);
    setEditingPrivatePersona(null);
    setDisplayName("");
    setStorageMode("local");
    setSystemPrompt("");
    setAvatarAssetId(null);
    setAvatarFile(null);
    setAvatarCropX(50);
    setAvatarCropY(50);
    setAvatarCropSize(100);
    setSelectedBaseModel(firstModelValue(modelGroups));
    setError(null);
    setStatus(null);
  }

  function startCreatingPersona(draft?: CustomModelDraft) {
    setEditingPersona(null);
    setEditingPrivatePersona(null);
    setDisplayName(draft?.displayName ?? "");
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
    setStorageMode(persona.visibility);
    setSelectedBaseModel(modelValue(version.base_backend_id, version.base_model_name));
    setSystemPrompt(version.system_prompt);
    setAvatarAssetId(version.avatar_asset_id ?? null);
    setAvatarFile(null);
    setAvatarCropX(version.avatar_crop_x ?? 50);
    setAvatarCropY(version.avatar_crop_y ?? 50);
    setAvatarCropSize(version.avatar_crop_size ?? 100);
    setError(null);
    setStatus(null);
  }

  function startEditingPrivatePersona(persona: PrivatePersona) {
    const version = persona.current_version;
    setIsEditorOpen(true);
    setEditingPrivatePersona(persona);
    setEditingPersona(null);
    setDisplayName(version.display_name);
    setStorageMode("local");
    setSelectedBaseModel(modelValue(version.base_backend_id, version.base_model_name));
    setSystemPrompt(version.system_prompt);
    setAvatarAssetId(version.avatar_asset_id ?? null);
    setAvatarFile(null);
    setAvatarCropX(version.avatar_crop_x ?? 50);
    setAvatarCropY(version.avatar_crop_y ?? 50);
    setAvatarCropSize(version.avatar_crop_size ?? 100);
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

      if (storageMode === "local") {
        const body = {
          displayName,
          baseBackendId: selected.backendId,
          baseBackendName: backendName,
          baseModelName: selected.modelName,
          systemPrompt,
          avatarAssetId: nextAvatarAssetId,
          avatarCropX,
          avatarCropY,
          avatarCropSize
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
        avatar_asset_id: nextAvatarAssetId,
        avatar_asset_changed: editingPersona
          ? nextAvatarAssetId !== (editingPersona.current_version.avatar_asset_id ?? null)
          : undefined,
        avatar_crop_x: avatarCropX,
        avatar_crop_y: avatarCropY,
        avatar_crop_size: avatarCropSize,
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
    try {
      let avatarAssetId: string | null = null;
      if (version.avatar_asset_id) {
        const response = await fetch(
          `/api/persona-avatars/${encodeURIComponent(version.avatar_asset_id)}`,
          { credentials: "include" }
        );
        if (!response.ok) {
          throw new Error(await responseErrorMessage(response));
        }
        const blob = await response.blob();
        const file = new File([blob], "profile-image", { type: blob.type });
        avatarAssetId = (await savePrivatePersonaAvatar(file)).id;
        uploadedAvatarAssetId = avatarAssetId;
      }
      await createPrivatePersona({
        displayName: version.display_name,
        baseBackendId: version.base_backend_id,
        baseBackendName: backendNameFor(modelGroups, version.base_backend_id),
        baseModelName: version.base_model_name,
        systemPrompt: version.system_prompt,
        avatarAssetId,
        avatarCropX: version.avatar_crop_x,
        avatarCropY: version.avatar_crop_y,
        avatarCropSize: version.avatar_crop_size,
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
              disabled={isSaving}
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
              disabled={isSaving}
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

type HostedPersonaAvatarAsset = {
  id: string;
  original_filename: string;
  mime_type: string;
  size_bytes: number;
  created_at: number;
};

async function uploadHostedPersonaAvatar(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/persona-avatars", {
    method: "POST",
    credentials: "include",
    body: formData
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }

  return ((await response.json()) as { asset: HostedPersonaAvatarAsset }).asset;
}

async function deleteHostedPersonaAvatar(assetId: string) {
  const response = await fetch(`/api/persona-avatars/${encodeURIComponent(assetId)}`, {
    method: "DELETE",
    credentials: "include"
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
}

async function hostedPersonaAvatarFile(assetId: string) {
  const response = await fetch(`/api/persona-avatars/${encodeURIComponent(assetId)}`, {
    credentials: "include"
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
  const blob = await response.blob();
  return new File([blob], "profile-image", { type: blob.type });
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
