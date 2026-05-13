import {
  FormEvent,
  useCallback,
  useEffect,
  useState
} from "react";
import {
  ChevronDown,
  Pencil,
  Power,
  RefreshCw,
  Save,
  Trash2
} from "lucide-react";
import { requestJson } from "./api";
import { RetroLoader } from "./common";
import { ModelCapabilityBadges } from "./modelCapabilities";
import { modelValue } from "./modelSelection";
import { DefaultPermissionTagControls, PermissionTagEditor } from "./permissionTags";
import { SettingsSaveBanner, ToggleSwitch } from "./settingsControls";
import {
  adminModelGroupTagsEqual,
  collectChangedAdminModelTagPatches,
  permissionTagPayload,
  permissionTagSetsEqual,
  updateAdminModelTagsInGroups
} from "./settingsModelHelpers";
import type {
  AdminBackendModelGroup,
  AdminModelsResponse,
  Backend,
  PermissionTag,
  UserBackendModelGroup,
  UserModelsResponse
} from "./types";

export function UserModelsPanel({ onModelsChanged }: { onModelsChanged: () => Promise<void> }) {
  const [groups, setGroups] = useState<UserBackendModelGroup[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [busyModelKey, setBusyModelKey] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const applyUserModelsResponse = useCallback((response: UserModelsResponse) => {
    setGroups(response.backends);
    setHasLoaded(true);
  }, []);

  const loadUserModels = useCallback(async () => {
    setError(null);

    try {
      const response = await requestJson<UserModelsResponse>("/api/user-models");
      applyUserModelsResponse(response);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load models");
      setHasLoaded(true);
    }
  }, [applyUserModelsResponse]);

  const refreshUserModels = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const response = await requestJson<UserModelsResponse>("/api/user-models/refresh", {
        method: "POST"
      });
      applyUserModelsResponse(response);
      await onModelsChanged();
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to refresh models");
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, [applyUserModelsResponse, onModelsChanged]);

  useEffect(() => {
    void (async () => {
      setIsRefreshing(true);
      await loadUserModels();
      await refreshUserModels();
    })();
  }, [loadUserModels, refreshUserModels]);

  async function toggleUserModel(backendId: string, modelName: string, isVisible: boolean) {
    const key = modelValue(backendId, modelName);
    setBusyModelKey(key);
    setError(null);
    setStatus(null);

    try {
      await requestJson("/api/user-models", {
        method: "PATCH",
        body: JSON.stringify({
          backend_id: backendId,
          model_name: modelName,
          is_visible: isVisible
        })
      });
      setGroups((current) =>
        current.map((group) =>
          group.backend.id === backendId
            ? {
                ...group,
                models: group.models.map((model) =>
                  model.name === modelName ? { ...model, is_visible: isVisible } : model
                )
              }
            : group
        )
      );
      setStatus(isVisible ? "Model shown in picker." : "Model hidden from picker.");
      await onModelsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update model");
    } finally {
      setBusyModelKey(null);
    }
  }

  return (
    <div className="settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Personal</p>
          <h1>Models</h1>
        </div>
        <button
          type="button"
          className="secondary-button refresh-button"
          onClick={() => void refreshUserModels()}
          disabled={isRefreshing}
        >
          {isRefreshing ? (
            <RetroLoader />
          ) : (
            <>
              <RefreshCw />
              <span>Refresh</span>
            </>
          )}
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      {status && <p className="status-message">{status}</p>}
      {!hasLoaded && <p className="status-message">Loading models...</p>}
      {hasLoaded && groups.length === 0 && isRefreshing && (
        <p className="status-message">Checking Ollama for models...</p>
      )}
      {hasLoaded && groups.length === 0 && !isRefreshing && (
        <p className="status-message">No models are currently available to your account.</p>
      )}

      <p className="status-message">
        These switches control which models appear in your picker. Admin model and tag permissions
        still decide which models your account can access.
      </p>

      <div className="model-access-list">
        {groups.map((group) => {
          const visibleCount = group.models.filter((model) => model.is_visible).length;

          return (
            <section key={group.backend.id} className="model-access-group">
              <div className="model-access-header">
                <div>
                  <h2>{group.backend.name}</h2>
                  <p>
                    {visibleCount} of {group.models.length} shown
                  </p>
                </div>
              </div>

              {group.models.length === 0 ? (
                <p className="status-message">No available models on this backend.</p>
              ) : (
                <div className="model-access-models">
                  {group.models.map((model) => {
                    const key = modelValue(group.backend.id, model.name);
                    const isBusy = busyModelKey === key;

                    return (
                      <article key={key} className="model-access-row">
                        <span className="model-access-main">
                          <span className="model-name">{model.name}</span>
                          <ModelCapabilityBadges model={model} />
                        </span>
                        <ToggleSwitch
                          label={model.is_visible ? "Shown" : "Hidden"}
                          checked={model.is_visible}
                          disabled={isBusy}
                          compact
                          onChange={(checked) =>
                            void toggleUserModel(group.backend.id, model.name, checked)
                          }
                        />
                      </article>
                    );
                  })}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}

export function AdminModelsAccessPanel({
  backends,
  editingBackend,
  backendBusyId,
  onCancelBackendEdit,
  onDeleteBackend,
  onEditBackend,
  onSaveBackend,
  onToggleBackend,
  modelRefreshRequest,
  onModelRefreshStateChange,
  onModelsChanged
}: {
  backends: Backend[];
  editingBackend: Backend | null;
  backendBusyId: string | null;
  onCancelBackendEdit: () => void;
  onDeleteBackend: (backend: Backend) => void;
  onEditBackend: (backend: Backend) => void;
  onSaveBackend: (
    backendId: string,
    body: Partial<Pick<Backend, "name" | "base_url" | "is_enabled">>
  ) => Promise<void>;
  onToggleBackend: (backendId: string, nextEnabled: boolean) => void;
  modelRefreshRequest: number;
  onModelRefreshStateChange: (isRefreshing: boolean) => void;
  onModelsChanged: () => Promise<void>;
}) {
  const [groups, setGroups] = useState<AdminBackendModelGroup[]>([]);
  const [savedGroups, setSavedGroups] = useState<AdminBackendModelGroup[]>([]);
  const [availableTags, setAvailableTags] = useState<PermissionTag[]>([]);
  const [defaultTags, setDefaultTags] = useState<PermissionTag[]>([]);
  const [savedDefaultTags, setSavedDefaultTags] = useState<PermissionTag[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [busyModelKey, setBusyModelKey] = useState<string | null>(null);
  const [busyModelsBackendId, setBusyModelsBackendId] = useState<string | null>(null);
  const [isSavingModelTags, setIsSavingModelTags] = useState(false);
  const [defaultTagApplyMode, setDefaultTagApplyMode] = useState<"new" | "all">("new");
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const applyAdminModelsResponse = useCallback((response: AdminModelsResponse) => {
    setGroups(response.backends);
    setSavedGroups(response.backends);
    setAvailableTags(response.available_tags);
    setDefaultTags(response.default_permission_tags);
    setSavedDefaultTags(response.default_permission_tags);
    setHasLoaded(true);
  }, []);

  const hasUnsavedModelTagChanges =
    defaultTagApplyMode !== "new" ||
    !permissionTagSetsEqual(defaultTags, savedDefaultTags) ||
    !adminModelGroupTagsEqual(groups, savedGroups);

  const loadAdminModels = useCallback(async () => {
    setError(null);

    try {
      const response = await requestJson<AdminModelsResponse>("/api/admin/models");
      applyAdminModelsResponse(response);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load models");
      setHasLoaded(true);
    }
  }, [applyAdminModelsResponse]);

  const refreshAdminModels = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);

    try {
      const response = await requestJson<AdminModelsResponse>("/api/admin/models/refresh", {
        method: "POST"
      });
      applyAdminModelsResponse(response);
      await onModelsChanged();
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to refresh models");
      setHasLoaded(true);
    } finally {
      setIsRefreshing(false);
    }
  }, [applyAdminModelsResponse, onModelsChanged]);

  useEffect(() => {
    void (async () => {
      setIsRefreshing(true);
      await loadAdminModels();
      await refreshAdminModels();
    })();
  }, [loadAdminModels, refreshAdminModels]);

  useEffect(() => {
    onModelRefreshStateChange(isRefreshing);
  }, [isRefreshing, onModelRefreshStateChange]);

  useEffect(() => {
    if (modelRefreshRequest > 0) {
      void refreshAdminModels();
    }
  }, [modelRefreshRequest, refreshAdminModels]);

  useEffect(() => {
    if (!saveStatus) {
      return;
    }

    const timeout = window.setTimeout(() => setSaveStatus(null), 3000);
    return () => window.clearTimeout(timeout);
  }, [saveStatus]);

  useEffect(() => {
    if (hasUnsavedModelTagChanges && saveStatus) {
      setSaveStatus(null);
    }
  }, [hasUnsavedModelTagChanges, saveStatus]);

  async function toggleModel(backendId: string, modelName: string, isEnabled: boolean) {
    const key = modelValue(backendId, modelName);
    setBusyModelKey(key);
    setError(null);
    setStatus(null);

    try {
      await requestJson("/api/admin/models", {
        method: "PATCH",
        body: JSON.stringify({
          backend_id: backendId,
          model_name: modelName,
          is_enabled: isEnabled
        })
      });
      setGroups((current) =>
        current.map((group) =>
          group.backend.id === backendId
            ? {
                ...group,
                models: group.models.map((model) =>
                  model.name === modelName ? { ...model, is_enabled: isEnabled } : model
                )
              }
            : group
        )
      );
      setStatus(isEnabled ? "Model enabled." : "Model disabled.");
      await onModelsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update model");
    } finally {
      setBusyModelKey(null);
    }
  }

  async function toggleBackend(group: AdminBackendModelGroup, isEnabled: boolean) {
    setBusyModelsBackendId(group.backend.id);
    setError(null);
    setStatus(null);

    try {
      await requestJson("/api/admin/models/backend", {
        method: "PATCH",
        body: JSON.stringify({
          backend_id: group.backend.id,
          model_names: group.models.map((model) => model.name),
          is_enabled: isEnabled
        })
      });
      setGroups((current) =>
        current.map((currentGroup) =>
          currentGroup.backend.id === group.backend.id
            ? {
                ...currentGroup,
                models: currentGroup.models.map((model) => ({
                  ...model,
                  is_enabled: isEnabled
                }))
              }
            : currentGroup
        )
      );
      setStatus(isEnabled ? "Backend models enabled." : "Backend models disabled.");
      await onModelsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update models");
    } finally {
      setBusyModelsBackendId(null);
    }
  }

  function updateModelTags(backendId: string, modelName: string, tags: PermissionTag[]) {
    setError(null);
    setStatus(null);
    setSaveStatus(null);
    setGroups((current) =>
      updateAdminModelTagsInGroups(current, backendId, modelName, { permission_tags: tags })
    );
  }

  function updateModelDefaultTags(
    backendId: string,
    modelName: string,
    tags: PermissionTag[]
  ) {
    setError(null);
    setStatus(null);
    setSaveStatus(null);
    setGroups((current) =>
      updateAdminModelTagsInGroups(current, backendId, modelName, {
        default_permission_tags: tags
      })
    );
  }

  async function saveModelTagChanges() {
    setIsSavingModelTags(true);
    setError(null);
    setStatus(null);
    setSaveStatus(null);

    try {
      const applyToExisting = defaultTagApplyMode === "all";
      const defaultTagsChanged = !permissionTagSetsEqual(defaultTags, savedDefaultTags);
      const modelTagPatches = collectChangedAdminModelTagPatches(
        groups,
        savedGroups,
        !applyToExisting
      );

      if (defaultTagsChanged || applyToExisting) {
        await requestJson<{ default_permission_tags: PermissionTag[] }>(
          "/api/admin/models/default-tags",
          {
            method: "PATCH",
            body: JSON.stringify({
              permission_tags: permissionTagPayload(defaultTags),
              apply_to_existing: applyToExisting
            })
          }
        );
      }

      for (const patch of modelTagPatches) {
        await requestJson<{
          permission_tags: PermissionTag[];
          default_permission_tags: PermissionTag[];
        }>("/api/admin/models/tags", {
          method: "PATCH",
          body: JSON.stringify(patch)
        });
      }

      await loadAdminModels();
      await onModelsChanged();
      setDefaultTagApplyMode("new");
      setSaveStatus("Model tag settings saved.");
    } catch (updateError) {
      setError(
        updateError instanceof Error ? updateError.message : "Failed to update model tag settings"
      );
    } finally {
      setIsSavingModelTags(false);
    }
  }

  function revertModelTagChanges() {
    setDefaultTags(savedDefaultTags);
    setGroups(savedGroups);
    setDefaultTagApplyMode("new");
    setSaveStatus(null);
    setError(null);
    setStatus(null);
  }

  return (
    <>
      {error && <p className="error">{error}</p>}
      {status && <p className="status-message">{status}</p>}
      {!hasLoaded && <p className="status-message">Loading models...</p>}

      <SettingsSaveBanner
        isDirty={hasUnsavedModelTagChanges}
        status={saveStatus}
        dirtyTitle="Unsaved model tag changes"
        dirtyDescription="Save tag edits, or apply current defaults to existing models."
        savedDescription="Saved changes are active for future generations."
      >
        <button
          type="button"
          className="secondary-button"
          disabled={isSavingModelTags}
          onClick={revertModelTagChanges}
        >
          Revert
        </button>
        <button
          type="button"
          disabled={isSavingModelTags}
          onClick={() => void saveModelTagChanges()}
        >
          <Save />
          <span>{isSavingModelTags ? "Saving..." : "Save"}</span>
        </button>
      </SettingsSaveBanner>

      <section className="settings-subsection permission-defaults backend-model-defaults">
        <div>
          <p className="eyebrow">Defaults</p>
          <h2>Default Model Tags</h2>
          <p className="status-message">
            New models get these tags the first time Vashti sees them. Empty means no one can use
            new models until tags are added. Existing model changes update only the default-tag layer
            and keep admin tags intact.
          </p>
        </div>
        <div className="permission-defaults-row">
          <PermissionTagEditor
            label="Default tags"
            tags={defaultTags}
            availableTags={availableTags}
            disabled={isSavingModelTags}
            onChange={(tags) => {
              setDefaultTags(tags);
              setStatus(null);
              setSaveStatus(null);
            }}
          />
          <div className="default-apply-mode">
            <button
              type="button"
              className={defaultTagApplyMode === "new" ? "segmented-option-active" : ""}
              onClick={() => {
                setDefaultTagApplyMode("new");
                setStatus(null);
                setSaveStatus(null);
              }}
            >
              New models only
            </button>
            <button
              type="button"
              className={defaultTagApplyMode === "all" ? "segmented-option-active" : ""}
              onClick={() => {
                setDefaultTagApplyMode("all");
                setStatus(null);
                setSaveStatus(null);
              }}
            >
              Existing models too
            </button>
          </div>
        </div>
      </section>

      <div className="backend-list">
        {backends.map((backend) => {
          if (editingBackend?.id === backend.id) {
            return (
              <BackendEditRow
                key={backend.id}
                backend={editingBackend}
                isBusy={backendBusyId === backend.id}
                onCancel={onCancelBackendEdit}
                onSave={onSaveBackend}
              />
            );
          }

          const group = groups.find((candidate) => candidate.backend.id === backend.id);
          const models = group?.models ?? [];
          const enabledCount = models.filter((model) => model.is_enabled).length;
          const isModelsBackendBusy = busyModelsBackendId === backend.id;
          const isBackendBusy = backendBusyId === backend.id;

          return (
            <details key={backend.id} className="backend-row backend-model-group">
              <summary className="backend-model-summary">
                <div className="backend-main">
                  <div>
                    <h2>{backend.name}</h2>
                    <p>{backend.base_url}</p>
                    {backend.last_error && <p className="backend-error">{backend.last_error}</p>}
                  </div>
                  <div className="badges">
                    <span className={backend.is_enabled ? "badge" : "badge badge-warning"}>
                      {backend.is_enabled ? "enabled" : "disabled"}
                    </span>
                    <span
                      className={
                        backend.last_health_status === "error" ? "badge badge-warning" : "badge"
                      }
                    >
                      {backend.last_health_status ?? "unknown"}
                    </span>
                    {backend.is_enabled && (
                      <span className="badge">
                        {enabledCount} / {models.length} models
                      </span>
                    )}
                  </div>
                </div>
                <div className="backend-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={isBackendBusy}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onToggleBackend(backend.id, !backend.is_enabled);
                    }}
                  >
                    <Power />
                    <span>{backend.is_enabled ? "Disable" : "Enable"}</span>
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={isBackendBusy}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onEditBackend(backend);
                    }}
                  >
                    <Pencil />
                    <span>Edit</span>
                  </button>
                  <button
                    type="button"
                    className="danger-button"
                    disabled={isBackendBusy}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      onDeleteBackend(backend);
                    }}
                  >
                    <Trash2 />
                    <span>Delete</span>
                  </button>
                  <span className="backend-model-toggle">
                    <ChevronDown />
                    <span className="backend-model-toggle-show">Models</span>
                    <span className="backend-model-toggle-hide">Models</span>
                  </span>
                </div>
              </summary>

              <div className="backend-model-panel">
                {!backend.is_enabled ? (
                  <p className="status-message">Enable this backend to manage its models.</p>
                ) : !group && isRefreshing ? (
                  <p className="status-message">Checking Ollama for models...</p>
                ) : !group ? (
                  <p className="status-message">No model data loaded for this backend yet.</p>
                ) : models.length === 0 ? (
                  <p className="status-message">No models returned by this backend.</p>
                ) : (
                  <>
                    <div className="model-access-actions backend-model-bulk-actions">
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={isModelsBackendBusy || models.length === 0}
                        onClick={() => void toggleBackend(group, true)}
                      >
                        Enable All
                      </button>
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={isModelsBackendBusy || models.length === 0}
                        onClick={() => void toggleBackend(group, false)}
                      >
                        Disable All
                      </button>
                    </div>
                    <div className="model-access-models">
                      {models.map((model) => {
                        const key = modelValue(backend.id, model.name);
                        const isBusy = busyModelKey === key || isModelsBackendBusy;

                        return (
                          <article key={key} className="model-access-row">
                            <span className="model-access-main">
                              <span className="model-name">{model.name}</span>
                              <ModelCapabilityBadges model={model} />
                              <DefaultPermissionTagControls
                                defaultTags={savedDefaultTags}
                                activeTags={model.default_permission_tags}
                                disabled={isBusy || isSavingModelTags}
                                onChange={(tags) =>
                                  updateModelDefaultTags(backend.id, model.name, tags)
                                }
                              />
                              <PermissionTagEditor
                                tags={model.permission_tags}
                                availableTags={availableTags}
                                disabled={isBusy || isSavingModelTags}
                                showEmpty={false}
                                onChange={(tags) => updateModelTags(backend.id, model.name, tags)}
                              />
                            </span>
                            <ToggleSwitch
                              label={model.is_enabled ? "On" : "Off"}
                              checked={model.is_enabled}
                              disabled={isBusy}
                              compact
                              onChange={(checked) => void toggleModel(backend.id, model.name, checked)}
                            />
                          </article>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            </details>
          );
        })}
      </div>
    </>
  );
}

function BackendEditRow({
  backend,
  isBusy,
  onCancel,
  onSave
}: {
  backend: Backend;
  isBusy: boolean;
  onCancel: () => void;
  onSave: (
    backendId: string,
    body: Partial<Pick<Backend, "name" | "base_url" | "is_enabled">>
  ) => Promise<void>;
}) {
  const [name, setName] = useState(backend.name);
  const [baseUrl, setBaseUrl] = useState(backend.base_url);
  const [isEnabled, setIsEnabled] = useState(backend.is_enabled);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onSave(backend.id, { name, base_url: baseUrl, is_enabled: isEnabled });
  }

  return (
    <article className="backend-row">
      <form className="backend-edit-form" onSubmit={submit}>
        <div className="backend-edit-grid">
          <label>
            <span>Name</span>
            <input required value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <label>
            <span>Base URL</span>
            <input
              required
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
            />
          </label>
        </div>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={isEnabled}
            onChange={(event) => setIsEnabled(event.target.checked)}
          />
          <span>Enabled</span>
        </label>
        <div className="backend-actions">
          <button type="button" className="secondary-button" disabled={isBusy} onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" disabled={isBusy}>
            <Save />
            <span>{isBusy ? "Saving..." : "Save Backend"}</span>
          </button>
        </div>
      </form>
    </article>
  );
}
