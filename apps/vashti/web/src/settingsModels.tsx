import {
  FormEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState
} from "react";
import {
  ChevronDown,
  Pencil,
  Power,
  RefreshCw,
  Save,
  Star,
  Trash2
} from "lucide-react";
import { requestJson } from "./api";
import { RetroLoader } from "./common";
import { ModelPicker } from "./ModelPicker";
import { ModelCapabilityBadges } from "./modelCapabilities";
import { compactModelName, modelValue } from "./modelSelection";
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

type UserModelPreferencePatch = {
  is_visible?: boolean;
  is_favorite?: boolean;
  is_default?: boolean;
};

type UserModelPreferenceAction = "visible" | "favorite" | "default";
type PendingFavoriteRemovalTimers = {
  timeout: number;
  interval: number;
};

function preferenceActionForPatch(patch: UserModelPreferencePatch): UserModelPreferenceAction {
  if (patch.is_favorite !== undefined) {
    return "favorite";
  }
  if (patch.is_visible !== undefined) {
    return "visible";
  }
  return "default";
}

export function UserModelsPanel({ onModelsChanged }: { onModelsChanged: () => Promise<void> }) {
  const [groups, setGroups] = useState<UserBackendModelGroup[]>([]);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [busyPreference, setBusyPreference] = useState<{
    key: string;
    action: UserModelPreferenceAction;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const primaryModelRowRefs = useRef(new Map<string, HTMLElement>());
  const pendingFavoriteAnchorRef = useRef<{
    value: string;
    top: number;
    scrollParent: HTMLElement | null;
  } | null>(null);
  const pendingFavoriteRemovalTimersRef = useRef(
    new Map<string, PendingFavoriteRemovalTimers>()
  );
  const [pendingFavoriteRemovals, setPendingFavoriteRemovals] = useState<Record<string, number>>(
    {}
  );
  const modelOptions = groups.flatMap((group) =>
    group.models.map((model) => ({
      backendId: group.backend.id,
      backendName: group.backend.name,
      model,
      value: modelValue(group.backend.id, model.name)
    }))
  );
  const visibleModelOptions = modelOptions.filter((option) => option.model.is_visible);
  const favoriteModelOptions = modelOptions.filter((option) => option.model.is_favorite);
  const defaultModelOption =
    modelOptions.find((option) => option.model.is_default) ?? null;
  const defaultModelValue = defaultModelOption?.value ?? "";
  const visibleModelGroups = groups
    .map((group) => ({
      backend: group.backend,
      models: group.models.filter((model) => model.is_visible)
    }))
    .filter((group) => group.models.length > 0);

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

  useLayoutEffect(() => {
    const pendingAnchor = pendingFavoriteAnchorRef.current;
    if (!pendingAnchor) {
      return;
    }

    const element = primaryModelRowRefs.current.get(pendingAnchor.value);
    pendingFavoriteAnchorRef.current = null;
    if (!element) {
      return;
    }

    const delta = element.getBoundingClientRect().top - pendingAnchor.top;
    if (Math.abs(delta) < 0.5) {
      return;
    }

    if (pendingAnchor.scrollParent) {
      pendingAnchor.scrollParent.scrollTop += delta;
    } else {
      window.scrollBy(0, delta);
    }
  }, [groups]);

  useEffect(
    () => () => {
      for (const timers of pendingFavoriteRemovalTimersRef.current.values()) {
        window.clearTimeout(timers.timeout);
        window.clearInterval(timers.interval);
      }
      pendingFavoriteRemovalTimersRef.current.clear();
    },
    []
  );

  async function updateUserModelPreference(
    backendId: string,
    modelName: string,
    patch: UserModelPreferencePatch
  ) {
    const key = modelValue(backendId, modelName);
    const action = preferenceActionForPatch(patch);
    setBusyPreference({ key, action });
    setError(null);

    try {
      const response = await requestJson<{
        backend_id: string;
        model_name: string;
        is_visible: boolean;
        is_favorite: boolean;
        is_default: boolean;
      }>("/api/user-models", {
        method: "PATCH",
        body: JSON.stringify({
          backend_id: backendId,
          model_name: modelName,
          ...patch
        })
      });
      setGroups((current) =>
        current.map((group) =>
          group.backend.id === response.backend_id
            ? {
                ...group,
                models: group.models.map((model) =>
                  model.name === response.model_name
                    ? {
                        ...model,
                        is_visible: response.is_visible,
                        is_favorite: response.is_favorite,
                        is_default: response.is_default
                      }
                    : {
                        ...model,
                        is_default: response.is_default ? false : model.is_default
                      }
                )
              }
            : {
                ...group,
                models: group.models.map((model) => ({
                  ...model,
                  is_default: response.is_default ? false : model.is_default
                }))
              }
        )
      );
      await onModelsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update model");
    } finally {
      setBusyPreference((current) =>
        current?.key === key && current.action === action ? null : current
      );
    }
  }

  async function selectDefaultModel(nextValue: string) {
    if (nextValue === defaultModelValue) {
      return;
    }

    if (!nextValue) {
      if (defaultModelOption) {
        await updateUserModelPreference(
          defaultModelOption.backendId,
          defaultModelOption.model.name,
          { is_default: false }
        );
      }
      return;
    }

    const selected = modelOptions.find((option) => option.value === nextValue);
    if (!selected) {
      return;
    }

    await updateUserModelPreference(selected.backendId, selected.model.name, {
      is_default: true
    });
  }

  function anchorPrimaryModelRow(value: string) {
    const element = primaryModelRowRefs.current.get(value);
    if (!element) {
      pendingFavoriteAnchorRef.current = null;
      return;
    }

    pendingFavoriteAnchorRef.current = {
      value,
      top: element.getBoundingClientRect().top,
      scrollParent: element.closest<HTMLElement>(".settings-content")
    };
  }

  function clearPendingFavoriteRemoval(value: string) {
    const timers = pendingFavoriteRemovalTimersRef.current.get(value);
    if (timers) {
      window.clearTimeout(timers.timeout);
      window.clearInterval(timers.interval);
      pendingFavoriteRemovalTimersRef.current.delete(value);
    }
    setPendingFavoriteRemovals((current) => {
      const next = { ...current };
      delete next[value];
      return next;
    });
  }

  function scheduleFavoriteRemoval(option: (typeof modelOptions)[number]) {
    clearPendingFavoriteRemoval(option.value);
    setPendingFavoriteRemovals((current) => ({ ...current, [option.value]: 5 }));

    const interval = window.setInterval(() => {
      setPendingFavoriteRemovals((current) => {
        if (current[option.value] === undefined) {
          return current;
        }
        return {
          ...current,
          [option.value]: Math.max(1, current[option.value] - 1)
        };
      });
    }, 1000);

    const timeout = window.setTimeout(() => {
      window.clearInterval(interval);
      pendingFavoriteRemovalTimersRef.current.delete(option.value);
      setPendingFavoriteRemovals((current) => {
        const next = { ...current };
        delete next[option.value];
        return next;
      });
      void updateUserModelPreference(option.backendId, option.model.name, {
        is_favorite: false
      });
    }, 5000);

    pendingFavoriteRemovalTimersRef.current.set(option.value, { timeout, interval });
  }

  function renderModelRow(
    option: (typeof modelOptions)[number],
    {
      keyPrefix = "model",
      showBackend = false
    }: {
      keyPrefix?: string;
      showBackend?: boolean;
    } = {}
  ) {
    const key = `${keyPrefix}:${option.value}`;
    const isFavoriteBusy =
      busyPreference?.key === option.value && busyPreference.action === "favorite";
    const isVisibilityBusy =
      busyPreference?.key === option.value && busyPreference.action === "visible";
    const pendingFavoriteRemovalSeconds = pendingFavoriteRemovals[option.value];
    const isFavoriteRemovalPending = pendingFavoriteRemovalSeconds !== undefined;
    const { model } = option;

    return (
      <article
        key={key}
        className={
          isFavoriteRemovalPending
            ? "model-access-row model-access-row-pending"
            : "model-access-row"
        }
        ref={
          keyPrefix === "model"
            ? (node) => {
                if (node) {
                  primaryModelRowRefs.current.set(option.value, node);
                } else {
                  primaryModelRowRefs.current.delete(option.value);
                }
              }
            : undefined
        }
      >
        {isFavoriteRemovalPending ? (
          <button
            type="button"
            className="secondary-button model-undo-button model-undo-remove-button"
            onClick={() => clearPendingFavoriteRemoval(option.value)}
          >
            <span>Undo Remove</span>
            <span className="model-undo-count">{pendingFavoriteRemovalSeconds}</span>
          </button>
        ) : (
          <button
            type="button"
            className={
              model.is_favorite
                ? "model-pref-button model-favorite-button model-pref-button-active"
                : "model-pref-button model-favorite-button"
            }
            title={model.is_favorite ? "Remove from favorites" : "Add to favorites"}
            disabled={isFavoriteBusy}
            onClick={() => {
              if (keyPrefix === "favorite" && model.is_favorite) {
                scheduleFavoriteRemoval(option);
                return;
              }
              if (keyPrefix === "model") {
                anchorPrimaryModelRow(option.value);
              }
              void updateUserModelPreference(option.backendId, model.name, {
                is_favorite: !model.is_favorite
              });
            }}
          >
            <Star />
          </button>
        )}
        <span className="model-access-main">
          <span className="model-name" title={model.name}>
            {compactModelName(model.name)}
          </span>
          {showBackend && <span className="model-subtitle">{option.backendName}</span>}
          <ModelCapabilityBadges model={model} />
        </span>
        <div className="model-row-actions">
          <ToggleSwitch
            label={model.is_visible ? "Shown" : "Hidden"}
            checked={model.is_visible}
            disabled={isVisibilityBusy}
            compact
            onChange={(checked) =>
              void updateUserModelPreference(option.backendId, model.name, {
                is_visible: checked
              })
            }
          />
        </div>
      </article>
    );
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

      <section className="settings-subsection default-model-panel">
        <div>
          <h2>Default Model</h2>
          <p className="status-message">
            Used after refresh, or when starting a chat from the home screen.
          </p>
        </div>
        <div className="default-model-picker">
          <ModelPicker
            groups={visibleModelGroups}
            personas={[]}
            privatePersonas={[]}
            isLoading={!hasLoaded}
            error={visibleModelOptions.length === 0 ? "No visible models" : null}
            value={defaultModelValue}
            onChange={(nextValue) => void selectDefaultModel(nextValue)}
          />
        </div>
      </section>

      <div className="model-access-list">
        {favoriteModelOptions.length > 0 && (
          <section className="model-access-group model-access-favorites">
            <div className="model-access-header">
              <div>
                <h2>Favorites</h2>
                <p>{favoriteModelOptions.length} favorite models</p>
              </div>
            </div>
            <div className="model-access-models">
              {favoriteModelOptions.map((option) =>
                renderModelRow(option, { keyPrefix: "favorite", showBackend: true })
              )}
            </div>
          </section>
        )}
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
                  {group.models.map((model) =>
                    renderModelRow({
                      backendId: group.backend.id,
                      backendName: group.backend.name,
                      model,
                      value: modelValue(group.backend.id, model.name)
                    })
                  )}
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
      await onModelsChanged();
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : "Failed to update models");
    } finally {
      setBusyModelsBackendId(null);
    }
  }

  function updateModelTags(backendId: string, modelName: string, tags: PermissionTag[]) {
    setError(null);
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
  }

  return (
    <>
      {error && <p className="error">{error}</p>}
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
              setSaveStatus(null);
            }}
          />
          <div className="default-apply-mode">
            <button
              type="button"
              className={defaultTagApplyMode === "new" ? "segmented-option-active" : ""}
              onClick={() => {
                setDefaultTagApplyMode("new");
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
                              <span className="model-name" title={model.name}>
                                {compactModelName(model.name)}
                              </span>
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
