import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  ArrowLeft,
  BrainCircuit,
  Clock3,
  ExternalLink,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  Trash2
} from "lucide-react";
import { ConfirmDialog } from "./common";
import {
  NoteModelScopeEditor,
  buildNoteModelOptions,
  normalizeNoteModelScope
} from "./notes/accessControls";
import {
  createMemory,
  forgetMemory,
  getMemory,
  getMemorySettings,
  listMemories,
  listMemoryVersions,
  permanentlyDeleteMemory,
  restoreMemory,
  restoreMemoryVersion,
  updateMemory,
  updateMemorySettings
} from "./memories/api";
import type {
  Memory,
  MemoryModelScope,
  MemorySettings,
  MemorySummary,
  MemoryVersion
} from "./memories/types";
import { SettingsPanel, SettingsSaveBanner, ToggleSwitch } from "./settingsControls";
import type { BackendModelGroup, Persona, SettingsGuard } from "./types";

type MemoryStatus = "active" | "forgotten";
type ConfirmAction =
  | { kind: "discard"; nextMemoryId: string | null }
  | { kind: "forget" }
  | { kind: "delete" }
  | { kind: "restore-version"; version: MemoryVersion };

const EMPTY_SCOPE: MemoryModelScope = { all_models: true, model_keys: [] };

export function MemoriesSettingsPanel({
  modelGroups,
  personas,
  onToolsChanged,
  onGuardChange
}: {
  modelGroups: BackendModelGroup[];
  personas: Persona[];
  onToolsChanged: () => Promise<void>;
  onGuardChange: (guard: SettingsGuard | null) => void;
}) {
  const [savedSettings, setSavedSettings] = useState<MemorySettings | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<MemorySettings | null>(null);
  const [memories, setMemories] = useState<MemorySummary[]>([]);
  const [total, setTotal] = useState(0);
  const [statusFilter, setStatusFilter] = useState<MemoryStatus>("active");
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [contentDraft, setContentDraft] = useState("");
  const [scopeDraft, setScopeDraft] = useState<MemoryModelScope>(EMPTY_SCOPE);
  const [versions, setVersions] = useState<MemoryVersion[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMemory, setIsLoadingMemory] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isMutating, setIsMutating] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmAction, setConfirmAction] = useState<ConfirmAction | null>(null);
  const loadRequestRef = useRef(0);
  const skipInitialListReloadRef = useRef(true);

  const modelOptions = useMemo(
    () => buildNoteModelOptions("server", modelGroups, personas, []),
    [modelGroups, personas]
  );
  const settingsDirty = Boolean(
    savedSettings &&
      settingsDraft &&
      settingsFingerprint(savedSettings) !== settingsFingerprint(settingsDraft)
  );
  const editorDirty = useMemo(() => {
    if (isCreating) {
      return Boolean(contentDraft.trim()) || scopeFingerprint(scopeDraft) !== scopeFingerprint(EMPTY_SCOPE);
    }
    if (!selectedMemory) return false;
    return (
      contentDraft !== selectedMemory.current_version.content ||
      scopeFingerprint(scopeDraft) !== scopeFingerprint(selectedMemory.model_scope)
    );
  }, [contentDraft, isCreating, scopeDraft, selectedMemory]);
  const isDirty = settingsDirty || editorDirty;
  const isEditing = isCreating || Boolean(selectedMemory);
  const modelAccessEnabled = Boolean(
    settingsDraft && (settingsDraft.allow_model_read || settingsDraft.allow_model_create)
  );

  const loadList = useCallback(async (filter: MemoryStatus, search: string) => {
    const requestId = loadRequestRef.current + 1;
    loadRequestRef.current = requestId;
    setIsLoading(true);
    setError(null);
    try {
      const response = await listMemories({ query: search, status: filter, limit: 100 });
      if (loadRequestRef.current !== requestId) return;
      setMemories(response.memories);
      setTotal(response.total);
    } catch (loadError) {
      if (loadRequestRef.current !== requestId) return;
      setError(loadError instanceof Error ? loadError.message : "Failed to load memories");
    } finally {
      if (loadRequestRef.current === requestId) setIsLoading(false);
    }
  }, []);

  const loadInitial = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [loadedSettings, response] = await Promise.all([
        getMemorySettings(),
        listMemories({ status: "active", limit: 100 })
      ]);
      setSavedSettings({ ...loadedSettings });
      setSettingsDraft({ ...loadedSettings });
      setMemories(response.memories);
      setTotal(response.total);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load memories");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadInitial();
    return () => {
      loadRequestRef.current += 1;
    };
  }, [loadInitial]);

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedQuery(query.trim()), 180);
    return () => window.clearTimeout(timeout);
  }, [query]);

  useEffect(() => {
    if (!savedSettings) return;
    if (skipInitialListReloadRef.current) {
      skipInitialListReloadRef.current = false;
      return;
    }
    void loadList(statusFilter, debouncedQuery);
  }, [debouncedQuery, loadList, savedSettings, statusFilter]);

  useEffect(() => {
    if (!status) return;
    const timeout = window.setTimeout(() => setStatus(null), 3000);
    return () => window.clearTimeout(timeout);
  }, [status]);

  useEffect(() => {
    if (isDirty && status) setStatus(null);
  }, [isDirty, status]);

  function updateSettings(update: (current: MemorySettings) => MemorySettings) {
    setStatus(null);
    setError(null);
    setSettingsDraft((current) => (current ? update(current) : current));
  }

  function setModelAccessEnabled(enabled: boolean) {
    updateSettings((current) => ({
      ...current,
      allow_model_read: enabled,
      allow_model_create: enabled,
      allow_model_edit: enabled,
      allow_model_forget: enabled
    }));
  }

  function setReadAllowed(allowed: boolean) {
    updateSettings((current) => ({
      ...current,
      allow_model_read: allowed,
      allow_model_edit: allowed ? current.allow_model_edit : false,
      allow_model_forget: allowed ? current.allow_model_forget : false
    }));
  }

  function setEditAllowed(allowed: boolean) {
    updateSettings((current) => ({
      ...current,
      allow_model_read: allowed || current.allow_model_read,
      allow_model_edit: allowed
    }));
  }

  function setForgetAllowed(allowed: boolean) {
    updateSettings((current) => ({
      ...current,
      allow_model_read: allowed || current.allow_model_read,
      allow_model_forget: allowed
    }));
  }

  function resetEditor() {
    if (selectedMemory) {
      setContentDraft(selectedMemory.current_version.content);
      setScopeDraft(cloneScope(selectedMemory.model_scope));
    } else {
      setContentDraft("");
      setScopeDraft(cloneScope(EMPTY_SCOPE));
    }
    setVersions(null);
  }

  const revertAll = useCallback(() => {
    if (savedSettings) setSettingsDraft({ ...savedSettings });
    if (selectedMemory) {
      setContentDraft(selectedMemory.current_version.content);
      setScopeDraft(cloneScope(selectedMemory.model_scope));
    } else {
      setContentDraft("");
      setScopeDraft(cloneScope(EMPTY_SCOPE));
    }
    setStatus(null);
    setError(null);
  }, [savedSettings, selectedMemory]);

  const saveAll = useCallback(async () => {
    if (isSaving || !settingsDraft) return false;
    if (isEditing && !contentDraft.trim()) {
      setError("Memory content cannot be empty.");
      return false;
    }
    setIsSaving(true);
    setStatus(null);
    setError(null);
    try {
      if (settingsDirty) {
        const updatedSettings = await updateMemorySettings(settingsDraft);
        setSavedSettings({ ...updatedSettings });
        setSettingsDraft({ ...updatedSettings });
        await onToolsChanged().catch(() => undefined);
      }

      let savedMemory: Memory | null = selectedMemory;
      if (editorDirty) {
        const normalizedScope = normalizeNoteModelScope(scopeDraft);
        savedMemory = isCreating
          ? await createMemory(contentDraft.trim(), normalizedScope)
          : selectedMemory
            ? await updateMemory(
                selectedMemory.id,
                selectedMemory.current_version.version_number,
                contentDraft.trim(),
                normalizedScope
              )
            : null;
        if (savedMemory) {
          setSelectedMemory(savedMemory);
          setIsCreating(false);
          setContentDraft(savedMemory.current_version.content);
          setScopeDraft(cloneScope(savedMemory.model_scope));
          setVersions(null);
        }
      }

      await loadList(statusFilter, debouncedQuery);
      setStatus(settingsDirty && editorDirty ? "Settings and memory saved." : editorDirty ? "Memory saved." : "Memory settings saved.");
      return true;
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save memories");
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [
    contentDraft,
    debouncedQuery,
    editorDirty,
    isCreating,
    isEditing,
    isSaving,
    loadList,
    onToolsChanged,
    scopeDraft,
    selectedMemory,
    settingsDirty,
    settingsDraft,
    statusFilter
  ]);

  useEffect(() => {
    onGuardChange({ isDirty, save: saveAll, discard: revertAll });
  }, [isDirty, onGuardChange, revertAll, saveAll]);

  useEffect(() => () => onGuardChange(null), [onGuardChange]);

  async function selectMemory(memoryId: string) {
    if (isDirty) {
      setConfirmAction({ kind: "discard", nextMemoryId: memoryId });
      return;
    }
    await openMemory(memoryId);
  }

  async function openMemory(memoryId: string) {
    setIsLoadingMemory(true);
    setError(null);
    setStatus(null);
    try {
      const memory = await getMemory(memoryId);
      setSelectedMemory(memory);
      setIsCreating(false);
      setContentDraft(memory.current_version.content);
      setScopeDraft(cloneScope(memory.model_scope));
      setVersions(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load memory");
    } finally {
      setIsLoadingMemory(false);
    }
  }

  function startNewMemory() {
    if (isDirty) {
      setConfirmAction({ kind: "discard", nextMemoryId: null });
      return;
    }
    beginNewMemory();
  }

  function beginNewMemory() {
    setSelectedMemory(null);
    setIsCreating(true);
    setContentDraft("");
    setScopeDraft(cloneScope(EMPTY_SCOPE));
    setVersions(null);
    setStatus(null);
    setError(null);
  }

  function closeEditor() {
    if (isDirty) {
      setConfirmAction({ kind: "discard", nextMemoryId: "" });
      return;
    }
    clearEditor();
  }

  function clearEditor() {
    setSelectedMemory(null);
    setIsCreating(false);
    setContentDraft("");
    setScopeDraft(cloneScope(EMPTY_SCOPE));
    setVersions(null);
    setStatus(null);
    setError(null);
  }

  async function loadHistory() {
    if (!selectedMemory) return;
    setIsMutating(true);
    setError(null);
    try {
      setVersions(await listMemoryVersions(selectedMemory.id));
    } catch (historyError) {
      setError(historyError instanceof Error ? historyError.message : "Failed to load memory history");
    } finally {
      setIsMutating(false);
    }
  }

  async function runConfirmAction() {
    if (!confirmAction) return;
    const action = confirmAction;
    setConfirmAction(null);
    if (action.kind === "discard") {
      revertAll();
      if (action.nextMemoryId === "") {
        clearEditor();
      } else if (action.nextMemoryId === null) {
        beginNewMemory();
      } else {
        await openMemory(action.nextMemoryId);
      }
      return;
    }
    if (!selectedMemory) return;
    setIsMutating(true);
    setError(null);
    try {
      if (action.kind === "forget") {
        await forgetMemory(selectedMemory.id, selectedMemory.current_version.version_number);
        clearEditor();
        await loadList(statusFilter, debouncedQuery);
        setStatus("Memory forgotten. You can restore it from Forgotten.");
      } else if (action.kind === "delete") {
        await permanentlyDeleteMemory(selectedMemory.id);
        clearEditor();
        await loadList(statusFilter, debouncedQuery);
        setStatus("Memory permanently deleted.");
      } else {
        const restored = await restoreMemoryVersion(
          selectedMemory.id,
          action.version.id,
          selectedMemory.current_version.version_number
        );
        setSelectedMemory(restored);
        setContentDraft(restored.current_version.content);
        setScopeDraft(cloneScope(restored.model_scope));
        setVersions(await listMemoryVersions(restored.id));
        await loadList(statusFilter, debouncedQuery);
        setStatus(`Version ${action.version.version_number} restored as a new version.`);
      }
    } catch (mutationError) {
      setError(mutationError instanceof Error ? mutationError.message : "Memory change failed");
    } finally {
      setIsMutating(false);
    }
  }

  async function restoreSelectedMemory() {
    if (!selectedMemory) return;
    setIsMutating(true);
    setError(null);
    try {
      await restoreMemory(selectedMemory.id, selectedMemory.current_version.version_number);
      clearEditor();
      await loadList(statusFilter, debouncedQuery);
      setStatus("Memory restored.");
    } catch (restoreError) {
      setError(restoreError instanceof Error ? restoreError.message : "Failed to restore memory");
    } finally {
      setIsMutating(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void saveAll();
  }

  return (
    <SettingsPanel
      eyebrow="Personal"
      title="Memories"
      width="standard"
      className="memories-settings-section"
      actions={
        !isEditing ? (
          <button
            type="button"
            className="secondary-button"
            disabled={isLoading}
            onClick={() => void loadList(statusFilter, debouncedQuery)}
          >
            <RefreshCw aria-hidden="true" />
            <span>Refresh</span>
          </button>
        ) : undefined
      }
    >
      <form
        className="settings-form settings-form-with-banner memories-settings-form"
        aria-busy={isSaving || isMutating}
        onSubmit={submit}
      >
        <SettingsSaveBanner
          isDirty={isDirty}
          status={status}
          dirtyTitle="Unsaved memory changes"
          dirtyDescription="Save or revert before leaving Memories."
          savedDescription="Your memory changes are active."
        >
          <button type="button" className="secondary-button" disabled={isSaving} onClick={revertAll}>
            Revert
          </button>
          <button type="submit" disabled={isSaving || isMutating}>
            {isSaving ? "Saving..." : "Save"}
          </button>
        </SettingsSaveBanner>

        {error && <p className="error">{error}</p>}

        {!isEditing && settingsDraft && (
          <>
            <section className="settings-subsection memories-access-section">
              <div className="tool-setting-heading">
                <BrainCircuit aria-hidden="true" />
                <div>
                  <strong>Let models use memories</strong>
                  <p>
                    Models only receive these tools when Memories is also turned on in that chat.
                  </p>
                </div>
              </div>
              <ToggleSwitch
                label={modelAccessEnabled ? "Model access is on" : "Model access is off"}
                description={
                  modelAccessEnabled
                    ? memoryAccessSummary(settingsDraft)
                    : "Models cannot currently read or create memories."
                }
                checked={modelAccessEnabled}
                onChange={setModelAccessEnabled}
              />
              {modelAccessEnabled && (
                <details className="tool-details memories-advanced-permissions">
                  <summary>What models can do</summary>
                  <ToggleSwitch
                    label="Find and read memories"
                    description="Search and open memories available to the current model."
                    checked={settingsDraft.allow_model_read}
                    onChange={setReadAllowed}
                  />
                  <ToggleSwitch
                    label="Create memories"
                    description="Store a concise memory scoped to the model that created it."
                    checked={settingsDraft.allow_model_create}
                    onChange={(allow_model_create) =>
                      updateSettings((current) => ({ ...current, allow_model_create }))
                    }
                  />
                  <ToggleSwitch
                    label="Edit memories"
                    description="Correct or refine accessible memories while preserving history."
                    checked={settingsDraft.allow_model_edit}
                    onChange={setEditAllowed}
                  />
                  <ToggleSwitch
                    label="Forget memories"
                    description="Move a memory to Forgotten. Models can never permanently delete it."
                    checked={settingsDraft.allow_model_forget}
                    onChange={setForgetAllowed}
                  />
                </details>
              )}
            </section>

            <section className="settings-subsection">
              <div className="tool-setting-heading">
                <Search aria-hidden="true" />
                <div>
                  <strong>Past chats</strong>
                  <p>Let models search completed messages from your server chat history.</p>
                </div>
              </div>
              <ToggleSwitch
                label={
                  settingsDraft.allow_model_chat_history
                    ? "Past-chat access is on"
                    : "Past-chat access is off"
                }
                description="Private chats stay on this device and are never included. Historical text is provided as quoted data, not instructions."
                checked={settingsDraft.allow_model_chat_history}
                onChange={(allow_model_chat_history) =>
                  updateSettings((current) => ({ ...current, allow_model_chat_history }))
                }
              />
            </section>

            <section className="memories-library" aria-label="Memory library">
              <div className="memories-library-toolbar">
                <div className="segmented-control memories-status-control" role="group" aria-label="Memory status">
                  <button
                    type="button"
                    className={statusFilter === "active" ? "active" : ""}
                    aria-pressed={statusFilter === "active"}
                    onClick={() => setStatusFilter("active")}
                  >
                    Memories
                  </button>
                  <button
                    type="button"
                    className={statusFilter === "forgotten" ? "active" : ""}
                    aria-pressed={statusFilter === "forgotten"}
                    onClick={() => setStatusFilter("forgotten")}
                  >
                    Forgotten
                  </button>
                </div>
                <button type="button" onClick={startNewMemory}>
                  <Plus aria-hidden="true" />
                  <span>New Memory</span>
                </button>
              </div>
              <label className="memories-search">
                <Search aria-hidden="true" />
                <span className="visually-hidden">Search memories</span>
                <input
                  type="search"
                  value={query}
                  placeholder="Search memories"
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
              <p className="memories-count">
                {isLoading ? "Loading..." : `${total} ${statusFilter === "active" ? "memories" : "forgotten"}`}
              </p>
              <div className="memories-list">
                {memories.map((memory) => (
                  <button
                    type="button"
                    key={memory.id}
                    className="memory-list-item"
                    disabled={isLoadingMemory}
                    onClick={() => void selectMemory(memory.id)}
                  >
                    <span>{memory.excerpt}</span>
                    <small>
                      v{memory.current_version_number} · {formatRelativeTime(memory.updated_at)} · {scopeSummary(memory.model_scope)}
                    </small>
                  </button>
                ))}
                {!isLoading && memories.length === 0 && (
                  <div className="memories-empty-state">
                    <BrainCircuit aria-hidden="true" />
                    <strong>{query ? "No matching memories" : statusFilter === "active" ? "No memories yet" : "Nothing forgotten"}</strong>
                    <span>
                      {query
                        ? "Try a different search."
                        : statusFilter === "active"
                          ? "Create a concise fact or preference you want models to remember."
                          : "Forgotten memories remain recoverable until you delete them permanently."}
                    </span>
                  </div>
                )}
              </div>
            </section>
          </>
        )}

        {isEditing && (
          <section className="memory-editor" aria-label={isCreating ? "New memory" : "Edit memory"}>
            <header className="memory-editor-header">
              <button type="button" className="icon-button" aria-label="Back to memories" title="Back to memories" onClick={closeEditor}>
                <ArrowLeft aria-hidden="true" />
              </button>
              <div>
                <p className="eyebrow">{selectedMemory?.deleted_at ? "Forgotten" : isCreating ? "New" : "Memory"}</p>
                <h2>{isCreating ? "New Memory" : `Version ${selectedMemory?.current_version.version_number ?? 1}`}</h2>
              </div>
              <div className="memory-editor-actions">
                {selectedMemory && !selectedMemory.deleted_at && (
                  <button type="button" className="secondary-button" disabled={isMutating || isDirty} onClick={() => setConfirmAction({ kind: "forget" })}>
                    <Trash2 aria-hidden="true" />
                    <span>Forget</span>
                  </button>
                )}
                {selectedMemory?.deleted_at && (
                  <>
                    <button type="button" disabled={isMutating} onClick={() => void restoreSelectedMemory()}>
                      <RotateCcw aria-hidden="true" />
                      <span>Restore</span>
                    </button>
                    <button type="button" className="danger-button" disabled={isMutating} onClick={() => setConfirmAction({ kind: "delete" })}>
                      <Trash2 aria-hidden="true" />
                      <span>Delete</span>
                    </button>
                  </>
                )}
              </div>
            </header>

            {selectedMemory && <MemorySource version={selectedMemory.current_version} />}

            <label className="setting-field memory-content-field">
              <span>What should Vashti remember?</span>
              <textarea
                autoFocus={isCreating}
                value={contentDraft}
                disabled={Boolean(selectedMemory?.deleted_at)}
                maxLength={32 * 1024}
                placeholder="A concise, standalone fact or preference"
                onChange={(event) => setContentDraft(event.target.value)}
              />
              <small>{contentDraft.length.toLocaleString()} / 32,768 characters</small>
            </label>

            {selectedMemory?.deleted_at ? (
              <section className="notes-model-scope memory-scope-readonly">
                <h3>Model access</h3>
                <p>
                  {scopeDraft.all_models
                    ? "This memory was available to every model before it was forgotten."
                    : scopeDraft.model_keys.length > 0
                      ? `This memory was available to ${scopeDraft.model_keys.length} selected model${scopeDraft.model_keys.length === 1 ? "" : "s"}.`
                      : "This memory was not available to any models."}
                </p>
              </section>
            ) : (
              <NoteModelScopeEditor
                scope={scopeDraft}
                options={modelOptions}
                onChange={setScopeDraft}
                title="Which models can use this memory?"
                description="Allow every model, or keep the memory scoped to selected base and custom models."
                emptyWarning="Choose at least one model, or this memory will not be available to models."
              />
            )}

            {selectedMemory && (
              <section className="memory-history-section">
                <div className="memory-history-heading">
                  <div>
                    <h3>History</h3>
                    <p>Every content change is preserved. Restoring creates a new version.</p>
                  </div>
                  <button type="button" className="secondary-button" disabled={isMutating} onClick={() => void loadHistory()}>
                    <Clock3 aria-hidden="true" />
                    <span>{versions ? "Refresh" : "Show History"}</span>
                  </button>
                </div>
                {versions && (
                  <div className="memory-version-list">
                    {versions.map((version) => (
                      <article key={version.id} className={version.id === selectedMemory.current_version.id ? "memory-version-current" : ""}>
                        <div className="memory-version-meta">
                          <strong>v{version.version_number}</strong>
                          <span>{actorLabel(version)} · {formatDateTime(version.created_at)}</span>
                        </div>
                        <p>{version.content}</p>
                        {version.source_chat_id && (
                          <a href={`/app/chats/${version.source_chat_id}`}>
                            Open source chat <ExternalLink aria-hidden="true" />
                          </a>
                        )}
                        {version.id !== selectedMemory.current_version.id && !selectedMemory.deleted_at && (
                          <button type="button" className="secondary-button" disabled={isMutating || isDirty} onClick={() => setConfirmAction({ kind: "restore-version", version })}>
                            Restore this version
                          </button>
                        )}
                      </article>
                    ))}
                  </div>
                )}
              </section>
            )}
          </section>
        )}
      </form>

      {confirmAction && (
        <ConfirmDialog
          title={confirmDialogCopy(confirmAction).title}
          message={confirmDialogCopy(confirmAction).message}
          confirmLabel={confirmDialogCopy(confirmAction).label}
          confirmTone={confirmAction.kind === "delete" || confirmAction.kind === "forget" ? "danger" : "primary"}
          isBusy={isMutating}
          onCancel={() => setConfirmAction(null)}
          onConfirm={() => void runConfirmAction()}
        />
      )}
    </SettingsPanel>
  );
}

function MemorySource({ version }: { version: MemoryVersion }) {
  return (
    <div className="memory-source">
      <BrainCircuit aria-hidden="true" />
      <div>
        <strong>{actorLabel(version)}</strong>
        <span>{formatDateTime(version.created_at)}</span>
      </div>
      {version.source_chat_id && (
        <a href={`/app/chats/${version.source_chat_id}`}>
          Source chat <ExternalLink aria-hidden="true" />
        </a>
      )}
    </div>
  );
}

function memoryAccessSummary(settings: MemorySettings) {
  const actions = [
    settings.allow_model_read ? "read" : null,
    settings.allow_model_create ? "create" : null,
    settings.allow_model_edit ? "edit" : null,
    settings.allow_model_forget ? "forget" : null
  ].filter(Boolean);
  return `Models may ${actions.join(", ")} memories when the chat toggle and admin permissions also allow it.`;
}

function cloneScope(scope: MemoryModelScope): MemoryModelScope {
  return { all_models: scope.all_models, model_keys: [...scope.model_keys] };
}

function settingsFingerprint(settings: MemorySettings) {
  return JSON.stringify(settings);
}

function scopeFingerprint(scope: MemoryModelScope) {
  return JSON.stringify(normalizeNoteModelScope(scope));
}

function actorLabel(version: MemoryVersion) {
  return version.actor_type === "model"
    ? `Remembered by ${version.actor_model_name ?? "a model"}`
    : "Saved by you";
}

function scopeSummary(scope: MemoryModelScope) {
  if (scope.all_models) return "all models";
  if (scope.model_keys.length === 0) return "no models";
  return `${scope.model_keys.length} model${scope.model_keys.length === 1 ? "" : "s"}`;
}

function formatDateTime(timestamp: number) {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(
    new Date(timestamp * 1000)
  );
}

function formatRelativeTime(timestamp: number) {
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - timestamp));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function confirmDialogCopy(action: ConfirmAction) {
  switch (action.kind) {
    case "discard":
      return {
        title: "Discard unsaved changes?",
        message: "Your unsaved memory and permission changes will be lost.",
        label: "Discard"
      };
    case "forget":
      return {
        title: "Forget this memory?",
        message: "The memory will stop appearing in searches. You can restore it from Forgotten.",
        label: "Forget"
      };
    case "delete":
      return {
        title: "Delete this memory permanently?",
        message: "This removes every version and cannot be undone.",
        label: "Delete Permanently"
      };
    case "restore-version":
      return {
        title: `Restore version ${action.version.version_number}?`,
        message: "The selected content will become a new current version. Existing history remains intact.",
        label: "Restore Version"
      };
  }
}
