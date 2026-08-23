import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HardDrive, NotebookPen, Server } from "lucide-react";
import {
  getPrivateNoteSettings,
  savePrivateNoteSettings,
  type PrivatePersona
} from "./privateChatStore";
import {
  NoteAccessLevelControl,
  NoteModelScopeEditor,
  buildNoteModelOptions,
  normalizeNoteModelScope
} from "./notes/accessControls";
import { getNoteSettings, updateNoteSettings } from "./notes/api";
import type { NoteSettings } from "./notes/types";
import { SettingsPanel, SettingsSaveBanner, ToggleSwitch } from "./settingsControls";
import type { BackendModelGroup, Persona, SettingsGuard } from "./types";

type NoteStorageMode = "server" | "device";

export function NotesSettingsPanel({
  modelGroups,
  personas,
  privatePersonas,
  initialStorageMode,
  onStorageModeChange,
  onToolsChanged,
  onGuardChange
}: {
  modelGroups: BackendModelGroup[];
  personas: Persona[];
  privatePersonas: PrivatePersona[];
  initialStorageMode: NoteStorageMode;
  onStorageModeChange: (storageMode: NoteStorageMode) => void;
  onToolsChanged: () => Promise<void>;
  onGuardChange: (guard: SettingsGuard | null) => void;
}) {
  const storageMode = initialStorageMode;
  const [savedSettings, setSavedSettings] = useState<NoteSettings | null>(null);
  const [draft, setDraft] = useState<NoteSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadRequestRef = useRef(0);

  const loadSettings = useCallback(async (mode: NoteStorageMode) => {
    const requestId = loadRequestRef.current + 1;
    loadRequestRef.current = requestId;
    setIsLoading(true);
    setSavedSettings(null);
    setDraft(null);
    setStatus(null);
    setError(null);
    try {
      const loaded = mode === "device" ? await getPrivateNoteSettings() : await getNoteSettings();
      if (loadRequestRef.current !== requestId) return;
      setSavedSettings(cloneSettings(loaded));
      setDraft(cloneSettings(loaded));
    } catch (loadError) {
      if (loadRequestRef.current !== requestId) return;
      setSavedSettings(null);
      setDraft(null);
      setError(loadError instanceof Error ? loadError.message : "Failed to load Notes settings");
    } finally {
      if (loadRequestRef.current === requestId) {
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void loadSettings(initialStorageMode);
    return () => {
      loadRequestRef.current += 1;
    };
  }, [initialStorageMode, loadSettings]);

  useEffect(() => {
    if (!status) return;
    const timeout = window.setTimeout(() => setStatus(null), 3000);
    return () => window.clearTimeout(timeout);
  }, [status]);

  const modelOptions = useMemo(
    () => buildNoteModelOptions(storageMode, modelGroups, personas, privatePersonas),
    [modelGroups, personas, privatePersonas, storageMode]
  );
  const isDirty = Boolean(
    savedSettings && draft && settingsFingerprint(savedSettings) !== settingsFingerprint(draft)
  );
  const modelAccessEnabled = Boolean(
    draft && (draft.allow_model_read || draft.allow_model_create)
  );

  function switchStorageMode(nextMode: NoteStorageMode) {
    if (nextMode === storageMode || isDirty || isLoading || isSaving) return;
    setStatus(null);
    onStorageModeChange(nextMode);
  }

  function updateDraft(update: (current: NoteSettings) => NoteSettings) {
    setStatus(null);
    setError(null);
    setDraft((current) => (current ? update(current) : current));
  }

  function setModelAccessEnabled(enabled: boolean) {
    updateDraft((current) => ({
      ...current,
      allow_model_read: enabled,
      allow_model_create: enabled,
      allow_model_edit: enabled,
      allow_model_trash: enabled
    }));
  }

  function setReadAllowed(allowed: boolean) {
    updateDraft((current) => ({
      ...current,
      allow_model_read: allowed,
      allow_model_edit: allowed ? current.allow_model_edit : false,
      allow_model_trash: allowed ? current.allow_model_trash : false
    }));
  }

  function setEditAllowed(allowed: boolean) {
    updateDraft((current) => ({
      ...current,
      allow_model_read: allowed || current.allow_model_read,
      allow_model_edit: allowed
    }));
  }

  function setTrashAllowed(allowed: boolean) {
    updateDraft((current) => ({
      ...current,
      allow_model_read: allowed || current.allow_model_read,
      allow_model_trash: allowed
    }));
  }

  const revertSettings = useCallback(() => {
    if (!savedSettings) return;
    setDraft(cloneSettings(savedSettings));
    setStatus(null);
    setError(null);
  }, [savedSettings]);

  const saveSettingsDraft = useCallback(async () => {
    if (!draft || isLoading || isSaving) return false;
    setIsSaving(true);
    setStatus(null);
    setError(null);
    try {
      const payload = {
        ...draft,
        default_model_scope: normalizeNoteModelScope(draft.default_model_scope)
      };
      const updated =
        storageMode === "device"
          ? await savePrivateNoteSettings(payload)
          : await updateNoteSettings(payload);
      setSavedSettings(cloneSettings(updated));
      setDraft(cloneSettings(updated));
      await onToolsChanged().catch(() => undefined);
      setStatus("Notes settings saved.");
      return true;
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save Notes settings");
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [draft, isLoading, isSaving, onToolsChanged, storageMode]);

  function saveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void saveSettingsDraft();
  }

  useEffect(() => {
    onGuardChange({
      isDirty,
      save: saveSettingsDraft,
      discard: revertSettings
    });
  }, [isDirty, onGuardChange, revertSettings, saveSettingsDraft]);

  useEffect(() => () => onGuardChange(null), [onGuardChange]);

  return (
    <SettingsPanel eyebrow="Personal" title="Notes" width="standard">
      <p className="settings-lead">
        Choose how models can use your notes whenever Notes is turned on for a chat. You can
        change an individual note at any time.
      </p>

      <div className="context-storage-switch" role="group" aria-label="Notes storage">
        <button
          type="button"
          className={storageMode === "server" ? "context-storage-option active" : "context-storage-option"}
          aria-pressed={storageMode === "server"}
          disabled={isDirty || isLoading || isSaving}
          title={isDirty ? "Save or revert your changes before switching" : undefined}
          onClick={() => switchStorageMode("server")}
        >
          <Server />
          <span>
            <strong>Server notes</strong>
            <small>Sync across your devices and work in standard chats.</small>
          </span>
        </button>
        <button
          type="button"
          className={storageMode === "device" ? "context-storage-option active" : "context-storage-option"}
          aria-pressed={storageMode === "device"}
          disabled={isDirty || isLoading || isSaving}
          title={isDirty ? "Save or revert your changes before switching" : undefined}
          onClick={() => switchStorageMode("device")}
        >
          <HardDrive />
          <span>
            <strong>Notes on this device</strong>
            <small>Stay in this app on this device and work only in private chats.</small>
          </span>
        </button>
      </div>

      {storageMode === "device" && (
        <p className="notes-settings-disclosure">
          Device notes are not stored on the server. When a model uses one, its content is sent to
          the selected model for that response.
        </p>
      )}

      {error && <p className="error">{error}</p>}
      {isLoading && <p className="status-message">Loading Notes settings...</p>}

      {draft && (
        <form
          className="settings-form settings-form-with-banner notes-settings-form"
          aria-busy={isSaving}
          onSubmit={saveSettings}
        >
          <SettingsSaveBanner
            isDirty={isDirty}
            status={status}
            dirtyTitle="Unsaved Notes changes"
            dirtyDescription="Save or revert your model-access changes."
            savedDescription="Your Notes settings are active."
          >
            <button type="button" className="secondary-button" disabled={isSaving} onClick={revertSettings}>
              Revert
            </button>
            <button type="submit" disabled={isSaving}>
              {isSaving ? "Saving..." : "Save"}
            </button>
          </SettingsSaveBanner>

          <section className="settings-subsection notes-model-access-summary">
            <div className="tool-setting-heading">
              <NotebookPen aria-hidden="true" />
              <div>
                <strong>Let models use notes</strong>
                <p>
                  When Notes is on in a chat, models can use the actions you allow below.
                </p>
              </div>
            </div>
            <ToggleSwitch
              label={modelAccessEnabled ? "Model access is on" : "Model access is off"}
              description={
                modelAccessEnabled
                  ? modelAccessSummary(draft)
                  : "Models cannot currently read or create notes."
              }
              checked={modelAccessEnabled}
              onChange={setModelAccessEnabled}
            />

            {modelAccessEnabled && (
              <details className="tool-details notes-advanced-permissions">
                <summary>What models can do</summary>
                <ToggleSwitch
                  label="Find and read notes"
                  description="Search and open notes that you have made available to models."
                  checked={draft.allow_model_read}
                  onChange={setReadAllowed}
                />
                <ToggleSwitch
                  label="Create notes"
                  description="Make new notes during a chat."
                  checked={draft.allow_model_create}
                  onChange={(allow_model_create) =>
                    updateDraft((current) => ({ ...current, allow_model_create }))
                  }
                />
                <ToggleSwitch
                  label="Edit notes"
                  description="Update available notes while keeping their version history."
                  checked={draft.allow_model_edit}
                  onChange={setEditAllowed}
                />
                <ToggleSwitch
                  label="Move notes to Trash"
                  description="Nothing is permanently deleted."
                  checked={draft.allow_model_trash}
                  onChange={setTrashAllowed}
                />
              </details>
            )}
          </section>

          <section className="settings-subsection notes-defaults-section">
            <div>
              <p className="eyebrow">Defaults for new notes</p>
              <h2>Notes you create</h2>
              <p className="status-message">
                These choices affect notes created from now on. Existing notes keep their current
                per-note access.
              </p>
            </div>
            <div className="notes-access-section">
              <h3>What can models do with new notes?</h3>
              <p>You can choose a different access level for any individual note later.</p>
              <NoteAccessLevelControl
                value={draft.default_ai_access}
                ariaLabel="Default model access for new notes"
                onChange={(default_ai_access) =>
                  updateDraft((current) => ({ ...current, default_ai_access }))
                }
              />
              <small>
                Full access means models may read, edit, and move the note to Trash. They can never
                permanently delete it.
              </small>
            </div>
            {draft.default_ai_access !== "none" && (
              <NoteModelScopeEditor
                scope={draft.default_model_scope}
                options={modelOptions}
                onChange={(default_model_scope) =>
                  updateDraft((current) => ({ ...current, default_model_scope }))
                }
                title="Which models can use new notes?"
                description="Allow every model, or choose specific models."
                emptyWarning="Choose at least one model, or new notes will not be available to models."
              />
            )}
            {!draft.allow_model_read && (
              <p className="notes-scope-warning" role="status">
                These defaults are saved for later, but models cannot read notes while Find and
                read notes is off.
              </p>
            )}
            <p className="notes-model-created-copy">
              A note created by a model starts with full access so that model can keep using it.
              You can restrict the note later.
            </p>
          </section>
        </form>
      )}
    </SettingsPanel>
  );
}

function modelAccessSummary(settings: NoteSettings) {
  if (settings.allow_model_read && settings.allow_model_create) {
    return "Models can find, read, and create notes. Advanced actions follow the choices below.";
  }
  if (settings.allow_model_read) {
    return "Models can find and read available notes, but cannot create them.";
  }
  return "Models can create notes, but cannot find or reopen them.";
}

function cloneSettings(settings: NoteSettings): NoteSettings {
  return {
    ...settings,
    default_model_scope: {
      ...settings.default_model_scope,
      model_keys: [...settings.default_model_scope.model_keys]
    }
  };
}

function settingsFingerprint(settings: NoteSettings) {
  return JSON.stringify({
    ...settings,
    default_model_scope: normalizeNoteModelScope(settings.default_model_scope)
  });
}
