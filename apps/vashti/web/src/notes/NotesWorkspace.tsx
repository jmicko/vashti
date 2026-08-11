import {
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  ArrowLeft,
  Check,
  Eye,
  FilePlus2,
  FileText,
  History,
  MoreHorizontal,
  Pin,
  PinOff,
  RefreshCw,
  Search,
  Settings2,
  Trash2,
  Undo2,
  X
} from "lucide-react";
import { ApiRequestError } from "../api";
import { ConfirmDialog, RetroLoader } from "../common";
import { MarkdownContent } from "../MarkdownContent";
import type { BackendModelGroup, Persona } from "../types";
import {
  createNote,
  getNote,
  getNoteSettings,
  listNotes,
  listNoteVersions,
  permanentlyDeleteNote,
  restoreNote,
  restoreNoteVersion,
  trashNote,
  updateNote,
  updateNoteSettings
} from "./api";
import { MarkdownToolbar } from "./MarkdownToolbar";
import type {
  Note,
  NoteAiAccess,
  NoteDraft,
  NoteListStatus,
  NoteModelScope,
  NoteSettings,
  NoteSort,
  NoteSummary,
  NoteVersion
} from "./types";

const AUTOSAVE_DELAY_MS = 700;
const NOTE_LIST_LIMIT = 100;

type Drawer = "details" | "history" | "settings" | null;
type SaveState = "idle" | "saving" | "saved" | "error" | "conflict";
type ConfirmAction =
  | { kind: "trash" }
  | { kind: "purge" }
  | { kind: "restore-version"; version: NoteVersion };

type ModelOption = {
  key: string;
  label: string;
  group: string;
};

export function NotesWorkspace({
  modelGroups,
  personas
}: {
  modelGroups: BackendModelGroup[];
  personas: Persona[];
}) {
  const [status, setStatus] = useState<NoteListStatus>("active");
  const [sort, setSort] = useState<NoteSort>("updated");
  const [searchInput, setSearchInput] = useState("");
  const [query, setQuery] = useState("");
  const [summaries, setSummaries] = useState<NoteSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoadingList, setIsLoadingList] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [listRefreshKey, setListRefreshKey] = useState(0);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const selectedNoteIdRef = useRef<string | null>(null);
  const [note, setNote] = useState<Note | null>(null);
  const [draft, setDraft] = useState<NoteDraft | null>(null);
  const [tagsText, setTagsText] = useState("");
  const [isLoadingNote, setIsLoadingNote] = useState(false);
  const [noteError, setNoteError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [editorMode, setEditorMode] = useState<"edit" | "preview">("edit");
  const [drawer, setDrawer] = useState<Drawer>(null);
  const [versions, setVersions] = useState<NoteVersion[]>([]);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [isLoadingVersions, setIsLoadingVersions] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [settings, setSettings] = useState<NoteSettings | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<NoteSettings | null>(null);
  const [isLoadingSettings, setIsLoadingSettings] = useState(false);
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [confirmAction, setConfirmAction] = useState<ConfirmAction | null>(null);
  const [isConfirming, setIsConfirming] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isMobileEditorOpen, setIsMobileEditorOpen] = useState(false);
  const [isListMenuOpen, setIsListMenuOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const titleRef = useRef<HTMLInputElement>(null);
  const noteRef = useRef<Note | null>(null);
  const draftRef = useRef<NoteDraft | null>(null);
  const listMenuRef = useRef<HTMLDivElement>(null);
  const listRequestRef = useRef(0);
  const noteRequestRef = useRef(0);
  const activeSaveRef = useRef<Promise<Note | null> | null>(null);
  const flushDraftRef = useRef<() => Promise<boolean>>(async () => true);
  const pendingTitleFocusRef = useRef<string | null>(null);
  const savedStateTimerRef = useRef<number | null>(null);

  const modelOptions = useMemo<ModelOption[]>(() => {
    const baseModels = modelGroups.flatMap((group) =>
      group.models.map((model) => ({
        key: `base:${group.backend.id}:${model.name}`,
        label: model.name,
        group: group.backend.name
      }))
    );
    const customModels = personas.map((persona) => ({
      key: `persona:${persona.id}`,
      label: persona.current_version.display_name,
      group: "Custom Models"
    }));
    return [...customModels, ...baseModels];
  }, [modelGroups, personas]);

  const isDirty = Boolean(note && draft && draftFingerprint(draft) !== noteFingerprint(note));
  const selectedVersion =
    versions.find((version) => version.id === selectedVersionId) ?? versions[0] ?? null;

  useEffect(() => {
    selectedNoteIdRef.current = selectedNoteId;
  }, [selectedNoteId]);

  useEffect(() => {
    if (note?.id !== pendingTitleFocusRef.current || !titleRef.current) {
      return;
    }
    pendingTitleFocusRef.current = null;
    titleRef.current.focus();
    titleRef.current.select();
  }, [note]);

  useEffect(() => {
    if (!isListMenuOpen) {
      return;
    }

    function closeListMenu(event: PointerEvent) {
      if (!listMenuRef.current?.contains(event.target as Node)) {
        setIsListMenuOpen(false);
      }
    }

    function closeListMenuWithKeyboard(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsListMenuOpen(false);
      }
    }

    document.addEventListener("pointerdown", closeListMenu);
    document.addEventListener("keydown", closeListMenuWithKeyboard);
    return () => {
      document.removeEventListener("pointerdown", closeListMenu);
      document.removeEventListener("keydown", closeListMenuWithKeyboard);
    };
  }, [isListMenuOpen]);

  useEffect(() => {
    const timeout = window.setTimeout(() => setQuery(searchInput.trim()), 180);
    return () => window.clearTimeout(timeout);
  }, [searchInput]);

  useEffect(() => {
    const requestId = ++listRequestRef.current;
    setIsLoadingList(true);
    setListError(null);

    void listNotes({ query, status, sort, limit: NOTE_LIST_LIMIT }).then(
      (response) => {
        if (requestId !== listRequestRef.current) {
          return;
        }
        setSummaries(response.notes);
        setTotal(response.total);
        setIsLoadingList(false);
      },
      (error) => {
        if (requestId !== listRequestRef.current) {
          return;
        }
        setListError(errorMessage(error, "Failed to load notes"));
        setIsLoadingList(false);
      }
    );
  }, [listRefreshKey, query, sort, status]);

  useEffect(() => {
    if (!selectedNoteId) {
      noteRef.current = null;
      draftRef.current = null;
      setNote(null);
      setDraft(null);
      setTagsText("");
      setNoteError(null);
      setDrawer(null);
      return;
    }

    const requestId = ++noteRequestRef.current;
    noteRef.current = null;
    draftRef.current = null;
    setNote(null);
    setDraft(null);
    setTagsText("");
    setIsLoadingNote(true);
    setNoteError(null);
    setSaveState("idle");
    setSaveMessage(null);
    setDrawer(null);
    void getNote(selectedNoteId).then(
      (nextNote) => {
        if (requestId !== noteRequestRef.current) {
          return;
        }
        applyLoadedNote(nextNote);
        setIsLoadingNote(false);
      },
      (error) => {
        if (requestId !== noteRequestRef.current) {
          return;
        }
        setNoteError(errorMessage(error, "Failed to load note"));
        setIsLoadingNote(false);
      }
    );
  }, [selectedNoteId]);

  useEffect(() => {
    if (!note || !draft || !isDirty || note.deleted_at || saveState === "conflict") {
      return;
    }
    if (!draft.title.trim()) {
      setSaveState("error");
      setSaveMessage("A title is required");
      return;
    }

    const timeout = window.setTimeout(() => {
      void saveDraft(note, draft);
    }, AUTOSAVE_DELAY_MS);
    return () => window.clearTimeout(timeout);
  }, [draft, isDirty, note, saveState]);

  useEffect(() => {
    function warnBeforeUnload(event: BeforeUnloadEvent) {
      if (!isDirty) {
        return;
      }
      event.preventDefault();
    }

    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, [isDirty]);

  useEffect(() => {
    return () => {
      void flushDraftRef.current();
      if (savedStateTimerRef.current !== null) {
        window.clearTimeout(savedStateTimerRef.current);
      }
    };
  }, []);

  function applyLoadedNote(nextNote: Note) {
    const nextDraft = noteToDraft(nextNote);
    noteRef.current = nextNote;
    draftRef.current = nextDraft;
    setNote(nextNote);
    setDraft(nextDraft);
    setTagsText(nextNote.tags.join(", "));
    setSaveState("idle");
    setSaveMessage(null);
    upsertSummary(nextNote);
  }

  function upsertSummary(nextNote: Note) {
    setSummaries((current) => {
      const nextSummary = summaryFromNote(nextNote);
      const without = current.filter((candidate) => candidate.id !== nextNote.id);
      if ((status === "trashed") !== Boolean(nextNote.deleted_at)) {
        return without;
      }
      return sortSummaries([...without, nextSummary], sort);
    });
  }

  async function saveDraft(sourceNote: Note, sourceDraft: NoteDraft) {
    const activeSave = activeSaveRef.current;
    if (activeSave) {
      const latestNote = await activeSave;
      if (!latestNote) {
        return null;
      }
      if (draftFingerprint(sourceDraft) === noteFingerprint(latestNote)) {
        return latestNote;
      }
      return saveDraft(latestNote, sourceDraft);
    }
    if (draftFingerprint(sourceDraft) === noteFingerprint(sourceNote)) {
      return sourceNote;
    }
    if (!sourceDraft.title.trim()) {
      setSaveState("error");
      setSaveMessage("A title is required");
      return null;
    }

    const saveOperation = persistDraft(sourceNote, sourceDraft);
    activeSaveRef.current = saveOperation;
    try {
      return await saveOperation;
    } finally {
      if (activeSaveRef.current === saveOperation) {
        activeSaveRef.current = null;
      }
    }
  }

  async function persistDraft(sourceNote: Note, sourceDraft: NoteDraft) {
    setSaveState("saving");
    setSaveMessage(null);
    try {
      const updated = await updateNote(sourceNote.id, {
        expected_version: sourceNote.current_version.version_number,
        title: sourceDraft.title.trim(),
        content: sourceDraft.content,
        tags: normalizeTags(sourceDraft.tags),
        is_pinned: sourceDraft.is_pinned,
        ai_access: sourceDraft.ai_access,
        model_scope: normalizeScope(sourceDraft.model_scope)
      });
      upsertSummary(updated);
      if (selectedNoteIdRef.current === updated.id) {
        noteRef.current = updated;
        setNote(updated);
        setTagsText((current) =>
          normalizeTags(sourceDraft.tags).join(", ") === normalizeTags(parseTags(current)).join(", ")
            ? normalizeTags(sourceDraft.tags).join(", ")
            : current
        );
        setSaveState("saved");
        setSaveMessage("Saved");
        if (savedStateTimerRef.current !== null) {
          window.clearTimeout(savedStateTimerRef.current);
        }
        savedStateTimerRef.current = window.setTimeout(() => {
          setSaveState((current) => (current === "saved" ? "idle" : current));
          setSaveMessage((current) => (current === "Saved" ? null : current));
        }, 1800);
      }
      return updated;
    } catch (error) {
      if (selectedNoteIdRef.current === sourceNote.id) {
        if (error instanceof ApiRequestError && error.code === "note_version_conflict") {
          setSaveState("conflict");
          setSaveMessage(error.message);
        } else {
          setSaveState("error");
          setSaveMessage(errorMessage(error, "Failed to save note"));
        }
      }
      return null;
    }
  }

  async function flushCurrentDraft() {
    const currentNote = noteRef.current;
    const currentDraft = draftRef.current;
    if (
      !currentNote ||
      !currentDraft ||
      currentNote.deleted_at ||
      draftFingerprint(currentDraft) === noteFingerprint(currentNote)
    ) {
      return true;
    }
    return Boolean(await saveDraft(currentNote, currentDraft));
  }

  flushDraftRef.current = flushCurrentDraft;

  async function selectNote(noteId: string) {
    if (noteId === selectedNoteId) {
      setIsMobileEditorOpen(true);
      return;
    }
    if (!(await flushCurrentDraft())) {
      return;
    }
    setSelectedNoteId(noteId);
    setIsMobileEditorOpen(true);
  }

  async function createNewNote() {
    if (isCreating) {
      return;
    }
    if (!(await flushCurrentDraft())) {
      return;
    }
    setIsCreating(true);
    setListError(null);
    try {
      const created = await createNote({ title: "Untitled note" });
      setStatus("active");
      setSearchInput("");
      setQuery("");
      setSummaries((current) => sortSummaries([summaryFromNote(created), ...current], sort));
      setTotal((current) => current + 1);
      pendingTitleFocusRef.current = created.id;
      setSelectedNoteId(created.id);
      setIsMobileEditorOpen(true);
    } catch (error) {
      setListError(errorMessage(error, "Failed to create note"));
    } finally {
      setIsCreating(false);
    }
  }

  async function reloadSelectedNote() {
    if (!selectedNoteId) {
      return;
    }
    setIsLoadingNote(true);
    setNoteError(null);
    try {
      applyLoadedNote(await getNote(selectedNoteId));
    } catch (error) {
      setNoteError(errorMessage(error, "Failed to reload note"));
    } finally {
      setIsLoadingNote(false);
    }
  }

  async function saveConflictCopy() {
    if (!draft) {
      return;
    }
    setIsCreating(true);
    try {
      const created = await createNote({
        ...draft,
        title: `${draft.title.trim() || "Untitled note"} (conflict copy)`,
        tags: normalizeTags(draft.tags),
        model_scope: normalizeScope(draft.model_scope)
      });
      setSummaries((current) => sortSummaries([summaryFromNote(created), ...current], sort));
      setTotal((current) => current + 1);
      setSelectedNoteId(created.id);
    } catch (error) {
      setSaveState("error");
      setSaveMessage(errorMessage(error, "Failed to save conflict copy"));
    } finally {
      setIsCreating(false);
    }
  }

  async function performConfirmedAction() {
    if (!confirmAction || !note) {
      return;
    }
    setIsConfirming(true);
    try {
      if (confirmAction.kind === "purge") {
        await permanentlyDeleteNote(note.id);
        removeCurrentFromList();
      } else if (confirmAction.kind === "trash") {
        const current = draft && isDirty ? await saveDraft(note, draft) : note;
        if (!current) {
          return;
        }
        await trashNote(current.id, current.current_version.version_number);
        removeCurrentFromList();
      } else {
        const current = draft && isDirty ? await saveDraft(note, draft) : note;
        if (!current) {
          return;
        }
        const restored = await restoreNoteVersion(
          current.id,
          confirmAction.version.id,
          current.current_version.version_number
        );
        applyLoadedNote(restored);
        await loadVersions(restored.id);
        setDrawer("history");
      }
      setConfirmAction(null);
    } catch (error) {
      setNoteError(errorMessage(error, "Note action failed"));
    } finally {
      setIsConfirming(false);
    }
  }

  async function restoreCurrentNote() {
    if (!note) {
      return;
    }
    try {
      await restoreNote(note.id, note.current_version.version_number);
      removeCurrentFromList();
    } catch (error) {
      setNoteError(errorMessage(error, "Failed to restore note"));
    }
  }

  function removeCurrentFromList() {
    if (!note) {
      return;
    }
    setSummaries((current) => current.filter((candidate) => candidate.id !== note.id));
    setTotal((current) => Math.max(0, current - 1));
    setSelectedNoteId(null);
    setIsMobileEditorOpen(false);
    setDrawer(null);
  }

  async function loadVersions(noteId: string) {
    setIsLoadingVersions(true);
    setHistoryError(null);
    try {
      const loaded = await listNoteVersions(noteId);
      setVersions(loaded);
      setSelectedVersionId(loaded[0]?.id ?? null);
    } catch (error) {
      setHistoryError(errorMessage(error, "Failed to load note history"));
    } finally {
      setIsLoadingVersions(false);
    }
  }

  async function openHistory() {
    if (!note) {
      return;
    }
    if (!(await flushCurrentDraft())) {
      return;
    }
    setDrawer("history");
    void loadVersions(note.id);
  }

  async function openSettings() {
    setDrawer("settings");
    if (settings) {
      setSettingsDraft(cloneSettings(settings));
      return;
    }
    setIsLoadingSettings(true);
    setSettingsError(null);
    try {
      const loaded = await getNoteSettings();
      setSettings(loaded);
      setSettingsDraft(cloneSettings(loaded));
    } catch (error) {
      setSettingsError(errorMessage(error, "Failed to load note settings"));
    } finally {
      setIsLoadingSettings(false);
    }
  }

  async function saveSettings() {
    if (!settingsDraft) {
      return;
    }
    setIsSavingSettings(true);
    setSettingsError(null);
    try {
      const updated = await updateNoteSettings({
        ...settingsDraft,
        default_model_scope: normalizeScope(settingsDraft.default_model_scope)
      });
      setSettings(updated);
      setSettingsDraft(cloneSettings(updated));
      setDrawer(null);
    } catch (error) {
      setSettingsError(errorMessage(error, "Failed to save note settings"));
    } finally {
      setIsSavingSettings(false);
    }
  }

  function updateDraft(patch: Partial<NoteDraft>) {
    const current = draftRef.current;
    if (!current) {
      return;
    }
    const next = { ...current, ...patch };
    draftRef.current = next;
    setDraft(next);
    if (saveState === "error" && saveMessage === "A title is required") {
      setSaveState("idle");
      setSaveMessage(null);
    }
  }

  function commitTags() {
    updateDraft({ tags: parseTags(tagsText) });
  }

  async function switchLibrary(nextStatus: NoteListStatus) {
    if (nextStatus === status || !(await flushCurrentDraft())) {
      return;
    }
    setStatus(nextStatus);
    setSelectedNoteId(null);
    setIsMobileEditorOpen(false);
    setDrawer(null);
  }

  return (
    <section className="notes-workspace">
      <header className="notes-workspace-header">
        <div>
          <p className="eyebrow">Personal</p>
          <h1>Notes</h1>
        </div>
        <div className="notes-workspace-actions">
          <button type="button" className="secondary-button" onClick={() => void openSettings()}>
            <Settings2 />
            <span>Note Access</span>
          </button>
          <button type="button" onClick={() => void createNewNote()} disabled={isCreating}>
            {isCreating ? <RetroLoader /> : <FilePlus2 />}
            <span>New Note</span>
          </button>
        </div>
      </header>

      <div className={isMobileEditorOpen ? "notes-layout notes-mobile-editor-open" : "notes-layout"}>
        <aside className="notes-library" aria-label="Note library">
          <div className="notes-library-controls">
            <div className="segmented-control notes-status-control">
              <button
                type="button"
                className={status === "active" ? "active" : ""}
                onClick={() => void switchLibrary("active")}
              >
                Notes
              </button>
              <button
                type="button"
                className={status === "trashed" ? "active" : ""}
                onClick={() => void switchLibrary("trashed")}
              >
                Trash
              </button>
            </div>
            <label className="notes-search">
              <Search />
              <span className="visually-hidden">Search notes</span>
              <input
                type="search"
                value={searchInput}
                placeholder="Search notes"
                onChange={(event) => setSearchInput(event.target.value)}
              />
              {searchInput && (
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Clear search"
                  onClick={() => setSearchInput("")}
                >
                  <X />
                </button>
              )}
            </label>
            <div className="notes-library-meta">
              <span>{total} {total === 1 ? "note" : "notes"}</span>
              <div ref={listMenuRef} className="notes-list-menu-wrap">
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Sort and refresh notes"
                  aria-expanded={isListMenuOpen}
                  onClick={() => setIsListMenuOpen((open) => !open)}
                >
                  <MoreHorizontal />
                </button>
                {isListMenuOpen && (
                  <div className="notes-list-menu">
                    <label>
                      Sort
                      <select value={sort} onChange={(event) => setSort(event.target.value as NoteSort)}>
                        <option value="updated">Recently updated</option>
                        <option value="created">Recently created</option>
                        <option value="title">Title</option>
                      </select>
                    </label>
                    <button
                      type="button"
                      className="menu-item"
                      onClick={() => {
                        setIsListMenuOpen(false);
                        setListRefreshKey((current) => current + 1);
                      }}
                    >
                      <RefreshCw />
                      <span>Refresh</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="notes-list">
            {isLoadingList && summaries.length === 0 ? (
              <div className="notes-list-state"><RetroLoader /></div>
            ) : listError ? (
              <div className="notes-list-state">
                <p className="error">{listError}</p>
                <button type="button" className="secondary-button" onClick={() => setListRefreshKey((key) => key + 1)}>
                  <RefreshCw /> Retry
                </button>
              </div>
            ) : summaries.length === 0 ? (
              <div className="notes-list-state">
                <FileText />
                <strong>{query ? "No matching notes" : status === "trashed" ? "Trash is empty" : "No notes yet"}</strong>
                {!query && status === "active" && (
                  <button type="button" className="secondary-button" onClick={() => void createNewNote()}>
                    <FilePlus2 /> New Note
                  </button>
                )}
              </div>
            ) : (
              summaries.map((summary) => (
                <button
                  type="button"
                  key={summary.id}
                  className={summary.id === selectedNoteId ? "note-list-item note-list-item-active" : "note-list-item"}
                  onClick={() => void selectNote(summary.id)}
                >
                  <span className="note-list-title-row">
                    <strong>{summary.title}</strong>
                    {summary.is_pinned && <Pin aria-label="Pinned" />}
                  </span>
                  <span className="note-list-excerpt">{summary.excerpt || "Empty note"}</span>
                  <span className="note-list-footer">
                    <span>{formatRelativeTime(summary.updated_at)}</span>
                    {summary.tags.length > 0 && <span>{summary.tags.slice(0, 2).join(" · ")}</span>}
                  </span>
                </button>
              ))
            )}
            {total > NOTE_LIST_LIMIT && (
              <p className="notes-list-limit">Showing the first {NOTE_LIST_LIMIT} results. Refine your search to narrow the list.</p>
            )}
          </div>
        </aside>

        <article className="note-editor-pane">
          {!selectedNoteId ? (
            <div className="notes-empty-editor">
              <FileText />
              <h2>{status === "trashed" ? "Select a trashed note" : "Select a note"}</h2>
              <p>{status === "trashed" ? "Restore or permanently remove notes from here." : "Choose a note from the library or create a new one."}</p>
            </div>
          ) : isLoadingNote ? (
            <div className="notes-empty-editor"><RetroLoader /></div>
          ) : noteError && !note ? (
            <div className="notes-empty-editor">
              <p className="error">{noteError}</p>
              <button type="button" className="secondary-button" onClick={() => void reloadSelectedNote()}>
                <RefreshCw /> Retry
              </button>
            </div>
          ) : note && draft ? (
            <>
              <header className="note-editor-header">
                <button
                  type="button"
                  className="icon-button notes-mobile-back"
                  aria-label="Back to notes"
                  onClick={() => setIsMobileEditorOpen(false)}
                >
                  <ArrowLeft />
                </button>
                <input
                  ref={titleRef}
                  className="note-title-input"
                  value={draft.title}
                  aria-label="Note title"
                  readOnly={Boolean(note.deleted_at)}
                  onChange={(event) => updateDraft({ title: event.target.value })}
                  onBlur={() => void flushCurrentDraft()}
                />
                <div className="note-save-status" role="status" aria-live="polite">
                  {saveState === "saving" ? <><RefreshCw className="spin" /> Saving</> : saveState === "saved" ? <><Check /> Saved</> : saveState === "error" || saveState === "conflict" ? saveMessage : null}
                </div>
                <div className="note-editor-actions">
                  {note.deleted_at ? (
                    <>
                      <button type="button" className="secondary-button" onClick={() => void restoreCurrentNote()}>
                        <Undo2 /> <span>Restore</span>
                      </button>
                      <button type="button" className="danger-button" onClick={() => setConfirmAction({ kind: "purge" })}>
                        <Trash2 /> <span>Delete Forever</span>
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="icon-button"
                        aria-label={draft.is_pinned ? "Unpin note" : "Pin note"}
                        title={draft.is_pinned ? "Unpin note" : "Pin note"}
                        onClick={() => updateDraft({ is_pinned: !draft.is_pinned })}
                      >
                        {draft.is_pinned ? <PinOff /> : <Pin />}
                      </button>
                      <button type="button" className="icon-button" aria-label="Version history" title="Version history" onClick={() => void openHistory()}>
                        <History />
                      </button>
                      <button type="button" className="icon-button" aria-label="Note access and tags" title="Note access and tags" onClick={() => setDrawer("details")}>
                        <Settings2 />
                      </button>
                      <button type="button" className="icon-button danger-button" aria-label="Move to trash" title="Move to trash" onClick={() => setConfirmAction({ kind: "trash" })}>
                        <Trash2 />
                      </button>
                    </>
                  )}
                </div>
              </header>

              {saveState === "conflict" && (
                <div className="note-conflict-banner" role="alert">
                  <div>
                    <strong>This note changed elsewhere.</strong>
                    <span>Your draft is still here. Reload the latest version or preserve this draft as a copy.</span>
                  </div>
                  <button type="button" className="secondary-button" onClick={() => void reloadSelectedNote()}>Reload</button>
                  <button type="button" onClick={() => void saveConflictCopy()} disabled={isCreating}>Save as Copy</button>
                </div>
              )}
              {noteError && <p className="error note-editor-error">{noteError}</p>}

              <div className="note-editor-modebar">
                <div className="segmented-control">
                  <button type="button" className={editorMode === "edit" ? "active" : ""} onClick={() => setEditorMode("edit")}>
                    Edit
                  </button>
                  <button type="button" className={editorMode === "preview" ? "active" : ""} onClick={() => setEditorMode("preview")}>
                    <Eye /> Preview
                  </button>
                </div>
                <span>v{note.current_version.version_number}</span>
              </div>

              {editorMode === "edit" && !note.deleted_at ? (
                <div className="note-markdown-editor" onBlur={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) {
                    void flushCurrentDraft();
                  }
                }}>
                  <MarkdownToolbar textareaRef={textareaRef} value={draft.content} onChange={(content) => updateDraft({ content })} />
                  <textarea
                    ref={textareaRef}
                    value={draft.content}
                    aria-label="Note content"
                    placeholder="Write in Markdown..."
                    spellCheck
                    onChange={(event) => updateDraft({ content: event.target.value })}
                    onKeyDown={(event) => {
                      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
                        event.preventDefault();
                        void flushCurrentDraft();
                      }
                    }}
                  />
                </div>
              ) : (
                <div className="note-preview" tabIndex={0}>
                  {draft.content.trim() ? <MarkdownContent content={draft.content} /> : <p className="notes-empty-copy">This note is empty.</p>}
                </div>
              )}
            </>
          ) : null}
        </article>

        {drawer && (
          <>
            <button type="button" className="notes-drawer-backdrop" aria-label="Close panel" onClick={() => setDrawer(null)} />
            <aside className="notes-drawer" aria-label={drawerTitle(drawer)}>
              <header>
                <div>
                  <p className="eyebrow">Notes</p>
                  <h2>{drawerTitle(drawer)}</h2>
                </div>
                <button type="button" className="icon-button" aria-label="Close panel" onClick={() => setDrawer(null)}><X /></button>
              </header>
              {drawer === "details" && draft && note && (
                <NoteDetails
                  draft={draft}
                  tagsText={tagsText}
                  modelOptions={modelOptions}
                  onDraftChange={updateDraft}
                  onTagsTextChange={setTagsText}
                  onTagsCommit={commitTags}
                />
              )}
              {drawer === "history" && note && (
                <NoteHistory
                  currentVersionId={note.current_version.id}
                  versions={versions}
                  selectedVersion={selectedVersion}
                  isLoading={isLoadingVersions}
                  error={historyError}
                  onRefresh={() => void loadVersions(note.id)}
                  onSelect={setSelectedVersionId}
                  onRestore={(version) => setConfirmAction({ kind: "restore-version", version })}
                />
              )}
              {drawer === "settings" && (
                <NoteAccessSettings
                  settings={settingsDraft}
                  isLoading={isLoadingSettings}
                  isSaving={isSavingSettings}
                  error={settingsError}
                  modelOptions={modelOptions}
                  onChange={setSettingsDraft}
                  onCancel={() => setDrawer(null)}
                  onSave={() => void saveSettings()}
                />
              )}
            </aside>
          </>
        )}
      </div>

      {confirmAction && (
        <ConfirmDialog
          title={confirmTitle(confirmAction)}
          message={confirmMessage(confirmAction, note)}
          confirmLabel={confirmAction.kind === "trash" ? "Move to Trash" : confirmAction.kind === "purge" ? "Delete Forever" : "Restore Version"}
          confirmTone={confirmAction.kind === "restore-version" ? "primary" : "danger"}
          isBusy={isConfirming}
          onCancel={() => setConfirmAction(null)}
          onConfirm={() => void performConfirmedAction()}
        />
      )}
    </section>
  );
}

function NoteDetails({
  draft,
  tagsText,
  modelOptions,
  onDraftChange,
  onTagsTextChange,
  onTagsCommit
}: {
  draft: NoteDraft;
  tagsText: string;
  modelOptions: ModelOption[];
  onDraftChange: (patch: Partial<NoteDraft>) => void;
  onTagsTextChange: (value: string) => void;
  onTagsCommit: () => void;
}) {
  return (
    <div className="notes-drawer-content">
      <label>
        Tags
        <input
          value={tagsText}
          placeholder="project, research"
          onChange={(event) => onTagsTextChange(event.target.value)}
          onBlur={onTagsCommit}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onTagsCommit();
              event.currentTarget.blur();
            }
          }}
        />
        <small>Separate tags with commas.</small>
      </label>
      <section className="notes-access-section">
        <h3>AI Access</h3>
        <p>Choose the maximum action a model may take with this note.</p>
        <AccessLevelControl value={draft.ai_access} onChange={(ai_access) => onDraftChange({ ai_access })} />
      </section>
      <ModelScopeEditor
        scope={draft.model_scope}
        options={modelOptions}
        onChange={(model_scope) => onDraftChange({ model_scope })}
      />
    </div>
  );
}

function NoteHistory({
  currentVersionId,
  versions,
  selectedVersion,
  isLoading,
  error,
  onRefresh,
  onSelect,
  onRestore
}: {
  currentVersionId: string;
  versions: NoteVersion[];
  selectedVersion: NoteVersion | null;
  isLoading: boolean;
  error: string | null;
  onRefresh: () => void;
  onSelect: (versionId: string) => void;
  onRestore: (version: NoteVersion) => void;
}) {
  if (isLoading && versions.length === 0) {
    return <div className="notes-drawer-state"><RetroLoader /></div>;
  }
  if (error) {
    return <div className="notes-drawer-state"><p className="error">{error}</p><button type="button" className="secondary-button" onClick={onRefresh}><RefreshCw /> Retry</button></div>;
  }

  return (
    <div className="note-history-layout">
      <div className="note-version-list">
        {versions.map((version) => (
          <button
            type="button"
            key={version.id}
            className={version.id === selectedVersion?.id ? "note-version-item note-version-item-active" : "note-version-item"}
            onClick={() => onSelect(version.id)}
          >
            <span><strong>v{version.version_number}</strong>{version.id === currentVersionId && <em>Current</em>}</span>
            <small>{version.actor_type === "model" ? version.actor_model_name ?? "Model" : "You"} · {formatDateTime(version.created_at)}</small>
          </button>
        ))}
      </div>
      {selectedVersion && (
        <div className="note-version-preview">
          <header>
            <div><strong>{selectedVersion.title}</strong><small>Version {selectedVersion.version_number}</small></div>
            {selectedVersion.id !== currentVersionId && (
              <button type="button" className="secondary-button" onClick={() => onRestore(selectedVersion)}><Undo2 /> Restore</button>
            )}
          </header>
          <MarkdownContent content={selectedVersion.content || "_Empty note_"} />
        </div>
      )}
    </div>
  );
}

function NoteAccessSettings({
  settings,
  isLoading,
  isSaving,
  error,
  modelOptions,
  onChange,
  onCancel,
  onSave
}: {
  settings: NoteSettings | null;
  isLoading: boolean;
  isSaving: boolean;
  error: string | null;
  modelOptions: ModelOption[];
  onChange: (settings: NoteSettings) => void;
  onCancel: () => void;
  onSave: () => void;
}) {
  if (isLoading || !settings) {
    return <div className="notes-drawer-state">{error ? <p className="error">{error}</p> : <RetroLoader />}</div>;
  }

  return (
    <div className="notes-drawer-content notes-settings-content">
      <p>These limits apply to every model before a note's own access setting is considered.</p>
      <div className="notes-permission-list">
        <PermissionToggle label="Read notes" checked={settings.allow_model_read} onChange={(allow_model_read) => onChange({ ...settings, allow_model_read })} />
        <PermissionToggle label="Create notes" checked={settings.allow_model_create} onChange={(allow_model_create) => onChange({ ...settings, allow_model_create })} />
        <PermissionToggle label="Edit notes" checked={settings.allow_model_edit} onChange={(allow_model_edit) => onChange({ ...settings, allow_model_edit })} />
        <PermissionToggle label="Move notes to trash" checked={settings.allow_model_trash} onChange={(allow_model_trash) => onChange({ ...settings, allow_model_trash })} />
      </div>
      <section className="notes-access-section">
        <h3>New Note Default</h3>
        <p>Default AI access for notes you create later.</p>
        <AccessLevelControl value={settings.default_ai_access} onChange={(default_ai_access) => onChange({ ...settings, default_ai_access })} />
      </section>
      <ModelScopeEditor scope={settings.default_model_scope} options={modelOptions} onChange={(default_model_scope) => onChange({ ...settings, default_model_scope })} title="New Note Model Scope" />
      {error && <p className="error">{error}</p>}
      <div className="notes-drawer-actions">
        <button type="button" className="secondary-button" disabled={isSaving} onClick={onCancel}>Cancel</button>
        <button type="button" disabled={isSaving} onClick={onSave}>{isSaving ? <RetroLoader /> : "Save"}</button>
      </div>
    </div>
  );
}

function AccessLevelControl({
  value,
  onChange
}: {
  value: NoteAiAccess;
  onChange: (value: NoteAiAccess) => void;
}) {
  const levels: Array<{ value: NoteAiAccess; label: string }> = [
    { value: "none", label: "None" },
    { value: "read", label: "Read" },
    { value: "edit", label: "Edit" },
    { value: "manage", label: "Manage" }
  ];
  return (
    <div className="segmented-control notes-access-control">
      {levels.map((level) => (
        <button type="button" key={level.value} className={value === level.value ? "active" : ""} onClick={() => onChange(level.value)}>
          {level.label}
        </button>
      ))}
    </div>
  );
}

function ModelScopeEditor({
  scope,
  options,
  onChange,
  title = "Model Scope"
}: {
  scope: NoteModelScope;
  options: ModelOption[];
  onChange: (scope: NoteModelScope) => void;
  title?: string;
}) {
  const [query, setQuery] = useState("");
  const filtered = options.filter((option) => `${option.label} ${option.group}`.toLowerCase().includes(query.toLowerCase()));
  const grouped = filtered.reduce<Map<string, ModelOption[]>>((current, option) => {
    const group = current.get(option.group) ?? [];
    group.push(option);
    current.set(option.group, group);
    return current;
  }, new Map());

  function toggleModel(key: string, checked: boolean) {
    const next = new Set(scope.model_keys);
    if (checked) {
      next.add(key);
    } else {
      next.delete(key);
    }
    onChange({ all_models: false, model_keys: [...next] });
  }

  return (
    <section className="notes-model-scope">
      <h3>{title}</h3>
      <p>Custom models are scoped independently from their base model.</p>
      <PermissionToggle label="All models" checked={scope.all_models} onChange={(all_models) => onChange({ all_models, model_keys: scope.model_keys })} />
      {!scope.all_models && (
        <>
          <label className="notes-scope-search">
            <Search />
            <span className="visually-hidden">Search models</span>
            <input value={query} placeholder="Search models" onChange={(event) => setQuery(event.target.value)} />
          </label>
          <div className="notes-model-options">
            {[...grouped.entries()].map(([group, groupOptions]) => (
              <section key={group}>
                <h4>{group}</h4>
                {groupOptions.map((option) => (
                  <PermissionToggle
                    key={option.key}
                    label={option.label}
                    checked={scope.model_keys.includes(option.key)}
                    onChange={(checked) => toggleModel(option.key, checked)}
                  />
                ))}
              </section>
            ))}
            {filtered.length === 0 && <p className="notes-empty-copy">No matching models.</p>}
          </div>
        </>
      )}
    </section>
  );
}

function PermissionToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="notes-permission-toggle">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function noteToDraft(note: Note): NoteDraft {
  return {
    title: note.current_version.title,
    content: note.current_version.content,
    tags: [...note.tags],
    is_pinned: note.is_pinned,
    ai_access: note.ai_access,
    model_scope: { ...note.model_scope, model_keys: [...note.model_scope.model_keys] }
  };
}

function summaryFromNote(note: Note): NoteSummary {
  return {
    id: note.id,
    title: note.current_version.title,
    excerpt: excerpt(note.current_version.content),
    current_version_id: note.current_version.id,
    current_version_number: note.current_version.version_number,
    tags: [...note.tags],
    is_pinned: note.is_pinned,
    ai_access: note.ai_access,
    model_scope: { ...note.model_scope, model_keys: [...note.model_scope.model_keys] },
    deleted_at: note.deleted_at,
    created_at: note.created_at,
    updated_at: note.updated_at
  };
}

function noteFingerprint(note: Note) {
  return draftFingerprint(noteToDraft(note));
}

function draftFingerprint(draft: NoteDraft) {
  return JSON.stringify({
    title: draft.title.trim(),
    content: draft.content,
    tags: normalizeTags(draft.tags),
    is_pinned: draft.is_pinned,
    ai_access: draft.ai_access,
    model_scope: normalizeScope(draft.model_scope)
  });
}

function normalizeTags(tags: string[]) {
  return [...new Set(tags.map((tag) => tag.trim()).filter(Boolean))].slice(0, 32);
}

function parseTags(value: string) {
  return normalizeTags(value.split(","));
}

function normalizeScope(scope: NoteModelScope): NoteModelScope {
  return {
    all_models: scope.all_models,
    model_keys: [...new Set(scope.model_keys)].sort()
  };
}

function sortSummaries(notes: NoteSummary[], sort: NoteSort) {
  return [...notes].sort((left, right) => {
    if (left.is_pinned !== right.is_pinned) {
      return left.is_pinned ? -1 : 1;
    }
    if (sort === "title") {
      return left.title.localeCompare(right.title);
    }
    return sort === "created" ? right.created_at - left.created_at : right.updated_at - left.updated_at;
  });
}

function excerpt(content: string) {
  return content.replace(/[#>*_`\[\]-]/g, " ").replace(/\s+/g, " ").trim().slice(0, 180);
}

function formatRelativeTime(timestamp: number) {
  const seconds = Math.max(0, Math.floor(Date.now() / 1000) - timestamp);
  if (seconds < 60) return "Just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(timestamp * 1000);
}

function formatDateTime(timestamp: number) {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(timestamp * 1000);
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function drawerTitle(drawer: Exclude<Drawer, null>) {
  if (drawer === "details") return "Note Access";
  if (drawer === "history") return "Version History";
  return "AI Note Defaults";
}

function confirmTitle(action: ConfirmAction) {
  if (action.kind === "trash") return "Move Note to Trash";
  if (action.kind === "purge") return "Delete Note Forever";
  return `Restore Version ${action.version.version_number}`;
}

function confirmMessage(action: ConfirmAction, note: Note | null) {
  if (action.kind === "trash") return `Move “${note?.current_version.title ?? "this note"}” to trash? You can restore it later.`;
  if (action.kind === "purge") return `Permanently delete “${note?.current_version.title ?? "this note"}” and all of its history? This cannot be undone.`;
  return `Restore version ${action.version.version_number}? The current content will remain available in history.`;
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
