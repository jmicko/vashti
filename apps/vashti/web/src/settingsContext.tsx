import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Blocks,
  Check,
  ChevronDown,
  HardDrive,
  Pencil,
  Plus,
  RefreshCw,
  Server,
  Trash2,
  X
} from "lucide-react";
import { requestJson } from "./api";
import { ConfirmDialog, RetroLoader } from "./common";
import {
  createPrivateContextBlock,
  createPrivateContextCategory,
  deletePrivateContextBlock,
  deletePrivateContextCategory,
  listPrivateContextLibrary,
  updatePrivateContextBlock,
  updatePrivateContextCategory
} from "./privateChatStore";
import { SettingsPanel } from "./settingsControls";
import type {
  ContextBlock,
  ContextCategory,
  ContextLibraryResponse,
  ContextSelectionMode
} from "./types";

type StorageMode = "server" | "device";
type EditorState =
  | { type: "category"; category: ContextCategory | null }
  | { type: "block"; block: ContextBlock | null };
type DeleteTarget =
  | { type: "category"; item: ContextCategory }
  | { type: "block"; item: ContextBlock };

const EMPTY_LIBRARY: ContextLibraryResponse = { categories: [], blocks: [] };

export function ContextSettingsPanel({
  onContextChanged
}: {
  onContextChanged: () => Promise<void>;
}) {
  const [storageMode, setStorageMode] = useState<StorageMode>("server");
  const [serverLibrary, setServerLibrary] = useState<ContextLibraryResponse>(EMPTY_LIBRARY);
  const [deviceLibrary, setDeviceLibrary] = useState<ContextLibraryResponse>(EMPTY_LIBRARY);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorState | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);

  const loadLibraries = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    const [serverResult, deviceResult] = await Promise.allSettled([
      requestJson<ContextLibraryResponse>("/api/context-library"),
      listPrivateContextLibrary()
    ]);
    if (serverResult.status === "fulfilled") {
      setServerLibrary(serverResult.value);
    }
    if (deviceResult.status === "fulfilled") {
      setDeviceLibrary(deviceResult.value);
    }
    const activeResult = storageMode === "server" ? serverResult : deviceResult;
    if (activeResult.status === "rejected") {
      setError(
        activeResult.reason instanceof Error
          ? activeResult.reason.message
          : "Failed to load context blocks"
      );
    }
    setIsLoading(false);
  }, [storageMode]);

  useEffect(() => {
    void loadLibraries();
  }, [loadLibraries]);

  const library = storageMode === "server" ? serverLibrary : deviceLibrary;
  const groupedBlocks = useMemo(() => {
    const groups = library.categories.map((category) => ({
      category,
      blocks: library.blocks.filter((block) => block.category_id === category.id)
    }));
    const uncategorized = library.blocks.filter((block) => !block.category_id);
    return uncategorized.length > 0
      ? [...groups, { category: null, blocks: uncategorized }]
      : groups;
  }, [library]);

  async function finishMutation() {
    await loadLibraries();
    await onContextChanged();
  }

  async function handleDelete() {
    if (!deleteTarget) return;
    setIsSaving(true);
    setError(null);
    try {
      if (storageMode === "server") {
        const path = deleteTarget.type === "category"
          ? `/api/context-categories/${deleteTarget.item.id}`
          : `/api/context-blocks/${deleteTarget.item.id}`;
        await requestJson(path, { method: "DELETE" });
      } else if (deleteTarget.type === "category") {
        await deletePrivateContextCategory(deleteTarget.item.id);
      } else {
        await deletePrivateContextBlock(deleteTarget.item.id);
      }
      setDeleteTarget(null);
      await finishMutation();
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Failed to delete context item");
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <SettingsPanel
      eyebrow="Personal"
      title="Context"
      width="standard"
      actions={
        <button type="button" className="secondary-button" disabled={isLoading} onClick={() => void loadLibraries()}>
          {isLoading ? <RetroLoader /> : <><RefreshCw /> Refresh</>}
        </button>
      }
    >
      <p className="settings-lead">
        Build reusable prompt fragments, then combine them per conversation without tying them to a model.
      </p>

      <div className="context-storage-switch" role="group" aria-label="Context storage">
        <button
          type="button"
          className={storageMode === "server" ? "context-storage-option active" : "context-storage-option"}
          onClick={() => setStorageMode("server")}
        >
          <Server />
          <span><strong>Server</strong><small>Available in standard chats on your devices.</small></span>
        </button>
        <button
          type="button"
          className={storageMode === "device" ? "context-storage-option active" : "context-storage-option"}
          onClick={() => setStorageMode("device")}
        >
          <HardDrive />
          <span><strong>This device</strong><small>Private-local chats only. Not synced.</small></span>
        </button>
      </div>

      <div className="context-library-toolbar">
        <div>
          <strong>{storageMode === "server" ? "Server library" : "Device library"}</strong>
          <span>{library.categories.length} categories · {library.blocks.length} blocks</span>
        </div>
        <div className="context-library-actions">
          <button type="button" className="secondary-button" onClick={() => setEditor({ type: "category", category: null })}>
            <Plus /> Category
          </button>
          <button type="button" onClick={() => setEditor({ type: "block", block: null })}>
            <Plus /> Block
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {isLoading ? (
        <div className="context-library-empty"><RetroLoader /></div>
      ) : groupedBlocks.length === 0 ? (
        <div className="context-library-empty">
          <Blocks />
          <strong>No context blocks yet</strong>
          <span>Create a category for organization, or add an uncategorized block.</span>
        </div>
      ) : (
        <div className="context-category-list">
          {groupedBlocks.map(({ category, blocks }) => (
            <section className="context-category" key={category?.id ?? "uncategorized"}>
              <header className="context-category-header">
                <div>
                  <strong>{category?.name ?? "Uncategorized"}</strong>
                  {category && (
                    <span>{category.selection_mode === "single" ? "Choose one" : "Choose multiple"}</span>
                  )}
                </div>
                {category && (
                  <div className="icon-action-row">
                    <button type="button" className="icon-button" title="Edit category" onClick={() => setEditor({ type: "category", category })}>
                      <Pencil />
                    </button>
                    <button type="button" className="icon-button danger-icon-button" title="Delete category" onClick={() => setDeleteTarget({ type: "category", item: category })}>
                      <Trash2 />
                    </button>
                  </div>
                )}
              </header>
              {blocks.length === 0 ? (
                <p className="context-category-empty">No blocks in this category.</p>
              ) : (
                <div className="context-block-list">
                  {blocks.map((block) => (
                    <article className="context-block-row" key={block.id}>
                      <div className="context-block-copy">
                        <strong>{block.current_version.name}</strong>
                        <span>v{block.current_version.version_number}</span>
                        <p>{block.current_version.content}</p>
                      </div>
                      <div className="icon-action-row">
                        <button type="button" className="icon-button" title="Edit block" onClick={() => setEditor({ type: "block", block })}>
                          <Pencil />
                        </button>
                        <button type="button" className="icon-button danger-icon-button" title="Delete block" onClick={() => setDeleteTarget({ type: "block", item: block })}>
                          <Trash2 />
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>
          ))}
        </div>
      )}

      {editor && (
        <ContextEditorDialog
          editor={editor}
          library={library}
          storageMode={storageMode}
          isSaving={isSaving}
          onCancel={() => setEditor(null)}
          onError={setError}
          onLibraryMutated={finishMutation}
          onSaved={async () => {
            setEditor(null);
            await finishMutation();
          }}
          setIsSaving={setIsSaving}
        />
      )}
      {deleteTarget && (
        <ConfirmDialog
          title={deleteTarget.type === "category" ? "Delete category?" : "Delete context block?"}
          message={deleteTarget.type === "category"
            ? "Blocks in this category will become uncategorized. Existing chat snapshots are unchanged."
            : "The block will leave your library. Existing chat snapshots are unchanged."}
          confirmLabel="Delete"
          isBusy={isSaving}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => void handleDelete()}
        />
      )}
    </SettingsPanel>
  );
}

function ContextEditorDialog({
  editor,
  library,
  storageMode,
  isSaving,
  onCancel,
  onError,
  onLibraryMutated,
  onSaved,
  setIsSaving
}: {
  editor: EditorState;
  library: ContextLibraryResponse;
  storageMode: StorageMode;
  isSaving: boolean;
  onCancel: () => void;
  onError: (message: string | null) => void;
  onLibraryMutated: () => Promise<void>;
  onSaved: () => Promise<void>;
  setIsSaving: (value: boolean) => void;
}) {
  const existingCategory = editor.type === "category" ? editor.category : null;
  const existingBlock = editor.type === "block" ? editor.block : null;
  const [name, setName] = useState(existingCategory?.name ?? existingBlock?.current_version.name ?? "");
  const [selectionMode, setSelectionMode] = useState<ContextSelectionMode>(existingCategory?.selection_mode ?? "single");
  const [categoryId, setCategoryId] = useState(existingBlock?.category_id ?? "");
  const [content, setContent] = useState(existingBlock?.current_version.content ?? "");
  const [isCategoryMenuOpen, setIsCategoryMenuOpen] = useState(false);
  const [isAddingCategory, setIsAddingCategory] = useState(false);
  const [newCategoryName, setNewCategoryName] = useState("");
  const [newCategorySelectionMode, setNewCategorySelectionMode] =
    useState<ContextSelectionMode>("single");
  const [createdCategories, setCreatedCategories] = useState<ContextCategory[]>([]);
  const [isCreatingCategory, setIsCreatingCategory] = useState(false);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const categoryPickerRef = useRef<HTMLDivElement>(null);

  const availableCategories = useMemo(() => {
    const categories = new Map<string, ContextCategory>();
    library.categories.forEach((category) => categories.set(category.id, category));
    createdCategories.forEach((category) => categories.set(category.id, category));
    return [...categories.values()].sort(
      (left, right) => left.sort_order - right.sort_order || left.name.localeCompare(right.name)
    );
  }, [createdCategories, library.categories]);
  const selectedCategory = availableCategories.find((category) => category.id === categoryId);

  useEffect(() => {
    function closeCategoryMenu(event: MouseEvent) {
      if (!categoryPickerRef.current?.contains(event.target as Node)) {
        setIsCategoryMenuOpen(false);
      }
    }

    function closeCategoryMenuWithEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsCategoryMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", closeCategoryMenu);
    document.addEventListener("keydown", closeCategoryMenuWithEscape);
    return () => {
      document.removeEventListener("mousedown", closeCategoryMenu);
      document.removeEventListener("keydown", closeCategoryMenuWithEscape);
    };
  }, []);

  function cancelInlineCategory() {
    setIsAddingCategory(false);
    setNewCategoryName("");
    setNewCategorySelectionMode("single");
    setDialogError(null);
  }

  async function createInlineCategory() {
    if (!newCategoryName.trim()) {
      setDialogError("Enter a category name");
      return;
    }

    setIsCreatingCategory(true);
    setDialogError(null);
    onError(null);
    try {
      let category: ContextCategory;
      if (storageMode === "server") {
        const response = await requestJson<{ category: ContextCategory }>(
          "/api/context-categories",
          {
            method: "POST",
            body: JSON.stringify({
              name: newCategoryName,
              selection_mode: newCategorySelectionMode
            })
          }
        );
        category = response.category;
      } else {
        category = await createPrivateContextCategory({
          name: newCategoryName,
          selectionMode: newCategorySelectionMode
        });
      }

      setCreatedCategories((current) => [...current, category]);
      setCategoryId(category.id);
      cancelInlineCategory();
      await onLibraryMutated();
    } catch (categoryError) {
      setDialogError(
        categoryError instanceof Error ? categoryError.message : "Failed to create category"
      );
    } finally {
      setIsCreatingCategory(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setIsSaving(true);
    setDialogError(null);
    onError(null);
    try {
      if (editor.type === "category") {
        if (storageMode === "server") {
          const path = existingCategory
            ? `/api/context-categories/${existingCategory.id}`
            : "/api/context-categories";
          await requestJson(path, {
            method: existingCategory ? "PATCH" : "POST",
            body: JSON.stringify({ name, selection_mode: selectionMode })
          });
        } else if (existingCategory) {
          await updatePrivateContextCategory(existingCategory.id, { name, selectionMode });
        } else {
          await createPrivateContextCategory({ name, selectionMode });
        }
      } else if (storageMode === "server") {
        const path = existingBlock ? `/api/context-blocks/${existingBlock.id}` : "/api/context-blocks";
        await requestJson(path, {
          method: existingBlock ? "PATCH" : "POST",
          body: JSON.stringify({ category_id: categoryId || null, name, content })
        });
      } else if (existingBlock) {
        await updatePrivateContextBlock(existingBlock.id, {
          categoryId: categoryId || null,
          name,
          content
        });
      } else {
        await createPrivateContextBlock({ categoryId: categoryId || null, name, content });
      }
      await onSaved();
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "Failed to save context item";
      setDialogError(message);
      onError(message);
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="confirm-backdrop" role="presentation">
      <section className="context-editor-dialog" role="dialog" aria-modal="true" aria-labelledby="context-editor-title">
        <header>
          <div>
            <p className="eyebrow">{storageMode === "server" ? "Server" : "This device"}</p>
            <h2 id="context-editor-title">
              {existingCategory || existingBlock ? "Edit" : "New"} {editor.type === "category" ? "Category" : "Context Block"}
            </h2>
          </div>
          <button type="button" className="icon-button" aria-label="Close editor" onClick={onCancel}><X /></button>
        </header>
        <form onSubmit={submit}>
          <label className="setting-field">
            <span>Name</span>
            <input value={name} maxLength={editor.type === "category" ? 80 : 120} autoFocus onChange={(event) => setName(event.target.value)} />
          </label>
          {editor.type === "category" ? (
            <div className="setting-field">
              <span>Selection</span>
              <div className="segmented-control">
                <button type="button" className={selectionMode === "single" ? "active" : ""} onClick={() => setSelectionMode("single")}>Choose one</button>
                <button type="button" className={selectionMode === "multiple" ? "active" : ""} onClick={() => setSelectionMode("multiple")}>Choose multiple</button>
              </div>
              <small>This rule is enforced when blocks are added to a conversation.</small>
            </div>
          ) : (
            <>
              <div className="setting-field">
                <span>Category</span>
                <div className="context-category-picker" ref={categoryPickerRef}>
                  <button
                    type="button"
                    className="context-category-picker-button"
                    aria-haspopup="listbox"
                    aria-expanded={isCategoryMenuOpen}
                    onClick={() => setIsCategoryMenuOpen((open) => !open)}
                  >
                    <span>{selectedCategory?.name ?? "Uncategorized"}</span>
                    <ChevronDown />
                  </button>
                  {isCategoryMenuOpen && (
                    <div className="context-category-picker-menu" role="listbox" aria-label="Context category">
                      <button
                        type="button"
                        className={categoryId === "" ? "context-category-picker-option selected" : "context-category-picker-option"}
                        role="option"
                        aria-selected={categoryId === ""}
                        onClick={() => {
                          setCategoryId("");
                          setIsCategoryMenuOpen(false);
                          cancelInlineCategory();
                        }}
                      >
                        <span>Uncategorized</span>
                        {categoryId === "" && <Check />}
                      </button>
                      {availableCategories.map((category) => (
                        <button
                          type="button"
                          className={category.id === categoryId ? "context-category-picker-option selected" : "context-category-picker-option"}
                          role="option"
                          aria-selected={category.id === categoryId}
                          key={category.id}
                          onClick={() => {
                            setCategoryId(category.id);
                            setIsCategoryMenuOpen(false);
                            cancelInlineCategory();
                          }}
                        >
                          <span>
                            <strong>{category.name}</strong>
                            <small>{category.selection_mode === "single" ? "Choose one" : "Choose multiple"}</small>
                          </span>
                          {category.id === categoryId && <Check />}
                        </button>
                      ))}
                      <button
                        type="button"
                        className="context-category-picker-option context-category-picker-add"
                        onClick={() => {
                          setIsCategoryMenuOpen(false);
                          setIsAddingCategory(true);
                        }}
                      >
                        <Plus />
                        <span>Add New Category</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
              {isAddingCategory && (
                <section className="context-inline-category-editor" aria-label="New category">
                  <header>
                    <strong>New Category</strong>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label="Cancel new category"
                      disabled={isCreatingCategory}
                      onClick={cancelInlineCategory}
                    >
                      <X />
                    </button>
                  </header>
                  <label className="setting-field">
                    <span>Name</span>
                    <input
                      value={newCategoryName}
                      maxLength={80}
                      autoFocus
                      onChange={(event) => setNewCategoryName(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          void createInlineCategory();
                        }
                      }}
                    />
                  </label>
                  <div className="setting-field">
                    <span>Selection</span>
                    <div className="segmented-control">
                      <button
                        type="button"
                        className={newCategorySelectionMode === "single" ? "active" : ""}
                        onClick={() => setNewCategorySelectionMode("single")}
                      >
                        Choose one
                      </button>
                      <button
                        type="button"
                        className={newCategorySelectionMode === "multiple" ? "active" : ""}
                        onClick={() => setNewCategorySelectionMode("multiple")}
                      >
                        Choose multiple
                      </button>
                    </div>
                  </div>
                  <button
                    type="button"
                    disabled={isCreatingCategory || !newCategoryName.trim()}
                    onClick={() => void createInlineCategory()}
                  >
                    {isCreatingCategory ? <RetroLoader /> : <><Plus /> Add Category</>}
                  </button>
                </section>
              )}
              <label className="setting-field">
                <span>Content</span>
                <textarea value={content} maxLength={60_000} rows={12} onChange={(event) => setContent(event.target.value)} />
                <small>Editing the name or content creates a new immutable version.</small>
              </label>
            </>
          )}
          {dialogError && <p className="error">{dialogError}</p>}
          <div className="dialog-actions">
            <button type="button" className="secondary-button" disabled={isSaving} onClick={onCancel}>Cancel</button>
            <button type="submit" disabled={isSaving || isCreatingCategory || isAddingCategory}>
              {isSaving ? <RetroLoader /> : "Save"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}
