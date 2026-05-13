import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Plus, X } from "lucide-react";
import type { PermissionTag } from "./types";

function mergePermissionTags(...tagGroups: PermissionTag[][]) {
  const merged = new Map<string, PermissionTag>();
  for (const tags of tagGroups) {
    for (const tag of tags) {
      if (!merged.has(tag.id)) {
        merged.set(tag.id, tag);
      }
    }
  }

  return Array.from(merged.values());
}

function permissionTagFromInput(value: string, availableTags: PermissionTag[]): PermissionTag | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const existing = availableTags.find(
    (tag) =>
      tag.id.toLowerCase() === trimmed.toLowerCase() ||
      tag.label.toLowerCase() === trimmed.toLowerCase()
  );
  return existing ?? { id: trimmed, label: trimmed, kind: "group" };
}

function isCoarseTouch() {
  if (typeof window === "undefined" || typeof window.matchMedia === "undefined") {
    return false;
  }

  return window.matchMedia("(hover: none) and (pointer: coarse)").matches;
}

function isInsidePermissionTag(target: EventTarget | null, tagId: string) {
  if (!(target instanceof Element)) {
    return false;
  }

  const tagElement = target.closest<HTMLElement>("[data-permission-tag-id]");
  return tagElement?.dataset.permissionTagId === tagId;
}

export function PermissionTagEditor({
  label,
  tags,
  availableTags,
  onChange,
  disabled = false,
  suggestionsKind,
  showEmpty = true
}: {
  label?: string;
  tags: PermissionTag[];
  availableTags: PermissionTag[];
  onChange: (tags: PermissionTag[]) => void;
  disabled?: boolean;
  suggestionsKind?: string;
  showEmpty?: boolean;
}) {
  const [value, setValue] = useState("");
  const [isSuggestionsOpen, setIsSuggestionsOpen] = useState(false);
  const [isInputFocused, setIsInputFocused] = useState(false);
  const [highlightedSuggestionIndex, setHighlightedSuggestionIndex] = useState(0);
  const [armedTagId, setArmedTagId] = useState<string | null>(null);
  const tagAddRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const inputId = useId();
  const suggestionListId = useId();
  const suggestions = useMemo(
    () =>
      suggestionsKind
        ? mergePermissionTags(availableTags.filter((tag) => tag.kind === suggestionsKind))
        : mergePermissionTags(availableTags),
    [availableTags, suggestionsKind]
  );
  const availableSuggestions = useMemo(() => {
    const existingIds = new Set(tags.map((tag) => tag.id));
    const query = value.trim().toLowerCase();
    return suggestions
      .filter((tag) => !existingIds.has(tag.id))
      .filter((tag) => {
        if (!query) {
          return true;
        }

        return (
          tag.id.toLowerCase().includes(query) ||
          tag.label.toLowerCase().includes(query)
        );
      })
      .slice(0, 8);
  }, [suggestions, tags, value]);
  const shouldShowSuggestions =
    !disabled && isInputFocused && isSuggestionsOpen && availableSuggestions.length > 0;
  const activeSuggestion =
    shouldShowSuggestions ? availableSuggestions[highlightedSuggestionIndex] ?? null : null;

  useEffect(() => {
    setHighlightedSuggestionIndex((current) =>
      Math.min(Math.max(0, current), Math.max(0, availableSuggestions.length - 1))
    );
  }, [availableSuggestions.length]);

  useEffect(() => {
    if (!armedTagId && !isSuggestionsOpen) {
      return;
    }

    function handleOutsideInteraction(event: MouseEvent | TouchEvent) {
      const target = event.target;
      const isInsideTagAdd = target instanceof Node && tagAddRef.current?.contains(target);

      if (armedTagId && !isInsidePermissionTag(target, armedTagId)) {
        setArmedTagId(null);
      }

      if (isSuggestionsOpen && !isInsideTagAdd) {
        if (event.type === "touchstart" && isCoarseTouch()) {
          setIsSuggestionsOpen(true);
          return;
        }

        setIsSuggestionsOpen(false);
        if (event.type === "mousedown") {
          inputRef.current?.blur();
          setIsInputFocused(false);
        }
      }
    }

    document.addEventListener("mousedown", handleOutsideInteraction);
    document.addEventListener("touchstart", handleOutsideInteraction);
    return () => {
      document.removeEventListener("mousedown", handleOutsideInteraction);
      document.removeEventListener("touchstart", handleOutsideInteraction);
    };
  }, [armedTagId, isSuggestionsOpen]);

  function addTag(selectedTag?: PermissionTag) {
    const tag = selectedTag ?? permissionTagFromInput(value, suggestions);
    if (!tag || tags.some((existing) => existing.id === tag.id)) {
      setValue("");
      setIsSuggestionsOpen(isInputFocused);
      return;
    }
    onChange([...tags, tag]);
    setValue("");
    setIsSuggestionsOpen(true);
    setHighlightedSuggestionIndex(0);
    window.requestAnimationFrame(() => inputRef.current?.focus());
  }

  return (
    <div className="permission-tag-editor">
      {label && <span>{label}</span>}
      <div className="permission-tag-row">
        <div className="permission-tags">
          {tags.length === 0 && showEmpty ? (
            <span className="permission-tag-empty">No tags</span>
          ) : (
            tags.map((tag) => (
              <PermissionTagChip
                key={tag.id}
                tag={tag}
                disabled={disabled}
                isArmed={armedTagId === tag.id}
                onArm={() => setArmedTagId(tag.id)}
                onAction={() => onChange(tags.filter((existing) => existing.id !== tag.id))}
              />
            ))
          )}
        </div>
        <div className="permission-tag-add" ref={tagAddRef}>
          <input
            ref={inputRef}
            id={inputId}
            type="search"
            name="vashti-permission-tag-search"
            value={value}
            disabled={disabled}
            placeholder="tag"
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="none"
            spellCheck={false}
            enterKeyHint="done"
            role="combobox"
            aria-autocomplete="list"
            aria-controls={suggestionListId}
            aria-expanded={shouldShowSuggestions}
            aria-activedescendant={
              activeSuggestion ? `${suggestionListId}-${activeSuggestion.id}` : undefined
            }
            style={{ width: `${Math.max(6, value.length + 2)}ch` }}
            onChange={(event) => {
              setValue(event.target.value);
              setIsSuggestionsOpen(true);
              setHighlightedSuggestionIndex(0);
            }}
            onFocus={() => {
              setIsInputFocused(true);
              setIsSuggestionsOpen(true);
              setHighlightedSuggestionIndex(0);
            }}
            onBlur={() => setIsInputFocused(false)}
            onPointerDown={() => setIsSuggestionsOpen(true)}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setIsSuggestionsOpen(true);
                setHighlightedSuggestionIndex((current) =>
                  availableSuggestions.length === 0
                    ? 0
                    : (current + 1) % availableSuggestions.length
                );
                return;
              }
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setIsSuggestionsOpen(true);
                setHighlightedSuggestionIndex((current) =>
                  availableSuggestions.length === 0
                    ? 0
                    : (current - 1 + availableSuggestions.length) % availableSuggestions.length
                );
                return;
              }
              if (event.key === "Enter") {
                event.preventDefault();
                event.stopPropagation();
                addTag(activeSuggestion ?? undefined);
              }
              if (event.key === "Escape") {
                setIsSuggestionsOpen(false);
              }
            }}
          />
          <button
            type="button"
            className="secondary-button"
            disabled={disabled}
            onClick={() => addTag()}
          >
            <Plus />
          </button>
          {shouldShowSuggestions && (
            <div id={suggestionListId} className="permission-tag-suggestions" role="listbox">
              {availableSuggestions.map((tag, index) => (
                <button
                  key={tag.id}
                  id={`${suggestionListId}-${tag.id}`}
                  type="button"
                  role="option"
                  aria-selected={index === highlightedSuggestionIndex}
                  className={`permission-tag-suggestion permission-tag-suggestion-${tag.kind} ${
                    index === highlightedSuggestionIndex
                      ? "permission-tag-suggestion-active"
                      : ""
                  }`}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setHighlightedSuggestionIndex(index)}
                  onClick={() => addTag(tag)}
                >
                  <span>{tag.label}</span>
                  <small>{tag.kind}</small>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function DefaultPermissionTagControls({
  label,
  defaultTags,
  activeTags,
  disabled = false,
  onChange
}: {
  label?: string;
  defaultTags: PermissionTag[];
  activeTags: PermissionTag[];
  disabled?: boolean;
  onChange: (tags: PermissionTag[]) => void;
}) {
  const [armedTagId, setArmedTagId] = useState<string | null>(null);
  const activeTagIds = new Set(activeTags.map((tag) => tag.id));
  const visibleDefaultTags = mergePermissionTags(defaultTags, activeTags);

  useEffect(() => {
    if (!armedTagId) {
      return;
    }

    const activeArmedTagId = armedTagId;
    function disarmOnOutsideClick(event: MouseEvent | TouchEvent) {
      if (!isInsidePermissionTag(event.target, activeArmedTagId)) {
        setArmedTagId(null);
      }
    }

    document.addEventListener("mousedown", disarmOnOutsideClick);
    document.addEventListener("touchstart", disarmOnOutsideClick);
    return () => {
      document.removeEventListener("mousedown", disarmOnOutsideClick);
      document.removeEventListener("touchstart", disarmOnOutsideClick);
    };
  }, [armedTagId]);

  function toggleTag(tag: PermissionTag) {
    if (activeTagIds.has(tag.id)) {
      onChange(activeTags.filter((activeTag) => activeTag.id !== tag.id));
      return;
    }

    onChange([...activeTags, tag]);
  }

  return (
    <div className="permission-tag-editor">
      {label && <span>{label}</span>}
      <div className="permission-tag-row">
        <div className="permission-tags permission-tags-defaults">
          {visibleDefaultTags.length === 0 ? (
            <span className="permission-tag-empty">No defaults</span>
          ) : (
            visibleDefaultTags.map((tag) => {
              const isActive = activeTagIds.has(tag.id);
              return (
                <PermissionTagChip
                  key={tag.id}
                  tag={tag}
                  disabled={disabled}
                  isArmed={armedTagId === tag.id}
                  isRemoved={!isActive}
                  sourceLabel="default"
                  actionIcon={isActive ? "remove" : "restore"}
                  actionTitle={
                    isActive ? "Remove this default from this model" : "Restore this default"
                  }
                  onArm={() => setArmedTagId(tag.id)}
                  onAction={() => toggleTag(tag)}
                />
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

function PermissionTagChip({
  tag,
  disabled = false,
  isArmed = false,
  isRemoved = false,
  sourceLabel,
  actionIcon = "remove",
  actionTitle = "Remove tag",
  onArm,
  onAction
}: {
  tag: PermissionTag;
  disabled?: boolean;
  isArmed?: boolean;
  isRemoved?: boolean;
  sourceLabel?: string;
  actionIcon?: "remove" | "restore";
  actionTitle?: string;
  onArm: () => void;
  onAction: () => void;
}) {
  return (
    <span
      data-permission-tag-id={tag.id}
      className={`permission-tag permission-tag-${tag.kind} ${
        isRemoved ? "permission-tag-default-removed" : ""
      } ${isArmed ? "permission-tag-armed" : ""}`}
    >
      <button
        type="button"
        className="permission-tag-label"
        disabled={disabled}
        onClick={onArm}
      >
        <span>{tag.label}</span>
        {sourceLabel && <span className="permission-tag-source">{sourceLabel}</span>}
      </button>
      <button
        type="button"
        className="permission-tag-action"
        disabled={disabled}
        title={actionTitle}
        aria-label={actionTitle}
        onClick={onAction}
      >
        {actionIcon === "restore" ? <Plus /> : <X />}
      </button>
    </span>
  );
}
