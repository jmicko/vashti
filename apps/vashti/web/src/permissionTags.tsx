import { useEffect, useId, useRef, useState } from "react";
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
  const [armedTagId, setArmedTagId] = useState<string | null>(null);
  const editorRef = useRef<HTMLDivElement | null>(null);
  const datalistId = useId();
  const suggestions = suggestionsKind
    ? availableTags.filter((tag) => tag.kind === suggestionsKind)
    : availableTags;

  useEffect(() => {
    if (!armedTagId) {
      return;
    }

    function disarmOnOutsideClick(event: MouseEvent | TouchEvent) {
      const target = event.target;
      if (target instanceof Node && editorRef.current?.contains(target)) {
        return;
      }

      setArmedTagId(null);
    }

    document.addEventListener("mousedown", disarmOnOutsideClick);
    document.addEventListener("touchstart", disarmOnOutsideClick);
    return () => {
      document.removeEventListener("mousedown", disarmOnOutsideClick);
      document.removeEventListener("touchstart", disarmOnOutsideClick);
    };
  }, [armedTagId]);

  function addTag() {
    const tag = permissionTagFromInput(value, suggestions);
    if (!tag || tags.some((existing) => existing.id === tag.id)) {
      setValue("");
      return;
    }
    onChange([...tags, tag]);
    setValue("");
  }

  return (
    <div className="permission-tag-editor" ref={editorRef}>
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
        <div className="permission-tag-add">
          <input
            value={value}
            disabled={disabled}
            list={datalistId}
            placeholder="tag"
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addTag();
              }
            }}
          />
          <datalist id={datalistId}>
            {suggestions.map((tag) => (
              <option key={tag.id} value={tag.label} />
            ))}
          </datalist>
          <button type="button" className="secondary-button" disabled={disabled} onClick={addTag}>
            <Plus />
          </button>
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
  const editorRef = useRef<HTMLDivElement | null>(null);
  const activeTagIds = new Set(activeTags.map((tag) => tag.id));
  const visibleDefaultTags = mergePermissionTags(defaultTags, activeTags);

  useEffect(() => {
    if (!armedTagId) {
      return;
    }

    function disarmOnOutsideClick(event: MouseEvent | TouchEvent) {
      const target = event.target;
      if (target instanceof Node && editorRef.current?.contains(target)) {
        return;
      }

      setArmedTagId(null);
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
    <div className="permission-tag-editor" ref={editorRef}>
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
