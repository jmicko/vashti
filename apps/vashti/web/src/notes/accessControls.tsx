import { Search } from "lucide-react";
import { useState } from "react";
import type { PrivatePersona } from "../privateChatStore";
import type { BackendModelGroup, Persona } from "../types";
import type { NoteAiAccess, NoteModelScope } from "./types";

export type NoteModelOption = {
  key: string;
  label: string;
  group: string;
};

export function buildNoteModelOptions(
  storageMode: "server" | "device",
  modelGroups: BackendModelGroup[],
  personas: Persona[],
  privatePersonas: PrivatePersona[]
): NoteModelOption[] {
  const baseModels = modelGroups.flatMap((group) =>
    group.models.map((model) => ({
      key: `base:${group.backend.id}:${model.name}`,
      label: model.name,
      group: group.backend.name
    }))
  );
  const customModels = (storageMode === "device" ? privatePersonas : personas).map((persona) => ({
    key: `persona:${persona.id}`,
    label: persona.current_version.display_name,
    group: "Custom models"
  }));
  return [...customModels, ...baseModels];
}

export function NoteAccessLevelControl({
  value,
  ariaLabel,
  onChange
}: {
  value: NoteAiAccess;
  ariaLabel: string;
  onChange: (value: NoteAiAccess) => void;
}) {
  const levels: Array<{ value: NoteAiAccess; label: string }> = [
    { value: "none", label: "No access" },
    { value: "read", label: "Read only" },
    { value: "edit", label: "Read and edit" },
    { value: "manage", label: "Full access" }
  ];
  return (
    <div className="segmented-control notes-access-control" role="group" aria-label={ariaLabel}>
      {levels.map((level) => (
        <button
          type="button"
          key={level.value}
          className={value === level.value ? "active" : ""}
          aria-pressed={value === level.value}
          onClick={() => onChange(level.value)}
        >
          {level.label}
        </button>
      ))}
    </div>
  );
}

export function NoteModelScopeEditor({
  scope,
  options,
  onChange,
  title,
  description,
  emptyWarning
}: {
  scope: NoteModelScope;
  options: NoteModelOption[];
  onChange: (scope: NoteModelScope) => void;
  title: string;
  description: string;
  emptyWarning: string;
}) {
  const [query, setQuery] = useState("");
  const filtered = options.filter((option) =>
    `${option.label} ${option.group}`.toLowerCase().includes(query.toLowerCase())
  );
  const grouped = filtered.reduce<Map<string, NoteModelOption[]>>((current, option) => {
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
      <p>{description}</p>
      <NotePermissionToggle
        label="Every model"
        checked={scope.all_models}
        onChange={(all_models) => onChange({ all_models, model_keys: scope.model_keys })}
      />
      {!scope.all_models && (
        <>
          {scope.model_keys.length === 0 && (
            <p className="notes-scope-warning" role="status">{emptyWarning}</p>
          )}
          <p className="notes-model-scope-helper">
            Custom models appear separately, even when they use the same base model.
          </p>
          <label className="notes-scope-search">
            <Search />
            <span className="visually-hidden">Search models</span>
            <input
              value={query}
              placeholder="Search models"
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                }
              }}
            />
          </label>
          <div className="notes-model-options">
            {[...grouped.entries()].map(([group, groupOptions]) => (
              <section key={group}>
                <h4>{group}</h4>
                {groupOptions.map((option) => (
                  <NotePermissionToggle
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

export function NotePermissionToggle({
  label,
  description,
  checked,
  onChange
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="notes-permission-toggle">
      <span>
        <strong>{label}</strong>
        {description && <small>{description}</small>}
      </span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
    </label>
  );
}

export function normalizeNoteModelScope(scope: NoteModelScope): NoteModelScope {
  return {
    all_models: scope.all_models,
    model_keys: [...new Set(scope.model_keys)].sort()
  };
}
