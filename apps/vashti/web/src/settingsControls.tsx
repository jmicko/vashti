import { type ReactNode, useEffect } from "react";
import { RetroLoader } from "./common";

export function SettingsPlaceholder({
  eyebrow,
  title,
  text
}: {
  eyebrow: string;
  title: string;
  text: string;
}) {
  return (
    <div className="settings-section">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="status-message">{text}</p>
    </div>
  );
}

export function SettingsSaveBanner({
  isDirty,
  status,
  dirtyTitle,
  dirtyDescription,
  savedDescription,
  children
}: {
  isDirty: boolean;
  status: string | null;
  dirtyTitle: string;
  dirtyDescription: string;
  savedDescription: string;
  children: ReactNode;
}) {
  const visibleStatus = isDirty ? null : status;
  const isVisible = isDirty || Boolean(visibleStatus);

  return (
    <div
      className={isVisible ? "sticky-save-slot" : "sticky-save-slot sticky-save-slot-placeholder"}
    >
      <div
        className={
          visibleStatus
            ? "sticky-save-bar settings-content-wide sticky-save-bar-saved"
            : "sticky-save-bar settings-content-wide"
        }
        aria-hidden={!isVisible}
      >
        <div>
          <strong>{visibleStatus ?? dirtyTitle}</strong>
          <span>{visibleStatus ? savedDescription : dirtyDescription}</span>
        </div>
        {isDirty && <div className="sticky-save-actions">{children}</div>}
      </div>
    </div>
  );
}

export function ToolPromptEditor({
  label,
  value,
  defaultValue,
  isChanged,
  onChange
}: {
  label: string;
  value: string;
  defaultValue: string;
  isChanged: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <div className="tool-prompt-editor">
      <label className={isChanged ? "setting-field setting-field-changed" : "setting-field"}>
        <span>{label}</span>
        <textarea value={value} onChange={(event) => onChange(event.target.value)} />
      </label>
      <button
        type="button"
        className="secondary-button prompt-reset-button"
        disabled={value === defaultValue}
        onClick={() => onChange(defaultValue)}
      >
        Reset to Default
      </button>
    </div>
  );
}

export function ToggleSwitch({
  icon,
  label,
  description,
  checked,
  disabled = false,
  compact = false,
  isChanged = false,
  onChange
}: {
  icon?: ReactNode;
  label: string;
  description?: string;
  checked: boolean;
  disabled?: boolean;
  compact?: boolean;
  isChanged?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      className={[
        "toggle-row",
        compact ? "toggle-row-compact" : "",
        isChanged ? "toggle-row-changed" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <span className="toggle-copy">
        {icon && <span className="tool-icon">{icon}</span>}
        <span className="toggle-title">
          <span>{label}</span>
          {isChanged && <span className="changed-badge">Changed</span>}
        </span>
        {description && <small>{description}</small>}
      </span>
      <span className="switch-control">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="switch-track" aria-hidden="true" />
      </span>
    </label>
  );
}

export function UnsavedSettingsDialog({
  isSaving,
  onCancel,
  onDiscard,
  onSave
}: {
  isSaving: boolean;
  onCancel: () => void;
  onDiscard: () => void;
  onSave: () => void;
}) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !isSaving) {
        onCancel();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isSaving, onCancel]);

  return (
    <div className="confirm-backdrop" role="presentation">
      <section
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="unsaved-settings-title"
      >
        <h2 id="unsaved-settings-title">Unsaved Settings</h2>
        <p>Save your app settings before leaving, or discard the unsaved changes.</p>
        <div className="dialog-actions">
          <button type="button" className="secondary-button" disabled={isSaving} onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="danger-button" disabled={isSaving} onClick={onDiscard}>
            Discard
          </button>
          <button type="button" disabled={isSaving} onClick={onSave}>
            {isSaving ? <RetroLoader /> : "Save"}
          </button>
        </div>
      </section>
    </div>
  );
}
