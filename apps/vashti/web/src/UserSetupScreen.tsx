import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BrainCircuit,
  Check,
  LogOut,
  MessageSquareText,
  NotebookPen,
  RotateCcw
} from "lucide-react";
import { requestJson } from "./api";
import { BrandMark, RetroLoader } from "./common";
import { ToggleSwitch } from "./settingsControls";
import type { SetupChoiceGroup, SetupStatusResponse, User } from "./types";

const NOTES_READ = "notes.allow_model_read";
const NOTES_EDIT = "notes.allow_model_edit";
const NOTES_TRASH = "notes.allow_model_trash";
const MEMORIES_READ = "memories.allow_model_read";
const MEMORIES_EDIT = "memories.allow_model_edit";
const MEMORIES_FORGET = "memories.allow_model_forget";

export function UserSetupScreen({
  user,
  onComplete,
  onSignOut
}: {
  user: User;
  onComplete: () => Promise<void>;
  onSignOut: () => Promise<void>;
}) {
  const [status, setStatus] = useState<SetupStatusResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, boolean>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isSigningOut, setIsSigningOut] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSetup = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await requestJson<SetupStatusResponse>("/api/user-setup");
      if (response.pending_count === 0) {
        await onComplete();
        return;
      }
      setStatus(response);
      setDraft(recommendedDraft(response.groups));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load your settings");
    } finally {
      setIsLoading(false);
    }
  }, [onComplete]);

  useEffect(() => {
    void loadSetup();
  }, [loadSetup]);

  const hasCustomChoices = useMemo(
    () =>
      Boolean(
        status?.groups.some((group) =>
          group.choices.some((choice) => draft[choice.key] !== choice.recommended_value)
        )
      ),
    [draft, status]
  );

  function updateChoice(key: string, value: boolean) {
    setError(null);
    setDraft((current) => normalizeDependencies({ ...current, [key]: value }, key, value));
  }

  function resetRecommended() {
    if (!status) return;
    setDraft(recommendedDraft(status.groups));
    setError(null);
  }

  async function saveChoices() {
    if (!status || isSaving) return;
    setIsSaving(true);
    setError(null);
    try {
      const choices = status.groups.flatMap((group) =>
        group.choices.map((choice) => ({
          key: choice.key,
          value: draft[choice.key] ?? choice.recommended_value
        }))
      );
      const response = await requestJson<SetupStatusResponse>("/api/user-setup", {
        method: "PATCH",
        body: JSON.stringify({ choices })
      });
      if (response.pending_count > 0) {
        setStatus(response);
        setDraft(recommendedDraft(response.groups));
        return;
      }
      await onComplete();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save your settings");
    } finally {
      setIsSaving(false);
    }
  }

  async function signOut() {
    if (isSigningOut) return;
    setIsSigningOut(true);
    setError(null);
    try {
      await onSignOut();
    } catch (signOutError) {
      setError(signOutError instanceof Error ? signOutError.message : "Failed to sign out");
      setIsSigningOut(false);
    }
  }

  return (
    <main className="user-setup-page">
      <header className="user-setup-topbar">
        <BrandMark compact />
        <div className="user-setup-account">
          <span>{user.display_name?.trim() || user.username}</span>
          <button
            type="button"
            className="secondary-button"
            aria-label="Sign Out"
            title="Sign Out"
            disabled={isSigningOut || isSaving}
            onClick={() => void signOut()}
          >
            <LogOut aria-hidden="true" />
            <span>{isSigningOut ? "Signing Out..." : "Sign Out"}</span>
          </button>
        </div>
      </header>

      <div className="user-setup-scroll">
        <section className="user-setup-content" aria-labelledby="user-setup-title">
          <div className="user-setup-intro">
            <p className="eyebrow">Your Settings</p>
            <h1 id="user-setup-title">Choose how models can use your data</h1>
            <p>
              Recommended choices are selected. Review them now; every setting remains available
              to change later.
            </p>
          </div>

          {error && (
            <div className="user-setup-error" role="alert">
              <p>{error}</p>
              <button
                type="button"
                className="secondary-button"
                disabled={isLoading || isSaving}
                onClick={() => void loadSetup()}
              >
                {status ? "Reload Choices" : "Retry"}
              </button>
            </div>
          )}

          {isLoading && !status ? (
            <div className="user-setup-loading">
              <RetroLoader />
              <span className="visually-hidden">Loading settings</span>
            </div>
          ) : (
            <div className="user-setup-groups">
              {status?.groups.map((group) => (
                <SetupGroup
                  key={group.id}
                  group={group}
                  draft={draft}
                  disabled={isLoading || isSaving || isSigningOut}
                  onChange={updateChoice}
                />
              ))}
            </div>
          )}
        </section>
      </div>

      {status && (
        <footer className="user-setup-actions">
          <div>
            <strong>
              {status.pending_count} {status.pending_count === 1 ? "choice" : "choices"}
            </strong>
            <span>New choices will appear here after future updates.</span>
          </div>
          <div className="user-setup-action-buttons">
            <button
              type="button"
              className="secondary-button"
              disabled={!hasCustomChoices || isLoading || isSaving || isSigningOut}
              onClick={resetRecommended}
            >
              <RotateCcw aria-hidden="true" />
              <span>Recommended</span>
            </button>
            <button
              type="button"
              disabled={isLoading || isSaving || isSigningOut}
              onClick={() => void saveChoices()}
            >
              <Check aria-hidden="true" />
              <span>{isSaving ? "Saving..." : "Save & Continue"}</span>
            </button>
          </div>
        </footer>
      )}
    </main>
  );
}

function SetupGroup({
  group,
  draft,
  disabled,
  onChange
}: {
  group: SetupChoiceGroup;
  draft: Record<string, boolean>;
  disabled: boolean;
  onChange: (key: string, value: boolean) => void;
}) {
  const GroupIcon = groupIcon(group.id);
  return (
    <section className={`user-setup-group user-setup-group-${group.id}`}>
      <div className="user-setup-group-heading">
        <GroupIcon aria-hidden="true" />
        <div>
          <h2>{group.title}</h2>
          <p>{group.description}</p>
        </div>
      </div>
      <div className="user-setup-choice-list">
        {group.choices.map((choice) => (
          <ToggleSwitch
            key={choice.key}
            compact
            label={choice.title}
            description={choice.description}
            checked={draft[choice.key] ?? choice.recommended_value}
            disabled={disabled}
            onChange={(value) => onChange(choice.key, value)}
          />
        ))}
      </div>
    </section>
  );
}

function groupIcon(groupId: string) {
  if (groupId === "notes") return NotebookPen;
  if (groupId === "memories") return BrainCircuit;
  return MessageSquareText;
}

function recommendedDraft(groups: SetupChoiceGroup[]) {
  return Object.fromEntries(
    groups.flatMap((group) =>
      group.choices.map((choice) => [choice.key, choice.recommended_value])
    )
  );
}

function normalizeDependencies(
  draft: Record<string, boolean>,
  changedKey: string,
  value: boolean
) {
  if (changedKey === NOTES_READ && !value) {
    if (NOTES_EDIT in draft) draft[NOTES_EDIT] = false;
    if (NOTES_TRASH in draft) draft[NOTES_TRASH] = false;
  }
  if ((changedKey === NOTES_EDIT || changedKey === NOTES_TRASH) && value && NOTES_READ in draft) {
    draft[NOTES_READ] = true;
  }
  if (changedKey === MEMORIES_READ && !value) {
    if (MEMORIES_EDIT in draft) draft[MEMORIES_EDIT] = false;
    if (MEMORIES_FORGET in draft) draft[MEMORIES_FORGET] = false;
  }
  if (
    (changedKey === MEMORIES_EDIT || changedKey === MEMORIES_FORGET) &&
    value &&
    MEMORIES_READ in draft
  ) {
    draft[MEMORIES_READ] = true;
  }
  return draft;
}
