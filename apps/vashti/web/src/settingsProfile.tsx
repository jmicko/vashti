import { FormEvent, useCallback, useEffect, useState } from "react";
import { Save, X } from "lucide-react";
import { requestJson } from "./api";
import { RetroLoader } from "./common";
import { SettingsSaveBanner } from "./settingsControls";
import {
  THEME_OPTIONS,
  applyTheme,
  normalizeTheme,
  storeAndApplyTheme,
  storedTheme,
  type ThemeId
} from "./theme";
import type { User, UserSettings } from "./types";

export function ProfileSettings({
  user,
  onUserChanged
}: {
  user: User;
  onUserChanged: (user: User) => void;
}) {
  const [displayName, setDisplayName] = useState(user.display_name ?? "");
  const [email, setEmail] = useState(user.email ?? "");
  const [theme, setTheme] = useState<ThemeId>(storedTheme());
  const [savedDisplayName, setSavedDisplayName] = useState(user.display_name ?? "");
  const [savedEmail, setSavedEmail] = useState(user.email ?? "");
  const [savedTheme, setSavedTheme] = useState<ThemeId>(storedTheme());
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadProfileSettings = useCallback(async () => {
    setError(null);

    try {
      const settings = await requestJson<UserSettings>("/api/user-settings");
      const nextTheme = normalizeTheme(settings.theme ?? storedTheme());
      setSavedDisplayName(user.display_name ?? "");
      setSavedEmail(user.email ?? "");
      setSavedTheme(nextTheme);
      setDisplayName(user.display_name ?? "");
      setEmail(user.email ?? "");
      setTheme(nextTheme);
      storeAndApplyTheme(nextTheme);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load profile settings");
    } finally {
      setHasLoaded(true);
    }
  }, [user.display_name, user.email]);

  useEffect(() => {
    void loadProfileSettings();
  }, [loadProfileSettings]);

  const isDirty =
    displayName.trim() !== savedDisplayName ||
    email.trim() !== savedEmail ||
    theme !== savedTheme;

  function updateTheme(nextTheme: ThemeId) {
    setTheme(nextTheme);
    applyTheme(nextTheme);
    setStatus(null);
  }

  function revertProfileSettings() {
    setDisplayName(savedDisplayName);
    setEmail(savedEmail);
    setTheme(savedTheme);
    applyTheme(savedTheme);
    setStatus(null);
    setError(null);
  }

  async function saveProfileSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setError(null);
    setStatus(null);

    try {
      if (displayName.trim() !== savedDisplayName || email.trim() !== savedEmail) {
        const response = await requestJson<{ user: User }>("/api/auth/profile", {
          method: "PATCH",
          body: JSON.stringify({
            display_name: displayName.trim() === "" ? null : displayName,
            email: email.trim() === "" ? null : email
          })
        });
        onUserChanged(response.user);
      }

      if (theme !== savedTheme) {
        await requestJson<UserSettings>("/api/user-settings", {
          method: "PATCH",
          body: JSON.stringify({ theme })
        });
      }

      const nextDisplayName = displayName.trim();
      const nextEmail = email.trim();
      setSavedDisplayName(nextDisplayName);
      setSavedEmail(nextEmail);
      setSavedTheme(theme);
      setDisplayName(nextDisplayName);
      setEmail(nextEmail);
      storeAndApplyTheme(theme);
      setStatus("Profile settings saved.");
      window.setTimeout(() => setStatus(null), 3000);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save profile");
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="settings-section profile-settings-section">
      <div className="section-header">
        <div>
          <p className="eyebrow">Account</p>
          <h1>Profile</h1>
        </div>
      </div>

      {!hasLoaded && <p className="status-message">Loading profile...</p>}
      {error && <p className="error">{error}</p>}

      <form className="settings-form profile-settings-form" onSubmit={saveProfileSettings}>
        <SettingsSaveBanner
          isDirty={isDirty}
          status={status}
          dirtyTitle="Unsaved profile changes"
          dirtyDescription="Save or revert your profile and theme changes."
          savedDescription="Your profile settings are up to date."
        >
          <button
            type="button"
            className="secondary-button"
            disabled={isSaving}
            onClick={revertProfileSettings}
          >
            <X />
            <span>Revert</span>
          </button>
          <button type="submit" disabled={isSaving}>
            <Save />
            <span>{isSaving ? <RetroLoader /> : "Save"}</span>
          </button>
        </SettingsSaveBanner>

        <section className="settings-subsection">
          <div>
            <p className="eyebrow">Profile Info</p>
            <h2>{user.username}</h2>
          </div>
          <label>
            <span>Name</span>
            <input
              autoComplete="name"
              value={displayName}
              placeholder={user.username}
              onChange={(event) => {
                setDisplayName(event.target.value);
                setStatus(null);
              }}
            />
          </label>
          <label>
            <span>Email</span>
            <input
              autoComplete="email"
              type="email"
              value={email}
              onChange={(event) => {
                setEmail(event.target.value);
                setStatus(null);
              }}
            />
          </label>
          <p className="status-message">
            Your username is used for login and user tags. The name field is for display only.
          </p>
        </section>

        <section className="settings-subsection">
          <div>
            <p className="eyebrow">Theme</p>
            <h2>Color Scheme</h2>
          </div>
          <div className="theme-options" role="radiogroup" aria-label="Color scheme">
            {THEME_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                className={
                  theme === option.id
                    ? "theme-option theme-option-active"
                    : "theme-option"
                }
                role="radio"
                aria-checked={theme === option.id}
                onClick={() => updateTheme(option.id)}
              >
                <span className={`theme-swatch theme-swatch-${option.id}`} aria-hidden="true" />
                <span>
                  <strong>{option.label}</strong>
                  <small>{option.description}</small>
                </span>
              </button>
            ))}
          </div>
        </section>
      </form>
    </div>
  );
}
