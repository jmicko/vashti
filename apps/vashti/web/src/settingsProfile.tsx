import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { CheckCircle2, Download, Save, X } from "lucide-react";
import { requestJson } from "./api";
import { RetroLoader } from "./common";
import { usePwa } from "./pwa";
import { SettingsPanel, SettingsSaveBanner } from "./settingsControls";
import {
  THEME_OPTIONS,
  applyTheme,
  normalizeTheme,
  storeAndApplyTheme,
  storedTheme,
  type ThemeId,
} from "./theme";
import type {
  SettingsGuard,
  User,
  UserSettings,
  VersionResponse,
} from "./types";
import {
  NativeConnectionsSettings,
  useNativeConnections,
} from "./nativeConnections";
import { NativeUpdateSettings } from "./nativeUpdates";
import { ImportPanel } from "./imports/ImportPanel";
import type { BackendModelGroup } from "./types";

export function ProfileSettings({
  user,
  onUserChanged,
  onGuardChange,
  view = "profile",
  modelGroups,
  onOpenImportedDocument,
}: {
  user: User;
  onUserChanged: (user: User) => void;
  onGuardChange: (guard: SettingsGuard | null) => void;
  view?: "profile" | "appearance" | "device";
  modelGroups: BackendModelGroup[];
  onOpenImportedDocument?: (chatId:string,mode:"server"|"device")=>void;
}) {
  const { installState, installError, isInstalling, install } = usePwa();
  const { isNative } = useNativeConnections();
  const [displayName, setDisplayName] = useState(user.display_name ?? "");
  const [email, setEmail] = useState(user.email ?? "");
  const [theme, setTheme] = useState<ThemeId>(storedTheme());
  const [savedDisplayName, setSavedDisplayName] = useState(
    user.display_name ?? "",
  );
  const [savedEmail, setSavedEmail] = useState(user.email ?? "");
  const [savedTheme, setSavedTheme] = useState<ThemeId>(storedTheme());
  const [hasLoaded, setHasLoaded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverVersion, setServerVersion] = useState<string | null>(null);
  const [importBusy, setImportBusy] = useState(false);
  const cancelImportRef = useRef<() => void>(() => {});
  const importBusyChanged = useCallback((busy: boolean, cancel: () => void) => {
    setImportBusy(busy); cancelImportRef.current = cancel;
  }, []);
  const themeRef = useRef(theme);
  const savedThemeRef = useRef(savedTheme);
  const statusTimerRef = useRef<number | null>(null);
  const loadRequestRef = useRef(0);

  function clearStatusTimer() {
    if (statusTimerRef.current !== null) {
      window.clearTimeout(statusTimerRef.current);
      statusTimerRef.current = null;
    }
  }

  const loadProfileSettings = useCallback(async () => {
    const request = ++loadRequestRef.current;
    setError(null);

    try {
      const settings = await requestJson<UserSettings>("/api/user-settings");
      if (request !== loadRequestRef.current) return;
      const nextTheme = normalizeTheme(settings.theme ?? storedTheme());
      themeRef.current = nextTheme;
      savedThemeRef.current = nextTheme;
      setSavedDisplayName(user.display_name ?? "");
      setSavedEmail(user.email ?? "");
      setSavedTheme(nextTheme);
      setDisplayName(user.display_name ?? "");
      setEmail(user.email ?? "");
      setTheme(nextTheme);
      storeAndApplyTheme(nextTheme);
    } catch (loadError) {
      if (request === loadRequestRef.current)
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load profile settings",
        );
    } finally {
      if (request === loadRequestRef.current) setHasLoaded(true);
    }
  }, [user.display_name, user.email]);

  useEffect(() => {
    void loadProfileSettings();
    return () => {
      loadRequestRef.current++;
    };
  }, [loadProfileSettings]);

  useEffect(() => {
    if (isNative) {
      return;
    }

    void requestJson<VersionResponse>("/api/version")
      .then((response) => setServerVersion(response.version))
      .catch(() => setServerVersion(null));
  }, [isNative]);

  useEffect(
    () => () => {
      clearStatusTimer();
      if (themeRef.current !== savedThemeRef.current) {
        applyTheme(savedThemeRef.current);
      }
    },
    [],
  );

  const isDirty =
    displayName.trim() !== savedDisplayName ||
    email.trim() !== savedEmail ||
    theme !== savedTheme;

  function updateTheme(nextTheme: ThemeId) {
    clearStatusTimer();
    themeRef.current = nextTheme;
    setTheme(nextTheme);
    applyTheme(nextTheme);
    setStatus(null);
  }

  function revertProfileSettings() {
    clearStatusTimer();
    themeRef.current = savedTheme;
    setDisplayName(savedDisplayName);
    setEmail(savedEmail);
    setTheme(savedTheme);
    applyTheme(savedTheme);
    setStatus(null);
    setError(null);
  }

  async function saveProfileSettings() {
    if (isSaving || !hasLoaded) return false;
    setIsSaving(true);
    setError(null);
    setStatus(null);
    clearStatusTimer();

    try {
      if (
        displayName.trim() !== savedDisplayName ||
        email.trim() !== savedEmail
      ) {
        const response = await requestJson<{ user: User }>(
          "/api/auth/profile",
          {
            method: "PATCH",
            body: JSON.stringify({
              display_name: displayName.trim() === "" ? null : displayName,
              email: email.trim() === "" ? null : email,
            }),
          },
        );
        onUserChanged(response.user);
      }

      if (theme !== savedTheme) {
        await requestJson<UserSettings>("/api/user-settings", {
          method: "PATCH",
          body: JSON.stringify({ theme }),
        });
      }

      const nextDisplayName = displayName.trim();
      const nextEmail = email.trim();
      themeRef.current = theme;
      savedThemeRef.current = theme;
      setSavedDisplayName(nextDisplayName);
      setSavedEmail(nextEmail);
      setSavedTheme(theme);
      setDisplayName(nextDisplayName);
      setEmail(nextEmail);
      storeAndApplyTheme(theme);
      setStatus("Profile settings saved.");
      statusTimerRef.current = window.setTimeout(() => {
        setStatus(null);
        statusTimerRef.current = null;
      }, 3000);
      return true;
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "Failed to save profile",
      );
      return false;
    } finally {
      setIsSaving(false);
    }
  }

  const actions = useRef({
    save: saveProfileSettings,
    discard: revertProfileSettings,
  });
  actions.current = {
    save: saveProfileSettings,
    discard: revertProfileSettings,
  };
  useEffect(() => {
    onGuardChange({
      isDirty: isDirty || importBusy,
      pendingJob: importBusy ? "import" : undefined,
      save: () => importBusy ? Promise.resolve(false) : actions.current.save(),
      discard: () => { cancelImportRef.current(); actions.current.discard(); },
    });
  }, [isDirty, importBusy, onGuardChange]);
  useEffect(() => () => onGuardChange(null), [onGuardChange]);

  return (
    <SettingsPanel
      eyebrow="Personal"
      title={
        view === "appearance"
          ? "Appearance"
          : view === "device"
            ? "This device"
            : "Profile"
      }
      className="profile-settings-section"
    >
      {!hasLoaded && <p className="status-message">Loading profile...</p>}
      {error && <p className="error">{error}</p>}

      <form
        className="settings-form settings-form-with-banner profile-settings-form"
        onSubmit={(event: FormEvent<HTMLFormElement>) => {
          event.preventDefault();
          void saveProfileSettings();
        }}
      >
        {view !== "device" && (
          <SettingsSaveBanner
            isDirty={isDirty}
            status={status}
            dirtyTitle="Unsaved changes"
            dirtyDescription="Save or revert your changes."
            savedDescription="Settings are up to date."
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
        )}

        {view === "profile" && (
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
              Your username is used for login and user tags. The name field is
              for display only.
            </p>
          </section>
        )}

        {view === "appearance" && (
          <section className="settings-subsection">
            <div>
              <p className="eyebrow">Theme</p>
              <h2>Color Scheme</h2>
            </div>
            <div
              className="theme-options"
              role="radiogroup"
              aria-label="Color scheme"
            >
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
                  disabled={!hasLoaded || isSaving}
                  onClick={() => updateTheme(option.id)}
                >
                  <span
                    className={`theme-swatch theme-swatch-${option.id}`}
                    aria-hidden="true"
                  />
                  <span>
                    <strong>{option.label}</strong>
                    <small>{option.description}</small>
                  </span>
                </button>
              ))}
            </div>
          </section>
        )}

        {view === "device" && (
          <>
            <NativeConnectionsSettings />
            <NativeUpdateSettings />

            {!isNative && (
              <section className="settings-subsection pwa-install-section">
                <div>
                  <p className="eyebrow">App</p>
                  <h2>Install Vashti</h2>
                </div>
                {installState === "installed" && (
                  <p className="pwa-install-status pwa-install-status-ready">
                    <CheckCircle2 aria-hidden="true" />
                    <span>Vashti is installed on this device.</span>
                  </p>
                )}
                {installState === "available" && (
                  <>
                    <p className="status-message">
                      Install Vashti for quick access on this device.
                    </p>
                    <button
                      type="button"
                      className="secondary-button pwa-install-button"
                      disabled={isInstalling}
                      onClick={() => void install()}
                    >
                      <Download aria-hidden="true" />
                      <span>
                        {isInstalling ? "Installing..." : "Install Vashti"}
                      </span>
                    </button>
                  </>
                )}
                {installState === "browser" && (
                  <p className="status-message">
                    Installation is available from your browser's app menu.
                  </p>
                )}
                {installState === "development" && (
                  <p className="status-message">
                    Installation is available from the built Vashti app.
                  </p>
                )}
                {installState === "insecure" && (
                  <p className="status-message">
                    App installation requires HTTPS or localhost.
                  </p>
                )}
                {installState === "unsupported" && (
                  <p className="status-message">
                    This browser does not support app installation.
                  </p>
                )}
                {installError && <p className="error">{installError}</p>}
                {serverVersion ? (
                  <a
                    className="button-link secondary-button pwa-install-button"
                    href={`https://vashti.chat/releases/v${serverVersion}/vashti-android.apk`}
                  >
                    <Download aria-hidden="true" />
                    <span>Download Android App v{serverVersion}</span>
                  </a>
                ) : (
                  <p className="status-message">
                    Android download unavailable because the server version
                    could not be determined.
                  </p>
                )}
              </section>
            )}
          </>
        )}
      </form>
      {view === "profile" && <ImportPanel groups={modelGroups} onBusyChange={importBusyChanged} onOpenImportedDocument={onOpenImportedDocument}/>}
    </SettingsPanel>
  );
}
