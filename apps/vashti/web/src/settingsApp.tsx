import {
  FormEvent,
  useCallback,
  useEffect,
  useState
} from "react";
import { Copy, Download, Lock, RefreshCw, Save } from "lucide-react";
import { requestJson } from "./api";
import { ConfirmDialog, RetroLoader } from "./common";
import { MarkdownContent } from "./MarkdownContent";
import { SettingsPanel, SettingsSaveBanner } from "./settingsControls";
import type {
  AppSettings,
  AppSettingsGuard,
  UpdateStatusResponse,
  VersionResponse
} from "./types";

export function AppSettingsPanel({
  onGuardChange,
  updateStatus,
  updateStatusError,
  onUpdateStatusChange,
  onRefreshUpdateStatus
}: {
  onGuardChange: (guard: AppSettingsGuard | null) => void;
  updateStatus: UpdateStatusResponse | null;
  updateStatusError: string | null;
  onUpdateStatusChange: (status: UpdateStatusResponse) => void;
  onRefreshUpdateStatus: () => Promise<UpdateStatusResponse | null>;
}) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [allowSignup, setAllowSignup] = useState(true);
  const [signupLimit, setSignupLimit] = useState(25);
  const [updateChannel, setUpdateChannel] =
    useState<AppSettings["update_channel"]>("stable");
  const [networkMode, setNetworkMode] = useState<AppSettings["network_mode"]>("lan_http");
  const [publicBaseUrl, setPublicBaseUrl] = useState("");
  const [trustProxyHeaders, setTrustProxyHeaders] = useState(false);
  const [networkPassword, setNetworkPassword] = useState("");
  const [networkAcknowledged, setNetworkAcknowledged] = useState(false);
  const [isNetworkUnlocked, setIsNetworkUnlocked] = useState(false);
  const [isSavingNetwork, setIsSavingNetwork] = useState(false);
  const [networkConfirmPending, setNetworkConfirmPending] = useState(false);
  const [proxyType, setProxyType] = useState<"caddy" | "nginx">("caddy");
  const [proxyDomain, setProxyDomain] = useState("");
  const [internalVashtiUrl, setInternalVashtiUrl] = useState(defaultInternalVashtiUrl);
  const [proxyCopied, setProxyCopied] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [appVersion, setAppVersion] = useState<VersionResponse | null>(null);
  const [isCheckingUpdate, setIsCheckingUpdate] = useState(false);
  const [isRequestingUpdate, setIsRequestingUpdate] = useState(false);
  const [updateConfirmPending, setUpdateConfirmPending] = useState(false);
  const [updateActionError, setUpdateActionError] = useState<string | null>(null);

  const loadSettings = useCallback(async () => {
    setIsLoading(true);
    setSaveStatus(null);
    setError(null);

    try {
      const [response, versionResponse] = await Promise.all([
        requestJson<AppSettings>("/api/settings"),
        requestJson<VersionResponse>("/api/version")
      ]);
      setSettings(response);
      setAllowSignup(response.allow_signup);
      setSignupLimit(response.signup_limit);
      setUpdateChannel(response.update_channel);
      setNetworkMode(response.network_mode);
      setPublicBaseUrl(response.public_base_url ?? "");
      setTrustProxyHeaders(response.trust_proxy_headers);
      setProxyDomain(domainFromPublicBaseUrl(response.public_base_url));
      setAppVersion(versionResponse);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load settings");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const isUpdateChannelDirty = Boolean(
    settings && settings.update_channel !== updateChannel
  );
  const isDirty = Boolean(
    settings &&
      (settings.allow_signup !== allowSignup ||
        settings.signup_limit !== signupLimit ||
        settings.update_channel !== updateChannel)
  );
  const updateOperationActive = Boolean(
    updateStatus && ["requested", "installing"].includes(updateStatus.operation.state)
  );
  const updateStatusMatchesChannel = updateStatus?.channel === updateChannel;

  const checkForUpdates = useCallback(async () => {
    setIsCheckingUpdate(true);
    setUpdateActionError(null);
    try {
      const response = await requestJson<UpdateStatusResponse>("/api/admin/update/check", {
        method: "POST"
      });
      onUpdateStatusChange(response);
      return response;
    } catch (checkError) {
      setUpdateActionError(
        checkError instanceof Error ? checkError.message : "Failed to check for updates"
      );
      return null;
    } finally {
      setIsCheckingUpdate(false);
    }
  }, [onUpdateStatusChange]);

  const saveSettingsDraft = useCallback(async () => {
    setIsSaving(true);
    setSaveStatus(null);
    setError(null);

    try {
      const channelChanged = settings?.update_channel !== updateChannel;
      const response = await requestJson<AppSettings>("/api/settings", {
        method: "PATCH",
        body: JSON.stringify({
          allow_signup: allowSignup,
          signup_limit: signupLimit,
          update_channel: updateChannel
        })
      });
      setSettings(response);
      setAllowSignup(response.allow_signup);
      setSignupLimit(response.signup_limit);
      setUpdateChannel(response.update_channel);
      setSaveStatus("App settings saved.");
      if (channelChanged) {
        void checkForUpdates();
      }
      return true;
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save settings");
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [allowSignup, checkForUpdates, settings?.update_channel, signupLimit, updateChannel]);

  const discardSettingsDraft = useCallback(() => {
    if (!settings) {
      return;
    }

    setAllowSignup(settings.allow_signup);
    setSignupLimit(settings.signup_limit);
    setUpdateChannel(settings.update_channel);
    setSaveStatus(null);
    setError(null);
  }, [settings]);

  useEffect(() => {
    onGuardChange({
      isDirty,
      save: saveSettingsDraft,
      discard: discardSettingsDraft
    });
  }, [discardSettingsDraft, isDirty, onGuardChange, saveSettingsDraft]);

  useEffect(() => () => onGuardChange(null), [onGuardChange]);

  useEffect(() => {
    if (!saveStatus) {
      return;
    }

    const timeout = window.setTimeout(() => setSaveStatus(null), 3000);
    return () => window.clearTimeout(timeout);
  }, [saveStatus]);

  useEffect(() => {
    if (isDirty && saveStatus) {
      setSaveStatus(null);
    }
  }, [isDirty, saveStatus]);

  async function saveSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await saveSettingsDraft();
  }

  const networkConfig = proxyConfig(proxyType, proxyDomain, internalVashtiUrl);

  function unlockNetworkSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!networkPassword.trim() || !networkAcknowledged) {
      setNetworkError("Enter your admin password and confirm the risk before unlocking.");
      return;
    }
    setNetworkError(null);
    setIsNetworkUnlocked(true);
  }

  async function saveNetworkSettings(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (networkMode === "public_https_proxy" && window.location.protocol !== "https:" && !networkConfirmPending) {
      setNetworkConfirmPending(true);
      return;
    }

    setIsSavingNetwork(true);
    setNetworkError(null);

    try {
      const response = await requestJson<AppSettings>("/api/settings/network", {
        method: "PATCH",
        body: JSON.stringify({
          network_mode: networkMode,
          public_base_url: publicBaseUrl.trim() || null,
          trust_proxy_headers: trustProxyHeaders,
          admin_password: networkPassword,
          acknowledge_risk: networkAcknowledged
        })
      });
      setSettings(response);
      setNetworkMode(response.network_mode);
      setPublicBaseUrl(response.public_base_url ?? "");
      setTrustProxyHeaders(response.trust_proxy_headers);
      setNetworkPassword("");
      setNetworkAcknowledged(false);
      setIsNetworkUnlocked(false);
      setNetworkConfirmPending(false);
    } catch (saveError) {
      setNetworkError(
        saveError instanceof Error ? saveError.message : "Failed to save network settings"
      );
    } finally {
      setIsSavingNetwork(false);
    }
  }

  async function copyProxyConfig() {
    await navigator.clipboard.writeText(networkConfig);
    setProxyCopied(true);
    window.setTimeout(() => setProxyCopied(false), 1400);
  }

  async function dismissNetworkRecoveryNotice() {
    setNetworkError(null);

    try {
      const response = await requestJson<AppSettings>(
        "/api/settings/network-recovery-notice/dismiss",
        { method: "POST" }
      );
      setSettings(response);
    } catch (dismissError) {
      setNetworkError(
        dismissError instanceof Error ? dismissError.message : "Failed to dismiss recovery notice"
      );
    }
  }

  async function requestUpdateInstall() {
    setIsRequestingUpdate(true);
    setUpdateActionError(null);
    try {
      const response = await requestJson<UpdateStatusResponse>("/api/admin/update/install", {
        method: "POST"
      });
      onUpdateStatusChange(response);
      setUpdateConfirmPending(false);
    } catch (installError) {
      setUpdateActionError(
        installError instanceof Error ? installError.message : "Failed to request the update"
      );
    } finally {
      setIsRequestingUpdate(false);
    }
  }

  return (
    <SettingsPanel
      eyebrow="Admin"
      title="App Settings"
      actions={
        <button
          type="button"
          className="secondary-button refresh-button"
          onClick={() => {
            void loadSettings();
            void onRefreshUpdateStatus();
          }}
          disabled={isLoading}
        >
          {isLoading ? <RetroLoader /> : "Refresh"}
        </button>
      }
    >

      {error && <p className="error">{error}</p>}
      {!settings && isLoading && <p className="status-message">Loading settings...</p>}
      {settings && (
        <>
          <SettingsSaveBanner
            isDirty={isDirty}
            status={saveStatus}
            dirtyTitle="Unsaved app changes"
            dirtyDescription="Save to apply these app settings."
            savedDescription="Saved changes are active."
          >
            <button
              type="button"
              className="secondary-button"
              disabled={isSaving}
              onClick={discardSettingsDraft}
            >
              Revert
            </button>
            <button type="button" disabled={isSaving} onClick={() => void saveSettingsDraft()}>
              <Save />
              <span>{isSaving ? "Saving..." : "Save"}</span>
            </button>
          </SettingsSaveBanner>

          <form className="settings-form" onSubmit={saveSettings}>
            <dl className="settings-meta">
              <div>
                <dt>Version</dt>
                <dd>{appVersion ? `v${appVersion.version}` : "Unknown"}</dd>
              </div>
            </dl>
            <label
              className={
                settings.allow_signup !== allowSignup
                  ? "checkbox-row setting-field setting-field-changed"
                  : "checkbox-row setting-field"
              }
            >
              <input
                type="checkbox"
                checked={allowSignup}
                onChange={(event) => setAllowSignup(event.target.checked)}
              />
              <span>Allow public account creation</span>
            </label>
            <label
              className={
                settings.signup_limit !== signupLimit
                  ? "setting-field setting-field-changed"
                  : "setting-field"
              }
            >
              <span>Signup limit</span>
              <input
                min="0"
                type="number"
                value={signupLimit}
                onChange={(event) => setSignupLimit(Number(event.target.value))}
              />
            </label>
            <p className="status-message">
              {settings.signup_count} public account signup
              {settings.signup_count === 1 ? "" : "s"} used.
            </p>
          </form>

          <section className="managed-update-panel">
            <div className="managed-update-heading">
              <div>
                <p className="eyebrow">Updates</p>
                <h2>Managed Updates</h2>
                <p className="status-message">
                  Vashti checks in the background, but installs only when an admin confirms one.
                </p>
              </div>
              <div className="managed-update-version">
                <span>Installed</span>
                <strong>
                  {updateStatus?.current_version ??
                    (appVersion ? `v${appVersion.version}` : "Unknown")}
                </strong>
              </div>
            </div>

            <div
              className={
                isUpdateChannelDirty
                  ? "setting-field setting-field-changed update-channel-field"
                  : "setting-field update-channel-field"
              }
            >
              <span>Release channel</span>
              <div className="segmented-control update-channel-control">
                <button
                  type="button"
                  className={updateChannel === "stable" ? "active" : undefined}
                  aria-pressed={updateChannel === "stable"}
                  onClick={() => setUpdateChannel("stable")}
                >
                  Stable
                </button>
                <button
                  type="button"
                  className={updateChannel === "prerelease" ? "active" : undefined}
                  aria-pressed={updateChannel === "prerelease"}
                  onClick={() => setUpdateChannel("prerelease")}
                >
                  Prerelease
                </button>
              </div>
              <small>
                {updateChannel === "stable"
                  ? "Notify only for promoted releases."
                  : "Include the current test release when it is newer than stable."}
              </small>
            </div>

            {updateStatusError && <p className="error">{updateStatusError}</p>}
            {updateActionError && <p className="error">{updateActionError}</p>}
            {updateStatus?.check_error && (
              <p className="error">Last update check failed: {updateStatus.check_error}</p>
            )}
            {isUpdateChannelDirty && (
              <p className="status-message">Save the release channel before checking.</p>
            )}
            {!isUpdateChannelDirty && updateStatus && !updateStatusMatchesChannel && (
              <p className="status-message">Refreshing the selected release channel...</p>
            )}

            {updateStatusMatchesChannel && updateStatus?.available ? (
              <div className="available-update" role="status">
                <div className="available-update-summary">
                  <div>
                    <strong>{updateStatus.available.version} is available</strong>
                    <span>
                      {updateStatus.available.release_status === "prerelease"
                        ? "Prerelease"
                        : "Stable release"}
                    </span>
                  </div>
                  {updateStatus.checked_at && (
                    <small>Checked {formatUpdateTimestamp(updateStatus.checked_at)}</small>
                  )}
                </div>
                {updateStatus.available.notes && (
                  <details className="update-release-notes">
                    <summary>Release notes</summary>
                    <MarkdownContent content={updateStatus.available.notes} />
                  </details>
                )}
              </div>
            ) : (
              updateStatusMatchesChannel &&
              updateStatus?.checked_at &&
              !updateStatus.check_error && (
                <p className="status-message">
                  This installation is up to date. Checked {formatUpdateTimestamp(updateStatus.checked_at)}.
                </p>
              )
            )}

            {updateStatus && !updateStatus.managed_updates && (
              <div className="info-box" role="status">
                <p className="eyebrow">One-time Setup Required</p>
                <p>
                  This installation predates managed updates. Run the current Vashti installer once
                  from a terminal to install the restricted system update service. Future updates
                  can then be installed here.
                </p>
              </div>
            )}
            {updateStatus?.target === null && (
              <p className="error">Managed updates are not available for this system target.</p>
            )}
            {updateStatus?.operation.message && updateStatus.operation.state !== "idle" && (
              <p
                className={
                  ["failed", "rolled_back"].includes(updateStatus.operation.state)
                    ? "error"
                    : "status-message"
                }
              >
                {updateStatus.operation.message}
              </p>
            )}

            <div className="managed-update-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={
                  isCheckingUpdate ||
                  isRequestingUpdate ||
                  isUpdateChannelDirty ||
                  updateOperationActive
                }
                onClick={() => void checkForUpdates()}
              >
                {isCheckingUpdate ? <RetroLoader /> : <RefreshCw />}
                <span>{isCheckingUpdate ? "Checking..." : "Check Now"}</span>
              </button>
              {updateStatusMatchesChannel && updateStatus?.available && (
                <button
                  type="button"
                  disabled={
                    !updateStatus.managed_updates ||
                    updateStatus.target === null ||
                    isRequestingUpdate ||
                    updateOperationActive
                  }
                  onClick={() => setUpdateConfirmPending(true)}
                >
                  <Download />
                  <span>{updateOperationActive ? "Updating..." : "Install Update"}</span>
                </button>
              )}
            </div>
          </section>
        </>
      )}
      {settings && (
        <section className="network-settings-panel">
          <div>
            <p className="eyebrow">Advanced</p>
            <h2>Network Access</h2>
            <p className="status-message">
              LAN mode keeps local HTTP login working. Public HTTPS Reverse Proxy mode is for
              deployments behind nginx, Caddy, or a tunnel on ports 80/443.
            </p>
          </div>

          {settings.network_recovery_notice && (
            <div className="info-box" role="status">
              <p className="eyebrow">Recovery</p>
              <p>{settings.network_recovery_notice}</p>
              <button
                type="button"
                className="secondary-button"
                onClick={() => void dismissNetworkRecoveryNotice()}
              >
                Dismiss
              </button>
            </div>
          )}

          <div
            className={
              isNetworkUnlocked
                ? "network-settings-box"
                : "network-settings-box network-settings-box-locked"
            }
          >
            {!isNetworkUnlocked && (
              <form className="settings-form network-unlock-form" onSubmit={unlockNetworkSettings}>
                <p className="error">
                  These settings can break login or make Vashti unreachable if configured
                  incorrectly.
                </p>
                {networkError && <p className="error">{networkError}</p>}
                <label>
                  <span>Admin password</span>
                  <input
                    autoComplete="current-password"
                    type="password"
                    value={networkPassword}
                    onChange={(event) => setNetworkPassword(event.target.value)}
                  />
                </label>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={networkAcknowledged}
                    onChange={(event) => setNetworkAcknowledged(event.target.checked)}
                  />
                  <span>I understand these settings may lock me out if configured incorrectly.</span>
                </label>
                <button type="submit">
                  <Lock />
                  <span>Unlock Network Settings</span>
                </button>
              </form>
            )}

            <form
              className={
                isNetworkUnlocked
                  ? "settings-form network-settings-form"
                  : "settings-form network-settings-form network-settings-form-locked"
              }
              onSubmit={saveNetworkSettings}
            >
              {networkError && <p className="error">{networkError}</p>}
              {!isNetworkUnlocked && (
                <p className="status-message">
                  Enter your admin password below to unlock editing. Current settings are shown here
                  for review.
                </p>
              )}
              {isNetworkUnlocked &&
                networkMode === "public_https_proxy" &&
                window.location.protocol !== "https:" && (
                  <div className="info-box" role="status">
                    <p className="eyebrow">HTTP Detected</p>
                    <p>
                      You are currently using plain HTTP. Saving Public HTTPS Reverse Proxy mode may
                      require you to access Vashti through the HTTPS proxy before login works again.
                    </p>
                  </div>
                )}
              <label>
                <span>Network Access Mode</span>
                <select
                  value={networkMode}
                  disabled={!isNetworkUnlocked}
                  onChange={(event) =>
                    setNetworkMode(event.target.value as AppSettings["network_mode"])
                  }
                >
                  <option value="lan_http">LAN / Local HTTP</option>
                  <option value="public_https_proxy">Public HTTPS Reverse Proxy</option>
                </select>
              </label>
              <label>
                <span>Public base URL</span>
                <input
                  value={publicBaseUrl}
                  disabled={!isNetworkUnlocked}
                  onChange={(event) => {
                    setPublicBaseUrl(event.target.value);
                    setProxyDomain(domainFromPublicBaseUrl(event.target.value));
                  }}
                  placeholder="https://chat.example.com"
                />
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={trustProxyHeaders}
                  disabled={!isNetworkUnlocked || networkMode !== "public_https_proxy"}
                  onChange={(event) => setTrustProxyHeaders(event.target.checked)}
                />
                <span>Trust reverse-proxy headers from nginx, Caddy, or tunnel</span>
              </label>

              {isNetworkUnlocked && (
                <div className="network-actions">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => {
                      setIsNetworkUnlocked(false);
                      setNetworkPassword("");
                      setNetworkAcknowledged(false);
                    }}
                  >
                    Cancel
                  </button>
                  <button type="submit" disabled={isSavingNetwork}>
                    <Save />
                    <span>{isSavingNetwork ? "Saving..." : "Save Network Mode"}</span>
                  </button>
                </div>
              )}
            </form>
          </div>

          <div className="proxy-config-panel">
            <div className="backend-edit-grid">
              <label>
                <span>Proxy type</span>
                <select value={proxyType} onChange={(event) => setProxyType(event.target.value as "caddy" | "nginx")}>
                  <option value="caddy">Caddy</option>
                  <option value="nginx">nginx</option>
                </select>
              </label>
              <label>
                <span>Public domain</span>
                <input
                  value={proxyDomain}
                  onChange={(event) => setProxyDomain(event.target.value)}
                  placeholder="chat.example.com"
                />
              </label>
              <label>
                <span>Internal Vashti URL</span>
                <input
                  value={internalVashtiUrl}
                  onChange={(event) => setInternalVashtiUrl(event.target.value)}
                  placeholder="http://192.168.1.55:7771"
                />
              </label>
            </div>
            <pre className="proxy-config-output">{networkConfig}</pre>
            <button type="button" className="secondary-button" onClick={() => void copyProxyConfig()}>
              <Copy />
              <span>{proxyCopied ? "Copied" : "Copy Config"}</span>
            </button>
          </div>
        </section>
      )}
      {networkConfirmPending && (
        <ConfirmDialog
          title="Enable Public HTTPS Mode?"
          message="You are not currently using HTTPS. If your reverse proxy is not ready, login may stop working until you access Vashti through HTTPS or use recover_network.txt."
          confirmLabel="Save Anyway"
          isBusy={isSavingNetwork}
          onCancel={() => setNetworkConfirmPending(false)}
          onConfirm={() => void saveNetworkSettings()}
        />
      )}
      {updateConfirmPending && updateStatus?.available && (
        <ConfirmDialog
          title={`Install ${updateStatus.available.version}?`}
          message="Vashti will download and verify the release before briefly restarting. The database and current binary are backed up and restored automatically if the new version fails its health check."
          confirmLabel="Install Update"
          confirmTone="primary"
          isBusy={isRequestingUpdate}
          onCancel={() => setUpdateConfirmPending(false)}
          onConfirm={() => void requestUpdateInstall()}
        />
      )}
    </SettingsPanel>
  );
}

function defaultInternalVashtiUrl() {
  if (typeof window === "undefined") {
    return "http://127.0.0.1:7771";
  }
  const host = window.location.hostname || "127.0.0.1";
  return `http://${host}:7771`;
}

function formatUpdateTimestamp(timestamp: number) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(timestamp * 1000));
}

function domainFromPublicBaseUrl(publicBaseUrl: string | null) {
  if (!publicBaseUrl) {
    return "";
  }
  try {
    return new URL(publicBaseUrl).host;
  } catch {
    return publicBaseUrl.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
  }
}

function proxyConfig(proxyType: "caddy" | "nginx", domain: string, internalUrl: string) {
  const cleanedDomain = domain.trim() || "chat.example.com";
  const cleanedInternalUrl = internalUrl.trim() || "http://127.0.0.1:7771";
  const internalTarget = cleanedInternalUrl.replace(/^https?:\/\//, "");

  if (proxyType === "caddy") {
    return `${cleanedDomain} {
    reverse_proxy ${internalTarget}
}`;
  }

  return `server {
    listen 80;
    server_name ${cleanedDomain};
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ${cleanedDomain};

    ssl_certificate /etc/letsencrypt/live/${cleanedDomain}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${cleanedDomain}/privkey.pem;

    location / {
        proxy_pass ${cleanedInternalUrl};

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        proxy_cookie_flags vashti_session secure httponly samesite=lax;
    }
}`;
}
