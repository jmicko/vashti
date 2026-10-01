import { useCallback, useEffect, useState } from "react";
import {
  ArrowLeft,
  ImagePlus,
  Images,
  Pencil,
  RefreshCw,
  Square,
  Settings2,
  X,
} from "lucide-react";
import { requestJson } from "../api";
import type { BackendModelGroup } from "../types";
import { ImageAssetProvider } from "./ImageAsset";
import { ImagePanel, NumberField } from "./controls";
import { imageControls } from "./GenerationControls";
import { imageError, imagePath, writeImage } from "./api";
import { ImageFlow } from "./ImageFlow";
import { ImageGallery } from "./ImageGallery";
import { workflowLabel, type Run } from "./types";
import type {
  ImageHost,
  ImageView,
  Preferences,
  ServiceSettings,
} from "./types";
import "./images.css";

type Props = {
  view: ImageView;
  sessionId?: string;
  newFlowKey: number;
  headerTarget: HTMLElement | null;
  isAdmin: boolean;
  modelGroups: BackendModelGroup[];
  onNavigate: (view: ImageView, sessionId?: string) => void;
  onOpenBackends: () => void;
};
export function ImagesWorkspace(props: Props) {
  return (
    <ImageAssetProvider>
      <Workspace {...props} />
    </ImageAssetProvider>
  );
}
function Workspace({
  view,
  sessionId,
  newFlowKey,
  headerTarget,
  isAdmin,
  modelGroups,
  onNavigate,
  onOpenBackends,
}: Props) {
  const [preferences, setPreferences] = useState<Preferences | null>(null);
  const [pendingPreferences, setPendingPreferences] =
    useState<Preferences | null>(null);
  const [hosts, setHosts] = useState<ImageHost[]>([]);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState("");
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [settings, setSettings] = useState<ServiceSettings | null>(null);
  const [busy, setBusy] = useState(false);
  const [activeRuns, setActiveRuns] = useState<Run[]>([]);
  const [createdFlow, setCreatedFlow] = useState<{
    id: string;
    key: string;
  } | null>(null);
  useEffect(() => {
    let live = true;
    const refresh = () => {
      void requestJson<Run[]>(`${imagePath}/runs/active`)
        .then((values) => {
          if (live) setActiveRuns(values);
        })
        .catch(() => {});
    };
    refresh();
    const timer = setInterval(refresh, 5000);
    window.addEventListener("vashti:image-flows-changed", refresh);
    return () => {
      live = false;
      clearInterval(timer);
      window.removeEventListener("vashti:image-flows-changed", refresh);
    };
  }, []);
  useEffect(() => {
    let live = true;
    requestJson<Preferences>(`${imagePath}/preferences`)
      .then((value) => {
        if (live) setPreferences(value);
      })
      .catch((error) => {
        if (live) setError(imageError(error));
      });
    return () => {
      live = false;
    };
  }, []);
  const refreshHosts = useCallback(async () => {
    setChecking(true);
    try {
      setHosts(await requestJson<ImageHost[]>(`${imagePath}/hosts`));
      setError("");
    } catch (error) {
      setError(imageError(error));
    } finally {
      setChecking(false);
    }
  }, []);
  useEffect(() => {
    void refreshHosts();
  }, [refreshHosts, modelGroups]);
  useEffect(() => {
    const focus = () => {
      void refreshHosts();
    };
    window.addEventListener("focus", focus);
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") void refreshHosts();
    }, 60000);
    return () => {
      clearInterval(timer);
      window.removeEventListener("focus", focus);
    };
  }, [refreshHosts]);
  function openPreferences() {
    setPendingPreferences(preferences ? structuredClone(preferences) : null);
    setPreferencesOpen(true);
    if (isAdmin)
      void requestJson<ServiceSettings>(`${imagePath}/settings`)
        .then(setSettings)
        .catch((error) => setError(imageError(error)));
  }
  async function savePreferences() {
    if (!pendingPreferences) return;
    setBusy(true);
    try {
      setPreferences(
        await writeImage<Preferences>(
          `${imagePath}/preferences`,
          pendingPreferences,
          "PUT",
        ),
      );
      setPreferencesOpen(false);
    } catch (error) {
      setError(imageError(error));
    } finally {
      setBusy(false);
    }
  }
  const flow = view !== "home" && view !== "gallery";
  return (
    <div
      className={`images-workspace${flow ? " images-conversation-workspace" : ""}`}
    >
      {activeRuns
        .filter((run) => !flow || run.session_id !== sessionId)
        .map((run) => (
          <div className="image-global-run" key={run.id}>
            <span role="status">
              {workflowLabel(run.request.draft.recipe.workflow)} · {run.status}
            </span>
            <button
              type="button"
              onClick={() => onNavigate("flow", run.session_id)}
            >
              View generation
            </button>
            <button
              type="button"
              className="images-icon"
              aria-label="Stop image generation"
              title="Stop image generation"
              disabled={run.status === "stopping"}
              onClick={() => {
                void writeImage<Run>(`${imagePath}/runs/${run.id}/stop`, {})
                  .then((next) =>
                    setActiveRuns((values) =>
                      values.map((value) =>
                        value.id === next.id ? next : value,
                      ),
                    ),
                  )
                  .catch((error) => setError(imageError(error)));
              }}
            >
              <Square size={18} />
            </button>
          </div>
        ))}
      {error && (
        <div className="images-error" role="alert">
          <span>{error}</span>
          <button
            type="button"
            className="images-icon"
            aria-label="Dismiss error"
            onClick={() => setError("")}
          >
            <X size={18} />
          </button>
        </div>
      )}
      {!flow && (
        <header className="images-page-header">
          <div className="images-actions">
            {view === "gallery" && (
              <button
                type="button"
                className="icon-button"
                aria-label="Back to Images"
                title="Back to Images"
                onClick={() => onNavigate("home")}
              >
                <ArrowLeft />
              </button>
            )}
            <h1>{view === "gallery" ? "Gallery" : "Images"}</h1>
          </div>
          <button
            type="button"
            className="icon-button"
            title="Image preferences"
            aria-label="Image preferences"
            onClick={openPreferences}
          >
            <Settings2 />
          </button>
        </header>
      )}
      {view === "home" ? (
        <div className="images-apps" aria-label="Image apps">
          <button
            type="button"
            disabled={!preferences}
            onClick={() => onNavigate("generate")}
          >
            <ImagePlus />
            <span>Create</span>
          </button>
          <button
            type="button"
            disabled={!preferences}
            onClick={() => onNavigate("edit")}
          >
            <Pencil />
            <span>Edit</span>
          </button>
          <button type="button" onClick={() => onNavigate("gallery")}>
            <Images />
            <span>Gallery</span>
          </button>
        </div>
      ) : view === "gallery" && preferences ? (
        <ImageGallery preferences={preferences} onNavigate={onNavigate} />
      ) : flow && preferences ? (
        <ImageFlow
          key={
            sessionId && createdFlow?.id === sessionId
              ? createdFlow.key
              : (sessionId ?? `new:${newFlowKey}`)
          }
          sessionId={sessionId}
          initialEditing={view === "edit"}
          headerTarget={headerTarget}
          preferences={preferences}
          onPreferences={setPreferences}
          hosts={hosts}
          checking={checking}
          modelGroups={modelGroups}
          onRefreshHosts={refreshHosts}
          onPreferencesOpen={openPreferences}
          onNavigate={(next, id) => {
            if (!sessionId && next === "flow" && id)
              setCreatedFlow({ id, key: `new:${newFlowKey}` });
            onNavigate(next, id);
          }}
          isAdmin={isAdmin}
          onOpenBackends={onOpenBackends}
        />
      ) : (
        <p role="status">Loading Images...</p>
      )}
      {preferencesOpen && pendingPreferences && (
        <ImagePanel
          title="Image preferences"
          onClose={() => setPreferencesOpen(false)}
        >
          <details open>
            <summary>Visible inference controls</summary>
            {imageControls.map(([key, label]) => (
              <label className="images-check" key={key}>
                <input
                  type="checkbox"
                  checked={!pendingPreferences.hidden_controls.includes(key)}
                  onChange={(event) =>
                    setPendingPreferences({
                      ...pendingPreferences,
                      hidden_controls: event.target.checked
                        ? pendingPreferences.hidden_controls.filter(
                            (value) => value !== key,
                          )
                        : [...pendingPreferences.hidden_controls, key],
                    })
                  }
                />
                {label}
              </label>
            ))}
          </details>
          <details>
            <summary>Chat image tool permissions</summary>
            {(["generate", "edit"] as const).map((operation) => (
              <fieldset key={operation}>
                <legend>
                  {operation === "generate" ? "Generate" : "Edit"}
                </legend>
                {imageControls
                  .filter(([key]) => key !== "count")
                  .map(([key, label]) => (
                    <label className="images-check" key={key}>
                      <input
                        type="checkbox"
                        checked={pendingPreferences[
                          operation
                        ].model_may_change.includes(key)}
                        onChange={(event) =>
                          setPendingPreferences({
                            ...pendingPreferences,
                            [operation]: {
                              ...pendingPreferences[operation],
                              model_may_change: event.target.checked
                                ? [
                                    ...pendingPreferences[operation]
                                      .model_may_change,
                                    key,
                                  ]
                                : pendingPreferences[
                                    operation
                                  ].model_may_change.filter(
                                    (value) => value !== key,
                                  ),
                            },
                          })
                        }
                      />
                      {label}
                    </label>
                  ))}
              </fieldset>
            ))}
          </details>
          <details>
            <summary>Inference hosts</summary>
            <div className="images-host-statuses">
              {hosts.map((host) => (
                <section key={host.backend.id}>
                  <strong>{host.backend.name}</strong>
                  <p>{host.service.message}</p>
                </section>
              ))}
              {!hosts.length && <p>No connected inference hosts.</p>}
            </div>
            <div className="images-actions">
              <button
                type="button"
                disabled={checking}
                onClick={() => void refreshHosts()}
              >
                <RefreshCw size={18} />
                Check hosts
              </button>
              {isAdmin && (
                <button type="button" onClick={onOpenBackends}>
                  Backends
                </button>
              )}
            </div>
          </details>
          {isAdmin && settings && (
            <details>
              <summary>Server storage and tool default</summary>
              <div className="images-settings">
                <NumberField
                  label="Storage per user (MiB)"
                  min={20}
                  max={1048576}
                  value={settings.max_user_bytes / 1048576}
                  onChange={(value) =>
                    setSettings({
                      ...settings,
                      max_user_bytes: value * 1048576,
                    })
                  }
                />
                <label>
                  Default host for chat tools
                  <select
                    value={settings.backend_id ?? ""}
                    onChange={(event) =>
                      setSettings({
                        ...settings,
                        backend_id: event.target.value || null,
                      })
                    }
                  >
                    <option value="">Automatic</option>
                    {hosts.map((host) => (
                      <option key={host.backend.id} value={host.backend.id}>
                        {host.backend.name}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setBusy(true);
                    void writeImage<ServiceSettings>(
                      `${imagePath}/settings`,
                      {
                        backend_id: settings.backend_id,
                        max_user_bytes: settings.max_user_bytes,
                      },
                      "PUT",
                    )
                      .then(setSettings)
                      .catch((error) => setError(imageError(error)))
                      .finally(() => setBusy(false));
                  }}
                >
                  Save server settings
                </button>
              </div>
            </details>
          )}
          <button
            type="button"
            className="primary-action"
            disabled={busy}
            onClick={() => void savePreferences()}
          >
            Save preferences
          </button>
        </ImagePanel>
      )}
    </div>
  );
}
