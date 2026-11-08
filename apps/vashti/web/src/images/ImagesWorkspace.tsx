import { useCallback, useEffect, useRef, useState } from "react";
import { ConfirmDialog } from "../common";
import {
  ArrowLeft,
  ImagePlus,
  Images,
  Pencil,
  RefreshCw,
  Square,
  Settings2,
  X,
  LockKeyhole,
  Server,
  Film,
} from "lucide-react";
import type { BackendModelGroup } from "../types";
import { ImageAssetProvider } from "./ImageAsset";
import { ImagePanel, NumberField } from "./controls";
import { imageControls } from "./GenerationControls";
import { imageError, requestJson, notifyImageFlows, writeImage } from "./api";
import { ImageStorageProvider, useImagePath, type ImageStorage } from "./storage";
import { ImageFlow } from "./ImageFlow";
import { ImageGallery } from "./ImageGallery";
import { workflowLabel, type Run } from "./types";
import type {
  ImageHost,
  ImageSession,
  ImageView,
  Preferences,
  ServiceSettings,
} from "./types";
import "./images.css";
import { VideoWorkspace } from "../videos/VideoWorkspace";
import type { VideoHost } from "../videos/types";

type Props = {
  storageMode: ImageStorage;
  view: ImageView;
  sessionId?: string;
  newFlowKey: number;
  headerTarget: HTMLElement | null;
  isAdmin: boolean;
  modelGroups: BackendModelGroup[];
  onNavigate: (view: ImageView, sessionId?: string, storageMode?: ImageStorage) => void;
  onOpenBackends: () => void;
};
export function ImagesWorkspace(props: Props) {
  return (
    <ImageStorageProvider mode={props.storageMode}>
      <ImageAssetProvider key={props.storageMode}>
        <Workspace {...props} />
      </ImageAssetProvider>
    </ImageStorageProvider>
  );
}
function Workspace({
  storageMode,
  view,
  sessionId,
  newFlowKey,
  headerTarget,
  isAdmin,
  modelGroups,
  onNavigate,
  onOpenBackends,
}: Props) {
  const imagePath = useImagePath();
  const [preferences, setPreferences] = useState<Preferences | null>(null);
  const [pendingPreferences, setPendingPreferences] =
    useState<Preferences | null>(null);
  const [hosts, setHosts] = useState<ImageHost[]>([]);
  const [videoHosts, setVideoHosts] = useState<VideoHost[]>([]);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState("");
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [settings, setSettings] = useState<ServiceSettings | null>(null);
  const [savedSettings, setSavedSettings] = useState<ServiceSettings | null>(null);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const afterClose = useRef<(() => void) | undefined>(undefined);
  const preferencesRequest = useRef(0);
  const [busy, setBusy] = useState(false);
  const [activeRuns, setActiveRuns] = useState<Run[]>([]);
  const [createdFlow, setCreatedFlow] = useState<{
    id: string;
    key: string;
  } | null>(null);
  const serverDirty = Boolean(settings && savedSettings &&
    (settings.backend_id !== savedSettings.backend_id || settings.max_user_bytes !== savedSettings.max_user_bytes));
  const personalDirty = Boolean(pendingPreferences && preferences && JSON.stringify(pendingPreferences) !== JSON.stringify(preferences));
  useEffect(() => {
    if (!preferencesOpen || (!personalDirty && !serverDirty)) return;
    const protect = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", protect);
    return () => window.removeEventListener("beforeunload", protect);
  }, [preferencesOpen, personalDirty, serverDirty]);

  function closePreferences(next?: () => void) {
    if (busy) return;
    afterClose.current = next;
    if (personalDirty || serverDirty) { setConfirmDiscard(true); return; }
    preferencesRequest.current++;
    setPreferencesOpen(false);
    next?.();
  }
  useEffect(() => {
    let live = true;
    const refresh = () => {
      void requestJson<Run[]>("/api/images/runs/active")
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
      const [images, videos] = await Promise.allSettled([
        requestJson<ImageHost[]>("/api/images/hosts"),
        requestJson<VideoHost[]>("/api/videos/hosts"),
      ]);
      setVideoHosts(videos.status === "fulfilled" ? videos.value : []);
      if (images.status === "rejected") throw images.reason;
      setHosts(images.value);
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
    const request = ++preferencesRequest.current;
    setPendingPreferences(preferences ? structuredClone(preferences) : null);
    setPreferencesOpen(true);
    setSettings(null);
    setSavedSettings(null);
    if (isAdmin)
      void requestJson<ServiceSettings>("/api/images/settings")
        .then((value) => {
          if (request !== preferencesRequest.current) return;
          setSettings(value); setSavedSettings(value);
        })
        .catch((error) => { if (request === preferencesRequest.current) setError(imageError(error)); });
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
      if (!serverDirty) setPreferencesOpen(false);
    } catch (error) {
      setError(imageError(error));
    } finally {
      setBusy(false);
    }
  }
  const flow = view !== "home" && view !== "gallery";
  return (
    <div
      className={`images-workspace${flow ? " images-conversation-workspace" : view === "gallery" ? " images-gallery-workspace" : ""}`}
    >
      {activeRuns
        .filter((run) => storageMode === "server" && (!flow || run.session_id !== sessionId))
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
      {view === "home" && (
        <div className="images-actions image-storage-switch" aria-label="Image storage">
          <button type="button" aria-pressed={storageMode === "server"} onClick={() => onNavigate("home", undefined, "server")}><Server size={18} />Server</button>
          <button type="button" aria-pressed={storageMode === "device"} onClick={() => onNavigate("home", undefined, "device")}><LockKeyhole size={18} />This device</button>
        </div>
      )}
      {view === "video" ? (
        <VideoWorkspace storageMode={storageMode} sessionId={sessionId} newFlowKey={newFlowKey} headerTarget={headerTarget} onNavigate={onNavigate} />
      ) : view === "home" ? (
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
          {videoHosts.length > 0 && <button type="button" onClick={() => onNavigate("video")}><Film /><span>Video</span></button>}
        </div>
      ) : view === "gallery" && preferences ? (
        <ImageGallery key={`${imagePath}:${sessionId ?? "all"}`} sessionId={sessionId} preferences={preferences} onNavigate={onNavigate} />
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
          onNewFlow={async (draft, mode) => {
            const next = await writeImage<ImageSession>(
              `${imagePath}/sessions`,
              { draft, mode },
            );
            setCreatedFlow(null);
            notifyImageFlows();
            onNavigate("flow", next.id);
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
          onClose={() => closePreferences()}
        >
          <fieldset disabled={busy}>
          {error && <p role="alert" className="images-error">{error}</p>}
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
                <button type="button" onClick={() => closePreferences(onOpenBackends)}>
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
                  disabled={busy || !serverDirty}
                  onClick={() => {
                    setBusy(true);
                    void writeImage<ServiceSettings>(
                      "/api/images/settings",
                      {
                        backend_id: settings.backend_id,
                        max_user_bytes: settings.max_user_bytes,
                      },
                      "PUT",
                    )
                      .then((value) => { setSettings(value); setSavedSettings(value); })
                      .catch((error) => setError(imageError(error)))
                      .finally(() => setBusy(false));
                  }}
                >
                  {serverDirty ? "Save server settings" : "Server settings saved"}
                </button>
              </div>
            </details>
          )}
          <button
            type="button"
            className="primary-action"
            disabled={busy || !personalDirty}
            onClick={() => void savePreferences()}
          >
            Save my preferences
          </button>
          </fieldset>
        </ImagePanel>
      )}
      {confirmDiscard && <ConfirmDialog title="Discard image preference changes?"
        message="Unsaved account preferences and server settings will be discarded. Your conversation settings are unchanged."
        confirmLabel="Discard changes" isBusy={false}
        onCancel={() => { setConfirmDiscard(false); afterClose.current = undefined; }}
        onConfirm={() => {
          preferencesRequest.current++;
          setConfirmDiscard(false);
          setPreferencesOpen(false);
          afterClose.current?.();
          afterClose.current = undefined;
        }} />}
    </div>
  );
}
