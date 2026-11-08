import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Download,
  Film,
  ImagePlus,
  Images,
  LockKeyhole,
  Play,
  RefreshCw,
  RotateCcw,
  Save,
  SendHorizontal,
  Settings2,
  Square,
  X,
} from "lucide-react";
import { ModelPicker } from "../ModelPicker";
import { modelParts, modelValue } from "../modelSelection";
import { ActionMenu } from "../ActionMenu";
import { PhotoViewer } from "../PhotoViewer";
import { saveFile } from "../saveFile";
import { ApiRequestError, requestJson } from "../api";
import { usesMobileInputBehavior } from "../viewport";
import { ImagePanel } from "../images/controls";
import { aspectRatios, closestAspect } from "../images/aspectRatio";
import { mediaModelGroups } from "../images/flow";
import type { ImageView } from "../images/types";
import {
  videoRepository,
  notifyVideos,
  serializeVideoMutation,
  type VideoStorage,
} from "./repository";
import {
  videoDefaults,
  videoActive,
  videoSize,
  nextVideoSeed,
  type VideoDraft,
  type VideoHost,
  type VideoRun,
  type VideoSession,
  type VideoStatus,
} from "./types";
import "./videos.css";

type Repo = ReturnType<typeof videoRepository>;
type SavedDraft = {
  draft: VideoDraft;
  session_id?: string;
  request_id: string;
};
const message = (e: unknown) =>
  e instanceof Error ? e.message : "Video request failed";
const readDraft = (run: VideoRun) => JSON.parse(run.draft_json) as VideoDraft;
export function VideoWorkspace({
  storageMode,
  sessionId,
  newFlowKey,
  headerTarget,
  onNavigate,
}: {
  storageMode: VideoStorage;
  sessionId?: string;
  newFlowKey: number;
  headerTarget: HTMLElement | null;
  onNavigate: (view: ImageView, id?: string, mode?: VideoStorage) => void;
}) {
  return (
    <VideoFlow
      key={`${storageMode}:${sessionId ?? newFlowKey}`}
      storageMode={storageMode}
      sessionId={sessionId}
      headerTarget={headerTarget}
      onNavigate={onNavigate}
    />
  );
}
function VideoFlow({
  storageMode,
  sessionId,
  headerTarget,
  onNavigate,
}: {
  storageMode: VideoStorage;
  sessionId?: string;
  headerTarget: HTMLElement | null;
  onNavigate: (view: ImageView, id?: string, mode?: VideoStorage) => void;
}) {
  const repo = useMemo(() => videoRepository(storageMode), [storageMode]);
  const [draft, setDraft] = useState<VideoDraft>({ ...videoDefaults });
  const draftRef = useRef(draft);
  draftRef.current = draft;
  const [defaults, setDefaults] = useState<VideoDraft>({ ...videoDefaults });
  const [runs, setRuns] = useState<VideoRun[]>([]);
  const runsRef = useRef(runs);
  runsRef.current = runs;
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [hosts, setHosts] = useState<VideoHost[]>([]);
  const [ready, setReady] = useState(false),
    [busy, setBusy] = useState(false),
    [settings, setSettings] = useState(false),
    [focused, setFocused] = useState(false);
  const [error, setError] = useState<string | null>(null),
    [notice, setNotice] = useState<string | null>(null);
  const [progress, setProgress] = useState<VideoStatus | null>(null);
  const [selected, setSelected] = useState<Record<string, string>>({});
  const [viewFrame, setViewFrame] = useState<string | null>(null);
  const session = useRef<VideoSession | null>(null),
    requestId = useRef<string>(crypto.randomUUID()),
    saved = useRef("");
  const accepted = useRef(false);
  const revisionQueue = useRef(Promise.resolve());
  const alive = useRef(true),
    loading = useRef(false);
  const file = useRef<HTMLInputElement>(null),
    input = useRef<HTMLTextAreaElement>(null),
    transcript = useRef<HTMLDivElement>(null),
    following = useRef(true);
  const active = runs.find((r) => videoActive(r.status));
  const selectedHost = hosts.find((h) => h.backend.id === draft.backend_id);
  const workflow = selectedHost?.capabilities.workflows.find(
    (w) => w.id === draft.workflow,
  );
  const groups = useMemo(
    () =>
      mediaModelGroups(
        hosts.map((h) => ({
          backend: h.backend,
          models: h.capabilities.workflows.map((w) => ({
            name: w.id,
            supports_images: true,
          })),
        })),
      ),
    [hosts],
  );
  useEffect(() => {
    alive.current = true;
    let current = true;
    void (async () => {
      try {
        const preferences = await repo.preferences();
        repo.guard();
        if (!current) return;
        setDefaults(preferences);
        const pending = await repoDraft(repo, sessionId);
        if (!current) return;
        const id = sessionId ?? pending?.session_id;
        let value = preferences;
        if (id) {
          const page = await repo.page(id);
          repo.guard();
          if (!current) return;
          session.current = page.session;
          setRuns(page.runs);
          value = JSON.parse(page.session.draft_json);
          saved.current = page.session.draft_json;
        }
        if (pending) {
          value = pending.draft;
          requestId.current = pending.request_id;
        }
        if (current) {
          setDraft(value);
          setReady(true);
        }
      } catch (e) {
        if (current) {
          setError(message(e));
          setReady(true);
        }
      }
    })();
    void requestJson<VideoHost[]>("/api/videos/hosts")
      .then((value) => {
        if (current) setHosts(value);
      })
      .catch((e) => {
        if (current) setError(message(e));
      });
    return () => {
      current = false;
      alive.current = false;
    };
  }, [repo, sessionId]);
  useEffect(() => {
    if (ready && !draft.backend_id && hosts.length)
      setDraft((value) => ({
        ...value,
        backend_id: hosts[0].backend.id,
        workflow: hosts[0].capabilities.workflows[0].id,
      }));
  }, [ready, hosts]);
  useEffect(() => {
    if (!ready || (!sessionId && accepted.current)) return;
    const value = draft;
    void writeRepoDraft(repo, sessionId, {
      draft: value,
      session_id: session.current?.id,
      request_id: requestId.current,
    }).catch((e) => setError(message(e)));
    if (
      !session.current ||
      saved.current === JSON.stringify(value) ||
      busy ||
      active
    )
      return;
    const timer = setTimeout(() => {
      revisionQueue.current = revisionQueue.current
        .catch(() => undefined)
        .then(async () => {
          const current = session.current;
          if (!current) return;
          repo.guard();
          const updated = await repo.save(current, value);
          repo.guard();
          session.current = updated;
          saved.current = JSON.stringify(value);
        })
        .catch((e) => {
          if (alive.current) setError(message(e));
        });
    }, 650);
    return () => clearTimeout(timer);
  }, [draft, ready, busy, Boolean(active), repo]);
  useEffect(() => {
    const id = session.current?.id;
    if (!ready || !id) return;
    let current = true,
      timer = 0,
      polling = false;
    async function poll() {
      if (polling || !current) return;
      polling = true;
      try {
        let pending = runsRef.current.find((r) => videoActive(r.status));
        if (!pending) {
          const page = await repo.page(id!);
          repo.guard();
          if (!current) return;
          setRuns(page.runs);
          pending = page.runs.find((r) => videoActive(r.status));
        }
        if (pending) {
          try {
            const status = await repo.status(pending.id);
            if (!current) return;
            setProgress(status);
            if (!videoActive(status.status)) {
              await repo.finish(id!, pending.id, status);
              if (!current) return;
              const next = await repo.page(id!);
              setRuns(next.runs);
              setProgress(null);
              if (status.status === "failed")
                setError(status.progress.error ?? "Video generation failed");
            }
          } catch (e) {
            if (
              storageMode === "device" &&
              e instanceof ApiRequestError &&
              e.status === 404
            ) {
              await repo.finish(id!, pending.id, {
                status: "failed",
                progress: {
                  error:
                    "Generation was interrupted or expired. The prompt and first frame are saved on this device.",
                },
              });
            }
            if (current) setError(message(e));
          }
        }
      } catch (e) {
        if (current) setError(message(e));
      } finally {
        polling = false;
        if (current)
          timer = window.setTimeout(
            poll,
            document.hidden
              ? 10000
              : runsRef.current.some((r) => videoActive(r.status))
                ? 1200
                : 15000,
          );
      }
    }
    const resume = () => {
      if (!document.hidden) {
        clearTimeout(timer);
        void poll();
      }
    };
    document.addEventListener("visibilitychange", resume);
    void poll();
    return () => {
      current = false;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", resume);
    };
  }, [repo, ready, session.current?.id, active?.id]);
  useEffect(() => {
    if (!transcript.current) return;
    const element = transcript.current,
      observer = new ResizeObserver(() => {
        if (following.current) element.scrollTop = element.scrollHeight;
      });
    if (element.firstElementChild) observer.observe(element.firstElementChild);
    return () => observer.disconnect();
  }, [ready]);
  function change(patch: Partial<VideoDraft>) {
    requestId.current = crypto.randomUUID();
    setNotice(null);
    setDraft((value) => ({ ...value, ...patch }));
  }
  async function send(retry?: VideoRun) {
    if (busy || loading.current || active || !ready) return;
    loading.current = true;
    setBusy(true);
    setError(null);
    let value = retry ? readDraft(retry) : draft;
    if (retry && storageMode === "device") {
      const origin = runs.find((r) => r.id === retry.group_id) ?? retry;
      value = {
        ...readDraft(origin),
        seed: nextVideoSeed(
          readDraft(origin),
          runs.filter((r) => r.group_id === retry.group_id).length,
        ),
      };
    }
    const nonce = retry ? crypto.randomUUID() : requestId.current;
    try {
      repo.guard();
      await revisionQueue.current;
      repo.guard();
      let current = session.current;
      if (!current) {
        current = await repo.create(value);
        repo.guard();
        session.current = current;
      }
      await writeRepoDraft(repo, sessionId, {
        draft: value,
        session_id: current.id,
        request_id: nonce,
      });
      repo.guard();
      const response = await repo.submit(current, value, nonce, retry?.id);
      repo.guard();
      notifyVideos();
      accepted.current = true;
      if (!alive.current) return;
      const page = await repo.page(current.id);
      repo.guard();
      session.current = page.session;
      setRuns(page.runs);
      saved.current = page.session.draft_json;
      if (!retry) {
        const next = { ...value, seed: nextVideoSeed(value) };
        setDraft(next);
        requestId.current = crypto.randomUUID();
        await writeRepoDraft(repo, current.id, {
          draft: next,
          session_id: current.id,
          request_id: requestId.current,
        });
      }
      setSelected((previous) => ({
        ...previous,
        [retry?.group_id ?? response.id]: response.id,
      }));
      setFocused(false);
      input.current?.blur();
      following.current = true;
      if (!sessionId) {
        const { deletePrivateImage } = await import("../privateChatStore");
        repo.guard();
        await serializeVideoMutation(`video:composer:${repo.mode}:new`, () =>
          deletePrivateImage(`video:composer:${repo.mode}:new`),
        );
        repo.guard();
        onNavigate("video", current.id, storageMode);
      }
    } catch (e) {
      if (alive.current) setError(message(e));
    } finally {
      loading.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function upload(file: File) {
    setBusy(true);
    setError(null);
    try {
      const result = await repo.frame(file);
      repo.guard();
      if (alive.current)
        change({
          first_frame_id: result.id,
          ...(workflow
            ? videoSize(result.width / result.height, workflow)
            : {}),
        });
    } catch (e) {
      if (alive.current) setError(message(e));
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  const header = (
    <div className="image-flow-header-controls">
      {storageMode === "device" && (
        <LockKeyhole size={18} aria-label="Device-only video flow" />
      )}
      <ModelPicker
        task="image"
        groups={groups}
        personas={[]}
        privatePersonas={[]}
        isLoading={!ready}
        error={null}
        value={
          draft.backend_id ? modelValue(draft.backend_id, draft.workflow) : ""
        }
        modelLabel={() => "MiniMax H3"}
        emptyLabel="No video models"
        ariaLabel="Video model"
        onChange={(value) => {
          const model = modelParts(value);
          if (model)
            change({ backend_id: model.backendId, workflow: model.modelName });
        }}
      />
      <ActionMenu label="Video options">
        <button
          type="button"
          onClick={() => onNavigate("home", undefined, storageMode)}
        >
          <ArrowLeft />
          Image apps
        </button>
        <button
          type="button"
          onClick={() =>
            void requestJson<VideoHost[]>("/api/videos/hosts")
              .then(setHosts)
              .catch((e) => setError(message(e)))
          }
        >
          <RefreshCw />
          Refresh models
        </button>
      </ActionMenu>
    </div>
  );
  const groupIds = [...new Set(runs.map((r) => r.group_id))];
  return (
    <div className="video-workspace image-flow">
      {headerTarget ? createPortal(header, headerTarget) : header}
      {error && (
        <div className="images-error video-error" role="alert">
          {error}
          <button
            type="button"
            className="message-icon-button"
            aria-label="Dismiss video error"
            onClick={() => setError(null)}
          >
            <X size={16} />
          </button>
        </div>
      )}
      <div
        className="video-transcript"
        ref={transcript}
        onClickCapture={(e) => {
          if (e.target instanceof Element && e.target.closest("summary"))
            following.current = false;
        }}
        onScroll={() => {
          const e = transcript.current;
          if (e)
            following.current =
              e.scrollHeight - e.scrollTop - e.clientHeight < 80;
        }}
      >
        <div className="video-thread">
          {!runs.length && (
            <div className="video-empty">
              <Film size={30} />
              <h1>Video</h1>
            </div>
          )}
          {groupIds.map((group) => {
            const versions = runs.filter((r) => r.group_id === group),
              run =
                versions.find((r) => r.id === selected[group]) ??
                versions[versions.length - 1],
              recipe = readDraft(run),
              index = versions.indexOf(run);
            return (
              <section className="video-turn" key={group}>
                <details className="video-prompt">
                  <summary>
                    <span>
                      {recipe.prompt.trim() ||
                        "Animate this image with gentle natural motion."}
                    </span>
                    <ChevronDown size={16} />
                  </summary>
                  <dl>
                    <dt>Resolution</dt>
                    <dd>
                      {recipe.width} × {recipe.height}
                    </dd>
                    <dt>Duration</dt>
                    <dd>{(recipe.frames / 24).toFixed(2)} s</dd>
                    <dt>Audio</dt>
                    <dd>{recipe.audio ? "On" : "Off"}</dd>
                    <dt>Seed</dt>
                    <dd>{recipe.seed}</dd>
                  </dl>
                  {recipe.first_frame_id && (
                    <button
                      type="button"
                      className="video-source"
                      aria-label="View original first frame"
                      onClick={() => setViewFrame(recipe.first_frame_id)}
                    >
                      <VideoFrame repo={repo} id={recipe.first_frame_id} />
                    </button>
                  )}
                </details>
                <article className="video-response">
                  <header>
                    <span>MiniMax H3</span>
                    {versions.length > 1 && (
                      <div className="video-versions">
                        <button
                          type="button"
                          className="message-icon-button"
                          disabled={index === 0}
                          aria-label="Previous video version"
                          onClick={() =>
                            setSelected((v) => ({
                              ...v,
                              [group]: versions[index - 1].id,
                            }))
                          }
                        >
                          <ChevronLeft size={18} />
                        </button>
                        <span>
                          {index + 1}/{versions.length}
                        </span>
                        <button
                          type="button"
                          className="message-icon-button"
                          disabled={index === versions.length - 1}
                          aria-label="Next video version"
                          onClick={() =>
                            setSelected((v) => ({
                              ...v,
                              [group]: versions[index + 1].id,
                            }))
                          }
                        >
                          <ChevronRight size={18} />
                        </button>
                      </div>
                    )}
                  </header>
                  {videoActive(run.status) ? (
                    <div
                      className="video-stage"
                      style={{
                        aspectRatio: `${readDraft(run).width}/${readDraft(run).height}`,
                      }}
                    >
                      {progress?.progress.preview ? (
                        <img
                          src={`data:image/png;base64,${progress.progress.preview.png_base64}`}
                          alt="Video generation preview"
                        />
                      ) : recipe.first_frame_id ? (
                        <VideoFrame repo={repo} id={recipe.first_frame_id} />
                      ) : null}
                      <div className="video-progress" role="status">
                        {progress?.progress.phase ?? run.status}
                        {progress?.progress.step
                          ? ` · ${progress.progress.step}/${progress.progress.steps}`
                          : ""}
                      </div>
                    </div>
                  ) : run.status === "complete" ? (
                    <VideoPlayer
                      key={run.id}
                      repo={repo}
                      run={run}
                      playing={playingId === run.id}
                      onActivate={() => setPlayingId(run.id)}
                    />
                  ) : (
                    <p role="status">
                      {run.error ??
                        (run.status === "cancelled"
                          ? "Generation stopped"
                          : "Video unavailable")}
                    </p>
                  )}
                  <footer>
                    <button
                      type="button"
                      className="message-icon-button"
                      aria-label="Regenerate video"
                      title="Regenerate video"
                      disabled={busy || Boolean(active)}
                      onClick={() => void send(run)}
                    >
                      <RefreshCw size={18} />
                    </button>
                    <button
                      type="button"
                      className="message-icon-button"
                      aria-label="Use video prompt and settings"
                      title="Use prompt and settings"
                      disabled={busy}
                      onClick={() => change(readDraft(run))}
                    >
                      <Settings2 size={18} />
                    </button>
                  </footer>
                </article>
              </section>
            );
          })}
        </div>
      </div>
      <div className="image-composer-dock video-composer-dock">
        <form
          className={`image-chat-composer${!focused ? " image-chat-composer-compact" : ""}`}
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <div className="video-prompt-row">
            {draft.first_frame_id && (
              <button
                type="button"
                className="video-input-preview"
                aria-label="View first frame"
                onClick={() => setViewFrame(draft.first_frame_id)}
              >
                <VideoFrame repo={repo} id={draft.first_frame_id} />
              </button>
            )}
            <textarea
              ref={input}
              aria-label="Video prompt"
              placeholder="Describe the motion..."
              rows={focused ? 3 : 1}
              value={draft.prompt}
              disabled={!ready || busy}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              onChange={(e) => change({ prompt: e.target.value })}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.altKey &&
                  !e.nativeEvent.isComposing &&
                  (e.ctrlKey || e.metaKey || !usesMobileInputBehavior())
                ) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
            />
          </div>
          <footer>
            <div className="images-actions">
              <button
                type="button"
                className="icon-button primary-action"
                title="Video inference settings"
                aria-label="Video inference settings"
                aria-expanded={settings}
                onClick={() => setSettings(true)}
              >
                <Settings2 />
              </button>
              <button
                type="button"
                className="icon-button primary-action"
                title="Choose first frame"
                aria-label="Choose first frame"
                disabled={busy || !ready}
                onClick={() => file.current?.click()}
              >
                <ImagePlus />
              </button>
              {draft.first_frame_id && (
                <button
                  type="button"
                  className="message-icon-button"
                  aria-label="Remove first frame"
                  title="Remove first frame"
                  disabled={busy}
                  onClick={() => change({ first_frame_id: null })}
                >
                  <X size={18} />
                </button>
              )}
            </div>
            {active ? (
              <button
                type="button"
                className="icon-button primary-action"
                aria-label="Stop video generation"
                onClick={() =>
                  void repo
                    .stop(active.id)
                    .then(() =>
                      setProgress({ status: "cancelled", progress: {} }),
                    )
                    .catch((e) => setError(message(e)))
                }
              >
                <Square />
              </button>
            ) : (
              <button
                type="submit"
                className="icon-button primary-action"
                aria-label="Send video prompt"
                disabled={busy || !ready || !workflow || !draft.first_frame_id}
              >
                <SendHorizontal />
              </button>
            )}
          </footer>
          <input
            ref={file}
            type="file"
            accept="image/*"
            className="image-file-input"
            onChange={(e) => {
              const value = e.target.files?.[0];
              e.target.value = "";
              if (value) void upload(value);
            }}
          />
        </form>
        {!hosts.length && ready && (
          <p className="video-caption">No ready video model is available.</p>
        )}
      </div>
      {settings && (
        <ImagePanel title="Video settings" onClose={() => setSettings(false)}>
          <div className="video-settings">
            <label>
              Aspect ratio
              <select
                aria-label="Video aspect ratio"
                value={closestAspect(draft.width, draft.height)}
                disabled={!workflow}
                onChange={(e) => {
                  if (!workflow) return;
                  if (e.target.value === "original") {
                    if (!draft.first_frame_id) return;
                    void repo
                      .frameBlob(draft.first_frame_id, false)
                      .then(createImageBitmap)
                      .then((bitmap) => {
                        try {
                          repo.guard();
                          if (alive.current)
                            change(
                              videoSize(bitmap.width / bitmap.height, workflow),
                            );
                        } finally {
                          bitmap.close();
                        }
                      })
                      .catch((e) => setError(message(e)));
                    return;
                  }
                  const [w, h] = e.target.value.split(":").map(Number);
                  change(videoSize(w / h, workflow));
                }}
              >
                <option value="original" disabled={!draft.first_frame_id}>
                  Original image
                </option>
                <option value="Custom" disabled>
                  {draft.width} × {draft.height}
                </option>
                {aspectRatios.map((r) => (
                  <option key={r}>{r}</option>
                ))}
              </select>
            </label>
            <label>
              Clip length
              <select
                aria-label="Video clip length"
                value={draft.frames}
                disabled={!workflow}
                onChange={(e) => change({ frames: Number(e.target.value) })}
              >
                {(workflow?.frames ?? [22, 39, 56, 73, 90, 107, 124]).map(
                  (n) => (
                    <option key={n} value={n}>
                      {(n / 24).toFixed(2)} s
                    </option>
                  ),
                )}
              </select>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={draft.audio}
                disabled={!workflow?.audio}
                onChange={(e) => change({ audio: e.target.checked })}
              />
              Audio
            </label>
            <label>
              Seed
              <input
                aria-label="Video seed"
                inputMode="numeric"
                pattern="[0-9]+"
                maxLength={19}
                value={draft.seed}
                onChange={(e) => {
                  if (/^\d*$/.test(e.target.value))
                    change({ seed: e.target.value });
                }}
                onBlur={() => {
                  try {
                    if (
                      !draft.seed ||
                      BigInt(draft.seed) > 9223372036854775807n
                    )
                      change({ seed: "0" });
                  } catch {
                    change({ seed: "0" });
                  }
                }}
              />
            </label>
            <label>
              Next seed
              <select
                aria-label="Video seed mode"
                value={draft.seed_mode}
                onChange={(e) =>
                  change({
                    seed_mode: e.target.value as VideoDraft["seed_mode"],
                  })
                }
              >
                <option value="increment">Increment</option>
                <option value="random">Random</option>
                <option value="fixed">Fixed</option>
              </select>
            </label>
            <div className="images-actions">
              <button
                type="button"
                onClick={() =>
                  void repo
                    .defaults(draft)
                    .then((value) => {
                      setDefaults(value);
                      setNotice(
                        storageMode === "device"
                          ? "Device defaults saved"
                          : "Account defaults saved",
                      );
                    })
                    .catch((e) => setError(message(e)))
                }
              >
                <Save size={17} />
                Save defaults
              </button>
              <button
                type="button"
                onClick={() =>
                  change({
                    ...defaults,
                    prompt: draft.prompt,
                    first_frame_id: draft.first_frame_id,
                  })
                }
              >
                <RotateCcw size={17} />
                Reset settings
              </button>
            </div>
            <button
              type="button"
              onClick={() => {
                onNavigate("video", undefined, storageMode);
              }}
            >
              <Film size={17} />
              New video flow
            </button>
            {notice && <p role="status">{notice}</p>}
          </div>
        </ImagePanel>
      )}
      {viewFrame && (
        <PhotoViewer
          id={viewFrame}
          title="First frame"
          onClose={() => setViewFrame(null)}
          onStep={() => {}}
          canPrevious={false}
          canNext={false}
        >
          <VideoFrame repo={repo} id={viewFrame} original />
        </PhotoViewer>
      )}
    </div>
  );
}

// The generic encrypted device store holds drafts separately from submitted recipes.
async function repoDraft(repo: Repo, id?: string) {
  repo.guard();
  const { readPrivateImage } = await import("../privateChatStore");
  const value = await readPrivateImage<SavedDraft>(
    `video:composer:${repo.mode}:${id ?? "new"}`,
  );
  repo.guard();
  return value;
}
async function writeRepoDraft(
  repo: Repo,
  id: string | undefined,
  value: SavedDraft,
) {
  repo.guard();
  const { writePrivateImage } = await import("../privateChatStore");
  await serializeVideoMutation(
    `video:composer:${repo.mode}:${id ?? "new"}`,
    async () => {
      repo.guard();
      await writePrivateImage(
        `video:composer:${repo.mode}:${id ?? "new"}`,
        value,
      );
    },
  );
  repo.guard();
}
function VideoFrame({
  repo,
  id,
  original = false,
}: {
  repo: Repo;
  id: string;
  original?: boolean;
}) {
  const [image, setImage] = useState<{ id: string; url: string } | null>(null);
  useEffect(() => {
    let current = true,
      url = "";
    void repo
      .frameBlob(id, !original)
      .then((blob) => {
        repo.guard();
        if (current) {
          url = URL.createObjectURL(blob);
          setImage({ id, url });
        }
      })
      .catch(() => {});
    return () => {
      current = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [repo, id, original]);
  return image?.id === id ? (
    <img src={image.url} alt="First frame" />
  ) : (
    <span className="video-frame-placeholder">
      <Images size={20} />
    </span>
  );
}
function VideoPlayer({
  repo,
  run,
  playing,
  onActivate,
}: {
  repo: Repo;
  run: VideoRun;
  playing: boolean;
  onActivate: () => void;
}) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
  const blob = useRef<Blob | null>(null);
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    if (playing) void load();
    else {
      setUrl("");
      blob.current = null;
      setBusy(false);
    }
    return () => {
      generation.current++;
      controller.current?.abort();
    };
  }, [playing]);
  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url],
  );
  const draft = readDraft(run);
  async function load(download = false) {
    const attempt = generation.current;
    controller.current?.abort();
    controller.current = new AbortController();
    setBusy(true);
    setError(null);
    try {
      const value =
        blob.current ?? (await repo.result(run.id, controller.current.signal));
      repo.guard();
      if (attempt !== generation.current) return;
      if (playing) blob.current = value;
      if (download) {
        const result = await saveFile(
          value,
          `vashti-${run.id}.mp4`,
          "video/mp4",
        );
        setError(result === "cancelled" ? "Save cancelled" : null);
      } else setUrl(URL.createObjectURL(value));
    } catch (e) {
      if (attempt === generation.current) setError(message(e));
    } finally {
      if (attempt === generation.current) setBusy(false);
    }
  }
  return (
    <>
      <div
        className="video-stage"
        style={{ aspectRatio: `${draft.width}/${draft.height}` }}
      >
        {url ? (
          <video
            key={run.id}
            controls
            playsInline
            autoPlay
            preload="metadata"
            src={url}
            onError={() =>
              setError(
                "This device could not play the AV1 clip. You can still download it.",
              )
            }
          />
        ) : (
          <>
            {draft.first_frame_id && (
              <VideoFrame repo={repo} id={draft.first_frame_id} />
            )}
            <button
              type="button"
              className="video-play"
              aria-label="Play video"
              disabled={busy}
              onClick={onActivate}
            >
              {busy ? <RefreshCw /> : <Play />}
            </button>
          </>
        )}
      </div>
      <button
        type="button"
        className="message-icon-button video-download"
        aria-label="Download video"
        title="Download video"
        disabled={busy}
        onClick={() => void load(true)}
      >
        <Download size={18} />
      </button>
      {error && (
        <p className="images-error" role="status">
          {error}
        </p>
      )}
    </>
  );
}
