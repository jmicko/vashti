import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ArrowLeft,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Copy,
  CopyPlus,
  Download,
  GitBranch,
  ImagePlus,
  Images,
  Paperclip,
  Pencil,
  RefreshCw,
  RotateCcw,
  SendHorizontal,
  Settings2,
  Sparkles,
  Square,
  X,
} from "lucide-react";
import { requestJson, requestMultipartJson } from "../api";
import { ModelPicker } from "../ModelPicker";
import { modelParts, modelValue } from "../modelSelection";
import { MessageTreeExplorer } from "../MessageTreeExplorer";
import { usesMobileInputBehavior } from "../viewport";
import type { BackendModelGroup } from "../types";
import { ImageAsset, downloadImage } from "./ImageAsset";
import { ImagePanel } from "./controls";
import { GenerationControls } from "./GenerationControls";
import {
  continuationLeaf,
  imageModelGroups,
  imageTreeMessages,
  imageTurns,
} from "./flow";
import { imagePath, writeImage } from "./api";
import {
  activeRun,
  canGenerate,
  newFlowWithSettings,
  profileFromDraft,
  resetDraftSettings,
  workflowLabel,
  type Draft,
  type ImageHost,
  type ImageResult,
  type ImageView,
  type Preferences,
  type Recipe,
  type Run,
} from "./types";
import { useImageFlow } from "./useImageFlow";
import { SwipeSurface } from "./SwipeSurface";

type Props = {
  sessionId?: string;
  initialEditing: boolean;
  headerTarget: HTMLElement | null;
  preferences: Preferences;
  onPreferences: (value: Preferences) => void;
  hosts: ImageHost[];
  checking: boolean;
  modelGroups: BackendModelGroup[];
  onRefreshHosts: () => Promise<void>;
  onPreferencesOpen: () => void;
  onNavigate: (view: ImageView, id?: string) => void;
  onNewFlow: (
    draft: Draft,
    mode: "generate" | "iterative" | "original",
  ) => Promise<void>;
  isAdmin: boolean;
  onOpenBackends: () => void;
};
export function ImageFlow({
  sessionId,
  initialEditing,
  headerTarget,
  preferences,
  onPreferences,
  hosts,
  checking,
  modelGroups,
  onRefreshHosts,
  onPreferencesOpen,
  onNavigate,
  onNewFlow,
  isAdmin,
  onOpenBackends,
}: Props) {
  const flow = useImageFlow(sessionId, initialEditing, preferences);
  const { draft, results, runs, active, leaf, busy } = flow;
  useEffect(() => {
    if (flow.session && !sessionId && !busy && flow.loaded)
      onNavigate("flow", flow.session.id);
  }, [flow.session?.id, sessionId, busy, flow.loaded]);
  const [expanded, setExpanded] = useState(false);
  const [defaultsSaved, setDefaultsSaved] = useState(false);
  useEffect(() => setDefaultsSaved(false), [draft]);
  const [treeOpen, setTreeOpen] = useState(false);
  const [confirmEndless, setConfirmEndless] = useState(false);
  const [viewer, setViewer] = useState<{ ids: string[]; index: number } | null>(
    null,
  );
  const [suggestion, setSuggestion] = useState<Run | null>(null);
  const suggestionIds = useRef(new Set<string>());
  const suggestionsInitialized = useRef(false);
  const [unseen, setUnseen] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const file = useRef<HTMLInputElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const end = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const scrollIntentUntil = useRef(0);
  function noteScrollIntent() {
    scrollIntentUntil.current = performance.now() + 240;
  }
  const groups = useMemo(() => imageModelGroups(hosts), [hosts]);
  const selectedHost = hosts.find(
    (host) => host.backend.id === draft?.backend_id,
  );
  const workflows = selectedHost?.service.capabilities?.images.workflows ?? [];
  const selectedWorkflow = workflows.find(
    (workflow) => workflow.id === draft?.recipe.workflow,
  );
  const editing = draft?.recipe.workflow === "qwen-image-2.1-edit/v1";
  const ready = Boolean(
    draft &&
    canGenerate(selectedHost?.service, draft.recipe, draft.input_ids.length),
  );
  const turns = useMemo(
    () => imageTurns(runs, results, leaf),
    [runs, results, leaf],
  );
  const tree = useMemo(
    () => imageTreeMessages(runs, results, leaf),
    [runs, results, leaf],
  );
  let root = tree.find((message) => message.id === leaf);
  const ancestors = new Set<string>();
  while (root?.parent_message_id && !ancestors.has(root.id)) {
    ancestors.add(root.id);
    const parent = tree.find(
      (message) => message.id === root!.parent_message_id,
    );
    if (!parent) break;
    root = parent;
  }
  const promptModels = useMemo(
    () =>
      modelGroups.map((group) => ({
        ...group,
        models: group.models.filter(
          (model) =>
            !model.capabilities?.includes("embedding") ||
            model.capabilities.includes("completion"),
        ),
      })),
    [modelGroups],
  );
  function patchRecipe(patch: Partial<Recipe>) {
    if (draft) flow.patch({ recipe: { ...draft.recipe, ...patch } });
  }
  function chooseModel(backendId: string, workflowId: string) {
    if (!draft) return;
    const workflow = hosts
      .find((host) => host.backend.id === backendId)
      ?.service.capabilities?.images.workflows.find(
        (workflow) => workflow.id === workflowId,
      );
    const changing = draft.recipe.workflow !== workflowId;
    const profile = workflow?.reference_images_max
      ? preferences.edit
      : preferences.generate;
    const recipe = changing
      ? {
          ...profile.recipe,
          workflow: workflowId,
          prompt: draft.recipe.prompt,
          negative_prompt: draft.recipe.negative_prompt,
          seed: draft.recipe.seed,
          flow_shift:
            workflowId === "qwen-image-2512-text-to-image/v1" ? 3.1 : null,
        }
      : { ...draft.recipe };
    flow.patch({
      backend_id: backendId,
      recipe,
      ...(workflow?.reference_images_max === 0 ? { input_ids: [] } : {}),
    });
  }
  useEffect(() => {
    if (!flow.loaded) return;
    if (!suggestionsInitialized.current) {
      suggestionsInitialized.current = true;
      for (const run of runs)
        if (!activeRun(run)) suggestionIds.current.add(run.id);
    }
    if (!draft || draft.backend_id || checking) return;
    const options = hosts.flatMap((host) =>
      (host.service.capabilities?.images.workflows ?? [])
        .filter((workflow) => workflow.state === "ready")
        .map((workflow) => ({ host, workflow })),
    );
    const matching =
      options.find((option) => option.workflow.id === draft.recipe.workflow) ??
      options.find(
        (option) =>
          Boolean(option.workflow.reference_images_max) ===
          Boolean(initialEditing || draft.input_ids.length),
      );
    if (matching) chooseModel(matching.host.backend.id, matching.workflow.id);
  }, [draft?.backend_id, draft === null, checking, hosts]);
  useEffect(() => {
    for (const run of runs) {
      if (
        run.status === "complete" &&
        run.request.improve_only &&
        run.effective_prompt &&
        !suggestionIds.current.has(run.id)
      ) {
        suggestionIds.current.add(run.id);
        setSuggestion(run);
      }
    }
  }, [runs, flow.loaded]);
  const contentSignature = `${results.length}:${leaf}:${active?.preview?.step ?? ""}:${active?.status ?? ""}:${runs.length}`;
  useEffect(() => {
    if (following.current)
      end.current?.scrollIntoView({ block: "end", behavior: "instant" });
    else if (active || results.length) setUnseen(true);
  }, [contentSignature]);
  useEffect(() => {
    const viewport = scroll.current;
    if (!viewport) return;
    const observer = new ResizeObserver(() => {
      if (following.current) viewport.scrollTop = viewport.scrollHeight;
    });
    if (end.current?.parentElement) observer.observe(end.current.parentElement);
    return () => observer.disconnect();
  }, [treeOpen]);
  useEffect(() => {
    if (!viewer) return;
    const key = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") moveViewer(-1);
      if (event.key === "ArrowRight") moveViewer(1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [Boolean(viewer)]);
  function moveViewer(delta: number) {
    setViewer((value) =>
      value
        ? {
            ...value,
            index: Math.max(
              0,
              Math.min(value.ids.length - 1, value.index + delta),
            ),
          }
        : value,
    );
  }
  function unavailable(backendId: string, id: string) {
    const host = hosts.find((value) => value.backend.id === backendId);
    const workflow = host?.service.capabilities?.images.workflows.find(
      (value) => value.id === id,
    );
    return !host || !["ready", "partial"].includes(host.service.state)
      ? (host?.service.message ?? "Host unavailable")
      : workflow?.state !== "ready"
        ? "Image model files are unavailable on this host"
        : undefined;
  }
  const picker = (
    <div className="image-flow-header-controls">
      <ModelPicker
        groups={groups}
        personas={[]}
        privatePersonas={[]}
        isLoading={checking && !hosts.length}
        error={null}
        value={
          draft?.backend_id
            ? modelValue(draft.backend_id, draft.recipe.workflow)
            : ""
        }
        modelLabel={workflowLabel}
        modelUnavailable={unavailable}
        emptyLabel="No image models"
        ariaLabel="Image model"
        onChange={(value) => {
          const parts = modelParts(value);
          if (parts) chooseModel(parts.backendId, parts.modelName);
        }}
      />
      <button
        type="button"
        className="icon-button"
        aria-label="Image inference settings"
        title="Image inference settings"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        <Settings2 />
      </button>
      <button
        type="button"
        className="icon-button"
        title="Image tree"
        aria-label="Image tree"
        disabled={!tree.length}
        onClick={() => setTreeOpen(true)}
      >
        <GitBranch />
      </button>
    </div>
  );
  async function send(
    improveOnly = false,
    retry?: ImageResult,
    confirmed = false,
    restore?: Draft,
  ) {
    if (!draft || active) return;
    const value = structuredClone(restore ?? draft);
    if (!retry && !restore)
      value.parent_result_id = editing ? draft.parent_result_id : leaf;
    if (
      !improveOnly &&
      value.count === null &&
      value.seed_mode === "fixed" &&
      !confirmed
    ) {
      setConfirmEndless(true);
      return;
    }
    const id = await flow.submit(value, improveOnly, retry, confirmed);
    if (id) {
      setConfirmEndless(false);
      if (!improveOnly) {
        following.current = true;
        setUnseen(false);
        setExpanded(false);
      }
      if (!sessionId) onNavigate("flow", id);
    }
  }
  async function upload(files: FileList | null) {
    if (!draft || !files?.length) return;
    if (draft.input_ids.length + files.length > 10) {
      flow.setError("Choose at most ten reference images.");
      return;
    }
    const chosenFiles = [...files];
    await flow.guard(async () => {
      const ids: string[] = [];
      for (const file of chosenFiles) {
        const form = new FormData();
        form.append("image", file);
        const value = await requestMultipartJson<{ id: string }>(
          `${imagePath}/assets`,
          form,
          { method: "POST" },
        );
        ids.push(value.id);
      }
      const snapshot = flow.getDraft();
      if (!snapshot) return;
      const input_ids = [...snapshot.input_ids, ...ids];
      flow.patch({
        input_ids,
        original_input_ids: input_ids,
        parent_result_id: snapshot.input_ids.length
          ? snapshot.parent_result_id
          : null,
      });
      const edit = hosts
        .find((host) => host.backend.id === snapshot.backend_id)
        ?.service.capabilities?.images.workflows.find(
          (workflow) =>
            workflow.reference_images_max > 0 && workflow.state === "ready",
        );
      if (
        snapshot.recipe.workflow !== "qwen-image-2.1-edit/v1" &&
        edit &&
        snapshot.backend_id
      ) {
        flow.patch({
          backend_id: snapshot.backend_id,
          recipe: {
            ...preferences.edit.recipe,
            prompt: snapshot.recipe.prompt,
            negative_prompt: snapshot.recipe.negative_prompt,
            workflow: edit.id,
          },
          input_ids,
          original_input_ids: input_ids,
          parent_result_id: null,
        });
      }
      await flow.save();
    });
  }
  function editResult(result: ImageResult) {
    if (!draft) return;
    const original =
      runs.find((run) => run.id === result.run_id)?.request.draft.backend_id ??
      draft.backend_id;
    const host = hosts.find((host) => host.backend.id === original);
    const edit = host?.service.capabilities?.images.workflows.find(
      (workflow) =>
        workflow.reference_images_max > 0 && workflow.state === "ready",
    );
    if (!edit || !host) {
      flow.setError(
        "This host has no ready image edit model. Choose an edit model first.",
      );
      return;
    }
    flow.chooseLeaf(result.id);
    flow.setEditMode("iterative");
    flow.patch({
      backend_id: host.backend.id,
      recipe: { ...preferences.edit.recipe, workflow: edit.id, prompt: "" },
      input_ids: [result.asset_id],
      original_input_ids: [result.asset_id],
      parent_result_id: result.id,
    });
    setViewer(null);
    setTreeOpen(false);
    input.current?.focus();
  }
  function reuse(run: Run) {
    if (!draft) return;
    flow.chooseLeaf(run.request.draft.parent_result_id);
    flow.patch(structuredClone(run.request.draft));
    input.current?.focus();
    setTreeOpen(false);
  }
  const pendingVisible =
    active && turns.some((turn) => turn.pending?.id === active.id);
  const noModels = !groups.some((group) =>
    group.models.some((model) => !unavailable(group.backend.id, model.name)),
  );
  const unsupportedAttention = Boolean(
    draft &&
      selectedWorkflow?.state === "ready" &&
      !selectedWorkflow.attention.includes(draft.recipe.attention),
  );
  const notice =
    checking && !hosts.length
      ? "Checking inference hosts..."
      : noModels
        ? "No image models are available on your connected inference hosts."
        : draft && !selectedHost
          ? "Choose an image model."
          : draft && unavailable(draft.backend_id ?? "", draft.recipe.workflow)
            ? selectedHost?.service.message
            : unsupportedAttention
              ? "This host does not support the saved attention mode. Select Standard to continue."
              : draft && !ready
                ? selectedWorkflow?.reference_images_min
                  ? "Add a reference image for this edit model."
                  : draft.input_ids.length
                    ? "Choose an edit model to use reference images."
                    : "These settings are not supported by the selected model."
                : "";

  return (
    <div className="image-flow">
      {headerTarget ? (
        createPortal(picker, headerTarget)
      ) : (
        <header className="image-flow-local-header">{picker}</header>
      )}
      {flow.error && (
        <div className="images-error" role="alert">
          <span>{flow.error}</span>
          <button
            type="button"
            className="images-icon"
            aria-label="Dismiss error"
            onClick={() => flow.setError("")}
          >
            <X size={18} />
          </button>
        </div>
      )}
      {treeOpen ? (
        <MessageTreeExplorer
          title="Image Tree"
          messages={tree}
          activeRootMessageId={root?.id ?? tree[0]?.id ?? null}
          onClose={() => setTreeOpen(false)}
          onOpenBranch={(id) => {
            const result = results.find((value) => value.id === id);
            if (result) {
              void flow.select(result, continuationLeaf(results, result.id));
              setTreeOpen(false);
            } else {
              const run = runs.find((value) => `prompt:${value.id}` === id);
              if (run) {
                const selectedChild = tree.find(
                  (node) => node.id === id,
                )?.active_child_message_id;
                const response =
                  results.find((value) => value.id === selectedChild) ??
                  [...results]
                    .reverse()
                    .find(
                      (value) =>
                        value.origin_run_id === run.id && value.selected,
                    );
                if (response)
                  void flow.select(
                    response,
                    continuationLeaf(results, response.id),
                  );
                else flow.chooseLeaf(run.request.draft.parent_result_id);
                setTreeOpen(false);
              }
            }
          }}
          renderNodePreview={(message) => {
            const result = results.find((value) => value.id === message.id);
            return result ? (
              <ImageAsset id={result.asset_id} alt="Image branch" />
            ) : (
              <span>{message.active_revision?.content_text.slice(0, 120)}</span>
            );
          }}
          renderFocusedContent={(message) => {
            const result = results.find((value) => value.id === message.id);
            return result ? (
              <>
                <ImageAsset id={result.asset_id} />
                <RecipeDetails recipe={result.recipe} />
              </>
            ) : (
              <p className="images-result-prompt">
                {message.active_revision?.content_text}
              </p>
            );
          }}
        />
      ) : (
        <>
          <div
            className="image-flow-scroll"
            ref={scroll}
            onWheel={noteScrollIntent}
            onTouchMove={noteScrollIntent}
            onKeyDownCapture={noteScrollIntent}
            onPointerDown={(event) => {
              if (event.target === event.currentTarget) noteScrollIntent();
            }}
            onScroll={() => {
              const el = scroll.current;
              if (el && performance.now() <= scrollIntentUntil.current) {
                following.current =
                  el.scrollHeight - el.scrollTop - el.clientHeight < 160;
                if (following.current) setUnseen(false);
              }
            }}
          >
            <div className="image-flow-messages">
              {flow.more && (
                <button
                  type="button"
                  className="images-load-older"
                  disabled={busy}
                  onClick={() => void flow.loadOlder()}
                >
                  Load earlier steps
                </button>
              )}
              {!turns.length && (
                <div className="image-flow-empty">
                  <ImagePlus />
                  <h1>
                    {initialEditing || editing
                      ? "Edit an image"
                      : "Create an image"}
                  </h1>
                </div>
              )}
              {turns.map((turn) => (
                <section className="image-turn" key={turn.run.id}>
                  <article className="image-prompt-bubble message-bubble message-bubble-user">
                    <details className="image-prompt-details">
                      <summary>
                        <span>{turn.run.request.draft.recipe.prompt}</span>
                        <ChevronDown size={16} />
                      </summary>
                      <div>
                        <RecipeDetails recipe={turn.run.request.draft.recipe} />
                        <div className="images-inputs">
                          {turn.run.request.draft.input_ids.map((id, index) => (
                            <ImageAsset
                              key={`${id}:${index}`}
                              id={id}
                              alt={`Reference ${index + 1}`}
                            />
                          ))}
                        </div>
                      </div>
                    </details>
                    <div className="images-message-actions">
                      <button
                        type="button"
                        className="message-tool-button"
                        aria-label="Edit prompt"
                        title="Edit prompt"
                        onClick={() => reuse(turn.run)}
                      >
                        <Pencil size={18} />
                      </button>
                      <button
                        type="button"
                        className="message-tool-button"
                        aria-label="Copy prompt"
                        title="Copy prompt"
                        onClick={() =>
                          void flow.guard(() =>
                            navigator.clipboard.writeText(
                              turn.run.request.draft.recipe.prompt,
                            ),
                          )
                        }
                      >
                        <Copy size={18} />
                      </button>
                    </div>
                  </article>
                  {turn.responses.map((result) => {
                    const variants = [...results]
                        .reverse()
                        .filter((value) => value.group_id === result.group_id),
                      index = variants.findIndex(
                        (value) => value.id === result.id,
                      );
                    const hostId =
                      runs.find((value) => value.id === result.run_id)?.request
                        .draft.backend_id ?? draft?.backend_id;
                    const service = hosts.find(
                      (value) => value.backend.id === hostId,
                    )?.service;
                    return (
                      <article
                        className="image-response-bubble message-bubble message-bubble-assistant"
                        key={result.group_id}
                      >
                        <header>
                          <strong>
                            {workflowLabel(result.recipe.workflow)}
                          </strong>
                          {variants.length > 1 && (
                            <div className="images-variants">
                              <button
                                type="button"
                                aria-label="Previous image version"
                                title="Previous image version"
                                disabled={!index}
                                onClick={() =>
                                  void flow.select(
                                    variants[index - 1],
                                    continuationLeaf(
                                      results,
                                      variants[index - 1].id,
                                    ),
                                  )
                                }
                              >
                                <ArrowLeft size={16} />
                              </button>
                              <span>
                                {index + 1}/{variants.length}
                              </span>
                              <button
                                type="button"
                                aria-label="Next image version"
                                title="Next image version"
                                disabled={index === variants.length - 1}
                                onClick={() =>
                                  void flow.select(
                                    variants[index + 1],
                                    continuationLeaf(
                                      results,
                                      variants[index + 1].id,
                                    ),
                                  )
                                }
                              >
                                <ArrowRight size={16} />
                              </button>
                            </div>
                          )}
                        </header>
                        <SwipeSurface
                          onStep={(direction) => {
                            const next = variants[index + direction];
                            if (next)
                              void flow.select(
                                next,
                                continuationLeaf(results, next.id),
                              );
                          }}
                        >
                          <ImageAsset
                            id={result.asset_id}
                            neighbors={[
                              variants[index - 1]?.asset_id,
                              variants[index + 1]?.asset_id,
                            ].filter(Boolean)}
                            onClick={() => {
                              setViewer({
                                ids: variants.map((value) => value.asset_id),
                                index,
                              });
                            }}
                          />
                        </SwipeSurface>
                        <div className="images-message-actions">
                          <button
                            type="button"
                            className="message-tool-button"
                            title="Regenerate image"
                            aria-label="Regenerate image"
                            disabled={
                              busy ||
                              Boolean(active) ||
                              !canGenerate(
                                service,
                                result.recipe,
                                result.input_ids.length,
                              )
                            }
                            onClick={() => void send(false, result)}
                          >
                            <RotateCcw size={18} />
                          </button>
                          <button
                            type="button"
                            className="message-tool-button"
                            title="Edit image"
                            aria-label="Edit image"
                            onClick={() => editResult(result)}
                          >
                            <Pencil size={18} />
                          </button>
                          <button
                            type="button"
                            className="message-tool-button"
                            title="Download image"
                            aria-label="Download image"
                            onClick={() =>
                              void flow.guard(() =>
                                downloadImage(result.asset_id),
                              )
                            }
                          >
                            <Download size={18} />
                          </button>
                          <button
                            type="button"
                            className="message-tool-button"
                            title="Continue from this image"
                            aria-label="Continue from this image"
                            onClick={() => void flow.select(result, result.id)}
                          >
                            <GitBranch size={18} />
                          </button>
                        </div>
                        <details className="images-result-details">
                          <summary>
                            Generation details{" "}
                            <small>
                              {(result.duration_ms / 1000).toFixed(1)}s
                            </small>
                          </summary>
                          <RecipeDetails recipe={result.recipe} />
                        </details>
                      </article>
                    );
                  })}
                  {turn.pending && (
                    <article className="image-response-bubble image-pending-bubble message-bubble message-bubble-assistant">
                      <header>
                        <strong>
                          {workflowLabel(
                            turn.pending.request.draft.recipe.workflow,
                          )}
                        </strong>
                        <span role="status">
                          {turn.pending.status}
                          {turn.pending.completed_count
                            ? ` · ${turn.pending.completed_count} saved`
                            : ""}
                        </span>
                      </header>
                      {turn.pending.preview && (
                        <figure>
                          <img
                            src={`data:image/png;base64,${turn.pending.preview.png_base64}`}
                            alt="Image generation preview"
                            style={{
                              aspectRatio: `${turn.pending.request.draft.recipe.width} / ${turn.pending.request.draft.recipe.height}`,
                            }}
                          />
                          <figcaption>
                            Step {turn.pending.preview.step}/
                            {turn.pending.preview.steps}
                          </figcaption>
                        </figure>
                      )}
                      {turn.pending.error && (
                        <p role="alert">{turn.pending.error}</p>
                      )}
                      {!activeRun(turn.pending) &&
                        turn.pending.status !== "complete" && (
                          <button
                            type="button"
                            disabled={busy || Boolean(active)}
                            onClick={() =>
                              void send(
                                false,
                                results.find(
                                  (result) =>
                                    result.id ===
                                    turn.pending!.request.retry_result_id,
                                ),
                                false,
                                turn.pending!.request.draft,
                              )
                            }
                          >
                            <RotateCcw size={18} />
                            Try again
                          </button>
                        )}
                    </article>
                  )}
                </section>
              ))}
              <div ref={end} />
            </div>
          </div>
          {(flow.notice || (active && !pendingVisible) || unseen) && (
            <div className="image-flow-jump">
              <button
                type="button"
                onClick={() => {
                  const target =
                    flow.notice ??
                    (active ? active.request.draft.parent_result_id : leaf);
                  const result =
                    flow.notice && results.find((value) => value.id === target);
                  if (result) void flow.select(result, result.id);
                  else if (active) flow.followRun(active);
                  else flow.chooseLeaf(target);
                  flow.setNotice(null);
                  setUnseen(false);
                  following.current = true;
                  requestAnimationFrame(() =>
                    end.current?.scrollIntoView({ block: "end" }),
                  );
                }}
              >
                {flow.notice
                  ? "View completed image"
                  : active && !pendingVisible
                    ? "View generation"
                    : "Latest images"}
              </button>
              <button
                type="button"
                className="images-icon"
                aria-label="Dismiss image notification"
                onClick={() => {
                  flow.setNotice(null);
                  setUnseen(false);
                }}
              >
                <X size={16} />
              </button>
            </div>
          )}
          {draft && (
            <div
              className={`image-composer-dock${expanded ? " image-composer-dock-expanded" : ""}`}
            >
              {notice && (
                <div className="image-connection-notice" role="status">
                  <span>{notice}</span>
                  {unsupportedAttention &&
                    selectedWorkflow?.attention.includes("standard") && (
                      <button
                        type="button"
                        onClick={() => patchRecipe({ attention: "standard" })}
                      >
                        Use Standard
                      </button>
                    )}
                  <button
                    type="button"
                    className="message-tool-button"
                    disabled={checking}
                    title="Refresh image models"
                    aria-label="Refresh image models"
                    onClick={() => void onRefreshHosts()}
                  >
                    <RefreshCw size={18} />
                  </button>
                  {noModels && (
                    <button
                      type="button"
                      className="message-tool-button"
                      title="Inference host details"
                      aria-label="Inference host details"
                      onClick={onPreferencesOpen}
                    >
                      <Settings2 size={18} />
                    </button>
                  )}
                  {noModels && isAdmin && (
                    <button type="button" onClick={onOpenBackends}>
                      Backends
                    </button>
                  )}
                </div>
              )}
              <form
                className="image-chat-composer"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (ready && !busy && !active && draft.recipe.prompt.trim())
                    void send();
                }}
              >
                <button
                  type="button"
                  className="image-composer-expand"
                  title={
                    expanded
                      ? "Collapse inference settings"
                      : "Expand inference settings"
                  }
                  aria-label={
                    expanded
                      ? "Collapse inference settings"
                      : "Expand inference settings"
                  }
                  aria-expanded={expanded}
                  onClick={() => setExpanded((value) => !value)}
                >
                  {expanded ? (
                    <ChevronDown size={18} />
                  ) : (
                    <ChevronUp size={18} />
                  )}
                </button>
                <div className="image-composer-body">
                  {draft.input_ids.length > 0 && (
                    <div className="images-inputs">
                      {draft.input_ids.map((id, index) => (
                        <div key={`${id}:${index}`}>
                          <ImageAsset id={id} alt={`Reference ${index + 1}`} />
                          <div className="images-actions">
                            <button
                              type="button"
                              className="message-tool-button"
                              title={`Move reference ${index + 1} left`}
                              aria-label={`Move reference ${index + 1} left`}
                              disabled={index === 0 || busy}
                              onClick={() => {
                                const ids = [...draft.input_ids];
                                [ids[index], ids[index - 1]] = [
                                  ids[index - 1],
                                  ids[index],
                                ];
                                flow.patch({
                                  input_ids: ids,
                                  original_input_ids: ids,
                                  parent_result_id: null,
                                });
                              }}
                            >
                              <ArrowLeft size={14} />
                            </button>
                            <button
                              type="button"
                              className="message-tool-button"
                              title={`Remove reference ${index + 1}`}
                              aria-label={`Remove reference ${index + 1}`}
                              disabled={busy}
                              onClick={() => {
                                const ids = draft.input_ids.filter(
                                  (_, position) => position !== index,
                                );
                                flow.patch({
                                  input_ids: ids,
                                  original_input_ids: ids,
                                  parent_result_id:
                                    index === 0 ? null : draft.parent_result_id,
                                });
                              }}
                            >
                              <X size={14} />
                            </button>
                            <button
                              type="button"
                              className="message-tool-button"
                              title="Import generation settings"
                              aria-label="Import generation settings"
                              disabled={busy}
                              onClick={() =>
                                void flow.guard(async () => {
                                  const value = await requestJson<{
                                    recipe: Recipe;
                                    required_reference_images: number;
                                  }>(`${imagePath}/assets/${id}/settings`);
                                  flow.patch({ recipe: value.recipe });
                                  setExpanded(true);
                                  if (value.required_reference_images)
                                    flow.setError(
                                      `This recipe needs ${value.required_reference_images} original reference images in their original order.`,
                                    );
                                })
                              }
                            >
                              <Settings2 size={14} />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  <textarea
                    ref={input}
                    aria-label="Image prompt"
                    placeholder={
                      editing
                        ? "Describe the changes..."
                        : "Describe an image..."
                    }
                    rows={expanded ? 5 : 2}
                    value={draft.recipe.prompt}
                    onChange={(event) =>
                      patchRecipe({ prompt: event.target.value })
                    }
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" &&
                        !event.shiftKey &&
                        !event.altKey &&
                        !event.nativeEvent.isComposing &&
                        event.keyCode !== 229 &&
                        (event.ctrlKey ||
                          event.metaKey ||
                          !usesMobileInputBehavior())
                      ) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                  />
                  {expanded && (
                    <div className="image-expanded-controls">
                      {editing && (
                        <fieldset>
                          <legend>Edit source</legend>
                          <div className="images-segments">
                            <button
                              type="button"
                              aria-pressed={flow.editMode === "iterative"}
                              onClick={() => flow.setEditMode("iterative")}
                            >
                              Latest image
                            </button>
                            <button
                              type="button"
                              aria-pressed={flow.editMode === "original"}
                              onClick={() => flow.setEditMode("original")}
                            >
                              Original references
                            </button>
                          </div>
                        </fieldset>
                      )}
                      <GenerationControls
                        draft={draft}
                        hidden={preferences.hidden_controls}
                        workflows={workflows}
                        editing={editing}
                        patchRecipe={patchRecipe}
                        patchDraft={flow.patch}
                      />
                      <details>
                        <summary>Prompt improvement</summary>
                        <div className="images-model-picker">
                          <span>Assistant model</span>
                          <ModelPicker
                            groups={promptModels}
                            personas={[]}
                            privatePersonas={[]}
                            isLoading={false}
                            error={null}
                            value={
                              draft.prompt_help
                                ? modelValue(
                                    draft.prompt_help.backend_id,
                                    draft.prompt_help.model_name,
                                  )
                                : ""
                            }
                            ariaLabel="Prompt assistant model"
                            onChange={(value) => {
                              const parts = modelParts(value);
                              if (parts)
                                flow.patch({
                                  prompt_help: {
                                    backend_id: parts.backendId,
                                    model_name: parts.modelName,
                                    guidance: draft.prompt_help?.guidance ?? "",
                                  },
                                });
                            }}
                          />
                        </div>
                        <label>
                          Guidance
                          <textarea
                            rows={2}
                            value={draft.prompt_help?.guidance ?? ""}
                            disabled={!draft.prompt_help}
                            onChange={(event) => {
                              if (draft.prompt_help)
                                flow.patch({
                                  prompt_help: {
                                    ...draft.prompt_help,
                                    guidance: event.target.value,
                                  },
                                });
                            }}
                          />
                        </label>
                        <label className="images-check">
                          <input
                            type="checkbox"
                            checked={draft.auto_improve}
                            disabled={!draft.prompt_help}
                            onChange={(event) =>
                              flow.patch({ auto_improve: event.target.checked })
                            }
                          />
                          Improve each image's prompt
                        </label>
                        <button
                          type="button"
                          disabled={
                            busy ||
                            Boolean(active) ||
                            !draft.prompt_help ||
                            !draft.recipe.prompt.trim()
                          }
                          onClick={() => void send(true)}
                        >
                          <Sparkles size={18} />
                          Improve once
                        </button>
                      </details>
                      <div className="images-actions">
                        <button type="button" onClick={onPreferencesOpen}>
                          Preferences
                        </button>
                        <button
                          type="button"
                          disabled={busy}
                          onClick={() =>
                            void flow.guard(async () => {
                              const operation = editing ? "edit" : "generate";
                              const value = {
                                ...preferences,
                                [operation]: profileFromDraft(
                                  preferences[operation],
                                  draft,
                                  flow.editMode,
                                ),
                              };
                              onPreferences(
                                await writeImage<Preferences>(
                                  `${imagePath}/preferences`,
                                  value,
                                  "PUT",
                                ),
                              );
                              setDefaultsSaved(true);
                            })
                          }
                        >
                          Save as account defaults
                        </button>
                        <button
                          type="button"
                          disabled={busy}
                          onClick={() => {
                            const profile = editing
                              ? preferences.edit
                              : preferences.generate;
                            if (editing)
                              flow.setEditMode(profile.edit_mode ?? "iterative");
                            flow.patch(resetDraftSettings(flow.getDraft()!, profile));
                          }}
                        >
                          Reset to account defaults
                        </button>
                        <button
                          type="button"
                          disabled={busy}
                          onClick={() =>
                            void flow.guard(async () => {
                              const current = flow.getDraft();
                              if (!current) return;
                              const next = newFlowWithSettings(current);
                              await flow.save();
                              await onNewFlow(
                                next,
                                editing ? flow.editMode : "generate",
                              );
                            })
                          }
                        >
                          <CopyPlus size={18} />
                          New flow with these settings
                        </button>
                        {defaultsSaved && (
                          <span role="status">Account defaults saved.</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
                <footer>
                  <div className="images-actions">
                    <button
                      type="button"
                      className="icon-button primary-action"
                      title="Add reference images"
                      aria-label="Add reference images"
                      disabled={busy || draft.input_ids.length >= 10}
                      onClick={() => file.current?.click()}
                    >
                      <Paperclip />
                    </button>
                    <button
                      type="button"
                      className="message-tool-button"
                      title="Gallery"
                      aria-label="Open gallery"
                      onClick={() => onNavigate("gallery")}
                    >
                      <Images size={18} />
                    </button>
                    <small role="status">
                      {flow.saving
                        ? "Saving..."
                        : flow.session &&
                            JSON.stringify(draft) === flow.persisted
                          ? "Saved"
                          : ""}
                    </small>
                  </div>
                  {active ? (
                    <button
                      type="button"
                      className="icon-button primary-action"
                      title="Stop generation"
                      aria-label="Stop image generation"
                      disabled={active.status === "stopping"}
                      onClick={() => void flow.stop()}
                    >
                      <Square />
                    </button>
                  ) : (
                    <button
                      type="submit"
                      className="icon-button primary-action"
                      title="Send image prompt"
                      aria-label="Send image prompt"
                      disabled={busy || !ready || !draft.recipe.prompt.trim()}
                    >
                      <SendHorizontal />
                    </button>
                  )}
                </footer>
                <input
                  ref={file}
                  className="image-file-input"
                  type="file"
                  multiple
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) => {
                    void upload(event.target.files);
                    event.target.value = "";
                  }}
                />
              </form>
            </div>
          )}
        </>
      )}
      {confirmEndless && (
        <ImagePanel
          title="Repeat the same image?"
          onClose={() => setConfirmEndless(false)}
        >
          <p>
            Fixed seed with unchanged settings repeats the same image until
            stopped.
          </p>
          <button type="button" onClick={() => setConfirmEndless(false)}>
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => void send(false, undefined, true)}
          >
            Generate anyway
          </button>
        </ImagePanel>
      )}
      {suggestion && (
        <ImagePanel
          title="Suggested prompt"
          onClose={() => setSuggestion(null)}
        >
          <p className="images-result-prompt">{suggestion.effective_prompt}</p>
          {draft?.recipe.prompt !== suggestion.request.draft.recipe.prompt && (
            <p>
              Your draft has changed. Using this suggestion will replace its
              text.
            </p>
          )}
          <button
            type="button"
            onClick={() => {
              if (suggestion.effective_prompt) {
                patchRecipe({ prompt: suggestion.effective_prompt });
                flow.patch({ prompt_version_id: suggestion.prompt_version_id });
              }
              setSuggestion(null);
            }}
          >
            Use this prompt
          </button>
          <button type="button" onClick={() => setSuggestion(null)}>
            Keep my prompt
          </button>
        </ImagePanel>
      )}
      {viewer && (
        <ImagePanel
          title={`Image ${viewer.index + 1} of ${viewer.ids.length}`}
          className="images-viewer-panel"
          onClose={() => setViewer(null)}
        >
          <SwipeSurface onStep={moveViewer}>
            <ImageAsset
              id={viewer.ids[viewer.index]}
              neighbors={viewer.ids.filter(
                (_, index) => Math.abs(index - viewer.index) === 1,
              )}
            />
          </SwipeSurface>
          <div className="images-actions">
            <button
              type="button"
              className="images-icon"
              title="Previous image"
              aria-label="Previous image"
              disabled={!viewer.index}
              onClick={() => moveViewer(-1)}
            >
              <ArrowLeft />
            </button>
            <button
              type="button"
              className="images-icon"
              title="Next image"
              aria-label="Next image"
              disabled={viewer.index === viewer.ids.length - 1}
              onClick={() => moveViewer(1)}
            >
              <ArrowRight />
            </button>
            <button
              type="button"
              className="images-icon"
              title="Download image"
              aria-label="Download image"
              onClick={() =>
                void flow.guard(() => downloadImage(viewer.ids[viewer.index]))
              }
            >
              <Download />
            </button>
            <button
              type="button"
              className="images-icon"
              title="Copy to gallery"
              aria-label="Copy to gallery"
              onClick={() =>
                void flow.guard(async () => {
                  await writeImage(
                    `${imagePath}/gallery/copy/${viewer.ids[viewer.index]}`,
                    {},
                  );
                })
              }
            >
              <Copy />
            </button>
            {results.some(
              (result) => result.asset_id === viewer.ids[viewer.index],
            ) && (
              <button
                type="button"
                onClick={() =>
                  editResult(
                    results.find(
                      (result) => result.asset_id === viewer.ids[viewer.index],
                    )!,
                  )
                }
              >
                Edit image
              </button>
            )}
          </div>
        </ImagePanel>
      )}
    </div>
  );
}

function RecipeDetails({ recipe }: { recipe: Recipe }) {
  return (
    <div className="image-recipe-details">
      <p className="images-result-prompt">{recipe.prompt}</p>
      <dl>
        <div>
          <dt>Model</dt>
          <dd>{workflowLabel(recipe.workflow)}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>
            {recipe.width} × {recipe.height}
          </dd>
        </div>
        <div>
          <dt>Seed</dt>
          <dd>{recipe.seed}</dd>
        </div>
        <div>
          <dt>Steps</dt>
          <dd>{recipe.steps}</dd>
        </div>
        <div>
          <dt>CFG</dt>
          <dd>{recipe.cfg}</dd>
        </div>
        <div>
          <dt>Attention</dt>
          <dd>{recipe.attention}</dd>
        </div>
        {recipe.flow_shift != null && (
          <div>
            <dt>Flow shift</dt>
            <dd>{recipe.flow_shift}</dd>
          </div>
        )}
      </dl>
      {recipe.negative_prompt && (
        <>
          <strong>Negative prompt</strong>
          <p className="images-result-prompt">{recipe.negative_prompt}</p>
        </>
      )}
    </div>
  );
}
