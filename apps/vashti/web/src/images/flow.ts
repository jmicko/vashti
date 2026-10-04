import type { BackendModelGroup, ChatMessage, ModelInfo } from "../types";
import {
  activeRun,
  imageLineage,
  workflowLabel,
  type ImageHost,
  type ImageResult,
  type Run,
  type Draft,
  type RunRequest,
} from "./types";

export function clearPromptAfterSend(request: RunRequest, mode: string, currentPrompt: string | undefined): boolean {
  return !request.improve_only && !request.retry_result_id && mode === "iterative" &&
    request.draft.recipe.workflow === "qwen-image-2.1-edit/v1" &&
    currentPrompt === request.draft.recipe.prompt;
}

export function advancedImageSeed(
  run: Run,
  previousCount: number,
  current: Draft,
  seedAtDispatch?: string,
): string | null {
  const submitted = run.request.draft;
  if (
    submitted.seed_mode !== "increment" ||
    current.recipe.workflow !== submitted.recipe.workflow ||
    current.backend_id !== submitted.backend_id
  )
    return null;
  const before = BigInt(submitted.recipe.seed) + BigInt(previousCount);
  if (
    current.recipe.seed !== before.toString() &&
    !(previousCount === 0 && current.recipe.seed === seedAtDispatch)
  )
    return null;
  const next = BigInt(submitted.recipe.seed) + BigInt(run.completed_count);
  return next <= 9223372036854775807n ? next.toString() : null;
}

export function imageModelGroups(hosts: ImageHost[]): BackendModelGroup[] {
  return hosts.map((host) => ({
    backend: host.backend,
    models: (host.service.capabilities?.images.workflows ?? []).map(
      (workflow) =>
        ({
          name: workflow.id,
          supports_images: workflow.reference_images_max > 0,
          capabilities: [],
          avatar_asset_id: null,
          avatar_crop_x: 0.5,
          avatar_crop_y: 0.5,
          avatar_crop_size: 1,
          background_asset_id: null,
          background_dim: 0.65,
          background_message_dim: 0.7,
          background_landscape_mode: "fill",
          background_landscape_x: 0.5,
          background_landscape_y: 0.5,
          background_landscape_scale: 1,
          background_portrait_mode: "fill",
          background_portrait_x: 0.5,
          background_portrait_y: 0.5,
          background_portrait_scale: 1,
        }) satisfies ModelInfo,
    ),
  }));
}

export function groupOrigin(
  results: ImageResult[],
  groupId: string,
): ImageResult | undefined {
  return [...results].reverse().find((result) => result.group_id === groupId);
}
function originRuns(results: ImageResult[]): Map<string, string> {
  const groups = new Map<string, string>();
  for (const result of [...results].reverse()) {
    if (!groups.has(result.group_id) || result.origin_run_id)
      groups.set(result.group_id, result.origin_run_id ?? result.run_id);
  }
  return groups;
}

export function continuationLeaf(
  results: ImageResult[],
  resultId: string,
): string {
  const visited = new Set<string>();
  let leaf = resultId;
  while (!visited.has(leaf)) {
    visited.add(leaf);
    const child = results.find(
      (result) => result.parent_result_id === leaf && result.selected,
    );
    if (!child) break;
    leaf = child.id;
  }
  return leaf;
}

export function hasImageAncestry(
  runs: Run[],
  results: ImageResult[],
  leaf: string | null,
): boolean {
  const byId = new Map(results.map((result) => [result.id, result]));
  const runIds = new Set(runs.map((run) => run.id));
  const visited = new Set<string>();
  let cursor = leaf;
  while (cursor && !visited.has(cursor)) {
    visited.add(cursor);
    const result = byId.get(cursor);
    if (!result || !runIds.has(result.origin_run_id ?? result.run_id))
      return false;
    cursor = result.parent_result_id;
  }
  return true;
}

export type ImageTurn = { run: Run; responses: ImageResult[]; pending?: Run };
export function imageTurns(
  runs: Run[],
  results: ImageResult[],
  leaf: string | null,
): ImageTurn[] {
  const path = imageLineage(results, leaf);
  const pathIds = new Set(path.map((result) => result.id));
  const groups = originRuns(results);
  const origins = new Set(path.map((result) => groups.get(result.group_id)));
  const pathGroups = new Set(path.map((result) => result.group_id));
  const byId = new Map(results.map((result) => [result.id, result]));
  const turns: ImageTurn[] = [];
  for (const run of [...runs].reverse()) {
    if (run.request.improve_only || run.request.retry_result_id) continue;
    const responses = [...results]
      .reverse()
      .filter(
        (result) =>
          groups.get(result.group_id) === run.id &&
          (pathIds.has(result.id) ||
            (!pathGroups.has(result.group_id) && result.selected)),
      );
    const parent = run.request.draft.parent_result_id;
    if (origins.has(run.id) || (!responses.length && parent === leaf)) {
      const retry = runs.find(
        (candidate) =>
          candidate.request.retry_result_id &&
          responses.some(
            (result) =>
              byId.get(candidate.request.retry_result_id!)?.group_id ===
              result.group_id,
          ),
      );
      turns.push({
        run,
        responses,
        pending:
          retry && (activeRun(retry) || retry.status !== "complete")
            ? retry
            : !responses.length || activeRun(run)
              ? run
              : undefined,
      });
    }
  }
  return turns;
}

// A view-only adapter lets image flows reuse chat's pan/zoom/tree navigation.
// These objects are never sent to the chat API or persisted as LLM messages.
export function imageTreeMessages(
  runs: Run[],
  results: ImageResult[],
  leaf: string | null,
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  const groups = originRuns(results);
  function node(
    id: string,
    parent: string | null,
    role: string,
    content: string,
    status: string,
    model: string | null,
    at: number,
  ): ChatMessage {
    const revision = {
      id,
      content_text: content,
      thinking_text: "",
      source: "written",
      created_at: at,
    };
    return {
      id,
      parent_message_id: parent,
      active_child_message_id: null,
      active_revision_id: id,
      role,
      status,
      is_deleted: false,
      backend_id: null,
      model_name: model,
      think_mode: null,
      done_reason: null,
      error_text: null,
      started_at: at,
      completed_at: null,
      created_at: at,
      updated_at: at,
      active_revision: revision,
      revisions: [revision],
      revision_count: 1,
      attachments: [],
      context_blocks: [],
    };
  }
  for (const run of [...runs].reverse()) {
    if (run.request.improve_only || run.request.retry_result_id) continue;
    messages.push(
      node(
        `prompt:${run.id}`,
        run.request.draft.parent_result_id,
        "user",
        run.request.draft.recipe.prompt,
        run.status,
        null,
        run.created_at,
      ),
    );
  }
  const promptIds = new Set(messages.map((message) => message.id));
  for (const result of [...results].reverse()) {
    const promptId = `prompt:${groups.get(result.group_id) ?? result.run_id}`;
    messages.push(
      node(
        result.id,
        promptIds.has(promptId) ? promptId : result.parent_result_id,
        "assistant",
        result.recipe.prompt,
        "complete",
        workflowLabel(result.recipe.workflow),
        result.created_at,
      ),
    );
  }
  const selected = imageLineage(results, leaf);
  const pathIds = selected.flatMap((result) => [
    `prompt:${groups.get(result.group_id) ?? result.run_id}`,
    result.id,
  ]);
  const byId = new Map(messages.map((message) => [message.id, message]));
  for (let index = 0; index < pathIds.length - 1; index++) {
    const message = byId.get(pathIds[index]);
    if (message) message.active_child_message_id = pathIds[index + 1];
  }
  return messages;
}
