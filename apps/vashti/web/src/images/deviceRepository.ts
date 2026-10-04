import { requestBlob, requestJson } from "../api";
import {
  deletePrivateImage,
  privateImageGuard,
  privateImageIds,
  readPrivateImage,
  writePrivateImage,
} from "../privateChatStore";
import { imageDataUrl } from "../saveImage";
import {
  activeRun,
  type Draft,
  type GalleryItem,
  type ImageResult,
  type ImageSession,
  type Preferences,
  type Recipe,
  type Run,
  type RunRequest,
} from "./types";
import { notifyImageFlows } from "./api";

type DeviceFlow = {
  session: ImageSession;
  runs: Run[];
  results: ImageResult[];
  hiddenGroups: string[];
};
type Asset = { data: string; width: number; height: number; recipe?: Recipe };
type Job = {
  session: string;
  cancel: AbortController;
  remote?: string;
  preview?: Run["preview"];
  status?: string;
};
const jobs = new Map<string, Job>();
const flowGuards = new WeakMap<DeviceFlow, () => void>();
let tail: Promise<void> = Promise.resolve();
const now = () => Math.floor(Date.now() / 1000);
const uuid = () => crypto.randomUUID();
async function serialized<T>(work: () => Promise<T>): Promise<T> {
  const guard = privateImageGuard();
  const task = tail
    .catch(() => {})
    .then(() => {
      guard();
      return work();
    });
  tail = task.then(
    () => {},
    () => {},
  );
  return task;
}
async function getFlow(id: string) {
  const guard = privateImageGuard();
  const flow = await readPrivateImage<DeviceFlow>(`flow:${id}`);
  guard();
  if (!flow) throw new Error("Device image flow was not found");
  flowGuards.set(flow, guard);
  // A reloaded WebView cannot resume a transient job. Keep completed images.
  for (const run of flow.runs)
    if (activeRun(run) && !jobs.has(run.id)) {
      run.status = "stopped";
      run.error = "Generation was interrupted. Regenerate to try again.";
    }
  for (const run of flow.runs) {
    const job = jobs.get(run.id);
    if (job && activeRun(run)) {
      run.preview = job.preview;
      run.status = job.status ?? run.status;
    }
  }
  return flow;
}
async function putFlow(flow: DeviceFlow) {
  flowGuards.get(flow)?.();
  const stored = structuredClone(flow);
  for (const run of stored.runs) run.preview = undefined;
  await writePrivateImage(`flow:${flow.session.id}`, stored);
}
function referencedAssets(flow: DeviceFlow) {
  return new Set([
    ...flow.session.draft.input_ids,
    ...flow.session.draft.original_input_ids,
    ...flow.results.flatMap((value) => [value.asset_id, ...value.input_ids]),
    ...flow.runs.flatMap((value) => [
      ...value.request.draft.input_ids,
      ...value.request.draft.original_input_ids,
    ]),
  ]);
}
async function collectAssets(candidates: Set<string>) {
  const guard = privateImageGuard();
  for (const flow of await flows())
    for (const id of referencedAssets(flow)) candidates.delete(id);
  for (const key of await privateImageIds("gallery:")) {
    const value = await readPrivateImage<GalleryItem>(key);
    if (value) candidates.delete(value.asset_id);
  }
  for (const id of candidates) {
    guard();
    await deletePrivateImage(`asset:${id}`);
  }
}
async function flows() {
  const result: DeviceFlow[] = [];
  for (const id of await privateImageIds("flow:"))
    result.push(await getFlow(id.slice(5)));
  return result.sort((a, b) => b.session.updated_at - a.session.updated_at);
}
async function asset(id: string) {
  const value = await readPrivateImage<Asset>(`asset:${id}`);
  if (!value) throw new Error("Device image was not found");
  return value;
}
function blobFromData(data: string) {
  const [header, body] = data.split(",");
  const mime = /^data:(image\/[^;]+);base64$/.exec(header)?.[1];
  if (!mime || !body) throw new Error("Device image could not be read");
  const bytes = Uint8Array.from(atob(body), (char) => char.charCodeAt(0));
  return new Blob([bytes], { type: mime });
}
export async function deviceImageBlob(id: string) {
  return blobFromData((await asset(id)).data);
}
export async function uploadDeviceImage(blob: Blob) {
  const guard = privateImageGuard();
  if (blob.size > 12 * 1024 * 1024)
    throw new Error("Choose an image smaller than 12 MiB");
  const pixels = await createImageBitmap(blob);
  try {
    if (pixels.width > 4096 || pixels.height > 4096)
      throw new Error("Choose an image up to 4096 pixels per side");
    // Match the server's reference normalization without a lossy re-encode.
    if (!["image/png", "image/jpeg", "image/webp"].includes(blob.type)) {
      const canvas = document.createElement("canvas");
      canvas.width = pixels.width;
      canvas.height = pixels.height;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Could not read image");
      context.drawImage(pixels, 0, 0);
      blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (value) =>
            value
              ? resolve(value)
              : reject(new Error("Could not convert image")),
          "image/png",
        ),
      );
    }
    if (blob.size > 12 * 1024 * 1024)
      throw new Error("Converted image is larger than 12 MiB");
    const value = {
      data: await imageDataUrl(blob),
      width: pixels.width,
      height: pixels.height,
    };
    guard();
    const id = uuid();
    await writePrivateImage(`asset:${id}`, value);
    return { id, width: value.width, height: value.height };
  } finally {
    pixels.close();
  }
}
function nextSeed(seed: string, mode: Draft["seed_mode"], offset: number) {
  if (mode === "random") {
    const words = crypto.getRandomValues(new Uint32Array(2));
    return (
      (BigInt(words[0] & 0x7fffffff) << 32n) |
      BigInt(words[1])
    ).toString();
  }
  const value = BigInt(seed || "0");
  const next = value + BigInt(mode === "increment" ? offset : 0);
  if (next < 0n || next > 9223372036854775807n)
    throw new Error("Seed range exhausted. Choose Random to regenerate.");
  return next.toString();
}
export async function stopDeviceImageFlow(session: string) {
  await serialized(async () => {
    const flow = await getFlow(session);
    for (const run of flow.runs.filter(activeRun)) {
      const job = jobs.get(run.id);
      job?.cancel.abort();
      run.status = "stopped";
      run.preview = undefined;
    }
    await putFlow(flow);
  });
}
async function execute(session: string, id: string, job: Job) {
  const guard = privateImageGuard();
  try {
    const original = (await getFlow(session)).runs.find(
      (run) => run.id === id,
    )!;
    const draft = structuredClone(original.request.draft);
    for (
      let index = 0;
      original.request.improve_only
        ? index < 1
        : draft.count === null || index < draft.count;
      index++
    ) {
      guard();
      if (job.cancel.signal.aborted) break;
      const references: string[] = [];
      for (const id of draft.input_ids)
        references.push((await asset(id)).data.split(",")[1]);
      const execution = structuredClone(draft);
      execution.recipe.seed = nextSeed(
        draft.recipe.seed,
        draft.seed_mode,
        index,
      );
      const started = performance.now();
      guard();
      const remote = await requestJson<{ id: string }>(
        "/api/images/private/flow",
        {
          method: "POST",
          signal: job.cancel.signal,
          body: JSON.stringify({
            draft: execution,
            reference_images: references,
            improve_only: original.request.improve_only,
          }),
        },
      );
      job.remote = remote.id;
      while (!job.cancel.signal.aborted) {
        guard();
        const status = await requestJson<{
          status: string;
          recipe: Recipe;
          preview?: Run["preview"];
        }>(`/api/images/private/${remote.id}`, { signal: job.cancel.signal });
        guard();
        if (status.status === "failed")
          throw new Error("Private image generation failed");
        if (status.status === "complete") {
          const output = original.request.improve_only
            ? null
            : await requestBlob(`/api/images/private/${remote.id}/result`, {
                signal: job.cancel.signal,
              });
          const data = output ? await imageDataUrl(output) : null;
          guard();
          await serialized(async () => {
            guard();
            const flow = await getFlow(session);
            const run = flow.runs.find((value) => value.id === id)!;
            if (!activeRun(run) || job.cancel.signal.aborted) return;
            run.preview = undefined;
            run.effective_prompt = status.recipe.prompt;
            if (data) {
              const assetId = uuid(),
                resultId = uuid();
              await writePrivateImage(`asset:${assetId}`, {
                data,
                width: status.recipe.width,
                height: status.recipe.height,
                recipe: status.recipe,
              });
              const retry = flow.results.find(
                (value) => value.id === original.request.retry_result_id,
              );
              const group = retry?.group_id ?? uuid();
              flow.results.forEach((value) => {
                if (value.group_id === group) value.selected = false;
              });
              const active = flow.session.active_result_id;
              const follow =
                !active ||
                active === draft.parent_result_id ||
                (retry &&
                  flow.results.some(
                    (value) => value.id === active && value.group_id === group,
                  ));
              flow.results.unshift({
                id: resultId,
                run_id: id,
                origin_run_id: retry
                  ? (retry.origin_run_id ?? retry.run_id)
                  : id,
                session_id: session,
                group_id: group,
                asset_id: assetId,
                parent_result_id: draft.parent_result_id,
                recipe: status.recipe,
                input_ids: [...draft.input_ids],
                duration_ms: Math.round(performance.now() - started),
                created_at: now(),
                selected: true,
              });
              run.completed_count++;
              job.preview = undefined;
              if (follow) flow.session.active_result_id = resultId;
            }
            flow.session.updated_at = now();
            await putFlow(flow);
          });
          break;
        }
        job.preview = status.preview;
        job.status = status.status;
        await new Promise<void>((resolve) => {
          const done = () => {
            clearTimeout(timer);
            job.cancel.signal.removeEventListener("abort", done);
            resolve();
          };
          const timer = setTimeout(done, 900);
          job.cancel.signal.addEventListener("abort", done, { once: true });
        });
      }
      guard();
      if (job.remote) {
        await requestJson(`/api/images/private/${job.remote}/stop`, {
          method: "POST",
          body: "{}",
        }).catch(() => {});
        job.remote = undefined;
      }
      notifyImageFlows();
    }
    await finish(job.cancel.signal.aborted ? "stopped" : "complete");
  } catch (error) {
    await finish(
      job.cancel.signal.aborted ? "stopped" : "failed",
      error instanceof Error ? error.message : "Generation failed",
    ).catch(() => {});
  } finally {
    try {
      guard();
      if (job.remote)
        await requestJson(`/api/images/private/${job.remote}/stop`, {
          method: "POST",
          body: "{}",
        }).catch(() => {});
    } catch {
      /* The original server expires uncollected jobs after disconnect. */
    }
    jobs.delete(id);
    notifyImageFlows();
  }
  async function finish(status: string, error: string | null = null) {
    guard();
    await serialized(async () => {
      guard();
      const flow = await getFlow(session);
      const run = flow.runs.find((value) => value.id === id)!;
      run.status = status;
      run.error = error;
      run.preview = undefined;
      await putFlow(flow);
    });
  }
}

export async function deviceImageRequest<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const guard = privateImageGuard();
  const parsed = new URL(url, "http://device.invalid");
  const parts = parsed.pathname
    .split("/")
    .filter(Boolean)
    .map(decodeURIComponent);
  const [resource, id, action] = parts;
  const method = options.method ?? "GET";
  const input = options.body ? JSON.parse(String(options.body)) : {};
  const offset = Math.max(0, Number(parsed.searchParams.get("offset")) || 0);
  const result = await serialized(async (): Promise<unknown> => {
    guard();
    if (resource === "preferences") {
      if (method === "PUT") {
        await writePrivateImage("preferences", input);
        return input;
      }
      return (
        (await readPrivateImage<Preferences>("preferences")) ??
        (await requestJson<Preferences>("/api/images/preferences"))
      );
    }
    if (resource === "assets" && action === "settings") {
      const value = await asset(id);
      return {
        width: value.width,
        height: value.height,
        recipe: value.recipe ?? null,
      };
    }
    if (resource === "sessions") {
      if (!id) {
        if (method === "GET")
          return (await flows())
            .map((value) => value.session)
            .slice(offset, offset + 50);
        const session: ImageSession = {
          id: uuid(),
          title: input.draft.recipe.prompt.trim().slice(0, 60) || "Image edit",
          mode: input.mode,
          draft: input.draft,
          revision: 1,
          updated_at: now(),
          active_result_id: null,
        };
        await putFlow({ session, runs: [], results: [], hiddenGroups: [] });
        return session;
      }
      const flow = await getFlow(id);
      if (action === "results") return flow.results.slice(offset, offset + 100);
      if (action === "runs" && method === "GET")
        return flow.runs.slice(offset, offset + 100);
      if (action === "runs" && method === "POST") {
        const request = input as RunRequest;
        const existing = flow.runs.find(
          (run) => run.request.client_request_id === request.client_request_id,
        );
        if (existing) return existing;
        if (flow.runs.some(activeRun))
          throw new Error("Stop the current generation first");
        const retry = flow.results.find(
          (value) => value.id === request.retry_result_id,
        );
        if (request.retry_result_id && !retry)
          throw new Error("Image version was not found");
        if (retry) {
          request.draft.recipe = structuredClone(retry.recipe);
          request.draft.input_ids = [...retry.input_ids];
          request.draft.parent_result_id = retry.parent_result_id;
          request.draft.backend_id = flow.runs.find(
            (value) => value.id === retry.run_id,
          )?.request.draft.backend_id;
          request.draft.auto_improve = false;
          request.draft.count = 1;
          request.improve_only = false;
          if (request.draft.seed_mode !== "random") {
            request.draft.seed_mode = "increment";
            const maximum = flow.results
              .filter((value) => value.group_id === retry.group_id)
              .reduce(
                (max, value) =>
                  BigInt(value.recipe.seed) > max
                    ? BigInt(value.recipe.seed)
                    : max,
                0n,
              );
            request.draft.recipe.seed = nextSeed(
              maximum.toString(),
              "increment",
              1,
            );
          }
        }
        if (
          !retry &&
          request.draft.recipe.workflow === "qwen-image-2.1-edit/v1"
        ) {
          if (flow.session.mode === "original")
            request.draft.input_ids = [...request.draft.original_input_ids];
          else if (request.draft.parent_result_id) {
            const parent = flow.results.find(
              (value) => value.id === request.draft.parent_result_id,
            );
            if (!parent || request.draft.input_ids[0] !== parent.asset_id)
              throw new Error(
                "The selected result must be the first image for this iterative edit",
              );
          }
          if (!request.draft.input_ids.length)
            throw new Error("Choose an image to edit");
        }
        if (
          request.draft.count === null &&
          request.draft.seed_mode === "fixed" &&
          !request.fixed_endless_confirmed &&
          !request.improve_only
        )
          throw new Error("Confirm endless generation with a fixed seed first");
        if (
          !request.draft.recipe.prompt.trim() &&
          (!request.draft.input_ids.length || request.improve_only)
        )
          throw new Error("Enter a prompt or choose an image to edit");
        // Validate every local reference before submitting anything to the server.
        for (const assetId of [
          ...request.draft.input_ids,
          ...request.draft.original_input_ids,
        ])
          await asset(assetId);
        if (
          request.draft.parent_result_id &&
          !flow.results.some(
            (value) => value.id === request.draft.parent_result_id,
          )
        )
          throw new Error("Image ancestry was not found");
        const run: Run = {
          id: uuid(),
          session_id: id,
          status: "queued",
          completed_count: 0,
          effective_prompt: null,
          prompt_version_id: null,
          error: null,
          request,
          created_at: now(),
        };
        flow.runs.unshift(run);
        if (!request.improve_only)
          flow.session.active_result_id = request.draft.parent_result_id;
        const job: Job = { session: id, cancel: new AbortController() };
        jobs.set(run.id, job);
        try {
          await putFlow(flow);
        } catch (error) {
          jobs.delete(run.id);
          throw error;
        }
        void execute(id, run.id, job).catch(() => {
          jobs.delete(run.id);
        });
        return run;
      }
      if (method === "DELETE") {
        for (const run of flow.runs) jobs.get(run.id)?.cancel.abort();
        guard();
        await deletePrivateImage(`flow:${id}`);
        guard();
        await collectAssets(referencedAssets(flow));
        return {};
      }
      if (action === "title")
        flow.session.title =
          String(input.title).trim().slice(0, 200) || "Image flow";
      else if (method === "PUT") {
        if (input.revision !== flow.session.revision)
          throw new Error(
            "Image settings changed in another window. Reopen the flow.",
          );
        flow.session.draft = input.draft;
        flow.session.mode = input.mode;
        flow.session.revision++;
      }
      if (method !== "GET") {
        flow.session.updated_at = now();
        await putFlow(flow);
      }
      return flow.session;
    }
    if (resource === "runs" || resource === "results") {
      const running = resource === "runs" ? jobs.get(id) : undefined;
      for (const flow of running
        ? [await getFlow(running.session)]
        : await flows()) {
        if (resource === "runs") {
          const run = flow.runs.find((value) => value.id === id);
          if (!run) continue;
          if (action === "stop") {
            jobs.get(id)?.cancel.abort();
            run.status = "stopped";
            run.preview = undefined;
            await putFlow(flow);
          }
          return run;
        }
        const selected = flow.results.find((value) => value.id === id);
        if (!selected) continue;
        let cursor: ImageResult | undefined = selected;
        const visited = new Set<string>();
        while (cursor && !visited.has(cursor.id)) {
          visited.add(cursor.id);
          for (const value of flow.results)
            if (value.group_id === cursor.group_id)
              value.selected = value.id === cursor.id;
          cursor = flow.results.find(
            (value) => value.id === cursor?.parent_result_id,
          );
        }
        flow.session.active_result_id = id;
        await putFlow(flow);
        return {};
      }
      throw new Error("Image version was not found");
    }
    if (resource === "gallery") {
      if (id === "copy") {
        const value = await asset(action),
          key = uuid();
        const item: GalleryItem = {
          id: key,
          asset_id: action,
          width: value.width,
          height: value.height,
          group_id: null,
          session_id: null,
          variants: 1,
          created_at: now(),
        };
        guard();
        await writePrivateImage(`gallery:${key}`, item);
        return item;
      }
      const items: GalleryItem[] = [];
      for (const flow of await flows()) {
        for (const value of flow.results.filter(
          (value) =>
            value.selected && !flow.hiddenGroups.includes(value.group_id),
        ))
          items.push({
            id: value.group_id,
            asset_id: value.asset_id,
            width: value.recipe.width,
            height: value.recipe.height,
            group_id: value.group_id,
            session_id: flow.session.id,
            variants: flow.results.filter(
              (other) => other.group_id === value.group_id,
            ).length,
            created_at: value.created_at,
          });
        if (
          id &&
          method === "DELETE" &&
          flow.results.some((value) => value.group_id === id)
        ) {
          flow.hiddenGroups.push(id);
          await putFlow(flow);
        }
      }
      for (const key of await privateImageIds("gallery:"))
        items.push((await readPrivateImage<GalleryItem>(key))!);
      if (id && method === "DELETE") {
        const removed = await readPrivateImage<GalleryItem>(`gallery:${id}`);
        guard();
        await deletePrivateImage(`gallery:${id}`);
        guard();
        if (removed) await collectAssets(new Set([removed.asset_id]));
        return {};
      }
      return items
        .sort((a, b) => b.created_at - a.created_at)
        .slice(offset, offset + 50);
    }
    throw new Error("Unsupported device image operation");
  });
  guard();
  return structuredClone(result) as T;
}
