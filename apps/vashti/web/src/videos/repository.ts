import { requestBlob, requestJson, requestMultipartJson } from "../api";
import {
  readPrivateImage,
  writePrivateImage,
  privateImageIds,
  privateImageGuard,
  deletePrivateImage,
  privateProfileIdentity,
} from "../privateChatStore";
import { imageDataUrl } from "../saveImage";
import {
  videoDefaults,
  videoActive,
  VIDEO_I2V,
  wireVideoDraft,
  type VideoDraft,
  type VideoPage,
  type VideoSession,
  type VideoRun,
  type VideoStatus,
} from "./types";

export type VideoStorage = "server" | "device";
const mutations = new Map<string, Promise<unknown>>();
export function serializeVideoMutation<T>(
  key: string,
  work: () => Promise<T>,
): Promise<T> {
  const next = (mutations.get(key) ?? Promise.resolve())
    .catch(() => undefined)
    .then(work);
  mutations.set(key, next);
  void next
    .finally(() => {
      if (mutations.get(key) === next) mutations.delete(key);
    })
    .catch(() => undefined);
  return next;
}
export function blobFromData(data: string): Blob {
  const comma = data.indexOf(","),
    binary = atob(data.slice(comma + 1));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: data.slice(5, comma).split(";")[0] });
}
export function notifyVideos() {
  window.dispatchEvent(new Event("vashti:image-flows-changed"));
}
export function videoRepository(mode: VideoStorage) {
  const guard = privateImageGuard();
  const thumbnails = new Map<string, Promise<Blob>>();
  const get = async <T>(key: string) => {
    guard();
    const result = await readPrivateImage<T>(`video:${key}`);
    guard();
    return result;
  };
  const put = async (key: string, value: unknown) => {
    guard();
    await writePrivateImage(`video:${key}`, value);
    guard();
  };
  const mutate = <T>(
    id: string,
    work: (page: VideoPage & { archived?: boolean }) => Promise<T>,
  ) =>
    serializeVideoMutation(`video:${id}`, async () => {
      const page = await get<VideoPage & { archived?: boolean }>(
        `session:${id}`,
      );
      if (!page) throw new Error("Video flow not found on this device");
      const result = await work(page);
      await put(`session:${id}`, page);
      return result;
    });
  return {
    mode,
    guard,
    async preferences() {
      return mode === "server"
        ? requestJson<VideoDraft>("/api/videos/preferences")
        : ((await get<VideoDraft>("preferences")) ?? { ...videoDefaults });
    },
    async defaults(draft: VideoDraft) {
      const value = {
        ...draft,
        prompt: "",
        first_frame_id: null,
        last_frame_id: null,
        reference_image_ids: [],
      };
      if (mode === "server")
        await requestJson("/api/videos/preferences", {
          method: "PUT",
          body: JSON.stringify(wireVideoDraft(value)),
        });
      else await put("preferences", value);
      return value;
    },
    async list(q = "", offset = 0) {
      if (mode === "server")
        return requestJson<VideoSession[]>(
          `/api/videos/sessions?${new URLSearchParams({ q, offset: String(offset) })}`,
        );
      const ids = await privateImageIds("video:session:");
      guard();
      const values = await Promise.all(
        ids.map((id) =>
          readPrivateImage<VideoPage & { archived?: boolean }>(id),
        ),
      );
      guard();
      return values
        .filter((v): v is VideoPage =>
          Boolean(
            v &&
            !v.archived &&
            v.session.title.toLowerCase().includes(q.toLowerCase()),
          ),
        )
        .map((v) => v.session)
        .sort((a, b) => b.updated_at - a.updated_at)
        .slice(offset, offset + 50);
    },
    async create(draft: VideoDraft) {
      if (mode === "server")
        return requestJson<VideoSession>("/api/videos/sessions", {
          method: "POST",
          body: JSON.stringify({ draft: wireVideoDraft(draft) }),
        });
      const session = {
        id: crypto.randomUUID(),
        title: "New video",
        draft_json: JSON.stringify(draft),
        revision: 1,
        updated_at: Date.now() / 1000,
      };
      await put(`session:${session.id}`, { session, runs: [] });
      notifyVideos();
      return session;
    },
    async page(id: string) {
      if (mode === "server")
        return requestJson<VideoPage>(`/api/videos/sessions/${id}`);
      const value = await get<VideoPage>(`session:${id}`);
      if (!value) throw new Error("Video flow not found on this device");
      return value;
    },
    async save(session: VideoSession, draft: VideoDraft) {
      if (mode === "server")
        return requestJson<VideoSession>(`/api/videos/sessions/${session.id}`, {
          method: "PUT",
          body: JSON.stringify({
            draft: wireVideoDraft(draft),
            revision: session.revision,
          }),
        });
      return mutate(session.id, async (page) => {
        if (page.session.revision !== session.revision)
          throw new Error(
            "This flow changed elsewhere. Reopen it before saving.",
          );
        page.session = {
          ...page.session,
          draft_json: JSON.stringify(draft),
          revision: session.revision + 1,
          updated_at: Date.now() / 1000,
        };
        return page.session;
      });
    },
    async rename(id: string, title: string) {
      if (mode === "server")
        await requestJson(`/api/videos/sessions/${id}/title`, {
          method: "PATCH",
          body: JSON.stringify({ title }),
        });
      else {
        await mutate(id, async (page) => {
          page.session.title = title;
        });
      }
      notifyVideos();
    },
    async remove(id: string) {
      if (mode === "server")
        await requestJson(`/api/videos/sessions/${id}`, { method: "DELETE" });
      else {
        await serializeVideoMutation(`video:${id}`, async () => {
          const page = await get<VideoPage>(`session:${id}`);
          if (!page) return;
          if (page.runs.some((r) => videoActive(r.status)))
            throw new Error("Stop the active video first");
          for (const run of page.runs) {
            guard();
            await deletePrivateImage(`video:clip:${run.id}`);
          }
          guard();
          await deletePrivateImage(`video:session:${id}`);
        });
      }
      await serializeVideoMutation(`video:composer:${mode}:${id}`, async () => {
        guard();
        await deletePrivateImage(`video:composer:${mode}:${id}`);
      });
      notifyVideos();
    },
    async frame(file: File) {
      guard();
      if (file.size > 12 * 1024 * 1024)
        throw new Error("Input images must be at most 12 MiB");
      const bitmap = await createImageBitmap(file);
      let blob: Blob = file;
      try {
        guard();
        if (bitmap.width > 4096 || bitmap.height > 4096)
          throw new Error("Input images must be at most 4096 pixels per side");
        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
          const canvas = document.createElement("canvas");
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
          blob = await new Promise<Blob>((resolve, reject) =>
            canvas.toBlob(
              (v) =>
                v ? resolve(v) : reject(new Error("Could not convert image")),
              "image/png",
            ),
          );
        }
        if (blob.size > 12 * 1024 * 1024)
          throw new Error("Converted input image exceeds 12 MiB");
        guard();
        if (mode === "server") {
          const form = new FormData();
          form.append("image", blob, "first-frame");
          return await requestMultipartJson<{
            id: string;
            width: number;
            height: number;
          }>("/api/videos/frames", form, { method: "POST" });
        }
        const id = crypto.randomUUID();
        const canvas = document.createElement("canvas");
        const scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height));
        canvas.width = Math.max(1, Math.round(bitmap.width * scale));
        canvas.height = Math.max(1, Math.round(bitmap.height * scale));
        canvas
          .getContext("2d")!
          .drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        const thumb = canvas.toDataURL("image/png");
        const data = await imageDataUrl(blob);
        await put(`frame:${id}`, {
          data,
          thumb,
          width: bitmap.width,
          height: bitmap.height,
        });
        return { id, width: bitmap.width, height: bitmap.height };
      } finally {
        bitmap.close();
      }
    },
    async frameBlob(id: string, thumbnail = true) {
      guard();
      const load = async () => {
        if (mode === "server")
          return requestBlob(`/api/videos/frames/${id}?thumbnail=${thumbnail}`);
        const frame = await get<{ data: string; thumb: string }>(`frame:${id}`);
        if (!frame) throw new Error("The input image is no longer available");
        return blobFromData(thumbnail ? frame.thumb : frame.data);
      };
      let pending = thumbnail ? thumbnails.get(id) : undefined;
      if (!pending) {
        pending = load();
        if (thumbnail) {
          thumbnails.set(id, pending);
          if (thumbnails.size > 16)
            thumbnails.delete(thumbnails.keys().next().value!);
          const current = pending;
          void pending.catch(() => {
            if (thumbnails.get(id) === current) thumbnails.delete(id);
          });
        }
      }
      const value = await pending;
      guard();
      return value;
    },
    async submit(
      session: VideoSession,
      draft: VideoDraft,
      request_id: string,
      retry_of?: string,
    ) {
      guard();
      const identity = mode === "device" ? privateProfileIdentity() : null;
      if (mode === "server")
        return requestJson<{ id: string }>(
          `/api/videos/sessions/${session.id}/runs`,
          {
            method: "POST",
            body: JSON.stringify({
              request_id,
              draft: wireVideoDraft(draft),
              retry_of: retry_of ?? null,
            }),
          },
        );
      const loadInput = async (id: string | null | undefined) => {
        if (!id) return null;
        const frame = await get<{ data: string }>(`frame:${id}`);
        if (!frame)
          throw new Error(
            "An input image is no longer available on this device",
          );
        return frame.data.split(",")[1];
      };
      const first_frame = await loadInput(draft.first_frame_id);
      const last_frame = await loadInput(draft.last_frame_id);
      const reference_images: string[] = [];
      for (const id of draft.reference_image_ids ?? [])
        reference_images.push((await loadInput(id))!);
      if (
        (first_frame?.length ?? 0) +
          (last_frame?.length ?? 0) +
          reference_images.reduce((total, image) => total + image.length, 0) +
          new TextEncoder().encode(draft.prompt).length +
          8192 >
        64 * 1024 * 1024
      )
        throw new Error(
          "The input images are too large together. Use smaller images.",
        );
      guard();
      const complete = await mutate(session.id, async (page) => {
        const existing = page.runs.find((r) => r.id === request_id);
        if (existing && existing.draft_json !== JSON.stringify(draft))
          throw new Error("This request ID was already used for another video");
        if (existing?.status === "complete") return true;
        if (existing) {
          existing.status = "queued";
          existing.error = null;
        } else {
          const run: VideoRun = {
            id: request_id,
            session_id: session.id,
            group_id: retry_of
              ? (page.runs.find((r) => r.id === retry_of)?.group_id ??
                request_id)
              : request_id,
            draft_json: JSON.stringify(draft),
            status: "queued",
            error: null,
            size_bytes: 0,
            duration_ms: 0,
            created_at: Date.now() / 1000,
          };
          page.runs.push(run);
          page.session = {
            ...page.session,
            draft_json: JSON.stringify(draft),
            title:
              page.session.title === "New video"
                ? draft.prompt.slice(0, 80) || "Animated image"
                : page.session.title,
            revision: page.session.revision + 1,
          };
        }
        return false;
      });
      if (complete) return { id: request_id };
      guard();
      const result = await requestJson<{ id: string }>("/api/videos/private", {
        method: "POST",
        expectedServerInstanceId: identity!.instance_id,
        body: JSON.stringify({
          ...(draft.workflow === VIDEO_I2V
            ? {}
            : {
                expected_user_id: identity!.user_id,
                last_frame,
                reference_images,
              }),
          request: { request_id, draft: wireVideoDraft(draft), retry_of: null },
          first_frame,
        }),
      });
      notifyVideos();
      return result;
    },
    async status(id: string) {
      guard();
      return requestJson<VideoStatus>(`/api/videos/runs/${id}`);
    },
    async stop(id: string) {
      guard();
      await requestJson(`/api/videos/runs/${id}/stop`, {
        method: "POST",
        body: "{}",
      });
    },
    async finish(sessionId: string, id: string, status: VideoStatus) {
      if (mode === "server") return;
      let size = 0;
      if (status.status === "complete") {
        const blob = await requestBlob(`/api/videos/runs/${id}/result`);
        guard();
        await put(`clip:${id}`, { data: await imageDataUrl(blob) });
        size = blob.size;
      }
      await mutate(sessionId, async (page) => {
        const run = page.runs.find((r) => r.id === id);
        if (run) {
          run.status = status.status;
          run.error = status.progress.error ?? null;
          run.size_bytes = size;
        }
      });
      notifyVideos();
    },
    async result(id: string, signal?: AbortSignal) {
      guard();
      if (mode === "server")
        return requestBlob(`/api/videos/runs/${id}/result`, { signal });
      const clip = await get<{ data: string }>(`clip:${id}`);
      if (!clip) throw new Error("The clip is not saved on this device");
      return blobFromData(clip.data);
    },
  };
}
