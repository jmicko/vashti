import { requestBlob } from "../api";
import {
  deletePrivateImage,
  privateImageGuard,
  readPrivateImage,
  writePrivateImage,
} from "../privateChatStore";
import { imageDataUrl } from "../saveImage";
import { deviceImagePath } from "./storage";
import { imageBlobFromData, imageThumbnail } from "./imageBytes";

let cacheWrites = Promise.resolve();
function cacheImage(id: string, blob: Blob, data?: string, thumbnail = false) {
  const guard = privateImageGuard();
  const indexKey = thumbnail ? "thumbnail-cache-index" : "cache-index";
  const key = (id: string) => thumbnail ? `thumbnail:server:${id}` : `cache:${id}`;
  cacheWrites = cacheWrites
    .catch(() => {})
    .then(async () => {
      guard();
      try {
        const current =
          (await readPrivateImage<{ id: string; bytes: number }[]>(
            indexKey,
          )) ?? [];
        const index = current.filter((value) => value.id !== id);
        if (!data || !current.some((value) => value.id === id)) {
          data ??= await imageDataUrl(blob);
          guard();
          await writePrivateImage(key(id), data);
        }
        index.unshift({ id, bytes: blob.size });
        while (
          index.length > (thumbnail ? 512 : 10) ||
          index.reduce((sum, value) => sum + value.bytes, 0) > (thumbnail ? 64 : 128) * 1024 * 1024
        ) {
          const removed = index.pop()!;
          guard();
          await deletePrivateImage(key(removed.id));
        }
        guard();
        await writePrivateImage(indexKey, index);
      } catch {
        // Do not retain an unindexed blob if quota or an index write fails.
        guard();
        await deletePrivateImage(key(id)).catch(() => {});
      }
    })
    .catch(() => {});
}
export async function loadImageBlob(path: string, id: string) {
  const guard = privateImageGuard();
  if (path === deviceImagePath) {
    const { deviceImageBlob } = await import("./deviceRepository");
    const blob = await deviceImageBlob(id);
    guard();
    return blob;
  }
  const key = `cache:${id}`;
  let data: string | null = null;
  try {
    data = await readPrivateImage<string>(key);
  } catch {
    guard();
  }
  if (data) {
    const mime = /^data:(image\/[^;]+);base64,/.exec(data)?.[1];
    if (mime) {
      try {
        const blob = imageBlobFromData(data);
        guard();
        cacheImage(id, blob, data);
        return blob;
      } catch {
        guard();
      }
    }
  }
  const blob = await requestBlob(`${path}/assets/${encodeURIComponent(id)}`);
  guard();
  // Original bytes are encrypted on disk; decoded pixels use a bounded RAM cache.
  cacheImage(id, blob);
  return blob;
}

export async function storeDeviceThumbnail(id: string, blob: Blob) {
  const guard = privateImageGuard();
  try {
    const preview = await imageThumbnail(blob);
    const data = await imageDataUrl(preview);
    guard();
    await writePrivateImage(`thumbnail:device:${id}`, data);
  } catch {
    // The original remains usable when a derived preview cannot be stored.
    guard();
  }
}

export async function loadImageThumbnail(path: string, id: string) {
  const guard = privateImageGuard();
  const device = path === deviceImagePath;
  const key = `thumbnail:${device ? "device" : "server"}:${id}`;
  try {
    const data = await readPrivateImage<string>(key);
    guard();
    if (data) {
      const blob = imageBlobFromData(data);
      if (!device) cacheImage(id, blob, data, true);
      return blob;
    }
  } catch {
    guard();
  }
  let preview: Blob;
  if (device) {
    preview = await imageThumbnail(await loadImageBlob(path, id));
    const data = await imageDataUrl(preview);
    guard();
    await writePrivateImage(key, data).catch(() => {});
  } else {
    try {
      preview = await requestBlob(`${path}/assets/${encodeURIComponent(id)}/thumbnail`);
    } catch (error) {
      guard();
      // Older servers do not expose previews yet. Cache the derived result once.
      if (!(error instanceof Error) || !("status" in error) || error.status !== 404) throw error;
      preview = await imageThumbnail(await loadImageBlob(path, id));
    }
    guard();
    cacheImage(id, preview, undefined, true);
  }
  guard();
  return preview;
}
