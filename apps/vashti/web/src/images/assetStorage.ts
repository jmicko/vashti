import { requestBlob } from "../api";
import {
  deletePrivateImage,
  privateImageGuard,
  readPrivateImage,
  writePrivateImage,
} from "../privateChatStore";
import { imageDataUrl } from "../saveImage";
import { deviceImagePath } from "./storage";

let cacheWrites = Promise.resolve();
function cacheImage(id: string, blob: Blob, data?: string) {
  const guard = privateImageGuard();
  cacheWrites = cacheWrites
    .catch(() => {})
    .then(async () => {
      guard();
      try {
        const current =
          (await readPrivateImage<{ id: string; bytes: number }[]>(
            "cache-index",
          )) ?? [];
        const index = current.filter((value) => value.id !== id);
        if (!data || !current.some((value) => value.id === id)) {
          data ??= await imageDataUrl(blob);
          guard();
          await writePrivateImage(`cache:${id}`, data);
        }
        index.unshift({ id, bytes: blob.size });
        while (
          index.length > 10 ||
          index.reduce((sum, value) => sum + value.bytes, 0) > 128 * 1024 * 1024
        ) {
          const removed = index.pop()!;
          guard();
          await deletePrivateImage(`cache:${removed.id}`);
        }
        guard();
        await writePrivateImage("cache-index", index);
      } catch {
        // Do not retain an unindexed blob if quota or an index write fails.
        guard();
        await deletePrivateImage(`cache:${id}`).catch(() => {});
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
        const blob = new Blob(
          [
            Uint8Array.from(atob(data.split(",")[1]), (value) =>
              value.charCodeAt(0),
            ),
          ],
          { type: mime },
        );
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
