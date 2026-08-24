import { useCallback, useSyncExternalStore } from "react";
import type { ImageDimensions } from "./avatarCrop";

type MediaKind = "avatar" | "background";

type MediaEntry = {
  kind: MediaKind;
  image: HTMLImageElement;
  snapshot: MediaSnapshot;
  listeners: Set<() => void>;
};

type MediaSnapshot = {
  dimensions: ImageDimensions | null;
  ready: boolean;
  failed: boolean;
};

const emptySnapshot: MediaSnapshot = {
  dimensions: null,
  ready: false,
  failed: false
};

const mediaCache = new Map<string, MediaEntry>();
const cacheLimits: Record<MediaKind, number> = {
  avatar: 64,
  background: 10
};

export function clearDecodedModelMediaCache() {
  mediaCache.clear();
}

export function useDecodedModelMedia(src: string | null, kind: MediaKind) {
  const entry = src ? ensureMediaEntry(src, kind) : null;
  const subscribe = useCallback(
    (listener: () => void) => {
      if (!entry) return () => undefined;
      entry.listeners.add(listener);
      return () => {
        entry.listeners.delete(listener);
        trimMediaCache(kind);
      };
    },
    [entry]
  );
  const getSnapshot = useCallback(() => entry?.snapshot ?? emptySnapshot, [entry]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function warmDecodedModelMedia(src: string, kind: MediaKind): Promise<void> {
  const entry = ensureMediaEntry(src, kind);
  if (entry.snapshot.ready || entry.snapshot.failed) return Promise.resolve();

  return new Promise((resolve) => {
    const listener = () => {
      if (!entry.snapshot.ready && !entry.snapshot.failed) return;
      entry.listeners.delete(listener);
      trimMediaCache(kind);
      resolve();
    };
    entry.listeners.add(listener);
  });
}

function ensureMediaEntry(src: string, kind: MediaKind) {
  const cached = mediaCache.get(src);
  if (cached) {
    mediaCache.delete(src);
    mediaCache.set(src, cached);
    return cached;
  }

  const image = new Image();
  image.decoding = "async";
  const entry: MediaEntry = {
    kind,
    image,
    snapshot: emptySnapshot,
    listeners: new Set()
  };
  mediaCache.set(src, entry);

  const markReady = () => {
    if (entry.snapshot.ready || entry.snapshot.failed) return;
    entry.snapshot = {
      ready: true,
      failed: false,
      dimensions: {
        width: image.naturalWidth,
        height: image.naturalHeight
      }
    };
    notify(entry);
  };
  const markFailed = () => {
    if (entry.snapshot.ready || entry.snapshot.failed) return;
    entry.snapshot = {
      ready: false,
      failed: true,
      dimensions: null
    };
    notify(entry);
  };

  image.addEventListener("load", markReady, { once: true });
  image.addEventListener("error", markFailed, { once: true });
  image.src = src;
  if (image.complete && image.naturalWidth > 0) {
    markReady();
  } else {
    void image.decode().then(markReady).catch(() => {
      if (image.complete && image.naturalWidth === 0) markFailed();
    });
  }

  trimMediaCache(kind);
  return entry;
}

function notify(entry: MediaEntry) {
  for (const listener of entry.listeners) listener();
}

function trimMediaCache(kind: MediaKind) {
  let count = 0;
  for (const entry of mediaCache.values()) {
    if (entry.kind === kind) count += 1;
  }
  if (count <= cacheLimits[kind]) return;

  for (const [src, entry] of mediaCache) {
    if (entry.kind !== kind || entry.listeners.size > 0) continue;
    mediaCache.delete(src);
    count -= 1;
    if (count <= cacheLimits[kind]) break;
  }
}
