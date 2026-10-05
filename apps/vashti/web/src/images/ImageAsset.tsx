import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { saveImage } from "../saveImage";
import { privateImageGuard } from "../privateChatStore";
import { loadImageBlob } from "./assetStorage";
import { useImagePath } from "./storage";
import { AssetCache, thumbnailCrop, type CachedImage } from "./assetCache";

function imageCache(path: string, thumbnail = false) {
  const guard = privateImageGuard();
  return new AssetCache(
    async (id) => {
      guard();
      const blob = await loadImageBlob(path, id);
      guard();
      let url = URL.createObjectURL(blob);
      try {
        const image = new Image();
        image.src = url;
        await image.decode();
        guard();
        if (!thumbnail)
          return {
            url,
            width: image.naturalWidth,
            height: image.naturalHeight,
            bytes: blob.size + image.naturalWidth * image.naturalHeight * 4,
          };
        // Only the gallery preview is cropped. Viewing, editing and exporting use original bytes.
        const crop = thumbnailCrop(image.naturalWidth, image.naturalHeight);
        const side = Math.min(320, crop.side);
        const canvas = document.createElement("canvas");
        canvas.width = side;
        canvas.height = side;
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Thumbnail renderer unavailable");
        context.drawImage(
          image,
          crop.x,
          crop.y,
          crop.side,
          crop.side,
          0,
          0,
          side,
          side,
        );
        const preview = await new Promise<Blob>((resolve, reject) =>
          canvas.toBlob(
            (value) =>
              value
                ? resolve(value)
                : reject(new Error("Thumbnail unavailable")),
            "image/png",
          ),
        );
        guard();
        URL.revokeObjectURL(url);
        url = URL.createObjectURL(preview);
        return {
          url,
          width: side,
          height: side,
          bytes: preview.size + side * side * 4,
        };
      } catch (error) {
        URL.revokeObjectURL(url);
        throw error;
      }
    },
    thumbnail ? 120 : 10,
    (thumbnail ? 32 : 64) * 1024 * 1024,
    thumbnail ? 2 : 3,
    undefined,
    guard,
  );
}
const CacheContext = createContext<{
  originals: AssetCache;
  thumbnails: AssetCache;
} | null>(null);
export function ImageAssetProvider({ children }: { children: ReactNode }) {
  const path = useImagePath();
  const [cache, setCache] = useState<{
    originals: AssetCache;
    thumbnails: AssetCache;
  } | null>(null);
  useEffect(() => {
    // A fresh cache for each effect lifetime also supports React's development remount.
    const current = {
      originals: imageCache(path),
      thumbnails: imageCache(path, true),
    };
    setCache(current);
    return () => {
      current.originals.dispose();
      current.thumbnails.dispose();
    };
  }, [path]);
  return cache ? (
    <CacheContext.Provider value={cache}>{children}</CacheContext.Provider>
  ) : null;
}

export type ImageAssetStatus = {
  id: string;
  state: "loading" | "ready" | "unavailable";
};

export function ImageAsset({
  id,
  alt = "Generated image",
  dimensions,
  onClick,
  onDimensions,
  onStatus,
  openLabel = "Open full image",
  neighbors = [],
  thumbnail = false,
}: {
  id: string;
  alt?: string;
  dimensions?: { width: number; height: number };
  onClick?: () => void;
  onDimensions?: (size: { width: number; height: number }) => void;
  onStatus?: (status: ImageAssetStatus) => void;
  openLabel?: string;
  neighbors?: string[];
  thumbnail?: boolean;
}) {
  const caches = useContext(CacheContext);
  const cache = caches
    ? thumbnail
      ? caches.thumbnails
      : caches.originals
    : null;
  const path = useImagePath();
  const [image, setImage] = useState<(CachedImage & { id: string }) | null>(
    null,
  );
  const [failed, setFailed] = useState(false);
  const [visible, setVisible] = useState(false);
  const [size, setSize] = useState<{
    id: string;
    width: number;
    height: number;
  } | null>(null);
  const dimensionsRef = useRef(onDimensions);
  dimensionsRef.current = onDimensions;
  const displayed = useRef<(() => void) | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const shouldLoad = visible || Boolean(onDimensions);
  const neighborKey = neighbors
    .filter((value) => value !== id)
    .slice(0, 2)
    .join(",");
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => setVisible(entries.some((e) => e.isIntersecting)),
      { rootMargin: thumbnail ? "300px" : "1000px" },
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [thumbnail]);
  useEffect(() => {
    if (!shouldLoad) {
      setImage(null);
      displayed.current?.();
      displayed.current = null;
      return;
    }
    let alive = true;
    let adopted = false;
    const owned = cache ?? imageCache(path, thumbnail);
    const acquired = owned.acquire(id);
    setFailed(false);
    const adopt = (value: CachedImage) => {
      if (!alive) return;
      displayed.current?.();
      displayed.current = () => {
        acquired.release();
        if (!cache) owned.dispose();
      };
      adopted = true;
      setImage({ ...value, id });
      setSize({ id, width: value.width, height: value.height });
      dimensionsRef.current?.({ width: value.width, height: value.height });
    };
    if (acquired.image) adopt(acquired.image);
    acquired.promise
      .then((value) => {
        if (!adopted) adopt(value);
      })
      .catch(() => {
        if (alive) {
          setFailed(true);
          setImage(null);
        }
      });
    return () => {
      alive = false;
      if (!adopted) {
        acquired.release();
        if (!cache) owned.dispose();
      }
    };
  }, [id, shouldLoad, cache, path, thumbnail]);
  useEffect(() => {
    if (!visible || !cache || !neighborKey) return;
    const entries = neighborKey.split(",").map((value) => cache.acquire(value));
    for (const entry of entries) void entry.promise.catch(() => {});
    return () => {
      for (const entry of entries) entry.release();
    };
  }, [neighborKey, visible, cache]);
  useEffect(
    () => () => {
      displayed.current?.();
      displayed.current = null;
    },
    [],
  );
  const frame =
    dimensions ?? (size?.id === id ? size : null) ?? cache?.peek(id);
  const state = failed ? "unavailable" : image?.id === id ? "ready" : "loading";
  useEffect(() => {
    onStatus?.({ id, state });
  }, [id, state, onStatus]);
  return (
    <div
      className="images-asset"
      ref={ref}
      style={
        frame ? { aspectRatio: `${frame.width} / ${frame.height}` } : undefined
      }
      aria-busy={visible && !failed && image?.id !== id}
    >
      {image ? (
        onClick ? (
          <button
            type="button"
            className="images-open-image"
            disabled={image.id !== id}
            onClick={onClick}
            aria-label={openLabel}
          >
            <img draggable={false} src={image.url} alt={alt} />
          </button>
        ) : (
          <img draggable={false} src={image.url} alt={alt} />
        )
      ) : (
        <span role="status">
          {failed ? "Image unavailable" : visible ? "Loading image…" : ""}
        </span>
      )}
    </div>
  );
}
export async function downloadImage(id: string, path = "/api/images") {
  await saveImage(await loadImageBlob(path, id), id);
}
