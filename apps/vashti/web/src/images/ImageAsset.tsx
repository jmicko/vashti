import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { requestBlob } from "../api";

type CachedImage = { url: string; width: number; height: number; bytes: number };
type Entry = { promise: Promise<CachedImage>; image?: CachedImage; users: number };
class AssetCache {
  private entries = new Map<string, Entry>();
  private closed = false;
  acquire(id: string) {
    let entry = this.entries.get(id);
    if (!entry) {
      const created: Entry = { users: 0, promise: Promise.resolve(null as unknown as CachedImage) };
      created.promise = requestBlob(`/api/images/assets/${encodeURIComponent(id)}`).then(async blob => {
        const url = URL.createObjectURL(blob);
        try {
          const image = new Image(); image.src = url; await image.decode();
          if (this.closed) throw new Error("Image workspace closed");
          const cached = { url, width: image.naturalWidth, height: image.naturalHeight, bytes: Math.max(blob.size, image.naturalWidth * image.naturalHeight * 4) };
          created.image = cached;
          this.trim();
          return cached;
        } catch (error) { URL.revokeObjectURL(url); throw error; }
      }).catch(error => { if (this.entries.get(id) === created) this.entries.delete(id); throw error; });
      entry = created; this.entries.set(id, entry);
    }
    this.entries.delete(id); this.entries.set(id, entry); entry.users++;
    const acquired = entry;
    return { promise: entry.promise, release: () => { acquired.users--; this.trim(); } };
  }
  private trim() {
    let bytes = [...this.entries.values()].reduce((total, entry) => total + (entry.image?.bytes ?? 0), 0);
    for (const [id, entry] of this.entries) {
      if (this.entries.size <= 10 && bytes <= 64 * 1024 * 1024) break;
      if (!entry.users && entry.image) {
        URL.revokeObjectURL(entry.image.url); bytes -= entry.image.bytes; this.entries.delete(id);
      }
    }
  }
  dispose() {
    this.closed = true;
    for (const entry of this.entries.values()) if (entry.image) URL.revokeObjectURL(entry.image.url);
    this.entries.clear();
  }
}
const CacheContext = createContext<AssetCache | null>(null);
export function ImageAssetProvider({ children }: { children: ReactNode }) {
  const [cache, setCache] = useState<AssetCache | null>(null);
  useEffect(() => {
    // A fresh cache for each effect lifetime also supports React's development remount.
    const current = new AssetCache(); setCache(current);
    return () => current.dispose();
  }, []);
  return cache ? <CacheContext.Provider value={cache}>{children}</CacheContext.Provider> : null;
}

export function ImageAsset({ id, alt = "Generated image", onClick, onDimensions, neighbors = [] }: {
  id: string; alt?: string; onClick?: () => void; onDimensions?: (size: { width: number; height: number }) => void; neighbors?: string[];
}) {
  const cache = useContext(CacheContext);
  const [image, setImage] = useState<(CachedImage & { id: string }) | null>(null);
  const [failed, setFailed] = useState(false);
  const [visible, setVisible] = useState(false);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const dimensionsRef = useRef(onDimensions); dimensionsRef.current = onDimensions;
  const displayed = useRef<(() => void) | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const shouldLoad = visible || Boolean(onDimensions);
  const neighborKey = neighbors.filter(value => value !== id).slice(0, 2).join(",");
  useEffect(() => {
    const observer = new IntersectionObserver(entries => setVisible(entries.some(e => e.isIntersecting)), { rootMargin: "300px" });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!shouldLoad) { setImage(null); displayed.current?.(); displayed.current = null; return; }
    let alive = true;
    let adopted = false;
    const owned = cache ?? new AssetCache();
    const acquired = owned.acquire(id);
    setFailed(false);
    acquired.promise.then(value => {
      if (!alive) return;
      displayed.current?.();
      displayed.current = () => { acquired.release(); if (!cache) owned.dispose(); };
      adopted = true;
      setImage({ ...value, id }); setSize({ width: value.width, height: value.height });
      dimensionsRef.current?.({ width: value.width, height: value.height });
    }).catch(() => { if (alive) { setFailed(true); setImage(null); } });
    return () => { alive = false; if (!adopted) { acquired.release(); if (!cache) owned.dispose(); } };
  }, [id, shouldLoad, cache]);
  useEffect(() => {
    if (!visible || !cache || !neighborKey) return;
    const entries = neighborKey.split(",").map(value => cache.acquire(value));
    for (const entry of entries) void entry.promise.catch(() => {});
    return () => { for (const entry of entries) entry.release(); };
  }, [neighborKey, visible, cache]);
  useEffect(() => () => { displayed.current?.(); displayed.current = null; }, []);
  return <div className="images-asset" ref={ref} style={size ? { aspectRatio: `${size.width} / ${size.height}` } : undefined} aria-busy={visible && !failed && image?.id !== id}>
    {image ? onClick ? <button type="button" className="images-open-image" disabled={image.id !== id} onClick={onClick} aria-label="Open full image"><img draggable={false} src={image.url} alt={alt} /></button> : <img draggable={false} src={image.url} alt={alt} /> : <span role="status">{failed ? "Image unavailable" : visible ? "Loading image…" : ""}</span>}
  </div>;
}
export async function downloadImage(id: string) {
  const blob = await requestBlob(`/api/images/assets/${encodeURIComponent(id)}`);
  const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
  anchor.href = url; anchor.download = `${id}.${blob.type === "image/jpeg" ? "jpg" : blob.type === "image/webp" ? "webp" : "png"}`;
  anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
