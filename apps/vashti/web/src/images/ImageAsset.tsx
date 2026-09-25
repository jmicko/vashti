import { useEffect, useRef, useState } from "react";
import { requestBlob } from "../api";
export function ImageAsset({ id, alt = "Generated image", onClick }: { id: string; alt?: string; onClick?: () => void }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const observer = new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting)) { setVisible(true); observer.disconnect(); } }, { rootMargin: "200px" });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) return;
    let alive = true; let objectUrl: string | undefined;
    setUrl(null); setFailed(false);
    requestBlob(`/api/images/assets/${encodeURIComponent(id)}`).then(blob => {
      if (!alive) return; objectUrl = URL.createObjectURL(blob); setUrl(objectUrl);
    }).catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [id, visible]);
  return <div className="images-asset" ref={ref}>
    {url ? onClick ? <button type="button" className="images-open-image" onClick={onClick} aria-label="Open full image"><img src={url} alt={alt} /></button> : <img src={url} alt={alt} /> : <span role="status">{failed ? "Image unavailable" : "Loading image…"}</span>}
  </div>;
}
export async function downloadImage(id: string) {
  const blob = await requestBlob(`/api/images/assets/${encodeURIComponent(id)}`);
  const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
  anchor.href = url; anchor.download = `${id}.${blob.type === "image/jpeg" ? "jpg" : blob.type === "image/webp" ? "webp" : "png"}`;
  anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
