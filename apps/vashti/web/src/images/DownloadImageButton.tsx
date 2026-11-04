import { useEffect, useRef, useState } from "react";
import { Check, Download, LoaderCircle } from "lucide-react";
import { downloadImage } from "./ImageAsset";

export function DownloadImageButton({
  id,
  path,
  disabled = false,
  className = "images-icon",
  onError,
}: {
  id: string;
  path: string;
  disabled?: boolean;
  className?: string;
  onError: (message: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ id: string; label: string } | null>(
    null,
  );
  const currentId = useRef(id);
  currentId.current = id;
  const live = useRef(false);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  useEffect(() => {
    if (!result) return;
    const timer = setTimeout(() => setResult(null), 4000);
    return () => clearTimeout(timer);
  }, [result]);
  const label = busy
    ? "Saving image..."
    : result?.id === id
      ? result.label
      : "Download image";
  async function save() {
    setBusy(true);
    setResult(null);
    try {
      const outcome = await downloadImage(id, path);
      if (live.current && currentId.current === id && outcome !== "cancelled")
        setResult({
          id,
          label: outcome === "saved" ? "Image saved" : "Download requested",
        });
    } catch (error) {
      if (live.current)
        onError(
          error instanceof Error ? error.message : "Could not download image",
        );
    } finally {
      if (live.current) setBusy(false);
    }
  }
  return (
    <>
      <button
        type="button"
        className={className}
        title={label}
        aria-label={label}
        disabled={disabled || busy}
        onClick={() => void save()}
      >
        {busy ? (
          <LoaderCircle className="image-download-progress" />
        ) : result?.id === id ? (
          <Check />
        ) : (
          <Download />
        )}
      </button>
      <span className="visually-hidden" role="status">
        {label === "Download image" ? "" : label}
      </span>
    </>
  );
}
