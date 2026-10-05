import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowLeft, Check, Layers, Paperclip, X } from "lucide-react";
import { ImageAsset } from "./ImageAsset";
import { ImagePanel } from "./controls";
import { appendGallery, galleryVariants } from "./gallery";
import { imageError, requestJson } from "./api";
import { deviceImagePath, useImagePath } from "./storage";
import { toggleReference } from "./references";
import type { GalleryItem, ImageResult } from "./types";

export function ImageReferencePicker({
  existingIds,
  limit,
  onAttach,
  onClose,
}: {
  existingIds: string[];
  limit: number;
  onAttach: (ids: string[]) => Promise<void>;
  onClose: () => void;
}) {
  const path = useImagePath();
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [stack, setStack] = useState<GalleryItem | null>(null);
  const [variants, setVariants] = useState<ImageResult[]>([]);
  const live = useRef(true);
  const body = useRef<HTMLDivElement>(null);
  const galleryScroll = useRef(0);
  const remaining = Math.max(0, limit - existingIds.length);

  useEffect(() => {
    live.current = true;
    void load(0);
    return () => {
      live.current = false;
    };
  }, [path]);
  useLayoutEffect(() => {
    if (body.current)
      body.current.scrollTop = stack ? 0 : galleryScroll.current;
  }, [stack]);

  async function load(nextOffset: number) {
    setLoading(true);
    setError("");
    try {
      const page = await requestJson<GalleryItem[]>(
        `${path}/gallery?offset=${nextOffset}`,
      );
      if (!live.current) return;
      setItems((values) => (nextOffset ? appendGallery(values, page) : page));
      setOffset(nextOffset + page.length);
      setHasMore(page.length === 50);
    } catch (error) {
      if (live.current) setError(imageError(error));
    } finally {
      if (live.current) setLoading(false);
    }
  }

  async function openStack(item: GalleryItem) {
    if (!stack) galleryScroll.current = body.current?.scrollTop ?? 0;
    setStack(item);
    setVariants([]);
    setBusy(true);
    setError("");
    try {
      const values = await galleryVariants(item, (offset) =>
        requestJson<ImageResult[]>(
          `${path}/sessions/${item.session_id}/results?offset=${offset}`,
        ),
      );
      if (live.current) setVariants(values);
    } catch (error) {
      if (live.current) setError(imageError(error));
    } finally {
      if (live.current) setBusy(false);
    }
  }

  async function attach() {
    if (busy || !selected.length || selected.length > remaining) return;
    setBusy(true);
    setError("");
    try {
      await onAttach(selected);
    } catch (error) {
      if (live.current) setError(imageError(error));
    } finally {
      if (live.current) setBusy(false);
    }
  }

  const tiles = stack
    ? variants.length
      ? variants.map((value) => ({
          id: value.id,
          asset_id: value.asset_id,
          variants: 1,
          session_id: null,
        }))
      : [stack]
    : items;
  return (
    <ImagePanel
      title={
        path === deviceImagePath
          ? "Attach from device gallery"
          : "Attach from gallery"
      }
      className="images-picker-panel"
      onClose={onClose}
    >
      <div className="images-picker-body" ref={body}>
        {stack && (
          <div className="images-picker-stack-header">
            <button
              type="button"
              className="images-icon"
              title="Back to gallery"
              aria-label="Back to gallery"
              disabled={busy}
              onClick={() => {
                setStack(null);
                setError("");
              }}
            >
              <ArrowLeft size={18} />
            </button>
            <strong>Image versions</strong>
          </div>
        )}
        {error && (
          <p className="images-error" role="alert">
            {error}
          </p>
        )}
        {busy && stack && !variants.length && (
          <p role="status">Loading versions...</p>
        )}
        <div
          className="images-gallery images-reference-gallery"
          aria-label={stack ? "Image versions" : "Gallery reference images"}
        >
          {tiles.map((item, index) => {
            const order = selected.indexOf(item.asset_id);
            const attached = existingIds.includes(item.asset_id);
            return (
              <div className="images-picker-tile" key={item.id}>
                <button
                  type="button"
                  className="images-gallery-tile"
                  aria-label={`${stack ? "Version" : "Image"} ${index + 1}${attached ? ", already attached" : ""}`}
                  aria-pressed={order >= 0 || attached}
                  disabled={
                    busy ||
                    attached ||
                    (order < 0 && selected.length >= remaining)
                  }
                  onClick={() =>
                    setSelected((values) =>
                      toggleReference(values, item.asset_id, remaining),
                    )
                  }
                >
                  <ImageAsset
                    id={item.asset_id}
                    alt=""
                    thumbnail
                    dimensions={{ width: 1, height: 1 }}
                  />
                  {(order >= 0 || attached) && (
                    <span className="images-picker-selected" aria-hidden="true">
                      {attached ? <Check size={16} /> : order + 1}
                    </span>
                  )}
                </button>
                {!stack && item.variants > 1 && item.session_id && (
                  <button
                    type="button"
                    className="images-picker-versions"
                    disabled={busy}
                    title="Choose an image version"
                    aria-label={`Choose a version of image ${index + 1}, ${item.variants} versions`}
                    onClick={() => void openStack(item)}
                  >
                    <Layers size={16} />
                    {item.variants}
                  </button>
                )}
              </div>
            );
          })}
        </div>
        {loading && !items.length && (
          <p className="images-empty" role="status">
            Loading images...
          </p>
        )}
        {!loading && !items.length && !error && (
          <p className="images-empty" role="status">
            No images in this gallery yet.
          </p>
        )}
        {!stack && (hasMore || error) && (
          <div className="images-actions images-pagination">
            <button
              type="button"
              disabled={loading || busy}
              onClick={() => void load(offset)}
            >
              {loading ? "Loading..." : error ? "Try again" : "Load more"}
            </button>
          </div>
        )}
        {stack && error && (
          <button
            type="button"
            disabled={busy}
            onClick={() => void openStack(stack)}
          >
            Try again
          </button>
        )}
      </div>
      <footer className="images-picker-footer">
        {selected.length > 0 && (
          <div
            className="images-picker-selection"
            aria-label="Selected reference images, in order"
          >
            {selected.map((id, index) => (
              <button
                type="button"
                key={id}
                disabled={busy}
                title={`Remove selected reference ${index + 1}`}
                aria-label={`Remove selected reference ${index + 1}`}
                onClick={() =>
                  setSelected((values) =>
                    values.filter((value) => value !== id),
                  )
                }
              >
                <ImageAsset
                  id={id}
                  alt=""
                  thumbnail
                  dimensions={{ width: 1, height: 1 }}
                />
                <span>{index + 1}</span>
                <X size={12} />
              </button>
            ))}
          </div>
        )}
        <small role="status">
          {selected.length} / {remaining} selected
        </small>
        <button
          type="button"
          disabled={busy || !selected.length || selected.length > remaining}
          onClick={() => void attach()}
        >
          <Paperclip size={18} />
          {busy ? "Please wait..." : "Attach"}
        </button>
      </footer>
    </ImagePanel>
  );
}
