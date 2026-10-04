import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  History,
  Layers,
  Minus,
  Pencil,
  Plus,
  Trash2,
} from "lucide-react";
import { ImageAsset, downloadImage } from "./ImageAsset";
import { ImagePanel } from "./controls";
import { SwipeSurface } from "./SwipeSurface";
import { appendGallery, galleryVariants } from "./gallery";
import { imageError, requestJson, notifyImageFlows, writeImage } from "./api";
import { useImagePath } from "./storage";
import {
  newDraft,
  type GalleryItem,
  type ImageSession,
  type ImageResult,
  type ImageView,
  type Preferences,
} from "./types";

export function ImageGallery({
  preferences,
  onNavigate,
}: {
  preferences: Preferences;
  onNavigate: (view: ImageView, id?: string) => void;
}) {
  const imagePath = useImagePath();
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [nextOffset, setNextOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [density, setDensity] = useState(1);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const mounted = useRef(true);
  const [viewer, setViewer] = useState<{
    item: GalleryItem;
    ids: string[];
    index: number;
    results: ImageResult[];
  } | null>(null);

  useEffect(() => {
    let live = true;
    mounted.current = true;
    requestJson<GalleryItem[]>(`${imagePath}/gallery?offset=0`)
      .then((values) => {
        if (!live) return;
        setItems(values);
        setNextOffset(values.length);
        setHasMore(values.length === 50);
      })
      .catch((error) => {
        if (live) setError(imageError(error));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
      mounted.current = false;
    };
  }, [imagePath]);

  async function loadMore() {
    setLoading(true);
    setError("");
    try {
      const page = await requestJson<GalleryItem[]>(
        `${imagePath}/gallery?offset=${nextOffset}`,
      );
      if (!mounted.current) return;
      setItems((values) => appendGallery(values, page));
      setNextOffset((value) => value + page.length);
      setHasMore(page.length === 50);
    } catch (error) {
      if (mounted.current) setError(imageError(error));
    } finally {
      if (mounted.current) setLoading(false);
    }
  }

  async function action(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (error) {
      if (mounted.current) setError(imageError(error));
    } finally {
      if (mounted.current) setBusy(false);
    }
  }

  async function edit(id: string) {
    const draft = newDraft(preferences.edit);
    draft.input_ids = [id];
    draft.original_input_ids = [id];
    const session = await writeImage<ImageSession>(`${imagePath}/sessions`, {
      mode: "iterative",
      draft,
    });
    if (!mounted.current) return;
    notifyImageFlows();
    onNavigate("flow", session.id);
  }

  async function open(item: GalleryItem) {
    setConfirmRemove(false);
    setViewer({ item, ids: [item.asset_id], index: 0, results: [] });
    if (!item.session_id || !item.group_id) return;
    const variants = await galleryVariants(item, (offset) =>
      requestJson<ImageResult[]>(
        `${imagePath}/sessions/${item.session_id}/results?offset=${offset}`,
      ),
    );
    if (!mounted.current || !variants.length) return;
    const ids = variants.map((value) => value.asset_id);
    setViewer((value) =>
      value?.item.id === item.id
        ? {
            item,
            ids,
            index: Math.max(0, ids.indexOf(item.asset_id)),
            results: variants,
          }
        : value,
    );
  }

  async function remove(item: GalleryItem) {
    await requestJson(`${imagePath}/gallery/${item.id}`, { method: "DELETE" });
    if (!mounted.current) return;
    setItems((values) => values.filter((value) => value.id !== item.id));
    setNextOffset((value) => Math.max(0, value - 1));
    setViewer(null);
    setConfirmRemove(false);
  }

  function move(delta: number) {
    setViewer((value) =>
      value
        ? {
            ...value,
            index: Math.max(
              0,
              Math.min(value.ids.length - 1, value.index + delta),
            ),
          }
        : value,
    );
    setConfirmRemove(false);
  }

  useEffect(() => {
    if (!viewer) return;
    const key = (event: KeyboardEvent) => {
      if (
        event.target instanceof HTMLElement &&
        event.target.closest("input,textarea,select,[contenteditable=true]")
      )
        return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        move(event.key === "ArrowLeft" ? -1 : 1);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [Boolean(viewer)]);

  const selected = viewer?.results[viewer.index];
  const alert = error && (
    <p className="images-error" role="alert">
      {error}
    </p>
  );
  return (
    <>
      {!viewer && alert}
      <div className="images-gallery-controls">
        <small role="status">
          {loading && !items.length
            ? "Loading images..."
            : `${items.length}${hasMore ? "+" : ""} images`}
        </small>
        <div
          className="images-actions"
          role="group"
          aria-label="Thumbnail size"
        >
          <button
            type="button"
            className="images-icon"
            title="Smaller thumbnails"
            aria-label="Smaller thumbnails"
            disabled={density === 0}
            onClick={() => setDensity((value) => value - 1)}
          >
            <Minus size={18} />
          </button>
          <button
            type="button"
            className="images-icon"
            title="Larger thumbnails"
            aria-label="Larger thumbnails"
            disabled={density === 2}
            onClick={() => setDensity((value) => value + 1)}
          >
            <Plus size={18} />
          </button>
        </div>
      </div>
      <div
        className={`images-gallery images-gallery-density-${density}`}
        aria-label="Image gallery"
      >
        {items.map((item, index) => (
          <button
            key={item.id}
            type="button"
            className="images-gallery-tile"
            disabled={busy}
            onClick={() => void action(() => open(item))}
            aria-label={`Open image ${index + 1}${item.variants > 1 ? `, ${item.variants} versions` : ""}`}
          >
            <ImageAsset
              id={item.asset_id}
              alt=""
              thumbnail
              dimensions={{ width: 1, height: 1 }}
            />
            {item.variants > 1 && (
              <span className="images-gallery-stack" aria-hidden="true">
                <Layers size={14} />
                {item.variants}
              </span>
            )}
          </button>
        ))}
      </div>
      {!items.length && !loading && !error && (
        <p className="images-empty" role="status">
          No images yet.
        </p>
      )}
      {(hasMore || (!items.length && error)) && (
        <div className="images-actions images-pagination">
          <button
            type="button"
            disabled={loading || busy}
            onClick={() => void loadMore()}
          >
            {loading ? "Loading..." : error ? "Try again" : "Load more"}
          </button>
        </div>
      )}
      {viewer && (
        <ImagePanel
          title={
            viewer.ids.length > 1
              ? `Version ${viewer.index + 1} of ${viewer.ids.length}`
              : "Image"
          }
          className="images-viewer-panel"
          onClose={() => {
            setViewer(null);
            setError("");
          }}
        >
          {alert}
          <SwipeSurface onStep={move}>
            <ImageAsset
              id={viewer.ids[viewer.index]}
              dimensions={
                selected
                  ? {
                      width: selected.recipe.width,
                      height: selected.recipe.height,
                    }
                  : viewer.item.width && viewer.item.height
                    ? { width: viewer.item.width, height: viewer.item.height }
                    : undefined
              }
              neighbors={viewer.ids.filter(
                (_, index) => Math.abs(index - viewer.index) === 1,
              )}
            />
          </SwipeSurface>
          <div className="images-actions">
            {viewer.ids.length > 1 && (
              <>
                <button
                  type="button"
                  className="images-icon"
                  disabled={viewer.index === 0}
                  title="Previous version"
                  aria-label="Previous version"
                  onClick={() => move(-1)}
                >
                  <ArrowLeft />
                </button>
                <button
                  type="button"
                  className="images-icon"
                  disabled={viewer.index === viewer.ids.length - 1}
                  title="Next version"
                  aria-label="Next version"
                  onClick={() => move(1)}
                >
                  <ArrowRight />
                </button>
              </>
            )}
            <button
              type="button"
              className="images-icon"
              disabled={busy}
              title="Edit image"
              aria-label="Edit image"
              onClick={() => void action(() => edit(viewer.ids[viewer.index]))}
            >
              <Pencil />
            </button>
            <button
              type="button"
              className="images-icon"
              disabled={busy}
              title="Download image"
              aria-label="Download image"
              onClick={() =>
                void action(() =>
                  downloadImage(viewer.ids[viewer.index], imagePath),
                )
              }
            >
              <Download />
            </button>
            {viewer.item.session_id && (
              <button
                type="button"
                className="images-icon"
                title="Open image flow"
                aria-label="Open image flow"
                onClick={() => onNavigate("flow", viewer.item.session_id!)}
              >
                <History />
              </button>
            )}
            <button
              type="button"
              className="images-icon"
              disabled={busy}
              title="Remove from gallery"
              aria-label="Remove from gallery"
              onClick={() => setConfirmRemove(true)}
            >
              <Trash2 />
            </button>
          </div>
          {confirmRemove && (
            <div className="images-gallery-remove">
              <p>
                Remove from gallery? Images still used in flows or chats are
                kept.
              </p>
              <div className="images-actions">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void action(() => remove(viewer.item))}
                >
                  Remove
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => setConfirmRemove(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </ImagePanel>
      )}
    </>
  );
}
