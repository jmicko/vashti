import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  History,
  Layers,
  Pencil,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { ImageAsset, downloadImage } from "./ImageAsset";
import { ImageComparison } from "./ImageComparison";
import { ActionMenu } from "../ActionMenu";
import { comparisonReference } from "./comparison";
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
  const pageRequest = useRef<Promise<GalleryItem[]> | null>(null);
  const pageSentinel = useRef<HTMLDivElement>(null);
  const nextPage = useRef<() => Promise<GalleryItem[]>>(async () => []);
  const variantsCache = useRef(new Map<string, ImageResult[]>());
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

  async function loadMore(): Promise<GalleryItem[]> {
    if (pageRequest.current) return pageRequest.current;
    setLoading(true);
    setError("");
    const request = (async () => {
      try {
        const page = await requestJson<GalleryItem[]>(
          `${imagePath}/gallery?offset=${nextOffset}`,
        );
        if (!mounted.current) return [];
        setItems((values) => appendGallery(values, page));
        setNextOffset((value) => value + page.length);
        setHasMore(page.length === 50);
        return page;
      } catch (error) {
        if (mounted.current) setError(imageError(error));
        return [];
      } finally {
        if (mounted.current) setLoading(false);
        pageRequest.current = null;
      }
    })();
    pageRequest.current = request;
    return request;
  }
  nextPage.current = loadMore;
  useEffect(() => {
    const target = pageSentinel.current;
    if (!target || !hasMore || loading || error || viewer) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting))
          void nextPage.current();
      },
      { rootMargin: "400px" },
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, [hasMore, loading, error, nextOffset, Boolean(viewer)]);

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

  function open(item: GalleryItem) {
    setConfirmRemove(false);
    setError("");
    setViewer({ item, ids: [item.asset_id], index: 0, results: [] });
  }

  useEffect(() => {
    const item = viewer?.item;
    if (!item?.session_id || !item.group_id) return;
    let live = true;
    const key = `${item.id}:${item.variants}:${item.asset_id}`;
    const cached = variantsCache.current.get(key);
    const request = cached
      ? Promise.resolve(cached)
      : galleryVariants(item, (offset) =>
          requestJson<ImageResult[]>(
            `${imagePath}/sessions/${item.session_id}/results?offset=${offset}`,
          ),
        );
    void request
      .then((variants) => {
        if (!live || !variants.length) return;
        variantsCache.current.set(key, variants);
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
      })
      .catch((error) => {
        if (live) setError(imageError(error));
      });
    return () => {
      live = false;
    };
  }, [viewer?.item.id, imagePath]);

  async function remove(item: GalleryItem) {
    await requestJson(`${imagePath}/gallery/${item.id}`, { method: "DELETE" });
    if (!mounted.current) return;
    setItems((values) => values.filter((value) => value.id !== item.id));
    setNextOffset((value) => Math.max(0, value - 1));
    setViewer(null);
    setConfirmRemove(false);
  }

  function moveVersion(delta: number) {
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

  const libraryIndex = viewer
    ? items.findIndex((item) => item.id === viewer.item.id)
    : -1;
  function move(delta: number) {
    const item = items[libraryIndex + delta];
    if (item) open(item);
    else if (delta > 0 && hasMore) {
      const previousId = viewer?.item.id;
      void loadMore().then((page) => {
        if (!mounted.current || !page.length) return;
        setViewer((current) =>
          current?.item.id === previousId
            ? { item: page[0], ids: [page[0].asset_id], index: 0, results: [] }
            : current,
        );
      });
    }
  }

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
        <ActionMenu label="Gallery view" icon={<SlidersHorizontal />}>
          <fieldset className="gallery-view-options">
            <legend>Thumbnail size</legend>
            {["Small", "Medium", "Large"].map((label, value) => (
              <label key={label}>
                <input
                  type="radio"
                  name="gallery-density"
                  checked={density === value}
                  onChange={() => setDensity(value)}
                />
                {label}
              </label>
            ))}
          </fieldset>
        </ActionMenu>
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
            onClick={() => open(item)}
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
        <div ref={pageSentinel} className="images-actions images-pagination">
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
        <ImageComparison
          fullscreen={{
            title: `Image ${libraryIndex + 1} of ${items.length}${hasMore ? "+" : ""}`,
            onClose: () => {
              setViewer(null);
              setError("");
            },
            canPrevious: libraryIndex > 0,
            canNext: libraryIndex < items.length - 1 || (hasMore && !loading),
          }}
          previousId={items[libraryIndex - 1]?.asset_id}
          nextId={items[libraryIndex + 1]?.asset_id}
          notice={alert}
          versions={
            viewer.ids.length > 1 && (
              <details className="photo-viewer-versions">
                <summary>
                  Version {viewer.index + 1} of {viewer.ids.length}
                </summary>
                <div
                  className="images-gallery-versions"
                  role="group"
                  aria-label="Image versions"
                >
                  <button
                    type="button"
                    className="images-icon"
                    aria-label="Previous version"
                    title="Previous version"
                    disabled={viewer.index === 0}
                    onClick={() => moveVersion(-1)}
                  >
                    <ArrowLeft size={18} />
                  </button>
                  <span>
                    Version {viewer.index + 1} of {viewer.ids.length}
                  </span>
                  <button
                    type="button"
                    className="images-icon"
                    aria-label="Next version"
                    title="Next version"
                    disabled={viewer.index === viewer.ids.length - 1}
                    onClick={() => moveVersion(1)}
                  >
                    <ArrowRight size={18} />
                  </button>
                </div>
              </details>
            )
          }
          secondaryActions={
            <>
              {viewer.item.session_id && (
                <button
                  type="button"
                  onClick={() => onNavigate("flow", viewer.item.session_id!)}
                >
                  <History />
                  Open image flow
                </button>
              )}
              <button
                type="button"
                disabled={busy}
                onClick={() => setConfirmRemove(true)}
              >
                <Trash2 />
                Remove from gallery
              </button>
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
            </>
          }
          id={viewer.ids[viewer.index]}
          referenceId={comparisonReference(selected)}
          onStep={move}
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
          neighbors={[
            items[libraryIndex - 1]?.asset_id,
            items[libraryIndex + 1]?.asset_id,
          ].filter((id): id is string => Boolean(id))}
        >
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
        </ImageComparison>
      )}
    </>
  );
}
