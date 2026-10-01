import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  History,
  Pencil,
  Trash2,
} from "lucide-react";
import { requestJson } from "../api";
import { ImageAsset, downloadImage } from "./ImageAsset";
import { ImagePanel } from "./controls";
import { SwipeSurface } from "./SwipeSurface";
import { imageError, imagePath, notifyImageFlows, writeImage } from "./api";
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
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [viewer, setViewer] = useState<{
    ids: string[];
    index: number;
    sessionId: string | null;
    results: ImageResult[];
  } | null>(null);
  useEffect(() => {
    let live = true;
    setLoading(true);
    requestJson<GalleryItem[]>(`${imagePath}/gallery?offset=${offset}`)
      .then((values) => {
        if (live) setItems(values);
      })
      .catch((error) => {
        if (live) setError(imageError(error));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [offset]);
  async function action(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (error) {
      setError(imageError(error));
    } finally {
      setBusy(false);
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
    notifyImageFlows();
    onNavigate("flow", session.id);
  }
  async function open(item: GalleryItem) {
    const results =
      item.session_id && item.group_id
        ? await requestJson<ImageResult[]>(
            `${imagePath}/sessions/${item.session_id}/results`,
          )
        : [];
    const variants = results
      .filter((value) => value.group_id === item.group_id)
      .reverse();
    const ids = variants.length
      ? variants.map((value) => value.asset_id)
      : [item.asset_id];
    setViewer({
      ids,
      index: Math.max(0, ids.indexOf(item.asset_id)),
      sessionId: item.session_id,
      results: variants,
    });
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
  }
  useEffect(() => {
    if (!viewer) return;
    const key = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [Boolean(viewer)]);
  return (
    <>
      {error && (
        <p className="images-error" role="alert">
          {error}
        </p>
      )}
      <div className="images-gallery">
        {items.map((item) => (
          <article key={item.id}>
            <ImageAsset
              id={item.asset_id}
              onClick={() => void action(() => open(item))}
            />
            <div className="images-result-toolbar">
              <small>
                {item.variants > 1 ? `${item.variants} versions` : ""}
              </small>
              <div className="images-actions">
                <button
                  type="button"
                  className="images-icon"
                  disabled={busy}
                  title="Edit image"
                  aria-label="Edit image"
                  onClick={() => void action(() => edit(item.asset_id))}
                >
                  <Pencil size={18} />
                </button>
                <button
                  type="button"
                  className="images-icon"
                  title="Download image"
                  aria-label="Download image"
                  onClick={() =>
                    void action(() => downloadImage(item.asset_id))
                  }
                >
                  <Download size={18} />
                </button>
                {item.session_id && (
                  <button
                    type="button"
                    className="images-icon"
                    title="Open image flow"
                    aria-label="Open image flow"
                    onClick={() => onNavigate("flow", item.session_id!)}
                  >
                    <History size={18} />
                  </button>
                )}
                <button
                  type="button"
                  className="images-icon"
                  title="Remove from gallery"
                  aria-label="Remove from gallery"
                  disabled={busy}
                  onClick={() =>
                    void action(async () => {
                      await requestJson(`${imagePath}/gallery/${item.id}`, {
                        method: "DELETE",
                      });
                      setItems((values) =>
                        values.filter((value) => value.id !== item.id),
                      );
                    })
                  }
                >
                  <Trash2 size={18} />
                </button>
              </div>
            </div>
          </article>
        ))}
      </div>
      {!items.length && (
        <p className="images-empty" role="status">
          {loading ? "Loading images..." : "No images yet."}
        </p>
      )}
      <div className="images-actions images-pagination">
        <button
          type="button"
          disabled={!offset}
          onClick={() => setOffset((value) => Math.max(0, value - 50))}
        >
          Newer
        </button>
        <button
          type="button"
          disabled={items.length < 50}
          onClick={() => setOffset((value) => value + 50)}
        >
          Older
        </button>
      </div>
      {viewer && (
        <ImagePanel
          title={`Image ${viewer.index + 1} of ${viewer.ids.length}`}
          className="images-viewer-panel"
          onClose={() => setViewer(null)}
        >
          <SwipeSurface onStep={move}>
            <ImageAsset
              id={viewer.ids[viewer.index]}
              neighbors={viewer.ids.filter(
                (_, index) => Math.abs(index - viewer.index) === 1,
              )}
            />
          </SwipeSurface>
          <div className="images-actions">
            <button
              type="button"
              className="images-icon"
              disabled={viewer.index === 0}
              title="Previous image"
              aria-label="Previous image"
              onClick={() => move(-1)}
            >
              <ArrowLeft />
            </button>
            <button
              type="button"
              className="images-icon"
              disabled={viewer.index === viewer.ids.length - 1}
              title="Next image"
              aria-label="Next image"
              onClick={() => move(1)}
            >
              <ArrowRight />
            </button>
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
              title="Download image"
              aria-label="Download image"
              onClick={() =>
                void action(() => downloadImage(viewer.ids[viewer.index]))
              }
            >
              <Download />
            </button>
            {viewer.sessionId && (
              <button
                type="button"
                onClick={() => {
                  onNavigate("flow", viewer.sessionId!);
                }}
              >
                Open flow
              </button>
            )}
          </div>
        </ImagePanel>
      )}
    </>
  );
}
