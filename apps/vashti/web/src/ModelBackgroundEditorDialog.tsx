import { useEffect, useState } from "react";
import { ImagePlus, Save, Trash2, X } from "lucide-react";
import { RetroLoader } from "./common";
import { ModelBackgroundPreview } from "./ModelBackground";
import type { ModelBackgroundMode } from "./types";

type BackgroundLayout = {
  mode: ModelBackgroundMode;
  x: number;
  y: number;
  scale: number;
};

export type ModelBackgroundEdit = {
  assetId: string | null;
  file: File | null;
  dim: number;
  messageDim: number;
  landscape: BackgroundLayout;
  portrait: BackgroundLayout;
};

export function ModelBackgroundEditorDialog({
  title,
  assetId,
  privateAssetId,
  dim,
  messageDim,
  landscape,
  portrait,
  inheritedBackground,
  isBusy,
  error,
  onCancel,
  onSave
}: {
  title: string;
  assetId: string | null;
  privateAssetId?: string | null;
  dim: number;
  messageDim: number;
  landscape: BackgroundLayout;
  portrait: BackgroundLayout;
  inheritedBackground?: {
    assetId: string;
    dim: number;
    messageDim: number;
    landscape: BackgroundLayout;
    portrait: BackgroundLayout;
  } | null;
  isBusy: boolean;
  error: string | null;
  onCancel: () => void;
  onSave: (edit: ModelBackgroundEdit) => void;
}) {
  const [nextAssetId, setNextAssetId] = useState(assetId);
  const [file, setFile] = useState<File | null>(null);
  const [nextDim, setNextDim] = useState(dim);
  const [nextMessageDim, setNextMessageDim] = useState(messageDim);
  const [nextLandscape, setNextLandscape] = useState(landscape);
  const [nextPortrait, setNextPortrait] = useState(portrait);
  const [orientation, setOrientation] = useState<"landscape" | "portrait">("landscape");
  const hasOwnImage = Boolean(nextAssetId || file);
  const previewAssetId =
    nextAssetId ?? (!file && inheritedBackground ? inheritedBackground.assetId : null);
  const previewDim = hasOwnImage ? nextDim : (inheritedBackground?.dim ?? nextDim);
  const previewMessageDim = hasOwnImage
    ? nextMessageDim
    : (inheritedBackground?.messageDim ?? nextMessageDim);
  const activeLayout =
    orientation === "landscape"
      ? hasOwnImage
        ? nextLandscape
        : (inheritedBackground?.landscape ?? nextLandscape)
      : hasOwnImage
        ? nextPortrait
        : (inheritedBackground?.portrait ?? nextPortrait);
  const canEditLayout = hasOwnImage;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape" && !isBusy) {
        onCancel();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isBusy, onCancel]);

  function updateLayout(patch: Partial<BackgroundLayout>) {
    if (orientation === "landscape") {
      setNextLandscape((current) => ({ ...current, ...patch }));
    } else {
      setNextPortrait((current) => ({ ...current, ...patch }));
    }
  }

  return (
    <div
      className="confirm-backdrop model-background-editor-backdrop"
      role="presentation"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget && !isBusy) {
          onCancel();
        }
      }}
    >
      <section
        className="model-background-editor-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-background-editor-title"
      >
        <header className="model-background-editor-header">
          <div>
            <p className="eyebrow">Chat Background</p>
            <h2 id="model-background-editor-title">{title}</h2>
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="Close background editor"
            disabled={isBusy}
            onClick={onCancel}
          >
            <X />
          </button>
        </header>

        {!hasOwnImage && inheritedBackground && (
          <p className="model-background-inherited">
            Using the server default. Upload an image to create a personal override.
          </p>
        )}

        <div className="model-background-editor-grid">
          <div className="model-background-preview-column">
            <div className="model-background-orientation" aria-label="Preview orientation">
              <button
                type="button"
                className={orientation === "landscape" ? "segmented-option-active" : ""}
                onClick={() => setOrientation("landscape")}
              >
                Landscape
              </button>
              <button
                type="button"
                className={orientation === "portrait" ? "segmented-option-active" : ""}
                onClick={() => setOrientation("portrait")}
              >
                Portrait
              </button>
            </div>
            <ModelBackgroundPreview
              assetId={previewAssetId}
              privateAssetId={!file && previewAssetId === privateAssetId ? privateAssetId : null}
              previewFile={file}
              layout={activeLayout}
              dim={previewDim}
              messageDim={previewMessageDim}
              orientation={orientation}
            />
            <div className="model-background-file-actions">
              <label className="secondary-button file-button">
                <ImagePlus />
                <span>{hasOwnImage ? "Replace image" : "Choose image"}</span>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/gif"
                  disabled={isBusy}
                  onChange={(event) => {
                    const nextFile = event.currentTarget.files?.[0] ?? null;
                    if (nextFile) {
                      setFile(nextFile);
                      setNextAssetId(null);
                    }
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              {hasOwnImage && (
                <button
                  type="button"
                  className="danger-button"
                  disabled={isBusy}
                  onClick={() => {
                    setNextAssetId(null);
                    setFile(null);
                  }}
                >
                  <Trash2 />
                  <span>Remove</span>
                </button>
              )}
            </div>
          </div>

          <div className="model-background-controls">
            <fieldset disabled={!canEditLayout || isBusy}>
              <legend>{orientation === "landscape" ? "Landscape layout" : "Portrait layout"}</legend>
              <div className="model-background-mode" aria-label="Background sizing">
                {(["fill", "fit", "stretch", "tile"] as ModelBackgroundMode[]).map((mode) => (
                  <button
                    type="button"
                    key={mode}
                    className={activeLayout.mode === mode ? "segmented-option-active" : ""}
                    onClick={() => updateLayout({ mode })}
                  >
                    {mode}
                  </button>
                ))}
              </div>

              <label className="model-background-dim-control">
                <span>Dim image</span>
                <output>{Math.round(previewDim * 100)}%</output>
                <input
                  type="range"
                  min="0"
                  max="90"
                  step="1"
                  value={Math.round(previewDim * 100)}
                  onChange={(event) => setNextDim(Number(event.currentTarget.value) / 100)}
                />
              </label>

              <label className="model-background-dim-control">
                <span>Dim behind messages</span>
                <output>{Math.round(previewMessageDim * 100)}%</output>
                <input
                  type="range"
                  min="0"
                  max="98"
                  step="1"
                  value={Math.round(previewMessageDim * 100)}
                  onChange={(event) =>
                    setNextMessageDim(Number(event.currentTarget.value) / 100)
                  }
                />
              </label>

              {activeLayout.mode === "tile" && (
                <label className="model-background-dim-control">
                  <span>Tile size</span>
                  <output>{Math.round(activeLayout.scale)}%</output>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="1"
                    value={activeLayout.scale}
                    onChange={(event) =>
                      updateLayout({ scale: Number(event.currentTarget.value) })
                    }
                  />
                </label>
              )}

              <div>
                <span className="model-background-control-label">Placement</span>
                <div className="model-background-position" aria-label="Background placement">
                  {[0, 50, 100].flatMap((y) =>
                    [0, 50, 100].map((x) => (
                      <button
                        type="button"
                        key={`${x}:${y}`}
                        className={activeLayout.x === x && activeLayout.y === y ? "active" : ""}
                        aria-label={`${positionWord(y, "top", "center", "bottom")} ${positionWord(x, "left", "center", "right")}`}
                        onClick={() => updateLayout({ x, y })}
                      >
                        <span />
                      </button>
                    ))
                  )}
                </div>
              </div>
            </fieldset>
          </div>
        </div>

        {error && <p className="error">{error}</p>}
        <div className="dialog-actions">
          <button
            type="button"
            className="secondary-button"
            disabled={isBusy}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={isBusy}
            onClick={() =>
              onSave({
                assetId: nextAssetId,
                file,
                dim: nextDim,
                messageDim: nextMessageDim,
                landscape: nextLandscape,
                portrait: nextPortrait
              })
            }
          >
            {isBusy ? <RetroLoader /> : <Save />}
            <span>{isBusy ? "Saving..." : "Save"}</span>
          </button>
        </div>
      </section>
    </div>
  );
}

function positionWord(value: number, low: string, middle: string, high: string) {
  return value === 0 ? low : value === 100 ? high : middle;
}
