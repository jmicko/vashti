import { useEffect, useLayoutEffect, useState, type ReactNode } from "react";
import { Columns2 } from "lucide-react";
import { ImageAsset, type ImageAssetStatus } from "./ImageAsset";
import { ComparisonPress } from "./comparison";
import { SwipeSurface } from "./SwipeSurface";

export function ImageComparison({
  id,
  alt = "Generated image",
  referenceId,
  dimensions,
  neighbors,
  onClick,
  onStep,
  actionsClassName = "images-actions",
  children,
}: {
  id: string;
  alt?: string;
  referenceId?: string;
  dimensions?: { width: number; height: number };
  neighbors?: string[];
  onClick?: () => void;
  onStep: (direction: number) => void;
  actionsClassName?: string;
  children: ReactNode;
}) {
  const [showReference, setShowReference] = useState(false);
  const [press] = useState(() => new ComparisonPress(setShowReference));
  const [resultStatus, setResultStatus] = useState<ImageAssetStatus | null>(null);
  const [referenceStatus, setReferenceStatus] =
    useState<ImageAssetStatus | null>(null);
  const available = Boolean(
    referenceId &&
      resultStatus?.id === id &&
      resultStatus.state === "ready" &&
      referenceStatus?.id === referenceId &&
      referenceStatus.state === "ready",
  );
  const showing = available && showReference;
  const failed =
    referenceStatus?.id === referenceId &&
    referenceStatus?.state === "unavailable";
  useLayoutEffect(() => {
    press.reset();
  }, [id, referenceId, press]);
  useLayoutEffect(() => press.setAvailable(available), [available, press]);
  useEffect(() => {
    const reset = () => press.reset();
    const visibility = () => {
      if (document.hidden) reset();
    };
    window.addEventListener("blur", reset);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("blur", reset);
      document.removeEventListener("visibilitychange", visibility);
      press.reset();
    };
  }, [press]);
  return (
    <>
      <SwipeSurface onStep={onStep}>
        <div
          className={`image-compare-frame${referenceId ? " image-compare-enabled" : ""}${showing ? " image-compare-show-reference" : ""}`}
        >
          <div
            className="image-compare-result"
            aria-hidden={showing}
            inert={showing || undefined}
          >
            <ImageAsset
              id={id}
              alt={alt}
              dimensions={dimensions}
              neighbors={neighbors}
              onClick={onClick}
              onStatus={referenceId ? setResultStatus : undefined}
            />
          </div>
          {referenceId && (
            <div className="image-compare-reference" aria-hidden={!showing}>
              <ImageAsset
                id={referenceId}
                alt="First reference used for this edit"
                onStatus={setReferenceStatus}
              />
            </div>
          )}
          {showing && <span className="image-compare-label">Reference 1</span>}
        </div>
      </SwipeSurface>
      <div className={actionsClassName}>
        {children}
        {referenceId && (
          <button
            type="button"
            className="image-compare-button"
            aria-label="Compare with first reference"
            aria-pressed={showing}
            title={
              failed
                ? "Reference image unavailable"
                : available
                  ? "Hold to compare with the first reference"
                  : "Loading reference image"
            }
            disabled={!available}
            onPointerDown={(event) => {
              if (!event.isPrimary || event.button !== 0) return;
              event.currentTarget.setPointerCapture(event.pointerId);
              press.start(`pointer:${event.pointerId}`);
            }}
            onPointerUp={(event) => press.end(`pointer:${event.pointerId}`)}
            onPointerCancel={(event) => press.end(`pointer:${event.pointerId}`)}
            onLostPointerCapture={(event) =>
              press.end(`pointer:${event.pointerId}`)
            }
            onBlur={() => press.reset()}
            onContextMenu={(event) => event.preventDefault()}
            onClick={(event) => {
              // Keyboard/screen-reader activation toggles; pointer release stays momentary.
              if (event.detail === 0) press.activate();
            }}
          >
            <Columns2 size={18} />
            Compare
          </button>
        )}
      </div>
      {failed && (
        <small role="status" className="image-compare-error">
          Reference image unavailable
        </small>
      )}
    </>
  );
}
